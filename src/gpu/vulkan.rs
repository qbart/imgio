// SPDX-FileCopyrightText: 2010-2026 Mark Callow, The Khronos Group Inc. and KTX-Software contributors
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from KTX-Software (via ktx-rs `src/vkloader.rs`) and modified; see NOTICE.md.
//! Vulkan texture upload (feature `vulkan`): port of KTX-Software `lib/src/vkloader.c` and
//! `vk_funcs.c` for [`TextureSource`]s.
//!
//! This is the Vulkan loader of AGENTS.md rule 9, one of the two modules that may use
//! `unsafe`: every Vulkan command is called through a function pointer on handles supplied by
//! the application. The functions taking Vulkan handles are `unsafe fn`; their contract is the
//! one of the Vulkan commands they call (valid handles that belong together, externally
//! synchronized queue and command pool, a live allocator).
//!
//! The Vulkan library is never loaded by imgio: commands come from the application through
//! [`VulkanFunctions`] (at least `vkGetInstanceProcAddr` and an instance), or are looked up in
//! the running process ([`load_vulkan_function`]), which must have linked or globally loaded
//! the Vulkan loader.
//!
//! Deviations from C: every `VkResult` is checked and the objects created so far are released
//! on failure; a missing command returns `Unsupported` instead of crashing; mapped memory is
//! written through bounds-checked slices; the linear-tiling copy honours the driver's row,
//! array and depth pitches on every path; mipmap blits use the image's layer count for both
//! subresources (C uses 1 for the destination, invalid for arrays); with `generate_mipmaps`
//! only level 0 is uploaded (C copies every level, then overwrites levels 1.. by blits).
#![allow(unsafe_code)]

use std::ffi::{CStr, c_void};
use std::sync::OnceLock;

pub use ash::vk;

use super::texture::{Shape, TextureSource, copy_tight};
use crate::{Error, Result};

const DEFAULT_FENCE_TIMEOUT: u64 = 100_000_000_000;

/// The imgio error of a failed Vulkan command.
fn vk_error(r: vk::Result) -> Error {
    match r {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            Error::OutOfMemory
        }
        vk::Result::ERROR_FORMAT_NOT_SUPPORTED => Error::Unsupported,
        r => Error::Gpu(r.as_raw()),
    }
}

fn check(r: vk::Result) -> Result<()> {
    if r == vk::Result::SUCCESS {
        Ok(())
    } else {
        Err(vk_error(r))
    }
}

/// Port of `ktxVulkanFunctions`: the Vulkan commands the upload uses. Unset entries are loaded
/// by [`VulkanDevice::new`].
#[derive(Clone, Copy, Default)]
pub struct VulkanFunctions {
    pub vk_get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub vk_get_device_proc_addr: Option<vk::PFN_vkGetDeviceProcAddr>,
    pub vk_allocate_command_buffers: Option<vk::PFN_vkAllocateCommandBuffers>,
    pub vk_allocate_memory: Option<vk::PFN_vkAllocateMemory>,
    pub vk_begin_command_buffer: Option<vk::PFN_vkBeginCommandBuffer>,
    pub vk_bind_buffer_memory: Option<vk::PFN_vkBindBufferMemory>,
    pub vk_bind_image_memory: Option<vk::PFN_vkBindImageMemory>,
    pub vk_cmd_blit_image: Option<vk::PFN_vkCmdBlitImage>,
    pub vk_cmd_copy_buffer_to_image: Option<vk::PFN_vkCmdCopyBufferToImage>,
    pub vk_cmd_pipeline_barrier: Option<vk::PFN_vkCmdPipelineBarrier>,
    pub vk_create_image: Option<vk::PFN_vkCreateImage>,
    pub vk_destroy_image: Option<vk::PFN_vkDestroyImage>,
    pub vk_create_buffer: Option<vk::PFN_vkCreateBuffer>,
    pub vk_destroy_buffer: Option<vk::PFN_vkDestroyBuffer>,
    pub vk_create_fence: Option<vk::PFN_vkCreateFence>,
    pub vk_destroy_fence: Option<vk::PFN_vkDestroyFence>,
    pub vk_end_command_buffer: Option<vk::PFN_vkEndCommandBuffer>,
    pub vk_free_command_buffers: Option<vk::PFN_vkFreeCommandBuffers>,
    pub vk_free_memory: Option<vk::PFN_vkFreeMemory>,
    pub vk_get_buffer_memory_requirements: Option<vk::PFN_vkGetBufferMemoryRequirements>,
    pub vk_get_image_memory_requirements: Option<vk::PFN_vkGetImageMemoryRequirements>,
    pub vk_get_image_subresource_layout: Option<vk::PFN_vkGetImageSubresourceLayout>,
    pub vk_get_physical_device_image_format_properties:
        Option<vk::PFN_vkGetPhysicalDeviceImageFormatProperties>,
    pub vk_get_physical_device_format_properties:
        Option<vk::PFN_vkGetPhysicalDeviceFormatProperties>,
    pub vk_get_physical_device_memory_properties:
        Option<vk::PFN_vkGetPhysicalDeviceMemoryProperties>,
    pub vk_map_memory: Option<vk::PFN_vkMapMemory>,
    pub vk_queue_submit: Option<vk::PFN_vkQueueSubmit>,
    pub vk_queue_wait_idle: Option<vk::PFN_vkQueueWaitIdle>,
    pub vk_unmap_memory: Option<vk::PFN_vkUnmapMemory>,
    pub vk_wait_for_fences: Option<vk::PFN_vkWaitForFences>,
}

/// The process (or `vulkan-1.dll`) searched for Vulkan commands (`ktxVulkanModuleHandle`).
static VULKAN_MODULE: OnceLock<Option<libloading::Library>> = OnceLock::new();

#[cfg(unix)]
fn vulkan_module() -> Option<libloading::Library> {
    // dlopen(NULL): the process and every library it loaded globally.
    Some(libloading::os::unix::Library::this().into())
}

#[cfg(windows)]
fn vulkan_module() -> Option<libloading::Library> {
    libloading::os::windows::Library::open_already_loaded("vulkan-1.dll")
        .ok()
        .map(Into::into)
}

#[cfg(not(any(unix, windows)))]
fn vulkan_module() -> Option<libloading::Library> {
    None
}

/// Port of `ktxLoadVulkanFunction`: looks a Vulkan command up in the running process (`None`
/// when the Vulkan loader is not linked or globally loaded, or has no such command).
pub fn load_vulkan_function(name: &CStr) -> vk::PFN_vkVoidFunction {
    let module = VULKAN_MODULE.get_or_init(vulkan_module).as_ref()?;
    // SAFETY: the symbol is a Vulkan command, only called through its PFN type.
    unsafe { module.get::<unsafe extern "system" fn()>(name.to_bytes_with_nul()) }
        .ok()
        .map(|s| *s)
}

/// Reinterprets a `PFN_vkVoidFunction` as the command's real type.
macro_rules! cast_pfn {
    ($pfn:expr, $t:ty) => {
        // SAFETY: the pointer was returned for the command named like `$t`.
        $pfn.map(|f| unsafe { std::mem::transmute::<unsafe extern "system" fn(), $t>(f) })
    };
}

/// A loaded command, or `Unsupported`.
macro_rules! vkf {
    ($funcs:expr, $name:ident) => {
        $funcs.$name.ok_or(Error::Unsupported)?
    };
}

/// Port of `ktxVulkanDeviceInfo`: the device on which textures are created, with the command
/// buffer used for uploads.
pub struct VulkanDevice {
    pub instance: vk::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: vk::Device,
    /// Queue the upload commands are submitted to (a graphics queue for mipmap blits).
    pub queue: vk::Queue,
    /// Pool of `cmd_buffer`; it must allow resetting single command buffers
    /// (`RESET_COMMAND_BUFFER`) to upload more than once.
    pub cmd_pool: vk::CommandPool,
    /// Command buffer allocated by [`VulkanDevice::new`].
    pub cmd_buffer: vk::CommandBuffer,
    /// Allocation callbacks for created objects (may be null).
    pub allocator: *const vk::AllocationCallbacks<'static>,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub functions: VulkanFunctions,
}

impl VulkanDevice {
    /// Port of `ktxVulkanDeviceInfo_ConstructEx`: loads the commands missing from `functions`
    /// and allocates the upload command buffer.
    ///
    /// With an `instance` and `functions` holding `vkGetInstanceProcAddr`, every other command
    /// is loaded through it; otherwise `vkGetInstanceProcAddr`, `vkGetDeviceProcAddr` and the
    /// physical device queries are looked up in the process ([`load_vulkan_function`]).
    ///
    /// Errors: `Unsupported` (a command could not be loaded), `OutOfMemory`, `Gpu`.
    ///
    /// # Safety
    /// The handles must be valid and belong together (`instance` may be null), `allocator`
    /// must be null or outlive the returned device and the textures created with it, and the
    /// commands in `functions` must be Vulkan commands of `instance` / `device`.
    pub unsafe fn new(
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        queue: vk::Queue,
        cmd_pool: vk::CommandPool,
        allocator: *const vk::AllocationCallbacks<'static>,
        functions: Option<&VulkanFunctions>,
    ) -> Result<VulkanDevice> {
        let mut f = functions.copied().unwrap_or_default();

        macro_rules! load_process {
            ($member:ident, $name:literal, $t:ty) => {
                if f.$member.is_none() {
                    f.$member = cast_pfn!(load_vulkan_function($name), $t);
                    if f.$member.is_none() {
                        return Err(Error::Unsupported);
                    }
                }
            };
        }
        macro_rules! load_instance {
            ($member:ident, $name:literal, $t:ty) => {
                if f.$member.is_none() {
                    let gipa = vkf!(f, vk_get_instance_proc_addr);
                    // SAFETY: valid (or null) instance and a NUL terminated command name.
                    f.$member = cast_pfn!(unsafe { gipa(instance, $name.as_ptr()) }, $t);
                    if f.$member.is_none() {
                        return Err(Error::Unsupported);
                    }
                }
            };
        }
        macro_rules! load_device {
            ($member:ident, $name:literal, $t:ty) => {
                if f.$member.is_none() {
                    let gdpa = vkf!(f, vk_get_device_proc_addr);
                    // SAFETY: valid device and a NUL terminated command name.
                    f.$member = cast_pfn!(unsafe { gdpa(device, $name.as_ptr()) }, $t);
                    if f.$member.is_none() {
                        return Err(Error::Unsupported);
                    }
                }
            };
        }

        if instance == vk::Instance::null() || functions.is_none() {
            // No functions given: take a shot at loading them from the process.
            load_process!(
                vk_get_instance_proc_addr,
                c"vkGetInstanceProcAddr",
                vk::PFN_vkGetInstanceProcAddr
            );
            if instance == vk::Instance::null() {
                // Instance-level commands cannot be loaded without an instance.
                load_process!(
                    vk_get_device_proc_addr,
                    c"vkGetDeviceProcAddr",
                    vk::PFN_vkGetDeviceProcAddr
                );
                load_process!(
                    vk_get_physical_device_format_properties,
                    c"vkGetPhysicalDeviceFormatProperties",
                    vk::PFN_vkGetPhysicalDeviceFormatProperties
                );
                load_process!(
                    vk_get_physical_device_image_format_properties,
                    c"vkGetPhysicalDeviceImageFormatProperties",
                    vk::PFN_vkGetPhysicalDeviceImageFormatProperties
                );
                load_process!(
                    vk_get_physical_device_memory_properties,
                    c"vkGetPhysicalDeviceMemoryProperties",
                    vk::PFN_vkGetPhysicalDeviceMemoryProperties
                );
            }
        }

        load_instance!(
            vk_get_physical_device_format_properties,
            c"vkGetPhysicalDeviceFormatProperties",
            vk::PFN_vkGetPhysicalDeviceFormatProperties
        );
        load_instance!(
            vk_get_physical_device_memory_properties,
            c"vkGetPhysicalDeviceMemoryProperties",
            vk::PFN_vkGetPhysicalDeviceMemoryProperties
        );
        load_instance!(
            vk_get_physical_device_image_format_properties,
            c"vkGetPhysicalDeviceImageFormatProperties",
            vk::PFN_vkGetPhysicalDeviceImageFormatProperties
        );
        load_instance!(
            vk_get_device_proc_addr,
            c"vkGetDeviceProcAddr",
            vk::PFN_vkGetDeviceProcAddr
        );
        load_device!(
            vk_allocate_command_buffers,
            c"vkAllocateCommandBuffers",
            vk::PFN_vkAllocateCommandBuffers
        );
        load_device!(
            vk_free_command_buffers,
            c"vkFreeCommandBuffers",
            vk::PFN_vkFreeCommandBuffers
        );
        load_device!(
            vk_allocate_memory,
            c"vkAllocateMemory",
            vk::PFN_vkAllocateMemory
        );
        load_device!(vk_free_memory, c"vkFreeMemory", vk::PFN_vkFreeMemory);
        load_device!(
            vk_begin_command_buffer,
            c"vkBeginCommandBuffer",
            vk::PFN_vkBeginCommandBuffer
        );
        load_device!(
            vk_end_command_buffer,
            c"vkEndCommandBuffer",
            vk::PFN_vkEndCommandBuffer
        );
        load_device!(
            vk_bind_buffer_memory,
            c"vkBindBufferMemory",
            vk::PFN_vkBindBufferMemory
        );
        load_device!(
            vk_bind_image_memory,
            c"vkBindImageMemory",
            vk::PFN_vkBindImageMemory
        );
        load_device!(vk_cmd_blit_image, c"vkCmdBlitImage", vk::PFN_vkCmdBlitImage);
        load_device!(
            vk_cmd_copy_buffer_to_image,
            c"vkCmdCopyBufferToImage",
            vk::PFN_vkCmdCopyBufferToImage
        );
        load_device!(
            vk_cmd_pipeline_barrier,
            c"vkCmdPipelineBarrier",
            vk::PFN_vkCmdPipelineBarrier
        );
        load_device!(vk_create_image, c"vkCreateImage", vk::PFN_vkCreateImage);
        load_device!(vk_destroy_image, c"vkDestroyImage", vk::PFN_vkDestroyImage);
        load_device!(vk_create_buffer, c"vkCreateBuffer", vk::PFN_vkCreateBuffer);
        load_device!(
            vk_destroy_buffer,
            c"vkDestroyBuffer",
            vk::PFN_vkDestroyBuffer
        );
        load_device!(vk_create_fence, c"vkCreateFence", vk::PFN_vkCreateFence);
        load_device!(vk_destroy_fence, c"vkDestroyFence", vk::PFN_vkDestroyFence);
        load_device!(
            vk_wait_for_fences,
            c"vkWaitForFences",
            vk::PFN_vkWaitForFences
        );
        load_device!(vk_map_memory, c"vkMapMemory", vk::PFN_vkMapMemory);
        load_device!(vk_unmap_memory, c"vkUnmapMemory", vk::PFN_vkUnmapMemory);
        load_device!(vk_queue_submit, c"vkQueueSubmit", vk::PFN_vkQueueSubmit);
        load_device!(
            vk_queue_wait_idle,
            c"vkQueueWaitIdle",
            vk::PFN_vkQueueWaitIdle
        );
        load_device!(
            vk_get_buffer_memory_requirements,
            c"vkGetBufferMemoryRequirements",
            vk::PFN_vkGetBufferMemoryRequirements
        );
        load_device!(
            vk_get_image_memory_requirements,
            c"vkGetImageMemoryRequirements",
            vk::PFN_vkGetImageMemoryRequirements
        );
        load_device!(
            vk_get_image_subresource_layout,
            c"vkGetImageSubresourceLayout",
            vk::PFN_vkGetImageSubresourceLayout
        );

        let get_memory_properties = vkf!(f, vk_get_physical_device_memory_properties);
        let allocate_command_buffers = vkf!(f, vk_allocate_command_buffers);
        let mut memory_properties = vk::PhysicalDeviceMemoryProperties::default();
        // SAFETY: valid physical device; plain output struct.
        unsafe { get_memory_properties(physical_device, &mut memory_properties) };

        // A separate command buffer for texture loading: image barriers and copies.
        let info = vk::CommandBufferAllocateInfo {
            command_pool: cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let mut cmd_buffer = vk::CommandBuffer::null();
        // SAFETY: valid device and pool; one command buffer is written.
        check(unsafe { allocate_command_buffers(device, &info, &mut cmd_buffer) })?;
        Ok(VulkanDevice {
            instance,
            physical_device,
            device,
            queue,
            cmd_pool,
            cmd_buffer,
            allocator,
            memory_properties,
            functions: f,
        })
    }

    /// Port of `ktxVulkanDeviceInfo_Destruct`: frees the upload command buffer.
    ///
    /// # Safety
    /// The device and command pool must still be valid and the command buffer not in use.
    pub unsafe fn destroy(self) {
        if let Some(free_command_buffers) = self.functions.vk_free_command_buffers {
            // SAFETY: caller contract.
            unsafe { free_command_buffers(self.device, self.cmd_pool, 1, &self.cmd_buffer) };
        }
    }

    /// Index of a memory type allowed by `type_bits` with `properties`.
    fn memory_type(&self, type_bits: u32, properties: vk::MemoryPropertyFlags) -> Result<u32> {
        let mp = &self.memory_properties;
        (0..mp.memory_type_count.min(vk::MAX_MEMORY_TYPES as u32))
            .find(|&i| {
                type_bits & (1 << i) != 0
                    && mp.memory_types[i as usize]
                        .property_flags
                        .contains(properties)
            })
            .ok_or(Error::Unsupported)
    }

    /// Port of `ktxTexture_VkUploadEx`: creates a Vulkan image holding the images of `src`.
    ///
    /// Optimal tiling copies the images through a host-visible staging buffer (padded rows
    /// are stripped on the way); linear tiling writes them to the mapped image memory at the
    /// driver's pitches. With `generate_mipmaps` the full mip chain is blitted from level 0.
    ///
    /// Errors: `InvalidArgument` (bad source shape), `Unsupported` (no `VkFormat`, format or
    /// size not supported by the device, missing command, no suitable memory type),
    /// `OutOfMemory`, `Gpu` (other `VkResult`), and the source's errors.
    ///
    /// # Safety
    /// The handles of `self` must still be valid; the queue and `cmd_buffer` must not be in
    /// use elsewhere during the call.
    pub unsafe fn upload(
        &mut self,
        src: &(impl TextureSource + ?Sized),
        options: &VulkanUploadOptions,
    ) -> Result<VulkanTexture> {
        // SAFETY: forwarded contract.
        unsafe { upload(self, src, options, None) }
    }

    /// Port of `ktxTexture_VkUploadEx_WithSuballocator`: [`VulkanDevice::upload`] with the
    /// image and staging memory allocated through `sub_allocator`.
    ///
    /// # Safety
    /// As [`VulkanDevice::upload`]; `sub_allocator` must allocate on `self.device`.
    pub unsafe fn upload_with_suballocator(
        &mut self,
        src: &(impl TextureSource + ?Sized),
        options: &VulkanUploadOptions,
        sub_allocator: &mut dyn SubAllocator,
    ) -> Result<VulkanTexture> {
        // SAFETY: forwarded contract.
        unsafe { upload(self, src, options, Some(sub_allocator)) }
    }
}

/// Port of `ktxVulkanTexture_subAllocatorCallbacks`: a memory suballocator.
///
/// # Safety
/// Implementations allocate, bind and map memory of the device the upload runs on: a mapped
/// pointer must stay valid for writes of the returned length until `unmap`.
pub unsafe trait SubAllocator {
    /// Allocates memory for `requirements`; returns a non-zero allocation id and its page
    /// count (only 1 page is supported).
    fn alloc_mem(
        &mut self,
        info: &vk::MemoryAllocateInfo<'_>,
        requirements: &vk::MemoryRequirements,
    ) -> std::result::Result<(u64, u64), vk::Result>;
    /// Binds `buffer` to allocation `id`.
    fn bind_buffer(&mut self, buffer: vk::Buffer, id: u64) -> vk::Result;
    /// Binds `image` to allocation `id`.
    fn bind_image(&mut self, image: vk::Image, id: u64) -> vk::Result;
    /// Maps `page` of allocation `id`; returns its address and length.
    fn map(
        &mut self,
        id: u64,
        page: u64,
    ) -> std::result::Result<(*mut c_void, vk::DeviceSize), vk::Result>;
    /// Unmaps `page` of allocation `id`.
    fn unmap(&mut self, id: u64, page: u64);
    /// Frees allocation `id`.
    fn free(&mut self, id: u64);
}

/// Options of [`VulkanDevice::upload`].
#[derive(Debug, Clone, Copy)]
pub struct VulkanUploadOptions {
    /// `OPTIMAL` (staging buffer copy) or `LINEAR` (host-written image).
    pub tiling: vk::ImageTiling,
    /// Image usage; `TRANSFER_DST` (optimal tiling) and `TRANSFER_SRC` (mipmaps) are added.
    pub usage: vk::ImageUsageFlags,
    /// Layout of the image after the upload.
    pub final_layout: vk::ImageLayout,
    /// Blit the full mip chain from level 0 (needs blit support for the format).
    pub generate_mipmaps: bool,
}

impl Default for VulkanUploadOptions {
    /// `ktxTexture_VkUpload`: optimal tiling, sampled, `SHADER_READ_ONLY_OPTIMAL`.
    fn default() -> Self {
        VulkanUploadOptions {
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::SAMPLED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            generate_mipmaps: false,
        }
    }
}

/// Port of `ktxVulkanTexture`: a Vulkan image created by an upload.
#[derive(Debug, Clone, Copy, Default)]
pub struct VulkanTexture {
    pub image: vk::Image,
    pub image_format: vk::Format,
    /// Layout of the image: the upload's `final_layout`.
    pub image_layout: vk::ImageLayout,
    /// Memory of the image (null with a suballocator).
    pub device_memory: vk::DeviceMemory,
    /// Allocation id of the image memory with a suballocator (0 otherwise).
    pub allocation_id: u64,
    /// View type matching the image: 2D, 2D array, cube, cube array or 3D.
    pub view_type: vk::ImageViewType,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub level_count: u32,
    /// Array layers of the image (layers times faces).
    pub layer_count: u32,
    vk_destroy_image: Option<vk::PFN_vkDestroyImage>,
    vk_free_memory: Option<vk::PFN_vkFreeMemory>,
}

impl VulkanTexture {
    /// Port of `ktxVulkanTexture_Destruct`: destroys the image and frees its memory.
    ///
    /// # Safety
    /// `device` must be the device the texture was created on (with `allocator` as passed to
    /// [`VulkanDevice::new`]) and the image no longer in use.
    pub unsafe fn destroy(
        &mut self,
        device: vk::Device,
        allocator: *const vk::AllocationCallbacks<'static>,
    ) {
        // SAFETY: caller contract.
        unsafe {
            if let Some(destroy_image) = self.vk_destroy_image {
                destroy_image(device, self.image, allocator);
            }
            if let Some(free_memory) = self.vk_free_memory {
                free_memory(device, self.device_memory, allocator);
            }
        }
        self.image = vk::Image::null();
        self.device_memory = vk::DeviceMemory::null();
    }

    /// Port of `ktxVulkanTexture_Destruct_WithSuballocator`: destroys the image and frees its
    /// memory through `sub_allocator`.
    ///
    /// # Safety
    /// As [`VulkanTexture::destroy`]; `sub_allocator` must be the one used for the upload.
    pub unsafe fn destroy_with_suballocator(
        &mut self,
        device: vk::Device,
        allocator: *const vk::AllocationCallbacks<'static>,
        sub_allocator: &mut dyn SubAllocator,
    ) {
        if let Some(destroy_image) = self.vk_destroy_image {
            // SAFETY: caller contract.
            unsafe { destroy_image(device, self.image, allocator) };
        }
        sub_allocator.free(self.allocation_id);
        self.image = vk::Image::null();
        self.allocation_id = 0;
    }
}

/// Memory from `vkAllocateMemory` or a suballocator.
#[derive(Clone, Copy)]
enum Mem {
    Device(vk::DeviceMemory),
    Sub(u64),
}

/// Host mapped memory, written through bounds-checked slices.
struct Mapped {
    ptr: *mut u8,
    len: usize,
}

impl Mapped {
    fn slice(&mut self) -> &mut [u8] {
        if self.ptr.is_null() {
            return &mut [];
        }
        // SAFETY: `ptr` is a live host mapping of `len` bytes while `self` is used (it is
        // unmapped only after the last write).
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn range(&mut self, offset: u64, len: usize) -> Result<&mut [u8]> {
        let start = usize::try_from(offset).map_err(|_| Error::InvalidData)?;
        let end = start.checked_add(len).ok_or(Error::InvalidData)?;
        let have = self.len;
        self.slice()
            .get_mut(start..end)
            .ok_or(Error::BufferTooSmall {
                needed: end,
                got: have,
            })
    }
}

/// The loaded commands used by an upload.
struct Fns {
    begin_command_buffer: vk::PFN_vkBeginCommandBuffer,
    end_command_buffer: vk::PFN_vkEndCommandBuffer,
    allocate_memory: vk::PFN_vkAllocateMemory,
    free_memory: vk::PFN_vkFreeMemory,
    bind_buffer_memory: vk::PFN_vkBindBufferMemory,
    bind_image_memory: vk::PFN_vkBindImageMemory,
    map_memory: vk::PFN_vkMapMemory,
    unmap_memory: vk::PFN_vkUnmapMemory,
    create_buffer: vk::PFN_vkCreateBuffer,
    destroy_buffer: vk::PFN_vkDestroyBuffer,
    create_image: vk::PFN_vkCreateImage,
    destroy_image: vk::PFN_vkDestroyImage,
    create_fence: vk::PFN_vkCreateFence,
    destroy_fence: vk::PFN_vkDestroyFence,
    wait_for_fences: vk::PFN_vkWaitForFences,
    queue_submit: vk::PFN_vkQueueSubmit,
    queue_wait_idle: vk::PFN_vkQueueWaitIdle,
    get_buffer_memory_requirements: vk::PFN_vkGetBufferMemoryRequirements,
    get_image_memory_requirements: vk::PFN_vkGetImageMemoryRequirements,
    get_image_subresource_layout: vk::PFN_vkGetImageSubresourceLayout,
    cmd_copy_buffer_to_image: vk::PFN_vkCmdCopyBufferToImage,
    cmd_pipeline_barrier: vk::PFN_vkCmdPipelineBarrier,
    cmd_blit_image: vk::PFN_vkCmdBlitImage,
    get_physical_device_image_format_properties: vk::PFN_vkGetPhysicalDeviceImageFormatProperties,
    get_physical_device_format_properties: vk::PFN_vkGetPhysicalDeviceFormatProperties,
}

impl Fns {
    fn new(f: &VulkanFunctions) -> Result<Fns> {
        Ok(Fns {
            begin_command_buffer: vkf!(f, vk_begin_command_buffer),
            end_command_buffer: vkf!(f, vk_end_command_buffer),
            allocate_memory: vkf!(f, vk_allocate_memory),
            free_memory: vkf!(f, vk_free_memory),
            bind_buffer_memory: vkf!(f, vk_bind_buffer_memory),
            bind_image_memory: vkf!(f, vk_bind_image_memory),
            map_memory: vkf!(f, vk_map_memory),
            unmap_memory: vkf!(f, vk_unmap_memory),
            create_buffer: vkf!(f, vk_create_buffer),
            destroy_buffer: vkf!(f, vk_destroy_buffer),
            create_image: vkf!(f, vk_create_image),
            destroy_image: vkf!(f, vk_destroy_image),
            create_fence: vkf!(f, vk_create_fence),
            destroy_fence: vkf!(f, vk_destroy_fence),
            wait_for_fences: vkf!(f, vk_wait_for_fences),
            queue_submit: vkf!(f, vk_queue_submit),
            queue_wait_idle: vkf!(f, vk_queue_wait_idle),
            get_buffer_memory_requirements: vkf!(f, vk_get_buffer_memory_requirements),
            get_image_memory_requirements: vkf!(f, vk_get_image_memory_requirements),
            get_image_subresource_layout: vkf!(f, vk_get_image_subresource_layout),
            cmd_copy_buffer_to_image: vkf!(f, vk_cmd_copy_buffer_to_image),
            cmd_pipeline_barrier: vkf!(f, vk_cmd_pipeline_barrier),
            cmd_blit_image: vkf!(f, vk_cmd_blit_image),
            get_physical_device_image_format_properties: vkf!(
                f,
                vk_get_physical_device_image_format_properties
            ),
            get_physical_device_format_properties: vkf!(
                f,
                vk_get_physical_device_format_properties
            ),
        })
    }
}

/// The objects created by an upload, released by [`Ctx::release`].
#[derive(Default)]
struct Objects {
    staging_buffer: vk::Buffer,
    staging_memory: Option<Mem>,
    fence: vk::Fence,
    image: vk::Image,
    image_memory: Option<Mem>,
}

/// One upload: device, commands and allocator.
struct Ctx<'d, 's> {
    dev: &'d VulkanDevice,
    fns: Fns,
    sub: Option<&'s mut dyn SubAllocator>,
}

// SAFETY (every method of `Ctx`): the device objects belong to `dev` (caller contract of the
// upload); each pointer passed to a command points to a local that outlives the call.
impl Ctx<'_, '_> {
    fn alloc(
        &mut self,
        req: &vk::MemoryRequirements,
        props: vk::MemoryPropertyFlags,
    ) -> Result<Mem> {
        let info = vk::MemoryAllocateInfo {
            allocation_size: req.size,
            memory_type_index: self.dev.memory_type(req.memory_type_bits, props)?,
            ..Default::default()
        };
        match self.sub.as_deref_mut() {
            None => {
                let mut memory = vk::DeviceMemory::null();
                check(unsafe {
                    (self.fns.allocate_memory)(
                        self.dev.device,
                        &info,
                        self.dev.allocator,
                        &mut memory,
                    )
                })?;
                Ok(Mem::Device(memory))
            }
            Some(s) => {
                let (id, pages) = s.alloc_mem(&info, req).map_err(vk_error)?;
                if id == 0 {
                    return Err(Error::OutOfMemory);
                }
                if pages > 1 {
                    // Sparse binding is unsupported, as in C.
                    s.free(id);
                    return Err(Error::Unsupported);
                }
                Ok(Mem::Sub(id))
            }
        }
    }

    fn free(&mut self, mem: Mem) {
        match (mem, self.sub.as_deref_mut()) {
            (Mem::Device(m), _) => unsafe {
                (self.fns.free_memory)(self.dev.device, m, self.dev.allocator)
            },
            (Mem::Sub(id), Some(s)) => s.free(id),
            (Mem::Sub(_), None) => {}
        }
    }

    fn bind_buffer(&mut self, buffer: vk::Buffer, mem: Mem) -> Result<()> {
        check(match (mem, self.sub.as_deref_mut()) {
            (Mem::Device(m), _) => unsafe {
                (self.fns.bind_buffer_memory)(self.dev.device, buffer, m, 0)
            },
            (Mem::Sub(id), Some(s)) => s.bind_buffer(buffer, id),
            (Mem::Sub(_), None) => vk::Result::ERROR_UNKNOWN,
        })
    }

    fn bind_image(&mut self, image: vk::Image, mem: Mem) -> Result<()> {
        check(match (mem, self.sub.as_deref_mut()) {
            (Mem::Device(m), _) => unsafe {
                (self.fns.bind_image_memory)(self.dev.device, image, m, 0)
            },
            (Mem::Sub(id), Some(s)) => s.bind_image(image, id),
            (Mem::Sub(_), None) => vk::Result::ERROR_UNKNOWN,
        })
    }

    fn map(&mut self, mem: Mem, size: vk::DeviceSize) -> Result<Mapped> {
        let (ptr, len) = match (mem, self.sub.as_deref_mut()) {
            (Mem::Device(m), _) => {
                let mut ptr: *mut c_void = std::ptr::null_mut();
                check(unsafe {
                    (self.fns.map_memory)(
                        self.dev.device,
                        m,
                        0,
                        size,
                        vk::MemoryMapFlags::empty(),
                        &mut ptr,
                    )
                })?;
                (ptr, size)
            }
            (Mem::Sub(id), Some(s)) => s.map(id, 0).map_err(vk_error)?,
            (Mem::Sub(_), None) => return Err(Error::Unsupported),
        };
        if ptr.is_null() {
            return Err(Error::OutOfMemory);
        }
        Ok(Mapped {
            ptr: ptr.cast::<u8>(),
            len: usize::try_from(len.min(size)).map_err(|_| Error::OutOfMemory)?,
        })
    }

    fn unmap(&mut self, mem: Mem) {
        match (mem, self.sub.as_deref_mut()) {
            (Mem::Device(m), _) => unsafe { (self.fns.unmap_memory)(self.dev.device, m) },
            (Mem::Sub(id), Some(s)) => s.unmap(id, 0),
            (Mem::Sub(_), None) => {}
        }
    }

    fn create_image(&self, info: &vk::ImageCreateInfo<'_>) -> Result<vk::Image> {
        let mut image = vk::Image::null();
        check(unsafe {
            (self.fns.create_image)(self.dev.device, info, self.dev.allocator, &mut image)
        })?;
        Ok(image)
    }

    fn image_requirements(&self, image: vk::Image) -> vk::MemoryRequirements {
        let mut req = vk::MemoryRequirements::default();
        unsafe { (self.fns.get_image_memory_requirements)(self.dev.device, image, &mut req) };
        req
    }

    /// Destroys the staging objects and fence, and the image too when `keep_image` is false.
    fn release(&mut self, o: &mut Objects, keep_image: bool) {
        let (device, allocator) = (self.dev.device, self.dev.allocator);
        unsafe {
            if o.fence != vk::Fence::null() {
                (self.fns.destroy_fence)(device, o.fence, allocator);
            }
            if o.staging_buffer != vk::Buffer::null() {
                (self.fns.destroy_buffer)(device, o.staging_buffer, allocator);
            }
        }
        if let Some(m) = o.staging_memory.take() {
            self.free(m);
        }
        o.staging_buffer = vk::Buffer::null();
        o.fence = vk::Fence::null();
        if !keep_image {
            if o.image != vk::Image::null() {
                unsafe { (self.fns.destroy_image)(device, o.image, allocator) };
            }
            if let Some(m) = o.image_memory.take() {
                self.free(m);
            }
            o.image = vk::Image::null();
        }
    }

    /// Port of `setImageLayout`: records an image layout transition barrier.
    fn set_image_layout(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange,
    ) {
        use vk::AccessFlags as A;
        use vk::ImageLayout as L;
        let mut barrier = vk::ImageMemoryBarrier {
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            old_layout,
            new_layout,
            image,
            subresource_range,
            ..Default::default()
        };
        // Source access: what must finish on the old layout before the transition.
        barrier.src_access_mask = match old_layout {
            L::UNDEFINED => A::empty(),
            L::PREINITIALIZED => A::HOST_WRITE,
            L::COLOR_ATTACHMENT_OPTIMAL => A::COLOR_ATTACHMENT_WRITE,
            L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => A::DEPTH_STENCIL_ATTACHMENT_WRITE,
            L::TRANSFER_SRC_OPTIMAL => A::TRANSFER_READ,
            L::TRANSFER_DST_OPTIMAL => A::TRANSFER_WRITE,
            L::SHADER_READ_ONLY_OPTIMAL => A::SHADER_READ,
            L::GENERAL => A::MEMORY_READ | A::MEMORY_WRITE,
            _ => A::empty(),
        };
        let mut dst_stage = vk::PipelineStageFlags::ALL_COMMANDS;
        // Destination access: the dependency of the new layout.
        match new_layout {
            L::TRANSFER_DST_OPTIMAL => barrier.dst_access_mask = A::TRANSFER_WRITE,
            L::TRANSFER_SRC_OPTIMAL => {
                barrier.src_access_mask |= A::TRANSFER_READ;
                barrier.dst_access_mask = A::TRANSFER_READ;
            }
            L::COLOR_ATTACHMENT_OPTIMAL => {
                barrier.src_access_mask = A::TRANSFER_READ;
                barrier.dst_access_mask = A::COLOR_ATTACHMENT_WRITE;
            }
            L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                barrier.dst_access_mask = A::DEPTH_STENCIL_ATTACHMENT_WRITE;
            }
            L::SHADER_READ_ONLY_OPTIMAL => {
                if barrier.src_access_mask.is_empty() {
                    barrier.src_access_mask = A::HOST_WRITE | A::TRANSFER_WRITE;
                }
                barrier.dst_access_mask = A::SHADER_READ;
                dst_stage =
                    vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::COMPUTE_SHADER;
            }
            L::GENERAL => barrier.dst_access_mask = A::MEMORY_READ | A::MEMORY_WRITE,
            _ => {}
        }
        unsafe {
            (self.fns.cmd_pipeline_barrier)(
                self.dev.cmd_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                dst_stage,
                vk::DependencyFlags::empty(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                1,
                &barrier,
            )
        };
    }

    /// Port of `generateMipmaps`: blits each level from the previous one, then transitions
    /// every level to the final layout.
    fn generate_mipmaps(
        &self,
        t: &VulkanTexture,
        filter: vk::Filter,
        initial_layout: vk::ImageLayout,
    ) {
        let range = |base_mip_level, level_count| vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level,
            level_count,
            base_array_layer: 0,
            layer_count: t.layer_count,
        };
        let layers = |mip_level| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level,
            base_array_layer: 0,
            layer_count: t.layer_count,
        };
        let corner = |level: u32| {
            let dim = |v: u32| v.checked_shr(level).unwrap_or(0).max(1) as i32;
            vk::Offset3D {
                x: dim(t.width),
                y: dim(t.height),
                z: dim(t.depth),
            }
        };
        // Base level to TRANSFER_SRC for blitting.
        self.set_image_layout(
            t.image,
            initial_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            range(0, 1),
        );
        for i in 1..t.level_count {
            let blit = vk::ImageBlit {
                src_subresource: layers(i - 1),
                src_offsets: [vk::Offset3D::default(), corner(i - 1)],
                dst_subresource: layers(i),
                dst_offsets: [vk::Offset3D::default(), corner(i)],
            };
            self.set_image_layout(
                t.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                range(i, 1),
            );
            unsafe {
                (self.fns.cmd_blit_image)(
                    self.dev.cmd_buffer,
                    t.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    t.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    1,
                    &blit,
                    filter,
                )
            };
            // Source of the next blit.
            self.set_image_layout(
                t.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                range(i, 1),
            );
        }
        // Every level is TRANSFER_SRC now.
        self.set_image_layout(
            t.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            t.image_layout,
            range(0, t.level_count),
        );
    }

    /// Ends the command buffer and submits it; waits on a fence (`fence`) or the queue.
    fn submit(&self, o: &mut Objects, fence: bool) -> Result<()> {
        let dev = self.dev;
        check(unsafe { (self.fns.end_command_buffer)(dev.cmd_buffer) })?;
        if fence {
            let info = vk::FenceCreateInfo::default();
            check(unsafe {
                (self.fns.create_fence)(dev.device, &info, dev.allocator, &mut o.fence)
            })?;
        }
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &dev.cmd_buffer,
            ..Default::default()
        };
        check(unsafe { (self.fns.queue_submit)(dev.queue, 1, &submit, o.fence) })?;
        if fence {
            check(unsafe {
                (self.fns.wait_for_fences)(dev.device, 1, &o.fence, vk::TRUE, DEFAULT_FENCE_TIMEOUT)
            })
        } else {
            check(unsafe { (self.fns.queue_wait_idle)(dev.queue) })
        }
    }

    fn begin(&self) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::default();
        check(unsafe { (self.fns.begin_command_buffer)(self.dev.cmd_buffer, &info) })
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The image created for a source: types, levels, format and blit filter.
struct Plan {
    shape: Shape,
    format: vk::Format,
    image_type: vk::ImageType,
    flags: vk::ImageCreateFlags,
    usage: vk::ImageUsageFlags,
    texture: VulkanTexture,
    /// Levels copied from the source.
    upload_levels: u32,
    filter: vk::Filter,
}

/// Port of the checks at the start of `ktxTexture_VkUploadEx_WithSuballocator`.
fn plan(
    ctx: &Ctx<'_, '_>,
    src: &(impl TextureSource + ?Sized),
    options: &VulkanUploadOptions,
) -> Result<Plan> {
    let shape = Shape::of(src)?;
    let raw = crate::ktx2::vk_format(shape.format).ok_or(Error::Unsupported)?;
    let format = vk::Format::from_raw(i32::try_from(raw).map_err(|_| Error::Unsupported)?);
    let cube = shape.faces == 6;
    let flags = if cube {
        vk::ImageCreateFlags::CUBE_COMPATIBLE
    } else {
        vk::ImageCreateFlags::empty()
    };
    let (image_type, view_type) = if shape.depth > 1 {
        (vk::ImageType::TYPE_3D, vk::ImageViewType::TYPE_3D)
    } else {
        let view = match (cube, shape.array) {
            (true, true) => vk::ImageViewType::CUBE_ARRAY,
            (true, false) => vk::ImageViewType::CUBE,
            (false, true) => vk::ImageViewType::TYPE_2D_ARRAY,
            (false, false) => vk::ImageViewType::TYPE_2D,
        };
        (vk::ImageType::TYPE_2D, view)
    };

    let mut usage = options.usage;
    if options.tiling == vk::ImageTiling::OPTIMAL {
        // Copies from the staging buffer.
        usage |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if options.generate_mipmaps {
        // Blits between levels.
        usage |= vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
    }
    let pd = ctx.dev.physical_device;
    let mut props = vk::ImageFormatProperties::default();
    // SAFETY: valid physical device; plain output struct.
    check(unsafe {
        (ctx.fns.get_physical_device_image_format_properties)(
            pd,
            format,
            image_type,
            options.tiling,
            usage,
            flags,
            &mut props,
        )
    })?;
    let layer_count = shape.images();
    if layer_count > props.max_array_layers {
        return Err(Error::Unsupported);
    }

    let mut filter = vk::Filter::LINEAR;
    let (level_count, upload_levels) = if options.generate_mipmaps {
        let mut fp = vk::FormatProperties::default();
        // SAFETY: valid physical device; plain output struct.
        unsafe { (ctx.fns.get_physical_device_format_properties)(pd, format, &mut fp) };
        let features = if options.tiling == vk::ImageTiling::OPTIMAL {
            fp.optimal_tiling_features
        } else {
            fp.linear_tiling_features
        };
        let needed = vk::FormatFeatureFlags::BLIT_DST | vk::FormatFeatureFlags::BLIT_SRC;
        if !features.contains(needed) {
            return Err(Error::Unsupported);
        }
        if !features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) {
            filter = vk::Filter::NEAREST;
        }
        (shape.max_levels(), 1)
    } else {
        (shape.levels, shape.levels)
    };
    if level_count > props.max_mip_levels {
        return Err(Error::Unsupported);
    }

    let texture = VulkanTexture {
        image_format: format,
        image_layout: options.final_layout,
        view_type,
        width: shape.width,
        height: shape.height,
        depth: shape.depth,
        level_count,
        layer_count,
        vk_destroy_image: Some(ctx.fns.destroy_image),
        vk_free_memory: Some(ctx.fns.free_memory),
        ..VulkanTexture::default()
    };
    Ok(Plan {
        shape,
        format,
        image_type,
        flags,
        usage,
        texture,
        upload_levels,
        filter,
    })
}

fn image_info(
    p: &Plan,
    tiling: vk::ImageTiling,
    initial: vk::ImageLayout,
) -> vk::ImageCreateInfo<'static> {
    vk::ImageCreateInfo {
        image_type: p.image_type,
        flags: p.flags,
        format: p.format,
        extent: vk::Extent3D {
            width: p.shape.width,
            height: p.shape.height,
            depth: p.shape.depth,
        },
        mip_levels: p.texture.level_count,
        array_layers: p.texture.layer_count,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling,
        usage: p.usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: initial,
        ..Default::default()
    }
}

/// Optimal tiling: staging buffer, `vkCmdCopyBufferToImage`, layout transitions.
fn upload_optimal(
    ctx: &mut Ctx<'_, '_>,
    o: &mut Objects,
    src: &(impl TextureSource + ?Sized),
    p: &Plan,
) -> Result<()> {
    let shape = &p.shape;
    let images = shape.images();
    // Region offsets must be multiples of the texel block size and of 4.
    let block = shape.format.block_bytes() as u64;
    let align = block / gcd(block, 4) * 4;
    // At most 32 levels (u32 sizes).
    let mut regions: [vk::BufferImageCopy; 32] = Default::default();
    let mut offset = 0u64;
    for level in 0..p.upload_levels {
        offset = offset.div_ceil(align) * align;
        let (width, height, depth) = shape.extent(level);
        *regions
            .get_mut(level as usize)
            .ok_or(Error::InvalidArgument)? = vk::BufferImageCopy {
            buffer_offset: offset,
            // Tightly packed rows.
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: level,
                base_array_layer: 0,
                layer_count: images,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width,
                height,
                depth,
            },
        };
        offset += shape.tight_image_bytes(level)? as u64 * u64::from(images);
    }
    let regions = regions
        .get(..p.upload_levels as usize)
        .ok_or(Error::InvalidArgument)?;

    // Host-visible staging buffer holding the tightly packed images.
    let info = vk::BufferCreateInfo {
        size: offset,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let dev = ctx.dev;
    check(unsafe {
        (ctx.fns.create_buffer)(dev.device, &info, dev.allocator, &mut o.staging_buffer)
    })?;
    let mut req = vk::MemoryRequirements::default();
    unsafe { (ctx.fns.get_buffer_memory_requirements)(dev.device, o.staging_buffer, &mut req) };
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let mem = ctx.alloc(&req, host)?;
    o.staging_memory = Some(mem);
    ctx.bind_buffer(o.staging_buffer, mem)?;
    let mut mapped = ctx.map(mem, req.size)?;
    // Fast path: images copied in bulk; padded rows are copied one by one (copy_tight).
    let filled = (|| -> Result<()> {
        for (level, region) in (0..p.upload_levels).zip(regions) {
            let size = shape.tight_image_bytes(level)?;
            let mut at = region.buffer_offset;
            for layer in 0..shape.layers {
                for face in 0..shape.faces {
                    let data = src.image_data(level, layer, face)?;
                    copy_tight(shape, level, data, mapped.range(at, size)?)?;
                    at += size as u64;
                }
            }
        }
        Ok(())
    })();
    ctx.unmap(mem);
    filled?;

    // Optimal tiled target image.
    o.image = ctx.create_image(&image_info(
        p,
        vk::ImageTiling::OPTIMAL,
        vk::ImageLayout::UNDEFINED,
    ))?;
    let req = ctx.image_requirements(o.image);
    let mem = ctx.alloc(&req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    o.image_memory = Some(mem);
    ctx.bind_image(o.image, mem)?;

    let mut t = p.texture;
    t.image = o.image;
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: p.upload_levels,
        base_array_layer: 0,
        layer_count: images,
    };
    ctx.begin()?;
    ctx.set_image_layout(
        o.image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        range,
    );
    unsafe {
        (ctx.fns.cmd_copy_buffer_to_image)(
            dev.cmd_buffer,
            o.staging_buffer,
            o.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            regions.len() as u32,
            regions.as_ptr(),
        )
    };
    if p.texture.level_count > p.upload_levels {
        ctx.generate_mipmaps(&t, p.filter, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    } else {
        ctx.set_image_layout(
            o.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            t.image_layout,
            range,
        );
    }
    ctx.submit(o, true)
}

/// Linear tiling: the images are written to the mapped image memory.
fn upload_linear(
    ctx: &mut Ctx<'_, '_>,
    o: &mut Objects,
    src: &(impl TextureSource + ?Sized),
    p: &Plan,
) -> Result<()> {
    let shape = &p.shape;
    o.image = ctx.create_image(&image_info(
        p,
        vk::ImageTiling::LINEAR,
        vk::ImageLayout::PREINITIALIZED,
    ))?;
    let req = ctx.image_requirements(o.image);
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let mem = ctx.alloc(&req, host)?;
    o.image_memory = Some(mem);
    ctx.bind_image(o.image, mem)?;
    let mut mapped = ctx.map(mem, req.size)?;
    let (device, image) = (ctx.dev.device, o.image);
    let get_layout = ctx.fns.get_image_subresource_layout;
    let filled = (|| -> Result<()> {
        for level in 0..p.upload_levels {
            let row = shape.row_bytes(level)?;
            let src_pitch = shape.src_row_pitch(level)?;
            let rows = shape.block_rows(level) as usize;
            let depth = shape.extent(level).2 as usize;
            for layer in 0..shape.layers {
                for face in 0..shape.faces {
                    let sub = vk::ImageSubresource {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level,
                        array_layer: layer * shape.faces + face,
                    };
                    let mut l = vk::SubresourceLayout::default();
                    // SAFETY: valid device and linear image; plain output struct.
                    unsafe { get_layout(device, image, &sub, &mut l) };
                    let data = src.image_data(level, layer, face)?;
                    let slice_pitch = (rows as u64) * l.row_pitch;
                    let tight = l.row_pitch == row as u64
                        && src_pitch == row
                        && (depth == 1 || l.depth_pitch == slice_pitch);
                    if tight {
                        // Fast path: one copy of the whole image.
                        let size = shape.tight_image_bytes(level)?;
                        let src = data.get(..size).ok_or(Error::InvalidData)?;
                        mapped.range(l.offset, size)?.copy_from_slice(src);
                        continue;
                    }
                    // Padded path: row by row at the driver's pitches.
                    for z in 0..depth {
                        for r in 0..rows {
                            let at = l.offset + z as u64 * l.depth_pitch + r as u64 * l.row_pitch;
                            let from = (z * rows + r) * src_pitch;
                            let src = data.get(from..from + row).ok_or(Error::InvalidData)?;
                            mapped.range(at, row)?.copy_from_slice(src);
                        }
                    }
                }
            }
        }
        Ok(())
    })();
    ctx.unmap(mem);
    filled?;

    let mut t = p.texture;
    t.image = o.image;
    ctx.begin()?;
    if p.texture.level_count > p.upload_levels {
        ctx.generate_mipmaps(&t, p.filter, vk::ImageLayout::PREINITIALIZED);
    } else {
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: t.level_count,
            base_array_layer: 0,
            layer_count: t.layer_count,
        };
        ctx.set_image_layout(
            o.image,
            vk::ImageLayout::PREINITIALIZED,
            t.image_layout,
            range,
        );
    }
    ctx.submit(o, false)
}

/// Port of `ktxTexture_VkUploadEx_WithSuballocator`.
///
/// # Safety
/// See [`VulkanDevice::upload`].
unsafe fn upload(
    dev: &mut VulkanDevice,
    src: &(impl TextureSource + ?Sized),
    options: &VulkanUploadOptions,
    sub: Option<&mut dyn SubAllocator>,
) -> Result<VulkanTexture> {
    let fns = Fns::new(&dev.functions)?;
    let mut ctx = Ctx { dev, fns, sub };
    let p = plan(&ctx, src, options)?;
    let mut o = Objects::default();
    let r = if options.tiling == vk::ImageTiling::OPTIMAL {
        upload_optimal(&mut ctx, &mut o, src, &p)
    } else {
        upload_linear(&mut ctx, &mut o, src, &p)
    };
    ctx.release(&mut o, r.is_ok());
    r?;
    let mut t = p.texture;
    t.image = o.image;
    match o.image_memory {
        Some(Mem::Device(m)) => t.device_memory = m,
        Some(Mem::Sub(id)) => {
            t.allocation_id = id;
            t.vk_free_memory = None;
        }
        None => {}
    }
    Ok(t)
}
