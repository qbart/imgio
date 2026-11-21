// SPDX-FileCopyrightText: 2010-2026 Mark Callow, The Khronos Group Inc. and KTX-Software contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from ktx-rs `tests/vkloader.rs`; see NOTICE.md.
//! Vulkan upload (`imgio::gpu::vulkan`, port of `vkloader.c`): textures are uploaded with every
//! path (optimal tiling fast / padded, linear tiling fast / padded, mipmap generation,
//! suballocator) to the first CPU Vulkan device (lavapipe) or the first device, read back and
//! compared with the source images. Skipped when there is no Vulkan loader or device.
#![cfg(all(feature = "vulkan", not(target_family = "wasm")))]
#![allow(unsafe_code)]

mod gpu_common;

use std::collections::HashMap;
use std::ffi::c_void;

use gpu_common::{DIR, TestResult, level_data, levels, pad_rows, pattern, texture};
use imgio::dds::DdsFile;
use imgio::gpu::TextureSource;
use imgio::gpu::vulkan::{
    SubAllocator, VulkanDevice, VulkanFunctions, VulkanTexture, VulkanUploadOptions,
    load_vulkan_function, vk,
};
use imgio::ktx2::Ktx2File;
use imgio::{Error, Format};

struct Ctx {
    entry: ash::Entry,
    instance: ash::Instance,
    pd: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
}

impl Ctx {
    /// A device and queue, or `None` (test skipped) without a Vulkan loader or device.
    fn new() -> Option<Ctx> {
        let ctx = Self::try_new();
        if ctx.is_none() {
            eprintln!("no Vulkan device: test skipped");
        }
        ctx
    }

    fn try_new() -> Option<Ctx> {
        // SAFETY: plain Vulkan setup; every handle is destroyed in Drop.
        unsafe {
            let entry = ash::Entry::load().ok()?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
            let ci = vk::InstanceCreateInfo::default().application_info(&app);
            let instance = entry.create_instance(&ci, None).ok()?;
            let pds = instance.enumerate_physical_devices().unwrap_or_default();
            // Prefer the CPU device (lavapipe) for reproducibility.
            let Some(pd) = pds
                .iter()
                .copied()
                .find(|&p| {
                    instance.get_physical_device_properties(p).device_type
                        == vk::PhysicalDeviceType::CPU
                })
                .or_else(|| pds.first().copied())
            else {
                instance.destroy_instance(None);
                return None;
            };
            let Some(qfi) = instance
                .get_physical_device_queue_family_properties(pd)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            else {
                instance.destroy_instance(None);
                return None;
            };
            let qfi = qfi as u32;
            let prio = [1.0f32];
            let qci = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qfi)
                .queue_priorities(&prio);
            let features = vk::PhysicalDeviceFeatures::default().image_cube_array(true);
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&qci))
                .enabled_features(&features);
            let Ok(device) = instance.create_device(pd, &dci, None) else {
                instance.destroy_instance(None);
                return None;
            };
            let queue = device.get_device_queue(qfi, 0);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(qfi)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .ok()?;
            Some(Ctx {
                entry,
                instance,
                pd,
                device,
                queue,
                pool,
            })
        }
    }

    fn device_info(&self) -> imgio::Result<VulkanDevice> {
        let funcs = VulkanFunctions {
            vk_get_instance_proc_addr: Some(self.entry.static_fn().get_instance_proc_addr),
            ..Default::default()
        };
        // SAFETY: handles of this context.
        unsafe {
            VulkanDevice::new(
                self.instance.handle(),
                self.pd,
                self.device.handle(),
                self.queue,
                self.pool,
                std::ptr::null(),
                Some(&funcs),
            )
        }
    }

    fn memory_type(&self, bits: u32, props: vk::MemoryPropertyFlags) -> TestResult<u32> {
        // SAFETY: valid physical device.
        let mp = unsafe { self.instance.get_physical_device_memory_properties(self.pd) };
        (0..mp.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0 && mp.memory_types[i as usize].property_flags.contains(props)
            })
            .ok_or_else(|| "no memory type".into())
    }

    /// Copies all layers of `level` of an image in TRANSFER_SRC_OPTIMAL to the host.
    fn read_level(&self, vt: &VulkanTexture, level: u32, size: usize) -> TestResult<Vec<u8>> {
        // SAFETY: objects of this context, destroyed before returning.
        unsafe {
            let d = &self.device;
            let buf = d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )?;
            let req = d.get_buffer_memory_requirements(buf);
            let mem = d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(self.memory_type(
                        req.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )?),
                None,
            )?;
            d.bind_buffer_memory(buf, mem, 0)?;
            let cb = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .command_buffer_count(1),
            )?[0];
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())?;
            let dim = |v: u32| (v >> level).max(1);
            let region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level,
                    base_array_layer: 0,
                    layer_count: vt.layer_count,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: dim(vt.width),
                    height: dim(vt.height),
                    depth: dim(vt.depth),
                },
            };
            d.cmd_copy_image_to_buffer(
                cb,
                vt.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buf,
                &[region],
            );
            d.end_command_buffer(cb)?;
            let cbs = [cb];
            d.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&cbs)],
                vk::Fence::null(),
            )?;
            d.queue_wait_idle(self.queue)?;
            let p = d.map_memory(mem, 0, size as u64, vk::MemoryMapFlags::empty())?;
            let out = std::slice::from_raw_parts(p.cast::<u8>(), size).to_vec();
            d.unmap_memory(mem);
            d.free_command_buffers(self.pool, &cbs);
            d.destroy_buffer(buf, None);
            d.free_memory(mem, None);
            Ok(out)
        }
    }

    /// Reads `rows` rows of `row` bytes of level 0, layer 0 of a linear image.
    fn read_linear(&self, vt: &VulkanTexture, row: usize, rows: usize) -> TestResult<Vec<u8>> {
        // SAFETY: linear image of this context with host-visible memory.
        unsafe {
            let layout = self.device.get_image_subresource_layout(
                vt.image,
                vk::ImageSubresource {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    array_layer: 0,
                },
            );
            let p = self
                .device
                .map_memory(
                    vt.device_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )?
                .cast::<u8>();
            let mut out = Vec::new();
            for r in 0..rows {
                let at = p.add(layout.offset as usize + r * layout.row_pitch as usize);
                out.extend_from_slice(std::slice::from_raw_parts(at, row));
            }
            self.device.unmap_memory(vt.device_memory);
            Ok(out)
        }
    }

    fn destroy(&self, mut vt: VulkanTexture) {
        // SAFETY: created on this device, no longer in use (uploads wait for the queue).
        unsafe { vt.destroy(self.device.handle(), std::ptr::null()) };
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        // SAFETY: every object of this context was destroyed by the tests.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn readback() -> VulkanUploadOptions {
    VulkanUploadOptions {
        usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
        final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ..Default::default()
    }
}

/// Uploads with optimal tiling to TRANSFER_SRC_OPTIMAL and compares every level.
fn upload_and_compare(
    ctx: &Ctx,
    vdi: &mut VulkanDevice,
    src: &dyn TextureSource,
    expected: &[Vec<u8>],
) -> TestResult<VulkanTexture> {
    // SAFETY: device of `ctx`, not used elsewhere.
    let vt = unsafe { vdi.upload(src, &readback()) }?;
    assert_eq!(vt.level_count as usize, expected.len());
    for (level, want) in expected.iter().enumerate() {
        let got = ctx.read_level(&vt, level as u32, want.len())?;
        assert!(got == *want, "level {level} differs");
    }
    Ok(vt)
}

#[test]
fn upload_shapes() -> TestResult {
    let Some(ctx) = Ctx::new() else {
        return Ok(());
    };
    let mut vdi = ctx.device_info()?;
    let f = Format::Rgba8Unorm;
    // 2D with mips, cubemap, array, cube array, 3D.
    let cases = [
        ((16, 8, 1), 4, 1, false, vk::ImageViewType::TYPE_2D),
        ((8, 8, 1), 2, 1, true, vk::ImageViewType::CUBE),
        ((8, 4, 1), 2, 3, false, vk::ImageViewType::TYPE_2D_ARRAY),
        ((4, 4, 1), 3, 2, true, vk::ImageViewType::CUBE_ARRAY),
        ((4, 4, 4), 3, 1, false, vk::ImageViewType::TYPE_3D),
    ];
    for (i, (dims, nlevels, layers, cube, view)) in cases.into_iter().enumerate() {
        let images = layers * if cube { 6 } else { 1 };
        let data = level_data(f, dims, nlevels, images, i as u32 * 10)?;
        let t = texture(f, dims, layers, cube, &data)?;
        let vt = upload_and_compare(&ctx, &mut vdi, &t, &data)?;
        assert_eq!(vt.view_type, view);
        assert_eq!(vt.layer_count, images);
        assert_eq!(vt.image_format, vk::Format::R8G8B8A8_UNORM);
        ctx.destroy(vt);
    }
    // SAFETY: device of `ctx`.
    unsafe { vdi.destroy() };
    Ok(())
}

/// Plain KTX2 (raw and zstd) and DDS files, compressed and uncompressed.
#[test]
fn upload_files() -> TestResult {
    let Some(ctx) = Ctx::new() else {
        return Ok(());
    };
    let mut vdi = ctx.device_info()?;
    for name in [
        "rgba8_mips",
        "rgba8_array_zstd19",
        "bc7_mips",
        "bc1_array",
        "rgba16f_cube",
        "astc6x5_cube_array",
    ] {
        let data = std::fs::read(format!("{DIR}ktx2/plain/{name}.ktx2"))?;
        let f = Ktx2File::parse(&data)?;
        let want = levels(&f)?;
        match upload_and_compare(&ctx, &mut vdi, &f, &want) {
            Ok(vt) => ctx.destroy(vt),
            // Not every device has ASTC.
            Err(e) if name.starts_with("astc") => eprintln!("{name}: {e}"),
            Err(e) => return Err(format!("{name}: {e}").into()),
        }
    }
    for name in ["bc3_cube_mips", "bc5_array", "r8", "rgb565", "rgba8_mips"] {
        let data = std::fs::read(format!("{DIR}dds/{name}.dds"))?;
        let f = DdsFile::parse(&data)?;
        let want = levels(&f)?;
        let vt =
            upload_and_compare(&ctx, &mut vdi, &f, &want).map_err(|e| format!("{name}: {e}"))?;
        ctx.destroy(vt);
    }
    // SAFETY: device of `ctx`.
    unsafe { vdi.destroy() };
    Ok(())
}

/// R8 5x3 with rows padded to 4 bytes: the staging copy strips the padding
/// (`optimalTilingPadCallback`).
#[test]
fn upload_padded_rows() -> TestResult {
    let Some(ctx) = Ctx::new() else {
        return Ok(());
    };
    let mut vdi = ctx.device_info()?;
    let tight = [pattern(15, 7), pattern(2, 8)];
    let padded = [pad_rows(&tight[0], 5), pad_rows(&tight[1], 2)];
    let mut t = imgio::gpu::TextureData::new(Format::R8Unorm, 5, 3).with_row_alignment(4);
    for l in &padded {
        t.push_level(l)?;
    }
    let vt = upload_and_compare(&ctx, &mut vdi, &t, &tight)?;
    ctx.destroy(vt);
    // SAFETY: device of `ctx`.
    unsafe { vdi.destroy() };
    Ok(())
}

#[test]
fn upload_linear_tiling() -> TestResult {
    let Some(ctx) = Ctx::new() else {
        return Ok(());
    };
    let mut vdi = ctx.device_info()?;
    let linear = VulkanUploadOptions {
        tiling: vk::ImageTiling::LINEAR,
        usage: vk::ImageUsageFlags::SAMPLED,
        final_layout: vk::ImageLayout::GENERAL,
        generate_mipmaps: false,
    };
    // Fast path when the driver's row pitch is tight, row copies otherwise.
    let data = [pattern(8 * 4 * 4, 3)];
    let t = texture(Format::Rgba8Unorm, (8, 4, 1), 1, false, &data)?;
    // SAFETY: device of `ctx`.
    let vt = unsafe { vdi.upload(&t, &linear) }?;
    assert_eq!(vt.image_layout, vk::ImageLayout::GENERAL);
    assert_eq!(ctx.read_linear(&vt, 32, 4)?, data[0]);
    ctx.destroy(vt);

    // Padded source rows (R8 5x3, padded to 8).
    let tight = pattern(15, 3);
    let padded = pad_rows(&tight, 5);
    let mut t = imgio::gpu::TextureData::new(Format::R8Unorm, 5, 3).with_row_alignment(4);
    t.push_level(&padded)?;
    // SAFETY: device of `ctx`.
    let vt = unsafe { vdi.upload(&t, &linear) }?;
    assert_eq!(ctx.read_linear(&vt, 5, 3)?, tight);
    ctx.destroy(vt);
    // SAFETY: device of `ctx`.
    unsafe { vdi.destroy() };
    Ok(())
}

#[test]
fn upload_generate_mipmaps() -> TestResult {
    let Some(ctx) = Ctx::new() else {
        return Ok(());
    };
    let mut vdi = ctx.device_info()?;
    let options = VulkanUploadOptions {
        generate_mipmaps: true,
        ..readback()
    };
    let level0 = pattern(16 * 4 * 4, 5);
    let t = texture(
        Format::Rgba8Unorm,
        (16, 4, 1),
        1,
        false,
        std::slice::from_ref(&level0),
    )?;
    // SAFETY: device of `ctx`.
    let vt = unsafe { vdi.upload(&t, &options) }?;
    // floor(log2(16)) + 1 levels, blitted from level 0.
    assert_eq!(vt.level_count, 5);
    assert_eq!(ctx.read_level(&vt, 0, level0.len())?, level0);
    ctx.destroy(vt);

    // A 2-layer array of constant layers: every level of each layer keeps its color
    // (the blits cover every layer).
    let colors = [[10u8, 20, 30, 255], [200, 100, 50, 128]];
    let data: Vec<u8> = colors.iter().flat_map(|c| c.repeat(8 * 8)).collect();
    let data = [data];
    let t = texture(Format::Rgba8Unorm, (8, 8, 1), 2, false, &data)?;
    // SAFETY: device of `ctx`.
    let vt = unsafe { vdi.upload(&t, &options) }?;
    assert_eq!(vt.level_count, 4);
    for level in 0..4u32 {
        let n = ((8 >> level) * (8 >> level)) as usize;
        let got = ctx.read_level(&vt, level, n * 4 * 2)?;
        let want: Vec<u8> = colors.iter().flat_map(|c| c.repeat(n)).collect();
        assert_eq!(got, want, "level {level}");
    }
    ctx.destroy(vt);
    // Compressed formats cannot be blitted.
    let bc = [vec![0u8; 16]];
    let t = texture(Format::Bc7Unorm, (4, 4, 1), 1, false, &bc)?;
    // SAFETY: device of `ctx`.
    let r = unsafe { vdi.upload(&t, &options) };
    assert_eq!(r.map(|_| ()), Err(Error::Unsupported));
    // SAFETY: device of `ctx`.
    unsafe { vdi.destroy() };
    Ok(())
}

/// A suballocator on plain `vkAllocateMemory`, counting frees.
struct SubAlloc {
    device: ash::Device,
    next_id: u64,
    allocs: HashMap<u64, vk::DeviceMemory>,
    freed: u32,
}

// SAFETY: maps whole allocations of the upload's device with vkMapMemory.
unsafe impl SubAllocator for SubAlloc {
    fn alloc_mem(
        &mut self,
        info: &vk::MemoryAllocateInfo<'_>,
        _requirements: &vk::MemoryRequirements,
    ) -> Result<(u64, u64), vk::Result> {
        // SAFETY: valid device and allocation info.
        let mem = unsafe { self.device.allocate_memory(info, None) }?;
        self.next_id += 1;
        self.allocs.insert(self.next_id, mem);
        Ok((self.next_id, 1))
    }
    fn bind_buffer(&mut self, buffer: vk::Buffer, id: u64) -> vk::Result {
        let Some(&mem) = self.allocs.get(&id) else {
            return vk::Result::ERROR_UNKNOWN;
        };
        // SAFETY: buffer and memory of this device.
        unsafe { self.device.bind_buffer_memory(buffer, mem, 0) }
            .err()
            .unwrap_or(vk::Result::SUCCESS)
    }
    fn bind_image(&mut self, image: vk::Image, id: u64) -> vk::Result {
        let Some(&mem) = self.allocs.get(&id) else {
            return vk::Result::ERROR_UNKNOWN;
        };
        // SAFETY: image and memory of this device.
        unsafe { self.device.bind_image_memory(image, mem, 0) }
            .err()
            .unwrap_or(vk::Result::SUCCESS)
    }
    fn map(&mut self, id: u64, _page: u64) -> Result<(*mut c_void, vk::DeviceSize), vk::Result> {
        let &mem = self.allocs.get(&id).ok_or(vk::Result::ERROR_UNKNOWN)?;
        // SAFETY: host-visible memory of this device.
        let p = unsafe {
            self.device
                .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }?;
        Ok((p, vk::WHOLE_SIZE))
    }
    fn unmap(&mut self, id: u64, _page: u64) {
        if let Some(&mem) = self.allocs.get(&id) {
            // SAFETY: mapped by `map`.
            unsafe { self.device.unmap_memory(mem) };
        }
    }
    fn free(&mut self, id: u64) {
        if let Some(mem) = self.allocs.remove(&id) {
            // SAFETY: no longer bound to a live object.
            unsafe { self.device.free_memory(mem, None) };
            self.freed += 1;
        }
    }
}

#[test]
fn upload_with_suballocator() -> TestResult {
    let Some(ctx) = Ctx::new() else {
        return Ok(());
    };
    let mut sub = SubAlloc {
        device: ctx.device.clone(),
        next_id: 0,
        allocs: HashMap::new(),
        freed: 0,
    };
    let data = level_data(Format::Rgba8Unorm, (8, 8, 1), 2, 1, 9)?;
    let t = texture(Format::Rgba8Unorm, (8, 8, 1), 1, false, &data)?;
    let mut vdi = ctx.device_info()?;
    // SAFETY: device of `ctx`; the suballocator allocates on it.
    let mut vt = unsafe { vdi.upload_with_suballocator(&t, &readback(), &mut sub) }?;
    assert_ne!(vt.allocation_id, 0);
    assert_eq!(vt.device_memory, vk::DeviceMemory::null());
    // The staging allocation was freed through the suballocator.
    assert_eq!((sub.freed, sub.allocs.len()), (1, 1));
    for (level, w) in data.iter().enumerate() {
        assert!(
            ctx.read_level(&vt, level as u32, w.len())? == *w,
            "level {level}"
        );
    }
    // SAFETY: created on this device with `sub`.
    unsafe { vt.destroy_with_suballocator(ctx.device.handle(), std::ptr::null(), &mut sub) };
    assert_eq!((sub.freed, sub.allocs.len()), (2, 0));

    // Linear tiling through the suballocator.
    let linear = VulkanUploadOptions {
        tiling: vk::ImageTiling::LINEAR,
        final_layout: vk::ImageLayout::GENERAL,
        ..Default::default()
    };
    let one = [data[0].clone()];
    let t = texture(Format::Rgba8Unorm, (8, 8, 1), 1, false, &one)?;
    // SAFETY: as above.
    let mut vt = unsafe { vdi.upload_with_suballocator(&t, &linear, &mut sub) }?;
    assert_eq!(sub.allocs.len(), 1);
    // SAFETY: as above.
    unsafe { vt.destroy_with_suballocator(ctx.device.handle(), std::ptr::null(), &mut sub) };
    assert!(sub.allocs.is_empty());
    // SAFETY: device of `ctx`.
    unsafe { vdi.destroy() };
    Ok(())
}

#[test]
fn upload_errors() -> TestResult {
    let Some(ctx) = Ctx::new() else {
        return Ok(());
    };
    let mut vdi = ctx.device_info()?;
    let options = VulkanUploadOptions::default();
    // Basis Universal payloads must be transcoded first.
    let data = std::fs::read(format!("{DIR}ktx2/etc1s_rgba.ktx2"))?;
    let basis = Ktx2File::parse(&data)?;
    // SAFETY (all uploads): device of `ctx`.
    let r = unsafe { vdi.upload(&basis, &options) };
    assert_eq!(r.map(|_| ()), Err(Error::Unsupported));
    // No VkFormat (ATC).
    let atc = [vec![0u8; 8]];
    let t = texture(Format::AtcRgb, (4, 4, 1), 1, false, &atc)?;
    let r = unsafe { vdi.upload(&t, &options) };
    assert_eq!(r.map(|_| ()), Err(Error::Unsupported));
    // An unset function pointer is reported instead of called.
    let rgba = [pattern(64, 1)];
    let t = texture(Format::Rgba8Unorm, (4, 4, 1), 1, false, &rgba)?;
    let saved = vdi.functions;
    vdi.functions.vk_create_image = None;
    let r = unsafe { vdi.upload(&t, &options) };
    assert_eq!(r.map(|_| ()), Err(Error::Unsupported));
    vdi.functions = saved;
    let vt = unsafe { vdi.upload(&t, &options) }?;
    assert_eq!(vt.view_type, vk::ImageViewType::TYPE_2D);
    assert_eq!(vt.image_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    ctx.destroy(vt);
    // SAFETY: device of `ctx`.
    unsafe { vdi.destroy() };
    Ok(())
}

/// `ktxVulkanDeviceInfo_Create` without an instance or functions: the commands are looked up
/// in the process, so the Vulkan loader must be globally loaded, as when an application links
/// it.
#[cfg(unix)]
#[test]
fn device_from_process_symbols() -> TestResult {
    let Some(ctx) = Ctx::new() else {
        return Ok(());
    };
    // SAFETY: loading the system Vulkan loader.
    let _global = unsafe {
        libloading::os::unix::Library::open(
            Some("libvulkan.so.1"),
            libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
        )
    }?;
    assert!(load_vulkan_function(c"vkGetInstanceProcAddr").is_some());
    assert!(load_vulkan_function(c"vkNoSuchCommandImgio").is_none());
    // SAFETY: handles of `ctx`.
    let mut vdi = unsafe {
        VulkanDevice::new(
            vk::Instance::null(),
            ctx.pd,
            ctx.device.handle(),
            ctx.queue,
            ctx.pool,
            std::ptr::null(),
            None,
        )
    }?;
    assert!(vdi.functions.vk_cmd_blit_image.is_some());
    assert_ne!(vdi.cmd_buffer, vk::CommandBuffer::null());
    let data = [pattern(64, 2)];
    let t = texture(Format::Rgba8Unorm, (4, 4, 1), 1, false, &data)?;
    let vt = upload_and_compare(&ctx, &mut vdi, &t, &data)?;
    ctx.destroy(vt);
    // SAFETY: device of `ctx`.
    unsafe { vdi.destroy() };
    Ok(())
}
