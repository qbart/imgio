# GPU upload

`imgio::gpu`: texture sources (always built) and the two opt-in bindings, the only non-Rust
code imgio talks to. Ported from KTX-Software `vkloader.c` / `glloader.c` via ktx-rs
(`src/vkloader.rs`, `src/glloader.rs`, `src/glupload.rs`, `src/vk2gl.rs`).

## Sources

- `TextureSource`: `format() -> Result<Format>`, `width`, `height`, `depth` (default 1),
  `level_count`, `layer_count`, `face_count`, `is_array` (default layers > 1),
  `row_alignment` (default 1), `image_data(level, layer, face)`. A 3D image holds every slice
  of its level. Rows (block rows) are padded to `row_alignment`; images are tightly packed.
- Implemented for `ktx2::Ktx2File` (plain payloads; Basis payloads -> `Unsupported`, transcode
  first), `dds::DdsFile`, and `TextureData<'a>` (caller data: `new(format, w, h)`,
  `with_layers`, `with_cube`, `with_depth`, `with_row_alignment`, `push_level(&[u8])` /
  `push_owned_level(Buffer)`; one entry per level in the KTX2 level layout, size checked).
- Shape rules (`InvalidArgument`): non-zero size, faces 1 or 6 (square, not 3D), layers > 1
  only for arrays, 3D without layers / faces, levels <= full chain, alignment 1/2/4/8.
- `gl_format(Format) -> Option<GlFormat { internal_format, base_internal_format, format, ty }>`:
  `vk2gl.h` entries for every imgio format, plus ETC1 (`GL_ETC1_RGB8_OES`), ATC, FXT1, PVRTC.
  BC1 -> RGBA DXT1; ASTC HDR uses the LDR enum.

## Vulkan (`vulkan` = `ash` + `libloading`)

- `VulkanDevice::new(instance, physical_device, device, queue, cmd_pool, allocator,
  Option<&VulkanFunctions>)` (`ktxVulkanDeviceInfo_ConstructEx`): commands missing from
  `VulkanFunctions` are loaded through `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr`, or from
  the process (`load_vulkan_function`, `dlopen(NULL)` / `vulkan-1.dll`) without an instance;
  allocates the upload command buffer. `destroy(self)` frees it.
- `upload(src, &VulkanUploadOptions { tiling, usage, final_layout, generate_mipmaps })` ->
  `VulkanTexture { image, image_format, image_layout, device_memory, allocation_id, view_type,
  width, height, depth, level_count, layer_count }`; `destroy(device, allocator)`.
  Default options = `ktxTexture_VkUpload` (optimal, sampled, shader read-only).
- Checks as C: format via `ktx2::vk_format` (`Unsupported` for ATC / FXT1), image format
  properties (layers, levels), blit features + linear filter for mipmaps.
- Optimal: host-visible staging buffer, one region per level covering all layer-faces, region
  offsets aligned to lcm(block bytes, 4); images copied in bulk, padded rows one by one
  (`optimalTilingPadCallback`); barriers UNDEFINED -> TRANSFER_DST -> final, fence wait.
- Linear: PREINITIALIZED host-visible image, written at `vkGetImageSubresourceLayout` offsets:
  bulk when row / depth pitches match, row by row otherwise; queue wait idle.
- `generate_mipmaps`: level 0 uploaded, the full chain blitted (`generateMipmaps`).
- Suballocator: `unsafe trait SubAllocator { alloc_mem, bind_buffer, bind_image, map, unmap,
  free }` (`ktxVulkanTexture_subAllocatorCallbacks`); `upload_with_suballocator`,
  `destroy_with_suballocator`. More than one page -> `Unsupported`.
- Errors: `Unsupported` (missing command, format, limits, memory type), `OutOfMemory`,
  `Gpu(VkResult)`. Objects created so far are released on failure.

## OpenGL (`opengl` = `libloading`)

- `GlContext::load(&mut dyn FnMut(&CStr) -> *const c_void)` (`ktxLoadOpenGL`) or
  `load_from_process()` (`dlopen(NULL)`; Windows: loaded GLES / opengl32 + wgl). Required:
  the commands of every GL version; optional: `glGetStringi`, `glGenerateMipmap`,
  `glTexImage3D`, `glCompressedTexImage3D`. Commands and capabilities are per `GlContext`.
- `upload(src, &GlUploadOptions { texture, generate_mipmaps, decode_fallback })` ->
  `GlTexture { name, target, decoded }` (`ktxTexture_GLUpload`): capabilities discovered at
  the first upload (`discoverContextCapabilities`: ES / core / compat, sized formats, sRGB,
  R16, cube map arrays, max level); targets 2D, 2D array, cube (per face target), cube array
  (`layers * 6` layer-faces), 3D; `GL_UNPACK_ALIGNMENT` = source row alignment, restored;
  `GL_TEXTURE_MAX_LEVEL`; `glGenerateMipmap` or `GL_GENERATE_MIPMAP`; generated names deleted
  on failure; unsized base format without sized-format support.
- Fallback (`GlDecodeFallback::{Never, Etc (default, as C), Always}`): a rejected compressed
  format (`INVALID_ENUM` / `INVALID_VALUE`) is decoded with `imgio::decode` and uploaded
  uncompressed; 16-bit EAC results narrowed to 8 bit without R16 support, sRGB -> linear
  without sRGB support.
- Errors: `Gpu(gl_error)`, `Unsupported` (no GL format, missing command, target), source errors.

## Not ported

1D / 1D array textures (imgio sources are 2D or 3D), KTX1 `GL_LUMINANCE`/`ALPHA` legacy paths,
the WebGL `GlCommands` split of ktx-rs, sparse suballocations (as C).

## Tests

`tests/gpu_vulkan.rs` (lavapipe preferred), `tests/gpu_gl.rs` (Mesa surfaceless EGL, desktop
core + ES 3.0 from process symbols); both skip without a device / context. `make test-gpu`.
Uploads are read back (image -> buffer copy, mapped linear memory, `glGetTexImage`,
`glGetCompressedTexImage`, framebuffer `glReadPixels`) and compared with the sources:
all shapes, KTX2 / DDS files (BC1/3/5/7, RGBA8, RGBA16F, R8, 565, zstd), padded rows,
generated mipmaps (array layers keep their color), suballocator, ETC1 / PVRTC / ATC software
fallback, user texture names, error paths.

## Audit

- 2026-11-21: Vulkan and OpenGL upload ported from ktx-rs; deviations from C: every
  `VkResult` checked with cleanup, mipmap blit dst layer count = image layers (C: 1, invalid
  for arrays), only level 0 uploaded when generating mipmaps, linear copies honour driver
  pitches, cube map arrays uploaded with `layers * 6`, fallback extended to every decodable
  format on request, no process-global GL state.
