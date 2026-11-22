# Changelog

## Unreleased

- CLI: `imgio lut` (bake / convert LUTs) and the `--lut` image option.
- ICC device links: `color::apply_link`, `Profile::is_device_link`, `Lut3d::to_device_link`.
- Colour conversions with LUT profiles are byte identical with LittleCMS (sampled curves and
  CLUTs evaluated in 16-bit fixed point like LittleCMS); crate license `Apache-2.0 AND MIT`.
- LUTs (`imgio::lut`): `.cube` read / write (with Resolve shapers), trilinear / tetrahedral
  apply, baking from filters and colour transforms, HALD CLUT and strip images.
- `imgio` command line tool (feature `cli`): info, convert, texture, pack, atlas, ibl, compare.
- Colour management (`imgio::color`, default feature `color`): ICC v2 / v4 matrix and LUT
  profiles, built-in colour spaces, CICP, conversions matching LittleCMS; `file::colorimetry`
  and colour managed `file::decode_to`.
- Image based lighting (`imgio::ibl`): equirect <-> cubemap, SH / convolved irradiance, GGX
  prefiltered specular mip chain, BRDF LUT.
- Texture atlases (`imgio::atlas`): MaxRects / skyline / shelf packing, smallest page search,
  padding, border, extrusion, rotation, trimming, duplicates, multiple pages, JSON layout.
- ICO / CUR files (`imgio::file::ico`, own code): DIB and PNG entries, cursors, multi-size
  encoding; `FileFormat::Ico`.
- Quality metrics (`imgio::metrics`): MSE / PSNR / max / mean error and SSIM (matches
  scikit-image) per channel and aggregated, difference image.
- Channel packing (`imgio::pack`): any output channel from any channel of any input,
  constants, invert / scale / bias, size modes, ORM and mask map presets, `split`.

## 0.1.0 (2026-11-22)

First functional release.

- GPU texture encoders for BC1-BC7, ETC1 / ETC2 / EAC, ASTC (astcenc 6.0 port, LZ RDO), PVRTC1/2,
  ATC and FXT1; decoders for all of them (`imgio::encode`, `imgio::decode`).
- Basis Universal v2.50 encoder and transcoder: ETC1S, UASTC LDR / HDR, ASTC HDR 6x6, ASTC LDR,
  XUASTC LDR, XUBC7; byte identical with the C++ reference.
- Containers: KTX2 (raw / Zstd / Basis payloads), KTX1, DDS (legacy and DX10), `.astc`, `.basis`;
  2D, arrays, cubemaps and 3D textures.
- Uncompressed format conversion, sRGB-correct mip generation (`imgio::mip`), one-call texture
  builder (`imgio::texture::build`).
- Optional Vulkan / OpenGL upload (`vulkan`, `opengl` features).
- Image files (`imgio::file`): format detection and decode / encode on byte slices for BMP,
  TGA, Radiance HDR, RenderMan zfile (own code), PNG / APNG, GIF, WebP, JPEG (decode), OpenEXR,
  TIFF, JPEG XL, JPEG 2000 (decode), OpenVDB (decode, partial) and AVIF, each crate-backed
  format behind its own cargo feature (pure Rust crates only).
- Image resize (`imgio::resize`): every uncompressed format, sRGB-correct, premultiplied
  alpha, every resampler filter, exact / contain / cover / width / height fits, crop.
- Image filters (`imgio::filter`): flip / rotate / transpose, box / Gaussian blur, unsharp mask,
  exposure / brightness / contrast / gamma / saturation, grayscale, swizzle, premultiply,
  normal map from height, HDR tone mapping.
- `R16Float`, `R32Float`, `Rgb8*`, `Bgra8*` pixel formats.
