# Changelog

## Unreleased

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
