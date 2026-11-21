# Texture builder

`imgio::texture::build(images, layers, faces, &BuildOptions { format, encode, mips, mip,
normal_map })` -> `Texture { format, width, height, layers, faces, levels }`.

- Images: `max(layers, 1) * faces` base surfaces (layer, then face), same size / format.
- Working format for the mips: RGBA32F for HDR / signed / float targets, RGBA8 with the sRGB
  label (so `mip` resamples in linear light) when the target or source is sRGB, RGBA16 for
  16-bit unorm targets, else RGBA8. Sources are converted labels-only (`convert`).
- `Mips::{None, Full, Count(n)}`; levels from `mip::generate_mips` (each from the base).
- `normal_map`: RGB as [-1, 1] vectors renormalized on every generated level.
- Every level encoded with `imgio::encode` (compressed targets) or `convert` (uncompressed).
- `Texture::to_ktx2(&WriteOptions)`, `to_dds(&DdsWriteOptions)`, `to_ktx1(key_values)`.
- Tests: BC7 sRGB chain written to every container; level 1 equals encoding the sRGB-correct
  mip directly; cubemap of normals keeps unit length on every mip; shape errors.

## Audit

- 2026-11-21: builder.
