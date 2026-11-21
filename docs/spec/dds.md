# DDS

`imgio::dds` (own implementation of the Microsoft DDS layout).

- `DdsFile::parse(&[u8])`: `format`, `width`, `height`, `level_count`, `layer_count`,
  `face_count`, `is_dx10`, `image_data(level, layer, face)`. Data order: every array element /
  cube face with its whole mip chain; images tightly packed (`Format::surface_bytes`).
- DX10 header: DXGI <-> `Format` (`dxgi_format`, `format_from_dxgi`; typeless BC read as unorm;
  DXGI B5G6R5 = `Rgb565`), TEXTURE2D only, `TEXTURECUBE` misc flag, array size.
- Legacy header: FourCC DXT1, DXT2/3, DXT4/5, ATI1/BC4U, BC4S, ATI2/BC5U, BC5S, D3DFMT 36 / 113
  / 116 (RGBA16, RGBA16F, RGBA32F); RGB / luminance masks for RGBA8, R8, RG8, RG16, 565 (both
  orders), RGBA4444. Cubemaps need all six faces. Volume textures, BGRA layouts and other
  formats without an imgio `Format`: `Unsupported`.
- Volume textures: legacy `DDSD_DEPTH` + `DDSCAPS2_VOLUME`, DX10 `TEXTURE3D` (array size 1);
  every level's depth slices back to back (level-major); `depth()`, `level_depth(level)`,
  `image_data` returns all slices of the level; `DdsDesc::depth` (0 = 2D) for the writer.
- `write(&DdsDesc, levels, &DdsWriteOptions { legacy })`: same `levels` input as
  `ktx2::write` (level-major, layers then faces), reordered to the DDS layout. DX10 header as
  basisu `build_dx10_header` (flags, pitch / linear size of level 0, caps, cube caps2, array
  size >= 1); `legacy: true` writes the pre-DX10 pixel format for formats that have one
  (not for arrays).
- Tests `tests/dds.rs`: 11 files written by `basisu -dds` (BC1/2/3/4/5/7, cube + mips, 2D
  array, RGBA8 / sRGB, R8, RG8, 565; `testdata/dds`) parse and are rewritten byte for byte;
  BGRA8 rejected; legacy header round trips for 15 formats (plain + cube).

## Audit

- 2026-11-21: reader + writer; byte-identical with basisu v2.50 `-dds` output.
- 2026-11-21: volume textures (DX10 and legacy headers), round-trip tested.
