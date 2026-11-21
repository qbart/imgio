# KTX 1

`imgio::ktx1` (own implementation of the KTX 1.1 layout).

- `Ktx1File::parse`: little / big endian (header and uncompressed elements by `glTypeSize`
  swapped), 2D / 3D with mips, arrays, cubemaps (3D: slices innermost; `image` returns all
  slices of a level; `Ktx1Desc::depth`, 0 = 2D, for the writer). `format`, `width`,
  `height`, `depth`, `level_depth`, `level_count`, `layer_count`, `face_count`, `is_srgb` (incl.
  PVRTC sRGB enums),
  `gl_internal_format`, `key_values`, `image(level, layer, face)` -> tightly packed texels.
- GL enums: compressed table (S3TC incl. RGB / RGBA / sRGB DXT1, RGTC, BPTC, ETC1, ETC2, EAC,
  ASTC LDR / sRGB (no HDR enum: ASTC reads as LDR), PVRTC1 / 2 incl. sRGB, ATC, FXT1);
  uncompressed via `gpu::gl_format` (sized internal format, or unsized format + type).
- Layout: per level `imageSize` (one face for non-array cubemaps, else all images), images
  layer then face, uncompressed rows padded to 4, cube faces and levels padded to 4. PVRTC1
  levels are stored with at least 2 x 2 blocks: the writer tiles small grids (PVRTC wraps),
  the reader crops and re-twiddles them.
- `write(&Ktx1Desc { format, w, h, layers, faces, levels, srgb }, levels, key_values)`:
  basisu `create_ktx_texture_file` conventions for compressed formats (`glType` /
  `glFormat` 0, `glTypeSize` 1, BC1 as RGB DXT1, base formats RGB / RGBA / RED / RG).
- Test `tests/ktx1.rs`: 16 files from `basisu -export_ktx` (BC1 / 3 / 4 / 5 / 6H / 7, ETC1 /
  2, EAC R11 / RG11, ASTC LDR / HDR 4x4, PVRTC1 with 1x1 mips, PVRTC2, mips, 2-layer array,
  cubemap with mips; `testdata/ktx1`): every image equals imgio's transcode of the source
  KTX2, rewrites are byte identical. Uncompressed round trip with row padding and key/values.

## Audit

- 2026-11-21: reader / writer, byte identical with basisu v2.50 `-export_ktx`.
- 2026-11-21: 3D textures (round trip).
