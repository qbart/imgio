# RenderMan zfile

`imgio::file::zfile` (own implementation; layout from the PRMan `zfile` display driver docs
and the OpenImageIO `zfile` plugin, read for format facts only).

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `0x2f0867ab`, writer's native byte order |
| 4 | 2 | width (u16) |
| 6 | 2 | height (u16) |
| 8 | 64 | world-to-screen matrix (16 f32) |
| 72 | 64 | world-to-camera matrix (16 f32) |
| 136 | 4 w h | depths (f32), rows top to bottom |

- The magic bytes give the byte order of every field (`ab 67 08 2f` little, `2f 08 67 ab`
  big endian); both decode. Note the on-disk order: world-to-screen first (OpenImageIO
  `ZfileHeader`).
- `decode` -> `R32Float`; `decode_zfile` -> `ZfileImage { image, world_to_screen,
  world_to_camera, big_endian }`. Trailing bytes ignored; short data `InvalidData`.
  Gzip-compressed zfiles (OpenImageIO option) are not recognized.
- `encode`: `R32Float` (`R16Float`, `R8Unorm`, `R16Unorm` widened exactly), matrices from
  `EncodeOptions` (identity by default, as OpenImageIO), little endian unless `big_endian`.
- Tests `tests/file_zfile.rs`: no reference tool; files built byte by byte in the test from
  the table (both byte orders, inf / large depths), encoder output equals those bytes,
  defaults, half input, fuzzing.

## Audit

- 2026-11-22: decoder (both byte orders) and encoder.
