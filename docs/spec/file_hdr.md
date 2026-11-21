# Radiance HDR (RGBE)

`imgio::file::hdr` (own implementation).

- Header: `#?` program line, variables until the blank line: `FORMAT=32-bit_rle_rgbe` /
  `32-bit_rle_xyze` (others `Unsupported`), `EXPOSURE=` (product reported in
  `HdrImage::exposure`, not applied, as ffmpeg / imagecodecs), others ignored.
- Resolution string: all eight forms (`-Y N +X M` standard, `+Y` bottom-up, `-X`
  right-to-left, X-major `±X M ±Y N` = column scanlines). Radiance Y grows upwards.
- Scanlines: new adaptive RLE (`2 2 hi lo`, 4 planes of runs (>128) / literals, width
  8..=32767), else flat or old-style RLE (`1 1 1 n`: repeat previous pixel `n << shift`,
  shift += 8 per consecutive record, clipped to the scanline; previous pixel carries across
  scanlines). Scanlines are buffered as decoded (no allocation before the data exists).
- RGBE -> float: e == 0 -> 0, else `m * 2^(e - 136)`, exact in f32 (ffmpeg `hdrdec`,
  imagecodecs / Walter `rgbe2float`; no +0.5). Output `Rgba32Float` (alpha 1): every RGBE
  value fits f32 exactly; half floats would clip above 65504 and round below 6e-5.
- Float -> RGBE (ffmpeg `hdrenc` `float2rgbe`): `v = max(r,g,b)`, `< 1e-32` -> 0; else
  `frexpf(v) = m * 2^e`, channels `clamp(trunc(c * m * 256 / v))`, exponent
  `clamp(e + 128)`. NaN -> 0, +inf -> `f32::MAX` (undefined in the references).
- Encode: `#?RADIANCE`, FORMAT, optional EXPOSURE, `-Y H +X W`; new RLE scanlines with
  Walter's `RGBE_WriteBytes_RLE` run finder (min run 4) -> scanlines byte-identical to
  imagecodecs `rgbe_encode`; flat for width outside 8..=32767 or `rle: false`. Any
  uncompressed input via `convert` (sRGB linearized, gray replicated, alpha dropped).
- Limits: 2^28 pixels, 256 pixels per input byte.
- Tests `tests/file_hdr.rs`, data `testdata/file/hdr`: ffmpeg and imagecodecs (RLE / flat)
  files decoded bit-exactly vs both references; hand-built `#?RGBE` + EXPOSURE, old RLE
  (incl. a 280-pixel run, no reference: ffmpeg 6.1 `decompress` counts each repeat as 4
  pixels, imagecodecs has no old RLE), the 8 orientations, width 5, XYZE; encoder golden
  files (ffmpeg / imagecodecs decode = imagecodecs encode of the source; RLE body
  byte-identical); conversion edge cases; fuzzing.

## Audit

- 2026-11-22: decoder (all orientations, flat / old / new RLE) and RLE encoder, bit-exact
  with ffmpeg and imagecodecs.
