# PNG / APNG (`imgio::file::png`)

Backed by the `png` crate 0.18 (feature `png`, default on; pure Rust: `fdeflate`, `flate2` with
`miniz_oxide`, `crc32fast`). Without the feature every function returns `Unsupported`.

- Decode (`decode`, `decode_with`): every color type / bit depth (gray 1/2/4/8/16, gray + alpha
  8/16, RGB 8/16, RGBA 8/16, palette 1/2/4/8), Adam7, tRNS (palette alpha; gray / RGB color keys
  become an alpha channel). Formats: gray `R8Unorm` / `R16Unorm`, gray + alpha (or gray + key)
  `Rg8Unorm` / `Rg16Unorm`, RGB / palette `Rgb8Srgb`, RGBA / RGB + key / palette + tRNS
  `Rgba8Srgb`, 16-bit RGB(A) `Rgba16Unorm` (RGB: alpha 65535). 8-bit color is always labeled sRGB.
- `info`: IHDR fields, tRNS presence, acTL (`frames`, `plays`, `default_image_is_first_frame`),
  `ColorInfo` (sRGB intent, gAMA, cHRM, decompressed iCCP, cICP) and `ColorInfo::is_srgb` (sRGB
  chunk / cICP 1,13 / no color chunks; false for iCCP, other cICP, gAMA != 1/2.2).
- APNG (`decode_animation[_with]`): canvas starts transparent black; per frame: save the rect if
  dispose PREVIOUS, blend (SOURCE copy; OVER with the APNG spec sample formula: `u = a_s * M`,
  `v = (M - a_s) * a_d`, `c = (c_s u + c_d v) / (u + v)`, `a = (u + v) / M`, M = 255 / 65535,
  truncating; source copied when `a_d = 0`), emit the canvas, then dispose (BACKGROUND clears the
  rect, PREVIOUS restores it; a first-frame PREVIOUS acts as BACKGROUND by construction). A
  default image without fcTL is decoded and dropped. Delay `(num, den)`, den 0 -> 100; `loops` =
  `num_plays`. png 0.18 writes Adam7 subframes with the IHDR row stride (handled).
- Limits: the IDAT image must fit `input_len * 1040` raw bytes (deflate ratio bound) before any
  allocation; `DecodeOptions::max_bytes` (default 1 GiB) bounds all frames together and the
  crate's buffers (`OutOfMemory`). Corrupt / truncated input -> `InvalidData`.
- Encode (`encode`, `encode_animation`): `R8Unorm`, `Rg8Unorm`, `Rgb8*`, `Rgba8*`, `R16Unorm`,
  `Rg16Unorm`, `Rgba16Unorm` native; `Bgra8*` swizzled, `Rgb565` / `Bgr565` / `Rgba4444` widened
  to 8-bit (`convert`); others `Unsupported`. Options: `Compression` (None, Fastest / Fast =
  fdeflate, Default = zlib 6, Best = 9, Level(n)), `Filter` (None, Sub, Up, Average, Paeth,
  Adaptive, MinEntropy), `palette` (8-bit RGB / RGBA with <= 256 colors over all frames ->
  PLTE + tRNS, translucent entries first, always a tRNS for RGBA input so it decodes as RGBA),
  `strip_opaque_alpha` (all frames opaque -> RGB / gray; the way to write 16-bit RGB),
  `srgb_chunk` (sRGB chunk for `*Srgb` input, default on). APNG: full-canvas frames, dispose
  NONE, blend SOURCE, delays reduced to 16-bit terms, `num_plays = loops`; frame 0 is the IDAT.
- Tests (`tests/file_png.rs`, `testdata/file/png`, `scripts/gen_file_testdata_pnggif.py`):
  66 files (script writer: every color type / depth, plain + Adam7, tRNS variants, color chunks,
  every dispose / blend op, hidden default image, palette / gray-alpha / RGB / RGBA16 APNG;
  Pillow, ffmpeg (Adam7 PNG, APNG) and imagecodecs (APNG) encoders). Expected hashes: libpng
  (imagecodecs `png_decode`) for still images, cross-checked with Pillow for 8-bit; APNG frames
  from ffmpeg (8-bit non-palette) or imagecodecs (palette, 16-bit, partial alpha), cross-checked
  against each other and Pillow where those are right. All exact. Encoder: round trips for
  every format x option set, palette / strip paths, APNG; 17 encoder outputs decoded exactly by
  Pillow / libpng (`--verify`, hashes in `encoded_verified.txt`). Corrupt input: truncations
  and byte flips (chunk CRCs repaired) of every file; size-limit checks.
- Reference quirks found: Pillow 13 fails on interlaced APNG frames after the first, blends
  OVER as a masked paste (wrong alpha for partial alpha), and does not restore PREVIOUS after an
  OVER frame (blends into its saved copy); it also compares 1/2/4-bit gray tRNS keys unscaled.
  ffmpeg blends palette APNG frames in index space and rounds OVER (+-1 vs the spec formula).

## Audit

- 2026-11-22: decode (all color types, Adam7, tRNS, APNG compositing), color info, encode
  (8/16-bit, palette, strip alpha, compression / filter options), APNG encode; exact vs libpng,
  ffmpeg, imagecodecs and Pillow.
