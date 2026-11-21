# BMP

`imgio::file::bmp` (own implementation).

- Headers: BITMAPCOREHEADER (12, OS/2 1.x, 3-byte palette entries), OS/2 2.x (16..64 bytes
  other than 40 / 52 / 56), BITMAPINFOHEADER (40), V2 (52, RGB masks), V3 (56, + alpha mask),
  V4 (108), V5 (124). Bit fields after a 40-byte header for `BI_BITFIELDS` (3 masks) /
  `BI_ALPHABITFIELDS` (4). Negative height = top-down.
- Depths: 1/2/4/8 palette (entries = `biClrUsed` or 2^bpp, clipped to the bytes before the
  pixel offset; out-of-range index = black), 16 (default 5:5:5), 24, 32 (default BGRX), any
  bit field masks for 16/24/32 (n-bit channel -> `round(v * 255 / (2^n - 1))`).
- RLE8 / RLE4: encoded runs, absolute runs (word aligned), end of line, delta, end of bitmap;
  pixels past the row end are dropped; a stream ending before the last row without end of
  bitmap is `InvalidData`.
- Output: all-gray palette -> `R8Unorm`, other palettes -> `Rgb8Srgb`; masks -> `Rgb8Srgb`,
  `Rgba8Srgb` when an alpha mask is set, or for 32-bit `BI_RGB` when any fourth byte is
  non-zero. RLE pixels never drawn are transparent: the image becomes `Rgba8Srgb` / `Rg8Unorm`
  with alpha 0 there (choice: browsers do the same; black would be indistinguishable).
- `BI_JPEG` / `BI_PNG` payloads go to `file::jpeg` / `file::png`; OS/2 Huffman 1D and RLE24:
  `Unsupported`.
- Limits: 2^28 pixels; RLE at most 512 pixels per compressed byte (no huge allocation from a
  small file). Uncompressed rows are expanded in parallel (`parallel::for_each_unit`).
- Encode (uncompressed, bottom-up, 72 DPI): `R8Unorm` -> 8-bit gray palette (decodes back to
  `R8Unorm`), RGB -> 24-bit INFOHEADER, RGBA -> 32-bit `BI_BITFIELDS` V4 header (alpha mask,
  `LCS_sRGB`). `Rg8Unorm` -> RGBA, `Bgra8*` / 565 / 4444 expanded; deeper formats
  `Unsupported`.
- Tests `tests/file_bmp.rs`, data `testdata/file/bmp` (`scripts/gen_file_testdata_own.py`):
  36 files (Pillow: 24/32-bit, P8, L8, 1-bit; ffmpeg: bgr24, bgra, rgb8, gray, monob, 565,
  555, 444; hand-built: OS/2 1.x / 2.x, 2/4-bit, top-down, RLE8 / RLE4 with and without
  holes, 5:5:5, 5:6:5, 4:4:4:4, BGRX / RGBX, ALPHABITFIELDS, V2 / V4 / V5, 10:10:10:2), exact
  vs Pillow / ffmpeg decodes where they read the variant (Pillow drops 32-bit BI_RGB alpha:
  color checked only), else vs the hand-built source; encoder output byte-identical to
  golden files that Pillow and ffmpeg decode to the source; truncation / byte-flip fuzzing.

## Audit

- 2026-11-22: decoder (all headers, depths, RLE, bit fields) and uncompressed encoder.
