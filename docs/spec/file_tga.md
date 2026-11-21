# TGA

`imgio::file::tga` (own implementation).

- Types 1 / 9 (color-mapped: 8 / 16-bit indices, 15 / 16 / 24 / 32-bit map entries, first
  entry index), 2 / 10 (truecolor 15 / 16 / 24 / 32), 3 / 11 (gray 8, gray + alpha 16);
  raw or RLE. RLE is one packet stream for the whole image (packets may cross scanlines);
  at most 128 pixels per byte of RLE data.
- Origins: descriptor bit 4 right-to-left, bit 5 top-to-bottom (all four). Image ID field
  skipped and returned by `decode_tga` (`TgaImage { image, image_id, attribute_type }`).
- TGA 2.0 footer + extension area: attribute type (byte 494) 3 / 4 keep alpha, other values
  drop it. Without it: descriptor alpha bits > 0 keep alpha (A1R5G5B5 bit, 32-bit fourth
  byte, gray alpha byte); with 0 alpha bits, 32-bit pixels / map entries and 16-bit gray
  keep alpha only if some alpha byte is non-zero; 15/16-bit color has none.
- Output: `Rgb8Srgb` / `Rgba8Srgb`, gray `R8Unorm` / `Rg8Unorm`; 5-bit channels
  `round(v * 255 / 31)`.
- Encode: `R8Unorm` -> type 3 / 11, RGB -> 24-bit, RGBA -> 32-bit (alpha bits 8), top-left
  origin, `EncodeOptions { rle }` (packets per scanline, runs >= 2), TGA 2.0 footer. Inputs
  as BMP (`Rg8Unorm` -> RGBA, packed formats expanded, deeper `Unsupported`).
- `looks_like_tga` (detection): footer, or a plausible header.
- Tests `tests/file_tga.rs`, data `testdata/file/tga`: 58 files (Pillow L / LA / P / RGB /
  RGBA raw and RLE, both origins; ffmpeg bgr24 / bgra / rgb555 / gray raw and RLE;
  hand-built: color maps 15/16/24/32 with 8/16-bit indices and a map offset, raw / RLE across
  scanlines, 15 / 16-bit truecolor, four origins, image IDs, alpha-bit and extension
  attribute rules), exact vs Pillow / ffmpeg where they read the variant, else vs the source;
  encoder golden files decoded to the source by Pillow and ffmpeg; fuzzing.

## Audit

- 2026-11-22: decoder (types 1/2/3/9/10/11, origins, 2.0 footer / extension) and raw / RLE
  encoder.
