# Encode

- `imgio::encode(src: &Surface, format, &EncodeOptions { quality }, dst)` / `encode_to_buffer`,
  `can_encode`. Source: any uncompressed format (converted to RGBA8 first, labels only).
  One row of blocks per work unit (`parallel::try_for_each_unit`); edge blocks read clamped
  pixels.
- `Quality` (Fastest, Fast, Default, High, Best): each encoder maps it to its settings.
- BC1: basisu `encode_bc1` (transcoder real-time encoder, 4-color opaque blocks): Fastest /
  Fast = flags 0, Default = high quality, High / Best = higher quality.
- BC3 = `encode_bc4`(A) + BC1; BC4 = `encode_bc4`(R); BC5 = R + G; BC2 = explicit alpha
  `(a * 15 + 127) / 255` + BC1.
- Golden test `tests/encode.rs` vs `testdata/encode/golden.txt` (`tools/basisu-golden/encode.cpp`,
  C++ basisu encoders with the same edge clamping) on `testdata/images/raw` (raw RGBA8 copies
  written by `scripts/gen_images.py`): BC1 x 3 qualities, BC3 x 3, BC4, BC5 on 3 images, all
  bit-exact. Round trips (PSNR) for BC1/2/3/4.

## Audit

- 2026-11-21: API + BC1-BC5 encoders (basisu real-time), bit-exact with C++ (24 cases).
