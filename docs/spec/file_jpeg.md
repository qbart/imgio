# JPEG (`imgio::file::jpeg`, feature `jpeg`, decode only)

Crate `jpeg-decoder` 0.3 (pure Rust).

- Baseline / extended (SOF0 / SOF1), progressive (SOF2), restart markers, any subsampling
  (4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1), lossless (SOF3, 2-16 bits).
- Gray -> `R8Unorm` (lossless > 8 bits `R16Unorm`); YCbCr / RGB / CMYK / YCCK (Adobe APP14) ->
  `Rgb8Srgb` (lossless RGB > 8 bits `Rgba16Unorm`); CMYK -> RGB like Pillow
  (`R = (255 - K) - C * (255 - K) / 255`).
- 12-bit DCT, hierarchical and arithmetic-coded files: `Unsupported`.
- EXIF orientation as metadata (`decode_with_metadata`, `exif_orientation`); pixels not rotated.
- Not bit-exact with libjpeg-turbo (own integer IDCT, 20-bit YCbCr fixed point). Measured vs
  libjpeg-turbo 3.2 (islow, fancy upsampling): gray / YCCK max 1 (mean <= 0.05); CMYK, 4:2:0,
  4:1:1 max 2 (mean <= 0.12); other subsamplings / progressive max 3 (mean <= 0.25); lossless
  exact.
- Tests (`tests/file_jpeg.rs`): files from libjpeg-turbo (Pillow, imagecodecs) and ffmpeg;
  truncation / flip loops.

## Audit

- 2026-11-22: decoder.
