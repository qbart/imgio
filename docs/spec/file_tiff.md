# TIFF (`imgio::file::tiff`, feature `tiff`)

Decoding is own code (IFDs, strips / tiles, predictors, color conversion) on `weezl` (LZW),
`miniz_oxide` (Deflate) and `zune-jpeg` (JPEG); encoding uses the `tiff` crate.

- Decode: classic and BigTIFF, both byte orders, strips / tiles, planar 1 / 2, FillOrder 2,
  multi-page (`page_count`, `decode_page`; `decode` = page 0). None, LZW (also old-style),
  Deflate / AdobeDeflate, PackBits, new-style JPEG (with / without JPEGTables); predictors 2
  and 3. CCITT, old-style JPEG, ZSTD, WebP: `Unsupported`.
- Samples: unsigned 1-32 bit, signed 8 / 16 / 32, float 16 / 32 / 64 (64 -> f32).
- Photometric: gray (min-is-white inverted for integers), RGB, palette (16-bit colormaps to
  8 bits like libtiff), CMYK (libtiff `TIFFReadRGBA` formula), YCbCr (8-bit, any
  subsampling, libtiff tables). Alpha: first extra sample (associated alpha returned as
  stored, see `PageInfo::alpha`); unspecified extra samples dropped; Orientation ignored.
- Output: <= 8 bit -> `R8Unorm` / `Rg8Unorm` / `Rgb8Srgb` / `Rgba8Srgb`; 16 bit ->
  `R16Unorm` / `Rg16Unorm` / `Rgba16Unorm`; half -> `R16Float` / `Rgb16Float` /
  `Rgba16Float`; u32 (normalized), signed (/ max), f32 / f64 -> `R32Float` / `Rgba32Float`.
- Encode: 8 / 16-bit gray, gray+alpha, RGB, RGBA; half and f32 gray / RGB(A); BGRA swizzled;
  others -> `Rgba32Float`. None, LZW, Deflate, PackBits, optional predictor, BigTIFF option,
  multi-page (`encode_pages`).
- Allocation limit: decoded bytes <= max(input * 4096, 16 MiB), at most 4 GiB.
- Tests (`tests/file_tiff.rs`): 53 files from tifffile / imagecodecs (libtiff) and Pillow
  (`scripts/gen_file_testdata_exrtiff.py`); decodes exact against libtiff except JPEG-in-TIFF
  (max 4, mean < 0.5 vs libjpeg-turbo: IDCT / upsampling rounding); encode -> tifffile decode
  exact; page info; truncation and byte-flip loops.

## Audit

- 2026-11-22: decoder / encoder.
