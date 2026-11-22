# OpenEXR (`imgio::file::exr`, feature `exr`)

Crate `exr` 1.74 (pure Rust; default `rayon` feature decompresses blocks in parallel).

- Decode: scanline / tiled, multi-part, multi-resolution (mip / rip), f16 / f32 / u32;
  NONE, RLE, ZIPS, ZIP, PIZ, PXR24, B44, B44A, DWAA, DWAB. HTJ2K, deep data and subsampled
  channels: `Unsupported`.
- `decode`: main layer of part 0 at full resolution in the display window. RGB(A) half ->
  `Rgb16Float` / `Rgba16Float`; any f32 / u32 -> `Rgba32Float`; Y / RY / BY -> RGB like
  OpenEXR `RgbaInputFile` (chromaticities, Rec. 709 default); single channel / Y ->
  `R16Float` / `R32Float`; u32 -> f32 (exact to 2^24; `decode_planes` gives raw u32).
  Display-window pixels outside the data window are 0 (alpha 1 if no alpha channel).
- `parts`, `decode_with` (part / layer / level / window), `decode_planes` (raw channels).
- Encode: one part from `Rgba16Float`, `Rgb16Float`, `Rgba32Float`, `R16Float` / `R32Float`
  (channel `Y`); other uncompressed formats -> `Rgba32Float` (sRGB linearized). NONE, RLE,
  ZIPS, ZIP, PIZ, PXR24, B44, B44A (DWA: `Unsupported`); scanlines or tiles, optional mips.
- Allocation limit: decoded bytes <= max(input * 4096, 64 MiB), at most 4 GiB.
- Tests (`tests/file_exr.rs`): 30 files written by the OpenEXR 3.x Python bindings
  (`scripts/gen_file_testdata_exrtiff.py`); decodes equal OpenEXR's (stored hashes, exact,
  also the lossy PXR24 / B44 / DWA files); encode -> OpenEXR decode exact for the lossless
  compressions; part / layer info; truncation and byte-flip loops.

## Audit

- 2026-11-22: decoder / encoder.
