# JPEG XL (`imgio::file::jxl`)

Feature `jxl`: decode with `jxl-oxide` 0.12 (conformant, pure Rust; `rayon` threads, `moxcms`
for embedded ICC profiles), lossless encode with `zune-jpegxl` 0.5.

- `decode` / `decode_with(&DecodeOptions { max_pixels, max_alloc_bytes })`: codestream
  (`FF 0A`) and ISOBMFF container, VarDCT and modular, any bit depth, alpha, orientation
  applied. Main image = first displayed keyframe. The header is parsed from a growing prefix
  first; canvases above `max_pixels` (default 2^28) are `Unsupported` before frame data is
  decoded; jxl-oxide's `AllocTracker` bounds its allocations (default 4 GiB). Truncated files
  (not all frames loaded) are `InvalidData`.
- Output: integer <= 8 bits -> `R8Unorm` / `Rg8Unorm` / `Rgb8Srgb` / `Rgba8Srgb`; 9..16 bits ->
  `R16Unorm` / `Rg16Unorm` / `Rgba16Unorm` (scaled `v * 65535 / (2^bits - 1)`, rounded; RGB gets
  alpha 65535); float or > 16-bit integer -> `R32Float` (gray) / `Rgba32Float` (gray + alpha
  replicates gray; RGB gets alpha 1.0). CMYK: `Unsupported`.
- Color: integer images are rendered to sRGB (gray: sRGB curve, D65), float images to linear
  sRGB (gray: linear), with `RenderingIntent::Relative`. XYB and enum encodings convert in
  jxl-oxide; ICC via moxcms. Files already in the target encoding are not touched (bit exact).
- `decode_animation`: every keyframe, composited canvas; delay = `duration * tps_den /
  tps_num` s as a reduced fraction; `loops` = `num_loops` (0 = forever). Wired into
  `file::decode_animation`.
- `info`: size, bits, float, gray, alpha, xyb, animated, container.
- `encode(&Surface, &EncodeOptions { effort, threads })`: lossless modular, tagged sRGB.
  `R8Unorm`, `Rg8Unorm`, `Rgb8*`, `Rgba8*`, `Bgra8*` (swizzled), `R16Unorm`, `Rgba16Unorm`;
  `Rg16Unorm` is stored as RGBA16 with R = G = B (zune's 16-bit gray + alpha path reads samples
  as `i16`). zune 0.5 declares the alpha of 16-bit RGBA as 8-bit: imgio regenerates zune's
  byte-aligned image header with an explicit 16-bit alpha channel (and checks the original
  prefix matches its own rebuild first, else `Unsupported`). Width or height < 2, floats,
  other formats: `Unsupported`.
- Tests (`tests/file_jxl.rs`, data `testdata/file/jxl` from
  `scripts/gen_file_testdata_jxlj2kvdb.py`, libjxl via imagecodecs 2026.11.11): 17 files with
  libjxl's decodes. Exact: lossless 8-bit RGBA / RGB container / gray / gray + alpha, 16-bit
  RGB / RGBA container / gray / gray + alpha, float32 RGBA (HDR > 1), float16 RGB, float32
  gray, 3-frame animation (delays 1/10 s). Bounded: lossy VarDCT 8-bit RGB / RGBA / gray max 1,
  16-bit lossy max 8 / 65535, 16-bit linear-transfer file color managed to sRGB vs float64
  reference max 8 / 65535 (jxl-oxide's f32 transfer curve). Encoder: 10 sources x 2 effort /
  thread settings round trip exactly through our decoder; their libjxl decodes were verified
  exact (`IMGIO_JXL_ENC_DUMP=dir` + `--verify-jxl-encoder dir`). Truncation at ~60 lengths
  per file and random bit flips (`IMGIO_JXL_FUZZ`, 3000 per file run once) never panic.

## Audit

- 2026-11-22: decoder (jxl-oxide 0.12.6), encoder (zune-jpegxl 0.5.2 + 16-bit alpha header
  fix), animation; libjxl-checked.
