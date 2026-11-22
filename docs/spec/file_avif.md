# AVIF (`imgio::file::avif`, feature `avif`)

- Decode: `avif-parse` (HEIF container, primary + alpha items), `re_rav1d` 0.1 (rav1d, Rust
  port of dav1d) without its `asm` feature, own YUV -> RGB. `re_rav1d` lists `cc` and
  `nasm-rs` as build dependencies, but its build script only uses them with `asm`: with it
  off, `main()` is empty and nothing but Rust is compiled.
- 8 / 10 / 12 bits, 4:2:0 / 4:2:2 / 4:4:4 / mono, alpha, full / limited range; matrices from
  `nclx` (else the AV1 sequence header): BT.601 / 470BG (also unspecified), 709, FCC, SMPTE 240,
  BT.2020 NCL, identity, YCgCo. Constant luminance / chroma-derived / ICtCp: `Unsupported`.
- YUV -> RGB reproduces libavif's built-in path bit for bit (f32, bilinear 9/3/3/1 chroma).
- Output: 8-bit `Rgb8Srgb` / `Rgba8Srgb`; 10 / 12-bit `Rgba16Unorm` (`v * 65535 / max`). PQ /
  HLG not applied; `irot` / `imir` / `clap` ignored; premultiplied alpha unpremultiplied;
  grids and animated AVIF (`avis`): `Unsupported`. `decode_planes`: raw Y / U / V / A.
- Encode: `ravif` (rav1e, no asm) from `Rgb8*`, `Rgba8*`, `Bgra8*`, `R8Unorm`, `Rg8Unorm`;
  4:4:4 BT.601 full range, 8 or 10 bits; quality / speed / alpha quality options.
- Tests (`tests/file_avif.rs`): files from libavif 1.4 / libaom (imagecodecs) and ffmpeg; YUV
  planes exact vs dav1d; RGB exact vs libavif's built-in conversion (10 / 12-bit, YCgCo,
  identity, mono); 8-bit YCbCr within 2 of libavif's libyuv path (11 for limited-range
  BT.709, where libyuv also deviates from the float formulas); encodes decode in libavif;
  truncation / flip loops.

## Audit

- 2026-11-22: decoder / encoder.
