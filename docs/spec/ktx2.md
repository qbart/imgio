# KTX2 (Basis payloads)

`imgio::ktx2::Ktx2File` borrows the file bytes.

- Header (80 bytes), level index (24 bytes/level), DFD (44 or 60 bytes, one basic block),
  key/values (`KTXanimData` -> video), supercompression global data.
- Payload detection as basisu `ktx2_transcoder::init`:
  | DFD model | vkFormat | scheme | payload |
  |---|---|---|---|
  | 163 ETC1S | 0 | 1 BasisLZ | ETC1S (alpha when DFD is 60 bytes) |
  | 166 UASTC LDR | 0 | 0 / 2 zstd | UASTC LDR 4x4 (alpha: channel RGBA/RRRG) |
  | 167 UASTC HDR | ASTC_4x4_SFLOAT | 0 / 2 | UASTC HDR 4x4 |
  | 162 ASTC | ASTC_6x6_SFLOAT | 0 / 2 | ASTC HDR 6x6 |
  | 168 | 0 | 4 (12-byte descs) or 1 (8-byte, v1.6 files) | UASTC HDR 6x6 intermediate |
  | 162 ASTC | ASTC_{4x4..12x12}_{UNORM,SRGB} (157..184) | 0 / 2 | ASTC LDR (DFD block dims must match) |
  | 169 | 0 | 5 (12-byte descs) | XUASTC LDR, block size from DFD texel block dims |
  | 170 | 0 | 6 (12-byte descs) | XUBC7 |
- Supercompressed slice descriptor arrays (schemes 4/5/6) must exactly fill the global data.
- Images are indexed `(level * layers + layer) * faces + face`. ETC1S/6x6i/XUASTC/XUBC7 slices
  come from the
  global data (offsets relative to the level); plain payloads are `(layer * faces + face)`
  images of `blocks * 16` bytes in the (inflated) level.
- Zstandard levels are inflated with `ruzstd` (pure Rust) into a vas `Buffer`, once per level.
- Not supported: depth > 0, non-Basis KTX2 (plain BCn etc.).
- `transcode(level, layer, face, format, dst)` reuses the container-independent
  `basis::transcode::transcode_slice`; `transcode_with` takes `basis::TranscodeOptions`
  (`Deblock::Auto` follows `DeblockFilterID`, `Off`/`Force` override it like C++).

Test: `tests/ktx2_transcode.rs`, 436 level/layer/target cases (XUBC7, ETC1S with mips and a 2-layer
array, UASTC zstd and raw, HDR 4x4, ASTC HDR 6x6, 6x6i) bit-exact with the C++ `ktx2_transcoder`.

## Audit

- 2026-11-21: initial reader written from the KTX 2.0 spec and basisu `ktx2_transcoder`.
- 2026-11-21: DFD models 169 (XUASTC LDR), 170 (XUBC7) and ASTC LDR vkFormats parsed.
