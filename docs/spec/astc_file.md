# .astc files

`imgio::astc_file`: ARM astcenc container. Header (16 bytes): magic `0x5CA1AB13` (LE), block x,
y, z (u8), width, height, depth (u24 LE); then 16-byte blocks, x fastest, then y, then z.

- `AstcFile::parse` (`block`, `width`, `height`, `depth`, `data`, `format(kind)` for 2D
  blocks: the header has no LDR / sRGB / HDR flag), `write(block, w, h, d, data)`.
- Test `tests/astc_file.rs`: 3 files from astcenc 6.0 (`-cl ... 4x4 / 6x5 / 12x12 -fast`,
  `testdata/astc/files`) parse and rewrite byte for byte; imgio's ASTC decoder output equals
  `astcenc -dl` (legacy RGBA8 DDS, read with `imgio::dds`).

## Audit

- 2026-11-21: reader / writer; decode cross-checked against astcenc 6.0.
