# ASTC decode

- `Format::Astc(AstcBlock, AstcKind)`: 14 2D footprints (4x4..12x12), kinds `Unorm`, `Srgb`
  (LDR profile, decode to RGBA8) and `Float` (HDR profile, decode to RGBA16F).
- Decoder: basisu `astc_helpers` (`unpack_block` + `decode_block`) ported via ktx-rs
  `src/basisu/astc_helpers.rs`; also kept for the UASTC -> ASTC transcoders (packer, BISE).
- LDR output follows the decode_unorm8 rule (top 8 bits of the UNORM16 result), like astcenc
  with `ASTCENC_FLG_USE_DECODE_UNORM8`.
- Error color: magenta (LDR), all ones / NaN (HDR). HDR blocks in LDR profiles are errors.
- 3D blocks are not supported.

Tests: `tests/astc_decode.rs` vs the ktx-rs port of Arm astcenc (generator
`tools/astc-golden`): synthetic 37x29 images in 7 block sizes x 3 kinds, plus random blocks.
All bit-exact.

## Audit

- 2026-11-21: initial port, bit-exact with astcenc on all 27 cases.
