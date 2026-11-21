# BC1-BC7 decode

API (`imgio::bc`):
- `decode(format, src, width, height, dst) -> Result<()>`: writes tightly packed rows of
  `format.decoded()`; partial edge blocks are cropped. Errors: `Unsupported` (not a BC format),
  `BufferTooSmall` (src or dst), `InvalidArgument` (size overflow).
- `decode_to_buffer(format, src, width, height) -> Result<Buffer>`: same, allocates a vas buffer.

Behavior:
- BC1: 4-color when c0 > c1, else 3-color + transparent black. BC2/BC3 color is always 4-color (D3D10).
- 565 expansion by bit replication; interpolants `(2a+b)/3` truncated (as basisu, Pillow).
- BC4/BC5 unsigned: integer interpolation, truncated. Signed: -128 endpoints clamp to -127,
  interpolants rounded down (floor). Output R8/RG8 (snorm as two's complement bytes).
- BC6H: port of basisu `unpack_bc6h` (with `+32` interpolation rounding); output RGBA16F, alpha 1.0.
- BC7: port of basisu `bc7u::unpack_bc7`. Invalid modes (BC6H and BC7) decode to zero.
- Multithreaded per block row through `parallel::for_each_unit` (std scoped threads, calling
  thread participates, inline below 256 KiB output).

Tests (`tests/bc_decode.rs`, data from `scripts/gen_bc_testdata.py`): random blocks, fixed seed.
- Pillow 13 reference for BC1-BC5, BC6H unsigned, BC7. Known Pillow deviations, accepted:
  invalid BC7 -> opaque black; BC5 snorm without -128 clamp (±1); BC6H without `+32` rounding (±1);
  BC6H signed wrong (not used).
- bcdec_rs (dev-dependency) bit-exact reference for BC6H (both) and BC7.

## Audit

- 2026-11-21: BC1-BC5 written from the D3D spec; BC6H/BC7 decoders and tables from Basis
  Universal via ktx-rs `src/basisu/gpu_texture.rs`, `uastc_tables.rs`, `hdr_tables.rs`.
