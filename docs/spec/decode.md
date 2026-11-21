# Decode

- `imgio::decode(format, src, width, height, dst) -> Result<()>`: any block-compressed format
  (`can_decode`) to tightly packed rows of `format.decoded()`; edge blocks cropped.
  Errors: `Unsupported`, `BufferTooSmall` (src or dst), `InvalidArgument` (size overflow).
- `decode_to_buffer(...) -> Result<Buffer>`: same, allocates (vas `Buffer`).
- `block::decode_surface`: shared driver, one row of blocks per work unit through
  `parallel::for_each_unit` (std scoped threads, caller participates, inline below 64K pixels).
  Per-family block decoders (`fn(Format, &[u8], &mut [u8]) -> Result<()>`, driver uses
  `try_for_each_unit`): `bc`, `etc`, `astc`, `atc`, `fxt1`, `pvrtc::decode_pvrtc2_block`.
- PVRTC1 (`pvrtc::decode_pvrtc1`): whole surface (each pixel blends the endpoints of the 2x2
  blocks around it, wrapping), rows in parallel; twiddled block order (basisu
  `pvrtc4_swizzle_uv`); block counts must be powers of two (`InvalidArgument`).
- Legacy decoders follow basisu `gpu_image::unpack`: ATC RGB / RGBA (BC4 alpha) full; FXT1
  only CC_MIXED without alpha, PVRTC2 only hard-transition non-interpolated blocks (what the
  Basis transcoder emits), other block modes `Unsupported`.
- Golden test `decode_legacy_formats` (tests/basis_transcode.rs): our transcode of every
  `.basis` level to PVRTC1/2, ATC, FXT1, decoded, vs basisu `gpu_image::unpack` hashes
  (`testdata/basis/golden_decode.txt`, `tools/basisu-golden/unpack.cpp` linked with the
  basisu encoder library). The tool forces alpha 255 for ATC RGB: C++ `unpack_atc` leaves
  alpha of mode-0 palette entries 1/2 uninitialized.

## Audit

- 2026-11-21: generic driver and top-level API (BC, ETC).
- 2026-11-21: PVRTC1/2, ATC, FXT1 decoders; bit-exact with basisu on 104 transcoded levels.
