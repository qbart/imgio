# Conversion

- `half::{to_f32, from_f32}`: exact widening; narrowing rounds to nearest even, overflow -> inf,
  NaN kept. Tested exhaustively (all 65536 halves roundtrip).
- `convert::convert(src: &Surface, dst_format, dst)` / `convert_to_buffer`: any uncompressed
  format -> RGBA8 (unorm/srgb), RGBA16F, RGBA32F, through f32 per channel, multithreaded per row.
  Unorm `v/255`, snorm `max(v/127,-1)`, to 8-bit: clamp [0,1], round. Missing G/B = 0, A = 1.
  sRGB transfer not applied.

## Audit

- 2026-11-21: initial half and convert.
