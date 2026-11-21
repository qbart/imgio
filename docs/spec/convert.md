# Conversion

- `half::{to_f32, from_f32}`: exact widening; narrowing rounds to nearest even, overflow -> inf,
  NaN kept. Tested exhaustively (all 65536 halves roundtrip).
- `convert::convert(src: &Surface, dst_format, dst)` / `convert_to_buffer`: any uncompressed
  format -> any uncompressed format, through f32 per channel, multithreaded per row.
  Unorm `v/max`, snorm `max(v/max,-1)`; writing clamps and rounds (snorm: round half away
  from zero), 565/4444 per channel `x*max+0.5`, RGB9E5 per `EXT_texture_shared_exponent`
  (clamp to 65408, exponent bump when the max mantissa rounds to 512). Missing G/B = 0, A = 1.
- `convert_with(..., &ConvertOptions { srgb_transfer })`: sRGB curve on RGB (never alpha) when
  exactly one side is sRGB (`srgb_to_linear` / `linear_to_srgb`, exact piecewise formulas);
  default is label-only.
- `Format::Rgba16Unorm` (VK 91) added for 16-bit sources.

## Audit

- 2026-11-21: initial half and convert.
- 2026-11-21: every uncompressed format writable, `Rgba16Unorm`, opt-in sRGB transfer.

- 2026-11-21: `Rgb8Unorm/Srgb`, `Bgra8Unorm/Srgb` (read / write; KTX2 vk 23/29/44/50 with
  red-blue-swapped DFD, DDS DXGI 87/91 + legacy 24/32-bit masks, KTX1 / GL `GL_RGB8`, `GL_BGRA`).
