# Formats

- `Format` enum: uncompressed (R8, RG8, RGBA8, RGBA16F, RGBA32F) and BC1-BC7 (unorm/srgb, snorm, ufloat/sfloat).
- Each format has a block footprint (`block_dim`), `block_bytes`, and `decoded()`: the
  uncompressed format it decodes to losslessly (BCn LDR -> RGBA8, BC4 -> R8, BC5 -> RG8, BC6H -> RGBA16F).
- Sizes use checked arithmetic (`surface_bytes`, `row_bytes`) and return `Error::InvalidArgument` on overflow.
- Transcode-only targets (no decoder yet): PVRTC1/2, ATC. Packed uncompressed: RGB565, BGR565,
  RGBA4444, RGB16F, RGB9E5 (readable by `convert`).

## Audit

- 2026-11-21: initial `Format` with uncompressed and BC1-BC7 variants.
- 2026-11-21: ETC/EAC, R16/RG16, ASTC, PVRTC, ATC, 565/4444, RGB16F, RGB9E5 added.
