# Formats

- `Format` enum: uncompressed (R8, RG8, RGBA8, RGBA16F, RGBA32F) and BC1-BC7 (unorm/srgb, snorm, ufloat/sfloat).
- Each format has a block footprint (`block_dim`), `block_bytes`, and `decoded()`: the
  uncompressed format it decodes to losslessly (BCn LDR -> RGBA8, BC4 -> R8, BC5 -> RG8, BC6H -> RGBA16F).
- Sizes use checked arithmetic (`surface_bytes`, `row_bytes`) and return `Error::InvalidArgument` on overflow.
- ASTC, ETC and Basis formats will be added with their codecs.

## Audit

- 2026-11-21: initial `Format` with uncompressed and BC1-BC7 variants.
