# Image types

- `Surface<'a>`: format + width + height + tightly packed `&[u8]`, size checked on creation.
- `mip_dim`, `mip_levels` (full chain to 1x1), `mip_ranges` (byte ranges of a packed chain,
  `vas::Vector<Range<usize>, Inline<32>>`, no heap).

## Audit

- 2026-11-21: initial `Surface` and mip helpers.
