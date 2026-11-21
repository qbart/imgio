# Decode

- `imgio::decode(format, src, width, height, dst) -> Result<()>`: any block-compressed format
  (`can_decode`) to tightly packed rows of `format.decoded()`; edge blocks cropped.
  Errors: `Unsupported`, `BufferTooSmall` (src or dst), `InvalidArgument` (size overflow).
- `decode_to_buffer(...) -> Result<Buffer>`: same, allocates (vas `Buffer`).
- `block::decode_surface`: shared driver, one row of blocks per work unit through
  `parallel::for_each_unit` (std scoped threads, caller participates, inline below 256 KiB).
  Per-family block decoders: `bc::decode_block`, `etc::decode_block`.

## Audit

- 2026-11-21: generic driver and top-level API (BC, ETC).
