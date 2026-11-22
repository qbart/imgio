# Image resize (`imgio::resize`)

- `resize(&Surface, width, height, &ResizeOptions) -> Image` for every uncompressed format
  (block formats: `Unsupported`); output keeps the source format.
- Pipeline: convert to RGBA f32 -> sRGB decode of RGB (sRGB formats or `srgb: Some(true)`) ->
  premultiply RGB by alpha (formats with alpha, `premultiply`, default on) -> one `R32Float`
  plane per channel resampled by `mip::resize_into` (basisu resampler; planes in parallel) ->
  unpremultiply (alpha clamped to [0, 1] for non-float formats; alpha 0 -> color 0) -> sRGB
  encode -> convert back.
- Options: `filter` (every `mip::Filter`, default Lanczos4), `filter_scale`, `wrap`, `fit`,
  `srgb`, `premultiply`.
- `Fit`: `Exact`, `Contain` (inside the box, aspect kept), `Cover` (exact size, source cropped
  centered to the aspect first), `Width`, `Height`; `target_size` computes the size (rounded,
  at least 1).
- `crop(&Surface, x, y, w, h)` copies a rectangle (uncompressed formats).
- Tests: opaque images equal `mip::resize` byte for byte; premultiplied box filter does not
  bleed transparent color; sRGB averages in linear light (black + white -> 188); a constant
  image stays exact for every integer / half format (f32 within 1e-6); fit sizes, cover
  window, crop bounds.

## Audit

- 2026-11-22: resize, fit modes, crop.
