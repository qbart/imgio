# WebP (`imgio::file::webp`, feature `webp`)

Crate `image-webp` 0.2 (pure Rust).

- Decode: lossy (VP8), lossless (VP8L), alpha (ALPH, filtered / compressed). Stills ->
  `Rgb8Srgb` / `Rgba8Srgb`; lossy uses libwebp's fancy upsampling and YUV -> RGB (bit-exact
  with libwebp).
- Animation: own `ANIM` / `ANMF` parsing, frames decoded by `image-webp`, composited like
  libwebp `WebPAnimDecoder` (transparent black canvas, ANIM background ignored as the spec
  allows, libwebp's non-premultiplied integer blend); delays `(ms, 1000)`; main image = frame 0.
- Encode: lossless only (no pure Rust lossy encoder) from `Rgb8*`, `Rgba8*`, `Bgra8*`,
  `R8Unorm`, `Rg8Unorm`.
- Tests (`tests/file_webp.rs`): files from libwebp 1.6 (imagecodecs, Pillow, ffmpeg
  `libwebp`); stills / alpha exact vs libwebp, animations exact vs Pillow (libwebp
  `WebPAnimDecoder`), encodes byte-identical to Pillow-verified files; truncation / flip loops.

## Audit

- 2026-11-22: decoder / lossless encoder.
