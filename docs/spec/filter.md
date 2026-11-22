# Image filters (`imgio::filter`)

Every uncompressed format; block formats return `Unsupported`. Output keeps the source format
unless the filter has a `format` option.

- Geometry (pixel copies, exact): `flip_horizontal`, `flip_vertical`, `rotate90` (clockwise),
  `rotate180`, `rotate270`, `transpose`.
- Other filters convert to RGBA f32, work, and convert back (integer formats clamped / rounded);
  per-pixel work and blur passes run in parallel over rows.
- `blur_box(radius)`, `blur_gaussian(sigma)` (radius `ceil(3 sigma)`): separable, edges
  repeated, linear light for sRGB formats, premultiplied alpha for formats with alpha.
- `unsharp_mask(sigma, amount, threshold)`: `v + amount * (v - blur)` on stored RGB values
  where `|v - blur| > threshold`.
- `adjust(Adjust { exposure, brightness, contrast, gamma, saturation })`: exposure in linear
  light (`* 2^stops`), then `(v - 0.5) * contrast + 0.5 + brightness`, `v^(1 / gamma)`,
  saturation around Rec. 709 luma, on stored (sRGB: encoded) values; alpha untouched.
- `grayscale(Luma::{Rec709, Rec601})`: luminance in linear light written to R, G, B.
- `swizzle([Channel; 4])` (R, G, B, A, Zero, One); `premultiply_alpha` /
  `unpremultiply_alpha` (sRGB formats in linear light; alpha 0 -> black).
- `normal_map(NormalMapOptions { strength, directx, wrap, format })`: Sobel gradient of the
  height (RGB luminance, else channel R), `n = normalize(-gx s, gy s, 1)` stored as
  `0.5 + 0.5 n` (OpenGL +Y up; `directx` flips green), alpha 1.
- `tone_map(ToneMapOptions { curve, exposure, format })`: linear input (sRGB decoded),
  `* 2^exposure`, curve per channel (`Clamp`, `Reinhard` `v / (1 + v)`, `ReinhardExtended`
  with white point, Narkowicz `Aces`), clamped, sRGB-encoded for sRGB outputs (default
  `Rgba8Srgb`); alpha clamped.
- Tests: geometry index maps and four quarter turns; constant images unchanged by blurs,
  impulse mass preserved, no transparent bleeding; unsharp overshoot / threshold; adjust,
  grayscale, swizzle, premultiply round trip (sRGB in linear light); normal directions for
  both conventions; tone curve values.

## Audit

- 2026-11-22: filters.
