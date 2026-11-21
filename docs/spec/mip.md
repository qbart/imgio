# Mipmaps and resizing

`imgio::mip`: `resize` / `resize_into` (any size), `generate_mips(src, levels, opts)` (levels
1..`levels`, each resampled from the base like KTX-Software `ktx create --generate-mipmap`,
levels in parallel through `parallel::try_for_each_item`).

- Resampler: port of basisu `basisu_resampler.cpp` / `basisu_resample_filters.cpp` (via
  ktx-rs `src/basisu`), separable, one resampler per channel sharing the contributor lists.
- `MipOptions { filter, filter_scale, wrap, srgb }`; defaults as `ktx create`: lanczos4, 1.0,
  clamp, sRGB from the format label. `Filter` = the 16 basisu filters (`name`, `from_name`,
  `ALL`); `Wrap` = clamp / wrap / reflect.
- Per pixel as KTX-Software `ImageT::resample`: unorm / 255 (65535), sRGB decode (clamped)
  for RGB, never alpha; samples clamped to [0, 1] for unorm (none for float); sRGB encode;
  unorm out: NaN / < 0 -> 0, > 1 -> max, else `v * max + 0.5` truncated.
- Formats: R8, RG8, RGBA8 (unorm / sRGB), R16, RG16, RGBA16 unorm, RGB16F / RGBA16F (via
  f32), RGBA32F (`can_resample`); others `Unsupported` (convert first). Max 16384 px.
- Golden test `tests/mip.rs`: 9 `ktx create --generate-mipmap` files (`testdata/ktx2/mips`,
  `tools/ktx2-golden/gen.py`): sRGB / linear, RGBA / RG / R, NPOT, 8 filters, filter scale,
  clamp / wrap / reflect; every level byte-identical.

## Audit

- 2026-11-21: resampler port, `resize`, `generate_mips`; bit-exact with `ktx create`.
