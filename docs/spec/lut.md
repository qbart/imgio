# LUTs (`imgio::lut`)

Spec: 1D / 3D colour lookup tables, `.cube` files, applying and baking them.

- `Lut3d`: `size^3` RGB f32 lattice (2..=256), entry `r + size * (g + size * b)`, per-channel
  domain (default 0..1). `Lut1d`: three curves of `size` points (2..=65536) over a domain.
- `Cube` (`.cube`): Adobe Cube LUT 1.0 (`TITLE`, `DOMAIN_MIN` / `DOMAIN_MAX`, `LUT_1D_SIZE` or
  `LUT_3D_SIZE`) and Resolve (1D shaper + 3D, `LUT_1D_INPUT_RANGE` / `LUT_3D_INPUT_RANGE`);
  `#` comments, CRLF, unknown keywords skipped; keywords after data, wrong entry count,
  non-finite numbers, bad sizes / domains are `InvalidData`. Writing uses shortest round trip
  decimals, domain lines only when not 0..1 (input ranges for shaper + 3D, uniform domains).
- Lookup: input normalized by the domain and clamped to 0..1 (NaN -> 0); 1D linear, 3D
  trilinear or tetrahedral (default; the 6 tetrahedra and tie rules of colour-science /
  OCIO), all in f64; shaper first.
- `apply`: gray / gray + alpha / RGB / RGBA surfaces, values as stored (sRGB labels not
  decoded), alpha copied, any uncompressed output (gray outputs take red), rows in parallel.
- Baking: `Lut3d::from_fn`, `Lut3d::bake(size, domain, op)` runs a per-pixel image operation on
  an `Rgba32Float` image of the lattice points (`filter::adjust`, `tone_map`, `grayscale`,
  `color::convert`, ...), `Lut3d::from_profiles` = `color::convert` between two profiles.
- Images: HALD CLUT (`level^2` table, `level^3` square image, entry `k` at pixel `k`) and strips
  (`size^2 x size`, pixel `(r + size * b, g)`, Unreal / Unity grading textures), both ways.

## Audit

- tests/lut.rs: colour-science 0.4.7 writes 4 files (Iridas 3D 17 with title, 3D 9 with a
  non-uniform domain, 1D 1024 over 0..2, Resolve 1D 256 shaper over -0.25..4 + 3D 21) and
  applies them (trilinear and tetrahedral, float64) to 2048 pixels in and far outside the
  domains, including tetrahedron boundaries: imgio within 1e-6 relative (f32 tables), alpha
  passed, `eval` equals `apply`.
- Write -> parse keeps every table bit exact; exact text for a size 2 identity; 9 malformed
  inputs rejected; comments / CRLF / unknown keywords accepted.
- Identity exact within 1e-6; HALD / strip layouts checked per pixel and round trip;
  baked lattice entries equal the filter / colour transform output; a baked 33^3 P3 -> sRGB
  LUT applied to an 8-bit image is within 4 levels (mean < 0.25) of LittleCMS.
