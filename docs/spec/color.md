# Colour management (`imgio::color`, feature `color`)

Spec: ICC profiles and conversions; parsing, built-ins, CICP profiles and ICC writing by
`moxcms`; transforms by imgio's own per pixel f64 pipeline (`color_pipeline.rs`).

- `Profile::from_icc` (v2 / v4 RGB and gray: matrix / TRC, `lut8`, `lut16`, `mAB`, `mBA`; Lab
  or XYZ PCS), `Profile::new(ColorSpace)` (sRGB, linear sRGB, Display P3, DCI-P3, Adobe RGB,
  ProPhoto, Rec. 2020, Rec. 2100 PQ / HLG, ACEScg, ACES 2065-1, gray gamma),
  `Profile::cicp` (H.273 code points), `Profile::icc` (v4 bytes).
- `convert(src, from, to, intent, format)`: stored values normalized like `convert`, gray /
  gray + alpha / RGB / RGBA, alpha copied, any uncompressed output (integer outputs clamp).
- Pipeline (LittleCMS model, no precalculated device link): device -> PCS XYZ D50 -> device.
  Matrix / TRC: curves (`curv` tables linear interpolation, gamma, parametric types 0..4),
  colorant matrix; inverse with analytic parametric inverses / monotonic table search, sign
  symmetric below 0 (extended range floats). Gray: Y * D50 / Y. LUT tags: lut8 / lut16 (matrix
  for XYZ input, input tables, CLUT, output tables; lut16 Lab legacy encoding), mAB (A, CLUT,
  M, matrix + offset, B), mBA (B, matrix + offset, M, CLUT, A); tetrahedral (LittleCMS
  `Eval3Inputs`), bilinear or linear CLUTs; stage values clamped to 0..1; XYZ u1Fixed15 and Lab
  encodings. Intent tables A2B0/1/2 with fallback to 0; absolute = relative scaled by
  media white points. No black point compensation (LittleCMS forces it only for v4 LUT
  profiles in perceptual / saturation; matrix shapers are unaffected).

## Audit

- tests/color.rs vs LittleCMS 2.19 8-bit transforms with `NOOPTIMIZE` (scripts/
  gen_color_testdata.py, hand-built profiles): 9 matrix / TRC cases (sRGB, Display P3, Adobe
  RGB, ProPhoto, Rec. 2020; perceptual, relative, absolute) and 2 gray cases are byte
  identical; 3 LUT profile cases (lut16 XYZ PCS, 17^3 / 33^3 grids, shaper curves) within 1
  except one near-black value at 2 (true value between), mean <= 0.016.
- moxcms's own transforms (a resampled 3D device link for LUT profiles) were 10+ levels off
  in steep regions; replaced by the pipeline above.
- Built-in sRGB vs LittleCMS sRGB, identity, Rec. 2020 float round trip, linear sRGB = sRGB
  EOTF (< 2e-4), CICP 1/13/0 = sRGB, every built-in transforms and survives an ICC write /
  read; alpha pass-through, gray / RGB mismatches, compressed input, corrupt profiles (no
  panics).
