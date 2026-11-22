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
  Matrix / TRC: curves (`curv` tables as LittleCMS evaluates 16-bit curves: input rounded to
  a word, `LinLerp1D` fixed point, 16-bit output; gamma, parametric types 0..4),
  colorant matrix; inverse with analytic parametric inverses / monotonic table search, sign
  symmetric below 0 (extended range floats). Gray: Y * D50 / Y. LUT tags: lut8 / lut16 (matrix
  for XYZ input, input tables, CLUT, output tables; lut16 Lab legacy encoding), mAB (A, CLUT,
  M, matrix + offset, B), mBA (B, matrix + offset, M, CLUT, A); CLUTs like LittleCMS
  `EvaluateCLUTfloatIn16`: 16-bit words, `TetrahedralInterp16` (trilinear for Lab indexed
  tables), `Eval1Input`, with their 32-bit wrap-around (2 input tables: f64 bilinear); stage
  values clamped to 0..1; XYZ u1Fixed15 and Lab
  encodings. Intent tables A2B0/1/2 with fallback to 0; absolute = relative scaled by
  media white points. No black point compensation (LittleCMS forces it only for v4 LUT
  profiles in perceptual / saturation; matrix shapers are unaffected).

## Files (`imgio::file::colorimetry`, `decode_to`)

- `colorimetry(data)`: PNG cICP > iCCP > sRGB > gAMA (+ cHRM); JPEG APP2 `ICC_PROFILE`
  segments joined by sequence; WebP `ICCP`; TIFF tag 34675 (classic / BigTIFF); AVIF `colr`
  `prof` / `rICC` before `nclx`; OpenEXR `chromaticities` (linear, Rec. 709 default);
  Radiance `PRIMARIES=` (linear); JPEG XL = sRGB (decoder renders); others unspecified.
- `decode_to(data, target, intent)`: unspecified = sRGB (integer) / linear sRGB (float), gray
  equivalents for gray images (gray profiles; cICP and RGB ICC profiles on gray images are
  ignored), unmatched embedded profiles (CMYK) ignored; returns the decode unchanged when the
  source is the target's built-in space. Output keeps the depth; float for linear targets and
  PQ / HLG sources; gray / RGB per target; 8-bit labeled `*Srgb` for an sRGB target.
- `Profile::from_primaries`: chromaticities -> XYZ, Bradford to D50 colorants, parametric
  gamma / sRGB / identity curves.

## Audit

- tests/color.rs vs LittleCMS 2.19 8-bit transforms with `NOOPTIMIZE` (scripts/
  gen_color_testdata.py, hand-built profiles): 9 matrix / TRC cases (sRGB, Display P3, Adobe
  RGB, ProPhoto, Rec. 2020; perceptual, relative, absolute), 2 gray cases and 3 LUT profile
  cases (lut16 XYZ PCS, 17^3 / 33^3 grids, shaper curves) are byte identical (LUT profiles
  were within 2 with f64 curve / CLUT interpolation; LittleCMS 2.14 sources read for the 16-bit
  evaluation).
- Device links (`apply_link`, `Profile::is_device_link`): the A2B0 table maps device values
  directly (RGB or gray each side); `convert` rejects links. tests/lut.rs: LittleCMS `linkicc`
  links (P3 -> sRGB mAB with sampled curves, gray 1D CLUT) match `transicc -c0 -l` within one
  16-bit word (f32 pipeline rounding edges: at most 4 values), the rest within its 4 decimal
  print.
- moxcms's own transforms (a resampled 3D device link for LUT profiles) were 10+ levels off
  in steep regions; replaced by the pipeline above.
- Built-in sRGB vs LittleCMS sRGB, identity, Rec. 2020 float round trip, linear sRGB = sRGB
  EOTF (< 2e-4), CICP 1/13/0 = sRGB, every built-in transforms and survives an ICC write /
  read; alpha pass-through, gray / RGB mismatches, compressed input, corrupt profiles (no
  panics).
- tests/color_files.rs: the embedded P3 profile is found byte for byte in Pillow PNG, JPEG,
  WebP, TIFF and AVIF files, a 245 KB profile over APP2 segments, cICP 9/16/0/1, gAMA + cHRM
  values, EXR chromaticities, Radiance primaries; a P3 PNG decoded to LittleCMS's sRGB is
  byte identical with LittleCMS (imgio's built-in sRGB within 1), gAMA + cHRM within 1 of a
  LittleCMS profile built from the chunks; plain sRGB files unchanged; P3, gray, PQ -> linear
  float and EXR -> linear sRGB targets (neutrals stay neutral within 0.1%).
