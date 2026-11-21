# Encode

- `imgio::encode(src: &Surface, format, &EncodeOptions { quality }, dst)` / `encode_to_buffer`,
  `can_encode`. Source: any uncompressed format (converted to RGBA8 first, labels only).
  One row of blocks per work unit (`parallel::try_for_each_unit`); edge blocks read clamped
  pixels.
- `Quality` (Fastest, Fast, Default, High, Best): each encoder maps it to its settings.
- BC1: basisu `encode_bc1` (transcoder real-time encoder, 4-color opaque blocks): Fastest /
  Fast = flags 0, Default = high quality, High / Best = higher quality.
- BC3 = `encode_bc4`(A) + BC1; BC4 = `encode_bc4`(R); BC5 = R + G; BC2 = explicit alpha
  `(a * 15 + 127) / 255` + BC1.
- BC7: `bc7f::fast_pack_bc7_auto_rgba` with `cPackBC7FlagDefault{Fastest, Fast, (Default),
  PartiallyAnalytical, NonAnalytical}` for Fastest .. Best (bc7e to come for the slow end).
- ETC1 / ETC2 RGB (sRGB): `etc1f::pack_etc1` (basisu real-time ETC1, single quality).
  ETC2 RGBA = EAC(A) + ETC1; EAC R11 / RG11 unorm = `pack_eac` on R (G); `Quality::High`+
  uses `pack_eac_high_quality`. Not yet: ETC2 RGB A1, signed EAC, a slower high-quality ETC1.
- BC6H unsigned: `fast_encode_bc6h` (transcoder's encoder) on RGB halves; source -> RGBA32F,
  `v > 0 ? half(min(v, 65504)) : 0`. Presets `{diff endpoint modes, 2-subset patterns,
  hq_ls, brute force weights}`: Fastest {0,0,-,-}, Fast {2,0,-,-}, Default {2,1,hq,-} (basisu
  default), High {4,8,hq,-}, Best {9,32,hq,brute}. Signed BC6H: no encoder yet.
- Golden test `tests/encode.rs` vs `testdata/encode/golden.txt` (`tools/basisu-golden/encode.cpp`,
  C++ basisu encoders with the same edge clamping) on `testdata/images/raw` (raw RGBA8 copies
  written by `scripts/gen_images.py`): BC1 x 3 qualities, BC3 x 3, BC4, BC5 on 3 images, all
  bit-exact; plus BC7 x 5 presets on the LDR images and BC6H x 5 on `raw/hdr_40x24.rgba16f`
  ETC1, ETC2 RGBA, EAC R11 / RG11 (normal / high quality) (65 lines). Round trips (PSNR / relative error) for BC1/2/3/4/7, BC6H.

## Audit

- 2026-11-21: API + BC1-BC5 encoders (basisu real-time), bit-exact with C++ (24 cases).
- 2026-11-21: ETC1 / ETC2 RGBA / EAC encoders (`etc1f`, `pack_eac[_high_quality]`), 65 cases.
- 2026-11-21: BC7 (`bc7f` presets) and BC6H (`fast_encode_bc6h` presets), 44 cases bit-exact.
