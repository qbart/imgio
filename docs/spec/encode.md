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
- BC7: `bc7f::fast_pack_bc7_auto_rgba` for Fastest (`cPackBC7FlagDefaultFastest`), Fast
  (`...Fast`), Default (`...PartiallyAnalytical`); `bc7e_scalar` (`src/bc7e.rs`, see bc7e.md)
  for High (`slow`) and Best (`slowest`), perceptual for `Bc7Srgb`. 64x48 test image: bc7f
  32.2 / 37.1 / 37.3 dB, bc7e slow / slowest 38.0 / 38.3 dB (10-50x slower).
- ETC1 / ETC2 RGB (sRGB): `etc1f::pack_etc1` (basisu real-time ETC1) up to Default; High /
  Best: `etc1hq` (own search: flip x individual / differential, each subblock fitted by basisu
  `etc1_optimizer` at slow / uber quality, the 2nd differential subblock constrained to the
  1st's base color like rg_etc1; solid blocks via `pack_etc1_block_solid_color`; perceptual
  for sRGB formats). 64x48 test image: 27.8 dB (etc1f) -> 29.2 dB (High / Best).
  ETC2 RGBA = EAC(A) + ETC1; EAC R11 / RG11 unorm = `pack_eac` on R (G); `Quality::High`+
  uses `pack_eac_high_quality`.
- ETC2 RGB A1 (punch-through; alpha < 128 = transparent): `etc1hq::encode_punchthrough_block`
  (own). Opaque blocks: differential-only `etc1_optimizer` search, opaque bit set. Blocks
  with transparent texels: opaque bit clear (raw index 2 transparent, 0 / 1 / 3 = base / +b /
  -b), 5-bit base colors searched around each subblock's opaque average and over the delta
  box of a fixed best candidate, every flip x table. No T / H / planar modes yet, so
  two-color subblock edges are weak (mosaic test image ~21 dB vs 29 dB for ETC1).
- Signed BC4 / BC5 / EAC R11 / RG11: source -> RGBA32F, `round(clamp(v, -1, 1) * 127) + 128`
  offset bytes through the unsigned encoders, endpoint / base bytes XOR 0x80 (offset binary
  -> two's complement keeps their order; signed EAC lacks the unsigned +4 bias).
- ASTC (all 2D blocks; unorm -> LDR, sRGB -> LDR_SRGB, float -> HDR profile): `src/astcenc`,
  the ktx-rs port of ARM astc-encoder (compressor + SSE4.1 / AVX2 copies picked at runtime;
  ktx-rs' `#[isa]` proc macro replaced by `isa_fn! { .. }` wrappers so no macro crate is
  needed). Presets fastest 0 / fast 10 / medium 60 / thorough 98 / exhaustive 100. One context,
  `available_parallelism` threads (1 below 64K pixels) each calling `astcenc_compress_image`;
  input RGBA8 (LDR) or RGBA32F (HDR). Golden: `astcenc::tests::compress_matches_astcenc` (ktx-rs
  golden from C++ astcenc: 8 block sizes x 4 profiles x 3 qualities x 4 flag sets, every ISA
  copy) + PSNR round trips (also multithreaded).
- PVRTC1 4bpp RGB / RGBA: basisu `encode_pvrtc1` (min/max endpoints + modulation fit against
  the bilinear upsample), whole image, power-of-two block counts (`InvalidArgument`). No
  encoders for ATC, PVRTC2, FXT1 (basisu has none outside ETC1S transcoding).
- BC6H unsigned: `fast_encode_bc6h` (transcoder's encoder) on RGB halves; source -> RGBA32F,
  `v > 0 ? half(min(v, 65504)) : 0`. Presets `{diff endpoint modes, 2-subset patterns,
  hq_ls, brute force weights}`: Fastest {0,0,-,-}, Fast {2,0,-,-}, Default {2,1,hq,-} (basisu
  default), High {4,8,hq,-}, Best {9,32,hq,brute}. Signed BC6H: no encoder yet.
- Golden test `tests/encode.rs` vs `testdata/encode/golden.txt` (`tools/basisu-golden/encode.cpp`,
  C++ basisu encoders with the same edge clamping) on `testdata/images/raw` (raw RGBA8 copies
  written by `scripts/gen_images.py`): BC1 x 3 qualities, BC3 x 3, BC4, BC5 on 3 images, all
  bit-exact; plus BC7 x 5 presets on the LDR images and BC6H x 5 on `raw/hdr_40x24.rgba16f`
  ETC1, ETC2 RGBA, EAC R11 / RG11 (normal / high quality), PVRTC1 RGB / RGBA on the
  top-left power-of-two crop; ETC2 RGBA at High compares only its EAC alpha halves
  (`etc2rgba_eac`) since High also switches the color half to `etc1hq` (65 lines). Round trips (PSNR / relative error) for BC1/2/3/4/7, BC6H.

## Audit

- 2026-11-21: API + BC1-BC5 encoders (basisu real-time), bit-exact with C++ (24 cases).
- 2026-11-21: PVRTC1 encoder exposed (`encode_pvrtc1`), 6 golden cases.
- 2026-11-21: ASTC encoder (astcenc port from ktx-rs), C++ golden passes on every ISA copy.
- 2026-11-21: ETC2 RGB A1 and signed BC4 / BC5 / EAC encoders (own; round-trip tested).
- 2026-11-21: high-quality ETC1 (`etc1hq`, etc1_optimizer search) for High / Best.
- 2026-11-21: BC7 High / Best = bc7e_scalar (12 API cases vs the bc7e golden); bc7f golden cut
  to the 3 presets the API uses (65 lines).
- 2026-11-21: ETC1 / ETC2 RGBA / EAC encoders (`etc1f`, `pack_eac[_high_quality]`), 65 cases.
- 2026-11-21: BC7 (`bc7f` presets) and BC6H (`fast_encode_bc6h` presets), 44 cases bit-exact.
