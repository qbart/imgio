# XUBC7 encoder (XBC7 slice streams)

Port of Basis Universal v2.50 `encoder/basisu_xbc7_encode.cpp` (`xbc7::pack_image`) in
`src/basis/xubc7/encoder.rs` (crate-private, reachable from `crate::basis`). Output: the XBC7
slice stream of one image, byte-identical to C++, decodable by `xubc7::decoder`. Public entry:
`encode_basis` / `ktx2::encode_basis` with `BasisCodec::Xubc7` (`basis_compressor::
encode_slices_to_xubc7`, see `basis_encode.md`).

## API

- `encode(rgba, width, height, &EncodeParams) -> Result<Buffer>`; `encode_blocks` also returns
  the coded logical BC7 blocks (what the stream decodes to; `pack_bc7` them for plain BC7).
  Tightly packed RGBA8, 1..=65535 per side (the decoder caps at 16384). Alpha is detected like
  `image::has_alpha` (any texel < 255); opaque images code 3-channel solids / mode 6 alpha.
- `EncodeParams` mirrors `xbc7::pack_options` minus debug / stats: `dct_q` [1,100] (100 =
  lossless weights), `weights`, `effort_level` [0,10], `optimize_weights_after_bc7f`,
  `bc7_pack_flags`, `bc7_encoder` (`Bc7f` / `Bc7eScalar`), `bc7e_level` [0,6], `perceptual`,
  `self_validate`, alt-pack / repeat / solid / endpoint / AC-truncation RDO enables and PSNR
  drops, `rdo_min_block_psnr`, `wt_dpcm_alpha_pct`, `stripes`, `zstd_level` (C++: fixed 19).
  `EncodeParams::pack_options()` = the C++ struct defaults; `set_rdo_level(0..=100)`.
- `Stripes`: `Auto` (`m_num_stripes = 0`), `Desired(n)` (`set_num_stripes_for_image`: >= 16
  block rows per stripe, 1 below 128 texel rows), `Exact(n)` (`m_num_stripes = n`, clamped to
  [1, min(block rows, 16)]).
- `BasisuParams` = the XUBC7 part of `basis_compressor_params` (quality -1 / 1..100, effort,
  RDO level, desired stripes 8, encoder, bc7e level, channel weights 9,11,1,11 + perceptual,
  `set_srgb(false)` = `-linear`); `EncodeParams::from_basisu` replays
  `encode_slices_to_xubc7` (weight re-opt at Q >= 90 and effort >= 3; bc7f flags `Default` at
  effort <= 1, `NonAnalytical` at 10). `EncodeParams::default()` = the CLI `-xubc7` default.

## Structure

1. Base pack, block rows in parallel: bc7f `fast_pack_bc7_auto_rgba` or `bc7e::
   compress_block_lut` (LUTs only at Q 100; LUT blocks are forced to lossless DPCM), optional
   bc7f alt-pack RDO, `unpack_bc7`, `canonicalize_endpoints`, optional
   `optimize_block_weights`.
2. Per stripe, stripes in parallel (each references only its rows, so running reuse RDO ->
   endpoint RDO -> coding per stripe equals the C++ pass order): block-reuse RDO, endpoint
   RDO, `pack_stripe` (Repeat / Solid DPCM / config reuse, endpoint DPCM sweep over 4
   neighbors + 2 subset-1 modes + 32 XY deltas with a 3% switch margin, weight DCT sweep
   (fewest ACs, then 2% error margin) vs lossless DPCM sweep by estimated octobits, optional
   AC truncation).
3. Merge in stripe order (bit streams appended without padding), byte-plane seek table of
   deltas for > 1 stripe, 7 byte header, `serialize` (zstd where it saves anything up to 128
   bytes, else >= size/400), tiny-mip fallback (`0xB8/0xB9`, raw BC7) when smaller.
4. Self validation: `decoder::decode` + `compare_block_full` against the coded blocks.

Reused from `decoder.rs` (made `pub(super)`): `Fx` (Q15.16), quant tables / level scale / span,
`dct_inverse`, `Tile`, `eval_weight_predictor`, candidate / XY delta / blob id constants. New:
`dct_forward` (`forward_4x4` butterflies, deadzone quantizer, per-bit-depth DC quantizer),
`endpoint_dpcm_encode` (packed prediction obtained by decoding zero residuals with
`bc7u::endpoint_dpcm_decode`; the packed values never exceed the endpoint mask, so the wrapped
residual bytes are identical to C++), `compare_block_configs` / `compare_block_full` (only the
used subsets / p-bits / planes, not struct equality), `is_solid_blk`, `canonicalize_endpoints`,
`bitwise_coder`, `blob_stream_writer`.

## Exactness

- Integer / fixed point everywhere except the RDO block PSNR (`xbc7_block_wsse_psnr`): `f32`
  in the C++ order, `sqrtf` (exact) and `log10f` (platform libm, same glibc call as C++).
  Thresholds compare in `f32` (`p >= orig - drop`, `primary - alt < drop`).
- C++ unsigned / wrapping semantics kept: weighted dist2 in `i32` then `u32`, `x * 1000 <
  best * 980` with `best = UINT64_MAX` (wrapping multiply; harmless by the sweep order).
- Zstandard via `zstd-pure-rs` 0.2 (libzstd 1.5.7 translation) = basisu's bundled 1.5.7:
  identical compressed blobs.
- Golden C++ built with `-ffp-contract=off`, no fast-math, SSE off (`tools/basisu-golden/
  build.sh`, `xubc7_encode.cpp`).

## Tests

- `encoder::tests::matches_basisu`: 84 lines of `testdata/encode/xubc7_golden.txt` (3 raw
  images + `:opaque` alpha-stripped copies + `:tiny` 6x5 crop + `:tall` 4x tiling = 7 images x
  12 parameter sets: CLI default, Q 20..90, effort 0..10, RDO 20..100, bc7e levels 0/2/4, sRGB
  and linear weights, forced 3 / 4 stripes). 69 lines are blob streams, 15 tiny mips; all
  byte-exact (size + FNV-1a 64).
- `xubc7_extra_golden` (ignored; `XUBC7_GOLDEN` / `XUBC7_RAW` env): offline check on larger
  images. 2026-11-21: 96 lines on kodim01 / kodim23 (768x512, 8 stripes), alpha0 (256x256),
  bicycle_alpha (328x340) + variants (incl. 256x1024 tall) all match.
- `round_trip`: decode == coded blocks; at Q 100 without weight re-optimization the decoded
  pixels equal direct bc7f / bc7e output; lossy Q 80 / Q 40 + RDO 60 decode above 30 dB and not
  above the base encoder.
- Speed: kodim01 768x512, the 12 golden parameter sets x {RGBA, opaque}: Rust 10.0 s on one
  core vs C++ (`-O2`, 1 thread) 9.4 s; 3.1 s on 24 threads (base pack by row, coding by stripe,
  so <= 8 stripes bound the coding pass). CLI default on kodim01: ~0.33 s single core.

## Not ported

Stats dashboard (`xbc7_pack_stats`, `xbc7_dct_stats`), debug printing, debug images / legends
(`m_debug_*`, `m_print_stats`): output-neutral. The `job_pool` is replaced by
`parallel::try_for_each_item` (output independent of thread count, like C++).

## Audit

- 2026-11-21: ported `basisu_xbc7_encode.cpp` + `encode_slices_to_xubc7` option setup; golden
  tool and 84-line golden; byte-exact on the first build (plus 96 offline lines on larger
  images).
- 2026-11-21: wired into `basis_compressor` (`encode_slices_to_xubc7` via
  `EncodeParams::from_basisu`) and `BasisCodec::Xubc7`: `.basis` and KTX2 files byte-identical
  to the 2.50 CLI (`tests/basis_encode_xubc7.rs`, 9 + 1 goldens).
