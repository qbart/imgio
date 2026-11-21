# ASTC LDR / XUASTC LDR encoder (basisu v2.50)

Bit-exact port of basisu v2.50 `astc_ldr::compress_image` (default build:
`BASISU_SUPPORT_ASTCENC` = 0, no SSE) in `src/basis/xuastc/enc/`. Public API: `encode_basis`
with `BasisCodec::AstcLdr(block)` / `BasisCodec::XuastcLdr(block)` (`basis_encode.md`); the
`basis_compressor` glue is `BasisCompressor::encode_slices_to_xuastc_or_astc_ldr` in
`src/basis/ll/comp.rs`.

## Pipeline (`enc::compress_image`)

1. Optional sharpening (`scd::sharpen_image`, gaussian `image_resample`).
2. Config: `HighLevelConfig` from the global config (`base_q = clamp(round(q * 2), 0, 200) / 2`,
   comp weights, `configure_encoder_effort_level`, force-disable subsets / RGB dual plane,
   decode mode). `max_candidate_limit` 16, or 64/32 (32/16 merged, fewer for huge images)
   with SCD.
3. Block encoders, each returning every block's candidate list + chosen index
   (`EncodeImageOutput`): `astcf::encode_image_astcf` (fast encoder `fencode`,
   `astc_ldrf`), `block_image::ldr_astc_block_encode_image` (full "basisu" encoder: 3
   superpasses over `lowlevel::LdrAstcLowlevelBlockEncoder`, DC / linear latent configs,
   blurring), `fast4x4::encode_image_fast_4x4` (effort 0, 4x4: `bc7f` -> ASTC). basisu's
   `cAuto` selection (in `comp.rs`): effort 0 astcf (fast 4x4 for 4x4), 1-2 astcf, 3 astcf +
   basisu merged when block > 64 texels or any source alpha, >= 4 both;
   `scd::merge_output_candidates` appends the second list and picks the lower WSSE.
4. SCD deblocking refinement (`scd::refine_output_for_deblocking`, default for blocks >= 80
   texels at effort >= 2; disables lossy supercompression), then `scd::cross_check_dct`.
5. Supercompression (`supercomp::supercompress`): full arithmetic, hybrid arithmetic + Zstd
   (weights) or full Zstd (`zstd_syntax`, `zstd-pure-rs` = libzstd 1.5.7, level 19 at effort
   >= 3 else 9; hybrid 9); images of <= 64 blocks always use full arithmetic. Lossy
   supercompression (quality < 100) may swap a block's chosen candidate for one that reuses a
   neighbor's config (+ endpoints) within the PSNR thresholds. Returns the bitstream
   (XUASTC payload) and the coded logical blocks (packed with `pack_astc_block` for ASTC LDR).

Modules: `common{,_encode,_surrogate}` (`basisu_astc_ldr_common.cpp`: CEM encoders, weight
refiner, `eval_solution`, surrogates), `partitions` (pattern vectors, LSH map,
`partitions_data`; the VP tree is unreachable with `BASISU_USE_LSH2/3` = 1 and not ported),
`grid_data` (`astc_block_grid_data`), `fencode{,_tables}`, `astcf`, `fast4x4`, `dct_enc`
(forward DCT, `code_block_weights`, `apply_weight_grid_dct`), `block_helpers` +
`lowlevel` (encode.cpp ~45-5996), `block_image` + `effort` (~6000-9621, 11070-11708), `scd`,
`arith_enc`, `supercomp`, `zstd_syntax`, `rand` (`basisu::rand`), `containers` (`hash_map`,
`std::priority_queue`), `dump` (test-only `.xud` loader).

## Exactness rules

- f32/f64 and float operation order as C++ (no fused ops); `(int)` casts as x86 `cvttss2si`
  (`common::ftoi`); C `roundf` = `f32::round`.
- Observable container order reproduced: libstdc++ `std::sort` (`ll::enc::std_sort_by`),
  `indirect_sort`, `std::priority_queue` heap order, `basisu::hash_map` slot order (Fibonacci
  hash, backwards probing, rehash order) with the C++ key hashes, `std::mt19937` + GCC 13
  `uniform_int/real` and `normal_distribution`, `static_vector` overflow drops.
- Thread-count independence (as C++): astcf / fast 4x4 / superpasses 0-1 per block; superpass 2
  reads frozen neighbor results, merged serially in raster order; SCD evaluates one checkerboard
  color per pass against committed state.
- C++ quirks kept on purpose: `weight_polish` reads rank `m_weights[idx]` but edits
  `m_weights[idx * num_planes]`; effort 1-2 `m_use_method2 = false` is re-enabled by
  `init_multi_subset_context`; SCD base+scale neighbor check reads row 0 / column 0; SCD
  partition mutation looks up the permutation with the new id; CEM 6/10 alpha refine uses
  unscaled high RGB; surrogate CEM 8/12 degenerate fix compares against channel 0's range.

Not ported: astcenc paths (`BASISU_SUPPORT_ASTCENC` = 0: the selections are ignored like the
default C++ build), debug images / statistics / final triple-decoder validation, `#if 0` code.
Not exposed in the public API: blurring, sharpening, heavy subset usage, SCD pass count,
encoder selection, subset / dual plane disabling, lossy thresholds (all reachable through
`AstcLdrEncodeConfig`).

## Verification

- Whole files: `tests/basis_encode_xuastc.rs` `matches_basisu_tool` encodes the 14
  `testdata/basisu_enc/cli_{astc,xuastc}*.basis` CLI goldens (`scripts/gen_basis_testdata.sh`)
  byte for byte with 1 and 4 threads (no header field differs); `round_trip` checks PSNR.
  Offline matrix (2026-11-21): 248 random CLI configs (all 14 block sizes, ASTC / XUASTC,
  effort 0-10, zstd / arith / hybrid, quality 10-100, sRGB / linear, mipmaps, deblocking
  modes, 8 images up to 128x96) + 6 configs on a 512x512 image: all byte-identical.
- Unit goldens (no C++ needed at test time; generators in `tools/basisu-golden/`):
  `basisu_astc_ldr_common.txt` (31 groups: CEM encoders, eval, surrogates on synthetic
  blocks), `basisu_astc_ldr_block.txt` (block-level full encoder, 200 cases),
  `basisu_astcf.txt` and `basisu_astc_ldr_image.txt` (candidate-list hashes of the astcf and
  basisu image encoders), `scd_*.pre.xud.zst` + `.cfg` (pre-SCD candidates -> SCD result hash),
  `sharpen_matches_cpp`, `rand` sequence, supercompressor re-encode of `testdata/basis` XUASTC
  files (byte-identical payloads).
- Stage-by-stage dumps (`#[ignore]` tests, `XU_DUMP_DIR`): a basisu CLI patched with
  `tools/basisu-golden/xuastc_dump.patch` writes each stage's candidate lists
  (`tools/basisu-golden/xuastc_dumps.sh`; loader `enc::dump`). Results at merge: astcf 203/203
  configs, basisu image encoder 220 cases / 18239 blocks (1-16 threads), SCD 29/29, merge
  22/22, supercompressor 123/123 bitstreams.

Speed (512x512, release, 24 threads / 1 thread; C++ CLI 1 thread in parentheses):
ASTC 6x6 0.22 s / 1.53 s (1.67 s), XUASTC 4x4 0.13 s / 0.78 s (0.75 s), XUASTC 12x12 (SCD)
0.57 s / 4.78 s (4.44 s), XUASTC 6x6 effort 1 0.06 s / 0.33 s.

## Audit

- 2026-11-21: full port (common, partitions, fencode/astcf, fast 4x4, DCT encode, full encoder
  block + image level, effort configs, SCD + sharpening + merge, arith / hybrid / zstd
  writers), `compress_image` glue, `.basis` v2 writer + key/value data, `encode_basis` API.
  All goldens and dump matrices above match.
