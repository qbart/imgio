# bc7e (BC7 encoder)

Port of Basis Universal v2.50 `encoder/basisu_bc7e_scalar.cpp` (scalar C++ translation of
`bc7e.ispc`) in `src/bc7e.rs` (crate-private).

## API

- `Params` mirrors `bc7e_compress_block_params` (`opaque` / `alpha` = `m_opaque_settings` /
  `m_alpha_settings`, `use_luts` = `m_use_luts`). Presets: `Params::ultrafast / veryfast / fast /
  basic / slow / veryslow / slowest(perceptual)`; `Params::new` = `bc7e_compress_block_params_init`.
- `compress_block(&[[u8; 4]; 16], &Params) -> [u8; 16]`: one row-major RGBA8 block, output =
  the C++ `uint64_t[2]` little-endian. `compress_block_lut` also returns the `pUsed_lut` flag.
- One-color optimal endpoint tables (`bc7e_compress_block_init`) built once in a `OnceLock`.
- Stateless and allocation free per block; callers parallelize over blocks.
- Not ported: `bc7e_compress_block_single_mode` (testing API, no preset uses it, along with
  the `forced_partition` plumbing).

## Exactness

Bit-exact with the C++ as built by g++ for x86-64 (SSE2 scalar float, no fast-math, no FMA):

- `f32` math in the C++ operation order; `(int)` / `(int64_t)` casts emulate `cvttss2si` /
  `cvttsd2si` (out of range / NaN -> `INT_MIN`), which matters for `(int)floor(x)` of least
  squares alpha endpoints that can blow up.
- `1.0f / sqrt(s)` (vector normalize, covariance axis): the C++ calls `double sqrt(double)`
  (`<cmath>` without `std::`), so the reciprocal is computed in `f64` and rounded to `f32`.
  `floor(float)` likewise goes through `double`, which is exact.
- `min` / `max` / `clamp` follow the C++ templates (`a < b ? a : b`, `clamp = min(max(..))`,
  NaN clamps to `lo`); `abs(x) > 0` is not `x != 0` for NaN.
- Unsigned arithmetic wraps (`wrapping_*`); weight / index selection keeps the C++ tie rules
  (strict `<` first-wins in the LUT builder, perceptual search and alpha search; the N = 4 / 8
  linear search picks the highest tied index like the `select` chains).
- The mode 7 LUT does not reset its best entry between pbit combinations (upstream quirk, kept).

## Tests

- `bc7e::tests::matches_basisu`: 3 raw images x 7 presets x perceptual {0, 1} = 42 FNV-1a
  hashes vs `testdata/encode/bc7e_golden.txt` (`tools/basisu-golden/bc7e.cpp`, edge pixels
  clamped like `imgio::encode`). All 42 match (release and debug builds).
- Decode sanity (MSE vs source via `decode_to_buffer(Bc7Unorm)`), solid blocks exact.
- Speed on par with the C++ `-O2` scalar build (64x48 RGBA, single thread): ultrafast ~0.7 ms,
  veryfast / fast ~3 ms, basic / slow ~4-5 ms, veryslow ~12-17 ms, slowest ~15-23 ms.

## Audit

- 2026-11-21: ported `bc7e_scalar` (all presets, LUT path, `pUsed_lut`), golden generator and
  42-case golden test, all bit-exact on the first build.
