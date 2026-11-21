# astcenc API (astc-encoder 6.0 port)

`src/astcenc` (crate-internal) ports the astc-encoder 6.0.0 C API: `astcenc_config_init`,
`astcenc_context_alloc[_isa]`, `astcenc_compress_image` / `_reset` / `_cancel`,
`astcenc_decompress_image`. `imgio::encode` uses it for 2D ASTC (see [encode.md](encode.md));
the features below are reachable through this API only, not through `EncodeOptions`.

- Multithreading: every thread calls `astcenc_compress_image` with its own index; the stages
  share work through `ParallelManager`s (`manage_avg`, LZ-RDO setup, `manage_compress`,
  LZ-RDO optimize), each with init / batches / wait / term as in C++. `term` runs its cleanup
  after releasing the manager lock (C++ runs it under the lock); otherwise a thread still
  leaving its task loop could deadlock against the cleanup's buffer lock.
- `validate_image`: a 3D image (`dim_z > 1`) with a 2D block is `BadParam`, as upstream.
- Alpha scaling, `a_scale_radius` (CLI `-a <radius>`, which also sets
  `ASTCENC_FLG_USE_ALPHA_WEIGHT`; radius clamped to 64): `compute_variance.rs` ports
  `astcenc_compute_variance.cpp` and `compute_averages`. The first thread allocates
  `input_alpha_averages` (one float per texel) and the task count; threads take strips of
  32 rows (3D: 16 rows x 16 slices) in granules of 16, cut them into 32x32 (3D: 16^3)
  regions, build zero-padded summed-area tables
  with the Brent-Kung prefix order (2D: X then Y; 3D: X, Y, Z), and write
  `sum / (2r + 1)^2` (3D: `^3`) per texel; all threads wait before the compression. Only the
  alpha lane is computed (C++ also builds RGB and N^2 lanes nobody reads). Loads: U8
  `a * (1 / 255)`, F16 / F32 alpha clamped to [0, 1] (F16 via the packs-saturating vector
  conversion), swizzle incl. ZERO / ONE. Compressor (2D blocks only): a block with no averaged
  alpha above `0.9 / (255 * (bx + 2(r - 1)) * (by + 2(r - 1)))` becomes a constant
  transparent black block (origin / min / mean / max zero, grayscale). 3D blocks compute the
  averages but do not use them (as upstream). `manage_compress.term` frees the table.
- Progress, `config.progress_callback: Option<AstcencProgressCallback>` (an
  `Arc<dyn Fn(f32) + Send + Sync>`; `AstcencConfig` is `Clone`, no longer `Copy`):
  `ParallelManager::init_progress(tasks, callback, start, end)`; `complete_task_assignment`
  reports `(1 - f) * start + f * end` (f = done / total) on the completing thread, outside the
  task lock and under a callback lock, when it exceeds the last report by
  `max(4096 / tasks * 100, 1)` or the stage is done. Compression reports 0..100 (0..33 with
  LZ-RDO), the LZ-RDO optimization 33..100; the averages and LZ-RDO setup stages report
  nothing. With several threads a worker may start the optimization before another reports
  33 (as in C++), so only each stage is monotonic.
- 3D input to the CLI: `-zdim <n>` loads `name_0.ext` .. `name_<n-1>.ext` as slices of one
  image (`load_uncomp_file`); 3D KTX (`pixel_depth`) / DDS (depth flag) inputs also load.
  The CLI rejects `-zdim` with a 2D block.

Tests (all byte identical, every ISA copy):
- `astcenc::cli_tests::alpha_scale_matches_astcenc_cli`: 18 `astcenc-avx2` files from
  `cutout_rgba_96x64` (transparent background, soft disc, opaque rectangle, ramp, 1..3 / 255
  specks): `-a 1/2/4/8` x 4x4 / 6x6 / 8x8 x fast / medium / thorough over `-cl` / `-cs`, five
  with `-rdo-lambda 20`, one `-cH` from FP16; 1 and 16 threads (the generator also checks the
  CLI's `-j 16` against `-j 1`).
- `astcenc::cli_tests::volume_matches_astcenc_cli`: 7 files of a 26x22x14 `-zdim 14` volume
  cut from the same image: 4x4x4 / 6x6x6 medium with and without `-rdo-lambda 20`, 3x3x3
  thorough, `-a 2` (4x4x4) and `-a 4 -rdo-lambda 20` (6x6x6).
- `astcenc::tests::compress_alpha_scale_matches_astcenc`: 180 C++ library hashes
  (`compress_alpha_scale_golden.txt`): 9 block sizes (6 2D, 3 3D) x radius 1 / 2 / 4 / 8 / 100 x
  4 profiles, rotating U8 / F16 / F32, qualities, swizzles (alpha from A, G, ZERO), the alpha
  weight flag and LZ-RDO; float specks of 0.0005..0.004 sit near the threshold (a 5% change of
  the averages breaks 15 hashes). 1-4 threads.
- `astcenc::tests::progress_callback_matches_parallel_manager`: the single-thread reports
  equal the C++ callback values (`compress_progress_golden.txt`, 1024x512 4x4 fastest, with
  and without LZ-RDO) on the calling thread; with 4 threads each stage rises to its end, on
  worker threads only.
- `astcenc::tests::compress_rejects_3d_image_with_2d_blocks`.

## Audit

- 2026-11-21: alpha scaling (`astcenc_compute_variance.cpp`, `manage_avg`, negligible alpha
  blocks), progress callback, `validate_image` 3D check, `term` outside the lock; CLI goldens
  for `-a` and `-zdim` 3D (incl. LZ-RDO), library goldens for alpha scaling and progress.
  Reference: astc-encoder 6.0.0 `astcenc_entry.cpp`, `astcenc_compute_variance.cpp`,
  `astcenc_internal_entry.h`, `astcenccli_toplevel.cpp`.
