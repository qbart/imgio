# Quality metrics (`imgio::metrics`)

Spec: compare two same-size uncompressed surfaces (formats may differ) on stored values
normalized like `convert` (unorm 0..1, snorm -1..1, floats as is), in f64.

- Per channel (channels of the first format) and aggregated over RGB and all channels: MSE,
  PSNR (`10 log10(peak^2 / mse)`, infinite when equal), max and mean absolute error, SSIM.
- SSIM = scikit-image `structural_similarity(data_range=peak, gaussian_weights=True,
  sigma=1.5, use_sample_covariance=False)`: 11x11 Gaussian (truncate 3.5), scipy `reflect`
  edges, K1 0.01, K2 0.03, mean without a 5 px border; NaN under 11x11 or when disabled.
  Aggregate = channel mean (`channel_axis`). Channels run in parallel.
- `peak` option (1 unorm, 2 snorm, max for HDR); `difference` renders `|a - b| * amplify` as
  RGBA8.

- `Thresholds { min_psnr, min_ssim, max_error, max_mean_error, max_mse, scope }`: `check`
  lists failed limits (`metric`, `channel`, value, limit) per channel (default) or on the RGB /
  all aggregate; NaN fails except SSIM, which is skipped when not computed.
- CLI `compare` (cli.md): limits as flags, `--scope`; two files or two directories (every
  image / texture under the reference matched by relative path: pass / fail / missing / extra /
  error), `--diff` (file, or a directory of `<path>.diff.png` for failures), `--json`
  (non-finite numbers are `null`: infinite PSNR or no SSIM), exit 1 on any failure, missing
  file or error (`--strict`: extras too).

## Audit

- tests/metrics.rs: 8 cases (RGB noise / JPEG, RGBA blur / identical, gray quantize / shift,
  R16 noise, RGBA16F HDR scale with peak = max) match numpy / scikit-image 0.26 within 1e-12
  relative (MSE, max, mean, SSIM, PSNR, aggregates); small images, disabled SSIM, mixed
  formats, size / peak / compressed errors, difference image; thresholds: per channel /
  RGB / all scopes, every metric, skipped SSIM.
- tests/cli.rs `compare_ci`: identical pair under strict limits, failed limit exit code,
  directory pass / fail / missing / extra counts in JSON, diffs only for failures, `--strict`.
