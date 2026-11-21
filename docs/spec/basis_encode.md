# Basis Universal encoder

`imgio::basis::encode_basis(images, opts) -> Result<Buffer>`: images to a `.basis` file with
the ported basisu `basis_compressor` (bit-exact with the C++ encoder of KTX-Software / ktx-rs).

## API (`src/basis/encode.rs`)

- `BasisCodec`: `Etc1s` (default), `UastcLdr4x4`, `UastcHdr4x4`, `AstcHdr6x6`,
  `UastcHdr6x6Intermediate`; `is_hdr()`.
- `BasisEncodeOptions` (`Default` = `basisu` tool defaults):
  `codec`, `tex_type: BasisTexType` (Tex2D), `etc1s_quality` 1..=255 (128, `-q`),
  `etc1s_level` 0..=6 (1, `-comp_level`), `uastc_level` 0..=4 (2), `uastc_rdo_lambda` (0 = no
  RDO, `-uastc_rdo_l`), `hdr_4x4_level` 0..=4 (1), `hdr_6x6_level` 0..=12 (2),
  `hdr_6x6_lambda` (0, `-lambda`), `srgb` (true: `set_srgb_options`), `mipmaps` (false),
  `mip_filter: mip::Filter` (Kaiser), `mip_wrap` (true), `y_flip` (false), `threads` (0 = all).
  Out-of-range values: `InvalidArgument`.
- Inputs: LDR codecs take RGBA8 (`Rgba8Unorm` / `Rgba8Srgb` as is; other uncompressed formats
  via `convert`, labels only; block-compressed ones decoded first). HDR codecs take linear
  float RGB (`convert` to RGBA32F with `srgb_transfer`, so sRGB 8-bit inputs are linearized);
  alpha ignored. Alpha is detected (any texel < 255), as basisu `m_check_for_alpha`.
- Errors: empty list / zero size / bad options / compressor read or validation failure
  (e.g. cubemap count not a multiple of 6, array sizes differ) -> `InvalidArgument`; other
  encoder failures -> `InvalidData`.
- Output depends on `threads` like basisu (ETC1S codebook threads, UASTC RDO jobs = min(threads,
  4), ASTC HDR 6x6 strips = threads when lambda is 0). UASTC LDR / HDR 4x4 blocks do not.

## Low level (`src/basis/ll`)

Copied from ktx-rs `src/basisu` (same names): `comp` (`basis_compressor`), `frontend`,
`backend`, `vq`, `huffman_enc`, `basis_file` (.basis writer), `etc_enc{,_tables}`, `enc`,
`gpu_texture`, `bc7enc`, `uastc_enc` (+ RDO), `uastc_hdr_4x4_enc`, `astc_hdr_enc`,
`astc_hdr_6x6_enc`, `astc_hdr_downsample_tables`, `job_pool`, and `scoped` (ktx-rs
`src/scoped.rs`, wasm-safe `thread::scope`). They use imgio's transcoder modules (identical
or superset of ktx-rs) and the `crate::mip` resampler (made `pub(crate)`).

Changes vs ktx-rs:
- `comp`: y flip and mipmap generation added back from C++ `basis_compressor`
  (`read_source_images`, `generate_mipmaps` via `image_resample` for LDR and HDR,
  `m_mip_{scale,filter,wrapping,fast,smallest_dimension}` params, C++ defaults; HDR mips use
  box and `clean_hdr_image` per level); HDR slices get their mip index; `clean_hdr_image` is a
  free function; the C++ warning prints are dropped; UASTC HDR 4x4 blocks run in jobs of 256
  on the job pool (C++ queues them too; ktx-rs ran them serially).
- `astc_hdr_6x6_enc::compress_photo`: strips run in parallel on the job pool (each strip owns
  its rows of `coded_blocks` / `final_astc_blocks` and its bit buffer; strips never read
  each other's blocks, as in C++).

Not ported (also absent in ktx-rs): source file loading, source resampling, user mip chains,
normal map renormalization, stats / output validation, KTX2 output (separate step), debug
images, OpenCL, XUASTC / ASTC LDR / XUBC7 encoders. `miniz.rs` (KTX2 zlib) and `ssim.rs`
(statistics) are not copied: nothing in the encoder uses them.

## Tests

- `src/basis/ll/tests/*` (ktx-rs `tests/basisu_*.rs`, golden data in
  `testdata/basisu_enc/basisu_*.txt`, hashes of the C++ output): comp, frontend (2), backend,
  etc, uastc_enc, uastc_rdo, uastc_hdr_4x4_enc, hdr6x6_enc, hdr6x6_parts, astc_hdr_enc,
  bc7enc, huffman_enc, vq (2): 15 tests, ~11.8k cases, all pass.
- `tests/basis_encode.rs`:
  - `matches_basisu_compressor`: `encode_basis` on the 13 comp golden inputs x the 13 configs
    the API can express (169 whole-file hashes) = C++ `basis_compressor`.
  - `matches_basisu_tool`: mipmaps (ETC1S, UASTC) and y flip vs the basisu v2.50 CLI
    (`testdata/basisu_enc/cli_*.basis`, `scripts/gen_basis_testdata.sh`): every level
    transcodes to identical pixels (2.50 adds a 31-byte key/value header extension).
  - Round trips of `testdata/images/raw` through `BasisFile::parse` + `transcode_to_buffer`:

    | Image | ETC1S q128 | ETC1S q255 | UASTC | UASTC RDO 1 |
    |---|---|---|---|---|
    | rgba_64x48 (noisy) RGB / A dB | 23.4 / 42.7 | 24.0 / 43.4 | 35.7 / 37.6 | 35.5 / 37.4 |
    | rgb_37x29 RGB dB | 21.7 | 22.5 | 35.2 | 35.2 |
    | mosaic_rgba_50x38 RGB / A dB | 19.8 / 46.7 | 19.8 / 46.7 | 65.2 / 59.9 | 55.7 / 59.9 |

    (imgio BC7 / ASTC 4x4 on rgba_64x48: ~37 dB.) hdr_40x24 relative RGB error: UASTC HDR 4x4
    0.031, ASTC HDR 6x6 0.043, 6x6i 0.043 (lambda 2000: 501 -> 396 bytes).
  - Mipmaps + 2D arrays (levels, sizes), HDR mipmaps, y flip, thread-count independence of
    UASTC HDR 4x4, F16 vs F32 input, bad options, compressed / R8 inputs.

Speed (512x512, 24 threads / 1 thread, MPix/s): ETC1S 2.9 / 1.5, UASTC 3.5 / 0.37, UASTC
RDO 1.0 0.6 / 0.13, UASTC HDR 4x4 3.2 / 0.40, ASTC HDR 6x6 3.1 / 0.52, 6x6i 2.9 / 0.53,
6x6i lambda 500 0.5 / 0.15.

## Audit

- 2026-11-21: encoder modules and golden tests copied from ktx-rs; `encode_basis` API; y flip
  and mipmaps from C++ `basisu_comp.cpp` / `basisu_enc.cpp` (`image_resample`); parallel
  UASTC HDR 4x4 blocks and ASTC HDR 6x6 strips. All golden tests pass.
