# Basis Universal encoder

`imgio::basis::encode_basis(images, opts) -> Result<Buffer>`: images to a `.basis` file with
the ported basisu `basis_compressor` (bit-exact with the C++ encoder of KTX-Software / ktx-rs,
and with the basisu v2.50 tool including its `.basis` key/value data).

## API (`src/basis/encode.rs`)

- `BasisCodec`: `Etc1s` (default), `UastcLdr4x4`, `UastcHdr4x4`, `AstcHdr6x6`,
  `UastcHdr6x6Intermediate`, `AstcLdr(AstcBlock)` (`-ldr_WxH`), `XuastcLdr(AstcBlock)`
  (`-ldr_WxHi`), `Xubc7` (`-xubc7`); all 14 `AstcBlock` sizes are basisu formats. `is_hdr()`,
  `tex_format()`.
- `BasisEncodeOptions` (`Default` = `basisu` tool defaults):
  `codec`, `tex_type: BasisTexType` (Tex2D), `etc1s_quality` 1..=255 (128, `-q`),
  `etc1s_level` 0..=6 (1, `-comp_level`), `uastc_level` 0..=4 (2), `uastc_rdo_lambda` (0 = no
  RDO, `-uastc_rdo_l`), `hdr_4x4_level` 0..=4 (1), `hdr_6x6_level` 0..=12 (2),
  `hdr_6x6_lambda` (0, `-lambda`), `srgb` (true: `set_srgb_options`), `mipmaps` (false),
  `mip_filter: mip::Filter` (Kaiser), `mip_wrap` (true), `y_flip` (false), `threads` (0 = all),
  `basisu_v2_keys` (true: the 2.50 key/value data, see below).
  ASTC LDR / XUASTC LDR (CLI `-effort E -quality Q`, mapped with
  `set_format_mode_and_quality_effort(mode, Q, E, false)` like the tool): `astc_ldr_effort`
  0..=10 (3), `xuastc_quality` 1..=100 (100 = no DCT, lossless supercompression; < 100 enables
  weight grid DCT at that quality + lossy supercompression; ignored by ASTC LDR),
  `xuastc_lossy_supercompression: Option<bool>` (`-xy` / `-xyd` override), `xuastc_syntax:
  XuastcSyntax` (`Zstd` default / `Arith` / `Hybrid`), `xuastc_deblocking: XuastcDeblocking`
  (`ScdFilterLargest` default = SCD + `DeblockFilterID` for blocks >= 80 texels; `Off`,
  `ScdFilterAll`, `ScdOnly`, `FilterLargest`, `FilterAll` = basisu `xuastc_ldr_deblocking_mode`
  1, 0, 2..5), `ldr_channel_weights: Option<[u32; 4]>` (1..=256; `None` = 9,11,1,11 if `srgb`
  else 1,1,1,1, as `-srgb` / `-linear`; also used by XUBC7). `srgb` also sets the ASTC sRGB
  decode mode and the file flag. Not exposed: blurring, sharpening, heavy subset usage, SCD pass count, lossy
  supercompression thresholds, encoder selection (`cAuto`), subset / dual plane disabling.
  Out-of-range values: `InvalidArgument`.
  XUBC7 (CLI `-xubc7 -quality Q -effort E -xubc7_rdo R -xubc7_stripes S -xubc7_bc7f |
  -xubc7_bc7e_scalar_level L`, mapped like the tool: `set_format_mode(cXUBC7)`, params, then
  `set_format_mode_and_quality_effort(cXUBC7, Q, E, false)`): `xubc7_quality` 1..=100 (100 =
  lossless weights, default), `xubc7_effort` 0..=10 (3), `xubc7_rdo_level` 0..=100 (0),
  `xubc7_encoder: Xubc7Encoder` (`Bc7f` default / `Bc7e(level 0..=6)`; `-xubc7_bc7e` = level 2),
  `xubc7_stripes` 1..=16 (8, desired; one stripe below 128 rows), `ldr_channel_weights`, and
  `srgb` -> `m_perceptual` (bc7e perceptual mode). The XUBC7 encoder runs on all cores
  (`crate::parallel`) whatever `threads` is; its output does not depend on it.
- Inputs: LDR codecs take RGBA8 (`Rgba8Unorm` / `Rgba8Srgb` as is; other uncompressed formats
  via `convert`, labels only; block-compressed ones decoded first). HDR codecs take linear
  float RGB (`convert` to RGBA32F with `srgb_transfer`, so sRGB 8-bit inputs are linearized);
  alpha ignored. Alpha is detected (any texel < 255), as basisu `m_check_for_alpha`.
- Errors: empty list / zero size / bad options / compressor read or validation failure
  (e.g. cubemap count not a multiple of 6, array sizes differ) -> `InvalidArgument`; errors of
  the ASTC / XUASTC LDR block encoder (`xuastc::enc::compress_image`) as returned; other
  encoder failures -> `InvalidData`.
- `.basis` key/value data (basisu 2.50 `basisu_basis_file.cpp`): an 8-byte header (`m_sig`
  0x564B u16, `m_num` u32, `m_crc16` of the entries u16) + per entry key length u8, value
  length u32, key without NUL, raw value; placed right after the 77-byte header
  (`m_extended_file_ofs/size`), the slice descriptors follow. Entries in order: the params'
  `m_key_values` (`DeblockFilterID` = "1\0" when the deblocking mode makes
  `cfg.m_scd_will_postfilter` true), then `BasisULibVers` = "2.50\0" (31 bytes in total
  alone), then for HDR `KTXmapRange` (8 bytes: f32 bits of the HDR image scale, offset 0).
  `basisu_v2_keys: false` drops the last two (the pre-2.0 / KTX-Software layout, byte-identical
  to before); `DeblockFilterID` is always written since it changes decoding.
  `ktx2::encode_basis` copies `DeblockFilterID` into the KTX2 key/values (as basisu).
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

basisu v2.50 additions in `comp` / `basis_file`: `set_format_mode`, the XUASTC / ASTC LDR part
of `set_format_mode_and_quality_effort`, `pick_format_mode` for tex formats 5..32 (block size
from the format; slices padded to it), `encode_slices_to_xuastc_or_astc_ldr` (config mapping
in `BasisCompressor::xuastc_or_astc_ldr_config`: `cAuto` encoder choice by effort (0: astcf
except 4x4; 1-2: astcf; 3: astcf + basisu merged if block > 64 texels or any source alpha;
>= 4: both), sharpening, DCT only for XUASTC with `m_quality_level` in [1, 100), SCD passes by
effort 0,0,2,8,10,12,14,16,20,20,32 (0 disables SCD); per slice: crop to the original size,
`compress_image`, pack the logical blocks with `pack_astc_block` (ASTC LDR payload = the
packed blocks, XUASTC = the bitstream), CRC16), key/value data (`m_key_values`,
`create_key_value_data`, `check_key`, `add_key_value`), `extract_source_blocks` skipped. Alpha:
as UASTC (no alpha slices; per-slice `has_alpha` flag, header `HasAlphaSlices`); sRGB flag =
`m_ktx2_and_basis_srgb_transfer_function`. `BasisuBackendOutput::tex_format` is now the public
`BasisTexFormat` (all 34 formats). astcenc selections behave as with `BASISU_SUPPORT_ASTCENC`
= 0 (ignored).

XUBC7 in `comp`: `m_format_mode` (`format_mode`, set by `set_format_mode`) and the
`m_xubc7_{effort_level, rdo_level, num_stripes, encoder, bc7e_scalar_level}` params (C++
defaults 3, 0, 8, bc7f, 2; clamped on assignment), the XUBC7 branches of
`set_format_mode_and_quality_effort` (effort clamp, quality -> `m_quality_level` in [1, 100]),
`pick_format_mode` (XUBC7 first: `uastc`, no HDR, 4x4), `extract_source_blocks` skipped,
`encode_slices_to_xubc7`: `xubc7_pack_options` = `xbc7::EncodeParams::from_basisu` (the
`pack_options` setup), per slice: crop to the original size, `xubc7::encoder::encode_blocks`,
every coded block must `pack_bc7` (C++ fills the BC7 `gpu_image`; unused without KTX2 / stats),
payload = the XBC7 stream, CRC16. Alpha and sRGB flags as XUASTC.

Not ported (also absent in ktx-rs): source file loading, source resampling, user mip chains,
normal map renormalization, stats / output validation, KTX2 output (separate step), debug
images, OpenCL. `miniz.rs` (KTX2 zlib) and `ssim.rs`
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
    (`testdata/basisu_enc/cli_*.basis`, `scripts/gen_basis_testdata.sh`): identical files;
    with `basisu_v2_keys: false` 31 bytes shorter and every level transcodes the same.
  - `hdr_v2_keys`: UASTC HDR 4x4 / ASTC HDR 6x6 / 6x6i key/value data = the 2.50 tool's
    (`testdata/basis/*hdr*.basis`).
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

ASTC LDR / XUASTC LDR goldens (`testdata/basisu_enc/cli_{astc,xuastc}*.basis`, basisu 2.50
`-basis -ldr_WxH[i]` + options; RGBA = rgba_64x48 / mosaic_rgba_50x38, RGB = rgb_37x29):

| File | Options | Level 0 RGB / A dB |
|---|---|---|
| cli_astc4x4_rgba_64x48 | `-ldr_4x4` | 34.6 / 37.9 |
| cli_astc4x4_e0_rgb_37x29 | `-ldr_4x4 -effort 0` (bc7f path) | 33.8 |
| cli_astc6x6_linear_mip_rgb_37x29 | `-ldr_6x6 -linear -mipmap` | 31.8 |
| cli_astc8x6_e1_rgba_64x48 | `-ldr_8x6 -effort 1` | 28.6 / 33.6 |
| cli_astc12x12_rgba_64x48 | `-ldr_12x12` (SCD, `DeblockFilterID`) | 22.5 / 30.3 |
| cli_xuastc4x4_zstd_rgb_37x29 | `-ldr_4x4i` (zstd) | 34.0 |
| cli_xuastc5x5_e5_mip_rgb_37x29 | `-ldr_5x5i -effort 5 -mipmap` | 29.0 |
| cli_xuastc6x6_arith_mip_rgba_64x48 | `-ldr_6x6i -xuastc_arith -mipmap` | 33.0 / 35.7 |
| cli_xuastc6x5_hybrid_linear_rgba_64x48 | `-ldr_6x5i -xuastc_hybrid -linear` | 32.0 / 35.2 |
| cli_xuastc6x6_q60_rgba_64x48 | `-ldr_6x6i -quality 60` (DCT, lossy) | 31.1 / 37.7 |
| cli_xuastc6x6_hybrid_q60_rgba_64x48 | `-ldr_6x6i -xuastc_hybrid -quality 60` | 31.1 / 37.7 |
| cli_xuastc8x6_arith_q60_mosaic_rgba_50x38 | `-ldr_8x6i -xuastc_arith -quality 60` | 40.7 / 47.4 |
| cli_xuastc12x12_rgba_64x48 | `-ldr_12x12i` (SCD, `DeblockFilterID`) | 22.5 / 30.3 |
| cli_xuastc12x12_nodeblock_rgba_64x48 | `-ldr_12x12i -xuastc_ldr_no_deblocking` | 22.4 / 30.2 |

- Unit tests `basis::encode::tests`: `writer_matches_basisu_goldens` injects each golden's
  `compress_image` outputs (payload; logical blocks = payload unpacked, or the XUASTC slice
  transcoded to ASTC and unpacked) and checks the whole file byte for byte (1 and 3 threads):
  sources, mipmaps, alpha / sRGB flags, config -> `DeblockFilterID`, ASTC repacking, writer.
  `writer_without_v2_keys`, `config_mapping` (effort -> encoders / SCD passes, deblocking modes,
  DCT / lossy rules, weights, syntax), `encoder_errors_propagate`.
XUBC7 goldens (`testdata/basisu_enc/cli_xubc7_*.basis`, basisu 2.50 `-basis -xubc7` +
options; mosaic RGB = `mosaic_rgb_50x38`, raw copy added to `testdata/images/raw`). The noisy
`rgba_64x48` / `rgb_37x29` default files mostly use the raw BC7 "tiny mip" fallback; the mosaics
and Q < 100 code real XBC7 streams:

| File | Options | Bytes | Level 0 RGB / A dB |
|---|---|---|---|
| cli_xubc7_mip_rgba_64x48 | `-mipmap` | 4338 | 36.6 / 39.0 |
| cli_xubc7_rgb_37x29 | defaults | 1414 | 35.7 / 72.4 |
| cli_xubc7_mosaic_rgba_50x38 | defaults | 775 | 56.7 / 55.2 |
| cli_xubc7_q60_rgba_64x48 | `-quality 60` | 2703 | 31.6 / 39.3 |
| cli_xubc7_e0_mosaic_rgb_50x38 | `-effort 0` | 1032 | 34.4 / 54.8 |
| cli_xubc7_e10_mip_mosaic_rgb_50x38 | `-effort 10 -mipmap` | 2057 | 34.2 / 61.7 |
| cli_xubc7_rdo60_mosaic_rgba_50x38 (+ `.ktx2`) | `-xubc7_rdo 60` | 757 | 54.1 / 59.9 |
| cli_xubc7_bc7e4_linear_rgba_64x48 | `-xubc7_bc7e_scalar_level 4 -linear` | 3118 | 37.6 / 39.3 |
| cli_xubc7_bc7e_q80_e5_mosaic_rgba_50x38 | `-xubc7_bc7e -quality 80 -effort 5` | 796 | 55.8 / 53.6 |

(Opaque sources decode with alpha < 255 in places: the coded BC7 blocks use alpha modes there too; identical bytes to C++, and the transcoder is golden-checked against C++.)

- `tests/basis_encode_xubc7.rs`: `matches_basisu_tool` (all 9 byte-exact, 1 and 4 threads),
  `ktx2_matches_basisu_tool` (`ktx2::encode_basis` with the CLI's key/values = the `-ktx2`
  file), `round_trip` (PSNR >= golden - 0.5 dB, RGBA8 transcode = BC7 transcode decoded, all
  levels, Q 30 and RDO 80 shrink), `bad_options`. Unit test
  `basis::encode::tests::xubc7_config_mapping` (options -> `xbc7::EncodeParams`).
- `tests/basis_encode_xuastc.rs`: `goldens_decode` (golden PSNRs, calibrates the thresholds),
  `bad_options`; `matches_basisu_tool` (byte-exact `encode_basis`, 1 and 4 threads) and
  `round_trip` (PSNR >= golden - 0.5 dB, all levels transcode, quality 30 < 100 in size).

## Audit

- 2026-11-21: encoder modules and golden tests copied from ktx-rs; `encode_basis` API; y flip
  and mipmaps from C++ `basisu_comp.cpp` / `basisu_enc.cpp` (`image_resample`); parallel
  UASTC HDR 4x4 blocks and ASTC HDR 6x6 strips. All golden tests pass.
- 2026-11-21: basisu 2.50 `.basis` key/value data (default on: ETC1S / UASTC files now equal
  the 2.50 tool's byte for byte), ASTC LDR / XUASTC LDR in `basis_compressor` + API + 14 CLI
  goldens; everything around the block encoder verified byte-exact by payload injection.
- 2026-11-21: XUBC7 wired into `basis_compressor` (`encode_slices_to_xubc7`, `format_mode`,
  `m_xubc7_*` params) + `BasisCodec::Xubc7` API; 9 CLI goldens + 1 KTX2 twin byte-exact on the
  first run.
