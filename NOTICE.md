# NOTICE

imgio is licensed under Apache-2.0 (see [LICENSE](LICENSE)). License texts for
third-party code and test data are in [LICENSES/](LICENSES).

## Third-party code

Translated or copied files carry the original copyright and SPDX identifier in their header and
state that they were modified (Apache-2.0 §4(b)).

| Path | Derived from | Copyright | License |
|---|---|---|---|
| `src/bc/bc6h.rs`, `src/bc/bc7.rs`, `src/bc/tables.rs`, `src/astc/helpers.rs`, `src/astc/tables.rs` | [Basis Universal](https://github.com/BinomialLLC/basis_universal) `basisu_gpu_texture.cpp` and tables, via [ktx-rs](https://github.com/qbart/ktx-rs) `src/basisu` | Binomial LLC | Apache-2.0 |

| `src/basis/ll/*` | Basis Universal transcoder, via ktx-rs `src/basisu`; `etc1f*.rs`, `bc7f*.rs`, `bc7u.rs`, `fxt1.rs` directly from `basisu_transcoder.cpp` / `basisu_etc1_mods.inl` | Binomial LLC | Apache-2.0 |
| `src/basis/ll/{astc_hdr_6x6_enc,astc_hdr_downsample_tables,astc_hdr_enc,backend,basis_file,bc7enc,comp,enc,etc_enc,etc_enc_tables,frontend,gpu_texture,huffman_enc,job_pool,uastc_enc,uastc_hdr_4x4_enc,vq}.rs` | Basis Universal encoder (`encoder/basisu_*.cpp`), via ktx-rs `src/basisu` | Binomial LLC | Apache-2.0 |
| `src/basis/ll/{comp,basis_file}.rs` (v2 parts), `src/basis/encode.rs` (ASTC / XUASTC LDR options) | Basis Universal v2.50 `basisu_comp.{h,cpp}` (`encode_slices_to_xuastc_or_astc_ldr`, `set_format_mode`, `set_format_mode_and_quality_effort`, XUASTC LDR params), `basisu_basis_file.cpp` (key/value data), `basisu_tool.cpp` (option mapping) | Binomial LLC | Apache-2.0 |
| `src/basis/xuastc/enc/*` | Basis Universal v2.50 encoder `basisu_astc_ldr_encode.cpp`, `basisu_astc_ldr_fencode.cpp`, `basisu_astc_ldr_common.cpp`, `basisu_astc_ldr_pseudoinv_tab.inl`, parts of `basisu_enc.h` (`rand`, `bitwise_coder`), `basisu_containers.h` (`hash_map`) and `basisu_transcoder_internal.h` (`arith_enc`, `grid_weight_dct`); `rand.rs` / `containers.rs` also reproduce the output order of libstdc++ (GCC 13) `std::mt19937`, its int / real / normal distributions and `std::priority_queue` (algorithms reimplemented, no code copied) | Binomial LLC | Apache-2.0 |
| `src/basis/ll/scoped.rs` | ktx-rs `src/scoped.rs` | KTX Rust port contributors | Apache-2.0 |
| `src/basis/ll/tests/*`, `testdata/basisu_enc/basisu_*.txt` | ktx-rs `tests/basisu_*.rs` and their golden data (hashes of C++ basisu encoder output) | Mark Callow, The Khronos Group Inc. and KTX-Software contributors | Apache-2.0 |
| `src/basis/xubc7/*`, `src/basis/deblock.rs`, `src/basis/pvrtc1_enc.rs` | Basis Universal `basisu_transcoder.cpp`, `basisu_xbc7_decoder.{h,inl}` | Binomial LLC | Apache-2.0 |
| `src/basis/xuastc/*` | Basis Universal `basisu_transcoder.cpp` (`astc_ldr_t`), `basisu_transcoder_internal.h` (`arith`), `basisu_idct.h`, `basisu_astc_cfgs.inl` | Binomial LLC | Apache-2.0 |
| `src/atc.rs`, `src/fxt1.rs`, `src/pvrtc.rs` | Basis Universal `basisu_gpu_texture.cpp`, `basisu_pvrtc1_4.{h,cpp}` (encoder library) | Binomial LLC | Apache-2.0 |
| `src/astcenc/*` | [astc-encoder](https://github.com/ARM-software/astc-encoder) (astcenc) 6.0.0, via ktx-rs `src/astcenc` (5.x port, `#[isa]` attributes rewritten as `isa_fn!` wrappers) | Arm Limited | Apache-2.0 |
| `src/astcenc/compute_variance.rs` | astc-encoder 6.0.0 `astcenc_compute_variance.cpp` and `compute_averages` / the alpha scaling hooks of `astcenc_entry.cpp` | Arm Limited | Apache-2.0 |
| `src/astcenc/enc/lz_optimize.rs` | astc-encoder 6.0.0 `astcenc_lz_optimize.cpp` (LZ rate-distortion optimization) and its hooks in `astcenc_entry.cpp` / `astcenc_compress_symbolic.cpp` | Epic Games, Inc. and Arm Limited | Apache-2.0 |
| `src/astcenc/tests.rs`, `tools/astcenc-golden/compress.cpp` | ktx-rs `tests/astcenc_compress.rs` and `scripts/golden/astcenc_compress_golden.cpp` (the generator updated for the astc-encoder 6.0 API) | Mark Callow, The Khronos Group Inc. and KTX-Software contributors | Apache-2.0 |
| `src/bc7e.rs` | Basis Universal v2.50 `encoder/basisu_bc7e_scalar.cpp` (scalar port of `bc7e.ispc`) | Binomial LLC | Apache-2.0 |
| `src/basis/xubc7/encoder.rs` | Basis Universal v2.50 `encoder/basisu_xbc7_encode.cpp` (XBC7 / XUBC7 encoder) and the `basisu_comp.cpp` XUBC7 option setup | Binomial LLC | Apache-2.0 |
| `src/mip/resampler.rs`, `src/mip/filters.rs` | Basis Universal `basisu_resampler.cpp`, `basisu_resample_filters.cpp`, via ktx-rs `src/basisu` | Binomial LLC | Apache-2.0 |
| `src/ktx2/basis_writer.rs` | Basis Universal `basisu_comp.cpp` (`create_ktx2_file`, `get_dfd`, DFD templates) | Binomial LLC | Apache-2.0 |
| `src/ktx2/dfd.rs` | [KTX-Software](https://github.com/KhronosGroup/KTX-Software) dfdutils (`createdfd.c`, `vk2dfd.inl`), via ktx-rs `src/dfd` | The Khronos Group Inc. | Apache-2.0 |
| `src/gpu/vulkan.rs` | [KTX-Software](https://github.com/KhronosGroup/KTX-Software) `lib/src/vkloader.c`, `vk_funcs.c`, via [ktx-rs](https://github.com/qbart/ktx-rs) `src/vkloader.rs` | Mark Callow, The Khronos Group Inc. and KTX-Software contributors | Apache-2.0 |
| `src/gpu/gl.rs`, `src/gpu/gl_format.rs` | KTX-Software `lib/src/glloader.c`, `gl_funcs.c`, `vk2gl.h`, via ktx-rs `src/glloader.rs`, `src/glupload.rs`, `src/vk2gl.rs` | Mark Callow, The Khronos Group Inc. and KTX-Software contributors | Apache-2.0 |
| `tests/gpu_vulkan.rs`, `tests/gpu_gl.rs`, `tests/gpu_common/*` | ktx-rs `tests/vkloader.rs`, `tests/glloader.rs`, `tests/glloader_es.rs`, `tests/gl_common/mod.rs` | Mark Callow, The Khronos Group Inc. and KTX-Software contributors | Apache-2.0 |
| `src/etc.rs` | [ktx-rs](https://github.com/qbart/ktx-rs) `src/etcunpack.rs` (port of [KTX-Software](https://github.com/KhronosGroup/KTX-Software) `etcunpack.cxx`) | Mark Callow, The Khronos Group Inc. and KTX-Software contributors | Apache-2.0 |

### Basis Universal NOTICE

Basis Universal™ Supercompressed GPU Texture Compression Library
Copyright © 2016–2026 Binomial LLC.
"Basis Universal" is a trademark of Binomial LLC.

## Dependencies

| Crate | Use | License |
|---|---|---|
| [vas](https://crates.io/crates/vas) | containers (`Buffer`) | MIT |
| [bytemuck](https://crates.io/crates/bytemuck) | SIMD vector casts in the astcenc kernels | Zlib OR Apache-2.0 OR MIT |
| [zstd-pure-rs](https://crates.io/crates/zstd-pure-rs) | KTX2 Zstandard supercompression (writer) | BSD-3-Clause OR GPL-2.0-only (used under BSD-3-Clause) |
| [ruzstd](https://crates.io/crates/ruzstd) | Zstandard decoding of KTX2 levels and XUASTC LDR streams | MIT |
| [ash](https://crates.io/crates/ash) | optional (feature `vulkan`): Vulkan types and command signatures; tests: Vulkan device setup | MIT OR Apache-2.0 |
| [libloading](https://crates.io/crates/libloading) | optional (features `vulkan`, `opengl`): looks up Vulkan / GL commands in the process; tests: global loading of the system libraries | ISC |
| [jxl-oxide](https://crates.io/crates/jxl-oxide) | optional (feature `jxl`): JPEG XL decoding (with its `moxcms` ICC and `rayon` thread features) | MIT OR Apache-2.0 |
| [zune-jpegxl](https://crates.io/crates/zune-jpegxl), [zune-core](https://crates.io/crates/zune-core) | optional (feature `jxl`): lossless JPEG XL encoding | MIT OR Apache-2.0 OR Zlib |
| [hayro-jpeg2000](https://crates.io/crates/hayro-jpeg2000) | optional (feature `jpeg2000`): JPEG 2000 decoding (with `fearless_simd`) | Apache-2.0 OR MIT |
| [miniz_oxide](https://crates.io/crates/miniz_oxide) | optional (feature `vdb`): zlib streams of OpenVDB grids; tests: inflates the stored uncompressed VDB file | MIT OR Zlib OR Apache-2.0 |
| [khronos-egl](https://crates.io/crates/khronos-egl) | tests only: surfaceless EGL contexts for the GL upload tests | MIT OR Apache-2.0 |
| [bcdec_rs](https://crates.io/crates/bcdec_rs) | tests only: reference BC6H/BC7 decoder | MIT |
| Basis Universal C++ (`ref/basisu`) | `tools/basisu-golden` only (offline generator, not linked into the crate) | Apache-2.0 |
| [ktx-rs](https://crates.io/crates/ktx-rs) | `tools/astc-golden` only (not a crate dependency): astcenc reference | Apache-2.0 AND BSD-3-Clause AND MIT AND Zlib |
| OpenVDB 10.0.1 C++ (Ubuntu 24.04 packages) | `tools/vdb-golden/gen.cpp` only (offline generator of `testdata/file/vdb`, not linked); its headers documented the file layout for imgio's own reader (no code copied) | MPL-2.0 |
| libjxl, OpenJPEG (via imagecodecs), Pillow 13, `opj_compress` 2.5.0 | `scripts/gen_file_testdata_jxlj2kvdb.py` only (offline reference encoders / decoders, not linked) | BSD-3-Clause, BSD-2-Clause, MIT-CMU |
| astc-encoder 6.0.0 C++ (`ref/astcenc`) | `tools/astcenc-golden`, `scripts/gen_astcenc_golden.sh`, `scripts/gen_astcenc_testdata.py` only (offline generators, not linked into the crate) | Apache-2.0 |

## Test data

Not included in the published crate.

| Path | Source | License |
|---|---|---|
| `testdata/bc/*.bc` | random blocks generated by `scripts/gen_bc_testdata.py` (imgio) | Apache-2.0 |
| `testdata/bc/*.expected` | decodes of the above by Pillow 13 (generated, no Pillow code) | Apache-2.0 |
| `testdata/etc/etcdec_golden.txt` | ktx-rs `tests/resources/golden/etcunpack.txt` (hashes of Ericsson etcdec output) | Apache-2.0 |
| `testdata/astc/*` | generated by `tools/astc-golden` (synthetic images, random blocks; ktx-rs astcenc output) | Apache-2.0 |
| `testdata/images/*` | generated by `scripts/gen_images.py` (imgio), incl. raw RGBA8 copies in `raw/` | Apache-2.0 |
| `testdata/encode/golden.txt` | hashes of C++ Basis Universal v2.50 block encoder output (`tools/basisu-golden/encode.cpp`) | Apache-2.0 |
| `testdata/encode/bc7e_golden.txt` | hashes of C++ Basis Universal v2.50 `bc7e_scalar` output (`tools/basisu-golden/bc7e.cpp`) | Apache-2.0 |
| `testdata/encode/xubc7_golden.txt` | sizes and hashes of C++ Basis Universal v2.50 `xbc7::pack_image` (XUBC7) streams of `testdata/images/raw` and derived crops / tilings (`tools/basisu-golden/xubc7_encode.cpp`) | Apache-2.0 |
| `testdata/basisu_enc/basisu_astc_ldr_common.txt` | hashes of C++ Basis Universal v2.50 `astc_ldr` common encoder API output on synthetic blocks (`tools/basisu-golden/astc_ldr_common.cpp`) | Apache-2.0 |
| `testdata/basisu_enc/basisu_astc_ldr_block.txt` | hashes of the C++ Basis Universal v2.50 XUASTC LDR full encoder block-level functions on synthetic blocks (`tools/basisu-golden/astc_ldr_block.cpp`) | Apache-2.0 |
| `testdata/basisu_enc/basisu_astc_ldr_image.txt`, `testdata/basisu_enc/basisu_astcf.txt` | hashes of the candidate lists of the C++ Basis Universal v2.50 full ("basisu") and fast ("astcf") ASTC LDR image encoders on `testdata/images`, from a debug-hooked build (`docs/spec/xuastc_encode.md`) | Apache-2.0 |
| `testdata/basisu_enc/scd_*` | crops of `testdata/images` (`rgba_64x48` 36x24 at (10,8), `mosaic_rgba_50x38` 22x15 at (5,5)) and the zstd-compressed candidate dumps of the Basis Universal v2.50 XUASTC LDR encoder before SCD, from a debug-hooked build (`docs/spec/xuastc_encode.md`) | Apache-2.0 |
| `testdata/basis/*.basis` | `testdata/images` encoded by the Basis Universal v2.50 CLI (`scripts/gen_basis_testdata.sh`) | Apache-2.0 |
| `testdata/basis/golden*.txt` | hashes of C++ Basis Universal v2.50 transcoder output (also with decode flags) and of its `gpu_image::unpack` decodes (`tools/basisu-golden`) | Apache-2.0 |
| `testdata/basisu_enc/cli_*.basis`, `testdata/basisu_enc/cli_*.ktx2` | `testdata/images` encoded by the Basis Universal v2.50 CLI (`scripts/gen_basis_testdata.sh`): ETC1S / UASTC, ASTC LDR / XUASTC LDR (`cli_astc*`, `cli_xuastc*`) and XUBC7 (`cli_xubc7*`, incl. one `-ktx2` twin) encoder references | Apache-2.0 |
| `testdata/ktx2/*.ktx2` | `testdata/images` encoded by the Basis Universal v2.50 CLI (`scripts/gen_basis_testdata.sh`) | Apache-2.0 |
| `testdata/ktx2/golden*.txt` | hashes of C++ Basis Universal v2.50 `ktx2_transcoder` output (also with decode flags) | Apache-2.0 |
| `testdata/ktx2/mips/*` | `testdata/images` PNGs mipmapped by `ktx create --generate-mipmap` (ktx-rs port; `tools/ktx2-golden/gen.py`) | Apache-2.0 |
| `testdata/ktx2/basis/*.basis` | `.basis` twins of `testdata/ktx2/*.ktx2`, same Basis Universal v2.50 CLI options (`scripts/gen_basis_testdata.sh`) | Apache-2.0 |
| `testdata/ktx2/plain/*` | synthetic raw images (`tools/ktx2-golden/gen.py`) packed by `ktx create --raw` (KTX-Software CLI, ktx-rs port) | Apache-2.0 |
| `testdata/astc/files/*` | `testdata/images` raw image (as TGA) encoded / decoded by ARM astcenc 6.0 (`ref/astcenc`) | Apache-2.0 |
| `testdata/astcenc/compress_golden.txt`, `testdata/astcenc/compress_lz_rdo_golden.txt` | hashes of C++ astc-encoder 6.0.0 compressor output on synthetic images, without and with LZ-RDO (`tools/astcenc-golden/compress.cpp`, `scripts/gen_astcenc_golden.sh`) | Apache-2.0 |
| `testdata/astcenc/compress_alpha_scale_golden.txt`, `testdata/astcenc/compress_progress_golden.txt` | hashes of C++ astc-encoder 6.0.0 compressor output with alpha scaling (`a_scale_radius`) on synthetic images with transparent regions, and the values its progress callback reports (`tools/astcenc-golden/compress.cpp`, `scripts/gen_astcenc_golden.sh`) | Apache-2.0 |
| `testdata/astcenc/cli/*.astc` | `testdata/images/raw` images encoded by the ARM astc-encoder 6.0.0 CLI (`astcenc-avx2`, `scripts/gen_astcenc_testdata.py`); `*_rdo*.astc` with `-rdo-lambda`, `tiled_rgba_320x240_*` from a 320x240 tiling of `rgba_64x48` made by the script; `cutout_96x64_*` with `-a <radius>` (alpha scaling) from `cutout_rgba_96x64` (`*_H_*` as FP16 DDS); `vol_26x22x14_*` 3D blocks from a `-zdim 14` TGA slice stack cut from `cutout_rgba_96x64` by the script | Apache-2.0 |
| `testdata/ktx1/*` | KTX2 test files exported by the Basis Universal v2.50 CLI (`basisu -export_ktx`; `cube_32.ktx2` from `testdata/images/rgba_32x32.png`) | Apache-2.0 |
| `testdata/dds/*.dds` | `testdata/images` PNGs written by the Basis Universal v2.50 CLI (`basisu -dds`) | Apache-2.0 |
| `testdata/bench/*.ktx2` | `scripts/gen_images.py` bench image encoded by the Basis Universal v2.50 CLI | Apache-2.0 |
| `testdata/file/jxl/*` | `testdata/images/raw` images and crops / gray / 16-bit / float derivatives encoded by libjxl via imagecodecs 2026.11.11 (`jpegxl_encode`), `*.raw` = libjxl decodes (`scripts/gen_file_testdata_jxlj2kvdb.py`) | Apache-2.0 |
| `testdata/file/jpeg2000/*` | `testdata/images/raw` images and derived gray / 12- / 16-bit / YCbCr variants encoded by OpenJPEG (imagecodecs `jpeg2k`, Pillow 13 with OpenJPEG 2.5.4, `opj_compress` 2.5.0 CLI), expected pixels from the source or OpenJPEG's decode (`scripts/gen_file_testdata_jxlj2kvdb.py`) | Apache-2.0 |
| `testdata/file/vdb/*` | synthetic grids written by the C++ OpenVDB 10.0.1 library (`tools/vdb-golden/gen.cpp`), expected values / grid info from OpenVDB reading them back (`scripts/gen_file_testdata_jxlj2kvdb.py`) | Apache-2.0 |
| `testdata/file/bmp/*` | `testdata/images/raw` images (and Pillow median-cut palettes of them) written by Pillow 13 (`pil_*`), ffmpeg 6.1 (`ff_*`) or built from the BMP spec by `scripts/gen_file_testdata_own.py`; `expected/` decodes checked against Pillow / ffmpeg (`cases.txt`); `enc/` imgio encoder output checked against Pillow and ffmpeg | Apache-2.0 |
| `testdata/file/tga/*` | `testdata/images/raw` images written by Pillow 13 (`pil_*`), ffmpeg 6.1 (`ff_*`) or built from the TGA 2.0 spec by `scripts/gen_file_testdata_own.py`; `expected/` checked against Pillow / ffmpeg; `enc/` imgio encoder output checked against Pillow and ffmpeg | Apache-2.0 |
| `testdata/file/hdr/*` | `testdata/images/raw/hdr_40x24.rgba16f` written by ffmpeg 6.1 (`ff_*`), imagecodecs 2026.11.11 `rgbe_encode` (`ic_*`) or built from the Radiance spec by `scripts/gen_file_testdata_own.py`; `expected/` checked against ffmpeg / imagecodecs; `enc/` imgio encoder output checked against both | Apache-2.0 |
