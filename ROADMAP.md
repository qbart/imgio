# Roadmap

- [x] Scaffold: crate, Makefile, licenses, `Error`/`Result`
- [x] Core types: `Format`, `Buffer` (vas)
- [x] Image/surface/mip types
- [x] Raw RGBA conversion (8-bit, 16F, 32F)
- [x] BC1-BC5 decode
- [x] BC6H / BC7 decode (multithreaded, tested vs Pillow and bcdec_rs)
- [x] ETC1 / ETC2 / EAC decode
- [x] ASTC decode (LDR, HDR), 2D blocks
- [x] Basis Universal transcoder (.basis): ETC1S, UASTC LDR, UASTC HDR 4x4/6x6i, ASTC HDR 6x6 -> all targets (incl. FXT1 from ETC1S)
- [x] Basis Universal v2 payloads: XUASTC LDR, ASTC LDR, XUBC7 -> all targets (bit-exact vs C++)
- [x] BCn / ASTC / ETC / PVRTC1 encoders (multithreaded)
- [x] Basis Universal encoders -> `.basis`: ETC1S, UASTC LDR 4x4 (+ RDO), UASTC HDR 4x4, ASTC HDR 6x6, UASTC HDR 6x6i,
  ASTC LDR and XUASTC LDR 4x4-12x12 (basisu v2.50, bit-exact with the CLI)
- [x] KTX2 reader for Basis payloads (BasisLZ, zstd, arrays, cubemaps)
- [x] KTX2 plain formats, KTX2 writer (raw / zstd; Basis payloads via `write_basis`)
- [x] DDS reader and writer (own implementation: legacy FourCC and DX10 headers, mips, arrays,
  cubemaps; every `Format` with a DXGI equivalent)
- [ ] Image file formats (encode/decode API on byte slices, pure Rust; table = target scope,
  crate = reference for the feature set):

  | Format | Reference crate | Decode | Encode |
  |--------|-----------------|--------|--------|
  | BMP | `image` | Yes, including RLE-compressed files | Uncompressed only |
  | TGA | `image` | Yes, including RLE | Basic |
  | GIF | `gif` | Yes, including animation | Yes, including animation |
  | WebP | `image-webp` | Lossy, lossless, alpha and animation | Lossless only |
  | PNG | `png` | Yes, including APNG | Yes, including APNG |
  | JPG | - | Yes | - |
  | EXR | `exr` | Layers, multi-resolution, f16/f32/u32, RLE/ZIP/PIZ/PXR24/B44 (no DWAA/DWAB, limited deep data) | Yes |
  | TIFF | `tiff` | Yes, incl. JPEG-compressed | Uncompressed, LZW, Deflate, PackBits, float samples, BigTIFF |
  | HDR (Radiance RGBE) | `image` (hdr) | Yes | Yes |
  | JPEG XL | `jxl-oxide` / `zune-jpegxl` | Yes (conformant) | Lossless only |
  | JPEG 2000 | `hayro-jpeg2000` | Yes | - (only C bindings exist) |
  | OpenVDB | `vdb-rs` | Partial | - (only C++ bindings exist) |
  | RenderMan zfile | own implementation | Yes | Yes |
  | AVIF | `avif-parse` + `rav1d` (decode), `ravif` / `rav1e` (encode) | Yes | Yes (asm features off: pure Rust only) |

  No bindings, ever: a format or direction without a pure Rust option is skipped (Ptex: only
  C++ bindings exist, not planned).

  - Done: JPEG XL (`jxl`): `jxl-oxide` decode (lossless bit-exact vs libjxl, lossy VarDCT
    within 1/255, float, animation, sRGB / linear sRGB output), `zune-jpegxl` lossless encode
    (8/16-bit, libjxl-verified).
  - Done: JPEG 2000 (`jpeg2000`): `hayro-jpeg2000` decode of JP2 / J2K, 5/3 and 9/7, 8/12/16-bit,
    gray / RGB(A) / sYCC, tiles, layers, resolution reduce, sub-sampled components (tested
    against OpenJPEG).
  - Done: OpenVDB (`vdb`, partial): own reader (`vdb-rs` links C Blosc): grid list, densify
    `float` / `double` grids to `R32Float` volumes, none / zip / active-mask compression, half,
    tiles, instances; Blosc grids listed only (bit-exact vs OpenVDB 10).
  - Done (own code, `docs/spec/file_{bmp,tga,hdr,zfile}.md`): BMP (all headers / depths,
    RLE4 / RLE8, bit fields; uncompressed encode), TGA (types 1/2/3/9/10/11, origins, 2.0
    footer; raw / RLE encode), Radiance HDR (all orientations, flat / old / new RLE; RLE
    encode, bit-exact with ffmpeg / imagecodecs), RenderMan zfile (both byte orders).
- [x] Optional Vulkan / OpenGL upload features (`vulkan`, `opengl`; `gpu::TextureSource`)
- [ ] Image resize: any uncompressed `Image` / `Surface` (8/16-bit, float, sRGB-correct),
  filters from the `mip` resampler (box, tent, Mitchell, Catmull-Rom, Kaiser, Lanczos, ...),
  fit / fill / exact sizes, crop; multithreaded
- [ ] Image filters: flip / rotate / transpose, blur (box, Gaussian), sharpen (unsharp mask),
  brightness / contrast / gamma / exposure, grayscale, channel swizzle, premultiply alpha,
  normal map from height, tone mapping (HDR -> LDR)

## Follow-ups

- [x] FXT1 transcode target (ETC1S; other payloads unsupported in C++ too)
- [x] Transcoder options (basisu decode flags: high quality, alpha to opaque, deblock, ...)
- [x] PVRTC1/2, ATC and FXT1 decoders (PVRTC2 / FXT1: the modes Basis Universal emits)
- [x] Parallel transcode of many levels/images (`transcode_all`)

## Current plan (in order): finish everything texture related

Image file formats wait until every step below is done. Ports keep bit-exactness with their
C++ / ktx-rs reference (golden hashes from offline builds, as for the transcoder).

1. [x] Conversion gaps: `Rgba16Unorm`, sRGB <-> linear transfer, every uncompressed format
   as a conversion target
2. [x] Mipmap generation: basisu resampler (box, tent, kaiser, lanczos, ...), sRGB aware,
   clamp / wrap, levels in parallel
3. [x] Encode API (`imgio::encode`, quality presets, multithreaded per block row) with
   BC1 / BC3 / BC4 / BC5 (basisu `encode_bc1` / `encode_bc4`) and BC2
4. [x] BC7 encoders: `bc7f` presets, basisu v2.50 `bc7e_scalar` (High / Best)
5. [x] BC6H encoder (unsigned; `fast_encode_bc6h` presets)
6. [x] ETC1 / ETC2 / EAC encoders: `etc1f`, EAC (normal / high quality)
7. [x] ASTC LDR / HDR encoder: astcenc (ktx-rs port), + LZ-RDO (`EncodeOptions::rdo_lambda`)
8. [x] PVRTC1 encoder exposure; own ATC, PVRTC2, FXT1 encoders (every compressed format encodes)
9. [x] Basis Universal encoders: ETC1S, UASTC LDR 4x4 (+ RDO), UASTC HDR 4x4, ASTC HDR 6x6,
   UASTC HDR 6x6i (ktx-rs `basisu::comp` port, + mipmaps / y flip) -> `.basis`
10. [x] KTX2 writer for Basis payloads (BasisLZ global data, UASTC, HDR, 6x6i, XUASTC, XUBC7)
11. [x] Basis Universal v2 encoders: [x] ASTC LDR, [x] XUASTC LDR (zstd / arith / hybrid, DCT,
    lossy supercompression, SCD deblocking; `docs/spec/xuastc_encode.md`), [x] XUBC7
    (`basis/xubc7/encoder.rs` + `BasisCodec::Xubc7`; `.basis` / KTX2 byte-identical to the 2.50
    CLI, `docs/spec/xubc7_encode.md`)
12. [x] DDS reader and writer
13. [x] Optional Vulkan / OpenGL upload features (ktx-rs `vkloader` / `glloader`)

Texture follow-ups found on the way: [x] high-quality ETC1 (basisu `etc1_optimizer` flip / diff
search), [x] ETC2 RGB A1 and signed BC4 / BC5 / EAC encoders, [x] ETC2 T / H / planar mode
encoding (ETC2 RGB / RGB A1 quality), [x] signed BC6H encoder, [x] ASTC encode speed vs C++
astcenc: ported to astc-encoder 6.0.0, byte identical with the 6.0 CLI; the remaining speed gap
is accepted since the output is byte identical. 1024x1024, 1 / 16 threads
(imgio vs `astcenc-avx2` 6.0): 4x4 medium 1.16 / 0.158 s vs 0.86 / 0.102 s (1.33x), 4x4 fast
0.70 / 0.098 vs 0.57 / 0.075 (1.22x), 6x6 medium 1.79 / 0.216 vs 1.30 / 0.143 (1.36x), 8x8
thorough 2.96 / 0.348 vs 2.29 / 0.257 (1.28x); was 1.99 s for 4x4 medium on 5.x. Vs a non-LTO
in-process C++ 6.0 build the gap is 1.06-1.2x (the CLI's gcc LTO is worth 15-20%; Rust LTO 3%),
a codegen gap spread over many kernels. [x] ASTC LZ-RDO (`-rdo-lambda`,
`EncodeOptions::rdo_lambda`): byte identical with the 6.0 CLI, thread-count independent;
1024x1024 4x4 medium lambda 20: 1.74 / 0.26 s vs 1.36 / 0.20 s. [x] astcenc alpha scaling
(`a_scale_radius`, CLI `-a`), progress callback and 3D blocks (`-zdim`, incl. LZ-RDO): byte
identical with the 6.0 CLI through the astcenc API (`docs/spec/astcenc.md`).

More texture work (in order): [x] `.astc` file reader / writer, [x] KTX1 reader / writer
(own, byte identical with basisu `-export_ktx`), [x] 3D textures (KTX2 depth, DDS volumes, KTX1 depth), [x] one-call
texture builder (mips -> encode -> KTX2 / DDS, per layer / face, normal map renormalization).

## Previous plan (done): basisu v2 payloads (XUASTC LDR, ASTC LDR, XUBC7)

Ported from C++ basisu (`ref/basisu`; ktx-rs has no groundwork here), bit-exact vs the C++
transcoder via `tools/basisu-golden`.

1. [x] Containers: tex formats 5..33 in `.basis` and KTX2 (DFD 169/170, ASTC LDR vkFormats),
   test files (`scripts/gen_basis_testdata.sh`), golden hashes for targets 0..40
2. [x] ASTC LDR -> ASTC passthrough and uncompressed targets (RGBA32, 565, 4444), 4x4 re-blocking
3. [x] ASTC LDR -> BC1/BC3/BC4/BC5/EAC via the real-time encoders already in `ll`
4. [x] `etc1f` real-time ETC1 encoder -> ETC1 / ETC2 RGBA targets
5. [x] `bc7f` real-time BC7 encoder -> BC7 target
6. [x] Deblocking filter and PVRTC1 encoder -> every ASTC LDR target
7. [x] XUASTC LDR decompression, zstd syntax (+ DCT weight grids)
8. [x] XUASTC LDR arithmetic and hybrid syntaxes
9. [x] XUASTC direct BC7 fast path (default transcode path)
10. [x] XUBC7 transcoder

Previous plan (done): image types, conversion, ETC/ASTC decode, Basis core, ETC1S, UASTC LDR,
UASTC HDR, KTX2 reader, golden tests and benches.
