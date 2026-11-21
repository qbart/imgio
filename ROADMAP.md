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
- [ ] BCn / ASTC encoders (multithreaded)
- [ ] Basis Universal encoders: ETC1S, UASTC LDR/HDR
- [x] KTX2 reader for Basis payloads (BasisLZ, zstd, arrays, cubemaps)
- [x] KTX2 plain formats, KTX2 writer (raw / zstd; Basis payload writing still open)
- [ ] DDS reader and writer (own implementation: legacy FourCC and DX10 headers, mips, arrays,
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
- [ ] Optional Vulkan / OpenGL upload features

## Follow-ups

- [x] FXT1 transcode target (ETC1S; other payloads unsupported in C++ too)
- [x] Transcoder options (basisu decode flags: high quality, alpha to opaque, deblock, ...)
- [x] PVRTC1/2, ATC and FXT1 decoders (PVRTC2 / FXT1: the modes Basis Universal emits)
- [x] Parallel transcode of many levels/images (`transcode_all`)

## Current plan (in order): basisu v2 payloads (XUASTC LDR, ASTC LDR, XUBC7)

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
