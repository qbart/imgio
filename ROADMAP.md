# Roadmap

- [x] Scaffold: crate, Makefile, licenses, `Error`/`Result`
- [x] Core types: `Format`, `Buffer` (vas)
- [x] Image/surface/mip types
- [x] Raw RGBA conversion (8-bit, 16F, 32F)
- [x] BC1-BC5 decode
- [x] BC6H / BC7 decode (multithreaded, tested vs Pillow and bcdec_rs)
- [x] ETC1 / ETC2 / EAC decode
- [x] ASTC decode (LDR, HDR), 2D blocks
- [x] Basis Universal transcoder (.basis): ETC1S, UASTC LDR, UASTC HDR 4x4/6x6i, ASTC HDR 6x6 -> all targets except FXT1
- [ ] Basis Universal v2 payloads: XUASTC LDR, ASTC LDR, XUBC7 (see current plan)
- [ ] BCn / ASTC encoders (multithreaded)
- [ ] Basis Universal encoders: ETC1S, UASTC LDR/HDR
- [x] KTX2 reader for Basis payloads (BasisLZ, zstd, arrays, cubemaps)
- [ ] KTX2 plain formats, KTX2 writer
- [ ] PNG, TGA, JPG
- [ ] Optional Vulkan / OpenGL upload features

## Follow-ups

- [ ] FXT1 transcode target
- [ ] PVRTC1/2 and ATC decoders
- [ ] Parallel transcode of many levels/images

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
7. [ ] XUASTC LDR decompression, zstd syntax (+ DCT weight grids)
8. [ ] XUASTC LDR arithmetic and hybrid syntaxes
9. [ ] XUASTC direct BC7 fast path (default transcode path)
10. [x] XUBC7 transcoder

Previous plan (done): image types, conversion, ETC/ASTC decode, Basis core, ETC1S, UASTC LDR,
UASTC HDR, KTX2 reader, golden tests and benches.
