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
- [ ] BCn / ASTC encoders (multithreaded)
- [ ] Basis Universal encoders: ETC1S, UASTC LDR/HDR
- [x] KTX2 reader for Basis payloads (BasisLZ, zstd, arrays, cubemaps)
- [ ] KTX2 plain formats, KTX2 writer
- [ ] PNG, TGA, JPG
- [ ] Optional Vulkan / OpenGL upload features

## Follow-ups

- [ ] FXT1 transcode target
- [ ] PVRTC1/2 and ATC decoders
- [ ] XUASTC LDR and ASTC LDR (basisu v2 tex formats 5+)
- [ ] Parallel transcode of many levels/images

## Current plan (in order)

1. [x] Image types: `Surface`, mip chain helpers
2. [x] Raw conversion: half/f32, R8/RG8/RGBA8/RGBA16F/RGBA32F -> RGBA8/RGBA32F
3. [x] ETC1/ETC2/EAC decode
4. [x] ASTC decode (LDR, HDR)
5. [x] Basis core: .basis file, bit reader, Huffman, tables
6. [x] ETC1S transcoder -> all LDR targets
7. [x] UASTC LDR 4x4 transcoder -> all LDR targets
8. [x] UASTC HDR (4x4, 6x6) transcoder -> BC6H, ASTC HDR, RGBA16F
9. [x] KTX2 reader for Basis payloads (BasisLZ, zstd)
10. [ ] Golden transcode tests and benches
