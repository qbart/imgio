# Roadmap

- [x] Scaffold: crate, Makefile, licenses, `Error`/`Result`
- [x] Core types: `Format`, `Buffer` (vas)
- [x] Image/surface/mip types
- [x] Raw RGBA conversion (8-bit, 16F, 32F)
- [x] BC1-BC5 decode
- [x] BC6H / BC7 decode (multithreaded, tested vs Pillow and bcdec_rs)
- [x] ETC1 / ETC2 / EAC decode
- [x] ASTC decode (LDR, HDR), 2D blocks
- [ ] Basis Universal transcoder: ETC1S, UASTC LDR, UASTC HDR -> all targets
- [ ] BCn / ASTC encoders (multithreaded)
- [ ] Basis Universal encoders: ETC1S, UASTC LDR/HDR
- [ ] PNG, TGA, JPG
- [ ] Optional Vulkan / OpenGL upload features

## Current plan (in order)

1. [x] Image types: `Surface`, mip chain helpers
2. [x] Raw conversion: half/f32, R8/RG8/RGBA8/RGBA16F/RGBA32F -> RGBA8/RGBA32F
3. [x] ETC1/ETC2/EAC decode
4. [x] ASTC decode (LDR, HDR)
5. [ ] Basis core: .basis file, bit reader, Huffman, tables
6. [ ] ETC1S transcoder -> all LDR targets
7. [ ] UASTC LDR 4x4 transcoder -> all LDR targets
8. [ ] UASTC HDR (4x4, 6x6) transcoder -> BC6H, ASTC HDR, RGBA16F
9. [ ] KTX2 reader for Basis payloads (BasisLZ, zstd)
10. [ ] Golden transcode tests and benches
