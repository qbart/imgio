# Roadmap

- [x] Scaffold: crate, Makefile, licenses, `Error`/`Result`
- [x] Core types: `Format`, `Buffer` (vas)
- [ ] Image/surface/mip types
- [ ] Raw RGBA decode/encode (8-bit, 16F, 32F)
- [x] BC1-BC5 decode
- [x] BC6H / BC7 decode (multithreaded, tested vs Pillow and bcdec_rs)
- [ ] ASTC decode (LDR, HDR)
- [ ] Basis Universal transcoder: ETC1S, UASTC LDR, UASTC HDR -> all targets
- [ ] BCn / ASTC encoders (multithreaded)
- [ ] Basis Universal encoders: ETC1S, UASTC LDR/HDR
- [ ] PNG, TGA, JPG
- [ ] Optional Vulkan / OpenGL upload features
