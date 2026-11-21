# Specs and audit logs

One file per subsystem. Write the spec before implementing; append audit entries
(date, change, reference used) as work lands.

- [error.md](error.md): error handling
- [format.md](format.md): pixel/texture formats
- [bc.md](bc.md): BC1-BC7 decode
- [image.md](image.md): surfaces and mip chains
- [convert.md](convert.md): half floats and uncompressed conversion
- [decode.md](decode.md): decode API and block driver
- [etc.md](etc.md): ETC1/ETC2/EAC decode
- [astc.md](astc.md): ASTC decode
- [astcenc.md](astcenc.md): astcenc (astc-encoder 6.0 port) API: alpha scaling, progress, 3D
- [basis.md](basis.md): Basis Universal container and transcoder
- [basis_encode.md](basis_encode.md): Basis Universal encoder (`.basis` output)
- [xuastc_encode.md](xuastc_encode.md): ASTC LDR / XUASTC LDR encoder core (basisu v2.50)
- [ktx2.md](ktx2.md): KTX2 container with Basis payloads
- [bc7e.md](bc7e.md): basisu bc7e_scalar BC7 encoder port
- [xubc7_encode.md](xubc7_encode.md): basisu XBC7 (XUBC7) encoder port
- [bench.md](bench.md): benchmarks
- [gpu.md](gpu.md): Vulkan / OpenGL texture upload (features `vulkan`, `opengl`)
