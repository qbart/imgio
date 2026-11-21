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
- [basis.md](basis.md): Basis Universal container and transcoder
- [ktx2.md](ktx2.md): KTX2 container with Basis payloads
