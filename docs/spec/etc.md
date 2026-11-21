# ETC1 / ETC2 / EAC decode

- Block decoders from ktx-rs `src/etcunpack.rs` (written from the Khronos Data Format
  Specification; etcdec itself is under the Ericsson SLA and is not used).
- ETC1 decoded with the ETC2 decoder (ETC1 is a subset). RGB formats -> RGBA8 (A = 255);
  punch-through transparent texels -> (0,0,0,0); ETC2 RGBA alpha from EAC 8-bit.
- EAC R11/RG11 -> R16/RG16 unorm/snorm, 11-bit values expanded to 16 bits like etcdec.
- Test: 66 random-block cases (all formats, odd sizes) vs etcdec FNV-1a hashes
  (`testdata/etc/etcdec_golden.txt`).

## Audit

- 2026-11-21: ported from ktx-rs; new formats Etc1Rgb, Etc2*, EacR11*/EacRg11*, R16*/Rg16*.
