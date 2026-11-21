# KTX2

`imgio::ktx2::Ktx2File` borrows the file bytes; `imgio::ktx2::write` creates plain files.

## Plain `VkFormat` payloads

- `vk::vk_format(Format)` / `format_from_vk(u32)`: every `Format` with a Vulkan equivalent (no
  ATC, FXT1). Written as: BC1 -> `BC1_RGBA` (133/134), ETC1 -> `ETC2_R8G8B8`, PVRTC1/2 ->
  `*_4BPP_UNORM_BLOCK_IMG`; read: `BC1_RGB` too, PVRTC sRGB as unorm.
- Reader: when no Basis payload matches and the DFD model is the vkFormat's standard model
  (from `dfd::dfd`), the file is plain: `format()`, `vk_format()`, `image_data(level, layer,
  face)` (zstd inflated once per level), `has_alpha` from an alpha DFD sample (BC1A: "alpha
  present"); `typeSize` must match. `tex_format()` is `Option` (`None` for plain); plain ASTC
  LDR / ASTC 6x6 HDR files are both plain and Basis (transcodable).
- `dfd::dfd(Format)`: port of KTX-Software `vk2dfd` / `createDFD{Unpacked,Packed,Compressed}`
  (little-endian; E5B9G9R9 special case), fixed-size words.
- `write(desc, levels, opts)`: `Ktx2Desc` (format, size, layers (0 = no array), faces, levels),
  `levels[i]` = all layer/face images of level i; `WriteOptions { supercompression: None |
  Zstd(level), key_values }`. Layout as `ktxTexture2_WriteToMemory`: header, level index,
  DFD, key/values sorted by key (each padded to 4, padding counted), levels smallest first,
  aligned to lcm(block bytes, 4) (1 when supercompressed); empty KVD -> offset 0. Zstandard
  via `zstd-pure-rs` (libzstd translation): byte-identical with `ktx create --zstd N`.
- Golden test `tests/ktx2_plain.rs`: 94 files from `ktx create --raw` (KTX-Software CLI as
  ported by ktx-rs; `tools/ktx2-golden/gen.py`): every vkFormat, mips, arrays, cubemaps,
  cube arrays, zstd 1/3/19. Reader returns every image; writer output is byte-identical.

## Basis payloads: writing

- `write_basis(basis_bytes, &BasisKtx2Options { zstd, key_values })`: port of basisu
  `create_ktx2_file` / `get_dfd` driven by a parsed `.basis`: shape from the slices (images ->
  layers, cubemap arrays -> faces, video frames -> layers with P-frame flags), basisu's DFD
  templates (+ transfer from the sRGB flag, UASTC RGBA channel, ASTC block size), vkFormat
  (ASTC 4x4 / 6x6 SFLOAT for UASTC HDR 4x4 / ASTC HDR 6x6, ASTC LDR UNORM / SRGB), BasisLZ global
  data (ETC1S) or 12-byte offset / length / profile descriptors (6x6i scheme 4, XUASTC 5, XUBC7
  6), optional Zstandard for UASTC LDR / HDR 4x4 / ASTC HDR 6x6 / ASTC LDR, key/values sorted
  with basisu's dummy padding key (16-byte KVD end when not supercompressed), SGD at 8, levels
  smallest first.
- `encode_basis(images, &BasisEncodeOptions, &BasisKtx2Options)`: `basis::encode_basis` +
  `write_basis`, adding `KTXwriter` "imgio <version>".
- Golden test `write_basis_matches_basisu`: every `testdata/ktx2/*.ktx2` (basisu `-ktx2`) vs
  `write_basis` of its `testdata/ktx2/basis/*.basis` twin (same CLI options): 13 files byte
  identical (ETC1S incl. array, UASTC raw / zstd, UASTC HDR 4x4, ASTC HDR 6x6, 6x6i, ASTC LDR,
  XUASTC, XUBC7). Round trip test: encode -> KTX2 -> transcode = `.basis` transcode.

## Basis payloads: reading

- Header (80 bytes), level index (24 bytes/level), DFD (44 or 60 bytes, one basic block),
  key/values (`KTXanimData` -> video), supercompression global data.
- Payload detection as basisu `ktx2_transcoder::init`:
  | DFD model | vkFormat | scheme | payload |
  |---|---|---|---|
  | 163 ETC1S | 0 | 1 BasisLZ | ETC1S (alpha when DFD is 60 bytes) |
  | 166 UASTC LDR | 0 | 0 / 2 zstd | UASTC LDR 4x4 (alpha: channel RGBA/RRRG) |
  | 167 UASTC HDR | ASTC_4x4_SFLOAT | 0 / 2 | UASTC HDR 4x4 |
  | 162 ASTC | ASTC_6x6_SFLOAT | 0 / 2 | ASTC HDR 6x6 |
  | 168 | 0 | 4 (12-byte descs) or 1 (8-byte, v1.6 files) | UASTC HDR 6x6 intermediate |
  | 162 ASTC | ASTC_{4x4..12x12}_{UNORM,SRGB} (157..184) | 0 / 2 | ASTC LDR (DFD block dims must match) |
  | 169 | 0 | 5 (12-byte descs) | XUASTC LDR, block size from DFD texel block dims |
  | 170 | 0 | 6 (12-byte descs) | XUBC7 |
- Supercompressed slice descriptor arrays (schemes 4/5/6) must exactly fill the global data.
- Images are indexed `(level * layers + layer) * faces + face`. ETC1S/6x6i/XUASTC/XUBC7 slices
  come from the
  global data (offsets relative to the level); plain payloads are `(layer * faces + face)`
  images of `blocks * 16` bytes in the (inflated) level.
- Zstandard levels are inflated with `ruzstd` (pure Rust) into a vas `Buffer`, once per level.
- 3D textures (plain formats only; Basis payloads are 2D): `depth()`, `level_depth(level)`;
  `image_data` returns the level's slices back to back; `Ktx2Desc::depth` (0 = 2D) for the
  writer, slices innermost. Not supported: plain vkFormats outside `Format`.
- `transcode(level, layer, face, format, dst)` reuses the container-independent
  `basis::transcode::transcode_slice`; `transcode_with` takes `basis::TranscodeOptions`
  (`Deblock::Auto` follows `DeblockFilterID`, `Off`/`Force` override it like C++).
- `transcode_all(format, opts)`: every level x layer x face in parallel (see basis.md).

Test: `tests/ktx2_transcode.rs`, 436 level/layer/target cases (XUBC7, ETC1S with mips and a 2-layer
array, UASTC zstd and raw, HDR 4x4, ASTC HDR 6x6, 6x6i) bit-exact with the C++ `ktx2_transcoder`.

## Audit

- 2026-11-21: initial reader written from the KTX 2.0 spec and basisu `ktx2_transcoder`.
- 2026-11-21: DFD models 169 (XUASTC LDR), 170 (XUBC7) and ASTC LDR vkFormats parsed.
- 2026-11-21: `write_basis` / `encode_basis` (basisu `create_ktx2_file` port), byte identical
  with basisu `-ktx2` on 13 files; `Ktx2File::key_values`.
- 2026-11-21: plain vkFormat reader + writer (raw / zstd), DFD generation; 94 KTX-Software
  reference files read and rewritten byte-identically.
- 2026-11-21: 3D textures (plain): reader + writer, 2 `ktx create --depth` files byte identical.
