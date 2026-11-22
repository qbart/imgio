# GPU memory (`imgio::memory`)

Spec: how much GPU memory textures take, and what other formats would take.

- `texture_bytes(format, Layout)`: sum over levels (each dimension halved, at least 1) of
  `surface_bytes` (whole blocks) x depth slices x layers x faces; `Layout::new_2d` /
  `with_mips` (full chain = `32 - lzcnt(max dim)` levels). Zero sizes, faces other than 1 / 6
  or too many levels are `InvalidArgument`; overflow checked (u64).
- `upload_format`: RGB8 -> RGBA8, RGB16F -> RGBA16F (3-byte texels are padded on GPUs).
- `of_container`: KTX2 / KTX1 / DDS / `.basis` / `.astc` headers by magic: layout, stored
  format, alpha, sRGB, decoded channels, bytes. Basis payloads (ETC1S, UASTC, XUASTC, XUBC7,
  HDR) have no fixed GPU format: `transcode_target` = the usual runtime choice (desktop
  BC1 / BC3 / BC7 / BC6H, mobile ETC2 / ASTC of the payload block / ASTC HDR).
- `analyze`: alpha opaque / binary (0 and 1 only) / full, gray (R = G = B everywhere), HDR,
  sRGB. `suggest`: desktop BC6H / BC4 / BC5 / BC7, mobile ASTC 6x6 (HDR, opaque or cutout)
  or 4x4 (partial alpha, two channel).
- CLI `imgio memory`: files and directories (recursive, known extensions), table or `--json`,
  totals with the change vs now, `--no-mips`, fixed `--desktop` / `--mobile` formats,
  `--budget SIZE` (exit 1 when the current total is larger). Plain containers are decoded
  (level 0) for the content; textures already in a target's block family keep their format.

## Audit

- tests/memory.rs: `of_container` + `texture_bytes` equal the summed stored image data of
  every uncompressed-payload DDS / KTX1 / KTX2 test file (KTX-Software, texconv, basisu and
  imgio writers; 2D, arrays, cubes, 3D, mips, NPOT; >= 100 files); Basis payloads give
  transcode estimates; known sizes (1024^2 RGBA8 / BC7 / BC1 chains, ASTC 6x6 rounding, cube
  arrays, volumes), errors; content classes and suggestions.
- tests/cli.rs: directory scan with a nested KTX2, JSON values equal the library, `--no-mips`,
  budget pass / fail / bad size.
