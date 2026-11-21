# OpenVDB (`imgio::file::vdb`, feature `vdb`)

Decode only, partial. imgio's own reader: `vdb-rs` 0.6 links the C Blosc library
unconditionally (`blosc-src`), and panics on several inputs, so it is not used. Zlib streams
are inflated with `miniz_oxide` (pure Rust). Written from the OpenVDB file layout (OpenVDB 10
`io/Archive`, `io/Compression.h`, `tree/RootNode.h` / `InternalNode.h` / `LeafNode.h`,
`math/Maps.h`); no code copied.

- File: magic `20 42 44 56 00 00 00 00`, file version >= 222 (OpenVDB 3.0+; older:
  `Unsupported`), grid offsets required, UUID, file metadata (skipped), grid descriptors
  (unique name, type with `_HalfFloat` suffix, instance parent, grid / block / end offsets).
- Grid: compression flags (`ZIP` 1, `ACTIVE_MASK` 2, `BLOSC` 4), metadata (`class`,
  `file_bbox_min` / `file_bbox_max` used), transform: `UniformScaleMap`, `ScaleMap`,
  `(Uniform)ScaleTranslateMap`, `TranslationMap`, `AffineMap`, `UnitaryMap` -> 4x4 matrix in
  OpenVDB's row-vector convention; `NonlinearFrustumMap` parsed, no matrix; others
  `Unsupported`.
- Tree `Tree_<T>_5_4_3` for `float`, `double`, `int32`, `int64`, `bool`, `vec3s`, `vec3d`,
  `vec3i` (topology parsed; other types listed with the metadata bbox): root background, tiles
  (4096^3), children; internal nodes 32^3 / 16^3 (child mask, value mask, compressed values,
  children in mask order); leaves 8^3 (value mask in topology; buffers: mask again + values).
  Root children's buffers are read in sorted-origin order (OpenVDB's map order).
- `io::readCompressedValues`: metadata byte 0-6 (background, -background, one or two stored
  inactive values, selection mask), active-mask compression stores only active values; data
  raw or zip (`i64` size, <= 0 = stored raw); half storage (`_HalfFloat`) stores buffer values
  as binary16 while background / tiles / inactive values keep full size; no half data is
  written for 0 values. Blosc payloads: `Unsupported` (its raw fallback chunks are read).
- API: `grids` -> `GridInfo { name, type_name, value_type, half_float, class, compression,
  instance_of, transform, active_bbox }` (bbox = active voxels and active tiles from the
  topology; for unparsable grids (Blosc, other types) the `file_bbox_*` metadata).
  `densify(data, grid, &DecodeOptions { max_voxels })` / `densify_region(.., min, max, ..)` ->
  `VdbVolume { width, height, depth, origin, data }`: `R32Float` voxels, x fastest, values as
  OpenVDB `getValue` (leaves, tiles, background; `double` and half converted to f32). Scalar
  float grids only (`float`, `double`), else `Unsupported`; `max_voxels` (default 2^26) ->
  `InvalidArgument` above. Instanced grids read the parent's tree.
- `decode`: middle z slice (`min.z + (max.z - min.z) / 2`) of the active bbox of the first
  `float` / `double` grid as an `R32Float` image; `Unsupported` without one.
- Robustness: every read bounds-checked; counts checked against the remaining input;
  descriptors only move forward; decompressed sizes are exact (known in advance); coordinates
  in i64, bboxes outside i32 -> `InvalidData`.

## Tests (`tests/file_vdb.rs`, data `testdata/file/vdb`)

Written by the C++ OpenVDB 10.0.1 library (`tools/vdb-golden/gen.cpp`, built by
`scripts/gen_file_testdata_jxlj2kvdb.py` `gen_vdb` against the Ubuntu 24.04 packages, env
`OVDB_PREFIX`); expected values = OpenVDB reading the file back (`getValue` over the active
bbox, `evalActiveVoxelBoundingBox`, `transform().baseMap()->getAffineMap()`).

- Files: uncompressed (stored zlib-wrapped, 175 KB raw), zip, active mask, zip + active mask
  (one leaf per inactive-value mode), level set (zip dropped by OpenVDB, -background inside),
  half float with an active 8^3 tile and an inactive non-background tile, multi-grid (int32
  with a root tile, vec3s, bool, double with an affine rotation, float, instance, sparse
  1001^3 bbox), Blosc.
- Exact: grid list (names, types, half, map, matrix, bbox) for 14 grids; densify bit-exact for
  9 float / double grids; `decode` = middle slice; regions beyond the bbox, voxel limit,
  non-scalar grids / Blosc -> `Unsupported`; detection; truncation at 200 lengths and 300
  random bit flips per file never panic.

## Audit

- 2026-11-22: own reader (versions 222+, none / zip / active mask, half, tiles, instances);
  bit-exact vs OpenVDB 10.0.1.
