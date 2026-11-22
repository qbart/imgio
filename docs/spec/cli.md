# Command line tool (`imgio` binary, feature `cli`)

Spec: one binary over the library; hand-written argument parsing (no extra dependencies);
errors are messages on stderr with exit code 1; no panics.

- `info`: containers (KTX2 / DDS / KTX1 / Basis: size, levels, layers, faces, format) and
  image files (format, size, pixel format, frames, colorimetry, ICO entries).
- `convert in out`: input any image file or texture container level 0 (decoded /
  transcoded); image options in a fixed order (colour management, crop, resize, flip, rotate,
  transpose, blurs, sharpen, adjust, grayscale, premultiply, swizzle, normal map, tone map,
  pixel format); output by extension: image files, or `.ktx2` / `.dds` / `.ktx` textures
  (`--format`, `--quality`, mips, `--normal`, `--zstd`, `--basis CODEC`). Image files that
  cannot store the pixel format get 8-bit sRGB RGBA.
- `texture -o out ins...`: arrays (several inputs) or cubes (`--cube`, 6 per cube).
- `pack`: `--r/--g/--b/--a file[:channel][:invert]` or constants, `--orm`, size modes.
- `atlas`: every `AtlasOptions` field as a flag, pages named `name<i>.ext`, `--json`.
- `ibl`: panorama -> cube KTX2 (mips), `--irradiance` (SH), `--specular` (GGX levels),
  `--brdf` (PNG RGBA16 or KTX2 RG16).
- `compare`: metrics table, `--diff` image.

## Audit

- tests/cli.rs: every command on generated PNGs, outputs read back with `info`, repeated
  conversion byte identical, error messages and exit codes.
