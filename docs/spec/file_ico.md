# ICO / CUR (`imgio::file::ico`, own code)

Spec: directory (`0, type 1 icon / 2 cursor, count`, 16-byte entries: width / height (0 =
256), colors, planes + bit count or cursor hotspot, size, offset) of PNG streams or DIBs.

- Decode: DIB = BITMAPINFOHEADER (height doubled) + XOR bitmap + 1-bit AND mask, decoded by
  prefixing a BMP file header and calling `bmp::decode` with the halved height; output
  `Rgba8Srgb`, alpha from the 4th byte for 32-bit DIBs, else from the AND mask (set =
  transparent). PNG entries via `png::decode` (native format).
- `decode` = largest entry: most pixels, then highest color depth (Windows behaviour; Pillow
  takes the lowest depth among equal sizes), first on ties. `decode_entry`, `decode_all`,
  `info` (sizes, bits, hotspots, PNG flag).
- 32-bit alpha follows the DIB bit count (Pillow uses the directory bit count, which cursors
  use for the hotspot).
- Encode: sizes 16..256 (Pillow's defaults) not above the smaller source side, or the source
  size under 16; Lanczos `resize` with `Fit::Contain`; 32-bit BGRA DIB + AND mask, or PNG
  (`PngEntries::Large` default = 256 entries, `All`, `None`); `hotspot` writes a cursor with
  per-entry scaled hotspots; `encode_entries` for given images (<= 256).

## Audit

- tests/file_ico.rs: 6 files (1 / 4 / 8-bit color and gray palettes, 24 / 32-bit, odd sizes,
  PNG entries incl. 256, Pillow PNG and BMP output, cursor) decode exactly like Pillow 13 per
  entry; directory fields and largest entry; 5 encoder cases decoded exactly by Pillow
  (`encoded_verified.txt`); errors; truncation / bit flips never panic.
