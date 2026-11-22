# Texture atlases (`imgio::atlas`)

Spec: pack sprites into pages; `build` renders pixels, `layout` packs sizes only.

- Packers (`packer.rs`, deterministic, first candidate wins ties): MaxRects (Jylänki 2010:
  BSSF, BLSF, BAF, bottom-left, contact point; split + prune, pruning only pairs with new free
  rectangles), skyline bottom-left, shelf best-height-fit. Online insertion in `SortOrder`
  (max side, area, height, width, perimeter, none; stable).
- Packed rectangle = trimmed size + 2 * extrude + padding; bin = page - 2 * border + padding
  (no padding needed at the far edges).
- Page size: `Smallest` (widths: powers of two, or ~40 steps of +1/7 from the lower bound,
  in parallel; per width binary search of the height; shrink to the used extent; pick area,
  then longer side, then width), `square` (binary search on the side), `Fixed`; limits
  `max_width/height`, `power_of_two`, `multiple_of`.
- `multipage`: greedy fill of maximum-size pages in sort order, then the smallest page per
  page; without it, sprites that do not fit are `InvalidArgument`.
- Pixels: sprites converted to the page format; `Trim::Alpha(t)` crops to alpha > t (no alpha:
  untouched; empty: 1x1); `dedupe` merges sprites with equal trimmed pixels (`alias_of`);
  rotation 90 degrees clockwise (TexturePacker convention); extrusion clamps to the region edge.
- Output: `Placement` (page, region, rotated, trim rect, source size, alias), `Atlas::uv`,
  `to_json` = TexturePacker JSON (Hash) shape plus `page` and `meta.pages`.

## Audit

- tests/atlas.rs: 7 algorithms x 5 option sets (rotation, power of two, square, multiple of
  4, padding, border, extrusion, trimming, sort orders): bounds, page constraints, no overlap
  of padded / extruded regions, every region and extruded pixel traced back to its source,
  determinism, >= 70% occupancy; trimming bounds, duplicates, fully transparent sprites;
  multipage, fixed pages, oversize and rotation-only fits, option / format errors; exact
  small layout and JSON text.
- 1000 random 8..128 px rectangles (padding 2): MaxRects BSSF 97.2% (rotation 98.2%) in
  0.2 s, skyline 93.8% (95.3%) in 10 ms, shelf 79.4% (82.6%) in 1 ms.
