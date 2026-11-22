//! Texture atlases: pack many images into one or more pages.
//!
//! [`build`] trims, deduplicates, packs and renders sprites; [`layout`] only packs rectangle
//! sizes (no pixels). Options cover the packing algorithm ([`Algorithm`]: MaxRects with five
//! heuristics, skyline, shelves), sort order, page size limits and the smallest-page search
//! (power of two, square, multiple of N, or a fixed size), padding between sprites, a page
//! border, edge extrusion (duplicated border pixels against filtering bleed), 90 degree
//! rotation, alpha trimming, duplicate merging and multiple pages.
//!
//! The page size search tries a range of widths (all powers of two, or about 40 steps from the
//! smallest possible width to the maximum, in parallel) and binary searches the smallest
//! height for each, keeping the smallest area (then the smaller longer side, then the smaller
//! width). Everything is deterministic: the same input and options give the same atlas.
//!
//! [`Atlas::to_json`] writes the layout in the TexturePacker "JSON (Hash)" shape that most
//! engines and frameworks read (`frames` by name with `frame`, `rotated`, `trimmed`,
//! `spriteSourceSize`, `sourceSize`, plus `page`; `meta` with page sizes). Rotated sprites are
//! turned 90 degrees clockwise, as in TexturePacker.

mod packer;

pub use packer::{Algorithm, MaxRectsHeuristic};

use crate::file::Image;
use crate::{Buffer, Error, Format, Result, Surface};
use packer::Bin;
use vas::Vector;
use vas::storage::Dynamic;

/// Order in which sprites are packed (larger first packs denser).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    /// Input order.
    None,
    /// Longer side, descending (then area).
    #[default]
    MaxSide,
    Area,
    Height,
    Width,
    Perimeter,
}

/// Page size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PageSize {
    /// The smallest page that fits (within the maximum and the constraints).
    #[default]
    Smallest,
    /// Exactly this size.
    Fixed(u32, u32),
}

/// Transparent border removal.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Trim {
    #[default]
    None,
    /// Crop to the pixels with alpha above this value (0..1; 0 keeps any non-zero alpha).
    /// Formats without alpha are never trimmed. A fully transparent sprite becomes 1x1.
    Alpha(f32),
}

/// Options of [`build`] and [`layout`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasOptions {
    pub algorithm: Algorithm,
    pub sort: SortOrder,
    pub size: PageSize,
    /// Page size limit (default 4096 x 4096).
    pub max_width: u32,
    pub max_height: u32,
    /// Page sides are powers of two.
    pub power_of_two: bool,
    /// Pages are square.
    pub square: bool,
    /// Page sides are multiples of this (e.g. 4 for block compression; default 1).
    pub multiple_of: u32,
    /// Empty pixels between sprites (default 2).
    pub padding: u32,
    /// Empty pixels along the page edges.
    pub border: u32,
    /// Sprite edge pixels repeated outward this many times (inside the padding budget: the
    /// extruded pixels are part of the packed rectangle).
    pub extrude: u32,
    /// Allow 90 degree (clockwise) rotation.
    pub rotation: bool,
    pub trim: Trim,
    /// Pack identical sprites (same pixels and trim) once; their placements share a region.
    pub dedupe: bool,
    /// Use more pages when the sprites do not fit one page of the maximum size.
    pub multipage: bool,
    /// Page pixel format (any uncompressed format; default `Rgba8Srgb`).
    pub format: Format,
}

impl Default for AtlasOptions {
    fn default() -> Self {
        AtlasOptions {
            algorithm: Algorithm::default(),
            sort: SortOrder::MaxSide,
            size: PageSize::Smallest,
            max_width: 4096,
            max_height: 4096,
            power_of_two: false,
            square: false,
            multiple_of: 1,
            padding: 2,
            border: 0,
            extrude: 0,
            rotation: false,
            trim: Trim::None,
            dedupe: false,
            multipage: false,
            format: Format::Rgba8Srgb,
        }
    }
}

/// Where a sprite went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub page: usize,
    /// Region of the page holding the (trimmed) sprite, without extrusion and padding; `w`
    /// and `h` are swapped when rotated.
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    /// Stored turned 90 degrees clockwise.
    pub rotated: bool,
    /// The kept rectangle of the source (the whole source without trimming).
    pub trim_x: u32,
    pub trim_y: u32,
    pub trim_w: u32,
    pub trim_h: u32,
    pub source_w: u32,
    pub source_h: u32,
    /// Index of the sprite whose region this one shares ([`AtlasOptions::dedupe`]), else its
    /// own index.
    pub alias_of: usize,
}

/// Page sizes and placements of [`layout`].
#[derive(Debug)]
pub struct Layout {
    /// (width, height) per page.
    pub pages: Vector<(u32, u32), Dynamic>,
    /// One per input, in input order.
    pub placements: Vector<Placement, Dynamic>,
}

/// Result of [`build`].
#[derive(Debug)]
pub struct Atlas {
    pub pages: Vector<Image, Dynamic>,
    /// One per input sprite, in input order.
    pub placements: Vector<Placement, Dynamic>,
}

impl Atlas {
    /// Normalized `[u0, v0, u1, v1]` of sprite `i`'s region (`None` for a bad index).
    pub fn uv(&self, i: usize) -> Option<[f32; 4]> {
        let p = self.placements.get(i)?;
        let pg = self.pages.get(p.page)?;
        let (w, h) = (pg.width as f32, pg.height as f32);
        Some([
            p.x as f32 / w,
            p.y as f32 / h,
            (p.x + p.w) as f32 / w,
            (p.y + p.h) as f32 / h,
        ])
    }

    /// TexturePacker "JSON (Hash)" description; `names` (one per sprite) become the frame
    /// keys (`InvalidArgument` if the count differs). `image` names page files: `{}` is
    /// replaced by the page index (e.g. `"atlas{}.png"`).
    pub fn to_json(&self, names: &[&str], image: &str) -> Result<Buffer> {
        let sizes: Vector<(u32, u32), Dynamic> = {
            let mut v = Vector::default();
            for p in self.pages.iter() {
                v.push((p.width, p.height))
                    .map_err(|_| Error::OutOfMemory)?;
            }
            v
        };
        json(&sizes, &self.placements, names, image)
    }
}

impl Layout {
    /// Like [`Atlas::to_json`].
    pub fn to_json(&self, names: &[&str], image: &str) -> Result<Buffer> {
        json(&self.pages, &self.placements, names, image)
    }
}

// ---------------------------------------------------------------------------------------------
// Packing

/// A rectangle to pack: `id` (unique item index), packed size.
#[derive(Clone, Copy)]
struct Item {
    id: usize,
    w: u32,
    h: u32,
}

/// (x, y, rotated) per item of a successful pack, in item order.
type Packed = Vector<(u32, u32, bool), Dynamic>;

fn pot_ceil(v: u32) -> u32 {
    v.max(1).checked_next_power_of_two().unwrap_or(1 << 31)
}

fn pot_floor(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        1 << (31 - v.leading_zeros())
    }
}

struct Ctx<'a> {
    o: &'a AtlasOptions,
    /// Largest page.
    max_w: u32,
    max_h: u32,
}

impl Ctx<'_> {
    /// The smallest allowed side >= `v` (`None` above the maximum).
    fn up(&self, v: u32, max: u32) -> Option<u32> {
        let m = self.o.multiple_of.max(1);
        let v = if self.o.power_of_two {
            pot_ceil(v)
        } else {
            v.div_ceil(m).checked_mul(m)?
        };
        let v = v.max(1);
        (v <= max).then_some(v)
    }

    /// Packs every item into a `pw` x `ph` page; `None` if one does not fit.
    fn pack(&self, items: &[Item], pw: u32, ph: u32) -> Result<Option<Packed>> {
        let o = self.o;
        let inner = |s: u32| (s + o.padding).checked_sub(2 * o.border);
        let (Some(bw), Some(bh)) = (inner(pw), inner(ph)) else {
            return Ok(None);
        };
        let mut bin = Bin::new(o.algorithm, bw, bh)?;
        let mut out = Packed::default();
        out.reserve(items.len()).map_err(|_| Error::OutOfMemory)?;
        for it in items {
            match bin.insert(it.w, it.h, o.rotation)? {
                Some(p) => out.push(p).map_err(|_| Error::OutOfMemory)?,
                None => return Ok(None),
            }
        }
        Ok(Some(out))
    }

    /// Used extent of a pack as a page size (before rounding).
    fn extent(&self, items: &[Item], packed: &Packed) -> (u32, u32) {
        let o = self.o;
        let (mut w, mut h) = (0, 0);
        for (it, &(x, y, rot)) in items.iter().zip(packed.iter()) {
            let (iw, ih) = if rot { (it.h, it.w) } else { (it.w, it.h) };
            w = w.max(x + iw);
            h = h.max(y + ih);
        }
        (
            (w + 2 * o.border).saturating_sub(o.padding).max(1),
            (h + 2 * o.border).saturating_sub(o.padding).max(1),
        )
    }

    /// Smallest page for `items` (all fit the maximum page).
    fn smallest(&self, items: &[Item]) -> Result<(u32, u32, Packed)> {
        let o = self.o;
        let need = |s: u32| (s + 2 * o.border).saturating_sub(o.padding).max(1);
        let (mut lw, mut lh, mut area) = (1u32, 1u32, 0u64);
        for it in items {
            let (a, b) = if o.rotation {
                (it.w.min(it.h), it.w.min(it.h))
            } else {
                (it.w, it.h)
            };
            lw = lw.max(need(a));
            lh = lh.max(need(b));
            area += u64::from(it.w) * u64::from(it.h);
        }
        let lw = self.up(lw, self.max_w).ok_or(Error::InvalidArgument)?;
        let lh = self.up(lh, self.max_h).ok_or(Error::InvalidArgument)?;

        if o.square {
            return self.smallest_square(items, lw.max(lh));
        }

        // Width candidates.
        let mut widths = Vector::<(u32, u32, u32, Option<Packed>), Dynamic>::default();
        let mut w = lw;
        loop {
            widths
                .push((w, 0, 0, None))
                .map_err(|_| Error::OutOfMemory)?;
            if w >= self.max_w {
                break;
            }
            let next = if o.power_of_two {
                w.saturating_mul(2)
            } else {
                w + (w / 7).max(o.multiple_of.max(1))
            };
            w = self
                .up(next.min(self.max_w), self.max_w)
                .unwrap_or(self.max_w);
            if widths.last().is_some_and(|l| l.0 >= w) {
                w = self.max_w;
                if widths.last().is_some_and(|l| l.0 >= w) {
                    break;
                }
            }
        }
        crate::parallel::try_for_each_item(&mut widths, |c| {
            let w = c.0;
            let min_h = (area / u64::from(w + o.padding))
                .min(u64::from(self.max_h))
                .saturating_sub(u64::from(o.padding)) as u32;
            let Some(mut lo) = self.up(lh.max(min_h), self.max_h) else {
                return Ok(());
            };
            let Some(mut best) = self.pack(items, w, self.max_h)? else {
                return Ok(());
            };
            let mut hi = self.max_h;
            while lo < hi {
                let mid = self.up(lo + (hi - lo) / 2, hi).unwrap_or(hi);
                let mid = if mid >= hi { lo } else { mid };
                match self.pack(items, w, mid)? {
                    Some(p) => {
                        best = p;
                        hi = mid;
                    }
                    None => {
                        lo = self.up(mid + 1, hi).unwrap_or(hi);
                    }
                }
            }
            // Shrink to the used extent.
            let (ew, eh) = self.extent(items, &best);
            c.1 = self.up(ew, w).unwrap_or(w);
            c.2 = self.up(eh, hi).unwrap_or(hi);
            c.3 = Some(best);
            Ok(())
        })?;
        let mut pick: Option<(u64, u32, u32, usize)> = None;
        for (i, c) in widths.iter().enumerate() {
            if c.3.is_none() {
                continue;
            }
            let key = (u64::from(c.1) * u64::from(c.2), c.1.max(c.2), c.1, i);
            if pick.is_none_or(|p| (key.0, key.1, key.2) < (p.0, p.1, p.2)) {
                pick = Some(key);
            }
        }
        let (_, _, _, i) = pick.ok_or(Error::InvalidArgument)?;
        let c = widths.swap_remove(i).ok_or(Error::InvalidArgument)?;
        let packed = c.3.ok_or(Error::InvalidArgument)?;
        Ok((c.1, c.2, packed))
    }
}

impl Ctx<'_> {
    /// Smallest square page (binary search on the side, from `lo`).
    fn smallest_square(&self, items: &[Item], lo: u32) -> Result<(u32, u32, Packed)> {
        let top = self.max_w.min(self.max_h);
        let mut lo = self.up(lo, top).ok_or(Error::InvalidArgument)?;
        let mut best = self.pack(items, top, top)?.ok_or(Error::InvalidArgument)?;
        let mut hi = top;
        while lo < hi {
            let mid = self.up(lo + (hi - lo) / 2, hi).unwrap_or(hi);
            let mid = if mid >= hi { lo } else { mid };
            match self.pack(items, mid, mid)? {
                Some(p) => {
                    best = p;
                    hi = mid;
                }
                None => lo = self.up(mid + 1, hi).unwrap_or(hi),
            }
        }
        Ok((hi, hi, best))
    }
}

/// Packs `items` into pages: (page size, items of the page, their positions).
type Pages = Vector<((u32, u32), Vector<Item, Dynamic>, Packed), Dynamic>;

fn pack_pages(o: &AtlasOptions, items: &[Item]) -> Result<Pages> {
    if o.multiple_of == 0 || o.max_width == 0 || o.max_height == 0 {
        return Err(Error::InvalidArgument);
    }
    let (mut max_w, mut max_h) = match o.size {
        PageSize::Fixed(w, h) => (w, h),
        PageSize::Smallest => (o.max_width, o.max_height),
    };
    if o.power_of_two && o.size == PageSize::Smallest {
        max_w = pot_floor(max_w);
        max_h = pot_floor(max_h);
    }
    if o.square && o.size == PageSize::Smallest {
        max_w = max_w.min(max_h);
        max_h = max_w;
    }
    if max_w == 0 || max_h == 0 {
        return Err(Error::InvalidArgument);
    }
    let cx = Ctx { o, max_w, max_h };

    // Split into pages greedily at the maximum size.
    let mut groups = Vector::<Vector<Item, Dynamic>, Dynamic>::default();
    let mut rest = Vector::<Item, Dynamic>::default();
    for it in items {
        rest.push(*it).map_err(|_| Error::OutOfMemory)?;
    }
    while !rest.is_empty() {
        let inner = |s: u32| (s + o.padding).checked_sub(2 * o.border);
        let (Some(bw), Some(bh)) = (inner(max_w), inner(max_h)) else {
            return Err(Error::InvalidArgument);
        };
        let mut bin = Bin::new(o.algorithm, bw, bh)?;
        let mut page = Vector::<Item, Dynamic>::default();
        let mut left = Vector::<Item, Dynamic>::default();
        for it in rest.iter() {
            if !left.is_empty() && !o.multipage {
                left.push(*it).map_err(|_| Error::OutOfMemory)?;
                continue;
            }
            match bin.insert(it.w, it.h, o.rotation)? {
                Some(_) => page.push(*it).map_err(|_| Error::OutOfMemory)?,
                None => left.push(*it).map_err(|_| Error::OutOfMemory)?,
            }
        }
        if page.is_empty() || (!left.is_empty() && !o.multipage) {
            // A sprite larger than a page, or too many sprites for one page.
            return Err(Error::InvalidArgument);
        }
        groups.push(page).map_err(|_| Error::OutOfMemory)?;
        rest = left;
    }

    let mut pages = Pages::default();
    for g in groups.iter() {
        let (w, h, packed) = match o.size {
            PageSize::Fixed(w, h) => {
                let p = cx.pack(g, w, h)?.ok_or(Error::InvalidArgument)?;
                (w, h, p)
            }
            PageSize::Smallest => cx.smallest(g)?,
        };
        let mut items = Vector::<Item, Dynamic>::default();
        for it in g.iter() {
            items.push(*it).map_err(|_| Error::OutOfMemory)?;
        }
        pages
            .push(((w, h), items, packed))
            .map_err(|_| Error::OutOfMemory)?;
    }
    Ok(pages)
}

fn sort_items(items: &mut Vector<Item, Dynamic>, order: SortOrder) {
    let key = |i: &Item| -> (u64, u64) {
        let (w, h) = (u64::from(i.w), u64::from(i.h));
        match order {
            SortOrder::None => (0, 0),
            SortOrder::MaxSide => (w.max(h), w * h),
            SortOrder::Area => (w * h, w.max(h)),
            SortOrder::Height => (h, w),
            SortOrder::Width => (w, h),
            SortOrder::Perimeter => (w + h, w * h),
        }
    };
    items.sort_stable_by(|a, b| key(b).cmp(&key(a)));
}

/// Packs rectangles of the given sizes (no pixels, trimming or deduplication). Errors:
/// `InvalidArgument` for zero sizes, a rectangle larger than a page, or too many for one page
/// without [`AtlasOptions::multipage`].
pub fn layout(sizes: &[(u32, u32)], opts: &AtlasOptions) -> Result<Layout> {
    let mut rects = Vector::<(u32, u32, u32, u32), Dynamic>::default();
    for &(w, h) in sizes {
        rects.push((0, 0, w, h)).map_err(|_| Error::OutOfMemory)?;
    }
    let mut alias = Vector::<usize, Dynamic>::default();
    for i in 0..sizes.len() {
        alias.push(i).map_err(|_| Error::OutOfMemory)?;
    }
    let (pages, placements) = place(sizes, &rects, &alias, opts)?;
    Ok(Layout { pages, placements })
}

/// Packs the unique sprites and returns page sizes and placements for every sprite.
/// `trims[i]` = (x, y, w, h) kept rectangle of source `sizes[i]`; `alias[i]` = first sprite
/// with the same content.
#[allow(clippy::type_complexity)]
fn place(
    sizes: &[(u32, u32)],
    trims: &[(u32, u32, u32, u32)],
    alias: &[usize],
    o: &AtlasOptions,
) -> Result<(Vector<(u32, u32), Dynamic>, Vector<Placement, Dynamic>)> {
    let grow = 2 * o.extrude + o.padding;
    let mut items = Vector::<Item, Dynamic>::default();
    for (i, t) in trims.iter().enumerate() {
        if t.2 == 0 || t.3 == 0 {
            return Err(Error::InvalidArgument);
        }
        if alias[i] == i {
            items
                .push(Item {
                    id: i,
                    w: t.2.checked_add(grow).ok_or(Error::InvalidArgument)?,
                    h: t.3.checked_add(grow).ok_or(Error::InvalidArgument)?,
                })
                .map_err(|_| Error::OutOfMemory)?;
        }
    }
    sort_items(&mut items, o.sort);
    let pages = pack_pages(o, &items)?;

    let mut placements = Vector::<Placement, Dynamic>::default();
    for (i, &(sw, sh)) in sizes.iter().enumerate() {
        let t = trims[i];
        placements
            .push(Placement {
                page: 0,
                x: 0,
                y: 0,
                w: t.2,
                h: t.3,
                rotated: false,
                trim_x: t.0,
                trim_y: t.1,
                trim_w: t.2,
                trim_h: t.3,
                source_w: sw,
                source_h: sh,
                alias_of: alias[i],
            })
            .map_err(|_| Error::OutOfMemory)?;
    }
    let mut page_sizes = Vector::<(u32, u32), Dynamic>::default();
    for (pi, (size, its, packed)) in pages.iter().enumerate() {
        page_sizes.push(*size).map_err(|_| Error::OutOfMemory)?;
        for (it, &(x, y, rot)) in its.iter().zip(packed.iter()) {
            let p = &mut placements[it.id];
            p.page = pi;
            p.x = x + o.border + o.extrude;
            p.y = y + o.border + o.extrude;
            p.rotated = rot;
            if rot {
                core::mem::swap(&mut p.w, &mut p.h);
            }
        }
    }
    for i in 0..placements.len() {
        let a = alias[i];
        if a != i {
            let src = placements[a];
            let p = &mut placements[i];
            (p.page, p.x, p.y, p.w, p.h, p.rotated) =
                (src.page, src.x, src.y, src.w, src.h, src.rotated);
        }
    }
    Ok((page_sizes, placements))
}

// ---------------------------------------------------------------------------------------------
// Pixels

fn f32_at(b: &[u8], i: usize) -> f32 {
    let k = i * 4;
    f32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]])
}

fn has_alpha(f: Format) -> bool {
    use Format::*;
    matches!(
        f,
        Rgba8Unorm
            | Rgba8Srgb
            | Bgra8Unorm
            | Bgra8Srgb
            | Rgba4444
            | Rgba16Float
            | Rgba16Unorm
            | Rgba32Float
    )
}

/// Kept rectangle of `s` for `trim`.
fn trim_rect(s: &Surface, trim: Trim) -> Result<(u32, u32, u32, u32)> {
    let (w, h) = (s.width(), s.height());
    let Trim::Alpha(t) = trim else {
        return Ok((0, 0, w, h));
    };
    if !has_alpha(s.format()) {
        return Ok((0, 0, w, h));
    }
    let f = crate::convert::convert_to_buffer(s, Format::Rgba32Float)?;
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            let a = f32_at(&f, (y as usize * w as usize + x as usize) * 4 + 3);
            if a > t {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x1 == 0 {
        return Ok((0, 0, 1, 1));
    }
    Ok((x0, y0, x1 - x0, y1 - y0))
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, &x| {
        (h ^ u64::from(x)).wrapping_mul(0x100_0000_01B3)
    })
}

/// Packs and renders `sprites` (any uncompressed formats) into atlas pages of
/// `opts.format`. Unused page pixels are zero (transparent black).
///
/// Errors: as [`layout`]; `Unsupported` for compressed formats.
pub fn build(sprites: &[Surface], opts: &AtlasOptions) -> Result<Atlas> {
    let fmt = opts.format;
    if fmt.is_compressed() {
        return Err(Error::Unsupported);
    }
    let bpp = fmt.block_bytes();
    // Every sprite in the page format, and its trim rectangle.
    let mut conv = Vector::<Buffer, Dynamic>::default();
    let mut sizes = Vector::<(u32, u32), Dynamic>::default();
    let mut trims = Vector::<(u32, u32, u32, u32), Dynamic>::default();
    for s in sprites {
        if s.format().is_compressed() {
            return Err(Error::Unsupported);
        }
        conv.push(crate::convert::convert_to_buffer(s, fmt)?)
            .map_err(|_| Error::OutOfMemory)?;
        sizes
            .push((s.width(), s.height()))
            .map_err(|_| Error::OutOfMemory)?;
        trims
            .push(trim_rect(s, opts.trim)?)
            .map_err(|_| Error::OutOfMemory)?;
    }
    // Duplicates: same trimmed size and pixels.
    let trimmed = |i: usize| -> Vector<u8, Dynamic> {
        let (tx, ty, tw, th) = trims[i];
        let sw = sizes[i].0 as usize;
        let mut v = Vector::<u8, Dynamic>::default();
        for y in ty as usize..(ty + th) as usize {
            let a = (y * sw + tx as usize) * bpp;
            let _ = v.extend_from_slice(&conv[i][a..a + tw as usize * bpp]);
        }
        v
    };
    let mut alias = Vector::<usize, Dynamic>::default();
    let mut hashes = Vector::<u64, Dynamic>::default();
    for i in 0..sprites.len() {
        let mut a = i;
        let h = if opts.dedupe { fnv(&trimmed(i)) } else { 0 };
        if opts.dedupe {
            for j in 0..i {
                if alias[j] == j
                    && hashes[j] == h
                    && trims[j].2 == trims[i].2
                    && trims[j].3 == trims[i].3
                    && trimmed(j)[..] == trimmed(i)[..]
                {
                    a = j;
                    break;
                }
            }
        }
        alias.push(a).map_err(|_| Error::OutOfMemory)?;
        hashes.push(h).map_err(|_| Error::OutOfMemory)?;
    }
    let (page_sizes, placements) = place(&sizes, &trims, &alias, opts)?;

    let mut pages = Vector::<Image, Dynamic>::default();
    for &(w, h) in page_sizes.iter() {
        pages
            .push(Image::new(fmt, w, h)?)
            .map_err(|_| Error::OutOfMemory)?;
    }
    let e = opts.extrude as i64;
    for (i, p) in placements.iter().enumerate() {
        if p.alias_of != i {
            continue;
        }
        let page = pages.get_mut(p.page).ok_or(Error::InvalidData)?;
        let pw = page.width as usize;
        let sw = sizes[i].0 as usize;
        let src = &conv[i];
        // Region pixel (rx, ry) (rotated layout) -> trimmed source pixel.
        let at = |rx: i64, ry: i64| -> usize {
            let rx = rx.clamp(0, i64::from(p.w) - 1) as u32;
            let ry = ry.clamp(0, i64::from(p.h) - 1) as u32;
            let (sx, sy) = if p.rotated {
                (ry, p.trim_h - 1 - rx)
            } else {
                (rx, ry)
            };
            ((p.trim_y + sy) as usize * sw + (p.trim_x + sx) as usize) * bpp
        };
        for ry in -e..i64::from(p.h) + e {
            for rx in -e..i64::from(p.w) + e {
                let dx = (i64::from(p.x) + rx) as usize;
                let dy = (i64::from(p.y) + ry) as usize;
                let d = (dy * pw + dx) * bpp;
                let s = at(rx, ry);
                if let (Some(dst), Some(px)) = (page.data.get_mut(d..d + bpp), src.get(s..s + bpp))
                {
                    dst.copy_from_slice(px);
                }
            }
        }
    }
    Ok(Atlas { pages, placements })
}

// ---------------------------------------------------------------------------------------------
// JSON

fn put(out: &mut Buffer, s: &str) -> Result<()> {
    out.extend_from_slice(s.as_bytes())
        .map_err(|_| Error::OutOfMemory)
}

fn put_str(out: &mut Buffer, s: &str) -> Result<()> {
    put(out, "\"")?;
    for c in s.chars() {
        match c {
            '"' => put(out, "\\\"")?,
            '\\' => put(out, "\\\\")?,
            '\n' => put(out, "\\n")?,
            '\r' => put(out, "\\r")?,
            '\t' => put(out, "\\t")?,
            c if (c as u32) < 0x20 => {
                let mut b = [0u8; 6];
                let hex = b"0123456789abcdef";
                b[..4].copy_from_slice(b"\\u00");
                b[4] = hex[(c as usize) >> 4];
                b[5] = hex[(c as usize) & 15];
                out.extend_from_slice(&b).map_err(|_| Error::OutOfMemory)?;
            }
            c => {
                let mut b = [0u8; 4];
                put(out, c.encode_utf8(&mut b))?;
            }
        }
    }
    put(out, "\"")
}

fn put_num(out: &mut Buffer, v: u64) -> Result<()> {
    let mut b = [0u8; 20];
    let mut i = b.len();
    let mut v = v;
    loop {
        i -= 1;
        b[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    out.extend_from_slice(&b[i..])
        .map_err(|_| Error::OutOfMemory)
}

fn put_rect(out: &mut Buffer, key: &str, v: &[(&str, u32)]) -> Result<()> {
    put_str(out, key)?;
    put(out, ":{")?;
    for (k, (name, n)) in v.iter().enumerate() {
        if k > 0 {
            put(out, ",")?;
        }
        put_str(out, name)?;
        put(out, ":")?;
        put_num(out, u64::from(*n))?;
    }
    put(out, "}")
}

fn page_name(image: &str, i: usize) -> Result<Buffer> {
    let mut b = Buffer::default();
    match image.split_once("{}") {
        Some((a, z)) => {
            put(&mut b, a)?;
            put_num(&mut b, i as u64)?;
            put(&mut b, z)?;
        }
        None => put(&mut b, image)?,
    }
    Ok(b)
}

fn json(
    pages: &[(u32, u32)],
    placements: &[Placement],
    names: &[&str],
    image: &str,
) -> Result<Buffer> {
    if names.len() != placements.len() {
        return Err(Error::InvalidArgument);
    }
    let mut out = Buffer::default();
    put(&mut out, "{\"frames\":{")?;
    for (i, (p, name)) in placements.iter().zip(names).enumerate() {
        if i > 0 {
            put(&mut out, ",")?;
        }
        put(&mut out, "\n")?;
        put_str(&mut out, name)?;
        put(&mut out, ":{")?;
        put_rect(
            &mut out,
            "frame",
            &[("x", p.x), ("y", p.y), ("w", p.trim_w), ("h", p.trim_h)],
        )?;
        put(&mut out, ",\"rotated\":")?;
        put(&mut out, if p.rotated { "true" } else { "false" })?;
        put(&mut out, ",\"trimmed\":")?;
        let trimmed = (p.trim_w, p.trim_h) != (p.source_w, p.source_h);
        put(&mut out, if trimmed { "true" } else { "false" })?;
        put(&mut out, ",")?;
        put_rect(
            &mut out,
            "spriteSourceSize",
            &[
                ("x", p.trim_x),
                ("y", p.trim_y),
                ("w", p.trim_w),
                ("h", p.trim_h),
            ],
        )?;
        put(&mut out, ",")?;
        put_rect(
            &mut out,
            "sourceSize",
            &[("w", p.source_w), ("h", p.source_h)],
        )?;
        put(&mut out, ",\"page\":")?;
        put_num(&mut out, p.page as u64)?;
        put(&mut out, "}")?;
    }
    put(&mut out, "\n},\n\"meta\":{\"app\":\"imgio\",\"version\":")?;
    put_str(&mut out, env!("CARGO_PKG_VERSION"))?;
    put(&mut out, ",\"image\":")?;
    let first = page_name(image, 0)?;
    put_str(
        &mut out,
        core::str::from_utf8(&first).map_err(|_| Error::InvalidData)?,
    )?;
    put(&mut out, ",")?;
    let (w0, h0) = pages.first().copied().unwrap_or((0, 0));
    put_rect(&mut out, "size", &[("w", w0), ("h", h0)])?;
    put(&mut out, ",\"scale\":\"1\",\"pages\":[")?;
    for (i, &(w, h)) in pages.iter().enumerate() {
        if i > 0 {
            put(&mut out, ",")?;
        }
        put(&mut out, "{\"image\":")?;
        let n = page_name(image, i)?;
        put_str(
            &mut out,
            core::str::from_utf8(&n).map_err(|_| Error::InvalidData)?,
        )?;
        put(&mut out, ",")?;
        put_rect(&mut out, "size", &[("w", w), ("h", h)])?;
        put(&mut out, "}")?;
    }
    put(&mut out, "]}}\n")?;
    Ok(out)
}
