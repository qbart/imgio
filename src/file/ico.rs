//! Windows icons and cursors (`.ico`, `.cur`; own implementation).
//!
//! A file is a directory of entries, each a PNG stream or a DIB (BITMAPINFOHEADER bitmap
//! with a doubled height, followed by a 1-bit AND mask). Cursors (type 2) store a hotspot per
//! entry where icons store planes / bit count.
//!
//! Decode (like Pillow's `IcoImagePlugin`): [`decode`] returns the largest entry (most pixels,
//! then highest color depth, first on ties), [`decode_entry`] any entry of [`info`]. PNG entries
//! decode with [`super::png`] (native format); DIB entries go through [`super::bmp`] and become
//! `Rgba8Srgb`: 32-bit entries take alpha from their fourth byte, other depths from the AND
//! mask (set bit = transparent). [`info`] lists the entries with their sizes, color depths and
//! cursor hotspots.
//!
//! Encode: [`encode`] scales the source (Lanczos, sRGB-correct, premultiplied, aspect kept like
//! Pillow's `thumbnail`) to every size of [`EncodeOptions::sizes`] that fits (default 16, 24,
//! 32, 48, 64, 128, 256, or the source size when it is under 16); [`encode_entries`] writes given
//! images. Entries are 32-bit BGRA DIBs with an AND mask (alpha 0 = transparent) or PNG streams
//! ([`PngEntries`]; PNG needs the `png` feature). A hotspot makes a cursor.

use super::Image;
use crate::resize::{Fit, ResizeOptions, resize};
use crate::{Buffer, Error, Format, Result, Surface};
use vas::Vector;
use vas::storage::Dynamic;

/// Default icon sizes.
pub const DEFAULT_SIZES: [u32; 7] = [16, 24, 32, 48, 64, 128, 256];

/// One directory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// Size from the directory (0 read as 256).
    pub width: u32,
    pub height: u32,
    /// Bits per pixel from the directory (icons; 0 when unknown or for cursors).
    pub bits: u16,
    /// Palette colors from the directory (0 = none or 256).
    pub colors: u8,
    /// Cursor hotspot (cursors only).
    pub hotspot: Option<(u16, u16)>,
    /// The entry is a PNG stream.
    pub png: bool,
    pub offset: usize,
    pub size: usize,
}

impl Entry {
    /// Pillow's color depth key: bit count, else log2 of the palette size, else 256.
    fn depth(&self) -> u32 {
        if self.bits != 0 {
            u32::from(self.bits)
        } else if self.colors != 0 {
            u32::from(self.colors).next_power_of_two().trailing_zeros()
        } else {
            256
        }
    }
}

/// Parsed directory.
#[derive(Debug)]
pub struct IconInfo {
    /// Type 2 (cursor) rather than 1 (icon).
    pub cursor: bool,
    pub entries: Vector<Entry, Dynamic>,
}

impl IconInfo {
    /// Index of the entry [`decode`] returns.
    pub fn largest(&self) -> Option<usize> {
        let mut best: Option<(usize, u64, u32)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let k = (u64::from(e.width) * u64::from(e.height), e.depth());
            if best.is_none_or(|(_, a, d)| k > (a, d)) {
                best = Some((i, k.0, k.1));
            }
        }
        best.map(|b| b.0)
    }
}

fn u16_at(d: &[u8], o: usize) -> Result<u16> {
    let b = d.get(o..o + 2).ok_or(Error::InvalidData)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(d: &[u8], o: usize) -> Result<u32> {
    let b = d.get(o..o + 4).ok_or(Error::InvalidData)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// True for a plausible ICO / CUR header (used by [`super::FileFormat::detect`]).
pub(crate) fn looks_like_ico(d: &[u8]) -> bool {
    let (Ok(t), Ok(n)) = (u16_at(d, 2), u16_at(d, 4)) else {
        return false;
    };
    d.get(..2) == Some(&[0, 0])
        && matches!(t, 1 | 2)
        && n > 0
        && d.get(9) == Some(&0)
        && u32_at(d, 18).is_ok_and(|o| o as usize >= 6 + 16 * usize::from(n))
}

/// Reads the directory.
pub fn info(data: &[u8]) -> Result<IconInfo> {
    if data.get(..2) != Some(&[0, 0]) {
        return Err(Error::InvalidData);
    }
    let cursor = match u16_at(data, 2)? {
        1 => false,
        2 => true,
        _ => return Err(Error::InvalidData),
    };
    let n = usize::from(u16_at(data, 4)?);
    if n == 0 {
        return Err(Error::InvalidData);
    }
    let mut entries = Vector::<Entry, Dynamic>::default();
    entries.reserve(n).map_err(|_| Error::OutOfMemory)?;
    for i in 0..n {
        let o = 6 + 16 * i;
        let d = data.get(o..o + 16).ok_or(Error::InvalidData)?;
        let dim = |v: u8| if v == 0 { 256 } else { u32::from(v) };
        let (a, b) = (u16_at(d, 4)?, u16_at(d, 6)?);
        let size = u32_at(d, 8)? as usize;
        let offset = u32_at(d, 12)? as usize;
        let body = data
            .get(offset..offset.checked_add(size).ok_or(Error::InvalidData)?)
            .ok_or(Error::InvalidData)?;
        entries
            .push(Entry {
                width: dim(d[0]),
                height: dim(d[1]),
                colors: d[2],
                bits: if cursor { 0 } else { b },
                hotspot: cursor.then_some((a, b)),
                png: body.starts_with(b"\x89PNG\r\n\x1a\n"),
                offset,
                size,
            })
            .map_err(|_| Error::OutOfMemory)?;
    }
    Ok(IconInfo { cursor, entries })
}

/// Decodes the largest entry.
pub fn decode(data: &[u8]) -> Result<Image> {
    let inf = info(data)?;
    let i = inf.largest().ok_or(Error::InvalidData)?;
    decode_entry(data, i)
}

/// Decodes entry `index` of [`info`].
pub fn decode_entry(data: &[u8], index: usize) -> Result<Image> {
    let inf = info(data)?;
    let e = inf.entries.get(index).ok_or(Error::InvalidArgument)?;
    let body = data
        .get(e.offset..e.offset + e.size)
        .ok_or(Error::InvalidData)?;
    if e.png {
        super::png::decode(body)
    } else {
        decode_dib(body)
    }
}

/// Every entry, in directory order.
pub fn decode_all(data: &[u8]) -> Result<Vector<Image, Dynamic>> {
    let n = info(data)?.entries.len();
    let mut out = Vector::<Image, Dynamic>::default();
    out.reserve(n).map_err(|_| Error::OutOfMemory)?;
    for i in 0..n {
        out.push(decode_entry(data, i)?)
            .map_err(|_| Error::OutOfMemory)?;
    }
    Ok(out)
}

/// A DIB entry: XOR bitmap through the BMP decoder, alpha from the 32-bit pixels or the AND
/// mask.
fn decode_dib(dib: &[u8]) -> Result<Image> {
    let hsize = u32_at(dib, 0)? as usize;
    if !(40..=124).contains(&hsize) || dib.len() < hsize {
        return Err(Error::InvalidData);
    }
    let w = i32::from_le_bytes([dib[4], dib[5], dib[6], dib[7]]);
    let h2 = i32::from_le_bytes([dib[8], dib[9], dib[10], dib[11]]);
    let bpp = usize::from(u16_at(dib, 14)?);
    let compression = u32_at(dib, 16)?;
    let used = u32_at(dib, 32)? as usize;
    if w <= 0 || h2 < 2 || compression > 3 {
        return Err(Error::InvalidData);
    }
    let (w, h) = (w as usize, h2 as usize / 2);
    let masks = if compression == 3 && hsize == 40 {
        12
    } else {
        0
    };
    let colors = if bpp <= 8 {
        if used == 0 || used > 1 << bpp {
            1 << bpp
        } else {
            used
        }
    } else {
        used.min(256)
    };
    let px_off = hsize + masks + 4 * colors;

    // BMP file = 14 byte file header + the DIB with the XOR height.
    let mut bmp = crate::buffer::zeroed(14 + dib.len())?;
    bmp[..2].copy_from_slice(b"BM");
    bmp[2..6].copy_from_slice(&((14 + dib.len()) as u32).to_le_bytes());
    bmp[10..14].copy_from_slice(&((14 + px_off) as u32).to_le_bytes());
    bmp[14..].copy_from_slice(dib);
    bmp[22..26].copy_from_slice(&(h as u32).to_le_bytes());
    let xor = super::bmp::decode(&bmp)?;
    if (xor.width as usize, xor.height as usize) != (w, h) {
        return Err(Error::InvalidData);
    }

    let mut img = Image::new(Format::Rgba8Srgb, w as u32, h as u32)?;
    let cpp = match xor.format {
        Format::R8Unorm => 1,
        Format::Rg8Unorm => 2,
        Format::Rgb8Srgb => 3,
        Format::Rgba8Srgb => 4,
        _ => return Err(Error::InvalidData),
    };
    for (o, s) in img
        .data
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(xor.data.chunks_exact(cpp))
    {
        *o = match *s {
            [g] => [g, g, g, 255],
            [g, a] => [g, g, g, a],
            [r, g, b] => [r, g, b, 255],
            [r, g, b, a] => [r, g, b, a],
            _ => [0; 4],
        };
    }
    let stride = (w * bpp).div_ceil(32) * 4;
    let xor_end = px_off + stride * h;
    if bpp == 32 {
        // Alpha = the fourth byte of each stored (bottom-up) pixel.
        let px = dib.get(px_off..xor_end).ok_or(Error::InvalidData)?;
        for y in 0..h {
            let row = &px[(h - 1 - y) * stride..];
            for x in 0..w {
                img.data[(y * w + x) * 4 + 3] = row[x * 4 + 3];
            }
        }
    } else {
        let mstride = w.div_ceil(32) * 4;
        let mask = dib
            .get(xor_end..xor_end + mstride * h)
            .ok_or(Error::InvalidData)?;
        for y in 0..h {
            let row = &mask[(h - 1 - y) * mstride..];
            for x in 0..w {
                let bit = row[x / 8] >> (7 - x % 8) & 1;
                img.data[(y * w + x) * 4 + 3] = if bit == 1 { 0 } else { 255 };
            }
        }
    }
    Ok(img)
}

/// Which entries are PNG streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PngEntries {
    /// 256-pixel entries (Windows Vista+ convention), DIBs below.
    #[default]
    Large,
    /// Every entry (smallest files; Pillow's default).
    All,
    /// None (DIBs only, for old readers).
    None,
}

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EncodeOptions<'a> {
    /// Square sizes to write ([`encode`]); empty = [`DEFAULT_SIZES`]. Sizes above the source
    /// (or 256) are skipped.
    pub sizes: &'a [u32],
    pub png: PngEntries,
    /// Write a cursor with this hotspot (in source pixels, scaled to each entry).
    pub hotspot: Option<(u32, u32)>,
}

/// RGBA8 sRGB copy of `src`.
fn rgba8(src: &Surface) -> Result<Image> {
    let (px, cpp) = super::bmp::ldr_pixels(src)?;
    let mut img = Image::new(Format::Rgba8Srgb, src.width(), src.height())?;
    for (o, s) in img
        .data
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(px.chunks_exact(cpp))
    {
        *o = match *s {
            [g] => [g, g, g, 255],
            [r, g, b] => [r, g, b, 255],
            [r, g, b, a] => [r, g, b, a],
            _ => [0; 4],
        };
    }
    Ok(img)
}

/// Encodes `src` at every fitting size of `opts.sizes`.
///
/// Errors: `Unsupported` for formats the BMP encoder cannot take (above 8 bits per channel),
/// `InvalidArgument` if no size fits.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    let base = rgba8(src)?;
    let (w, h) = (src.width(), src.height());
    let mut sizes: Vector<u32, Dynamic> = Vector::default();
    if w.min(h) < 16 && opts.sizes.is_empty() {
        sizes.push(0).map_err(|_| Error::OutOfMemory)?;
    } else {
        let list: &[u32] = if opts.sizes.is_empty() {
            &DEFAULT_SIZES
        } else {
            opts.sizes
        };
        let mut l: Vector<u32, Dynamic> = Vector::default();
        for &s in list {
            if s > 0 && s <= w.min(h).min(256) && !l.contains(&s) {
                l.push(s).map_err(|_| Error::OutOfMemory)?;
            }
        }
        l.sort_unstable();
        sizes = l;
    }
    if sizes.is_empty() {
        return Err(Error::InvalidArgument);
    }
    let ropts = ResizeOptions {
        fit: Fit::Contain,
        ..Default::default()
    };
    let mut images = Vector::<Image, Dynamic>::default();
    let mut hot = Vector::<Option<(u32, u32)>, Dynamic>::default();
    for &s in sizes.iter() {
        let img = if s == 0 || (s == w && s == h) {
            rgba8(&base.surface()?)?
        } else {
            resize(&base.surface()?, s, s, &ropts)?
        };
        let hs = opts.hotspot.map(|(x, y)| {
            let sc = |v: u32, from: u32, to: u32| {
                ((u64::from(v) * u64::from(to) + u64::from(from) / 2) / u64::from(from.max(1)))
                    .min(u64::from(to.saturating_sub(1))) as u32
            };
            (sc(x, w, img.width), sc(y, h, img.height))
        });
        hot.push(hs).map_err(|_| Error::OutOfMemory)?;
        images.push(img).map_err(|_| Error::OutOfMemory)?;
    }
    let surfaces: Vector<Surface, Dynamic> = {
        let mut v = Vector::default();
        for i in images.iter() {
            v.push(i.surface()?).map_err(|_| Error::OutOfMemory)?;
        }
        v
    };
    write(&surfaces, &hot, opts.png)
}

/// Writes the given images as entries (each at most 256 x 256), as an icon or, with
/// `hotspots` (one per image), a cursor.
pub fn encode_entries(
    images: &[Surface],
    hotspots: Option<&[(u32, u32)]>,
    png: PngEntries,
) -> Result<Buffer> {
    let mut hot = Vector::<Option<(u32, u32)>, Dynamic>::default();
    for i in 0..images.len() {
        let h = match hotspots {
            Some(hs) => Some(*hs.get(i).ok_or(Error::InvalidArgument)?),
            None => None,
        };
        hot.push(h).map_err(|_| Error::OutOfMemory)?;
    }
    write(images, &hot, png)
}

fn write(images: &[Surface], hot: &[Option<(u32, u32)>], png: PngEntries) -> Result<Buffer> {
    let n = images.len();
    if n == 0 || n > usize::from(u16::MAX) {
        return Err(Error::InvalidArgument);
    }
    let cursor = hot.first().is_some_and(|h| h.is_some());
    let mut bodies = Vector::<Buffer, Dynamic>::default();
    for s in images {
        let (w, h) = (s.width(), s.height());
        if w > 256 || h > 256 {
            return Err(Error::InvalidArgument);
        }
        let img = rgba8(s)?;
        let as_png = match png {
            PngEntries::All => true,
            PngEntries::Large => w == 256 || h == 256,
            PngEntries::None => false,
        };
        let body = if as_png {
            let o = super::png::EncodeOptions {
                srgb_chunk: false,
                ..Default::default()
            };
            super::png::encode(&img.surface()?, &o)?
        } else {
            dib(&img)?
        };
        bodies.push(body).map_err(|_| Error::OutOfMemory)?;
    }
    let total = 6 + 16 * n + bodies.iter().map(|b| b.len()).sum::<usize>();
    if total > u32::MAX as usize {
        return Err(Error::Unsupported);
    }
    let mut out = Buffer::default();
    out.reserve(total).map_err(|_| Error::OutOfMemory)?;
    let mut push = |b: &[u8]| out.extend_from_slice(b).map_err(|_| Error::OutOfMemory);
    push(&[0, 0])?;
    push(&(if cursor { 2u16 } else { 1 }).to_le_bytes())?;
    push(&(n as u16).to_le_bytes())?;
    let mut offset = 6 + 16 * n;
    for ((s, b), h) in images.iter().zip(bodies.iter()).zip(hot) {
        let dim = |v: u32| if v >= 256 { 0 } else { v as u8 };
        push(&[dim(s.width()), dim(s.height()), 0, 0])?;
        match h {
            Some((x, y)) => {
                push(&(*x).min(0xFFFF).to_le_bytes()[..2])?;
                push(&(*y).min(0xFFFF).to_le_bytes()[..2])?;
            }
            None => {
                push(&1u16.to_le_bytes())?;
                push(&32u16.to_le_bytes())?;
            }
        }
        push(&(b.len() as u32).to_le_bytes())?;
        push(&(offset as u32).to_le_bytes())?;
        offset += b.len();
    }
    for b in bodies.iter() {
        push(b)?;
    }
    Ok(out)
}

/// 32-bit BGRA DIB (bottom-up) + AND mask (alpha 0 = transparent).
fn dib(img: &Image) -> Result<Buffer> {
    let (w, h) = (img.width as usize, img.height as usize);
    let stride = w * 4;
    let mstride = w.div_ceil(32) * 4;
    let size = 40 + stride * h + mstride * h;
    let mut b = crate::buffer::zeroed(size)?;
    b[0..4].copy_from_slice(&40u32.to_le_bytes());
    b[4..8].copy_from_slice(&(w as u32).to_le_bytes());
    b[8..12].copy_from_slice(&(2 * h as u32).to_le_bytes());
    b[12..14].copy_from_slice(&1u16.to_le_bytes());
    b[14..16].copy_from_slice(&32u16.to_le_bytes());
    b[20..24].copy_from_slice(&((stride * h + mstride * h) as u32).to_le_bytes());
    for y in 0..h {
        let src = &img.data[(h - 1 - y) * stride..(h - y) * stride];
        let row = 40 + y * stride;
        let mrow = 40 + stride * h + y * mstride;
        for x in 0..w {
            let p = &src[x * 4..x * 4 + 4];
            b[row + x * 4..row + x * 4 + 4].copy_from_slice(&[p[2], p[1], p[0], p[3]]);
            if p[3] == 0 {
                b[mrow + x / 8] |= 0x80 >> (x % 8);
            }
        }
    }
    Ok(b)
}
