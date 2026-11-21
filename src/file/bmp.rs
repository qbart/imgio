//! Windows / OS/2 bitmap (own implementation).
//!
//! Decoding covers BITMAPCOREHEADER (OS/2 1.x), OS/2 2.x (16..64 byte) headers,
//! BITMAPINFOHEADER and its V2 (52), V3 (56), V4 (108) and V5 (124) extensions; 1/2/4/8-bit
//! palettes, 16-bit (5:5:5 default, bit fields), 24-bit and 32-bit (default BGRX, bit fields,
//! alpha bit fields); RLE8 and RLE4 with end-of-line, delta and end-of-bitmap escapes; bottom-up
//! and top-down rows. Embedded JPEG / PNG streams (`BI_JPEG`, `BI_PNG`) go to
//! [`super::jpeg`] / [`super::png`]. OS/2 Huffman 1D and RLE24 are `Unsupported`.
//!
//! Decoded formats:
//! - Palette images whose palette is all gray (`r == g == b`) decode to `R8Unorm`, other palette
//!   images to `Rgb8Srgb`.
//! - Bit field / 16 / 24 / 32-bit images decode to `Rgb8Srgb`, or `Rgba8Srgb` when an alpha mask
//!   is present (V3+ header or `BI_ALPHABITFIELDS`), or when a 32-bit `BI_RGB` file has a
//!   non-zero byte in its fourth (unused) channel. A channel of `n` bits expands to 8 bits as
//!   `round(v * 255 / (2^n - 1))`.
//! - RLE pixels never written (skipped by delta / end-of-line / end-of-bitmap) are transparent:
//!   such images decode to `Rgba8Srgb` (or `Rg8Unorm` for a gray palette) with alpha 0 there
//!   and 255 elsewhere; fully drawn RLE images use the palette formats above.
//!
//! Limits: at most 2^28 pixels; RLE files at most 512 pixels per compressed byte (runs
//! reach 127, more only with large skipped areas).
//!
//! Encoding is uncompressed and bottom-up: `R8Unorm` as an 8-bit gray palette
//! (BITMAPINFOHEADER), RGB as 24-bit (BITMAPINFOHEADER), RGBA as 32-bit `BI_BITFIELDS` with a
//! V4 header (alpha mask `0xff000000`, sRGB color space). `Rg8Unorm` (gray + alpha) is written
//! as RGBA, `Bgra8*` / `Rgb565` / `Bgr565` / `Rgba4444` are expanded to 8 bits; other formats
//! are `Unsupported` (no silent precision loss).

use super::Image;
use crate::convert::convert_to_buffer;
use crate::parallel::for_each_unit;
use crate::{Buffer, Error, Format, Result, Surface};

/// Encoder options (BMP is always written uncompressed).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EncodeOptions {}

/// Most pixels a decoder accepts.
pub(crate) const MAX_PIXELS: u64 = 1 << 28;
/// Most pixels per compressed byte for RLE files.
const RLE_PIXELS_PER_BYTE: u64 = 512;

const BI_RGB: u32 = 0;
const BI_RLE8: u32 = 1;
const BI_RLE4: u32 = 2;
const BI_BITFIELDS: u32 = 3;
const BI_JPEG: u32 = 4;
const BI_PNG: u32 = 5;
const BI_ALPHABITFIELDS: u32 = 6;

fn bytes<const N: usize>(d: &[u8], o: usize) -> Result<[u8; N]> {
    d.get(o..)
        .and_then(|s| s.first_chunk::<N>())
        .copied()
        .ok_or(Error::InvalidData)
}

fn u16_at(d: &[u8], o: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(bytes(d, o)?))
}

fn u32_at(d: &[u8], o: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(bytes(d, o)?))
}

/// One bit field channel.
#[derive(Debug, Clone, Copy, Default)]
struct Channel {
    mask: u32,
    shift: u32,
    max: u64,
}

impl Channel {
    fn new(mask: u32) -> Self {
        if mask == 0 {
            return Channel::default();
        }
        let shift = mask.trailing_zeros();
        let bits = 32 - (mask >> shift).leading_zeros();
        Channel {
            mask,
            shift,
            max: (1u64 << bits) - 1,
        }
    }

    fn get(&self, p: u32) -> u8 {
        if self.max == 0 {
            return 0;
        }
        let v = u64::from((p & self.mask) >> self.shift);
        ((v * 255 + self.max / 2) / self.max) as u8
    }
}

struct Header {
    width: usize,
    height: usize,
    top_down: bool,
    bpp: usize,
    compression: u32,
    masks: [u32; 4],
    palette: [[u8; 3]; 256],
    colors: usize,
    pixels: usize,
}

impl Header {
    fn parse(d: &[u8]) -> Result<Header> {
        if d.get(..2) != Some(b"BM") {
            return Err(Error::InvalidData);
        }
        let off_bits = u32_at(d, 10)? as usize;
        let hsize = u32_at(d, 14)? as usize;
        let core = hsize == 12;
        let (w, h, bpp, compression, colors_used) = if core {
            (
                i64::from(u16_at(d, 18)?),
                i64::from(u16_at(d, 20)?),
                u16_at(d, 24)?,
                BI_RGB,
                0,
            )
        } else if hsize >= 16 {
            let opt = |min: usize, o: usize| {
                if hsize >= min { u32_at(d, o) } else { Ok(0) }
            };
            (
                i64::from(i32::from_le_bytes(bytes(d, 18)?)),
                i64::from(i32::from_le_bytes(bytes(d, 22)?)),
                u16_at(d, 28)?,
                opt(20, 30)?,
                opt(36, 46)?,
            )
        } else {
            return Err(Error::InvalidData);
        };
        // OS/2 2.x headers: 16..=64 bytes except the Windows sizes.
        let os2v2 = !core && hsize <= 64 && !matches!(hsize, 40 | 52 | 56);
        if os2v2 && compression >= 3 {
            return Err(Error::Unsupported); // Huffman 1D, RLE24
        }
        let header_end = 14usize.saturating_add(hsize);
        let mut pal_start = header_end;
        let mut masks = [0u32; 4];
        if matches!(compression, BI_BITFIELDS | BI_ALPHABITFIELDS) {
            if hsize >= 52 {
                for (c, m) in masks.iter_mut().enumerate().take(3) {
                    *m = u32_at(d, 54 + 4 * c)?;
                }
                if hsize >= 56 {
                    masks[3] = u32_at(d, 66)?;
                }
            } else {
                let n = if compression == BI_ALPHABITFIELDS {
                    4
                } else {
                    3
                };
                for (c, m) in masks.iter_mut().enumerate().take(n) {
                    *m = u32_at(d, header_end + 4 * c)?;
                }
                pal_start += 4 * n;
            }
        }
        if w <= 0 || h == 0 {
            return Err(Error::InvalidData);
        }
        let (width, height) = (w as usize, h.unsigned_abs() as usize);
        if (width as u64).saturating_mul(height as u64) > MAX_PIXELS {
            return Err(Error::InvalidData);
        }
        let bpp = bpp as usize;
        let ok = match compression {
            BI_RGB => matches!(bpp, 1 | 2 | 4 | 8 | 16 | 24 | 32),
            BI_RLE8 => bpp == 8,
            BI_RLE4 => bpp == 4,
            BI_BITFIELDS | BI_ALPHABITFIELDS => matches!(bpp, 16 | 24 | 32),
            BI_JPEG | BI_PNG => true,
            _ => return Err(Error::Unsupported),
        };
        if !ok {
            return Err(Error::InvalidData);
        }
        if off_bits < header_end || off_bits > d.len() {
            return Err(Error::InvalidData);
        }
        let mut palette = [[0u8; 3]; 256];
        let mut colors = 0;
        if bpp <= 8 && !matches!(compression, BI_JPEG | BI_PNG) {
            let entry = if core { 3 } else { 4 };
            let max = 1usize << bpp;
            let wanted = if colors_used == 0 || colors_used as usize > max {
                max
            } else {
                colors_used as usize
            };
            colors = wanted.min(off_bits.saturating_sub(pal_start) / entry);
            for (i, p) in palette.iter_mut().enumerate().take(colors) {
                let [b, g, r] = bytes::<3>(d, pal_start + i * entry)?;
                *p = [r, g, b];
            }
        }
        Ok(Header {
            width,
            height,
            top_down: h < 0,
            bpp,
            compression,
            masks,
            palette,
            colors,
            pixels: off_bits,
        })
    }

    fn gray_palette(&self) -> bool {
        self.colors > 0
            && self
                .palette
                .iter()
                .take(self.colors)
                .all(|p| p[0] == p[1] && p[1] == p[2])
    }

    fn entry(&self, i: usize) -> [u8; 3] {
        if i < self.colors {
            self.palette.get(i).copied().unwrap_or_default()
        } else {
            [0; 3]
        }
    }

    /// Output row of stored row `y`.
    fn out_row(&self, y: usize) -> usize {
        if self.top_down {
            y
        } else {
            self.height - 1 - y
        }
    }
}

/// Decodes the main image.
pub fn decode(data: &[u8]) -> Result<Image> {
    let h = Header::parse(data)?;
    let px = data.get(h.pixels..).ok_or(Error::InvalidData)?;
    match h.compression {
        BI_JPEG => super::jpeg::decode(px),
        BI_PNG => super::png::decode(px),
        BI_RLE8 | BI_RLE4 => decode_rle(&h, px),
        _ if h.bpp <= 8 => decode_indexed(&h, px),
        _ => decode_masked(&h, px),
    }
}

fn row_stride(h: &Header) -> Result<(usize, usize)> {
    let bits = h.width.checked_mul(h.bpp).ok_or(Error::InvalidData)?;
    let stride = bits.div_ceil(32) * 4;
    let used = bits.div_ceil(8);
    let need = stride
        .checked_mul(h.height - 1)
        .and_then(|v| v.checked_add(used))
        .ok_or(Error::InvalidData)?;
    Ok((stride, need))
}

fn index_at(row: &[u8], bits: usize, x: usize) -> usize {
    let bit = x * bits;
    let byte = row.get(bit / 8).copied().unwrap_or(0) as usize;
    (byte >> (8 - bits - bit % 8)) & ((1 << bits) - 1)
}

fn decode_indexed(h: &Header, px: &[u8]) -> Result<Image> {
    let (stride, need) = row_stride(h)?;
    if px.len() < need {
        return Err(Error::InvalidData);
    }
    let gray = h.gray_palette();
    let format = if gray {
        Format::R8Unorm
    } else {
        Format::Rgb8Srgb
    };
    let mut img = Image::new(format, h.width as u32, h.height as u32)?;
    let cpp = if gray { 1 } else { 3 };
    let pixels = h.width * h.height;
    for_each_unit(&mut img.data, h.width * cpp, pixels, |y, out| {
        let sy = h.out_row(y);
        let row = px.get(sy * stride..).unwrap_or_default();
        for (x, o) in out.chunks_exact_mut(cpp).enumerate() {
            let c = h.entry(index_at(row, h.bpp, x));
            o.copy_from_slice(&c[..cpp]);
        }
    });
    Ok(img)
}

fn decode_masked(h: &Header, px: &[u8]) -> Result<Image> {
    let (stride, need) = row_stride(h)?;
    if px.len() < need {
        return Err(Error::InvalidData);
    }
    let bpp = h.bpp / 8;
    let mut masks = h.masks;
    if h.compression == BI_RGB {
        masks = match h.bpp {
            16 => [0x7C00, 0x03E0, 0x001F, 0],
            _ => [0xFF_0000, 0xFF00, 0xFF, 0],
        };
        // 32-bit BI_RGB: the fourth byte is alpha when any pixel sets it.
        if h.bpp == 32 {
            let any = (0..h.height).any(|y| {
                px.get(y * stride..)
                    .and_then(|r| r.get(..h.width * 4))
                    .is_some_and(|r| r.as_chunks::<4>().0.iter().any(|p| p[3] != 0))
            });
            if any {
                masks[3] = 0xFF00_0000;
            }
        }
    }
    let ch = masks.map(Channel::new);
    let alpha = masks[3] != 0;
    let (format, cpp) = if alpha {
        (Format::Rgba8Srgb, 4)
    } else {
        (Format::Rgb8Srgb, 3)
    };
    let mut img = Image::new(format, h.width as u32, h.height as u32)?;
    let pixels = h.width * h.height;
    for_each_unit(&mut img.data, h.width * cpp, pixels, |y, out| {
        let sy = h.out_row(y);
        let row = px.get(sy * stride..).unwrap_or_default();
        for (s, o) in row.chunks_exact(bpp).zip(out.chunks_exact_mut(cpp)) {
            let p = s
                .iter()
                .enumerate()
                .fold(0u32, |p, (k, &b)| p | (u32::from(b) << (8 * k)));
            let v = [ch[0].get(p), ch[1].get(p), ch[2].get(p), ch[3].get(p)];
            o.copy_from_slice(&v[..cpp]);
        }
    });
    Ok(img)
}

fn decode_rle(h: &Header, s: &[u8]) -> Result<Image> {
    let (w, ht) = (h.width, h.height);
    let pixels = w * ht;
    if pixels as u64 > (s.len() as u64).saturating_mul(RLE_PIXELS_PER_BYTE) {
        return Err(Error::InvalidData);
    }
    let rle4 = h.compression == BI_RLE4;
    let mut idx = crate::buffer::zeroed(pixels)?;
    let mut drawn = crate::buffer::zeroed(pixels)?;
    let (mut x, mut y, mut i) = (0usize, 0usize, 0usize);
    let mut put = |x: usize, y: usize, v: u8| {
        if x < w && y < ht {
            let o = h.out_row(y) * w + x;
            if let (Some(a), Some(b)) = (idx.get_mut(o), drawn.get_mut(o)) {
                *a = v;
                *b = 1;
            }
        }
    };
    let byte = |i: usize| s.get(i).copied().ok_or(Error::InvalidData);
    let mut ended = false;
    while y < ht {
        let Ok(a) = byte(i) else { break };
        let b = byte(i + 1)?;
        i += 2;
        if a > 0 {
            for k in 0..a as usize {
                let v = match (rle4, k % 2) {
                    (false, _) => b,
                    (true, 0) => b >> 4,
                    (true, _) => b & 15,
                };
                put(x, y, v);
                x += 1;
            }
            continue;
        }
        match b {
            0 => {
                x = 0;
                y += 1;
            }
            1 => {
                ended = true;
                break;
            }
            2 => {
                x += byte(i)? as usize;
                y += byte(i + 1)? as usize;
                i += 2;
            }
            n => {
                let n = n as usize;
                let len = if rle4 { n.div_ceil(2) } else { n };
                let run = s.get(i..i + len).ok_or(Error::InvalidData)?;
                for k in 0..n {
                    let v = if rle4 {
                        let c = run.get(k / 2).copied().unwrap_or(0);
                        if k % 2 == 0 { c >> 4 } else { c & 15 }
                    } else {
                        run.get(k).copied().unwrap_or(0)
                    };
                    put(x, y, v);
                    x += 1;
                }
                i += len + (len & 1);
            }
        }
    }
    if !ended && y < ht {
        return Err(Error::InvalidData); // truncated stream
    }
    let gray = h.gray_palette();
    let holes = drawn.contains(&0);
    let (format, cpp) = match (gray, holes) {
        (true, false) => (Format::R8Unorm, 1),
        (true, true) => (Format::Rg8Unorm, 2),
        (false, false) => (Format::Rgb8Srgb, 3),
        (false, true) => (Format::Rgba8Srgb, 4),
    };
    let mut img = Image::new(format, w as u32, ht as u32)?;
    for ((o, &v), &d) in img
        .data
        .chunks_exact_mut(cpp)
        .zip(idx.iter())
        .zip(drawn.iter())
    {
        let c = if d != 0 { h.entry(v as usize) } else { [0; 3] };
        let a = if d != 0 { 255 } else { 0 };
        match cpp {
            1 => o.copy_from_slice(&c[..1]),
            2 => o.copy_from_slice(&[c[0], a]),
            3 => o.copy_from_slice(&c),
            _ => o.copy_from_slice(&[c[0], c[1], c[2], a]),
        }
    }
    Ok(img)
}

/// 8-bit pixels of `src` for the LDR encoders, tightly packed, with 1 (gray), 3 (RGB) or 4
/// (RGBA) channels. Gray + alpha becomes RGBA; packed formats are expanded; formats above 8
/// bits per channel are `Unsupported`.
pub(crate) fn ldr_pixels(src: &Surface) -> Result<(Buffer, usize)> {
    let copy = |cpp: usize| -> Result<(Buffer, usize)> {
        let mut b = crate::buffer::zeroed(src.data().len())?;
        b.copy_from_slice(src.data());
        Ok((b, cpp))
    };
    match src.format() {
        Format::R8Unorm => copy(1),
        Format::Rgb8Unorm | Format::Rgb8Srgb => copy(3),
        Format::Rgba8Unorm | Format::Rgba8Srgb => copy(4),
        Format::Rgb565 | Format::Bgr565 => Ok((convert_to_buffer(src, Format::Rgb8Unorm)?, 3)),
        Format::Bgra8Unorm | Format::Bgra8Srgb | Format::Rgba4444 => {
            Ok((convert_to_buffer(src, Format::Rgba8Unorm)?, 4))
        }
        Format::Rg8Unorm => {
            let mut b = crate::buffer::zeroed(src.data().len() * 2)?;
            for (o, s) in b
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(src.data().as_chunks::<2>().0)
            {
                o.copy_from_slice(&[s[0], s[0], s[0], s[1]]);
            }
            Ok((b, 4))
        }
        _ => Err(Error::Unsupported),
    }
}

fn put(dst: &mut [u8], o: usize, v: &[u8]) {
    if let Some(d) = dst.get_mut(o..o + v.len()) {
        d.copy_from_slice(v);
    }
}

/// Encodes `src` (uncompressed, bottom-up).
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    let _ = opts;
    let (px, cpp) = ldr_pixels(src)?;
    let (w, h) = (src.width() as usize, src.height() as usize);
    if w > i32::MAX as usize || h > i32::MAX as usize {
        return Err(Error::Unsupported);
    }
    let (bpp, hsize, colors) = match cpp {
        1 => (8, 40, 256),
        3 => (24, 40, 0),
        _ => (32, 108, 0),
    };
    let stride = (w * bpp / 8).div_ceil(4) * 4;
    let off = 14 + hsize + colors * 4;
    let image_size = stride.checked_mul(h).ok_or(Error::Unsupported)?;
    let size = off + image_size;
    if size > u32::MAX as usize {
        return Err(Error::Unsupported);
    }
    let mut hd = [0u8; 14 + 108 + 1024];
    put(&mut hd, 0, b"BM");
    put(&mut hd, 2, &(size as u32).to_le_bytes());
    put(&mut hd, 10, &(off as u32).to_le_bytes());
    put(&mut hd, 14, &(hsize as u32).to_le_bytes());
    put(&mut hd, 18, &(w as u32).to_le_bytes());
    put(&mut hd, 22, &(h as u32).to_le_bytes());
    put(&mut hd, 26, &1u16.to_le_bytes());
    put(&mut hd, 28, &(bpp as u16).to_le_bytes());
    let compression = if cpp == 4 { BI_BITFIELDS } else { BI_RGB };
    put(&mut hd, 30, &compression.to_le_bytes());
    put(&mut hd, 34, &(image_size as u32).to_le_bytes());
    put(&mut hd, 38, &2835u32.to_le_bytes()); // 72 DPI
    put(&mut hd, 42, &2835u32.to_le_bytes());
    put(&mut hd, 46, &(colors as u32).to_le_bytes());
    if cpp == 4 {
        let masks = [0xFF_0000u32, 0xFF00, 0xFF, 0xFF00_0000];
        for (c, m) in masks.iter().enumerate() {
            put(&mut hd, 54 + 4 * c, &m.to_le_bytes());
        }
        put(&mut hd, 70, b"BGRs"); // LCS_sRGB, little endian 'sRGB'
    }
    for i in 0..colors {
        let g = i as u8;
        put(&mut hd, 14 + hsize + 4 * i, &[g, g, g, 0]);
    }
    let mut out = crate::buffer::zeroed(size)?;
    out.get_mut(..off)
        .ok_or(Error::InvalidData)?
        .copy_from_slice(hd.get(..off).ok_or(Error::InvalidData)?);
    let body = out.get_mut(off..).ok_or(Error::InvalidData)?;
    for_each_unit(body, stride, w * h, |y, row| {
        let sy = h - 1 - y;
        let Some(s) = px.get(sy * w * cpp..(sy + 1) * w * cpp) else {
            return;
        };
        match cpp {
            1 => row[..w].copy_from_slice(s),
            3 => {
                for (o, p) in row
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(s.as_chunks::<3>().0)
                {
                    o.copy_from_slice(&[p[2], p[1], p[0]]);
                }
            }
            _ => {
                for (o, p) in row
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(s.as_chunks::<4>().0)
                {
                    o.copy_from_slice(&[p[2], p[1], p[0], p[3]]);
                }
            }
        }
    });
    Ok(out)
}
