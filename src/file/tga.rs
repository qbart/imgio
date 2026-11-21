//! Truevision TGA (own implementation).
//!
//! Decoding covers image types 1 / 9 (color-mapped, raw / RLE), 2 / 10 (truecolor) and 3 / 11
//! (gray): 8 / 16-bit indices into 15 / 16 / 24 / 32-bit color maps (with a first-entry
//! offset), 15 / 16 / 24 / 32-bit truecolor, 8-bit gray and 16-bit gray + alpha; all four
//! origins (descriptor bits 4 / 5), the image ID field, and the TGA 2.0 footer and extension
//! area. RLE packets may cross scanlines (the whole image is one packet stream).
//!
//! Decoded formats: color `Rgb8Srgb` / `Rgba8Srgb`, gray `R8Unorm` / `Rg8Unorm`. 5-bit
//! channels expand as `round(v * 255 / 31)`. Alpha is decided in this order:
//! 1. the extension area attribute type: 3 (alpha) or 4 (premultiplied alpha) keep it, any other
//!    value drops it;
//! 2. else the descriptor's alpha bit count: non-zero keeps the 16-bit (A1R5G5B5) bit, the
//!    32-bit fourth byte or the gray alpha byte;
//! 3. else 32-bit pixels / color-map entries and 16-bit gray keep alpha only when some alpha
//!    byte is non-zero (writers that leave the count at 0); 15 / 16-bit color has no alpha.
//!
//! Limits: at most 2^28 pixels; RLE data cannot claim more than 128 pixels per byte (the most a
//! packet header can describe).
//!
//! Encoding: gray (`R8Unorm`) as type 3 / 11, RGB as 24-bit and RGBA as 32-bit (8 alpha bits)
//! type 2 / 10, top-left origin, raw or RLE ([`EncodeOptions::rle`]; packets never cross
//! scanlines), followed by the TGA 2.0 footer. Input formats as for BMP
//! ([`super::bmp`]): `Rg8Unorm` becomes RGBA, packed formats expand, deeper formats are
//! `Unsupported`.

use super::Image;
use super::bmp::{MAX_PIXELS, ldr_pixels};
use crate::parallel::for_each_unit;
use crate::{Buffer, Error, Format, Result, Surface};

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EncodeOptions {
    /// Run-length encode the pixels (image types 10 / 11).
    pub rle: bool,
}

/// A decoded TGA with its metadata.
#[derive(Debug)]
pub struct TgaImage {
    pub image: Image,
    /// The image ID field (free-form, up to 255 bytes).
    pub image_id: Buffer,
    /// The extension area attribute type (0 no alpha, 1 / 2 undefined, 3 alpha,
    /// 4 premultiplied alpha), when the file has a TGA 2.0 extension area.
    pub attribute_type: Option<u8>,
}

const FOOTER_SIG: &[u8] = b"TRUEVISION-XFILE.\0";

/// Decodes the main image.
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_tga(data).map(|t| t.image)
}

fn expand5(v: u16) -> u8 {
    ((u32::from(v & 31) * 255 + 15) / 31) as u8
}

/// A 15 / 16-bit A1R5G5B5 value as RGBA.
fn rgb555(v: u16) -> [u8; 4] {
    [
        expand5(v >> 10),
        expand5(v >> 5),
        expand5(v),
        if v & 0x8000 != 0 { 255 } else { 0 },
    ]
}

/// A stored pixel / color-map entry of `bits` bits as RGBA.
fn color(p: &[u8], bits: u8) -> [u8; 4] {
    match (bits, p) {
        (15 | 16, [a, b, ..]) => rgb555(u16::from_le_bytes([*a, *b])),
        (24, [b, g, r, ..]) => [*r, *g, *b, 255],
        (32, [b, g, r, a, ..]) => [*r, *g, *b, *a],
        _ => [0, 0, 0, 255],
    }
}

/// Decodes a TGA with its ID field and extension attribute type.
pub fn decode_tga(d: &[u8]) -> Result<TgaImage> {
    let h: &[u8; 18] = d.first_chunk().ok_or(Error::InvalidData)?;
    let id_len = h[0] as usize;
    let (cmap_type, img_type) = (h[1], h[2]);
    let cmap_first = u16::from_le_bytes([h[3], h[4]]) as usize;
    let cmap_len = u16::from_le_bytes([h[5], h[6]]) as usize;
    let cmap_bits = h[7];
    let w = u16::from_le_bytes([h[12], h[13]]) as usize;
    let ht = u16::from_le_bytes([h[14], h[15]]) as usize;
    let (bpp, desc) = (h[16], h[17]);
    if w == 0 || ht == 0 || cmap_type > 1 {
        return Err(Error::InvalidData);
    }
    let base = img_type & 7;
    let rle = match img_type {
        1..=3 => false,
        9..=11 => true,
        0 => return Err(Error::InvalidData), // no image data
        _ => return Err(Error::Unsupported),
    };
    let mapped = base == 1;
    let ok = match base {
        1 => cmap_type == 1 && matches!(bpp, 8 | 16) && matches!(cmap_bits, 15 | 16 | 24 | 32),
        2 => matches!(bpp, 15 | 16 | 24 | 32),
        _ => matches!(bpp, 8 | 16),
    };
    if !ok {
        return Err(Error::InvalidData);
    }
    let image_id = {
        let id = d.get(18..18 + id_len).ok_or(Error::InvalidData)?;
        let mut b = crate::buffer::zeroed(id.len())?;
        b.copy_from_slice(id);
        b
    };
    let cmap_start = 18 + id_len;
    let entry_bytes = (cmap_bits as usize).div_ceil(8);
    let cmap_size = if cmap_type == 1 {
        cmap_len * entry_bytes
    } else {
        0
    };
    let cmap = d
        .get(cmap_start..cmap_start + cmap_size)
        .ok_or(Error::InvalidData)?;
    let data_start = cmap_start + cmap_size;

    // TGA 2.0 footer and extension area.
    let mut attribute_type = None;
    if d.len() >= 18 + 26 && d.ends_with(FOOTER_SIG) {
        let f = d.len() - 26;
        let ext = u32::from_le_bytes(
            d.get(f..f + 4)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0; 4]),
        ) as usize;
        if ext >= data_start && ext.saturating_add(495) <= f {
            attribute_type = d.get(ext + 494).copied();
        }
    }

    let pixels = w * ht;
    if pixels as u64 > MAX_PIXELS {
        return Err(Error::InvalidData);
    }
    let pb = (bpp as usize).div_ceil(8);
    let body = d.get(data_start..).ok_or(Error::InvalidData)?;
    let mut raw_buf;
    let raw: &[u8] = if rle {
        if pixels as u64 > (body.len() as u64).saturating_mul(128) {
            return Err(Error::InvalidData);
        }
        raw_buf = crate::buffer::zeroed(pixels * pb)?;
        unpack_rle(body, pb, &mut raw_buf)?;
        &raw_buf
    } else {
        body.get(..pixels * pb).ok_or(Error::InvalidData)?
    };

    // Alpha decision (see the module docs).
    let alpha_bits = desc & 15;
    let src_bits = if mapped { cmap_bits } else { bpp };
    let gray = base == 3;
    let can_alpha = if gray {
        bpp == 16
    } else {
        matches!(src_bits, 16 | 32)
    };
    let has_alpha = can_alpha
        && match attribute_type {
            Some(a) => a == 3 || a == 4,
            None if alpha_bits > 0 => true,
            None if gray => raw.as_chunks::<2>().0.iter().any(|p| p[1] != 0),
            None if src_bits == 32 && mapped => cmap.as_chunks::<4>().0.iter().any(|p| p[3] != 0),
            None if src_bits == 32 => raw.as_chunks::<4>().0.iter().any(|p| p[3] != 0),
            None => false,
        };

    let (format, cpp) = match (gray, has_alpha) {
        (true, false) => (Format::R8Unorm, 1),
        (true, true) => (Format::Rg8Unorm, 2),
        (false, false) => (Format::Rgb8Srgb, 3),
        (false, true) => (Format::Rgba8Srgb, 4),
    };
    let mut img = Image::new(format, w as u32, ht as u32)?;
    let (rtl, top) = (desc & 0x10 != 0, desc & 0x20 != 0);
    let entry = |p: &[u8]| -> [u8; 4] {
        if gray {
            return match p {
                [g, a, ..] if bpp == 16 => [*g, *a, 0, 0],
                [g, ..] => [*g, 255, 0, 0],
                _ => [0, 255, 0, 0],
            };
        }
        if mapped {
            let i = match p {
                [a, b, ..] if bpp == 16 => u16::from_le_bytes([*a, *b]) as usize,
                [a, ..] => *a as usize,
                _ => 0,
            };
            return i
                .checked_sub(cmap_first)
                .and_then(|i| cmap.get(i * entry_bytes..(i + 1) * entry_bytes))
                .map_or([0, 0, 0, 255], |e| color(e, cmap_bits));
        }
        color(p, bpp)
    };
    for_each_unit(&mut img.data, w * cpp, pixels, |y, out| {
        let sy = if top { y } else { ht - 1 - y };
        for (x, o) in out.chunks_exact_mut(cpp).enumerate() {
            let sx = if rtl { w - 1 - x } else { x };
            let i = (sy * w + sx) * pb;
            let v = entry(raw.get(i..i + pb).unwrap_or_default());
            o.copy_from_slice(&v[..cpp]);
        }
    });
    Ok(TgaImage {
        image: img,
        image_id,
        attribute_type,
    })
}

/// Expands an RLE packet stream into `out` (`pb` bytes per pixel).
fn unpack_rle(s: &[u8], pb: usize, out: &mut [u8]) -> Result<()> {
    let (mut i, mut o) = (0usize, 0usize);
    while o < out.len() {
        let c = *s.get(i).ok_or(Error::InvalidData)?;
        i += 1;
        let n = ((c & 0x7F) as usize + 1) * pb;
        let n = n.min(out.len() - o);
        let dst = out.get_mut(o..o + n).ok_or(Error::InvalidData)?;
        if c & 0x80 != 0 {
            let p = s.get(i..i + pb).ok_or(Error::InvalidData)?;
            i += pb;
            for d in dst.chunks_mut(pb) {
                d.copy_from_slice(p.get(..d.len()).ok_or(Error::InvalidData)?);
            }
        } else {
            dst.copy_from_slice(s.get(i..i + n).ok_or(Error::InvalidData)?);
            i += n;
        }
        o += n;
    }
    Ok(())
}

fn push(out: &mut Buffer, v: &[u8]) -> Result<()> {
    out.extend_from_slice(v).map_err(|_| Error::OutOfMemory)
}

/// Appends one scanline (`cpp` bytes per pixel) as RLE packets.
fn rle_row(out: &mut Buffer, row: &[u8], cpp: usize) -> Result<()> {
    let px = |i: usize| row.get(i * cpp..(i + 1) * cpp);
    let n = row.len() / cpp;
    let mut i = 0;
    while i < n {
        let mut run = 1;
        while i + run < n && run < 128 && px(i + run) == px(i) {
            run += 1;
        }
        if run >= 2 {
            push(out, &[0x80 | (run - 1) as u8])?;
            push(out, px(i).unwrap_or_default())?;
            i += run;
            continue;
        }
        // Raw packet up to the next run of 2 (the first pixel never starts one).
        let start = i;
        i += 1;
        while i < n && i - start < 128 && (i + 1 >= n || px(i + 1) != px(i)) {
            i += 1;
        }
        push(out, &[(i - start - 1) as u8])?;
        push(
            out,
            row.get(start * cpp..i * cpp).ok_or(Error::InvalidData)?,
        )?;
    }
    Ok(())
}

/// Encodes `src`.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    let (px, cpp) = ldr_pixels(src)?;
    let (w, h) = (src.width() as usize, src.height() as usize);
    if w > 0xFFFF || h > 0xFFFF {
        return Err(Error::Unsupported);
    }
    // Stored byte order: gray, BGR, BGRA.
    let mut pixels = crate::buffer::zeroed(px.len())?;
    for (o, p) in pixels.chunks_exact_mut(cpp).zip(px.chunks_exact(cpp)) {
        match (o, p) {
            ([o0, o1, o2], [r, g, b]) => (*o0, *o1, *o2) = (*b, *g, *r),
            ([o0, o1, o2, o3], [r, g, b, a]) => (*o0, *o1, *o2, *o3) = (*b, *g, *r, *a),
            ([o0], [g]) => *o0 = *g,
            _ => {}
        }
    }
    let img_type = match (cpp, opts.rle) {
        (1, false) => 3u8,
        (1, true) => 11,
        (_, false) => 2,
        (_, true) => 10,
    };
    let mut hd = [0u8; 18];
    hd[2] = img_type;
    hd[12..14].copy_from_slice(&(w as u16).to_le_bytes());
    hd[14..16].copy_from_slice(&(h as u16).to_le_bytes());
    hd[16] = (cpp * 8) as u8;
    hd[17] = 0x20 | if cpp == 4 { 8 } else { 0 };
    let mut out = Buffer::default();
    out.reserve(18 + pixels.len() + 26)
        .map_err(|_| Error::OutOfMemory)?;
    push(&mut out, &hd)?;
    if opts.rle {
        for row in pixels.chunks_exact(w * cpp) {
            rle_row(&mut out, row, cpp)?;
        }
    } else {
        push(&mut out, &pixels)?;
    }
    push(&mut out, &[0; 8])?;
    push(&mut out, FOOTER_SIG)?;
    Ok(out)
}

/// True when `data` has the TGA 2.0 footer or a plausible TGA header (no signature exists).
pub fn looks_like_tga(data: &[u8]) -> bool {
    if data.len() >= 44 && data.ends_with(b"TRUEVISION-XFILE.\0") {
        return true;
    }
    let Some(h) = data.get(..18) else {
        return false;
    };
    let (cmap_type, img_type, bpp) = (h[1], h[2], h[16]);
    let (w, hh) = (
        u16::from_le_bytes([h[12], h[13]]),
        u16::from_le_bytes([h[14], h[15]]),
    );
    cmap_type <= 1
        && matches!(img_type, 1 | 2 | 3 | 9 | 10 | 11)
        && matches!(bpp, 8 | 15 | 16 | 24 | 32)
        && w > 0
        && hh > 0
        && (img_type & 7 != 1 || cmap_type == 1)
}
