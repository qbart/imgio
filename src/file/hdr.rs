//! Radiance RGBE (own implementation).
//!
//! Decoding: `#?` program line (`#?RADIANCE`, `#?RGBE`, ...), header variables (`FORMAT=` must
//! be `32-bit_rle_rgbe` or `32-bit_rle_xyze`, `EXPOSURE=` values multiply, others are ignored),
//! any of the eight resolution strings (`-Y N +X M` standard, `+Y` / `-X` flips and the X-major
//! transposed forms such as `+X M -Y N`), and per scanline flat, old-style RLE (`1 1 1 n`
//! repeats the previous pixel, consecutive counts shift by 8 bits) or new adaptive RLE (`2 2`
//! marker, four run-length coded channel planes; only for scanlines of 8..=32767 pixels).
//!
//! Output is `Rgba32Float` (alpha 1.0): RGBE has an 8-bit mantissa but a 2^-135 .. 2^127 range,
//! so every value is exact in f32 while half floats (max 65504, 10-bit subnormal floor near
//! 6e-8) would clip or round bright and dark pixels. The conversion is the one of ffmpeg's
//! `hdr` decoder and imagecodecs `rgbe_decode` (Bruce Walter's `rgbe.c`): exponent byte 0 is
//! 0.0, else `m * 2^(e - 136)` (no +0.5 mantissa offset), bit-exact. EXPOSURE is reported in
//! [`HdrImage`], not applied (as in both references). XYZE files decode their raw XYZ values.
//!
//! Limits: at most 2^28 pixels and 256 pixels per input byte (new-style RLE stays below 16;
//! only extreme old-style RLE runs exceed it).
//!
//! Encoding writes `#?RADIANCE`, `FORMAT=32-bit_rle_rgbe`, optional `EXPOSURE`, `-Y H +X W`
//! and new-style RLE scanlines (Walter's run finder, minimum run 4) when the width is 8..=32767
//! and [`EncodeOptions::rle`] is set, flat scanlines otherwise. Float to RGBE matches ffmpeg's
//! `hdr` encoder: `v = max(r, g, b)`; below 1e-32 all bytes 0; else with `v = m * 2^e`
//! (`m` in [0.5, 1)) each channel is `clamp(trunc(c * 2^(8 - e)), 0, 255)` and the exponent
//! byte `clamp(e + 128, 0, 255)`. NaN channels count as 0 and +inf as `f32::MAX` (both
//! undefined in the references). Input: any uncompressed format via [`crate::convert`]
//! (`*Srgb` formats are linearized, gray / gray + alpha formats replicate gray to RGB,
//! alpha is dropped).

use super::Image;
use super::bmp::MAX_PIXELS;
use crate::convert::{ConvertOptions, convert_with};
use crate::parallel::for_each_unit;
use crate::{Buffer, Error, Format, Result, Surface};

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    /// New-style RLE scanlines (when the width allows it); flat otherwise. Default true.
    pub rle: bool,
    /// Written as an `EXPOSURE=` header line when set.
    pub exposure: Option<f32>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            rle: true,
            exposure: None,
        }
    }
}

/// A decoded Radiance file with its header values.
#[derive(Debug)]
pub struct HdrImage {
    /// `Rgba32Float` pixels, top row first.
    pub image: Image,
    /// Product of the `EXPOSURE=` header values (1.0 without any); divide by it to undo the
    /// exposure applied when the file was written.
    pub exposure: f32,
    /// `FORMAT=32-bit_rle_xyze`: the channels are CIE XYZ, not RGB.
    pub xyz: bool,
}

/// Most pixels per input byte (new-style RLE reaches about 16).
const PIXELS_PER_BYTE: u64 = 256;

/// Decodes the main image.
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_hdr(data).map(|h| h.image)
}

/// `m * 2^(e - 136)` as in ffmpeg / Walter `rgbe.c` (exact in f32).
pub fn rgbe_to_f32(rgbe: [u8; 4]) -> [f32; 3] {
    let e = rgbe[3];
    if e == 0 {
        return [0.0; 3];
    }
    let x = i32::from(e) - 136; // -135 ..= 119
    let scale = if x >= -126 {
        f32::from_bits(((x + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << (x + 149))
    };
    [
        f32::from(rgbe[0]) * scale,
        f32::from(rgbe[1]) * scale,
        f32::from(rgbe[2]) * scale,
    ]
}

/// `frexpf`: `v = m * 2^e` with `m` in [0.5, 1) for finite positive `v`.
fn frexp(v: f32) -> (f32, i32) {
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32;
    if exp == 0 {
        // Subnormal: normalize first.
        let (m, e) = frexp(v * f32::from_bits(((64 + 127) as u32) << 23));
        return (m, e - 64);
    }
    let m = f32::from_bits((bits & 0x807F_FFFF) | (126 << 23));
    (m, exp - 126)
}

/// Float RGB to RGBE as in ffmpeg's `hdr` encoder (`float2rgbe`).
pub fn f32_to_rgbe(rgb: [f32; 3]) -> [u8; 4] {
    let c = rgb.map(|c| {
        if c.is_nan() {
            0.0
        } else if c == f32::INFINITY {
            f32::MAX
        } else {
            c
        }
    });
    let v = c[0].max(c[1]).max(c[2]);
    if v < 1e-32 {
        return [0; 4];
    }
    let (m, e) = frexp(v);
    let v = m * 256.0 / v;
    let q = |x: f32| (x * v).clamp(0.0, 255.0) as u8;
    [q(c[0]), q(c[1]), q(c[2]), (e + 128).clamp(0, 255) as u8]
}

/// Next header line (without `\n` / `\r\n`) and the offset after it.
fn line(d: &[u8], pos: usize) -> Result<(&[u8], usize)> {
    let rest = d.get(pos..).ok_or(Error::InvalidData)?;
    let n = rest
        .iter()
        .position(|&b| b == b'\n')
        .ok_or(Error::InvalidData)?;
    let l = rest.get(..n).ok_or(Error::InvalidData)?;
    Ok((l.strip_suffix(b"\r").unwrap_or(l), pos + n + 1))
}

/// One resolution string axis: (is Y, positive direction, size).
fn axis(sign: &str, n: &str) -> Result<(bool, bool, usize)> {
    let (pos, y) = match sign {
        "-Y" => (false, true),
        "+Y" => (true, true),
        "-X" => (false, false),
        "+X" => (true, false),
        _ => return Err(Error::InvalidData),
    };
    let n: usize = n.parse().map_err(|_| Error::InvalidData)?;
    if n == 0 {
        return Err(Error::InvalidData);
    }
    Ok((y, pos, n))
}

/// Decodes a Radiance file with its header values.
pub fn decode_hdr(d: &[u8]) -> Result<HdrImage> {
    if !d.starts_with(b"#?") {
        return Err(Error::InvalidData);
    }
    let (_, mut pos) = line(d, 0)?;
    let mut exposure = 1.0f32;
    let mut xyz = false;
    loop {
        let (l, next) = line(d, pos)?;
        pos = next;
        if l.is_empty() {
            break;
        }
        if let Some(f) = l.strip_prefix(b"FORMAT=") {
            match f.trim_ascii() {
                b"32-bit_rle_rgbe" => xyz = false,
                b"32-bit_rle_xyze" => xyz = true,
                _ => return Err(Error::Unsupported),
            }
        } else if let Some(e) = l.strip_prefix(b"EXPOSURE=") {
            let v: f32 = std::str::from_utf8(e)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .ok_or(Error::InvalidData)?;
            exposure *= v;
        }
    }
    let (res, start) = line(d, pos)?;
    let res = std::str::from_utf8(res).map_err(|_| Error::InvalidData)?;
    let mut t = res.split_ascii_whitespace();
    let mut tok = || t.next().ok_or(Error::InvalidData);
    let major = axis(tok()?, tok()?)?;
    let minor = axis(tok()?, tok()?)?;
    if major.0 == minor.0 {
        return Err(Error::InvalidData);
    }
    let (w, h) = if major.0 {
        (minor.2, major.2)
    } else {
        (major.2, minor.2)
    };
    let pixels = (w as u64).saturating_mul(h as u64);
    let body = d.get(start..).ok_or(Error::InvalidData)?;
    if pixels > MAX_PIXELS || pixels > (body.len() as u64).saturating_mul(PIXELS_PER_BYTE) {
        return Err(Error::InvalidData);
    }
    let (count, len) = (major.2, minor.2);
    // Scanlines in stored order; the buffer grows with the data actually decoded, so a
    // truncated file fails before any image-sized allocation.
    let mut stored = Buffer::default();
    let mut prev: Option<[u8; 4]> = None;
    let mut p = 0usize;
    for _ in 0..count {
        let at = stored.len();
        stored
            .resize(at + len * 4, 0)
            .map_err(|_| Error::OutOfMemory)?;
        let line = stored.get_mut(at..).ok_or(Error::InvalidData)?;
        p = read_scanline(body, p, line, &mut prev)?;
    }
    // Index along an axis of image position `k` (and back: the mapping is an involution).
    // Radiance Y grows upwards: -Y walks rows top to bottom.
    let coord = |(is_y, pos, n): (bool, bool, usize), k: usize| -> usize {
        if pos != is_y { k } else { n - 1 - k }
    };
    let mut image = Image::new(Format::Rgba32Float, w as u32, h as u32)?;
    for_each_unit(&mut image.data, w * 16, w * h, |y, row| {
        for (x, o) in row.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            let (a, b) = if major.0 { (y, x) } else { (x, y) };
            let o_in = (coord(major, a) * len + coord(minor, b)) * 4;
            let s = stored
                .get(o_in..)
                .and_then(|s| s.first_chunk::<4>())
                .copied()
                .unwrap_or_default();
            let [r, g, b] = rgbe_to_f32(s);
            for (k, v) in [r, g, b, 1.0].into_iter().enumerate() {
                o[k * 4..k * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    });
    Ok(HdrImage {
        image,
        exposure,
        xyz,
    })
}

fn byte(s: &[u8], i: usize) -> Result<u8> {
    s.get(i).copied().ok_or(Error::InvalidData)
}

/// Reads one scanline of `out.len() / 4` pixels starting at `p`; returns the next offset.
fn read_scanline(
    s: &[u8],
    mut p: usize,
    out: &mut [u8],
    prev: &mut Option<[u8; 4]>,
) -> Result<usize> {
    let len = out.len() / 4;
    let head = s.get(p..p + 4).ok_or(Error::InvalidData)?;
    if (8..=0x7FFF).contains(&len) && head[0] == 2 && head[1] == 2 && head[2] & 0x80 == 0 {
        if (usize::from(head[2]) << 8 | usize::from(head[3])) != len {
            return Err(Error::InvalidData);
        }
        p += 4;
        for c in 0..4 {
            let mut j = 0;
            while j < len {
                let code = byte(s, p)? as usize;
                p += 1;
                if code > 128 {
                    let n = code - 128;
                    let v = byte(s, p)?;
                    p += 1;
                    if j + n > len {
                        return Err(Error::InvalidData);
                    }
                    for k in j..j + n {
                        if let Some(o) = out.get_mut(k * 4 + c) {
                            *o = v;
                        }
                    }
                    j += n;
                } else {
                    if code == 0 || j + code > len {
                        return Err(Error::InvalidData);
                    }
                    let src = s.get(p..p + code).ok_or(Error::InvalidData)?;
                    for (k, &v) in src.iter().enumerate() {
                        if let Some(o) = out.get_mut((j + k) * 4 + c) {
                            *o = v;
                        }
                    }
                    p += code;
                    j += code;
                }
            }
        }
        if let Some(l) = out.last_chunk::<4>() {
            *prev = Some(*l);
        }
        return Ok(p);
    }
    // Flat or old-style RLE.
    let mut j = 0;
    let mut shift = 0u32;
    while j < len {
        let px: [u8; 4] = s
            .get(p..)
            .and_then(|r| r.first_chunk::<4>())
            .copied()
            .ok_or(Error::InvalidData)?;
        p += 4;
        if px[0] == 1 && px[1] == 1 && px[2] == 1 {
            let last = prev.ok_or(Error::InvalidData)?;
            let n = match (px[3], shift) {
                (0, _) => 0,
                (_, 32..) => len - j,
                (c, s) => (u64::from(c) << s).min((len - j) as u64) as usize,
            };
            for k in j..j + n {
                if let Some(o) = out.get_mut(k * 4..k * 4 + 4) {
                    o.copy_from_slice(&last);
                }
            }
            j += n;
            shift += 8;
        } else {
            if let Some(o) = out.get_mut(j * 4..j * 4 + 4) {
                o.copy_from_slice(&px);
            }
            *prev = Some(px);
            j += 1;
            shift = 0;
        }
    }
    Ok(p)
}

fn push(out: &mut Buffer, v: &[u8]) -> Result<()> {
    out.extend_from_slice(v).map_err(|_| Error::OutOfMemory)
}

/// Appends one channel plane as new-style RLE (Walter `RGBE_WriteBytes_RLE`, minimum run 4).
fn rle_plane(out: &mut Buffer, data: &[u8]) -> Result<()> {
    const MIN_RUN: usize = 4;
    let n = data.len();
    let at = |i: usize| data.get(i).copied().unwrap_or(0);
    let mut cur = 0;
    while cur < n {
        let mut beg = cur;
        let (mut run, mut old_run) = (0usize, 0usize);
        while run < MIN_RUN && beg < n {
            beg += run;
            old_run = run;
            run = 1;
            while beg + run < n && run < 127 && at(beg) == at(beg + run) {
                run += 1;
            }
        }
        if old_run > 1 && old_run == beg - cur {
            push(out, &[128 + old_run as u8, at(cur)])?;
            cur = beg;
        }
        while cur < beg {
            let k = (beg - cur).min(128);
            push(out, &[k as u8])?;
            push(out, data.get(cur..cur + k).ok_or(Error::InvalidData)?)?;
            cur += k;
        }
        if run >= MIN_RUN {
            push(out, &[128 + run as u8, at(beg)])?;
            cur += run;
        }
    }
    Ok(())
}

/// Linear float RGB of `src` (`Rgba32Float` layout).
fn float_pixels(src: &Surface) -> Result<Buffer> {
    let f = src.format();
    let len = Format::Rgba32Float.surface_bytes(src.width(), src.height())?;
    let mut buf = crate::buffer::zeroed(len)?;
    let opts = ConvertOptions {
        srgb_transfer: true,
    };
    convert_with(src, Format::Rgba32Float, &opts, &mut buf)?;
    let gray = matches!(
        f,
        Format::R8Unorm
            | Format::R8Snorm
            | Format::R16Unorm
            | Format::R16Snorm
            | Format::R16Float
            | Format::R32Float
            | Format::Rg8Unorm
            | Format::Rg8Snorm
            | Format::Rg16Unorm
            | Format::Rg16Snorm
    );
    if gray {
        for p in buf.as_chunks_mut::<16>().0 {
            let r = [p[0], p[1], p[2], p[3]];
            p[4..8].copy_from_slice(&r);
            p[8..12].copy_from_slice(&r);
        }
    }
    Ok(buf)
}

/// Encodes `src`.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let floats = float_pixels(src)?;
    let mut rgbe = crate::buffer::zeroed(w * h * 4)?;
    for_each_unit(&mut rgbe, w * 4, w * h, |y, row| {
        let s = floats.get(y * w * 16..(y + 1) * w * 16).unwrap_or_default();
        for (o, p) in row
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(s.as_chunks::<16>().0)
        {
            let f = |k: usize| f32::from_le_bytes([p[k], p[k + 1], p[k + 2], p[k + 3]]);
            o.copy_from_slice(&f32_to_rgbe([f(0), f(4), f(8)]));
        }
    });
    let mut out = Buffer::default();
    push(&mut out, b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n")?;
    if let Some(e) = opts.exposure {
        push(&mut out, format!("EXPOSURE={e}\n").as_bytes())?;
    }
    push(&mut out, format!("\n-Y {h} +X {w}\n").as_bytes())?;
    let rle = opts.rle && (8..=0x7FFF).contains(&w);
    if !rle {
        push(&mut out, &rgbe)?;
        return Ok(out);
    }
    let mut plane = crate::buffer::zeroed(w)?;
    for row in rgbe.chunks_exact(w * 4) {
        push(&mut out, &[2, 2, (w >> 8) as u8, (w & 0xFF) as u8])?;
        for c in 0..4 {
            for (d, p) in plane.iter_mut().zip(row.as_chunks::<4>().0) {
                *d = p[c];
            }
            rle_plane(&mut out, &plane)?;
        }
    }
    Ok(out)
}
