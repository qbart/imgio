//! What a file says about its colours, and colour managed decoding.
//!
//! [`colorimetry`] reads the colour description of a file without decoding its pixels:
//!
//! - PNG: cICP, then iCCP, then sRGB, then gAMA (+ cHRM), as the PNG specification orders them;
//! - JPEG: APP2 `ICC_PROFILE` segments (joined in sequence order);
//! - WebP: the `ICCP` chunk; TIFF: tag 34675 (InterColorProfile) of the first image;
//! - AVIF: `colr` properties (`prof` / `rICC` ICC profile first, else `nclx` code points);
//! - OpenEXR: the `chromaticities` attribute (linear light; Rec. 709 when absent);
//! - Radiance HDR: a `PRIMARIES=` header line (linear light);
//! - JPEG XL: [`Colorimetry::Srgb`] (imgio's JPEG XL decoder already renders to sRGB, or
//!   linear sRGB for float images);
//! - everything else (BMP, TGA, GIF, ICO, zfile, OpenVDB, JPEG 2000): unspecified.
//!
//! [`decode_to`] decodes the main image and converts it from that description to a target
//! [`Profile`] with [`crate::color::convert`] (unspecified = sRGB for integer samples, linear
//! sRGB for float samples; gray images use the gray equivalent). Embedded profiles that do not
//! fit the decoded pixels (a CMYK profile of a JPEG the decoder already turned into RGB) are
//! ignored like an unspecified description.

use super::{FileFormat, Image};
use crate::color::{ColorSpace, Intent, Profile};
use crate::{Buffer, Error, Format, Result};

pub use crate::color::Transfer;

/// The colour description of a file.
#[derive(Debug)]
pub enum Colorimetry {
    /// Nothing stated.
    Unspecified,
    /// sRGB stated (or rendered to sRGB by the decoder).
    Srgb,
    /// An embedded ICC profile.
    Icc(Buffer),
    /// ITU-T H.273 code points.
    Cicp {
        primaries: u8,
        transfer: u8,
        matrix: u8,
        full_range: bool,
    },
    /// Chromaticities (red, green, blue, white `x, y`) and a transfer.
    Primaries {
        xy: [[f32; 2]; 4],
        transfer: Transfer,
    },
    /// Only a transfer (PNG gAMA without cHRM: sRGB primaries).
    Transfer(Transfer),
}

fn buf(b: &[u8]) -> Result<Buffer> {
    let mut v = crate::buffer::zeroed(b.len())?;
    v.copy_from_slice(b);
    Ok(v)
}

fn be32(d: &[u8], o: usize) -> Option<u32> {
    let b = d.get(o..o + 4)?;
    Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// The colour description of `data` (see the module documentation). Errors only for
/// malformed colour data that a decoder would also reject; unknown formats are
/// `Unspecified`.
pub fn colorimetry(data: &[u8]) -> Result<Colorimetry> {
    Ok(match FileFormat::detect(data) {
        Some(FileFormat::Png) => png(data)?,
        Some(FileFormat::Jpeg) => jpeg(data)?,
        Some(FileFormat::WebP) => webp(data)?,
        Some(FileFormat::Tiff) => tiff(data)?,
        Some(FileFormat::Avif) => avif(data)?,
        Some(FileFormat::Exr) => exr(data),
        Some(FileFormat::Hdr) => hdr(data),
        Some(FileFormat::JpegXl) => Colorimetry::Srgb,
        _ => Colorimetry::Unspecified,
    })
}

fn png(data: &[u8]) -> Result<Colorimetry> {
    let c = super::png::info(data)?.color;
    if let Some([p, t, m, f]) = c.cicp {
        return Ok(Colorimetry::Cicp {
            primaries: p,
            transfer: t,
            matrix: m,
            full_range: f != 0,
        });
    }
    if let Some(icc) = c.icc_profile {
        return Ok(Colorimetry::Icc(icc));
    }
    if c.srgb_intent.is_some() {
        return Ok(Colorimetry::Srgb);
    }
    let Some(g) = c.gamma.filter(|&g| g > 0) else {
        return Ok(Colorimetry::Unspecified);
    };
    let transfer = if g == 100_000 {
        Transfer::Linear
    } else {
        Transfer::Gamma(100_000.0 / g as f32)
    };
    Ok(match c.chromaticities {
        Some(ch) => {
            let v = |i: usize| ch[i] as f32 / 100_000.0;
            Colorimetry::Primaries {
                xy: [[v(2), v(3)], [v(4), v(5)], [v(6), v(7)], [v(0), v(1)]],
                transfer,
            }
        }
        None => Colorimetry::Transfer(transfer),
    })
}

/// APP2 `ICC_PROFILE\0` segments before the first scan, joined by sequence number.
fn jpeg(d: &[u8]) -> Result<Colorimetry> {
    let mut parts: [Option<&[u8]>; 256] = [None; 256];
    let mut count = 0usize;
    let mut o = 2;
    while o + 4 <= d.len() {
        if d[o] != 0xFF {
            break;
        }
        let m = d[o + 1];
        if m == 0xFF {
            o += 1;
            continue;
        }
        if m == 0xDA || m == 0xD9 {
            break;
        }
        if (0xD0..=0xD7).contains(&m) || m == 0x01 {
            o += 2;
            continue;
        }
        let len = usize::from(u16::from_be_bytes([d[o + 2], d[o + 3]]));
        let Some(seg) = d.get(o + 4..o + 2 + len).filter(|_| len >= 2) else {
            break;
        };
        if m == 0xE2 && seg.len() > 14 && &seg[..12] == b"ICC_PROFILE\0" {
            let (seq, n) = (usize::from(seg[12]), usize::from(seg[13]));
            if seq >= 1 && seq <= n {
                parts[seq - 1] = Some(&seg[14..]);
                count = count.max(n);
            }
        }
        o += 2 + len;
    }
    if count == 0 || parts[..count].iter().any(|p| p.is_none()) {
        return Ok(Colorimetry::Unspecified);
    }
    let mut icc = Buffer::default();
    for p in parts[..count].iter().flatten() {
        icc.extend_from_slice(p).map_err(|_| Error::OutOfMemory)?;
    }
    Ok(Colorimetry::Icc(icc))
}

fn webp(d: &[u8]) -> Result<Colorimetry> {
    let size = d
        .get(4..8)
        .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
    let end = (size + 8).min(d.len());
    let mut o = 12;
    while o + 8 <= end {
        let n = u32::from_le_bytes([d[o + 4], d[o + 5], d[o + 6], d[o + 7]]) as usize;
        let Some(body) = d.get(o + 8..o + 8 + n) else {
            break;
        };
        if &d[o..o + 4] == b"ICCP" {
            return Ok(Colorimetry::Icc(buf(body)?));
        }
        o += 8 + n + (n & 1);
    }
    Ok(Colorimetry::Unspecified)
}

/// Tag 34675 of the first IFD (classic TIFF and BigTIFF, either byte order).
fn tiff(d: &[u8]) -> Result<Colorimetry> {
    let le = d.starts_with(b"II");
    let u16_at = |o: usize| -> Option<u64> {
        let b = d.get(o..o + 2)?;
        Some(u64::from(if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        }))
    };
    let u32_at = |o: usize| -> Option<u64> {
        let b = d.get(o..o + 4)?;
        let a = [b[0], b[1], b[2], b[3]];
        Some(u64::from(if le {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        }))
    };
    let u64_at = |o: usize| -> Option<u64> {
        let b: [u8; 8] = d.get(o..o + 8)?.try_into().ok()?;
        Some(if le {
            u64::from_le_bytes(b)
        } else {
            u64::from_be_bytes(b)
        })
    };
    let find = || -> Option<&[u8]> {
        let big = u16_at(2)? == 43;
        let (ifd, cnt_sz, ent_sz) = if big {
            (u64_at(8)?, 8, 20)
        } else {
            (u32_at(4)?, 2, 12)
        };
        let ifd = usize::try_from(ifd).ok()?;
        let n = if big { u64_at(ifd)? } else { u16_at(ifd)? };
        for i in 0..n.min(4096) as usize {
            let e = ifd + cnt_sz + i * ent_sz;
            if u16_at(e)? != 34675 {
                continue;
            }
            let (count, val) = if big {
                (u64_at(e + 4)?, e + 12)
            } else {
                (u32_at(e + 4)?, e + 8)
            };
            let count = usize::try_from(count).ok()?;
            let inline = if big { 8 } else { 4 };
            let off = if count <= inline {
                val
            } else if big {
                usize::try_from(u64_at(val)?).ok()?
            } else {
                usize::try_from(u32_at(val)?).ok()?
            };
            return d.get(off..off.checked_add(count)?);
        }
        None
    };
    Ok(match find() {
        Some(icc) => Colorimetry::Icc(buf(icc)?),
        None => Colorimetry::Unspecified,
    })
}

/// ISOBMFF boxes of `d`: (type, payload).
fn boxes(d: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8])> {
    let mut o = 0usize;
    core::iter::from_fn(move || {
        let size = be32(d, o)? as usize;
        let ty: [u8; 4] = d.get(o + 4..o + 8)?.try_into().ok()?;
        let (hdr, size) = match size {
            1 => {
                let b: [u8; 8] = d.get(o + 8..o + 16)?.try_into().ok()?;
                (16, usize::try_from(u64::from_be_bytes(b)).ok()?)
            }
            0 => (8, d.len() - o),
            s => (8, s),
        };
        if size < hdr {
            return None;
        }
        let body = d.get(o + hdr..o.checked_add(size)?)?;
        o += size;
        Some((ty, body))
    })
}

/// `colr` properties in `meta/iprp/ipco`: an ICC profile first, else `nclx`.
fn avif(d: &[u8]) -> Result<Colorimetry> {
    let Some((_, meta)) = boxes(d).find(|(t, _)| t == b"meta") else {
        return Ok(Colorimetry::Unspecified);
    };
    let meta = meta.get(4..).unwrap_or(&[]);
    let ipco = boxes(meta)
        .find(|(t, _)| t == b"iprp")
        .and_then(|(_, iprp)| boxes(iprp).find(|(t, _)| t == b"ipco"))
        .map(|(_, b)| b);
    let Some(ipco) = ipco else {
        return Ok(Colorimetry::Unspecified);
    };
    let mut nclx = None;
    for (t, b) in boxes(ipco) {
        if &t != b"colr" || b.len() < 4 {
            continue;
        }
        match &b[..4] {
            b"prof" | b"rICC" => return Ok(Colorimetry::Icc(buf(&b[4..])?)),
            b"nclx" if b.len() >= 11 && nclx.is_none() => {
                let v = |i: usize| u16::from_be_bytes([b[i], b[i + 1]]);
                nclx = Some(Colorimetry::Cicp {
                    primaries: v(4).min(255) as u8,
                    transfer: v(6).min(255) as u8,
                    matrix: v(8).min(255) as u8,
                    full_range: b[10] & 0x80 != 0,
                });
            }
            _ => {}
        }
    }
    Ok(nclx.unwrap_or(Colorimetry::Unspecified))
}

const REC709: [[f32; 2]; 4] = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.3127, 0.3290]];

/// The `chromaticities` attribute of the first header (8 little endian f32).
fn exr(d: &[u8]) -> Colorimetry {
    let mut o = 8;
    let cstr = |o: usize| -> Option<(&[u8], usize)> {
        let rest = d.get(o..)?;
        let n = rest.iter().position(|&c| c == 0)?;
        Some((&rest[..n], o + n + 1))
    };
    let mut xy = REC709;
    while let Some((name, next)) = cstr(o) {
        if name.is_empty() {
            break;
        }
        let Some((ty, next)) = cstr(next) else { break };
        let Some(size) = d
            .get(next..next + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        else {
            break;
        };
        let v = next + 4;
        if name == b"chromaticities"
            && ty == b"chromaticities"
            && size == 32
            && let Some(b) = d.get(v..v + 32)
        {
            let f =
                |i: usize| f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
            xy = [[f(0), f(1)], [f(2), f(3)], [f(4), f(5)], [f(6), f(7)]];
        }
        o = v + size;
    }
    Colorimetry::Primaries {
        xy,
        transfer: Transfer::Linear,
    }
}

/// A `PRIMARIES= rx ry gx gy bx by wx wy` header line, else Rec. 709 (linear).
fn hdr(d: &[u8]) -> Colorimetry {
    let head = &d[..d.len().min(4096)];
    let end = head
        .windows(2)
        .position(|w| w == b"\n\n")
        .unwrap_or(head.len());
    let mut xy = REC709;
    for line in head[..end].split(|&c| c == b'\n') {
        let Some(rest) = line.strip_prefix(b"PRIMARIES=") else {
            continue;
        };
        let Ok(s) = core::str::from_utf8(rest) else {
            continue;
        };
        let mut v = [0.0f32; 8];
        let mut n = 0;
        for (dst, t) in v.iter_mut().zip(s.split_whitespace()) {
            match t.parse() {
                Ok(x) => {
                    *dst = x;
                    n += 1;
                }
                Err(_) => break,
            }
        }
        if n == 8 {
            xy = [[v[0], v[1]], [v[2], v[3]], [v[4], v[5]], [v[6], v[7]]];
        }
    }
    Colorimetry::Primaries {
        xy,
        transfer: Transfer::Linear,
    }
}

/// Channels of a decoded format: (gray?, has alpha).
fn layout(f: Format) -> Option<(bool, bool)> {
    Some(match crate::resize::channels(f)? {
        1 => (true, false),
        2 => (true, true),
        3 => (false, false),
        _ => (false, true),
    })
}

/// The output format of a conversion: same depth, gray <-> RGB as the target needs, 8-bit
/// labeled sRGB for an sRGB target (float gray drops alpha: there is no gray + alpha float
/// format).
fn out_format(f: Format, gray_out: bool, srgb: bool, float: bool) -> Format {
    use Format::*;
    let (_, alpha) = layout(f).unwrap_or((false, true));
    let float = float || f.is_hdr();
    let wide = matches!(f, R16Unorm | Rg16Unorm | Rgba16Unorm);
    match (gray_out, alpha, float, wide) {
        (true, _, true, _) => R32Float,
        (false, _, true, _) => Rgba32Float,
        (true, false, false, true) => R16Unorm,
        (true, true, false, true) => Rg16Unorm,
        (false, _, false, true) => Rgba16Unorm,
        (true, false, false, false) => R8Unorm,
        (true, true, false, false) => Rg8Unorm,
        (false, false, false, false) => {
            if srgb {
                Rgb8Srgb
            } else {
                Rgb8Unorm
            }
        }
        (false, true, false, false) => {
            if srgb {
                Rgba8Srgb
            } else {
                Rgba8Unorm
            }
        }
    }
}

/// The source profile of decoded pixels of `fmt` described by `c`, or `None` for "the
/// default" (sRGB / linear sRGB).
fn source_profile(c: &Colorimetry, gray: bool) -> Result<Option<Profile>> {
    let tr = |t: Transfer| -> Result<Profile> {
        match t {
            Transfer::Linear => Profile::new(ColorSpace::Gray(1.0)),
            Transfer::Srgb => Profile::new(ColorSpace::GraySrgb),
            Transfer::Gamma(g) => Profile::new(ColorSpace::Gray(g)),
        }
    };
    Ok(match c {
        Colorimetry::Unspecified | Colorimetry::Srgb => None,
        Colorimetry::Icc(b) => {
            let p = Profile::from_icc(b).ok();
            let want = if gray {
                crate::color::Channels::Gray
            } else {
                crate::color::Channels::Rgb
            };
            p.filter(|p| p.data_channels() == want)
        }
        Colorimetry::Cicp {
            primaries,
            transfer,
            matrix,
            full_range,
        } => {
            if gray {
                None
            } else {
                Profile::cicp(*primaries, *transfer, *matrix, *full_range).ok()
            }
        }
        Colorimetry::Primaries { xy, transfer } => Some(if gray {
            tr(*transfer)?
        } else {
            Profile::from_primaries(*xy, *transfer)?
        }),
        Colorimetry::Transfer(t) => Some(if gray {
            tr(*t)?
        } else {
            Profile::from_primaries(
                [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.3127, 0.3290]],
                *t,
            )?
        }),
    })
}

/// Decodes the main image of `data` and converts it to `target` (see the module docs). The
/// output keeps the sample depth (8-bit, 16-bit or float; RGB without alpha stays without
/// alpha for 8-bit only) except that linear targets (linear sRGB, ACES, linear gray) and PQ /
/// HLG sources give float samples; it becomes gray for a gray target and RGB for an RGB
/// target.
///
/// Errors: as [`super::decode`]; `Unsupported` without the `color` feature or for a CMYK /
/// Lab target.
pub fn decode_to(data: &[u8], target: &Profile, intent: Intent) -> Result<Image> {
    let img = super::decode(data)?;
    let (gray, _) = layout(img.format).ok_or(Error::Unsupported)?;
    let gray_out = match target.data_channels() {
        crate::color::Channels::Gray => true,
        crate::color::Channels::Rgb => false,
        crate::color::Channels::Other => return Err(Error::Unsupported),
    };
    let c = colorimetry(data)?;
    let float = img.format.is_hdr();
    let src = match source_profile(&c, gray)? {
        Some(p) => p,
        None => {
            let space = match (gray, float) {
                (true, true) => ColorSpace::Gray(1.0),
                (true, false) => ColorSpace::GraySrgb,
                (false, true) => ColorSpace::LinearSrgb,
                (false, false) => ColorSpace::Srgb,
            };
            if target.space() == Some(space) {
                return Ok(img);
            }
            Profile::new(space)?
        }
    };
    let srgb = target.space() == Some(ColorSpace::Srgb);
    // Linear targets and PQ / HLG sources need float samples.
    let linear = matches!(
        target.space(),
        Some(ColorSpace::LinearSrgb | ColorSpace::AcesCg | ColorSpace::Aces2065)
    ) || target.space() == Some(ColorSpace::Gray(1.0));
    let hdr = matches!(
        c,
        Colorimetry::Cicp {
            transfer: 16 | 18,
            ..
        }
    );
    let fmt = out_format(img.format, gray_out, srgb, linear || hdr);
    crate::color::convert(&img.surface()?, &src, target, intent, fmt)
}
