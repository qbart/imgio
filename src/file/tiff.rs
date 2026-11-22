//! TIFF and BigTIFF.
//!
//! Needs the `tiff` cargo feature; without it every function returns `Unsupported`.
//!
//! Decoding is imgio's own code (IFD parsing, strips / tiles, predictors, color conversion) on
//! top of the `weezl` (LZW), `miniz_oxide` (Deflate) and `zune-jpeg` (JPEG) codecs:
//! - Classic and BigTIFF, both byte orders, strips and tiles, planar configuration 1 and 2,
//!   FillOrder 2, multi-page ([`page_count`], [`decode_page`]; [`decode`] = page 0).
//! - Compressions none, LZW (also old-style), Deflate / AdobeDeflate, PackBits, JPEG
//!   (new-style, with or without JPEGTables); predictors 2 (horizontal) and 3 (floating
//!   point) with LZW / Deflate (ignored with other compressions, like libtiff). Other compressions (CCITT, old-style JPEG, ZSTD, WebP, ...) are `Unsupported`.
//! - Unsigned 1/2/4/8/16/32-bit, signed 8/16/32-bit and float 16/32/64-bit samples.
//! - Photometric gray (min-is-black, min-is-white: integer samples inverted, floats as
//!   stored), RGB, palette (16-bit colormaps reduced to 8 bits like libtiff), CMYK (InkSet
//!   CMYK, converted like libtiff `TIFFReadRGBA*`: `R = (max - C) * (max - K) / max`,
//!   truncated; floats `(1 - C) * (1 - K)`), YCbCr (8-bit, any subsampling, converted to RGB
//!   with libtiff's tables from `YCbCrCoefficients` and `ReferenceBlackWhite`; JPEG YCbCr is
//!   converted by the JPEG decoder).
//! - Alpha: the first extra sample when it is associated (premultiplied, returned as stored,
//!   not un-premultiplied: see [`PageInfo::alpha`]) or unassociated alpha; unspecified extra
//!   samples are dropped. `Orientation` is ignored (rows as stored).
//!
//! Output formats: unsigned <= 8 bits (and palette, CMYK, YCbCr, JPEG) -> `R8Unorm` /
//! `Rg8Unorm` / `Rgb8Srgb` / `Rgba8Srgb` (sub-byte samples scaled to 0..255); 16 bits ->
//! `R16Unorm` / `Rg16Unorm` / `Rgba16Unorm`; half -> `R16Float` / `Rgb16Float` /
//! `Rgba16Float` (gray + alpha: RGBA); unsigned 32-bit (normalized to 0..1), signed (divided
//! by the type's max) and 32/64-bit float (64 -> f32, rounded) -> `R32Float` / `Rgba32Float`.
//!
//! Encoding uses the `tiff` crate: `R8Unorm`, `Rg8Unorm`, `Rgb8*`, `Rgba8*`, `R16Unorm`,
//! `Rg16Unorm`, `Rgba16Unorm`, `R16Float`, `Rgb16Float`, `Rgba16Float`, `R32Float`,
//! `Rgba32Float` natively (alpha as unassociated alpha); `Bgra8*` is swizzled, other
//! uncompressed formats are converted to `Rgba32Float`. Compressions none, LZW, Deflate,
//! PackBits, optional predictor (horizontal for integers, floating point for floats),
//! classic or BigTIFF, several pages ([`encode_pages`]).

use super::Image;
use crate::{Buffer, Error, Result, Surface};

/// Encoder compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    None,
    Lzw,
    #[default]
    Deflate,
    PackBits,
}

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    pub compression: Compression,
    /// Predictor 2 (integer samples) / 3 (float samples); LZW and Deflate only.
    pub predictor: bool,
    /// Write BigTIFF (64-bit offsets).
    pub bigtiff: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            compression: Compression::Deflate,
            predictor: true,
            bigtiff: false,
        }
    }
}

/// Alpha of a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alpha {
    None,
    /// Premultiplied (returned as stored).
    Associated,
    Unassociated,
}

/// Description of one page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageInfo {
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u16,
    pub bits_per_sample: u16,
    /// TIFF SampleFormat: 1 unsigned, 2 signed, 3 float.
    pub sample_format: u16,
    /// TIFF PhotometricInterpretation.
    pub photometric: u16,
    /// TIFF Compression.
    pub compression: u16,
    /// TIFF PlanarConfiguration (1 chunky, 2 planar).
    pub planar: u16,
    /// Tile size (`None` for strips).
    pub tile: Option<(u32, u32)>,
    pub alpha: Alpha,
    pub bigtiff: bool,
    pub little_endian: bool,
}

/// Decodes page 0.
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_page(data, 0)
}

/// Number of pages (IFDs in the main chain).
pub fn page_count(data: &[u8]) -> Result<usize> {
    #[cfg(feature = "tiff")]
    {
        imp::page_count(data)
    }
    #[cfg(not(feature = "tiff"))]
    {
        let _ = data;
        Err(Error::Unsupported)
    }
}

/// Describes page `page`.
pub fn page_info(data: &[u8], page: usize) -> Result<PageInfo> {
    #[cfg(feature = "tiff")]
    {
        imp::page_info(data, page)
    }
    #[cfg(not(feature = "tiff"))]
    {
        let _ = (data, page);
        Err(Error::Unsupported)
    }
}

/// Decodes page `page`.
pub fn decode_page(data: &[u8], page: usize) -> Result<Image> {
    #[cfg(feature = "tiff")]
    {
        imp::decode_page(data, page)
    }
    #[cfg(not(feature = "tiff"))]
    {
        let _ = (data, page);
        Err(Error::Unsupported)
    }
}

/// Encodes `src` as a one-page file.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    encode_pages(core::slice::from_ref(src), opts)
}

/// Encodes one page per surface.
pub fn encode_pages(pages: &[Surface], opts: &EncodeOptions) -> Result<Buffer> {
    #[cfg(feature = "tiff")]
    {
        enc::encode_pages(pages, opts)
    }
    #[cfg(not(feature = "tiff"))]
    {
        let _ = (pages, opts);
        Err(Error::Unsupported)
    }
}

#[cfg(feature = "tiff")]
mod imp {
    use super::*;
    use crate::Format;
    use vas::Vector;
    use vas::storage::Dynamic;

    /// Largest decoded page, in bytes.
    const MAX_BYTES: u64 = 1 << 32;
    /// Smallest output a tiny file may still claim.
    const MIN_LIMIT: u64 = 1 << 24;
    /// Largest plausible decompression ratio (LZW tops out near 2700:1).
    const MAX_RATIO: u64 = 4096;
    const MAX_PAGES: usize = 1 << 16;

    // Tags.
    const IMAGE_WIDTH: u16 = 256;
    const IMAGE_LENGTH: u16 = 257;
    const BITS_PER_SAMPLE: u16 = 258;
    const COMPRESSION: u16 = 259;
    const PHOTOMETRIC: u16 = 262;
    const FILL_ORDER: u16 = 266;
    const STRIP_OFFSETS: u16 = 273;
    const SAMPLES_PER_PIXEL: u16 = 277;
    const ROWS_PER_STRIP: u16 = 278;
    const STRIP_BYTE_COUNTS: u16 = 279;
    const PLANAR: u16 = 284;
    const PREDICTOR: u16 = 317;
    const COLOR_MAP: u16 = 320;
    const TILE_WIDTH: u16 = 322;
    const TILE_LENGTH: u16 = 323;
    const TILE_OFFSETS: u16 = 324;
    const TILE_BYTE_COUNTS: u16 = 325;
    const INK_SET: u16 = 332;
    const EXTRA_SAMPLES: u16 = 338;
    const SAMPLE_FORMAT: u16 = 339;
    const JPEG_TABLES: u16 = 347;
    const YCBCR_COEFFICIENTS: u16 = 529;
    const YCBCR_SUBSAMPLING: u16 = 530;
    const REFERENCE_BLACK_WHITE: u16 = 532;

    fn oom<T>(_: T) -> Error {
        Error::OutOfMemory
    }

    /// Byte access with the file's byte order.
    #[derive(Clone, Copy)]
    struct Rd<'a> {
        d: &'a [u8],
        le: bool,
        big: bool,
    }

    impl<'a> Rd<'a> {
        fn bytes<const N: usize>(&self, off: u64) -> Result<[u8; N]> {
            let o = usize::try_from(off).map_err(|_| Error::InvalidData)?;
            let s = self.d.get(o..o.checked_add(N).ok_or(Error::InvalidData)?);
            let mut b: [u8; N] = s
                .and_then(|s| s.try_into().ok())
                .ok_or(Error::InvalidData)?;
            if !self.le {
                b.reverse();
            }
            Ok(b)
        }
        fn u16(&self, off: u64) -> Result<u16> {
            Ok(u16::from_le_bytes(self.bytes(off)?))
        }
        fn u32(&self, off: u64) -> Result<u32> {
            Ok(u32::from_le_bytes(self.bytes(off)?))
        }
        fn u64(&self, off: u64) -> Result<u64> {
            Ok(u64::from_le_bytes(self.bytes(off)?))
        }
        fn slice(&self, off: u64, len: u64) -> Result<&'a [u8]> {
            let o = usize::try_from(off).map_err(|_| Error::InvalidData)?;
            let l = usize::try_from(len).map_err(|_| Error::InvalidData)?;
            self.d
                .get(o..o.checked_add(l).ok_or(Error::InvalidData)?)
                .ok_or(Error::InvalidData)
        }
    }

    #[derive(Clone, Copy)]
    struct Entry {
        tag: u16,
        typ: u16,
        count: u64,
        /// Absolute offset of the value bytes.
        at: u64,
    }

    fn type_size(t: u16) -> Option<u64> {
        Some(match t {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 | 13 => 4,
            5 | 10 | 12 | 16 | 17 | 18 => 8,
            _ => return None,
        })
    }

    fn header(d: &[u8]) -> Result<(Rd<'_>, u64)> {
        let le = match d.get(..2) {
            Some(b"II") => true,
            Some(b"MM") => false,
            _ => return Err(Error::InvalidData),
        };
        let r = Rd { d, le, big: false };
        match r.u16(2)? {
            42 => Ok((r, r.u32(4)? as u64)),
            43 => {
                if r.u16(4)? != 8 || r.u16(6)? != 0 {
                    return Err(Error::InvalidData);
                }
                let r = Rd { big: true, ..r };
                Ok((r, r.u64(8)?))
            }
            _ => Err(Error::InvalidData),
        }
    }

    /// Entries of the IFD at `off` and the next IFD offset.
    fn read_ifd(r: &Rd, off: u64) -> Result<(Vector<Entry, Dynamic>, u64)> {
        let (n, mut p, esize) = if r.big {
            (r.u64(off)?, off + 8, 20)
        } else {
            (r.u16(off)? as u64, off + 2, 12)
        };
        if n.saturating_mul(esize) > r.d.len() as u64 {
            return Err(Error::InvalidData);
        }
        let mut v = Vector::<Entry, Dynamic>::default();
        v.reserve(n as usize).map_err(oom)?;
        for _ in 0..n {
            let tag = r.u16(p)?;
            let typ = r.u16(p + 2)?;
            let (count, inline_len, val) = if r.big {
                (r.u64(p + 4)?, 8, p + 12)
            } else {
                (r.u32(p + 4)? as u64, 4, p + 8)
            };
            p += esize;
            let Some(sz) = type_size(typ) else {
                continue; // unknown type: skip, as libtiff does
            };
            let len = count.checked_mul(sz).ok_or(Error::InvalidData)?;
            let at = if len <= inline_len {
                val
            } else if r.big {
                r.u64(val)?
            } else {
                r.u32(val)? as u64
            };
            if at.checked_add(len).is_none_or(|e| e > r.d.len() as u64) {
                // Out-of-file values only matter for tags we read: drop the entry.
                continue;
            }
            v.push(Entry {
                tag,
                typ,
                count,
                at,
            })
            .map_err(oom)?;
        }
        let next = if r.big { r.u64(p)? } else { r.u32(p)? as u64 };
        Ok((v, next))
    }

    /// Offsets of every IFD in the main chain.
    fn ifds(r: &Rd, first: u64) -> Result<Vector<u64, Dynamic>> {
        let mut v = Vector::<u64, Dynamic>::default();
        let mut off = first;
        while off != 0 {
            if v.contains(&off) {
                break; // cycle
            }
            if v.len() >= MAX_PAGES {
                return Err(Error::InvalidData);
            }
            v.push(off).map_err(oom)?;
            let n = if r.big {
                r.u64(off)?
            } else {
                r.u16(off)? as u64
            };
            let esize = if r.big { 20 } else { 12 };
            let pos = off + if r.big { 8 } else { 2 } + n.saturating_mul(esize);
            off = if r.big {
                r.u64(pos)?
            } else {
                r.u32(pos)? as u64
            };
        }
        if v.is_empty() {
            return Err(Error::InvalidData);
        }
        Ok(v)
    }

    struct Ifd<'a> {
        r: Rd<'a>,
        e: Vector<Entry, Dynamic>,
    }

    impl<'a> Ifd<'a> {
        fn find(&self, tag: u16) -> Option<Entry> {
            self.e.iter().find(|e| e.tag == tag).copied()
        }
        /// Integer values of a tag (BYTE / SHORT / LONG / LONG8 / IFD types).
        fn uints(&self, tag: u16) -> Result<Option<Vector<u64, Dynamic>>> {
            let Some(e) = self.find(tag) else {
                return Ok(None);
            };
            let sz = type_size(e.typ).ok_or(Error::InvalidData)?;
            let mut v = Vector::<u64, Dynamic>::default();
            v.reserve(e.count as usize).map_err(oom)?;
            for i in 0..e.count {
                let p = e.at + i * sz;
                let x = match e.typ {
                    1 | 7 => self.r.slice(p, 1)?.first().copied().unwrap_or(0) as u64,
                    3 => self.r.u16(p)? as u64,
                    4 | 13 => self.r.u32(p)? as u64,
                    16 | 18 => self.r.u64(p)?,
                    _ => return Err(Error::InvalidData),
                };
                v.push(x).map_err(oom)?;
            }
            Ok(Some(v))
        }
        fn uint(&self, tag: u16) -> Result<Option<u64>> {
            Ok(self.uints(tag)?.and_then(|v| v.first().copied()))
        }
        /// Float values (RATIONAL / FLOAT / DOUBLE, also integers).
        fn floats(&self, tag: u16) -> Result<Option<Vector<f32, Dynamic>>> {
            let Some(e) = self.find(tag) else {
                return Ok(None);
            };
            let mut v = Vector::<f32, Dynamic>::default();
            v.reserve(e.count as usize).map_err(oom)?;
            for i in 0..e.count {
                let x = match e.typ {
                    5 => {
                        let (n, d) = (self.r.u32(e.at + i * 8)?, self.r.u32(e.at + i * 8 + 4)?);
                        if d == 0 { 0.0 } else { n as f32 / d as f32 }
                    }
                    10 => {
                        let n = self.r.u32(e.at + i * 8)? as i32;
                        let d = self.r.u32(e.at + i * 8 + 4)? as i32;
                        if d == 0 { 0.0 } else { n as f32 / d as f32 }
                    }
                    11 => f32::from_bits(self.r.u32(e.at + i * 4)?),
                    12 => f64::from_bits(self.r.u64(e.at + i * 8)?) as f32,
                    _ => return Ok(None),
                };
                v.push(x).map_err(oom)?;
            }
            Ok(Some(v))
        }
        fn bytes(&self, tag: u16) -> Result<Option<&'a [u8]>> {
            match self.find(tag) {
                Some(e) if matches!(e.typ, 1 | 2 | 6 | 7) => Ok(Some(self.r.slice(e.at, e.count)?)),
                _ => Ok(None),
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Fmt {
        Uint,
        Int,
        Float,
    }

    /// A parsed page.
    struct Page<'a> {
        r: Rd<'a>,
        w: u32,
        h: u32,
        spp: usize,
        bps: u32,
        fmt: Fmt,
        photometric: u16,
        compression: u16,
        predictor: u16,
        planar: u16,
        fill_order: u16,
        alpha: Alpha,
        /// Index of the alpha sample.
        alpha_at: usize,
        colormap: Option<Vector<u64, Dynamic>>,
        subsampling: (u32, u32),
        coeffs: [f32; 3],
        ref_bw: [f32; 6],
        jpeg_tables: Option<&'a [u8]>,
        offsets: Vector<u64, Dynamic>,
        counts: Vector<u64, Dynamic>,
        /// Chunk size (tile or strip width x rows).
        cw: u32,
        ch: u32,
        tiled: bool,
    }

    fn page_ifd(data: &[u8], page: usize) -> Result<Ifd<'_>> {
        let (r, first) = header(data)?;
        let list = ifds(&r, first)?;
        let off = *list.get(page).ok_or(Error::InvalidArgument)?;
        let (e, _) = read_ifd(&r, off)?;
        Ok(Ifd { r, e })
    }

    pub fn page_count(data: &[u8]) -> Result<usize> {
        let (r, first) = header(data)?;
        Ok(ifds(&r, first)?.len())
    }

    fn parse(ifd: Ifd<'_>) -> Result<Page<'_>> {
        let req = |t| -> Result<u64> { ifd.uint(t)?.ok_or(Error::InvalidData) };
        let w = u32::try_from(req(IMAGE_WIDTH)?).map_err(|_| Error::InvalidData)?;
        let h = u32::try_from(req(IMAGE_LENGTH)?).map_err(|_| Error::InvalidData)?;
        if w == 0 || h == 0 {
            return Err(Error::InvalidData);
        }
        let spp = ifd.uint(SAMPLES_PER_PIXEL)?.unwrap_or(1);
        if spp == 0 || spp > 64 {
            return Err(Error::InvalidData);
        }
        let spp = spp as usize;
        let bps_list = ifd.uints(BITS_PER_SAMPLE)?;
        let bps = bps_list
            .as_ref()
            .and_then(|v| v.first().copied())
            .unwrap_or(1);
        if bps_list
            .as_ref()
            .is_some_and(|v| v.iter().any(|&b| b != bps))
        {
            return Err(Error::Unsupported); // mixed bit depths
        }
        let fmt_list = ifd.uints(SAMPLE_FORMAT)?;
        let f = fmt_list
            .as_ref()
            .and_then(|v| v.first().copied())
            .unwrap_or(1);
        if fmt_list.as_ref().is_some_and(|v| v.iter().any(|&x| x != f)) {
            return Err(Error::Unsupported); // mixed sample formats
        }
        let fmt = match f {
            1 | 4 => Fmt::Uint, // 4 = undefined: treated as unsigned
            2 => Fmt::Int,
            3 => Fmt::Float,
            _ => return Err(Error::Unsupported),
        };
        let bps = bps as u32;
        let ok_bits = match fmt {
            Fmt::Uint => matches!(bps, 1 | 2 | 4 | 8 | 16 | 32),
            Fmt::Int => matches!(bps, 8 | 16 | 32),
            Fmt::Float => matches!(bps, 16 | 32 | 64),
        };
        if !ok_bits {
            return Err(Error::Unsupported);
        }
        let compression = ifd.uint(COMPRESSION)?.unwrap_or(1) as u16;
        let photometric = match ifd.uint(PHOTOMETRIC)? {
            Some(p) => p as u16,
            None if spp >= 3 => 2,
            None => 1,
        };
        let extra = ifd.uints(EXTRA_SAMPLES)?;
        let extra_n = extra.as_ref().map_or(0, |v| v.len());
        let color_n = match photometric {
            0 | 1 | 3 => 1,
            2 | 6 => 3,
            5 => 4,
            _ => return Err(Error::Unsupported),
        };
        if spp < color_n {
            return Err(Error::InvalidData);
        }
        let _ = extra_n;
        let alpha = match extra.as_ref().and_then(|v| v.first().copied()) {
            Some(1) if spp > color_n => Alpha::Associated,
            Some(2) if spp > color_n => Alpha::Unassociated,
            _ => Alpha::None,
        };
        let planar = ifd.uint(PLANAR)?.unwrap_or(1) as u16;
        if planar != 1 && planar != 2 {
            return Err(Error::InvalidData);
        }
        // Like libtiff, only LZW and Deflate apply the predictor.
        let predictor = match compression {
            5 | 8 | 32946 => ifd.uint(PREDICTOR)?.unwrap_or(1) as u16,
            _ => 1,
        };
        let fill_order = ifd.uint(FILL_ORDER)?.unwrap_or(1) as u16;
        if photometric == 5 && ifd.uint(INK_SET)?.unwrap_or(1) != 1 {
            return Err(Error::Unsupported);
        }
        let colormap = if photometric == 3 {
            let m = ifd.uints(COLOR_MAP)?.ok_or(Error::InvalidData)?;
            if fmt != Fmt::Uint || bps > 16 || m.len() as u64 != 3u64 << bps {
                return Err(Error::InvalidData);
            }
            Some(m)
        } else {
            None
        };
        let subsampling = match ifd.uints(YCBCR_SUBSAMPLING)? {
            Some(v) if v.len() >= 2 => (v[0] as u32, v[1] as u32),
            _ => (2, 2),
        };
        let mut coeffs = [0.299f32, 0.587, 0.114];
        if let Some(c) = ifd.floats(YCBCR_COEFFICIENTS)?
            && c.len() >= 3
        {
            coeffs.copy_from_slice(&c[..3]);
        }
        let full = ((1u64 << bps.min(32)) - 1) as f32;
        let half = (1u64 << (bps.min(32) - 1)) as f32;
        let mut ref_bw = [0.0, full, half, full, half, full];
        if let Some(c) = ifd.floats(REFERENCE_BLACK_WHITE)?
            && c.len() >= 6
        {
            ref_bw.copy_from_slice(&c[..6]);
        }
        let tw = ifd.uint(TILE_WIDTH)?;
        let (offsets, counts, cw, ch, tiled) = if let Some(tw) = tw {
            let th = req(TILE_LENGTH)?;
            if tw == 0 || th == 0 || tw > 1 << 16 || th > 1 << 16 {
                return Err(Error::InvalidData);
            }
            (
                ifd.uints(TILE_OFFSETS)?.ok_or(Error::InvalidData)?,
                ifd.uints(TILE_BYTE_COUNTS)?.ok_or(Error::InvalidData)?,
                tw as u32,
                th as u32,
                true,
            )
        } else {
            let rps = ifd.uint(ROWS_PER_STRIP)?.unwrap_or(h as u64).min(h as u64);
            if rps == 0 {
                return Err(Error::InvalidData);
            }
            (
                ifd.uints(STRIP_OFFSETS)?.ok_or(Error::InvalidData)?,
                ifd.uints(STRIP_BYTE_COUNTS)?.ok_or(Error::InvalidData)?,
                w,
                rps as u32,
                false,
            )
        };
        let jpeg_tables = ifd.bytes(JPEG_TABLES)?;
        Ok(Page {
            r: ifd.r,
            w,
            h,
            spp,
            bps,
            fmt,
            photometric,
            compression,
            predictor,
            planar,
            fill_order,
            alpha,
            alpha_at: color_n,
            colormap,
            subsampling,
            coeffs,
            ref_bw,
            jpeg_tables,
            offsets,
            counts,
            cw,
            ch,
            tiled,
        })
    }

    pub fn page_info(data: &[u8], page: usize) -> Result<PageInfo> {
        let p = parse(page_ifd(data, page)?)?;
        Ok(PageInfo {
            width: p.w,
            height: p.h,
            samples_per_pixel: p.spp as u16,
            bits_per_sample: p.bps as u16,
            sample_format: match p.fmt {
                Fmt::Uint => 1,
                Fmt::Int => 2,
                Fmt::Float => 3,
            },
            photometric: p.photometric,
            compression: p.compression,
            planar: p.planar,
            tile: p.tiled.then_some((p.cw, p.ch)),
            alpha: p.alpha,
            bigtiff: p.r.big,
            little_endian: p.r.le,
        })
    }

    /// Bytes per decoded sample in the intermediate buffer.
    fn sample_bytes(bps: u32) -> usize {
        match bps {
            0..=8 => 1,
            16 => 2,
            32 => 4,
            _ => 8,
        }
    }

    /// Intermediate image: `spp` little-endian samples per pixel, `sb` bytes each.
    struct Raw {
        data: Buffer,
        /// Image width in pixels.
        w: usize,
        spp: usize,
        sb: usize,
        /// Bits per sample of the values (<= 8 for 1-byte samples).
        bps: u32,
        fmt: Fmt,
        /// Photometric after chunk decoding (YCbCr / JPEG may already be RGB).
        photometric: u16,
    }

    pub fn decode_page(data: &[u8], page: usize) -> Result<Image> {
        let p = parse(page_ifd(data, page)?)?;
        let raw = decode_raw(&p, data.len())?;
        to_image(&p, raw)
    }

    // ----- chunk decoding -----

    fn decode_raw(p: &Page, file_len: usize) -> Result<Raw> {
        let jpeg = p.compression == 7;
        if !matches!(p.compression, 1 | 5 | 7 | 8 | 32946 | 32773) {
            return Err(Error::Unsupported);
        }
        let ycc_sub = p.photometric == 6 && !jpeg && p.subsampling != (1, 1);
        if p.photometric == 6 && !jpeg {
            if p.bps != 8 || p.spp < 3 || p.fmt != Fmt::Uint || (ycc_sub && p.planar == 2) {
                return Err(Error::Unsupported);
            }
            if ycc_sub && !matches!(p.subsampling, (1 | 2 | 4, 1 | 2 | 4)) {
                return Err(Error::InvalidData);
            }
        }
        if jpeg && (p.bps != 8 || p.fmt != Fmt::Uint || p.planar != 1) {
            return Err(Error::Unsupported);
        }
        let (w, h) = (p.w as usize, p.h as usize);
        let planes = if p.planar == 2 { p.spp } else { 1 };
        let spc = if p.planar == 2 { 1 } else { p.spp }; // samples per chunk pixel
        let across = w.div_ceil(p.cw as usize);
        let down = h.div_ceil(p.ch as usize);
        let per_plane = across.checked_mul(down).ok_or(Error::InvalidData)?;
        let total = per_plane.checked_mul(planes).ok_or(Error::InvalidData)?;
        if p.offsets.len() < total || p.counts.len() < total {
            return Err(Error::InvalidData);
        }
        // Intermediate layout: JPEG / YCbCr give 8-bit pixels.
        let (out_spp, out_bps, out_pm) = if p.photometric == 6 && !jpeg && p.planar == 1 {
            (p.spp, 8, 2u16) // converted to RGB per chunk
        } else if jpeg {
            let pm = if p.photometric == 6 { 2 } else { p.photometric };
            (p.spp, 8, pm)
        } else {
            (p.spp, p.bps, p.photometric)
        };
        let sb = sample_bytes(out_bps);
        let bytes = (w as u64)
            .saturating_mul(h as u64)
            .saturating_mul(out_spp as u64)
            .saturating_mul(sb as u64);
        let limit = (file_len as u64)
            .saturating_mul(MAX_RATIO)
            .clamp(MIN_LIMIT, MAX_BYTES);
        if bytes > limit {
            return Err(Error::InvalidData);
        }
        let mut raw = Raw {
            data: crate::buffer::zeroed(bytes as usize)?,
            w,
            spp: out_spp,
            sb,
            bps: out_bps,
            fmt: if jpeg { Fmt::Uint } else { p.fmt },
            photometric: out_pm,
        };
        let mut chunk = Buffer::default();
        let mut tmp = Buffer::default();
        for plane in 0..planes {
            for ci in 0..per_plane {
                let idx = plane * per_plane + ci;
                let (cx, cy) = ((ci % across) * p.cw as usize, (ci / across) * p.ch as usize);
                // Rows / columns of this chunk inside the image.
                let vis_w = (p.cw as usize).min(w - cx);
                let vis_h = (p.ch as usize).min(h - cy);
                // Rows stored in the chunk: tiles are full size, strips end at the image.
                let rows = if p.tiled { p.ch as usize } else { vis_h };
                let off = *p.offsets.get(idx).ok_or(Error::InvalidData)?;
                let cnt = *p.counts.get(idx).ok_or(Error::InvalidData)?;
                let src = p.r.slice(off, cnt)?;
                if jpeg {
                    decode_jpeg(p, src, rows, &mut tmp)?;
                    place_bytes(
                        &mut raw,
                        &tmp,
                        p.cw as usize * p.spp,
                        cx,
                        cy,
                        vis_w,
                        vis_h,
                        0,
                        p.spp,
                    )?;
                    continue;
                }
                let src = if p.fill_order == 2 {
                    tmp.clear();
                    tmp.reserve(src.len()).map_err(oom)?;
                    for &b in src {
                        tmp.push(b.reverse_bits()).map_err(oom)?;
                    }
                    &tmp[..]
                } else {
                    src
                };
                if ycc_sub {
                    let (sh, sv) = (p.subsampling.0 as usize, p.subsampling.1 as usize);
                    let units = (p.cw as usize).div_ceil(sh) * rows.div_ceil(sv);
                    let unit = sh * sv + 2 + (p.spp - 3);
                    let len = units.checked_mul(unit).ok_or(Error::InvalidData)?;
                    resize(&mut chunk, len)?;
                    decompress(p.compression, src, &mut chunk)?;
                    ycbcr_sub_place(p, &chunk, &mut raw, cx, cy, vis_w, vis_h)?;
                    continue;
                }
                let row_bytes = (p.cw as usize * spc * p.bps as usize).div_ceil(8);
                let len = row_bytes.checked_mul(rows).ok_or(Error::InvalidData)?;
                resize(&mut chunk, len)?;
                decompress(p.compression, src, &mut chunk)?;
                unpredict(p, &mut chunk, row_bytes, spc)?;
                if p.photometric == 6 {
                    // Full resolution YCbCr (8-bit): convert in place to RGB.
                    ycbcr_full(p, &mut chunk, spc)?;
                }
                if p.bps < 8 {
                    place_bits(
                        &mut raw, &chunk, row_bytes, p.bps, cx, cy, vis_w, vis_h, plane, spc,
                    )?;
                } else {
                    place_bytes(
                        &mut raw, &chunk, row_bytes, cx, cy, vis_w, vis_h, plane, spc,
                    )?;
                }
            }
        }
        Ok(raw)
    }

    fn resize(b: &mut Buffer, len: usize) -> Result<()> {
        b.clear();
        b.resize(len, 0).map_err(oom)
    }

    /// Decompresses `src` into all of `out`.
    fn decompress(compression: u16, src: &[u8], out: &mut [u8]) -> Result<()> {
        match compression {
            1 => {
                let s = src.get(..out.len()).ok_or(Error::InvalidData)?;
                out.copy_from_slice(s);
                Ok(())
            }
            32773 => packbits(src, out),
            5 => lzw(src, out),
            8 | 32946 => inflate(src, out),
            _ => Err(Error::Unsupported),
        }
    }

    fn packbits(src: &[u8], out: &mut [u8]) -> Result<()> {
        let (mut i, mut o) = (0usize, 0usize);
        while o < out.len() {
            let n = *src.get(i).ok_or(Error::InvalidData)? as i8;
            i += 1;
            if n >= 0 {
                let len = n as usize + 1;
                let s = src.get(i..i + len).ok_or(Error::InvalidData)?;
                let d = out.get_mut(o..o + len).ok_or(Error::InvalidData)?;
                d.copy_from_slice(s);
                i += len;
                o += len;
            } else if n != -128 {
                let len = (1 - n as isize) as usize;
                let v = *src.get(i).ok_or(Error::InvalidData)?;
                i += 1;
                out.get_mut(o..o + len).ok_or(Error::InvalidData)?.fill(v);
                o += len;
            }
        }
        Ok(())
    }

    fn lzw(src: &[u8], out: &mut [u8]) -> Result<()> {
        use weezl::{BitOrder, LzwStatus, decode::Decoder};
        // Old-style (pre-6.0 libtiff) LZW: LSB-first codes, no early change.
        let old = src.first() == Some(&0) && src.get(1).is_some_and(|b| b & 1 != 0);
        let mut dec = if old {
            Decoder::new(BitOrder::Lsb, 8)
        } else {
            Decoder::with_tiff_size_switch(BitOrder::Msb, 8)
        };
        let (mut i, mut o) = (0usize, 0usize);
        while o < out.len() {
            let r = dec.decode_bytes(
                src.get(i..).unwrap_or(&[]),
                out.get_mut(o..).unwrap_or(&mut []),
            );
            i += r.consumed_in;
            o += r.consumed_out;
            match r.status {
                Ok(LzwStatus::Ok) if r.consumed_in == 0 && r.consumed_out == 0 => {
                    return Err(Error::InvalidData);
                }
                Ok(LzwStatus::Ok) => {}
                Ok(LzwStatus::Done | LzwStatus::NoProgress) | Err(_) => break,
            }
        }
        if o < out.len() {
            return Err(Error::InvalidData);
        }
        Ok(())
    }

    fn inflate(src: &[u8], out: &mut [u8]) -> Result<()> {
        use miniz_oxide::inflate::TINFLStatus;
        use miniz_oxide::inflate::core::{DecompressorOxide, decompress, inflate_flags::*};
        let mut d = DecompressorOxide::new();
        let flags = TINFL_FLAG_PARSE_ZLIB_HEADER
            | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF
            | TINFL_FLAG_IGNORE_ADLER32;
        let (status, _, n) = decompress(&mut d, src, out, 0, flags);
        match status {
            TINFLStatus::Done | TINFLStatus::HasMoreOutput if n == out.len() => Ok(()),
            _ => Err(Error::InvalidData),
        }
    }

    /// Undoes the predictor on decompressed rows and makes samples little-endian.
    fn unpredict(p: &Page, chunk: &mut [u8], row_bytes: usize, spc: usize) -> Result<()> {
        let sb = (p.bps as usize).div_ceil(8);
        match p.predictor {
            1 | 2 => {
                if !p.r.le && sb > 1 {
                    for s in chunk.chunks_exact_mut(sb) {
                        s.reverse();
                    }
                }
                if p.predictor == 2 {
                    if p.bps < 8 {
                        return Err(Error::Unsupported);
                    }
                    for row in chunk.chunks_exact_mut(row_bytes) {
                        hor_acc(row, sb, spc);
                    }
                }
                Ok(())
            }
            3 => {
                if p.fmt != Fmt::Float {
                    return Err(Error::InvalidData);
                }
                let wc = row_bytes / sb;
                let mut tmp = Buffer::default();
                resize(&mut tmp, row_bytes)?;
                for row in chunk.chunks_exact_mut(row_bytes) {
                    for i in spc..row.len() {
                        row[i] = row[i].wrapping_add(row[i - spc]);
                    }
                    tmp.copy_from_slice(row);
                    for i in 0..wc {
                        for k in 0..sb {
                            // Byte planes are most significant first.
                            row[i * sb + k] = tmp[(sb - 1 - k) * wc + i];
                        }
                    }
                }
                Ok(())
            }
            _ => Err(Error::Unsupported),
        }
    }

    /// Horizontal differencing decode on little-endian samples of `sb` bytes.
    fn hor_acc(row: &mut [u8], sb: usize, spc: usize) {
        let stride = sb * spc;
        let mut i = stride;
        while i + sb <= row.len() {
            match sb {
                1 => row[i] = row[i].wrapping_add(row[i - stride]),
                2 => {
                    let a = u16::from_le_bytes([row[i], row[i + 1]]);
                    let b = u16::from_le_bytes([row[i - stride], row[i - stride + 1]]);
                    row[i..i + 2].copy_from_slice(&a.wrapping_add(b).to_le_bytes());
                }
                4 => {
                    let a = u32::from_le_bytes([row[i], row[i + 1], row[i + 2], row[i + 3]]);
                    let j = i - stride;
                    let b = u32::from_le_bytes([row[j], row[j + 1], row[j + 2], row[j + 3]]);
                    row[i..i + 4].copy_from_slice(&a.wrapping_add(b).to_le_bytes());
                }
                _ => {
                    let mut a = [0u8; 8];
                    let mut b = [0u8; 8];
                    a.copy_from_slice(&row[i..i + 8]);
                    b.copy_from_slice(&row[i - stride..i - stride + 8]);
                    let s = u64::from_le_bytes(a).wrapping_add(u64::from_le_bytes(b));
                    row[i..i + 8].copy_from_slice(&s.to_le_bytes());
                }
            }
            i += sb;
        }
    }

    /// Copies the visible part of a chunk of byte-aligned samples into the raw image.
    #[allow(clippy::too_many_arguments)]
    fn place_bytes(
        raw: &mut Raw,
        chunk: &[u8],
        row_bytes: usize,
        cx: usize,
        cy: usize,
        vis_w: usize,
        vis_h: usize,
        plane: usize,
        spc: usize,
    ) -> Result<()> {
        let sb = raw.sb;
        let w = raw.w;
        let px = raw.spp * sb;
        for y in 0..vis_h {
            let s = chunk.get(y * row_bytes..).ok_or(Error::InvalidData)?;
            let d0 = ((cy + y) * w + cx) * px;
            if spc == raw.spp {
                let n = vis_w * px;
                let src = s.get(..n).ok_or(Error::InvalidData)?;
                raw.data
                    .get_mut(d0..d0 + n)
                    .ok_or(Error::InvalidData)?
                    .copy_from_slice(src);
            } else {
                for x in 0..vis_w {
                    let src = s.get(x * sb..x * sb + sb).ok_or(Error::InvalidData)?;
                    let d = d0 + x * px + plane * sb;
                    raw.data
                        .get_mut(d..d + sb)
                        .ok_or(Error::InvalidData)?
                        .copy_from_slice(src);
                }
            }
        }
        Ok(())
    }

    /// Unpacks sub-byte samples (MSB first) of the visible part of a chunk.
    #[allow(clippy::too_many_arguments)]
    fn place_bits(
        raw: &mut Raw,
        chunk: &[u8],
        row_bytes: usize,
        bps: u32,
        cx: usize,
        cy: usize,
        vis_w: usize,
        vis_h: usize,
        plane: usize,
        spc: usize,
    ) -> Result<()> {
        let w = raw.w;
        let mask = (1u16 << bps) - 1;
        for y in 0..vis_h {
            let s = chunk
                .get(y * row_bytes..(y + 1) * row_bytes)
                .ok_or(Error::InvalidData)?;
            for x in 0..vis_w {
                for c in 0..spc {
                    let bit = (x * spc + c) * bps as usize;
                    let byte = *s.get(bit / 8).ok_or(Error::InvalidData)?;
                    let v = (byte as u16 >> (8 - bps as usize - bit % 8)) & mask;
                    let d = ((cy + y) * w + cx + x) * raw.spp + plane + c;
                    *raw.data.get_mut(d).ok_or(Error::InvalidData)? = v as u8;
                }
            }
        }
        Ok(())
    }

    /// libtiff `TIFFYCbCrToRGB` tables (8-bit).
    struct Ycc {
        cr_r: [i32; 256],
        cb_b: [i32; 256],
        cr_g: [i32; 256],
        cb_g: [i32; 256],
        y: [i32; 256],
    }

    impl Ycc {
        fn new(luma: [f32; 3], rbw: [f32; 6]) -> Ycc {
            const SHIFT: i32 = 16;
            let fix = |x: f32| ((x * (1i64 << SHIFT) as f32) as f64 + 0.5) as i32;
            let one_half = 1i32 << (SHIFT - 1);
            let clamp = |f: f32, lo: f32, hi: f32| {
                if f < lo {
                    lo
                } else if f > hi {
                    hi
                } else {
                    f
                }
            };
            // NaN maps to `lo` (libtiff behaviour), hence the negated comparison.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            let clampw = |f: f32, lo: f32, hi: f32| {
                if !(f >= lo) {
                    lo
                } else if f > hi {
                    hi
                } else {
                    f
                }
            };
            let code2v = |c: i32, rb: f32, rw: f32, cr: f32| {
                let d = rw - rb;
                ((c - rb as i32) as f32 * cr) / if d != 0.0 { d } else { 1.0 }
            };
            let f1 = 2.0 - 2.0 * luma[0];
            let d1 = fix(clamp(f1, 0.0, 2.0));
            let f2 = luma[0] * f1 / luma[1];
            let d2 = -fix(clamp(f2, 0.0, 2.0));
            let f3 = 2.0 - 2.0 * luma[2];
            let d3 = fix(clamp(f3, 0.0, 2.0));
            let f4 = luma[2] * f3 / luma[1];
            let d4 = -fix(clamp(f4, 0.0, 2.0));
            let mut t = Ycc {
                cr_r: [0; 256],
                cb_b: [0; 256],
                cr_g: [0; 256],
                cb_g: [0; 256],
                y: [0; 256],
            };
            for i in 0..256usize {
                let x = i as i32 - 128;
                let cr = clampw(
                    code2v(x, rbw[4] - 128.0, rbw[5] - 128.0, 127.0),
                    -128.0 * 32.0,
                    128.0 * 32.0,
                ) as i32;
                let cb = clampw(
                    code2v(x, rbw[2] - 128.0, rbw[3] - 128.0, 127.0),
                    -128.0 * 32.0,
                    128.0 * 32.0,
                ) as i32;
                t.cr_r[i] = d1.wrapping_mul(cr).wrapping_add(one_half) >> SHIFT;
                t.cb_b[i] = d3.wrapping_mul(cb).wrapping_add(one_half) >> SHIFT;
                t.cr_g[i] = d2.wrapping_mul(cr);
                t.cb_g[i] = d4.wrapping_mul(cb).wrapping_add(one_half);
                t.y[i] = clampw(
                    code2v(x + 128, rbw[0], rbw[1], 255.0),
                    -128.0 * 32.0,
                    128.0 * 32.0,
                ) as i32;
            }
            t
        }

        fn rgb(&self, y: u8, cb: u8, cr: u8) -> [u8; 3] {
            let (y, cb, cr) = (y as usize, cb as usize, cr as usize);
            let yv = self.y[y];
            let r = yv + self.cr_r[cr];
            let g = yv + (self.cb_g[cb].wrapping_add(self.cr_g[cr]) >> 16);
            let b = yv + self.cb_b[cb];
            [
                r.clamp(0, 255) as u8,
                g.clamp(0, 255) as u8,
                b.clamp(0, 255) as u8,
            ]
        }
    }

    /// Full resolution chunky / planar YCbCr: only chunky is converted here (planar is
    /// converted after assembly).
    fn ycbcr_full(p: &Page, chunk: &mut [u8], spc: usize) -> Result<()> {
        if spc < 3 {
            return Ok(()); // planar: converted in `to_image`
        }
        let t = Ycc::new(p.coeffs, p.ref_bw);
        for px in chunk.chunks_exact_mut(spc) {
            let rgb = t.rgb(px[0], px[1], px[2]);
            px[..3].copy_from_slice(&rgb);
        }
        Ok(())
    }

    /// Subsampled chunky YCbCr: data units of `sh * sv` Y, Cb, Cr (+ extra samples) ->
    /// RGB(+extra) pixels, chroma replicated over the unit (libtiff `putcontig8bitYCbCr*`).
    fn ycbcr_sub_place(
        p: &Page,
        chunk: &[u8],
        raw: &mut Raw,
        cx: usize,
        cy: usize,
        vis_w: usize,
        vis_h: usize,
    ) -> Result<()> {
        let t = Ycc::new(p.coeffs, p.ref_bw);
        let (sh, sv) = (p.subsampling.0 as usize, p.subsampling.1 as usize);
        let extra = p.spp - 3;
        let unit = sh * sv + 2 + extra;
        let units_x = (p.cw as usize).div_ceil(sh);
        let w = p.w as usize;
        for uy in 0..vis_h.div_ceil(sv) {
            for ux in 0..vis_w.div_ceil(sh) {
                let u = chunk
                    .get((uy * units_x + ux) * unit..(uy * units_x + ux + 1) * unit)
                    .ok_or(Error::InvalidData)?;
                let (cb, cr) = (u[sh * sv], u[sh * sv + 1]);
                for dy in 0..sv {
                    for dx in 0..sh {
                        let (x, y) = (ux * sh + dx, uy * sv + dy);
                        if x >= vis_w || y >= vis_h {
                            continue;
                        }
                        let rgb = t.rgb(u[dy * sh + dx], cb, cr);
                        let d = ((cy + y) * w + cx + x) * raw.spp;
                        let px = raw.data.get_mut(d..d + raw.spp).ok_or(Error::InvalidData)?;
                        px[..3].copy_from_slice(&rgb);
                        for e in 0..extra {
                            px[3 + e] = u[sh * sv + 2 + e];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Decodes a JPEG chunk (tables + abbreviated stream) to `cw * rows` pixels of `spp` bytes.
    fn decode_jpeg(p: &Page, src: &[u8], rows: usize, out: &mut Buffer) -> Result<()> {
        use zune_jpeg::JpegDecoder;
        use zune_jpeg::zune_core::bytestream::ZCursor;
        use zune_jpeg::zune_core::colorspace::ColorSpace;
        use zune_jpeg::zune_core::options::DecoderOptions;
        let mut stream = Buffer::default();
        match p.jpeg_tables {
            Some(t) if t.len() >= 4 && src.starts_with(&[0xFF, 0xD8]) => {
                let t = t.strip_suffix(&[0xFF, 0xD9]).unwrap_or(t);
                stream.extend_from_slice(t).map_err(oom)?;
                stream
                    .extend_from_slice(src.get(2..).unwrap_or(&[]))
                    .map_err(oom)?;
            }
            _ => stream.extend_from_slice(src).map_err(oom)?,
        }
        let cs = match (p.photometric, p.spp) {
            (1 | 0, 1) => ColorSpace::Luma,
            (2 | 6, 3) => ColorSpace::RGB,
            _ => return Err(Error::Unsupported),
        };
        let opts = DecoderOptions::default()
            .set_max_width(p.cw as usize)
            .set_max_height(rows.max(p.ch as usize))
            .jpeg_set_out_colorspace(cs);
        let mut dec = JpegDecoder::new_with_options(ZCursor::new(&stream[..]), opts);
        dec.decode_headers().map_err(|_| Error::InvalidData)?;
        let info = dec.info().ok_or(Error::InvalidData)?;
        let (jw, jh) = (info.width as usize, info.height as usize);
        let comps = cs.num_components();
        let size = dec.output_buffer_size().ok_or(Error::InvalidData)?;
        let mut pix = Buffer::default();
        resize(&mut pix, size)?;
        dec.decode_into(&mut pix).map_err(|_| Error::InvalidData)?;
        // Into a cw x rows chunk (JPEG strips may be shorter / tiles padded).
        resize(out, p.cw as usize * rows * comps)?;
        let n = jw.min(p.cw as usize) * comps;
        for y in 0..jh.min(rows) {
            let s = pix
                .get(y * jw * comps..y * jw * comps + n)
                .ok_or(Error::InvalidData)?;
            let d0 = y * p.cw as usize * comps;
            out.get_mut(d0..d0 + n)
                .ok_or(Error::InvalidData)?
                .copy_from_slice(s);
        }
        if jh < rows.min(p.ch as usize) && !p.tiled {
            return Err(Error::InvalidData);
        }
        Ok(())
    }

    // ----- color conversion -----

    /// Sample `i` of the raw image as a float (integers normalized to 0..1 / -1..1).
    fn norm(raw: &Raw, i: usize) -> f32 {
        let sb = raw.sb;
        let Some(b) = raw.data.get(i * sb..i * sb + sb) else {
            return 0.0;
        };
        match (raw.fmt, sb) {
            (Fmt::Uint, 1) => b[0] as f32 / ((1u32 << raw.bps) - 1) as f32,
            (Fmt::Uint, 2) => u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0,
            (Fmt::Uint, 4) => {
                (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64 / 4_294_967_295.0) as f32
            }
            (Fmt::Int, 1) => b[0] as i8 as f32 / 127.0,
            (Fmt::Int, 2) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32767.0,
            (Fmt::Int, 4) => {
                (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64 / 2_147_483_647.0) as f32
            }
            (Fmt::Float, 2) => crate::half::to_f32(u16::from_le_bytes([b[0], b[1]])),
            (Fmt::Float, 4) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            (Fmt::Float, 8) => {
                let mut a = [0u8; 8];
                a.copy_from_slice(b);
                f64::from_le_bytes(a) as f32
            }
            _ => 0.0,
        }
    }

    /// Raw integer sample (unsigned, <= 16 bits).
    fn uint(raw: &Raw, i: usize) -> u32 {
        match raw.sb {
            1 => raw.data.get(i).copied().unwrap_or(0) as u32,
            _ => raw
                .data
                .get(i * 2..i * 2 + 2)
                .map_or(0, |b| u16::from_le_bytes([b[0], b[1]]) as u32),
        }
    }

    fn to_image(p: &Page, raw: Raw) -> Result<Image> {
        let (w, h) = (p.w, p.h);
        let n = w as usize * h as usize;
        let spp = raw.spp;
        let pm = raw.photometric;
        let has_alpha = p.alpha != Alpha::None;
        let ai = p.alpha_at;
        let color_n = match pm {
            0 | 1 | 3 => 1,
            2 | 6 => 3,
            _ => 4,
        };
        let gray = color_n == 1 && pm != 3;
        let small = raw.fmt == Fmt::Uint && raw.bps <= 8;
        let palette = pm == 3;
        let format = if palette || small {
            match (gray, has_alpha) {
                (true, false) => Format::R8Unorm,
                (true, true) => Format::Rg8Unorm,
                (false, false) => Format::Rgb8Srgb,
                (false, true) => Format::Rgba8Srgb,
            }
        } else if raw.fmt == Fmt::Uint && raw.bps == 16 {
            match (gray, has_alpha) {
                (true, false) => Format::R16Unorm,
                (true, true) => Format::Rg16Unorm,
                _ => Format::Rgba16Unorm,
            }
        } else if raw.fmt == Fmt::Float && raw.bps == 16 {
            match (gray, has_alpha) {
                (true, false) => Format::R16Float,
                (false, false) => Format::Rgb16Float,
                _ => Format::Rgba16Float,
            }
        } else if gray && !has_alpha {
            Format::R32Float
        } else {
            Format::Rgba32Float
        };
        let mut img = Image::new(format, w, h)?;
        let px = format.block_bytes();
        let invert = pm == 0 && raw.fmt != Fmt::Float;
        let ycc_planar = pm == 6;
        let ycc = ycc_planar.then(|| Ycc::new(p.coeffs, p.ref_bw));
        let cmap = p.colormap.as_ref();
        let cmap16 = cmap.is_some_and(|m| m.iter().any(|&v| v >= 256));
        for (i, out) in img.data.chunks_exact_mut(px).enumerate().take(n) {
            let base = i * spp;
            match format {
                Format::R8Unorm | Format::Rg8Unorm | Format::Rgb8Srgb | Format::Rgba8Srgb => {
                    let max = (1u32 << raw.bps) - 1;
                    let scale = |v: u32| ((v * 255 + max / 2) / max) as u8;
                    let mut c = [0u8; 4];
                    if palette {
                        let idx = uint(&raw, base) as usize;
                        let m = cmap.ok_or(Error::InvalidData)?;
                        let k = 1usize << p.bps;
                        for (j, cj) in c.iter_mut().take(3).enumerate() {
                            let v = *m.get(j * k + idx).ok_or(Error::InvalidData)?;
                            *cj = if cmap16 { (v >> 8) as u8 } else { v as u8 };
                        }
                    } else if pm == 5 {
                        let k = 255 - scale(uint(&raw, base + 3)) as u32;
                        for (j, cj) in c.iter_mut().take(3).enumerate() {
                            let v = 255 - scale(uint(&raw, base + j)) as u32;
                            *cj = (k * v / 255) as u8;
                        }
                    } else if let Some(t) = &ycc {
                        let s = |j| uint(&raw, base + j) as u8;
                        c[..3].copy_from_slice(&t.rgb(s(0), s(1), s(2)));
                    } else {
                        for (j, cj) in c.iter_mut().take(color_n).enumerate() {
                            let v = uint(&raw, base + j);
                            *cj = scale(if invert { max - v } else { v });
                        }
                    }
                    let a = if has_alpha {
                        scale(uint(&raw, base + ai))
                    } else {
                        255
                    };
                    match format {
                        Format::R8Unorm => out[0] = c[0],
                        Format::Rg8Unorm => {
                            out[0] = c[0];
                            out[1] = a;
                        }
                        Format::Rgb8Srgb => out.copy_from_slice(&c[..3]),
                        _ => {
                            out[..3].copy_from_slice(&c[..3]);
                            out[3] = a;
                        }
                    }
                }
                Format::R16Unorm | Format::Rg16Unorm | Format::Rgba16Unorm => {
                    let mut c = [0u16; 4];
                    if pm == 5 {
                        let k = 65535 - uint(&raw, base + 3) as u64;
                        for (j, cj) in c.iter_mut().take(3).enumerate() {
                            let v = 65535 - uint(&raw, base + j) as u64;
                            *cj = (v * k / 65535) as u16;
                        }
                    } else {
                        for (j, cj) in c.iter_mut().take(color_n).enumerate() {
                            let v = uint(&raw, base + j) as u16;
                            *cj = if invert { 65535 - v } else { v };
                        }
                    }
                    if color_n == 1 && format == Format::Rgba16Unorm {
                        c[1] = c[0];
                        c[2] = c[0];
                    }
                    let a = if has_alpha {
                        uint(&raw, base + ai) as u16
                    } else {
                        65535
                    };
                    let vals: &[u16] = match format {
                        Format::R16Unorm => &c[..1],
                        Format::Rg16Unorm => &[c[0], a],
                        _ => &[c[0], c[1], c[2], a],
                    };
                    for (d, v) in out.as_chunks_mut::<2>().0.iter_mut().zip(vals) {
                        d.copy_from_slice(&v.to_le_bytes());
                    }
                }
                _ => {
                    let mut c = [0.0f32; 4];
                    if pm == 5 {
                        let k = 1.0 - norm(&raw, base + 3);
                        for (j, cj) in c.iter_mut().take(3).enumerate() {
                            *cj = (1.0 - norm(&raw, base + j)) * k;
                        }
                    } else {
                        for (j, cj) in c.iter_mut().take(color_n).enumerate() {
                            let v = norm(&raw, base + j);
                            *cj = if invert {
                                if raw.fmt == Fmt::Int { -v } else { 1.0 - v }
                            } else {
                                v
                            };
                        }
                    }
                    if color_n == 1 {
                        c[1] = c[0];
                        c[2] = c[0];
                    }
                    c[3] = if has_alpha {
                        norm(&raw, base + ai)
                    } else {
                        1.0
                    };
                    let ch = match format {
                        Format::R16Float | Format::R32Float => 1,
                        Format::Rgb16Float => 3,
                        _ => 4,
                    };
                    let half = matches!(
                        format,
                        Format::R16Float | Format::Rgb16Float | Format::Rgba16Float
                    );
                    for (j, v) in c.iter().take(ch).enumerate() {
                        if half {
                            // Half samples: copy the stored bits (exact, keeps NaN payloads).
                            let bits = if pm == 5 || (color_n == 1 && j > 0 && j < 3) {
                                crate::half::from_f32(*v)
                            } else {
                                let src = if j == 3 {
                                    base + ai
                                } else {
                                    base + j.min(color_n - 1)
                                };
                                raw.data
                                    .get(src * 2..src * 2 + 2)
                                    .map_or(0, |b| u16::from_le_bytes([b[0], b[1]]))
                            };
                            let bits = if j == 3 && !has_alpha { 0x3C00 } else { bits };
                            out[j * 2..j * 2 + 2].copy_from_slice(&bits.to_le_bytes());
                        } else {
                            out[j * 4..j * 4 + 4].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                }
            }
        }
        Ok(img)
    }
}

#[cfg(feature = "tiff")]
mod enc {
    use super::*;
    use crate::Format;
    use std::io::Cursor;
    use tiff::encoder::colortype::{self, ColorType};
    use tiff::encoder::{TiffEncoder, TiffKind, TiffKindBig, TiffKindStandard};
    use tiff::tags::{PhotometricInterpretation, SampleFormat, Tag};

    macro_rules! color {
        ($name:ident, $inner:ty, $pi:ident, [$($b:expr),*], $sf:ident) => {
            struct $name;
            impl ColorType for $name {
                type Inner = $inner;
                const TIFF_VALUE: PhotometricInterpretation = PhotometricInterpretation::$pi;
                const BITS_PER_SAMPLE: &'static [u16] = &[$($b),*];
                const SAMPLE_FORMAT: &'static [SampleFormat] =
                    &[$({ let _ = $b; SampleFormat::$sf }),*];
                fn horizontal_predict(row: &[Self::Inner], result: &mut Vec<Self::Inner>) {
                    result.extend_from_slice(row);
                }
            }
        };
    }
    color!(GrayA8, u8, BlackIsZero, [8, 8], Uint);
    color!(GrayA16, u16, BlackIsZero, [16, 16], Uint);
    color!(Gray16F, u16, BlackIsZero, [16], IEEEFP);
    color!(Rgb16F, u16, RGB, [16, 16, 16], IEEEFP);
    color!(Rgba16F, u16, RGB, [16, 16, 16, 16], IEEEFP);

    fn map(_: tiff::TiffError) -> Error {
        Error::InvalidData
    }

    fn oom<T>(_: T) -> Error {
        Error::OutOfMemory
    }

    /// Stored layout of a format: (samples, bytes per sample, float, alpha).
    fn layout(f: Format) -> Option<(usize, usize, bool, bool)> {
        Some(match f {
            Format::R8Unorm => (1, 1, false, false),
            Format::Rg8Unorm => (2, 1, false, true),
            Format::Rgb8Unorm | Format::Rgb8Srgb => (3, 1, false, false),
            Format::Rgba8Unorm | Format::Rgba8Srgb => (4, 1, false, true),
            Format::R16Unorm => (1, 2, false, false),
            Format::Rg16Unorm => (2, 2, false, true),
            Format::Rgba16Unorm => (4, 2, false, true),
            Format::R16Float => (1, 2, true, false),
            Format::Rgb16Float => (3, 2, true, false),
            Format::Rgba16Float => (4, 2, true, true),
            Format::R32Float => (1, 4, true, false),
            Format::Rgba32Float => (4, 4, true, true),
            _ => return None,
        })
    }

    /// Applies the predictor to `data` (rows of `w * n` little-endian samples of `sb` bytes).
    fn predict(data: &mut [u8], w: usize, n: usize, sb: usize, float: bool) -> Result<()> {
        let row = w * n * sb;
        if row == 0 {
            return Ok(());
        }
        let mut tmp = vas::Vector::<u8, vas::storage::Dynamic>::default();
        tmp.resize(row, 0).map_err(oom)?;
        for r in data.chunks_exact_mut(row) {
            if float {
                // Byte planes, most significant first, then byte differences.
                let wc = w * n;
                for i in 0..wc {
                    for k in 0..sb {
                        tmp[(sb - 1 - k) * wc + i] = r[i * sb + k];
                    }
                }
                for i in (n..row).rev() {
                    tmp[i] = tmp[i].wrapping_sub(tmp[i - n]);
                }
                r.copy_from_slice(&tmp);
            } else {
                let stride = n * sb;
                for i in (stride..row).step_by(sb).rev() {
                    match sb {
                        1 => r[i] = r[i].wrapping_sub(r[i - stride]),
                        _ => {
                            let a = u16::from_le_bytes([r[i], r[i + 1]]);
                            let b = u16::from_le_bytes([r[i - stride], r[i - stride + 1]]);
                            r[i..i + 2].copy_from_slice(&a.wrapping_sub(b).to_le_bytes());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn encode_pages(pages: &[Surface], opts: &EncodeOptions) -> Result<Buffer> {
        if pages.is_empty() {
            return Err(Error::InvalidArgument);
        }
        let mut cur = Cursor::new(Vec::new());
        if opts.bigtiff {
            let e = TiffEncoder::new_big(&mut cur).map_err(map)?;
            write_all(e, pages, opts)?;
        } else {
            let e = TiffEncoder::new(&mut cur).map_err(map)?;
            write_all(e, pages, opts)?;
        }
        let bytes = cur.into_inner();
        let mut out = crate::buffer::zeroed(bytes.len())?;
        out.copy_from_slice(&bytes);
        Ok(out)
    }

    fn write_all<K: TiffKind>(
        e: TiffEncoder<&mut Cursor<Vec<u8>>, K>,
        pages: &[Surface],
        opts: &EncodeOptions,
    ) -> Result<()> {
        let comp = match opts.compression {
            Compression::None => tiff::encoder::Compression::Uncompressed,
            Compression::Lzw => tiff::encoder::Compression::Lzw,
            Compression::Deflate => {
                tiff::encoder::Compression::Deflate(tiff::encoder::DeflateLevel::Balanced)
            }
            Compression::PackBits => tiff::encoder::Compression::Packbits,
        };
        let mut e = e.with_compression(comp);
        for s in pages {
            page(&mut e, s, opts)?;
        }
        Ok(())
    }

    fn page<K: TiffKind>(
        e: &mut TiffEncoder<&mut Cursor<Vec<u8>>, K>,
        src: &Surface,
        opts: &EncodeOptions,
    ) -> Result<()> {
        // Storable copy of the pixels.
        let (format, mut data) = match src.format() {
            f if layout(f).is_some() => {
                let mut b = crate::buffer::zeroed(src.data().len())?;
                b.copy_from_slice(src.data());
                (f, b)
            }
            Format::Bgra8Unorm | Format::Bgra8Srgb => {
                let mut b = crate::buffer::zeroed(src.data().len())?;
                for (d, s) in b
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(src.data().as_chunks::<4>().0)
                {
                    d.copy_from_slice(&[s[2], s[1], s[0], s[3]]);
                }
                (Format::Rgba8Srgb, b)
            }
            f if crate::convert::can_read(f) => (
                Format::Rgba32Float,
                crate::convert::convert_to_buffer(src, Format::Rgba32Float)?,
            ),
            _ => return Err(Error::Unsupported),
        };
        let (n, sb, float, alpha) = layout(format).ok_or(Error::Unsupported)?;
        let (w, h) = (src.width(), src.height());
        // The predictor is only defined for LZW and Deflate (libtiff ignores it otherwise).
        let predictor =
            opts.predictor && matches!(opts.compression, Compression::Lzw | Compression::Deflate);
        if predictor {
            predict(&mut data, w as usize, n, sb, float)?;
        }
        let pred: u16 = match (predictor, float) {
            (false, _) => 1,
            (true, false) => 2,
            (true, true) => 3,
        };
        macro_rules! go {
            ($ct:ty, $conv:expr) => {{
                let v = $conv;
                let mut im = e.new_image::<$ct>(w, h).map_err(map)?;
                im.encoder().write_tag(Tag::Predictor, pred).map_err(map)?;
                if alpha {
                    im.encoder()
                        .write_tag(Tag::ExtraSamples, &[2u16][..])
                        .map_err(map)?;
                }
                im.write_data(&v).map_err(map)?;
            }};
        }
        let u16s = || -> Result<Vec<u16>> {
            let mut v = Vec::new();
            v.try_reserve_exact(data.len() / 2).map_err(oom)?;
            v.extend(
                data.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| u16::from_le_bytes([b[0], b[1]])),
            );
            Ok(v)
        };
        let f32s = || -> Result<Vec<f32>> {
            let mut v = Vec::new();
            v.try_reserve_exact(data.len() / 4).map_err(oom)?;
            v.extend(
                data.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
            Ok(v)
        };
        match format {
            Format::R8Unorm => go!(colortype::Gray8, &data[..]),
            Format::Rg8Unorm => go!(GrayA8, &data[..]),
            Format::Rgb8Unorm | Format::Rgb8Srgb => go!(colortype::RGB8, &data[..]),
            Format::Rgba8Unorm | Format::Rgba8Srgb => go!(colortype::RGBA8, &data[..]),
            Format::R16Unorm => go!(colortype::Gray16, u16s()?),
            Format::Rg16Unorm => go!(GrayA16, u16s()?),
            Format::Rgba16Unorm => go!(colortype::RGBA16, u16s()?),
            Format::R16Float => go!(Gray16F, u16s()?),
            Format::Rgb16Float => go!(Rgb16F, u16s()?),
            Format::Rgba16Float => go!(Rgba16F, u16s()?),
            Format::R32Float => go!(colortype::Gray32Float, f32s()?),
            Format::Rgba32Float => go!(colortype::RGBA32Float, f32s()?),
            _ => return Err(Error::Unsupported),
        }
        let _ = (TiffKindStandard, TiffKindBig);
        Ok(())
    }
}
