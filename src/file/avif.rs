//! AVIF (still images).
//!
//! Needs the `avif` cargo feature; without it every function returns `Unsupported`.
//!
//! Decoding: `avif-parse` reads the HEIF container (primary item, alpha auxiliary item),
//! `re_rav1d` (rav1d, the Rust port of dav1d, built without its assembly) decodes the AV1
//! payloads, and imgio converts YUV to RGB.
//!
//! - 8, 10 and 12 bits; 4:2:0, 4:2:2, 4:4:4 and monochrome; alpha; full and limited range.
//! - Matrix coefficients from the `colr` `nclx` box (falling back to the AV1 sequence header):
//!   BT.601 / BT.470BG (also for unspecified), BT.709, FCC, SMPTE 240, BT.2020 NCL, identity
//!   (GBR) and YCgCo. Constant-luminance, chroma-derived and ICtCp matrices are `Unsupported`.
//! - The YUV -> RGB conversion reproduces libavif's built-in path bit for bit (float32
//!   arithmetic, bilinear chroma upsampling with 9/3/3/1 weights, the same edge handling).
//!   libavif links libyuv in many builds and then uses libyuv's integer conversion for some
//!   8-bit cases, which differs by up to 2.
//! - Output: 8-bit files `Rgb8Srgb` / `Rgba8Srgb`, 10 / 12-bit files `Rgba16Unorm` (values
//!   converted at the file's depth, then scaled `v * 65535 / max`). Transfer characteristics
//!   (PQ, HLG) are not applied; `irot` / `imir` / `clap` are ignored; premultiplied alpha is
//!   unpremultiplied. Grids and animated AVIF (`avis`) are `Unsupported`.
//! - [`decode_planes`] returns the decoded Y, U, V (and alpha) planes as stored.
//!
//! Encoding: `ravif` (rav1e without assembly) from `Rgb8*`, `Rgba8*`, `Bgra8*`, gray
//! `R8Unorm` and gray + alpha `Rg8Unorm`, as 4:4:4 BT.601 full range, 8 or 10 bits.

use super::Image;
use crate::{Buffer, Result, Surface};

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    /// Color quality `1.0..=100.0` (default 80).
    pub quality: f32,
    /// Alpha quality `1.0..=100.0` (default 80).
    pub alpha_quality: f32,
    /// Encoder speed `1..=10`: 1 = slowest, best compression (default 6).
    pub speed: u8,
    /// Bits per sample in the file: 8 or 10 (default 10; better quality even for 8-bit input).
    pub bit_depth: u8,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            quality: 80.0,
            alpha_quality: 80.0,
            speed: 6,
            bit_depth: 10,
        }
    }
}

/// Chroma subsampling of the decoded planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    /// 4:2:0 (chroma halved horizontally and vertically).
    Yuv420,
    /// 4:2:2 (chroma halved horizontally).
    Yuv422,
    /// 4:4:4.
    Yuv444,
    /// Monochrome (no chroma planes).
    Mono,
}

impl Subsampling {
    /// `(x, y)` chroma shifts.
    pub fn shifts(self) -> (u32, u32) {
        match self {
            Subsampling::Yuv420 => (1, 1),
            Subsampling::Yuv422 => (1, 0),
            Subsampling::Yuv444 | Subsampling::Mono => (0, 0),
        }
    }
}

/// Decoded AVIF planes, as stored in the file.
///
/// Samples are bytes for 8-bit files and `u16` little endian otherwise, rows tightly packed;
/// chroma planes are `((width + sx) >> sx) x ((height + sy) >> sy)` for shifts `(sx, sy)`.
#[derive(Debug)]
pub struct Planes {
    pub width: u32,
    pub height: u32,
    /// 8, 10 or 12.
    pub depth: u8,
    pub subsampling: Subsampling,
    /// Full (`true`) or limited ("video") range.
    pub full_range: bool,
    /// CICP colour primaries, transfer characteristics and matrix coefficients (from the
    /// `nclx` box, else the AV1 sequence header).
    pub primaries: u16,
    pub transfer: u16,
    pub matrix: u16,
    pub y: Buffer,
    /// Empty for monochrome images.
    pub u: Buffer,
    pub v: Buffer,
    /// Alpha plane (`width x height`, same sample size as `y`), converted to full range.
    pub alpha: Option<Buffer>,
    /// The color planes are premultiplied by alpha (`prem` reference).
    pub premultiplied: bool,
}

/// Largest image decoded, in pixels (libavif's default limit).
pub const MAX_PIXELS: u64 = 16384 * 16384;

/// Decodes the image (`Rgb8Srgb` / `Rgba8Srgb` for 8-bit files, `Rgba16Unorm` otherwise).
pub fn decode(data: &[u8]) -> Result<Image> {
    planes_to_image(&decode_planes(data)?)
}

/// Decodes the primary image's planes without color conversion.
pub fn decode_planes(data: &[u8]) -> Result<Planes> {
    #[cfg(feature = "avif")]
    {
        imp::decode_planes(data)
    }
    #[cfg(not(feature = "avif"))]
    {
        let _ = data;
        Err(crate::Error::Unsupported)
    }
}

/// Encodes `src`.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    #[cfg(feature = "avif")]
    {
        imp::encode(src, opts)
    }
    #[cfg(not(feature = "avif"))]
    {
        let _ = (src, opts);
        Err(crate::Error::Unsupported)
    }
}

/// How the color planes map to RGB.
#[derive(Clone, Copy)]
enum Mode {
    /// Y'CbCr with luma coefficients `kr`, `kb`.
    Ycbcr(f32, f32),
    /// GBR (matrix 0).
    Identity,
    /// YCgCo (matrix 8).
    Ycgco,
}

fn mode(matrix: u16) -> Result<Mode> {
    Ok(match matrix {
        0 => Mode::Identity,
        1 => Mode::Ycbcr(0.2126, 0.0722),
        4 => Mode::Ycbcr(0.30, 0.11),
        7 => Mode::Ycbcr(0.212, 0.087),
        8 => Mode::Ycgco,
        9 => Mode::Ycbcr(0.2627, 0.0593),
        10..=14 => return Err(crate::Error::Unsupported),
        // 5, 6 (BT.601), 2 (unspecified) and reserved values: BT.601, as libavif.
        _ => Mode::Ycbcr(0.299, 0.114),
    })
}

/// Sample `i` of a plane (u8, or u16 little endian when `wide`).
#[inline]
fn sample(p: &[u8], i: usize, wide: bool) -> usize {
    if wide {
        p.get(2 * i..2 * i + 2)
            .map_or(0, |b| usize::from(u16::from_le_bytes([b[0], b[1]])))
    } else {
        p.get(i).map_or(0, |&b| usize::from(b))
    }
}

/// Converts decoded planes to RGB(A) exactly like libavif's built-in conversion
/// (`avifImageYUVAnyToRGBAnySlow`, bilinear chroma upsampling).
pub fn planes_to_image(p: &Planes) -> Result<Image> {
    use crate::Error;
    use crate::Format;
    let (w, h) = (p.width as usize, p.height as usize);
    if !matches!(p.depth, 8 | 10 | 12) || w == 0 || h == 0 {
        return Err(Error::InvalidArgument);
    }
    let wide = p.depth > 8;
    let bps = if wide { 2 } else { 1 };
    let (sx, sy) = p.subsampling.shifts();
    let (cw, ch) = ((w + sx as usize) >> sx, (h + sy as usize) >> sy);
    let mono = p.subsampling == Subsampling::Mono;
    if p.y.len() != w * h * bps
        || (!mono && (p.u.len() != cw * ch * bps || p.v.len() != cw * ch * bps))
        || p.alpha.as_ref().is_some_and(|a| a.len() != w * h * bps)
    {
        return Err(Error::InvalidArgument);
    }
    let mode = mode(p.matrix)?;
    let max = (1usize << p.depth) - 1;
    let maxf = max as f32;

    // libavif unorm -> float tables.
    let (bias_y, range_y, range_uv) = if p.full_range {
        (0.0, maxf, maxf)
    } else {
        let s = (p.depth - 8) as u32;
        (
            (16u32 << s) as f32,
            (219u32 << s) as f32,
            (224u32 << s) as f32,
        )
    };
    let bias_uv = (1u32 << (p.depth - 1)) as f32;
    let mut table_y = crate::buffer::zeroed((max + 1) * 4)?;
    let mut table_uv = crate::buffer::zeroed((max + 1) * 4)?;
    for i in 0..=max {
        let ty = (i as f32 - bias_y) / range_y;
        let tuv = (i as f32 - bias_uv) / range_uv;
        table_y[i * 4..i * 4 + 4].copy_from_slice(&ty.to_ne_bytes());
        table_uv[i * 4..i * 4 + 4].copy_from_slice(&tuv.to_ne_bytes());
    }
    let lookup = |t: &[u8], v: usize| -> f32 {
        let v = v.min(max) * 4;
        f32::from_ne_bytes([t[v], t[v + 1], t[v + 2], t[v + 3]])
    };
    // Identity uses the luma range for every plane.
    let chroma_table: &[u8] = if matches!(mode, Mode::Identity) {
        &table_y
    } else {
        &table_uv
    };

    let has_alpha = p.alpha.is_some();
    let (fmt, out_ch, out_bps) = match (wide, has_alpha) {
        (false, false) => (Format::Rgb8Srgb, 3, 1),
        (false, true) => (Format::Rgba8Srgb, 4, 1),
        (true, _) => (Format::Rgba16Unorm, 4, 2),
    };
    let mut img = Image::new(fmt, p.width, p.height)?;
    let row_bytes = w * out_ch * out_bps;
    let unpremultiply = p.premultiplied && has_alpha;
    crate::parallel::for_each_unit(&mut img.data, row_bytes, w * h, |j, row| {
        // Vertical neighbour row for bilinear upsampling (libavif's edge rules).
        let cj = j >> sy;
        let nj = if sy == 0 || j == 0 || (j == h - 1 && j % 2 != 0) {
            cj
        } else if j % 2 != 0 {
            cj + 1
        } else {
            cj - 1
        };
        for i in 0..w {
            let yv = lookup(&table_y, sample(&p.y, j * w + i, wide));
            let (r, g, b) = if mono {
                (yv, yv, yv)
            } else {
                let ci = i >> sx;
                let ni = if sx == 0 || i == 0 || (i == w - 1 && i % 2 != 0) {
                    ci
                } else if i % 2 != 0 {
                    ci + 1
                } else {
                    ci - 1
                };
                let c = |plane: &[u8], y: usize, x: usize| {
                    lookup(chroma_table, sample(plane, y * cw + x, wide))
                };
                let up = |plane: &[u8]| -> f32 {
                    if sx == 0 {
                        c(plane, cj, ci)
                    } else {
                        // 4:2:2 uses the same row twice.
                        c(plane, cj, ci) * (9.0 / 16.0)
                            + c(plane, cj, ni) * (3.0 / 16.0)
                            + c(plane, nj, ci) * (3.0 / 16.0)
                            + c(plane, nj, ni) * (1.0 / 16.0)
                    }
                };
                let (cb, cr) = (up(&p.u), up(&p.v));
                match mode {
                    Mode::Identity => (cr, yv, cb),
                    Mode::Ycgco => {
                        let t = yv - cb;
                        (t + cr, yv + cb, t - cr)
                    }
                    Mode::Ycbcr(kr, kb) => {
                        let kg = 1.0 - kr - kb;
                        let r = yv + (2.0 * (1.0 - kr)) * cr;
                        let b = yv + (2.0 * (1.0 - kb)) * cb;
                        let g =
                            yv - ((2.0 * ((kr * (1.0 - kr) * cr) + (kb * (1.0 - kb) * cb))) / kg);
                        (r, g, b)
                    }
                }
            };
            let mut rgb = [r, g, b].map(|v| v.clamp(0.0, 1.0));
            let alpha = p.alpha.as_ref().map(|a| sample(a, j * w + i, wide));
            if unpremultiply && let Some(a) = alpha {
                let af = a as f32 / maxf;
                for v in &mut rgb {
                    *v = if af > 0.0 { (*v / af).min(1.0) } else { 0.0 };
                }
            }
            let px = &mut row[i * out_ch * out_bps..(i + 1) * out_ch * out_bps];
            let q = |v: f32| (0.5 + v * maxf) as usize;
            if wide {
                let s16 = |v: usize| (((v.min(max) * 65535 + max / 2) / max) as u16).to_le_bytes();
                for (k, v) in rgb.iter().enumerate() {
                    px[2 * k..2 * k + 2].copy_from_slice(&s16(q(*v)));
                }
                px[6..8].copy_from_slice(&s16(alpha.unwrap_or(max)));
            } else {
                for (k, v) in rgb.iter().enumerate() {
                    px[k] = q(*v).min(255) as u8;
                }
                if let Some(a) = alpha {
                    px[3] = a.min(255) as u8;
                }
            }
        }
    });
    Ok(img)
}

#[cfg(feature = "avif")]
mod imp {
    use super::{EncodeOptions, MAX_PIXELS, Planes, Subsampling};
    use crate::{Buffer, Error, Format, Result, Surface};
    use re_rav1d::dav1d::{self, PixelLayout, PlanarImageComponent as Comp, Settings, pixel};

    /// An ISO BMFF box: type and payload.
    struct Bx<'a> {
        ty: [u8; 4],
        data: &'a [u8],
    }

    fn be16(d: &[u8], o: usize) -> Option<u32> {
        d.get(o..o + 2)
            .map(|b| u32::from(u16::from_be_bytes([b[0], b[1]])))
    }

    fn be32(d: &[u8], o: usize) -> Option<u32> {
        d.get(o..o + 4)
            .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Calls `f` for each box of a box list until it returns `Some`.
    fn find_box<'a, T>(mut d: &'a [u8], mut f: impl FnMut(Bx<'a>) -> Option<T>) -> Option<T> {
        while d.len() >= 8 {
            let size = be32(d, 0)? as usize;
            let ty = [d[4], d[5], d[6], d[7]];
            let (head, size) = match size {
                0 => (8, d.len()),
                1 => (
                    16,
                    usize::try_from(u64::from(be32(d, 8)?) << 32 | u64::from(be32(d, 12)?)).ok()?,
                ),
                s => (8, s),
            };
            if size < head || size > d.len() {
                return None;
            }
            if let Some(r) = f(Bx {
                ty,
                data: &d[head..size],
            }) {
                return Some(r);
            }
            d = &d[size..];
        }
        None
    }

    fn child<'a>(d: &'a [u8], ty: &[u8; 4]) -> Option<&'a [u8]> {
        find_box(d, |b| (&b.ty == ty).then_some(b.data))
    }

    /// The `nclx` colour information of the primary item: (primaries, transfer, matrix,
    /// full range).
    fn nclx(file: &[u8]) -> Option<(u16, u16, u16, bool)> {
        let meta = child(file, b"meta")?.get(4..)?;
        let pitm = child(meta, b"pitm")?;
        let primary = if pitm.first()? == &0 {
            be16(pitm, 4)?
        } else {
            be32(pitm, 4)?
        };
        let iprp = child(meta, b"iprp")?;
        let ipco = child(iprp, b"ipco")?;
        let mut found = None;
        find_box(iprp, |ipma| {
            if &ipma.ty != b"ipma" {
                return None;
            }
            let d = ipma.data;
            let (version, flags) = (*d.first()?, *d.get(3)?);
            let count = be32(d, 4)?;
            let mut o = 8;
            for _ in 0..count {
                let item = if version < 1 {
                    be16(d, o)?
                } else {
                    be32(d, o)?
                };
                o += if version < 1 { 2 } else { 4 };
                let n = usize::from(*d.get(o)?);
                o += 1;
                for _ in 0..n {
                    let index = if flags & 1 != 0 {
                        let v = be16(d, o)? & 0x7FFF;
                        o += 2;
                        v
                    } else {
                        let v = u32::from(*d.get(o)?) & 0x7F;
                        o += 1;
                        v
                    };
                    if item != primary || index == 0 {
                        continue;
                    }
                    let mut k = 0;
                    let prop = find_box(ipco, |b| {
                        k += 1;
                        (k == index).then_some(b)
                    })?;
                    if &prop.ty == b"colr" && prop.data.get(..4) == Some(b"nclx") {
                        let p = prop.data;
                        found = Some((
                            be16(p, 4)? as u16,
                            be16(p, 6)? as u16,
                            be16(p, 8)? as u16,
                            p.get(10)? & 0x80 != 0,
                        ));
                    }
                }
            }
            Some(())
        });
        found
    }

    /// Decodes one AV1 payload (a still frame).
    fn decode_av1<T>(obu: T, threads: u32) -> Result<dav1d::Picture>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        let mut s = Settings::new();
        s.set_n_threads(threads);
        s.set_max_frame_delay(1);
        s.set_frame_size_limit(MAX_PIXELS as u32);
        s.set_all_layers(false);
        let mut dec = dav1d::Decoder::with_settings(&s).map_err(|_| Error::InvalidData)?;
        match dec.send_data(obu, None, None, None) {
            Ok(()) => {}
            Err(e) if e.is_again() => {}
            Err(_) => return Err(Error::InvalidData),
        }
        for _ in 0..16 {
            match dec.get_picture() {
                Ok(p) => return Ok(p),
                Err(e) if e.is_again() => match dec.send_pending_data() {
                    Ok(()) => {}
                    Err(e) if e.is_again() => {}
                    Err(_) => return Err(Error::InvalidData),
                },
                Err(_) => return Err(Error::InvalidData),
            }
        }
        Err(Error::InvalidData)
    }

    /// Copies a plane of `w x h` samples (`bps` bytes each) out of a strided picture plane.
    fn copy_plane(pic: &dav1d::Picture, c: Comp, w: usize, h: usize, bps: usize) -> Result<Buffer> {
        let stride = pic.stride(c) as usize;
        let src = pic.plane(c);
        let row = w * bps;
        if stride < row || src.len() < stride * (h - 1) + row {
            return Err(Error::InvalidData);
        }
        let mut out = crate::buffer::zeroed(row * h)?;
        for (y, dst) in out.chunks_exact_mut(row).enumerate() {
            let s = &src[y * stride..y * stride + row];
            if bps == 2 {
                // dav1d stores 16-bit samples in native byte order.
                for (d, s) in dst
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(s.as_chunks::<2>().0)
                {
                    *d = u16::from_ne_bytes(*s).to_le_bytes();
                }
            } else {
                dst.copy_from_slice(s);
            }
        }
        Ok(out)
    }

    fn depth_of(pic: &dav1d::Picture) -> Result<u8> {
        match pic.bits_per_component().map(|b| b.0) {
            Some(8) => Ok(8),
            Some(10) => Ok(10),
            Some(12) => Ok(12),
            _ => Err(Error::Unsupported),
        }
    }

    fn map_parse(e: avif_parse::Error) -> Error {
        match e {
            avif_parse::Error::Unsupported(_) => Error::Unsupported,
            avif_parse::Error::OutOfMemory => Error::OutOfMemory,
            _ => Error::InvalidData,
        }
    }

    pub fn decode_planes(data: &[u8]) -> Result<Planes> {
        let avif = avif_parse::read_avif(&mut &data[..]).map_err(map_parse)?;
        let meta = avif.primary_item_metadata().map_err(map_parse)?;
        let px = u64::from(meta.max_frame_width.get()) * u64::from(meta.max_frame_height.get());
        if px > MAX_PIXELS {
            return Err(Error::Unsupported);
        }
        let threads = if px >= 512 * 512 { 0 } else { 1 };
        let premultiplied = avif.premultiplied_alpha;
        let alpha_item = avif.alpha_item;
        let pic = decode_av1(avif.primary_item, threads)?;
        let (w, h) = (pic.width() as usize, pic.height() as usize);
        if w == 0 || h == 0 || (w * h) as u64 > MAX_PIXELS {
            return Err(Error::InvalidData);
        }
        let depth = depth_of(&pic)?;
        let bps = if depth > 8 { 2 } else { 1 };
        let subsampling = match pic.pixel_layout() {
            PixelLayout::I400 => Subsampling::Mono,
            PixelLayout::I420 => Subsampling::Yuv420,
            PixelLayout::I422 => Subsampling::Yuv422,
            PixelLayout::I444 => Subsampling::Yuv444,
        };
        let (sx, sy) = subsampling.shifts();
        let (cw, ch) = ((w + sx as usize) >> sx, (h + sy as usize) >> sy);
        let y = copy_plane(&pic, Comp::Y, w, h, bps)?;
        let (u, v) = if subsampling == Subsampling::Mono {
            (Buffer::default(), Buffer::default())
        } else {
            (
                copy_plane(&pic, Comp::U, cw, ch, bps)?,
                copy_plane(&pic, Comp::V, cw, ch, bps)?,
            )
        };
        let (primaries, transfer, matrix, full_range) = match nclx(data) {
            // The range always comes from the AV1 sequence header, as in libavif.
            Some((cp, tc, mc, _)) => (cp, tc, mc, pic.color_range() == pixel::YUVRange::Full),
            None => (
                // av-data's CICP enums carry the code points as discriminants.
                pic.color_primaries() as u16,
                pic.transfer_characteristic() as u16,
                pic.matrix_coefficients() as u16,
                pic.color_range() == pixel::YUVRange::Full,
            ),
        };
        drop(pic);

        let alpha = match alpha_item {
            None => None,
            Some(item) => {
                let a = decode_av1(item, threads)?;
                if a.width() as usize != w || a.height() as usize != h {
                    return Err(Error::InvalidData);
                }
                let ad = depth_of(&a)?;
                let mut plane = copy_plane(&a, Comp::Y, w, h, if ad > 8 { 2 } else { 1 })?;
                if a.color_range() != pixel::YUVRange::Full || ad != depth {
                    plane =
                        alpha_to_full(&plane, ad, a.color_range() == pixel::YUVRange::Full, depth)?;
                }
                Some(plane)
            }
        };
        Ok(Planes {
            width: w as u32,
            height: h as u32,
            depth,
            subsampling,
            full_range,
            primaries,
            transfer,
            matrix,
            y,
            u,
            v,
            alpha,
            premultiplied,
        })
    }

    /// Alpha to full range (libavif `avifLimitedToFullY`) and the color depth.
    fn alpha_to_full(src: &[u8], depth: u8, full: bool, out_depth: u8) -> Result<Buffer> {
        let wide_in = depth > 8;
        let n = src.len() / if wide_in { 2 } else { 1 };
        let out_bps = if out_depth > 8 { 2 } else { 1 };
        let mut out = crate::buffer::zeroed(n * out_bps)?;
        let (max, omax) = ((1i64 << depth) - 1, (1i64 << out_depth) - 1);
        let s = i64::from(depth - 8);
        for i in 0..n {
            let mut v = super::sample(src, i, wide_in) as i64;
            if !full {
                v = (((v - (16 << s)) * max) / ((219 << s) as i64)).clamp(0, max);
            }
            let o = (v * omax + max / 2) / max;
            if out_bps == 2 {
                out[2 * i..2 * i + 2].copy_from_slice(&(o as u16).to_le_bytes());
            } else {
                out[i] = o as u8;
            }
        }
        Ok(out)
    }

    pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
        use ravif::{BitDepth, Img, RGB8, RGBA8};
        let q_ok = |q: f32| (1.0..=100.0).contains(&q);
        if !q_ok(opts.quality) || !q_ok(opts.alpha_quality) || !(1..=10).contains(&opts.speed) {
            return Err(Error::InvalidArgument);
        }
        let depth = match opts.bit_depth {
            8 => BitDepth::Eight,
            10 => BitDepth::Ten,
            _ => return Err(Error::InvalidArgument),
        };
        let (w, h) = (src.width() as usize, src.height() as usize);
        if w == 0 || h == 0 {
            return Err(Error::InvalidArgument);
        }
        let enc = ravif::Encoder::new()
            .with_quality(opts.quality)
            .with_alpha_quality(opts.alpha_quality)
            .with_speed(opts.speed)
            .with_bit_depth(depth);
        let d = src.data();
        let rgba = |px: &mut dyn Iterator<Item = RGBA8>| -> Result<ravif::EncodedImage> {
            let mut v = Vec::new();
            v.try_reserve(w * h).map_err(|_| Error::OutOfMemory)?;
            v.extend(px);
            enc.encode_rgba(Img::new(&v[..], w, h))
                .map_err(|_| Error::InvalidArgument)
        };
        let rgb = |px: &mut dyn Iterator<Item = RGB8>| -> Result<ravif::EncodedImage> {
            let mut v = Vec::new();
            v.try_reserve(w * h).map_err(|_| Error::OutOfMemory)?;
            v.extend(px);
            enc.encode_rgb(Img::new(&v[..], w, h))
                .map_err(|_| Error::InvalidArgument)
        };
        let out = match src.format() {
            Format::Rgb8Srgb | Format::Rgb8Unorm => rgb(&mut d
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| RGB8::new(p[0], p[1], p[2])))?,
            Format::Rgba8Srgb | Format::Rgba8Unorm => rgba(
                &mut d
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| RGBA8::new(p[0], p[1], p[2], p[3])),
            )?,
            Format::Bgra8Srgb | Format::Bgra8Unorm => rgba(
                &mut d
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| RGBA8::new(p[2], p[1], p[0], p[3])),
            )?,
            Format::R8Unorm => rgb(&mut d.iter().map(|&g| RGB8::new(g, g, g)))?,
            Format::Rg8Unorm => rgba(
                &mut d
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| RGBA8::new(p[0], p[0], p[0], p[1])),
            )?,
            _ => return Err(Error::Unsupported),
        };
        let mut buf = Buffer::default();
        buf.extend_from_slice(&out.avif_file)
            .map_err(|_| Error::OutOfMemory)?;
        Ok(buf)
    }
}
