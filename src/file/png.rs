//! PNG and APNG (`png` crate, pure Rust).
//!
//! Decode: every color type and bit depth (1/2/4/8/16-bit gray, gray + alpha, RGB, RGBA,
//! palette), Adam7 interlacing, tRNS (palette alpha, gray / RGB color keys become an alpha
//! channel). Output formats follow [`crate::file`]: 8-bit gray `R8Unorm`, gray + alpha
//! `Rg8Unorm`, RGB / palette `Rgb8Srgb`, RGBA / palette with tRNS `Rgba8Srgb`, 16-bit gray
//! `R16Unorm`, gray + alpha `Rg16Unorm`, RGB / RGBA `Rgba16Unorm` (RGB: alpha 65535). 8-bit color
//! is labeled sRGB whatever the color chunks say; [`info`] exposes gAMA / cHRM / sRGB / iCCP /
//! cICP and [`ColorInfo::is_srgb`].
//!
//! APNG: [`decode_animation`] composites every frame on a canvas that starts transparent black
//! (dispose ops NONE / BACKGROUND / PREVIOUS, blend ops SOURCE / OVER with the APNG
//! specification's integer formula, frame offsets); a default image that is not part of the
//! animation (IDAT before the first fcTL) is skipped there but is what [`decode`] returns.
//! Delays are `(delay_num, delay_den)` with a zero denominator read as 100; `loops` is
//! `num_plays` (0 = forever).
//!
//! Encode: 8- and 16-bit gray, gray + alpha, RGB, RGBA, picked from the surface format
//! (`R8Unorm`, `Rg8Unorm`, `Rgb8*`, `Rgba8*`, `R16Unorm`, `Rg16Unorm`, `Rgba16Unorm`;
//! `Bgra8*`, `Rgb565`, `Bgr565` and `Rgba4444` are widened to 8-bit losslessly; anything else is
//! `Unsupported`). Optional lossless palette reduction and opaque alpha stripping (the way to
//! write 16-bit RGB). [`encode_animation`] writes an APNG of full-canvas frames.
//!
//! Needs the `png` cargo feature; without it every function returns `Unsupported`.

use super::{Animation, Image};
use crate::{Buffer, Result, Surface};

/// Default [`DecodeOptions::max_bytes`]: 1 GiB.
pub const DEFAULT_MAX_BYTES: usize = 1 << 30;

/// Decoder options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Upper bound for the decoded pixels (all frames of an animation together) and the
    /// decoder's internal buffers; larger files fail with `OutOfMemory`.
    pub max_bytes: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions {
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

/// Deflate effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// Stored (uncompressed) deflate blocks; disables filtering.
    None,
    /// `fdeflate` ultra fast mode.
    Fastest,
    /// `fdeflate` with a better ratio.
    Fast,
    /// Balanced speed / size (zlib level 6 class).
    #[default]
    Default,
    /// Slower, slightly smaller files.
    Best,
    /// Explicit zlib level 1..=9 (clamped).
    Level(u8),
}

/// Row filter strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Filter {
    /// No filtering.
    None,
    /// Every row `Sub`.
    Sub,
    /// Every row `Up`.
    Up,
    /// Every row `Average`.
    Average,
    /// Every row `Paeth`.
    Paeth,
    /// Per row, the filter with the smallest sum of absolute differences.
    #[default]
    Adaptive,
    /// Per row, the filter with the lowest entropy (slower).
    MinEntropy,
}

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    /// Deflate effort (default [`Compression::Default`]).
    pub compression: Compression,
    /// Row filters (default [`Filter::Adaptive`]).
    pub filter: Filter,
    /// Store 8-bit RGB / RGBA input with at most 256 distinct colors as an indexed image
    /// (PLTE, plus tRNS for RGBA input). Lossless: the file decodes to the same format and
    /// pixels. Default `false`.
    pub palette: bool,
    /// Drop an alpha channel that is fully opaque (RGBA -> RGB, gray + alpha -> gray); the
    /// only way to write 16-bit RGB. The pixels are kept, the decoded format loses its alpha
    /// (8-bit), or keeps `Rgba16Unorm` (16-bit RGB decodes with alpha 65535). Default `false`.
    pub strip_opaque_alpha: bool,
    /// Write an sRGB chunk (perceptual intent) for `*Srgb` input formats. Default `true`.
    pub srgb_chunk: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            compression: Compression::Default,
            filter: Filter::Adaptive,
            palette: false,
            strip_opaque_alpha: false,
            srgb_chunk: true,
        }
    }
}

/// PNG color type (IHDR).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorType {
    Gray,
    Rgb,
    Indexed,
    GrayAlpha,
    Rgba,
}

/// Color space chunks.
#[derive(Debug, Default)]
pub struct ColorInfo {
    /// sRGB chunk: rendering intent (0 perceptual, 1 relative colorimetric, 2 saturation,
    /// 3 absolute colorimetric).
    pub srgb_intent: Option<u8>,
    /// gAMA: file gamma x 100000 (45455 = 1 / 2.2).
    pub gamma: Option<u32>,
    /// cHRM: white x, y, red x, y, green x, y, blue x, y (x 100000).
    pub chromaticities: Option<[u32; 8]>,
    /// iCCP: the decompressed ICC profile.
    pub icc_profile: Option<Buffer>,
    /// cICP: colour primaries, transfer function, matrix coefficients, full range flag.
    pub cicp: Option<[u8; 4]>,
}

impl ColorInfo {
    /// True when the samples are sRGB encoded: an sRGB chunk, a cICP with sRGB primaries and
    /// transfer (1, 13), or no color chunks at all (the PNG default). An ICC profile, another
    /// cICP or a gAMA other than 1/2.2 (45455 +- 100) makes it false; cHRM alone is ignored.
    pub fn is_srgb(&self) -> bool {
        if self.srgb_intent.is_some() {
            return true;
        }
        if let Some(c) = self.cicp {
            return c[0] == 1 && c[1] == 13;
        }
        if self.icc_profile.is_some() {
            return false;
        }
        self.gamma.is_none_or(|g| g.abs_diff(45455) <= 100)
    }
}

/// Animation control (acTL).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationInfo {
    /// Frames of the animation (without a hidden default image).
    pub frames: u32,
    /// Number of plays, 0 = forever.
    pub plays: u32,
    /// False when the IDAT image is a default image that is not part of the animation.
    pub default_image_is_first_frame: bool,
}

/// Header and metadata of a PNG file.
#[derive(Debug)]
pub struct PngInfo {
    pub width: u32,
    pub height: u32,
    pub color_type: ColorType,
    /// Bits per sample (1, 2, 4, 8 or 16).
    pub bit_depth: u8,
    pub interlaced: bool,
    /// A tRNS chunk is present (palette alpha or a gray / RGB color key).
    pub has_trns: bool,
    /// acTL: present for APNG.
    pub animation: Option<AnimationInfo>,
    pub color: ColorInfo,
}

/// Reads the header and the chunks before the image data.
pub fn info(data: &[u8]) -> Result<PngInfo> {
    #[cfg(feature = "png")]
    {
        imp::info(data)
    }
    #[cfg(not(feature = "png"))]
    {
        let _ = data;
        Err(crate::Error::Unsupported)
    }
}

/// Decodes the main image (the IDAT image: the first frame of an APNG, or its default image).
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_with(data, &DecodeOptions::default())
}

/// [`decode`] with options.
pub fn decode_with(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    #[cfg(feature = "png")]
    {
        imp::decode(data, opts)
    }
    #[cfg(not(feature = "png"))]
    {
        let _ = (data, opts);
        Err(crate::Error::Unsupported)
    }
}

/// Decodes every frame (composited, canvas sized); a PNG without acTL is one frame.
pub fn decode_animation(data: &[u8]) -> Result<Animation> {
    decode_animation_with(data, &DecodeOptions::default())
}

/// [`decode_animation`] with options.
pub fn decode_animation_with(data: &[u8], opts: &DecodeOptions) -> Result<Animation> {
    #[cfg(feature = "png")]
    {
        imp::decode_animation(data, opts)
    }
    #[cfg(not(feature = "png"))]
    {
        let _ = (data, opts);
        Err(crate::Error::Unsupported)
    }
}

/// Encodes `src`.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    #[cfg(feature = "png")]
    {
        imp::encode(core::slice::from_ref(src), None, opts)
    }
    #[cfg(not(feature = "png"))]
    {
        let _ = (src, opts);
        Err(crate::Error::Unsupported)
    }
}

/// Encodes an APNG: every frame is a full canvas (same format and size), written with
/// dispose NONE / blend SOURCE and its delay (reduced to fit 16 bits); `num_plays` =
/// `anim.loops`. The first frame is the IDAT image (shown by non-APNG decoders).
pub fn encode_animation(anim: &Animation, opts: &EncodeOptions) -> Result<Buffer> {
    #[cfg(feature = "png")]
    {
        imp::encode_animation(anim, opts)
    }
    #[cfg(not(feature = "png"))]
    {
        let _ = (anim, opts);
        Err(crate::Error::Unsupported)
    }
}

/// Reduces `num / den` to 16-bit terms (`den` 0 means 1/100 s units, kept as 100).
#[cfg_attr(not(feature = "png"), allow(dead_code))]
fn delay_u16(num: u32, den: u32) -> (u16, u16) {
    let den = if den == 0 { 100 } else { den };
    let g = gcd(num, den).max(1);
    let (mut n, mut d) = (u64::from(num / g), u64::from(den / g));
    if n > 0xFFFF || d > 0xFFFF {
        // Scale both down, keeping the ratio as close as possible.
        let s = n.max(d).div_ceil(0xFFFF);
        n = (n + s / 2) / s;
        d = ((d + s / 2) / s).max(1);
    }
    (n.min(0xFFFF) as u16, d.min(0xFFFF) as u16)
}

#[cfg_attr(not(feature = "png"), allow(dead_code))]
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(feature = "png")]
mod imp {
    use super::{AnimationInfo, ColorInfo, ColorType, DecodeOptions, EncodeOptions, PngInfo};
    use crate::file::{Animation, Frame, Image};
    use crate::{Buffer, Error, Format, Result, Surface};
    use std::io::Cursor;
    use vas::Vector;
    use vas::storage::Dynamic;

    /// Deflate cannot expand data by more than ~1032:1.
    const MAX_INFLATE_RATIO: usize = 1040;

    fn err(e: png::DecodingError) -> Error {
        match e {
            png::DecodingError::LimitsExceeded => Error::OutOfMemory,
            _ => Error::InvalidData,
        }
    }

    fn enc_err(e: png::EncodingError) -> Error {
        match e {
            png::EncodingError::LimitsExceeded => Error::OutOfMemory,
            png::EncodingError::IoError(_) => Error::OutOfMemory,
            _ => Error::InvalidArgument,
        }
    }

    fn buffer_from(src: &[u8]) -> Result<Buffer> {
        let mut b = Buffer::default();
        b.extend_from_slice(src).map_err(|_| Error::OutOfMemory)?;
        Ok(b)
    }

    fn reader<'a>(data: &'a [u8], opts: &DecodeOptions) -> Result<png::Reader<Cursor<&'a [u8]>>> {
        let limits = png::Limits {
            bytes: opts.max_bytes,
        };
        let mut dec = png::Decoder::new_with_limits(Cursor::new(data), limits);
        dec.set_transformations(png::Transformations::EXPAND);
        dec.set_ignore_text_chunk(true);
        let r = dec.read_info().map_err(err)?;
        // The IDAT image must be backed by compressed data: reject headers whose image could
        // not come from this many bytes before allocating for it.
        let i = r.info();
        let bits =
            u64::from(i.width) * u64::from(samples(i.color_type)) * u64::from(i.bit_depth as u8);
        let raw = (bits.div_ceil(8) + 1).saturating_mul(u64::from(i.height));
        let max = (data.len() as u64).saturating_mul(MAX_INFLATE_RATIO as u64);
        if raw > max {
            return Err(Error::InvalidData);
        }
        Ok(r)
    }

    fn samples(ct: png::ColorType) -> u32 {
        match ct {
            png::ColorType::Grayscale | png::ColorType::Indexed => 1,
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Rgb => 3,
            png::ColorType::Rgba => 4,
        }
    }

    pub(super) fn info(data: &[u8]) -> Result<PngInfo> {
        let r = reader(data, &DecodeOptions::default())?;
        let i = r.info();
        let color_type = match i.color_type {
            png::ColorType::Grayscale => ColorType::Gray,
            png::ColorType::Rgb => ColorType::Rgb,
            png::ColorType::Indexed => ColorType::Indexed,
            png::ColorType::GrayscaleAlpha => ColorType::GrayAlpha,
            png::ColorType::Rgba => ColorType::Rgba,
        };
        let chromaticities = i.chrm_chunk.map(|c| {
            [
                c.white.0, c.white.1, c.red.0, c.red.1, c.green.0, c.green.1, c.blue.0, c.blue.1,
            ]
            .map(|v| v.into_scaled())
        });
        let icc_profile = match &i.icc_profile {
            Some(p) => Some(buffer_from(p)?),
            None => None,
        };
        let color = ColorInfo {
            srgb_intent: i.srgb.map(|s| s as u8),
            gamma: i.gama_chunk.map(|g| g.into_scaled()),
            chromaticities,
            icc_profile,
            cicp: i.coding_independent_code_points.map(|c| {
                [
                    c.color_primaries,
                    c.transfer_function,
                    c.matrix_coefficients,
                    u8::from(c.is_video_full_range_image),
                ]
            }),
        };
        let animation = i.animation_control.map(|a| AnimationInfo {
            frames: a.num_frames,
            plays: a.num_plays,
            default_image_is_first_frame: i.frame_control.is_some(),
        });
        Ok(PngInfo {
            width: i.width,
            height: i.height,
            color_type,
            bit_depth: i.bit_depth as u8,
            interlaced: i.interlaced,
            has_trns: i.trns.is_some(),
            animation,
            color,
        })
    }

    /// The decoded format of the reader's output (after `EXPAND`).
    fn out_format(r: &png::Reader<Cursor<&[u8]>>) -> Result<Format> {
        use png::{BitDepth::*, ColorType::*};
        Ok(match r.output_color_type() {
            (Grayscale, Eight) => Format::R8Unorm,
            (GrayscaleAlpha, Eight) => Format::Rg8Unorm,
            (Rgb, Eight) => Format::Rgb8Srgb,
            (Rgba, Eight) => Format::Rgba8Srgb,
            (Grayscale, Sixteen) => Format::R16Unorm,
            (GrayscaleAlpha, Sixteen) => Format::Rg16Unorm,
            (Rgb | Rgba, Sixteen) => Format::Rgba16Unorm,
            _ => return Err(Error::InvalidData),
        })
    }

    /// Decodes the next (sub)frame into an image of `fmt` (16-bit samples little endian, RGB16
    /// gets alpha 65535).
    fn next_image(
        r: &mut png::Reader<Cursor<&[u8]>>,
        buf: &mut [u8],
        fmt: Format,
    ) -> Result<Image> {
        let out = r.next_frame(buf).map_err(err)?;
        let mut img = Image::new(fmt, out.width, out.height)?;
        // png 0.18 expands Adam7 passes with the IHDR row stride, also for smaller APNG
        // subframes; progressive rows use the subframe's own line size.
        let line = if r.info().interlaced {
            r.output_line_size(r.info().width)
                .ok_or(Error::InvalidData)?
        } else {
            out.line_size
        };
        let row_bytes = fmt.row_bytes(out.width)?;
        let rgb16 =
            out.color_type == png::ColorType::Rgb && out.bit_depth == png::BitDepth::Sixteen;
        let rows = buf.chunks_exact(line.max(1)).take(out.height as usize);
        let wide = matches!(
            fmt,
            Format::R16Unorm | Format::Rg16Unorm | Format::Rgba16Unorm
        );
        for (src, dst) in rows.zip(img.data.chunks_exact_mut(row_bytes)) {
            if rgb16 {
                let px = src.as_chunks::<6>().0.iter();
                for (s, d) in px.zip(dst.as_chunks_mut::<8>().0.iter_mut()) {
                    *d = [s[1], s[0], s[3], s[2], s[5], s[4], 0xFF, 0xFF];
                }
            } else if wide {
                let px = src.as_chunks::<2>().0.iter();
                for (s, d) in px.zip(dst.as_chunks_mut::<2>().0.iter_mut()) {
                    *d = [s[1], s[0]];
                }
            } else {
                dst.copy_from_slice(src.get(..row_bytes).ok_or(Error::InvalidData)?);
            }
        }
        Ok(img)
    }

    fn frame_buffer(r: &png::Reader<Cursor<&[u8]>>) -> Result<Buffer> {
        let len = r.output_buffer_size().ok_or(Error::OutOfMemory)?;
        crate::buffer::zeroed(len)
    }

    pub(super) fn decode(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
        let mut r = reader(data, opts)?;
        let fmt = out_format(&r)?;
        let mut buf = frame_buffer(&r)?;
        next_image(&mut r, &mut buf, fmt)
    }

    /// Alpha channel of `fmt`: (bytes per pixel, alpha byte offset, 16-bit).
    fn alpha_layout(fmt: Format) -> Option<(usize, usize, bool)> {
        match fmt {
            Format::Rgba8Srgb => Some((4, 3, false)),
            Format::Rg8Unorm => Some((2, 1, false)),
            Format::Rgba16Unorm => Some((8, 6, true)),
            Format::Rg16Unorm => Some((4, 2, true)),
            _ => None,
        }
    }

    /// APNG `OVER` with the specification's integer formula (premultiplied mix, truncating).
    fn over8(s: &[u8], d: &mut [u8], ai: usize) {
        let sa = u32::from(s[ai]);
        if sa == 255 {
            d.copy_from_slice(s);
        } else if sa != 0 {
            let da = u32::from(d[ai]);
            if da == 0 {
                d.copy_from_slice(s);
                return;
            }
            let u = sa * 255;
            let v = (255 - sa) * da;
            let al = u + v;
            for c in 0..ai {
                d[c] = ((u32::from(s[c]) * u + u32::from(d[c]) * v) / al) as u8;
            }
            d[ai] = (al / 255) as u8;
        }
    }

    fn over16(s: &[u8], d: &mut [u8], ai: usize) {
        let get = |b: &[u8], i: usize| u64::from(u16::from_le_bytes([b[i], b[i + 1]]));
        let sa = get(s, ai);
        if sa == 0xFFFF {
            d.copy_from_slice(s);
        } else if sa != 0 {
            let da = get(d, ai);
            if da == 0 {
                d.copy_from_slice(s);
                return;
            }
            let u = sa * 0xFFFF;
            let v = (0xFFFF - sa) * da;
            let al = u + v;
            for c in (0..ai).step_by(2) {
                let x = ((get(s, c) * u + get(d, c) * v) / al) as u16;
                d[c..c + 2].copy_from_slice(&x.to_le_bytes());
            }
            d[ai..ai + 2].copy_from_slice(&((al / 0xFFFF) as u16).to_le_bytes());
        }
    }

    /// Byte range of row `y`, columns `x..x + w` of `canvas`.
    fn span(canvas: &Image, x: u32, y: u32, w: u32) -> Result<core::ops::Range<usize>> {
        if x.checked_add(w).is_none_or(|e| e > canvas.width) || y >= canvas.height {
            return Err(Error::InvalidData);
        }
        let bpp = canvas.format.block_bytes();
        let row = canvas.format.row_bytes(canvas.width)?;
        let start = y as usize * row + x as usize * bpp;
        Ok(start..start + w as usize * bpp)
    }

    fn blend(canvas: &mut Image, fr: &Image, x: u32, y: u32, over: bool) -> Result<()> {
        let fb = fr.format.row_bytes(fr.width)?;
        let alpha = if over { alpha_layout(fr.format) } else { None };
        for (j, src) in fr.data.chunks_exact(fb.max(1)).enumerate() {
            let range = span(canvas, x, y + j as u32, fr.width)?;
            let dst = canvas.data.get_mut(range).ok_or(Error::InvalidData)?;
            match alpha {
                None => dst.copy_from_slice(src),
                Some((bpp, ai, wide)) => {
                    for (s, d) in src.chunks_exact(bpp).zip(dst.chunks_exact_mut(bpp)) {
                        if wide {
                            over16(s, d, ai);
                        } else {
                            over8(s, d, ai);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Copies the `w` x `h` rectangle at (`x`, `y`) of `canvas` out (`save`) or back in.
    fn rect(
        canvas: &mut Image,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        saved: &mut Buffer,
        save: bool,
    ) -> Result<()> {
        let bpp = canvas.format.block_bytes();
        let rb = w as usize * bpp;
        if save {
            saved.clear();
        }
        for j in 0..h {
            let range = span(canvas, x, y + j, w)?;
            if save {
                let src = canvas.data.get(range).ok_or(Error::InvalidData)?;
                saved
                    .extend_from_slice(src)
                    .map_err(|_| Error::OutOfMemory)?;
            } else {
                let o = j as usize * rb;
                let src = saved.get(o..o + rb).ok_or(Error::InvalidData)?;
                canvas
                    .data
                    .get_mut(range)
                    .ok_or(Error::InvalidData)?
                    .copy_from_slice(src);
            }
        }
        Ok(())
    }

    fn clear(canvas: &mut Image, x: u32, y: u32, w: u32, h: u32) -> Result<()> {
        for j in 0..h {
            let range = span(canvas, x, y + j, w)?;
            canvas
                .data
                .get_mut(range)
                .ok_or(Error::InvalidData)?
                .fill(0);
        }
        Ok(())
    }

    pub(super) fn decode_animation(data: &[u8], opts: &DecodeOptions) -> Result<Animation> {
        let mut r = reader(data, opts)?;
        let fmt = out_format(&r)?;
        let mut buf = frame_buffer(&r)?;
        let Some(actl) = r.info().animation_control else {
            return Animation::still(next_image(&mut r, &mut buf, fmt)?);
        };
        let (w, h) = (r.info().width, r.info().height);
        if r.info().frame_control.is_none() {
            // Default image outside the animation: decode and drop it.
            next_image(&mut r, &mut buf, fmt)?;
        }
        let mut canvas = Image::new(fmt, w, h)?;
        let canvas_bytes = canvas.data.len();
        let mut frames = Vector::<Frame, Dynamic>::default();
        let mut saved = Buffer::default();
        let mut total = 0usize;
        for _ in 0..actl.num_frames {
            let sub = next_image(&mut r, &mut buf, fmt)?;
            let fc = r.info().frame_control.ok_or(Error::InvalidData)?;
            if sub.width != fc.width
                || sub.height != fc.height
                || fc.x_offset.checked_add(fc.width).is_none_or(|e| e > w)
                || fc.y_offset.checked_add(fc.height).is_none_or(|e| e > h)
            {
                return Err(Error::InvalidData);
            }
            total = total.saturating_add(canvas_bytes);
            if total > opts.max_bytes {
                return Err(Error::OutOfMemory);
            }
            let (x, y) = (fc.x_offset, fc.y_offset);
            let previous = fc.dispose_op == png::DisposeOp::Previous;
            if previous {
                rect(&mut canvas, x, y, fc.width, fc.height, &mut saved, true)?;
            }
            blend(&mut canvas, &sub, x, y, fc.blend_op == png::BlendOp::Over)?;
            let image = Image::from_surface(&canvas.surface()?)?;
            let den = if fc.delay_den == 0 { 100 } else { fc.delay_den };
            frames
                .push(Frame {
                    image,
                    delay: (u32::from(fc.delay_num), u32::from(den)),
                })
                .map_err(|_| Error::OutOfMemory)?;
            match fc.dispose_op {
                png::DisposeOp::None => {}
                png::DisposeOp::Background => clear(&mut canvas, x, y, fc.width, fc.height)?,
                png::DisposeOp::Previous => {
                    rect(&mut canvas, x, y, fc.width, fc.height, &mut saved, false)?
                }
            }
        }
        if frames.is_empty() {
            return Err(Error::InvalidData);
        }
        Ok(Animation {
            frames,
            loops: actl.num_plays,
        })
    }

    // ---------------------------------------------------------------------------------------
    // Encoder

    /// Pixels ready for the PNG writer.
    struct Prepared {
        color: png::ColorType,
        depth: png::BitDepth,
        /// Big endian samples (or palette indices, one byte each).
        data: Vec<u8>,
        srgb: bool,
    }

    /// Converts `src` to PNG samples: native layouts or lossless widening.
    fn prepare(src: &Surface) -> Result<Prepared> {
        use png::{BitDepth, ColorType as C};
        let fmt = src.format();
        let d = src.data();
        let (color, depth, data): (C, BitDepth, Vec<u8>) = match fmt {
            Format::R8Unorm => (C::Grayscale, BitDepth::Eight, d.to_vec()),
            Format::Rg8Unorm => (C::GrayscaleAlpha, BitDepth::Eight, d.to_vec()),
            Format::Rgb8Unorm | Format::Rgb8Srgb => (C::Rgb, BitDepth::Eight, d.to_vec()),
            Format::Rgba8Unorm | Format::Rgba8Srgb => (C::Rgba, BitDepth::Eight, d.to_vec()),
            Format::Bgra8Unorm | Format::Bgra8Srgb => {
                let mut v = d.to_vec();
                for p in v.as_chunks_mut::<4>().0 {
                    p.swap(0, 2);
                }
                (C::Rgba, BitDepth::Eight, v)
            }
            Format::R16Unorm | Format::Rg16Unorm | Format::Rgba16Unorm => {
                let mut v = d.to_vec();
                for p in v.as_chunks_mut::<2>().0 {
                    p.swap(0, 1);
                }
                let c = match fmt {
                    Format::R16Unorm => C::Grayscale,
                    Format::Rg16Unorm => C::GrayscaleAlpha,
                    _ => C::Rgba,
                };
                (c, BitDepth::Sixteen, v)
            }
            Format::Rgb565 | Format::Bgr565 => {
                let v = crate::convert::convert_to_buffer(src, Format::Rgb8Unorm)?;
                (C::Rgb, BitDepth::Eight, v.to_vec())
            }
            Format::Rgba4444 => {
                let v = crate::convert::convert_to_buffer(src, Format::Rgba8Unorm)?;
                (C::Rgba, BitDepth::Eight, v.to_vec())
            }
            _ => return Err(Error::Unsupported),
        };
        Ok(Prepared {
            color,
            depth,
            data,
            srgb: fmt.is_srgb(),
        })
    }

    /// (samples per pixel with alpha, color type without alpha) for alpha color types.
    fn alpha_info(c: png::ColorType) -> Option<(usize, png::ColorType)> {
        match c {
            png::ColorType::Rgba => Some((4, png::ColorType::Rgb)),
            png::ColorType::GrayscaleAlpha => Some((2, png::ColorType::Grayscale)),
            _ => None,
        }
    }

    /// (pixel bytes, color bytes) of a color type with alpha.
    fn alpha_split(p: &Prepared, n: usize) -> (usize, usize) {
        let s = if p.depth == png::BitDepth::Sixteen {
            2
        } else {
            1
        };
        (n * s, (n - 1) * s)
    }

    /// True when `p` has an alpha channel and every pixel is opaque.
    fn opaque(p: &Prepared) -> bool {
        let Some((n, _)) = alpha_info(p.color) else {
            return false;
        };
        let (px, keep) = alpha_split(p, n);
        p.data
            .chunks_exact(px)
            .all(|q| q.iter().skip(keep).all(|&b| b == 0xFF))
    }

    /// Drops the alpha channel.
    fn strip_alpha(p: &mut Prepared) {
        let Some((n, c)) = alpha_info(p.color) else {
            return;
        };
        let (px, keep) = alpha_split(p, n);
        let mut out = Vec::with_capacity(p.data.len() / px * keep);
        for q in p.data.chunks_exact(px) {
            out.extend_from_slice(q.get(..keep).unwrap_or_default());
        }
        p.data = out;
        p.color = c;
    }

    /// Lossless palette for 8-bit RGB / RGBA frames with at most 256 colors (shared by all
    /// frames): (PLTE, tRNS for RGBA input).
    fn palette(frames: &[Prepared]) -> Option<(Vec<u8>, Option<Vec<u8>>)> {
        let first = frames.first()?;
        let n = match (first.color, first.depth) {
            (png::ColorType::Rgb, png::BitDepth::Eight) => 3,
            (png::ColorType::Rgba, png::BitDepth::Eight) => 4,
            _ => return None,
        };
        let mut colors: Vec<u32> = Vec::new();
        for f in frames {
            for p in f.data.chunks_exact(n) {
                let key = u32::from_le_bytes([p[0], p[1], p[2], if n == 4 { p[3] } else { 255 }]);
                if let Err(pos) = colors.binary_search(&key) {
                    if colors.len() == 256 {
                        return None;
                    }
                    colors.insert(pos, key);
                }
            }
        }
        // Translucent entries first so tRNS stays short.
        colors.sort_by_key(|&c| (c >> 24 == 255, c));
        let plte = colors
            .iter()
            .flat_map(|c| c.to_le_bytes()[..3].to_vec())
            .collect();
        let trns = (n == 4).then(|| {
            let t: Vec<u8> = colors
                .iter()
                .map(|c| (c >> 24) as u8)
                .take_while(|&a| a != 255)
                .collect();
            // Keep one entry even when every color is opaque: the file must decode as RGBA.
            if t.is_empty() { vec![255] } else { t }
        });
        Some((plte, trns))
    }

    fn to_indices(p: &mut Prepared, plte: &[u8], trns: Option<&[u8]>) -> Result<()> {
        let n = if p.color == png::ColorType::Rgba {
            4
        } else {
            3
        };
        let mut keys: Vec<(u32, u8)> = plte
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let a = trns.and_then(|t| t.get(i).copied()).unwrap_or(255);
                (u32::from_le_bytes([c[0], c[1], c[2], a]), i as u8)
            })
            .collect();
        keys.sort_unstable();
        let mut out = Vec::with_capacity(p.data.len() / n);
        for q in p.data.chunks_exact(n) {
            let key = u32::from_le_bytes([q[0], q[1], q[2], if n == 4 { q[3] } else { 255 }]);
            let i = keys
                .binary_search_by_key(&key, |k| k.0)
                .map_err(|_| Error::InvalidArgument)?;
            out.push(keys.get(i).map_or(0, |k| k.1));
        }
        p.data = out;
        p.color = png::ColorType::Indexed;
        Ok(())
    }

    fn compression(c: super::Compression) -> png::DeflateCompression {
        use super::Compression as K;
        match c {
            K::None => png::DeflateCompression::NoCompression,
            K::Fastest | K::Fast => png::DeflateCompression::FdeflateUltraFast,
            K::Default => png::DeflateCompression::Level(6),
            K::Best => png::DeflateCompression::Level(9),
            K::Level(l) => png::DeflateCompression::Level(l.clamp(1, 9)),
        }
    }

    fn filter(f: super::Filter) -> png::Filter {
        use super::Filter as F;
        match f {
            F::None => png::Filter::NoFilter,
            F::Sub => png::Filter::Sub,
            F::Up => png::Filter::Up,
            F::Average => png::Filter::Avg,
            F::Paeth => png::Filter::Paeth,
            F::Adaptive => png::Filter::Adaptive,
            F::MinEntropy => png::Filter::MinEntropy,
        }
    }

    /// Writes `surfaces` (one: still PNG; several: APNG frames with `delays`).
    pub(super) fn encode(
        surfaces: &[Surface],
        anim: Option<(&[(u32, u32)], u32)>,
        opts: &EncodeOptions,
    ) -> Result<Buffer> {
        let first = surfaces.first().ok_or(Error::InvalidArgument)?;
        let (w, h) = (first.width(), first.height());
        let mut frames = Vec::with_capacity(surfaces.len());
        for s in surfaces {
            if s.format() != first.format() || s.width() != w || s.height() != h {
                return Err(Error::InvalidArgument);
            }
            frames.push(prepare(s)?);
        }
        // Strip alpha only when every frame is opaque (one IHDR for all frames).
        if opts.strip_opaque_alpha && frames.iter().all(opaque) {
            for f in &mut frames {
                strip_alpha(f);
            }
        }
        let pal = if opts.palette { palette(&frames) } else { None };
        if let Some((plte, trns)) = &pal {
            for f in &mut frames {
                to_indices(f, plte, trns.as_deref())?;
            }
        }
        let head = frames.first().ok_or(Error::InvalidArgument)?;
        let mut out = Vec::new();
        {
            let mut e = png::Encoder::new(&mut out, w, h);
            e.set_color(head.color);
            e.set_depth(head.depth);
            e.set_deflate_compression(compression(opts.compression));
            e.set_filter(filter(opts.filter));
            if opts.srgb_chunk && head.srgb {
                e.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
            }
            if let Some((plte, trns)) = pal {
                e.set_palette(plte);
                if let Some(t) = trns {
                    e.set_trns(t);
                }
            }
            if let Some((_, plays)) = anim {
                e.set_animated(frames.len() as u32, plays)
                    .map_err(enc_err)?;
                e.set_dispose_op(png::DisposeOp::None).map_err(enc_err)?;
                e.set_blend_op(png::BlendOp::Source).map_err(enc_err)?;
            }
            let mut wr = e.write_header().map_err(enc_err)?;
            for (i, f) in frames.iter().enumerate() {
                if let Some((delays, _)) = anim {
                    let (n, d) = delays.get(i).copied().unwrap_or((0, 100));
                    let (n, d) = super::delay_u16(n, d);
                    wr.set_frame_delay(n, d).map_err(enc_err)?;
                }
                wr.write_image_data(&f.data).map_err(enc_err)?;
            }
            wr.finish().map_err(enc_err)?;
        }
        buffer_from(&out)
    }

    pub(super) fn encode_animation(anim: &Animation, opts: &EncodeOptions) -> Result<Buffer> {
        let mut surfaces = Vec::with_capacity(anim.frames.len());
        let mut delays = Vec::with_capacity(anim.frames.len());
        for f in anim.frames.iter() {
            surfaces.push(f.image.surface()?);
            delays.push(f.delay);
        }
        encode(&surfaces, Some((&delays, anim.loops)), opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_fit_u16() {
        assert_eq!(delay_u16(1, 10), (1, 10));
        assert_eq!(delay_u16(3, 0), (3, 100));
        assert_eq!(delay_u16(200_000, 1_000_000), (1, 5));
        let (n, d) = delay_u16(100_003, 1_000_000);
        assert!(n > 0 && d > 0 && ((n as f64 / d as f64) - 0.100003).abs() < 1e-4);
    }

    #[test]
    fn srgb_detection() {
        let mut c = ColorInfo::default();
        assert!(c.is_srgb());
        c.gamma = Some(100000);
        assert!(!c.is_srgb());
        c.srgb_intent = Some(0);
        assert!(c.is_srgb());
    }
}
