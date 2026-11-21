//! JPEG XL: decode (codestream and ISOBMFF container, VarDCT and modular, 8/16-bit and float,
//! alpha, animation) with `jxl-oxide`, lossless encode with `zune-jpegxl`.
//!
//! Needs the `jxl` cargo feature; without it every function returns `Unsupported`.
//!
//! Decoded pixel formats (by the header's sample type):
//! - integer samples of up to 8 bits: `R8Unorm` (gray), `Rg8Unorm` (gray + alpha), `Rgb8Srgb`,
//!   `Rgba8Srgb`;
//! - integer samples of 9 to 16 bits: `R16Unorm`, `Rg16Unorm`, `Rgba16Unorm` (RGB: alpha 65535);
//!   12-bit and other depths are scaled to the full 16-bit range (`v * 65535 / (2^bits - 1)`,
//!   rounded);
//! - float samples, and integer samples above 16 bits: `R32Float` (gray) or `Rgba32Float`
//!   (gray + alpha replicates gray to RGB; RGB gets alpha 1.0).
//!
//! Color management: integer images are rendered in sRGB (gray: the sRGB transfer curve,
//! D65), float images in linear sRGB (gray: linear). jxl-oxide converts XYB (lossy VarDCT) and
//! enum color encodings (other primaries, PQ / HLG, gamma, linear) natively and embedded ICC
//! profiles with `moxcms` (pure Rust). Lossless files already stored as sRGB (or linear sRGB
//! for float) are returned unchanged. CMYK images are `Unsupported`.
//!
//! The encoder writes lossless modular codestreams (`zune-jpegxl`, always tagged sRGB): 8-bit
//! gray, gray + alpha, RGB and RGBA (`Bgra8*` is swizzled first), 16-bit gray and RGBA.
//! 16-bit gray + alpha (`Rg16Unorm`) is stored as RGBA with R = G = B = gray (zune's 16-bit
//! gray + alpha path is broken), so it decodes as `Rgba16Unorm`. zune-jpegxl declares the
//! alpha of 16-bit RGBA as 8-bit; imgio rewrites the (byte-aligned) image header with a 16-bit
//! alpha channel. Other formats, and images narrower or shorter than 2 pixels, are
//! `Unsupported`.

use super::{Animation, Image};
use crate::{Buffer, Error, Result, Surface};

/// Decoder limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Largest accepted canvas, in pixels (`width * height`). Default `1 << 28`.
    pub max_pixels: u64,
    /// Upper bound of the decoder's internal allocations, in bytes. Default 4 GiB.
    pub max_alloc_bytes: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions {
            max_pixels: 1 << 28,
            max_alloc_bytes: 4 << 30,
        }
    }
}

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    /// Encoder effort, 0..=127 (higher: slower, smaller). Default 4.
    pub effort: u8,
    /// Worker threads (0 = single threaded). Default 4.
    pub threads: u8,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            effort: 4,
            threads: 4,
        }
    }
}

/// Header facts of a JPEG XL file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Info {
    /// Canvas size with orientation applied.
    pub width: u32,
    pub height: u32,
    /// Bits per color sample.
    pub bits_per_sample: u32,
    /// Samples are floating point.
    pub float: bool,
    /// Gray (1 color channel) rather than RGB.
    pub gray: bool,
    /// Has an alpha channel.
    pub alpha: bool,
    /// Lossy XYB (VarDCT or modular XYB) encoding.
    pub xyb: bool,
    /// Has an animation header.
    pub animated: bool,
    /// Wrapped in the ISOBMFF container (`JXL ` signature box) rather than a bare codestream.
    pub container: bool,
}

/// Reads the header.
pub fn info(data: &[u8]) -> Result<Info> {
    imp::info(data, &DecodeOptions::default())
}

/// Decodes the main image (the first displayed frame of an animation).
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_with(data, &DecodeOptions::default())
}

/// Decodes the main image with explicit limits.
pub fn decode_with(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    imp::decode(data, opts)
}

/// Decodes every displayed frame (composited canvases). Delays come from the frame durations
/// and the animation's ticks per second; a still image is one frame with zero delay.
pub fn decode_animation(data: &[u8]) -> Result<Animation> {
    decode_animation_with(data, &DecodeOptions::default())
}

/// [`decode_animation`] with explicit limits.
pub fn decode_animation_with(data: &[u8], opts: &DecodeOptions) -> Result<Animation> {
    imp::decode_animation(data, opts)
}

/// Encodes `src` losslessly.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    imp::encode(src, opts)
}

#[cfg(not(feature = "jxl"))]
mod imp {
    use super::*;

    pub fn info(_: &[u8], _: &DecodeOptions) -> Result<Info> {
        Err(Error::Unsupported)
    }
    pub fn decode(_: &[u8], _: &DecodeOptions) -> Result<Image> {
        Err(Error::Unsupported)
    }
    pub fn decode_animation(_: &[u8], _: &DecodeOptions) -> Result<Animation> {
        Err(Error::Unsupported)
    }
    pub fn encode(_: &Surface, _: &EncodeOptions) -> Result<Buffer> {
        Err(Error::Unsupported)
    }
}

#[cfg(feature = "jxl")]
mod imp {
    use super::super::Frame;
    use super::*;
    use crate::Format;
    use jxl_oxide::color::{ColourSpace, TransferFunction};
    use jxl_oxide::image::BitDepth;
    use jxl_oxide::{
        AllocTracker, EnumColourEncoding, FrameBufferSample, InitializeResult, JxlImage,
        RenderingIntent,
    };
    use vas::Vector;
    use vas::storage::Dynamic;

    fn bad<E>(_: E) -> Error {
        Error::InvalidData
    }

    /// Parses the whole file. The header is parsed first (from a growing prefix) so oversized
    /// canvases are rejected before any frame data is decoded.
    fn open(data: &[u8], opts: &DecodeOptions) -> Result<JxlImage> {
        let tracker = AllocTracker::with_limit(opts.max_alloc_bytes);
        let mut uninit = JxlImage::builder().alloc_tracker(tracker).build_uninit();
        let (mut pos, mut end, mut chunk) = (0usize, 0usize, 4096usize);
        let mut image = loop {
            end = end.saturating_add(chunk).min(data.len());
            chunk = chunk.saturating_mul(2);
            let n = uninit
                .feed_bytes(data.get(pos..end).ok_or(Error::InvalidData)?)
                .map_err(bad)?;
            pos += n;
            match uninit.try_init().map_err(bad)? {
                InitializeResult::Initialized(image) => break image,
                InitializeResult::NeedMoreData(u) if end < data.len() => uninit = u,
                InitializeResult::NeedMoreData(_) => return Err(Error::InvalidData),
            }
        };
        let (w, h) = (image.width(), image.height());
        if w == 0 || h == 0 || u64::from(w) * u64::from(h) > opts.max_pixels {
            return Err(Error::Unsupported);
        }
        while pos < data.len() {
            let n = image
                .feed_bytes(data.get(pos..).ok_or(Error::InvalidData)?)
                .map_err(bad)?;
            if n == 0 {
                break;
            }
            pos += n;
        }
        image.finalize().map_err(bad)?;
        Ok(image)
    }

    fn header_info(image: &JxlImage, data: &[u8]) -> Info {
        let meta = &image.image_header().metadata;
        let (bits_per_sample, float) = match meta.bit_depth {
            BitDepth::IntegerSample { bits_per_sample } => (bits_per_sample, false),
            BitDepth::FloatSample {
                bits_per_sample, ..
            } => (bits_per_sample, true),
        };
        Info {
            width: image.width(),
            height: image.height(),
            bits_per_sample,
            float,
            gray: meta.grayscale(),
            alpha: meta.ec_info.iter().any(|ec| ec.is_alpha()),
            xyb: meta.xyb_encoded,
            animated: meta.animation.is_some(),
            container: !data.starts_with(&[0xFF, 0x0A]),
        }
    }

    pub fn info(data: &[u8], opts: &DecodeOptions) -> Result<Info> {
        let image = open(data, opts)?;
        Ok(header_info(&image, data))
    }

    /// Sample type of the decoded image.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Depth {
        U8,
        U16,
        F32,
    }

    /// Opens, requests the output color encoding, picks the output format.
    fn prepare(data: &[u8], opts: &DecodeOptions) -> Result<(JxlImage, Depth, Format, usize)> {
        let mut image = open(data, opts)?;
        let info = header_info(&image, data);
        let depth = if info.float || info.bits_per_sample > 16 {
            Depth::F32
        } else if info.bits_per_sample > 8 {
            Depth::U16
        } else {
            Depth::U8
        };
        let ri = RenderingIntent::Relative;
        let enc = match (info.gray, depth == Depth::F32) {
            (false, false) => EnumColourEncoding::srgb(ri),
            (false, true) => EnumColourEncoding::srgb_linear(ri),
            (true, false) => EnumColourEncoding::gray_srgb(ri),
            (true, true) => EnumColourEncoding {
                colour_space: ColourSpace::Grey,
                tf: TransferFunction::Linear,
                ..EnumColourEncoding::gray_srgb(ri)
            },
        };
        image.request_color_encoding(enc);
        let pf = image.pixel_format();
        if pf.has_black() {
            return Err(Error::Unsupported);
        }
        let channels = pf.channels();
        let format = match (depth, channels) {
            (Depth::U8, 1) => Format::R8Unorm,
            (Depth::U8, 2) => Format::Rg8Unorm,
            (Depth::U8, 3) => Format::Rgb8Srgb,
            (Depth::U8, 4) => Format::Rgba8Srgb,
            (Depth::U16, 1) => Format::R16Unorm,
            (Depth::U16, 2) => Format::Rg16Unorm,
            (Depth::U16, 3 | 4) => Format::Rgba16Unorm,
            (Depth::F32, 1) => Format::R32Float,
            (Depth::F32, 2..=4) => Format::Rgba32Float,
            _ => return Err(Error::Unsupported),
        };
        Ok((image, depth, format, channels))
    }

    /// Streams samples (interleaved, `channels` per pixel) to `put(index, sample)`.
    fn stream<S: FrameBufferSample + Copy>(
        render: &jxl_oxide::Render,
        total: usize,
        mut put: impl FnMut(usize, S),
    ) -> Result<()> {
        let mut st = render.stream();
        let mut chunk = [S::default(); 4096];
        let mut done = 0usize;
        while done < total {
            let n = st.write_to_buffer(&mut chunk);
            if n == 0 {
                break;
            }
            for (i, &v) in chunk.iter().take(n).enumerate() {
                put(done + i, v);
            }
            done += n;
        }
        if done == total {
            Ok(())
        } else {
            Err(Error::InvalidData)
        }
    }

    /// Renders keyframe `index` into an [`Image`].
    fn render(
        image: &JxlImage,
        index: usize,
        depth: Depth,
        format: Format,
        channels: usize,
    ) -> Result<(Image, u32)> {
        let render = image.render_frame(index).map_err(bad)?;
        let duration = render.duration();
        let (w, h) = (image.width(), image.height());
        {
            let st = render.stream();
            if st.width() != w || st.height() != h || st.channels() as usize != channels {
                return Err(Error::InvalidData);
            }
        }
        let mut out = Image::new(format, w, h)?;
        let pixels = (w as usize).saturating_mul(h as usize);
        let total = pixels.saturating_mul(channels);
        let oc = format.block_bytes() / sample_bytes(depth);
        // Output channel of source channel `c` (gray + alpha expands to RGBA for floats).
        let map = |c: usize| -> &'static [usize] {
            match (channels == 2 && oc == 4, c) {
                (true, 0) => &[0, 1, 2],
                (true, _) => &[3],
                (false, 0) => &[0],
                (false, 1) => &[1],
                (false, 2) => &[2],
                (false, _) => &[3],
            }
        };
        let data = &mut out.data[..];
        match depth {
            Depth::U8 => stream::<u8>(&render, total, |i, v| {
                if let Some(d) = data.get_mut(i) {
                    *d = v;
                }
            })?,
            Depth::U16 => {
                if channels == 3 {
                    for px in data.as_chunks_mut::<8>().0 {
                        px[6..8].copy_from_slice(&u16::MAX.to_le_bytes());
                    }
                }
                stream::<u16>(&render, total, |i, v| {
                    let (p, c) = (i / channels, i % channels);
                    for &o in map(c) {
                        let at = (p * oc + o) * 2;
                        if let Some(d) = data.get_mut(at..at + 2) {
                            d.copy_from_slice(&v.to_le_bytes());
                        }
                    }
                })?
            }
            Depth::F32 => {
                if oc == 4 && (channels == 1 || channels == 3) {
                    for px in data.as_chunks_mut::<16>().0 {
                        px[12..16].copy_from_slice(&1f32.to_le_bytes());
                    }
                }
                stream::<f32>(&render, total, |i, v| {
                    let (p, c) = (i / channels, i % channels);
                    for &o in map(c) {
                        let at = (p * oc + o) * 4;
                        if let Some(d) = data.get_mut(at..at + 4) {
                            d.copy_from_slice(&v.to_le_bytes());
                        }
                    }
                })?
            }
        }
        Ok((out, duration))
    }

    fn sample_bytes(depth: Depth) -> usize {
        match depth {
            Depth::U8 => 1,
            Depth::U16 => 2,
            Depth::F32 => 4,
        }
    }

    pub fn decode(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
        let (image, depth, format, channels) = prepare(data, opts)?;
        if !image.is_loading_done() || image.num_loaded_keyframes() == 0 {
            return Err(Error::InvalidData);
        }
        Ok(render(&image, 0, depth, format, channels)?.0)
    }

    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    pub fn decode_animation(data: &[u8], opts: &DecodeOptions) -> Result<Animation> {
        let (image, depth, format, channels) = prepare(data, opts)?;
        let n = image.num_loaded_keyframes();
        if !image.is_loading_done() || n == 0 {
            return Err(Error::InvalidData);
        }
        let anim = image.image_header().metadata.animation.as_ref();
        let (tps_num, tps_den, loops) = anim.map_or((0, 1, 0), |a| {
            (a.tps_numerator, a.tps_denominator, a.num_loops)
        });
        let mut frames = Vector::<Frame, Dynamic>::default();
        for i in 0..n {
            let (img, duration) = render(&image, i, depth, format, channels)?;
            // Seconds = duration * tps_den / tps_num.
            let delay = if anim.is_none() || tps_num == 0 || duration == 0 {
                (0, 1)
            } else {
                let (mut num, mut den) =
                    (u64::from(duration) * u64::from(tps_den), u64::from(tps_num));
                let g = gcd(num, den).max(1);
                (num, den) = (num / g, den / g);
                while num > u64::from(u32::MAX) || den > u64::from(u32::MAX) {
                    num = (num / 2).max(1);
                    den = (den / 2).max(1);
                }
                (num as u32, den as u32)
            };
            frames
                .push(Frame { image: img, delay })
                .map_err(|_| Error::OutOfMemory)?;
        }
        Ok(Animation { frames, loops })
    }

    /// `Write` sink appending to a [`Buffer`].
    struct Sink<'a>(&'a mut Buffer);

    impl std::io::Write for Sink<'_> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .extend_from_slice(buf)
                .map_err(|_| std::io::Error::from(std::io::ErrorKind::OutOfMemory))?;
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
        use zune_core::bit_depth::BitDepth as ZDepth;
        use zune_core::colorspace::ColorSpace as Zcs;
        use zune_core::options::EncoderOptions;

        let (w, h) = (src.width(), src.height());
        if w < 2 || h < 2 {
            return Err(Error::Unsupported);
        }
        let (cs, depth) = match src.format() {
            Format::R8Unorm => (Zcs::Luma, ZDepth::Eight),
            Format::Rg8Unorm => (Zcs::LumaA, ZDepth::Eight),
            Format::Rgb8Unorm | Format::Rgb8Srgb => (Zcs::RGB, ZDepth::Eight),
            Format::Rgba8Unorm | Format::Rgba8Srgb | Format::Bgra8Unorm | Format::Bgra8Srgb => {
                (Zcs::RGBA, ZDepth::Eight)
            }
            Format::R16Unorm => (Zcs::Luma, ZDepth::Sixteen),
            Format::Rg16Unorm | Format::Rgba16Unorm => (Zcs::RGBA, ZDepth::Sixteen),
            _ => return Err(Error::Unsupported),
        };
        // zune-jpegxl reads 16-bit samples in native byte order; BGRA needs a swizzle. Its
        // 16-bit gray + alpha path reads samples as i16 (values >= 32768 break), so 16-bit
        // gray + alpha is stored as RGBA with R = G = B = gray.
        let swizzled;
        let pixels: &[u8] = if matches!(src.format(), Format::Bgra8Unorm | Format::Bgra8Srgb) {
            swizzled = crate::convert::convert_to_buffer(src, Format::Rgba8Unorm)?;
            &swizzled
        } else if src.format() == Format::Rg16Unorm {
            let mut b = crate::buffer::zeroed(src.data().len().saturating_mul(2))?;
            for (d, s) in b
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(src.data().as_chunks::<4>().0)
            {
                let g = u16::from_le_bytes([s[0], s[1]]).to_ne_bytes();
                let a = u16::from_le_bytes([s[2], s[3]]).to_ne_bytes();
                for (i, v) in [g, g, g, a].iter().enumerate() {
                    d[i * 2..i * 2 + 2].copy_from_slice(v);
                }
            }
            swizzled = b;
            &swizzled
        } else if depth == ZDepth::Sixteen && cfg!(target_endian = "big") {
            let mut b = crate::buffer::zeroed(src.data().len())?;
            for (d, s) in b
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(src.data().as_chunks::<2>().0)
            {
                d.copy_from_slice(&u16::from_le_bytes([s[0], s[1]]).to_ne_bytes());
            }
            swizzled = b;
            &swizzled
        } else {
            src.data()
        };
        let zopts = EncoderOptions::new(w as usize, h as usize, cs, depth)
            .set_effort(opts.effort.min(127))
            .set_num_threads(opts.threads);
        let mut out = Buffer::default();
        zune_jpegxl::JxlSimpleEncoder::new(pixels, zopts)
            .encode(Sink(&mut out))
            .map_err(|e| match e {
                zune_jpegxl::JxlEncodeErrors::IoErrors(_) => Error::OutOfMemory,
                _ => Error::InvalidArgument,
            })?;
        if cs == Zcs::RGBA && depth == ZDepth::Sixteen {
            out = fix_alpha_depth(out, w, h)?;
        }
        Ok(out)
    }

    /// LSB-first bit writer (the JPEG XL bit order).
    #[derive(Default)]
    struct Bits {
        bytes: Vector<u8, Dynamic>,
        acc: u64,
        n: u32,
    }

    impl Bits {
        fn put(&mut self, nbits: u32, v: u64) -> Result<()> {
            self.acc |= (v & ((1u64 << nbits) - 1)) << self.n;
            self.n += nbits;
            while self.n >= 8 {
                self.bytes
                    .push(self.acc as u8)
                    .map_err(|_| Error::OutOfMemory)?;
                self.acc >>= 8;
                self.n -= 8;
            }
            Ok(())
        }
        fn pad(&mut self) -> Result<()> {
            if self.n > 0 {
                self.put(8 - self.n, 0)?;
            }
            Ok(())
        }
    }

    /// zune-jpegxl's hand-written 16-bit RGBA image header (signature, size, metadata), with
    /// the alpha extra channel either "all default" (as zune writes it: 8-bit alpha) or
    /// explicitly 16-bit.
    fn zune_header(w: u32, h: u32, alpha16: bool) -> Result<Vector<u8, Dynamic>> {
        let mut b = Bits::default();
        b.put(16, 0x0AFF)?;
        b.put(1, 0)?; // not small
        for (size, ratio) in [(h, true), (w, false)] {
            let s = u64::from(size - 1);
            match s {
                0..512 => (b.put(2, 0)?, b.put(9, s)?),
                512..8192 => (b.put(2, 1)?, b.put(13, s)?),
                8192..262144 => (b.put(2, 2)?, b.put(18, s)?),
                _ => (b.put(2, 3)?, b.put(30, s)?),
            };
            if ratio {
                b.put(3, 0)?;
            }
        }
        b.put(1, 0)?; // metadata not all default
        b.put(1, 0)?; // no extra fields
        b.put(1, 0)?; // integer samples
        b.put(2, 3)?; // bits_per_sample = 1 + u(6)
        b.put(6, 15)?;
        b.put(1, 0)?; // modular_16bit_buffers = false
        b.put(2, 1)?; // one extra channel
        if alpha16 {
            b.put(1, 0)?; // not d_alpha
            b.put(2, 0)?; // type = alpha
            b.put(1, 0)?; // integer samples
            b.put(2, 3)?; // bits_per_sample = 1 + u(6)
            b.put(6, 15)?;
            b.put(2, 0)?; // dim_shift 0
            b.put(2, 0)?; // empty name
            b.put(1, 0)?; // not premultiplied
        } else {
            b.put(1, 1)?; // d_alpha: 8-bit alpha
        }
        b.put(1, 0)?; // not XYB
        b.put(1, 1)?; // sRGB color encoding
        b.put(2, 0)?; // no extensions
        b.put(1, 1)?; // default transform data
        b.pad()?;
        Ok(b.bytes)
    }

    /// zune-jpegxl 0.5 declares the alpha channel of 16-bit images with the default (8-bit)
    /// bit depth although it codes 16-bit samples; rewrite the byte-aligned image header with
    /// a 16-bit alpha channel. The frame data after it is unchanged.
    fn fix_alpha_depth(enc: Buffer, w: u32, h: u32) -> Result<Buffer> {
        let old = zune_header(w, h, false)?;
        if !enc.starts_with(&old) {
            return Err(Error::Unsupported);
        }
        let new = zune_header(w, h, true)?;
        let mut out = Buffer::default();
        out.extend_from_slice(&new)
            .and_then(|_| out.extend_from_slice(enc.get(old.len()..).unwrap_or(&[])))
            .map_err(|_| Error::OutOfMemory)?;
        Ok(out)
    }
}
