//! WebP: lossy (VP8), lossless (VP8L), alpha (ALPH, filtered / compressed) and animation.
//!
//! Needs the `webp` cargo feature (crate `image-webp`); without it every function returns
//! `Unsupported`.
//!
//! - Stills decode to `Rgb8Srgb`, or `Rgba8Srgb` when the file has alpha. Lossy images use
//!   libwebp's "fancy" (bilinear) chroma upsampling and YUV -> RGB arithmetic, bit-exact with
//!   libwebp's decoder.
//! - Animations: imgio parses the `ANIM` / `ANMF` chunks itself, decodes each frame with
//!   `image-webp` and composites it like libwebp's `WebPAnimDecoder` (the reference used by
//!   Pillow and browsers): canvas starts transparent black (the `ANIM` background colour is
//!   ignored, as the spec allows), dispose-to-background clears to transparent black, blending
//!   uses libwebp's non-premultiplied integer formula. Frames are `Rgba8Srgb`, delays are
//!   `(milliseconds, 1000)`. The main image of an animated file is its first frame.
//! - Encoding is lossless only (`image-webp` has no lossy encoder): `Rgb8*`, `Rgba8*`,
//!   `Bgra8*` (reordered), gray `R8Unorm` and gray + alpha `Rg8Unorm`.

use super::Image;
use crate::{Buffer, Result, Surface};

/// Encoder options (lossless VP8L).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    /// Use the VP8L predictor transform (smaller files, slower). Default `true`.
    pub predictor_transform: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            predictor_transform: true,
        }
    }
}

/// Largest canvas / frame decoded, in pixels (the format's own limit is 16384 x 16384).
pub const MAX_PIXELS: u64 = 1 << 28;
/// Largest total size of the composited frames returned by [`decode_animation`], in bytes.
pub const MAX_ANIMATION_BYTES: u64 = 1 << 31;

/// Decodes the main image (the first frame of an animation).
pub fn decode(data: &[u8]) -> Result<Image> {
    #[cfg(feature = "webp")]
    {
        imp::decode(data)
    }
    #[cfg(not(feature = "webp"))]
    {
        let _ = data;
        Err(crate::Error::Unsupported)
    }
}

/// Decodes every frame (composited, canvas sized). A still image is one frame.
pub fn decode_animation(data: &[u8]) -> Result<super::Animation> {
    #[cfg(feature = "webp")]
    {
        imp::decode_animation(data)
    }
    #[cfg(not(feature = "webp"))]
    {
        let _ = data;
        Err(crate::Error::Unsupported)
    }
}

/// Encodes `src` losslessly.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    #[cfg(feature = "webp")]
    {
        imp::encode(src, opts)
    }
    #[cfg(not(feature = "webp"))]
    {
        let _ = (src, opts);
        Err(crate::Error::Unsupported)
    }
}

#[cfg(feature = "webp")]
mod imp {
    use super::{MAX_ANIMATION_BYTES, MAX_PIXELS};
    use crate::file::{Animation, Frame, Image};
    use crate::{Buffer, Error, Format, Result, Surface};
    use image_webp::{ColorType, DecodingError, EncoderParams, WebPDecoder, WebPEncoder};
    use std::io::Cursor;
    use vas::Vector;
    use vas::storage::Dynamic;

    /// A RIFF chunk: FourCC and payload.
    #[derive(Clone, Copy)]
    struct Chunk<'a> {
        id: [u8; 4],
        data: &'a [u8],
    }

    /// Splits a chunk list (RIFF body or ANMF payload); a truncated chunk is `InvalidData`.
    fn chunks(data: &[u8]) -> Result<Vector<Chunk<'_>, Dynamic>> {
        let mut list = Vector::default();
        let mut rest = data;
        while rest.len() >= 8 {
            let size = u32::from_le_bytes([rest[4], rest[5], rest[6], rest[7]]) as usize;
            let payload = rest
                .get(8..)
                .and_then(|r| r.get(..size))
                .ok_or(Error::InvalidData)?;
            list.push(Chunk {
                id: [rest[0], rest[1], rest[2], rest[3]],
                data: payload,
            })
            .map_err(|_| Error::OutOfMemory)?;
            rest = rest.get(8 + size + (size & 1)..).unwrap_or(&[]);
        }
        Ok(list)
    }

    fn map_err(e: DecodingError) -> Error {
        match e {
            DecodingError::ImageTooLarge | DecodingError::MemoryLimitExceeded => Error::Unsupported,
            _ => Error::InvalidData,
        }
    }

    /// The RIFF body (after `RIFF <size> WEBP`), clamped to the declared RIFF size.
    fn body(data: &[u8]) -> Result<&[u8]> {
        if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WEBP" {
            return Err(Error::InvalidData);
        }
        let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let end = size.checked_add(8).ok_or(Error::InvalidData)?;
        if size < 4 || end > data.len() {
            return Err(Error::InvalidData);
        }
        Ok(&data[12..end])
    }

    fn u24(b: &[u8]) -> u32 {
        u32::from(b[0]) | (u32::from(b[1]) << 8) | (u32::from(b[2]) << 16)
    }

    fn check_pixels(w: u32, h: u32) -> Result<()> {
        if w == 0 || h == 0 {
            return Err(Error::InvalidData);
        }
        if u64::from(w) * u64::from(h) > MAX_PIXELS {
            return Err(Error::Unsupported);
        }
        Ok(())
    }

    /// Animated files: `(canvas width, height)` from the `VP8X` chunk.
    fn animated_canvas(body: &[u8]) -> Result<Option<(u32, u32)>> {
        let list = chunks(body)?;
        let first = list.first().ok_or(Error::InvalidData)?;
        if &first.id != b"VP8X" {
            return Ok(None);
        }
        let d = first.data;
        if d.len() < 10 {
            return Err(Error::InvalidData);
        }
        if d[0] & 0x02 == 0 {
            return Ok(None);
        }
        Ok(Some((u24(&d[4..7]) + 1, u24(&d[7..10]) + 1)))
    }

    /// Decodes a still file with image-webp.
    fn decode_still(data: &[u8]) -> Result<Image> {
        let mut dec = WebPDecoder::new(Cursor::new(data)).map_err(map_err)?;
        if dec.is_animated() {
            return Err(Error::InvalidData);
        }
        let (w, h) = dec.dimensions();
        check_pixels(w, h)?;
        let fmt = if dec.has_alpha() {
            Format::Rgba8Srgb
        } else {
            Format::Rgb8Srgb
        };
        let mut img = Image::new(fmt, w, h)?;
        if dec.output_buffer_size() != Some(img.data.len()) {
            return Err(Error::InvalidData);
        }
        dec.read_image(&mut img.data).map_err(map_err)?;
        Ok(img)
    }

    pub fn decode(data: &[u8]) -> Result<Image> {
        let b = body(data)?;
        if animated_canvas(b)?.is_some() {
            let mut anim = decode_animation(data)?;
            return anim
                .frames
                .remove(0)
                .map(|f| f.image)
                .ok_or(Error::InvalidData);
        }
        decode_still(data)
    }

    /// One `ANMF` frame.
    struct AnimFrame<'a> {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        duration: u32,
        blend: bool,
        dispose_bg: bool,
        alph: Option<&'a [u8]>,
        /// `VP8 ` or `VP8L` chunk.
        bitstream: Chunk<'a>,
    }

    impl AnimFrame<'_> {
        /// libwebp demux `has_alpha`: an `ALPH` chunk or the VP8L header's alpha bit.
        fn has_alpha(&self) -> bool {
            self.alph.is_some()
                || (&self.bitstream.id == b"VP8L"
                    && self.bitstream.data.get(4).is_some_and(|b| b & 0x10 != 0))
        }
    }

    fn parse_frame(d: &[u8]) -> Result<AnimFrame<'_>> {
        if d.len() < 16 {
            return Err(Error::InvalidData);
        }
        let flags = d[15];
        let mut alph = None;
        let mut bitstream = None;
        for &c in chunks(&d[16..])?.iter() {
            match &c.id {
                b"ALPH" if alph.is_none() && bitstream.is_none() => alph = Some(c.data),
                b"VP8 " | b"VP8L" if bitstream.is_none() => bitstream = Some(c),
                _ => {}
            }
        }
        let bitstream = bitstream.ok_or(Error::InvalidData)?;
        if &bitstream.id == b"VP8L" {
            alph = None;
        }
        Ok(AnimFrame {
            x: u24(&d[0..3]) * 2,
            y: u24(&d[3..6]) * 2,
            w: u24(&d[6..9]) + 1,
            h: u24(&d[9..12]) + 1,
            duration: u24(&d[12..15]),
            blend: flags & 0x02 == 0,
            dispose_bg: flags & 0x01 != 0,
            alph,
            bitstream,
        })
    }

    fn push(out: &mut Buffer, bytes: &[u8]) -> Result<()> {
        out.extend_from_slice(bytes).map_err(|_| Error::OutOfMemory)
    }

    fn push_chunk(out: &mut Buffer, id: &[u8; 4], data: &[u8]) -> Result<()> {
        let len = u32::try_from(data.len()).map_err(|_| Error::InvalidData)?;
        push(out, id)?;
        push(out, &len.to_le_bytes())?;
        push(out, data)?;
        if data.len() & 1 == 1 {
            push(out, &[0])?;
        }
        Ok(())
    }

    /// Decodes one frame's bitstream to RGBA (opaque VP8 frames get alpha 255; VP8L frames
    /// keep their decoded alpha, as libwebp does).
    fn decode_frame(f: &AnimFrame) -> Result<Buffer> {
        // A standalone extended file: VP8X (alpha flag unless plain VP8) + ALPH? + VP8/VP8L.
        let alpha = f.alph.is_some() || &f.bitstream.id == b"VP8L";
        let mut file = Buffer::default();
        file.reserve(f.bitstream.data.len() + f.alph.map_or(0, <[u8]>::len) + 64)
            .map_err(|_| Error::OutOfMemory)?;
        push(&mut file, b"RIFF\0\0\0\0WEBP")?;
        let mut vp8x = [0u8; 10];
        vp8x[0] = if alpha { 0x10 } else { 0 };
        vp8x[4..7].copy_from_slice(&(f.w - 1).to_le_bytes()[..3]);
        vp8x[7..10].copy_from_slice(&(f.h - 1).to_le_bytes()[..3]);
        push_chunk(&mut file, b"VP8X", &vp8x)?;
        if let Some(a) = f.alph {
            push_chunk(&mut file, b"ALPH", a)?;
        }
        push_chunk(&mut file, &f.bitstream.id, f.bitstream.data)?;
        let riff = u32::try_from(file.len() - 8).map_err(|_| Error::InvalidData)?;
        file[4..8].copy_from_slice(&riff.to_le_bytes());

        let img = decode_still(&file)?;
        if img.width != f.w || img.height != f.h {
            return Err(Error::InvalidData);
        }
        if img.format == Format::Rgba8Srgb {
            return Ok(img.data);
        }
        let mut rgba = crate::buffer::zeroed(img.data.len() / 3 * 4)?;
        for (d, p) in rgba
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(img.data.as_chunks::<3>().0)
        {
            d.copy_from_slice(&[p[0], p[1], p[2], 255]);
        }
        Ok(rgba)
    }

    /// libwebp `BlendPixelNonPremult` (anim_decode.c), pixels as RGBA bytes.
    fn blend_pixel(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
        let src_a = u32::from(src[3]);
        if src_a == 0 {
            return dst;
        }
        let dst_a = u32::from(dst[3]);
        let dst_factor_a = (dst_a * (256 - src_a)) >> 8;
        let blend_a = src_a + dst_factor_a;
        let scale = (1u32 << 24) / blend_a;
        let ch = |i: usize| {
            let v = u32::from(src[i]) * src_a + u32::from(dst[i]) * dst_factor_a;
            ((v * scale) >> 24) as u8
        };
        [ch(0), ch(1), ch(2), blend_a as u8]
    }

    /// Blends `curr` row pixels with alpha < 255 against `prev` (libwebp `BlendPixelRowNonPremult`).
    fn blend_row(curr: &mut [u8], prev: &[u8]) {
        for (c, p) in curr
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(prev.as_chunks::<4>().0)
        {
            if c[3] != 255 {
                *c = blend_pixel(*c, *p);
            }
        }
    }

    pub fn decode_animation(data: &[u8]) -> Result<Animation> {
        let b = body(data)?;
        let Some((cw, ch)) = animated_canvas(b)? else {
            return Animation::still(decode_still(data)?);
        };
        check_pixels(cw, ch)?;
        let canvas_bytes = cw as usize * ch as usize * 4;

        let mut loops = None;
        let mut frames_raw: Vector<&[u8], Dynamic> = Vector::default();
        for c in chunks(b)?.iter() {
            match &c.id {
                b"ANIM" => {
                    let d = c.data.get(4..6).ok_or(Error::InvalidData)?;
                    loops = Some(u32::from(u16::from_le_bytes([d[0], d[1]])));
                }
                b"ANMF" => {
                    frames_raw.push(c.data).map_err(|_| Error::OutOfMemory)?;
                }
                _ => {}
            }
        }
        let loops = loops.ok_or(Error::InvalidData)?;
        if frames_raw.is_empty() {
            return Err(Error::InvalidData);
        }
        if frames_raw.len() as u64 * canvas_bytes as u64 > MAX_ANIMATION_BYTES {
            return Err(Error::Unsupported);
        }

        let mut curr = crate::buffer::zeroed(canvas_bytes)?;
        let mut prev_disposed = crate::buffer::zeroed(canvas_bytes)?;
        let mut out = Animation {
            frames: Vector::default(),
            loops,
        };
        let stride = cw as usize * 4;
        let mut prev: Option<(u32, u32, u32, u32, bool)> = None; // x, y, w, h, dispose_bg
        let mut prev_key = false;
        for (i, &raw) in frames_raw.iter().enumerate() {
            let f = parse_frame(raw)?;
            check_pixels(f.w, f.h)?;
            if f.x + f.w > cw || f.y + f.h > ch {
                return Err(Error::InvalidData);
            }
            let full = |w: u32, h: u32| w == cw && h == ch;
            // libwebp IsKeyFrame.
            let key = match prev {
                None => true,
                Some((_, _, pw, ph, pdisp)) => {
                    ((!f.has_alpha() || !f.blend) && full(f.w, f.h))
                        || (pdisp && (full(pw, ph) || prev_key))
                }
            };
            if key {
                curr.fill(0);
            } else {
                curr.copy_from_slice(&prev_disposed);
            }
            let pixels = decode_frame(&f)?;
            let fw = f.w as usize * 4;
            let (fx, fy) = (f.x as usize * 4, f.y as usize);
            for (row, src) in pixels.chunks_exact(fw).enumerate() {
                let o = (fy + row) * stride + fx;
                curr[o..o + fw].copy_from_slice(src);
            }
            if i > 0 && f.blend && !key {
                for row in 0..f.h as usize {
                    let y = f.y as usize + row;
                    let line = y * stride;
                    let (x0, x1) = (f.x as usize, (f.x + f.w) as usize);
                    let mut ranges = [(x0, x1), (0, 0)];
                    if let Some((px, py, pw, ph, true)) = prev {
                        // Blend only outside the previous (disposed) frame rectangle
                        // (libwebp FindBlendRangeAtRow).
                        let (px, py, pw, ph) = (px as usize, py as usize, pw as usize, ph as usize);
                        if y >= py && y < py + ph && x0 < px + pw && x1 > px {
                            ranges = [(x0, px.max(x0)), ((px + pw).min(x1), x1)];
                        }
                    }
                    for (a, z) in ranges {
                        if z > a {
                            let r = line + a * 4..line + z * 4;
                            blend_row(&mut curr[r.clone()], &prev_disposed[r]);
                        }
                    }
                }
            }
            let mut image = Image::new(Format::Rgba8Srgb, cw, ch)?;
            image.data.copy_from_slice(&curr);
            out.frames
                .push(Frame {
                    image,
                    delay: (f.duration, 1000),
                })
                .map_err(|_| Error::OutOfMemory)?;

            prev_disposed.copy_from_slice(&curr);
            if f.dispose_bg {
                for row in 0..f.h as usize {
                    let o = (fy + row) * stride + fx;
                    prev_disposed[o..o + fw].fill(0);
                }
            }
            prev = Some((f.x, f.y, f.w, f.h, f.dispose_bg));
            prev_key = key;
        }
        Ok(out)
    }

    pub fn encode(src: &Surface, opts: &crate::file::webp::EncodeOptions) -> Result<Buffer> {
        let (w, h) = (src.width(), src.height());
        if w > 16384 || h > 16384 {
            return Err(Error::InvalidArgument);
        }
        let converted;
        let (pixels, color): (&[u8], ColorType) = match src.format() {
            Format::Rgb8Srgb | Format::Rgb8Unorm => (src.data(), ColorType::Rgb8),
            Format::Rgba8Srgb | Format::Rgba8Unorm => (src.data(), ColorType::Rgba8),
            Format::R8Unorm => (src.data(), ColorType::L8),
            Format::Rg8Unorm => (src.data(), ColorType::La8),
            Format::Bgra8Srgb | Format::Bgra8Unorm => {
                converted = crate::convert::convert_to_buffer(src, Format::Rgba8Unorm)?;
                (&converted[..], ColorType::Rgba8)
            }
            _ => return Err(Error::Unsupported),
        };
        let mut out = Vec::new();
        let mut enc = WebPEncoder::new(&mut out);
        let mut params = EncoderParams::default();
        params.use_predictor_transform = opts.predictor_transform;
        enc.set_params(params);
        enc.encode(pixels, w, h, color)
            .map_err(|_| Error::InvalidArgument)?;
        let mut buf = Buffer::default();
        buf.extend_from_slice(&out)
            .map_err(|_| Error::OutOfMemory)?;
        Ok(buf)
    }
}
