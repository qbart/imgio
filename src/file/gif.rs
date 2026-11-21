//! GIF, including animation (`gif` crate, `color_quant` NeuQuant; pure Rust).
//!
//! Decode: global and local color tables, interlaced images, transparency, every disposal
//! method. Frames are composited on a canvas of the logical screen size that starts
//! transparent black, the way browsers render GIFs: disposal "background" clears the frame
//! rectangle to transparent (the background color index is ignored), "previous" restores it,
//! "none" / "keep" leave it; frames are clipped to the screen; pixel indices outside the color
//! table are opaque black. Output is `Rgba8Srgb`. Delays are `(centiseconds, 100)`.
//!
//! Loop count: no NETSCAPE2.0 extension = 1 play; loop count 0 = forever (`loops` 0); loop
//! count N = N + 1 plays (the first play plus N repeats, as Chrome and Firefox do; Pillow
//! reports the raw N as `info["loop"]` and leaves it out when there is no extension).
//!
//! Encode: RGBA8 / RGB8 / BGRA8 (unorm or sRGB), gray `R8Unorm` and gray + alpha `Rg8Unorm`;
//! each frame gets its own color table. Alpha < 128 becomes the transparent index (color
//! (0, 0, 0)), other alpha is dropped. A frame with at most 256 colors (255 with
//! transparency) is stored exactly, otherwise it is quantized with NeuQuant, optionally with
//! Floyd-Steinberg dithering. Frames are quantized and LZW compressed in parallel.
//!
//! Needs the `gif` cargo feature; without it every function returns `Unsupported`.

use super::{Animation, Image};
use crate::{Buffer, Result, Surface};

/// Default [`DecodeOptions::max_bytes`]: 1 GiB.
pub const DEFAULT_MAX_BYTES: usize = 1 << 30;

/// Decoder options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Upper bound for the decoded pixels (all frames of an animation together); larger
    /// files fail with `OutOfMemory`.
    pub max_bytes: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions {
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    /// Floyd-Steinberg dithering when a frame has more colors than a color table holds.
    /// Default `false`.
    pub dither: bool,
    /// NeuQuant sampling factor, 1 (best, slowest) ..= 30 (fastest); frames below 64K
    /// pixels per step use a smaller factor (every pixel for small frames). Default 10.
    pub quantizer_speed: u8,
    /// Number of plays (0 = forever, 1 = no loop extension); `None` uses
    /// [`Animation::loops`] for [`encode_animation`] and 1 for [`encode`].
    pub loops: Option<u32>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            dither: false,
            quantizer_speed: 10,
            loops: None,
        }
    }
}

/// Decodes the main image (the first frame, composited on the transparent canvas).
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_with(data, &DecodeOptions::default())
}

/// [`decode`] with options.
pub fn decode_with(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    #[cfg(feature = "gif")]
    {
        // Decoding stops after the first frame: it is the only one.
        let mut anim = imp::decode(data, opts, true)?;
        anim.frames
            .pop()
            .map(|f| f.image)
            .ok_or(crate::Error::InvalidData)
    }
    #[cfg(not(feature = "gif"))]
    {
        let _ = (data, opts);
        Err(crate::Error::Unsupported)
    }
}

/// Decodes every frame (composited, canvas sized).
pub fn decode_animation(data: &[u8]) -> Result<Animation> {
    decode_animation_with(data, &DecodeOptions::default())
}

/// [`decode_animation`] with options.
pub fn decode_animation_with(data: &[u8], opts: &DecodeOptions) -> Result<Animation> {
    #[cfg(feature = "gif")]
    {
        imp::decode(data, opts, false)
    }
    #[cfg(not(feature = "gif"))]
    {
        let _ = (data, opts);
        Err(crate::Error::Unsupported)
    }
}

/// Encodes `src` as a still GIF.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    #[cfg(feature = "gif")]
    {
        imp::encode(&[(*src, (0, 1))], opts.loops.unwrap_or(1), opts)
    }
    #[cfg(not(feature = "gif"))]
    {
        let _ = (src, opts);
        Err(crate::Error::Unsupported)
    }
}

/// Encodes an animated GIF: every frame is a full canvas (same format and size); delays are
/// rounded to centiseconds.
pub fn encode_animation(anim: &Animation, opts: &EncodeOptions) -> Result<Buffer> {
    #[cfg(feature = "gif")]
    {
        let mut frames = Vec::with_capacity(anim.frames.len());
        for f in anim.frames.iter() {
            frames.push((f.image.surface()?, f.delay));
        }
        imp::encode(&frames, opts.loops.unwrap_or(anim.loops), opts)
    }
    #[cfg(not(feature = "gif"))]
    {
        let _ = (anim, opts);
        Err(crate::Error::Unsupported)
    }
}

/// Frame delay in centiseconds (`den` 0 is read as 100).
#[cfg_attr(not(feature = "gif"), allow(dead_code))]
fn centiseconds(num: u32, den: u32) -> u16 {
    let den = u64::from(if den == 0 { 100 } else { den });
    let cs = (u64::from(num) * 100 + den / 2) / den;
    cs.min(0xFFFF) as u16
}

#[cfg(feature = "gif")]
mod imp {
    use super::{DecodeOptions, EncodeOptions};
    use crate::file::{Animation, Frame, Image};
    use crate::{Buffer, Error, Format, Result, Surface};
    use std::borrow::Cow;
    use std::num::NonZeroU64;
    use vas::Vector;
    use vas::storage::Dynamic;

    /// An LZW code (at least 3 bits) expands to at most 4096 pixels.
    const MAX_PIXELS_PER_BYTE: u64 = 4096 * 8 / 3;
    /// Canvas pixels always allowed (screens larger than the frames they show).
    const FREE_CANVAS_PIXELS: u64 = 1 << 22;

    fn err(e: gif::DecodingError) -> Error {
        match e {
            gif::DecodingError::OutOfMemory | gif::DecodingError::MemoryLimit => Error::OutOfMemory,
            _ => Error::InvalidData,
        }
    }

    fn enc_err(e: gif::EncodingError) -> Error {
        match e {
            gif::EncodingError::Io(_) => Error::OutOfMemory,
            _ => Error::InvalidArgument,
        }
    }

    /// Clips the `w` x `h` rectangle at (`x`, `y`) to the canvas: (x0, y0, x1, y1), with
    /// x0 >= x and y0 >= y; empty (x0 == x1 or y0 == y1) when it is outside.
    fn clip(c: &Image, x: u16, y: u16, w: u16, h: u16) -> (usize, usize, usize, usize) {
        let (x, y) = (usize::from(x), usize::from(y));
        let x1 = (x + usize::from(w)).min(c.width as usize).max(x);
        let y1 = (y + usize::from(h)).min(c.height as usize).max(y);
        if x1 == x || y1 == y {
            return (x, y, x, y);
        }
        (x, y, x1, y1)
    }

    pub(super) fn decode(data: &[u8], opts: &DecodeOptions, first_only: bool) -> Result<Animation> {
        let mut o = gif::DecodeOptions::new();
        o.set_color_output(gif::ColorOutput::Indexed);
        let limit = NonZeroU64::new(opts.max_bytes as u64).ok_or(Error::OutOfMemory)?;
        o.set_memory_limit(gif::MemoryLimit::Bytes(limit));
        let mut d = o.read_info(data).map_err(err)?;
        let (w, h) = (u32::from(d.width()), u32::from(d.height()));
        let budget = (data.len() as u64).saturating_mul(MAX_PIXELS_PER_BYTE);
        if w == 0 || h == 0 || u64::from(w) * u64::from(h) > budget.max(FREE_CANVAS_PIXELS) {
            return Err(Error::InvalidData);
        }
        let mut global = [0u8; 768];
        let global_len = d.global_palette().map_or(0, |p| p.len().min(768));
        if let Some(p) = d.global_palette() {
            global[..global_len].copy_from_slice(p.get(..global_len).unwrap_or_default());
        }
        let mut canvas = Image::new(Format::Rgba8Srgb, w, h)?;
        let canvas_bytes = canvas.data.len();
        let row = w as usize * 4;
        let mut frames = Vector::<Frame, Dynamic>::default();
        let mut saved = Buffer::default();
        let mut indices = Buffer::default();
        let mut total = 0usize;
        while let Some(fi) = d.next_frame_info().map_err(err)? {
            let (fx, fy, fw, fh) = (fi.left, fi.top, fi.width, fi.height);
            let (transparent, dispose, delay) = (fi.transparent, fi.dispose, fi.delay);
            let mut local = [0u8; 768];
            let local_len = fi.palette.as_ref().map(|p| p.len().min(768));
            if let (Some(n), Some(p)) = (local_len, &fi.palette) {
                local[..n].copy_from_slice(p.get(..n).unwrap_or_default());
            }
            let pixels = u64::from(fw) * u64::from(fh);
            if pixels > budget {
                return Err(Error::InvalidData);
            }
            total = total.saturating_add(canvas_bytes);
            if total > opts.max_bytes {
                return Err(Error::OutOfMemory);
            }
            indices.clear();
            indices
                .resize(d.buffer_size(), 0)
                .map_err(|_| Error::OutOfMemory)?;
            d.read_into_buffer(&mut indices).map_err(err)?;
            let pal = match local_len {
                Some(n) => &local[..n],
                None => &global[..global_len],
            };
            let (x0, y0, x1, y1) = clip(&canvas, fx, fy, fw, fh);
            if dispose == gif::DisposalMethod::Previous {
                saved.clear();
                for y in y0..y1 {
                    let src = canvas.data.get(y * row + x0 * 4..y * row + x1 * 4);
                    saved
                        .extend_from_slice(src.unwrap_or_default())
                        .map_err(|_| Error::OutOfMemory)?;
                }
            }
            let fw = usize::from(fw);
            for y in y0..y1 {
                let src_row = (y - usize::from(fy)) * fw;
                let dst = canvas
                    .data
                    .get_mut(y * row + x0 * 4..y * row + x1 * 4)
                    .ok_or(Error::InvalidData)?;
                let src = indices
                    .get(src_row + x0 - usize::from(fx)..src_row + x1 - usize::from(fx))
                    .ok_or(Error::InvalidData)?;
                for (&i, px) in src.iter().zip(dst.as_chunks_mut::<4>().0.iter_mut()) {
                    if Some(i) == transparent {
                        continue;
                    }
                    let o = usize::from(i) * 3;
                    *px = match pal.get(o..o + 3) {
                        Some(c) => [c[0], c[1], c[2], 255],
                        None => [0, 0, 0, 255],
                    };
                }
            }
            let image = Image::from_surface(&canvas.surface()?)?;
            frames
                .push(Frame {
                    image,
                    delay: (u32::from(delay), 100),
                })
                .map_err(|_| Error::OutOfMemory)?;
            if first_only {
                break;
            }
            match dispose {
                gif::DisposalMethod::Background => {
                    for y in y0..y1 {
                        if let Some(r) = canvas.data.get_mut(y * row + x0 * 4..y * row + x1 * 4) {
                            r.fill(0);
                        }
                    }
                }
                gif::DisposalMethod::Previous => {
                    let n = (x1 - x0) * 4;
                    for (j, y) in (y0..y1).enumerate() {
                        let src = saved.get(j * n..(j + 1) * n);
                        let dst = canvas.data.get_mut(y * row + x0 * 4..y * row + x1 * 4);
                        if let (Some(s), Some(d)) = (src, dst) {
                            d.copy_from_slice(s);
                        }
                    }
                }
                _ => {}
            }
        }
        if frames.is_empty() {
            return Err(Error::InvalidData);
        }
        let loops = match d.repeat() {
            gif::Repeat::Infinite => 0,
            gif::Repeat::Finite(n) => u32::from(n) + 1,
        };
        Ok(Animation { frames, loops })
    }

    // ---------------------------------------------------------------------------------------
    // Encoder

    /// RGBA8 copy of a supported input surface.
    fn to_rgba(src: &Surface) -> Result<Vec<u8>> {
        let d = src.data();
        let px = src.width() as usize * src.height() as usize;
        let mut out = Vec::with_capacity(px * 4);
        match src.format() {
            Format::Rgba8Unorm | Format::Rgba8Srgb => out.extend_from_slice(d),
            Format::Bgra8Unorm | Format::Bgra8Srgb => {
                for p in d.as_chunks::<4>().0 {
                    out.extend_from_slice(&[p[2], p[1], p[0], p[3]]);
                }
            }
            Format::Rgb8Unorm | Format::Rgb8Srgb => {
                for p in d.as_chunks::<3>().0 {
                    out.extend_from_slice(&[p[0], p[1], p[2], 255]);
                }
            }
            Format::R8Unorm => {
                for &g in d {
                    out.extend_from_slice(&[g, g, g, 255]);
                }
            }
            Format::Rg8Unorm => {
                for p in d.as_chunks::<2>().0 {
                    out.extend_from_slice(&[p[0], p[0], p[0], p[1]]);
                }
            }
            _ => return Err(Error::Unsupported),
        }
        Ok(out)
    }

    fn rgb_key(p: &[u8; 4]) -> u32 {
        u32::from_be_bytes([0, p[0], p[1], p[2]])
    }

    /// One frame: input, then its color table and indices.
    struct Job<'a> {
        src: Surface<'a>,
        delay: u16,
        dispose: gif::DisposalMethod,
        transparent: bool,
        out: Option<gif::Frame<'static>>,
    }

    /// Exact color table when the opaque colors fit, else `None`.
    fn exact_palette(rgba: &[u8], limit: usize) -> Option<Vec<u32>> {
        let mut colors: Vec<u32> = Vec::new();
        for p in rgba.as_chunks::<4>().0.iter().filter(|p| p[3] >= 128) {
            let k = rgb_key(p);
            if let Err(pos) = colors.binary_search(&k) {
                if colors.len() == limit {
                    return None;
                }
                colors.insert(pos, k);
            }
        }
        Some(colors)
    }

    /// Quantizes one frame into `job.out`.
    fn quantize(job: &mut Job, opts: &EncodeOptions) -> Result<()> {
        let rgba = to_rgba(&job.src)?;
        let (w, h) = (job.src.width() as usize, job.src.height() as usize);
        let limit = if job.transparent { 255 } else { 256 };
        let mut palette: Vec<u8>;
        let mut idx = vec![0u8; w * h];
        // The transparent index follows the colors (255 after a NeuQuant table); without
        // transparency no pixel gets it.
        let trns;
        if let Some(colors) = exact_palette(&rgba, limit) {
            trns = colors.len().min(255) as u8;
            palette = colors
                .iter()
                .flat_map(|c| c.to_be_bytes()[1..].to_vec())
                .collect();
            for (p, o) in rgba.as_chunks::<4>().0.iter().zip(idx.iter_mut()) {
                *o = if p[3] < 128 {
                    trns
                } else {
                    colors.binary_search(&rgb_key(p)).map_or(0, |i| i as u8)
                };
            }
        } else {
            let opaque: Vec<u8> = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|p| p[3] >= 128)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect();
            // NeuQuant learns from pixels / factor samples: small frames use every pixel.
            let full = (opaque.len() / 4 / 65536).max(1);
            let speed = usize::from(opts.quantizer_speed.clamp(1, 30)).min(full) as i32;
            let nq = color_quant::NeuQuant::new(speed, limit, &opaque);
            trns = 255;
            palette = nq.color_map_rgb();
            map_pixels(&rgba, w, &palette, &nq, opts.dither, trns, &mut idx);
        }
        let transparent = if job.transparent {
            // Pad the table so the transparent index exists, its color black.
            palette.resize(usize::from(trns) * 3 + 3, 0);
            Some(trns)
        } else {
            None
        };
        let mut f = gif::Frame {
            width: w as u16,
            height: h as u16,
            delay: job.delay,
            dispose: job.dispose,
            transparent,
            palette: Some(palette),
            buffer: Cow::Owned(idx),
            ..gif::Frame::default()
        };
        f.make_lzw_pre_encoded();
        job.out = Some(f);
        Ok(())
    }

    /// Maps pixels to the NeuQuant table, optionally with Floyd-Steinberg error diffusion.
    fn map_pixels(
        rgba: &[u8],
        w: usize,
        palette: &[u8],
        nq: &color_quant::NeuQuant,
        dither: bool,
        trns: u8,
        idx: &mut [u8],
    ) {
        // Error rows (current, next) per channel, with one pixel of padding on each side.
        let mut cur = vec![[0i32; 3]; w + 2];
        let mut next = vec![[0i32; 3]; w + 2];
        for (y, row) in rgba.chunks_exact(w * 4).enumerate() {
            for (x, p) in row.as_chunks::<4>().0.iter().enumerate() {
                let o = y * w + x;
                if p[3] < 128 {
                    if let Some(i) = idx.get_mut(o) {
                        *i = trns;
                    }
                    continue;
                }
                let mut v = [p[0], p[1], p[2]];
                if dither {
                    let e = cur.get(x + 1).copied().unwrap_or_default();
                    for c in 0..3 {
                        v[c] = (i32::from(v[c]) + e[c] / 16).clamp(0, 255) as u8;
                    }
                }
                let i = nq.index_of(&[v[0], v[1], v[2], 255]);
                if let Some(o) = idx.get_mut(o) {
                    *o = i as u8;
                }
                if dither {
                    let q = palette.get(i * 3..i * 3 + 3).unwrap_or(&[0u8, 0, 0][..]);
                    for c in 0..3 {
                        let e = i32::from(v[c]) - i32::from(q[c]);
                        if let Some(r) = cur.get_mut(x + 2) {
                            r[c] += e * 7;
                        }
                        for (dx, k) in [(0, 3), (1, 5), (2, 1)] {
                            if let Some(r) = next.get_mut(x + dx) {
                                r[c] += e * k;
                            }
                        }
                    }
                }
            }
            std::mem::swap(&mut cur, &mut next);
            next.iter_mut().for_each(|e| *e = [0; 3]);
        }
    }

    pub(super) fn encode(
        frames: &[(Surface, (u32, u32))],
        loops: u32,
        opts: &EncodeOptions,
    ) -> Result<Buffer> {
        let (first, _) = frames.first().ok_or(Error::InvalidArgument)?;
        let (w, h) = (first.width(), first.height());
        let (Ok(w16), Ok(h16)) = (u16::try_from(w), u16::try_from(h)) else {
            return Err(Error::InvalidArgument);
        };
        let mut jobs = Vec::with_capacity(frames.len());
        for (s, (num, den)) in frames {
            if s.format() != first.format() || s.width() != w || s.height() != h {
                return Err(Error::InvalidArgument);
            }
            let transparent = match s.format() {
                Format::Rgba8Unorm | Format::Rgba8Srgb | Format::Bgra8Unorm | Format::Bgra8Srgb => {
                    s.data().as_chunks::<4>().0.iter().any(|p| p[3] < 128)
                }
                Format::Rg8Unorm => s.data().as_chunks::<2>().0.iter().any(|p| p[1] < 128),
                _ => false,
            };
            jobs.push(Job {
                src: *s,
                delay: super::centiseconds(*num, *den),
                dispose: gif::DisposalMethod::Keep,
                transparent,
                out: None,
            });
        }
        // Transparent pixels must show the cleared canvas, not the previous frame: with any
        // transparency, every frame clears itself (disposal "background") and has a
        // transparent index (Pillow fills disposed areas with it).
        if jobs.iter().any(|j| j.transparent) {
            for j in &mut jobs {
                j.transparent = true;
                j.dispose = gif::DisposalMethod::Background;
            }
        }
        crate::parallel::try_for_each_item(&mut jobs, |j| quantize(j, opts))?;
        let mut out = Vec::new();
        {
            let mut e = gif::Encoder::new(&mut out, w16, h16, &[]).map_err(enc_err)?;
            let repeat = match loops {
                0 => gif::Repeat::Infinite,
                n => gif::Repeat::Finite(u16::try_from(n - 1).unwrap_or(u16::MAX)),
            };
            e.set_repeat(repeat).map_err(enc_err)?;
            for j in &jobs {
                let f = j.out.as_ref().ok_or(Error::InvalidArgument)?;
                e.write_lzw_pre_encoded_frame(f).map_err(enc_err)?;
            }
            e.into_inner().map_err(enc_err)?;
        }
        let mut b = Buffer::default();
        b.extend_from_slice(&out).map_err(|_| Error::OutOfMemory)?;
        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_round_to_centiseconds() {
        assert_eq!(centiseconds(1, 10), 10);
        assert_eq!(centiseconds(7, 100), 7);
        assert_eq!(centiseconds(3, 0), 3);
        assert_eq!(centiseconds(40, 1000), 4);
        assert_eq!(centiseconds(1000, 1), 0xFFFF);
    }
}
