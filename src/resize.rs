//! Image resizing and cropping for every uncompressed format.
//!
//! [`resize`] converts the source to f32 channel planes, decodes sRGB to linear light (sRGB
//! formats, or [`ResizeOptions::srgb`]), premultiplies color by alpha (formats with an alpha
//! channel, [`ResizeOptions::premultiply`]) so transparent pixels do not bleed their color,
//! resamples every plane with the [`crate::mip`] resampler (one plane per thread), then undoes
//! the premultiply / sRGB steps and converts back to the source format. Opaque images give the
//! same result as [`crate::mip::resize`].
//!
//! [`Fit`] picks the output size from the requested box; [`crop`] cuts a rectangle.

use crate::file::Image;
use crate::mip::{Filter, MipOptions, Wrap, srgb_decode, srgb_encode};
use crate::{Buffer, Error, Format, Result, Surface};
use vas::Vector;
use vas::storage::Dynamic;

/// How the requested size is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fit {
    /// Exactly `width` x `height` (the aspect ratio may change).
    #[default]
    Exact,
    /// The largest size inside `width` x `height` with the source aspect ratio.
    Contain,
    /// Exactly `width` x `height`: the source is cropped (centered) to that aspect ratio first.
    Cover,
    /// `width` wide, height from the aspect ratio (`height` is ignored).
    Width,
    /// `height` high, width from the aspect ratio (`width` is ignored).
    Height,
}

/// Options of [`resize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResizeOptions {
    pub filter: Filter,
    /// Filter support scale (1.0 = nominal; > 1 blurs, < 1 sharpens).
    pub filter_scale: f32,
    pub wrap: Wrap,
    pub fit: Fit,
    /// Resample RGB in linear light: `None` follows the format (`Format::is_srgb`).
    pub srgb: Option<bool>,
    /// Weight color by alpha while resampling (formats with an alpha channel). Default on.
    pub premultiply: bool,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        ResizeOptions {
            filter: Filter::Lanczos4,
            filter_scale: 1.0,
            wrap: Wrap::Clamp,
            fit: Fit::Exact,
            srgb: None,
            premultiply: true,
        }
    }
}

/// Number of meaningful channels of an uncompressed format (`None` for block formats).
pub(crate) fn channels(format: Format) -> Option<usize> {
    use Format::*;
    if format.is_compressed() {
        return None;
    }
    Some(match format {
        R8Unorm | R8Snorm | R16Unorm | R16Snorm | R16Float | R32Float => 1,
        Rg8Unorm | Rg8Snorm | Rg16Unorm | Rg16Snorm => 2,
        Rgb8Unorm | Rgb8Srgb | Rgb565 | Bgr565 | Rgb16Float | Rgb9e5Float => 3,
        _ => 4,
    })
}

fn scaled(len: u32, num: u32, den: u32) -> u32 {
    ((u64::from(len) * u64::from(num) + u64::from(den) / 2) / u64::from(den.max(1))).max(1) as u32
}

/// The output size of a `src_w` x `src_h` image resized into `width` x `height` with `fit`.
pub fn target_size(
    src_w: u32,
    src_h: u32,
    width: u32,
    height: u32,
    fit: Fit,
) -> Result<(u32, u32)> {
    let need_w = !matches!(fit, Fit::Height);
    let need_h = !matches!(fit, Fit::Width);
    if src_w == 0 || src_h == 0 || (need_w && width == 0) || (need_h && height == 0) {
        return Err(Error::InvalidArgument);
    }
    Ok(match fit {
        Fit::Exact | Fit::Cover => (width, height),
        Fit::Width => (width, scaled(src_h, width, src_w)),
        Fit::Height => (scaled(src_w, height, src_h), height),
        Fit::Contain => {
            // Compare width / src_w with height / src_h without rounding.
            if u64::from(width) * u64::from(src_h) <= u64::from(height) * u64::from(src_w) {
                (width, scaled(src_h, width, src_w).min(height))
            } else {
                (scaled(src_w, height, src_h).min(width), height)
            }
        }
    })
}

/// The centered source rectangle with the aspect ratio of `width` x `height` (for [`Fit::Cover`]).
fn cover_window(sw: u32, sh: u32, width: u32, height: u32) -> (u32, u32, u32, u32) {
    if u64::from(sw) * u64::from(height) > u64::from(sh) * u64::from(width) {
        let cw = scaled(sh, width, height).min(sw);
        ((sw - cw) / 2, 0, cw, sh)
    } else {
        let ch = scaled(sw, height, width).min(sh);
        (0, (sh - ch) / 2, sw, ch)
    }
}

/// Copies the `w` x `h` rectangle at (`x`, `y`) of an uncompressed surface.
pub fn crop(src: &Surface, x: u32, y: u32, w: u32, h: u32) -> Result<Image> {
    let format = src.format();
    if format.is_compressed() {
        return Err(Error::Unsupported);
    }
    if w == 0
        || h == 0
        || x.checked_add(w).is_none_or(|e| e > src.width())
        || y.checked_add(h).is_none_or(|e| e > src.height())
    {
        return Err(Error::InvalidArgument);
    }
    let mut img = Image::new(format, w, h)?;
    let (bpp, src_row, dst_row) = (
        format.block_bytes(),
        format.row_bytes(src.width())?,
        format.row_bytes(w)?,
    );
    for r in 0..h as usize {
        let at = (y as usize + r) * src_row + x as usize * bpp;
        let line = src.data().get(at..at + dst_row).ok_or(Error::InvalidData)?;
        img.data[r * dst_row..(r + 1) * dst_row].copy_from_slice(line);
    }
    Ok(img)
}

fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
}

/// One channel plane being resampled (`R32Float` bytes).
struct Plane {
    src: Buffer,
    dst: Buffer,
}

/// Resizes an uncompressed surface; the result has the source format and the size given by
/// [`ResizeOptions::fit`] (see [`target_size`]).
///
/// Errors: `Unsupported` for block-compressed formats, `InvalidArgument` for zero sizes or
/// sizes beyond the resampler limit.
pub fn resize(src: &Surface, width: u32, height: u32, opts: &ResizeOptions) -> Result<Image> {
    let format = src.format();
    let n = channels(format).ok_or(Error::Unsupported)?;
    let (sw, sh) = (src.width(), src.height());
    let (ow, oh) = target_size(sw, sh, width, height, opts.fit)?;
    let (cx, cy, cw, ch) = if opts.fit == Fit::Cover {
        cover_window(sw, sh, ow, oh)
    } else {
        (0, 0, sw, sh)
    };
    let srgb = opts.srgb.unwrap_or(format.is_srgb());
    let premul = opts.premultiply && n == 4;
    let float = format.is_hdr();

    // Source -> RGBA f32 (cropped for Cover) -> planes.
    let rgba = crate::convert::convert_to_buffer(src, Format::Rgba32Float)?;
    let (cwu, chu) = (cw as usize, ch as usize);
    let mut planes = Vector::<Plane, Dynamic>::default();
    planes.reserve(n).map_err(|_| Error::OutOfMemory)?;
    for _ in 0..n {
        planes
            .push(Plane {
                src: crate::buffer::zeroed(cwu * chu * 4)?,
                dst: crate::buffer::zeroed(ow as usize * oh as usize * 4)?,
            })
            .map_err(|_| Error::OutOfMemory)?;
    }
    for yy in 0..chu {
        for xx in 0..cwu {
            let s = (cy as usize + yy) * sw as usize + cx as usize + xx;
            let mut px = [0.0f32; 4];
            for (c, v) in px.iter_mut().enumerate() {
                *v = f32_at(&rgba, s * 4 + c);
            }
            if srgb {
                for v in px.iter_mut().take(3.min(n)) {
                    *v = srgb_decode(*v);
                }
            }
            if premul {
                let a = px[3];
                for v in px.iter_mut().take(3) {
                    *v *= a;
                }
            }
            let d = (yy * cwu + xx) * 4;
            for (c, p) in planes.iter_mut().enumerate() {
                p.src[d..d + 4].copy_from_slice(&px[c].to_le_bytes());
            }
        }
    }

    let mip = MipOptions {
        filter: opts.filter,
        filter_scale: opts.filter_scale,
        wrap: opts.wrap,
        srgb: Some(false),
    };
    crate::parallel::try_for_each_item(&mut planes, |p| {
        let s = Surface::new(Format::R32Float, cw, ch, &p.src)?;
        crate::mip::resize_into(&s, ow, oh, &mip, &mut p.dst)
    })?;

    // Planes -> RGBA f32 -> source format.
    let count = ow as usize * oh as usize;
    let mut out = crate::buffer::zeroed(count * 16)?;
    for i in 0..count {
        let mut px = [0.0, 0.0, 0.0, 1.0f32];
        for (c, p) in planes.iter().enumerate() {
            px[c] = f32_at(&p.dst, i);
        }
        if premul {
            let a = if float { px[3] } else { px[3].clamp(0.0, 1.0) };
            for v in px.iter_mut().take(3) {
                *v = if a > 0.0 { *v / a } else { 0.0 };
            }
        }
        if srgb {
            for v in px.iter_mut().take(3.min(n)) {
                *v = srgb_encode(*v);
            }
        }
        for (c, v) in px.iter().enumerate() {
            out[i * 16 + c * 4..i * 16 + c * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    let mut img = Image::new(format, ow, oh)?;
    crate::convert::convert(
        &Surface::new(Format::Rgba32Float, ow, oh, &out)?,
        format,
        &mut img.data,
    )?;
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba8(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&f(x, y));
            }
        }
        v
    }

    #[test]
    fn sizes() -> Result<()> {
        assert_eq!(target_size(200, 100, 50, 50, Fit::Contain)?, (50, 25));
        assert_eq!(target_size(100, 200, 50, 50, Fit::Contain)?, (25, 50));
        assert_eq!(target_size(200, 100, 64, 64, Fit::Cover)?, (64, 64));
        assert_eq!(target_size(200, 100, 50, 0, Fit::Width)?, (50, 25));
        assert_eq!(target_size(200, 100, 0, 10, Fit::Height)?, (20, 10));
        assert_eq!(target_size(3, 1000, 10, 10, Fit::Contain)?, (1, 10));
        assert!(target_size(0, 10, 5, 5, Fit::Exact).is_err());
        assert_eq!(cover_window(200, 100, 64, 64), (50, 0, 100, 100));
        assert_eq!(cover_window(100, 200, 64, 32), (0, 75, 100, 50));
        Ok(())
    }

    #[test]
    fn opaque_matches_mip_resize() -> Result<()> {
        let px = rgba8(37, 29, |x, y| {
            [(x * 7) as u8, (y * 9) as u8, ((x + y) * 3) as u8, 255]
        });
        for f in [Format::Rgba8Srgb, Format::Rgba8Unorm] {
            let s = Surface::new(f, 37, 29, &px)?;
            let got = resize(&s, 16, 40, &ResizeOptions::default())?;
            let want = crate::mip::resize(&s, 16, 40, &MipOptions::default())?;
            assert_eq!(&got.data[..], &want[..], "{f:?}");
        }
        Ok(())
    }

    #[test]
    fn premultiplied_alpha_does_not_bleed() -> Result<()> {
        // Transparent red next to opaque green, box-filtered to one pixel.
        let px = [255, 0, 0, 0, 0, 255, 0, 255];
        let s = Surface::new(Format::Rgba8Unorm, 2, 1, &px)?;
        let opts = ResizeOptions {
            filter: Filter::Box,
            ..ResizeOptions::default()
        };
        let img = resize(&s, 1, 1, &opts)?;
        assert_eq!(&img.data[..], &[0, 255, 0, 128]);
        let straight = resize(
            &s,
            1,
            1,
            &ResizeOptions {
                premultiply: false,
                ..opts
            },
        )?;
        assert_eq!(&straight.data[..], &[128, 128, 0, 128]);
        Ok(())
    }

    #[test]
    fn srgb_averages_in_linear_light() -> Result<()> {
        let px = [0, 0, 0, 255, 255, 255, 255, 255];
        let opts = ResizeOptions {
            filter: Filter::Box,
            ..ResizeOptions::default()
        };
        let lin = resize(&Surface::new(Format::Rgba8Srgb, 2, 1, &px)?, 1, 1, &opts)?;
        assert_eq!(&lin.data[..], &[188, 188, 188, 255]);
        let un = resize(&Surface::new(Format::Rgba8Unorm, 2, 1, &px)?, 1, 1, &opts)?;
        assert_eq!(&un.data[..], &[128, 128, 128, 255]);
        Ok(())
    }

    #[test]
    fn every_uncompressed_format_keeps_a_constant_image() -> Result<()> {
        let bytes = [0.25f32, 0.5, 0.75, 1.0].map(f32::to_le_bytes).concat();
        let rgba = Surface::new(Format::Rgba32Float, 1, 1, &bytes)?;
        for f in [
            Format::R8Unorm,
            Format::R8Snorm,
            Format::Rg8Unorm,
            Format::R16Unorm,
            Format::Rg16Snorm,
            Format::Rgba8Unorm,
            Format::Rgba8Srgb,
            Format::Rgb8Srgb,
            Format::Bgra8Srgb,
            Format::Rgb565,
            Format::Bgr565,
            Format::Rgba4444,
            Format::Rgb16Float,
            Format::Rgb9e5Float,
            Format::Rgba16Float,
            Format::Rgba16Unorm,
            Format::Rgba32Float,
            Format::R16Float,
            Format::R32Float,
        ] {
            let one = crate::convert::convert_to_buffer(&rgba, f)?;
            let src: Vec<u8> = (0..9 * 7).flat_map(|_| one.iter().copied()).collect();
            let img = resize(
                &Surface::new(f, 9, 7, &src)?,
                4,
                13,
                &ResizeOptions::default(),
            )?;
            for p in img.data.chunks_exact(one.len()) {
                if matches!(f, Format::R32Float | Format::Rgba32Float) {
                    // f32 resampling sums rounded weights: within an ulp or two.
                    for (a, b) in p.as_chunks::<4>().0.iter().zip(one.as_chunks::<4>().0) {
                        let (a, b) = (f32::from_le_bytes(*a), f32::from_le_bytes(*b));
                        assert!((a - b).abs() <= 1e-6, "{f:?}: {a} vs {b}");
                    }
                } else {
                    assert_eq!(p, &one[..], "{f:?}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn cover_crops_and_crop_copies() -> Result<()> {
        // Left half black, right half white, 4x2 -> cover 1x1 keeps the center (gray).
        let px = rgba8(4, 2, |x, _| if x < 2 { [0, 0, 0, 255] } else { [255; 4] });
        let s = Surface::new(Format::Rgba8Unorm, 4, 2, &px)?;
        let opts = ResizeOptions {
            filter: Filter::Box,
            fit: Fit::Cover,
            ..ResizeOptions::default()
        };
        let img = resize(&s, 1, 1, &opts)?;
        assert_eq!(&img.data[..], &[128, 128, 128, 255]);
        let c = crop(&s, 1, 1, 2, 1)?;
        assert_eq!(&c.data[..], &[0, 0, 0, 255, 255, 255, 255, 255]);
        assert!(crop(&s, 3, 0, 2, 1).is_err());
        let bc1 = Surface::new(Format::Bc1Unorm, 4, 4, &[0; 8])?;
        assert_eq!(resize(&bc1, 2, 2, &opts).err(), Some(Error::Unsupported));
        Ok(())
    }
}
