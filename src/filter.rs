//! Image filters for every uncompressed format.
//!
//! Geometric operations ([`flip_horizontal`], [`flip_vertical`], [`rotate90`], [`rotate180`],
//! [`rotate270`], [`transpose`]) copy pixels, so they are exact for every uncompressed format.
//! The other filters work on an RGBA f32 copy and convert back to the source format (integer
//! formats are clamped and rounded):
//!
//! - [`blur_box`], [`blur_gaussian`]: separable, edge pixels repeated, computed in linear light
//!   (sRGB formats) with premultiplied alpha (formats with alpha), like [`crate::resize`].
//! - [`unsharp_mask`]: `v + amount * (v - blur(v))` on the stored values where the difference
//!   exceeds `threshold` (alpha untouched).
//! - [`adjust`]: exposure (in linear light), then brightness, contrast, gamma and saturation on
//!   the stored (for sRGB: encoded) values.
//! - [`grayscale`]: luminance in linear light, written to R, G and B.
//! - [`swizzle`], [`premultiply_alpha`], [`unpremultiply_alpha`] (sRGB: in linear light).
//! - [`normal_map`]: tangent-space normals from a height map (Sobel), and [`tone_map`]: HDR to
//!   display range; both write a chosen output format.
//!
//! Per-pixel work and every blur pass run in parallel over rows.

use crate::file::Image;
use crate::mip::{srgb_decode, srgb_encode};
use crate::{Buffer, Error, Format, Result, Surface};

/// Bytes of one RGBA f32 pixel.
const PX: usize = 16;

fn check(src: &Surface) -> Result<()> {
    if src.format().is_compressed() {
        Err(Error::Unsupported)
    } else {
        Ok(())
    }
}

/// True when the format stores an alpha channel.
fn has_alpha(format: Format) -> bool {
    use Format::*;
    matches!(
        format,
        Rgba8Unorm
            | Rgba8Srgb
            | Bgra8Unorm
            | Bgra8Srgb
            | Rgba4444
            | Rgba16Float
            | Rgba16Unorm
            | Rgba32Float
    )
}

fn get(b: &[u8], i: usize) -> [f32; 4] {
    core::array::from_fn(|c| {
        let k = i * PX + c * 4;
        f32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]])
    })
}

fn put(b: &mut [u8], i: usize, px: [f32; 4]) {
    for (c, v) in px.iter().enumerate() {
        let k = i * PX + c * 4;
        b[k..k + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// The RGBA f32 copy of `src`.
fn to_work(src: &Surface) -> Result<Buffer> {
    check(src)?;
    crate::convert::convert_to_buffer(src, Format::Rgba32Float)
}

/// Converts RGBA f32 `work` (`w` x `h`) to an image of `format`.
fn from_work(work: &[u8], w: u32, h: u32, format: Format) -> Result<Image> {
    let mut img = Image::new(format, w, h)?;
    let s = Surface::new(Format::Rgba32Float, w, h, work)?;
    crate::convert::convert(&s, format, &mut img.data)?;
    Ok(img)
}

/// Applies `f` to every pixel of RGBA f32 `work`, rows in parallel.
fn map_pixels(work: &mut [u8], w: u32, h: u32, f: impl Fn(&mut [f32; 4]) + Sync) {
    let row = w as usize * PX;
    crate::parallel::for_each_unit(work, row, w as usize * h as usize, |_, line| {
        for i in 0..line.len() / PX {
            let mut px = get(line, i);
            f(&mut px);
            put(line, i, px);
        }
    });
}

fn per_pixel(src: &Surface, f: impl Fn(&mut [f32; 4]) + Sync) -> Result<Image> {
    let (w, h) = (src.width(), src.height());
    let mut work = to_work(src)?;
    map_pixels(&mut work, w, h, f);
    from_work(&work, w, h, src.format())
}

// ---------------------------------------------------------------------------------------------
// Geometry

/// Builds a `w` x `h` image of `src`'s format where pixel (x, y) is `src` pixel `at(x, y)`.
fn remap(src: &Surface, w: u32, h: u32, at: impl Fn(u32, u32) -> (u32, u32)) -> Result<Image> {
    check(src)?;
    let format = src.format();
    let bpp = format.block_bytes();
    let src_row = format.row_bytes(src.width())?;
    let mut img = Image::new(format, w, h)?;
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = at(x, y);
            let s = sy as usize * src_row + sx as usize * bpp;
            let d = (y as usize * w as usize + x as usize) * bpp;
            let p = src.data().get(s..s + bpp).ok_or(Error::InvalidData)?;
            img.data[d..d + bpp].copy_from_slice(p);
        }
    }
    Ok(img)
}

/// Mirrors left to right.
pub fn flip_horizontal(src: &Surface) -> Result<Image> {
    let (w, h) = (src.width(), src.height());
    remap(src, w, h, |x, y| (w - 1 - x, y))
}

/// Mirrors top to bottom.
pub fn flip_vertical(src: &Surface) -> Result<Image> {
    let (w, h) = (src.width(), src.height());
    remap(src, w, h, |x, y| (x, h - 1 - y))
}

/// Rotates 90 degrees clockwise.
pub fn rotate90(src: &Surface) -> Result<Image> {
    let (w, h) = (src.width(), src.height());
    remap(src, h, w, |x, y| (y, h - 1 - x))
}

/// Rotates 180 degrees.
pub fn rotate180(src: &Surface) -> Result<Image> {
    let (w, h) = (src.width(), src.height());
    remap(src, w, h, |x, y| (w - 1 - x, h - 1 - y))
}

/// Rotates 90 degrees counter-clockwise.
pub fn rotate270(src: &Surface) -> Result<Image> {
    let (w, h) = (src.width(), src.height());
    remap(src, h, w, |x, y| (w - 1 - y, x))
}

/// Swaps rows and columns.
pub fn transpose(src: &Surface) -> Result<Image> {
    let (w, h) = (src.width(), src.height());
    remap(src, h, w, |x, y| (y, x))
}

// ---------------------------------------------------------------------------------------------
// Blur

/// Normalized symmetric kernel: weights for offsets `-r..=r`.
fn kernel(weights: impl Iterator<Item = f32>) -> Result<Vec<f32>> {
    let mut k = Vec::new();
    for v in weights {
        k.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
        k.push(v);
    }
    let sum: f32 = k.iter().sum();
    if sum.is_nan() || sum <= 0.0 {
        return Err(Error::InvalidArgument);
    }
    for v in &mut k {
        *v /= sum;
    }
    Ok(k)
}

/// One separable pass over RGBA f32 `src` into `dst` (horizontal or vertical), edges clamped.
fn convolve(src: &[u8], dst: &mut [u8], w: u32, h: u32, k: &[f32], horizontal: bool) {
    let r = (k.len() / 2) as i64;
    let (wi, hi) = (i64::from(w), i64::from(h));
    let row = w as usize * PX;
    crate::parallel::for_each_unit(dst, row, w as usize * h as usize, |y, line| {
        let y = y as i64;
        for x in 0..wi {
            let mut acc = [0.0f32; 4];
            for (j, &wt) in k.iter().enumerate() {
                let o = j as i64 - r;
                let (sx, sy) = if horizontal {
                    ((x + o).clamp(0, wi - 1), y)
                } else {
                    (x, (y + o).clamp(0, hi - 1))
                };
                let p = get(src, (sy * wi + sx) as usize);
                for c in 0..4 {
                    acc[c] += wt * p[c];
                }
            }
            put(line, x as usize, acc);
        }
    });
}

/// Separable blur of the working copy, in linear light and premultiplied as configured.
fn blur_work(src: &Surface, k: &[f32], linear: bool) -> Result<Buffer> {
    let (w, h) = (src.width(), src.height());
    let format = src.format();
    let srgb = linear && format.is_srgb();
    let premul = has_alpha(format);
    let mut a = to_work(src)?;
    map_pixels(&mut a, w, h, |p| {
        if srgb {
            for v in p.iter_mut().take(3) {
                *v = srgb_decode(*v);
            }
        }
        if premul {
            let al = p[3];
            for v in p.iter_mut().take(3) {
                *v *= al;
            }
        }
    });
    let mut b = crate::buffer::zeroed(a.len())?;
    convolve(&a, &mut b, w, h, k, true);
    convolve(&b, &mut a, w, h, k, false);
    let float = format.is_hdr();
    map_pixels(&mut a, w, h, |p| {
        if premul {
            let al = if float { p[3] } else { p[3].clamp(0.0, 1.0) };
            for v in p.iter_mut().take(3) {
                *v = if al > 0.0 { *v / al } else { 0.0 };
            }
        }
        if srgb {
            for v in p.iter_mut().take(3) {
                *v = srgb_encode(*v);
            }
        }
    });
    Ok(a)
}

/// Box blur over a `(2 * radius + 1)` square window. `radius` 0 copies the image.
pub fn blur_box(src: &Surface, radius: u32) -> Result<Image> {
    check(src)?;
    if radius > 4096 {
        return Err(Error::InvalidArgument);
    }
    let k = kernel((0..2 * radius + 1).map(|_| 1.0))?;
    let work = blur_work(src, &k, true)?;
    from_work(&work, src.width(), src.height(), src.format())
}

fn gaussian_kernel(sigma: f32) -> Result<Vec<f32>> {
    if !(sigma > 0.0 && sigma <= 1024.0) {
        return Err(Error::InvalidArgument);
    }
    let r = (sigma * 3.0).ceil() as i32;
    kernel((-r..=r).map(|o| (-(o * o) as f32 / (2.0 * sigma * sigma)).exp()))
}

/// Gaussian blur with standard deviation `sigma` pixels (kernel radius `ceil(3 * sigma)`).
pub fn blur_gaussian(src: &Surface, sigma: f32) -> Result<Image> {
    check(src)?;
    let k = gaussian_kernel(sigma)?;
    let work = blur_work(src, &k, true)?;
    from_work(&work, src.width(), src.height(), src.format())
}

/// Unsharp mask: `v + amount * (v - gaussian(v, sigma))` on RGB where `|v - blur| > threshold`
/// (values in 0..1 units; alpha untouched).
pub fn unsharp_mask(src: &Surface, sigma: f32, amount: f32, threshold: f32) -> Result<Image> {
    check(src)?;
    if !amount.is_finite() || threshold.is_nan() || threshold < 0.0 {
        return Err(Error::InvalidArgument);
    }
    let (w, h) = (src.width(), src.height());
    let k = gaussian_kernel(sigma)?;
    let blurred = blur_work(src, &k, false)?;
    let mut work = to_work(src)?;
    let row = w as usize * PX;
    crate::parallel::for_each_unit(&mut work, row, w as usize * h as usize, |y, line| {
        for x in 0..w as usize {
            let mut p = get(line, x);
            let b = get(&blurred, y * w as usize + x);
            for c in 0..3 {
                let d = p[c] - b[c];
                if d.abs() > threshold {
                    p[c] += amount * d;
                }
            }
            put(line, x, p);
        }
    });
    from_work(&work, w, h, src.format())
}

// ---------------------------------------------------------------------------------------------
// Color

/// Options of [`adjust`]; the default changes nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adjust {
    /// Exposure in stops (linear light: values times `2^exposure`).
    pub exposure: f32,
    /// Added to RGB (0..1 units).
    pub brightness: f32,
    /// RGB scale around 0.5 (1 = unchanged).
    pub contrast: f32,
    /// RGB to the power `1 / gamma` (1 = unchanged).
    pub gamma: f32,
    /// Distance from the luma (0 = gray, 1 = unchanged).
    pub saturation: f32,
}

impl Default for Adjust {
    fn default() -> Self {
        Adjust {
            exposure: 0.0,
            brightness: 0.0,
            contrast: 1.0,
            gamma: 1.0,
            saturation: 1.0,
        }
    }
}

/// Rec. 709 luma weights.
const REC709: [f32; 3] = [0.2126, 0.7152, 0.0722];
/// Rec. 601 luma weights.
const REC601: [f32; 3] = [0.299, 0.587, 0.114];

/// Exposure, brightness, contrast, gamma and saturation (in that order; alpha untouched).
pub fn adjust(src: &Surface, a: &Adjust) -> Result<Image> {
    let ok = [a.exposure, a.brightness, a.contrast, a.saturation]
        .iter()
        .all(|v| v.is_finite())
        && a.gamma > 0.0
        && a.gamma.is_finite();
    if !ok {
        return Err(Error::InvalidArgument);
    }
    let srgb = src.format().is_srgb();
    let scale = a.exposure.exp2();
    per_pixel(src, |p| {
        if a.exposure != 0.0 {
            for v in p.iter_mut().take(3) {
                *v = if srgb {
                    srgb_encode(srgb_decode(*v) * scale)
                } else {
                    *v * scale
                };
            }
        }
        for v in p.iter_mut().take(3) {
            *v = (*v - 0.5) * a.contrast + 0.5 + a.brightness;
            if a.gamma != 1.0 {
                *v = v.max(0.0).powf(1.0 / a.gamma);
            }
        }
        if a.saturation != 1.0 {
            let l = REC709[0] * p[0] + REC709[1] * p[1] + REC709[2] * p[2];
            for v in p.iter_mut().take(3) {
                *v = l + (*v - l) * a.saturation;
            }
        }
    })
}

/// Luma weights of [`grayscale`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Luma {
    /// Rec. 709 / sRGB primaries.
    #[default]
    Rec709,
    /// Rec. 601.
    Rec601,
}

/// Replaces RGB by its luminance (computed in linear light for sRGB formats).
pub fn grayscale(src: &Surface, luma: Luma) -> Result<Image> {
    let wts = match luma {
        Luma::Rec709 => REC709,
        Luma::Rec601 => REC601,
    };
    let srgb = src.format().is_srgb();
    per_pixel(src, |p| {
        let lin = |v: f32| if srgb { srgb_decode(v) } else { v };
        let y = wts[0] * lin(p[0]) + wts[1] * lin(p[1]) + wts[2] * lin(p[2]);
        let y = if srgb { srgb_encode(y) } else { y };
        p[0] = y;
        p[1] = y;
        p[2] = y;
    })
}

/// A source of one output channel in [`swizzle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    R,
    G,
    B,
    A,
    Zero,
    One,
}

/// Rearranges channels: output channel `i` (R, G, B, A) comes from `map[i]`.
pub fn swizzle(src: &Surface, map: [Channel; 4]) -> Result<Image> {
    per_pixel(src, |p| {
        let s = *p;
        for (d, m) in p.iter_mut().zip(map) {
            *d = match m {
                Channel::R => s[0],
                Channel::G => s[1],
                Channel::B => s[2],
                Channel::A => s[3],
                Channel::Zero => 0.0,
                Channel::One => 1.0,
            };
        }
    })
}

/// Multiplies RGB by alpha (in linear light for sRGB formats).
pub fn premultiply_alpha(src: &Surface) -> Result<Image> {
    let srgb = src.format().is_srgb();
    per_pixel(src, |p| {
        let a = p[3];
        for v in p.iter_mut().take(3) {
            *v = if srgb {
                srgb_encode(srgb_decode(*v) * a)
            } else {
                *v * a
            };
        }
    })
}

/// Divides RGB by alpha (alpha 0 gives black; in linear light for sRGB formats).
pub fn unpremultiply_alpha(src: &Surface) -> Result<Image> {
    let srgb = src.format().is_srgb();
    per_pixel(src, |p| {
        let a = p[3];
        for v in p.iter_mut().take(3) {
            let lin = if srgb { srgb_decode(*v) } else { *v };
            let u = if a > 0.0 { lin / a } else { 0.0 };
            *v = if srgb { srgb_encode(u) } else { u };
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Normal maps and tone mapping

/// Options of [`normal_map`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalMapOptions {
    /// Height scale (slope multiplier).
    pub strength: f32,
    /// Green points down (DirectX convention) instead of up (OpenGL).
    pub directx: bool,
    /// Sample across the edges (tiling textures) instead of repeating edge pixels.
    pub wrap: bool,
    /// Output format (normals are stored as `0.5 + 0.5 * n`).
    pub format: Format,
}

impl Default for NormalMapOptions {
    fn default() -> Self {
        NormalMapOptions {
            strength: 1.0,
            directx: false,
            wrap: false,
            format: Format::Rgba8Unorm,
        }
    }
}

/// Tangent-space normal map from a height map (the luminance of RGB formats, channel R
/// otherwise), with a Sobel gradient. Alpha is 1.
pub fn normal_map(src: &Surface, opts: &NormalMapOptions) -> Result<Image> {
    check(src)?;
    if !opts.strength.is_finite() || opts.format.is_compressed() {
        return Err(Error::InvalidArgument);
    }
    let (w, h) = (src.width(), src.height());
    let rgb = !matches!(
        src.format(),
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
    let work = to_work(src)?;
    let (wi, hi) = (i64::from(w), i64::from(h));
    let height = |x: i64, y: i64| -> f32 {
        let (x, y) = if opts.wrap {
            (x.rem_euclid(wi), y.rem_euclid(hi))
        } else {
            (x.clamp(0, wi - 1), y.clamp(0, hi - 1))
        };
        let p = get(&work, (y * wi + x) as usize);
        if rgb {
            REC709[0] * p[0] + REC709[1] * p[1] + REC709[2] * p[2]
        } else {
            p[0]
        }
    };
    let mut out = crate::buffer::zeroed(work.len())?;
    let row = w as usize * PX;
    crate::parallel::for_each_unit(&mut out, row, w as usize * h as usize, |y, line| {
        let y = y as i64;
        for x in 0..wi {
            let hh = |dx: i64, dy: i64| height(x + dx, y + dy);
            let gx = (hh(1, -1) + 2.0 * hh(1, 0) + hh(1, 1))
                - (hh(-1, -1) + 2.0 * hh(-1, 0) + hh(-1, 1));
            // Image rows go down; +Y of the normal points up (OpenGL) unless `directx`.
            let gy = (hh(-1, 1) + 2.0 * hh(0, 1) + hh(1, 1))
                - (hh(-1, -1) + 2.0 * hh(0, -1) + hh(1, -1));
            let (nx, ny) = (-gx * opts.strength, gy * opts.strength);
            let ny = if opts.directx { -ny } else { ny };
            let len = (nx * nx + ny * ny + 1.0).sqrt();
            let n = [nx / len, ny / len, 1.0 / len];
            put(
                line,
                x as usize,
                [0.5 + 0.5 * n[0], 0.5 + 0.5 * n[1], 0.5 + 0.5 * n[2], 1.0],
            );
        }
    });
    from_work(&out, w, h, opts.format)
}

/// A tone mapping curve of [`tone_map`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ToneMap {
    /// Clamp to 0..1.
    Clamp,
    /// `v / (1 + v)`.
    #[default]
    Reinhard,
    /// `v * (1 + v / white^2) / (1 + v)`: `white` maps to 1.
    ReinhardExtended { white: f32 },
    /// Narkowicz's ACES filmic fit.
    Aces,
}

/// Options of [`tone_map`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneMapOptions {
    pub curve: ToneMap,
    /// Exposure in stops applied before the curve.
    pub exposure: f32,
    /// Output format; sRGB formats are encoded with the sRGB curve.
    pub format: Format,
}

impl Default for ToneMapOptions {
    fn default() -> Self {
        ToneMapOptions {
            curve: ToneMap::Reinhard,
            exposure: 0.0,
            format: Format::Rgba8Srgb,
        }
    }
}

/// Maps linear HDR values (per channel, alpha clamped) into display range.
pub fn tone_map(src: &Surface, opts: &ToneMapOptions) -> Result<Image> {
    check(src)?;
    let white_ok = match opts.curve {
        ToneMap::ReinhardExtended { white } => white > 0.0 && white.is_finite(),
        _ => true,
    };
    if !white_ok || !opts.exposure.is_finite() || opts.format.is_compressed() {
        return Err(Error::InvalidArgument);
    }
    let (w, h) = (src.width(), src.height());
    let srgb_in = src.format().is_srgb();
    let scale = opts.exposure.exp2();
    let mut work = to_work(src)?;
    map_pixels(&mut work, w, h, |p| {
        for v in p.iter_mut().take(3) {
            let x = if srgb_in { srgb_decode(*v) } else { *v };
            let x = if x.is_nan() { 0.0 } else { x.max(0.0) * scale };
            let y = match opts.curve {
                ToneMap::Clamp => x,
                ToneMap::Reinhard => x / (1.0 + x),
                ToneMap::ReinhardExtended { white } => x * (1.0 + x / (white * white)) / (1.0 + x),
                ToneMap::Aces => (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
            };
            // `from_work` stores sRGB formats as encoded values: encode here.
            let y = if y.is_finite() {
                y.clamp(0.0, 1.0)
            } else {
                1.0
            };
            *v = if opts.format.is_srgb() {
                srgb_encode(y)
            } else {
                y
            };
        }
        p[3] = p[3].clamp(0.0, 1.0);
    });
    from_work(&work, w, h, opts.format)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(f: Format, w: u32, h: u32, px: &[u8]) -> Result<Surface<'_>> {
        Surface::new(f, w, h, px)
    }

    #[test]
    fn geometry() -> Result<()> {
        // 3x2 RGBA8: pixel value = index.
        let px: Vec<u8> = (0..6u8).flat_map(|i| [i; 4]).collect();
        let s = img(Format::Rgba8Unorm, 3, 2, &px)?;
        let ids = |i: &Image| i.data.iter().step_by(4).copied().collect::<Vec<u8>>();
        assert_eq!(ids(&flip_horizontal(&s)?), [2, 1, 0, 5, 4, 3]);
        assert_eq!(ids(&flip_vertical(&s)?), [3, 4, 5, 0, 1, 2]);
        let r = rotate90(&s)?;
        assert_eq!((r.width, r.height, ids(&r)), (2, 3, vec![3, 0, 4, 1, 5, 2]));
        assert_eq!(ids(&rotate180(&s)?), [5, 4, 3, 2, 1, 0]);
        assert_eq!(ids(&rotate270(&s)?), [2, 5, 1, 4, 0, 3]);
        assert_eq!(ids(&transpose(&s)?), [0, 3, 1, 4, 2, 5]);
        // Four quarter turns give the original back, for a 2-byte format too.
        let p16: Vec<u8> = (0..12u8).collect();
        let s16 = img(Format::R16Unorm, 3, 2, &p16)?;
        let mut cur = Image::from_surface(&s16)?;
        for _ in 0..4 {
            cur = rotate90(&cur.surface()?)?;
        }
        assert_eq!(&cur.data[..], &p16[..]);
        let bc = img(Format::Bc1Unorm, 4, 4, &[0; 8])?;
        assert_eq!(flip_vertical(&bc).err(), Some(Error::Unsupported));
        Ok(())
    }

    #[test]
    fn blurs() -> Result<()> {
        // A constant image stays constant; an impulse spreads and keeps its sum.
        let flat = vec![100u8; 5 * 5 * 4];
        let s = img(Format::Rgba8Unorm, 5, 5, &flat)?;
        assert_eq!(&blur_gaussian(&s, 1.5)?.data[..], &flat[..]);
        assert_eq!(&blur_box(&s, 2)?.data[..], &flat[..]);
        let mut imp = vec![0f32; 9 * 9];
        imp[40] = 81.0;
        let bytes: Vec<u8> = imp.iter().flat_map(|v| v.to_le_bytes()).collect();
        let s = img(Format::R32Float, 9, 9, &bytes)?;
        let b = blur_box(&s, 1)?;
        let vals: Vec<f32> = b
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert!((vals.iter().sum::<f32>() - 81.0).abs() < 1e-3);
        assert!((vals[40] - 9.0).abs() < 1e-5 && (vals[30] - 9.0).abs() < 1e-5 && vals[20] == 0.0);
        // Transparent pixels do not bleed into opaque ones.
        let px = [255, 0, 0, 0, 0, 255, 0, 255];
        let b = blur_box(&img(Format::Rgba8Unorm, 2, 1, &px)?, 1)?;
        assert_eq!(&b.data[4..7], &[0, 255, 0]);
        assert!(blur_gaussian(&s, 0.0).is_err());
        Ok(())
    }

    #[test]
    fn unsharp_and_adjust() -> Result<()> {
        // A step edge gets overshoot on both sides; a flat image is unchanged.
        let px: Vec<u8> = (0..8u32)
            .flat_map(|x| {
                if x < 4 {
                    [64, 64, 64, 255]
                } else {
                    [192, 192, 192, 255]
                }
            })
            .collect();
        let s = img(Format::Rgba8Unorm, 8, 1, &px)?;
        let u = unsharp_mask(&s, 1.0, 1.0, 0.0)?;
        assert!(u.data[3 * 4] < 64 && u.data[4 * 4] > 192);
        assert_eq!(&unsharp_mask(&s, 1.0, 1.0, 0.5)?.data[..], &px[..]);
        assert_eq!(&adjust(&s, &Adjust::default())?.data[..], &px[..]);
        let a = adjust(
            &s,
            &Adjust {
                brightness: 0.1,
                ..Adjust::default()
            },
        )?;
        assert_eq!((a.data[0], a.data[3]), (90, 255));
        let a = adjust(
            &s,
            &Adjust {
                exposure: 1.0,
                ..Adjust::default()
            },
        )?;
        assert_eq!(a.data[0], 128);
        let a = adjust(
            &s,
            &Adjust {
                saturation: 0.0,
                contrast: 0.0,
                ..Adjust::default()
            },
        )?;
        assert_eq!(a.data[0], 128);
        assert!(
            adjust(
                &s,
                &Adjust {
                    gamma: 0.0,
                    ..Adjust::default()
                }
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn channels_and_alpha() -> Result<()> {
        let s = img(Format::Rgba8Unorm, 1, 1, &[255, 0, 0, 128])?;
        assert_eq!(&grayscale(&s, Luma::Rec709)?.data[..], &[54, 54, 54, 128]);
        assert_eq!(&grayscale(&s, Luma::Rec601)?.data[..], &[76, 76, 76, 128]);
        let sw = swizzle(&s, [Channel::A, Channel::Zero, Channel::R, Channel::One])?;
        assert_eq!(&sw.data[..], &[128, 0, 255, 255]);
        let pm = premultiply_alpha(&s)?;
        assert_eq!(&pm.data[..], &[128, 0, 0, 128]);
        assert_eq!(
            &unpremultiply_alpha(&pm.surface()?)?.data[..],
            &[255, 0, 0, 128]
        );
        // sRGB: 50% white premultiplied in linear light is sRGB 188.
        let s = img(Format::Rgba8Srgb, 1, 1, &[255, 255, 255, 128])?;
        assert_eq!(premultiply_alpha(&s)?.data[0], 188);
        Ok(())
    }

    #[test]
    fn normals_and_tone_mapping() -> Result<()> {
        // Flat height -> +Z; a ramp rising to the right tilts the normal to -X.
        let s = img(Format::R8Unorm, 4, 4, &[100; 16])?;
        let n = normal_map(&s, &NormalMapOptions::default())?;
        assert_eq!(&n.data[..4], &[128, 128, 255, 255]);
        let ramp: Vec<u8> = (0..16u8).map(|i| (i % 4) * 40).collect();
        let n = normal_map(
            &img(Format::R8Unorm, 4, 4, &ramp)?,
            &NormalMapOptions::default(),
        )?;
        assert!(n.data[4 + 4 * 4] < 128 && n.data[5 + 4 * 4] == 128);
        // Rows growing downwards: OpenGL green < 128 (normal points down), DirectX > 128.
        let rows: Vec<u8> = (0..16u8).map(|i| (i / 4) * 40).collect();
        let s = img(Format::R8Unorm, 4, 4, &rows)?;
        let gl = normal_map(&s, &NormalMapOptions::default())?;
        let dx = normal_map(
            &s,
            &NormalMapOptions {
                directx: true,
                ..NormalMapOptions::default()
            },
        )?;
        assert!(gl.data[1 + 20] > 128 && dx.data[1 + 20] < 128);
        // Tone mapping: 1.0 -> 0.5 (Reinhard), linear unorm output.
        let hdr: Vec<u8> = [1.0f32, 3.0, 0.0, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let s = img(Format::Rgba32Float, 1, 1, &hdr)?;
        let t = tone_map(
            &s,
            &ToneMapOptions {
                format: Format::Rgba8Unorm,
                ..ToneMapOptions::default()
            },
        )?;
        assert_eq!(&t.data[..], &[128, 191, 0, 255]);
        let t = tone_map(&s, &ToneMapOptions::default())?;
        assert_eq!(t.format, Format::Rgba8Srgb);
        assert_eq!(t.data[0], 188);
        let t = tone_map(
            &s,
            &ToneMapOptions {
                curve: ToneMap::ReinhardExtended { white: 3.0 },
                format: Format::Rgba8Unorm,
                ..ToneMapOptions::default()
            },
        )?;
        assert_eq!(t.data[1], 255);
        Ok(())
    }
}
