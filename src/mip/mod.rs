//! Resizing and mipmap generation with the Basis Universal resampler (separable, per channel),
//! following KTX-Software `ktx create --generate-mipmap` (`ImageT::resample`): every level is
//! resampled from the base image, sRGB channels are linearized first (alpha never), unorm
//! results are clamped to [0, 1] and rounded, float results are not clamped.

mod filters;
mod resampler;

use crate::parallel::try_for_each_item;
use crate::{Buffer, Error, Format, Result, Surface, half};
use resampler::{BoundaryOp, Resampler, Status};
use vas::Vector;
use vas::storage::Dynamic;

/// Resampling filter (basisu names in parentheses).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Filter {
    /// `box`
    Box,
    /// `tent` (bilinear)
    Tent,
    /// `bell`
    Bell,
    /// `b-spline`
    BSpline,
    /// `mitchell`
    Mitchell,
    /// `blackman`
    Blackman,
    /// `lanczos3`
    Lanczos3,
    /// `lanczos4` (KTX-Software default)
    #[default]
    Lanczos4,
    /// `lanczos6`
    Lanczos6,
    /// `lanczos12`
    Lanczos12,
    /// `kaiser`
    Kaiser,
    /// `gaussian`
    Gaussian,
    /// `catmullrom`
    CatmullRom,
    /// `quadratic_interp`
    QuadraticInterp,
    /// `quadratic_approx`
    QuadraticApprox,
    /// `quadratic_mix`
    QuadraticMix,
}

impl Filter {
    /// Every filter.
    pub const ALL: [Filter; 16] = {
        use Filter::*;
        [
            Box,
            Tent,
            Bell,
            BSpline,
            Mitchell,
            Blackman,
            Lanczos3,
            Lanczos4,
            Lanczos6,
            Lanczos12,
            Kaiser,
            Gaussian,
            CatmullRom,
            QuadraticInterp,
            QuadraticApprox,
            QuadraticMix,
        ]
    };

    /// The filter with basisu name `name`.
    pub fn from_name(name: &str) -> Option<Filter> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }

    /// The basisu filter name.
    pub fn name(self) -> &'static str {
        use Filter::*;
        match self {
            Box => "box",
            Tent => "tent",
            Bell => "bell",
            BSpline => "b-spline",
            Mitchell => "mitchell",
            Blackman => "blackman",
            Lanczos3 => "lanczos3",
            Lanczos4 => "lanczos4",
            Lanczos6 => "lanczos6",
            Lanczos12 => "lanczos12",
            Kaiser => "kaiser",
            Gaussian => "gaussian",
            CatmullRom => "catmullrom",
            QuadraticInterp => "quadratic_interp",
            QuadraticApprox => "quadratic_approx",
            QuadraticMix => "quadratic_mix",
        }
    }
}

/// Sampling outside the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Wrap {
    /// Repeat the edge pixels (KTX-Software default).
    #[default]
    Clamp,
    /// Tile.
    Wrap,
    /// Mirror.
    Reflect,
}

/// Options of [`resize`] and [`generate_mips`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MipOptions {
    pub filter: Filter,
    /// Filter support scale (1.0 = nominal; > 1 blurs, < 1 sharpens).
    pub filter_scale: f32,
    pub wrap: Wrap,
    /// Resample RGB in linear light: `None` follows the format (`Format::is_srgb`).
    pub srgb: Option<bool>,
}

impl Default for MipOptions {
    fn default() -> Self {
        MipOptions {
            filter: Filter::Lanczos4,
            filter_scale: 1.0,
            wrap: Wrap::Clamp,
            srgb: None,
        }
    }
}

/// How a format's channels are stored.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Comp {
    U8,
    U16,
    F16,
    F32,
}

fn layout(format: Format) -> Option<(Comp, usize)> {
    use Format::*;
    Some(match format {
        R8Unorm => (Comp::U8, 1),
        Rg8Unorm => (Comp::U8, 2),
        Rgba8Unorm | Rgba8Srgb => (Comp::U8, 4),
        R16Unorm => (Comp::U16, 1),
        Rg16Unorm => (Comp::U16, 2),
        Rgba16Unorm => (Comp::U16, 4),
        Rgb16Float => (Comp::F16, 3),
        Rgba16Float => (Comp::F16, 4),
        Rgba32Float => (Comp::F32, 4),
        _ => return None,
    })
}

/// True when [`resize`] / [`generate_mips`] support `format` (8/16-bit unorm and float
/// layouts; convert other formats first).
pub fn can_resample(format: Format) -> bool {
    layout(format).is_some()
}

fn srgb_decode(v: f32) -> f32 {
    let v = if v < 0.04045 {
        v * (1.0f32 / 12.92f32)
    } else {
        ((v + 0.055) * (1.0f32 / 1.055f32)).powf(2.4)
    };
    v.clamp(0.0, 1.0)
}

fn srgb_encode(v: f32) -> f32 {
    if v < 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0f32 / 2.4f32) - 0.055
    }
}

/// Resamples `src` to `width` x `height` into `dst` (same format, tightly packed).
pub fn resize_into(
    src: &Surface,
    width: u32,
    height: u32,
    opts: &MipOptions,
    dst: &mut [u8],
) -> Result<()> {
    let format = src.format();
    let (comp, n) = layout(format).ok_or(Error::Unsupported)?;
    let (sw, sh) = (src.width(), src.height());
    let max = resampler::BASISU_RESAMPLER_MAX_DIMENSION as u32;
    if width == 0 || height == 0 || sw.max(sh) > max || width.max(height) > max {
        return Err(Error::InvalidArgument);
    }
    let len = format.surface_bytes(width, height)?;
    let got = dst.len();
    let dst = dst
        .get_mut(..len)
        .ok_or(Error::BufferTooSmall { needed: len, got })?;
    let hdr = matches!(comp, Comp::F16 | Comp::F32);
    let srgb = opts.srgb.unwrap_or(format.is_srgb());
    let boundary = match opts.wrap {
        Wrap::Clamp => BoundaryOp::Clamp,
        Wrap::Wrap => BoundaryOp::Wrap,
        Wrap::Reflect => BoundaryOp::Reflect,
    };
    let status = |r: &Resampler| match r.status() {
        Status::Okay => Ok(()),
        Status::OutOfMemory => Err(Error::OutOfMemory),
        Status::BadFilterName | Status::ScanBufferFull => Err(Error::InvalidArgument),
    };
    let mut rs = Vector::<Resampler, Dynamic>::default();
    rs.reserve(n).map_err(|_| Error::OutOfMemory)?;
    for i in 0..n {
        let (cx, cy) = match rs.first() {
            Some(r0) if i > 0 => (r0.get_clist_x(), r0.get_clist_y()),
            _ => (None, None),
        };
        let r = Resampler::new(
            sw as i32,
            sh as i32,
            width as i32,
            height as i32,
            boundary,
            0.0,
            if hdr { 0.0 } else { 1.0 },
            Some(opts.filter.name()),
            cx,
            cy,
            opts.filter_scale,
            opts.filter_scale,
            0.0,
            0.0,
        );
        status(&r)?;
        rs.push(r).map_err(|_| Error::OutOfMemory)?;
    }
    let zeros = |len: usize| -> Result<Vector<f32, Dynamic>> {
        let mut v = Vector::<f32, Dynamic>::default();
        v.resize(len, 0.0).map_err(|_| Error::OutOfMemory)?;
        Ok(v)
    };
    let (swu, wu) = (sw as usize, width as usize);
    // One row of source samples / output samples per channel.
    let mut samples = zeros(swu * n)?;
    let mut lines = zeros(wu * n)?;
    let bpp = format.block_bytes();
    let cb = bpp / n;
    let read = |p: &[u8]| -> f32 {
        match comp {
            Comp::U8 => f32::from(p[0]) * (1.0 / 255.0),
            Comp::U16 => f32::from(u16::from_le_bytes([p[0], p[1]])) * (1.0 / 65535.0),
            Comp::F16 => half::to_f32(u16::from_le_bytes([p[0], p[1]])),
            Comp::F32 => f32::from_le_bytes([p[0], p[1], p[2], p[3]]),
        }
    };
    let write = |p: &mut [u8], out: f32| {
        let unorm = |one: f32| {
            if out.is_nan() || out < 0.0 {
                0.0
            } else if out > 1.0 {
                one
            } else {
                out * one + 0.5
            }
        };
        match comp {
            Comp::U8 => p[0] = unorm(255.0) as u8,
            Comp::U16 => p[..2].copy_from_slice(&(unorm(65535.0) as u16).to_le_bytes()),
            Comp::F16 => p[..2].copy_from_slice(&half::from_f32(out).to_le_bytes()),
            Comp::F32 => p[..4].copy_from_slice(&out.to_le_bytes()),
        }
    };
    let data = src.data();
    let src_row = format.row_bytes(sw)?;
    let dst_row = format.row_bytes(width)?;
    let mut target_y = 0usize;
    for sy in 0..sh as usize {
        let line = &data[sy * src_row..(sy + 1) * src_row];
        for (x, p) in line.chunks_exact(bpp).enumerate() {
            for c in 0..n {
                let v = read(&p[c * cb..]);
                samples[c * swu + x] = if srgb && c < 3 { srgb_decode(v) } else { v };
            }
        }
        for (c, r) in rs.iter_mut().enumerate() {
            if !r.put_line(&samples[c * swu..(c + 1) * swu]) {
                status(r)?;
            }
        }
        // Retrieve every finished output line.
        loop {
            let mut have = true;
            for (c, r) in rs.iter_mut().enumerate() {
                match r.get_line() {
                    Some(l) => {
                        let out = &mut lines[c * wu..(c + 1) * wu];
                        let k = out.len().min(l.len());
                        out[..k].copy_from_slice(&l[..k]);
                    }
                    None => have = false,
                }
            }
            if !have {
                break;
            }
            let Some(out) = dst.get_mut(target_y * dst_row..(target_y + 1) * dst_row) else {
                break;
            };
            for (x, p) in out.chunks_exact_mut(bpp).enumerate() {
                for c in 0..n {
                    let lin = lines[c * wu + x];
                    write(
                        &mut p[c * cb..],
                        if srgb && c < 3 { srgb_encode(lin) } else { lin },
                    );
                }
            }
            target_y += 1;
        }
    }
    if target_y != height as usize {
        return Err(Error::InvalidData);
    }
    Ok(())
}

/// Like [`resize_into`] but allocates the output.
pub fn resize(src: &Surface, width: u32, height: u32, opts: &MipOptions) -> Result<Buffer> {
    let mut buf = crate::buffer::zeroed(src.format().surface_bytes(width, height)?)?;
    resize_into(src, width, height, opts, &mut buf)?;
    Ok(buf)
}

/// Generates mip levels 1..`levels` of `src` (level 0), each resampled from `src`, levels in
/// parallel. `levels` = total level count including the base (at most the full chain).
pub fn generate_mips(
    src: &Surface,
    levels: u32,
    opts: &MipOptions,
) -> Result<Vector<Buffer, Dynamic>> {
    let (w, h) = (src.width(), src.height());
    if levels == 0 || levels > crate::image::mip_levels(w, h) {
        return Err(Error::InvalidArgument);
    }
    if !can_resample(src.format()) {
        return Err(Error::Unsupported);
    }
    let mut out = Vector::<(u32, Buffer), Dynamic>::default();
    out.reserve(levels as usize - 1)
        .map_err(|_| Error::OutOfMemory)?;
    for level in 1..levels {
        let (lw, lh) = crate::image::mip_dim(w, h, level);
        let buf = crate::buffer::zeroed(src.format().surface_bytes(lw, lh)?)?;
        out.push((level, buf)).map_err(|_| Error::OutOfMemory)?;
    }
    try_for_each_item(&mut out, |(level, buf)| {
        let (lw, lh) = crate::image::mip_dim(w, h, *level);
        resize_into(src, lw, lh, opts, buf)
    })?;
    let mut levels_out = Vector::<Buffer, Dynamic>::default();
    levels_out
        .reserve(out.len())
        .map_err(|_| Error::OutOfMemory)?;
    while let Some((_, b)) = out.pop() {
        levels_out.push(b).map_err(|_| Error::OutOfMemory)?;
    }
    levels_out.reverse();
    Ok(levels_out)
}
