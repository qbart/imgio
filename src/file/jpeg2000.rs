//! JPEG 2000 (decode only), via the pure Rust `hayro-jpeg2000` crate.
//!
//! Needs the `jpeg2000` cargo feature; without it every function returns `Unsupported`.
//! There is no pure Rust encoder, so [`encode`] always returns [`Error::Unsupported`].
//!
//! - Containers: JP2 files and raw J2K codestreams (`.j2k` / `.j2c`).
//! - Coding: reversible 5/3 (lossless) and irreversible 9/7 wavelets, RCT / ICT component
//!   transforms, tiles, quality layers, every progression order, image / tile offsets,
//!   palettes (resolved), sYCC (converted to RGB). [`DecodeOptions::reduce`] skips the
//!   highest resolution levels (each halves the size, rounded up).
//! - Sub-sampled components are upsampled by `hayro-jpeg2000` by sample replication: each
//!   component sample covers its `XRsiz` x `YRsiz` area of the reference grid (nearest
//!   neighbour). A raw codestream with three components and sub-sampled chroma is treated as
//!   YCbCr (sYCC) and converted to RGB, as OpenJPEG does. When every component has the same
//!   sub-sampling the image is decoded at the reduced size instead.
//! - Output: gray `R8Unorm`, gray + alpha `Rg8Unorm`, RGB `Rgb8Srgb`, RGBA `Rgba8Srgb` when
//!   every component has at most 8 bits; otherwise `R16Unorm` / `Rg16Unorm` / `Rgba16Unorm`
//!   (RGB gets alpha 65535). Decoded samples are rounded (ties to even), clamped to
//!   `[0, 2^p - 1]` and scaled to the output depth as `round(v * (2^n - 1) / (2^p - 1))`
//!   (identity for `p == n`; e.g. 12-bit `v` -> `(v * 65535 + 2047) / 4095`). Signed
//!   components come out offset to the unsigned range. Embedded ICC profiles with one or three
//!   channels are ignored (samples returned as gray / RGB); CMYK, CIELab and other colour
//!   spaces are `Unsupported`.

use super::Image;
use crate::{Buffer, Error, Result, Surface};

/// Default [`DecodeOptions::max_pixels`]: 64 Mpixels (8192 x 8192).
pub const DEFAULT_MAX_PIXELS: u64 = 1 << 26;

/// Decoder options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Number of highest resolution levels to skip (0 = full size); each level halves the
    /// width and height. Clamped to the levels the file has.
    pub reduce: u8,
    /// Largest accepted full-resolution image (width x height); larger files return
    /// [`Error::Unsupported`] before any pixel memory is allocated.
    pub max_pixels: u64,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions {
            reduce: 0,
            max_pixels: DEFAULT_MAX_PIXELS,
        }
    }
}

/// Header information of a JPEG 2000 file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Info {
    /// Full-resolution width.
    pub width: u32,
    /// Full-resolution height.
    pub height: u32,
    /// Output channels: 1 (gray), 3 (RGB) or 4 (CMYK / other) colour channels, plus alpha.
    pub channels: u32,
    /// Whether the last channel is alpha.
    pub has_alpha: bool,
    /// Bit depth of the first component.
    pub bit_depth: u8,
    /// `true` for a raw J2K codestream, `false` for a JP2 file.
    pub codestream: bool,
}

/// Encoder options (there is no JPEG 2000 encoder).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EncodeOptions {}

/// Decodes the image at full resolution.
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_with(data, &DecodeOptions::default())
}

/// Always [`Error::Unsupported`]: no pure Rust JPEG 2000 encoder exists.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    let _ = (src, opts);
    Err(Error::Unsupported)
}

#[cfg(not(feature = "jpeg2000"))]
/// Reads the header.
pub fn info(data: &[u8]) -> Result<Info> {
    let _ = data;
    Err(Error::Unsupported)
}

#[cfg(not(feature = "jpeg2000"))]
/// Decodes the image with options.
pub fn decode_with(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    let _ = (data, opts);
    Err(Error::Unsupported)
}

#[cfg(feature = "jpeg2000")]
pub use imp::{decode_with, info};

#[cfg(feature = "jpeg2000")]
mod imp {
    use super::{DecodeOptions, Info};
    use crate::file::Image;
    use crate::{Error, Format, Result};
    use hayro_jpeg2000 as hj;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    const CODESTREAM_MAGIC: &[u8] = &[0xFF, 0x4F, 0xFF, 0x51];

    /// Runs crate code, turning a panic on hostile input into `InvalidData`.
    fn guard<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
        catch_unwind(AssertUnwindSafe(f)).unwrap_or(Err(Error::InvalidData))
    }

    fn map_err(e: hj::DecodeError) -> Error {
        use hj::{DecodeError as D, FormatError as F, MarkerError as M, ValidationError as V};
        match e {
            D::Format(F::Unsupported) | D::Marker(M::Unsupported) => Error::Unsupported,
            D::Validation(V::ImageTooLarge) => Error::Unsupported,
            _ => Error::InvalidData,
        }
    }

    fn settings(target: Option<(u32, u32)>) -> hj::DecodeSettings {
        hj::DecodeSettings {
            resolve_palette_indices: true,
            strict: false,
            target_resolution: target,
        }
    }

    /// Colour channels of a supported colour space.
    fn color_channels(cs: &hj::ColorSpace) -> Result<u32> {
        match cs {
            hj::ColorSpace::Gray => Ok(1),
            hj::ColorSpace::RGB => Ok(3),
            hj::ColorSpace::Icc { num_channels, .. } if matches!(num_channels, 1 | 3) => {
                Ok(u32::from(*num_channels))
            }
            _ => Err(Error::Unsupported),
        }
    }

    fn header(data: &[u8]) -> Result<Info> {
        let img = hj::Image::new(data, &settings(None)).map_err(map_err)?;
        let has_alpha = img.has_alpha();
        Ok(Info {
            width: img.width(),
            height: img.height(),
            channels: u32::from(img.color_space().num_channels()) + u32::from(has_alpha),
            has_alpha,
            bit_depth: img.original_bit_depth(),
            codestream: data.starts_with(CODESTREAM_MAGIC),
        })
    }

    /// Reads the header (dimensions at full resolution, after component sub-sampling).
    pub fn info(data: &[u8]) -> Result<Info> {
        guard(|| header(data))
    }

    /// Decodes the image with options.
    pub fn decode_with(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
        guard(|| decode_inner(data, opts))
    }

    fn decode_inner(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
        let full = header(data)?;
        if full.width == 0 || full.height == 0 {
            return Err(Error::InvalidData);
        }
        if u64::from(full.width) * u64::from(full.height) > opts.max_pixels {
            return Err(Error::Unsupported);
        }
        // hayro skips `floor(log2(width / target))` levels (the minimum over both axes):
        // `width >> r` always gives exactly `r`.
        let target = (opts.reduce > 0).then(|| {
            let r = u32::from(opts.reduce.min(31));
            ((full.width >> r).max(1), (full.height >> r).max(1))
        });
        let img = hj::Image::new(data, &settings(target)).map_err(map_err)?;
        let nc = color_channels(img.color_space())?;
        let has_alpha = img.has_alpha();
        let (w, h) = (img.width(), img.height());
        let pixels = (w as usize)
            .checked_mul(h as usize)
            .ok_or(Error::InvalidData)?;
        if pixels == 0 {
            return Err(Error::InvalidData);
        }

        let mut ctx = hj::DecoderContext::default();
        let decoded = img.decode(&mut ctx).map_err(map_err)?;
        let comps = decoded.components();
        let total = nc as usize + usize::from(has_alpha);
        if comps.len() < total {
            return Err(Error::InvalidData);
        }
        // Colour channels first; alpha is sorted last by the channel definitions.
        let mut planes = Vec::with_capacity(total);
        planes.extend(comps.iter().take(nc as usize));
        if has_alpha {
            planes.extend(comps.last());
        }
        if planes
            .iter()
            .any(|c| c.samples().len() < pixels || !(1..=31).contains(&c.bit_depth()))
        {
            return Err(Error::InvalidData);
        }
        let wide = planes.iter().any(|c| c.bit_depth() > 8);
        let format = match (total, wide) {
            (1, false) => Format::R8Unorm,
            (2, false) => Format::Rg8Unorm,
            (3, false) => Format::Rgb8Srgb,
            (4, false) => Format::Rgba8Srgb,
            (1, true) => Format::R16Unorm,
            (2, true) => Format::Rg16Unorm,
            (3 | 4, true) => Format::Rgba16Unorm,
            _ => return Err(Error::Unsupported),
        };
        let mut out = Image::new(format, w, h)?;
        let out_ch = if wide && total == 3 { 4 } else { total };
        let out_max: u64 = if wide { 65535 } else { 255 };
        let bytes = if wide { 2 } else { 1 };
        let px_bytes = out_ch * bytes;
        for (ci, plane) in planes.iter().enumerate() {
            let p = u32::from(plane.bit_depth());
            let m = (1u64 << p) - 1;
            let mf = m as f32;
            let samples = plane.samples().get(..pixels).ok_or(Error::InvalidData)?;
            for (px, &s) in out.data.chunks_exact_mut(px_bytes).zip(samples) {
                // NaN casts to 0; `as` saturates.
                let v = (s.round_ties_even().clamp(0.0, mf) as u64).min(m);
                let o = if m == out_max {
                    v
                } else {
                    (v * out_max + m / 2) / m
                };
                if let Some(dst) = px.get_mut(ci * bytes..(ci + 1) * bytes) {
                    if wide {
                        dst.copy_from_slice(&(o as u16).to_le_bytes());
                    } else {
                        dst.fill(o as u8);
                    }
                }
            }
        }
        if out_ch != total {
            for px in out.data.chunks_exact_mut(px_bytes) {
                if let Some(a) = px.get_mut(6..8) {
                    a.copy_from_slice(&[0xFF, 0xFF]);
                }
            }
        }
        Ok(out)
    }
}
