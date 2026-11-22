//! Colour management: ICC profiles and conversions between colour spaces.
//!
//! [`Profile`] is an ICC profile: parsed from bytes ([`Profile::from_icc`]: v2 / v4 RGB and
//! gray, matrix / TRC and LUT based (`lut8`, `lut16`, `mAB`, `mBA`), Lab or XYZ connection
//! space), built in ([`Profile::new`], [`ColorSpace`]: sRGB, linear sRGB, Display P3, DCI-P3, Adobe RGB,
//! ProPhoto, Rec. 2020 and Rec. 2100 PQ / HLG, ACEScg, ACES 2065-1, gray gamma), or made from
//! CICP code points ([`Profile::cicp`], as PNG `cICP`, AVIF `nclx`, video streams).
//! [`convert`] transforms an uncompressed surface between two profiles with a rendering
//! intent and writes any uncompressed format; alpha is passed through.
//!
//! The engine is the pure Rust `moxcms` crate (cargo feature `color`), driven through its f64
//! transforms (16-bit tone curve tables, f64 matrices and LUT interpolation, no fixed point and
//! no CPU-specific paths, so the result does not depend on the machine). Values are the stored
//! channel values normalized like [`crate::convert`] (an `*Srgb` format label does not decode
//! anything: the profile says what the numbers mean); float inputs above 1 are extended range.
//! Rows run in parallel. Without the feature every function returns `Unsupported`.
//!
//! Checked against LittleCMS 2.19 (see docs/spec/color.md).

use crate::file::Image;
use crate::{Buffer, Error, Format, Result, Surface};

/// Built-in colour spaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorSpace {
    /// IEC 61966-2-1 sRGB.
    Srgb,
    /// sRGB primaries, linear transfer.
    LinearSrgb,
    /// Display P3 (P3 primaries, D65, sRGB transfer).
    DisplayP3,
    /// DCI-P3 (DCI white, gamma 2.6).
    DciP3,
    /// Adobe RGB (1998).
    AdobeRgb,
    /// ProPhoto RGB (ROMM, D50, gamma 1.8).
    ProPhoto,
    /// Rec. 2020 with the Rec. 709 transfer.
    Rec2020,
    /// Rec. 2100 PQ (SMPTE ST 2084).
    Rec2100Pq,
    /// Rec. 2100 HLG.
    Rec2100Hlg,
    /// ACEScg (AP1, linear).
    AcesCg,
    /// ACES 2065-1 (AP0, linear).
    Aces2065,
    /// Gray with a pure gamma transfer (2.2 typical; 1 = linear).
    Gray(f32),
    /// Gray with the sRGB transfer (D50 relative, like any gray ICC profile).
    GraySrgb,
}

/// A transfer function (for [`Profile::from_primaries`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transfer {
    Linear,
    /// The sRGB curve.
    Srgb,
    /// Pure power with this decoding exponent (PNG gAMA 45455 -> 2.2).
    Gamma(f32),
}

/// ICC rendering intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Intent {
    #[default]
    Perceptual,
    RelativeColorimetric,
    Saturation,
    AbsoluteColorimetric,
}

/// An ICC profile.
#[derive(Debug, Clone)]
pub struct Profile {
    #[cfg(feature = "color")]
    inner: moxcms::ColorProfile,
    /// The built-in space this profile was made from.
    space: Option<ColorSpace>,
}

/// Channels of a profile's data colour space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    Gray,
    Rgb,
    /// Anything else (CMYK, Lab, ...), which [`convert`] does not take.
    Other,
}

#[cfg(feature = "color")]
mod imp {
    use super::*;
    use crate::color_pipeline::{Side, media_white};
    use moxcms::{
        CicpColorPrimaries, CicpProfile, ColorProfile, DataColorSpace, MatrixCoefficients,
        ToneReprCurve, TransferCharacteristics,
    };

    fn cms(_: moxcms::CmsError) -> Error {
        Error::InvalidData
    }

    fn srgb_trc() -> ToneReprCurve {
        ToneReprCurve::Parametric(vec![2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045])
    }

    fn mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        core::array::from_fn(|i| core::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    }

    /// `m x = v` (Cramer).
    fn solve(m: [[f64; 3]; 3], v: [f64; 3]) -> Option<[f64; 3]> {
        let det = |m: [[f64; 3]; 3]| {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        let d = det(m);
        if d.abs() < 1e-12 {
            return None;
        }
        Some(core::array::from_fn(|c| {
            let mut k = m;
            for (r, row) in k.iter_mut().enumerate() {
                row[c] = v[r];
            }
            det(k) / d
        }))
    }

    /// Bradford adaptation from white `w` (XYZ, Y = 1) to D50.
    fn bradford(w: [f64; 3]) -> Option<[[f64; 3]; 3]> {
        const B: [[f64; 3]; 3] = [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ];
        let d50 = crate::color_pipeline::D50;
        let mv = |v: [f64; 3]| -> [f64; 3] {
            core::array::from_fn(|i| (0..3).map(|k| B[i][k] * v[k]).sum())
        };
        let (s, d) = (mv(w), mv(d50));
        let diag: [[f64; 3]; 3] = core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                if i == j && s[i] != 0.0 {
                    d[i] / s[i]
                } else {
                    0.0
                }
            })
        });
        // B^-1 diag B
        let unit =
            |c: usize| -> [f64; 3] { core::array::from_fn(|r| if r == c { 1.0 } else { 0.0 }) };
        let cols = [solve(B, unit(0))?, solve(B, unit(1))?, solve(B, unit(2))?];
        let binv: [[f64; 3]; 3] = core::array::from_fn(|r| core::array::from_fn(|c| cols[c][r]));
        Some(mul(binv, mul(diag, B)))
    }

    impl Profile {
        pub(super) fn parse_icc(data: &[u8]) -> Result<Profile> {
            Ok(Profile {
                inner: ColorProfile::new_from_slice(data).map_err(cms)?,
                space: None,
            })
        }

        pub(super) fn builtin(space: ColorSpace) -> Profile {
            let inner = match space {
                ColorSpace::Srgb => ColorProfile::new_srgb(),
                ColorSpace::LinearSrgb => {
                    let mut p = ColorProfile::new_srgb();
                    let lin = || Some(ToneReprCurve::Lut(Vec::new()));
                    (p.red_trc, p.green_trc, p.blue_trc) = (lin(), lin(), lin());
                    p.cicp = None;
                    p
                }
                ColorSpace::DisplayP3 => ColorProfile::new_display_p3(),
                ColorSpace::DciP3 => ColorProfile::new_dci_p3(),
                ColorSpace::AdobeRgb => ColorProfile::new_adobe_rgb(),
                ColorSpace::ProPhoto => ColorProfile::new_pro_photo_rgb(),
                ColorSpace::Rec2020 => ColorProfile::new_bt2020(),
                ColorSpace::Rec2100Pq => ColorProfile::new_bt2020_pq(),
                ColorSpace::Rec2100Hlg => ColorProfile::new_bt2020_hlg(),
                ColorSpace::AcesCg => ColorProfile::new_aces_cg_linear(),
                ColorSpace::Aces2065 => ColorProfile::new_aces_aces_2065_1_linear(),
                ColorSpace::Gray(g) => ColorProfile::new_gray_with_gamma(g),
                ColorSpace::GraySrgb => {
                    let mut p = ColorProfile::new_gray_with_gamma(2.2);
                    p.gray_trc = Some(srgb_trc());
                    p
                }
            };
            Profile {
                inner,
                space: Some(space),
            }
        }

        pub(super) fn from_cicp(
            primaries: u8,
            transfer: u8,
            matrix: u8,
            full_range: bool,
        ) -> Result<Profile> {
            let cp = CicpColorPrimaries::try_from(primaries).map_err(cms)?;
            let tc = TransferCharacteristics::try_from(transfer).map_err(cms)?;
            let mc = MatrixCoefficients::try_from(matrix).map_err(cms)?;
            let c = CicpProfile {
                color_primaries: cp,
                transfer_characteristics: tc,
                matrix_coefficients: mc,
                full_range,
            };
            let p = ColorProfile::new_from_cicp(c);
            if p.red_trc.is_none() && p.gray_trc.is_none() {
                return Err(Error::Unsupported);
            }
            Ok(Profile {
                inner: p,
                space: None,
            })
        }

        /// RGB profile from chromaticities (red, green, blue, white `x, y`) and a transfer:
        /// colorants adapted to D50 with Bradford, like ICC v4 display profiles.
        pub(super) fn primaries(xy: [[f32; 2]; 4], t: Transfer) -> Result<Profile> {
            let xyz = |p: [f32; 2]| -> Result<[f64; 3]> {
                let (x, y) = (f64::from(p[0]), f64::from(p[1]));
                if y <= 0.0 || !x.is_finite() || !y.is_finite() {
                    return Err(Error::InvalidData);
                }
                Ok([x / y, 1.0, (1.0 - x - y) / y])
            };
            let (r, g, b, w) = (xyz(xy[0])?, xyz(xy[1])?, xyz(xy[2])?, xyz(xy[3])?);
            let m = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
            let s = solve(m, w).ok_or(Error::InvalidData)?;
            let rgb_xyz: [[f64; 3]; 3] =
                core::array::from_fn(|i| core::array::from_fn(|j| m[i][j] * s[j]));
            let a = bradford(w).ok_or(Error::InvalidData)?;
            let c = mul(a, rgb_xyz);
            let mut p = ColorProfile::new_srgb();
            let col = |j: usize| moxcms::Xyzd {
                x: c[0][j],
                y: c[1][j],
                z: c[2][j],
            };
            (p.red_colorant, p.green_colorant, p.blue_colorant) = (col(0), col(1), col(2));
            p.chromatic_adaptation = Some(moxcms::Matrix3d { v: a });
            let trc = match t {
                Transfer::Linear => ToneReprCurve::Lut(Vec::new()),
                Transfer::Srgb => srgb_trc(),
                Transfer::Gamma(g) => ToneReprCurve::Parametric(vec![g]),
            };
            (p.red_trc, p.green_trc, p.blue_trc) =
                (Some(trc.clone()), Some(trc.clone()), Some(trc));
            p.cicp = None;
            p.description = None;
            Ok(Profile {
                inner: p,
                space: None,
            })
        }

        pub(super) fn to_icc(&self) -> Result<Buffer> {
            let v = self.inner.encode().map_err(cms)?;
            let mut b = crate::buffer::zeroed(v.len())?;
            b.copy_from_slice(&v);
            Ok(b)
        }

        pub(super) fn channels(&self) -> Channels {
            match self.inner.color_space {
                DataColorSpace::Gray => Channels::Gray,
                DataColorSpace::Rgb => Channels::Rgb,
                _ => Channels::Other,
            }
        }
    }

    pub(super) fn convert(
        src: &Surface,
        from: &Profile,
        to: &Profile,
        intent: Intent,
        format: Format,
    ) -> Result<Image> {
        let (fc, tc) = (from.channels(), to.channels());
        if fc == Channels::Other || tc == Channels::Other {
            return Err(Error::Unsupported);
        }
        let n_in = super::format_channels(src.format()).ok_or(Error::Unsupported)?;
        let n_out = super::format_channels(format).ok_or(Error::Unsupported)?;
        // Gray data needs a gray profile and vice versa (gray + alpha formats: Rg*).
        let gray_in = n_in <= 2;
        let gray_out = n_out <= 2;
        if gray_in != (fc == Channels::Gray) || gray_out != (tc == Channels::Gray) {
            return Err(Error::InvalidArgument);
        }
        let it = match intent {
            Intent::Perceptual => 0,
            Intent::RelativeColorimetric => 1,
            Intent::Saturation => 2,
            Intent::AbsoluteColorimetric => 3,
        };
        let sin = Side::new(&from.inner, it, true)?;
        let sout = Side::new(&to.inner, it, false)?;
        // Absolute colorimetric: scale by the media white points (LittleCMS, adaptation 1).
        let scale: [f64; 3] = if it == 3 {
            let (a, b) = (media_white(&from.inner), media_white(&to.inner));
            core::array::from_fn(|i| if b[i] != 0.0 { a[i] / b[i] } else { 1.0 })
        } else {
            [1.0; 3]
        };
        let (w, h) = (src.width() as usize, src.height() as usize);
        let rgba = crate::convert::convert_to_buffer(src, Format::Rgba32Float)?;
        let mut out = crate::buffer::zeroed(w * h * 16)?;
        let (ci, co) = (if gray_in { 1 } else { 3 }, if gray_out { 1 } else { 3 });
        let alpha_in = super::has_alpha(n_in);
        crate::parallel::for_each_unit(&mut out, w * 16, w * h * 32, |y, row| {
            let at = |i: usize| -> f64 {
                let k = i * 4;
                f64::from(f32::from_le_bytes([
                    rgba[k],
                    rgba[k + 1],
                    rgba[k + 2],
                    rgba[k + 3],
                ]))
            };
            for x in 0..w {
                let p = (y * w + x) * 4;
                let dv = if ci == 1 {
                    [at(p); 3]
                } else {
                    [at(p), at(p + 1), at(p + 2)]
                };
                let xyz = sin.device_to_xyz(dv);
                let xyz: [f64; 3] = core::array::from_fn(|i| xyz[i] * scale[i]);
                let o = sout.xyz_to_device(xyz);
                let a = if alpha_in {
                    at(p + if gray_in { 1 } else { 3 }) as f32
                } else {
                    1.0
                };
                let v: [f32; 4] = if co == 1 {
                    let g = o[0] as f32;
                    // Gray + alpha formats store alpha in G.
                    [g, a, g, a]
                } else {
                    [o[0] as f32, o[1] as f32, o[2] as f32, a]
                };
                for (c, f) in v.iter().enumerate() {
                    row[x * 16 + c * 4..x * 16 + c * 4 + 4].copy_from_slice(&f.to_le_bytes());
                }
            }
        });
        let s = Surface::new(Format::Rgba32Float, w as u32, h as u32, &out)?;
        let mut img = Image::new(format, w as u32, h as u32)?;
        crate::convert::convert(&s, format, &mut img.data)?;
        Ok(img)
    }
}

impl Profile {
    /// Parses an ICC profile (`InvalidData` if malformed, `Unsupported` without the `color`
    /// feature).
    pub fn from_icc(data: &[u8]) -> Result<Profile> {
        #[cfg(feature = "color")]
        {
            Profile::parse_icc(data)
        }
        #[cfg(not(feature = "color"))]
        {
            let _ = data;
            Err(Error::Unsupported)
        }
    }

    /// A built-in colour space (`Unsupported` without the `color` feature).
    pub fn new(space: ColorSpace) -> Result<Profile> {
        #[cfg(feature = "color")]
        {
            Ok(Profile::builtin(space))
        }
        #[cfg(not(feature = "color"))]
        {
            let _ = space;
            Err(Error::Unsupported)
        }
    }

    /// An RGB profile from chromaticities (red, green, blue, white `x, y`) and a transfer
    /// (Bradford adapted to D50). `InvalidData` for degenerate primaries.
    pub fn from_primaries(xy: [[f32; 2]; 4], transfer: Transfer) -> Result<Profile> {
        #[cfg(feature = "color")]
        {
            Profile::primaries(xy, transfer)
        }
        #[cfg(not(feature = "color"))]
        {
            let _ = (xy, transfer);
            Err(Error::Unsupported)
        }
    }

    /// The built-in space of a profile made with [`Profile::new`].
    pub fn space(&self) -> Option<ColorSpace> {
        self.space
    }

    /// sRGB.
    pub fn srgb() -> Result<Profile> {
        Profile::new(ColorSpace::Srgb)
    }

    /// From ITU-T H.273 code points (colour primaries, transfer characteristics, matrix
    /// coefficients, full range). `InvalidData` for unknown values, `Unsupported` for
    /// combinations without an RGB / gray meaning.
    pub fn cicp(primaries: u8, transfer: u8, matrix: u8, full_range: bool) -> Result<Profile> {
        #[cfg(feature = "color")]
        {
            Profile::from_cicp(primaries, transfer, matrix, full_range)
        }
        #[cfg(not(feature = "color"))]
        {
            let _ = (primaries, transfer, matrix, full_range);
            Err(Error::Unsupported)
        }
    }

    /// The profile as ICC v4 bytes.
    pub fn icc(&self) -> Result<Buffer> {
        #[cfg(feature = "color")]
        {
            self.to_icc()
        }
        #[cfg(not(feature = "color"))]
        {
            Err(Error::Unsupported)
        }
    }

    /// Gray, RGB or other.
    pub fn data_channels(&self) -> Channels {
        #[cfg(feature = "color")]
        {
            self.channels()
        }
        #[cfg(not(feature = "color"))]
        {
            Channels::Other
        }
    }
}

/// Colour channels of an uncompressed format (1 gray, 2 gray + alpha, 3 RGB, 4 RGBA).
#[cfg(feature = "color")]
fn format_channels(f: Format) -> Option<usize> {
    crate::resize::channels(f)
}

#[cfg(feature = "color")]
fn has_alpha(n: usize) -> bool {
    n == 2 || n == 4
}

/// Converts `src` (gray, gray + alpha, RGB or RGBA; values as stored) from profile `from` to
/// `to` and writes `format` (gray formats need gray profiles, RGB formats RGB profiles:
/// `InvalidArgument` otherwise). Alpha is copied (1 when the source has none).
///
/// Errors: `Unsupported` for compressed formats, CMYK / Lab profiles, a transform the engine
/// cannot build, or without the `color` feature.
pub fn convert(
    src: &Surface,
    from: &Profile,
    to: &Profile,
    intent: Intent,
    format: Format,
) -> Result<Image> {
    #[cfg(feature = "color")]
    {
        imp::convert(src, from, to, intent, format)
    }
    #[cfg(not(feature = "color"))]
    {
        let _ = (src, from, to, intent, format);
        Err(Error::Unsupported)
    }
}
