//! Conversion between uncompressed formats.
//!
//! Values are converted numerically per channel through f32: unorm `v / max`, snorm
//! `max(v / max, -1)`, floats as is; to unorm / snorm by clamping and rounding, to RGB9E5 with
//! the shared exponent rule of the format. Missing channels become green/blue 0 and alpha 1.
//! By default sRGB and unorm only differ in the label; [`ConvertOptions::srgb_transfer`]
//! applies the sRGB curve to RGB (not alpha) when exactly one side is sRGB.

use crate::parallel::for_each_unit;
use crate::{Buffer, Error, Format, Result, Surface, half};

fn read(format: Format, p: &[u8]) -> [f32; 4] {
    let un = |v: u8| v as f32 / 255.0;
    let sn = |v: u8| (v as i8 as f32 / 127.0).max(-1.0);
    let un16 = |i: usize| u16::from_le_bytes([p[i], p[i + 1]]) as f32 / 65535.0;
    let sn16 = |i: usize| (i16::from_le_bytes([p[i], p[i + 1]]) as f32 / 32767.0).max(-1.0);
    let h = |i: usize| half::to_f32(u16::from_le_bytes([p[i], p[i + 1]]));
    let f = |i: usize| f32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]);
    match format {
        Format::R8Unorm => [un(p[0]), 0.0, 0.0, 1.0],
        Format::R8Snorm => [sn(p[0]), 0.0, 0.0, 1.0],
        Format::Rg8Unorm => [un(p[0]), un(p[1]), 0.0, 1.0],
        Format::Rg8Snorm => [sn(p[0]), sn(p[1]), 0.0, 1.0],
        Format::R16Unorm => [un16(0), 0.0, 0.0, 1.0],
        Format::R16Snorm => [sn16(0), 0.0, 0.0, 1.0],
        Format::Rg16Unorm => [un16(0), un16(2), 0.0, 1.0],
        Format::Rg16Snorm => [sn16(0), sn16(2), 0.0, 1.0],
        Format::Rgba8Unorm | Format::Rgba8Srgb => [un(p[0]), un(p[1]), un(p[2]), un(p[3])],
        Format::Bgra8Unorm | Format::Bgra8Srgb => [un(p[2]), un(p[1]), un(p[0]), un(p[3])],
        Format::Rgb8Unorm | Format::Rgb8Srgb => [un(p[0]), un(p[1]), un(p[2]), 1.0],
        Format::Rgb565 | Format::Bgr565 | Format::Rgba4444 => {
            let v = u16::from_le_bytes([p[0], p[1]]) as u32;
            let n = |x: u32, max: f32| x as f32 / max;
            match format {
                Format::Rgb565 => [
                    n(v >> 11, 31.0),
                    n((v >> 5) & 63, 63.0),
                    n(v & 31, 31.0),
                    1.0,
                ],
                Format::Bgr565 => [
                    n(v & 31, 31.0),
                    n((v >> 5) & 63, 63.0),
                    n(v >> 11, 31.0),
                    1.0,
                ],
                _ => [
                    n(v >> 12, 15.0),
                    n((v >> 8) & 15, 15.0),
                    n((v >> 4) & 15, 15.0),
                    n(v & 15, 15.0),
                ],
            }
        }
        Format::Rgb16Float => [h(0), h(2), h(4), 1.0],
        Format::Rgb9e5Float => {
            let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            let scale = 2f32.powi((v >> 27) as i32 - 15 - 9);
            let m = |s: u32| ((v >> s) & 511) as f32 * scale;
            [m(0), m(9), m(18), 1.0]
        }
        Format::Rgba16Float => [h(0), h(2), h(4), h(6)],
        Format::Rgba16Unorm => [un16(0), un16(2), un16(4), un16(6)],
        Format::Rgba32Float => [f(0), f(4), f(8), f(12)],
        Format::R16Float => [h(0), 0.0, 0.0, 1.0],
        Format::R32Float => [f(0), 0.0, 0.0, 1.0],
        _ => [0.0; 4],
    }
}

/// sRGB electro-optical transfer: encoded [0, 1] -> linear.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse of [`srgb_to_linear`]: linear [0, 1] -> encoded.
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// RGB9E5 encoding (`EXT_texture_shared_exponent`): 9-bit mantissas, 5-bit shared exponent.
fn rgb9e5(v: [f32; 4]) -> u32 {
    const N: i32 = 9;
    const B: i32 = 15;
    const MAX: f32 = 65408.0; // (2^9 - 1) / 2^9 * 2^(31 - 15)
    let c = [0, 1, 2].map(|i| {
        let x = v[i];
        if x.is_nan() { 0.0 } else { x.clamp(0.0, MAX) }
    });
    let maxc = c[0].max(c[1]).max(c[2]);
    // floor(log2(maxc)) from the float exponent (maxc is normal or zero here, or a denormal).
    let log2 = if maxc > 0.0 {
        maxc.log2().floor() as i32
    } else {
        -B - 1
    };
    let mut exp = log2.max(-B - 1) + 1 + B;
    let scale = |e: i32| 2f32.powi(e - B - N);
    let max_s = (maxc / scale(exp) + 0.5).floor() as i32;
    if max_s == 1 << N {
        exp += 1;
    }
    let m = c.map(|x| ((x / scale(exp) + 0.5).floor() as u32).min(511));
    m[0] | (m[1] << 9) | (m[2] << 18) | ((exp as u32) << 27)
}

fn write(format: Format, v: [f32; 4], out: &mut [u8]) {
    let u8n = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let s8n = |x: f32| (x.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8;
    let u16n = |x: f32| ((x.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_le_bytes();
    let s16n = |x: f32| ((x.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes();
    let bits = |x: f32, max: f32| (x.clamp(0.0, 1.0) * max + 0.5) as u16;
    let mut put16 = |i: usize, b: [u8; 2]| out[i * 2..i * 2 + 2].copy_from_slice(&b);
    match format {
        Format::R8Unorm => out[0] = u8n(v[0]),
        Format::R8Snorm => out[0] = s8n(v[0]),
        Format::Rg8Unorm => {
            out[0] = u8n(v[0]);
            out[1] = u8n(v[1]);
        }
        Format::Rg8Snorm => {
            out[0] = s8n(v[0]);
            out[1] = s8n(v[1]);
        }
        Format::Rgba8Unorm | Format::Rgba8Srgb => {
            for c in 0..4 {
                out[c] = u8n(v[c]);
            }
        }
        Format::Bgra8Unorm | Format::Bgra8Srgb => {
            out[..4].copy_from_slice(&[u8n(v[2]), u8n(v[1]), u8n(v[0]), u8n(v[3])]);
        }
        Format::Rgb8Unorm | Format::Rgb8Srgb => {
            out[..3].copy_from_slice(&[u8n(v[0]), u8n(v[1]), u8n(v[2])]);
        }
        Format::R16Unorm => put16(0, u16n(v[0])),
        Format::R16Snorm => put16(0, s16n(v[0])),
        Format::Rg16Unorm | Format::Rgba16Unorm => {
            let n = if format == Format::Rg16Unorm { 2 } else { 4 };
            for (c, &x) in v[..n].iter().enumerate() {
                put16(c, u16n(x));
            }
        }
        Format::Rg16Snorm => {
            put16(0, s16n(v[0]));
            put16(1, s16n(v[1]));
        }
        Format::Rgb565 => put16(
            0,
            ((bits(v[0], 31.0) << 11) | (bits(v[1], 63.0) << 5) | bits(v[2], 31.0)).to_le_bytes(),
        ),
        Format::Bgr565 => put16(
            0,
            ((bits(v[2], 31.0) << 11) | (bits(v[1], 63.0) << 5) | bits(v[0], 31.0)).to_le_bytes(),
        ),
        Format::Rgba4444 => put16(
            0,
            ((bits(v[0], 15.0) << 12)
                | (bits(v[1], 15.0) << 8)
                | (bits(v[2], 15.0) << 4)
                | bits(v[3], 15.0))
            .to_le_bytes(),
        ),
        Format::Rgb16Float | Format::Rgba16Float => {
            let n = if format == Format::Rgb16Float { 3 } else { 4 };
            for (c, &x) in v[..n].iter().enumerate() {
                put16(c, half::from_f32(x).to_le_bytes());
            }
        }
        Format::Rgb9e5Float => out[..4].copy_from_slice(&rgb9e5(v).to_le_bytes()),
        Format::Rgba32Float => {
            for c in 0..4 {
                out[c * 4..c * 4 + 4].copy_from_slice(&v[c].to_le_bytes());
            }
        }
        Format::R16Float => put16(0, half::from_f32(v[0]).to_le_bytes()),
        Format::R32Float => out[..4].copy_from_slice(&v[0].to_le_bytes()),
        _ => {}
    }
}

/// True when [`convert`] can read `format`.
pub fn can_read(format: Format) -> bool {
    !format.is_compressed()
}

/// True when [`convert`] can write `format` (every uncompressed format).
pub fn can_write(format: Format) -> bool {
    !format.is_compressed()
}

/// Options of [`convert_with`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConvertOptions {
    /// Apply the sRGB curve to RGB when exactly one of source / destination is sRGB
    /// (decode sRGB -> linear, or encode linear -> sRGB). Default: labels only.
    pub srgb_transfer: bool,
}

/// Converts `src` to `dst_format` into `dst` (tightly packed rows). Multithreaded for large images.
pub fn convert(src: &Surface, dst_format: Format, dst: &mut [u8]) -> Result<()> {
    convert_with(src, dst_format, &ConvertOptions::default(), dst)
}

/// Like [`convert`], with options.
pub fn convert_with(
    src: &Surface,
    dst_format: Format,
    opts: &ConvertOptions,
    dst: &mut [u8],
) -> Result<()> {
    let sf = src.format();
    if !can_read(sf) || !can_write(dst_format) {
        return Err(Error::Unsupported);
    }
    let (w, h) = (src.width(), src.height());
    let len = dst_format.surface_bytes(w, h)?;
    let got = dst.len();
    let dst = dst
        .get_mut(..len)
        .ok_or(Error::BufferTooSmall { needed: len, got })?;
    let (sp, dp) = (sf.block_bytes(), dst_format.block_bytes());
    let (src_row, dst_row) = (sf.row_bytes(w)?, dst_format.row_bytes(w)?);
    let data = src.data();
    if sf == dst_format {
        dst.copy_from_slice(data);
        return Ok(());
    }
    let decode = opts.srgb_transfer && sf.is_srgb() && !dst_format.is_srgb();
    let encode = opts.srgb_transfer && !sf.is_srgb() && dst_format.is_srgb();
    let pixels = (w as usize).saturating_mul(h as usize);
    for_each_unit(dst, dst_row, pixels, |y, row| {
        let s = &data[y * src_row..(y + 1) * src_row];
        for (sp, dp) in s.chunks_exact(sp).zip(row.chunks_exact_mut(dp)) {
            let mut v = read(sf, sp);
            if decode {
                for c in &mut v[..3] {
                    *c = srgb_to_linear(c.clamp(0.0, 1.0));
                }
            } else if encode {
                for c in &mut v[..3] {
                    *c = linear_to_srgb(c.clamp(0.0, 1.0));
                }
            }
            write(dst_format, v, dp);
        }
    });
    Ok(())
}

/// Like [`convert`] but allocates the output.
pub fn convert_to_buffer(src: &Surface, dst_format: Format) -> Result<Buffer> {
    let len = dst_format.surface_bytes(src.width(), src.height())?;
    let mut buf = crate::buffer::zeroed(len)?;
    convert(src, dst_format, &mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba8_float_roundtrip() -> Result<()> {
        let px: Vec<u8> = (0..=255u8).collect();
        let s = Surface::new(Format::Rgba8Unorm, 64, 1, &px)?;
        for f in [Format::Rgba16Float, Format::Rgba32Float] {
            let mid = convert_to_buffer(&s, f)?;
            let back = convert_to_buffer(&Surface::new(f, 64, 1, &mid)?, Format::Rgba8Unorm)?;
            assert_eq!(&back[..], &px[..]);
        }
        Ok(())
    }

    #[test]
    fn packed() -> Result<()> {
        let b = 0xF81Fu16.to_le_bytes();
        let s = Surface::new(Format::Rgb565, 1, 1, &b)?;
        assert_eq!(
            &convert_to_buffer(&s, Format::Rgba8Unorm)?[..],
            &[255, 0, 255, 255]
        );
        let b = 0x0F80u16.to_le_bytes();
        let s = Surface::new(Format::Rgba4444, 1, 1, &b)?;
        assert_eq!(
            &convert_to_buffer(&s, Format::Rgba8Unorm)?[..],
            &[0, 255, 136, 0]
        );
        // 9e5: mantissa 256 with exponent 15 -> 0.5
        let v: u32 = (15 << 27) | (256 << 18) | (256 << 9) | 256;
        let b = v.to_le_bytes();
        let s = Surface::new(Format::Rgb9e5Float, 1, 1, &b)?;
        assert_eq!(
            &convert_to_buffer(&s, Format::Rgba8Unorm)?[..],
            &[128, 128, 128, 255]
        );
        Ok(())
    }

    #[test]
    fn channels() -> Result<()> {
        let s = Surface::new(Format::Rg8Snorm, 2, 1, &[0x81, 0x7F, 0x80, 0])?;
        let out = convert_to_buffer(&s, Format::Rgba32Float)?;
        let f: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(f, [-1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0]);
        let s = Surface::new(Format::R8Unorm, 1, 1, &[200])?;
        assert_eq!(
            &convert_to_buffer(&s, Format::Rgba8Unorm)?[..],
            &[200, 0, 0, 255]
        );
        assert!(convert_to_buffer(&s, Format::Bc7Unorm).is_err());
        Ok(())
    }

    /// Every uncompressed format: RGBA8 -> format -> RGBA8 keeps what the format can hold.
    #[test]
    fn every_format_roundtrips() -> Result<()> {
        use Format::*;
        let px = [0u8, 51, 102, 255, 255, 204, 153, 0];
        let s = Surface::new(Rgba8Unorm, 2, 1, &px)?;
        for (f, want) in [
            (R8Unorm, [0, 0, 0, 255, 255, 0, 0, 255]),
            (Rg8Unorm, [0, 51, 0, 255, 255, 204, 0, 255]),
            (R16Unorm, [0, 0, 0, 255, 255, 0, 0, 255]),
            (Rg16Unorm, [0, 51, 0, 255, 255, 204, 0, 255]),
            (Rgba16Unorm, px),
            (Rgba16Float, px),
            (Rgba32Float, px),
            (Rgb16Float, [0, 51, 102, 255, 255, 204, 153, 255]),
            (Rgb9e5Float, [0, 51, 102, 255, 255, 204, 153, 255]),
            (Rgb565, [0, 53, 99, 255, 255, 202, 156, 255]),
            (Bgr565, [0, 53, 99, 255, 255, 202, 156, 255]),
            (Rgba4444, [0, 51, 102, 255, 255, 204, 153, 0]),
        ] {
            let mid = convert_to_buffer(&s, f)?;
            let back = convert_to_buffer(&Surface::new(f, 2, 1, &mid)?, Rgba8Unorm)?;
            assert_eq!(&back[..], &want, "{f:?}");
        }
        // Snorm keeps the non-negative half.
        let mid = convert_to_buffer(&s, Rg16Snorm)?;
        let back = convert_to_buffer(&Surface::new(Rg16Snorm, 2, 1, &mid)?, Rgba8Unorm)?;
        assert_eq!(&back[..], &[0, 51, 0, 255, 255, 204, 0, 255]);
        Ok(())
    }

    #[test]
    fn srgb_transfer() -> Result<()> {
        let s = Surface::new(Format::Rgba8Srgb, 1, 1, &[188, 0, 255, 188])?;
        let opts = ConvertOptions {
            srgb_transfer: true,
        };
        let mut out = [0u8; 16];
        convert_with(&s, Format::Rgba32Float, &opts, &mut out)?;
        let f = |i: usize| f32::from_le_bytes([out[i], out[i + 1], out[i + 2], out[i + 3]]);
        assert!((f(0) - 0.5029).abs() < 1e-3, "{}", f(0));
        assert_eq!((f(4), f(8)), (0.0, 1.0));
        assert!((f(12) - 188.0 / 255.0).abs() < 1e-6); // alpha untouched
        // Back to sRGB bytes.
        let lin = Surface::new(Format::Rgba32Float, 1, 1, &out)?;
        let mut back = [0u8; 4];
        convert_with(&lin, Format::Rgba8Srgb, &opts, &mut back)?;
        assert_eq!(back, [188, 0, 255, 188]);
        // Default: labels only.
        convert(&s, Format::Rgba8Unorm, &mut back)?;
        assert_eq!(back, [188, 0, 255, 188]);
        Ok(())
    }

    #[test]
    fn rgb9e5_encoding() {
        let enc = |r: f32| rgb9e5([r, 0.0, 0.0, 1.0]);
        assert_eq!(enc(0.0), 0);
        assert_eq!(enc(0.5), (15 << 27) | 256);
        assert_eq!(enc(1.0), (16 << 27) | 256);
        assert_eq!(enc(65408.0), (31 << 27) | 511);
        assert_eq!(enc(1e9), (31 << 27) | 511);
        // Rounding up to 512 bumps the exponent.
        assert_eq!(enc(0.999_99), (16 << 27) | 256);
    }
}
