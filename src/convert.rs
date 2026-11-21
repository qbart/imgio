//! Conversion between uncompressed formats.
//!
//! Values are converted numerically per channel: unorm `v / 255`, snorm `max(v / 127, -1)`,
//! floats as is; to 8-bit unorm by clamping to [0, 1] and rounding. Missing channels become
//! green/blue 0 and alpha 1. sRGB transfer functions are not applied (sRGB and unorm only
//! differ in the label).

use crate::parallel::for_each_unit;
use crate::{Buffer, Error, Format, Result, Surface, half};

fn read(format: Format, p: &[u8]) -> [f32; 4] {
    let un = |v: u8| v as f32 / 255.0;
    let sn = |v: u8| (v as i8 as f32 / 127.0).max(-1.0);
    let h = |i: usize| half::to_f32(u16::from_le_bytes([p[i], p[i + 1]]));
    let f = |i: usize| f32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]);
    match format {
        Format::R8Unorm => [un(p[0]), 0.0, 0.0, 1.0],
        Format::R8Snorm => [sn(p[0]), 0.0, 0.0, 1.0],
        Format::Rg8Unorm => [un(p[0]), un(p[1]), 0.0, 1.0],
        Format::Rg8Snorm => [sn(p[0]), sn(p[1]), 0.0, 1.0],
        Format::Rgba8Unorm | Format::Rgba8Srgb => [un(p[0]), un(p[1]), un(p[2]), un(p[3])],
        Format::Rgba16Float => [h(0), h(2), h(4), h(6)],
        Format::Rgba32Float => [f(0), f(4), f(8), f(12)],
        _ => [0.0; 4],
    }
}

fn write(format: Format, v: [f32; 4], out: &mut [u8]) {
    let u8n = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    match format {
        Format::Rgba8Unorm | Format::Rgba8Srgb => {
            for c in 0..4 {
                out[c] = u8n(v[c]);
            }
        }
        Format::Rgba16Float => {
            for c in 0..4 {
                out[c * 2..c * 2 + 2].copy_from_slice(&half::from_f32(v[c]).to_le_bytes());
            }
        }
        Format::Rgba32Float => {
            for c in 0..4 {
                out[c * 4..c * 4 + 4].copy_from_slice(&v[c].to_le_bytes());
            }
        }
        _ => {}
    }
}

/// True when [`convert`] can read `format`.
pub fn can_read(format: Format) -> bool {
    !format.is_compressed()
}

/// True when [`convert`] can write `format`.
pub fn can_write(format: Format) -> bool {
    matches!(
        format,
        Format::Rgba8Unorm | Format::Rgba8Srgb | Format::Rgba16Float | Format::Rgba32Float
    )
}

/// Converts `src` to `dst_format` into `dst` (tightly packed rows). Multithreaded for large images.
pub fn convert(src: &Surface, dst_format: Format, dst: &mut [u8]) -> Result<()> {
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
    for_each_unit(dst, dst_row, |y, row| {
        let s = &data[y * src_row..(y + 1) * src_row];
        for (sp, dp) in s.chunks_exact(sp).zip(row.chunks_exact_mut(dp)) {
            write(dst_format, read(sf, sp), dp);
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
}
