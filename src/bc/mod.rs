//! BC1-BC7 (DXT / BPTC / RGTC) block-compressed formats.

mod bc1;
mod bc6h;
mod bc7;
mod tables;

use crate::parallel::for_each_unit;
use crate::{Buffer, Error, Format, Result};

const MAX_BLOCK_OUT: usize = 16 * 8; // 16 pixels of RGBA16F

/// Decodes one block of `format` into 16 pixels of `format.decoded()`, row major.
/// Invalid BC6H/BC7 blocks decode to zero, as on GPUs.
fn decode_block(format: Format, block: &[u8], out: &mut [u8; MAX_BLOCK_OUT]) {
    use Format::*;
    let half = |b: &[u8]| -> [u8; 8] {
        let mut h = [0u8; 8];
        h.copy_from_slice(&b[..8]);
        h
    };
    let mut rgba = [[0u8; 4]; 16];
    match format {
        Bc1Unorm | Bc1Srgb => bc1::color(&half(block), false, &mut rgba),
        Bc2Unorm | Bc2Srgb => {
            bc1::color(&half(&block[8..]), true, &mut rgba);
            bc1::alpha4(&half(block), &mut rgba);
        }
        Bc3Unorm | Bc3Srgb => {
            bc1::color(&half(&block[8..]), true, &mut rgba);
            let mut a = [0u8; 16];
            bc1::bc4u(&half(block), &mut a);
            for (p, a) in rgba.iter_mut().zip(a) {
                p[3] = a;
            }
        }
        Bc4Unorm | Bc4Snorm | Bc5Unorm | Bc5Snorm => {
            let signed = matches!(format, Bc4Snorm | Bc5Snorm);
            let channels = if matches!(format, Bc4Unorm | Bc4Snorm) {
                1
            } else {
                2
            };
            for c in 0..channels {
                let mut v = [0u8; 16];
                let b = half(&block[c * 8..]);
                if signed {
                    bc1::bc4s(&b, &mut v);
                } else {
                    bc1::bc4u(&b, &mut v);
                }
                for (i, v) in v.iter().enumerate() {
                    out[i * channels + c] = *v;
                }
            }
            return;
        }
        Bc6hUfloat | Bc6hSfloat => {
            let mut b = [0u8; 16];
            b.copy_from_slice(&block[..16]);
            let mut rgb = [[0u16; 3]; 16];
            if !bc6h::unpack_bc6h(&b, &mut rgb, format == Bc6hSfloat) {
                rgb = [[0; 3]; 16];
            }
            for (i, p) in rgb.iter().enumerate() {
                let o = &mut out[i * 8..i * 8 + 8];
                o[0..2].copy_from_slice(&p[0].to_le_bytes());
                o[2..4].copy_from_slice(&p[1].to_le_bytes());
                o[4..6].copy_from_slice(&p[2].to_le_bytes());
                o[6..8].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0
            }
            return;
        }
        Bc7Unorm | Bc7Srgb => {
            let mut b = [0u8; 16];
            b.copy_from_slice(&block[..16]);
            if !bc7::unpack_bc7(&b, &mut rgba) {
                rgba = [[0; 4]; 16];
            }
        }
        _ => return,
    }
    for (i, p) in rgba.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(p);
    }
}

fn is_bc(format: Format) -> bool {
    use Format::*;
    matches!(
        format,
        Bc1Unorm
            | Bc1Srgb
            | Bc2Unorm
            | Bc2Srgb
            | Bc3Unorm
            | Bc3Srgb
            | Bc4Unorm
            | Bc4Snorm
            | Bc5Unorm
            | Bc5Snorm
            | Bc6hUfloat
            | Bc6hSfloat
            | Bc7Unorm
            | Bc7Srgb
    )
}

/// Decodes a `width` x `height` BC surface from `src` into `dst`, laid out as tightly packed
/// rows of `format.decoded()`. Runs multithreaded for large surfaces.
///
/// `src` needs `format.surface_bytes(width, height)` bytes and `dst` needs
/// `format.decoded().surface_bytes(width, height)` bytes; extra bytes are ignored.
pub fn decode(format: Format, src: &[u8], width: u32, height: u32, dst: &mut [u8]) -> Result<()> {
    if !is_bc(format) {
        return Err(Error::Unsupported);
    }
    let out_fmt = format.decoded();
    let src_len = format.surface_bytes(width, height)?;
    let dst_len = out_fmt.surface_bytes(width, height)?;
    if src.len() < src_len {
        return Err(Error::BufferTooSmall {
            needed: src_len,
            got: src.len(),
        });
    }
    if dst.len() < dst_len {
        return Err(Error::BufferTooSmall {
            needed: dst_len,
            got: dst.len(),
        });
    }
    let block_bytes = format.block_bytes();
    let src_row = format.row_bytes(width)?;
    let px_bytes = out_fmt.block_bytes();
    let dst_row = out_fmt.row_bytes(width)?;
    let w = width as usize;
    // One unit = one row of blocks = up to 4 pixel rows of output.
    for_each_unit(&mut dst[..dst_len], dst_row * 4, |by, rows| {
        let src_blocks = &src[by * src_row..(by + 1) * src_row];
        let pixel_rows = rows.len() / dst_row;
        let mut out = [0u8; MAX_BLOCK_OUT];
        for (bx, block) in src_blocks.chunks_exact(block_bytes).enumerate() {
            decode_block(format, block, &mut out);
            let x0 = bx * 4;
            let cols = (w - x0).min(4);
            for y in 0..pixel_rows {
                let d = y * dst_row + x0 * px_bytes;
                let s = y * 4 * px_bytes;
                rows[d..d + cols * px_bytes].copy_from_slice(&out[s..s + cols * px_bytes]);
            }
        }
    });
    Ok(())
}

/// Like [`decode`] but allocates the output buffer.
pub fn decode_to_buffer(format: Format, src: &[u8], width: u32, height: u32) -> Result<Buffer> {
    let len = format.decoded().surface_bytes(width, height)?;
    let mut buf = crate::buffer::zeroed(len)?;
    decode(format, src, width, height, &mut buf)?;
    Ok(buf)
}
