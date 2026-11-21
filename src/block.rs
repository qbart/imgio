//! Shared driver that decodes a block-compressed surface block by block.

use crate::parallel::for_each_unit;
use crate::{Error, Format, Result};

/// Largest decoded block: 12x12 pixels of RGBA16F.
pub(crate) const MAX_BLOCK_OUT: usize = 12 * 12 * 8;

/// Decodes one block of a format into `block_w * block_h` pixels of `format.decoded()`, row major.
pub(crate) type BlockDecoder = fn(Format, &[u8], &mut [u8]);

/// Decodes a `width` x `height` surface of `format` from `src` into tightly packed rows of
/// `format.decoded()` in `dst`, one row of blocks per work unit (multithreaded).
pub(crate) fn decode_surface(
    format: Format,
    src: &[u8],
    width: u32,
    height: u32,
    dst: &mut [u8],
    decode_block: BlockDecoder,
) -> Result<()> {
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
    let (bw, bh) = format.block_dim();
    let (bw, bh) = (bw as usize, bh as usize);
    let px = out_fmt.block_bytes();
    if bw * bh * px > MAX_BLOCK_OUT {
        return Err(Error::Unsupported);
    }
    let block_bytes = format.block_bytes();
    let src_row = format.row_bytes(width)?;
    let dst_row = out_fmt.row_bytes(width)?;
    let w = width as usize;
    let pixels = (width as usize).saturating_mul(height as usize);
    for_each_unit(&mut dst[..dst_len], dst_row * bh, pixels, |by, rows| {
        let src_blocks = &src[by * src_row..(by + 1) * src_row];
        let pixel_rows = rows.len() / dst_row;
        let mut out = [0u8; MAX_BLOCK_OUT];
        let out = &mut out[..bw * bh * px];
        for (bx, block) in src_blocks.chunks_exact(block_bytes).enumerate() {
            decode_block(format, block, out);
            let x0 = bx * bw;
            let cols = (w - x0).min(bw);
            for y in 0..pixel_rows {
                let d = y * dst_row + x0 * px;
                let s = y * bw * px;
                rows[d..d + cols * px].copy_from_slice(&out[s..s + cols * px]);
            }
        }
    });
    Ok(())
}
