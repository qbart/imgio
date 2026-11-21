//! ASTC LDR 4x4..12x12 payloads (plain ASTC LDR, and XUASTC LDR once decompressed) to GPU
//! formats, following basisu `basisu_lowlevel_xuastc_ldr_transcoder::transcode_slice`.
//!
//! ASTC targets with the same block size are a copy. Every other target decodes the blocks to
//! an RGBA8 image (padded to whole blocks, rows of blocks in parallel) and then emits the
//! target from it, 4x4 block by 4x4 block. Edge blocks read clamped pixels: x is clamped to the
//! padded width, y to the image height, or to the padded height when the whole image path of
//! the C++ transcoder is used (deblocking, PVRTC1).

use crate::astc::helpers::{DecodePixels, LogAstcBlock, decode_block, unpack_block};
use crate::parallel::try_for_each_unit;
use crate::{AstcBlock, AstcKind, Buffer, Error, Format, Result};

/// Decoded ASTC image, RGBA8, padded to whole source blocks.
pub(crate) struct Image {
    pub data: Buffer,
    pub width: usize,
}

/// One image of ASTC LDR blocks.
pub(crate) struct AstcLdrImage<'a> {
    /// Physical 16-byte blocks, row major.
    pub blocks: &'a [u8],
    pub block: AstcBlock,
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub width: u32,
    pub height: u32,
    /// ASTC sRGB decode profile (otherwise linear LDR).
    pub srgb: bool,
    /// Apply the deblocking filter before emitting non-ASTC targets.
    pub deblock: bool,
}

/// True when ASTC LDR payloads with `block` size can be transcoded to `format`.
pub(crate) fn can_transcode(block: AstcBlock, format: Format) -> bool {
    use Format::*;
    match format {
        Astc(b, AstcKind::Unorm | AstcKind::Srgb) => b == block,
        Rgba8Unorm | Rgba8Srgb | Rgb565 | Bgr565 | Rgba4444 => true,
        _ => false,
    }
}

/// Decodes one physical ASTC LDR block into `out` (`w * h` RGBA8 pixels).
fn decode_ldr(block: &[u8; 16], w: u32, h: u32, srgb: bool, out: &mut [[u8; 4]]) -> Result<()> {
    let mut log = LogAstcBlock::default();
    if !unpack_block(block, &mut log, w, h) {
        return Err(Error::InvalidData);
    }
    let mut px = if srgb {
        DecodePixels::Srgb8(out)
    } else {
        DecodePixels::Ldr8(out)
    };
    if !decode_block(&log, &mut px, w, h) {
        return Err(Error::InvalidData);
    }
    Ok(())
}

/// Decodes every block of `src` into a padded RGBA8 image (multithreaded).
pub(crate) fn decode_image(src: &AstcLdrImage) -> Result<Image> {
    let (bw, bh) = src.block.dim();
    let (bxs, bys) = (src.blocks_x as usize, src.blocks_y as usize);
    if src.blocks.len() < bxs * bys * 16 {
        return Err(Error::InvalidData);
    }
    let (width, height) = (bxs * bw as usize, bys * bh as usize);
    let mut data = crate::buffer::zeroed(width * height * 4)?;
    let row_bytes = width * 4;
    let pixels = width.saturating_mul(height);
    try_for_each_unit(&mut data, row_bytes * bh as usize, pixels, |by, rows| {
        let mut px = [[0u8; 4]; crate::astc::helpers::MAX_BLOCK_PIXELS];
        let px = &mut px[..(bw * bh) as usize];
        let blocks = &src.blocks[by * bxs * 16..(by + 1) * bxs * 16];
        for (bx, block) in blocks.as_chunks::<16>().0.iter().enumerate() {
            decode_ldr(block, bw, bh, src.srgb, px)?;
            for (y, line) in px.chunks_exact(bw as usize).enumerate() {
                let d = y * row_bytes + bx * bw as usize * 4;
                rows[d..d + bw as usize * 4].copy_from_slice(line.as_flattened());
            }
        }
        Ok(())
    })?;
    Ok(Image { data, width })
}

/// basisu `mul_8`: `v * q / 255` rounded.
fn mul_8(v: u8, q: u32) -> u16 {
    let v = u32::from(v) * q + 128;
    ((v + (v >> 8)) >> 8) as u16
}

/// Writes the uncompressed `format` for the `width` x `height` area of `img` (multithreaded).
fn write_uncompressed(
    img: &Image,
    width: usize,
    height: usize,
    format: Format,
    dst: &mut [u8],
) -> Result<()> {
    let bpp = format.block_bytes();
    let row = width * bpp;
    try_for_each_unit(&mut dst[..row * height], row, width * height, |y, out| {
        let src = &img.data[y * img.width * 4..(y * img.width + width) * 4];
        for (o, p) in out.chunks_exact_mut(bpp).zip(src.as_chunks::<4>().0) {
            let [r, g, b, a] = *p;
            let v = match format {
                Format::Rgba8Unorm | Format::Rgba8Srgb => {
                    o.copy_from_slice(p);
                    continue;
                }
                Format::Rgb565 => (mul_8(r, 31) << 11) | (mul_8(g, 63) << 5) | mul_8(b, 31),
                Format::Bgr565 => (mul_8(b, 31) << 11) | (mul_8(g, 63) << 5) | mul_8(r, 31),
                Format::Rgba4444 => {
                    (mul_8(r, 15) << 12) | (mul_8(g, 15) << 8) | (mul_8(b, 15) << 4) | mul_8(a, 15)
                }
                _ => return Err(Error::Unsupported),
            };
            o.copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    })
}

/// Transcodes `src` to `format` into `dst` (tightly packed, size checked by the caller).
pub(crate) fn transcode(src: &AstcLdrImage, format: Format, dst: &mut [u8]) -> Result<()> {
    if !can_transcode(src.block, format) {
        return Err(Error::Unsupported);
    }
    let blocks = src.blocks_x as usize * src.blocks_y as usize * 16;
    if let Format::Astc(..) = format {
        let s = src.blocks.get(..blocks).ok_or(Error::InvalidData)?;
        let d = dst.get_mut(..blocks).ok_or(Error::InvalidArgument)?;
        d.copy_from_slice(s);
        return Ok(());
    }
    if src.deblock {
        return Err(Error::Unsupported); // deblocking filter not ported yet
    }
    let img = decode_image(src)?;
    write_uncompressed(&img, src.width as usize, src.height as usize, format, dst)
}
