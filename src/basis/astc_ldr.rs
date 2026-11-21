//! ASTC LDR 4x4..12x12 payloads (plain ASTC LDR, and XUASTC LDR once decompressed) to GPU
//! formats, following basisu `basisu_lowlevel_xuastc_ldr_transcoder::transcode_slice`.
//!
//! ASTC targets with the same block size are a copy. Every other target decodes the blocks to
//! an RGBA8 image (padded to whole blocks, rows of blocks in parallel) and then emits the
//! target from it, 4x4 block by 4x4 block. Edge blocks read clamped pixels: x is clamped to the
//! padded width, y to the image height, or to the padded height when the whole image path of
//! the C++ transcoder is used (deblocking, PVRTC1).

use super::ll::dxt::Dxt1Block;
use super::ll::eac::EacBlock;
use super::ll::etc::Color32;
use super::ll::uastc_bc1::{encode_bc1, encode_bc4};
use super::ll::uastc_etc::pack_eac;
use crate::astc::helpers::{DecodePixels, LogAstcBlock, decode_block, unpack_block};
use crate::parallel::try_for_each_unit;
use crate::{AstcBlock, AstcKind, Buffer, Error, Format, Result};

/// Decoded ASTC image, RGBA8, padded to whole source blocks.
pub(crate) struct Image {
    pub data: Buffer,
    pub width: usize,
    pub height: usize,
}

impl Image {
    /// basisu `vector2D::extract_block_clamped`: the 4x4 block at pixel (`x`, `y`), reading
    /// clamped coordinates (x to the width, y to `max_y` rows).
    fn block(&self, x: usize, y: usize, max_y: usize) -> [[u8; 4]; 16] {
        let mut out = [[0u8; 4]; 16];
        for (i, o) in out.iter_mut().enumerate() {
            let px = (x + (i & 3)).min(self.width - 1);
            let py = (y + (i >> 2)).min(max_y - 1);
            let at = (py * self.width + px) * 4;
            o.copy_from_slice(&self.data[at..at + 4]);
        }
        out
    }
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
        Bc1Unorm | Bc1Srgb | Bc3Unorm | Bc3Srgb | Bc4Unorm | Bc5Unorm => true,
        EacR11Unorm | EacRg11Unorm => true,
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
    Ok(Image {
        data,
        width,
        height,
    })
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

/// One channel of a 4x4 block.
fn channel(px: &[[u8; 4]; 16], c: usize) -> [u8; 16] {
    px.map(|p| p[c])
}

fn bc1(px: &[[u8; 4]; 16], out: &mut [u8]) {
    let colors = px.map(|c| Color32 { c });
    let mut b = Dxt1Block::default();
    encode_bc1(&mut b, &colors, 0);
    out.copy_from_slice(&b.bytes);
}

fn bc4(px: &[[u8; 4]; 16], c: usize, out: &mut [u8]) {
    let mut b = [0u8; 8];
    encode_bc4(&mut b, &channel(px, c));
    out.copy_from_slice(&b);
}

fn eac(px: &[[u8; 4]; 16], c: usize, out: &mut [u8]) {
    let mut b = EacBlock::default();
    pack_eac(&mut b, &channel(px, c));
    out.copy_from_slice(&b.bytes);
}

/// basisu `transcode_4x4_block` for block targets (default channels: R, and R+A for two).
fn encode_4x4(format: Format, px: &[[u8; 4]; 16], out: &mut [u8]) -> Result<()> {
    use Format::*;
    match format {
        Bc1Unorm | Bc1Srgb => bc1(px, out),
        Bc3Unorm | Bc3Srgb => {
            bc4(px, 3, &mut out[..8]);
            bc1(px, &mut out[8..]);
        }
        Bc4Unorm => bc4(px, 0, out),
        Bc5Unorm => {
            bc4(px, 0, &mut out[..8]);
            bc4(px, 3, &mut out[8..]);
        }
        EacR11Unorm => eac(px, 0, out),
        EacRg11Unorm => {
            eac(px, 0, &mut out[..8]);
            eac(px, 3, &mut out[8..]);
        }
        _ => return Err(Error::Unsupported),
    }
    Ok(())
}

/// Encodes 4x4 blocks of `img` (`width` x `height` area) to `format`, rows in parallel.
/// `max_y`: rows readable by the edge clamp.
fn write_blocks(
    img: &Image,
    width: usize,
    height: usize,
    max_y: usize,
    format: Format,
    dst: &mut [u8],
) -> Result<()> {
    let bb = format.block_bytes();
    let row = width.div_ceil(4) * bb;
    let rows = height.div_ceil(4);
    try_for_each_unit(&mut dst[..row * rows], row, width * height, |by, out| {
        for (bx, o) in out.chunks_exact_mut(bb).enumerate() {
            encode_4x4(format, &img.block(bx * 4, by * 4, max_y), o)?;
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
    let (w, h) = (src.width as usize, src.height as usize);
    if !format.is_compressed() {
        return write_uncompressed(&img, w, h, format, dst);
    }
    // 4x4 sources map block to block (padding rows are real decoded pixels); the C++
    // row-buffered path for other sizes clamps edge rows to the image height.
    let max_y = if src.block == AstcBlock::B4x4 {
        img.height
    } else {
        h.min(img.height)
    };
    write_blocks(&img, w, h, max_y, format, dst)
}
