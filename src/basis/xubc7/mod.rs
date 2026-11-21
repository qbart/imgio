// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! XUBC7 (supercompressed BC7) payloads to GPU formats, following basisu
//! `basisu_lowlevel_xubc7_transcoder::transcode_slice`.
//!
//! The XBC7 stream is decoded to logical BC7 blocks first (stripes in parallel), then every
//! target is produced from them in parallel: BC7 is a repack, ASTC 4x4 goes latent to latent
//! through `bc7f::fast_pack_astc`, everything else from the decoded pixels exactly like the
//! ASTC LDR 4x4 path (`transcode_4x4_block`, `encode_pvrtc1`).

mod decoder;

use super::astc_ldr::{BlockOpts, Image, write_blocks, write_uncompressed};
use super::ll::bc7f_astc::fast_pack_astc;
use super::ll::bc7u::{self, LogBc7Block};
use crate::parallel::try_for_each_unit;
use crate::{AstcBlock, AstcKind, Error, Format, Result};

/// True when XUBC7 payloads can be transcoded to `format` (basisu `basis_is_format_supported`).
pub(crate) fn can_transcode(format: Format) -> bool {
    use Format::*;
    matches!(
        format,
        Astc(AstcBlock::B4x4, AstcKind::Unorm | AstcKind::Srgb)
            | Rgba8Unorm
            | Rgba8Srgb
            | Rgb565
            | Bgr565
            | Rgba4444
            | Bc1Unorm
            | Bc1Srgb
            | Bc3Unorm
            | Bc3Srgb
            | Bc4Unorm
            | Bc5Unorm
            | Bc7Unorm
            | Bc7Srgb
            | EacR11Unorm
            | EacRg11Unorm
            | Etc1Rgb
            | Etc2Rgba
            | Etc2RgbaSrgb
            | Pvrtc1Rgb
            | Pvrtc1Rgba
    )
}

/// One XUBC7 image.
pub(crate) struct Xubc7Image<'a> {
    /// The XBC7 stream.
    pub data: &'a [u8],
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub width: u32,
    pub height: u32,
    /// Container alpha flag (PVRTC1 RGBA falls back to RGB without it).
    pub has_alpha: bool,
    /// Decode flag effects.
    pub opts: BlockOpts,
}

/// Transcodes `src` to `format` into `dst` (tightly packed, size checked by the caller).
pub(crate) fn transcode(src: &Xubc7Image, format: Format, dst: &mut [u8]) -> Result<()> {
    if !can_transcode(format) {
        return Err(Error::Unsupported);
    }
    let pvrtc1 = matches!(format, Format::Pvrtc1Rgb | Format::Pvrtc1Rgba);
    if pvrtc1 && (!src.width.is_power_of_two() || !src.height.is_power_of_two()) {
        return Err(Error::InvalidArgument);
    }
    let decoded = decoder::decode(src.data, src.blocks_x, src.blocks_y)?;
    let blocks = decoded.blocks.as_slice();
    let (bxs, bys) = (src.blocks_x as usize, src.blocks_y as usize);
    if blocks.len() != bxs * bys {
        return Err(Error::InvalidData);
    }
    let pixels = bxs * bys * 16;
    let rows = |by: usize| &blocks[by * bxs..(by + 1) * bxs];

    match format {
        Format::Bc7Unorm | Format::Bc7Srgb | Format::Astc(..) => {
            let astc = matches!(format, Format::Astc(..));
            let dst = dst.get_mut(..pixels).ok_or(Error::InvalidArgument)?;
            try_for_each_unit(dst, bxs * 16, pixels, |by, out| {
                for (log, o) in rows(by).iter().zip(out.as_chunks_mut::<16>().0) {
                    let mut bc7 = [0u8; 16];
                    if !bc7u::pack_bc7(log, &mut bc7) {
                        return Err(Error::InvalidData);
                    }
                    if astc {
                        if !fast_pack_astc(o, &bc7, src.opts.bc7_flags()) {
                            return Err(Error::InvalidData);
                        }
                    } else {
                        *o = bc7;
                    }
                }
                Ok(())
            })
        }
        _ => {
            let img = decode_pixels(blocks, bxs, bys)?;
            let (w, h) = (src.width as usize, src.height as usize);
            if !format.is_compressed() {
                write_uncompressed(&img, w, h, format, dst)
            } else if pvrtc1 {
                super::pvrtc1_enc::encode_pvrtc1(
                    &img.data,
                    img.width,
                    img.height,
                    format == Format::Pvrtc1Rgba && src.has_alpha,
                    src.opts.from_alpha,
                    src.blocks_x,
                    src.blocks_y,
                    dst,
                )
            } else {
                write_blocks(&img, w, h, img.height, format, true, src.opts, dst)
            }
        }
    }
}

/// Decodes logical blocks to an RGBA8 image padded to whole blocks (rows in parallel).
fn decode_pixels(blocks: &[LogBc7Block], bxs: usize, bys: usize) -> Result<Image> {
    let (width, height) = (bxs * 4, bys * 4);
    let mut data = crate::buffer::zeroed(width * height * 4)?;
    let row_bytes = width * 4;
    try_for_each_unit(&mut data, row_bytes * 4, width * height, |by, out| {
        let row = blocks
            .get(by * bxs..(by + 1) * bxs)
            .ok_or(Error::InvalidData)?;
        let mut px = [[0u8; 4]; 16];
        for (bx, log) in row.iter().enumerate() {
            if !bc7u::unpack_pixels(log, &mut px) {
                return Err(Error::InvalidData);
            }
            for (y, line) in px.as_chunks::<4>().0.iter().enumerate() {
                let d = y * row_bytes + bx * 16;
                out[d..d + 16].copy_from_slice(line.as_flattened());
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
