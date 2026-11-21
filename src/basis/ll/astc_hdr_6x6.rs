// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal (basisu_transcoder.cpp,
// `basisu_lowlevel_astc_hdr_6x6_transcoder`); modified.
//! ASTC HDR 6x6 (plain ASTC blocks) to ASTC HDR 6x6, BC6H, RGB(A) half and RGB9E5.

use super::etc1s::TranscodeStatus;
use super::uastc_hdr_6x6i::transcode_astc_6x6_blocks;
use super::{
    BlockFormat, TranscoderTextureFormat, basis_get_bytes_per_block_or_pixel,
    basis_validate_output_buffer_size,
};

/// Port of `basisu_lowlevel_astc_hdr_6x6_transcoder`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LowlevelAstcHdr6x6Transcoder;

impl LowlevelAstcHdr6x6Transcoder {
    pub fn new() -> Self {
        Self
    }

    /// Port of `transcode_image` (`num_blocks_x/y` are source 6x6 blocks).
    #[allow(clippy::too_many_arguments)]
    pub fn transcode_image(
        &self,
        target_format: TranscoderTextureFormat,
        output: &mut [u8],
        output_blocks_buf_size_in_blocks_or_pixels: u32,
        compressed_data: &[u8],
        num_blocks_x: u32,
        num_blocks_y: u32,
        orig_width: u32,
        orig_height: u32,
        slice_offset: u64,
        slice_length: u32,
        decode_flags: u32,
    ) -> TranscodeStatus {
        use TranscoderTextureFormat as F;
        let Some(src) = usize::try_from(slice_offset)
            .ok()
            .and_then(|s| compressed_data.get(s..s.checked_add(slice_length as usize)?))
        else {
            return TranscodeStatus::Failed;
        };
        if !basis_validate_output_buffer_size(
            target_format,
            output_blocks_buf_size_in_blocks_or_pixels,
            orig_width,
            orig_height,
            0,
            0,
        ) {
            return TranscodeStatus::Failed;
        }
        let fmt = match target_format {
            F::ASTC_HDR_6x6_RGBA => BlockFormat::AstcHdr6x6,
            F::BC6H => BlockFormat::Bc6h,
            F::RGB_HALF => BlockFormat::RgbHalf,
            F::RGBA_HALF => BlockFormat::RgbaHalf,
            F::RGB_9E5 => BlockFormat::Rgb9e5,
            _ => return TranscodeStatus::Failed,
        };
        let total = (num_blocks_x as usize) * (num_blocks_y as usize);
        let blocks = src.as_chunks::<16>().0;
        if blocks.len() < total {
            return TranscodeStatus::Failed;
        }
        transcode_astc_6x6_blocks(
            output,
            &blocks[..total],
            num_blocks_x,
            num_blocks_y,
            fmt,
            basis_get_bytes_per_block_or_pixel(target_format),
            orig_width,
            orig_height,
            0,
            0,
            decode_flags,
        )
    }
}
