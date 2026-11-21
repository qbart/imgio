// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of `ldr_astc_block_encode_image_fast_4x4` (`basisu_astc_ldr_encode.cpp`, effort 0 +
//! 4x4): every block is packed to BC7 by `bc7f` in ASTC compatible mode and transcoded to a
//! single ASTC LDR 4x4 candidate.

use super::astcf::finish_candidate;
use super::common::CemEncodeParams;
use super::dct_enc::GridWeightDct;
use super::{AstcLdrEncodeConfig, BlockInfo, EncodeBlockOutput, EncodeImageOutput};
use crate::astc::helpers as ah;
use crate::basis::ll::bc7f::{
    PACK_BC7_FLAG_ASTC_COMPATIBLE, PACK_BC7_FLAG_DISABLE_RGB_DUAL_PLANE, PACK_BC7_FLAG_PBIT_OPT,
    PACK_BC7_FLAG_PBIT_OPT_MODE6, PACK_BC7_FLAG_USE_2_SUBSETS_RGB,
    PACK_BC7_FLAG_USE_2_SUBSETS_RGBA, PACK_BC7_FLAG_USE_3_SUBSETS_RGB,
    PACK_BC7_FLAG_USE_DUAL_PLANE_RGB, PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA,
    PACK_BC7_FLAG_USE_TRIVIAL_MODE6,
};
use crate::basis::ll::bc7f_astc::pack_from_pixels;
use crate::basis::ll::comp::Image;
use crate::basis::ll::job_pool::map_jobs;
use crate::{Error, Result};

/// `ldr_astc_block_encode_image_fast_4x4`. `cfg` supplies the force-disable flags; see
/// [`super::astcf::encode_image_astcf`] for the other parameters.
pub(crate) fn encode_image_fast_4x4(
    img: &Image,
    cfg: &AstcLdrEncodeConfig,
    use_dct: bool,
    base_q: f32,
    cem_params: &CemEncodeParams,
    threads: u32,
) -> Result<EncodeImageOutput> {
    const BW: u32 = 4;
    const BH: u32 = 4;
    let num_blocks_x = img.get_block_width(BW);
    let num_blocks_y = img.get_block_height(BH);

    let mut flags = PACK_BC7_FLAG_PBIT_OPT
        | PACK_BC7_FLAG_PBIT_OPT_MODE6
        | PACK_BC7_FLAG_USE_TRIVIAL_MODE6
        | PACK_BC7_FLAG_USE_2_SUBSETS_RGB
        | PACK_BC7_FLAG_ASTC_COMPATIBLE
        | PACK_BC7_FLAG_USE_DUAL_PLANE_RGB
        | PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA
        | PACK_BC7_FLAG_USE_3_SUBSETS_RGB
        | PACK_BC7_FLAG_USE_2_SUBSETS_RGBA;
    if cfg.force_disable_subsets {
        flags &= !(PACK_BC7_FLAG_USE_2_SUBSETS_RGB
            | PACK_BC7_FLAG_USE_3_SUBSETS_RGB
            | PACK_BC7_FLAG_USE_2_SUBSETS_RGBA);
    }
    if cfg.force_disable_rgb_dual_plane {
        flags &= !PACK_BC7_FLAG_USE_DUAL_PLANE_RGB;
        flags |= PACK_BC7_FLAG_DISABLE_RGB_DUAL_PLANE;
    }

    let grid_coder = use_dct.then(|| GridWeightDct::new(BW, BH));

    let rows = map_jobs(threads, num_blocks_y as usize, |by| -> Result<_> {
        let mut row = Vec::with_capacity(num_blocks_x as usize);
        for bx in 0..num_blocks_x {
            let mut px = [[0u8; 4]; 16];
            img.extract_block_clamped(&mut px, bx * BW, by as u32 * BH, BW, BH);
            let mut out = EncodeBlockOutput::default();
            if !pack_from_pixels(&mut out.log_blk, &px, flags) {
                return Err(Error::InvalidData);
            }
            if out.log_blk.solid_color_flag_ldr {
                out.trial_mode_index = -1;
                out.sse = 0;
            } else {
                finish_candidate(
                    &mut out,
                    &px,
                    BW,
                    BH,
                    0,
                    grid_coder.as_ref(),
                    base_q,
                    cem_params,
                )?;
            }
            let mut phys = [0u8; 16];
            if !ah::pack_astc_block(&mut phys, &out.log_blk) {
                return Err(Error::InvalidData);
            }
            let info = BlockInfo {
                out_blocks: vec![out],
                ..Default::default()
            };
            row.push((info, phys));
        }
        Ok(row)
    })
    .ok_or(Error::InvalidData)?;

    let mut out = EncodeImageOutput::new(num_blocks_x, num_blocks_y);
    for (by, row) in rows.into_iter().enumerate() {
        for (bx, (info, phys)) in row?.into_iter().enumerate() {
            let i = bx + by * num_blocks_x as usize;
            out.block_info[i] = info;
            out.packed_phys_blocks[i] = phys;
        }
    }
    Ok(out)
}
