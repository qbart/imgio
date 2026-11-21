// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! SCD ("smart candidate deblocking") refinement and other `compress_image` helpers: port of
//! `basisu_astc_ldr_encode.cpp` `calc_block_wsse` .. `refine_output_for_deblocking`,
//! `sharpen_image`, `merge_output_candidates` and `cross_check_dct`.
//!
//! Threading: like the C++ code, the candidate selection passes evaluate one checkerboard color
//! at a time (blocks of one color only read the other color's committed choices and the
//! candidate image, which is updated after the pass), and the mutation passes touch only their
//! own block, so the output does not depend on the thread count.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use super::block_helpers::{deblock_block_region, weighted_color_error};
use super::common::{CemEncodeParams, decode_endpoints};
use super::effort::HighLevelConfig;
use super::fencode::cem_encode;
use super::partitions::{NUM_PART3_MAPPINGS, PART3_MAPPING, PartitionsData};
use super::rand::BasisuRand;
use super::{AstcLdrEncodeConfig, BLUR_ID_EXP, DctSyms, EncodeBlockOutput, EncodeImageOutput};
use crate::astc::helpers::{
    self as ah, CEM_HDR_RGB_HDR_ALPHA, CEM_HDR_RGB_LDR_ALPHA, CEM_LDR_LUM_ALPHA_BASE_PLUS_OFS,
    CEM_LDR_LUM_ALPHA_DIRECT, CEM_LDR_LUM_DIRECT, CEM_LDR_RGB_BASE_PLUS_OFFSET,
    CEM_LDR_RGB_BASE_SCALE, CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A, CEM_LDR_RGB_DIRECT,
    CEM_LDR_RGBA_BASE_PLUS_OFFSET, CEM_LDR_RGBA_DIRECT, DecodePixels, LogAstcBlock,
    MAX_BLOCK_PIXELS,
};
use crate::basis::ll::comp::Image;
use crate::basis::ll::enc::{ColorRgba, hash_hsieh};
use crate::basis::ll::job_pool::{for_each_chunk_mut, map_jobs};
use crate::basis::xuastc::dct as xdct;
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

// ---- convars (default values) ----

/// `g_astc_refine_cross_block_penalty_weight`.
const ASTC_REFINE_CROSS_BLOCK_PENALTY_WEIGHT: f32 = 2.5;
/// `g_astc_refine_max_new_blocks`.
const ASTC_REFINE_MAX_NEW_BLOCKS: i32 = 16;
/// `g_astc_refine_try_nonprimary_candidates`.
const ASTC_REFINE_TRY_NONPRIMARY_CANDIDATES: bool = true;
/// `g_astc_refine_mutate_part_id_prob`.
const ASTC_REFINE_MUTATE_PART_ID_PROB: i32 = 10;
/// `g_astc_refine_mutate_endpoint_edge`.
const ASTC_REFINE_MUTATE_ENDPOINT_EDGE: bool = true;
/// `g_astc_refine_mutate_dct`.
const ASTC_REFINE_MUTATE_DCT: bool = true;
/// `g_astc_refine_mutate_endpoints`.
const ASTC_REFINE_MUTATE_ENDPOINTS: bool = true;
/// `g_astc_refine_base_seed`.
const ASTC_REFINE_BASE_SEED: i32 = 0;

/// `basist::astc_ldr_t::DCT_MAX_ARITH_COEFF_MAG`.
const DCT_MAX_ARITH_COEFF_MAG: i32 = 255;

/// `NUM_SIMILAR_PATS`.
const NUM_SIMILAR_PATS: usize = 16;

type Vec4F = [f32; 4];

// ---- small helpers (basisu image / astc_helpers methods) ----

/// `color_rgba::get_vec4F`.
#[inline]
fn vec4(c: ColorRgba) -> Vec4F {
    [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32]
}

/// `image::blit_clamped`.
fn blit_clamped(
    dst: &mut Image,
    src: &Image,
    src_x: i32,
    src_y: i32,
    src_w: i32,
    src_h: i32,
    dst_x: i32,
    dst_y: i32,
) {
    for y in 0..src_h {
        for x in 0..src_w {
            dst.set_clipped(dst_x + x, dst_y + y, src.get_clamped(src_x + x, src_y + y));
        }
    }
}

/// `image::set_block_clipped`.
fn set_block_clipped(dst: &mut Image, src: &[ColorRgba], dst_x: u32, dst_y: u32, w: u32, h: u32) {
    let mut i = 0;
    for y in 0..h {
        for x in 0..w {
            dst.set_clipped((dst_x + x) as i32, (dst_y + y) as i32, src[i]);
            i += 1;
        }
    }
}

/// `image::blit(src, src_x, src_y, src_w, src_h, dst_x, dst_y, set_clipped_to_black = true)`.
fn blit_black(
    dst: &mut Image,
    src: &Image,
    src_x: i32,
    src_y: i32,
    src_w: i32,
    src_h: i32,
    dst_x: i32,
    dst_y: i32,
) {
    for y in 0..src_h {
        let sy = src_y + y;
        for x in 0..src_w {
            let sx = src_x + x;
            let clipped =
                sx < 0 || sx >= src.get_width() as i32 || sy < 0 || sy >= src.get_height() as i32;
            let c = if clipped {
                [0, 0, 0, 255]
            } else {
                src.get(sx as u32, sy as u32)
            };
            dst.set_clipped(dst_x + x, dst_y + y, c);
        }
    }
}

/// `astc_helpers::decode_block_xuastc_ldr` (equals `decode_block` for the LDR modes).
fn decode_xuastc(
    log_blk: &LogAstcBlock,
    pixels: &mut [ColorRgba],
    block_width: u32,
    block_height: u32,
    srgb: bool,
) -> bool {
    let mut px = if srgb {
        DecodePixels::Srgb8(pixels)
    } else {
        DecodePixels::Ldr8(pixels)
    };
    ah::decode_block(log_blk, &mut px, block_width, block_height)
}

/// `astc_helpers::cem_is_ldr_base_scale`.
#[inline]
fn cem_is_ldr_base_scale(cem_index: u32) -> bool {
    cem_index == CEM_LDR_RGB_BASE_SCALE || cem_index == CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A
}

/// `astc_helpers::does_cem_have_alpha`.
#[inline]
fn does_cem_have_alpha(mode: u32) -> bool {
    matches!(
        mode,
        CEM_LDR_LUM_ALPHA_DIRECT
            | CEM_LDR_LUM_ALPHA_BASE_PLUS_OFS
            | CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A
            | CEM_LDR_RGBA_DIRECT
            | CEM_LDR_RGBA_BASE_PLUS_OFFSET
            | CEM_HDR_RGB_LDR_ALPHA
            | CEM_HDR_RGB_HDR_ALPHA
    )
}

/// `basisu::clamp` (never panics, unlike `Ord::clamp` with `lo > hi`).
#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `basisu::fast_roundf_int`.
#[inline]
fn fast_roundf_int(x: f32) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    }
}

/// `grid_weight_dct::decode_block_weights(q, plane_index, log_blk, nullptr, nullptr, dct_work,
/// &syms)` for a `block_width` x `block_height` grid coder (symbol path).
pub(crate) fn decode_block_weights_syms(
    block_width: u32,
    block_height: u32,
    q: f32,
    plane_index: u32,
    log_blk: &mut LogAstcBlock,
    syms: &DctSyms,
) -> bool {
    let mut s = xdct::DctSyms {
        dc_sym: syms.dc_sym,
        ..Default::default()
    };
    // Every coefficient advances the zigzag index by at least one, so the C++ loop always stops
    // within the first 64 entries.
    let n = syms.coeffs.len().min(s.coeffs.len());
    for (d, c) in s.coeffs.iter_mut().zip(&syms.coeffs[..n]) {
        d.num_zeros = c.num_zeros;
        d.coeff = c.coeff;
    }
    s.num_coeffs = n;
    xdct::decode_block_weights(block_width, block_height, q, plane_index, log_blk, &s)
}

// ---- error metrics ----

/// `calc_block_wsse`.
fn calc_block_wsse(
    orig_img: &Image,
    sx: i32,
    sy: i32,
    block_width: i32,
    block_height: i32,
    tile_img: &Image,
    tx: i32,
    ty: i32,
    p: &CemEncodeParams,
) -> u64 {
    let mut total_err = 0u64;
    for y in 0..block_height {
        let oy = sy + y;
        if oy < 0 || oy >= orig_img.get_height() as i32 {
            continue;
        }
        for x in 0..block_width {
            let ox = sx + x;
            if ox < 0 || ox >= orig_img.get_width() as i32 {
                continue;
            }
            total_err += u64::from(weighted_color_error(
                &orig_img.get(ox as u32, oy as u32),
                &tile_img.get((tx + x) as u32, (ty + y) as u32),
                p,
            ));
        }
    }
    total_err
}

/// `calc_color_delta`.
#[inline]
fn calc_color_delta(a: ColorRgba, b: ColorRgba) -> Vec4F {
    let d = |c: usize| (i32::from(a[c]) - i32::from(b[c])) as f32;
    [d(0), d(1), d(2), d(3)]
}

/// `calc_penalty`.
#[inline]
fn calc_penalty(orig: &Vec4F, cand: &Vec4F, comp_weights: &Vec4F) -> f64 {
    let mut delta = [0.0f32; 4];
    for i in 0..4 {
        let d = orig[i] - cand[i];
        delta[i] = (d * d) * comp_weights[i];
    }
    f64::from(delta[0] + delta[1] + delta[2] + delta[3])
}

/// `calc_cross_block_boundary_delta_mismatch`.
fn calc_cross_block_boundary_delta_mismatch(
    orig_img: &Image,
    candidate_img: &Image,
    block_width: u32,
    block_height: u32,
    bx: u32,
    by: u32,
    p: &CemEncodeParams,
) -> u64 {
    let (ofs_x, ofs_y) = ((block_width * bx) as i32, (block_height * by) as i32);
    let (bw, bh) = (block_width as i32, block_height as i32);
    let comp_weights = [
        p.comp_weights[0] as f32,
        p.comp_weights[1] as f32,
        p.comp_weights[2] as f32,
        p.comp_weights[3] as f32,
    ];
    let delta = |img: &Image, x0: i32, y0: i32, x1: i32, y1: i32| {
        calc_color_delta(img.get_clamped(x0, y0), img.get_clamped(x1, y1))
    };

    let mut penalty = 0.0f64;
    for x in 0..bw {
        let (px, ty, by_) = (ofs_x + x, ofs_y, ofs_y + bh - 1);
        let orig_delta_top = delta(orig_img, px, ty, px, ty - 1);
        let cand_delta_top = delta(candidate_img, px, ty, px, ty - 1);
        penalty += calc_penalty(&orig_delta_top, &cand_delta_top, &comp_weights);

        let orig_delta_bot = delta(orig_img, px, by_, px, by_ + 1);
        let cand_delta_bot = delta(candidate_img, px, by_, px, by_ + 1);
        penalty += calc_penalty(&orig_delta_bot, &cand_delta_bot, &comp_weights);
    }
    for y in 0..bh {
        let (py, lx, rx) = (ofs_y + y, ofs_x, ofs_x + bw - 1);
        let orig_delta_left = delta(orig_img, lx, py, lx - 1, py);
        let cand_delta_left = delta(candidate_img, lx, py, lx - 1, py);
        penalty += calc_penalty(&orig_delta_left, &cand_delta_left, &comp_weights);

        let orig_delta_right = delta(orig_img, rx, py, rx + 1, py);
        let cand_delta_right = delta(candidate_img, rx, py, rx + 1, py);
        penalty += calc_penalty(&orig_delta_right, &cand_delta_right, &comp_weights);
    }

    (penalty * f64::from(ASTC_REFINE_CROSS_BLOCK_PENALTY_WEIGHT)).round() as u64
}

/// `calc_ycbcr`: (Y, Cb, Cr) of the first three components.
#[inline]
fn calc_ycbcr(c: &Vec4F) -> [f32; 3] {
    let (r, g, b) = (c[0], c[1], c[2]);
    let y = r * 0.212600f32 + g * 0.715200f32 + b * 0.072200f32;
    let cb = r * -0.114572f32 + g * -0.385428f32 + b * 0.500000f32;
    let cr = r * 0.500000f32 + g * -0.454153f32 + b * -0.045847f32;
    [y, cb, cr]
}

/// `calc_chroma_loss_penalty`.
fn calc_chroma_loss_penalty(
    orig_img: &Image,
    candidate_img: &Image,
    block_width: u32,
    block_height: u32,
    bx: u32,
    by: u32,
    p: &CemEncodeParams,
) -> u64 {
    let mut orig_block = [[0u8; 4]; MAX_BLOCK_PIXELS];
    let mut cand_block = [[0u8; 4]; MAX_BLOCK_PIXELS];
    orig_img.extract_block_clamped(
        &mut orig_block,
        bx * block_width,
        by * block_height,
        block_width,
        block_height,
    );
    candidate_img.extract_block_clamped(
        &mut cand_block,
        bx * block_width,
        by * block_height,
        block_width,
        block_height,
    );

    let total_block_pixels = block_width * block_height;
    let (mut avg_orig, mut avg_cand) = ([0.0f32; 4], [0.0f32; 4]);
    for i in 0..total_block_pixels as usize {
        let (o, c) = (vec4(orig_block[i]), vec4(cand_block[i]));
        for k in 0..4 {
            avg_orig[k] += o[k];
            avg_cand[k] += c[k];
        }
    }
    let s = 1.0f32 / total_block_pixels as f32;
    for k in 0..4 {
        avg_orig[k] *= s;
        avg_cand[k] *= s;
    }

    let o = calc_ycbcr(&avg_orig);
    let c = calc_ycbcr(&avg_cand);
    let delta = [o[0] - c[0], o[1] - c[1], o[2] - c[2]];
    let mut penalty = (delta[1] * delta[1]) + (delta[2] * delta[2]);
    let penalty_weight =
        ((total_block_pixels as f32 * 0.25f32) * p.comp_weights[1] as f32) * (14.0f32 * 14.0f32);
    penalty *= penalty_weight;
    penalty.round() as u64
}

/// `calc_block_sse`.
fn calc_block_sse(
    block_width: u32,
    block_height: u32,
    block_pixels: &[ColorRgba],
    log_astc_blk: &LogAstcBlock,
    enc_cfg: &HighLevelConfig,
) -> u64 {
    let mut dec_pixels = [[0u8; 4]; MAX_BLOCK_PIXELS];
    if !decode_xuastc(
        log_astc_blk,
        &mut dec_pixels,
        block_width,
        block_height,
        enc_cfg.cem_enc_params.decode_mode_srgb,
    ) {
        return u64::MAX;
    }
    let total_block_pixels = (block_width * block_height) as usize;
    let mut total_err = 0u64;
    for i in 0..total_block_pixels {
        total_err += u64::from(weighted_color_error(
            &block_pixels[i],
            &dec_pixels[i],
            &enc_cfg.cem_enc_params,
        ));
    }
    total_err
}

/// `deblocking_thread_state`.
struct DeblockingThreadState {
    /// bw*3 x bh*3
    deblock_orig: Image,
    /// bw*3 x bh*3
    deblock_staging: Image,
    /// (bw+2) x (bh+2)
    deblock_temp: Image,
}

impl DeblockingThreadState {
    fn new(block_width: u32, block_height: u32) -> Self {
        Self {
            deblock_orig: Image::with_size(block_width * 3, block_height * 3),
            deblock_staging: Image::with_size(block_width * 3, block_height * 3),
            deblock_temp: Image::with_size(block_width + 2, block_height + 2),
        }
    }
}

/// Whether base+scale candidates of block (bx, by) get penalized (shared by
/// `deblocking_find_best_candidate` and `deblocking_compute_error`). The neighbor coordinates
/// reproduce the C++ code exactly: horizontal neighbors use row 0 and vertical ones column 0.
fn penalize_isolated_base_ofs(
    num_blocks_x: u32,
    num_blocks_y: u32,
    bx: u32,
    by: u32,
    enc_out: &EncodeImageOutput,
) -> bool {
    let (mut total_neighbors_base_ofs, mut total_neighbors_examined) = (0u32, 0u32);
    for k in 0..4 {
        let (mut nbx, mut nby) = (0i32, 0i32);
        if k < 2 {
            nbx = bx as i32 + if k & 1 != 0 { -1 } else { 1 };
        } else {
            nby = by as i32 + if (k - 2) & 1 != 0 { -1 } else { 1 };
        }
        if nbx < 0 || nbx >= num_blocks_x as i32 {
            continue;
        }
        if nby < 0 || nby >= num_blocks_y as i32 {
            continue;
        }
        let nbi = enc_out.block(nbx as u32, nby as u32);
        let neighbor_block = &nbi.out_blocks[nbi.packed_out_block_index as usize].log_blk;
        if neighbor_block.solid_color_flag_ldr {
            continue;
        }
        if cem_is_ldr_base_scale(u32::from(neighbor_block.color_endpoint_modes[0])) {
            total_neighbors_base_ofs += 1;
        }
        total_neighbors_examined += 1;
    }
    if total_neighbors_examined == 0 {
        return false;
    }
    const PENALIZE_ISOLATED_BASE_OFS_FRACT: u32 = 8; // div 16
    let neighbors_base_ofs_fract = (total_neighbors_base_ofs << 4) / total_neighbors_examined;
    neighbors_base_ofs_fract < PENALIZE_ISOLATED_BASE_OFS_FRACT
}

/// `deblocking_find_best_candidate`: returns `(new_best_packed_block_index, best_err)`.
fn deblocking_find_best_candidate(
    block_width: u32,
    block_height: u32,
    num_blocks_x: u32,
    num_blocks_y: u32,
    bx: u32,
    by: u32,
    thread_state: &mut DeblockingThreadState,
    candidate_img: &Image,
    orig_img: &Image,
    global_cfg: &AstcLdrEncodeConfig,
    enc_cfg: &HighLevelConfig,
    enc_out: &EncodeImageOutput,
) -> Result<(u32, u64)> {
    let srgb = global_cfg.astc_decode_mode_srgb;
    let blk_info = enc_out.block(bx, by);
    let out_blocks = &blk_info.out_blocks;
    let packed = blk_info.packed_out_block_index;

    let cur_sse = out_blocks
        .get(packed as usize)
        .ok_or(Error::InvalidData)?
        .sse;
    if cur_sse == 0 {
        return Ok((packed, 0));
    }

    // if a candidate has a WSSE worse than this, ignore it
    let skip_sse_thresh = u64::MAX;

    let mut best_err = u64::MAX;
    let mut best_cand_index = 0u32;
    let (bw, bh) = (block_width as i32, block_height as i32);

    // grab the 3x3 block region around the block from the original image and the current output
    blit_clamped(
        &mut thread_state.deblock_orig,
        orig_img,
        (bx as i32 - 1) * bw,
        (by as i32 - 1) * bh,
        bw * 3,
        bh * 3,
        0,
        0,
    );
    blit_clamped(
        &mut thread_state.deblock_staging,
        candidate_img,
        (bx as i32 - 1) * bw,
        (by as i32 - 1) * bh,
        bw * 3,
        bh * 3,
        0,
        0,
    );

    let penalize_isolated_base_ofs_blocks =
        penalize_isolated_base_ofs(num_blocks_x, num_blocks_y, bx, by, enc_out);

    let mut block_pixels = [[0u8; 4]; MAX_BLOCK_PIXELS];
    let p = &enc_cfg.cem_enc_params;

    for (cand_index, out_blk) in out_blocks.iter().enumerate() {
        if out_blk.sse > skip_sse_thresh {
            continue;
        }
        if !decode_xuastc(
            &out_blk.log_blk,
            &mut block_pixels,
            block_width,
            block_height,
            srgb,
        ) {
            return Err(Error::InvalidData);
        }

        // insert the candidate block into the center of the 3x3 block staging image
        set_block_clipped(
            &mut thread_state.deblock_staging,
            &block_pixels,
            block_width,
            block_height,
            block_width,
            block_height,
        );

        let block_wsse = if global_cfg.scd_will_postfilter {
            deblock_block_region(
                bw,
                bh,
                &thread_state.deblock_staging,
                bw,
                bh,
                &mut thread_state.deblock_temp,
            );
            calc_block_wsse(
                &thread_state.deblock_orig,
                bw - 1,
                bh - 1,
                bw + 2,
                bh + 2,
                &thread_state.deblock_temp,
                0,
                0,
                p,
            )
        } else {
            let mut block_wsse = 0u64;
            for y in 0..bh {
                for x in 0..bw {
                    block_wsse += u64::from(weighted_color_error(
                        &thread_state.deblock_orig.get_clamped(bw + x, bh + y),
                        &block_pixels[(x + y * bw) as usize],
                        p,
                    ));
                }
            }
            // boost this wsse so the other penalties (tuned with filtering enabled) are roughly
            // relative to it
            let num_filtered_texels = u64::from((block_width + 2) * (block_height + 2));
            let num_unfiltered_texels = u64::from(block_width * block_height);
            (block_wsse * num_filtered_texels) / num_unfiltered_texels
        };

        let block_artifact_penalty_wsse = calc_cross_block_boundary_delta_mismatch(
            &thread_state.deblock_orig,
            &thread_state.deblock_staging,
            block_width,
            block_height,
            1,
            1,
            p,
        );
        let chroma_loss_penalty_wsse = if global_cfg.scd_preserve_chroma {
            calc_chroma_loss_penalty(
                &thread_state.deblock_orig,
                &thread_state.deblock_staging,
                block_width,
                block_height,
                1,
                1,
                p,
            )
        } else {
            0
        };

        let mut cand_err = block_wsse;
        if penalize_isolated_base_ofs_blocks
            && !out_blk.log_blk.solid_color_flag_ldr
            && cem_is_ldr_base_scale(u32::from(out_blk.log_blk.color_endpoint_modes[0]))
        {
            // penalize base-ofs candidates if there are not enough base-ofs neighbors
            cand_err = cand_err.wrapping_mul(5) / 4;
        }
        cand_err = cand_err
            .wrapping_add(block_artifact_penalty_wsse)
            .wrapping_add(chroma_loss_penalty_wsse);

        if cand_index as u32 != packed && out_blk.log_blk.solid_color_flag_ldr {
            // don't switch to solid unless it REALLY seems to help
            cand_err = cand_err.wrapping_mul(8);
        }

        if cand_err < best_err {
            best_err = cand_err;
            best_cand_index = cand_index as u32;
        }
    }

    Ok((best_cand_index, best_err))
}

/// `deblocking_compute_error` (debug statistics only).
fn deblocking_compute_error(
    block_width: u32,
    block_height: u32,
    num_blocks_x: u32,
    num_blocks_y: u32,
    bx: u32,
    by: u32,
    candidate_img: &mut Image,
    orig_img: &Image,
    global_cfg: &AstcLdrEncodeConfig,
    enc_cfg: &HighLevelConfig,
    enc_out: &EncodeImageOutput,
    deblock_orig: &mut Image,
    deblock_temp: &mut Image,
) -> Result<u64> {
    let srgb = global_cfg.astc_decode_mode_srgb;
    let blk_info = enc_out.block(bx, by);
    let (bw, bh) = (block_width as i32, block_height as i32);
    let (sx, sy) = ((bx * block_width) as i32, (by * block_height) as i32);

    blit_black(deblock_orig, candidate_img, sx, sy, bw, bh, 0, 0);

    let penalize_isolated_base_ofs_blocks =
        penalize_isolated_base_ofs(num_blocks_x, num_blocks_y, bx, by, enc_out);

    let out_blk = blk_info
        .out_blocks
        .get(blk_info.packed_out_block_index as usize)
        .ok_or(Error::InvalidData)?;
    let mut block_pixels = [[0u8; 4]; MAX_BLOCK_PIXELS];
    if !decode_xuastc(
        &out_blk.log_blk,
        &mut block_pixels,
        block_width,
        block_height,
        srgb,
    ) {
        return Err(Error::InvalidData);
    }
    set_block_clipped(
        candidate_img,
        &block_pixels,
        bx * block_width,
        by * block_height,
        block_width,
        block_height,
    );

    let p = &enc_cfg.cem_enc_params;
    let block_wsse = if global_cfg.scd_will_postfilter {
        deblock_block_region(bw, bh, candidate_img, sx, sy, deblock_temp);
        calc_block_wsse(
            orig_img,
            sx - 1,
            sy - 1,
            bw + 2,
            bh + 2,
            deblock_temp,
            0,
            0,
            p,
        )
    } else {
        let mut block_wsse = 0u64;
        for y in 0..bh {
            for x in 0..bw {
                block_wsse += u64::from(weighted_color_error(
                    &orig_img.get_clamped(sx + x, sy + y),
                    &block_pixels[(x + y * bw) as usize],
                    p,
                ));
            }
        }
        let num_filtered_texels = u64::from((block_width + 2) * (block_height + 2));
        let num_unfiltered_texels = u64::from(block_width * block_height);
        (block_wsse * num_filtered_texels) / num_unfiltered_texels
    };

    let block_artifact_penalty_wsse = calc_cross_block_boundary_delta_mismatch(
        orig_img,
        candidate_img,
        block_width,
        block_height,
        bx,
        by,
        p,
    );
    let chroma_loss_penalty_wsse = if global_cfg.scd_preserve_chroma {
        calc_chroma_loss_penalty(
            orig_img,
            candidate_img,
            block_width,
            block_height,
            bx,
            by,
            p,
        )
    } else {
        0
    };

    let mut cand_err = block_wsse;
    if penalize_isolated_base_ofs_blocks
        && !out_blk.log_blk.solid_color_flag_ldr
        && cem_is_ldr_base_scale(u32::from(out_blk.log_blk.color_endpoint_modes[0]))
    {
        cand_err = cand_err.wrapping_mul(5) / 4;
    }
    cand_err = cand_err
        .wrapping_add(block_artifact_penalty_wsse)
        .wrapping_add(chroma_loss_penalty_wsse);

    blit_black(candidate_img, deblock_orig, 0, 0, bw, bh, sx, sy);

    Ok(cand_err)
}

/// `deblocking_compute_overall_error` (debug statistics only).
fn deblocking_compute_overall_error(
    block_width: u32,
    block_height: u32,
    num_blocks_x: u32,
    num_blocks_y: u32,
    orig_img: &Image,
    global_cfg: &AstcLdrEncodeConfig,
    enc_cfg: &HighLevelConfig,
    enc_out: &EncodeImageOutput,
) -> Result<u64> {
    let (width, height) = (orig_img.get_width(), orig_img.get_height());
    let mut candidate_img = Image::with_size(width, height);
    let srgb = global_cfg.astc_decode_mode_srgb;
    let mut block_pixels = [[0u8; 4]; MAX_BLOCK_PIXELS];
    for by in 0..num_blocks_y {
        for bx in 0..num_blocks_x {
            let blk_info = enc_out.block(bx, by);
            let best_blk = blk_info
                .out_blocks
                .get(blk_info.packed_out_block_index as usize)
                .ok_or(Error::InvalidData)?;
            if !decode_xuastc(
                &best_blk.log_blk,
                &mut block_pixels,
                block_width,
                block_height,
                srgb,
            ) {
                return Err(Error::InvalidData);
            }
            set_block_clipped(
                &mut candidate_img,
                &block_pixels,
                bx * block_width,
                by * block_height,
                block_width,
                block_height,
            );
        }
    }

    let mut deblock_orig = Image::with_size(block_width, block_height);
    let mut deblock_temp = Image::with_size(block_width + 2, block_height + 2);
    let mut overall_err = 0u64;
    for by in 0..num_blocks_y {
        for bx in 0..num_blocks_x {
            overall_err = overall_err.wrapping_add(deblocking_compute_error(
                block_width,
                block_height,
                num_blocks_x,
                num_blocks_y,
                bx,
                by,
                &mut candidate_img,
                orig_img,
                global_cfg,
                enc_cfg,
                enc_out,
                &mut deblock_orig,
                &mut deblock_temp,
            )?);
        }
    }
    Ok(overall_err)
}

/// `similar_pats[2]` / `similar_pat_perm_index[2]` of `refine_output_for_deblocking`: for each
/// ASTC partition seed, up to [`NUM_SIMILAR_PATS`] similar seeds (`u16::MAX` = none) and the
/// best subset permutation index.
struct SimilarPats {
    /// `[num_subsets - 2][seed][r]`
    pats: [Vec<[u16; NUM_SIMILAR_PATS]>; 2],
    perm_index: [Vec<[u8; NUM_SIMILAR_PATS]>; 2],
}

impl SimilarPats {
    fn new(part_data_p2: &PartitionsData, part_data_p3: &PartitionsData) -> Result<Self> {
        let mut s = Self {
            pats: [
                vec![[u16::MAX; NUM_SIMILAR_PATS]; 1024],
                vec![[u16::MAX; NUM_SIMILAR_PATS]; 1024],
            ],
            perm_index: [
                vec![[0u8; NUM_SIMILAR_PATS]; 1024],
                vec![[0u8; NUM_SIMILAR_PATS]; 1024],
            ],
        };
        for num_subsets in 2..=3u32 {
            let pat_data = if num_subsets == 2 {
                part_data_p2
            } else {
                part_data_p3
            };
            let si = (num_subsets - 2) as usize;
            let num_unique_subset_pats = pat_data.total_unique_patterns;
            for unique_pat_index in 0..num_unique_subset_pats {
                let seed_index = pat_data.unique_index_to_part_seed[unique_pat_index as usize];
                let seed_index = usize::try_from(seed_index).map_err(|_| Error::InvalidData)?;
                let desired_pat_vec = pat_data
                    .partition_pats
                    .get(unique_pat_index as usize)
                    .ok_or(Error::InvalidData)?;

                let mut results = [0u32; NUM_SIMILAR_PATS + 1];
                let num_results = pat_data.part_lhs_map.find(
                    desired_pat_vec,
                    &mut results,
                    (NUM_SIMILAR_PATS + 1) as u32,
                    false,
                );

                let mut dst_index = 0usize;
                for &similar_unique_pat_index in results.iter().take(num_results as usize) {
                    if similar_unique_pat_index == unique_pat_index {
                        continue;
                    }
                    let similar_seed = pat_data
                        .unique_index_to_part_seed
                        .get(similar_unique_pat_index as usize)
                        .ok_or(Error::InvalidData)?;
                    s.pats[si][seed_index][dst_index] = *similar_seed as u16;

                    let total_perms = if num_subsets == 2 {
                        2
                    } else {
                        NUM_PART3_MAPPINGS as u32
                    };
                    let (mut best_dist, mut best_perm_index) = (u32::MAX, 0u32);
                    let similar_pat_vec = pat_data
                        .partition_pats
                        .get(similar_unique_pat_index as usize)
                        .ok_or(Error::InvalidData)?;
                    for perm_index in 0..total_perms {
                        let permuted = if num_subsets == 2 {
                            desired_pat_vec.get_permuted2(perm_index)
                        } else {
                            desired_pat_vec.get_permuted3(perm_index)
                        };
                        let dist = permuted.get_squared_distance(similar_pat_vec) as u32;
                        if dist < best_dist {
                            best_dist = dist;
                            best_perm_index = perm_index;
                        }
                    }
                    s.perm_index[si][seed_index][dst_index] = best_perm_index as u8;

                    dst_index += 1;
                    if dst_index >= NUM_SIMILAR_PATS {
                        break;
                    }
                }
            }
        }
        Ok(s)
    }
}

/// Read-only inputs of the mutation pass.
struct MutateCtx<'a> {
    pass: u32,
    block_width: u32,
    block_height: u32,
    num_blocks_x: u32,
    max_new_blocks: u32,
    block_std_dev: &'a [f32],
    similar: &'a SimilarPats,
    orig_img: &'a Image,
    candidate_img: &'a Image,
    global_cfg: &'a AstcLdrEncodeConfig,
    enc_cfg: &'a HighLevelConfig,
    rnd_mag: f32,
}

/// One block of `mutate_candidates` (the body of the C++ job loop). `false` on failure.
fn mutate_block(
    ctx: &MutateCtx,
    block_index: u32,
    blk_info: &mut super::BlockInfo,
    sorted_candidates: &mut Vec<u64>,
) -> bool {
    let (block_width, block_height) = (ctx.block_width, ctx.block_height);
    let pass = ctx.pass;
    let bx = block_index % ctx.num_blocks_x;
    let by = block_index / ctx.num_blocks_x;

    sorted_candidates.clear();

    // first retire old mutated candidates (older than 4 passes)
    {
        let cur_out_blocks = std::mem::take(&mut blk_info.out_blocks);
        let mut new_out_blocks: Vec<EncodeBlockOutput> =
            Vec::with_capacity(1 + cur_out_blocks.len() / 2);
        let mut new_packed_out_block_index: i64 = -1;
        for (i, blk) in cur_out_blocks.into_iter().enumerate() {
            let mut should_remove = false;
            if i as u32 == blk_info.packed_out_block_index {
                new_packed_out_block_index = new_out_blocks.len() as i64;
            } else if u32::from(blk.blur_id) >= u32::from(BLUR_ID_EXP) {
                let blk_pass = i32::from(blk.blur_id) - i32::from(BLUR_ID_EXP);
                if blk_pass <= pass as i32 - 4 {
                    should_remove = true;
                }
            }
            if !should_remove {
                sorted_candidates.push((blk.sse << 16) | new_out_blocks.len() as u64);
                new_out_blocks.push(blk);
            }
        }
        if new_packed_out_block_index < 0 {
            return false;
        }
        blk_info.packed_out_block_index = new_packed_out_block_index as u32;
        blk_info.out_blocks = new_out_blocks;
    }

    // keys are unique (the low 16 bits hold the index)
    sorted_candidates.sort_unstable();

    // generate new candidates via mutation
    let h: u64 = 1u64
        .wrapping_add(u64::from(pass).wrapping_mul(4096 * 4096))
        .wrapping_add(u64::from(block_index));
    let mut rnd = BasisuRand::new();
    rnd.seed((ASTC_REFINE_BASE_SEED as u32).wrapping_add(hash_hsieh(&h.to_le_bytes())));

    let packed = blk_info.packed_out_block_index as usize;
    if blk_info.out_blocks[packed].log_blk.solid_color_flag_ldr {
        return true;
    }

    let mut orig_block = [[0u8; 4]; MAX_BLOCK_PIXELS];
    ctx.orig_img.extract_block_clamped(
        &mut orig_block,
        bx * block_width,
        by * block_height,
        block_width,
        block_height,
    );

    let std_dev = ctx.block_std_dev[block_index as usize];
    let num_rand_blocks = clampi(
        (std_dev * 0.25f32).round() as i32,
        1,
        ctx.max_new_blocks as i32,
    ) as u32
        + 2;

    let tabs = ah::dequant_tables();

    for q in 0..num_rand_blocks {
        let mut best_index = packed;
        let mut choose_nonprimary_cand_flag = false;

        if ASTC_REFINE_TRY_NONPRIMARY_CANDIDATES
            && q >= num_rand_blocks - 2
            && sorted_candidates.len() >= 2
        {
            let n = sorted_candidates.len() as u32;
            let ix = rnd.irand(0, 10u32.min(n) as i32 - 1) as u32;
            let mut rnd_cand_index = (sorted_candidates[ix as usize] & 0xFFFF) as usize;
            best_index = rnd_cand_index;
            if blk_info.out_blocks[best_index].log_blk.solid_color_flag_ldr {
                rnd_cand_index = (sorted_candidates[((ix + 1) % n) as usize] & 0xFFFF) as usize;
                best_index = rnd_cand_index;
            }
            choose_nonprimary_cand_flag = true;
        }

        if blk_info.out_blocks[best_index].log_blk.solid_color_flag_ldr {
            continue;
        }
        let best_blk = blk_info.out_blocks[best_index].clone();

        let log_blk = &best_blk.log_blk;
        let cem_index = u32::from(log_blk.color_endpoint_modes[0]);
        let num_cem_vals = ah::get_num_cem_values(cem_index) as usize;
        let mut new_log_blk = *log_blk;
        let endpoint_tab = tabs.endpoint_tab(u32::from(log_blk.endpoint_ise_range));

        let mut any_changed = false;
        let mut new_packed_dct_plane_data: [DctSyms; 2] = Default::default();
        let mut has_new_packed_dct_plane_data = false;

        if choose_nonprimary_cand_flag {
            new_log_blk.user_mode |= 8;
        }

        if ASTC_REFINE_MUTATE_PART_ID_PROB != 0
            && new_log_blk.num_partitions >= 2
            && rnd.iprob(ASTC_REFINE_MUTATE_PART_ID_PROB)
        {
            // partition pattern mutation
            let r = rnd.irand(0, NUM_SIMILAR_PATS as i32 - 1) as usize;
            let si = (new_log_blk.num_partitions - 2) as usize;
            let Some(k) = ctx.similar.pats[si]
                .get(new_log_blk.partition_id as usize)
                .map(|v| v[r])
            else {
                return false;
            };
            if k != u16::MAX {
                new_log_blk.partition_id = k;
                // (indexed by the NEW partition id, as in the C++ code)
                let perm_index = ctx.similar.perm_index[si][k as usize][r] as usize;
                // now permute the endpoints if necessary so they roughly match
                if perm_index != 0 {
                    let np = new_log_blk.num_partitions as usize;
                    let mut cur_endpoints = [0u8; ah::MAX_ENDPOINTS as usize];
                    cur_endpoints[..num_cem_vals * np]
                        .copy_from_slice(&new_log_blk.endpoints[..num_cem_vals * np]);
                    for src_index in 0..np {
                        let dst_index = if np == 2 {
                            src_index ^ 1
                        } else {
                            PART3_MAPPING[perm_index][src_index] as usize
                        };
                        new_log_blk.endpoints
                            [dst_index * num_cem_vals..(dst_index + 1) * num_cem_vals]
                            .copy_from_slice(
                                &cur_endpoints
                                    [src_index * num_cem_vals..(src_index + 1) * num_cem_vals],
                            );
                    }
                }
                new_log_blk.user_mode |= 1;
                any_changed = true;
            }
        }

        if !any_changed && ASTC_REFINE_MUTATE_ENDPOINT_EDGE && rnd.iprob(10) {
            // lerp endpoint towards adjacent block edge mutation
            let subset_index = if new_log_blk.num_partitions == 1 {
                0
            } else {
                rnd.irand(0, i32::from(new_log_blk.num_partitions) - 1) as usize
            };

            let mut avg_edge = [0.0f32; 4];
            let edge_index = rnd.irand(0, 3);
            let add = |avg: &mut Vec4F, c: ColorRgba| {
                let v = vec4(c);
                for i in 0..4 {
                    avg[i] += v[i];
                }
            };
            if edge_index == 0 || edge_index == 2 {
                // top (0) or bottom (2)
                for x in 0..block_width {
                    let src_x = (bx * block_width + x) as i32;
                    let src_y = (by as i32 * block_height as i32)
                        + if edge_index == 0 {
                            -1
                        } else {
                            block_height as i32
                        };
                    add(&mut avg_edge, ctx.candidate_img.get_clamped(src_x, src_y));
                }
                let s = 1.0f32 / block_width as f32;
                avg_edge.iter_mut().for_each(|v| *v *= s);
            } else {
                // right (1) or left (3)
                for y in 0..block_height {
                    let src_x = (bx as i32 * block_width as i32)
                        + if edge_index == 1 {
                            block_width as i32
                        } else {
                            -1
                        };
                    let src_y = (by * block_height + y) as i32;
                    add(&mut avg_edge, ctx.candidate_img.get_clamped(src_x, src_y));
                }
                let s = 1.0f32 / block_height as f32;
                avg_edge.iter_mut().for_each(|v| *v *= s);
            }

            let e0 = num_cem_vals * subset_index;
            let (mut le, mut he) = ([0u8; 4], [0u8; 4]);
            decode_endpoints(
                cem_index,
                &new_log_blk.endpoints[e0..],
                u32::from(new_log_blk.endpoint_ise_range),
                &mut le,
                &mut he,
                None,
            );

            let lrp = rnd.frand(0.0, 0.3);
            let ei = rnd.irand(0, 2);
            let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
            for i in 0..4 {
                if ei == 0 || ei == 2 {
                    le[i] = clampi(
                        fast_roundf_int(lerp(le[i] as f32, avg_edge[i], lrp)),
                        0,
                        255,
                    ) as u8;
                }
                if ei == 1 || ei == 2 {
                    he[i] = clampi(
                        fast_roundf_int(lerp(he[i] as f32, avg_edge[i], lrp)),
                        0,
                        255,
                    ) as u8;
                }
            }

            let fendpoints: [f32; 8] = [
                le[0] as f32,
                he[0] as f32,
                le[1] as f32,
                he[1] as f32,
                le[2] as f32,
                he[2] as f32,
                le[3] as f32,
                he[3] as f32,
            ];
            let mut new_endpoints = [0u8; 8];
            cem_encode(
                cem_index,
                &fendpoints,
                u32::from(new_log_blk.endpoint_ise_range),
                &mut new_endpoints,
                true,
                false,
            );

            let dst = &mut new_log_blk.endpoints[e0..e0 + num_cem_vals];
            if dst != &new_endpoints[..num_cem_vals] {
                dst.copy_from_slice(&new_endpoints[..num_cem_vals]);
                new_log_blk.user_mode |= 2;
                any_changed = true;
            }
        }

        let num_planes = if new_log_blk.dual_plane { 2usize } else { 1 };

        if !any_changed && ctx.global_cfg.use_dct && ASTC_REFINE_MUTATE_DCT && rnd.iprob(10) {
            // DCT coefficient mutation
            if rnd.iprob(10) {
                // DC
                let plane_to_change = if new_log_blk.dual_plane {
                    rnd.irand(0, 1) as usize
                } else {
                    0
                };
                for plane_index in 0..num_planes {
                    let new_coeffs = &mut new_packed_dct_plane_data[plane_index];
                    *new_coeffs = best_blk.packed_dct_plane_data[plane_index].clone();
                    if plane_to_change == plane_index {
                        let mut new_dc_sym = new_coeffs.dc_sym as i32;
                        new_dc_sym += if rnd.bit() { -1 } else { 1 };
                        new_dc_sym = clampi(new_dc_sym, 0, new_coeffs.num_dc_levels as i32 - 1);
                        if new_dc_sym != new_coeffs.dc_sym as i32 {
                            new_coeffs.dc_sym = new_dc_sym as u32;
                            any_changed = true;
                            has_new_packed_dct_plane_data = true;
                            new_log_blk.user_mode |= 16;
                        }
                    }
                }
            }

            if !any_changed {
                // AC
                for plane_index in 0..num_planes {
                    let new_coeffs = &mut new_packed_dct_plane_data[plane_index];
                    *new_coeffs = best_blk.packed_dct_plane_data[plane_index].clone();

                    let mut total_coeffs = new_coeffs.coeffs.len();
                    if total_coeffs != 0 && new_coeffs.coeffs[total_coeffs - 1].coeff == i16::MAX {
                        total_coeffs -= 1;
                    }
                    if total_coeffs == 0 {
                        continue;
                    }

                    let rnd_coeff_index = rnd.irand(0, total_coeffs as i32 - 1) as usize;
                    let mut cur_coeff = i32::from(new_coeffs.coeffs[rnd_coeff_index].coeff);
                    if rnd.iprob(10) {
                        cur_coeff = -cur_coeff;
                    } else if rnd.bit() {
                        cur_coeff += 1;
                    } else {
                        cur_coeff -= 1;
                    }

                    if cur_coeff == 0 || cur_coeff.abs() > DCT_MAX_ARITH_COEFF_MAG {
                        continue;
                    }

                    new_coeffs.coeffs[rnd_coeff_index].coeff = cur_coeff as i16;
                    new_coeffs.max_coeff_mag =
                        new_coeffs.max_coeff_mag.max(cur_coeff.unsigned_abs());
                    any_changed = true;
                    has_new_packed_dct_plane_data = true;
                    new_log_blk.user_mode |= 4;
                }
            }
        }

        if !any_changed && ASTC_REFINE_MUTATE_ENDPOINTS {
            // gaussian endpoint CEM value mutation
            let subset_index = if log_blk.num_partitions == 1 {
                0
            } else {
                rnd.irand(0, i32::from(log_blk.num_partitions) - 1) as usize
            };
            let e = usize::from(rnd.bit());

            let (mut mutate_rgb, mut mutate_alpha) = (true, false);
            if does_cem_have_alpha(cem_index) {
                let c = rnd.irand(0, 2);
                mutate_rgb = c == 0 || c == 2;
                mutate_alpha = c == 1 || c == 2;
            }

            let levels = ah::get_ise_levels(u32::from(log_blk.endpoint_ise_range)) as i32;
            let mut mutate_val = |rnd: &mut BasisuRand, idx: usize, changed: &mut bool| -> bool {
                let Some(&ise) = new_log_blk.endpoints.get(idx) else {
                    return false;
                };
                let Some(&cur_rank) = endpoint_tab.ise_to_rank.get(ise as usize) else {
                    return false;
                };
                let cur_rank = i32::from(cur_rank);
                let new_rank = clampi(
                    (cur_rank as f32 + rnd.gaussian(0.0, ctx.rnd_mag)).round() as i32,
                    0,
                    levels - 1,
                );
                if new_rank != cur_rank {
                    *changed = true;
                }
                let Some(&v) = endpoint_tab.rank_to_ise.get(new_rank as usize) else {
                    return false;
                };
                new_log_blk.endpoints[idx] = v;
                true
            };

            if mutate_rgb {
                let mut e_ofs = subset_index * num_cem_vals;
                let (e_stride, nc);
                match cem_index {
                    CEM_LDR_LUM_DIRECT | CEM_LDR_LUM_ALPHA_DIRECT => {
                        e_ofs += e;
                        e_stride = 2;
                        nc = 1;
                    }
                    CEM_LDR_RGB_BASE_SCALE | CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => {
                        // e=0 -> scale, e=1 -> base rgb
                        e_ofs += if e != 0 { 0 } else { 3 };
                        e_stride = 1;
                        nc = if e != 0 { 3 } else { 1 };
                    }
                    CEM_LDR_RGB_DIRECT
                    | CEM_LDR_RGB_BASE_PLUS_OFFSET
                    | CEM_LDR_RGBA_DIRECT
                    | CEM_LDR_RGBA_BASE_PLUS_OFFSET => {
                        e_ofs += e;
                        e_stride = 2;
                        nc = 3;
                    }
                    _ => {
                        e_stride = 0;
                        nc = 0;
                    }
                }
                for c in 0..nc {
                    if !mutate_val(&mut rnd, e_ofs + c * e_stride, &mut any_changed) {
                        return false;
                    }
                }
            }

            if mutate_alpha {
                let mut e_ofs = subset_index * num_cem_vals;
                match cem_index {
                    CEM_LDR_LUM_ALPHA_DIRECT => e_ofs += 2 + e,
                    CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => e_ofs += 4 + e,
                    CEM_LDR_RGBA_DIRECT | CEM_LDR_RGBA_BASE_PLUS_OFFSET => e_ofs += 6 + e,
                    _ => {}
                }
                if !mutate_val(&mut rnd, e_ofs, &mut any_changed) {
                    return false;
                }
            }
        }

        if !any_changed {
            continue;
        }

        let mut new_out_block = EncodeBlockOutput {
            trial_mode_index: best_blk.trial_mode_index,
            blur_id: (u32::from(BLUR_ID_EXP) + pass) as u16,
            ..Default::default()
        };

        if ctx.enc_cfg.use_dct {
            new_out_block.packed_dct_plane_data = if has_new_packed_dct_plane_data {
                new_packed_dct_plane_data
            } else {
                best_blk.packed_dct_plane_data.clone()
            };
            for plane_index in 0..num_planes {
                if !decode_block_weights_syms(
                    block_width,
                    block_height,
                    ctx.enc_cfg.base_q,
                    plane_index as u32,
                    &mut new_log_blk,
                    &new_out_block.packed_dct_plane_data[plane_index],
                ) {
                    return false;
                }
            }
        }

        new_out_block.log_blk = new_log_blk;
        new_out_block.sse = calc_block_sse(
            block_width,
            block_height,
            &orig_block,
            &new_log_blk,
            ctx.enc_cfg,
        );

        blk_info.out_blocks.push(new_out_block);
    }

    true
}

/// `mutate_candidates`: retires old mutated candidates and adds new random mutations of the
/// best candidates, for every block.
fn mutate_candidates(ctx: &MutateCtx, threads: u32, enc_out: &mut EncodeImageOutput) -> Result<()> {
    let failed = AtomicBool::new(false);
    const CHUNK: usize = 16;
    let ok = for_each_chunk_mut(threads, &mut enc_out.block_info, CHUNK, |first, chunk| {
        let mut sorted_candidates: Vec<u64> = Vec::with_capacity(1024);
        for (i, blk_info) in chunk.iter_mut().enumerate() {
            if failed.load(Ordering::Relaxed) {
                return;
            }
            if !mutate_block(ctx, (first + i) as u32, blk_info, &mut sorted_candidates) {
                failed.store(true, Ordering::Relaxed);
                return;
            }
        }
    });
    if !ok || failed.load(Ordering::Relaxed) {
        return Err(Error::InvalidData);
    }
    Ok(())
}

/// `refine_output_for_deblocking`: SCD. Picks, per block, the candidate that minimizes the
/// error after the deblocking post filter plus cross-block-boundary and chroma penalties, over
/// `global_cfg.num_scd_passes` checkerboard passes, adding mutated candidates on the middle
/// passes. `part_data_p2/p3` are the encoder's `enc_out.m_part_data_p2/p3`.
pub(crate) fn refine_output_for_deblocking(
    orig_img: &Image,
    global_cfg: &AstcLdrEncodeConfig,
    enc_cfg: &HighLevelConfig,
    part_data_p2: &PartitionsData,
    part_data_p3: &PartitionsData,
    enc_out: &mut EncodeImageOutput,
    threads: u32,
) -> Result<()> {
    let (width, height) = (orig_img.get_width(), orig_img.get_height());
    let block_width = global_cfg.astc_block_width;
    let block_height = global_cfg.astc_block_height;
    if block_width == 0 || block_height == 0 || width == 0 || height == 0 {
        return Err(Error::InvalidArgument);
    }
    let total_block_pixels = block_width * block_height;
    let num_blocks_x = width.div_ceil(block_width);
    let num_blocks_y = height.div_ceil(block_height);
    let total_blocks = num_blocks_x * num_blocks_y;
    if enc_out.blocks_x != num_blocks_x
        || enc_out.blocks_y != num_blocks_y
        || enc_out.block_info.len() != total_blocks as usize
    {
        return Err(Error::InvalidArgument);
    }
    let srgb = global_cfg.astc_decode_mode_srgb;

    let mut candidate_img = Image::with_size(width, height);
    let mut num_max_candidates = 1usize;
    let mut block_std_dev = vec![0.0f32; total_blocks as usize];
    let mut block_pixels = [[0u8; 4]; MAX_BLOCK_PIXELS];

    for by in 0..num_blocks_y {
        for bx in 0..num_blocks_x {
            let blk_info = enc_out.block(bx, by);
            let best_blk = blk_info
                .out_blocks
                .get(blk_info.packed_out_block_index as usize)
                .ok_or(Error::InvalidData)?;
            if !decode_xuastc(
                &best_blk.log_blk,
                &mut block_pixels,
                block_width,
                block_height,
                srgb,
            ) {
                return Err(Error::InvalidData);
            }

            let (mut sum, mut sum2) = ([0.0f32; 4], [0.0f32; 4]);
            for px in block_pixels.iter().take(total_block_pixels as usize) {
                let p = vec4(*px);
                for c in 0..4 {
                    sum[c] += p[c];
                    sum2[c] += p[c] * p[c];
                }
            }
            let oo_total_texels = 1.0f32 / total_block_pixels as f32;
            let mut std_dev = [0.0f32; 4];
            for c in 0..4 {
                let v = sum2[c] - (sum[c] * sum[c]) * oo_total_texels;
                // `vec4F::component_max(v, 0)`
                let v = if v > 0.0 { v } else { 0.0 };
                std_dev[c] = (v * oo_total_texels).sqrt();
            }
            let fmax = |a: f32, b: f32| if a > b { a } else { b };
            let block_stddev = fmax(fmax(fmax(std_dev[0], std_dev[1]), std_dev[2]), std_dev[3]);
            for dy in -1i32..=1 {
                let y = by as i32 + dy;
                if y < 0 || y >= num_blocks_y as i32 {
                    continue;
                }
                for dx in -1i32..=1 {
                    let x = bx as i32 + dx;
                    if x < 0 || x >= num_blocks_x as i32 {
                        continue;
                    }
                    let s = &mut block_std_dev[(x + y * num_blocks_x as i32) as usize];
                    *s = fmax(*s, block_stddev);
                }
            }

            set_block_clipped(
                &mut candidate_img,
                &block_pixels,
                bx * block_width,
                by * block_height,
                block_width,
                block_height,
            );
            num_max_candidates = num_max_candidates.max(blk_info.out_blocks.len());
        }
    }

    if num_max_candidates == 1 {
        // No candidates to try for any block.
        return Ok(());
    }

    // checkerboard block lists
    let (mut white_list, mut black_list) = (Vec::new(), Vec::new());
    for y in 0..num_blocks_y {
        for x in 0..num_blocks_x {
            if (x ^ y) & 1 != 0 {
                black_list.push(x + y * num_blocks_x);
            } else {
                white_list.push(x + y * num_blocks_x);
            }
        }
    }

    let max_new_blocks = if u64::from(width) * u64::from(height) >= 2048 * 2048 {
        ((ASTC_REFINE_MAX_NEW_BLOCKS + 1) / 2) as u32
    } else {
        ASTC_REFINE_MAX_NEW_BLOCKS as u32
    };
    let num_passes = global_cfg.num_scd_passes.clamp(2, 256);
    let mutation_enabled = true;

    let similar = SimilarPats::new(part_data_p2, part_data_p3)?;

    const CHUNK: usize = 16;
    for pass in 0..num_passes {
        let job_block_list = if pass & 1 != 0 {
            &black_list
        } else {
            &white_list
        };

        let mut new_best_packed_block_indices = vec![-1i64; total_blocks as usize];

        if !job_block_list.is_empty() {
            let num_jobs = job_block_list.len().div_ceil(CHUNK);
            let enc_out_ref: &EncodeImageOutput = enc_out;
            let cand_ref = &candidate_img;
            let results = map_jobs(threads, num_jobs, |j| -> Result<Vec<(u32, u32)>> {
                let mut thread_state = DeblockingThreadState::new(block_width, block_height);
                let mut out = Vec::with_capacity(CHUNK);
                for &block_index in job_block_list.iter().skip(j * CHUNK).take(CHUNK) {
                    let bx = block_index % num_blocks_x;
                    let by = block_index / num_blocks_x;
                    let (new_index, _best_err) = deblocking_find_best_candidate(
                        block_width,
                        block_height,
                        num_blocks_x,
                        num_blocks_y,
                        bx,
                        by,
                        &mut thread_state,
                        cand_ref,
                        orig_img,
                        global_cfg,
                        enc_cfg,
                        enc_out_ref,
                    )?;
                    out.push((block_index, new_index));
                }
                Ok(out)
            })
            .ok_or(Error::InvalidData)?;
            for r in results {
                for (block_index, new_index) in r? {
                    new_best_packed_block_indices[block_index as usize] = i64::from(new_index);
                }
            }
        }

        // now commit changed blocks to candidate_img
        for by in 0..num_blocks_y {
            for bx in 0..num_blocks_x {
                let bi = (bx + by * num_blocks_x) as usize;
                let new_best = new_best_packed_block_indices[bi];
                if new_best < 0 {
                    continue;
                }
                let blk_info = &mut enc_out.block_info[bi];
                if i64::from(blk_info.packed_out_block_index) == new_best {
                    continue;
                }
                blk_info.packed_out_block_index = new_best as u32;
                let out_blk = blk_info
                    .out_blocks
                    .get(new_best as usize)
                    .ok_or(Error::InvalidData)?;
                if !decode_xuastc(
                    &out_blk.log_blk,
                    &mut block_pixels,
                    block_width,
                    block_height,
                    srgb,
                ) {
                    return Err(Error::InvalidData);
                }
                set_block_clipped(
                    &mut candidate_img,
                    &block_pixels,
                    bx * block_width,
                    by * block_height,
                    block_width,
                    block_height,
                );
            }
        }

        if mutation_enabled && pass >= 4 && pass < num_passes - 2 {
            let rnd_mag = if pass >= 16 {
                0.9f32
            } else if pass >= 10 {
                1.5
            } else if pass >= 6 {
                1.7
            } else if pass >= 4 {
                2.5
            } else {
                1.7
            };
            let ctx = MutateCtx {
                pass,
                block_width,
                block_height,
                num_blocks_x,
                max_new_blocks,
                block_std_dev: &block_std_dev,
                similar: &similar,
                orig_img,
                candidate_img: &candidate_img,
                global_cfg,
                enc_cfg,
                rnd_mag,
            };
            mutate_candidates(&ctx, threads, enc_out)?;
        }
    }

    Ok(())
}

/// `sharpen_image`: difference-of-Gaussians sharpening of the RGB channels.
pub(crate) fn sharpen_image(input: &Image, global_cfg: &AstcLdrEncodeConfig) -> Result<Image> {
    let amount = global_cfg.sharpen_amount;
    if input.get_width() <= 4 && input.get_height() <= 4 {
        return Ok(input.clone());
    }

    let (w, h) = (input.get_width(), input.get_height());
    let mut blur1 = Image::with_size(w, h);
    let mut blur2 = Image::with_size(w, h);
    if !crate::basis::ll::comp::image_resample(
        input, &mut blur1, false, "gaussian", 1.05, false, 0, 4,
    ) {
        return Err(Error::InvalidArgument);
    }
    if !crate::basis::ll::comp::image_resample(
        input, &mut blur2, false, "gaussian", 1.3, false, 0, 4,
    ) {
        return Err(Error::InvalidArgument);
    }

    let mut out = Image::with_size(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = input.get(x, y);
            let b1 = blur1.get(x, y);
            let b2 = blur2.get(x, y);
            let mut o = [0, 0, 0, i[3]];
            for c in 0..3 {
                let k = (i[c] as f32 + amount * (b1[c] as f32 - b2[c] as f32)).round() as i32;
                o[c] = clampi(k, 0, 255) as u8;
            }
            *out.get_mut(x, y) = o;
        }
    }
    Ok(out)
}

/// `merge_output_candidates`: appends `enc_out2`'s candidates (no de-dup) to `enc_out`; returns
/// the number of blocks whose best candidate came from `enc_out2`.
pub(crate) fn merge_output_candidates(
    enc_out: &mut EncodeImageOutput,
    enc_out2: &EncodeImageOutput,
) -> Result<u32> {
    if enc_out.blocks_x < enc_out2.blocks_x || enc_out.blocks_y < enc_out2.blocks_y {
        return Err(Error::InvalidArgument);
    }
    let mut total_better_blocks = 0;
    for by in 0..enc_out2.blocks_y {
        for bx in 0..enc_out2.blocks_x {
            let i1 = (bx + by * enc_out.blocks_x) as usize;
            let i2 = (bx + by * enc_out2.blocks_x) as usize;
            let blk_info2 = &enc_out2.block_info[i2];
            let blk_info1 = &mut enc_out.block_info[i1];
            let orig_out_block_size = blk_info1.out_blocks.len() as u32;
            blk_info1
                .out_blocks
                .extend(blk_info2.out_blocks.iter().cloned());

            let best_err1 = blk_info1
                .out_blocks
                .get(blk_info1.packed_out_block_index as usize)
                .ok_or(Error::InvalidData)?
                .sse;
            let best_err2 = blk_info2
                .out_blocks
                .get(blk_info2.packed_out_block_index as usize)
                .ok_or(Error::InvalidData)?
                .sse;
            if best_err2 < best_err1 {
                total_better_blocks += 1;
                blk_info1.packed_out_block_index =
                    orig_out_block_size + blk_info2.packed_out_block_index;
                enc_out.packed_phys_blocks[i1] = enc_out2.packed_phys_blocks[i2];
            }
        }
    }
    Ok(total_better_blocks)
}

/// `cross_check_dct`: sanity checks that every chosen block's DCT symbols decode to its weight
/// grid. `Err(InvalidData)` on any mismatch.
pub(crate) fn cross_check_dct(
    block_width: u32,
    block_height: u32,
    num_blocks_x: u32,
    num_blocks_y: u32,
    enc_out: &EncodeImageOutput,
    enc_cfg: &HighLevelConfig,
) -> Result<()> {
    for by in 0..num_blocks_y {
        for bx in 0..num_blocks_x {
            let blk_info = enc_out
                .block_info
                .get((bx + by * enc_out.blocks_x) as usize)
                .ok_or(Error::InvalidData)?;
            let out_blk = blk_info
                .out_blocks
                .get(blk_info.packed_out_block_index as usize)
                .ok_or(Error::InvalidData)?;
            let log_blk = &out_blk.log_blk;
            if log_blk.solid_color_flag_ldr {
                continue;
            }

            let mut temp_log_blk = *log_blk;
            let num_planes = if log_blk.dual_plane { 2 } else { 1 };
            for plane_index in 0..num_planes {
                let syms = &out_blk.packed_dct_plane_data[plane_index];
                if !decode_block_weights_syms(
                    block_width,
                    block_height,
                    enc_cfg.base_q,
                    plane_index as u32,
                    &mut temp_log_blk,
                    syms,
                ) {
                    return Err(Error::InvalidData);
                }
                let max_coeff = syms
                    .coeffs
                    .iter()
                    .filter(|c| c.coeff != i16::MAX)
                    .map(|c| i32::from(c.coeff).abs())
                    .fold(0, i32::max);
                if max_coeff > syms.max_coeff_mag as i32 {
                    return Err(Error::InvalidData);
                }
            }

            // the decoded weight grid must match the stored weight grid
            let total_grid_weights = usize::from(temp_log_blk.grid_width)
                * usize::from(temp_log_blk.grid_height)
                * num_planes;
            if log_blk.weights[..total_grid_weights] != temp_log_blk.weights[..total_grid_weights] {
                return Err(Error::InvalidData);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use crate::basis::xuastc::enc::dump;

    /// Encoder configuration written next to the dumps by the patched C++ build
    /// (`XU_DUMP_CFG`): bw bh srgb w0 w1 w2 w3 scd postfilter chroma passes dct base_q effort
    /// sharpen sharpen_amount.
    struct DumpCfg {
        global: AstcLdrEncodeConfig,
        enc: HighLevelConfig,
    }

    fn parse_cfg(s: &str) -> DumpCfg {
        let v: Vec<&str> = s.split_whitespace().collect();
        let u = |i: usize| v[i].parse::<u32>().unwrap();
        let f = |i: usize| v[i].parse::<f32>().unwrap();
        let global = AstcLdrEncodeConfig {
            astc_block_width: u(0),
            astc_block_height: u(1),
            astc_decode_mode_srgb: u(2) != 0,
            comp_weights: [u(3), u(4), u(5), u(6)],
            scd_enabled: u(7) != 0,
            scd_will_postfilter: u(8) != 0,
            scd_preserve_chroma: u(9) != 0,
            num_scd_passes: u(10),
            use_dct: u(11) != 0,
            effort_level: u(13),
            sharpen_flag: u(14) != 0,
            sharpen_amount: f(15),
            ..Default::default()
        };
        let mut enc = HighLevelConfig::default();
        enc.cem_enc_params.comp_weights = global.comp_weights;
        enc.cem_enc_params.decode_mode_srgb = global.astc_decode_mode_srgb;
        enc.base_q = f(12);
        enc.use_dct = global.use_dct;
        enc.block_width = global.astc_block_width;
        enc.block_height = global.astc_block_height;
        DumpCfg { global, enc }
    }

    /// First difference between two encoder outputs, if any.
    fn compare(a: &EncodeImageOutput, b: &EncodeImageOutput) -> Option<String> {
        if a.blocks_x != b.blocks_x || a.blocks_y != b.blocks_y {
            return Some("block dims".into());
        }
        for (i, (x, y)) in a.block_info.iter().zip(&b.block_info).enumerate() {
            if x.out_blocks.len() != y.out_blocks.len() {
                return Some(format!(
                    "block {i}: {} vs {} candidates",
                    x.out_blocks.len(),
                    y.out_blocks.len()
                ));
            }
            for (c, (p, q)) in x.out_blocks.iter().zip(&y.out_blocks).enumerate() {
                if p.trial_mode_index != q.trial_mode_index
                    || p.blur_id != q.blur_id
                    || p.sse != q.sse
                    || !dump::log_blk_eq(&p.log_blk, &q.log_blk)
                    || p.packed_dct_plane_data != q.packed_dct_plane_data
                {
                    return Some(format!(
                        "block {i} cand {c}: tm {}/{} blur {}/{} sse {}/{} mode {}/{}",
                        p.trial_mode_index,
                        q.trial_mode_index,
                        p.blur_id,
                        q.blur_id,
                        p.sse,
                        q.sse,
                        p.log_blk.user_mode,
                        q.log_blk.user_mode
                    ));
                }
            }
            if x.packed_out_block_index != y.packed_out_block_index {
                return Some(format!(
                    "block {i}: packed index {} vs {}",
                    x.packed_out_block_index, y.packed_out_block_index
                ));
            }
        }
        None
    }

    /// Runs SCD (+ `cross_check_dct`) on `pre` like `compress_image` does.
    fn run_scd(cfg: &DumpCfg, img: &Image, enc_out: &mut EncodeImageOutput, threads: u32) {
        let (bw, bh) = (cfg.global.astc_block_width, cfg.global.astc_block_height);
        if cfg.global.scd_enabled {
            let p2 = PartitionsData::new(2, bw, bh, true);
            let p3 = PartitionsData::new(3, bw, bh, true);
            refine_output_for_deblocking(img, &cfg.global, &cfg.enc, &p2, &p3, enc_out, threads)
                .unwrap();
        }
        if cfg.global.use_dct {
            let (nbx, nby) = (enc_out.blocks_x, enc_out.blocks_y);
            cross_check_dct(bw, bh, nbx, nby, enc_out, &cfg.enc).unwrap();
        }
    }

    /// Compares against every `NAME.cfg` + `NAME.pre.xud` + `NAME.final.xud` in `$XU_DUMP_DIR`.
    #[test]
    #[ignore = "needs XU_DUMP_DIR (C++ encoder dumps with .cfg files)"]
    fn scd_matches_cpp_dumps() {
        let Ok(dir) = std::env::var("XU_DUMP_DIR") else {
            return;
        };
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let p = e.unwrap().path();
                let n = p.file_name()?.to_str()?.strip_suffix(".cfg")?.to_string();
                Some(n)
            })
            .collect();
        names.sort();
        let mut failures = Vec::new();
        for name in &names {
            let cfg = parse_cfg(&std::fs::read_to_string(format!("{dir}/{name}.cfg")).unwrap());
            let (Some(pre), Some(fin)) = (
                dump::load(&format!("{name}.pre.xud")),
                dump::load(&format!("{name}.final.xud")),
            ) else {
                continue;
            };
            let mut out = pre.out.clone();
            run_scd(&cfg, &pre.image, &mut out, 4);
            match compare(&out, &fin.out) {
                None => eprintln!("{name}: OK"),
                Some(m) => {
                    eprintln!("{name}: MISMATCH {m}");
                    failures.push(name.clone());
                }
            }
            if name.contains("e5") {
                let mut out1 = pre.out.clone();
                run_scd(&cfg, &pre.image, &mut out1, 1);
                assert!(
                    compare(&out, &out1).is_none(),
                    "{name}: thread count dependent"
                );
            }
        }
        assert!(failures.is_empty(), "mismatches: {failures:?}");
    }

    fn fnv1a(data: &[u8]) -> u64 {
        let mut h: u64 = 1_469_598_103_934_665_603;
        for &b in data {
            h ^= u64::from(b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        h
    }

    /// Serializes the block data of `out` like the C++ `xu_dump_enc_out` (minus the image and
    /// with `m_uses_suboptimal_cem_encoding` = 0), for hashing.
    fn serialize(out: &EncodeImageOutput) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&out.blocks_x.to_le_bytes());
        v.extend_from_slice(&out.blocks_y.to_le_bytes());
        for bi in &out.block_info {
            v.extend_from_slice(&bi.packed_out_block_index.to_le_bytes());
            v.extend_from_slice(&[
                bi.low_freq_block_flag as u8,
                bi.super_strong_edges as u8,
                bi.very_strong_edges as u8,
                bi.strong_edges as u8,
            ]);
            v.extend_from_slice(&(bi.out_blocks.len() as u32).to_le_bytes());
            for ob in &bi.out_blocks {
                v.extend_from_slice(&ob.trial_mode_index.to_le_bytes());
                v.extend_from_slice(&ob.blur_id.to_le_bytes());
                v.extend_from_slice(&ob.sse.to_le_bytes());
                let b = &ob.log_blk;
                v.extend_from_slice(&[
                    b.error_flag as u8,
                    b.solid_color_flag_ldr as u8,
                    b.solid_color_flag_hdr as u8,
                    b.user_mode,
                    b.grid_width,
                    b.grid_height,
                    b.dual_plane as u8,
                    b.weight_ise_range,
                    b.endpoint_ise_range,
                    b.color_component_selector,
                    b.num_partitions,
                    0,
                ]);
                v.extend_from_slice(&b.partition_id.to_le_bytes());
                v.extend_from_slice(&b.color_endpoint_modes);
                v.extend_from_slice(&b.weights);
                v.extend_from_slice(&b.endpoints);
                for p in &ob.packed_dct_plane_data {
                    for x in [
                        p.dc_sym,
                        p.num_dc_levels,
                        p.max_coeff_mag,
                        p.max_zigzag_index,
                    ] {
                        v.extend_from_slice(&x.to_le_bytes());
                    }
                    v.extend_from_slice(&(p.coeffs.len() as u32).to_le_bytes());
                    for c in &p.coeffs {
                        v.extend_from_slice(&c.num_zeros.to_le_bytes());
                        v.extend_from_slice(&c.coeff.to_le_bytes());
                    }
                }
            }
        }
        v
    }

    /// Regression fixtures in `testdata/basisu_enc/`: `scd_NAME.cfg` + the zstd-compressed C++
    /// `.pre.xud` dump; the hash is [`serialize`] of the C++ `.final.xud` (see
    /// `print_fixture_hashes`).
    const FIXTURES: [(&str, u64); 2] = [
        ("scd_xu12x12", 7_813_062_417_089_735_249),
        ("scd_xu8x8_q60", 6_045_416_495_415_117_020),
    ];

    fn load_fixture(name: &str) -> (DumpCfg, dump::Dump) {
        use std::io::Read;
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/basisu_enc");
        let cfg = parse_cfg(&std::fs::read_to_string(format!("{dir}/{name}.cfg")).unwrap());
        let zst = std::fs::read(format!("{dir}/{name}.pre.xud.zst")).unwrap();
        let mut dec = ruzstd::decoding::StreamingDecoder::new(&zst[..]).unwrap();
        let mut data = Vec::new();
        dec.read_to_end(&mut data).unwrap();
        (cfg, dump::parse(&data))
    }

    #[test]
    fn scd_regression_fixtures() {
        for (name, hash) in FIXTURES {
            let (cfg, pre) = load_fixture(name);
            for threads in [1, 3] {
                let mut out = pre.out.clone();
                run_scd(&cfg, &pre.image, &mut out, threads);
                assert_eq!(fnv1a(&serialize(&out)), hash, "{name} threads={threads}");
            }
        }
    }

    /// Prints [`FIXTURES`] hashes from the C++ `scd_NAME.final.xud` dumps in `$XU_DUMP_DIR`.
    #[test]
    #[ignore = "needs XU_DUMP_DIR (C++ encoder dumps)"]
    fn print_fixture_hashes() {
        for (name, _) in FIXTURES {
            if let Some(fin) = dump::load(&format!("{name}.final.xud")) {
                eprintln!("(\"{name}\", {}),", fnv1a(&serialize(&fin.out)));
            }
        }
    }

    fn raw_image(name: &str, w: u32, h: u32) -> Image {
        let path = format!(
            "{}/testdata/images/raw/{name}.rgba",
            env!("CARGO_MANIFEST_DIR")
        );
        let data = std::fs::read(path).unwrap();
        let mut img = Image::with_size(w, h);
        for y in 0..h {
            for x in 0..w {
                let o = ((x + y * w) * 4) as usize;
                *img.get_mut(x, y) = [data[o], data[o + 1], data[o + 2], data[o + 3]];
            }
        }
        img
    }

    /// FNV-1a of the sharpened image the C++ encoder dumped (`-xuastc_ldr_sharpen A`).
    #[test]
    fn sharpen_matches_cpp() {
        for (name, w, h, amount, hash) in [
            ("rgba_64x48", 64, 48, 1.5f32, 7_533_324_829_638_658_780u64),
            ("mosaic_rgba_50x38", 50, 38, 4.0, 8_148_697_971_885_737_607),
        ] {
            let img = raw_image(name, w, h);
            let cfg = AstcLdrEncodeConfig {
                sharpen_flag: true,
                sharpen_amount: amount,
                ..Default::default()
            };
            let out = sharpen_image(&img, &cfg).unwrap();
            let bytes: Vec<u8> = out.get_pixels().iter().flatten().copied().collect();
            assert_eq!(fnv1a(&bytes), hash, "{name}");
        }
    }

    /// `merge_output_candidates(first, basisu)` == the merged `.pre.xud` for every
    /// `NAME.basisu.xud` in `$XU_DUMP_DIR`.
    #[test]
    #[ignore = "needs XU_DUMP_DIR (C++ encoder dumps)"]
    fn merge_matches_cpp_dumps() {
        let Ok(dir) = std::env::var("XU_DUMP_DIR") else {
            return;
        };
        let mut n = 0;
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            let Some(name) = p
                .file_name()
                .and_then(|f| f.to_str())
                .and_then(|f| f.strip_suffix(".basisu.xud"))
            else {
                continue;
            };
            let first = dump::load(&format!("{name}.first.xud")).unwrap();
            let basisu = dump::load(&format!("{name}.basisu.xud")).unwrap();
            let pre = dump::load(&format!("{name}.pre.xud")).unwrap();
            let mut out = first.out.clone();
            merge_output_candidates(&mut out, &basisu.out).unwrap();
            assert_eq!(compare(&out, &pre.out), None, "{name}");
            n += 1;
        }
        eprintln!("{n} merges OK");
    }
}
