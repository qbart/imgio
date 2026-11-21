// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! XUASTC LDR supercompression: the bitstream writer half of `astc_ldr::compress_image`
//! (`basisu_astc_ldr_encode.cpp`) for the full arithmetic and the hybrid arithmetic + Zstd
//! syntaxes, the syntax selection, and the helpers shared with the full Zstd writer
//! ([`super::zstd_syntax`]): lossy candidate replacement, block comparisons, solid color and
//! endpoint DPCM prediction.

#![allow(clippy::too_many_arguments)]

use super::arith_enc::ArithEnc;
use super::zstd_syntax::{compress_image_full_zstd, zstd_compress};
use super::{
    AstcLdrEncodeConfig, BlockInfo, CompressOutput, DctSyms, EncodeBlockOutput, EncodeImageOutput,
    XuastcSyntax,
};
use crate::astc::helpers::{
    self as ah, CEM_LDR_RGB_BASE_PLUS_OFFSET, CEM_LDR_RGB_DIRECT, CEM_LDR_RGBA_BASE_PLUS_OFFSET,
    CEM_LDR_RGBA_DIRECT, FIRST_VALID_ENDPOINT_ISE_RANGE, FIRST_VALID_WEIGHT_ISE_RANGE,
    LogAstcBlock,
};
use crate::basis::file::BASIS_ASTC_ORDER;
use crate::basis::ll::comp::Image;
use crate::basis::ll::hdr_tables::REUSE_XY_DELTAS;
use crate::basis::ll::huffman_enc::BitwiseCoder;
use crate::basis::xuastc::arith::{BitModel, DataModel, GammaContexts};
use crate::basis::xuastc::endpoints::{
    MAX_CEM_VALS, cem_supports_bc, convert_endpoints_across_cems, used_blue_contraction,
};
use crate::basis::xuastc::{
    ARITH_HEADER_MARKER, ARITH_HEADER_MARKER_BITS, DCT_MEAN_LEVELS0, DCT_RUN_LEN_EOB_SYM_INDEX,
    FINAL_SYNC_MARKER, FINAL_SYNC_MARKER_BITS, MAX_CONFIG_REUSE_NEIGHBORS, MODE_RAW,
    MODE_REUSE_CFG_ENDPOINTS_LEFT, MODE_RUN, MODE_SOLID, MODE_TOTAL, ModeTable, NUM_GROUPS,
    OTM_NUM_CCS, OTM_NUM_CEMS, OTM_NUM_GRID_ANISOS, OTM_NUM_GRID_SIZES, OTM_NUM_SUBSETS,
    PART_HASH_SIZE, TrialMode, cem_to_ldrcem_index, fib_hash, grid_aniso, group_index, mode_table,
    prev_solid_color, tables, total_unique_patterns,
};
use crate::{Error, Result};

/// `DCT_MAX_ARITH_COEFF_MAG`.
pub(super) const DCT_MAX_ARITH_COEFF_MAG: u32 = 255;
/// `DCT_MEAN_LEVELS1`.
const DCT_MEAN_LEVELS1: u32 = crate::basis::xuastc::dct::DCT_MEAN_LEVELS1;
/// `DISABLE_FASTER_FORMAT_TOTAL_BLOCKS_THRESH`.
const DISABLE_FASTER_FORMAT_TOTAL_BLOCKS_THRESH: u32 = 64;
/// `BIG_FLOAT_VAL`.
const BIG_FLOAT_VAL: f32 = 1e30;

// ---------------------------------------------------------------------------------------
// Shared state

/// Everything the writers need besides the candidates (from `compress_image`'s locals).
pub(super) struct Ctx<'a> {
    pub enc_out: &'a EncodeImageOutput,
    pub tms: &'static ModeTable,
    pub size_index: usize,
    pub width: u32,
    pub height: u32,
    pub bw: u32,
    pub bh: u32,
    pub bxs: u32,
    pub bys: u32,
    pub has_alpha: bool,
    pub srgb: bool,
    pub use_dct: bool,
    pub int_q: u32,
    pub effort_level: u32,
    pub lossy: bool,
    pub replacement_min_psnr: f32,
    pub psnr_trial_diff_thresh: f32,
    pub psnr_trial_diff_thresh_edge: f32,
    pub total_comp_weights: f32,
}

impl Ctx<'_> {
    pub fn info(&self, bx: u32, by: u32) -> Result<&BlockInfo> {
        self.enc_out
            .block_info
            .get((bx + by * self.bxs) as usize)
            .ok_or(Error::InvalidArgument)
    }

    /// The trial mode of a (non solid) candidate.
    pub fn tm(&self, blk_out: &EncodeBlockOutput) -> Result<TrialMode> {
        let i = u32::try_from(blk_out.trial_mode_index).map_err(|_| Error::InvalidArgument)?;
        self.tms.get(i).map_err(|_| Error::InvalidArgument)
    }

    /// Lossy supercompression: possibly replaces the chosen candidate with another one (within
    /// the PSNR thresholds) whose configuration (pass 0: and endpoints) matches a coded neighbor.
    /// Returns the candidate index to code.
    pub fn select_candidate(
        &self,
        blk_info: &BlockInfo,
        coded: &[LogAstcBlock],
        bx: u32,
        by: u32,
    ) -> Result<usize> {
        let packed = blk_info.packed_out_block_index as usize;
        let chosen = blk_info
            .out_blocks
            .get(packed)
            .ok_or(Error::InvalidArgument)?;
        let mut best = packed;
        let total_block_pixels = (self.bw * self.bh) as f32;
        let ref_wpsnr = calc_wpsnr(chosen.sse, self.total_comp_weights, total_block_pixels);
        if !self.lossy
            || ref_wpsnr < self.replacement_min_psnr
            || chosen.log_blk.solid_color_flag_ldr
        {
            return Ok(best);
        }
        let psnr_thresh = if blk_info.strong_edges {
            self.psnr_trial_diff_thresh_edge
        } else {
            self.psnr_trial_diff_thresh
        };
        let mut best_alt_wpsnr = 0.0f32;
        for pass in 0..2 {
            let mut found = false;
            for (iter, trial) in blk_info.out_blocks.iter().enumerate() {
                if iter == packed {
                    continue;
                }
                let trial_wpsnr =
                    calc_wpsnr(trial.sse, self.total_comp_weights, total_block_pixels);
                if trial_wpsnr < ref_wpsnr - psnr_thresh {
                    continue;
                }
                if trial_wpsnr < best_alt_wpsnr {
                    continue;
                }
                let t = &trial.log_blk;
                if t.solid_color_flag_ldr {
                    continue;
                }
                for i in 0..MAX_CONFIG_REUSE_NEIGHBORS {
                    let Some(n) = self.neighbor(coded, bx, by, i) else {
                        continue;
                    };
                    if n.solid_color_flag_ldr {
                        continue;
                    }
                    let accept = if pass == 0 {
                        compare_log_block_configs_and_endpoints(t, n)
                    } else {
                        compare_log_block_configs(t, n)
                    };
                    if accept {
                        best_alt_wpsnr = trial_wpsnr;
                        best = iter;
                        found = true;
                        break;
                    }
                }
            }
            if found {
                break;
            }
        }
        Ok(best)
    }

    /// Coded neighbor `i` (0 = left, 1 = upper, 2 = upper left).
    pub fn neighbor<'c>(
        &self,
        coded: &'c [LogAstcBlock],
        bx: u32,
        by: u32,
        i: u32,
    ) -> Option<&'c LogAstcBlock> {
        let (dx, dy) = match i {
            0 => (1, 0),
            1 => (0, 1),
            _ => (1, 1),
        };
        if bx < dx || by < dy {
            return None;
        }
        coded.get(((bx - dx) + (by - dy) * self.bxs) as usize)
    }

    /// The previous coded block (left, else upper).
    pub fn prev<'c>(
        &self,
        coded: &'c [LogAstcBlock],
        bx: u32,
        by: u32,
    ) -> Option<&'c LogAstcBlock> {
        if bx > 0 {
            coded.get((bx - 1 + by * self.bxs) as usize)
        } else if by > 0 {
            coded.get((bx + (by - 1) * self.bxs) as usize)
        } else {
            None
        }
    }

    /// First neighbor whose configuration, partition id and endpoints equal `cur`'s.
    pub fn full_cfg_endpoint_reuse_index(
        &self,
        coded: &[LogAstcBlock],
        bx: u32,
        by: u32,
        cur: &LogAstcBlock,
    ) -> Option<u32> {
        (0..MAX_CONFIG_REUSE_NEIGHBORS).find(|&i| {
            self.neighbor(coded, bx, by, i).is_some_and(|n| {
                !n.solid_color_flag_ldr && compare_log_block_configs_and_endpoints(cur, n)
            })
        })
    }

    /// Solid color DPCM deltas (3 or 4 components) against the previous coded block.
    pub fn solid_deltas(
        &self,
        coded: &[LogAstcBlock],
        bx: u32,
        by: u32,
        cur: &LogAstcBlock,
    ) -> ([u32; 4], usize) {
        let prev = prev_solid_color(self.prev(coded, bx, by));
        let mut d = [0u32; 4];
        for (i, v) in d.iter_mut().enumerate() {
            let c = u32::from(cur.solid_color(i) >> 8);
            *v = c.wrapping_sub(prev[i]) & 0xFF;
        }
        (d, if self.has_alpha { 4 } else { 3 })
    }

    /// The endpoint DPCM predictor search. `reuse_model` adds the reuse index price
    /// (arithmetic syntaxes only). Returns the reuse delta index and predicted endpoints.
    pub fn find_endpoint_pred(
        &self,
        coded: &[LogAstcBlock],
        bx: u32,
        by: u32,
        cur: &LogAstcBlock,
        num_parts: usize,
        use_bc: &[bool; 4],
        reuse_model: Option<&DataModel>,
    ) -> Option<(u32, [[u8; MAX_CEM_VALS]; 4])> {
        let cur_cem = u32::from(cur.color_endpoint_modes[0]);
        let vals = ah::get_num_cem_values(cur_cem) as usize;
        let erange = u32::from(cur.endpoint_ise_range);
        let e2r = &ah::dequant_tables().endpoint_tab(erange).ise_to_rank;
        let n = (vals * num_parts) as f32;
        let mut best_trial_bits = BIG_FLOAT_VAL;
        let mut best: Option<(u32, [[u8; MAX_CEM_VALS]; 4])> = None;
        for (reuse_index, d) in REUSE_XY_DELTAS.iter().enumerate() {
            let rx = bx as i32 + i32::from(d[0]);
            let ry = by as i32 + i32::from(d[1]);
            if rx < 0 || ry < 0 || rx >= self.bxs as i32 || ry >= self.bys as i32 {
                continue;
            }
            let Some(t) = coded.get((rx as u32 + ry as u32 * self.bxs) as usize) else {
                continue;
            };
            if t.solid_color_flag_ldr {
                continue;
            }
            let mut src = [0u8; MAX_CEM_VALS];
            src.copy_from_slice(&t.endpoints[..MAX_CEM_VALS]);
            let mut pred = [[0u8; MAX_CEM_VALS]; 4];
            let ok = (0..num_parts).all(|p| {
                convert_endpoints_across_cems(
                    u32::from(t.color_endpoint_modes[0]),
                    u32::from(t.endpoint_ise_range),
                    &src,
                    cur_cem,
                    erange,
                    &mut pred[p],
                    use_bc[p],
                )
            });
            if !ok {
                continue;
            }
            let mut delta2 = 0i64;
            for (p, pv) in pred.iter().enumerate().take(num_parts) {
                for (v, &pe) in pv.iter().enumerate().take(vals) {
                    let c = i64::from(e2r[usize::from(cur.endpoints[p * vals + v])]);
                    let q = i64::from(e2r[usize::from(pe)]);
                    delta2 += (c - q) * (c - q);
                }
            }
            let mse = delta2 as f32 / n;
            let k_const = 2.047_095_6_f32;
            let bits_per_sym = (0.5f32 * mse.max(1e-9).log2() + k_const).clamp(0.05, 8.0);
            let mut total_est_bits = bits_per_sym * n;
            if let Some(m) = reuse_model {
                total_est_bits += m.get_price(reuse_index as u32);
            }
            if total_est_bits < best_trial_bits {
                best_trial_bits = total_est_bits;
                best = Some((reuse_index as u32, pred));
                if delta2 == 0 {
                    break;
                }
            }
        }
        best
    }
}

/// Weighted PSNR of a candidate from its weighted SSE.
fn calc_wpsnr(sse: u64, total_comp_weights: f32, total_block_pixels: f32) -> f32 {
    let wmse = sse as f32 / (total_comp_weights * total_block_pixels);
    if wmse > 1e-5 {
        20.0f32 * (255.0f32 / wmse.sqrt()).log10()
    } else {
        10000.0
    }
}

/// `compare_log_block_configs`.
pub(super) fn compare_log_block_configs(t: &LogAstcBlock, n: &LogAstcBlock) -> bool {
    if n.solid_color_flag_ldr {
        return false;
    }
    t.color_endpoint_modes[0] == n.color_endpoint_modes[0]
        && t.dual_plane == n.dual_plane
        && t.color_component_selector == n.color_component_selector
        && t.num_partitions == n.num_partitions
        && t.partition_id == n.partition_id
        && t.grid_width == n.grid_width
        && t.grid_height == n.grid_height
        && t.endpoint_ise_range == n.endpoint_ise_range
        && t.weight_ise_range == n.weight_ise_range
}

fn total_endpoint_vals(b: &LogAstcBlock) -> usize {
    (usize::from(b.num_partitions)
        * ah::get_num_cem_values(u32::from(b.color_endpoint_modes[0])) as usize)
        .min(b.endpoints.len())
}

/// `compare_log_block_configs_and_endpoints`.
pub(super) fn compare_log_block_configs_and_endpoints(t: &LogAstcBlock, n: &LogAstcBlock) -> bool {
    if !compare_log_block_configs(t, n) {
        return false;
    }
    let e = total_endpoint_vals(t);
    t.endpoints[..e] == n.endpoints[..e]
}

/// `compare_log_blocks_for_equality`.
pub(crate) fn compare_log_blocks_for_equality(t: &LogAstcBlock, n: &LogAstcBlock) -> bool {
    if t.solid_color_flag_ldr {
        return n.solid_color_flag_ldr && (0..4).all(|i| t.solid_color(i) == n.solid_color(i));
    } else if n.solid_color_flag_ldr {
        return false;
    }
    if !compare_log_block_configs(t, n) {
        return false;
    }
    let e = total_endpoint_vals(t);
    if t.endpoints[..e] != n.endpoints[..e] {
        return false;
    }
    let w = ((if t.dual_plane { 2 } else { 1 })
        * usize::from(t.grid_width)
        * usize::from(t.grid_height))
    .min(t.weights.len());
    t.weights[..w] == n.weights[..w]
}

/// `separate_tm_index`: (cem, subset, ccs, grid size, grid aniso) of a trial mode.
pub(super) fn separate_tm_index(bw: u32, bh: u32, tm: &TrialMode) -> [u32; 5] {
    let cem_index = u32::from(tm.cem);
    let subset_index = u32::from(tm.num_parts) - 1;
    let ccs_index = (i32::from(tm.ccs_index) + 1) as u32;
    let grid_size = u32::from(u32::from(tm.grid_w) >= bw - 1 && u32::from(tm.grid_h) >= bh - 1);
    let aniso = grid_aniso(u32::from(tm.grid_w), u32::from(tm.grid_h), bw, bh);
    [cem_index, subset_index, ccs_index, grid_size, aniso]
}

/// `partitions_data::m_part_seed_to_unique_index` (inverse of the transcoder's unique pattern
/// tables) and `m_total_unique_patterns`.
pub(super) fn part_seed_to_unique(
    size_index: usize,
    num_parts: u32,
    seed: u16,
) -> Result<(u32, u32)> {
    let t: &[u16] = match num_parts {
        2 => tables::UNIQUE_TO_SEED_P2.get(size_index),
        3 => tables::UNIQUE_TO_SEED_P3.get(size_index),
        _ => None,
    }
    .copied()
    .ok_or(Error::InvalidArgument)?;
    let total = total_unique_patterns(size_index, num_parts);
    let u = t
        .iter()
        .take(total as usize)
        .position(|&s| s == seed)
        .ok_or(Error::InvalidArgument)?;
    Ok((u as u32, total))
}

/// Blue contraction flags of each partition's endpoints.
pub(super) fn endpoints_use_bc(cur: &LogAstcBlock, num_parts: usize) -> [bool; 4] {
    let cem = u32::from(cur.color_endpoint_modes[0]);
    let vals = ah::get_num_cem_values(cem) as usize;
    let mut r = [false; 4];
    if cem_supports_bc(cem) {
        for (p, u) in r.iter_mut().enumerate().take(num_parts) {
            *u = used_blue_contraction(
                cem,
                &cur.endpoints[p * vals..],
                u32::from(cur.endpoint_ise_range),
            );
        }
    }
    r
}

/// The RGB(A) direct trial mode's "is base+offset" flag.
pub(super) fn is_base_ofs(cur: &LogAstcBlock) -> bool {
    let c = u32::from(cur.color_endpoint_modes[0]);
    c == CEM_LDR_RGB_BASE_PLUS_OFFSET || c == CEM_LDR_RGBA_BASE_PLUS_OFFSET
}

pub(super) fn tm_is_direct(tm: &TrialMode) -> bool {
    let c = u32::from(tm.cem);
    c == CEM_LDR_RGB_DIRECT || c == CEM_LDR_RGBA_DIRECT
}

/// Endpoint DPCM symbols: `imod(rank(cur) - rank(pred), levels)` per value.
pub(super) fn endpoint_dpcm_syms(
    cur: &LogAstcBlock,
    pred: &[[u8; MAX_CEM_VALS]; 4],
    num_parts: usize,
    mut out: impl FnMut(u32),
) {
    let cem = u32::from(cur.color_endpoint_modes[0]);
    let vals = ah::get_num_cem_values(cem) as usize;
    let erange = u32::from(cur.endpoint_ise_range);
    let levels = ah::get_ise_levels(erange) as i32;
    let e2r = &ah::dequant_tables().endpoint_tab(erange).ise_to_rank;
    for (p, pv) in pred.iter().enumerate().take(num_parts) {
        for (v, &pe) in pv.iter().enumerate().take(vals) {
            let c = i32::from(e2r[usize::from(cur.endpoints[p * vals + v])]);
            let q = i32::from(e2r[usize::from(pe)]);
            out((c - q).rem_euclid(levels.max(1)) as u32);
        }
    }
}

/// Weight DPCM symbols of every plane: `imod(rank(w) - rank(prev w), levels)`.
pub(super) fn weight_dpcm_syms(cur: &LogAstcBlock, mut out: impl FnMut(u32)) {
    let planes = if cur.dual_plane { 2 } else { 1 };
    let total = usize::from(cur.grid_width) * usize::from(cur.grid_height);
    let wrange = u32::from(cur.weight_ise_range);
    let levels = ah::get_ise_levels(wrange) as i32;
    let w2r = &ah::dequant_tables().weight_tab(wrange).ise_to_rank;
    for plane in 0..planes {
        let mut prev_w = levels / 2;
        for i in 0..total {
            let w = i32::from(w2r[usize::from(cur.weights[plane + i * planes])]);
            out((w - prev_w).rem_euclid(levels.max(1)) as u32);
            prev_w = w;
        }
    }
}

/// The Zstd weight streams of the hybrid and full Zstd syntaxes.
#[derive(Default)]
pub(super) struct WeightStreams {
    pub mean0_bits: BitwiseCoder,
    pub mean1_bytes: Vec<u8>,
    pub run_bytes: Vec<u8>,
    pub coeff_bytes: Vec<u8>,
    pub sign_bits: BitwiseCoder,
    pub weight2_bits: BitwiseCoder,
    pub weight3_bits: BitwiseCoder,
    pub weight4_bits: BitwiseCoder,
    pub weight8_bits: Vec<u8>,
}

impl WeightStreams {
    /// DCT vs DPCM decision of the byte stream syntaxes (no DCT if a coefficient magnitude is
    /// too large or a plane has too many coefficients).
    pub fn use_dct(
        use_dct_cfg: bool,
        cur: &LogAstcBlock,
        blk_out: &EncodeBlockOutput,
    ) -> Result<bool> {
        if !use_dct_cfg {
            return Ok(false);
        }
        let planes = if cur.dual_plane { 2 } else { 1 };
        let thresh = (u32::from(cur.grid_width) * u32::from(cur.grid_height) * 45 + 64) >> 7;
        for syms in blk_out.packed_dct_plane_data.iter().take(planes) {
            if syms.coeffs.is_empty() {
                return Err(Error::InvalidArgument);
            }
            if syms.max_coeff_mag > DCT_MAX_ARITH_COEFF_MAG {
                return Ok(false);
            }
            if syms.coeffs.len() > thresh as usize {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn put_dct(&mut self, syms: &DctSyms) {
        if syms.num_dc_levels == DCT_MEAN_LEVELS1 {
            self.mean1_bytes.push(syms.dc_sym as u8);
        } else {
            self.mean0_bits.put_bits(syms.dc_sym, 4);
        }
        for c in &syms.coeffs {
            if c.coeff == i16::MAX {
                self.run_bytes.push(DCT_RUN_LEN_EOB_SYM_INDEX as u8);
            } else {
                self.run_bytes.push(c.num_zeros as u8);
                self.sign_bits.put_bits(u32::from(c.coeff < 0), 1);
                self.coeff_bytes
                    .push((c.coeff.unsigned_abs() as i32 - 1) as u8);
            }
        }
    }

    pub fn put_dpcm(&mut self, cur: &LogAstcBlock) {
        let levels = ah::get_ise_levels(u32::from(cur.weight_ise_range));
        weight_dpcm_syms(cur, |w| {
            if levels <= 4 {
                self.weight2_bits.put_bits(w, 2);
            } else if levels <= 8 {
                self.weight3_bits.put_bits(w, 4);
            } else if levels <= 16 {
                self.weight4_bits.put_bits(w, 4);
            } else {
                self.weight8_bits.push(w as u8);
            }
        });
    }

    pub fn flush(&mut self) {
        self.mean0_bits.flush();
        self.sign_bits.flush();
        self.weight2_bits.flush();
        self.weight3_bits.flush();
        self.weight4_bits.flush();
    }
}

// ---------------------------------------------------------------------------------------
// Entry points

/// `astc_ldr::compress_image`'s supercompression stage: writes the XUASTC LDR bitstream for the
/// candidates in `enc_out` with the configured syntax and returns it with the coded blocks.
pub(crate) fn supercompress(
    img: &Image,
    cfg: &AstcLdrEncodeConfig,
    enc_out: &EncodeImageOutput,
) -> Result<CompressOutput> {
    supercompress_dims(
        img.get_width(),
        img.get_height(),
        img.has_alpha(3),
        cfg,
        enc_out,
    )
}

/// [`supercompress`] given the image dimensions and alpha flag (`image::has_alpha()`).
pub(crate) fn supercompress_dims(
    width: u32,
    height: u32,
    has_alpha: bool,
    cfg: &AstcLdrEncodeConfig,
    enc_out: &EncodeImageOutput,
) -> Result<CompressOutput> {
    let (bw, bh) = (cfg.astc_block_width, cfg.astc_block_height);
    let size_index = BASIS_ASTC_ORDER
        .iter()
        .position(|b| b.dim() == (bw, bh))
        .ok_or(Error::InvalidArgument)?;
    let tms = mode_table(size_index).ok_or(Error::InvalidArgument)?;
    if !(1..=65535).contains(&width) || !(1..=65535).contains(&height) {
        return Err(Error::InvalidArgument);
    }
    let (bxs, bys) = (width.div_ceil(bw), height.div_ceil(bh));
    if enc_out.blocks_x != bxs
        || enc_out.blocks_y != bys
        || enc_out.block_info.len() != (bxs * bys) as usize
    {
        return Err(Error::InvalidArgument);
    }
    if !(1.0..=100.0).contains(&cfg.dct_quality) {
        return Err(Error::InvalidArgument);
    }
    let int_q = ((cfg.dct_quality * 2.0f32).round() as i32).clamp(0, 200) as u32;
    let ctx = Ctx {
        enc_out,
        tms,
        size_index,
        width,
        height,
        bw,
        bh,
        bxs,
        bys,
        has_alpha,
        srgb: cfg.astc_decode_mode_srgb,
        use_dct: cfg.use_dct,
        int_q,
        effort_level: cfg.effort_level,
        // SCD is incompatible with lossy supercompression.
        lossy: cfg.lossy_supercompression && !cfg.scd_enabled,
        replacement_min_psnr: if has_alpha {
            cfg.replacement_min_psnr_alpha
        } else {
            cfg.replacement_min_psnr
        },
        psnr_trial_diff_thresh: if has_alpha {
            cfg.psnr_trial_diff_thresh_alpha
        } else {
            cfg.psnr_trial_diff_thresh
        },
        psnr_trial_diff_thresh_edge: if has_alpha {
            cfg.psnr_trial_diff_thresh_edge_alpha
        } else {
            cfg.psnr_trial_diff_thresh_edge
        },
        total_comp_weights: (cfg.comp_weights[0]
            + cfg.comp_weights[1]
            + cfg.comp_weights[2]
            + cfg.comp_weights[3]) as f32,
    };
    let mut syntax = cfg.compressed_syntax;
    // Full arithmetic coding on the smallest images avoids the Zstd overhead.
    if bxs * bys <= DISABLE_FASTER_FORMAT_TOTAL_BLOCKS_THRESH {
        syntax = XuastcSyntax::FullArith;
    }
    match syntax {
        XuastcSyntax::FullZstd => compress_image_full_zstd(&ctx),
        XuastcSyntax::FullArith => compress_image_arith(&ctx, false),
        XuastcSyntax::HybridArithZstd => compress_image_arith(&ctx, true),
    }
}

// ---------------------------------------------------------------------------------------
// Arithmetic syntaxes

/// `basist::astc_ldr_t::prev_block_state`.
#[derive(Clone, Copy, Default)]
struct PrevBlockState {
    was_solid_color: bool,
    used_weight_dct: bool,
    first_endpoint_uses_bc: bool,
    reused_full_cfg: bool,
    used_part_hash: bool,
    tm_index: i32,
    base_cem_index: u32,
    subset_index: u32,
    ccs_index: u32,
    grid_size: u32,
    grid_aniso: u32,
}

/// Model index from a flag of the left and upper states (missing neighbors count as set).
fn ctx2(
    left: Option<&PrevBlockState>,
    upper: Option<&PrevBlockState>,
    f: fn(&PrevBlockState) -> bool,
) -> usize {
    usize::from(left.is_none_or(f)) | (usize::from(upper.is_none_or(f)) << 1)
}

/// The adaptive models of `compress_image`.
struct Models {
    mode: DataModel,
    solid_color_dpcm: [DataModel; 4],
    raw_endpoints: Vec<DataModel>,
    dpcm_endpoints: Vec<DataModel>,
    raw_weights: Vec<DataModel>,
    is_base_ofs: BitModel,
    use_dct: [BitModel; 4],
    use_dpcm_endpoints: BitModel,
    cem_index: [DataModel; 8],
    subset_index: Vec<DataModel>,
    ccs_index: Vec<DataModel>,
    grid_size: Vec<DataModel>,
    grid_aniso: Vec<DataModel>,
    dct_run_len: DataModel,
    dct_coeff_mag: DataModel,
    submodes: Vec<DataModel>,
    endpoints_use_bc: [BitModel; 4],
    endpoint_reuse_delta: DataModel,
    weight_mean: [DataModel; 2],
    config_reuse: [DataModel; 4],
    use_part_hash: [BitModel; 4],
    part2_hash_index: DataModel,
    part3_hash_index: DataModel,
    run_len: GammaContexts,
}

impl Models {
    fn new() -> Self {
        let n = |syms: u32| DataModel::new(syms, false);
        let range_models = |first: u32, count: u32| {
            (0..count)
                .map(|i| n(ah::get_ise_levels(first + i)))
                .collect::<Vec<_>>()
        };
        let rep = |count: u32, syms: u32| (0..count).map(|_| n(syms)).collect::<Vec<_>>();
        Self {
            mode: n(MODE_TOTAL),
            solid_color_dpcm: std::array::from_fn(|_| DataModel::new(256, true)),
            raw_endpoints: range_models(FIRST_VALID_ENDPOINT_ISE_RANGE, 17),
            dpcm_endpoints: range_models(FIRST_VALID_ENDPOINT_ISE_RANGE, 17),
            raw_weights: range_models(FIRST_VALID_WEIGHT_ISE_RANGE, 12),
            is_base_ofs: BitModel::default(),
            use_dct: [BitModel::default(); 4],
            use_dpcm_endpoints: BitModel::default(),
            cem_index: std::array::from_fn(|_| n(OTM_NUM_CEMS)),
            subset_index: rep(OTM_NUM_SUBSETS, OTM_NUM_SUBSETS),
            ccs_index: rep(OTM_NUM_CCS, OTM_NUM_CCS),
            grid_size: rep(OTM_NUM_GRID_SIZES, OTM_NUM_GRID_SIZES),
            grid_aniso: rep(OTM_NUM_GRID_ANISOS, OTM_NUM_GRID_ANISOS),
            dct_run_len: n(65),
            dct_coeff_mag: n(255),
            submodes: vec![DataModel::default(); NUM_GROUPS],
            endpoints_use_bc: [BitModel::default(); 4],
            endpoint_reuse_delta: n(REUSE_XY_DELTAS.len() as u32),
            weight_mean: [n(DCT_MEAN_LEVELS0), n(DCT_MEAN_LEVELS1)],
            config_reuse: std::array::from_fn(|_| n(MAX_CONFIG_REUSE_NEIGHBORS + 1)),
            use_part_hash: [BitModel::default(); 4],
            part2_hash_index: DataModel::new(PART_HASH_SIZE as u32, true),
            part3_hash_index: DataModel::new(PART_HASH_SIZE as u32, true),
            run_len: GammaContexts::default(),
        }
    }
}

fn model(v: &mut [DataModel], i: u32) -> Result<&mut DataModel> {
    v.get_mut(i as usize).ok_or(Error::InvalidArgument)
}

/// `compress_image` (full arithmetic or hybrid arithmetic + Zstd syntax).
fn compress_image_arith(ctx: &Ctx, use_faster_format: bool) -> Result<CompressOutput> {
    let (bxs, bys) = (ctx.bxs, ctx.bys);
    let mut ws = WeightStreams::default();
    let mut enc = ArithEnc::new(1024 * 1024);

    enc.put_bits(ARITH_HEADER_MARKER, ARITH_HEADER_MARKER_BITS);
    enc.put_bits(ctx.size_index as u32, 4);
    enc.put_bit(u32::from(ctx.srgb));
    enc.put_bits(ctx.width, 16);
    enc.put_bits(ctx.height, 16);
    enc.put_bit(u32::from(ctx.has_alpha));
    enc.put_bits(u32::from(ctx.use_dct), 1);
    if ctx.use_dct {
        enc.put_bits(ctx.int_q, 8);
    }

    let mut m = Models::new();
    let mut states = vec![PrevBlockState::default(); (bxs * bys) as usize];
    let mut coded = vec![LogAstcBlock::default(); (bxs * bys) as usize];
    let mut part2_hash = [-1i32; PART_HASH_SIZE];
    let mut part3_hash = [-1i32; PART_HASH_SIZE];
    let mut cur_run_len = 0u32;

    for by in 0..bys {
        for bx in 0..bxs {
            let idx = (bx + by * bxs) as usize;
            let left = (bx > 0).then(|| states[idx - 1]);
            let upper = (by > 0).then(|| states[idx - bxs as usize]);
            let diag = (bx > 0 && by > 0).then(|| states[idx - bxs as usize - 1]);
            let (left, upper, diag) = (left.as_ref(), upper.as_ref(), diag.as_ref());
            let pred_state = left.or(upper);
            let blk_info = ctx.info(bx, by)?;
            let packed = blk_info.packed_out_block_index as usize;

            // Runs
            if bx > 0 || by > 0 {
                let cur = &blk_info
                    .out_blocks
                    .get(packed)
                    .ok_or(Error::InvalidArgument)?
                    .log_blk;
                let prev = *ctx.prev(&coded, bx, by).ok_or(Error::InvalidArgument)?;
                if compare_log_blocks_for_equality(cur, &prev) {
                    cur_run_len += 1;
                    coded[idx] = prev;
                    let p = pred_state.ok_or(Error::InvalidArgument)?;
                    states[idx] = PrevBlockState {
                        reused_full_cfg: true,
                        ..*p
                    };
                    continue;
                }
            }
            if cur_run_len > 0 {
                enc.encode(MODE_RUN, &mut m.mode);
                enc.put_gamma(cur_run_len, &mut m.run_len);
                cur_run_len = 0;
            }

            let best = ctx.select_candidate(blk_info, &coded, bx, by)?;
            let blk_out = blk_info
                .out_blocks
                .get(best)
                .ok_or(Error::InvalidArgument)?;
            let cur = blk_out.log_blk;
            coded[idx] = cur;
            let mut st = PrevBlockState::default();

            if blk_out.trial_mode_index < 0 {
                if !cur.solid_color_flag_ldr {
                    return Err(Error::InvalidArgument);
                }
                enc.encode(MODE_SOLID, &mut m.mode);
                let (d, n) = ctx.solid_deltas(&coded, bx, by, &cur);
                for (i, &v) in d.iter().enumerate().take(n) {
                    enc.encode(v, &mut m.solid_color_dpcm[i]);
                }
                states[idx] = PrevBlockState {
                    was_solid_color: true,
                    // Bias the statistics towards using DCT (most common case).
                    used_weight_dct: ctx.use_dct,
                    first_endpoint_uses_bc: true,
                    tm_index: -1,
                    base_cem_index: CEM_LDR_RGB_DIRECT,
                    reused_full_cfg: false,
                    used_part_hash: true,
                    ..PrevBlockState::default()
                };
                continue;
            }
            if cur.solid_color_flag_ldr {
                return Err(Error::InvalidArgument);
            }
            let tm = ctx.tm(blk_out)?;
            let cur_cem = u32::from(cur.color_endpoint_modes[0]);
            let num_parts = usize::from(tm.num_parts);

            if let Some(ri) = ctx.full_cfg_endpoint_reuse_index(&coded, bx, by, &cur) {
                // Full configuration, partition id and endpoint reuse.
                enc.encode(MODE_REUSE_CFG_ENDPOINTS_LEFT + ri, &mut m.mode);
                let rs = *[left, upper, diag][ri as usize].ok_or(Error::InvalidArgument)?;
                st.tm_index = i32::from(blk_out.trial_mode_index);
                st.base_cem_index = rs.base_cem_index;
                st.subset_index = rs.subset_index;
                st.ccs_index = rs.ccs_index;
                st.grid_size = rs.grid_size;
                st.grid_aniso = rs.grid_aniso;
                st.used_part_hash = rs.used_part_hash;
                st.reused_full_cfg = true;
                if cem_supports_bc(cur_cem) {
                    st.first_endpoint_uses_bc = used_blue_contraction(
                        cur_cem,
                        &cur.endpoints,
                        u32::from(cur.endpoint_ise_range),
                    );
                }
            } else {
                enc.encode(MODE_RAW, &mut m.mode);

                // Configuration + partition id reuse.
                let neighbor_cfg_match = (0..MAX_CONFIG_REUSE_NEIGHBORS).find(|&i| {
                    let Some(ns) = [left, upper, diag][i as usize] else {
                        return false;
                    };
                    let Some(n) = ctx.neighbor(&coded, bx, by, i) else {
                        return false;
                    };
                    ns.tm_index == i32::from(blk_out.trial_mode_index)
                        && n.color_endpoint_modes[0] == cur.color_endpoint_modes[0]
                        && n.partition_id == cur.partition_id
                });
                let reuse_ctx = ctx2(left, upper, |s| s.reused_full_cfg);
                if let Some(ni) = neighbor_cfg_match {
                    enc.encode(ni, &mut m.config_reuse[reuse_ctx]);
                    let rs = *[left, upper, diag][ni as usize].ok_or(Error::InvalidArgument)?;
                    st.tm_index = i32::from(blk_out.trial_mode_index);
                    st.base_cem_index = rs.base_cem_index;
                    st.subset_index = rs.subset_index;
                    st.ccs_index = rs.ccs_index;
                    st.grid_size = rs.grid_size;
                    st.grid_aniso = rs.grid_aniso;
                    st.used_part_hash = rs.used_part_hash;
                    st.reused_full_cfg = true;
                } else {
                    enc.encode(MAX_CONFIG_REUSE_NEIGHBORS, &mut m.config_reuse[reuse_ctx]);
                    let [cem_index, subset_index, ccs_index, grid_size, aniso] =
                        separate_tm_index(ctx.bw, ctx.bh, &tm);
                    let submodes =
                        ctx.tms
                            .candidates(cem_index, subset_index, ccs_index, grid_size, aniso);
                    let submode_index = submodes
                        .iter()
                        .position(|&s| i32::from(s) == i32::from(blk_out.trial_mode_index))
                        .ok_or(Error::InvalidArgument)?
                        as u32;
                    let p = pred_state.copied().unwrap_or(PrevBlockState {
                        base_cem_index: CEM_LDR_RGB_DIRECT,
                        ..PrevBlockState::default()
                    });
                    enc.encode(
                        cem_index,
                        &mut m.cem_index[cem_to_ldrcem_index(p.base_cem_index)],
                    );
                    enc.encode(subset_index, model(&mut m.subset_index, p.subset_index)?);
                    enc.encode(ccs_index, model(&mut m.ccs_index, p.ccs_index)?);
                    enc.encode(grid_size, model(&mut m.grid_size, p.grid_size)?);
                    enc.encode(aniso, model(&mut m.grid_aniso, p.grid_aniso)?);
                    if submodes.len() > 1 {
                        let g = group_index(cem_index, subset_index, ccs_index, grid_size, aniso)
                            .ok_or(Error::InvalidArgument)?;
                        let sm = m.submodes.get_mut(g).ok_or(Error::InvalidArgument)?;
                        if sm.num_data_syms() == 0 {
                            sm.init(submodes.len() as u32, true);
                        }
                        enc.encode(submode_index, sm);
                    }
                    st.tm_index = i32::from(blk_out.trial_mode_index);
                    st.base_cem_index = cem_index;
                    st.subset_index = subset_index;
                    st.ccs_index = ccs_index;
                    st.grid_size = grid_size;
                    st.grid_aniso = aniso;
                    st.reused_full_cfg = false;

                    if tm_is_direct(&tm) {
                        enc.encode_bit(u32::from(is_base_ofs(&cur)), &mut m.is_base_ofs);
                    }
                    if tm.num_parts > 1 {
                        let parts = u32::from(tm.num_parts);
                        let (unique, total) =
                            part_seed_to_unique(ctx.size_index, parts, cur.partition_id)?;
                        let hctx = ctx2(left, upper, |s| s.used_part_hash);
                        let hash = if parts == 2 {
                            &mut part2_hash
                        } else {
                            &mut part3_hash
                        };
                        let h = fib_hash(unique, PART_HASH_SIZE);
                        if hash[h] != unique as i32 {
                            enc.encode_bit(0, &mut m.use_part_hash[hctx]);
                            enc.put_truncated_binary(unique, total);
                            st.used_part_hash = false;
                        } else {
                            enc.encode_bit(1, &mut m.use_part_hash[hctx]);
                            let hm = if parts == 2 {
                                &mut m.part2_hash_index
                            } else {
                                &mut m.part3_hash_index
                            };
                            enc.encode(h as u32, hm);
                            st.used_part_hash = true;
                        }
                        hash[h] = unique as i32;
                    } else {
                        st.used_part_hash = true; // bias to true
                    }
                }

                // Endpoints
                let bc_ctx = ctx2(left, upper, |s| s.first_endpoint_uses_bc);
                let use_bc = endpoints_use_bc(&cur, num_parts);
                if cem_supports_bc(cur_cem) {
                    st.first_endpoint_uses_bc = use_bc[0];
                }
                let erange = u32::from(cur.endpoint_ise_range);
                let emodel = erange
                    .checked_sub(FIRST_VALID_ENDPOINT_ISE_RANGE)
                    .ok_or(Error::InvalidArgument)?;
                let pred = ctx.find_endpoint_pred(
                    &coded,
                    bx,
                    by,
                    &cur,
                    num_parts,
                    &use_bc,
                    Some(&m.endpoint_reuse_delta),
                );
                if let Some((reuse_index, predicted)) = pred {
                    enc.encode_bit(1, &mut m.use_dpcm_endpoints);
                    enc.encode(reuse_index, &mut m.endpoint_reuse_delta);
                    if cem_supports_bc(cur_cem) {
                        for &u in use_bc.iter().take(num_parts) {
                            enc.encode_bit(u32::from(u), &mut m.endpoints_use_bc[bc_ctx]);
                        }
                    }
                    let dm = model(&mut m.dpcm_endpoints, emodel)?;
                    endpoint_dpcm_syms(&cur, &predicted, num_parts, |v| enc.encode(v, dm));
                } else {
                    enc.encode_bit(0, &mut m.use_dpcm_endpoints);
                    let rm = model(&mut m.raw_endpoints, emodel)?;
                    let n = num_parts * ah::get_num_cem_values(cur_cem) as usize;
                    for &e in cur.endpoints.iter().take(n) {
                        enc.encode(u32::from(e), rm);
                    }
                }
            }

            // Weights
            let planes = if cur.dual_plane { 2 } else { 1 };
            let use_dct_ctx = if ctx.use_dct {
                ctx2(left, upper, |s| s.used_weight_dct)
            } else {
                0
            };
            if use_faster_format {
                let use_dct = WeightStreams::use_dct(ctx.use_dct, &cur, blk_out)?;
                if ctx.use_dct {
                    enc.encode_bit(u32::from(use_dct), &mut m.use_dct[use_dct_ctx]);
                }
                if use_dct {
                    st.used_weight_dct = true;
                    for syms in blk_out.packed_dct_plane_data.iter().take(planes) {
                        ws.put_dct(syms);
                    }
                } else {
                    ws.put_dpcm(&cur);
                }
            } else {
                let wmodel = u32::from(cur.weight_ise_range)
                    .checked_sub(FIRST_VALID_WEIGHT_ISE_RANGE)
                    .ok_or(Error::InvalidArgument)?;
                let mut total_dpcm_bits = 0.0f32;
                {
                    let wm = model(&mut m.raw_weights, wmodel)?;
                    weight_dpcm_syms(&cur, |w| total_dpcm_bits += wm.get_price(w));
                }
                const FORBID_DCT_BITS: f32 = 1e8;
                let mut total_dct_bits = 0.0f32;
                if ctx.use_dct {
                    let planes_syms = || blk_out.packed_dct_plane_data.iter().take(planes);
                    if planes_syms().any(|s| s.max_coeff_mag > DCT_MAX_ARITH_COEFF_MAG) {
                        total_dct_bits = FORBID_DCT_BITS;
                    }
                    if total_dct_bits < FORBID_DCT_BITS {
                        for syms in planes_syms() {
                            let mi = usize::from(syms.num_dc_levels == DCT_MEAN_LEVELS1);
                            total_dct_bits += m.weight_mean[mi].get_price(syms.dc_sym);
                            for c in &syms.coeffs {
                                if c.coeff == i16::MAX {
                                    total_dct_bits +=
                                        m.dct_run_len.get_price(DCT_RUN_LEN_EOB_SYM_INDEX);
                                } else {
                                    total_dct_bits +=
                                        m.dct_run_len.get_price(u32::from(c.num_zeros));
                                    total_dct_bits += 1.0;
                                    total_dct_bits += m
                                        .dct_coeff_mag
                                        .get_price(u32::from(c.coeff.unsigned_abs()) - 1);
                                }
                            }
                        }
                    }
                }
                let udm = &m.use_dct[use_dct_ctx];
                let use_dct = ctx.use_dct
                    && total_dct_bits < FORBID_DCT_BITS
                    && total_dct_bits + udm.get_price(true)
                        <= total_dpcm_bits + udm.get_price(false);
                if ctx.use_dct {
                    enc.encode_bit(u32::from(use_dct), &mut m.use_dct[use_dct_ctx]);
                }
                if use_dct {
                    st.used_weight_dct = true;
                    for syms in blk_out.packed_dct_plane_data.iter().take(planes) {
                        let mi = usize::from(syms.num_dc_levels == DCT_MEAN_LEVELS1);
                        enc.encode(syms.dc_sym, &mut m.weight_mean[mi]);
                        for c in &syms.coeffs {
                            if c.coeff == i16::MAX {
                                enc.encode(DCT_RUN_LEN_EOB_SYM_INDEX, &mut m.dct_run_len);
                            } else {
                                enc.encode(u32::from(c.num_zeros), &mut m.dct_run_len);
                                enc.put_bit(u32::from(c.coeff < 0));
                                enc.encode(
                                    u32::from(c.coeff.unsigned_abs()) - 1,
                                    &mut m.dct_coeff_mag,
                                );
                            }
                        }
                    }
                } else {
                    let wm = model(&mut m.raw_weights, wmodel)?;
                    weight_dpcm_syms(&cur, |w| enc.encode(w, wm));
                }
            }
            states[idx] = st;
        }
        if cur_run_len > 0 {
            enc.encode(MODE_RUN, &mut m.mode);
            enc.put_gamma(cur_run_len, &mut m.run_len);
            cur_run_len = 0;
        }
    }

    enc.put_bits(FINAL_SYNC_MARKER, FINAL_SYNC_MARKER_BITS);
    enc.flush();
    let arith = enc.into_data_buf();

    let mut comp_data = Vec::new();
    if use_faster_format {
        ws.flush();
        const ZSTD_LEVEL: i32 = 9;
        let parts = [
            zstd_compress(ws.mean0_bits.get_bytes(), ZSTD_LEVEL)?,
            zstd_compress(&ws.mean1_bytes, ZSTD_LEVEL)?,
            zstd_compress(&ws.run_bytes, ZSTD_LEVEL)?,
            zstd_compress(&ws.coeff_bytes, ZSTD_LEVEL)?,
            ws.sign_bits.get_bytes().clone(),
            zstd_compress(ws.weight2_bits.get_bytes(), ZSTD_LEVEL)?,
            zstd_compress(ws.weight3_bits.get_bytes(), ZSTD_LEVEL)?,
            zstd_compress(ws.weight4_bits.get_bytes(), ZSTD_LEVEL)?,
            zstd_compress(&ws.weight8_bits, ZSTD_LEVEL)?,
        ];
        comp_data.push(crate::basis::xuastc::SYNTAX_HYBRID_ARITH_ZSTD);
        comp_data.extend_from_slice(&len_u32(arith.len())?.to_le_bytes());
        for p in &parts {
            comp_data.extend_from_slice(&len_u32(p.len())?.to_le_bytes());
        }
        comp_data.extend_from_slice(&0u32.to_le_bytes()); // m_unused
        comp_data.extend_from_slice(&arith);
        for p in &parts {
            comp_data.extend_from_slice(p);
        }
    } else {
        comp_data.push(XuastcSyntax::FullArith as u8);
        comp_data.extend_from_slice(&arith);
    }
    len_u32(comp_data.len())?;
    Ok(CompressOutput {
        comp_data,
        coded_blocks: coded,
    })
}

pub(super) fn len_u32(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::InvalidArgument)
}

#[cfg(test)]
#[path = "supercomp_tests.rs"]
mod tests;
