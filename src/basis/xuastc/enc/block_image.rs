// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! The full ("basisu") ASTC LDR / XUASTC LDR image encoder: `ldr_astc_block_encode_image` and
//! its helpers (`encoder_config_manager`, `encode_block_to_dc_latent`,
//! `encode_block_to_linear_latent`, `enforce_max_candidate_limit`) from
//! `basisu_astc_ldr_encode.cpp`.
//!
//! Data flow (identical to C++, independent of the thread count): superpass 0 encodes every
//! block independently; superpass 1 re-encodes the worst blocks (by WSSE) independently;
//! superpass 2 reads the (frozen) superpass 0/1 results of each block's left, upper and
//! upper-left neighbors and writes per-block side lists, which are then merged serially in
//! raster order. Debug images, statistics and the final multi-decoder validation are not
//! ported.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::if_same_then_else,
    clippy::vec_box
)]

use std::sync::Mutex;

use super::block_helpers::{
    SOBEL_X, SOBEL_Y, TrialFlags, compute_lost_dct_energy, compute_preserved_dct_energy,
    compute_sobel, decode_block_xuastc_ldr, encode_trial, encode_trial_subsets, eval_error,
    polish_block_weights, weighted_color_error,
};
use super::common::{
    ASTC_LDR_MAX_BLOCK_PIXELS, CemEncodeParams, PixelStats, cem_encode_pixels, dot3, saturate,
};
use super::dct_enc::{Dct2f, GridWeightDct, apply_weight_grid_dct, code_block_weights};
use super::effort::HighLevelConfig;
use super::grid_data::find_astc_block_grid_data;
use super::lowlevel::{
    EncodeBlockStats, LdrAstcLowlevelBlockEncoder, LdrAstcLowlevelBlockEncoderParams,
};
use super::partitions::{PartitionsData, get_precomputed_texel_partition};
use super::{
    BLUR_ID_AC_LATENT_BASE, BLUR_ID_DC_LATENT_BASE, BLUR_ID_GAUSSIAN_ALTERNATE, BlockInfo,
    EncodeBlockOutput, EncodeImageOutput,
};
use crate::astc::helpers::{
    self as ah, BISE_16_LEVELS, BISE_64_LEVELS, BISE_256_LEVELS, LogAstcBlock,
};
use crate::basis::file::BASIS_ASTC_ORDER;
use crate::basis::ll::comp::{Image, image_resample};
use crate::basis::ll::enc::{ColorRgba, indirect_sort};
use crate::basis::ll::job_pool::map_jobs;
use crate::basis::xuastc::{MAX_CONFIG_REUSE_NEIGHBORS, ModeTable, mode_table};
use crate::{Error, Result};

/// `TOTAL_BLURRED_IMAGES`.
const TOTAL_BLURRED_IMAGES: usize = 3;
/// `selective_blurring`.
const SELECTIVE_BLURRING: bool = true;
/// `TOTAL_RGBA_CHAN_PAIRS`.
const TOTAL_RGBA_CHAN_PAIRS: usize = 6;
/// `g_rgba_chan_pairs`.
const G_RGBA_CHAN_PAIRS: [[u8; 2]; TOTAL_RGBA_CHAN_PAIRS] =
    [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]];
/// `cInvalidIndex`.
const INVALID_INDEX: i32 = -1;
/// `cPiD`.
const PI_D: f64 = std::f64::consts::PI;

const MAXP: usize = ASTC_LDR_MAX_BLOCK_PIXELS;

/// `astc_helpers::get_block_size_index`.
fn block_size_index(bw: u32, bh: u32) -> Result<usize> {
    BASIS_ASTC_ORDER
        .iter()
        .position(|b| b.dim() == (bw, bh))
        .ok_or(Error::InvalidArgument)
}

/// `astc_helpers::decode_block_xuastc_ldr` (always succeeds for the encoder's blocks).
fn decode_xu(blk: &LogAstcBlock, out: &mut [ColorRgba], bw: u32, bh: u32, srgb: bool) -> bool {
    if blk.error_flag || out.len() < (bw * bh) as usize {
        return false;
    }
    decode_block_xuastc_ldr(blk, out, bw, bh, srgb);
    true
}

#[inline]
fn maxf(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[inline]
fn minf(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `encoder_config_manager`.
struct EncoderConfigManager {
    max_std_dev: f32,
    sobel_energy: f32,
    is_lum_only: bool,
    block_dct_energy: Vec<f32>,
    filter_horizontally_flag: bool,
    low_freq_block_flag: bool,
    total_active_chans: u32,
    chan_pair_correlations: [f32; TOTAL_RGBA_CHAN_PAIRS],
    active_chan_flags: [bool; 4],
    min_corr: f32,
    max_corr: f32,
    used_alpha_encoder_modes: bool,
    temp_cem_enc_params: CemEncodeParams<'static>,
    // work buffers
    block_floats: Vec<f32>,
    block_dct: Vec<f32>,
}

impl EncoderConfigManager {
    fn new() -> Self {
        Self {
            max_std_dev: 0.0,
            sobel_energy: 0.0,
            is_lum_only: false,
            block_dct_energy: Vec::new(),
            filter_horizontally_flag: false,
            low_freq_block_flag: false,
            total_active_chans: 0,
            chan_pair_correlations: [0.0; TOTAL_RGBA_CHAN_PAIRS],
            active_chan_flags: [false; 4],
            min_corr: 0.0,
            max_corr: 0.0,
            used_alpha_encoder_modes: false,
            temp_cem_enc_params: CemEncodeParams::default(),
            block_floats: Vec::new(),
            block_dct: Vec::new(),
        }
    }

    /// `encoder_config_manager::init`.
    fn init(
        &mut self,
        bx: u32,
        by: u32,
        block_width: u32,
        block_height: u32,
        total_block_pixels: u32,
        pixel_stats: &PixelStats,
        dct: &Dct2f,
        enc_cfg: &HighLevelConfig,
        sobel_xy: &Image,
    ) {
        let tbp = total_block_pixels as usize;

        self.max_std_dev = 0.0;
        for i in 0..4 {
            self.max_std_dev = maxf(self.max_std_dev, pixel_stats.rgba_stats[i].std_dev);
        }

        // Note: C++ only ends up examining the first row (its outer loop breaks while the
        // block is still luminance only).
        self.is_lum_only = true;
        'outer: for y in 0..block_height {
            for x in 0..block_width {
                let c = pixel_stats.pixels[(x + y * block_width) as usize];
                let is_lum_texel = c[0] == c[1] && c[0] == c[2];
                if !is_lum_texel {
                    self.is_lum_only = false;
                    break;
                }
            }
            if self.is_lum_only {
                break 'outer;
            }
        }

        self.block_dct_energy.clear();
        self.block_dct_energy.resize(tbp, 0.0);

        self.filter_horizontally_flag = false;
        self.low_freq_block_flag = false;

        {
            self.block_floats.clear();
            self.block_floats.resize(tbp, 0.0);
            self.block_dct.clear();
            self.block_dct.resize(tbp, 0.0);

            for c in 0..4 {
                for i in 0..tbp {
                    self.block_floats[i] = pixel_stats.pixels_f[i][c];
                }

                dct.forward(&self.block_floats, &mut self.block_dct);

                let w = enc_cfg.cem_enc_params.comp_weights[c] as f32;
                for y in 0..block_height {
                    for x in 0..block_width {
                        let k = (x + y * block_width) as usize;
                        let d = self.block_dct[k];
                        self.block_dct_energy[k] += w * (d * d);
                    }
                }
            }

            // Wipe DC
            self.block_dct_energy[0] = 0.0;

            let tot_energy = compute_preserved_dct_energy(
                block_width,
                block_height,
                &self.block_dct_energy,
                block_width,
                block_height,
            );

            let h_energy_lost = compute_lost_dct_energy(
                block_width,
                block_height,
                &self.block_dct_energy,
                block_width / 2,
                block_height,
            );
            let v_energy_lost = compute_lost_dct_energy(
                block_width,
                block_height,
                &self.block_dct_energy,
                block_width,
                block_height / 2,
            );

            self.filter_horizontally_flag = h_energy_lost < v_energy_lost;

            let mut hv2_lost_energy_fract =
                compute_lost_dct_energy(block_width, block_height, &self.block_dct_energy, 2, 2);
            if tot_energy != 0.0 {
                hv2_lost_energy_fract /= tot_energy;
            }

            const LOW_FREQ_BLOCK_LOST_ENERGY_FRACT_THRESH: f32 = 0.03;
            const LOW_FREQ_BLOCK_MAX_STD_DEV_TRESH: f32 = 1.0 / 255.0;
            if hv2_lost_energy_fract < LOW_FREQ_BLOCK_LOST_ENERGY_FRACT_THRESH
                || self.max_std_dev < LOW_FREQ_BLOCK_MAX_STD_DEV_TRESH
            {
                self.low_freq_block_flag = true;
            }
        }

        self.active_chan_flags = [false; 4];
        self.total_active_chans = 0;
        for i in 0..4 {
            if pixel_stats.rgba_stats[i].range > 0.0 {
                self.active_chan_flags[i] = true;
                self.total_active_chans += 1;
            }
        }

        // def=max correlation for each channel pair (or 1 if one of the channels is inactive)
        self.chan_pair_correlations = [1.0; TOTAL_RGBA_CHAN_PAIRS];

        self.min_corr = 1.0;
        self.max_corr = 0.0;

        for pair_index in 0..TOTAL_RGBA_CHAN_PAIRS {
            let chan_a = G_RGBA_CHAN_PAIRS[pair_index][0] as usize;
            let chan_b = G_RGBA_CHAN_PAIRS[pair_index][1] as usize;

            if self.active_chan_flags[chan_a] && self.active_chan_flags[chan_b] {
                // `comparative_stats<float>::calc_pearson`
                let sa = &pixel_stats.rgba_stats[chan_a];
                let sb = &pixel_stats.rgba_stats[chan_b];
                let mut cov = 0.0f32;
                for i in 0..tbp {
                    let fa = pixel_stats.pixels_f[i][chan_a];
                    let fb = pixel_stats.pixels_f[i][chan_b];
                    let da = fa - sa.avg;
                    let db = fb - sb.avg;
                    cov += da * db;
                }
                cov /= tbp as f32;
                let dv = sa.std_dev * sb.std_dev;
                let pearson = if dv != 0.0 { cov / dv } else { 0.0 };

                self.chan_pair_correlations[pair_index] = pearson.abs();

                let c = pearson.abs();
                self.min_corr = minf(self.min_corr, c);
                self.max_corr = maxf(self.max_corr, c);
            }
        }

        self.used_alpha_encoder_modes = pixel_stats.has_alpha;

        self.sobel_energy = 0.0;
        for y in 0..block_height {
            for x in 0..block_width {
                let s = sobel_xy.get_clamped(
                    (bx * block_width + x) as i32,
                    (by * block_height + y) as i32,
                );
                let e = s[0] as i32 * s[0] as i32
                    + s[1] as i32 * s[1] as i32
                    + s[2] as i32 * s[2] as i32
                    + s[3] as i32 * s[3] as i32;
                self.sobel_energy += e as f32;
            }
        }

        self.sobel_energy /= total_block_pixels as f32;
    }

    /// `encoder_config_manager::select`: returns the low-level block encoder parameters.
    fn select<'a>(
        &'a mut self,
        superpass_index: u32,
        bx: u32,
        by: u32,
        ctx: &'a Ctx,
        pixel_stats: &PixelStats,
    ) -> LdrAstcLowlevelBlockEncoderParams<'a> {
        let enc_cfg = ctx.cfg;
        let mut p = LdrAstcLowlevelBlockEncoderParams::new(
            &ctx.sobel_xy,
            &ctx.part_data_p2,
            &ctx.part_data_p3,
            &enc_cfg.cem_enc_params,
            &ctx.modes.modes,
            ctx.modes,
            &ctx.dct,
        );
        let block_width = ctx.block_width;
        let block_height = ctx.block_height;

        p.block_width = block_width;
        p.block_height = block_height;
        p.total_block_pixels = ctx.total_block_pixels;
        p.bx = bx;
        p.by = by;

        let ang_dot = saturate(dot3(
            &pixel_stats.zero_rel_axis3,
            &pixel_stats.mean_rel_axis3,
        ));
        let pca_axis_angles = ang_dot.acos() * (180.0f32 / PI_D as f32);

        p.use_alpha_or_opaque_modes = self.used_alpha_encoder_modes;
        p.use_lum_direct_modes = self.is_lum_only;

        let filter_by_pca_angles_flag = if superpass_index == 1 {
            enc_cfg.filter_by_pca_angles_flag_p2
        } else {
            enc_cfg.filter_by_pca_angles_flag
        };
        if !filter_by_pca_angles_flag {
            p.use_direct_modes = true;
            p.use_base_scale_modes = true;
        } else {
            p.use_direct_modes =
                self.total_active_chans == 0 || pca_axis_angles > enc_cfg.use_direct_angle_thresh;
            p.use_base_scale_modes = pca_axis_angles <= enc_cfg.use_base_scale_angle_thresh;
        }

        p.grid_hv_filtering = enc_cfg.grid_hv_filtering;
        p.filter_horizontally_flag = self.filter_horizontally_flag;

        p.use_small_grids_only = self.low_freq_block_flag && enc_cfg.low_freq_block_filtering;

        p.subsets_enabled = enc_cfg.subsets_enabled
            && (!self.low_freq_block_flag || !enc_cfg.subsets_edge_filtering);

        p.subsets_edge_filtering = enc_cfg.subsets_edge_filtering;

        p.use_blue_contraction = enc_cfg.use_blue_contraction;
        p.final_encode_try_base_ofs = enc_cfg.use_base_ofs;

        p.superbucket_max_to_retain = enc_cfg.superbucket_max_to_retain;

        p.final_shortlist_fraction = enc_cfg.final_shortlist_fraction;
        p.final_shortlist_min_size = enc_cfg.final_shortlist_min_size;
        p.final_shortlist_max_size = enc_cfg.final_shortlist_max_size;

        p.part2_fraction_to_keep = enc_cfg.part2_fraction_to_keep;
        p.part3_fraction_to_keep = enc_cfg.part3_fraction_to_keep;
        p.base_parts2 = enc_cfg.base_parts2;
        p.base_parts3 = enc_cfg.base_parts3;
        p.gradient_descent_flag = enc_cfg.gradient_descent_flag;
        p.polish_weights_flag = enc_cfg.polish_weights_flag;
        p.qcd_enabled_flag = enc_cfg.qcd_enabled_flag;
        p.encode_trial_early_out_thresh = enc_cfg.encode_trial_early_out_thresh;
        p.encode_trial_subsets_early_out_thresh = enc_cfg.encode_trial_subsets_early_out_thresh;
        p.bucket_pruning_passes = enc_cfg.bucket_pruning_passes;

        p.alpha_cems = self.used_alpha_encoder_modes;

        p.early_stop_wpsnr = enc_cfg.early_stop_wpsnr;
        p.early_stop2_wpsnr = enc_cfg.early_stop2_wpsnr;

        p.final_encode_always_try_rgb_direct = enc_cfg.final_encode_always_try_rgb_direct;

        // Determine DP usage
        if enc_cfg.force_all_dual_plane_chan_evals {
            for i in 0..4 {
                p.dp_active_chans[i] = self.active_chan_flags[i];
            }
        } else {
            for i in 0..3 {
                p.dp_active_chans[i] = false;
            }

            // Being very conservative with alpha here - always let the analytical evaluator
            // consider it.
            p.dp_active_chans[3] = pixel_stats.has_alpha;

            if !enc_cfg.disable_rgb_dual_plane {
                let rg_corr = self.chan_pair_correlations[0];
                let rb_corr = self.chan_pair_correlations[1];
                let gb_corr = self.chan_pair_correlations[2];

                let mut desired_dp_chan_rgb: i32 = -1;

                let min_p = minf(minf(rg_corr, rb_corr), gb_corr);

                if min_p < enc_cfg.strong_dp_decorr_thresh_rgb {
                    let has_r = self.active_chan_flags[0];
                    let has_g = self.active_chan_flags[1];

                    let total_active_chans_rgb =
                        self.active_chan_flags[..3].iter().filter(|&&f| f).count();

                    if total_active_chans_rgb == 2 {
                        if !has_r {
                            desired_dp_chan_rgb = 1;
                        } else if !has_g {
                            desired_dp_chan_rgb = 0;
                        } else {
                            desired_dp_chan_rgb = 0;
                        }
                    } else if total_active_chans_rgb == 3 {
                        if rg_corr < gb_corr && rb_corr < gb_corr {
                            desired_dp_chan_rgb = 0;
                        } else if rg_corr < rb_corr && gb_corr < rb_corr {
                            desired_dp_chan_rgb = 1;
                        } else {
                            desired_dp_chan_rgb = 2;
                        }
                    }
                }

                if desired_dp_chan_rgb != -1 {
                    p.dp_active_chans[desired_dp_chan_rgb as usize] = true;
                }
            }
        }

        if !p.dp_active_chans.iter().any(|&f| f) {
            p.use_dual_planes = false;
        }

        if superpass_index == 1 {
            p.base_parts2 = enc_cfg.base_parts2_p2;
            p.base_parts3 = enc_cfg.base_parts3_p2;
            p.part2_fraction_to_keep = 1;
            p.part3_fraction_to_keep = 1;

            p.superbucket_max_to_retain = enc_cfg.superbucket_max_to_retain_p2;
            p.final_shortlist_max_size = enc_cfg.final_shortlist_max_size_p2;

            if enc_cfg.second_pass_force_subsets_enabled {
                p.subsets_enabled = true;
            }
            p.subsets_edge_filtering = false;

            if enc_cfg.force_all_dp_chans_p2 {
                p.dp_active_chans = self.active_chan_flags;
                p.use_dual_planes = true;

                if !p.dp_active_chans.iter().any(|&f| f) {
                    p.use_dual_planes = false;
                }
            }

            p.gradient_descent_flag = true;
            p.polish_weights_flag = true;

            p.use_direct_modes = true;

            p.early_stop_wpsnr = enc_cfg.early_stop_wpsnr + 2.0;
            p.early_stop2_wpsnr = enc_cfg.early_stop2_wpsnr + 2.0;

            if enc_cfg.second_pass_total_weight_refine_passes != 0 {
                self.temp_cem_enc_params = enc_cfg.cem_enc_params;
                self.temp_cem_enc_params.total_weight_refine_passes =
                    enc_cfg.second_pass_total_weight_refine_passes;
                self.temp_cem_enc_params.worst_weight_nudging_flag = true;
                self.temp_cem_enc_params.endpoint_refinement_flag = true;
                p.enc_params = &self.temp_cem_enc_params;
            }
        }

        p
    }
}

/// `filter_block` (only reachable from experimental C++ paths that are compiled out:
/// `BLOCK_PREFILTER_BY_GRID_DIM` / `BLOCK_POSTFILTERING` are false).
fn filter_block(
    block_width: u32,
    block_height: u32,
    grid_width: u32,
    grid_height: u32,
    pixel_stats: &PixelStats,
    upsampled_block: &mut [ColorRgba],
) -> Result<()> {
    let num_block_samples = (block_width * block_height) as usize;
    let num_grid_samples = (grid_width * grid_height) as usize;

    let grid_data = find_astc_block_grid_data(block_width, block_height, grid_width, grid_height)
        .ok_or(Error::InvalidArgument)?;

    // rows=output grid samples, cols=input texels
    let d = &grid_data.downsample_matrix;
    // rows=output texels, cols=input grid samples
    let u = &grid_data.upsample_matrix;

    for c in 0..4 {
        let mut downsampled_block = [0.0f32; MAXP];

        if c == 3 && !pixel_stats.has_alpha {
            for px in upsampled_block.iter_mut().take(num_block_samples) {
                px[3] = 255;
            }
            break;
        }

        for g in 0..num_grid_samples {
            let mut sum = 0.0f32;
            for i in 0..num_block_samples {
                sum += pixel_stats.pixels[i][c] as f32 * d[g * num_block_samples + i];
            }
            downsampled_block[g] = sum;
        }

        for k in 0..num_block_samples {
            let mut sum = 0.0f32;
            for i in 0..num_grid_samples {
                sum += downsampled_block[i] * u[k * num_grid_samples + i];
            }
            upsampled_block[k][c] = ((sum + 0.5) as i32).clamp(0, 255) as u8;
        }
    }
    Ok(())
}

/// XUASTC LDR trial mode indices of `encode_block_to_dc_latent`.
const S_TM_INDEX2: [u32; 14] = [
    675, 675, 974, 974, 1263, 974, 1263, 974, 1263, 1747, 1747, 2131, 2131, 2425,
];
const S_TM_INDEX3: [u32; 14] = [
    679, 679, 978, 978, 1267, 978, 1267, 978, 1267, 1751, 1751, 2135, 2135, 2429,
];

/// `encode_block_to_dc_latent`.
fn encode_block_to_dc_latent(
    block_width: u32,
    block_height: u32,
    pixel_stats: &PixelStats,
    out_blocks: &mut Vec<EncodeBlockOutput>,
    enc_cfg: &HighLevelConfig,
    part_data_p2: &PartitionsData,
    part_data_p3: &PartitionsData,
    best_2subset_seed_ids: &mut Vec<u64>,
    best_3subset_seed_ids: &mut Vec<u64>,
) -> Result<()> {
    best_2subset_seed_ids.clear();
    best_2subset_seed_ids.reserve(part_data_p2.total_unique_patterns as usize);
    best_3subset_seed_ids.clear();
    best_3subset_seed_ids.reserve(part_data_p3.total_unique_patterns as usize);

    let block_size_index = block_size_index(block_width, block_height)?;
    let total_block_pixels = (block_width * block_height) as usize;
    let srgb = enc_cfg.cem_enc_params.decode_mode_srgb;

    let mut log_blk = LogAstcBlock {
        grid_width: 3,
        grid_height: 2,
        weight_ise_range: BISE_16_LEVELS as u8,
        ..Default::default()
    };
    log_blk.color_endpoint_modes[0] = 6;
    log_blk.color_endpoint_modes[1] = 6;

    let tabs = ah::dequant_tables();

    for subsets in 2..=3u32 {
        let mut new_output_blk = EncodeBlockOutput::default();
        let mut best_log_blk = LogAstcBlock::default();
        let mut best_err = u64::MAX;

        let pat_data = if subsets == 3 {
            part_data_p3
        } else {
            part_data_p2
        };

        log_blk.num_partitions = subsets as u8;
        log_blk.endpoint_ise_range = if subsets == 3 {
            BISE_64_LEVELS
        } else {
            BISE_256_LEVELS
        } as u8;
        log_blk.color_endpoint_modes[2] = if subsets == 3 { 6 } else { 0 };

        let endpoint_quant_tab = &tabs
            .endpoint_tab(u32::from(log_blk.endpoint_ise_range))
            .val_to_ise;

        for unique_part_iter in 0..pat_data.total_unique_patterns as usize {
            let part_seed = pat_data.unique_index_to_part_seed[unique_part_iter] as u32;

            log_blk.partition_id = part_seed as u16;

            let mut sums = [[0.0f32; 4]; 3];
            let mut num = [0u32; 3];

            for y in 0..block_height {
                for x in 0..block_width {
                    let p = get_precomputed_texel_partition(
                        block_width,
                        block_height,
                        part_seed,
                        x,
                        y,
                        u32::from(log_blk.num_partitions),
                    ) as usize;
                    let pf = &pixel_stats.pixels_f[(x + y * block_width) as usize];
                    sums[p][0] += pf[0];
                    sums[p][1] += pf[1];
                    sums[p][2] += pf[2];
                    num[p] += 1;
                }
            }

            for p in 0..subsets as usize {
                if num[p] == 0 {
                    return Err(Error::InvalidData);
                }
                // vec4F /= float (per component division)
                let s = num[p] as f32;
                for c in 0..4 {
                    sums[p][c] /= s;
                }

                for c in 0..3 {
                    let v = ((255.0f32 * sums[p][c]).round() as i32).clamp(0, 255);
                    log_blk.endpoints[p * 6 + c] = endpoint_quant_tab[v as usize];
                }

                log_blk.endpoints[p * 6 + 3] =
                    endpoint_quant_tab[if subsets == 3 { 248 } else { 255 }];
            }

            let mut best_k_err = u64::MAX;

            let k_start = if subsets == 3 { 13 } else { 15 };
            for k in k_start..16u8 {
                for i in 0..6 {
                    log_blk.weights[i] = k;
                }

                let mut unpacked_pixels_astc = [[0u8; 4]; MAXP];
                if !decode_xu(
                    &log_blk,
                    &mut unpacked_pixels_astc,
                    block_width,
                    block_height,
                    srgb,
                ) {
                    return Err(Error::InvalidData);
                }

                let mut total_err = 0u64;
                for i in 0..total_block_pixels {
                    total_err += u64::from(weighted_color_error(
                        &pixel_stats.pixels[i],
                        &unpacked_pixels_astc[i],
                        &enc_cfg.cem_enc_params,
                    ));
                }

                best_k_err = best_k_err.min(total_err);

                if total_err < best_err {
                    best_err = total_err;
                    best_log_blk = log_blk;
                }
            }

            let id = (best_k_err << 10) | u64::from(log_blk.partition_id);
            if subsets == 2 {
                best_2subset_seed_ids.push(id);
            } else {
                best_3subset_seed_ids.push(id);
            }
        }

        best_2subset_seed_ids.sort_unstable();
        best_3subset_seed_ids.sort_unstable();

        new_output_blk.log_blk = best_log_blk;
        new_output_blk.trial_mode_index = if best_log_blk.num_partitions == 2 {
            S_TM_INDEX2[block_size_index]
        } else {
            S_TM_INDEX3[block_size_index]
        } as i16;
        new_output_blk.sse = best_err;
        new_output_blk.blur_id =
            BLUR_ID_DC_LATENT_BASE + u16::from(best_log_blk.num_partitions) - 2;

        out_blocks.push(new_output_blk);
    }

    Ok(())
}

const S_TM2_INDEX_02: [u32; 14] = [
    717, 717, 1016, 1016, 1305, 1016, 1305, 1016, 1305, 1789, 1789, 2173, 2173, 2467,
];
const S_TM2_INDEX_13: [u32; 14] = [215; 14];
const S_TM3_INDEX_02: [u32; 14] = [
    721, 721, 1020, 1020, 1309, 1020, 1309, 1020, 1309, 1793, 1793, 2177, 2177, 2471,
];
const S_TM3_INDEX_13: [u32; 14] = [219; 14];

const NUM_WEIGHT_GRIDS: usize = 4 * 4;
const LL_GRID_WIDTH: [u32; NUM_WEIGHT_GRIDS] = [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2];
const LL_GRID_HEIGHT: [u32; NUM_WEIGHT_GRIDS] = [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3];
const LL_WEIGHT_GRIDS: [[u8; 6]; NUM_WEIGHT_GRIDS] = [
    [0, 0, 0, 15, 15, 15],
    [0, 15, 0, 15, 0, 15],
    [15, 15, 15, 0, 0, 0],
    [15, 0, 15, 0, 15, 0],
    // set 2
    [0, 0, 0, 4, 8, 15],
    [0, 4, 0, 8, 0, 15],
    [15, 8, 4, 0, 0, 0],
    [15, 0, 8, 0, 4, 0],
    // set 3
    [2, 4, 6, 9, 11, 13],
    [2, 9, 4, 11, 6, 13],
    [9, 11, 13, 2, 4, 6],
    [9, 2, 11, 4, 13, 6],
    // set 4
    [0, 7, 0, 7, 15, 7],
    [0, 7, 7, 15, 0, 7],
    [15, 7, 15, 7, 0, 7],
    [7, 0, 15, 7, 7, 0],
];

/// `encode_block_to_linear_latent`.
fn encode_block_to_linear_latent(
    num_patterns_to_try: u32,
    block_width: u32,
    block_height: u32,
    pixel_stats: &PixelStats,
    out_blocks: &mut Vec<EncodeBlockOutput>,
    enc_cfg: &HighLevelConfig,
    best_2subset_seed_ids: &[u64],
    best_3subset_seed_ids: &[u64],
) -> Result<()> {
    let block_size_index = block_size_index(block_width, block_height)?;
    let total_grid_samples = 6usize;
    let weight_ise_range = BISE_16_LEVELS;
    let tabs = ah::dequant_tables();
    let weight_dequant_tab = &tabs.weight_tab(weight_ise_range).ise_to_val;

    let mut dequantized_block_weights = [[0u8; MAXP]; NUM_WEIGHT_GRIDS];
    for g in 0..NUM_WEIGHT_GRIDS {
        let mut dequantized_grid_weights = [0u8; MAXP];
        for i in 0..total_grid_samples {
            dequantized_grid_weights[i] = weight_dequant_tab[LL_WEIGHT_GRIDS[g][i] as usize];
        }
        ah::upsample_weight_grid(
            block_width,
            block_height,
            LL_GRID_WIDTH[g],
            LL_GRID_HEIGHT[g],
            &dequantized_grid_weights,
            &mut dequantized_block_weights[g],
        );
    }

    for subsets in 2..=3u32 {
        for grid_index in 0..NUM_WEIGHT_GRIDS {
            let mut log_blk = LogAstcBlock {
                grid_width: LL_GRID_WIDTH[grid_index] as u8,
                grid_height: LL_GRID_HEIGHT[grid_index] as u8,
                num_partitions: subsets as u8,
                endpoint_ise_range: if subsets == 3 {
                    BISE_16_LEVELS
                } else {
                    BISE_64_LEVELS
                } as u8,
                weight_ise_range: weight_ise_range as u8,
                ..Default::default()
            };
            log_blk.color_endpoint_modes[0] = 8;
            log_blk.color_endpoint_modes[1] = 8;
            if subsets == 3 {
                log_blk.color_endpoint_modes[2] = 8;
            }
            log_blk.weights[..total_grid_samples]
                .copy_from_slice(&LL_WEIGHT_GRIDS[grid_index][..total_grid_samples]);

            let mut best_err = u64::MAX;
            let mut best_log_blk = LogAstcBlock::default();

            for pat_index in 0..num_patterns_to_try as usize {
                let ids = if subsets == 2 {
                    best_2subset_seed_ids
                } else {
                    best_3subset_seed_ids
                };
                let part_seed = (*ids.get(pat_index).ok_or(Error::InvalidData)? & 1023) as u32;

                log_blk.partition_id = part_seed as u16;

                let mut subset_pixels = [[[0u8; 4]; MAXP]; 3];
                let mut subset_weights = [[0u8; MAXP]; 3];
                let mut num_subset_pixels = [0usize; 3];

                for y in 0..block_height {
                    for x in 0..block_width {
                        let s = get_precomputed_texel_partition(
                            block_width,
                            block_height,
                            part_seed,
                            x,
                            y,
                            u32::from(log_blk.num_partitions),
                        ) as usize;
                        let n = num_subset_pixels[s];
                        let k = (y * block_width + x) as usize;
                        subset_pixels[s][n] = pixel_stats.pixels[k];
                        subset_weights[s][n] = dequantized_block_weights[grid_index][k];
                        num_subset_pixels[s] = n + 1;
                    }
                }

                let mut total_err = 0u64;
                for s in 0..subsets as usize {
                    let mut stats = PixelStats::default();
                    stats.init(num_subset_pixels[s] as u32, &subset_pixels[s]);

                    let mut cem_enc_params = CemEncodeParams {
                        decode_mode_srgb: enc_cfg.cem_enc_params.decode_mode_srgb,
                        max_ls_passes: 1,
                        ..Default::default()
                    };
                    cem_enc_params.forced_weight_vals0 = Some(&subset_weights[s]);

                    let mut temp_weights = [0u8; MAXP];

                    let subset_err = cem_encode_pixels(
                        8,
                        -1,
                        &stats,
                        &cem_enc_params,
                        u32::from(log_blk.endpoint_ise_range),
                        BISE_64_LEVELS,
                        &mut log_blk.endpoints[s * 6..],
                        &mut temp_weights,
                        None,
                        u64::MAX,
                        true,
                        None,
                    );

                    if subset_err == u64::MAX {
                        total_err = u64::MAX;
                        break;
                    }

                    total_err += subset_err;

                    if total_err > best_err {
                        break;
                    }
                }

                if total_err < best_err {
                    best_err = total_err;
                    best_log_blk = log_blk;
                }
            }

            if best_err != u64::MAX {
                let tm_index = if subsets == 3 {
                    if (grid_index & 3) == 0 || (grid_index & 3) == 2 {
                        S_TM3_INDEX_02[block_size_index]
                    } else {
                        S_TM3_INDEX_13[block_size_index]
                    }
                } else if (grid_index & 3) == 0 || (grid_index & 3) == 2 {
                    S_TM2_INDEX_02[block_size_index]
                } else {
                    S_TM2_INDEX_13[block_size_index]
                };

                out_blocks.push(EncodeBlockOutput {
                    log_blk: best_log_blk,
                    trial_mode_index: tm_index as i16,
                    sse: best_err,
                    blur_id: BLUR_ID_AC_LATENT_BASE
                        + grid_index as u16
                        + (subsets as u16 - 2) * NUM_WEIGHT_GRIDS as u16,
                    ..Default::default()
                });
            }
        }
    }

    Ok(())
}

/// `enforce_max_candidate_limit`. The comparator (WSSE, then index) is a strict total order,
/// so `std::partial_sort`'s result equals the first `max_candidates` of a full sort.
fn enforce_max_candidate_limit(block_info_out: &mut BlockInfo, max_candidates: u32) {
    let max_candidates = max_candidates.max(1) as usize;
    let n = block_info_out.out_blocks.len();
    if n <= max_candidates {
        return;
    }

    let blocks = &block_info_out.out_blocks;
    let mut sorted_indices: Vec<u32> = (0..n as u32).collect();
    sorted_indices.sort_by(|&a, &b| {
        blocks[a as usize]
            .sse
            .cmp(&blocks[b as usize].sse)
            .then(a.cmp(&b))
    });

    let packed = block_info_out.packed_out_block_index;
    let new_packed_out_block_index = sorted_indices[..max_candidates]
        .iter()
        .rposition(|&i| i == packed);

    let new_blocks: Vec<EncodeBlockOutput> = match new_packed_out_block_index {
        None => {
            // one more than requested, not the end of the world
            let mut v = Vec::with_capacity(max_candidates + 1);
            v.push(blocks[packed as usize].clone());
            v.extend(
                sorted_indices[..max_candidates]
                    .iter()
                    .map(|&i| blocks[i as usize].clone()),
            );
            v
        }
        Some(_) => sorted_indices[..max_candidates]
            .iter()
            .map(|&i| blocks[i as usize].clone())
            .collect(),
    };

    block_info_out.out_blocks = new_blocks;
    block_info_out.packed_out_block_index = new_packed_out_block_index.unwrap_or(0) as u32;
}

/// `ldr_astc_block_encode_image_output::block_info_superpass1`.
#[derive(Clone, Debug)]
struct BlockInfoSuperpass1 {
    config_reuse_neighbor_out_block_indices: [i32; MAX_CONFIG_REUSE_NEIGHBORS as usize],
    config_reuse_new_neighbor_out_block_flags: [bool; MAX_CONFIG_REUSE_NEIGHBORS as usize],
    new_out_config_reuse_blocks: Vec<EncodeBlockOutput>,
    new_out_config_endpoint_reuse_blocks: Vec<EncodeBlockOutput>,
}

impl Default for BlockInfoSuperpass1 {
    fn default() -> Self {
        Self {
            config_reuse_neighbor_out_block_indices: [INVALID_INDEX; 3],
            config_reuse_new_neighbor_out_block_flags: [false; 3],
            new_out_config_reuse_blocks: Vec::new(),
            new_out_config_endpoint_reuse_blocks: Vec::new(),
        }
    }
}

/// Shared read-only state of one `ldr_astc_block_encode_image` call.
struct Ctx<'a> {
    img: &'a Image,
    cfg: &'a HighLevelConfig,
    block_width: u32,
    block_height: u32,
    total_block_pixels: u32,
    max_candidate_limit: u32,
    sobel_xy: Image,
    blurred: Vec<Image>,
    part_data_p2: PartitionsData,
    part_data_p3: PartitionsData,
    modes: &'static ModeTable,
    dct: Dct2f,
    grid_coder: GridWeightDct,
}

/// Result of a superpass 0/1 block job.
struct BlockJobOut {
    info: BlockInfo,
    phys: [u8; 16],
    pixel_stats: Option<Box<PixelStats>>,
}

/// `ldr_astc_lowlevel_block_encoder_pool`: encoders are fully reset by `full_encode`, so
/// which one a job gets never affects the output.
struct EncoderPool(Mutex<Vec<Box<LdrAstcLowlevelBlockEncoder>>>);

impl EncoderPool {
    fn acquire(&self) -> Box<LdrAstcLowlevelBlockEncoder> {
        self.0
            .lock()
            .ok()
            .and_then(|mut v| v.pop())
            .unwrap_or_else(|| Box::new(LdrAstcLowlevelBlockEncoder::new()))
    }

    fn release(&self, e: Box<LdrAstcLowlevelBlockEncoder>) {
        if let Ok(mut v) = self.0.lock() {
            v.push(e);
        }
    }
}

/// Runs `code_block_weights` + `decode_block_weights` on every plane of `log_blk` (the
/// superpass 2 DCT round trip), storing the symbols in `out`.
fn dct_round_trip(
    ctx: &Ctx,
    log_blk: &mut LogAstcBlock,
    out: &mut EncodeBlockOutput,
) -> Result<()> {
    let num_planes = if log_blk.dual_plane { 2 } else { 1 };
    for plane_index in 0..num_planes {
        let syms = &mut out.packed_dct_plane_data[plane_index];
        if !code_block_weights(
            &ctx.grid_coder,
            ctx.cfg.base_q,
            plane_index as u32,
            log_blk,
            syms,
        ) {
            return Err(Error::InvalidData);
        }

        // ensure existing weights get blown away
        let n = log_blk.grid_width as usize * log_blk.grid_height as usize;
        for i in 0..n {
            log_blk.weights[i * num_planes + plane_index] = 0;
        }

        if !ctx
            .grid_coder
            .decode_block_weights(ctx.cfg.base_q, plane_index as u32, log_blk, syms)
        {
            return Err(Error::InvalidData);
        }
    }
    Ok(())
}

/// Superpass 0/1 block encode (the `superpass_index != 2` branch of the C++ job lambda).
fn encode_block_pass01(
    ctx: &Ctx,
    bx: u32,
    by: u32,
    superpass_index: u32,
    stored_pixel_stats: Option<&PixelStats>,
    pool: &EncoderPool,
) -> Result<BlockJobOut> {
    let cfg = ctx.cfg;
    let block_width = ctx.block_width;
    let block_height = ctx.block_height;
    let total_block_pixels = ctx.total_block_pixels;
    let tbp = total_block_pixels as usize;
    let srgb = cfg.cem_enc_params.decode_mode_srgb;

    let mut block_pixels = [[0u8; 4]; MAXP];
    ctx.img.extract_block_clamped(
        &mut block_pixels,
        bx * block_width,
        by * block_height,
        block_width,
        block_height,
    );

    let mut new_pixel_stats: Option<Box<PixelStats>> = None;
    let pixel_stats: &PixelStats = match stored_pixel_stats {
        Some(ps) if superpass_index != 0 => ps,
        _ => {
            let mut ps = Box::<PixelStats>::default();
            ps.init(total_block_pixels, &block_pixels[..tbp]);
            new_pixel_stats = Some(ps);
            new_pixel_stats.as_deref().ok_or(Error::InvalidData)?
        }
    };

    let mut info = BlockInfo::default();
    let mut phys = [0u8; 16];

    let is_purely_solid_block = pixel_stats.min == pixel_stats.max;

    // early out on totally solid blocks
    if is_purely_solid_block {
        let mut out = EncodeBlockOutput::default();
        let log_blk = &mut out.log_blk;
        log_blk.solid_color_flag_ldr = true;
        for c in 0..4 {
            let v = u16::from(pixel_stats.min[c]);
            log_blk.set_solid_color(c, v | (v << 8));
        }
        out.sse = eval_error(
            block_width,
            block_height,
            &out.log_blk,
            pixel_stats,
            &cfg.cem_enc_params,
        );

        info.low_freq_block_flag = true;
        info.packed_out_block_index = 0;

        if !ah::pack_astc_block(&mut phys, &out.log_blk) {
            return Err(Error::InvalidData);
        }
        info.out_blocks.push(out);

        return Ok(BlockJobOut {
            info,
            phys,
            pixel_stats: new_pixel_stats,
        });
    }

    let out_blocks = &mut info.out_blocks;

    let mut cfg_man = EncoderConfigManager::new();
    cfg_man.init(
        bx,
        by,
        block_width,
        block_height,
        total_block_pixels,
        pixel_stats,
        &ctx.dct,
        cfg,
        &ctx.sobel_xy,
    );

    let mut encoder = pool.acquire();
    let mut enc_block_stats = EncodeBlockStats::default();

    {
        let enc_blk_params = cfg_man.select(superpass_index, bx, by, ctx, pixel_stats);

        // solid color - must be first
        {
            let mut out = EncodeBlockOutput::default();
            out.log_blk.solid_color_flag_ldr = true;
            for c in 0..4 {
                let v = ((pixel_stats.mean_f[c] * 255.0).round() as i32).clamp(0, 255) as u16;
                out.log_blk.set_solid_color(c, v | (v << 8));
            }
            out.sse = eval_error(
                block_width,
                block_height,
                &out.log_blk,
                pixel_stats,
                &cfg.cem_enc_params,
            );
            out_blocks.push(out);
        }

        if !encoder.full_encode(
            &enc_blk_params,
            pixel_stats,
            out_blocks,
            0,
            &mut enc_block_stats,
        ) {
            pool.release(encoder);
            return Err(Error::InvalidData);
        }
    }

    // --------------------- BLOCK BLURRING
    const BLUR_STD_DEV_THRESH: f32 = 10.0 / 255.0;
    const BLUR_SOBEL_ENERGY_THRESH: f32 = 10000.0;
    let use_blurs = (cfg.blurring_enabled_p1
        && (!SELECTIVE_BLURRING
            || (cfg_man.max_std_dev > BLUR_STD_DEV_THRESH
                && cfg_man.sobel_energy > BLUR_SOBEL_ENERGY_THRESH)))
        || (cfg.blurring_enabled_p2 && superpass_index == 1);
    if use_blurs {
        for i in 0..TOTAL_BLURRED_IMAGES {
            let blur_id = u32::from(BLUR_ID_GAUSSIAN_ALTERNATE) + i as u32;
            let blurred_img = ctx.blurred.get(i).ok_or(Error::InvalidData)?;

            let mut block_pixels_blurred = [[0u8; 4]; MAXP];
            blurred_img.extract_block_clamped(
                &mut block_pixels_blurred,
                bx * block_width,
                by * block_height,
                block_width,
                block_height,
            );

            let mut pixel_stats_blurred = Box::<PixelStats>::default();
            pixel_stats_blurred.init(total_block_pixels, &block_pixels_blurred[..tbp]);

            cfg_man.init(
                bx,
                by,
                block_width,
                block_height,
                total_block_pixels,
                &pixel_stats_blurred,
                &ctx.dct,
                cfg,
                &ctx.sobel_xy,
            );

            let enc_blk_params_blurred =
                cfg_man.select(superpass_index, bx, by, ctx, &pixel_stats_blurred);

            if !encoder.full_encode(
                &enc_blk_params_blurred,
                &pixel_stats_blurred,
                out_blocks,
                blur_id,
                &mut enc_block_stats,
            ) {
                pool.release(encoder);
                return Err(Error::InvalidData);
            }
        }
    }

    // --------------------- EXPERIMENTAL - DC/LINEAR LATENT CODING
    if cfg.try_simplified_latent_configs
        && !pixel_stats.has_alpha
        && cfg_man.max_std_dev > (2.0 / 255.0)
    {
        let mut best_2subset_seed_ids = Vec::new();
        let mut best_3subset_seed_ids = Vec::new();

        encode_block_to_dc_latent(
            block_width,
            block_height,
            pixel_stats,
            out_blocks,
            cfg,
            &ctx.part_data_p2,
            &ctx.part_data_p3,
            &mut best_2subset_seed_ids,
            &mut best_3subset_seed_ids,
        )?;

        encode_block_to_linear_latent(
            8,
            block_width,
            block_height,
            pixel_stats,
            out_blocks,
            cfg,
            &best_2subset_seed_ids,
            &best_3subset_seed_ids,
        )?;
    }

    // --------------------- WEIGHT GRID DCT CODING
    if cfg.use_dct {
        for out_block in out_blocks.iter_mut() {
            if !apply_weight_grid_dct(
                block_width,
                block_height,
                &ctx.grid_coder,
                out_block,
                pixel_stats,
                true,
                cfg,
                &ctx.part_data_p2,
                &ctx.part_data_p3,
                true,
            ) {
                pool.release(encoder);
                return Err(Error::InvalidData);
            }
        }
    }

    // Find best output block
    let mut best_out_blocks_err = u64::MAX;
    let mut best_out_blocks_index = 0usize;
    let mut best_out_blocks_log_astc_blk = LogAstcBlock::default();

    for (out_block_iter, ob) in out_blocks.iter_mut().enumerate() {
        let mut dec_pixels = [[0u8; 4]; MAXP];
        if !decode_xu(
            &ob.log_blk,
            &mut dec_pixels,
            block_width,
            block_height,
            srgb,
        ) {
            pool.release(encoder);
            return Err(Error::InvalidData);
        }

        let mut total_err = 0u64;
        for i in 0..tbp {
            total_err += u64::from(weighted_color_error(
                &block_pixels[i],
                &dec_pixels[i],
                &cfg.cem_enc_params,
            ));
        }

        // if not blurred
        if ob.blur_id == 0 && ob.sse != total_err {
            // "output block SSE invalid"
            pool.release(encoder);
            return Err(Error::InvalidData);
        }

        // Replace m_sse with the actual WSSE vs. the original source block (in case it was
        // blurred)
        ob.sse = total_err;

        if total_err < best_out_blocks_err {
            best_out_blocks_err = total_err;
            best_out_blocks_log_astc_blk = ob.log_blk;
            best_out_blocks_index = out_block_iter;
        }
    }

    info.low_freq_block_flag = cfg_man.low_freq_block_flag;
    info.super_strong_edges = encoder.super_strong_edges;
    info.very_strong_edges = encoder.very_strong_edges;
    info.strong_edges = encoder.strong_edges;
    pool.release(encoder);

    info.packed_out_block_index = best_out_blocks_index as u32;

    enforce_max_candidate_limit(&mut info, ctx.max_candidate_limit);

    // Create packed ASTC block
    if !ah::pack_astc_block(&mut phys, &best_out_blocks_log_astc_blk) {
        return Err(Error::InvalidData);
    }

    Ok(BlockJobOut {
        info,
        phys,
        pixel_stats: new_pixel_stats,
    })
}

/// Superpass 2 block job: re-encodes the block with its left / upper / upper-left neighbors'
/// chosen configurations (and endpoints).
fn encode_block_pass2(
    ctx: &Ctx,
    bx: u32,
    by: u32,
    num_blocks_x: u32,
    infos: &[BlockInfo],
    pixel_stats: &PixelStats,
) -> Result<BlockInfoSuperpass1> {
    let cfg = ctx.cfg;
    let block_width = ctx.block_width;
    let block_height = ctx.block_height;
    let idx = |x: u32, y: u32| (x + y * num_blocks_x) as usize;

    let out_block_info = &infos[idx(bx, by)];
    let mut sp1 = BlockInfoSuperpass1::default();
    let flags = TrialFlags {
        gradient_descent_flag: cfg.gradient_descent_flag,
        polish_weights_flag: cfg.polish_weights_flag,
        qcd_enabled_flag: cfg.qcd_enabled_flag,
        use_blue_contraction: cfg.use_blue_contraction,
    };

    let is_purely_solid_block = pixel_stats.min == pixel_stats.max;

    // if void extent, just skip
    if is_purely_solid_block {
        return Ok(sp1);
    }

    for neighbor_index in 0..MAX_CONFIG_REUSE_NEIGHBORS as usize {
        let neighbor_info = match neighbor_index {
            // Left
            0 if bx != 0 => &infos[idx(bx - 1, by)],
            // Up
            1 if by != 0 => &infos[idx(bx, by - 1)],
            // Diagonal
            2 if bx != 0 && by != 0 => &infos[idx(bx - 1, by - 1)],
            _ => continue,
        };

        let neighbor_output = neighbor_info
            .out_blocks
            .get(neighbor_info.packed_out_block_index as usize)
            .ok_or(Error::InvalidData)?;

        // Best neighbor was solid, skip it (TODO: reusing it is possible)
        if neighbor_output.log_blk.solid_color_flag_ldr {
            continue;
        }

        let neighbor_tm_index = neighbor_output.trial_mode_index;
        let neighbor_log_blk = &neighbor_output.log_blk;

        let neighbor_actual_cem = neighbor_log_blk.color_endpoint_modes[0];
        let neighbor_partition_id = neighbor_log_blk.partition_id;

        // See if we've already encoded this full config
        let already_existing_out_block_index = out_block_info.out_blocks.iter().position(|ob| {
            ob.trial_mode_index == neighbor_tm_index
                && ob.log_blk.color_endpoint_modes[0] == neighbor_actual_cem
                && ob.log_blk.partition_id == neighbor_partition_id
        });

        let pat_for = |num_partitions: u8| -> Result<_> {
            if num_partitions > 1 {
                let part_data = if num_partitions == 2 {
                    &ctx.part_data_p2
                } else {
                    &ctx.part_data_p3
                };
                let part_unique_index =
                    part_data.part_seed_to_unique_index[neighbor_log_blk.partition_id as usize];
                let pat = part_data
                    .partition_pats
                    .get(part_unique_index as usize)
                    .ok_or(Error::InvalidData)?;
                Ok(Some(pat))
            } else {
                Ok(None)
            }
        };

        if let Some(i) = already_existing_out_block_index {
            // We already have an output block using this neighbor trial mode, skip
            sp1.config_reuse_neighbor_out_block_indices[neighbor_index] = i as i32;
            sp1.config_reuse_new_neighbor_out_block_flags[neighbor_index] = false;
        } else {
            // Re-encode using the neighbor's full config (tm, base+ofs, partition ID)
            let mut new_log_block = LogAstcBlock::default();

            let status = if neighbor_log_blk.num_partitions > 1 {
                let pat = pat_for(neighbor_log_blk.num_partitions)?.ok_or(Error::InvalidData)?;
                let refine_only_flag = false;
                encode_trial_subsets(
                    block_width,
                    block_height,
                    pixel_stats,
                    u32::from(neighbor_log_blk.color_endpoint_modes[0]),
                    u32::from(neighbor_log_blk.num_partitions),
                    u32::from(neighbor_log_blk.partition_id),
                    pat,
                    u32::from(neighbor_log_blk.endpoint_ise_range),
                    u32::from(neighbor_log_blk.weight_ise_range),
                    u32::from(neighbor_log_blk.grid_width),
                    u32::from(neighbor_log_blk.grid_height),
                    cfg.encode_trial_subsets_early_out_thresh,
                    &mut new_log_block,
                    &cfg.cem_enc_params,
                    refine_only_flag,
                    flags,
                    None,
                )
            } else {
                encode_trial(
                    block_width,
                    block_height,
                    pixel_stats,
                    u32::from(neighbor_log_blk.color_endpoint_modes[0]),
                    neighbor_log_blk.dual_plane,
                    if neighbor_log_blk.dual_plane {
                        i32::from(neighbor_log_blk.color_component_selector)
                    } else {
                        -1
                    },
                    u32::from(neighbor_log_blk.endpoint_ise_range),
                    u32::from(neighbor_log_blk.weight_ise_range),
                    u32::from(neighbor_log_blk.grid_width),
                    u32::from(neighbor_log_blk.grid_height),
                    cfg.encode_trial_early_out_thresh,
                    &mut new_log_block,
                    &cfg.cem_enc_params,
                    flags,
                    None,
                )
            };

            if !status {
                // "encode_trial/encode_trial_subsets failed in superpass 1!"
                return Err(Error::InvalidData);
            }

            sp1.config_reuse_neighbor_out_block_indices[neighbor_index] =
                sp1.new_out_config_reuse_blocks.len() as i32;
            sp1.config_reuse_new_neighbor_out_block_flags[neighbor_index] = true;

            let mut new_output_blk = EncodeBlockOutput::default();

            if cfg.use_dct {
                dct_round_trip(ctx, &mut new_log_block, &mut new_output_blk)?;
            }

            new_output_blk.trial_mode_index = neighbor_tm_index;
            new_output_blk.log_blk = new_log_block;
            new_output_blk.sse = eval_error(
                block_width,
                block_height,
                &new_log_block,
                pixel_stats,
                &cfg.cem_enc_params,
            );

            sp1.new_out_config_reuse_blocks.push(new_output_blk);
        }

        {
            // Re-encode using the neighbor's full config (tm, base+ofs, partition ID) AND its
            // endpoints
            let mut new_log_block = *neighbor_log_blk;

            // Start with fresh 0 weights, then polish them.
            new_log_block.weights = [0; ah::MAX_GRID_WEIGHTS as usize];

            let mut improved_flag = false;

            let pat = pat_for(neighbor_log_blk.num_partitions)?;

            if !polish_block_weights(
                block_width,
                block_height,
                pixel_stats,
                &mut new_log_block,
                &cfg.cem_enc_params,
                pat,
                &mut improved_flag,
                cfg.gradient_descent_flag,
                cfg.polish_weights_flag,
                cfg.qcd_enabled_flag,
            ) {
                // "polish_block_weights failed in superpass 1!"
                return Err(Error::InvalidData);
            }

            let mut new_output_blk = EncodeBlockOutput::default();

            if cfg.use_dct {
                dct_round_trip(ctx, &mut new_log_block, &mut new_output_blk)?;
            }

            new_output_blk.trial_mode_index = neighbor_tm_index;
            new_output_blk.log_blk = new_log_block;
            new_output_blk.sse = eval_error(
                block_width,
                block_height,
                &new_log_block,
                pixel_stats,
                &cfg.cem_enc_params,
            );

            sp1.new_out_config_endpoint_reuse_blocks
                .push(new_output_blk);
        }
    }

    Ok(sp1)
}

/// `ldr_astc_block_encode_image`. `threads` = basisu job pool total threads (the output does
/// not depend on it).
pub(crate) fn ldr_astc_block_encode_image(
    orig_img: &Image,
    enc_cfg: &HighLevelConfig,
    max_candidate_limit: u32,
    threads: u32,
) -> Result<EncodeImageOutput> {
    let block_width = enc_cfg.block_width;
    let block_height = enc_cfg.block_height;
    let width = orig_img.get_width();
    let height = orig_img.get_height();
    if width == 0 || height == 0 {
        return Err(Error::InvalidArgument);
    }
    let size_index = block_size_index(block_width, block_height)?;
    let total_block_pixels = block_width * block_height;
    let num_blocks_x = orig_img.get_block_width(block_width);
    let num_blocks_y = orig_img.get_block_height(block_height);
    let total_blocks = num_blocks_x * num_blocks_y;

    let part_data_p2 = PartitionsData::new(2, block_width, block_height, true);
    let part_data_p3 = PartitionsData::new(3, block_width, block_height, true);

    let mut blurred = Vec::new();
    if enc_cfg.blurring_enabled_p1 || enc_cfg.blurring_enabled_p2 {
        for i in 0..TOTAL_BLURRED_IMAGES {
            let f = i as f32 / (TOTAL_BLURRED_IMAGES - 1) as f32;
            let w = 1.1f32 + (2.0f32 - 1.1f32) * f;
            let mut b = Image::with_size(width, height);
            if !image_resample(orig_img, &mut b, false, "gaussian", w, false, 0, 4) {
                return Err(Error::InvalidData);
            }
            blurred.push(b);
        }
    }

    let mut sobel_x = Image::new();
    let mut sobel_y = Image::new();
    compute_sobel(orig_img, &mut sobel_x, &SOBEL_X);
    compute_sobel(orig_img, &mut sobel_y, &SOBEL_Y);

    let mut sobel_xy = Image::with_size(width, height);
    for y in 0..height {
        for x in 0..width {
            let sx = sobel_x.get(x, y);
            let sy = sobel_y.get(x, y);
            let f = |c: usize| {
                ((sx[c] as i32 - 128).abs() + (sy[c] as i32 - 128).abs()).clamp(0, 255) as u8
            };
            *sobel_xy.get_mut(x, y) = [f(0), f(1), f(2), f(3)];
        }
    }

    let modes = mode_table(size_index).ok_or(Error::InvalidArgument)?;

    let dct = Dct2f::new(block_height, block_width).ok_or(Error::InvalidArgument)?;
    let grid_coder = GridWeightDct::new(block_width, block_height);

    let ctx = Ctx {
        img: orig_img,
        cfg: enc_cfg,
        block_width,
        block_height,
        total_block_pixels,
        max_candidate_limit,
        sobel_xy,
        blurred,
        part_data_p2,
        part_data_p3,
        modes,
        dct,
        grid_coder,
    };

    let mut out = EncodeImageOutput::new(num_blocks_x, num_blocks_y);
    let mut pixel_stats: Vec<Box<PixelStats>> = Vec::with_capacity(total_blocks as usize);
    let mut recompress_flags = vec![false; total_blocks as usize];

    let pool = EncoderPool(Mutex::new(Vec::new()));

    let total_superpasses = if enc_cfg.third_superpass_try_neighbors {
        3
    } else if enc_cfg.second_superpass_refinement {
        2
    } else {
        1
    };

    let mut total_blocks_to_recompress = 0u32;
    let mut superpass2: Vec<BlockInfoSuperpass1> = Vec::new();

    for superpass_index in 0..total_superpasses {
        if superpass_index == 1
            && (!enc_cfg.second_superpass_refinement || total_blocks_to_recompress == 0)
        {
            continue;
        }

        match superpass_index {
            0 => {
                let res = map_jobs(threads, total_blocks as usize, |i| {
                    let (bx, by) = (i as u32 % num_blocks_x, i as u32 / num_blocks_x);
                    encode_block_pass01(&ctx, bx, by, 0, None, &pool)
                })
                .ok_or(Error::InvalidData)?;
                for (i, r) in res.into_iter().enumerate() {
                    let r = r?;
                    out.block_info[i] = r.info;
                    out.packed_phys_blocks[i] = r.phys;
                    pixel_stats.push(r.pixel_stats.ok_or(Error::InvalidData)?);
                }
            }
            1 => {
                let todo: Vec<usize> = (0..total_blocks as usize)
                    .filter(|&i| recompress_flags[i])
                    .collect();
                let res = map_jobs(threads, todo.len(), |j| {
                    let i = todo[j];
                    let (bx, by) = (i as u32 % num_blocks_x, i as u32 / num_blocks_x);
                    encode_block_pass01(&ctx, bx, by, 1, Some(&pixel_stats[i]), &pool)
                })
                .ok_or(Error::InvalidData)?;
                for (j, r) in res.into_iter().enumerate() {
                    let r = r?;
                    let i = todo[j];
                    out.block_info[i] = r.info;
                    out.packed_phys_blocks[i] = r.phys;
                }
            }
            _ => {
                let infos = &out.block_info;
                let res = map_jobs(threads, total_blocks as usize, |i| {
                    let (bx, by) = (i as u32 % num_blocks_x, i as u32 / num_blocks_x);
                    encode_block_pass2(&ctx, bx, by, num_blocks_x, infos, &pixel_stats[i])
                })
                .ok_or(Error::InvalidData)?;
                superpass2 = res.into_iter().collect::<Result<Vec<_>>>()?;
            }
        }

        if superpass_index == 0
            && enc_cfg.second_superpass_refinement
            && enc_cfg.second_superpass_fract_to_recompress > 0.0
        {
            let mut block_wsse_indices = vec![0u32; total_blocks as usize];
            let block_wsses: Vec<f32> = out
                .block_info
                .iter()
                .map(|bi| {
                    bi.out_blocks
                        .get(bi.packed_out_block_index as usize)
                        .map_or(0.0, |b| b.sse as f32)
                })
                .collect();

            indirect_sort(&mut block_wsse_indices, &block_wsses);

            if block_wsses[block_wsse_indices[total_blocks as usize - 1] as usize] > 0.0 {
                total_blocks_to_recompress = ((total_blocks as f32
                    * enc_cfg.second_superpass_fract_to_recompress)
                    .round() as u32)
                    .clamp(0, total_blocks);

                for i in 0..total_blocks_to_recompress as usize {
                    let block_index = block_wsse_indices[total_blocks as usize - 1 - i];
                    recompress_flags[block_index as usize] = true;
                }
            }
        }
    }

    if enc_cfg.third_superpass_try_neighbors {
        // Merge pass 2's output into pass 0's/1's output, which can be done safely now.
        for (i, sp1) in superpass2.iter().enumerate() {
            let out_block_info = &mut out.block_info[i];

            for neighbor_index in 0..MAX_CONFIG_REUSE_NEIGHBORS as usize {
                let new_neighbor_index =
                    sp1.config_reuse_neighbor_out_block_indices[neighbor_index];

                // Can't reuse neighbor's best output block
                if new_neighbor_index == INVALID_INDEX {
                    continue;
                }

                // Reuses an existing, already encoded output block which matches the neighbor
                if !sp1.config_reuse_new_neighbor_out_block_flags[neighbor_index] {
                    continue;
                }

                let new_output_blk = sp1
                    .new_out_config_reuse_blocks
                    .get(new_neighbor_index as usize)
                    .ok_or(Error::InvalidData)?;
                merge_candidate(
                    out_block_info,
                    &mut out.packed_phys_blocks[i],
                    new_output_blk,
                )?;
            }

            for new_output_blk in &sp1.new_out_config_endpoint_reuse_blocks {
                merge_candidate(
                    out_block_info,
                    &mut out.packed_phys_blocks[i],
                    new_output_blk,
                )?;
            }

            enforce_max_candidate_limit(out_block_info, max_candidate_limit);
        }
    }

    // Consistency/sanity cross checking (`ASTC_LDR_CONSISTENCY_CHECKING`)
    for (i, bi) in out.block_info.iter().enumerate() {
        let best = bi
            .out_blocks
            .get(bi.packed_out_block_index as usize)
            .ok_or(Error::InvalidData)?;
        let mut best_sse = u64::MAX;
        let mut best_out_block_index = 0usize;
        for (j, ob) in bi.out_blocks.iter().enumerate() {
            if ob.sse < best_sse {
                best_sse = ob.sse;
                best_out_block_index = j;
            }
        }
        if best_out_block_index != bi.packed_out_block_index as usize {
            return Err(Error::InvalidData);
        }
        if best.sse
            != eval_error(
                block_width,
                block_height,
                &best.log_blk,
                &pixel_stats[i],
                &enc_cfg.cem_enc_params,
            )
        {
            return Err(Error::InvalidData);
        }
        let mut packed_block = [0u8; 16];
        if !ah::pack_astc_block(&mut packed_block, &best.log_blk)
            || packed_block != out.packed_phys_blocks[i]
        {
            return Err(Error::InvalidData);
        }
    }

    Ok(out)
}

/// Appends a superpass 2 candidate; makes it the chosen block if it beats the current one
/// (`BU_CHECK_NEIGHBOR_BEST`).
fn merge_candidate(
    out_block_info: &mut BlockInfo,
    phys: &mut [u8; 16],
    new_output_blk: &EncodeBlockOutput,
) -> Result<()> {
    let new_out_block_index = out_block_info.out_blocks.len();
    out_block_info.out_blocks.push(new_output_blk.clone());

    let cur = out_block_info
        .out_blocks
        .get(out_block_info.packed_out_block_index as usize)
        .ok_or(Error::InvalidData)?;
    // See if the solution has improved
    if new_output_blk.sse < cur.sse {
        out_block_info.packed_out_block_index = new_out_block_index as u32;
        if !ah::pack_astc_block(phys, &new_output_blk.log_blk) {
            return Err(Error::InvalidData);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! `#[ignore]`d comparison against C++ dumps (`$XU_DUMP_DIR/cases.txt`, see
    //! `docs/spec/xuastc_encode.md`) and a fast regression test against hashes of the C++
    //! output (`testdata/basisu_enc/basisu_astc_ldr_image.txt`).
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

    use super::super::dump::{self, log_blk_eq};
    use super::super::effort::configure_encoder_effort_level;
    use super::*;
    use std::collections::BTreeMap;

    /// One C++ dump case: `name file bw bh effort srgb dct q scd blur merge`.
    struct Case {
        name: String,
        file: String,
        bw: u32,
        bh: u32,
        effort: i32,
        srgb: bool,
        dct: bool,
        q: f32,
        scd: bool,
        blur: bool,
        merge: bool,
    }

    fn parse_case(l: &str) -> Case {
        let f: Vec<&str> = l.split_whitespace().collect();
        Case {
            name: f[0].into(),
            file: f[1].into(),
            bw: f[2].parse().unwrap(),
            bh: f[3].parse().unwrap(),
            effort: f[4].parse().unwrap(),
            srgb: f[5] == "1",
            dct: f[6] == "1",
            q: f[7].parse().unwrap(),
            scd: f[8] == "1",
            blur: f[9] == "1",
            merge: f[10] == "1",
        }
    }

    /// The `HighLevelConfig` and `max_candidate_limit` `astc_ldr::compress_image` derives from
    /// what `basis_compressor` sets (no sharpening, default force-disable flags).
    fn config_for(c: &Case, total_blocks: u32) -> (HighLevelConfig, u32) {
        let mut cfg = HighLevelConfig {
            block_width: c.bw,
            block_height: c.bh,
            use_dct: c.dct,
            blurring_enabled_p1: c.blur,
            blurring_enabled_p2: c.blur,
            ..Default::default()
        };
        let int_q = ((c.q * 2.0).round() as i32).clamp(0, 200);
        cfg.base_q = int_q as f32 / 2.0;
        cfg.cem_enc_params.comp_weights = if c.srgb { [9, 11, 1, 11] } else { [1, 1, 1, 1] };
        configure_encoder_effort_level(c.effort, &mut cfg);
        cfg.cem_enc_params.decode_mode_srgb = c.srgb;
        let mut limit = 16;
        if c.scd {
            limit = if total_blocks >= 512 * 512 {
                if c.merge { 8 } else { 16 }
            } else if total_blocks >= 256 * 256 {
                if c.merge { 16 } else { 32 }
            } else if c.merge {
                32
            } else {
                64
            };
        }
        (cfg, limit)
    }

    /// Canonical byte serialization of the block data (the `.xud` block section layout).
    fn serialize(o: &EncodeImageOutput) -> Vec<u8> {
        let mut v = Vec::new();
        let w32 = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
        w32(&mut v, o.blocks_x);
        w32(&mut v, o.blocks_y);
        for bi in &o.block_info {
            w32(&mut v, bi.packed_out_block_index);
            v.extend_from_slice(&[
                bi.low_freq_block_flag as u8,
                bi.super_strong_edges as u8,
                bi.very_strong_edges as u8,
                bi.strong_edges as u8,
            ]);
            w32(&mut v, bi.out_blocks.len() as u32);
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
                ]);
                v.extend_from_slice(&b.partition_id.to_le_bytes());
                v.extend_from_slice(&b.color_endpoint_modes);
                v.extend_from_slice(&b.weights);
                v.extend_from_slice(&b.endpoints);
                for s in &ob.packed_dct_plane_data {
                    for x in [
                        s.dc_sym,
                        s.num_dc_levels,
                        s.max_coeff_mag,
                        s.max_zigzag_index,
                    ] {
                        w32(&mut v, x);
                    }
                    w32(&mut v, s.coeffs.len() as u32);
                    for c in &s.coeffs {
                        v.extend_from_slice(&c.num_zeros.to_le_bytes());
                        v.extend_from_slice(&c.coeff.to_le_bytes());
                    }
                }
            }
        }
        v
    }

    fn fnv64(b: &[u8]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Compares two outputs; returns (matching blocks, first mismatch description).
    fn compare(want: &EncodeImageOutput, got: &EncodeImageOutput) -> (usize, Option<String>) {
        let mut ok = 0;
        let mut first = None;
        for (i, (a, b)) in want.block_info.iter().zip(&got.block_info).enumerate() {
            let mut why = None;
            if a.out_blocks.len() != b.out_blocks.len() {
                why = Some(format!(
                    "count want {} got {}",
                    a.out_blocks.len(),
                    b.out_blocks.len()
                ));
            } else if a.packed_out_block_index != b.packed_out_block_index {
                why = Some(format!(
                    "packed want {} got {}",
                    a.packed_out_block_index, b.packed_out_block_index
                ));
            } else if (
                a.low_freq_block_flag,
                a.super_strong_edges,
                a.very_strong_edges,
                a.strong_edges,
            ) != (
                b.low_freq_block_flag,
                b.super_strong_edges,
                b.very_strong_edges,
                b.strong_edges,
            ) {
                why = Some("flags".into());
            } else {
                for (j, (x, y)) in a.out_blocks.iter().zip(&b.out_blocks).enumerate() {
                    if x.trial_mode_index != y.trial_mode_index
                        || x.blur_id != y.blur_id
                        || x.sse != y.sse
                        || !log_blk_eq(&x.log_blk, &y.log_blk)
                        || x.packed_dct_plane_data != y.packed_dct_plane_data
                    {
                        if std::env::var("XU_VERBOSE").is_ok() {
                            println!(
                                "want {:?}\n{:?}\ngot  {:?}\n{:?}",
                                x.log_blk,
                                x.packed_dct_plane_data,
                                y.log_blk,
                                y.packed_dct_plane_data
                            );
                        }
                        why = Some(format!(
                            "cand {j}: tm {}/{} blur {}/{} sse {}/{}",
                            x.trial_mode_index,
                            y.trial_mode_index,
                            x.blur_id,
                            y.blur_id,
                            x.sse,
                            y.sse
                        ));
                        break;
                    }
                }
            }
            match why {
                None => ok += 1,
                Some(w) => {
                    if first.is_none() {
                        first = Some(format!(
                            "block {i} ({},{}): {w}",
                            i as u32 % want.blocks_x,
                            i as u32 / want.blocks_x
                        ));
                    }
                }
            }
        }
        (ok, first)
    }

    fn cases() -> Option<(String, Vec<Case>)> {
        let dir = std::env::var("XU_DUMP_DIR").ok()?;
        let txt = std::fs::read_to_string(format!("{dir}/cases.txt")).ok()?;
        let filter = std::env::var("XU_CASE").unwrap_or_default();
        let v = txt
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(parse_case)
            .filter(|c| c.name.starts_with(&filter))
            .collect();
        Some((dir, v))
    }

    #[test]
    #[ignore = "needs XU_DUMP_DIR (C++ encoder dumps, cases.txt)"]
    fn basisu_image_encoder_matches_cpp_dumps() {
        let Some((_dir, cases)) = cases() else {
            return;
        };
        let threads = std::env::var("XU_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(4);
        let (mut tot_ok, mut tot) = (0, 0);
        let mut bad = Vec::new();
        for c in &cases {
            let d = dump::load(&c.file).unwrap();
            let (cfg, limit) = config_for(c, d.out.blocks_x * d.out.blocks_y);
            let t = std::time::Instant::now();
            let got = ldr_astc_block_encode_image(&d.image, &cfg, limit, threads).unwrap();
            let el = t.elapsed().as_secs_f64();
            let (ok, first) = compare(&d.out, &got);
            let n = d.out.block_info.len();
            tot_ok += ok;
            tot += n;
            println!("{}: {ok}/{n} blocks match ({el:.3}s)", c.name);
            if let Some(f) = first {
                println!("  first mismatch: {f}");
                bad.push(c.name.clone());
            }
        }
        println!("total {tot_ok}/{tot} blocks match, {} bad cases", bad.len());
        assert!(bad.is_empty(), "mismatching cases: {bad:?}");
    }

    /// Prints the golden lines of `testdata/basisu_enc/basisu_astc_ldr_image.txt` from the C++
    /// dumps of the regression cases.
    #[test]
    #[ignore = "needs XU_DUMP_DIR (C++ encoder dumps, cases.txt)"]
    fn print_golden_hashes() {
        let Some((_dir, cases)) = cases() else {
            return;
        };
        for c in &cases {
            if let Some(r) = REGRESSION_CASES.iter().find(|r| r.0 == c.name) {
                let d = dump::load(&c.file).unwrap();
                assert!(d.image == load_raw(r.1), "{}: raw image differs", c.name);
                println!("{}\t{:016x}", c.name, fnv64(&serialize(&d.out)));
            }
        }
    }

    /// (dump case name, raw image, `cases.txt` line).
    const REGRESSION_CASES: [(&str, &str, &str); 4] = [
        (
            "1_6x6_rgb_37x29_e3_s1",
            "rgb_37x29.rgba",
            "1_6x6_rgb_37x29_e3_s1 x 6 6 3 1 1 60 0 0 0",
        ),
        (
            "0_4x4_rgb_37x29_e2_s0",
            "rgb_37x29.rgba",
            "0_4x4_rgb_37x29_e2_s0 x 4 4 2 0 0 60 0 0 0",
        ),
        (
            "1_8x6_mosaic_rgba_50x38_e3_s0",
            "mosaic_rgba_50x38.rgba",
            "1_8x6_mosaic_rgba_50x38_e3_s0 x 8 6 3 0 1 60 0 0 0",
        ),
        (
            "1_12x12_mosaic_rgba_50x38_e3_s1_scd",
            "mosaic_rgba_50x38.rgba",
            "1_12x12_mosaic_rgba_50x38_e3_s1_scd x 12 12 3 1 1 60 1 0 0",
        ),
    ];

    fn load_raw(name: &str) -> Image {
        let (w, h) = match name {
            "rgb_37x29.rgba" => (37, 29),
            "rgba_64x48.rgba" => (64, 48),
            "mosaic_rgba_50x38.rgba" => (50, 38),
            _ => panic!("unknown image {name}"),
        };
        let path = format!("{}/testdata/images/raw/{name}", env!("CARGO_MANIFEST_DIR"));
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

    #[test]
    fn basisu_image_encoder_matches_golden_hashes() {
        let golden: BTreeMap<String, String> = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/basisu_enc/basisu_astc_ldr_image.txt"
        ))
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        for (i, (name, raw, line)) in REGRESSION_CASES.iter().enumerate() {
            let c = parse_case(line);
            let img = load_raw(raw);
            let bx = img.get_block_width(c.bw);
            let by = img.get_block_height(c.bh);
            let (cfg, limit) = config_for(&c, bx * by);
            let out = ldr_astc_block_encode_image(&img, &cfg, limit, 1 + i as u32 * 3).unwrap();
            let got = format!("{:016x}", fnv64(&serialize(&out)));
            assert_eq!(golden.get(*name), Some(&got), "{name}");
        }
    }

    /// Throughput (`cargo test --release speed -- --ignored --nocapture`).
    #[test]
    #[ignore = "timing only"]
    fn basisu_image_encoder_speed() {
        let src = load_raw("rgba_64x48.rgba");
        let (w, h) = (256u32, 256u32);
        let mut img = Image::with_size(w, h);
        for y in 0..h {
            for x in 0..w {
                // mirror-tile the source so blocks stay natural
                let (tx, ty) = (x / 64, y / 48);
                let sx = if tx % 2 == 0 { x % 64 } else { 63 - x % 64 };
                let sy = if ty % 2 == 0 { y % 48 } else { 47 - y % 48 };
                *img.get_mut(x, y) = src.get(sx, sy);
            }
        }
        let all = std::thread::available_parallelism().map_or(1, |n| n.get() as u32);
        for line in [
            "a x 6 6 3 1 0 60 0 0 0",
            "b x 6 6 3 1 1 60 0 0 0",
            "c x 4 4 3 1 0 60 0 0 0",
            "d x 12 12 3 1 1 60 0 0 0",
        ] {
            let c = parse_case(line);
            let (cfg, limit) =
                config_for(&c, img.get_block_width(c.bw) * img.get_block_height(c.bh));
            for threads in [1, all] {
                let t = std::time::Instant::now();
                ldr_astc_block_encode_image(&img, &cfg, limit, threads).unwrap();
                let el = t.elapsed().as_secs_f64();
                println!(
                    "{}x{} dct={} effort 3, {threads} threads: {:.3} MPix/s",
                    c.bw,
                    c.bh,
                    c.dct,
                    (w * h) as f64 / el / 1e6
                );
            }
        }
    }
}
