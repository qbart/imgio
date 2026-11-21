// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of `basisu_astc_ldr_encode.cpp` lines ~3560-5996: the full ("basisu") low-level block
//! encoder `ldr_astc_lowlevel_block_encoder` (partition triage, trial mode superbuckets,
//! analytic estimates, surrogate encoded shortlist buckets, bucket pruning and the final
//! polished encodes). The C++ encoder pool becomes an encoder object owned by each worker;
//! every block starts from an empty state of the same table sizes, so the output does not
//! depend on which worker encodes which block.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use super::EncodeBlockOutput;
use super::block_helpers::{
    BIG_FLOAT_VAL, LogSurrogateAstcBlk, TrialFlags, compute_ac_energy_from_dct,
    compute_lost_dct_energy_prefix_sum, compute_psnr_from_wsse, encode_surrogate_trial,
    encode_surrogate_trial_subsets, encode_trial, encode_trial_subsets, estimate_partition_sse,
    estimate_partition2, estimate_partition3, eval_error, fast_roundf_int, fast_roundf_pos_int,
    min_f, prepare_dct_energy_prefix_table,
};
use super::common::{CemEncodeParams, FLAG_DISABLE_QUANT, PixelStats};
use super::containers::{BuHashMap, BuKey, StdPriorityQueue};
use super::dct_enc::Dct2f;
use super::grid_data::find_astc_block_grid_data;
use super::partitions::PartitionsData;
use crate::astc::helpers::{
    BISE_64_LEVELS, BISE_256_LEVELS, CEM_LDR_LUM_ALPHA_DIRECT, CEM_LDR_LUM_DIRECT,
    CEM_LDR_RGB_BASE_PLUS_OFFSET, CEM_LDR_RGB_BASE_SCALE, CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A,
    CEM_LDR_RGB_DIRECT, CEM_LDR_RGBA_BASE_PLUS_OFFSET, CEM_LDR_RGBA_DIRECT, LogAstcBlock,
    MAX_BLOCK_PIXELS, get_ise_levels,
};
use crate::basis::ll::comp::Image;
use crate::basis::ll::enc::{hash_hsieh, indirect_sort, std_sort_by};
use crate::basis::xuastc::{
    ModeTable, OTM_NUM_CCS, OTM_NUM_CEMS, OTM_NUM_GRID_ANISOS, OTM_NUM_GRID_SIZES, OTM_NUM_SUBSETS,
    TrialMode,
};

const MAXP: usize = MAX_BLOCK_PIXELS;

pub(crate) const EXPECTED_SUPERBUCKET_HASH_SIZE: usize = 8192;
pub(crate) const EXPECTED_SHORTLIST_HASH_SIZE: usize = 4096;
pub(crate) const MAX_BASE_PARTS2: u32 = 128;
pub(crate) const MAX_BASE_PARTS3: u32 = 128;
pub(crate) const PART_ESTIMATE_STAGE1_MULTIPLIER: u32 = 4;
const MAX_EST_PARTS: usize = (MAX_BASE_PARTS2 * PART_ESTIMATE_STAGE1_MULTIPLIER) as usize;

const BLUE_CONTRACTION_BASE_OFS_DISCOUNT: f32 = 0.9;
const SKIP_IF_BUCKET_WORSE_MULTIPLIER: f32 = 5.0;

/// `astc_helpers::does_cem_have_alpha`.
fn does_cem_have_alpha(cem: u32) -> bool {
    matches!(cem, 4 | 5 | 10 | 12 | 13 | 14 | 15)
}

/// `astc_helpers::cem_is_ldr_base_scale`.
fn cem_is_ldr_base_scale(cem: u32) -> bool {
    cem == CEM_LDR_RGB_BASE_SCALE || cem == CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A
}

/// `ldr_astc_lowlevel_block_encoder_params`. [`Self::new`] sets the C++ `clear()` defaults.
#[derive(Clone, Copy)]
pub(crate) struct LdrAstcLowlevelBlockEncoderParams<'a> {
    pub bx: u32,
    pub by: u32,
    pub block_width: u32,
    pub block_height: u32,
    pub total_block_pixels: u32,

    pub orig_img_sobel_xy_t: &'a Image,

    pub part_data_p2: &'a PartitionsData,
    pub part_data_p3: &'a PartitionsData,

    pub enc_params: &'a CemEncodeParams<'a>,

    /// RGB or alpha trial lists (`m_num_trial_modes` = `trial_modes.len()`).
    pub trial_modes: &'a [TrialMode],
    /// `m_pGrouped_trial_modes` (only [`ModeTable::candidates`] is used).
    pub grouped_trial_modes: &'a ModeTable,

    /// [block_complexity_index]
    pub superbucket_max_to_retain: [u32; 3],

    pub shortlist_buckets_to_examine_fract: f32,
    pub shortlist_buckets_to_examine_min: u32,
    pub shortlist_buckets_to_examine_max: u32,

    pub num_similar_modes_in_bucket_to_shortlist_fract: f32,
    pub num_similar_modes_in_bucket_to_shortlist_fract_min: u32,
    pub num_similar_modes_in_bucket_to_shortlist_fract_max: u32,

    pub final_shortlist_fraction: [f32; 3],
    pub final_shortlist_min_size: [u32; 3],
    pub final_shortlist_max_size: [u32; 3],

    pub use_superbuckets: bool,
    pub bucket_pruning_passes: bool,

    /// True if this is a trial mode list containing alpha.
    pub alpha_cems: bool,

    pub use_alpha_or_opaque_modes: bool,
    pub use_lum_direct_modes: bool,
    pub use_base_scale_modes: bool,
    pub use_direct_modes: bool,
    pub use_dual_planes: bool,

    pub grid_hv_filtering: bool,
    pub filter_horizontally_flag: bool,
    pub use_small_grids_only: bool,

    pub dp_active_chans: [bool; 4],

    pub subsets_enabled: bool,
    pub subsets_edge_filtering: bool,

    pub gradient_descent_flag: bool,
    pub polish_weights_flag: bool,
    pub qcd_enabled_flag: bool,

    pub encode_trial_early_out_thresh: f32,
    pub encode_trial_subsets_early_out_thresh: f32,

    pub final_encode_try_base_ofs: bool,
    pub final_encode_always_try_rgb_direct: bool,

    pub brute_force_est_parts: bool,
    pub disable_part_est_stage2: bool,
    pub use_fast_part_est_stage2: bool,

    pub use_blue_contraction: bool,

    pub use_parts_std_dev_thresh: f32,
    pub use_parts_std_dev_thresh2: f32,
    pub sobel_energy_thresh1: f32,
    pub sobel_energy_thresh2: f32,
    pub sobel_energy_thresh3: f32,

    pub part2_fraction_to_keep: u32,
    pub part3_fraction_to_keep: u32,
    pub base_parts2: u32,
    pub base_parts3: u32,

    pub early_stop_wpsnr: f32,
    pub early_stop2_wpsnr: f32,

    /// At block size.
    pub dct2f: &'a Dct2f,
}

impl<'a> LdrAstcLowlevelBlockEncoderParams<'a> {
    /// Constructor + `clear()`; the pointer members are passed in.
    pub fn new(
        orig_img_sobel_xy_t: &'a Image,
        part_data_p2: &'a PartitionsData,
        part_data_p3: &'a PartitionsData,
        enc_params: &'a CemEncodeParams<'a>,
        trial_modes: &'a [TrialMode],
        grouped_trial_modes: &'a ModeTable,
        dct2f: &'a Dct2f,
    ) -> Self {
        let s = 0.2f32;
        Self {
            bx: 0,
            by: 0,
            block_width: 0,
            block_height: 0,
            total_block_pixels: 0,
            orig_img_sobel_xy_t,
            part_data_p2,
            part_data_p3,
            enc_params,
            trial_modes,
            grouped_trial_modes,
            superbucket_max_to_retain: [4, 8, 16],
            shortlist_buckets_to_examine_fract: 1.0,
            shortlist_buckets_to_examine_min: 1,
            shortlist_buckets_to_examine_max: 1024,
            num_similar_modes_in_bucket_to_shortlist_fract: 0.33,
            num_similar_modes_in_bucket_to_shortlist_fract_min: 2,
            num_similar_modes_in_bucket_to_shortlist_fract_max: 4096,
            final_shortlist_fraction: [0.2, 0.3, 0.5],
            final_shortlist_min_size: [1, 1, 1],
            final_shortlist_max_size: [4096, 4096, 4096],
            use_superbuckets: true,
            bucket_pruning_passes: true,
            alpha_cems: false,
            use_alpha_or_opaque_modes: false,
            use_lum_direct_modes: false,
            use_base_scale_modes: false,
            use_direct_modes: false,
            use_dual_planes: true,
            grid_hv_filtering: false,
            filter_horizontally_flag: false,
            use_small_grids_only: false,
            dp_active_chans: [true; 4],
            subsets_enabled: false,
            subsets_edge_filtering: true,
            gradient_descent_flag: true,
            polish_weights_flag: true,
            qcd_enabled_flag: true,
            encode_trial_early_out_thresh: 0.1,
            encode_trial_subsets_early_out_thresh: 0.1,
            final_encode_try_base_ofs: true,
            final_encode_always_try_rgb_direct: false,
            brute_force_est_parts: false,
            disable_part_est_stage2: false,
            use_fast_part_est_stage2: true,
            use_blue_contraction: true,
            use_parts_std_dev_thresh: (8.0f32 / 255.0) * s,
            use_parts_std_dev_thresh2: (40.0f32 / 255.0) * s,
            sobel_energy_thresh1: 3200.0 * s,
            sobel_energy_thresh2: 30000.0 * s,
            sobel_energy_thresh3: 50000.0 * s,
            part2_fraction_to_keep: 2,
            part3_fraction_to_keep: 2,
            base_parts2: 32,
            base_parts3: 32,
            early_stop_wpsnr: 0.0,
            early_stop2_wpsnr: 0.0,
            dct2f,
        }
    }

    fn trial_flags(&self) -> TrialFlags {
        TrialFlags {
            gradient_descent_flag: self.gradient_descent_flag,
            polish_weights_flag: self.polish_weights_flag,
            qcd_enabled_flag: self.qcd_enabled_flag,
            use_blue_contraction: self.use_blue_contraction,
        }
    }

    fn part_data(&self, num_parts: u32) -> &'a PartitionsData {
        if num_parts == 2 {
            self.part_data_p2
        } else {
            self.part_data_p3
        }
    }
}

/// `trial_surrogate`.
#[derive(Clone, Copy, Default)]
pub(crate) struct TrialSurrogate {
    pub trial_mode_index: u32,
    pub err: f32,
    pub log_blk: LogSurrogateAstcBlk,
}

/// `encode_block_stats`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EncodeBlockStats {
    pub total_superbuckets_created: u32,
    pub total_buckets_created: u32,
    pub total_surrogate_encodes: u32,
    pub total_shortlist_candidates: u32,
    pub total_full_encodes: u32,
}

/// `chan_mse_est`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ChanMseEst {
    pub ep: f32,
    pub wp: f32,
}

/// `weight_terms`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WeightTerms {
    pub mean: f32,
    pub var: f32,
    pub endpoint_factor: f32,
    pub weight_spread_scale: f32,
}

impl WeightTerms {
    /// `calc`.
    pub fn calc(&mut self, n: u32, weights: &[f32]) {
        let n_us = n as usize;
        let mut weight_total = 0.0f32;
        for &w in &weights[..n_us] {
            weight_total += w;
        }
        self.mean = weight_total / n as f32;
        let mut weight_var = 0.0f32;
        for &w in &weights[..n_us] {
            let d = w - self.mean;
            weight_var += d * d;
        }
        self.var = weight_var / n as f32;
        self.endpoint_factor =
            (1.0 + 2.0 * self.var + 2.0 * self.mean * self.mean - 2.0 * self.mean) * (3.0 / 2.0);
        self.endpoint_factor = self.endpoint_factor.clamp(0.25, 1.50);
        const UNIFORM_VAR: f32 = 1.0 / 12.0;
        let s = self.var / UNIFORM_VAR;
        self.weight_spread_scale = s.clamp(0.0, 1.0);
    }
}

/// `compute_quantized_channel_mse_estimates` (`BASISU_MODIFIED_WEIGHT_QUANT_MSE_ESTIMATE`).
pub(crate) fn compute_quantized_channel_mse_estimates(
    num_endpoint_levels: u32,
    num_weight_levels: u32,
    span_size: f32,
    weight_gamma: f32,
    weight_terms: Option<&WeightTerms>,
) -> ChanMseEst {
    let dep = 1.0f32 / (num_endpoint_levels - 1) as f32;
    let dw = 1.0f32 / num_weight_levels as f32;
    let mut ep_lower = (dep * dep) / 12.0 * (2.0f32 / 3.0);
    let mut wq_lower = (dw * dw) / 12.0 * weight_gamma * (span_size * span_size);
    if let Some(t) = weight_terms {
        ep_lower *= t.endpoint_factor;
        wq_lower *= t.weight_spread_scale;
    }
    ChanMseEst {
        ep: ep_lower,
        wp: wq_lower,
    }
}

/// `compute_quantized_channel_endpoint_mse_estimate`.
pub(crate) fn compute_quantized_channel_endpoint_mse_estimate(
    num_endpoint_levels: u32,
    weight_terms: Option<&WeightTerms>,
) -> f32 {
    let dep = 1.0f32 / (num_endpoint_levels - 1) as f32;
    let mut ep_lower = (dep * dep) * (1.0f32 / 12.0) * (2.0f32 / 3.0);
    if let Some(t) = weight_terms {
        ep_lower *= t.endpoint_factor;
    }
    ep_lower
}

/// `compute_quantized_channel_weight_mse_estimate`.
pub(crate) fn compute_quantized_channel_weight_mse_estimate(
    num_weight_levels: u32,
    span_size: f32,
    weight_gamma: f32,
    weight_terms: Option<&WeightTerms>,
) -> f32 {
    let dw = 1.0f32 / num_weight_levels as f32;
    let mut wq_lower = (dw * dw) * (1.0f32 / 12.0) * weight_gamma * (span_size * span_size);
    if let Some(t) = weight_terms {
        wq_lower *= t.weight_spread_scale;
    }
    wq_lower
}

/// `shortlist_bucket`. Hashing/equality use only the configuration fields.
#[derive(Clone, Copy)]
pub(crate) struct ShortlistBucket {
    pub examined_flag: bool,
    pub grid_width: i8,
    pub grid_height: i8,
    pub ccs_index: i8,
    pub cem_index: u8,
    pub num_parts: u8,
    pub unique_seed_index: u16,
    pub surrogate_log_blk: LogSurrogateAstcBlk,
    pub sse: f32,
}

impl ShortlistBucket {
    pub fn new(
        grid_width: i32,
        grid_height: i32,
        cem_index: u32,
        ccs_index: i32,
        num_parts: u32,
        unique_seed_index: u32,
    ) -> Self {
        Self {
            examined_flag: false,
            grid_width: grid_width as i8,
            grid_height: grid_height as i8,
            ccs_index: ccs_index as i8,
            cem_index: cem_index as u8,
            num_parts: num_parts as u8,
            unique_seed_index: unique_seed_index as u16,
            surrogate_log_blk: LogSurrogateAstcBlk::default(),
            sse: 0.0,
        }
    }
}

impl BuKey for ShortlistBucket {
    fn bu_hash(&self) -> u64 {
        let mut h = 0u64;
        h ^= hash_hsieh(&[self.grid_width as u8]) as u64;
        h ^= hash_hsieh(&[self.grid_height as u8]) as u64;
        h ^= hash_hsieh(&[self.ccs_index as u8]) as u64;
        h ^= hash_hsieh(&[self.cem_index]) as u64;
        h ^= hash_hsieh(&[self.num_parts]) as u64;
        h ^= hash_hsieh(&self.unique_seed_index.to_le_bytes()) as u64;
        h
    }

    fn bu_eq(&self, r: &Self) -> bool {
        self.grid_width == r.grid_width
            && self.grid_height == r.grid_height
            && self.cem_index == r.cem_index
            && self.ccs_index == r.ccs_index
            && self.num_parts == r.num_parts
            && self.unique_seed_index == r.unique_seed_index
    }
}

/// `trial_mode_index_vec` (`static_vector<uint16_t, 16>`: pushes past 16 are dropped).
#[derive(Clone, Copy, Default)]
pub(crate) struct TrialModeIndexVec {
    data: [u16; 16],
    size: u32,
}

impl TrialModeIndexVec {
    pub fn push(&mut self, v: u16) {
        if (self.size as usize) < self.data.len() {
            self.data[self.size as usize] = v;
            self.size += 1;
        }
    }

    pub fn as_slice(&self) -> &[u16] {
        &self.data[..self.size as usize]
    }

    pub fn len(&self) -> usize {
        self.size as usize
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

/// `trial_mode_estimate_superbucket_key` (8 bytes, hashed as a little endian `uint64_t`).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct SuperbucketKey {
    pub cem_index: u8,
    pub ccs_index: i8,
    pub subset_unique_index: u16,
    pub num_subsets: u8,
}

impl SuperbucketKey {
    fn as_u64(&self) -> u64 {
        self.cem_index as u64
            | ((self.ccs_index as u8 as u64) << 8)
            | ((self.subset_unique_index as u64) << 16)
            | ((self.num_subsets as u64) << 32)
    }
}

impl BuKey for SuperbucketKey {
    fn bu_hash(&self) -> u64 {
        let mut x = self.as_u64();
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x
    }

    fn bu_eq(&self, r: &Self) -> bool {
        self == r
    }
}

/// `trial_mode_estimate`.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct TrialModeEstimate {
    pub superbucket_key: SuperbucketKey,
    pub trial_mode_index: u32,
    pub wsse: f32,
}

fn tme_less(a: &TrialModeEstimate, b: &TrialModeEstimate) -> bool {
    a.wsse < b.wsse
}

/// `ranked_shortlist_bucket`.
#[derive(Clone, Copy)]
pub(crate) struct RankedShortlistBucket {
    pub bucket: ShortlistBucket,
    pub trial_mode_indices: TrialModeIndexVec,
}

/// `ldr_astc_lowlevel_block_encoder` (per-thread scratch state + the block encoder).
pub(crate) struct LdrAstcLowlevelBlockEncoder {
    trial_modes_to_estimate: Vec<u32>,
    superbucket_hash: BuHashMap<SuperbucketKey, Vec<u32>>,
    trial_mode_estimate_priority_queue: StdPriorityQueue<TrialModeEstimate>,
    dct_work: Vec<f32>,
    shortlist_hash0: BuHashMap<ShortlistBucket, TrialModeIndexVec>,
    shortlist_hash1: BuHashMap<ShortlistBucket, TrialModeIndexVec>,
    trial_surrogates: Vec<TrialSurrogate>,

    pub sobel_energy: f32,
    pub max_std_dev: f32,
    /// [0,2]
    pub block_complexity_index: u32,
    pub strong_edges: bool,
    pub very_strong_edges: bool,
    pub super_strong_edges: bool,
    pub used_superbuckets: bool,

    /// [rgb(a) direct / rgbs][est_part]
    best_parts2: [[i32; MAX_EST_PARTS]; 2],
    num_est_parts2: [i32; 2],
    best_parts3: [[i32; MAX_EST_PARTS]; 2],
    num_est_parts3: [i32; 2],

    ranked_buckets: Vec<RankedShortlistBucket>,
    order: Vec<u32>,
}

impl Default for LdrAstcLowlevelBlockEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl LdrAstcLowlevelBlockEncoder {
    pub fn new() -> Self {
        Self {
            trial_modes_to_estimate: Vec::new(),
            superbucket_hash: BuHashMap::new(),
            trial_mode_estimate_priority_queue: StdPriorityQueue::default(),
            dct_work: Vec::new(),
            shortlist_hash0: BuHashMap::new(),
            shortlist_hash1: BuHashMap::new(),
            trial_surrogates: Vec::new(),
            sobel_energy: 0.0,
            max_std_dev: 0.0,
            block_complexity_index: 0,
            strong_edges: false,
            very_strong_edges: false,
            super_strong_edges: false,
            used_superbuckets: false,
            best_parts2: [[0; MAX_EST_PARTS]; 2],
            num_est_parts2: [0; 2],
            best_parts3: [[0; MAX_EST_PARTS]; 2],
            num_est_parts3: [0; 2],
            ranked_buckets: Vec::new(),
            order: Vec::new(),
        }
    }

    /// `clear`.
    pub fn clear(&mut self) {
        self.trial_modes_to_estimate.clear();
        self.superbucket_hash.reset();
        self.trial_surrogates.clear();
        self.sobel_energy = 0.0;
        self.max_std_dev = 0.0;
        self.block_complexity_index = 0;
        self.strong_edges = false;
        self.very_strong_edges = false;
        self.super_strong_edges = false;
        self.used_superbuckets = false;
        self.best_parts2 = [[0; MAX_EST_PARTS]; 2];
        self.num_est_parts2 = [0; 2];
        self.best_parts3 = [[0; MAX_EST_PARTS]; 2];
        self.num_est_parts3 = [0; 2];
        self.ranked_buckets.clear();
    }

    /// `init`: block edge / complexity classification.
    pub fn init(
        &mut self,
        p: &LdrAstcLowlevelBlockEncoderParams,
        pixel_stats: &PixelStats,
    ) -> bool {
        self.sobel_energy = 0.0;
        for y in 0..p.block_height {
            for x in 0..p.block_width {
                let s = p.orig_img_sobel_xy_t.get_clamped(
                    (p.bx * p.block_width + x) as i32,
                    (p.by * p.block_height + y) as i32,
                );
                let s = s.map(|v| v as i32);
                self.sobel_energy += (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3]) as f32;
            }
        }
        self.sobel_energy /= p.total_block_pixels as f32;

        self.max_std_dev = 0.0;
        for i in 0..4 {
            let v = pixel_stats.rgba_stats[i].std_dev;
            if self.max_std_dev < v {
                self.max_std_dev = v;
            }
        }

        self.strong_edges = self.max_std_dev > p.use_parts_std_dev_thresh
            && self.sobel_energy > p.sobel_energy_thresh1;
        self.very_strong_edges = self.max_std_dev > p.use_parts_std_dev_thresh2
            && self.sobel_energy > p.sobel_energy_thresh2;
        self.super_strong_edges = self.max_std_dev > p.use_parts_std_dev_thresh2
            && self.sobel_energy > p.sobel_energy_thresh3;
        self.block_complexity_index = if self.super_strong_edges {
            2
        } else if self.very_strong_edges {
            1
        } else {
            0
        };
        true
    }

    /// Stage 2 of `partition_triage` for one subset count / CEM class.
    fn partition_triage_stage2(
        &self,
        p: &LdrAstcLowlevelBlockEncoderParams,
        pixel_stats: &PixelStats,
        stats: &mut EncodeBlockStats,
        num_subsets: u32,
        s: usize,
        best_parts_temp: &[i32],
        fraction_to_keep: u32,
        best_parts_out: &mut [i32; MAX_EST_PARTS],
    ) -> i32 {
        let total_parts = best_parts_temp.len();
        let mut cem_to_surrogate_encode = if p.alpha_cems {
            CEM_LDR_RGBA_DIRECT
        } else {
            CEM_LDR_RGB_DIRECT
        };
        if s != 0 {
            cem_to_surrogate_encode = if p.alpha_cems {
                CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A
            } else {
                CEM_LDR_RGB_BASE_SCALE
            };
        }
        let part_data = p.part_data(num_subsets);
        let mut part_sses = vec![0.0f32; total_parts];
        for i in 0..total_parts {
            let unique_seed_index = best_parts_temp[i] as usize;
            let part_seed_index = part_data.unique_index_to_part_seed[unique_seed_index] as u32;
            let pat = &part_data.partition_pats[unique_seed_index];
            let sse = if p.use_fast_part_est_stage2 {
                estimate_partition_sse(
                    cem_to_surrogate_encode,
                    pixel_stats,
                    p.block_width,
                    p.block_height,
                    num_subsets,
                    pat,
                )
            } else {
                let mut slb = LogSurrogateAstcBlk::default();
                stats.total_surrogate_encodes += 1;
                encode_surrogate_trial_subsets(
                    p.block_width,
                    p.block_height,
                    pixel_stats,
                    cem_to_surrogate_encode,
                    num_subsets,
                    part_seed_index,
                    pat,
                    BISE_256_LEVELS,
                    BISE_64_LEVELS,
                    p.block_width,
                    p.block_height,
                    &mut slb,
                    p.enc_params,
                    0,
                )
            };
            part_sses[i] = sse;
        }
        let mut ranks = vec![0u32; total_parts];
        indirect_sort(&mut ranks, &part_sses);
        let n = (total_parts as u32).div_ceil(fraction_to_keep).max(1) as i32;
        for i in 0..n as usize {
            best_parts_out[i] = best_parts_temp[ranks[i] as usize];
        }
        n
    }

    /// `partition_triage`.
    pub fn partition_triage(
        &mut self,
        p: &LdrAstcLowlevelBlockEncoderParams,
        pixel_stats: &PixelStats,
        stats: &mut EncodeBlockStats,
    ) -> bool {
        self.num_est_parts2 = [0; 2];
        self.num_est_parts3 = [0; 2];
        if !p.subsets_enabled {
            return true;
        }
        if p.subsets_edge_filtering && !self.strong_edges {
            return true;
        }

        for num_subsets in [2u32, 3] {
            let base = if num_subsets == 2 {
                p.base_parts2
            } else {
                p.base_parts3
            };
            let mut total_parts = if self.super_strong_edges {
                base * PART_ESTIMATE_STAGE1_MULTIPLIER
            } else if self.very_strong_edges {
                base * 2
            } else {
                base
            };
            total_parts = total_parts.min(MAX_EST_PARTS as u32);
            let part_data = p.part_data(num_subsets);
            total_parts = total_parts.min(part_data.total_unique_patterns);
            if total_parts == 0 {
                continue;
            }

            let mut temp = [0i32; MAX_EST_PARTS];
            let mut total = total_parts as i32;
            let has_est = if num_subsets == 2 {
                estimate_partition2(
                    p.block_width,
                    p.block_height,
                    pixel_stats,
                    &mut temp,
                    &mut total,
                    part_data,
                    p.brute_force_est_parts,
                )
            } else {
                estimate_partition3(
                    p.block_width,
                    p.block_height,
                    pixel_stats,
                    &mut temp,
                    &mut total,
                    part_data,
                    p.brute_force_est_parts,
                )
            };
            if !has_est {
                continue;
            }
            let fraction_to_keep = if num_subsets == 2 {
                p.part2_fraction_to_keep
            } else {
                p.part3_fraction_to_keep
            };

            for s in 0..2usize {
                if s != 0 && !p.use_base_scale_modes {
                    continue;
                }
                let n;
                let mut out = [0i32; MAX_EST_PARTS];
                if p.disable_part_est_stage2 {
                    n = total;
                    out[..total as usize].copy_from_slice(&temp[..total as usize]);
                } else {
                    n = self.partition_triage_stage2(
                        p,
                        pixel_stats,
                        stats,
                        num_subsets,
                        s,
                        &temp[..total as usize],
                        fraction_to_keep,
                        &mut out,
                    );
                }
                if num_subsets == 2 {
                    self.num_est_parts2[s] = n;
                    self.best_parts2[s][..n as usize].copy_from_slice(&out[..n as usize]);
                } else {
                    self.num_est_parts3[s] = n;
                    self.best_parts3[s][..n as usize].copy_from_slice(&out[..n as usize]);
                }
            }
        }
        true
    }

    /// `trivial_triage`: collects the candidate trial modes.
    pub fn trivial_triage(&mut self, p: &LdrAstcLowlevelBlockEncoderParams) -> bool {
        self.trial_modes_to_estimate.clear();
        for cem_index in CEM_LDR_LUM_DIRECT..OTM_NUM_CEMS {
            if does_cem_have_alpha(cem_index) != p.alpha_cems {
                continue;
            }
            if does_cem_have_alpha(cem_index) != p.use_alpha_or_opaque_modes {
                continue;
            }
            let accept_flag = match cem_index {
                CEM_LDR_LUM_DIRECT | CEM_LDR_LUM_ALPHA_DIRECT => p.use_lum_direct_modes,
                CEM_LDR_RGB_DIRECT | CEM_LDR_RGBA_DIRECT => p.use_direct_modes,
                CEM_LDR_RGB_BASE_SCALE | CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => {
                    p.use_base_scale_modes
                }
                _ => false,
            };
            if !accept_flag {
                continue;
            }
            let s = cem_is_ldr_base_scale(cem_index) as usize;
            for subsets_index in 0..OTM_NUM_SUBSETS {
                if subsets_index == 1 && self.num_est_parts2[s] == 0 {
                    continue;
                }
                if subsets_index == 2 && self.num_est_parts3[s] == 0 {
                    continue;
                }
                let ccs_max_index = if p.use_dual_planes { OTM_NUM_CCS } else { 1 };
                for ccs_index in 0..ccs_max_index {
                    if ccs_index != 0 && !p.dp_active_chans[(ccs_index - 1) as usize] {
                        continue;
                    }
                    for grid_size_index in 0..OTM_NUM_GRID_SIZES {
                        if grid_size_index != 0 && p.use_small_grids_only {
                            continue;
                        }
                        for grid_anisos_index in 0..OTM_NUM_GRID_ANISOS {
                            if p.grid_hv_filtering {
                                if grid_anisos_index == 1 && p.filter_horizontally_flag {
                                    continue;
                                }
                                if grid_anisos_index == 2 && !p.filter_horizontally_flag {
                                    continue;
                                }
                            }
                            let c = p.grouped_trial_modes.candidates(
                                cem_index,
                                subsets_index,
                                ccs_index,
                                grid_size_index,
                                grid_anisos_index,
                            );
                            self.trial_modes_to_estimate
                                .extend(c.iter().map(|&v| v as u32));
                        }
                    }
                }
            }
        }
        !self.trial_modes_to_estimate.is_empty()
    }

    /// `analytic_triage`: superbucket analytic estimation, fills the shortlist buckets.
    pub fn analytic_triage(
        &mut self,
        p: &LdrAstcLowlevelBlockEncoderParams,
        pixel_stats: &PixelStats,
        stats: &mut EncodeBlockStats,
    ) -> bool {
        if self.shortlist_hash0.table_size() != EXPECTED_SHORTLIST_HASH_SIZE {
            self.shortlist_hash0.clear();
            self.shortlist_hash0
                .reserve(EXPECTED_SHORTLIST_HASH_SIZE / 2);
        } else {
            self.shortlist_hash0.reset();
        }

        self.used_superbuckets = false;

        if p.use_superbuckets {
            self.used_superbuckets = true;
            if self.superbucket_hash.table_size() != EXPECTED_SUPERBUCKET_HASH_SIZE {
                self.superbucket_hash.clear();
                self.superbucket_hash
                    .reserve(EXPECTED_SUPERBUCKET_HASH_SIZE >> 1);
            } else {
                self.superbucket_hash.reset();
            }

            // Create superbuckets.
            for j in 0..self.trial_modes_to_estimate.len() {
                let trial_mode_iter = self.trial_modes_to_estimate[j];
                let tm = &p.trial_modes[trial_mode_iter as usize];
                let mut key = SuperbucketKey {
                    cem_index: tm.cem,
                    ccs_index: tm.ccs_index,
                    subset_unique_index: 0,
                    num_subsets: tm.num_parts,
                };
                if tm.num_parts == 1 {
                    let (e, created) = self.superbucket_hash.insert(key, Vec::new());
                    let v = &mut self.superbucket_hash.entry_mut(e).1;
                    if created {
                        v.reserve(256);
                    }
                    v.push(trial_mode_iter);
                } else {
                    let s = cem_is_ldr_base_scale(tm.cem as u32) as usize;
                    let n = if tm.num_parts == 2 {
                        self.num_est_parts2[s]
                    } else {
                        self.num_est_parts3[s]
                    };
                    for est_part_iter in 0..n.max(0) as usize {
                        let part_unique_index = if tm.num_parts == 2 {
                            self.best_parts2[s][est_part_iter]
                        } else {
                            self.best_parts3[s][est_part_iter]
                        };
                        key.subset_unique_index = part_unique_index as u16;
                        let (e, created) = self.superbucket_hash.insert(key, Vec::new());
                        let v = &mut self.superbucket_hash.entry_mut(e).1;
                        if created {
                            v.reserve(256);
                        }
                        v.push(trial_mode_iter);
                    }
                }
            }

            let total_block_texels = p.total_block_pixels;
            let inv_total_block_texels = 1.0f32 / total_block_texels as f32;

            self.trial_mode_estimate_priority_queue.clear();
            let max_pq_size =
                p.superbucket_max_to_retain[self.block_complexity_index as usize] as usize;

            const SLAM_TO_LINE_WEIGHT: f32 = 1.5;
            const QUANT_ERROR_WEIGHT: f32 = 1.0;
            const SCALE_ERROR_WEIGHT: f32 = 3.0;
            const BLUE_CONTRACTION_ENDPOINT_QUANT_DISCOUNT: f32 = 0.5;

            let mut order = std::mem::take(&mut self.order);
            self.superbucket_hash.iter_order(&mut order);

            let bw = p.block_width;
            let bh = p.block_height;

            for &ei in &order {
                let (key, val) = self.superbucket_hash.entry(ei as usize);
                let key = *key;
                let pq = &mut self.trial_mode_estimate_priority_queue;

                let mut log_blk = LogSurrogateAstcBlk::default();
                let mut part_data: Option<&PartitionsData> = None;

                let worst_wsse_found_so_far = if pq.len() >= max_pq_size {
                    pq.top().map_or(1e9, |t| t.wsse)
                } else {
                    1e9
                };

                let slam_to_line_wsse = if key.num_subsets == 1 {
                    encode_surrogate_trial(
                        bw,
                        bh,
                        pixel_stats,
                        key.cem_index as u32,
                        key.ccs_index as i32,
                        BISE_256_LEVELS,
                        BISE_64_LEVELS,
                        bw,
                        bh,
                        &mut log_blk,
                        p.enc_params,
                        FLAG_DISABLE_QUANT,
                    )
                } else {
                    let pd = p.part_data(if key.num_subsets == 3 { 3 } else { 2 });
                    part_data = Some(pd);
                    let unique_seed_index = key.subset_unique_index as usize;
                    let part_seed_index = pd.unique_index_to_part_seed[unique_seed_index] as u32;
                    let pat = &pd.partition_pats[unique_seed_index];
                    encode_surrogate_trial_subsets(
                        bw,
                        bh,
                        pixel_stats,
                        key.cem_index as u32,
                        key.num_subsets as u32,
                        part_seed_index,
                        pat,
                        BISE_256_LEVELS,
                        BISE_64_LEVELS,
                        bw,
                        bh,
                        &mut log_blk,
                        p.enc_params,
                        FLAG_DISABLE_QUANT,
                    )
                };
                stats.total_surrogate_encodes += 1;

                if SLAM_TO_LINE_WEIGHT * slam_to_line_wsse >= worst_wsse_found_so_far {
                    continue;
                }

                let ns = key.num_subsets as usize;
                let mut can_use_base_ofs = false;
                if key.cem_index as u32 == CEM_LDR_RGB_DIRECT
                    || key.cem_index as u32 == CEM_LDR_RGBA_DIRECT
                {
                    let mut max_span_size = 0.0f32;
                    for si in 0..ns {
                        for c in 0..4 {
                            let span =
                                (log_blk.endpoints[si][1][c] - log_blk.endpoints[si][0][c]).abs();
                            if max_span_size < span {
                                max_span_size = span;
                            }
                        }
                    }
                    can_use_base_ofs = max_span_size < 0.25;
                }

                let mut e0 = [0f32; MAXP];
                let mut e1 = [0f32; MAXP];
                let mut ps0 = [0f32; MAXP];
                let mut ps1 = [0f32; MAXP];

                p.dct2f.forward(&log_blk.weights0, &mut e0);
                let total_w0_ac = compute_ac_energy_from_dct(bw, bh, &mut e0);
                prepare_dct_energy_prefix_table(bw, bh, &e0, &mut ps0);

                let mut total_w1_ac = 0.0f32;
                if key.ccs_index >= 0 {
                    p.dct2f.forward(&log_blk.weights1, &mut e1);
                    total_w1_ac = compute_ac_energy_from_dct(bw, bh, &mut e1);
                    prepare_dct_energy_prefix_table(bw, bh, &e1, &mut ps1);
                }

                let mut w0_terms = WeightTerms::default();
                let mut w1_terms = WeightTerms::default();
                w0_terms.calc(total_block_texels, &log_blk.weights0);
                if key.ccs_index >= 0 {
                    w1_terms.calc(total_block_texels, &log_blk.weights1);
                }

                let mut subset_spans = [[0f32; 4]; 4];
                let mut subset_pixels = [0u32; 4];
                for si in 0..ns {
                    subset_spans[si] = std::array::from_fn(|c| {
                        log_blk.endpoints[si][1][c] - log_blk.endpoints[si][0][c]
                    });
                    let mut tsp = p.total_block_pixels;
                    if key.num_subsets > 1
                        && let Some(pd) = part_data
                    {
                        tsp = pd.partition_pat_histograms[key.subset_unique_index as usize][si]
                            as u32;
                    }
                    subset_pixels[si] = tsp;
                }

                let skip_chan = |si: usize, c: usize, span_size: f32| {
                    span_size == 0.0
                        && (log_blk.endpoints[si][1][c] == 0.0
                            || log_blk.endpoints[si][1][c] == 1.0)
                };

                for &trial_mode_index in val {
                    let tm = &p.trial_modes[trial_mode_index as usize];
                    let tgw = tm.grid_w as u32;
                    let tgh = tm.grid_h as u32;
                    let Some(grid_data) = find_astc_block_grid_data(bw, bh, tgw, tgh) else {
                        continue;
                    };
                    let total_endpoint_levels = get_ise_levels(tm.endpoint_range as u32);
                    let total_weight_levels = get_ise_levels(tm.weight_range as u32);
                    let num_effective_e_levels = if can_use_base_ofs {
                        (total_endpoint_levels * 2).min(256)
                    } else {
                        total_endpoint_levels
                    };
                    let mut qe0 = compute_quantized_channel_endpoint_mse_estimate(
                        num_effective_e_levels,
                        None,
                    );
                    let qe1 = if key.ccs_index >= 0 {
                        qe0 * w1_terms.endpoint_factor
                    } else {
                        0.0
                    };
                    qe0 *= w0_terms.endpoint_factor;

                    let mut total_e_quant_wsse = 0.0f32;
                    for si in 0..ns {
                        for c in 0..4 {
                            let span_size = subset_spans[si][c].abs();
                            if skip_chan(si, c, span_size) {
                                continue;
                            }
                            let chan_n =
                                p.enc_params.comp_weights[c] as f32 * subset_pixels[si] as f32;
                            total_e_quant_wsse += (if key.ccs_index as i32 == c as i32 {
                                qe1
                            } else {
                                qe0
                            }) * chan_n;
                        }
                    }

                    if tm.cem as u32 == CEM_LDR_RGB_DIRECT || tm.cem as u32 == CEM_LDR_RGBA_DIRECT {
                        total_e_quant_wsse *= BLUE_CONTRACTION_ENDPOINT_QUANT_DISCOUNT;
                    }

                    let mut total_wsse_so_far = (SLAM_TO_LINE_WEIGHT * slam_to_line_wsse)
                        + (QUANT_ERROR_WEIGHT * total_e_quant_wsse);
                    if total_wsse_so_far >= worst_wsse_found_so_far {
                        continue;
                    }

                    let lost0 = compute_lost_dct_energy_prefix_sum(bw, &ps0, tgw, tgh, total_w0_ac)
                        * inv_total_block_texels;
                    let mut lost1 = 0.0f32;
                    if key.ccs_index >= 0 {
                        lost1 = compute_lost_dct_energy_prefix_sum(bw, &ps1, tgw, tgh, total_w1_ac)
                            * inv_total_block_texels;
                    }

                    if lost0 != 0.0 || lost1 != 0.0 {
                        let mut total_scale_wsse = 0.0f32;
                        for si in 0..ns {
                            for c in 0..4 {
                                let span_size = subset_spans[si][c].abs();
                                if skip_chan(si, c, span_size) {
                                    continue;
                                }
                                let chan_n =
                                    p.enc_params.comp_weights[c] as f32 * subset_pixels[si] as f32;
                                let lost = if key.ccs_index as i32 == c as i32 {
                                    lost1
                                } else {
                                    lost0
                                };
                                total_scale_wsse += lost * (span_size * span_size) * chan_n;
                            }
                        }
                        total_wsse_so_far += SCALE_ERROR_WEIGHT * total_scale_wsse;
                        if total_wsse_so_far >= worst_wsse_found_so_far {
                            continue;
                        }
                    }

                    let mut total_w_quant_wsse = 0.0f32;
                    for si in 0..ns {
                        for c in 0..4 {
                            let span_size = subset_spans[si][c].abs();
                            if skip_chan(si, c, span_size) {
                                continue;
                            }
                            let terms = if key.ccs_index as i32 == c as i32 {
                                &w1_terms
                            } else {
                                &w0_terms
                            };
                            let chan_w_mse = compute_quantized_channel_weight_mse_estimate(
                                total_weight_levels,
                                span_size,
                                grid_data.weight_gamma,
                                Some(terms),
                            );
                            let chan_n =
                                p.enc_params.comp_weights[c] as f32 * subset_pixels[si] as f32;
                            total_w_quant_wsse += chan_w_mse * chan_n;
                        }
                    }

                    let total_wsse = total_wsse_so_far + (QUANT_ERROR_WEIGHT * total_w_quant_wsse);

                    let est = TrialModeEstimate {
                        superbucket_key: key,
                        trial_mode_index,
                        wsse: total_wsse,
                    };
                    if pq.len() >= max_pq_size {
                        if pq.top().is_some_and(|t| total_wsse < t.wsse) {
                            pq.pop(tme_less);
                            pq.push(est, tme_less);
                        }
                    } else {
                        pq.push(est, tme_less);
                    }
                }
            }
            self.order = order;

            stats.total_superbuckets_created += self.superbucket_hash.len() as u32;

            let total_estimates_to_retain = self.trial_mode_estimate_priority_queue.len();
            for _ in 0..total_estimates_to_retain {
                let Some(&est) = self.trial_mode_estimate_priority_queue.top() else {
                    break;
                };
                let tm = &p.trial_modes[est.trial_mode_index as usize];
                let (e, _) = self.shortlist_hash0.insert(
                    ShortlistBucket::new(
                        tm.grid_w as i32,
                        tm.grid_h as i32,
                        tm.cem as u32,
                        tm.ccs_index as i32,
                        tm.num_parts as u32,
                        est.superbucket_key.subset_unique_index as u32,
                    ),
                    TrialModeIndexVec::default(),
                );
                self.shortlist_hash0
                    .entry_mut(e)
                    .1
                    .push(est.trial_mode_index as u16);
                self.trial_mode_estimate_priority_queue.pop(tme_less);
            }
        } else {
            for j in 0..self.trial_modes_to_estimate.len() {
                let trial_mode_iter = self.trial_modes_to_estimate[j];
                let tm = &p.trial_modes[trial_mode_iter as usize];
                if tm.num_parts > 1 {
                    let s = cem_is_ldr_base_scale(tm.cem as u32) as usize;
                    let n = if tm.num_parts == 2 {
                        self.num_est_parts2[s]
                    } else {
                        self.num_est_parts3[s]
                    };
                    for est_part_iter in 0..n.max(0) as usize {
                        let part_unique_index = if tm.num_parts == 2 {
                            self.best_parts2[s][est_part_iter]
                        } else {
                            self.best_parts3[s][est_part_iter]
                        };
                        let (e, _) = self.shortlist_hash0.insert(
                            ShortlistBucket::new(
                                tm.grid_w as i32,
                                tm.grid_h as i32,
                                tm.cem as u32,
                                tm.ccs_index as i32,
                                tm.num_parts as u32,
                                part_unique_index as u32,
                            ),
                            TrialModeIndexVec::default(),
                        );
                        self.shortlist_hash0
                            .entry_mut(e)
                            .1
                            .push(trial_mode_iter as u16);
                    }
                } else {
                    let (e, _) = self.shortlist_hash0.insert(
                        ShortlistBucket::new(
                            tm.grid_w as i32,
                            tm.grid_h as i32,
                            tm.cem as u32,
                            tm.ccs_index as i32,
                            1,
                            0,
                        ),
                        TrialModeIndexVec::default(),
                    );
                    self.shortlist_hash0
                        .entry_mut(e)
                        .1
                        .push(trial_mode_iter as u16);
                }
            }
        }

        stats.total_buckets_created += self.shortlist_hash0.len() as u32;
        true
    }

    /// `surrogate_encode_shortlist_bucket_representatives`.
    pub fn surrogate_encode_shortlist_bucket_representatives(
        &mut self,
        p: &LdrAstcLowlevelBlockEncoderParams,
        pixel_stats: &PixelStats,
        stats: &mut EncodeBlockStats,
    ) -> bool {
        let mut order = std::mem::take(&mut self.order);
        self.shortlist_hash0.iter_order(&mut order);
        for &ei in &order {
            let (bucket, indices) = self.shortlist_hash0.entry_mut(ei as usize);
            let (mut max_e, mut max_w) = (0u32, 0u32);
            for &ti in indices.as_slice() {
                let tm = &p.trial_modes[ti as usize];
                max_e = max_e.max(tm.endpoint_range as u32);
                max_w = max_w.max(tm.weight_range as u32);
            }
            if bucket.num_parts == 1 {
                bucket.sse = encode_surrogate_trial(
                    p.block_width,
                    p.block_height,
                    pixel_stats,
                    bucket.cem_index as u32,
                    bucket.ccs_index as i32,
                    max_e,
                    max_w,
                    bucket.grid_width as u32,
                    bucket.grid_height as u32,
                    &mut bucket.surrogate_log_blk,
                    p.enc_params,
                    0,
                );
            } else {
                let pd = p.part_data(bucket.num_parts as u32);
                let usi = bucket.unique_seed_index as usize;
                let part_seed_index = pd.unique_index_to_part_seed[usi] as u32;
                bucket.sse = encode_surrogate_trial_subsets(
                    p.block_width,
                    p.block_height,
                    pixel_stats,
                    bucket.cem_index as u32,
                    bucket.num_parts as u32,
                    part_seed_index,
                    &pd.partition_pats[usi],
                    max_e,
                    max_w,
                    bucket.grid_width as u32,
                    bucket.grid_height as u32,
                    &mut bucket.surrogate_log_blk,
                    p.enc_params,
                    0,
                );
            }
            stats.total_surrogate_encodes += 1;
            if bucket.cem_index as u32 == CEM_LDR_RGB_DIRECT
                || bucket.cem_index as u32 == CEM_LDR_RGBA_DIRECT
            {
                bucket.sse *= BLUE_CONTRACTION_BASE_OFS_DISCOUNT;
            }
        }
        self.order = order;
        true
    }

    /// `prune_shortlist_buckets`: optional pruning passes, then ranks the buckets by SSE.
    pub fn prune_shortlist_buckets(&mut self, p: &LdrAstcLowlevelBlockEncoderParams) -> bool {
        let mut order = std::mem::take(&mut self.order);

        if p.bucket_pruning_passes {
            if self.shortlist_hash1.table_size() != EXPECTED_SHORTLIST_HASH_SIZE {
                self.shortlist_hash1.clear();
                self.shortlist_hash1
                    .reserve(EXPECTED_SHORTLIST_HASH_SIZE / 2);
            } else {
                self.shortlist_hash1.reset();
            }

            const NUM_PRUNE_PASSES: u32 = 3;
            for prune_pass in 0..NUM_PRUNE_PASSES {
                let src = &mut self.shortlist_hash0;
                let dst = &mut self.shortlist_hash1;
                src.iter_order(&mut order);
                for &ei in &order {
                    src.entry_mut(ei as usize).0.examined_flag = false;
                }
                dst.reset();

                for &ei in &order {
                    let ei = ei as usize;
                    let bucket = src.entry(ei).0;
                    if bucket.examined_flag {
                        continue;
                    }
                    let mut copy_into_dst = |src: &BuHashMap<_, _>, i: usize| {
                        let (k, v): &(ShortlistBucket, TrialModeIndexVec) = src.entry(i);
                        dst.insert(*k, *v);
                    };

                    match prune_pass {
                        0 => {
                            // Dual plane groups: only accept the best CCS index.
                            if bucket.ccs_index >= 0 {
                                let mut ccs_buckets = [None; 4];
                                let mut best_ccs_index = -1i32;
                                let mut best_ccs_err = BIG_FLOAT_VAL;
                                for c in 0..4u32 {
                                    let f = src.find(&ShortlistBucket::new(
                                        bucket.grid_width as i32,
                                        bucket.grid_height as i32,
                                        bucket.cem_index as u32,
                                        c as i32,
                                        bucket.num_parts as u32,
                                        bucket.unique_seed_index as u32,
                                    ));
                                    ccs_buckets[c as usize] = f;
                                    let Some(fi) = f else {
                                        continue;
                                    };
                                    src.entry_mut(fi).0.examined_flag = true;
                                    let err = src.entry(fi).0.sse;
                                    if err < best_ccs_err {
                                        best_ccs_err = err;
                                        best_ccs_index = c as i32;
                                    }
                                }
                                if best_ccs_index >= 0
                                    && let Some(bi) = ccs_buckets[best_ccs_index as usize]
                                {
                                    copy_into_dst(src, bi);
                                }
                            } else {
                                copy_into_dst(src, ei);
                            }
                        }
                        1 => {
                            // Same # of weight samples: compare WxH vs. HxW.
                            if bucket.grid_width != bucket.grid_height {
                                let alt = src.find(&ShortlistBucket::new(
                                    bucket.grid_height as i32,
                                    bucket.grid_width as i32,
                                    bucket.cem_index as u32,
                                    bucket.ccs_index as i32,
                                    bucket.num_parts as u32,
                                    bucket.unique_seed_index as u32,
                                ));
                                match alt {
                                    None => copy_into_dst(src, ei),
                                    Some(ai) => {
                                        src.entry_mut(ai).0.examined_flag = true;
                                        let alt_sse = src.entry(ai).0.sse;
                                        let fract = if bucket.sse > 0.0 {
                                            alt_sse / bucket.sse
                                        } else {
                                            0.0
                                        };
                                        const ALT_RES_SSE_THRESH: f32 = 0.2;
                                        if fract < 1.0 - ALT_RES_SSE_THRESH {
                                            copy_into_dst(src, ai);
                                        } else if fract > 1.0 + ALT_RES_SSE_THRESH {
                                            copy_into_dst(src, ei);
                                        } else {
                                            copy_into_dst(src, ai);
                                            copy_into_dst(src, ei);
                                        }
                                    }
                                }
                            } else {
                                copy_into_dst(src, ei);
                            }
                        }
                        _ => {
                            // RGB direct vs. base+scale bucket groups.
                            let cem = bucket.cem_index as u32;
                            let alt_cem = match cem {
                                CEM_LDR_RGB_DIRECT => Some(CEM_LDR_RGB_BASE_SCALE),
                                CEM_LDR_RGB_BASE_SCALE => Some(CEM_LDR_RGB_DIRECT),
                                CEM_LDR_RGBA_DIRECT => Some(CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A),
                                CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => Some(CEM_LDR_RGBA_DIRECT),
                                _ => None,
                            };
                            if let Some(alt_cem) = alt_cem {
                                let alt = src.find(&ShortlistBucket::new(
                                    bucket.grid_width as i32,
                                    bucket.grid_height as i32,
                                    alt_cem,
                                    bucket.ccs_index as i32,
                                    bucket.num_parts as u32,
                                    bucket.unique_seed_index as u32,
                                ));
                                match alt {
                                    None => copy_into_dst(src, ei),
                                    Some(ai) => {
                                        src.entry_mut(ai).0.examined_flag = true;
                                        let alt_sse = src.entry(ai).0.sse;
                                        let fract = if bucket.sse > 0.0 {
                                            alt_sse / bucket.sse
                                        } else {
                                            0.0
                                        };
                                        const ALT_RES_SSE_THRESH: f32 = 0.1;
                                        if fract < 1.0 - ALT_RES_SSE_THRESH {
                                            copy_into_dst(src, ai);
                                        } else if fract > 1.0 + ALT_RES_SSE_THRESH {
                                            copy_into_dst(src, ei);
                                        } else {
                                            copy_into_dst(src, ai);
                                            copy_into_dst(src, ei);
                                        }
                                    }
                                }
                            } else {
                                copy_into_dst(src, ei);
                            }
                        }
                    }

                    src.entry_mut(ei).0.examined_flag = true;
                }

                self.shortlist_hash0.swap(&mut self.shortlist_hash1);
            }
        }

        self.shortlist_hash0.iter_order(&mut order);
        self.ranked_buckets.clear();
        for &ei in &order {
            let (bucket, indices) = self.shortlist_hash0.entry(ei as usize);
            self.ranked_buckets.push(RankedShortlistBucket {
                bucket: *bucket,
                trial_mode_indices: *indices,
            });
        }
        self.order = order;

        if self.ranked_buckets.is_empty() {
            return true;
        }

        // std::sort by bucket SSE (sorting an index permutation gives the same order).
        let mut idx: Vec<u32> = (0..self.ranked_buckets.len() as u32).collect();
        {
            let rb = &self.ranked_buckets;
            std_sort_by(&mut idx, |&a, &b| {
                rb[a as usize].bucket.sse < rb[b as usize].bucket.sse
            });
        }
        let sorted: Vec<RankedShortlistBucket> = idx
            .iter()
            .map(|&i| self.ranked_buckets[i as usize])
            .collect();
        self.ranked_buckets = sorted;
        true
    }

    /// `rank_and_sort_shortlist_buckets`: surrogate encodes the most promising modes of each
    /// examined bucket and sorts the resulting shortlist by error.
    pub fn rank_and_sort_shortlist_buckets(
        &mut self,
        p: &LdrAstcLowlevelBlockEncoderParams,
        pixel_stats: &PixelStats,
        stats: &mut EncodeBlockStats,
    ) -> bool {
        let shortlist_trials = &mut self.trial_surrogates;
        shortlist_trials.clear();

        let num_ranked = self.ranked_buckets.len() as u32;
        let mut num_buckets_to_examine =
            fast_roundf_int(num_ranked as f32 * p.shortlist_buckets_to_examine_fract) as u32;
        num_buckets_to_examine = num_buckets_to_examine
            .max(p.shortlist_buckets_to_examine_min)
            .min(p.shortlist_buckets_to_examine_max);
        num_buckets_to_examine = num_buckets_to_examine.max(1).min(num_ranked);

        let mut best_err_so_far = BIG_FLOAT_VAL;
        let bw = p.block_width;
        let bh = p.block_height;

        for bucket_index in 0..num_buckets_to_examine as usize {
            let rb = &self.ranked_buckets[bucket_index];
            let bucket = &rb.bucket;
            let indices = rb.trial_mode_indices.as_slice();

            if best_err_so_far != BIG_FLOAT_VAL
                && bucket.sse > best_err_so_far * SKIP_IF_BUCKET_WORSE_MULTIPLIER
            {
                continue;
            }
            best_err_so_far = min_f(best_err_so_far, bucket.sse);

            if indices.len() == 1 {
                shortlist_trials.push(TrialSurrogate {
                    trial_mode_index: indices[0] as u32,
                    err: bucket.sse,
                    log_blk: bucket.surrogate_log_blk,
                });
                continue;
            }

            let num_modes_in_bucket = indices.len() as u32;
            let mut num_to_shortlist = fast_roundf_pos_int(
                num_modes_in_bucket as f32 * p.num_similar_modes_in_bucket_to_shortlist_fract,
            ) as u32;
            num_to_shortlist = num_to_shortlist
                .max(p.num_similar_modes_in_bucket_to_shortlist_fract_min)
                .min(p.num_similar_modes_in_bucket_to_shortlist_fract_max);
            num_to_shortlist = num_to_shortlist.max(1).min(num_modes_in_bucket);

            let mut bucket_indices: Vec<u32> = (0..num_modes_in_bucket).collect();

            if num_to_shortlist < num_modes_in_bucket {
                let mut sse_estimates = vec![0.0f32; num_modes_in_bucket as usize];
                let slb = &bucket.surrogate_log_blk;
                let bucket_e_levels = slb.num_endpoint_levels;
                let bucket_w_levels = slb.num_weight_levels;
                let bucket_base_sse = bucket.sse;

                let Some(grid_data) = find_astc_block_grid_data(
                    bw,
                    bh,
                    bucket.grid_width as u32,
                    bucket.grid_height as u32,
                ) else {
                    continue;
                };
                let part_data = if bucket.num_parts == 1 {
                    None
                } else {
                    Some(p.part_data(bucket.num_parts as u32))
                };

                let mut can_use_base_ofs = false;
                if bucket.cem_index as u32 == CEM_LDR_RGB_DIRECT
                    || bucket.cem_index as u32 == CEM_LDR_RGBA_DIRECT
                {
                    let mut max_span_size = 0.0f32;
                    for pi in 0..bucket.num_parts as usize {
                        for c in 0..4 {
                            let span = (slb.endpoints[pi][1][c] - slb.endpoints[pi][0][c]).abs();
                            if max_span_size < span {
                                max_span_size = span;
                            }
                        }
                    }
                    can_use_base_ofs = max_span_size < 0.25;
                }

                let texels_in_part = |pi: usize| -> u32 {
                    match part_data {
                        Some(pd) => {
                            pd.partition_pat_histograms[bucket.unique_seed_index as usize][pi]
                                as u32
                        }
                        None => bw * bh,
                    }
                };

                let sse_est = |e_levels: u32, w_levels: u32| -> ChanMseEst {
                    let mut est = ChanMseEst::default();
                    for pi in 0..bucket.num_parts as usize {
                        let total_texels_in_part = texels_in_part(pi);
                        for c in 0..4 {
                            let span_size =
                                (slb.endpoints[pi][1][c] - slb.endpoints[pi][0][c]).abs();
                            let mut cm = compute_quantized_channel_mse_estimates(
                                if can_use_base_ofs {
                                    (e_levels * 2).min(256)
                                } else {
                                    e_levels
                                },
                                w_levels,
                                span_size,
                                grid_data.weight_gamma,
                                None,
                            );
                            if span_size == 0.0
                                && (slb.endpoints[pi][1][c] == 1.0
                                    || slb.endpoints[pi][1][c] == 0.0)
                            {
                                cm.ep = 0.0;
                                cm.wp = 0.0;
                            }
                            let cw = p.enc_params.comp_weights[c] as f32;
                            est.ep += cm.ep * cw * total_texels_in_part as f32;
                            est.wp += cm.wp * cw * total_texels_in_part as f32;
                        }
                    }
                    est
                };

                let bucket_sse_est = sse_est(bucket_e_levels, bucket_w_levels);

                for (j, &ti) in indices.iter().enumerate() {
                    let tm = &p.trial_modes[ti as usize];
                    let tm_e_levels = get_ise_levels(tm.endpoint_range as u32);
                    let tm_w_levels = get_ise_levels(tm.weight_range as u32);
                    let mut mode_sse_est = sse_est(tm_e_levels, tm_w_levels);
                    if tm_e_levels == bucket_e_levels {
                        mode_sse_est.ep = 0.0;
                    } else {
                        mode_sse_est.ep -= bucket_sse_est.ep;
                        if mode_sse_est.ep < 0.0 {
                            mode_sse_est.ep = 0.0;
                        }
                    }
                    if tm_w_levels == bucket_w_levels {
                        mode_sse_est.wp = 0.0;
                    } else {
                        mode_sse_est.wp -= bucket_sse_est.wp;
                        if mode_sse_est.wp < 0.0 {
                            mode_sse_est.wp = 0.0;
                        }
                    }
                    sse_estimates[j] = bucket_base_sse + mode_sse_est.ep + mode_sse_est.wp;
                }

                indirect_sort(&mut bucket_indices, &sse_estimates);
            }

            for q in 0..num_to_shortlist as usize {
                let j = bucket_indices[q] as usize;
                let trial_mode_index = indices[j] as u32;
                let tm = &p.trial_modes[trial_mode_index as usize];
                let mut s = TrialSurrogate {
                    trial_mode_index,
                    ..Default::default()
                };
                if bucket.num_parts == 1 {
                    s.err = encode_surrogate_trial(
                        bw,
                        bh,
                        pixel_stats,
                        bucket.cem_index as u32,
                        bucket.ccs_index as i32,
                        tm.endpoint_range as u32,
                        tm.weight_range as u32,
                        bucket.grid_width as u32,
                        bucket.grid_height as u32,
                        &mut s.log_blk,
                        p.enc_params,
                        0,
                    );
                } else {
                    let pd = p.part_data(bucket.num_parts as u32);
                    let usi = bucket.unique_seed_index as usize;
                    let part_seed_index = pd.unique_index_to_part_seed[usi] as u32;
                    s.err = encode_surrogate_trial_subsets(
                        bw,
                        bh,
                        pixel_stats,
                        bucket.cem_index as u32,
                        bucket.num_parts as u32,
                        part_seed_index,
                        &pd.partition_pats[usi],
                        tm.endpoint_range as u32,
                        tm.weight_range as u32,
                        bucket.grid_width as u32,
                        bucket.grid_height as u32,
                        &mut s.log_blk,
                        p.enc_params,
                        0,
                    );
                }
                stats.total_surrogate_encodes += 1;
                if bucket.cem_index as u32 == CEM_LDR_RGB_DIRECT
                    || bucket.cem_index as u32 == CEM_LDR_RGBA_DIRECT
                {
                    s.err *= BLUE_CONTRACTION_BASE_OFS_DISCOUNT;
                }
                shortlist_trials.push(s);
            }
        }

        if shortlist_trials.is_empty() {
            return false;
        }

        // shortlist_trials.sort() (std::sort by error).
        let mut idx: Vec<u32> = (0..shortlist_trials.len() as u32).collect();
        std_sort_by(&mut idx, |&a, &b| {
            shortlist_trials[a as usize].err < shortlist_trials[b as usize].err
        });
        let sorted: Vec<TrialSurrogate> =
            idx.iter().map(|&i| shortlist_trials[i as usize]).collect();
        *shortlist_trials = sorted;

        stats.total_shortlist_candidates += shortlist_trials.len() as u32;
        true
    }

    /// `final_polish_encode_from_shortlist`: real encodes of the top shortlist entries,
    /// appended to `out_blocks`.
    pub fn final_polish_encode_from_shortlist(
        &mut self,
        p: &LdrAstcLowlevelBlockEncoderParams,
        pixel_stats: &PixelStats,
        out_blocks: &mut Vec<EncodeBlockOutput>,
        blur_id: u32,
        stats: &mut EncodeBlockStats,
    ) -> bool {
        let shortlist_trials = &self.trial_surrogates;
        let ci = self.block_complexity_index as usize;
        let shortlist_fract = p.final_shortlist_fraction[ci];
        let mut max_shortlist_trials =
            (shortlist_trials.len() as f32 * shortlist_fract).round() as u32;
        max_shortlist_trials = max_shortlist_trials
            .max(p.final_shortlist_min_size[ci])
            .min(p.final_shortlist_max_size[ci]);
        let total_shortlist_trials = max_shortlist_trials
            .max(1)
            .min(shortlist_trials.len() as u32);

        const EARLY_STOP2_SHORTLIST_ITER_INDEX: u32 = 5;
        let bw = p.block_width;
        let bh = p.block_height;
        let flags = p.trial_flags();

        // Returns true if the encoding loop should stop (early out).
        let emit = |log_blk: LogAstcBlock,
                    trial_mode_index: u32,
                    shortlist_iter: u32,
                    out_blocks: &mut Vec<EncodeBlockOutput>| {
            let sse = eval_error(bw, bh, &log_blk, pixel_stats, p.enc_params);
            out_blocks.push(EncodeBlockOutput {
                trial_mode_index: trial_mode_index as i16,
                blur_id: blur_id as u16,
                log_blk,
                packed_dct_plane_data: Default::default(),
                sse,
            });
            if p.early_stop_wpsnr != 0.0 || p.early_stop2_wpsnr != 0.0 {
                let wpsnr =
                    compute_psnr_from_wsse(bw, bh, sse, p.enc_params.get_total_comp_weights());
                if p.early_stop_wpsnr != 0.0 && wpsnr >= p.early_stop_wpsnr {
                    return true;
                }
                if shortlist_iter >= EARLY_STOP2_SHORTLIST_ITER_INDEX
                    && p.early_stop2_wpsnr != 0.0
                    && wpsnr >= p.early_stop2_wpsnr
                {
                    return true;
                }
            }
            false
        };

        for shortlist_iter in 0..total_shortlist_trials {
            let st = &shortlist_trials[shortlist_iter as usize];
            let trial_mode_index = st.trial_mode_index;
            let tm = &p.trial_modes[trial_mode_index as usize];
            let tm_cem = tm.cem as u32;
            let num_parts = tm.num_parts as u32;

            let pat = if num_parts > 1 {
                let pd = p.part_data(num_parts);
                let part_seed_index = st.log_blk.seed_index;
                let unique = pd.part_seed_to_unique_index[(part_seed_index & 1023) as usize];
                Some((part_seed_index, &pd.partition_pats[unique.max(0) as usize]))
            } else {
                None
            };

            let run = |cem: u32, log_blk: &mut LogAstcBlock, tdef: Option<&mut bool>| -> bool {
                match pat {
                    Some((seed, pat)) => encode_trial_subsets(
                        bw,
                        bh,
                        pixel_stats,
                        cem,
                        num_parts,
                        seed,
                        pat,
                        tm.endpoint_range as u32,
                        tm.weight_range as u32,
                        tm.grid_w as u32,
                        tm.grid_h as u32,
                        p.encode_trial_subsets_early_out_thresh,
                        log_blk,
                        p.enc_params,
                        false,
                        flags,
                        tdef,
                    ),
                    None => encode_trial(
                        bw,
                        bh,
                        pixel_stats,
                        cem,
                        tm.ccs_index != -1,
                        tm.ccs_index as i32,
                        tm.endpoint_range as u32,
                        tm.weight_range as u32,
                        tm.grid_w as u32,
                        tm.grid_h as u32,
                        p.encode_trial_early_out_thresh,
                        log_blk,
                        p.enc_params,
                        flags,
                        tdef,
                    ),
                }
            };

            let mut base_ofs_succeeded_flag = false;

            if p.final_encode_try_base_ofs
                && (tm_cem == CEM_LDR_RGB_DIRECT || tm_cem == CEM_LDR_RGBA_DIRECT)
            {
                let base_ofs_cem_index = if tm_cem == CEM_LDR_RGB_DIRECT {
                    CEM_LDR_RGB_BASE_PLUS_OFFSET
                } else {
                    CEM_LDR_RGBA_BASE_PLUS_OFFSET
                };
                let mut alt = LogAstcBlock::default();
                let mut tdef = false;
                let status = run(base_ofs_cem_index, &mut alt, Some(&mut tdef));
                stats.total_full_encodes += 1;
                if status {
                    if emit(alt, trial_mode_index, shortlist_iter, out_blocks) {
                        break;
                    }
                    base_ofs_succeeded_flag = !tdef;
                }
            }

            if p.final_encode_always_try_rgb_direct || !base_ofs_succeeded_flag {
                let mut log_astc_blk = LogAstcBlock::default();
                let status = run(tm_cem, &mut log_astc_blk, None);
                stats.total_full_encodes += 1;
                if !status {
                    return false;
                }
                if emit(log_astc_blk, trial_mode_index, shortlist_iter, out_blocks) {
                    break;
                }
            }
        }
        true
    }

    /// `full_encode`: encodes one block (candidates are appended to `out_blocks`).
    pub fn full_encode(
        &mut self,
        p: &LdrAstcLowlevelBlockEncoderParams,
        pixel_stats: &PixelStats,
        out_blocks: &mut Vec<EncodeBlockOutput>,
        blur_id: u32,
        stats: &mut EncodeBlockStats,
    ) -> bool {
        self.clear();
        self.init(p, pixel_stats)
            && self.partition_triage(p, pixel_stats, stats)
            && self.trivial_triage(p)
            && self.analytic_triage(p, pixel_stats, stats)
            && self.surrogate_encode_shortlist_bucket_representatives(p, pixel_stats, stats)
            && self.prune_shortlist_buckets(p)
            && self.rank_and_sort_shortlist_buckets(p, pixel_stats, stats)
            && self.final_polish_encode_from_shortlist(p, pixel_stats, out_blocks, blur_id, stats)
    }
}
