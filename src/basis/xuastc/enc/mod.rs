// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! XUASTC LDR / ASTC LDR encoder: port of basisu v2.50 `astc_ldr::compress_image`
//! (`encoder/basisu_astc_ldr_encode.cpp`, `basisu_astc_ldr_fencode.cpp`,
//! `basisu_astc_ldr_common.cpp`), default build (`BASISU_SUPPORT_ASTCENC` = 0).
//!
//! See `docs/spec/xuastc_encode.md`. Module map (C++ source in parentheses):
//! - [`common`] (+ `common_encode`, `common_surrogate`): CEM endpoint/weight encoding, solution
//!   evaluation, surrogates (`basisu_astc_ldr_common.cpp` up to `partition_pattern_vec`).
//! - [`partitions`], [`grid_data`]: partition patterns, LSH map, `partitions_data`,
//!   `astc_block_grid_data`.
//! - [`fencode`], [`astcf`]: the fast "astcf" encoder (`basisu_astc_ldr_fencode.cpp`,
//!   `ldr_astc_block_encode_image_astcf`); [`fast4x4`]: effort 0 4x4 (bc7f based).
//! - [`block_helpers`], [`lowlevel`], [`block_image`], [`effort`]: the full ("basisu") encoder.
//! - [`dct_enc`]: forward weight grid DCT, `code_block_weights`, `apply_weight_grid_dct`.
//! - [`scd`]: SCD deblocking refinement, sharpening, candidate merging, DCT cross check.
//! - [`arith_enc`], [`supercomp`], [`zstd_syntax`]: the XUASTC LDR bitstream writers.
//! - [`rand`], [`containers`]: order-exact `basisu::rand`, `hash_map`, `std::priority_queue`.

// Debug-only C++ helpers are ported but not all reachable from `compress_image`.
#![allow(dead_code)]

pub(crate) mod arith_enc;
pub(crate) mod astcf;
pub(crate) mod block_helpers;
pub(crate) mod block_image;
pub(crate) mod common;
pub(crate) mod common_encode;
pub(crate) mod common_surrogate;
pub(crate) mod containers;
pub(crate) mod dct_enc;
pub(crate) mod effort;
pub(crate) mod fast4x4;
pub(crate) mod fencode;
pub(crate) mod fencode_tables;
pub(crate) mod grid_data;
pub(crate) mod lowlevel;
pub(crate) mod partitions;
pub(crate) mod rand;
pub(crate) mod scd;
pub(crate) mod supercomp;
#[cfg(test)]
mod tests_block;
#[cfg(test)]
mod tests_common;
pub(crate) mod zstd_syntax;

use crate::astc::helpers::LogAstcBlock;
use crate::basis::ll::comp::Image;
use crate::{Error, Result};

/// `basist::astc_ldr_t::xuastc_ldr_syntax`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum XuastcSyntax {
    /// `cFullArith`.
    FullArith = 0,
    /// `cHybridArithZStd`.
    HybridArithZstd = 1,
    /// `cFullZStd` (the `basis_compressor` default).
    #[default]
    FullZstd = 2,
}

pub(crate) const EFFORT_LEVEL_MIN: u32 = 0;
pub(crate) const EFFORT_LEVEL_MAX: u32 = 10;
pub(crate) const EFFORT_LEVEL_DEF: u32 = 3;
pub(crate) const DCT_QUALITY_MIN: i32 = 1;
pub(crate) const DCT_QUALITY_MAX: i32 = 100;

/// `astc_ldr::astc_ldr_encode_config` (C++ defaults; debug fields dropped).
#[derive(Clone, Debug)]
pub(crate) struct AstcLdrEncodeConfig {
    pub astc_block_width: u32,
    pub astc_block_height: u32,
    pub astc_decode_mode_srgb: bool,
    pub compressed_syntax: XuastcSyntax,
    pub effort_level: u32,
    pub dct_quality: f32,
    pub use_dct: bool,
    pub lossy_supercompression: bool,
    pub comp_weights: [u32; 4],
    pub replacement_min_psnr: f32,
    pub psnr_trial_diff_thresh: f32,
    pub psnr_trial_diff_thresh_edge: f32,
    pub replacement_min_psnr_alpha: f32,
    pub psnr_trial_diff_thresh_alpha: f32,
    pub psnr_trial_diff_thresh_edge_alpha: f32,
    pub block_blurring_p1: bool,
    pub block_blurring_p2: bool,
    pub force_disable_subsets: bool,
    pub force_disable_rgb_dual_plane: bool,
    pub use_astcf: bool,
    pub merge_basisu_into_output: bool,
    pub scd_enabled: bool,
    pub scd_will_postfilter: bool,
    pub scd_preserve_chroma: bool,
    pub num_scd_passes: u32,
    pub sharpen_flag: bool,
    pub sharpen_amount: f32,
    pub try_simplified_latent_configs: bool,
}

impl Default for AstcLdrEncodeConfig {
    fn default() -> Self {
        Self {
            astc_block_width: 6,
            astc_block_height: 6,
            astc_decode_mode_srgb: true,
            compressed_syntax: XuastcSyntax::FullArith,
            effort_level: 3,
            dct_quality: 85.0,
            use_dct: false,
            lossy_supercompression: false,
            comp_weights: [1, 1, 1, 1],
            replacement_min_psnr: 35.0,
            psnr_trial_diff_thresh: 1.5,
            psnr_trial_diff_thresh_edge: 1.0,
            replacement_min_psnr_alpha: 38.0,
            psnr_trial_diff_thresh_alpha: 0.75,
            psnr_trial_diff_thresh_edge_alpha: 0.5,
            block_blurring_p1: false,
            block_blurring_p2: false,
            force_disable_subsets: false,
            force_disable_rgb_dual_plane: false,
            use_astcf: false,
            merge_basisu_into_output: false,
            scd_enabled: false,
            scd_will_postfilter: true,
            scd_preserve_chroma: true,
            num_scd_passes: 12,
            sharpen_flag: false,
            sharpen_amount: 4.0,
            try_simplified_latent_configs: false,
        }
    }
}

/// `basist::astc_ldr_t::dct_syms::coeff`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DctCoeff {
    pub num_zeros: u16,
    /// `INT16_MAX` if invalid.
    pub coeff: i16,
}

/// `basist::astc_ldr_t::dct_syms` (encoder side: every field of the C++ struct).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DctSyms {
    pub dc_sym: u32,
    pub num_dc_levels: u32,
    pub coeffs: Vec<DctCoeff>,
    pub max_coeff_mag: u32,
    pub max_zigzag_index: u32,
}

impl DctSyms {
    pub fn clear(&mut self) {
        self.dc_sym = 0;
        self.num_dc_levels = 0;
        self.coeffs.clear();
        self.max_coeff_mag = 0;
        self.max_zigzag_index = 0;
    }
}

// `m_blur_id` values (blur index, or codec identifier).
pub(crate) const BLUR_ID_GAUSSIAN_ALTERNATE: u16 = 1;
pub(crate) const BLUR_ID_GRID_DIM_BASE: u16 = 32;
pub(crate) const BLUR_ID_DC_LATENT_BASE: u16 = 128;
pub(crate) const BLUR_ID_AC_LATENT_BASE: u16 = 138;
pub(crate) const BLUR_ID_BEST_CANDIDATE: u16 = 256;
pub(crate) const BLUR_ID_ASTCENC: u16 = 512;
pub(crate) const BLUR_ID_ASTCF: u16 = 513;
pub(crate) const BLUR_ID_EXP: u16 = 1024;

/// `encode_block_output`.
#[derive(Clone, Debug)]
pub(crate) struct EncodeBlockOutput {
    /// -1 = solid, no trial mode.
    pub trial_mode_index: i16,
    /// Blur index, or codec identifier.
    pub blur_id: u16,
    pub log_blk: LogAstcBlock,
    /// Packed per-plane DCT data.
    pub packed_dct_plane_data: [DctSyms; 2],
    pub sse: u64,
}

impl Default for EncodeBlockOutput {
    fn default() -> Self {
        Self {
            trial_mode_index: -1,
            blur_id: 0,
            log_blk: LogAstcBlock::default(),
            packed_dct_plane_data: Default::default(),
            sse: 0,
        }
    }
}

/// `ldr_astc_block_encode_image_output::block_info` (without `m_pixel_stats`).
#[derive(Clone, Debug, Default)]
pub(crate) struct BlockInfo {
    pub out_blocks: Vec<EncodeBlockOutput>,
    /// Index of the chosen block (typically the best out block by WSSE).
    pub packed_out_block_index: u32,
    pub low_freq_block_flag: bool,
    pub super_strong_edges: bool,
    pub very_strong_edges: bool,
    pub strong_edges: bool,
}

/// `ldr_astc_block_encode_image_output` (block data only; the partition tables and trial mode
/// tables live with whoever needs them).
#[derive(Clone, Debug, Default)]
pub(crate) struct EncodeImageOutput {
    pub blocks_x: u32,
    pub blocks_y: u32,
    /// `m_image_block_info`, row major.
    pub block_info: Vec<BlockInfo>,
    /// `m_packed_phys_blocks`, row major.
    pub packed_phys_blocks: Vec<[u8; 16]>,
}

impl EncodeImageOutput {
    pub fn new(blocks_x: u32, blocks_y: u32) -> Self {
        let n = (blocks_x * blocks_y) as usize;
        Self {
            blocks_x,
            blocks_y,
            block_info: vec![BlockInfo::default(); n],
            packed_phys_blocks: vec![[0; 16]; n],
        }
    }

    pub fn block(&self, bx: u32, by: u32) -> &BlockInfo {
        &self.block_info[(bx + by * self.blocks_x) as usize]
    }
}

/// Result of [`compress_image`].
pub(crate) struct CompressOutput {
    /// The XUASTC LDR bitstream (the `.basis` slice payload for XUASTC formats).
    pub comp_data: Vec<u8>,
    /// The coded logical blocks, row major (packed to physical ASTC for ASTC LDR formats).
    pub coded_blocks: Vec<LogAstcBlock>,
}

/// `MAX_WIDTH` / `MAX_HEIGHT` of `basisu_astc_ldr_encode.cpp`.
const MAX_DIM: u32 = 65535;

/// `astc_ldr::compress_image` (default build: no astcenc). `threads` = basisu job pool total
/// threads (the output does not depend on it).
pub(crate) fn compress_image(
    actual_orig_img: &Image,
    orig_global_cfg: &AstcLdrEncodeConfig,
    threads: u32,
) -> Result<CompressOutput> {
    let mut global_cfg = orig_global_cfg.clone();
    if global_cfg.scd_enabled {
        // SCD is incompatible with lossy supercompression.
        global_cfg.lossy_supercompression = false;
    }

    let sharpened;
    let orig_img = if global_cfg.sharpen_flag && global_cfg.sharpen_amount > 0.0 {
        sharpened = scd::sharpen_image(actual_orig_img, &global_cfg)?;
        &sharpened
    } else {
        actual_orig_img
    };

    let (width, height) = (orig_img.get_width(), orig_img.get_height());
    if !(1..=MAX_DIM).contains(&width) || !(1..=MAX_DIM).contains(&height) {
        return Err(Error::InvalidArgument);
    }
    let (block_width, block_height) = (global_cfg.astc_block_width, global_cfg.astc_block_height);
    if fencode::find_astc_block_size_index(block_width, block_height).is_none() {
        return Err(Error::InvalidArgument);
    }
    let num_blocks_x = width.div_ceil(block_width);
    let num_blocks_y = height.div_ceil(block_height);
    let total_blocks = num_blocks_x * num_blocks_y;

    if !(1.0..=100.0).contains(&global_cfg.dct_quality) {
        return Err(Error::InvalidArgument);
    }
    if global_cfg
        .comp_weights
        .iter()
        .any(|w| !(1..=256).contains(w))
    {
        return Err(Error::InvalidArgument);
    }
    if global_cfg.effort_level > EFFORT_LEVEL_MAX {
        return Err(Error::InvalidArgument);
    }

    let mut enc_cfg = effort::HighLevelConfig {
        block_width,
        block_height,
        use_dct: global_cfg.use_dct,
        base_q: astcf::base_q_from_dct_quality(global_cfg.dct_quality),
        blurring_enabled_p1: global_cfg.block_blurring_p1,
        blurring_enabled_p2: global_cfg.block_blurring_p2,
        try_simplified_latent_configs: global_cfg.try_simplified_latent_configs,
        ..Default::default()
    };
    enc_cfg.cem_enc_params.comp_weights = global_cfg.comp_weights;
    effort::configure_encoder_effort_level(global_cfg.effort_level as i32, &mut enc_cfg);
    if global_cfg.force_disable_subsets {
        enc_cfg.subsets_enabled = false;
        enc_cfg.second_pass_force_subsets_enabled = false;
    }
    if global_cfg.force_disable_rgb_dual_plane {
        enc_cfg.disable_rgb_dual_plane = true;
        enc_cfg.force_all_dp_chans_p2 = false;
    }
    enc_cfg.cem_enc_params.decode_mode_srgb = global_cfg.astc_decode_mode_srgb;

    // Per-encoder candidate limit (`compress_image`).
    let max_candidate_limit = astcf::max_candidate_limit(&global_cfg, total_blocks);

    // Every encoder initializes the same 2/3 subset partition tables (LSH, no VP tree).
    let part_data_p2 = std::sync::Arc::new(partitions::PartitionsData::new(
        2,
        block_width,
        block_height,
        true,
    ));
    let part_data_p3 = std::sync::Arc::new(partitions::PartitionsData::new(
        3,
        block_width,
        block_height,
        true,
    ));

    let run_basisu = |cfg: &effort::HighLevelConfig| -> Result<EncodeImageOutput> {
        if global_cfg.effort_level == 0 && block_width == 4 && block_height == 4 {
            fast4x4::encode_image_fast_4x4(
                orig_img,
                &global_cfg,
                cfg.use_dct,
                cfg.base_q,
                &cfg.cem_enc_params,
                threads,
            )
        } else {
            block_image::ldr_astc_block_encode_image(orig_img, cfg, max_candidate_limit, threads)
        }
    };

    let mut enc_out = if global_cfg.use_astcf {
        let mut out = astcf::encode_image_astcf(
            orig_img,
            block_width,
            block_height,
            &global_cfg,
            enc_cfg.use_dct,
            enc_cfg.base_q,
            &enc_cfg.cem_enc_params,
            max_candidate_limit,
            (part_data_p2.clone(), part_data_p3.clone()),
            threads,
        )?;
        if global_cfg.merge_basisu_into_output {
            let out2 = run_basisu(&enc_cfg)?;
            scd::merge_output_candidates(&mut out, &out2)?;
        }
        out
    } else {
        run_basisu(&enc_cfg)?
    };

    if global_cfg.scd_enabled {
        scd::refine_output_for_deblocking(
            orig_img,
            &global_cfg,
            &enc_cfg,
            &part_data_p2,
            &part_data_p3,
            &mut enc_out,
            threads,
        )?;
    }

    // Sanity check the decoded weight grids.
    if global_cfg.use_dct {
        scd::cross_check_dct(
            block_width,
            block_height,
            num_blocks_x,
            num_blocks_y,
            &enc_out,
            &enc_cfg,
        )?;
    }

    supercomp::supercompress(orig_img, &global_cfg, &enc_out)
}

#[cfg(test)]
pub(crate) mod dump;
