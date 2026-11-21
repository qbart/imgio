// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! `ldr_astc_block_encode_image_high_level_config` and `configure_encoder_effort_level`
//! (`basisu_astc_ldr_encode.cpp`).

use super::common::CemEncodeParams;

/// `astc_ldr::WEIGHT_REFINER_MAX_PASSES`.
pub(crate) const WEIGHT_REFINER_MAX_PASSES: u32 = 17;

/// `ldr_astc_block_encode_image_high_level_config` (debug fields and the job pool dropped).
#[derive(Clone, Debug)]
pub(crate) struct HighLevelConfig {
    pub block_width: u32,
    pub block_height: u32,

    pub second_superpass_refinement: bool,
    pub second_superpass_fract_to_recompress: f32,

    pub third_superpass_try_neighbors: bool,

    pub base_q: f32,
    pub use_dct: bool,

    pub subsets_enabled: bool,
    pub subsets_edge_filtering: bool,

    pub filter_by_pca_angles_flag: bool,

    pub use_direct_angle_thresh: f32,
    pub use_base_scale_angle_thresh: f32,

    /// Much slower, test on base.
    pub force_all_dual_plane_chan_evals: bool,
    /// DP can be on alpha only, if block has alpha.
    pub disable_rgb_dual_plane: bool,
    pub strong_dp_decorr_thresh_rgb: f32,

    pub use_base_ofs: bool,
    pub use_blue_contraction: bool,

    pub grid_hv_filtering: bool,
    pub low_freq_block_filtering: bool,

    pub superbucket_max_to_retain: [u32; 3],

    pub final_shortlist_fraction: [f32; 3],
    pub final_shortlist_min_size: [u32; 3],
    pub final_shortlist_max_size: [u32; 3],

    pub part2_fraction_to_keep: u32,
    pub part3_fraction_to_keep: u32,
    pub base_parts2: u32,
    pub base_parts3: u32,

    pub early_stop_wpsnr: f32,
    pub early_stop2_wpsnr: f32,

    pub blurring_enabled_p1: bool,
    pub blurring_enabled_p2: bool,

    pub gradient_descent_flag: bool,
    pub polish_weights_flag: bool,
    /// Gradient descent must be enabled too.
    pub qcd_enabled_flag: bool,
    pub bucket_pruning_passes: bool,
    pub encode_trial_early_out_thresh: f32,
    pub encode_trial_subsets_early_out_thresh: f32,

    // 2nd superpass options
    pub base_parts2_p2: u32,
    pub base_parts3_p2: u32,
    pub superbucket_max_to_retain_p2: [u32; 3],
    pub final_shortlist_max_size_p2: [u32; 3],
    pub second_pass_total_weight_refine_passes: u32,
    pub second_pass_force_subsets_enabled: bool,
    pub force_all_dp_chans_p2: bool,
    pub final_encode_always_try_rgb_direct: bool,
    pub filter_by_pca_angles_flag_p2: bool,

    pub try_simplified_latent_configs: bool,

    pub cem_enc_params: CemEncodeParams<'static>,
}

impl Default for HighLevelConfig {
    fn default() -> Self {
        Self {
            block_width: 6,
            block_height: 6,
            second_superpass_refinement: true,
            second_superpass_fract_to_recompress: 0.075,
            third_superpass_try_neighbors: true,
            base_q: 75.0,
            use_dct: false,
            subsets_enabled: true,
            subsets_edge_filtering: true,
            filter_by_pca_angles_flag: true,
            use_direct_angle_thresh: 0.25,
            use_base_scale_angle_thresh: 7.0,
            force_all_dual_plane_chan_evals: false,
            disable_rgb_dual_plane: false,
            strong_dp_decorr_thresh_rgb: 0.998,
            use_base_ofs: true,
            use_blue_contraction: true,
            grid_hv_filtering: true,
            low_freq_block_filtering: true,
            superbucket_max_to_retain: [4, 8, 16],
            final_shortlist_fraction: [0.25, 0.33, 0.5],
            final_shortlist_min_size: [1, 1, 1],
            final_shortlist_max_size: [4096, 4096, 4096],
            part2_fraction_to_keep: 2,
            part3_fraction_to_keep: 2,
            base_parts2: 32,
            base_parts3: 32,
            early_stop_wpsnr: 0.0,
            early_stop2_wpsnr: 0.0,
            blurring_enabled_p1: false,
            blurring_enabled_p2: false,
            gradient_descent_flag: true,
            polish_weights_flag: true,
            qcd_enabled_flag: true,
            bucket_pruning_passes: true,
            encode_trial_early_out_thresh: 0.1,
            encode_trial_subsets_early_out_thresh: 0.1,
            base_parts2_p2: 64,
            base_parts3_p2: 64,
            superbucket_max_to_retain_p2: [16, 32, 256],
            final_shortlist_max_size_p2: [4096, 4096, 4096],
            second_pass_total_weight_refine_passes: WEIGHT_REFINER_MAX_PASSES,
            second_pass_force_subsets_enabled: true,
            force_all_dp_chans_p2: false,
            final_encode_always_try_rgb_direct: false,
            filter_by_pca_angles_flag_p2: true,
            try_simplified_latent_configs: false,
            cem_enc_params: CemEncodeParams::default(),
        }
    }
}

/// `configure_encoder_effort_level` (levels outside 0..=10 act as 3).
pub(crate) fn configure_encoder_effort_level(level: i32, cfg: &mut HighLevelConfig) {
    match level {
        10 => {
            cfg.second_superpass_refinement = true;
            cfg.third_superpass_try_neighbors = true;

            cfg.subsets_enabled = true;
            cfg.use_blue_contraction = true;
            cfg.use_base_ofs = true;

            cfg.encode_trial_early_out_thresh = 0.01;
            cfg.encode_trial_subsets_early_out_thresh = 0.01;

            cfg.force_all_dual_plane_chan_evals = true;
            cfg.filter_by_pca_angles_flag = false;

            cfg.superbucket_max_to_retain = [256, 256, 256];

            cfg.base_parts2 = 128;
            cfg.base_parts3 = 128;
            cfg.part2_fraction_to_keep = 1;
            cfg.part3_fraction_to_keep = 1;

            cfg.final_shortlist_fraction = [1.0, 1.0, 1.0];
            cfg.final_shortlist_max_size = [128, 128, 128];

            // Second superpass
            cfg.second_superpass_fract_to_recompress = 0.075;
            cfg.superbucket_max_to_retain_p2 = [1024, 1024, 1024];
            cfg.final_shortlist_max_size_p2 = [256, 256, 256];
            cfg.base_parts2_p2 = 128;
            cfg.base_parts3_p2 = 128;
            cfg.force_all_dp_chans_p2 = true;
            cfg.filter_by_pca_angles_flag_p2 = false;

            cfg.final_encode_always_try_rgb_direct = true;

            cfg.early_stop_wpsnr = 90.0;
            cfg.early_stop2_wpsnr = 90.0;
            cfg.grid_hv_filtering = false;
            cfg.low_freq_block_filtering = false;

            cfg.cem_enc_params.use_exhaustive_weight_eval = true;
        }
        9 => {
            cfg.second_superpass_refinement = true;
            cfg.third_superpass_try_neighbors = true;

            cfg.subsets_enabled = true;
            cfg.use_blue_contraction = true;
            cfg.use_base_ofs = true;

            cfg.encode_trial_early_out_thresh = 0.01;
            cfg.encode_trial_subsets_early_out_thresh = 0.01;

            cfg.force_all_dual_plane_chan_evals = false;
            cfg.filter_by_pca_angles_flag = true;

            cfg.superbucket_max_to_retain = [16, 32, 64];

            cfg.base_parts2 = 32;
            cfg.base_parts3 = 32;
            cfg.part2_fraction_to_keep = 2;
            cfg.part3_fraction_to_keep = 2;

            cfg.final_shortlist_fraction = [1.0, 1.0, 1.0];
            cfg.final_shortlist_max_size = [16, 32, 64];

            // Second superpass
            cfg.second_superpass_fract_to_recompress = 0.15;
            cfg.superbucket_max_to_retain_p2 = [16, 64, 256];
            cfg.final_shortlist_max_size_p2 = [32, 64, 128];
            cfg.base_parts2_p2 = 64;
            cfg.base_parts3_p2 = 64;
            cfg.force_all_dp_chans_p2 = false;
            cfg.filter_by_pca_angles_flag_p2 = false;

            cfg.final_encode_always_try_rgb_direct = false;

            cfg.early_stop_wpsnr = 75.0;
            cfg.early_stop2_wpsnr = 70.0;

            cfg.cem_enc_params.use_exhaustive_weight_eval = true;
        }
        8 => {
            // old 9
            cfg.second_superpass_refinement = true;
            cfg.third_superpass_try_neighbors = true;

            cfg.subsets_enabled = true;
            cfg.use_blue_contraction = true;
            cfg.use_base_ofs = true;

            cfg.encode_trial_early_out_thresh = 0.01;
            cfg.encode_trial_subsets_early_out_thresh = 0.01;

            cfg.force_all_dual_plane_chan_evals = false;
            cfg.filter_by_pca_angles_flag = true;

            cfg.superbucket_max_to_retain = [8, 16, 32];

            cfg.base_parts2 = 32;
            cfg.base_parts3 = 32;
            cfg.part2_fraction_to_keep = 2;
            cfg.part3_fraction_to_keep = 2;

            cfg.final_shortlist_fraction = [1.0, 1.0, 1.0];
            cfg.final_shortlist_max_size = [4, 12, 24];

            // Second superpass
            cfg.second_superpass_fract_to_recompress = 0.15;
            cfg.superbucket_max_to_retain_p2 = [16, 64, 256];
            cfg.final_shortlist_max_size_p2 = [8, 16, 32];
            cfg.base_parts2_p2 = 64;
            cfg.base_parts3_p2 = 64;
            cfg.force_all_dp_chans_p2 = false;
            cfg.filter_by_pca_angles_flag_p2 = false;

            cfg.final_encode_always_try_rgb_direct = false;

            cfg.early_stop_wpsnr = 75.0;
            cfg.early_stop2_wpsnr = 70.0;

            cfg.cem_enc_params.use_exhaustive_weight_eval = true;
        }
        7 => {
            cfg.second_superpass_refinement = true;
            cfg.third_superpass_try_neighbors = true;

            cfg.subsets_enabled = true;
            cfg.use_blue_contraction = true;
            cfg.use_base_ofs = true;

            cfg.disable_rgb_dual_plane = false;
            cfg.strong_dp_decorr_thresh_rgb = 0.9;

            cfg.force_all_dual_plane_chan_evals = false;
            cfg.filter_by_pca_angles_flag = true;

            cfg.superbucket_max_to_retain = [3, 7, 12];

            cfg.base_parts2 = 12;
            cfg.base_parts3 = 12;
            cfg.part2_fraction_to_keep = 2;
            cfg.part3_fraction_to_keep = 2;

            cfg.final_shortlist_fraction = [1.0, 1.0, 1.0];
            cfg.final_shortlist_max_size = [2, 4, 8];

            cfg.gradient_descent_flag = true;
            cfg.polish_weights_flag = true;
            cfg.qcd_enabled_flag = true;

            cfg.bucket_pruning_passes = false;
            cfg.cem_enc_params.max_ls_passes = 1;

            // Second superpass
            cfg.second_superpass_fract_to_recompress = 0.075;
            cfg.superbucket_max_to_retain_p2 = [4, 16, 32];
            cfg.final_shortlist_max_size_p2 = [4, 16, 32];
            cfg.base_parts2_p2 = 32;
            cfg.base_parts3_p2 = 8;
            cfg.force_all_dp_chans_p2 = false;
            cfg.filter_by_pca_angles_flag_p2 = true;

            cfg.early_stop_wpsnr = 65.0;
            cfg.early_stop2_wpsnr = 60.0;
        }
        6 => {
            cfg.second_superpass_refinement = true;
            cfg.third_superpass_try_neighbors = true;

            cfg.subsets_enabled = true;
            cfg.use_blue_contraction = true;
            cfg.use_base_ofs = true;

            cfg.disable_rgb_dual_plane = false;
            cfg.strong_dp_decorr_thresh_rgb = 0.75;

            cfg.force_all_dual_plane_chan_evals = false;
            cfg.filter_by_pca_angles_flag = true;

            cfg.superbucket_max_to_retain = [2, 5, 10];

            cfg.base_parts2 = 12;
            cfg.base_parts3 = 10;
            cfg.part2_fraction_to_keep = 2;
            cfg.part3_fraction_to_keep = 2;

            cfg.final_shortlist_fraction = [1.0, 1.0, 1.0];
            cfg.final_shortlist_max_size = [1, 4, 8];

            cfg.gradient_descent_flag = true;
            cfg.polish_weights_flag = true;
            cfg.qcd_enabled_flag = true;

            cfg.bucket_pruning_passes = false;
            cfg.cem_enc_params.max_ls_passes = 1;

            // Second superpass
            cfg.second_superpass_fract_to_recompress = 0.075;
            cfg.superbucket_max_to_retain_p2 = [2, 8, 16];
            cfg.final_shortlist_max_size_p2 = [2, 8, 16];
            cfg.base_parts2_p2 = 32;
            cfg.base_parts3_p2 = 8;
            cfg.force_all_dp_chans_p2 = false;
            cfg.filter_by_pca_angles_flag_p2 = true;

            cfg.early_stop_wpsnr = 65.0;
            cfg.early_stop2_wpsnr = 60.0;
        }
        5 | 4 => {
            cfg.second_superpass_refinement = true;
            cfg.third_superpass_try_neighbors = true;

            cfg.subsets_enabled = true;
            cfg.use_blue_contraction = true;
            cfg.use_base_ofs = true;

            cfg.disable_rgb_dual_plane = false;
            cfg.strong_dp_decorr_thresh_rgb = 0.75;

            cfg.force_all_dual_plane_chan_evals = false;
            cfg.filter_by_pca_angles_flag = true;

            cfg.superbucket_max_to_retain = [1, 4, 8];

            if level == 5 {
                cfg.base_parts2 = 12;
                cfg.base_parts3 = 8;
            } else {
                cfg.base_parts2 = 8;
                cfg.base_parts3 = 4;
            }
            cfg.part2_fraction_to_keep = 2;
            cfg.part3_fraction_to_keep = 2;

            cfg.final_shortlist_fraction = [1.0, 1.0, 1.0];
            cfg.final_shortlist_max_size = [1, 4, 8];

            cfg.gradient_descent_flag = true;
            cfg.polish_weights_flag = true;
            cfg.qcd_enabled_flag = false;

            cfg.bucket_pruning_passes = false;
            cfg.cem_enc_params.max_ls_passes = 1;

            // Second superpass
            cfg.second_superpass_fract_to_recompress = 0.075;
            cfg.superbucket_max_to_retain_p2 = [2, 8, 16];
            cfg.final_shortlist_max_size_p2 = [2, 8, 16];
            cfg.base_parts2_p2 = 32;
            cfg.base_parts3_p2 = 8;
            cfg.force_all_dp_chans_p2 = false;
            cfg.filter_by_pca_angles_flag_p2 = true;

            cfg.early_stop_wpsnr = 65.0;
            cfg.early_stop2_wpsnr = 60.0;
        }
        2 => {
            // Level 2+ have subsets and RGB dual-plane enabled
            cfg.second_superpass_refinement = false;
            cfg.third_superpass_try_neighbors = true;

            cfg.subsets_enabled = true;
            cfg.use_blue_contraction = true;
            cfg.use_base_ofs = false;
            cfg.disable_rgb_dual_plane = false;

            cfg.force_all_dual_plane_chan_evals = false;
            cfg.filter_by_pca_angles_flag = true;

            cfg.superbucket_max_to_retain = [1, 2, 3];

            cfg.base_parts2 = 1;
            cfg.base_parts3 = 0;
            cfg.part2_fraction_to_keep = 1;
            cfg.part3_fraction_to_keep = 1;

            cfg.final_shortlist_fraction = [1.0, 1.0, 1.0];
            cfg.final_shortlist_max_size = [1, 2, 3];

            cfg.gradient_descent_flag = false;
            cfg.polish_weights_flag = true;
            cfg.qcd_enabled_flag = false;

            cfg.bucket_pruning_passes = false;
            cfg.cem_enc_params.max_ls_passes = 1;

            // Second superpass
            cfg.second_superpass_fract_to_recompress = 0.04;
            cfg.second_pass_force_subsets_enabled = true;
            cfg.superbucket_max_to_retain_p2 = [1, 2, 8];
            cfg.final_shortlist_max_size_p2 = [1, 2, 8];
            cfg.base_parts2_p2 = 16;
            cfg.base_parts3_p2 = 0;
            cfg.force_all_dp_chans_p2 = false;
            cfg.filter_by_pca_angles_flag_p2 = true;

            cfg.early_stop_wpsnr = 45.0;
            cfg.early_stop2_wpsnr = 40.0;
        }
        1 | 0 => {
            cfg.second_superpass_refinement = false;
            cfg.third_superpass_try_neighbors = false;

            cfg.subsets_enabled = false;
            cfg.use_blue_contraction = true;
            cfg.use_base_ofs = false;
            cfg.disable_rgb_dual_plane = true;

            cfg.force_all_dual_plane_chan_evals = false;
            cfg.filter_by_pca_angles_flag = true;

            cfg.superbucket_max_to_retain = [1, 1, 1];

            cfg.base_parts2 = 0;
            cfg.base_parts3 = 0;
            cfg.part2_fraction_to_keep = 1;
            cfg.part3_fraction_to_keep = 1;

            cfg.final_shortlist_fraction = [1.0, 1.0, 1.0];
            cfg.final_shortlist_max_size = [1, 1, 1];

            cfg.gradient_descent_flag = false;
            cfg.polish_weights_flag = level == 1;
            cfg.qcd_enabled_flag = false;

            cfg.bucket_pruning_passes = false;
            cfg.cem_enc_params.max_ls_passes = 1;

            cfg.early_stop_wpsnr = 45.0;
            cfg.early_stop2_wpsnr = 40.0;
        }
        // default / 3
        _ => {
            cfg.second_superpass_refinement = true;
            cfg.third_superpass_try_neighbors = true;

            cfg.subsets_enabled = true;
            cfg.use_blue_contraction = true;
            cfg.use_base_ofs = false;

            cfg.disable_rgb_dual_plane = false;
            cfg.strong_dp_decorr_thresh_rgb = 0.75;

            cfg.force_all_dual_plane_chan_evals = false;
            cfg.filter_by_pca_angles_flag = true;

            cfg.superbucket_max_to_retain = [1, 4, 8];

            cfg.base_parts2 = 4;
            cfg.base_parts3 = 2;
            cfg.part2_fraction_to_keep = 2;
            cfg.part3_fraction_to_keep = 2;

            cfg.final_shortlist_fraction = [1.0, 1.0, 1.0];
            cfg.final_shortlist_max_size = [1, 4, 8];

            cfg.gradient_descent_flag = true;
            cfg.polish_weights_flag = true;
            cfg.qcd_enabled_flag = false;

            cfg.bucket_pruning_passes = false;
            cfg.cem_enc_params.max_ls_passes = 1;

            // Second superpass
            cfg.second_superpass_fract_to_recompress = 0.075;
            cfg.superbucket_max_to_retain_p2 = [2, 8, 16];
            cfg.final_shortlist_max_size_p2 = [2, 8, 16];
            cfg.base_parts2_p2 = 32;
            cfg.base_parts3_p2 = 8;
            cfg.force_all_dp_chans_p2 = false;
            cfg.filter_by_pca_angles_flag_p2 = true;

            cfg.early_stop_wpsnr = 65.0;
            cfg.early_stop2_wpsnr = 60.0;
        }
    }
}
