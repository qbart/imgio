// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! `basisu_astc_ldr_common.cpp`: the per-CEM endpoint/weight encoders (`try_cem*`,
//! `encode_cem*`, `weight_refiner`) and the `cem_encode_pixels` dispatcher.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::implicit_saturating_sub,
    clippy::manual_clamp,
    clippy::manual_range_contains,
    clippy::needless_option_as_deref
)]

use super::common::{
    ASTC_LDR_MAX_BLOCK_PIXELS, BIG_FLOAT_VAL, CemEncodeParams, MAX_CEM_ENDPOINT_VALS, PixelStats,
    Vec3F, Vec4F, WEIGHT_REFINER_MAX_PASSES, apply_delta_to_bise_endpoint_val,
    apply_delta_to_bise_weight_val, blue_contract_dec, calc_pca_3d, calc_pca_4d,
    cem_encode_ldr_rgb_or_rgba_base_offset, cem_encode_ldr_rgb_or_rgba_base_scale,
    cem_encode_ldr_rgb_or_rgba_direct, cem8_or_12_used_blue_contraction, clamp3, clamp4, clampf,
    compute_least_squares_endpoints_1d, compute_least_squares_endpoints_2d,
    compute_least_squares_endpoints_3d, compute_least_squares_endpoints_4d, decode_endpoints, dot3,
    dot4, encode_cem0_4_endpoints, eval_solution, eval_solution_dp, get_ls_weights_ise, mul4,
    normalize4, precise_round_bise_endpoint_val, saturate, sub4, used_blue_contraction, v4,
};
use crate::astc::helpers::{
    self as ah, BISE_2_LEVELS, BISE_32_LEVELS, BISE_64_LEVELS, BISE_256_LEVELS,
    CEM_LDR_LUM_ALPHA_DIRECT, CEM_LDR_LUM_DIRECT, CEM_LDR_RGB_BASE_PLUS_OFFSET,
    CEM_LDR_RGB_BASE_SCALE, CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A, CEM_LDR_RGB_DIRECT,
    CEM_LDR_RGBA_BASE_PLUS_OFFSET, CEM_LDR_RGBA_DIRECT,
};
use crate::basis::ll::enc::{ColorRgba, color_distance_linear, indirect_sort};

const MAXP: usize = ASTC_LDR_MAX_BLOCK_PIXELS;
const NV: usize = MAX_CEM_ENDPOINT_VALS;

#[inline]
fn ncem(cem_index: u32) -> usize {
    ah::get_num_cem_values(cem_index) as usize
}

#[inline]
fn to_color(f: &Vec4F) -> ColorRgba {
    let mut c = [0u8; 4];
    for i in 0..4 {
        c[i] = super::common::ftoi((f[i] * 255.0).round()).clamp(0, 255) as u8;
    }
    c
}

/// `refine_cem8_or_12_endpoints`: refine ISE quantized endpoints from float endpoints.
fn refine_cem8_or_12_endpoints(
    cem_index: u32,
    endpoint_ise_range: u32,
    trial_endpoint_vals: &mut [u8],
    low_color_f: &Vec4F,
    high_color_f: &Vec4F,
    endpoints_are_swapped: bool,
) {
    if endpoint_ise_range == BISE_256_LEVELS {
        return;
    }
    let total_comps = if cem_index == CEM_LDR_RGBA_DIRECT {
        4
    } else {
        3
    };
    let total_endpoint_vals = ncem(cem_index);
    let num_levels = ah::get_ise_levels(endpoint_ise_range) as i32;
    let t = ah::dequant_tables().endpoint_tab(endpoint_ise_range);
    let (dq, ise_to_rank, rank_to_ise) = (&t.ise_to_val, &t.ise_to_rank, &t.rank_to_ise);

    let orig_used_bc =
        cem8_or_12_used_blue_contraction(cem_index, trial_endpoint_vals, endpoint_ise_range);
    let mut first_comp = 0;
    let mut refined = [0u8; NV];
    refined[..total_endpoint_vals].copy_from_slice(&trial_endpoint_vals[..total_endpoint_vals]);

    if orig_used_bc {
        for e in 0..2 {
            let mut best_err = BIG_FLOAT_VAL;
            let mut best_vals = [0u8; 3];
            for b_delta in -1..=1 {
                for k in 0..9 {
                    let r_delta = (k % 3) - 1;
                    let g_delta = (k / 3) - 1;
                    let comp_deltas = [r_delta, g_delta, b_delta];
                    let mut trial_vals = [0u8; 3];
                    for c in 0..3 {
                        let enc_val = trial_endpoint_vals[c * 2 + e] as usize;
                        let orig_rank = ise_to_rank[enc_val] as i32;
                        let new_rank = (orig_rank + comp_deltas[c]).clamp(0, num_levels - 1);
                        trial_vals[c] = rank_to_ise[new_rank as usize];
                    }
                    let dec = blue_contract_dec(
                        dq[trial_vals[0] as usize] as i32,
                        dq[trial_vals[1] as usize] as i32,
                        dq[trial_vals[2] as usize] as i32,
                        255,
                    );
                    let mut dec_f = [0f32; 3];
                    for c in 0..3 {
                        dec_f[c] = dec[c] as f32 * (1.0f32 / 255.0);
                    }
                    let desired = if endpoints_are_swapped == (e == 0) {
                        high_color_f
                    } else {
                        low_color_f
                    };
                    let mut trial_err = 0f32;
                    for c in 0..3 {
                        let d = desired[c] - dec_f[c];
                        trial_err += d * d;
                    }
                    if trial_err < best_err {
                        best_err = trial_err;
                        best_vals = trial_vals;
                    }
                }
            }
            for c in 0..3 {
                refined[c * 2 + e] = best_vals[c];
            }
        }
        // just refine A now (if it exists)
        first_comp = 3;
    }

    if first_comp < total_comps {
        for e in 0..2 {
            for c in first_comp..total_comps {
                let idx = c * 2 + e;
                let orig_rank = ise_to_rank[trial_endpoint_vals[idx] as usize] as i32;
                let mut best_rank = orig_rank;
                let mut best_err = BIG_FLOAT_VAL;
                for v_delta in -1..=1 {
                    let new_rank = (orig_rank + v_delta).clamp(0, num_levels - 1);
                    let new_ise = rank_to_ise[new_rank as usize];
                    let dequant_val = dq[new_ise as usize] as f32 * (1.0f32 / 255.0);
                    let orig_val = if endpoints_are_swapped == (e == 0) {
                        high_color_f[c]
                    } else {
                        low_color_f[c]
                    };
                    let err = (dequant_val - orig_val).abs();
                    if err < best_err {
                        best_err = err;
                        best_rank = new_rank;
                    }
                }
                refined[idx] = rank_to_ise[best_rank as usize];
            }
        }
    }

    let refined_used_bc = cem8_or_12_used_blue_contraction(cem_index, &refined, endpoint_ise_range);
    if refined_used_bc == orig_used_bc {
        trial_endpoint_vals[..total_endpoint_vals].copy_from_slice(&refined[..total_endpoint_vals]);
    }
}

/// Builds the 3 degenerate-endpoint fix trials of `try_cem0_or_4` / `try_cem4_dp_a`.
fn cem0_4_fixed(
    t: u32,
    cem_has_alpha: bool,
    endpoint_ise_range: u32,
    trial: &[u8; NV],
    lum_l: f32,
    lum_h: f32,
    a_l: f32,
    a_h: f32,
) -> [u8; NV] {
    let l_delta = if lum_l < lum_h { -1 } else { 1 };
    let a_delta = if a_l < a_h { -1 } else { 1 };
    let mut fixed = *trial;
    let ad =
        |v: u8, d: i32| apply_delta_to_bise_endpoint_val(endpoint_ise_range, v as i32, d) as u8;
    if t & 1 != 0 {
        if trial[0] == trial[1] && lum_l != lum_h {
            fixed[0] = ad(trial[0], l_delta);
        }
        if cem_has_alpha && trial[2] == trial[3] && a_l != a_h {
            fixed[2] = ad(trial[2], a_delta);
        }
    }
    if t & 2 != 0 {
        if trial[0] == trial[1] && lum_l != lum_h {
            fixed[1] = ad(trial[1], -l_delta);
        }
        if cem_has_alpha && trial[2] == trial[3] && a_l != a_h {
            fixed[3] = ad(trial[3], -a_delta);
        }
    }
    fixed
}

fn cem0_4_any_degen(
    cem_has_alpha: bool,
    v: &[u8; NV],
    lum_l: f32,
    lum_h: f32,
    a_l: f32,
    a_h: f32,
) -> bool {
    (v[0] == v[1] && lum_l != lum_h) || (cem_has_alpha && v[2] == v[3] && a_l != a_h)
}

/// `try_cem0_or_4`: direct L/LA, single plane.
fn try_cem0_or_4(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    lum_l: f32,
    lum_h: f32,
    a_l: f32,
    a_h: f32,
    trial_endpoint_vals_out: &mut [u8],
    trial_weight_vals_out: &mut [u8],
    trial_blk_error: &mut u64,
) -> bool {
    let cem_has_alpha = cem_index == CEM_LDR_LUM_ALPHA_DIRECT;
    let nv = ncem(cem_index);
    let n = ps.num_pixels as usize;
    let mut trial = [0u8; NV];
    let mut tw = [0u8; MAXP];
    encode_cem0_4_endpoints(
        cem_index,
        lum_l,
        lum_h,
        a_l,
        a_h,
        endpoint_ise_range,
        &mut trial,
    );
    let mut trial_err = eval_solution(
        ps,
        cem_index,
        &trial,
        endpoint_ise_range,
        &mut tw,
        weight_ise_range,
        enc_params,
    );
    let mut improved_flag = false;
    if trial_err < *trial_blk_error {
        *trial_blk_error = trial_err;
        trial_endpoint_vals_out[..nv].copy_from_slice(&trial[..nv]);
        trial_weight_vals_out[..n].copy_from_slice(&tw[..n]);
        improved_flag = true;
    }
    if cem0_4_any_degen(cem_has_alpha, &trial, lum_l, lum_h, a_l, a_h) {
        for t in 1..=3 {
            let fixed = cem0_4_fixed(
                t,
                cem_has_alpha,
                endpoint_ise_range,
                &trial,
                lum_l,
                lum_h,
                a_l,
                a_h,
            );
            trial_err = eval_solution(
                ps,
                cem_index,
                &fixed,
                endpoint_ise_range,
                &mut tw,
                weight_ise_range,
                enc_params,
            );
            if trial_err < *trial_blk_error {
                *trial_blk_error = trial_err;
                trial_endpoint_vals_out[..nv].copy_from_slice(&fixed[..nv]);
                trial_weight_vals_out[..n].copy_from_slice(&tw[..n]);
                improved_flag = true;
            }
        }
    }
    improved_flag
}

/// `try_cem4_dp_a`: LA direct, dual plane alpha.
fn try_cem4_dp_a(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    lum_l: f32,
    lum_h: f32,
    a_l: f32,
    a_h: f32,
    trial_endpoint_vals_out: &mut [u8],
    trial_weight_vals0_out: &mut [u8],
    trial_weight_vals1_out: &mut [u8],
    trial_blk_error: &mut u64,
) -> bool {
    let cem_has_alpha = cem_index == CEM_LDR_LUM_ALPHA_DIRECT;
    let nv = ncem(cem_index);
    let n = ps.num_pixels as usize;
    let mut trial = [0u8; NV];
    let mut tw0 = [0u8; MAXP];
    let mut tw1 = [0u8; MAXP];
    encode_cem0_4_endpoints(
        cem_index,
        lum_l,
        lum_h,
        a_l,
        a_h,
        endpoint_ise_range,
        &mut trial,
    );
    let mut trial_err = eval_solution_dp(
        ps,
        cem_index,
        3,
        &trial,
        endpoint_ise_range,
        &mut tw0,
        &mut tw1,
        weight_ise_range,
        enc_params,
    );
    let mut improved_flag = false;
    if trial_err < *trial_blk_error {
        *trial_blk_error = trial_err;
        trial_endpoint_vals_out[..nv].copy_from_slice(&trial[..nv]);
        trial_weight_vals0_out[..n].copy_from_slice(&tw0[..n]);
        trial_weight_vals1_out[..n].copy_from_slice(&tw1[..n]);
        improved_flag = true;
    }
    if cem0_4_any_degen(cem_has_alpha, &trial, lum_l, lum_h, a_l, a_h) {
        for t in 1..=3 {
            let fixed = cem0_4_fixed(
                t,
                cem_has_alpha,
                endpoint_ise_range,
                &trial,
                lum_l,
                lum_h,
                a_l,
                a_h,
            );
            trial_err = eval_solution_dp(
                ps,
                cem_index,
                3,
                &fixed,
                endpoint_ise_range,
                &mut tw0,
                &mut tw1,
                weight_ise_range,
                enc_params,
            );
            if trial_err < *trial_blk_error {
                *trial_blk_error = trial_err;
                trial_endpoint_vals_out[..nv].copy_from_slice(&fixed[..nv]);
                trial_weight_vals0_out[..n].copy_from_slice(&tw0[..n]);
                trial_weight_vals1_out[..n].copy_from_slice(&tw1[..n]);
                improved_flag = true;
            }
        }
    }
    improved_flag
}

/// The 3 degenerate-endpoint fix trials of `try_cem8_12(_dp)`; `None` when blue contraction
/// usage changed.
fn cem8_12_fixed(
    t: u32,
    cem_index: u32,
    endpoint_ise_range: u32,
    trial: &[u8; NV],
    low_color: &ColorRgba,
    high_color: &ColorRgba,
    s0: u32,
    s1: u32,
    is_blue_contracted: bool,
) -> Option<[u8; NV]> {
    let num_comps = if cem_index == CEM_LDR_RGB_DIRECT {
        3
    } else {
        4
    };
    let mut fixed = *trial;
    let ad =
        |v: u8, d: i32| apply_delta_to_bise_endpoint_val(endpoint_ise_range, v as i32, d) as u8;
    if t & 1 != 0 {
        for c in 0..num_comps {
            let (li, hi) = (c * 2, c * 2 + 1);
            if trial[li] == trial[hi] && low_color[c] != high_color[c] {
                let delta = if s0 <= s1 { -1 } else { 1 };
                fixed[li] = ad(trial[li], delta);
            }
        }
    }
    if t & 2 != 0 {
        for c in 0..num_comps {
            let (li, hi) = (c * 2, c * 2 + 1);
            if trial[li] == trial[hi] && low_color[c] != high_color[c] {
                let delta = if s0 <= s1 { 1 } else { -1 };
                fixed[hi] = ad(trial[hi], delta);
            }
        }
    }
    if cem8_or_12_used_blue_contraction(cem_index, &fixed, endpoint_ise_range) != is_blue_contracted
    {
        return None;
    }
    Some(fixed)
}

fn cem8_12_sums(cem_index: u32, trial: &[u8; NV], endpoint_ise_range: u32) -> (u32, u32) {
    let (mut dec_l, mut dec_h) = ([0u8; 4], [0u8; 4]);
    decode_endpoints(
        cem_index,
        trial,
        endpoint_ise_range,
        &mut dec_l,
        &mut dec_h,
        None,
    );
    let mut s0 = dec_l.iter().map(|&v| v as u32).sum::<u32>();
    let mut s1 = dec_h.iter().map(|&v| v as u32).sum::<u32>();
    if cem8_or_12_used_blue_contraction(cem_index, trial, endpoint_ise_range) {
        std::mem::swap(&mut s0, &mut s1);
    }
    (s0, s1)
}

/// `try_cem8_12`: direct RGB/RGBA, single plane. Returns true if improved.
fn try_cem8_12(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    low_color_f: &Vec4F,
    high_color_f: &Vec4F,
    trial_endpoint_vals_out: &mut [u8],
    trial_weight_vals_out: &mut [u8],
    trial_blk_error: &mut u64,
    trial_used_blue_contraction: &mut bool,
    try_blue_contract: bool,
    tried_used_blue_contraction: &mut bool,
) -> bool {
    let nv = ncem(cem_index);
    let n = ps.num_pixels as usize;
    let low_color = to_color(low_color_f);
    let high_color = to_color(high_color_f);
    let mut trial = [0u8; NV];
    let mut tw = [0u8; MAXP];
    let res = cem_encode_ldr_rgb_or_rgba_direct(
        cem_index,
        endpoint_ise_range,
        &low_color,
        &high_color,
        &mut trial,
        try_blue_contract,
    );
    *tried_used_blue_contraction = res.is_blue_contracted;
    if endpoint_ise_range < BISE_256_LEVELS {
        refine_cem8_or_12_endpoints(
            cem_index,
            endpoint_ise_range,
            &mut trial,
            low_color_f,
            high_color_f,
            res.endpoints_are_swapped,
        );
    }
    let mut trial_err = eval_solution(
        ps,
        cem_index,
        &trial,
        endpoint_ise_range,
        &mut tw,
        weight_ise_range,
        enc_params,
    );
    let mut improved_flag = false;
    if trial_err < *trial_blk_error {
        *trial_blk_error = trial_err;
        trial_endpoint_vals_out[..nv].copy_from_slice(&trial[..nv]);
        trial_weight_vals_out[..n].copy_from_slice(&tw[..n]);
        *trial_used_blue_contraction = res.is_blue_contracted;
        improved_flag = true;
    }
    if res.any_degen {
        let (s0, s1) = cem8_12_sums(cem_index, &trial, endpoint_ise_range);
        for t in 1..=3 {
            let Some(fixed) = cem8_12_fixed(
                t,
                cem_index,
                endpoint_ise_range,
                &trial,
                &low_color,
                &high_color,
                s0,
                s1,
                res.is_blue_contracted,
            ) else {
                continue;
            };
            trial_err = eval_solution(
                ps,
                cem_index,
                &fixed,
                endpoint_ise_range,
                &mut tw,
                weight_ise_range,
                enc_params,
            );
            if trial_err < *trial_blk_error {
                *trial_blk_error = trial_err;
                trial_endpoint_vals_out[..nv].copy_from_slice(&fixed[..nv]);
                trial_weight_vals_out[..n].copy_from_slice(&tw[..n]);
                *trial_used_blue_contraction = res.is_blue_contracted;
                improved_flag = true;
            }
        }
    }
    improved_flag
}

/// `try_cem8_12_dp`: direct RGB/RGBA, dual plane.
fn try_cem8_12_dp(
    cem_index: u32,
    ccs_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    low_color_f: &Vec4F,
    high_color_f: &Vec4F,
    trial_endpoint_vals_out: &mut [u8],
    trial_weight_vals0_out: &mut [u8],
    trial_weight_vals1_out: &mut [u8],
    trial_blk_error: &mut u64,
    trial_used_blue_contraction: &mut bool,
    try_blue_contract: bool,
    tried_used_blue_contraction: &mut bool,
) -> bool {
    let nv = ncem(cem_index);
    let n = ps.num_pixels as usize;
    let mut improved_flag = false;
    let low_color = to_color(low_color_f);
    let high_color = to_color(high_color_f);
    let mut trial = [0u8; NV];
    let mut tw0 = [0u8; MAXP];
    let mut tw1 = [0u8; MAXP];
    let res = cem_encode_ldr_rgb_or_rgba_direct(
        cem_index,
        endpoint_ise_range,
        &low_color,
        &high_color,
        &mut trial,
        try_blue_contract,
    );
    *tried_used_blue_contraction = res.is_blue_contracted;
    if endpoint_ise_range < BISE_256_LEVELS {
        refine_cem8_or_12_endpoints(
            cem_index,
            endpoint_ise_range,
            &mut trial,
            low_color_f,
            high_color_f,
            res.endpoints_are_swapped,
        );
    }
    let mut trial_err = eval_solution_dp(
        ps,
        cem_index,
        ccs_index,
        &trial,
        endpoint_ise_range,
        &mut tw0,
        &mut tw1,
        weight_ise_range,
        enc_params,
    );
    if trial_err < *trial_blk_error {
        *trial_blk_error = trial_err;
        trial_endpoint_vals_out[..nv].copy_from_slice(&trial[..nv]);
        trial_weight_vals0_out[..n].copy_from_slice(&tw0[..n]);
        trial_weight_vals1_out[..n].copy_from_slice(&tw1[..n]);
        *trial_used_blue_contraction = res.is_blue_contracted;
        improved_flag = true;
    }
    if res.any_degen {
        let (s0, s1) = cem8_12_sums(cem_index, &trial, endpoint_ise_range);
        for t in 1..=3 {
            let Some(fixed) = cem8_12_fixed(
                t,
                cem_index,
                endpoint_ise_range,
                &trial,
                &low_color,
                &high_color,
                s0,
                s1,
                res.is_blue_contracted,
            ) else {
                continue;
            };
            trial_err = eval_solution_dp(
                ps,
                cem_index,
                ccs_index,
                &fixed,
                endpoint_ise_range,
                &mut tw0,
                &mut tw1,
                weight_ise_range,
                enc_params,
            );
            if trial_err < *trial_blk_error {
                *trial_blk_error = trial_err;
                trial_endpoint_vals_out[..nv].copy_from_slice(&fixed[..nv]);
                trial_weight_vals0_out[..n].copy_from_slice(&tw0[..n]);
                trial_weight_vals1_out[..n].copy_from_slice(&tw1[..n]);
                // (C++ does not update trial_used_blue_contraction here)
                improved_flag = true;
            }
        }
    }
    improved_flag
}

/// `compute_endpoint_distance`.
fn compute_endpoint_distance(
    low_color1: &ColorRgba,
    high_color1: &ColorRgba,
    low_color2: &ColorRgba,
    high_color2: &ColorRgba,
) -> u32 {
    let ll = color_distance_linear(low_color2, low_color1, true) as u32;
    let lh = color_distance_linear(low_color2, high_color1, true) as u32;
    let hl = color_distance_linear(high_color2, low_color1, true) as u32;
    let hh = color_distance_linear(high_color2, high_color1, true) as u32;
    let d0 = ll + hh;
    let d1 = lh + hl;
    d0.min(d1)
}

/// `compare_base_ofs_vs_direct_encoding`: true if the direct encoding would be better.
fn compare_base_ofs_vs_direct_encoding(
    cem_index: u32,
    trial_endpoint_vals: &[u8],
    endpoint_ise_range: u32,
    low_color: &ColorRgba,
    high_color: &ColorRgba,
    try_blue_contract: bool,
) -> bool {
    let (mut bl, mut bh) = ([0u8; 4], [0u8; 4]);
    decode_endpoints(
        cem_index,
        trial_endpoint_vals,
        endpoint_ise_range,
        &mut bl,
        &mut bh,
        None,
    );
    let baseofs_best_d = compute_endpoint_distance(&bl, &bh, low_color, high_color);
    if baseofs_best_d == 0 {
        return false;
    }
    let mut direct = [0u8; NV];
    let direct_cem = if cem_index == CEM_LDR_RGB_BASE_PLUS_OFFSET {
        CEM_LDR_RGB_DIRECT
    } else {
        CEM_LDR_RGBA_DIRECT
    };
    let _ = cem_encode_ldr_rgb_or_rgba_direct(
        direct_cem,
        endpoint_ise_range,
        low_color,
        high_color,
        &mut direct,
        try_blue_contract,
    );
    let (mut dl, mut dh) = ([0u8; 4], [0u8; 4]);
    decode_endpoints(
        direct_cem,
        &direct,
        endpoint_ise_range,
        &mut dl,
        &mut dh,
        None,
    );
    let direct_best_d = compute_endpoint_distance(&dl, &dh, low_color, high_color);
    direct_best_d < baseofs_best_d
}

/// `try_cem9_13_sp_or_dp`: base+offset RGB/RGBA, single (`ccs_index` = -1) or dual plane.
/// `try_direct_encoding_flag`: the base+ofs encoding clamped or direct may be better.
fn try_cem9_13_sp_or_dp(
    cem_index: u32,
    ccs_index: i32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    low_color_f: &Vec4F,
    high_color_f: &Vec4F,
    trial_endpoint_vals_out: &mut [u8],
    trial_weight_vals0_out: &mut [u8],
    mut trial_weight_vals1_out: Option<&mut [u8]>,
    trial_blk_error: &mut u64,
    trial_used_blue_contraction: &mut bool,
    try_blue_contract: bool,
    tried_used_blue_contraction: &mut bool,
    try_direct_encoding_flag: &mut bool,
) -> bool {
    let num_comps = if cem_index == CEM_LDR_RGB_BASE_PLUS_OFFSET {
        3
    } else {
        4
    };
    let nv = ncem(cem_index);
    let n = ps.num_pixels as usize;
    let low_color = to_color(low_color_f);
    let high_color = to_color(high_color_f);
    let mut trial = [0u8; NV];
    let mut tw0 = [0u8; MAXP];
    let mut tw1 = [0u8; MAXP];

    let mut res = cem_encode_ldr_rgb_or_rgba_base_offset(
        cem_index,
        endpoint_ise_range,
        &low_color,
        &high_color,
        &mut trial,
        try_blue_contract,
    );
    *tried_used_blue_contraction = res.used_blue_contraction;
    *try_direct_encoding_flag = res.delta_clamped;
    if res.failed_flag {
        return false;
    }
    if !*try_direct_encoding_flag
        && compare_base_ofs_vs_direct_encoding(
            cem_index,
            &trial,
            endpoint_ise_range,
            &low_color,
            &high_color,
            try_blue_contract,
        )
    {
        *try_direct_encoding_flag = true;
    }

    let mut improved_flag = false;

    // Evaluates the current `trial` endpoints and keeps them if better (`check_direct`: the
    // later passes also update the direct encoding flag).
    let mut eval_and_keep = |trial: &[u8; NV],
                             res: &super::common::RgbBaseOffsetRes,
                             check_direct: bool,
                             tw0: &mut [u8; MAXP],
                             tw1: &mut [u8; MAXP],
                             w1_out: Option<&mut [u8]>,
                             improved_flag: &mut bool,
                             try_direct_encoding_flag: &mut bool| {
        let trial_err = if ccs_index == -1 {
            eval_solution(
                ps,
                cem_index,
                trial,
                endpoint_ise_range,
                tw0,
                weight_ise_range,
                enc_params,
            )
        } else {
            eval_solution_dp(
                ps,
                cem_index,
                ccs_index as u32,
                trial,
                endpoint_ise_range,
                tw0,
                tw1,
                weight_ise_range,
                enc_params,
            )
        };
        if trial_err < *trial_blk_error {
            *trial_blk_error = trial_err;
            trial_endpoint_vals_out[..nv].copy_from_slice(&trial[..nv]);
            trial_weight_vals0_out[..n].copy_from_slice(&tw0[..n]);
            if let Some(w1) = w1_out {
                if ccs_index == -1 {
                    w1[..n].fill(0);
                } else {
                    w1[..n].copy_from_slice(&tw1[..n]);
                }
            }
            *trial_used_blue_contraction = res.used_blue_contraction;
            *improved_flag = true;
            if check_direct {
                if res.delta_clamped {
                    *try_direct_encoding_flag = true;
                }
                if !*try_direct_encoding_flag
                    && compare_base_ofs_vs_direct_encoding(
                        cem_index,
                        trial,
                        endpoint_ise_range,
                        &low_color,
                        &high_color,
                        try_blue_contract,
                    )
                {
                    *try_direct_encoding_flag = true;
                }
            }
        }
    };

    eval_and_keep(
        &trial,
        &res,
        false,
        &mut tw0,
        &mut tw1,
        trial_weight_vals1_out.as_deref_mut(),
        &mut improved_flag,
        try_direct_encoding_flag,
    );

    if res.any_degen {
        let (mut dec_l, mut dec_h) = ([0u8; 4], [0u8; 4]);
        decode_endpoints(
            cem_index,
            &trial,
            endpoint_ise_range,
            &mut dec_l,
            &mut dec_h,
            None,
        );
        // The packing in these modes is so complex that we're going to approximate the biasing.
        let num_ise_levels = ah::get_ise_levels(endpoint_ise_range) as i32;
        let vals_per_ise_level = (256 + num_ise_levels - 1) / num_ise_levels;
        for p in 1..=3u32 {
            let (mut tl, mut th) = (low_color, high_color);
            for c in 0..num_comps {
                if low_color[c] == high_color[c] {
                    continue;
                }
                if dec_l[c] != dec_h[c] {
                    continue;
                }
                let delta = if low_color[c] < high_color[c] { -1 } else { 1 };
                if p & 1 != 0 {
                    tl[c] = (tl[c] as i32 + vals_per_ise_level * delta).clamp(0, 255) as u8;
                }
                if p & 2 != 0 {
                    th[c] = (th[c] as i32 + vals_per_ise_level * -delta).clamp(0, 255) as u8;
                }
            }
            res = cem_encode_ldr_rgb_or_rgba_base_offset(
                cem_index,
                endpoint_ise_range,
                &tl,
                &th,
                &mut trial,
                try_blue_contract,
            );
            if res.failed_flag {
                continue;
            }
            eval_and_keep(
                &trial,
                &res,
                true,
                &mut tw0,
                &mut tw1,
                trial_weight_vals1_out.as_deref_mut(),
                &mut improved_flag,
                try_direct_encoding_flag,
            );
        }
    } else {
        // Factor in the quantization introduced into the low (base) color, and apply this to
        // the offset, for gain.
        let (mut dec_l, mut dec_h) = ([0u8; 4], [0u8; 4]);
        decode_endpoints(
            cem_index,
            &trial,
            endpoint_ise_range,
            &mut dec_l,
            &mut dec_h,
            None,
        );
        if res.endpoints_swapped {
            dec_l = low_color; // high color is the quantized base
        } else {
            dec_h = high_color; // low color is the quantized base
        }
        res = cem_encode_ldr_rgb_or_rgba_base_offset(
            cem_index,
            endpoint_ise_range,
            &dec_l,
            &dec_h,
            &mut trial,
            try_blue_contract,
        );
        if !res.failed_flag {
            eval_and_keep(
                &trial,
                &res,
                true,
                &mut tw0,
                &mut tw1,
                trial_weight_vals1_out.as_deref_mut(),
                &mut improved_flag,
                try_direct_encoding_flag,
            );
        }
    }
    improved_flag
}

/// `encode_cem0_4`: L/LA direct, single plane.
fn encode_cem0_4(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    endpoint_vals: &mut [u8],
    weight_vals: &mut [u8],
    mut cur_blk_error: u64,
) -> u64 {
    let cem_has_alpha = cem_index == CEM_LDR_LUM_ALPHA_DIRECT;
    let total_endpoint_vals = ncem(cem_index);
    let total_weights = ps.num_pixels as usize;
    let n = ps.num_pixels as usize;

    let mut lum_l = BIG_FLOAT_VAL;
    let mut lum_h = -BIG_FLOAT_VAL;
    let mut pixel1 = [0f32; MAXP];
    let mut pixel2 = [[0f32; 2]; MAXP];
    for i in 0..n {
        let px = &ps.pixels_f[i];
        let l = (px[0] + px[1] + px[2]) * (1.0f32 / 3.0);
        pixel1[i] = l;
        pixel2[i] = [l, px[3]];
        lum_l = if l < lum_l { l } else { lum_l };
        lum_h = if lum_h < l { l } else { lum_h };
    }
    let a_l = ps.min_f[3];
    let a_h = ps.max_f[3];
    let min_pixel2 = [lum_l, a_l];
    let max_pixel2 = [lum_h, a_h];

    let mut trial_endpoints = [0u8; NV];
    let mut trial_weights = [0u8; MAXP];
    let mut trial_blk_error = u64::MAX;

    try_cem0_or_4(
        cem_index,
        ps,
        enc_params,
        endpoint_ise_range,
        weight_ise_range,
        lum_l,
        lum_h,
        a_l,
        a_h,
        &mut trial_endpoints,
        &mut trial_weights,
        &mut trial_blk_error,
    );
    if trial_blk_error == u64::MAX {
        return cur_blk_error;
    }
    if trial_blk_error < cur_blk_error {
        cur_blk_error = trial_blk_error;
        endpoint_vals[..total_endpoint_vals]
            .copy_from_slice(&trial_endpoints[..total_endpoint_vals]);
        weight_vals[..total_weights].copy_from_slice(&trial_weights[..total_weights]);
    }

    const NUM_LS_OPT_PASSES: u32 = 3;
    let lsw = get_ls_weights_ise(weight_ise_range);
    for _pass in 0..NUM_LS_OPT_PASSES {
        let mut xl = [lum_l, a_l];
        let mut xh = [lum_h, a_h];
        let ls_res = if cem_has_alpha {
            compute_least_squares_endpoints_2d(
                ps.num_pixels,
                &trial_weights,
                lsw,
                &mut xl,
                &mut xh,
                &pixel2,
                &min_pixel2,
                &max_pixel2,
            )
        } else {
            let (mut l0, mut h0) = (xl[0], xh[0]);
            let r = compute_least_squares_endpoints_1d(
                ps.num_pixels,
                &trial_weights,
                lsw,
                &mut l0,
                &mut h0,
                &pixel1,
                lum_l,
                lum_h,
            );
            xl[0] = l0;
            xh[0] = h0;
            r
        };
        if !ls_res {
            break;
        }
        try_cem0_or_4(
            cem_index,
            ps,
            enc_params,
            endpoint_ise_range,
            weight_ise_range,
            xl[0],
            xh[0],
            xl[1],
            xh[1],
            &mut trial_endpoints,
            &mut trial_weights,
            &mut trial_blk_error,
        );
        if trial_blk_error >= cur_blk_error {
            break;
        }
        cur_blk_error = trial_blk_error;
        endpoint_vals[..total_endpoint_vals]
            .copy_from_slice(&trial_endpoints[..total_endpoint_vals]);
        weight_vals[..total_weights].copy_from_slice(&trial_weights[..total_weights]);
    }
    cur_blk_error
}

/// `encode_cem4_dp_a`: LA direct, dual plane alpha.
fn encode_cem4_dp_a(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    endpoint_vals: &mut [u8],
    weight_vals0: &mut [u8],
    weight_vals1: &mut [u8],
    mut cur_blk_error: u64,
) -> u64 {
    let total_endpoint_vals = ncem(cem_index);
    let total_weights = ps.num_pixels as usize;
    let mut alpha_vals = [0f32; MAXP];
    for i in 0..total_weights {
        alpha_vals[i] = ps.pixels_f[i][3];
    }

    // First get plane0's low/high (lum)
    let mut lum_endpoints = [0u8; NV];
    let mut lum_weights0 = [0u8; MAXP];
    let lum_blk_error = encode_cem0_4(
        CEM_LDR_LUM_DIRECT,
        ps,
        enc_params,
        endpoint_ise_range,
        weight_ise_range,
        &mut lum_endpoints,
        &mut lum_weights0,
        u64::MAX,
    );
    if lum_blk_error == u64::MAX {
        return cur_blk_error;
    }
    let dq = &ah::dequant_tables()
        .endpoint_tab(endpoint_ise_range)
        .ise_to_val;
    let lum_l = dq[lum_endpoints[0] as usize] as f32 * (1.0f32 / 255.0);
    let lum_h = dq[lum_endpoints[1] as usize] as f32 * (1.0f32 / 255.0);
    let a_l = ps.min_f[3];
    let a_h = ps.max_f[3];

    let mut trial_endpoints = [0u8; NV];
    let mut trial_weights0 = [0u8; MAXP];
    let mut trial_weights1 = [0u8; MAXP];
    let mut trial_blk_error = u64::MAX;

    let mut did_improve = try_cem4_dp_a(
        cem_index,
        ps,
        enc_params,
        endpoint_ise_range,
        weight_ise_range,
        lum_l,
        lum_h,
        a_l,
        a_h,
        &mut trial_endpoints,
        &mut trial_weights0,
        &mut trial_weights1,
        &mut trial_blk_error,
    );
    if !did_improve {
        return cur_blk_error;
    }
    if trial_blk_error < cur_blk_error {
        cur_blk_error = trial_blk_error;
        endpoint_vals[..total_endpoint_vals]
            .copy_from_slice(&trial_endpoints[..total_endpoint_vals]);
        weight_vals0[..total_weights].copy_from_slice(&trial_weights0[..total_weights]);
        weight_vals1[..total_weights].copy_from_slice(&trial_weights1[..total_weights]);
    }

    const NUM_LS_OPT_PASSES: u32 = 3;
    let lsw = get_ls_weights_ise(weight_ise_range);
    for _pass in 0..NUM_LS_OPT_PASSES {
        let (mut xl, mut xh) = (ps.min_f[3], ps.max_f[3]);
        let ls_res = compute_least_squares_endpoints_1d(
            ps.num_pixels,
            &trial_weights1,
            lsw,
            &mut xl,
            &mut xh,
            &alpha_vals,
            ps.min_f[3],
            ps.max_f[3],
        );
        if !ls_res {
            break;
        }
        did_improve = try_cem4_dp_a(
            cem_index,
            ps,
            enc_params,
            endpoint_ise_range,
            weight_ise_range,
            lum_l,
            lum_h,
            xl,
            xh,
            &mut trial_endpoints,
            &mut trial_weights0,
            &mut trial_weights1,
            &mut trial_blk_error,
        );
        if !did_improve {
            break;
        }
        cur_blk_error = trial_blk_error;
        endpoint_vals[..total_endpoint_vals]
            .copy_from_slice(&trial_endpoints[..total_endpoint_vals]);
        weight_vals0[..total_weights].copy_from_slice(&trial_weights0[..total_weights]);
        weight_vals1[..total_weights].copy_from_slice(&trial_weights1[..total_weights]);
    }
    cur_blk_error
}

/// `weight_refiner`: perturbs weight ranks to seed alternative least squares solutions.
pub(crate) struct WeightRefiner {
    total_pixels: u32,
    weight_ise_range: u32,
    num_weight_levels: u32,
    /// Ranks, not ISE.
    start_weights: [u8; MAXP],
    min_weight: u32,
    max_weight: u32,
    sum_weight: u32,
}

impl Default for WeightRefiner {
    fn default() -> Self {
        Self {
            total_pixels: 0,
            weight_ise_range: 0,
            num_weight_levels: 0,
            start_weights: [0; MAXP],
            min_weight: 0,
            max_weight: 0,
            sum_weight: 0,
        }
    }
}

impl WeightRefiner {
    /// `init`.
    pub fn init(&mut self, weight_ise_range: u32, total_pixels: u32, initial_ise_weights: &[u8]) {
        self.weight_ise_range = weight_ise_range;
        self.total_pixels = total_pixels;
        self.num_weight_levels = ah::get_ise_levels(weight_ise_range);
        let ise_to_rank = &ah::dequant_tables()
            .weight_tab(weight_ise_range)
            .ise_to_rank;
        for i in 0..total_pixels as usize {
            self.start_weights[i] = ise_to_rank
                .get(initial_ise_weights[i] as usize)
                .copied()
                .unwrap_or(0);
        }
        self.min_weight = u32::MAX;
        self.max_weight = 0;
        self.sum_weight = 0;
        for i in 0..total_pixels as usize {
            let w = self.start_weights[i] as u32;
            self.sum_weight += w;
            self.min_weight = self.min_weight.min(w);
            self.max_weight = self.max_weight.max(w);
        }
    }

    /// `refine`.
    pub fn refine(&self, pass_index: u32, trial_ise_weights: &mut [u8]) {
        let rank_to_ise = &ah::dequant_tables()
            .weight_tab(self.weight_ise_range)
            .rank_to_ise;
        let n = self.total_pixels as usize;
        let nl = self.num_weight_levels;
        let (minw, maxw) = (self.min_weight, self.max_weight);
        let mwri = nl as i32 - 1;
        let lerp_case = |ly: i32, hy: i32, out: &mut [u8]| {
            for i in 0..n {
                let f = (mwri as f32 * (self.start_weights[i] as f32 - ly as f32)
                    / (hy as f32 - ly as f32)
                    + 0.5)
                    .floor();
                let s = clampf(f, 0.0, mwri as f32) as i32;
                out[i] = rank_to_ise[s as usize];
            }
        };
        let mid_case = |scale: f32, mode: u32, out: &mut [u8]| {
            let mid_weight = self.sum_weight as f32 / self.total_pixels as f32;
            for i in 0..n {
                let v = self.start_weights[i] as i32;
                let scaled = (v as f32 - mid_weight) * scale + (nl as f32 * 0.5);
                let fv = match mode {
                    1 if (v as f32) >= mid_weight => v as f32,
                    2 if (v as f32) < mid_weight => v as f32,
                    _ => scaled,
                };
                let v = super::common::ftoi(fv.round()).clamp(0, nl as i32 - 1);
                out[i] = rank_to_ise[v as usize];
            }
        };
        match pass_index {
            0..=2 | 7..=9 | 15 | 16 => {
                for i in 0..n {
                    let mut v = self.start_weights[i] as u32;
                    match pass_index {
                        0 => {
                            if v == minw && v < nl - 1 {
                                v += 1;
                            }
                        }
                        1 => {
                            if v == maxw && v > 0 {
                                v -= 1;
                            }
                        }
                        2 => {
                            if v == minw && v < nl - 1 {
                                v += 1;
                            } else if v == maxw && v > 0 {
                                v -= 1;
                            }
                        }
                        7 => {
                            if v == minw && v < nl - 1 {
                                v += 1;
                                if v < nl - 1 {
                                    v += 1;
                                }
                            }
                        }
                        8 => {
                            if v == maxw && v > 0 {
                                v -= 1;
                                if v > 0 {
                                    v -= 1;
                                }
                            }
                        }
                        9 => {
                            if v == minw && v < nl - 1 {
                                v += 1;
                                if v < nl - 1 {
                                    v += 1;
                                }
                            } else if v == maxw && v > 0 {
                                v -= 1;
                                if v > 0 {
                                    v -= 1;
                                }
                            }
                        }
                        15 => {
                            if v < nl - 1 {
                                v += 1;
                            }
                        }
                        _ => {
                            if v != 0 {
                                v -= 1;
                            }
                        }
                    }
                    trial_ise_weights[i] = rank_to_ise[v as usize];
                }
            }
            3 => lerp_case(-1, mwri + 1, trial_ise_weights),
            4 => lerp_case(-2, mwri + 2, trial_ise_weights),
            5 => lerp_case(-1, mwri + 2, trial_ise_weights),
            6 => lerp_case(-2, mwri + 1, trial_ise_weights),
            10 => mid_case(0.8, 0, trial_ise_weights),
            11 => mid_case(0.9, 0, trial_ise_weights),
            12 => mid_case(1.1, 0, trial_ise_weights),
            // 13: only weights below the mean are scaled
            13 => mid_case(0.8, 1, trial_ise_weights),
            // 14: only weights at or above the mean are scaled
            14 => mid_case(0.8, 2, trial_ise_weights),
            _ => trial_ise_weights[..n].fill(0),
        }
    }
}

/// One `try_cem9_13_sp_or_dp` / `try_cem8_12` round of `encode_cem8_12_9_13(_dp)`: tries with
/// the requested blue contraction, then (if it was used) without it.
struct Rgb812Ctx<'a, 'p> {
    cem_index: u32,
    ccs_index: i32,
    ps: &'a PixelStats,
    enc_params: &'a CemEncodeParams<'p>,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    cem_is_base_offset: bool,
    use_blue_contraction: bool,
}

struct Rgb812Trial {
    endpoints: [u8; NV],
    weights0: [u8; MAXP],
    weights1: [u8; MAXP],
    error: u64,
    used_blue_contraction: bool,
    tried_used_blue_contraction: bool,
}

impl Rgb812Ctx<'_, '_> {
    fn try_once(
        &self,
        t: &mut Rgb812Trial,
        l: &Vec4F,
        h: &Vec4F,
        try_bc: bool,
        direct_flag: &mut Option<&mut bool>,
    ) {
        if self.cem_is_base_offset {
            let mut try_direct = false;
            let w1 = if self.ccs_index >= 0 {
                Some(&mut t.weights1[..])
            } else {
                None
            };
            try_cem9_13_sp_or_dp(
                self.cem_index,
                self.ccs_index,
                self.ps,
                self.enc_params,
                self.endpoint_ise_range,
                self.weight_ise_range,
                l,
                h,
                &mut t.endpoints,
                &mut t.weights0,
                w1,
                &mut t.error,
                &mut t.used_blue_contraction,
                try_bc,
                &mut t.tried_used_blue_contraction,
                &mut try_direct,
            );
            if try_direct {
                if let Some(f) = direct_flag.as_deref_mut() {
                    *f = true;
                }
            }
        } else if self.ccs_index >= 0 {
            try_cem8_12_dp(
                self.cem_index,
                self.ccs_index as u32,
                self.ps,
                self.enc_params,
                self.endpoint_ise_range,
                self.weight_ise_range,
                l,
                h,
                &mut t.endpoints,
                &mut t.weights0,
                &mut t.weights1,
                &mut t.error,
                &mut t.used_blue_contraction,
                try_bc,
                &mut t.tried_used_blue_contraction,
            );
        } else {
            try_cem8_12(
                self.cem_index,
                self.ps,
                self.enc_params,
                self.endpoint_ise_range,
                self.weight_ise_range,
                l,
                h,
                &mut t.endpoints,
                &mut t.weights0,
                &mut t.error,
                &mut t.used_blue_contraction,
                try_bc,
                &mut t.tried_used_blue_contraction,
            );
        }
    }

    fn try_both(
        &self,
        t: &mut Rgb812Trial,
        l: &Vec4F,
        h: &Vec4F,
        direct_flag: &mut Option<&mut bool>,
    ) {
        self.try_once(t, l, h, self.use_blue_contraction, direct_flag);
        if t.tried_used_blue_contraction {
            // Try without blue contraction for a minor gain.
            self.try_once(t, l, h, false, direct_flag);
        }
    }
}

/// `encode_cem8_12_9_13`: RGB/RGBA direct or base+offset, single plane.
fn encode_cem8_12_9_13(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    endpoint_vals: &mut [u8],
    weight_vals: &mut [u8],
    mut cur_blk_error: u64,
    use_blue_contraction: bool,
    mut try_direct_encoding_flag: Option<&mut bool>,
) -> u64 {
    if let Some(f) = try_direct_encoding_flag.as_deref_mut() {
        *f = false;
    }
    let cem_has_alpha =
        cem_index == CEM_LDR_RGBA_DIRECT || cem_index == CEM_LDR_RGBA_BASE_PLUS_OFFSET;
    let cem_is_base_offset =
        cem_index == CEM_LDR_RGB_BASE_PLUS_OFFSET || cem_index == CEM_LDR_RGBA_BASE_PLUS_OFFSET;
    let total_endpoint_vals = ncem(cem_index);
    let total_weights = ps.num_pixels as usize;
    let n = total_weights;

    let mut best_l = BIG_FLOAT_VAL;
    let mut best_h = -BIG_FLOAT_VAL;
    for c in 0..n {
        let px = sub4(&ps.pixels_f[c], &ps.mean_f);
        let p = if cem_has_alpha {
            dot4(&px, &ps.mean_rel_axis4)
        } else {
            dot3(&px, &ps.mean_rel_axis3)
        };
        if p < best_l {
            best_l = p;
        }
        if p > best_h {
            best_h = p;
        }
    }
    let axis = if cem_has_alpha {
        ps.mean_rel_axis4
    } else {
        v4(&ps.mean_rel_axis3)
    };
    let mut low_color_f = super::common::add4(&mul4(&axis, best_l), &ps.mean_f);
    let mut high_color_f = super::common::add4(&mul4(&axis, best_h), &ps.mean_f);
    clamp4(&mut low_color_f, 0.0, 1.0);
    clamp4(&mut high_color_f, 0.0, 1.0);

    let ctx = Rgb812Ctx {
        cem_index,
        ccs_index: -1,
        ps,
        enc_params,
        endpoint_ise_range,
        weight_ise_range,
        cem_is_base_offset,
        use_blue_contraction,
    };
    let mut t = Rgb812Trial {
        endpoints: [0; NV],
        weights0: [0; MAXP],
        weights1: [0; MAXP],
        error: u64::MAX,
        used_blue_contraction: false,
        tried_used_blue_contraction: false,
    };

    ctx.try_both(
        &mut t,
        &low_color_f,
        &high_color_f,
        &mut try_direct_encoding_flag,
    );

    if t.error == u64::MAX {
        return cur_blk_error;
    }
    if t.error < cur_blk_error {
        cur_blk_error = t.error;
        endpoint_vals[..total_endpoint_vals].copy_from_slice(&t.endpoints[..total_endpoint_vals]);
        weight_vals[..total_weights].copy_from_slice(&t.weights0[..total_weights]);
    }

    let lsw = get_ls_weights_ise(weight_ise_range);
    let ls = |w: &[u8], xl: &mut Vec4F, xh: &mut Vec4F| {
        if cem_has_alpha {
            compute_least_squares_endpoints_4d(
                ps.num_pixels,
                w,
                lsw,
                xl,
                xh,
                &ps.pixels_f,
                &ps.min_f,
                &ps.max_f,
            )
        } else {
            compute_least_squares_endpoints_3d(
                ps.num_pixels,
                w,
                lsw,
                xl,
                xh,
                &ps.pixels_f,
                &ps.min_f,
                &ps.max_f,
            )
        }
    };

    for _pass in 0..enc_params.max_ls_passes {
        let (mut xl, mut xh) = ([0f32; 4], [0f32; 4]);
        if !ls(&t.weights0, &mut xl, &mut xh) {
            break;
        }
        ctx.try_both(&mut t, &xl, &xh, &mut try_direct_encoding_flag);
        if t.error >= cur_blk_error {
            break;
        }
        cur_blk_error = t.error;
        endpoint_vals[..total_endpoint_vals].copy_from_slice(&t.endpoints[..total_endpoint_vals]);
        weight_vals[..total_weights].copy_from_slice(&t.weights0[..total_weights]);
    }

    if enc_params.total_weight_refine_passes != 0
        && weight_ise_range != BISE_2_LEVELS
        && weight_ise_range != BISE_64_LEVELS
    {
        let mut refiner = WeightRefiner::default();
        refiner.init(weight_ise_range, ps.num_pixels, weight_vals);
        for pass in 0..enc_params.total_weight_refine_passes {
            refiner.refine(pass, &mut t.weights0);
            let (mut xl, mut xh) = ([0f32; 4], [0f32; 4]);
            if !ls(&t.weights0, &mut xl, &mut xh) {
                continue;
            }
            ctx.try_both(&mut t, &xl, &xh, &mut try_direct_encoding_flag);
            if t.error < cur_blk_error {
                cur_blk_error = t.error;
                endpoint_vals[..total_endpoint_vals]
                    .copy_from_slice(&t.endpoints[..total_endpoint_vals]);
                weight_vals[..total_weights].copy_from_slice(&t.weights0[..total_weights]);
            }
        }
    }

    const N: usize = 4;
    if enc_params.worst_weight_nudging_flag
        && n > N
        && weight_ise_range != BISE_2_LEVELS
        && weight_ise_range != BISE_64_LEVELS
    {
        const NUM_NUDGING_PASSES: u32 = 1;
        for _pass in 0..NUM_NUDGING_PASSES {
            let (mut l, mut h) = ([0u8; 4], [0u8; 4]);
            decode_endpoints(
                cem_index,
                endpoint_vals,
                endpoint_ise_range,
                &mut l,
                &mut h,
                None,
            );
            let mut dir: Vec4F = [
                (h[0] as i32 - l[0] as i32) as f32,
                (h[1] as i32 - l[1] as i32) as f32,
                (h[2] as i32 - l[2] as i32) as f32,
                if cem_has_alpha {
                    (h[3] as i32 - l[3] as i32) as f32
                } else {
                    0.0
                },
            ];
            normalize4(&mut dir);

            let mut errs = [0f32; MAXP];
            let mut delta_dots = [0f32; MAXP];
            for i in 0..n {
                let ofs = sub4(&ps.pixels_f[i], &ps.mean_f);
                let proj = dot4(&dir, &ofs);
                let proj_vec = super::common::add4(&ps.mean_f, &mul4(&dir, proj));
                let delta_vec = sub4(&ps.pixels_f[i], &proj_vec);
                delta_dots[i] = dot4(&dir, &delta_vec);
                errs[i] = if cem_has_alpha {
                    dot4(&delta_vec, &delta_vec)
                } else {
                    dot3(&delta_vec, &delta_vec)
                };
            }
            let mut errs_indices = [0u32; MAXP];
            indirect_sort(&mut errs_indices[..n], &errs[..n]);

            t.weights0[..total_weights].copy_from_slice(&weight_vals[..total_weights]);
            for i in 0..N {
                let idx = errs_indices[n - 1 - i] as usize;
                let delta_to_apply = if delta_dots[idx] > 0.0 { 1 } else { -1 };
                t.weights0[idx] = apply_delta_to_bise_weight_val(
                    weight_ise_range,
                    t.weights0[idx] as i32,
                    delta_to_apply,
                ) as u8;
            }
            let (mut xl, mut xh) = ([0f32; 4], [0f32; 4]);
            if !ls(&t.weights0, &mut xl, &mut xh) {
                break;
            }
            ctx.try_both(&mut t, &xl, &xh, &mut try_direct_encoding_flag);
            if t.error < cur_blk_error {
                cur_blk_error = t.error;
                endpoint_vals[..total_endpoint_vals]
                    .copy_from_slice(&t.endpoints[..total_endpoint_vals]);
                weight_vals[..total_weights].copy_from_slice(&t.weights0[..total_weights]);
            } else {
                break;
            }
        }
    }

    if enc_params.endpoint_refinement_flag {
        let num_comps = if cem_has_alpha { 4 } else { 3 };
        for c in 0..num_comps {
            let mut base = [0u8; NV];
            base[..total_endpoint_vals].copy_from_slice(&endpoint_vals[..total_endpoint_vals]);
            for dl in -1..=1 {
                for dh in -1..=1 {
                    if dl == 0 && dh == 0 {
                        continue;
                    }
                    t.endpoints[..total_endpoint_vals]
                        .copy_from_slice(&base[..total_endpoint_vals]);
                    t.endpoints[c * 2] = apply_delta_to_bise_endpoint_val(
                        endpoint_ise_range,
                        t.endpoints[c * 2] as i32,
                        dl,
                    ) as u8;
                    t.endpoints[c * 2 + 1] = apply_delta_to_bise_endpoint_val(
                        endpoint_ise_range,
                        t.endpoints[c * 2 + 1] as i32,
                        dh,
                    ) as u8;
                    if !use_blue_contraction
                        && used_blue_contraction(cem_index, &t.endpoints, endpoint_ise_range)
                    {
                        continue;
                    }
                    t.error = eval_solution(
                        ps,
                        cem_index,
                        &t.endpoints,
                        endpoint_ise_range,
                        &mut t.weights0,
                        weight_ise_range,
                        enc_params,
                    );
                    if t.error < cur_blk_error {
                        cur_blk_error = t.error;
                        endpoint_vals[..total_endpoint_vals]
                            .copy_from_slice(&t.endpoints[..total_endpoint_vals]);
                        weight_vals[..total_weights].copy_from_slice(&t.weights0[..total_weights]);
                    }
                }
            }
        }
    }

    cur_blk_error
}

/// `encode_cem8_12_9_13_dp`: RGB/RGBA direct or base+offset, dual plane.
fn encode_cem8_12_9_13_dp(
    cem_index: u32,
    ccs_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    endpoint_vals: &mut [u8],
    weight_vals0: &mut [u8],
    weight_vals1: &mut [u8],
    mut cur_blk_error: u64,
    use_blue_contraction: bool,
    mut try_direct_encoding_flag: Option<&mut bool>,
) -> u64 {
    if let Some(f) = try_direct_encoding_flag.as_deref_mut() {
        *f = false;
    }
    let (cem_has_alpha, cem_is_base_offset) = match cem_index {
        CEM_LDR_RGB_DIRECT => (false, false),
        CEM_LDR_RGBA_DIRECT => (true, false),
        CEM_LDR_RGB_BASE_PLUS_OFFSET => (false, true),
        CEM_LDR_RGBA_BASE_PLUS_OFFSET => (true, true),
        // C++ returns `false` (0) here.
        _ => return 0,
    };
    let ccs = ccs_index as usize;
    let total_endpoint_vals = ncem(cem_index);
    let total_weights = ps.num_pixels as usize;
    let n = total_weights;

    // Remove influence of the 2nd plane's values, recalc principle axis on other values.
    let mut flattened = [[0f32; 4]; MAXP];
    for i in 0..n {
        flattened[i] = ps.pixels_f[i];
        flattened[i][ccs] = 0.0;
        if !cem_has_alpha {
            flattened[i][3] = 0.0;
        }
    }
    let mut flattened_mean = ps.mean_f;
    flattened_mean[ccs] = 0.0;
    if !cem_has_alpha {
        flattened_mean[3] = 0.0;
    }
    let flattened_axis = if !cem_has_alpha {
        calc_pca_3d(&flattened[..n], &flattened_mean)
    } else {
        calc_pca_4d(&flattened[..n], &flattened_mean)
    };

    let mut best_l = BIG_FLOAT_VAL;
    let mut best_h = -BIG_FLOAT_VAL;
    for c in 0..n {
        let px = sub4(&flattened[c], &flattened_mean);
        let p = dot4(&px, &flattened_axis);
        if p < best_l {
            best_l = p;
        }
        if p > best_h {
            best_h = p;
        }
    }
    let mut low_color_f = super::common::add4(&flattened_mean, &mul4(&flattened_axis, best_l));
    let mut high_color_f = super::common::add4(&flattened_mean, &mul4(&flattened_axis, best_h));
    clamp4(&mut low_color_f, 0.0, 1.0);
    clamp4(&mut high_color_f, 0.0, 1.0);
    low_color_f[ccs] = ps.min_f[ccs];
    high_color_f[ccs] = ps.max_f[ccs];

    let ctx = Rgb812Ctx {
        cem_index,
        ccs_index: ccs_index as i32,
        ps,
        enc_params,
        endpoint_ise_range,
        weight_ise_range,
        cem_is_base_offset,
        use_blue_contraction,
    };
    let mut t = Rgb812Trial {
        endpoints: [0; NV],
        weights0: [0; MAXP],
        weights1: [0; MAXP],
        error: u64::MAX,
        used_blue_contraction: false,
        tried_used_blue_contraction: false,
    };

    ctx.try_both(
        &mut t,
        &low_color_f,
        &high_color_f,
        &mut try_direct_encoding_flag,
    );
    if t.error == u64::MAX {
        return cur_blk_error;
    }

    let keep = |t: &Rgb812Trial, ev: &mut [u8], w0: &mut [u8], w1: &mut [u8]| {
        ev[..total_endpoint_vals].copy_from_slice(&t.endpoints[..total_endpoint_vals]);
        w0[..total_weights].copy_from_slice(&t.weights0[..total_weights]);
        w1[..total_weights].copy_from_slice(&t.weights1[..total_weights]);
    };

    if t.error < cur_blk_error {
        cur_blk_error = t.error;
        keep(&t, endpoint_vals, weight_vals0, weight_vals1);
    }

    let mut flattened_min_f = ps.min_f;
    flattened_min_f[ccs] = 0.0;
    let mut flattened_max_f = ps.max_f;
    flattened_max_f[ccs] = 0.0;
    let lsw = get_ls_weights_ise(weight_ise_range);

    // Plane 0 solve, with the ccs channel taken from the current decoded endpoints.
    let solve_plane0 = |t: &Rgb812Trial, xl: &mut Vec4F, xh: &mut Vec4F| -> bool {
        if !compute_least_squares_endpoints_4d(
            ps.num_pixels,
            &t.weights0,
            lsw,
            xl,
            xh,
            &flattened,
            &flattened_min_f,
            &flattened_max_f,
        ) {
            return false;
        }
        let (mut dl, mut dh) = ([0u8; 4], [0u8; 4]);
        decode_endpoints(
            cem_index,
            &t.endpoints,
            endpoint_ise_range,
            &mut dl,
            &mut dh,
            None,
        );
        xl[ccs] = dl[ccs] as f32 * (1.0f32 / 255.0);
        xh[ccs] = dh[ccs] as f32 * (1.0f32 / 255.0);
        true
    };

    for _pass in 0..enc_params.max_ls_passes {
        let (mut xl, mut xh) = ([0f32; 4], [0f32; 4]);
        if !solve_plane0(&t, &mut xl, &mut xh) {
            break;
        }
        ctx.try_both(&mut t, &xl, &xh, &mut try_direct_encoding_flag);
        if t.error >= cur_blk_error {
            break;
        }
        cur_blk_error = t.error;
        keep(&t, endpoint_vals, weight_vals0, weight_vals1);
    }

    let ccs_bounds_min = ps.min_f[ccs];
    let ccs_bounds_max = ps.max_f[ccs];
    let mut ccs_vals = [0f32; MAXP];

    // Plane 1 solve: other channels from the current decoded endpoints.
    let solve_plane1 =
        |t: &Rgb812Trial, ccs_vals: &[f32], vl: &mut Vec4F, vh: &mut Vec4F| -> bool {
            let (mut xl, mut xh) = (0f32, 0f32);
            if !compute_least_squares_endpoints_1d(
                ps.num_pixels,
                &t.weights1,
                lsw,
                &mut xl,
                &mut xh,
                ccs_vals,
                ccs_bounds_min,
                ccs_bounds_max,
            ) {
                return false;
            }
            let (mut dl, mut dh) = ([0u8; 4], [0u8; 4]);
            decode_endpoints(
                cem_index,
                &t.endpoints,
                endpoint_ise_range,
                &mut dl,
                &mut dh,
                None,
            );
            for c in 0..4 {
                if c == ccs {
                    vl[c] = xl;
                    vh[c] = xh;
                } else {
                    vl[c] = dl[c] as f32 * (1.0f32 / 255.0);
                    vh[c] = dh[c] as f32 * (1.0f32 / 255.0);
                }
            }
            true
        };

    if ccs_bounds_min != ccs_bounds_max {
        for i in 0..n {
            ccs_vals[i] = ps.pixels_f[i][ccs];
        }
        for _pass in 0..enc_params.max_ls_passes {
            let (mut vl, mut vh) = ([0f32; 4], [0f32; 4]);
            if !solve_plane1(&t, &ccs_vals, &mut vl, &mut vh) {
                break;
            }
            ctx.try_both(&mut t, &vl, &vh, &mut try_direct_encoding_flag);
            if t.error >= cur_blk_error {
                break;
            }
            cur_blk_error = t.error;
            keep(&t, endpoint_vals, weight_vals0, weight_vals1);
        }
    }

    if enc_params.total_weight_refine_passes != 0
        && weight_ise_range != BISE_2_LEVELS
        && weight_ise_range != BISE_64_LEVELS
    {
        let mut refiner = WeightRefiner::default();
        refiner.init(weight_ise_range, ps.num_pixels, weight_vals0);
        for pass in 0..enc_params.total_weight_refine_passes {
            refiner.refine(pass, &mut t.weights0);
            let (mut xl, mut xh) = ([0f32; 4], [0f32; 4]);
            if !solve_plane0(&t, &mut xl, &mut xh) {
                break;
            }
            ctx.try_both(&mut t, &xl, &xh, &mut try_direct_encoding_flag);
            if t.error >= cur_blk_error {
                continue;
            }
            cur_blk_error = t.error;
            keep(&t, endpoint_vals, weight_vals0, weight_vals1);
        }

        if ccs_bounds_min != ccs_bounds_max {
            refiner.init(weight_ise_range, ps.num_pixels, weight_vals1);
            for pass in 0..WEIGHT_REFINER_MAX_PASSES {
                refiner.refine(pass, &mut t.weights1);
                let (mut vl, mut vh) = ([0f32; 4], [0f32; 4]);
                if !solve_plane1(&t, &ccs_vals, &mut vl, &mut vh) {
                    break;
                }
                ctx.try_both(&mut t, &vl, &vh, &mut try_direct_encoding_flag);
                if t.error >= cur_blk_error {
                    continue;
                }
                cur_blk_error = t.error;
                keep(&t, endpoint_vals, weight_vals0, weight_vals1);
            }
        }
    }

    cur_blk_error
}

/// `try_cem6_10`: RGB/RGBA base+scale, single plane. Returns true if improved.
fn try_cem6_10(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    scale: f32,
    low_a_f: f32,
    high_color_f: &Vec4F,
    trial_endpoint_vals_out: &mut [u8],
    trial_weight_vals_out: &mut [u8],
    trial_blk_error: &mut u64,
) -> bool {
    let nv = ncem(cem_index);
    let n = ps.num_pixels as usize;
    let mut trial = [0u8; NV];
    let mut tw = [0u8; MAXP];
    cem_encode_ldr_rgb_or_rgba_base_scale(
        cem_index,
        endpoint_ise_range,
        scale,
        low_a_f,
        high_color_f,
        &mut trial,
    );
    let mut trial_err = eval_solution(
        ps,
        cem_index,
        &trial,
        endpoint_ise_range,
        &mut tw,
        weight_ise_range,
        enc_params,
    );
    let mut improved_flag = false;
    if trial_err < *trial_blk_error {
        *trial_blk_error = trial_err;
        trial_endpoint_vals_out[..nv].copy_from_slice(&trial[..nv]);
        trial_weight_vals_out[..n].copy_from_slice(&tw[..n]);
        improved_flag = true;
    }
    for delta in [-1, 1] {
        let mut fixed = trial;
        fixed[3] =
            apply_delta_to_bise_endpoint_val(endpoint_ise_range, fixed[3] as i32, delta) as u8;
        trial_err = eval_solution(
            ps,
            cem_index,
            &fixed,
            endpoint_ise_range,
            &mut tw,
            weight_ise_range,
            enc_params,
        );
        if trial_err < *trial_blk_error {
            *trial_blk_error = trial_err;
            trial_endpoint_vals_out[..nv].copy_from_slice(&fixed[..nv]);
            trial_weight_vals_out[..n].copy_from_slice(&tw[..n]);
            improved_flag = true;
        }
    }
    improved_flag
}

/// `try_cem6_10_dp`: RGB/RGBA base+scale, dual plane.
fn try_cem6_10_dp(
    cem_index: u32,
    ccs_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    scale: f32,
    low_a_f: f32,
    high_color_f: &Vec4F,
    trial_endpoint_vals_out: &mut [u8],
    trial_weight_vals0_out: &mut [u8],
    trial_weight_vals1_out: &mut [u8],
    trial_blk_error: &mut u64,
) -> bool {
    let nv = ncem(cem_index);
    let n = ps.num_pixels as usize;
    let mut trial = [0u8; NV];
    let mut tw0 = [0u8; MAXP];
    let mut tw1 = [0u8; MAXP];
    cem_encode_ldr_rgb_or_rgba_base_scale(
        cem_index,
        endpoint_ise_range,
        scale,
        low_a_f,
        high_color_f,
        &mut trial,
    );
    let mut trial_err = eval_solution_dp(
        ps,
        cem_index,
        ccs_index,
        &trial,
        endpoint_ise_range,
        &mut tw0,
        &mut tw1,
        weight_ise_range,
        enc_params,
    );
    let mut improved_flag = false;
    if trial_err < *trial_blk_error {
        *trial_blk_error = trial_err;
        trial_endpoint_vals_out[..nv].copy_from_slice(&trial[..nv]);
        trial_weight_vals0_out[..n].copy_from_slice(&tw0[..n]);
        trial_weight_vals1_out[..n].copy_from_slice(&tw1[..n]);
        improved_flag = true;
    }
    for delta in [-1, 1] {
        let mut fixed = trial;
        fixed[3] =
            apply_delta_to_bise_endpoint_val(endpoint_ise_range, fixed[3] as i32, delta) as u8;
        trial_err = eval_solution_dp(
            ps,
            cem_index,
            ccs_index,
            &fixed,
            endpoint_ise_range,
            &mut tw0,
            &mut tw1,
            weight_ise_range,
            enc_params,
        );
        if trial_err < *trial_blk_error {
            *trial_blk_error = trial_err;
            trial_endpoint_vals_out[..nv].copy_from_slice(&fixed[..nv]);
            trial_weight_vals0_out[..n].copy_from_slice(&tw0[..n]);
            trial_weight_vals1_out[..n].copy_from_slice(&tw1[..n]);
            improved_flag = true;
        }
    }
    improved_flag
}

/// Dequantized [0,64] weight of an ISE (or raw, for `BISE_64_LEVELS`) weight.
#[inline]
fn weight_to_raw(weight_ise_range: u32, w: u8) -> u32 {
    if weight_ise_range == BISE_64_LEVELS {
        w as u32
    } else {
        // invalid on raw weights: C++ uses get_weight_tab(min(BISE_32_LEVELS, range))
        ah::dequant_tables()
            .weight_tab(weight_ise_range.min(BISE_32_LEVELS))
            .ise_to_val[w as usize] as u32
    }
}

/// `encode_cem6_10`: RGB/RGBA base+scale, single plane.
fn encode_cem6_10(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    endpoint_vals: &mut [u8],
    weight_vals: &mut [u8],
    mut cur_blk_error: u64,
) -> u64 {
    let cem_has_alpha = cem_index == CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A;
    let total_endpoint_vals = ncem(cem_index);
    let total_weights = ps.num_pixels as usize;
    let n = total_weights;
    let axis = &ps.zero_rel_axis3;

    let mut best_l = BIG_FLOAT_VAL;
    let mut best_h = -BIG_FLOAT_VAL;
    for c in 0..n {
        let p = dot3(&ps.pixels_f[c], axis);
        if p < best_l {
            best_l = p;
        }
        if p > best_h {
            best_h = p;
        }
    }

    const MAX_S: f32 = 255.0 / 256.0;
    const EPS: f32 = 1e-6;

    let mut trial_blk_error = u64::MAX;
    let mut trial_endpoints = [0u8; NV];
    let mut trial_weights = [0u8; MAXP];
    let mut best_blk_endpoints = [0u8; NV];
    let mut best_blk_weights = [0u8; MAXP];

    let mut low3: Vec3F = [best_l * axis[0], best_l * axis[1], best_l * axis[2]];
    clamp3(&mut low3, 0.0, 1.0);
    let mut high3: Vec3F = [best_h * axis[0], best_h * axis[1], best_h * axis[2]];
    clamp3(&mut high3, 0.0, 1.0);

    let mut scale = MAX_S;
    let d = dot3(&low3, &high3);
    let nrm = dot3(&high3, &high3);
    if nrm > 0.0 {
        scale = saturate(d / nrm);
    }
    scale = if scale < MAX_S { scale } else { MAX_S };

    let low_a = ps.min_f[3];
    let high_color_f: Vec4F = [high3[0], high3[1], high3[2], ps.max_f[3]];

    try_cem6_10(
        cem_index,
        ps,
        enc_params,
        endpoint_ise_range,
        weight_ise_range,
        scale,
        low_a,
        &high_color_f,
        &mut trial_endpoints,
        &mut trial_weights,
        &mut trial_blk_error,
    );
    let mut best_blk_error = trial_blk_error;
    best_blk_endpoints[..total_endpoint_vals]
        .copy_from_slice(&trial_endpoints[..total_endpoint_vals]);
    best_blk_weights[..total_weights].copy_from_slice(&trial_weights[..total_weights]);

    const NUM_PASSES: u32 = 2;
    for _pass in 0..NUM_PASSES {
        let (mut actual_l, mut actual_h) = ([0u8; 4], [0u8; 4]);
        let mut actual_scale = 0f32;
        decode_endpoints(
            cem_index,
            &trial_endpoints,
            endpoint_ise_range,
            &mut actual_l,
            &mut actual_h,
            Some(&mut actual_scale),
        );
        let s255 = 1.0f32 / 255.0;
        let actual_high_f: Vec3F = [
            actual_h[0] as f32 * s255,
            actual_h[1] as f32 * s255,
            actual_h[2] as f32 * s255,
        ];

        let mut pa = [0f32; 3];
        let mut pb = [0f32; 3];
        let (mut a, mut b, mut c) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let px = &ps.pixels_f[i];
            let iw = weight_to_raw(weight_ise_range, trial_weights[i]);
            let tt = iw as f32 * (1.0f32 / 64.0);
            let bi = tt;
            let ai = 1.0 - tt;
            for k in 0..3 {
                pa[k] += px[k] * ai;
                pb[k] += px[k] * bi;
            }
            a += ai * ai;
            b += ai * bi;
            c += bi * bi;
        }

        let mut new_high = actual_high_f;
        let mut new_scale = actual_scale;
        let mut h2 = dot3(&actual_high_f, &actual_high_f);
        if h2 > EPS && a > EPS {
            new_scale = (dot3(&pa, &actual_high_f) / h2 - b) / a;
            new_scale = clampf(new_scale, 0.0, MAX_S);
        }
        let den = a * new_scale * new_scale + 2.0 * b * new_scale + c;
        if den > EPS {
            for k in 0..3 {
                new_high[k] = (pb[k] + pa[k] * new_scale) / den;
            }
        }
        h2 = dot3(&new_high, &new_high);
        if h2 > EPS && a > EPS {
            new_scale = (dot3(&pa, &new_high) / h2 - b) / a;
            new_scale = clampf(new_scale, 0.0, MAX_S);
        }

        try_cem6_10(
            cem_index,
            ps,
            enc_params,
            endpoint_ise_range,
            weight_ise_range,
            new_scale,
            actual_l[3] as f32 * s255,
            &[
                new_high[0],
                new_high[1],
                new_high[2],
                actual_h[3] as f32 * s255,
            ],
            &mut trial_endpoints,
            &mut trial_weights,
            &mut trial_blk_error,
        );
        if trial_blk_error >= best_blk_error {
            break;
        }
        best_blk_error = trial_blk_error;
        best_blk_endpoints[..total_endpoint_vals]
            .copy_from_slice(&trial_endpoints[..total_endpoint_vals]);
        best_blk_weights[..total_weights].copy_from_slice(&trial_weights[..total_weights]);
    }

    if cem_has_alpha {
        // Try to refine low a/high given the current selectors.
        let bounds_min = ps.min_f[3];
        let bounds_max = ps.max_f[3];
        if bounds_min != bounds_max {
            let mut a_vals = [0f32; MAXP];
            for i in 0..n {
                a_vals[i] = ps.pixels_f[i][3];
            }
            const TOTAL_PASSES: u32 = 1;
            for _pass in 0..TOTAL_PASSES {
                let (mut xl, mut xh) = (0f32, 0f32);
                if !compute_least_squares_endpoints_1d(
                    ps.num_pixels,
                    &best_blk_weights,
                    get_ls_weights_ise(weight_ise_range),
                    &mut xl,
                    &mut xh,
                    &a_vals,
                    bounds_min,
                    bounds_max,
                ) {
                    break;
                }
                let (mut actual_l, mut actual_h) = ([0u8; 4], [0u8; 4]);
                let mut actual_scale = 0f32;
                decode_endpoints(
                    cem_index,
                    &trial_endpoints,
                    endpoint_ise_range,
                    &mut actual_l,
                    &mut actual_h,
                    Some(&mut actual_scale),
                );
                // NOTE: C++ passes the unscaled [0,255] high RGB here.
                try_cem6_10(
                    cem_index,
                    ps,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    actual_scale,
                    xl,
                    &[
                        actual_h[0] as f32,
                        actual_h[1] as f32,
                        actual_h[2] as f32,
                        xh,
                    ],
                    &mut trial_endpoints,
                    &mut trial_weights,
                    &mut trial_blk_error,
                );
                if trial_blk_error < best_blk_error {
                    best_blk_error = trial_blk_error;
                    best_blk_endpoints[..total_endpoint_vals]
                        .copy_from_slice(&trial_endpoints[..total_endpoint_vals]);
                    best_blk_weights[..total_weights]
                        .copy_from_slice(&trial_weights[..total_weights]);
                } else {
                    break;
                }
            }
        }
    }

    if best_blk_error < cur_blk_error {
        cur_blk_error = best_blk_error;
        // (C++ copies the trial arrays, which always hold the best solution here)
        endpoint_vals[..total_endpoint_vals]
            .copy_from_slice(&trial_endpoints[..total_endpoint_vals]);
        weight_vals[..total_weights].copy_from_slice(&trial_weights[..total_weights]);
    }
    cur_blk_error
}

/// `encode_cem10_dp_a`: RGBA base+scale, dual plane alpha (ccs_index 3).
fn encode_cem10_dp_a(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    endpoint_vals: &mut [u8],
    weight_vals0: &mut [u8],
    weight_vals1: &mut [u8],
    mut cur_blk_error: u64,
) -> u64 {
    const NUM_MODE10_ENDPOINTS: usize = 6;
    let n = ps.num_pixels as usize;
    // RGB uses plane0, alpha plane1. So solve RGB first.
    let mut rgba_endpoint_vals = [0u8; NV];
    let mut rgb_weight_vals = [0u8; MAXP];
    let mut a_weight_vals = [0u8; MAXP];
    let rgb_blk_error = encode_cem6_10(
        CEM_LDR_RGB_BASE_SCALE,
        ps,
        enc_params,
        endpoint_ise_range,
        weight_ise_range,
        &mut rgba_endpoint_vals,
        &mut rgb_weight_vals,
        u64::MAX,
    );
    if rgb_blk_error == u64::MAX {
        return cur_blk_error;
    }
    let q = &ah::dequant_tables()
        .endpoint_tab(endpoint_ise_range)
        .val_to_ise;
    rgba_endpoint_vals[4] = q[ps.min[3] as usize];
    rgba_endpoint_vals[5] = q[ps.max[3] as usize];

    let rgba_blk_error = eval_solution_dp(
        ps,
        cem_index,
        3,
        &rgba_endpoint_vals,
        endpoint_ise_range,
        &mut rgb_weight_vals,
        &mut a_weight_vals,
        weight_ise_range,
        enc_params,
    );
    if rgba_blk_error < cur_blk_error {
        cur_blk_error = rgba_blk_error;
        endpoint_vals[..NUM_MODE10_ENDPOINTS]
            .copy_from_slice(&rgba_endpoint_vals[..NUM_MODE10_ENDPOINTS]);
        weight_vals0[..n].copy_from_slice(&rgb_weight_vals[..n]);
        weight_vals1[..n].copy_from_slice(&a_weight_vals[..n]);
        if cur_blk_error == 0 {
            return cur_blk_error;
        }
    }

    let bounds_min = ps.min_f[3];
    let bounds_max = ps.max_f[3];
    if bounds_min != bounds_max {
        let mut a_vals = [0f32; MAXP];
        for i in 0..n {
            a_vals[i] = ps.pixels_f[i][3];
        }
        const TOTAL_PASSES: u32 = 2;
        let mut trial_ev = [0u8; NV];
        let mut trial_rgb_w = [0u8; MAXP];
        let mut trial_a_w = [0u8; MAXP];
        for pass in 0..TOTAL_PASSES {
            let (mut xl, mut xh) = (0f32, 0f32);
            let sel: &[u8] = if pass != 0 {
                &trial_a_w
            } else {
                &a_weight_vals
            };
            if !compute_least_squares_endpoints_1d(
                ps.num_pixels,
                sel,
                get_ls_weights_ise(weight_ise_range),
                &mut xl,
                &mut xh,
                &a_vals,
                bounds_min,
                bounds_max,
            ) {
                break;
            }
            trial_ev[..NUM_MODE10_ENDPOINTS]
                .copy_from_slice(&rgba_endpoint_vals[..NUM_MODE10_ENDPOINTS]);
            trial_ev[4] = precise_round_bise_endpoint_val(xl, endpoint_ise_range);
            trial_ev[5] = precise_round_bise_endpoint_val(xh, endpoint_ise_range);
            let trial_err = eval_solution_dp(
                ps,
                cem_index,
                3,
                &trial_ev,
                endpoint_ise_range,
                &mut trial_rgb_w,
                &mut trial_a_w,
                weight_ise_range,
                enc_params,
            );
            if trial_err < cur_blk_error {
                cur_blk_error = trial_err;
                endpoint_vals[..NUM_MODE10_ENDPOINTS]
                    .copy_from_slice(&trial_ev[..NUM_MODE10_ENDPOINTS]);
                weight_vals0[..n].copy_from_slice(&trial_rgb_w[..n]);
                weight_vals1[..n].copy_from_slice(&trial_a_w[..n]);
            } else {
                break;
            }
        }
    }
    cur_blk_error
}

/// `encode_cem6_10_dp_rgb`: RGB/RGBA base+scale, dual plane R, G or B.
fn encode_cem6_10_dp_rgb(
    cem_index: u32,
    ccs_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    endpoint_vals: &mut [u8],
    weight_vals0: &mut [u8],
    weight_vals1: &mut [u8],
    mut cur_blk_error: u64,
) -> u64 {
    const NUM_MODE10_ENDPOINTS: usize = 6;
    let n = ps.num_pixels as usize;
    let ccs = (ccs_index as usize).min(2);
    // First solve using a single plane, then introduce the other plane's weights.
    let mut sp_endpoint_vals = [0u8; NV];
    let mut sp_weight_vals = [0u8; MAXP];
    let _ = encode_cem6_10(
        cem_index,
        ps,
        enc_params,
        endpoint_ise_range,
        weight_ise_range,
        &mut sp_endpoint_vals,
        &mut sp_weight_vals,
        u64::MAX,
    );

    // Now compute both plane's weights using the initial H/s values
    let mut tw0 = [0u8; MAXP];
    let mut tw1 = [0u8; MAXP];
    let dp_blk_error = eval_solution_dp(
        ps,
        cem_index,
        ccs_index,
        &sp_endpoint_vals,
        endpoint_ise_range,
        &mut tw0,
        &mut tw1,
        weight_ise_range,
        enc_params,
    );
    if dp_blk_error < cur_blk_error {
        cur_blk_error = dp_blk_error;
        endpoint_vals[..NUM_MODE10_ENDPOINTS]
            .copy_from_slice(&sp_endpoint_vals[..NUM_MODE10_ENDPOINTS]);
        weight_vals0[..n].copy_from_slice(&tw0[..n]);
        weight_vals1[..n].copy_from_slice(&tw1[..n]);
        if cur_blk_error == 0 {
            return cur_blk_error;
        }
    }

    // Compute refined H/s values using the current weights.
    const MAX_S: f32 = 255.0 / 256.0;
    const EPS: f32 = 1e-6;
    let mut pa = [0f32; 3];
    let mut pb = [0f32; 3];
    let mut a = [0f32; 3];
    let mut b = [0f32; 3];
    let mut c = [0f32; 3];
    for i in 0..n {
        let w0 = weight_to_raw(weight_ise_range, tw0[i]) as f32 * (1.0f32 / 64.0);
        let w1 = weight_to_raw(weight_ise_range, tw1[i]) as f32 * (1.0f32 / 64.0);
        let mut w = [w0, w0, w0];
        w[ccs] = w1;
        let p = &ps.pixels_f[i];
        for k in 0..3 {
            let aa = 1.0 - w[k];
            let bb = w[k];
            pa[k] += aa * p[k];
            pb[k] += bb * p[k];
            a[k] += aa * aa;
            b[k] += aa * bb;
            c[k] += bb * bb;
        }
    }

    let (mut actual_l, mut actual_h) = ([0u8; 4], [0u8; 4]);
    let mut actual_scale = 0f32;
    decode_endpoints(
        cem_index,
        &sp_endpoint_vals,
        endpoint_ise_range,
        &mut actual_l,
        &mut actual_h,
        Some(&mut actual_scale),
    );
    let s255 = 1.0f32 / 255.0;
    let h: Vec3F = [
        actual_h[0] as f32 * s255,
        actual_h[1] as f32 * s255,
        actual_h[2] as f32 * s255,
    ];
    let s1 = h[0] * pa[0] + h[1] * pa[1] + h[2] * pa[2];
    let (mut s2, mut s3) = (0f32, 0f32);
    for k in 0..3 {
        let h2 = h[k] * h[k];
        s2 += h2 * a[k];
        s3 += h2 * b[k];
    }
    let mut new_s = actual_scale;
    if s2 > EPS {
        new_s = (s1 - s3) / s2;
    }
    new_s = clampf(new_s, 0.0, MAX_S);

    let mut new_h = [0f32; 3];
    for k in 0..3 {
        let den = a[k] * new_s * new_s + 2.0 * b[k] * new_s + c[k];
        let mut hc = 0f32;
        if den > EPS {
            let num = pb[k] + new_s * pa[k];
            hc = num / den;
        }
        new_h[k] = hc;
    }

    try_cem6_10_dp(
        cem_index,
        ccs_index,
        ps,
        enc_params,
        endpoint_ise_range,
        weight_ise_range,
        new_s,
        actual_l[3] as f32 * s255,
        &[new_h[0], new_h[1], new_h[2], actual_h[3] as f32 * s255],
        endpoint_vals,
        weight_vals0,
        weight_vals1,
        &mut cur_blk_error,
    );
    cur_blk_error
}

/// `cem_encode_pixels`: encodes one partition's pixels with the given LDR CEM, single plane
/// (`ccs_index` = -1) or dual plane (`ccs_index` 0..=3, `weight_vals1` required). Returns the
/// new best error (`cur_blk_error` unchanged if not improved; outputs only written on improve).
pub(crate) fn cem_encode_pixels(
    cem_index: u32,
    ccs_index: i32,
    pixel_stats: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    endpoint_vals: &mut [u8],
    weight_vals0: &mut [u8],
    weight_vals1: Option<&mut [u8]>,
    cur_blk_error: u64,
    use_blue_contraction: bool,
    mut try_direct_encoding_flag: Option<&mut bool>,
) -> u64 {
    global_init_check();
    let dual_plane = ccs_index >= 0;
    if let Some(f) = try_direct_encoding_flag.as_deref_mut() {
        *f = false;
    }
    if pixel_stats.num_pixels == 0 {
        return u64::MAX;
    }
    let mut dummy = [0u8; MAXP];
    let w1: &mut [u8] = match weight_vals1 {
        Some(w) => w,
        None => &mut dummy,
    };
    match cem_index {
        CEM_LDR_LUM_DIRECT => encode_cem0_4(
            cem_index,
            pixel_stats,
            enc_params,
            endpoint_ise_range,
            weight_ise_range,
            endpoint_vals,
            weight_vals0,
            cur_blk_error,
        ),
        CEM_LDR_LUM_ALPHA_DIRECT => {
            if dual_plane {
                encode_cem4_dp_a(
                    cem_index,
                    pixel_stats,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    endpoint_vals,
                    weight_vals0,
                    w1,
                    cur_blk_error,
                )
            } else {
                encode_cem0_4(
                    cem_index,
                    pixel_stats,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    endpoint_vals,
                    weight_vals0,
                    cur_blk_error,
                )
            }
        }
        CEM_LDR_RGB_DIRECT
        | CEM_LDR_RGBA_DIRECT
        | CEM_LDR_RGB_BASE_PLUS_OFFSET
        | CEM_LDR_RGBA_BASE_PLUS_OFFSET => {
            if dual_plane {
                encode_cem8_12_9_13_dp(
                    cem_index,
                    ccs_index as u32,
                    pixel_stats,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    endpoint_vals,
                    weight_vals0,
                    w1,
                    cur_blk_error,
                    use_blue_contraction,
                    try_direct_encoding_flag,
                )
            } else {
                encode_cem8_12_9_13(
                    cem_index,
                    pixel_stats,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    endpoint_vals,
                    weight_vals0,
                    cur_blk_error,
                    use_blue_contraction,
                    try_direct_encoding_flag,
                )
            }
        }
        CEM_LDR_RGB_BASE_SCALE => {
            if dual_plane {
                encode_cem6_10_dp_rgb(
                    cem_index,
                    ccs_index as u32,
                    pixel_stats,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    endpoint_vals,
                    weight_vals0,
                    w1,
                    cur_blk_error,
                )
            } else {
                encode_cem6_10(
                    cem_index,
                    pixel_stats,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    endpoint_vals,
                    weight_vals0,
                    cur_blk_error,
                )
            }
        }
        CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => {
            if dual_plane {
                if ccs_index == 3 {
                    encode_cem10_dp_a(
                        cem_index,
                        pixel_stats,
                        enc_params,
                        endpoint_ise_range,
                        weight_ise_range,
                        endpoint_vals,
                        weight_vals0,
                        w1,
                        cur_blk_error,
                    )
                } else {
                    encode_cem6_10_dp_rgb(
                        cem_index,
                        ccs_index as u32,
                        pixel_stats,
                        enc_params,
                        endpoint_ise_range,
                        weight_ise_range,
                        endpoint_vals,
                        weight_vals0,
                        w1,
                        cur_blk_error,
                    )
                }
            } else {
                encode_cem6_10(
                    cem_index,
                    pixel_stats,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    endpoint_vals,
                    weight_vals0,
                    cur_blk_error,
                )
            }
        }
        _ => u64::MAX,
    }
}

#[inline]
fn global_init_check() {
    super::common::global_init();
}

#[allow(dead_code)]
fn _unused(a: &Vec4F) -> f32 {
    saturate(a[0])
}
