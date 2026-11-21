// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! `basisu_astc_ldr_common.cpp`: the float "surrogate" CEM encoders (fast approximate
//! endpoint/weight estimation with optional quantization simulation).

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
    ASTC_LDR_MAX_BLOCK_PIXELS, BIG_FLOAT_VAL, CemEncodeParams, FLAG_DISABLE_QUANT, FLAG_NO_ERROR,
    PixelStats, REALLY_SMALL_FLOAT_VAL, Vec3F, Vec4F, calc_pca_4d, clamp3, clampf, dot3, dot4,
    fast_floorf_int, fast_roundf_pos_int, saturate, sub4,
};
use crate::astc::helpers::{
    self as ah, BISE_64_LEVELS, CEM_LDR_LUM_ALPHA_DIRECT, CEM_LDR_LUM_DIRECT,
    CEM_LDR_RGB_BASE_SCALE, CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A, CEM_LDR_RGB_DIRECT,
    CEM_LDR_RGBA_DIRECT,
};

const MAXP: usize = ASTC_LDR_MAX_BLOCK_PIXELS;

/// `surrogate_evaluate_rgba_sp`: single plane weights (in [0,1]) and weighted float error.
pub(crate) fn surrogate_evaluate_rgba_sp(
    ps: &PixelStats,
    l: &Vec4F,
    h: &Vec4F,
    weights0: &mut [f32],
    num_weight_levels: u32,
    enc_params: &CemEncodeParams,
    flags: u32,
) -> f32 {
    let cw = &enc_params.comp_weights;
    let (wr, wg, wb, wa) = (cw[0] as f32, cw[1] as f32, cw[2] as f32, cw[3] as f32);
    let mut total_err = 0f32;
    let compute_error = (flags & FLAG_NO_ERROR) == 0;

    let (mut lr, mut lg, mut lb, mut la) = (l[0], l[1], l[2], l[3]);
    let (mut dr, mut dg, mut db, mut da) = (h[0] - lr, h[1] - lg, h[2] - lb, h[3] - la);
    let delta_col_nrm = dr * dr + dg * dg + db * db + da * da;

    let quant = (flags & FLAG_DISABLE_QUANT) == 0;
    let f = if quant {
        (num_weight_levels - 1) as f32 / (delta_col_nrm + REALLY_SMALL_FLOAT_VAL)
    } else {
        1.0f32 / (delta_col_nrm + REALLY_SMALL_FLOAT_VAL)
    };
    let inv_weight_levels = 1.0f32 / (num_weight_levels.wrapping_sub(1)) as f32;

    lr *= -dr;
    lg *= -dg;
    lb *= -db;
    la *= -da;
    dr *= f;
    dg *= f;
    db *= f;
    da *= f;
    let l_sum = if quant {
        (lr + lg + lb + la) * f + 0.5
    } else {
        (lr + lg + lb + la) * f
    };

    for i in 0..ps.num_pixels as usize {
        let p = &ps.pixels_f[i];
        let (r, g, b, a) = (p[0], p[1], p[2], p[3]);
        let d = r * dr + g * dg + b * db + a * da + l_sum;
        let mut w = if quant {
            fast_floorf_int(d) as f32 * inv_weight_levels
        } else {
            d
        };
        if w < 0.0 {
            w = 0.0;
        } else if w > 1.0 {
            w = 1.0;
        }
        weights0[i] = w;
        if compute_error {
            let one_minus_w = 1.0 - w;
            let dec_r = l[0] * one_minus_w + h[0] * w;
            let dec_g = l[1] * one_minus_w + h[1] * w;
            let dec_b = l[2] * one_minus_w + h[2] * w;
            let dec_a = l[3] * one_minus_w + h[3] * w;
            let diff_r = r - dec_r;
            let diff_g = g - dec_g;
            let diff_b = b - dec_b;
            let diff_a = a - dec_a;
            total_err += (wr * diff_r * diff_r)
                + (wg * diff_g * diff_g)
                + (wb * diff_b * diff_b)
                + (wa * diff_a * diff_a);
        }
    }
    total_err
}

/// `surrogate_evaluate_rgba_dp`: dual plane (plane 1 = `ccs_index`).
pub(crate) fn surrogate_evaluate_rgba_dp(
    ccs_index: u32,
    ps: &PixelStats,
    l: &Vec4F,
    h: &Vec4F,
    weights0: &mut [f32],
    weights1: &mut [f32],
    num_weight_levels: u32,
    enc_params: &CemEncodeParams,
    flags: u32,
) -> f32 {
    let ccs = (ccs_index & 3) as usize;
    let inv_weight_levels = 1.0f32 / (num_weight_levels.wrapping_sub(1)) as f32;
    let c0 = (ccs + 1) & 3;
    let c1 = (ccs + 2) & 3;
    let c2 = (ccs + 3) & 3;
    let (orig_lx, orig_ly, orig_lz, orig_lw) = (l[c0], l[c1], l[c2], l[ccs]);
    let (orig_hx, orig_hy, orig_hz, orig_hw) = (h[c0], h[c1], h[c2], h[ccs]);
    let cw = &enc_params.comp_weights;
    let (wx, wy, wz, ww) = (cw[c0] as f32, cw[c1] as f32, cw[c2] as f32, cw[ccs] as f32);
    let mut total_err = 0f32;
    let compute_error = (flags & FLAG_NO_ERROR) == 0;
    let quant = (flags & FLAG_DISABLE_QUANT) == 0;
    let n = ps.num_pixels as usize;

    // Plane 0
    {
        let (mut dx, mut dy, mut dz) = (orig_hx - orig_lx, orig_hy - orig_ly, orig_hz - orig_lz);
        let delta_col_nrm = dx * dx + dy * dy + dz * dz;
        let f = if quant {
            (num_weight_levels - 1) as f32 / (delta_col_nrm + REALLY_SMALL_FLOAT_VAL)
        } else {
            1.0f32 / (delta_col_nrm + REALLY_SMALL_FLOAT_VAL)
        };
        let (mut lx, mut ly, mut lz) = (orig_lx, orig_ly, orig_lz);
        lx *= -dx;
        ly *= -dy;
        lz *= -dz;
        dx *= f;
        dy *= f;
        dz *= f;
        let l_sum = if quant {
            (lx + ly + lz) * f + 0.5
        } else {
            (lx + ly + lz) * f
        };
        for i in 0..n {
            let p = &ps.pixels_f[i];
            let (x, y, z) = (p[c0], p[c1], p[c2]);
            let d = x * dx + y * dy + z * dz + l_sum;
            let mut weight = if quant {
                fast_floorf_int(d) as f32 * inv_weight_levels
            } else {
                d
            };
            if weight < 0.0 {
                weight = 0.0;
            } else if weight > 1.0 {
                weight = 1.0;
            }
            weights0[i] = weight;
            if compute_error {
                let omw = 1.0 - weight;
                let dec_x = orig_lx * omw + orig_hx * weight;
                let dec_y = orig_ly * omw + orig_hy * weight;
                let dec_z = orig_lz * omw + orig_hz * weight;
                let diff_x = x - dec_x;
                let diff_y = y - dec_y;
                let diff_z = z - dec_z;
                total_err +=
                    (wx * diff_x * diff_x) + (wy * diff_y * diff_y) + (wz * diff_z * diff_z);
            }
        }
    }

    // Plane 1
    {
        let delta_w = orig_hw - orig_lw;
        let f = if delta_w.abs() > REALLY_SMALL_FLOAT_VAL {
            if quant {
                (num_weight_levels - 1) as f32 / delta_w
            } else {
                1.0 / delta_w
            }
        } else {
            0.0
        };
        for i in 0..n {
            let w = ps.pixels_f[i][ccs];
            let mut weight = if quant {
                fast_floorf_int((w - orig_lw) * f + 0.5) as f32 * inv_weight_levels
            } else {
                (w - orig_lw) * f
            };
            if weight < 0.0 {
                weight = 0.0;
            } else if weight > 1.0 {
                weight = 1.0;
            }
            weights1[i] = weight;
            // Without quantization the error is 0 if L/H cover the entire span.
            if compute_error && (quant || w < orig_lw || w > orig_hw) {
                let omw = 1.0 - weight;
                let dec_w = orig_lw * omw + orig_hw * weight;
                let diff_w = w - dec_w;
                total_err += ww * diff_w * diff_w;
            }
        }
    }
    total_err
}

/// `surrogate_quant_endpoint_val`.
pub(crate) fn surrogate_quant_endpoint_val(e: f32, num_endpoint_levels: u32, flags: u32) -> f32 {
    if flags & FLAG_DISABLE_QUANT != 0 {
        return e;
    }
    let lm1 = num_endpoint_levels.wrapping_sub(1) as f32;
    let inv = 1.0f32 / lm1;
    fast_roundf_pos_int(e * lm1) as f32 * inv
}

/// `surrogate_quant_endpoint`.
pub(crate) fn surrogate_quant_endpoint(e: &Vec4F, num_endpoint_levels: u32, flags: u32) -> Vec4F {
    if flags & FLAG_DISABLE_QUANT != 0 {
        return *e;
    }
    let lm1 = num_endpoint_levels.wrapping_sub(1) as f32;
    let inv = 1.0f32 / lm1;
    let mut res = [0f32; 4];
    for i in 0..4 {
        res[i] = fast_roundf_pos_int(e[i] * lm1) as f32 * inv;
    }
    res
}

/// `get_num_weight_levels`: `BISE_64_LEVELS` = raw weights ([0,64], 65 levels).
fn get_num_weight_levels(weight_ise_range: u32) -> u32 {
    if weight_ise_range == BISE_64_LEVELS {
        65
    } else {
        ah::get_ise_levels(weight_ise_range)
    }
}

/// `cem_surrogate_encode_cem6_10_sp`.
fn cem_surrogate_encode_cem6_10_sp(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    low_endpoint: &mut Vec4F,
    high_endpoint: &mut Vec4F,
    s: &mut f32,
    weights0: &mut [f32],
    flags: u32,
) -> f32 {
    let num_endpoint_levels = ah::get_ise_levels(endpoint_ise_range);
    let num_weight_levels = get_num_weight_levels(weight_ise_range);
    let cem_has_alpha = cem_index == CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A;
    let axis = &ps.zero_rel_axis3;

    let mut d_min = BIG_FLOAT_VAL;
    let mut d_max = -BIG_FLOAT_VAL;
    for i in 0..ps.num_pixels as usize {
        let dot = dot3(&ps.pixels_f[i], axis);
        if dot < d_min {
            d_min = dot;
        }
        if dot > d_max {
            d_max = dot;
        }
    }
    let mut low3: Vec3F = [d_min * axis[0], d_min * axis[1], d_min * axis[2]];
    clamp3(&mut low3, 0.0, 1.0);
    let mut high3: Vec3F = [d_max * axis[0], d_max * axis[1], d_max * axis[2]];
    clamp3(&mut high3, 0.0, 1.0);

    const MAX_S: f32 = 255.0 / 256.0;
    let mut scale = MAX_S;
    let d = dot3(&low3, &high3);
    let nrm = dot3(&high3, &high3);
    if nrm > 0.0 {
        scale = d / nrm;
    }
    scale = clampf(scale, 0.0, MAX_S);
    scale = surrogate_quant_endpoint_val(scale * (256.0f32 / 255.0), num_endpoint_levels, flags);
    *s = scale;

    *high_endpoint = surrogate_quant_endpoint(
        &[
            high3[0],
            high3[1],
            high3[2],
            if cem_has_alpha { ps.max_f[3] } else { 1.0 },
        ],
        num_endpoint_levels,
        flags,
    );
    *low_endpoint = [
        high_endpoint[0] * scale,
        high_endpoint[1] * scale,
        high_endpoint[2] * scale,
        if cem_has_alpha { ps.min_f[3] } else { 1.0 },
    ];
    surrogate_evaluate_rgba_sp(
        ps,
        low_endpoint,
        high_endpoint,
        weights0,
        num_weight_levels,
        enc_params,
        flags,
    )
}

/// `cem_surrogate_encode_cem6_10_dp`.
fn cem_surrogate_encode_cem6_10_dp(
    cem_index: u32,
    ccs_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    low_endpoint: &mut Vec4F,
    high_endpoint: &mut Vec4F,
    s: &mut f32,
    weights0: &mut [f32],
    weights1: &mut [f32],
    flags: u32,
) -> f32 {
    let num_weight_levels = get_num_weight_levels(weight_ise_range);
    let mut temp_weights = [0f32; MAXP];
    cem_surrogate_encode_cem6_10_sp(
        if ccs_index == 3 {
            CEM_LDR_RGB_BASE_SCALE
        } else {
            cem_index
        },
        ps,
        enc_params,
        endpoint_ise_range,
        weight_ise_range,
        low_endpoint,
        high_endpoint,
        s,
        &mut temp_weights,
        flags,
    );
    if ccs_index == 3 {
        low_endpoint[3] = ps.min_f[3];
        high_endpoint[3] = ps.max_f[3];
    }
    surrogate_evaluate_rgba_dp(
        ccs_index,
        ps,
        low_endpoint,
        high_endpoint,
        weights0,
        weights1,
        num_weight_levels,
        enc_params,
        flags,
    )
}

/// Spreads endpoints that quantized to the same value although the channel varies (shared by
/// `cem_surrogate_encode_cem8_12_sp/dp`).
fn fix_degenerate_endpoints(
    ps: &PixelStats,
    num_comps: usize,
    num_endpoint_levels: u32,
    low_endpoint: &mut Vec4F,
    high_endpoint: &mut Vec4F,
) {
    for i in 0..num_comps {
        if low_endpoint[i] == high_endpoint[i] && ps.min_f[i] != ps.max_f[i] {
            let inv = 1.0f32 / num_endpoint_levels.wrapping_sub(1) as f32;
            let mut best_dist = BIG_FLOAT_VAL;
            let (mut best_l, mut best_h) = (0f32, 0f32);
            for ld in -2..=0 {
                let actual_l = saturate(low_endpoint[i] + ld as f32 * inv);
                for hd in 0..=2 {
                    let actual_h = saturate(high_endpoint[i] + hd as f32 * inv);
                    let v0 = actual_l + (actual_h - actual_l) * (1.0f32 / 3.0);
                    let v1 = actual_l + (actual_h - actual_l) * (2.0f32 / 3.0);
                    // (C++ compares against channel 0's bounds for every channel)
                    let dist0 = v0 - ps.min_f[0];
                    let dist1 = v1 - ps.max_f[0];
                    let total_dist = dist0 * dist0 + dist1 * dist1;
                    if total_dist < best_dist {
                        best_dist = total_dist;
                        best_l = actual_l;
                        best_h = actual_h;
                    }
                }
            }
            low_endpoint[i] = best_l;
            high_endpoint[i] = best_h;
        }
    }
}

#[inline]
fn sum4(v: &Vec4F) -> f32 {
    dot4(v, &[1.0; 4])
}

/// `cem_surrogate_encode_cem8_12_sp`.
fn cem_surrogate_encode_cem8_12_sp(
    cem_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    low_endpoint: &mut Vec4F,
    high_endpoint: &mut Vec4F,
    weights0: &mut [f32],
    flags: u32,
) -> f32 {
    let num_endpoint_levels = ah::get_ise_levels(endpoint_ise_range);
    let num_weight_levels = get_num_weight_levels(weight_ise_range);
    let cem_has_alpha = cem_index == CEM_LDR_RGBA_DIRECT;
    let num_comps = if cem_has_alpha { 4 } else { 3 };

    let mut d_min = BIG_FLOAT_VAL;
    let mut d_max = -BIG_FLOAT_VAL;
    let (mut l_idx, mut h_idx) = (0usize, 0usize);
    for i in 0..ps.num_pixels as usize {
        let p = sub4(&ps.pixels_f[i], &ps.mean_f);
        let dot = if cem_has_alpha {
            dot4(&p, &ps.mean_rel_axis4)
        } else {
            dot3(&p, &ps.mean_rel_axis3)
        };
        if dot < d_min {
            d_min = dot;
            l_idx = i;
        }
        if dot > d_max {
            d_max = dot;
            h_idx = i;
        }
    }
    *low_endpoint = surrogate_quant_endpoint(&ps.pixels_f[l_idx], num_endpoint_levels, flags);
    *high_endpoint = surrogate_quant_endpoint(&ps.pixels_f[h_idx], num_endpoint_levels, flags);
    if !cem_has_alpha {
        low_endpoint[3] = 1.0;
        high_endpoint[3] = 1.0;
    }
    if sum4(low_endpoint) > sum4(high_endpoint) {
        std::mem::swap(low_endpoint, high_endpoint);
    }
    if flags & FLAG_DISABLE_QUANT == 0 {
        fix_degenerate_endpoints(
            ps,
            num_comps,
            num_endpoint_levels,
            low_endpoint,
            high_endpoint,
        );
    }
    surrogate_evaluate_rgba_sp(
        ps,
        low_endpoint,
        high_endpoint,
        weights0,
        num_weight_levels,
        enc_params,
        flags,
    )
}

/// `cem_surrogate_encode_cem8_12_dp`.
fn cem_surrogate_encode_cem8_12_dp(
    cem_index: u32,
    ccs_index: u32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    low_endpoint: &mut Vec4F,
    high_endpoint: &mut Vec4F,
    weights0: &mut [f32],
    weights1: &mut [f32],
    flags: u32,
) -> f32 {
    let ccs = (ccs_index & 3) as usize;
    let num_endpoint_levels = ah::get_ise_levels(endpoint_ise_range);
    let num_weight_levels = get_num_weight_levels(weight_ise_range);
    let cem_has_alpha = cem_index == CEM_LDR_RGBA_DIRECT;
    let num_comps = if cem_has_alpha { 4 } else { 3 };
    let n = ps.num_pixels as usize;

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
    let flattened_axis = calc_pca_4d(&flattened[..n], &flattened_mean);

    let mut best_dl = BIG_FLOAT_VAL;
    let mut best_dh = -BIG_FLOAT_VAL;
    let (mut best_l_index, mut best_h_index) = (0usize, 0usize);
    for c in 0..n {
        let px = sub4(&flattened[c], &flattened_mean);
        let p = dot4(&px, &flattened_axis);
        if p < best_dl {
            best_dl = p;
            best_l_index = c;
        }
        if p > best_dh {
            best_dh = p;
            best_h_index = c;
        }
    }
    let mut low_color_f = ps.pixels_f[best_l_index];
    let mut high_color_f = ps.pixels_f[best_h_index];
    low_color_f[ccs] = 0.0;
    high_color_f[ccs] = 0.0;
    if !cem_has_alpha {
        low_color_f[3] = 1.0;
        high_color_f[3] = 1.0;
    }
    if sum4(&low_color_f) > sum4(&high_color_f) {
        std::mem::swap(&mut low_color_f, &mut high_color_f);
    }
    low_color_f[ccs] = ps.min_f[ccs];
    high_color_f[ccs] = ps.max_f[ccs];
    if !cem_has_alpha {
        low_color_f[3] = 1.0;
        high_color_f[3] = 1.0;
    }
    *low_endpoint = surrogate_quant_endpoint(&low_color_f, num_endpoint_levels, flags);
    *high_endpoint = surrogate_quant_endpoint(&high_color_f, num_endpoint_levels, flags);
    if flags & FLAG_DISABLE_QUANT == 0 {
        fix_degenerate_endpoints(
            ps,
            num_comps,
            num_endpoint_levels,
            low_endpoint,
            high_endpoint,
        );
    }
    surrogate_evaluate_rgba_dp(
        ccs_index,
        ps,
        low_endpoint,
        high_endpoint,
        weights0,
        weights1,
        num_weight_levels,
        enc_params,
        flags,
    )
}

/// `cem_surrogate_encode_cem0_4_sp_or_dp`.
fn cem_surrogate_encode_cem0_4_sp_or_dp(
    cem_index: u32,
    ccs_index: i32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    low_endpoint: &mut Vec4F,
    high_endpoint: &mut Vec4F,
    weights0: &mut [f32],
    weights1: &mut [f32],
    flags: u32,
) -> f32 {
    let cem_has_alpha = cem_index == CEM_LDR_LUM_ALPHA_DIRECT;
    let dual_plane = ccs_index == 3;
    let num_endpoint_levels = ah::get_ise_levels(endpoint_ise_range);
    let num_weight_levels = get_num_weight_levels(weight_ise_range);

    let mut lum_l = BIG_FLOAT_VAL;
    let mut lum_h = -BIG_FLOAT_VAL;
    for i in 0..ps.num_pixels as usize {
        let px = &ps.pixels_f[i];
        let l = (px[0] + px[1] + px[2]) * (1.0f32 / 3.0);
        lum_l = if l < lum_l { l } else { lum_l };
        lum_h = if lum_h < l { l } else { lum_h };
    }
    let a_l = if cem_has_alpha { ps.min_f[3] } else { 1.0 };
    let a_h = if cem_has_alpha { ps.max_f[3] } else { 1.0 };
    *low_endpoint =
        surrogate_quant_endpoint(&[lum_l, lum_l, lum_l, a_l], num_endpoint_levels, flags);
    *high_endpoint =
        surrogate_quant_endpoint(&[lum_h, lum_h, lum_h, a_h], num_endpoint_levels, flags);
    if dual_plane {
        surrogate_evaluate_rgba_dp(
            3,
            ps,
            low_endpoint,
            high_endpoint,
            weights0,
            weights1,
            num_weight_levels,
            enc_params,
            flags,
        )
    } else {
        surrogate_evaluate_rgba_sp(
            ps,
            low_endpoint,
            high_endpoint,
            weights0,
            num_weight_levels,
            enc_params,
            flags,
        )
    }
}

/// `cem_surrogate_encode_pixels`: CEM 0/4/6/8/10/12 (others return `BIG_FLOAT_VAL`).
/// `s` receives the CEM 6/10 scale.
pub(crate) fn cem_surrogate_encode_pixels(
    cem_index: u32,
    ccs_index: i32,
    ps: &PixelStats,
    enc_params: &CemEncodeParams,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    low_endpoint: &mut Vec4F,
    high_endpoint: &mut Vec4F,
    s: &mut f32,
    weights0: &mut [f32],
    weights1: &mut [f32],
    flags: u32,
) -> f32 {
    let dual_plane = ccs_index >= 0;
    match cem_index {
        CEM_LDR_LUM_DIRECT | CEM_LDR_LUM_ALPHA_DIRECT => cem_surrogate_encode_cem0_4_sp_or_dp(
            cem_index,
            ccs_index,
            ps,
            enc_params,
            endpoint_ise_range,
            weight_ise_range,
            low_endpoint,
            high_endpoint,
            weights0,
            weights1,
            flags,
        ),
        CEM_LDR_RGB_BASE_SCALE | CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => {
            if dual_plane {
                cem_surrogate_encode_cem6_10_dp(
                    cem_index,
                    ccs_index as u32,
                    ps,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    low_endpoint,
                    high_endpoint,
                    s,
                    weights0,
                    weights1,
                    flags,
                )
            } else {
                cem_surrogate_encode_cem6_10_sp(
                    cem_index,
                    ps,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    low_endpoint,
                    high_endpoint,
                    s,
                    weights0,
                    flags,
                )
            }
        }
        CEM_LDR_RGB_DIRECT | CEM_LDR_RGBA_DIRECT => {
            if dual_plane {
                cem_surrogate_encode_cem8_12_dp(
                    cem_index,
                    ccs_index as u32,
                    ps,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    low_endpoint,
                    high_endpoint,
                    weights0,
                    weights1,
                    flags,
                )
            } else {
                cem_surrogate_encode_cem8_12_sp(
                    cem_index,
                    ps,
                    enc_params,
                    endpoint_ise_range,
                    weight_ise_range,
                    low_endpoint,
                    high_endpoint,
                    weights0,
                    flags,
                )
            }
        }
        _ => BIG_FLOAT_VAL,
    }
}
