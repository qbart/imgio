// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of `astc_ldr_t::grid_weight_dct` decoding: dequantizes the run/coefficient symbols of
//! one weight plane, applies the separable inverse DCT (`basisu_idct.h`) and requantizes the
//! grid weights. Float math follows the C++ operation order exactly.

use super::endpoints::decode_endpoints;
use super::tables::IDCT;
use crate::astc::helpers::{self as ah, BISE_8_LEVELS, LogAstcBlock, MAX_GRID_WEIGHTS};

const DEADZONE_ALPHA: f32 = 0.5;
const SCALED_WEIGHT_BASE_CODING_SCALE: f32 = 0.5;

/// `DCT_MEAN_LEVELS1`.
pub(super) const DCT_MEAN_LEVELS1: u32 = 33;

/// Results of `scale_quant_steps()` per weight ISE range.
#[allow(clippy::excessive_precision)]
const SCALE_QUANT_STEPS: [f32; 12] = [
    1.51333141, 1.41198814, 1.35588217, 1.31743157, 1.28835952, 1.24573100, 1.21481407, 1.19067919,
    1.15431654, 1.12734985, 1.10601568, 1.07348967,
];

/// DC element modified (see basisu).
const BASELINE_JPEG_Y: [[i32; 8]; 8] = [
    [4, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
];

fn coding_scale(weight_ise_range: u32) -> f32 {
    if weight_ise_range <= BISE_8_LEVELS {
        1.0 / 8.0
    } else {
        SCALED_WEIGHT_BASE_CODING_SCALE
    }
}

/// `grid_weight_dct::get_num_weight_dc_levels`.
pub(super) fn num_weight_dc_levels(weight_ise_range: u32) -> u32 {
    (64.0 * coding_scale(weight_ise_range)) as u32 + 1
}

/// `dct_syms::coeff`: zeros skipped before `coeff`.
#[derive(Clone, Copy, Default)]
pub(super) struct Coeff {
    pub num_zeros: u16,
    pub coeff: i16,
}

/// `dct_syms` (the coefficient list holds at most 64 entries).
pub(super) struct DctSyms {
    pub dc_sym: u32,
    pub coeffs: [Coeff; 64],
    pub num_coeffs: usize,
}

impl Default for DctSyms {
    fn default() -> Self {
        Self {
            dc_sym: 0,
            coeffs: [Coeff::default(); 64],
            num_coeffs: 0,
        }
    }
}

/// `generate_zigzag_order(width, height)` into `out`.
fn zigzag(width: usize, height: usize, out: &mut [usize; 64]) {
    let mut idx = 0;
    for s in 0..width + height - 1 {
        let x_start = if s < height { 0 } else { s - height + 1 };
        let x_end = if s < width { s } else { width - 1 };
        let mut push = |x: usize| {
            if let Some(o) = out.get_mut(idx) {
                *o = x + (s - x) * width;
            }
            idx += 1;
        };
        if s & 1 == 1 {
            (x_start..=x_end).rev().for_each(&mut push);
        } else {
            (x_start..=x_end).for_each(&mut push);
        }
    }
}

/// `grid_weight_dct::compute_level_scale` (adaptive quantization).
fn compute_level_scale(q: f32, span_len: f32, weight_ise_range: u32) -> f32 {
    let q = q.clamp(1.0, 100.0);
    let mut level_scale = if q < 50.0 {
        5000.0 / q
    } else {
        200.0 - 2.0 * q
    };
    level_scale *= 1.0f32 / 100.0;
    let span_floor = 14.0f32;
    let mut adaptive_factor = 64.0 / span_len.max(span_floor);
    adaptive_factor *= SCALE_QUANT_STEPS
        .get(weight_ise_range as usize)
        .copied()
        .unwrap_or(1.0);
    level_scale *= adaptive_factor;
    level_scale
}

/// `sample_quant_table_state`.
struct QuantState {
    q: f32,
    sx: f32,
    sy: f32,
    level_scale: f32,
}

/// `grid_weight_dct::sample_quant_table`.
fn sample_quant_table(state: &QuantState, x: u32, y: u32) -> i32 {
    if state.q >= 100.0 {
        return 1;
    }
    let ry = y as f32 * state.sy;
    let rx = x as f32 * state.sx;
    let i = rx.min(7.0);
    let j = ry.min(7.0);
    let (i0, j0) = (i as i32, j as i32);
    let (i1, j1) = ((i0 + 1).min(7), (j0 + 1).min(7));
    let ti = i - i0 as f32;
    let tj = j - j0 as f32;
    let at = |jj: i32, ii: i32| BASELINE_JPEG_Y[jj as usize & 7][ii as usize & 7] as f32;
    let a = (1.0 - ti) * at(j0, i0) + ti * at(j0, i1);
    let b = (1.0 - ti) * at(j1, i0) + ti * at(j1, i1);
    let base = (1.0 - tj) * a + tj * b;
    ((base * state.level_scale + 0.5) as i32).max(1)
}

/// `grid_weight_dct::dequant_deadzone`.
fn dequant_deadzone(q: i32, l: i32, alpha: f32, x: u32, y: u32) -> f32 {
    if (x == 1 && y == 0) || (x == 0 && y == 1) {
        return q as f32 * l as f32;
    }
    if q == 0 || l <= 0 {
        return 0.0;
    }
    let tau = alpha * l as f32;
    let mag = tau + q.unsigned_abs() as f32 * l as f32;
    if q < 0 { -mag } else { mag }
}

/// `squaref((float)h - (float)l)`.
fn sq(h: u8, l: u8) -> f32 {
    let d = f32::from(h) - f32::from(l);
    d * d
}

/// `grid_weight_dct::get_max_span_len`.
fn max_span_len(blk: &LogAstcBlock, plane_index: u32) -> f32 {
    let cem = u32::from(blk.color_endpoint_modes[0]);
    let range = u32::from(blk.endpoint_ise_range);
    if blk.dual_plane {
        let (l, h) = decode_endpoints(cem, &blk.endpoints, range);
        let ccs = usize::from(blk.color_component_selector);
        let mut span = 0.0f32;
        for c in 0..4 {
            if (plane_index == 1) == (c == ccs) {
                span += sq(h[c], l[c]);
            }
        }
        span.sqrt()
    } else {
        let n = ah::get_num_cem_values(cem) as usize;
        let mut span = 0.0f32;
        for i in 0..usize::from(blk.num_partitions) {
            let (l, h) = decode_endpoints(cem, blk.endpoints.get(n * i..).unwrap_or(&[]), range);
            let part = (sq(h[0], l[0]) + sq(h[1], l[1]) + sq(h[2], l[2]) + sq(h[3], l[3])).sqrt();
            span = part.max(span);
        }
        span
    }
}

/// `idct_1d_N` on `n` values read at `src[k * stride]`, written to `dst[x * stride]`.
fn idct_1d(n: usize, src: &[f32], dst: &mut [f32], stride: usize) {
    let Some(tab) = IDCT.get(n.wrapping_sub(2)) else {
        return;
    };
    let mut s = [0.0f32; 12];
    for (k, row) in tab.iter().enumerate().take(n) {
        let v = src.get(k * stride).copied().unwrap_or(0.0);
        if v != 0.0 {
            for (acc, &c) in s.iter_mut().zip(row).take(n) {
                if c != 0.0 {
                    *acc += c * v;
                }
            }
        }
    }
    for (x, &v) in s.iter().enumerate().take(n) {
        if let Some(d) = dst.get_mut(x * stride) {
            *d = v;
        }
    }
}

/// `idct_2d`: columns (`rows` long) first, then rows (`cols` long).
fn idct_2d(src: &[f32; 64], dst: &mut [f32; 64], rows: usize, cols: usize) {
    let mut temp = [0.0f32; 144];
    for c in 0..cols {
        idct_1d(rows, &src[c..], &mut temp[c..], cols);
    }
    for r in 0..rows {
        let o = r * cols;
        idct_1d(cols, &temp[o..o + cols], &mut dst[o..o + cols], 1);
    }
}

/// `grid_weight_dct::decode_block_weights` (symbol path): decodes plane `plane_index` of
/// `blk` (configuration and endpoints already set) for a `block_w` x `block_h` block.
pub(super) fn decode_block_weights(
    block_w: u32,
    block_h: u32,
    q: f32,
    plane_index: u32,
    blk: &mut LogAstcBlock,
    syms: &DctSyms,
) -> bool {
    let (gw, gh) = (u32::from(blk.grid_width), u32::from(blk.grid_height));
    if gw < 2 || gh < 2 || gw > block_w || gh > block_h || gw * gh > MAX_GRID_WEIGHTS {
        return false;
    }
    let total = (gw * gh) as usize;
    let num_planes = if blk.dual_plane { 2 } else { 1 };
    let wrange = u32::from(blk.weight_ise_range);
    let tabs = ah::dequant_tables();
    let quant_tab = &tabs.weight_tab(wrange).val_to_ise;
    if quant_tab.len() < 65 {
        return false;
    }

    let span_len = max_span_len(blk, plane_index);
    let level_scale = compute_level_scale(q, span_len, wrange);
    let mean_weight = syms.dc_sym as f32 / coding_scale(wrange);

    let mut order = [0usize; 64];
    zigzag(gw as usize, gh as usize, &mut order);
    let state = QuantState {
        q,
        sx: 8.0 / block_w as f32,
        sy: 8.0 / block_h as f32,
        level_scale,
    };

    let mut dct_weights = [0.0f32; 64];
    let mut zig_idx = 1usize;
    for c in syms.coeffs.iter().take(syms.num_coeffs) {
        let run_len = usize::from(c.num_zeros);
        if run_len + zig_idx > total {
            return false;
        }
        zig_idx += run_len;
        if zig_idx >= total {
            break;
        }
        let dct_idx = order[zig_idx];
        let (y, x) = ((dct_idx as u32) / gw, (dct_idx as u32) % gw);
        let quant = sample_quant_table(&state, x, y);
        dct_weights[dct_idx] = dequant_deadzone(i32::from(c.coeff), quant, DEADZONE_ALPHA, x, y);
        zig_idx += 1;
    }

    let mut idct_weights = [0.0f32; 64];
    idct_2d(&dct_weights, &mut idct_weights, gh as usize, gw as usize);

    for (i, &w) in idct_weights.iter().enumerate().take(total) {
        let v = mean_weight + w;
        // `fast_roundf_int`
        let r = if v >= 0.0 {
            (v + 0.5) as i32
        } else {
            (v - 0.5) as i32
        };
        let Some(slot) = blk.weights.get_mut(i * num_planes + plane_index as usize) else {
            return false;
        };
        *slot = quant_tab[r.clamp(0, 64) as usize];
    }
    true
}
