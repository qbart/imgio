// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of `basisu_astc_ldr_common.cpp` (namespace `astc_ldr`) up to `partition_pattern_vec`:
//! CEM endpoint / weight encoding, solution evaluation and surrogate encoders.
//!
//! Split into: this file (types, tables, endpoint packing helpers, color decoding, PCA,
//! `pixel_stats_t`, solution evaluation), [`super::common_encode`] (`try_cem*` / `encode_cem*` /
//! `weight_refiner` / `cem_encode_pixels`) and [`super::common_surrogate`] (the float surrogate
//! encoders). Everything is re-exported from here.
//!
//! Porting notes:
//! - `vec4F` / `vec3F` are plain `[f32; 4]` / `[f32; 3]` arrays; the small helpers below keep the
//!   exact float operation order of basisu's `vec<N, T>` (`dot` accumulates left to right,
//!   `normalize_in_place` multiplies by `1 / length`).
//! - `cem_encode_params::m_pForced_weight_vals0/1` are `Option<&[u8]>` fields of
//!   [`CemEncodeParams`] (hence its lifetime parameter).
//! - C++ overloads get distinct names: `get_colors(l, h, ..)` = [`get_colors`],
//!   `get_colors(cem, endpoints, ..)` = [`get_colors_cem`] (same for `get_colors_raw_weights`),
//!   the endpoint-only `encode_cem0_4` = [`encode_cem0_4_endpoints`].
//! - `astc_ldr::global_init` builds a lazily initialized static table ([`global_init`] is
//!   optional: every user initializes it on first use).

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::implicit_saturating_sub,
    clippy::manual_clamp,
    clippy::manual_range_contains,
    clippy::needless_option_as_deref
)]

#[allow(unused_imports)]
pub(crate) use super::common_encode::*;
#[allow(unused_imports)]
pub(crate) use super::common_surrogate::*;

use crate::astc::helpers::{
    self as ah, BISE_32_LEVELS, BISE_64_LEVELS, BISE_256_LEVELS, CEM_LDR_LUM_ALPHA_DIRECT,
    CEM_LDR_RGB_BASE_PLUS_OFFSET, CEM_LDR_RGB_BASE_SCALE, CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A,
    CEM_LDR_RGBA_BASE_PLUS_OFFSET, CEM_LDR_RGBA_DIRECT,
};
use crate::basis::ll::astc_hdr_enc::G_ISE_WEIGHT_LERPS;
use crate::basis::ll::enc::ColorRgba;
use crate::basis::xuastc::endpoints::{MAX_CEM_VALS, requantize_ise_endpoints};
use std::sync::OnceLock;

/// `ASTC_LDR_MAX_BLOCK_WIDTH`.
pub(crate) const ASTC_LDR_MAX_BLOCK_WIDTH: u32 = ah::MAX_BLOCK_DIM;
/// `ASTC_LDR_MAX_BLOCK_HEIGHT`.
pub(crate) const ASTC_LDR_MAX_BLOCK_HEIGHT: u32 = ah::MAX_BLOCK_DIM;
/// `ASTC_LDR_MAX_BLOCK_PIXELS`.
pub(crate) const ASTC_LDR_MAX_BLOCK_PIXELS: usize = ah::MAX_BLOCK_PIXELS;
/// `ASTC_LDR_MAX_RAW_WEIGHTS`.
pub(crate) const ASTC_LDR_MAX_RAW_WEIGHTS: usize = ah::MAX_WEIGHT_INTERPOLANT_VALUE as usize + 1;
/// `WEIGHT_REFINER_MAX_PASSES`.
pub(crate) const WEIGHT_REFINER_MAX_PASSES: u32 = 17;
/// `astc_helpers::MAX_CEM_ENDPOINT_VALS`.
pub(crate) const MAX_CEM_ENDPOINT_VALS: usize = 8;

/// `cFlagDisableQuant`.
pub(crate) const FLAG_DISABLE_QUANT: u32 = 1;
/// `cFlagNoError`.
pub(crate) const FLAG_NO_ERROR: u32 = 2;

/// `BIG_FLOAT_VAL`.
pub(crate) const BIG_FLOAT_VAL: f32 = 1e30;
/// `REALLY_SMALL_FLOAT_VAL`.
pub(crate) const REALLY_SMALL_FLOAT_VAL: f32 = 0.000000125;

const MAXP: usize = ASTC_LDR_MAX_BLOCK_PIXELS;

/// `vec4F`.
pub(crate) type Vec4F = [f32; 4];
/// `vec3F`.
pub(crate) type Vec3F = [f32; 3];

// ---------------------------------------------------------------------------------------------
// vec helpers (exact basisu `vec<N, float>` operation order)

#[inline]
pub(crate) fn sub4(a: &Vec4F, b: &Vec4F) -> Vec4F {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[inline]
pub(crate) fn add4(a: &Vec4F, b: &Vec4F) -> Vec4F {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// `vec * s`.
#[inline]
pub(crate) fn mul4(a: &Vec4F, s: f32) -> Vec4F {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

/// `vec4F::dot_product`.
#[inline]
pub(crate) fn dot4(a: &Vec4F, b: &Vec4F) -> f32 {
    let mut r = a[0] * b[0];
    r += a[1] * b[1];
    r += a[2] * b[2];
    r += a[3] * b[3];
    r
}

/// `vec4F::dot_product3` / `vec3F::dot_product`.
#[inline]
pub(crate) fn dot3(a: &[f32], b: &[f32]) -> f32 {
    let mut r = a[0] * b[0];
    r += a[1] * b[1];
    r += a[2] * b[2];
    r
}

/// `vec4F::normalize_in_place`.
#[inline]
pub(crate) fn normalize4(a: &mut Vec4F) {
    let len = dot4(a, a).sqrt();
    if len != 0.0 {
        let s = 1.0 / len;
        for v in a.iter_mut() {
            *v *= s;
        }
    }
}

/// `vec3F(vec4F)`.
#[inline]
pub(crate) fn v3(a: &Vec4F) -> Vec3F {
    [a[0], a[1], a[2]]
}

/// `vec4F(vec3F)` (w = 0).
#[inline]
pub(crate) fn v4(a: &Vec3F) -> Vec4F {
    [a[0], a[1], a[2], 0.0]
}

/// `basisu::clamp` / `clampf` on floats.
#[inline]
pub(crate) fn clampf(v: f32, l: f32, h: f32) -> f32 {
    if v < l {
        l
    } else if v > h {
        h
    } else {
        v
    }
}

/// `saturate`.
#[inline]
pub(crate) fn saturate(v: f32) -> f32 {
    clampf(v, 0.0, 1.0)
}

#[inline]
pub(crate) fn clamp4(a: &mut Vec4F, l: f32, h: f32) {
    for v in a.iter_mut() {
        *v = clampf(*v, l, h);
    }
}

#[inline]
pub(crate) fn clamp3(a: &mut Vec3F, l: f32, h: f32) {
    for v in a.iter_mut() {
        *v = clampf(*v, l, h);
    }
}

/// C `(int)x` as compiled for x86-64 (`cvttss2si`): truncation, `INT_MIN` when out of range.
#[inline]
pub(crate) fn ftoi(x: f32) -> i32 {
    if x >= -2147483648.0 && x < 2147483648.0 {
        x as i32
    } else {
        i32::MIN
    }
}

/// `fast_floorf_int`.
#[inline]
pub(crate) fn fast_floorf_int(x: f32) -> i32 {
    let xi = ftoi(x);
    if x < 0.0 && x != xi as f32 {
        xi.wrapping_sub(1)
    } else {
        xi
    }
}

/// `fast_roundf_pos_int`.
#[inline]
pub(crate) fn fast_roundf_pos_int(x: f32) -> i32 {
    ftoi(x + 0.5)
}

// ---------------------------------------------------------------------------------------------
// Parameters / stats

/// `astc_ldr::cem_encode_params`. `forced_weight_vals0/1` are `m_pForced_weight_vals0/1`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CemEncodeParams<'a> {
    pub comp_weights: [u32; 4],
    pub decode_mode_srgb: bool,
    pub forced_weight_vals0: Option<&'a [u8]>,
    pub forced_weight_vals1: Option<&'a [u8]>,
    pub max_ls_passes: u32,
    pub total_weight_refine_passes: u32,
    pub worst_weight_nudging_flag: bool,
    pub endpoint_refinement_flag: bool,
    pub use_exhaustive_weight_eval: bool,
}

impl Default for CemEncodeParams<'_> {
    fn default() -> Self {
        Self {
            comp_weights: [1, 1, 1, 1],
            decode_mode_srgb: true,
            forced_weight_vals0: None,
            forced_weight_vals1: None,
            max_ls_passes: 3,
            total_weight_refine_passes: 0,
            worst_weight_nudging_flag: false,
            endpoint_refinement_flag: false,
            use_exhaustive_weight_eval: false,
        }
    }
}

impl CemEncodeParams<'_> {
    /// `get_total_comp_weights`.
    pub fn get_total_comp_weights(&self) -> f32 {
        (self.comp_weights[0] + self.comp_weights[1] + self.comp_weights[2] + self.comp_weights[3])
            as f32
    }
}

/// `basisu::stats<float>` (the fields `calc_simplified_with_range` computes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Stats {
    pub n: u32,
    pub total: f32,
    pub avg: f32,
    pub var: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub range: f32,
}

impl Stats {
    /// `calc_simplified_with_range(n, pVals, stride)`.
    pub fn calc_simplified_with_range(&mut self, n: u32, vals: impl Iterator<Item = f32> + Clone) {
        *self = Stats::default();
        if n == 0 {
            return;
        }
        self.n = n;
        for (i, v) in vals.clone().take(n as usize).enumerate() {
            self.total += v;
            if i == 0 {
                self.min = v;
                self.max = v;
            } else {
                self.min = if v < self.min { v } else { self.min };
                self.max = if self.max < v { v } else { self.max };
            }
        }
        self.range = self.max - self.min;
        let nd = n as f32;
        self.avg = self.total / nd;
        for v in vals.take(n as usize) {
            let d = v - self.avg;
            let d2 = d * d;
            self.var += d2;
        }
        self.var /= nd;
        self.std_dev = self.var.sqrt();
    }
}

/// `astc_ldr::pixel_stats_t`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PixelStats {
    pub num_pixels: u32,
    pub pixels: [ColorRgba; MAXP],
    pub pixels_f: [Vec4F; MAXP],
    pub min: ColorRgba,
    pub max: ColorRgba,
    pub min_f: Vec4F,
    pub max_f: Vec4F,
    pub mean_f: Vec4F,
    /// Always 3D, ignoring alpha.
    pub mean_rel_axis3: Vec3F,
    pub zero_rel_axis3: Vec3F,
    /// Always 4D.
    pub mean_rel_axis4: Vec4F,
    pub has_alpha: bool,
    pub rgba_stats: [Stats; 4],
}

impl Default for PixelStats {
    fn default() -> Self {
        Self {
            num_pixels: 0,
            pixels: [[0; 4]; MAXP],
            pixels_f: [[0.0; 4]; MAXP],
            min: [0; 4],
            max: [0; 4],
            min_f: [0.0; 4],
            max_f: [0.0; 4],
            mean_f: [0.0; 4],
            mean_rel_axis3: [0.0; 3],
            zero_rel_axis3: [0.0; 3],
            mean_rel_axis4: [0.0; 4],
            has_alpha: false,
            rgba_stats: [Stats::default(); 4],
        }
    }
}

impl PixelStats {
    /// `clear`.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// `pixel_stats_t::init` (`pixels.len()` must be in 1..=144).
    pub fn init(&mut self, num_pixels: u32, pixels: &[ColorRgba]) {
        let n = (num_pixels as usize).min(MAXP).min(pixels.len());
        self.num_pixels = n as u32;
        self.has_alpha = false;
        self.min = [255; 4];
        self.max = [0; 4];
        self.mean_f = [0.0; 4];
        let s = 1.0f32 / 255.0;
        for i in 0..n {
            let px = pixels[i];
            self.pixels[i] = px;
            self.pixels_f[i] = [
                px[0] as f32 * s,
                px[1] as f32 * s,
                px[2] as f32 * s,
                px[3] as f32 * s,
            ];
            self.mean_f = add4(&self.mean_f, &self.pixels_f[i]);
            for c in 0..4 {
                self.min[c] = self.min[c].min(px[c]);
                self.max[c] = self.max[c].max(px[c]);
            }
        }
        self.mean_f = mul4(&self.mean_f, 1.0 / n as f32);
        clamp4(&mut self.mean_f, 0.0, 1.0);
        for c in 0..4 {
            self.min_f[c] = self.min[c] as f32 * s;
            self.max_f[c] = self.max[c] as f32 * s;
        }
        self.has_alpha = self.min[3] < 255;
        self.mean_rel_axis3 = v3(&calc_pca_3d(&self.pixels_f[..n], &self.mean_f));
        self.zero_rel_axis3 = v3(&calc_pca_3d(&self.pixels_f[..n], &[0.0; 4]));
        self.mean_rel_axis4 = calc_pca_4d(&self.pixels_f[..n], &self.mean_f);
        for c in 0..4 {
            let it = self.pixels_f[..n].iter().map(move |p| p[c]);
            self.rgba_stats[c].calc_simplified_with_range(n as u32, it);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Tables

struct LsTables {
    /// `g_astc_ls_raw_weights_ise`.
    raw: [[f32; 4]; ASTC_LDR_MAX_RAW_WEIGHTS],
    /// `g_astc_ls_weights_ise` (`basisu_astc_hdr_common.cpp`).
    ise: [[[f32; 4]; 32]; 12],
}

fn ls_tables() -> &'static LsTables {
    static T: OnceLock<LsTables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = LsTables {
            raw: [[0.0; 4]; ASTC_LDR_MAX_RAW_WEIGHTS],
            ise: [[[0.0; 4]; 32]; 12],
        };
        for iw in 0..=64usize {
            let w = iw as f32 * (1.0f32 / 64.0);
            t.raw[iw] = [w * w, (1.0 - w) * w, (1.0 - w) * (1.0 - w), w];
        }
        for range in 0..12usize {
            let n = G_ISE_WEIGHT_LERPS[range][0] as usize;
            for i in 0..n {
                let w = G_ISE_WEIGHT_LERPS[range][1 + i] as f32 * (1.0f32 / 64.0);
                t.ise[range][i] = [w * w, (1.0 - w) * w, (1.0 - w) * (1.0 - w), w];
            }
        }
        t
    })
}

/// `astc_ldr::global_init`: builds the least squares weight tables (also done lazily on use).
pub(crate) fn global_init() {
    let _ = ls_tables();
}

/// `get_ls_weights_ise`: raw [0,64] weights for `BISE_64_LEVELS`, else ISE order weights.
pub(crate) fn get_ls_weights_ise(weight_ise_range: u32) -> &'static [[f32; 4]] {
    let t = ls_tables();
    if weight_ise_range == BISE_64_LEVELS {
        &t.raw
    } else {
        &t.ise[(weight_ise_range as usize).min(11)]
    }
}

// ---------------------------------------------------------------------------------------------
// Blue contraction / endpoint ISE helpers

/// `blue_contract_enc`.
pub(crate) fn blue_contract_enc(
    orig: ColorRgba,
    did_clamp: &mut bool,
    encoded_b: i32,
) -> ColorRgba {
    let tr = orig[0] as i32 * 2 - encoded_b;
    let tg = orig[1] as i32 * 2 - encoded_b;
    if !(0..=255).contains(&tr) || !(0..=255).contains(&tg) {
        *did_clamp = true;
    }
    [
        tr.clamp(0, 255) as u8,
        tg.clamp(0, 255) as u8,
        orig[2],
        orig[3],
    ]
}

/// `blue_contract_dec`.
pub(crate) fn blue_contract_dec(enc_r: i32, enc_g: i32, enc_b: i32, enc_a: i32) -> ColorRgba {
    [
        ((enc_r + enc_b) >> 1) as u8,
        ((enc_g + enc_b) >> 1) as u8,
        enc_b as u8,
        enc_a as u8,
    ]
}

/// `astc_helpers::bit_transfer_signed_enc`.
fn bit_transfer_signed_enc(a: &mut i32, b: &mut i32) {
    let bit = (*b & 0x80) != 0;
    *b = (*b << 1) & 0xFF;
    *a &= 0x3F;
    *a <<= 1;
    if bit {
        *a |= 0x80;
    }
}

/// `astc_helpers::cem8_or_12_used_blue_contraction`.
pub(crate) fn cem8_or_12_used_blue_contraction(
    _cem_index: u32,
    endpoint_vals: &[u8],
    endpoint_ise_index: u32,
) -> bool {
    let tab = &ah::dequant_tables()
        .endpoint_tab(endpoint_ise_index)
        .ise_to_val;
    let d = |i: usize| tab[endpoint_vals[i] as usize] as u32;
    let s0 = d(0) + d(2) + d(4);
    let s1 = d(1) + d(3) + d(5);
    s1 < s0
}

/// `astc_helpers::used_blue_contraction`.
pub(crate) fn used_blue_contraction(
    cem_index: u32,
    endpoint_vals: &[u8],
    endpoint_ise_index: u32,
) -> bool {
    crate::basis::xuastc::endpoints::used_blue_contraction(
        cem_index,
        endpoint_vals,
        endpoint_ise_index,
    )
}

/// `astc_helpers::apply_delta_to_bise_endpoint_val`.
pub(crate) fn apply_delta_to_bise_endpoint_val(
    endpoint_ise_range: u32,
    ise_val: i32,
    delta: i32,
) -> i32 {
    if delta == 0 {
        return ise_val;
    }
    let num_ise_levels = ah::get_ise_levels(endpoint_ise_range) as i32;
    let t = ah::dequant_tables().endpoint_tab(endpoint_ise_range);
    let cur_rank = t.ise_to_rank[ise_val as usize] as i32;
    let new_rank = (cur_rank + delta).clamp(0, num_ise_levels - 1);
    t.rank_to_ise[new_rank as usize] as i32
}

/// `apply_delta_to_bise_weight_val`.
pub(crate) fn apply_delta_to_bise_weight_val(
    weight_ise_range: u32,
    ise_val: i32,
    delta: i32,
) -> i32 {
    if delta == 0 {
        return ise_val;
    }
    let num_ise_levels = ah::get_ise_levels(weight_ise_range) as i32;
    let t = ah::dequant_tables().weight_tab(weight_ise_range);
    let cur_rank = t.ise_to_rank[ise_val as usize] as i32;
    let new_rank = (cur_rank + delta).clamp(0, num_ise_levels - 1);
    t.rank_to_ise[new_rank as usize] as i32
}

/// Nearest ISE symbol to a [0,1] value by checking the quantized neighbor ranks.
fn round_to_endpoint_ise(v: f32, endpoint_ise_range: u32) -> Option<u8> {
    let t = ah::dequant_tables().endpoint_tab(endpoint_ise_range);
    let iv = ftoi((v * 255.0).round()).clamp(0, 255);
    let mut best_err = BIG_FLOAT_VAL;
    let mut ise_index = None;
    for iscale_delta in -1..=1 {
        let trial = apply_delta_to_bise_endpoint_val(
            endpoint_ise_range,
            t.val_to_ise[iv as usize] as i32,
            iscale_delta,
        );
        let dequant_val = t.ise_to_val[trial as usize] as f32 * (1.0f32 / 255.0);
        let dequant_err = (dequant_val - v).abs();
        if dequant_err < best_err {
            best_err = dequant_err;
            ise_index = Some(trial as u8);
        }
    }
    ise_index
}

/// `precise_round_bise_endpoint_val`: v must be [0,1].
pub(crate) fn precise_round_bise_endpoint_val(v: f32, endpoint_ise_range: u32) -> u8 {
    let v = saturate(v);
    round_to_endpoint_ise(v, endpoint_ise_range).unwrap_or(0)
}

/// `cem_encode_ldr_rgb_or_rgba_direct_result`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CemEncodeLdrRgbOrRgbaDirectResult {
    pub is_blue_contracted: bool,
    pub endpoints_are_swapped: bool,
    pub any_degen: bool,
}

/// `cem_encode_ldr_rgb_or_rgba_direct` (CEM 8/12). Endpoints may be swapped.
pub(crate) fn cem_encode_ldr_rgb_or_rgba_direct(
    cem_index: u32,
    endpoint_ise_range: u32,
    l: &ColorRgba,
    h: &ColorRgba,
    endpoint_vals: &mut [u8],
    try_blue_contract: bool,
) -> CemEncodeLdrRgbOrRgbaDirectResult {
    let mut res = CemEncodeLdrRgbOrRgbaDirectResult::default();
    let has_alpha = cem_index == CEM_LDR_RGBA_DIRECT;
    let t = ah::dequant_tables().endpoint_tab(endpoint_ise_range);
    let q = |v: u8| t.val_to_ise[v as usize] as i32;
    let dq = |v: i32| t.ise_to_val[v as usize] as i32;

    let (mut enc_l, mut enc_h) = (*l, *h);
    if try_blue_contract {
        let enc_v4 = q(enc_l[2]);
        let enc_v5 = q(enc_h[2]);
        let (dec_v4, dec_v5) = (dq(enc_v4), dq(enc_v5));
        let mut did_clamp = false;
        // yes, they're swapped in the spec
        enc_l = blue_contract_enc(*h, &mut did_clamp, dec_v5);
        enc_h = blue_contract_enc(*l, &mut did_clamp, dec_v4);
        if !did_clamp {
            res.is_blue_contracted = true;
            res.endpoints_are_swapped = true;
        } else {
            enc_l = *l;
            enc_h = *h;
        }
    }

    let (mut v0, mut v2, mut v4) = (q(enc_l[0]), q(enc_l[1]), q(enc_l[2]));
    let (mut v1, mut v3, mut v5) = (q(enc_h[0]), q(enc_h[1]), q(enc_h[2]));
    let (mut v6, mut v7) = (0, 0);
    if has_alpha {
        v6 = q(enc_l[3]);
        v7 = q(enc_h[3]);
    }

    res.any_degen = (v0 == v1 && l[0] != h[0])
        || (v2 == v3 && l[1] != h[1])
        || (v4 == v5 && l[2] != h[2])
        || (has_alpha && v6 == v7 && l[3] != h[3]);

    let mut s0 = dq(v0) + dq(v2) + dq(v4);
    let mut s1 = dq(v1) + dq(v3) + dq(v5);

    if s1 == s0 && res.is_blue_contracted {
        // if sums are equal we can't use blue contraction at all, so undo it
        enc_l = *l;
        enc_h = *h;
        res.is_blue_contracted = false;
        res.endpoints_are_swapped = false;
        v0 = q(enc_l[0]);
        v2 = q(enc_l[1]);
        v4 = q(enc_l[2]);
        v1 = q(enc_h[0]);
        v3 = q(enc_h[1]);
        v5 = q(enc_h[2]);
        if has_alpha {
            v6 = q(enc_l[3]);
            v7 = q(enc_h[3]);
        }
        s0 = dq(v0) + dq(v2) + dq(v4);
        s1 = dq(v1) + dq(v3) + dq(v5);
    }

    let should_swap = if s1 >= s0 {
        res.is_blue_contracted
    } else {
        !res.is_blue_contracted
    };
    if should_swap {
        res.endpoints_are_swapped = !res.endpoints_are_swapped;
        std::mem::swap(&mut v0, &mut v1);
        std::mem::swap(&mut v2, &mut v3);
        std::mem::swap(&mut v4, &mut v5);
        std::mem::swap(&mut v6, &mut v7);
    }

    endpoint_vals[0] = v0 as u8;
    endpoint_vals[1] = v1 as u8;
    endpoint_vals[2] = v2 as u8;
    endpoint_vals[3] = v3 as u8;
    endpoint_vals[4] = v4 as u8;
    endpoint_vals[5] = v5 as u8;
    if has_alpha {
        endpoint_vals[6] = v6 as u8;
        endpoint_vals[7] = v7 as u8;
    }
    res
}

/// `cem_encode_ldr_rgb_or_rgba_base_scale` (CEM 6/10). Cannot fail.
pub(crate) fn cem_encode_ldr_rgb_or_rgba_base_scale(
    cem_index: u32,
    endpoint_ise_range: u32,
    scale: f32,
    l_a: f32,
    h: &Vec4F,
    endpoint_vals: &mut [u8],
) {
    let has_alpha = cem_index == CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A;
    let total_vals_to_pack = if has_alpha { 6 } else { 4 };
    let mut vals_to_pack = [0f32; 6];
    vals_to_pack[0] = h[0];
    vals_to_pack[1] = h[1];
    vals_to_pack[2] = h[2];
    vals_to_pack[3] = clampf(scale * (256.0f32 / 255.0), 0.0, 1.0);
    if has_alpha {
        vals_to_pack[4] = l_a;
        vals_to_pack[5] = h[3];
    }
    for c in 0..total_vals_to_pack {
        if let Some(v) = round_to_endpoint_ise(vals_to_pack[c], endpoint_ise_range) {
            endpoint_vals[c] = v;
        }
    }
}

/// `rgb_base_offset_res`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RgbBaseOffsetRes {
    pub failed_flag: bool,
    pub used_blue_contraction: bool,
    pub blue_contraction_clamped: bool,
    pub delta_clamped: bool,
    pub any_degen: bool,
    pub endpoints_swapped: bool,
}

/// `basist::astc_ldr_t::pack_base_offset` with all flags:
/// returns (status, blue_contraction_clamped, base_ofs_clamped, endpoints_swapped).
pub(crate) fn pack_base_offset(
    cem_index: u32,
    dst_ise_endpoint_range: u32,
    packed_endpoints: &mut [u8],
    l: &ColorRgba,
    h: &ColorRgba,
    mut use_blue_contraction: bool,
    auto_disable_blue_contraction_if_clamped: bool,
) -> (bool, bool, bool, bool) {
    let mut bc_clamped = false;
    let mut base_ofs_clamped = false;
    let mut endpoints_swapped = false;
    if cem_index != CEM_LDR_RGB_BASE_PLUS_OFFSET && cem_index != CEM_LDR_RGBA_BASE_PLUS_OFFSET {
        return (false, bc_clamped, base_ofs_clamped, endpoints_swapped);
    }
    let (mut pack_l, mut pack_h) = (*l, *h);
    if use_blue_contraction {
        let enc_l = blue_contract_enc(pack_l, &mut bc_clamped, pack_l[2] as i32);
        let enc_h = blue_contract_enc(pack_h, &mut bc_clamped, pack_h[2] as i32);
        if bc_clamped && auto_disable_blue_contraction_if_clamped {
            use_blue_contraction = false;
        } else {
            pack_h = enc_l;
            pack_l = enc_h;
            endpoints_swapped = true;
        }
    }

    let (mut dr, mut dg, mut db, mut da) = (0i32, 0i32, 0i32, 0i32);
    let mut low_clamp = -32;
    for pass in 0..4 {
        let d = |c: usize| pack_h[c] as i32 - pack_l[c] as i32;
        base_ofs_clamped = false;
        let (or, og, ob, oa) = (d(0), d(1), d(2), d(3));
        dr = or.clamp(low_clamp, 31);
        dg = og.clamp(low_clamp, 31);
        db = ob.clamp(low_clamp, 31);
        da = oa.clamp(low_clamp, 31);
        if dr != or || dg != og || db != ob || da != oa {
            base_ofs_clamped = true;
        }
        let s = dr + dg + db;
        if (s < 0) == use_blue_contraction {
            break;
        }
        if s == 0 {
            if db > -32 {
                db -= 1;
            } else if dr > -32 {
                dr -= 1;
            } else if dg > -32 {
                dg -= 1;
            }
            break;
        }
        if pass == 3 {
            break;
        }
        if pass == 1 {
            low_clamp = -31;
        }
        std::mem::swap(&mut pack_l, &mut pack_h);
        endpoints_swapped = !endpoints_swapped;
    }

    let (mut v0, mut v2, mut v4) = (pack_l[0] as i32, pack_l[1] as i32, pack_l[2] as i32);
    let (mut v1, mut v3, mut v5) = (dr, dg, db);
    bit_transfer_signed_enc(&mut v1, &mut v0);
    bit_transfer_signed_enc(&mut v3, &mut v2);
    bit_transfer_signed_enc(&mut v5, &mut v4);
    let mut e = [
        v0 as u8, v1 as u8, v2 as u8, v3 as u8, v4 as u8, v5 as u8, 0, 0,
    ];
    if cem_index == CEM_LDR_RGBA_BASE_PLUS_OFFSET {
        let (mut v6, mut v7) = (pack_l[3] as i32, da);
        bit_transfer_signed_enc(&mut v7, &mut v6);
        e[6] = v6 as u8;
        e[7] = v7 as u8;
    }
    let mut dst = [0u8; MAX_CEM_VALS];
    let n = ah::get_num_cem_values(cem_index) as usize;
    dst[..n].copy_from_slice(&packed_endpoints[..n]);
    let status = requantize_ise_endpoints(
        cem_index,
        BISE_256_LEVELS,
        &e,
        dst_ise_endpoint_range,
        &mut dst,
    );
    packed_endpoints[..n].copy_from_slice(&dst[..n]);
    (status, bc_clamped, base_ofs_clamped, endpoints_swapped)
}

/// `cem_encode_ldr_rgb_or_rgba_base_offset` (CEM 9/13). Endpoints may be swapped.
pub(crate) fn cem_encode_ldr_rgb_or_rgba_base_offset(
    cem_index: u32,
    endpoint_ise_range: u32,
    orig_l: &ColorRgba,
    orig_h: &ColorRgba,
    endpoint_vals: &mut [u8],
    use_blue_contract: bool,
) -> RgbBaseOffsetRes {
    let has_alpha = cem_index == CEM_LDR_RGBA_BASE_PLUS_OFFSET;
    let mut res = RgbBaseOffsetRes::default();
    let (status, bc_clamped, delta_clamped, swapped) = pack_base_offset(
        cem_index,
        endpoint_ise_range,
        endpoint_vals,
        orig_l,
        orig_h,
        use_blue_contract,
        true,
    );
    let _ = bc_clamped;
    res.delta_clamped = delta_clamped;
    res.endpoints_swapped = swapped;
    if !status {
        res.failed_flag = true;
        return res;
    }
    // Verify the actual BC status by unpacking to be absolutely sure
    res.used_blue_contraction = used_blue_contraction(cem_index, endpoint_vals, endpoint_ise_range);
    let (mut dec_l, mut dec_h) = ([0u8; 4], [0u8; 4]);
    decode_endpoints(
        cem_index,
        endpoint_vals,
        endpoint_ise_range,
        &mut dec_l,
        &mut dec_h,
        None,
    );
    let num_comps = if has_alpha { 4 } else { 3 };
    for c in 0..num_comps {
        if orig_l[c] != orig_h[c] {
            continue;
        }
        if dec_l[c] == dec_h[c] {
            res.any_degen = true;
            break;
        }
    }
    res
}

/// `encode_cem0_4` (endpoint-only overload): L or LA direct.
pub(crate) fn encode_cem0_4_endpoints(
    cem_index: u32,
    lum_l: f32,
    lum_h: f32,
    a_l: f32,
    a_h: f32,
    endpoint_ise_range: u32,
    endpoints: &mut [u8],
) {
    endpoints[0] = precise_round_bise_endpoint_val(lum_l, endpoint_ise_range);
    endpoints[1] = precise_round_bise_endpoint_val(lum_h, endpoint_ise_range);
    if cem_index == CEM_LDR_LUM_ALPHA_DIRECT {
        endpoints[2] = precise_round_bise_endpoint_val(a_l, endpoint_ise_range);
        endpoints[3] = precise_round_bise_endpoint_val(a_h, endpoint_ise_range);
    }
}

// ---------------------------------------------------------------------------------------------
// Color decoding

fn expand_endpoints(l: &ColorRgba, h: &ColorRgba, decode_mode_srgb: bool) -> ([u32; 4], [u32; 4]) {
    let mut le = [0u32; 4];
    let mut he = [0u32; 4];
    for c in 0..4 {
        let (a, b) = (l[c] as u32, h[c] as u32);
        if decode_mode_srgb {
            le[c] = (a << 8) | 0x80;
            he[c] = (b << 8) | 0x80;
        } else {
            le[c] = (a << 8) | a;
            he[c] = (b << 8) | b;
        }
    }
    (le, he)
}

fn interp_color(le: &[u32; 4], he: &[u32; 4], w: u32, rgb_only: bool) -> ColorRgba {
    let k = |c: usize| (ah::weight_interpolate(le[c], he[c], w) >> 8) as u8;
    [k(0), k(1), k(2), if rgb_only { 255 } else { k(3) }]
}

/// `get_colors(l, h, weight_ise_index, pColors, decode_mode_srgb)`: colors in ISE order.
pub(crate) fn get_colors(
    l: &ColorRgba,
    h: &ColorRgba,
    weight_ise_index: u32,
    colors: &mut [ColorRgba],
    decode_mode_srgb: bool,
) -> u32 {
    let total_weights = ah::get_ise_levels(weight_ise_index);
    let rgb_only = l[3] == 255 && h[3] == 255;
    let (le, he) = expand_endpoints(l, h, decode_mode_srgb);
    let lerps = &G_ISE_WEIGHT_LERPS[(weight_ise_index as usize).min(11)];
    for i in 0..total_weights as usize {
        colors[i] = interp_color(&le, &he, lerps[1 + i] as u32, rgb_only);
    }
    total_weights
}

/// `get_colors_raw_weights(l, h, pColors, decode_mode_srgb)`: 65 colors (weights 0..=64).
pub(crate) fn get_colors_raw_weights(
    l: &ColorRgba,
    h: &ColorRgba,
    colors: &mut [ColorRgba],
    decode_mode_srgb: bool,
) -> u32 {
    let rgb_only = l[3] == 255 && h[3] == 255;
    let (le, he) = expand_endpoints(l, h, decode_mode_srgb);
    for w in 0..=64u32 {
        colors[w as usize] = interp_color(&le, &he, w, rgb_only);
    }
    ASTC_LDR_MAX_RAW_WEIGHTS as u32
}

/// `decode_endpoints_ise20`: endpoint values are BISE 256 levels.
pub(crate) fn decode_endpoints_ise20(
    cem_index: u32,
    endpoint_vals: &[u8],
    l: &mut ColorRgba,
    h: &mut ColorRgba,
) {
    let mut e = [[0i32; 2]; 4];
    ah::decode_endpoint(cem_index, &mut e, endpoint_vals);
    for c in 0..4 {
        l[c] = e[c][0] as u8;
        h[c] = e[c][1] as u8;
    }
}

/// `decode_endpoints`; `scale` receives the CEM 6/10 scale (`v3 / 256`).
pub(crate) fn decode_endpoints(
    cem_index: u32,
    endpoint_vals: &[u8],
    endpoint_ise_index: u32,
    l: &mut ColorRgba,
    h: &mut ColorRgba,
    scale: Option<&mut f32>,
) {
    let n = (ah::get_num_cem_values(cem_index) as usize).min(MAX_CEM_ENDPOINT_VALS);
    let tab = &ah::dequant_tables()
        .endpoint_tab(endpoint_ise_index)
        .ise_to_val;
    let mut dq = [0u8; MAX_CEM_ENDPOINT_VALS];
    for i in 0..n {
        dq[i] = tab[endpoint_vals[i] as usize];
    }
    decode_endpoints_ise20(cem_index, &dq, l, h);
    if let Some(s) = scale {
        if cem_index == CEM_LDR_RGB_BASE_SCALE || cem_index == CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A {
            *s = dq[3] as f32 * (1.0f32 / 256.0);
        }
    }
}

/// `get_colors(cem_index, pEndpoint_vals, endpoint_ise_index, weight_ise_index, ..)`.
pub(crate) fn get_colors_cem(
    cem_index: u32,
    endpoint_vals: &[u8],
    endpoint_ise_index: u32,
    weight_ise_index: u32,
    colors: &mut [ColorRgba],
    decode_mode_srgb: bool,
) -> u32 {
    let (mut l, mut h) = ([0u8; 4], [0u8; 4]);
    decode_endpoints(
        cem_index,
        endpoint_vals,
        endpoint_ise_index,
        &mut l,
        &mut h,
        None,
    );
    get_colors(&l, &h, weight_ise_index, colors, decode_mode_srgb)
}

/// `get_colors_raw_weights(cem_index, pEndpoint_vals, endpoint_ise_index, ..)`: 65 colors.
pub(crate) fn get_colors_raw_weights_cem(
    cem_index: u32,
    endpoint_vals: &[u8],
    endpoint_ise_index: u32,
    colors: &mut [ColorRgba],
    decode_mode_srgb: bool,
) -> u32 {
    let (mut l, mut h) = ([0u8; 4], [0u8; 4]);
    decode_endpoints(
        cem_index,
        endpoint_vals,
        endpoint_ise_index,
        &mut l,
        &mut h,
        None,
    );
    get_colors_raw_weights(&l, &h, colors, decode_mode_srgb)
}

// ---------------------------------------------------------------------------------------------
// PCA

/// `calc_pca_4D`.
pub(crate) fn calc_pca_4d(pixels: &[Vec4F], mean_f: &Vec4F) -> Vec4F {
    let (mut m00, mut m01, mut m02, mut m03) = (0f32, 0f32, 0f32, 0f32);
    let (mut m11, mut m12, mut m13) = (0f32, 0f32, 0f32);
    let (mut m22, mut m23) = (0f32, 0f32);
    let mut m33 = 0f32;
    for p in pixels {
        let v = sub4(p, mean_f);
        m00 += v[0] * v[0];
        m01 += v[0] * v[1];
        m02 += v[0] * v[2];
        m03 += v[0] * v[3];
        m11 += v[1] * v[1];
        m12 += v[1] * v[2];
        m13 += v[1] * v[3];
        m22 += v[2] * v[2];
        m23 += v[2] * v[3];
        m33 += v[3] * v[3];
    }
    let mut v: Vec4F = [0.6, 0.75, 0.4, 0.75];
    const NUM_POW_ITERS: u32 = 6;
    for i in 0..NUM_POW_ITERS {
        let mut w = [
            m00 * v[0] + m01 * v[1] + m02 * v[2] + m03 * v[3],
            m01 * v[0] + m11 * v[1] + m12 * v[2] + m13 * v[3],
            m02 * v[0] + m12 * v[1] + m22 * v[2] + m23 * v[3],
            m03 * v[0] + m13 * v[1] + m23 * v[2] + m33 * v[3],
        ];
        if i & 1 != 0 {
            normalize4(&mut w);
        }
        v = w;
    }
    if dot4(&v, &v) < 1e-5 {
        v = [0.5; 4];
    }
    v
}

/// `calc_pca_3D` (w = 0).
pub(crate) fn calc_pca_3d(pixels: &[Vec4F], mean_f: &Vec4F) -> Vec4F {
    let mut cov = [0f32; 6];
    for v in pixels {
        let r = v[0] - mean_f[0];
        let g = v[1] - mean_f[1];
        let b = v[2] - mean_f[2];
        cov[0] += r * r;
        cov[1] += r * g;
        cov[2] += r * b;
        cov[3] += g * g;
        cov[4] += g * b;
        cov[5] += b * b;
    }
    let (mut xr, mut xg, mut xb) = (0.9f32, 1.0f32, 0.7f32);
    for _ in 0..3 {
        let mut r = xr * cov[0] + xg * cov[1] + xb * cov[2];
        let mut g = xr * cov[1] + xg * cov[3] + xb * cov[4];
        let mut b = xr * cov[2] + xg * cov[4] + xb * cov[5];
        let mx = |a: f32, b: f32| if a > b { a } else { b };
        let mut m = mx(mx(r.abs(), g.abs()), b.abs());
        if m > 1e-10 {
            m = 1.0 / m;
            r *= m;
            g *= m;
            b *= m;
        }
        xr = r;
        xg = g;
        xb = b;
    }
    let nrm = xr * xr + xg * xg + xb * xb;
    let mut axis: Vec4F = [0.57735027, 0.57735027, 0.57735027, 0.0];
    if nrm > 1e-5 {
        let inv_nrm = 1.0 / nrm.sqrt();
        xr *= inv_nrm;
        xg *= inv_nrm;
        xb *= inv_nrm;
        axis = [xr, xg, xb, 0.0];
    }
    axis
}

// ---------------------------------------------------------------------------------------------
// Solution evaluation

#[inline]
fn square_of_diff(a: u8, b: u8) -> u32 {
    let d = a as i32 - b as i32;
    (d * d) as u32
}

#[inline]
fn all_one(params: &CemEncodeParams) -> bool {
    params.comp_weights == [1, 1, 1, 1]
}

/// High endpoint color: the last color or the 2nd depending on the weight BISE range.
#[inline]
fn high_color_index(total_weights: u32, weight_ise_index: u32) -> usize {
    if total_weights == 65
        || G_ISE_WEIGHT_LERPS[(weight_ise_index as usize).min(11)][total_weights as usize] == 64
    {
        total_weights as usize - 1
    } else {
        1
    }
}

#[inline]
fn val_to_ise_tab(weight_ise_index: u32) -> Option<&'static [u8]> {
    if weight_ise_index <= BISE_32_LEVELS {
        Some(&ah::dequant_tables().weight_tab(weight_ise_index).val_to_ise)
    } else {
        None
    }
}

/// Static `eval_solution(pixel_stats, total_weights, pWeight_colors, ..)`.
fn eval_solution_colors(
    ps: &PixelStats,
    total_weights: u32,
    weight_colors: &[ColorRgba],
    weight_vals: &mut [u8],
    weight_ise_index: u32,
    params: &CemEncodeParams,
) -> u64 {
    let all_one_weights = all_one(params);
    let exhaustive_flag = params.use_exhaustive_weight_eval || !all_one_weights;
    let cw = &params.comp_weights;
    let n = ps.num_pixels as usize;
    let mut total_err = 0u64;

    if let Some(forced) = params.forced_weight_vals0 {
        for c in 0..n {
            let px = &ps.pixels[c];
            let w = forced[c] as usize;
            let wc = &weight_colors[w];
            let err = if all_one_weights {
                square_of_diff(px[0], wc[0])
                    + square_of_diff(px[1], wc[1])
                    + square_of_diff(px[2], wc[2])
                    + square_of_diff(px[3], wc[3])
            } else {
                cw[0] * square_of_diff(px[0], wc[0])
                    + cw[1] * square_of_diff(px[1], wc[1])
                    + cw[2] * square_of_diff(px[2], wc[2])
                    + cw[3] * square_of_diff(px[3], wc[3])
            };
            total_err += err as u64;
            weight_vals[c] = w as u8;
        }
    } else if exhaustive_flag {
        for c in 0..n {
            let px = &ps.pixels[c];
            let mut best_err = u32::MAX;
            let mut best_sel = 0usize;
            for i in 0..total_weights as usize {
                let wc = &weight_colors[i];
                let err = if all_one_weights {
                    square_of_diff(px[0], wc[0])
                        + square_of_diff(px[1], wc[1])
                        + square_of_diff(px[2], wc[2])
                        + square_of_diff(px[3], wc[3])
                } else {
                    cw[0] * square_of_diff(px[0], wc[0])
                        + cw[1] * square_of_diff(px[1], wc[1])
                        + cw[2] * square_of_diff(px[2], wc[2])
                        + cw[3] * square_of_diff(px[3], wc[3])
                };
                if err < best_err {
                    best_err = err;
                    best_sel = i;
                }
            }
            total_err += best_err as u64;
            weight_vals[c] = best_sel as u8;
        }
    } else {
        // low endpoint must be the first color (w=0)
        let e_l = weight_colors[0];
        let e_h = weight_colors[high_color_index(total_weights, weight_ise_index)];
        let dr = e_h[0] as i32 - e_l[0] as i32;
        let dg = e_h[1] as i32 - e_l[1] as i32;
        let db = e_h[2] as i32 - e_l[2] as i32;
        let da = e_h[3] as i32 - e_l[3] as i32;
        // computes weights between [0,64]
        let f = 64.0f32 / ((dr * dr + dg * dg + db * db + da * da) as f32 + 0.00000125);
        let tab = val_to_ise_tab(weight_ise_index);
        let sofs =
            -(e_l[0] as i32 * dr + e_l[1] as i32 * dg + e_l[2] as i32 * db + e_l[3] as i32 * da);
        for c in 0..n {
            let px = &ps.pixels[c];
            let dotv = px[0] as i32 * dr
                + px[1] as i32 * dg
                + px[2] as i32 * db
                + px[3] as i32 * da
                + sofs;
            let sel = ftoi(dotv as f32 * f + 0.5).clamp(0, 64);
            let best_sel = match tab {
                Some(t) => t[sel as usize] as usize,
                None => sel as usize,
            };
            let wc = &weight_colors[best_sel];
            total_err += (square_of_diff(px[0], wc[0])
                + square_of_diff(px[1], wc[1])
                + square_of_diff(px[2], wc[2])
                + square_of_diff(px[3], wc[3])) as u64;
            weight_vals[c] = best_sel as u8;
        }
    }
    total_err
}

/// `eval_solution`: evaluates against raw weights [0,64] (`BISE_64_LEVELS`) or ISE weights.
pub(crate) fn eval_solution(
    pixel_stats: &PixelStats,
    cem_index: u32,
    endpoint_vals: &[u8],
    endpoint_ise_index: u32,
    weight_vals: &mut [u8],
    weight_ise_index: u32,
    params: &CemEncodeParams,
) -> u64 {
    let mut weight_colors = [[0u8; 4]; ASTC_LDR_MAX_RAW_WEIGHTS];
    let num_weights = if weight_ise_index == BISE_64_LEVELS {
        get_colors_raw_weights_cem(
            cem_index,
            endpoint_vals,
            endpoint_ise_index,
            &mut weight_colors,
            params.decode_mode_srgb,
        )
    } else {
        get_colors_cem(
            cem_index,
            endpoint_vals,
            endpoint_ise_index,
            weight_ise_index,
            &mut weight_colors,
            params.decode_mode_srgb,
        )
    };
    eval_solution_colors(
        pixel_stats,
        num_weights,
        &weight_colors,
        weight_vals,
        weight_ise_index,
        params,
    )
}

/// Static `eval_solution_dp(ccs_index, pixel_stats, total_weights, pWeight_colors, ..)`.
fn eval_solution_dp_colors(
    ccs_index: u32,
    ps: &PixelStats,
    total_weights: u32,
    weight_colors: &[ColorRgba],
    weight_vals0: &mut [u8],
    weight_vals1: &mut [u8],
    weight_ise_index: u32,
    params: &CemEncodeParams,
) -> u64 {
    let ccs = ccs_index as usize;
    let all_one_weights = all_one(params);
    let exhaustive_flag = params.use_exhaustive_weight_eval || !all_one_weights;
    let cw = &params.comp_weights;
    let n = ps.num_pixels as usize;
    let mut total_err = 0u64;

    if let Some(forced) = params.forced_weight_vals0 {
        for c in 0..n {
            let px = &ps.pixels[c];
            let w = forced[c] as usize;
            let mut err = 0u32;
            for o in 0..4 {
                if o != ccs {
                    err += cw[o] * square_of_diff(px[o], weight_colors[w][o]);
                }
            }
            total_err += err as u64;
            weight_vals0[c] = w as u8;
        }
    } else if exhaustive_flag {
        for c in 0..n {
            let px = &ps.pixels[c];
            let mut best_err = u32::MAX;
            let mut best_sel = 0usize;
            for i in 0..total_weights as usize {
                let mut err = 0u32;
                for o in 0..4 {
                    if o != ccs {
                        err += cw[o] * square_of_diff(px[o], weight_colors[i][o]);
                    }
                }
                if err < best_err {
                    best_err = err;
                    best_sel = i;
                }
            }
            total_err += best_err as u64;
            weight_vals0[c] = best_sel as u8;
        }
    } else {
        let mut e_l = weight_colors[0];
        let mut e_h = weight_colors[high_color_index(total_weights, weight_ise_index)];
        e_l.swap(ccs, 3);
        e_h.swap(ccs, 3);
        let dr = e_h[0] as i32 - e_l[0] as i32;
        let dg = e_h[1] as i32 - e_l[1] as i32;
        let db = e_h[2] as i32 - e_l[2] as i32;
        let f = 64.0f32 / ((dr * dr + dg * dg + db * db) as f32 + 0.00000125);
        let sofs = -(e_l[0] as i32 * dr + e_l[1] as i32 * dg + e_l[2] as i32 * db);
        let tab = val_to_ise_tab(weight_ise_index);
        for c in 0..n {
            let mut px = ps.pixels[c];
            px.swap(ccs, 3);
            let dotv = px[0] as i32 * dr + px[1] as i32 * dg + px[2] as i32 * db + sofs;
            let sel = ftoi(dotv as f32 * f + 0.5).clamp(0, 64);
            let best_sel = match tab {
                Some(t) => t[sel as usize] as usize,
                None => sel as usize,
            };
            let mut wc = weight_colors[best_sel];
            wc.swap(ccs, 3);
            total_err += (square_of_diff(px[0], wc[0])
                + square_of_diff(px[1], wc[1])
                + square_of_diff(px[2], wc[2])) as u64;
            weight_vals0[c] = best_sel as u8;
        }
    }

    if let Some(forced) = params.forced_weight_vals1 {
        for c in 0..n {
            let px = &ps.pixels[c];
            let w = forced[c] as usize;
            let err = square_of_diff(px[ccs], weight_colors[w][ccs]);
            total_err += (err * cw[ccs]) as u64;
            weight_vals1[c] = w as u8;
        }
    } else if exhaustive_flag {
        for c in 0..n {
            let px = &ps.pixels[c];
            let mut best_err = u32::MAX;
            let mut best_sel = 0usize;
            for i in 0..total_weights as usize {
                let err = square_of_diff(px[ccs], weight_colors[i][ccs]);
                if err < best_err {
                    best_err = err;
                    best_sel = i;
                }
            }
            total_err += (best_err * cw[ccs]) as u64;
            weight_vals1[c] = best_sel as u8;
        }
    } else {
        let e_l = weight_colors[0];
        let e_h = weight_colors[high_color_index(total_weights, weight_ise_index)];
        let la = e_l[ccs] as i32;
        let ha = e_h[ccs] as i32;
        let da = ha - la;
        let f = 64.0f32 / (da as f32 + 0.00000125);
        let tab = val_to_ise_tab(weight_ise_index);
        for c in 0..n {
            let pxa = ps.pixels[c][ccs] as i32;
            let sel = ftoi((pxa - la) as f32 * f + 0.5).clamp(0, 64);
            let best_sel = match tab {
                Some(t) => t[sel as usize] as usize,
                None => sel as usize,
            };
            total_err += square_of_diff(pxa as u8, weight_colors[best_sel][ccs]) as u64;
            weight_vals1[c] = best_sel as u8;
        }
    }
    total_err
}

/// `eval_solution_dp`: dual plane evaluation (plane 1 = `ccs_index`).
pub(crate) fn eval_solution_dp(
    pixel_stats: &PixelStats,
    cem_index: u32,
    ccs_index: u32,
    endpoint_vals: &[u8],
    endpoint_ise_index: u32,
    weight_vals0: &mut [u8],
    weight_vals1: &mut [u8],
    weight_ise_index: u32,
    params: &CemEncodeParams,
) -> u64 {
    let mut weight_colors = [[0u8; 4]; ASTC_LDR_MAX_RAW_WEIGHTS];
    let num_weights = if weight_ise_index == BISE_64_LEVELS {
        get_colors_raw_weights_cem(
            cem_index,
            endpoint_vals,
            endpoint_ise_index,
            &mut weight_colors,
            params.decode_mode_srgb,
        )
    } else {
        get_colors_cem(
            cem_index,
            endpoint_vals,
            endpoint_ise_index,
            weight_ise_index,
            &mut weight_colors,
            params.decode_mode_srgb,
        )
    };
    eval_solution_dp_colors(
        ccs_index,
        pixel_stats,
        num_weights,
        &weight_colors,
        weight_vals0,
        weight_vals1,
        weight_ise_index,
        params,
    )
}

// ---------------------------------------------------------------------------------------------
// Least squares endpoint solvers

/// `compute_least_squares_endpoints_1D`.
pub(crate) fn compute_least_squares_endpoints_1d(
    n: u32,
    selectors: &[u8],
    selector_weights: &[[f32; 4]],
    xl: &mut f32,
    xh: &mut f32,
    vals: &[f32],
    bounds_min: f32,
    bounds_max: f32,
) -> bool {
    let (mut z00, mut z10, mut z11) = (0f32, 0f32, 0f32);
    let (mut q00_r, mut t_r) = (0f32, 0f32);
    for i in 0..n as usize {
        let sw = &selector_weights[selectors[i] as usize];
        z00 += sw[0];
        z10 += sw[1];
        z11 += sw[2];
        let w = sw[3];
        q00_r += w * vals[i];
        t_r += vals[i];
    }
    let q10_r = t_r - q00_r;
    let z01 = z10;
    let mut det = z00 * z11 - z01 * z10;
    if det.abs() < 1e-8 {
        return false;
    }
    det = 1.0 / det;
    let iz00 = z11 * det;
    let iz01 = -z01 * det;
    let iz10 = -z10 * det;
    let iz11 = z00 * det;
    *xh = iz00 * q00_r + iz01 * q10_r;
    *xl = iz10 * q00_r + iz11 * q10_r;
    let (mut l, mut h) = (saturate(*xl), saturate(*xh));
    if bounds_min == bounds_max {
        l = bounds_min;
        h = bounds_max;
    }
    *xl = l;
    *xh = h;
    true
}

/// Shared N-D solver (`compute_least_squares_endpoints_2D/3D/4D`): `nc` solved components,
/// components >= `nc` of `xl`/`xh` are zeroed when `zero_rest`.
fn compute_least_squares_endpoints_nd<const N: usize>(
    nc: usize,
    n: u32,
    selectors: &[u8],
    selector_weights: &[[f32; 4]],
    xl: &mut [f32; N],
    xh: &mut [f32; N],
    colors: &[[f32; N]],
    bounds_min: &[f32; N],
    bounds_max: &[f32; N],
    zero_rest: bool,
) -> bool {
    let (mut z00, mut z10, mut z11) = (0f32, 0f32, 0f32);
    let mut q00 = [0f32; 4];
    let mut t = [0f32; 4];
    for i in 0..n as usize {
        let sw = &selector_weights[selectors[i] as usize];
        z00 += sw[0];
        z10 += sw[1];
        z11 += sw[2];
        let w = sw[3];
        for c in 0..nc {
            q00[c] += w * colors[i][c];
            t[c] += colors[i][c];
        }
    }
    let mut q10 = [0f32; 4];
    for c in 0..nc {
        q10[c] = t[c] - q00[c];
    }
    let z01 = z10;
    let mut det = z00 * z11 - z01 * z10;
    if det.abs() < 1e-8 {
        return false;
    }
    det = 1.0 / det;
    let iz00 = z11 * det;
    let iz01 = -z01 * det;
    let iz10 = -z10 * det;
    let iz11 = z00 * det;
    for c in 0..nc {
        xh[c] = iz00 * q00[c] + iz01 * q10[c];
        xl[c] = iz10 * q00[c] + iz11 * q10[c];
    }
    if zero_rest {
        for c in nc..N {
            xh[c] = 0.0;
            xl[c] = 0.0;
        }
    }
    for c in 0..nc {
        let (mut l, mut h) = (saturate(xl[c]), saturate(xh[c]));
        if bounds_min[c] == bounds_max[c] {
            l = bounds_min[c];
            h = bounds_max[c];
        }
        xl[c] = l;
        xh[c] = h;
    }
    true
}

/// `compute_least_squares_endpoints_2D`.
pub(crate) fn compute_least_squares_endpoints_2d(
    n: u32,
    selectors: &[u8],
    selector_weights: &[[f32; 4]],
    xl: &mut [f32; 2],
    xh: &mut [f32; 2],
    colors: &[[f32; 2]],
    bounds_min: &[f32; 2],
    bounds_max: &[f32; 2],
) -> bool {
    compute_least_squares_endpoints_nd(
        2,
        n,
        selectors,
        selector_weights,
        xl,
        xh,
        colors,
        bounds_min,
        bounds_max,
        false,
    )
}

/// `compute_least_squares_endpoints_3D` (w of the results = 0).
pub(crate) fn compute_least_squares_endpoints_3d(
    n: u32,
    selectors: &[u8],
    selector_weights: &[[f32; 4]],
    xl: &mut Vec4F,
    xh: &mut Vec4F,
    colors: &[Vec4F],
    bounds_min: &Vec4F,
    bounds_max: &Vec4F,
) -> bool {
    compute_least_squares_endpoints_nd(
        3,
        n,
        selectors,
        selector_weights,
        xl,
        xh,
        colors,
        bounds_min,
        bounds_max,
        true,
    )
}

/// `compute_least_squares_endpoints_4D`.
pub(crate) fn compute_least_squares_endpoints_4d(
    n: u32,
    selectors: &[u8],
    selector_weights: &[[f32; 4]],
    xl: &mut Vec4F,
    xh: &mut Vec4F,
    colors: &[Vec4F],
    bounds_min: &Vec4F,
    bounds_max: &Vec4F,
) -> bool {
    compute_least_squares_endpoints_nd(
        4,
        n,
        selectors,
        selector_weights,
        xl,
        xh,
        colors,
        bounds_min,
        bounds_max,
        false,
    )
}
