// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of `basisu_astc_ldr_fencode.cpp` / `.h` (namespace `astc_ldrf`): the fast "astcf"
//! ASTC LDR block encoder. Single subset blocks are shortlisted analytically (PCA line fit
//! errors, endpoint/weight quantization noise estimates and DCT energy lost to weight grid
//! downsampling) and then encoded with least squares refinement; 2/3 subset blocks reuse a
//! "carrier" single subset configuration with partitions picked by bitmask distance (2
//! subsets) or k-means + LSH lookup (3 subsets).
//!
//! Float math follows the C++ operation order exactly (`f32`, `f64` where C++ uses `double`).
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::collapsible_else_if
)]

use super::dct_enc::Dct2f;
use super::fencode_tables::{
    AB_SUM_TAB, INTERVALS_RECIP, PSEUDOINVERSE_COEFFS, SINGLE_SUBSET_CONFIGS_LA,
    SINGLE_SUBSET_CONFIGS_RGBA, THREE_SUBSET_CONFIGS_RGBA, TWO_SUBSET_CONFIGS_RGBA,
};
use super::grid_data::{AstcBlockGridData, find_astc_block_grid_data};
use super::partitions::{PartitionPatternVec, PartitionsData, texel_partitions};
use crate::astc::helpers::{
    self as ah, BISE_32_LEVELS, BISE_64_LEVELS, BISE_256_LEVELS, LogAstcBlock, MAX_BLOCK_PIXELS,
    MAX_GRID_WEIGHTS, WeightedSample,
};
use crate::basis::ll::bc7f_astc::{
    bit_transfer_signed_enc, blue_contract_enc, requantize_from_256,
};
use std::sync::{Arc, OnceLock};

const SMALL_EPS: f32 = 0.001;
const TINY_EPS: f32 = 0.00001;
const BIG_FLOAT_VAL: f32 = 1e+30;

/// `MAX_CANDIDATES`.
pub(crate) const MAX_CANDIDATES: u32 = 512;
/// `MAX_UNIQUE_2SUBSET_PATS`.
pub(crate) const MAX_UNIQUE_2SUBSET_PATS: usize = 838;
/// `MAX_UNIQUE_3SUBSET_PATS`.
pub(crate) const MAX_UNIQUE_3SUBSET_PATS: usize = 626;
/// `NUM_DOT_THRESH_FRACTS`.
pub(crate) const NUM_DOT_THRESH_FRACTS: u32 = 6;

/// `cUserModeISEValues`.
pub(crate) const USER_MODE_ISE_VALUES: u8 = 0;
/// `cUserModeRankValues`.
pub(crate) const USER_MODE_RANK_VALUES: u8 = 1;

// cem_index
const CEM6: usize = 0;
const CEM8: usize = 1;
const CEM10: usize = 2;
const CEM12: usize = 3;
const CEM_TOTAL_INDICES: usize = 4;

const DEF_SCALE_WEIGHT: f32 = 1.0;
const DEF_QUANT_WEIGHT: f32 = 1.0;

// ------------------------------------------------------------------------------------------
// Small helpers (basisu math semantics).

/// `basisu::minimum` (`(a < b) ? a : b`).
#[inline]
fn fmin(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `basisu::maximum` (`(a > b) ? a : b`).
#[inline]
fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `basisu::clamp`.
#[inline]
fn fclamp(v: f32, l: f32, h: f32) -> f32 {
    if v < l {
        l
    } else if v > h {
        h
    } else {
        v
    }
}

#[inline]
fn sq(v: f32) -> f32 {
    v * v
}

/// `(int)x` as compiled for x86-64 (`cvttss2si`; out of range / NaN gives `INT_MIN`).
#[inline]
fn f2i(x: f32) -> i32 {
    if (-2147483648.0..2147483648.0).contains(&x) {
        x as i32
    } else {
        i32::MIN
    }
}

/// `fast_roundf_int`.
#[inline]
fn fast_roundf_int(x: f32) -> i32 {
    if x >= 0.0 { f2i(x + 0.5) } else { f2i(x - 0.5) }
}

/// `clamp_flag`.
#[inline]
fn clamp_flag(v: f32, l: f32, h: f32, did_clamp: &mut bool) -> f32 {
    if v < l {
        *did_clamp = true;
        l
    } else if v > h {
        *did_clamp = true;
        h
    } else {
        v
    }
}

#[inline]
fn vec4_dot(a: &[f32], b: &[f32]) -> f32 {
    (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]) + (a[3] * b[3])
}

#[inline]
fn vec3_dot(a: &[f32], b: &[f32]) -> f32 {
    (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])
}

/// `vec_normalize(pVec, num_comps)`.
fn vec_normalize(v: &mut [f32; 4], num_comps: usize) -> f32 {
    let mut len2 = 0.0f32;
    for i in 0..num_comps {
        len2 += v[i] * v[i];
    }
    let scale = if len2 > TINY_EPS {
        1.0 / len2.sqrt()
    } else {
        0.0
    };
    for i in 0..num_comps {
        v[i] *= scale;
    }
    len2
}

#[inline]
fn vec3_squared_dist(a: &[f32], b: &[f32]) -> f32 {
    sq(a[0] - b[0]) + sq(a[1] - b[1]) + sq(a[2] - b[2])
}

#[inline]
fn vec4_squared_dist(a: &[f32], b: &[f32]) -> f32 {
    sq(a[0] - b[0]) + sq(a[1] - b[1]) + sq(a[2] - b[2]) + sq(a[3] - b[3])
}

#[inline]
fn extract_bits(v: u32, ofs: u32, len: u32) -> u32 {
    (v >> ofs) & ((1u32 << len) - 1)
}

/// `popcount64`.
#[inline]
pub(crate) fn popcount64(x: u64) -> u32 {
    x.count_ones()
}

// ------------------------------------------------------------------------------------------
// Rank / value lookup tables.

struct RankTabs {
    /// `get_endpoint_tab(range).get_rank_to_val(rank)`, `[range][rank]`.
    ep_rank_to_val: [[u8; 256]; 21],
    /// `get_endpoint_tab(range).get_val_to_rank(val)`.
    ep_val_to_rank: [[u8; 256]; 21],
    /// `get_weight_tab(range).get_rank_to_val(rank)`, `[range][rank]` (range <= 11).
    w_rank_to_val: [[u8; 32]; 12],
    /// `get_weight_tab(range).get_val_to_rank(val)`, `[range][0..=64]`.
    w_val_to_rank: [[u8; 65]; 12],
}

fn rank_tabs() -> &'static RankTabs {
    static T: OnceLock<RankTabs> = OnceLock::new();
    T.get_or_init(|| {
        let dt = ah::dequant_tables();
        let mut t = RankTabs {
            ep_rank_to_val: [[0; 256]; 21],
            ep_val_to_rank: [[0; 256]; 21],
            w_rank_to_val: [[0; 32]; 12],
            w_val_to_rank: [[0; 65]; 12],
        };
        for r in ah::FIRST_VALID_ENDPOINT_ISE_RANGE..=ah::LAST_VALID_ENDPOINT_ISE_RANGE {
            let tab = dt.endpoint_tab(r);
            for (rank, &ise) in tab.rank_to_ise.iter().enumerate().take(256) {
                t.ep_rank_to_val[r as usize][rank] =
                    tab.ise_to_val.get(ise as usize).copied().unwrap_or(0);
            }
            for (v, &ise) in tab.val_to_ise.iter().enumerate().take(256) {
                t.ep_val_to_rank[r as usize][v] =
                    tab.ise_to_rank.get(ise as usize).copied().unwrap_or(0);
            }
        }
        for r in 0..=BISE_32_LEVELS {
            let tab = dt.weight_tab(r);
            for (rank, &ise) in tab.rank_to_ise.iter().enumerate().take(32) {
                t.w_rank_to_val[r as usize][rank] =
                    tab.ise_to_val.get(ise as usize).copied().unwrap_or(0);
            }
            for (v, &ise) in tab.val_to_ise.iter().enumerate().take(65) {
                t.w_val_to_rank[r as usize][v] =
                    tab.ise_to_rank.get(ise as usize).copied().unwrap_or(0);
            }
        }
        t
    })
}

#[inline]
fn ep_rank_to_val(range: u32, rank: u32) -> u32 {
    u32::from(rank_tabs().ep_rank_to_val[(range as usize).min(20)][(rank & 255) as usize])
}

#[inline]
fn ep_val_to_rank(range: u32, val: u32) -> u32 {
    u32::from(rank_tabs().ep_val_to_rank[(range as usize).min(20)][(val & 255) as usize])
}

#[inline]
fn w_rank_to_val(range: u32, rank: u32) -> u32 {
    u32::from(rank_tabs().w_rank_to_val[(range as usize).min(11)][(rank & 31) as usize])
}

#[inline]
fn w_val_to_rank(range: u32, val: u32) -> u32 {
    u32::from(rank_tabs().w_val_to_rank[(range as usize).min(11)][(val as usize).min(64)])
}

// ------------------------------------------------------------------------------------------
// bitmask192

/// `bitmask192`: a = bits 0-63, b = 64-127, c = 128-191.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Bitmask192 {
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

impl Bitmask192 {
    pub const fn new(a: u64, b: u64, c: u64) -> Self {
        Self { a, b, c }
    }
    pub const fn zero() -> Self {
        Self::new(0, 0, 0)
    }
    pub const fn all() -> Self {
        Self::new(u64::MAX, u64::MAX, u64::MAX)
    }
    pub fn is_zero(&self) -> bool {
        (self.a | self.b | self.c) == 0
    }
    pub fn popcount(&self) -> u32 {
        popcount64(self.a) + popcount64(self.b) + popcount64(self.c)
    }
    pub fn not(&self) -> Self {
        Self::new(!self.a, !self.b, !self.c)
    }
    pub fn and(&self, r: &Self) -> Self {
        Self::new(self.a & r.a, self.b & r.b, self.c & r.c)
    }
    pub fn or(&self, r: &Self) -> Self {
        Self::new(self.a | r.a, self.b | r.b, self.c | r.c)
    }
    pub fn xor(&self, r: &Self) -> Self {
        Self::new(self.a ^ r.a, self.b ^ r.b, self.c ^ r.c)
    }
    /// `operator<` (c is the most significant qword).
    pub fn less(&self, r: &Self) -> bool {
        (self.c, self.b, self.a) < (r.c, r.b, r.a)
    }
    pub fn intersects(&self, r: &Self) -> bool {
        ((self.a & r.a) | (self.b & r.b) | (self.c & r.c)) != 0
    }
    pub fn contains_all(&self, r: &Self) -> bool {
        (self.a & r.a) == r.a && (self.b & r.b) == r.b && (self.c & r.c) == r.c
    }
    /// `set_bit(index, bit_val)`.
    pub fn set_bit(&mut self, index: u32, bit_val: u32) {
        let (w, s) = match index {
            0..=63 => (&mut self.a, index),
            64..=127 => (&mut self.b, index - 64),
            _ => (&mut self.c, (index - 128) & 63),
        };
        *w &= !(1u64 << s);
        *w |= u64::from(bit_val & 1) << s;
    }
    /// `is_bit_set(index)`.
    pub fn is_bit_set(&self, index: u32) -> bool {
        match index {
            0..=63 => (self.a >> index) & 1 != 0,
            64..=127 => (self.b >> (index - 64)) & 1 != 0,
            _ => (self.c >> ((index - 128) & 63)) & 1 != 0,
        }
    }
    /// `lsb_mask(num_bits)`.
    pub fn lsb_mask(num_bits: u32) -> Self {
        match num_bits {
            0 => Self::zero(),
            1..=63 => Self::new((1u64 << num_bits) - 1, 0, 0),
            64 => Self::new(u64::MAX, 0, 0),
            65..=127 => Self::new(u64::MAX, (1u64 << (num_bits - 64)) - 1, 0),
            128 => Self::new(u64::MAX, u64::MAX, 0),
            129..=191 => Self::new(u64::MAX, u64::MAX, (1u64 << (num_bits - 128)) - 1),
            _ => Self::all(),
        }
    }
}

// ------------------------------------------------------------------------------------------
// pixelbuf

const PB_ROW: usize = 16;
const PB_COMP: usize = 16 * 16;
const PB_SIZE: usize = PB_COMP * 4;

/// `pixelbuf` (owning its `PIXELBUF_SIZE_IN_FLOATS` floats, `[c][y][x]` with pitches 16/256).
#[derive(Clone)]
pub(crate) struct PixelBuf {
    pub buf: [f32; PB_SIZE],
    pub width: u32,
    pub height: u32,
}

impl PixelBuf {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            buf: [0.0; PB_SIZE],
            width,
            height,
        }
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32, c: u32) -> f32 {
        self.buf[x as usize + y as usize * PB_ROW + c as usize * PB_COMP]
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, c: u32, v: f32) {
        self.buf[x as usize + y as usize * PB_ROW + c as usize * PB_COMP] = v;
    }

    #[inline]
    fn pixel4(&self, x: u32, y: u32) -> [f32; 4] {
        [
            self.get(x, y, 0),
            self.get(x, y, 1),
            self.get(x, y, 2),
            self.get(x, y, 3),
        ]
    }

    /// `pixelbuf_load_block(pbuf, src_img, 0, 0, num_comps)` from tightly packed RGBA pixels.
    fn load_block(&mut self, pixels: &[[u8; 4]], num_comps: u32) {
        let (w, h) = (self.width, self.height);
        for y in 0..h {
            for x in 0..w {
                let p = pixels[(x + y * w) as usize];
                self.set(x, y, 0, f32::from(p[0]));
                self.set(x, y, 1, f32::from(p[1]));
                self.set(x, y, 2, f32::from(p[2]));
                let a = if num_comps == 3 {
                    255.0
                } else {
                    f32::from(p[3])
                };
                self.set(x, y, 3, a);
            }
        }
    }

    /// `pixelbuf_set_comp_to_val`.
    fn set_comp_to_val(&mut self, comp: u32, val: f32) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.set(x, y, comp, val);
            }
        }
    }

    /// `pixelbuf_swap_comp_with_alpha`.
    fn swap_comp_with_alpha(&mut self, comp: u32) {
        if comp >= 3 {
            return;
        }
        for y in 0..self.height {
            for x in 0..self.width {
                let c = self.get(x, y, comp);
                let a = self.get(x, y, 3);
                self.set(x, y, comp, a);
                self.set(x, y, 3, c);
            }
        }
    }
}

#[inline]
fn does_cem_have_alpha(cem: u32) -> bool {
    cem == 4 || cem >= 10
}

#[inline]
fn get_num_cem_chans(cem: u32) -> u32 {
    if cem == 4 || cem >= 10 { 4 } else { 3 }
}

#[inline]
fn is_cem_6_or_10(cem: u32) -> bool {
    cem == 6 || cem == 10
}

#[inline]
fn is_cem_8_or_12(cem: u32) -> bool {
    cem == 8 || cem == 12
}

#[inline]
fn is_cem_9_or_13(cem: u32) -> bool {
    cem == 9 || cem == 13
}

/// `is_lblock_ise`.
#[inline]
pub(crate) fn is_lblock_ise(b: &LogAstcBlock) -> bool {
    b.user_mode == USER_MODE_ISE_VALUES
}

fn total_endpoint_vals(b: &LogAstcBlock) -> usize {
    (0..usize::from(b.num_partitions).min(4))
        .map(|i| ah::get_num_cem_values(u32::from(b.color_endpoint_modes[i])) as usize)
        .sum::<usize>()
        .min(ah::MAX_ENDPOINTS as usize)
}

fn total_weights(b: &LogAstcBlock) -> usize {
    ((if b.dual_plane { 2 } else { 1 }) * usize::from(b.grid_width) * usize::from(b.grid_height))
        .min(MAX_GRID_WEIGHTS as usize)
}

/// `convert_rank_lblock_to_ise`.
pub(crate) fn convert_rank_lblock_to_ise(b: &mut LogAstcBlock) {
    if b.solid_color_flag_ldr || is_lblock_ise(b) {
        return;
    }
    let dt = ah::dequant_tables();
    let et = &dt.endpoint_tab(u32::from(b.endpoint_ise_range)).rank_to_ise;
    for i in 0..total_endpoint_vals(b) {
        b.endpoints[i] = et.get(b.endpoints[i] as usize).copied().unwrap_or(0);
    }
    let wt = &dt.weight_tab(u32::from(b.weight_ise_range)).rank_to_ise;
    for i in 0..total_weights(b) {
        b.weights[i] = wt.get(b.weights[i] as usize).copied().unwrap_or(0);
    }
    b.user_mode = USER_MODE_ISE_VALUES;
}

/// `convert_ise_lblock_to_rank`.
pub(crate) fn convert_ise_lblock_to_rank(b: &mut LogAstcBlock) {
    if b.solid_color_flag_ldr || !is_lblock_ise(b) {
        return;
    }
    let dt = ah::dequant_tables();
    let et = &dt.endpoint_tab(u32::from(b.endpoint_ise_range)).ise_to_rank;
    for i in 0..total_endpoint_vals(b) {
        b.endpoints[i] = et.get(b.endpoints[i] as usize).copied().unwrap_or(0);
    }
    let wt = &dt.weight_tab(u32::from(b.weight_ise_range)).ise_to_rank;
    for i in 0..total_weights(b) {
        b.weights[i] = wt.get(b.weights[i] as usize).copied().unwrap_or(0);
    }
    b.user_mode = USER_MODE_RANK_VALUES;
}

// ------------------------------------------------------------------------------------------
// XUASTC LDR block decoding (`astc_helpers::decode_block_xuastc_ldr`,
// `xuastc_ldr_block_decoder`).

/// `astc_helpers::set_ldr_solid_block`.
pub(crate) fn set_ldr_solid_block(blk: &mut LogAstcBlock, r: u32, g: u32, b: u32, a: u32) {
    blk.clear();
    blk.solid_color_flag_ldr = true;
    for (c, v) in [r, g, b, a].into_iter().enumerate() {
        let v = v & 255;
        blk.set_solid_color(c, (v | (v << 8)) as u16);
    }
}

/// `astc_helpers::is_block_xuastc_ldr`.
pub(crate) fn is_block_xuastc_ldr(b: &LogAstcBlock) -> bool {
    if b.error_flag {
        return false;
    }
    if b.solid_color_flag_ldr {
        return true;
    }
    if b.solid_color_flag_hdr || b.num_partitions > 3 || (b.dual_plane && b.num_partitions > 1) {
        return false;
    }
    for i in 1..usize::from(b.num_partitions) {
        if b.color_endpoint_modes[0] != b.color_endpoint_modes[i] {
            return false;
        }
    }
    matches!(
        u32::from(b.color_endpoint_modes[0]),
        ah::CEM_LDR_LUM_DIRECT
            | ah::CEM_LDR_LUM_ALPHA_DIRECT
            | ah::CEM_LDR_RGB_BASE_SCALE
            | ah::CEM_LDR_RGB_DIRECT
            | ah::CEM_LDR_RGB_BASE_PLUS_OFFSET
            | ah::CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A
            | ah::CEM_LDR_RGBA_DIRECT
            | ah::CEM_LDR_RGBA_BASE_PLUS_OFFSET
    )
}

/// `astc_helpers::upsample_weight_grid_xuastc_ldr` (single plane variant).
pub(crate) fn upsample_weight_grid_xuastc_ldr(
    bx: u32,
    by: u32,
    wx: u32,
    wy: u32,
    src: &[u8],
    dst: &mut [u8],
) {
    let scale_x = (1024 + bx / 2) / (bx - 1);
    let scale_y = (1024 + by / 2) / (by - 1);
    let gy_inc = scale_y * (wy - 1);
    let gx_inc = scale_x * (wx - 1);
    let mut gyu = 32u32;
    for ty in 0..by {
        let gy = gyu >> 6;
        gyu += gy_inc;
        let (jy, fy) = (gy >> 4, gy & 0xf);
        let mut gxu = 32u32;
        for tx in 0..bx {
            let gx = gxu >> 6;
            gxu += gx_inc;
            let (jx, fx) = (gx >> 4, gx & 0xf);
            let w11 = (fx * fy + 8) >> 4;
            let w10 = fy - w11;
            let w01 = fx - w11;
            let w00 = (16 - fx - fy).wrapping_add(w11);
            let (sx, sy) = (jx, jy);
            let s = |i: u32| u32::from(src[i as usize]);
            let mut total = 8u32;
            if w00 != 0 {
                total += s(sx + sy * wx) * w00;
            }
            if w01 != 0 {
                total += s(sx + 1 + sy * wx) * w01;
            }
            if w10 != 0 {
                total += s(sx + (sy + 1) * wx) * w10;
            }
            if w11 != 0 {
                total += s(sx + 1 + (sy + 1) * wx) * w11;
            }
            dst[(tx + ty * bx) as usize] = (total >> 4) as u8;
        }
    }
}

/// Decoded endpoints of a non-solid XUASTC LDR block, expanded to 16 bits.
struct DecodedEndpoints {
    /// `[subset][comp][l/h]`.
    e: [[[i32; 2]; 4]; 3],
}

fn decode_block_endpoints(blk: &LogAstcBlock, srgb: bool) -> Option<DecodedEndpoints> {
    let erange = u32::from(blk.endpoint_ise_range);
    let ise_to_val = &ah::dequant_tables().endpoint_tab(erange).ise_to_val;
    let cem = u32::from(blk.color_endpoint_modes[0]);
    let n = ah::get_num_cem_values(cem) as usize;
    let mut out = DecodedEndpoints {
        e: [[[0; 2]; 4]; 3],
    };
    let nparts = usize::from(blk.num_partitions);
    if !(1..=3).contains(&nparts) || n * nparts > ah::MAX_ENDPOINTS as usize {
        return None;
    }
    for s in 0..nparts {
        let mut dq = [0u8; 8];
        for i in 0..n {
            dq[i] = *ise_to_val.get(blk.endpoints[s * n + i] as usize)?;
        }
        let mut e = [[0i32; 2]; 4];
        ah::decode_endpoint(cem, &mut e, &dq);
        for c in 0..4 {
            for lh in 0..2 {
                let v = e[c][lh];
                out.e[s][c][lh] = if srgb { (v << 8) | 0x80 } else { (v << 8) | v };
            }
        }
    }
    Some(out)
}

#[inline]
fn weight_interpolate(l: i32, h: i32, w: i32) -> u32 {
    ((l * (64 - w) + h * w + 32) >> 6) as u32
}

/// `astc_helpers::decode_block_xuastc_ldr` (8-bit output, sRGB or linear decode mode).
pub(crate) fn decode_block_xuastc_ldr(
    blk: &LogAstcBlock,
    out: &mut [[u8; 4]],
    bw: u32,
    bh: u32,
    srgb: bool,
) -> bool {
    let n = (bw * bh) as usize;
    if out.len() < n || blk.error_flag || blk.solid_color_flag_hdr {
        return false;
    }
    if blk.solid_color_flag_ldr {
        let c = [0, 1, 2, 3].map(|i| (blk.solid_color(i) >> 8) as u8);
        out[..n].fill(c);
        return true;
    }
    let (gw, gh) = (u32::from(blk.grid_width), u32::from(blk.grid_height));
    if gw < 2 || gh < 2 || gw > bw || gh > bh || gw * gh > MAX_GRID_WEIGHTS {
        return false;
    }
    let Some(ep) = decode_block_endpoints(blk, srgb) else {
        return false;
    };
    let wq = &ah::dequant_tables()
        .weight_tab(u32::from(blk.weight_ise_range))
        .ise_to_val;
    let num_grid = (gw * gh) as usize;
    let mut dq = [[0u8; 144]; 2];
    let mut up = [[0u8; 144]; 2];
    let num_planes = if blk.dual_plane { 2 } else { 1 };
    for i in 0..num_grid * num_planes {
        let Some(&v) = wq.get(blk.weights[i] as usize) else {
            return false;
        };
        dq[i % num_planes][i / num_planes] = v;
    }
    let weights: &[[u8; 144]; 2] = if gw < bw || gh < bh {
        for p in 0..num_planes {
            upsample_weight_grid_xuastc_ldr(bw, bh, gw, gh, &dq[p], &mut up[p]);
        }
        &up
    } else {
        &dq
    };
    let ccs = if blk.dual_plane {
        usize::from(blk.color_component_selector)
    } else {
        usize::MAX
    };
    let part = &texel_partitions()[(blk.partition_id & 1023) as usize];
    let mut part_shift = if blk.num_partitions == 3 { 2 } else { 0 };
    part_shift += u32::from(n >= 31) * 4;
    for y in 0..bh {
        for x in 0..bw {
            let i = (x + y * bw) as usize;
            let subset = if blk.num_partitions > 1 {
                ((u32::from(part[(x + y * 12) as usize]) >> part_shift) & 3).min(2) as usize
            } else {
                0
            };
            let w0 = i32::from(weights[0][i]);
            let w1 = i32::from(weights[1][i]);
            let e = &ep.e[subset];
            let mut c = [0u8; 4];
            for (ch, cv) in c.iter_mut().enumerate() {
                let w = if ch == ccs { w1 } else { w0 };
                *cv = (weight_interpolate(e[ch][0], e[ch][1], w) >> 8) as u8;
            }
            out[i] = c;
        }
    }
    true
}

/// `astc_helpers::xuastc_ldr_block_decoder`: decodes single texels on the fly while the
/// block's weights change (the weights are passed to [`Self::decode_texel`]).
struct XuastcLdrBlockDecoder<'a> {
    solid: Option<[u8; 4]>,
    blk_width: u32,
    grid_width: u32,
    dual_plane: bool,
    ccs: usize,
    num_partitions: u8,
    upsample_weights: &'a [WeightedSample],
    weight_dequant: &'static [u8],
    ep: DecodedEndpoints,
    part: &'static [u8; 144],
    part_shift: u32,
    is_upsampling: bool,
}

impl<'a> XuastcLdrBlockDecoder<'a> {
    fn new(
        blk: &LogAstcBlock,
        bw: u32,
        bh: u32,
        srgb: bool,
        upsample_weights: &'a [WeightedSample],
    ) -> Option<Self> {
        let part = &texel_partitions()[(blk.partition_id & 1023) as usize];
        if blk.solid_color_flag_ldr {
            return Some(Self {
                solid: Some([0, 1, 2, 3].map(|i| (blk.solid_color(i) >> 8) as u8)),
                blk_width: bw,
                grid_width: 0,
                dual_plane: false,
                ccs: usize::MAX,
                num_partitions: 1,
                upsample_weights,
                weight_dequant: &[],
                ep: DecodedEndpoints {
                    e: [[[0; 2]; 4]; 3],
                },
                part,
                part_shift: 0,
                is_upsampling: false,
            });
        }
        let ep = decode_block_endpoints(blk, srgb)?;
        let mut part_shift = if blk.num_partitions == 3 { 2 } else { 0 };
        part_shift += u32::from(bw * bh >= 31) * 4;
        Some(Self {
            solid: None,
            blk_width: bw,
            grid_width: u32::from(blk.grid_width),
            dual_plane: blk.dual_plane,
            ccs: usize::from(blk.color_component_selector),
            num_partitions: blk.num_partitions,
            upsample_weights,
            weight_dequant: &ah::dequant_tables()
                .weight_tab(u32::from(blk.weight_ise_range))
                .ise_to_val,
            ep,
            part,
            part_shift,
            is_upsampling: u32::from(blk.grid_width) != bw || u32::from(blk.grid_height) != bh,
        })
    }

    #[inline]
    fn dq(&self, w: u8) -> i32 {
        i32::from(self.weight_dequant.get(w as usize).copied().unwrap_or(0))
    }

    /// `decode_texel(x, y, dst_c)`.
    #[allow(unused_assignments)]
    fn decode_texel(&self, weights: &[u8; 64], x: u32, y: u32) -> [u8; 4] {
        if let Some(c) = self.solid {
            return c;
        }
        let texel_index = (x + y * self.blk_width) as usize;
        let gw = self.grid_width as usize;
        let (mut w0, mut w1) = (0i32, 0i32);
        if !self.is_upsampling {
            if self.dual_plane {
                w0 = self.dq(weights[texel_index * 2]);
                w1 = self.dq(weights[texel_index * 2 + 1]);
            } else {
                w0 = self.dq(weights[texel_index]);
            }
        } else {
            let ws = &self.upsample_weights[texel_index];
            let [[a00, a01], [a10, a11]] = ws.weights.map(|r| r.map(i32::from));
            let idx0 = usize::from(ws.src_x) + usize::from(ws.src_y) * gw;
            if !self.dual_plane {
                w0 = 8 + a00 * self.dq(weights[idx0]);
                if a01 != 0 {
                    w0 += a01 * self.dq(weights[idx0 + 1]);
                }
                if a10 != 0 {
                    w0 += a10 * self.dq(weights[idx0 + gw]);
                }
                if a11 != 0 {
                    w0 += a11 * self.dq(weights[idx0 + 1 + gw]);
                }
                w0 >>= 4;
                w1 >>= 4;
            } else {
                let idx0 = idx0 * 2;
                w0 = 8 + a00 * self.dq(weights[idx0]);
                w1 = 8 + a00 * self.dq(weights[idx0 + 1]);
                if a01 != 0 {
                    let i1 = idx0 + 2;
                    w0 += a01 * self.dq(weights[i1]);
                    w1 += a01 * self.dq(weights[i1 + 1]);
                }
                if a10 != 0 {
                    let i2 = idx0 + gw * 2;
                    w0 += a10 * self.dq(weights[i2]);
                    w1 += a10 * self.dq(weights[i2 + 1]);
                }
                if a11 != 0 {
                    let i3 = idx0 + 2 + gw * 2;
                    w0 += a11 * self.dq(weights[i3]);
                    w1 += a11 * self.dq(weights[i3 + 1]);
                }
                w0 >>= 4;
                w1 >>= 4;
            }
        }
        let mut subset = 0usize;
        let mut cw = [w0; 4];
        if self.dual_plane {
            cw[self.ccs & 3] = w1;
        } else if self.num_partitions > 1 {
            let v = u32::from(self.part[(y * 12 + x) as usize]);
            subset = ((v >> self.part_shift) & 3).min(2) as usize;
        }
        let e = &self.ep.e[subset];
        let mut c = [0u8; 4];
        for ch in 0..4 {
            c[ch] = (weight_interpolate(e[ch][0], e[ch][1], cw[ch]) >> 8) as u8;
        }
        c
    }
}

// ------------------------------------------------------------------------------------------
// Contexts.

/// `astc_ldrf::single_subset_enc_context`.
#[derive(Clone, Debug)]
pub(crate) struct SingleSubsetEncContext {
    pub block_width: u32,
    pub block_height: u32,
    pub block_size_index: u32,
    pub total_block_pixels: u32,
    pub max_candidates: u32,
    pub num_ls_iterations: u32,
    pub chan_weights: [u32; 4],
    pub dct: Dct2f,
    /// `m_astc_decode_mode == cDecodeModeSRGB8` (else `cDecodeModeLDR8`).
    pub astc_decode_mode_srgb: bool,
    pub disable_dual_plane: bool,
    pub weight_polishing: bool,
    pub has_alpha: bool,
    pub try_base_ofs: bool,
    pub higher_effort_bc: bool,
}

/// `astc_ldrf::subset_enc_context` (a `single_subset_enc_context` plus the subset state).
#[derive(Clone, Debug)]
pub(crate) struct SubsetEncContext {
    pub ss: SingleSubsetEncContext,
    pub max_subsets: u32,
    pub num_carrier_candidates: u32,
    pub num_pattern_candidates: u32,
    pub two_subset_var_thresh: f32,
    pub three_subset_var_thresh: f32,
    pub two_subset_dot_thresh_fract_index: u32,
    pub two_subset_pat_bitmask: Vec<Bitmask192>,
    pub num_unique_two_subset_pats: u32,
    /// `m_pUnique_two_subset_pats` (unique 2-subset index to ASTC seed).
    pub unique_two_subset_pats: Vec<u16>,
    pub use_method1: bool,
    pub use_method2: bool,
    pub part_data_p2: Option<Arc<PartitionsData>>,
    pub part_data_p3: Option<Arc<PartitionsData>>,
}

impl SubsetEncContext {
    /// Wraps a single subset context (multi subset fields zero until
    /// [`init_multi_subset_context`]).
    pub fn new(ss: SingleSubsetEncContext) -> Self {
        Self {
            ss,
            max_subsets: 1,
            num_carrier_candidates: 0,
            num_pattern_candidates: 0,
            two_subset_var_thresh: 0.0,
            three_subset_var_thresh: 0.0,
            two_subset_dot_thresh_fract_index: 0,
            two_subset_pat_bitmask: Vec::new(),
            num_unique_two_subset_pats: 0,
            unique_two_subset_pats: Vec::new(),
            use_method1: false,
            use_method2: false,
            part_data_p2: None,
            part_data_p3: None,
        }
    }
}

/// `astc_ldrf::subset_enc_thread_context`.
#[derive(Clone, Debug, Default)]
pub(crate) struct SubsetEncThreadContext {
    pub pat_vec: PartitionPatternVec,
}

/// `astc_ldrf::init_single_subset_context`. `None` on an invalid block size or candidate
/// count.
pub(crate) fn init_single_subset_context(
    block_width: u32,
    block_height: u32,
    astc_decode_mode_srgb: bool,
    chan_weights: &[u32; 4],
    max_candidates: u32,
    num_ls_iterations: u32,
    disable_dual_plane: bool,
    has_alpha: bool,
    weight_polishing: bool,
) -> Option<SingleSubsetEncContext> {
    let idx = find_astc_block_size_index(block_width, block_height)?;
    if max_candidates > MAX_CANDIDATES {
        return None;
    }
    Some(SingleSubsetEncContext {
        block_width,
        block_height,
        block_size_index: idx,
        total_block_pixels: block_width * block_height,
        max_candidates,
        num_ls_iterations,
        chan_weights: *chan_weights,
        dct: Dct2f::new(block_height, block_width)?,
        astc_decode_mode_srgb,
        disable_dual_plane,
        weight_polishing,
        has_alpha,
        try_base_ofs: true,
        higher_effort_bc: true,
    })
}

/// `astc_helpers::find_astc_block_size_index`.
pub(crate) fn find_astc_block_size_index(w: u32, h: u32) -> Option<u32> {
    crate::basis::file::BASIS_ASTC_ORDER
        .iter()
        .position(|b| b.dim() == (w, h))
        .map(|i| i as u32)
}

/// `astc_ldrf::init_multi_subset_context` (consumes the single subset context).
pub(crate) fn init_multi_subset_context(
    ctx: &mut SubsetEncContext,
    max_subsets: u32,
    num_carrier_candidates: u32,
    num_pattern_candidates: u32,
    two_subset_var_thresh: f32,
    two_subset_dot_thresh_fract_index: u32,
    three_subset_var_thresh: f32,
    part_data_p2: Arc<PartitionsData>,
    part_data_p3: Arc<PartitionsData>,
) -> bool {
    ctx.max_subsets = max_subsets;
    ctx.num_carrier_candidates = num_carrier_candidates;
    ctx.num_pattern_candidates = num_pattern_candidates;
    ctx.two_subset_var_thresh = two_subset_var_thresh;
    ctx.three_subset_var_thresh = three_subset_var_thresh;
    ctx.two_subset_dot_thresh_fract_index = two_subset_dot_thresh_fract_index;

    let size_index = ctx.ss.block_size_index as usize;
    let n = crate::basis::xuastc::total_unique_patterns(size_index, 2);
    ctx.num_unique_two_subset_pats = n;
    ctx.unique_two_subset_pats.clear();
    ctx.two_subset_pat_bitmask.clear();
    ctx.use_method1 = true;
    ctx.use_method2 = true;
    ctx.part_data_p2 = Some(part_data_p2);
    ctx.part_data_p3 = Some(part_data_p3);

    let (bw, bh) = (ctx.ss.block_width, ctx.ss.block_height);
    for unique_index in 0..n {
        let Ok(seed_id) = crate::basis::xuastc::part_seed(size_index, 2, unique_index) else {
            return false;
        };
        ctx.unique_two_subset_pats.push(seed_id);
        let mut pat = Bitmask192::zero();
        let mut bit_ofs = 0;
        for y in 0..bh {
            for x in 0..bw {
                let s = super::partitions::get_precomputed_texel_partition(
                    bw,
                    bh,
                    u32::from(seed_id),
                    x,
                    y,
                    2,
                );
                if s != 0 {
                    pat.set_bit(bit_ofs, 1);
                }
                bit_ofs += 1;
            }
        }
        ctx.two_subset_pat_bitmask.push(pat);
    }
    true
}

// ------------------------------------------------------------------------------------------
// Error metrics.

/// `compute_block_error`: channel weighted SSE of the decoded block vs. `src_block`.
fn compute_block_error(
    lblk: &LogAstcBlock,
    src_block: &PixelBuf,
    ctx: &SingleSubsetEncContext,
) -> f64 {
    let mut tmp;
    let blk = if is_lblock_ise(lblk) {
        lblk
    } else {
        tmp = *lblk;
        convert_rank_lblock_to_ise(&mut tmp);
        &tmp
    };
    let mut px = [[0u8; 4]; MAX_BLOCK_PIXELS];
    if !decode_block_xuastc_ldr(
        blk,
        &mut px,
        ctx.block_width,
        ctx.block_height,
        ctx.astc_decode_mode_srgb,
    ) {
        return f64::MAX;
    }
    let w = ctx.chan_weights.map(|v| v as f32);
    let mut wsse = 0.0f64;
    let mut i = 0;
    for y in 0..ctx.block_height {
        for x in 0..ctx.block_width {
            let p = px[i];
            for c in 0..4 {
                wsse += f64::from(sq(f32::from(p[c]) - src_block.get(x, y, c as u32)) * w[c]);
            }
            i += 1;
        }
    }
    wsse
}

/// `calc_weighted_error`.
#[inline]
fn calc_weighted_error(c: [u8; 4], p: [f32; 4], ctx: &SingleSubsetEncContext) -> f32 {
    (sq(f32::from(c[0]) - p[0]) * ctx.chan_weights[0] as f32)
        + (sq(f32::from(c[1]) - p[1]) * ctx.chan_weights[1] as f32)
        + (sq(f32::from(c[2]) - p[2]) * ctx.chan_weights[2] as f32)
        + (sq(f32::from(c[3]) - p[3]) * ctx.chan_weights[3] as f32)
}

// ------------------------------------------------------------------------------------------
// Covariance / PCA.

// Covariance matrix element indices.
const COV_RR: usize = 0;
const COV_RG: usize = 1;
const COV_RB: usize = 2;
const COV_RA: usize = 3;
const COV_GG: usize = 4;
const COV_GB: usize = 5;
const COV_GA: usize = 6;
const COV_BB: usize = 7;
const COV_BA: usize = 8;
const COV_AA: usize = 9;

const COVAR_COL_INDICES: [[usize; 4]; 4] = [
    [COV_RR, COV_RG, COV_RB, COV_RA],
    [COV_RG, COV_GG, COV_GB, COV_GA],
    [COV_RB, COV_GB, COV_BB, COV_BA],
    [COV_RA, COV_GA, COV_BA, COV_AA],
];

const CORR_RG: usize = 0;
const CORR_RB: usize = 1;
const CORR_RA: usize = 2;
const CORR_GB: usize = 3;
const CORR_GA: usize = 4;
const CORR_BA: usize = 5;

/// `compute_block_moments`.
fn compute_block_moments(
    m: &mut [f32; 10],
    sums: &mut [f32; 4],
    pb: &PixelBuf,
    num_comps: u32,
    zero_centered: bool,
) {
    for y in 0..pb.height {
        for x in 0..pb.width {
            let r = pb.get(x, y, 0);
            let g = pb.get(x, y, 1);
            let b = pb.get(x, y, 2);
            if !zero_centered {
                sums[0] += r;
                sums[1] += g;
                sums[2] += b;
            }
            m[COV_RR] += r * r;
            m[COV_RG] += r * g;
            m[COV_RB] += r * b;
            m[COV_GG] += g * g;
            m[COV_GB] += g * b;
            m[COV_BB] += b * b;
            if num_comps == 4 {
                let a = pb.get(x, y, 3);
                if !zero_centered {
                    sums[3] += a;
                }
                m[COV_RA] += r * a;
                m[COV_GA] += g * a;
                m[COV_BA] += b * a;
                m[COV_AA] += a * a;
            }
        }
    }
}

/// `calc_unnormalized_covariance`.
fn calc_unnormalized_covariance(
    m: &[f32; 10],
    sums: &[f32; 4],
    cov: &mut [f32; 10],
    num_comps: u32,
    total_pixels: u32,
    zero_centered: bool,
) {
    if zero_centered {
        *cov = *m;
        return;
    }
    let oo = 1.0f32 / total_pixels as f32;
    let (sr, sg, sb) = (sums[0], sums[1], sums[2]);
    cov[COV_RR] = m[COV_RR] - (sr * sr * oo);
    cov[COV_RG] = m[COV_RG] - (sr * sg * oo);
    cov[COV_RB] = m[COV_RB] - (sr * sb * oo);
    cov[COV_GG] = m[COV_GG] - (sg * sg * oo);
    cov[COV_GB] = m[COV_GB] - (sg * sb * oo);
    cov[COV_BB] = m[COV_BB] - (sb * sb * oo);
    cov[COV_RA] = 0.0;
    cov[COV_GA] = 0.0;
    cov[COV_BA] = 0.0;
    cov[COV_AA] = 0.0;
    if num_comps == 4 {
        let sa = sums[3];
        cov[COV_RA] = m[COV_RA] - (sr * sa * oo);
        cov[COV_GA] = m[COV_GA] - (sg * sa * oo);
        cov[COV_BA] = m[COV_BA] - (sb * sa * oo);
        cov[COV_AA] = m[COV_AA] - (sa * sa * oo);
    }
}

/// `corr_pair`.
#[inline]
fn corr_pair(xy: f32, xx: f32, yy: f32) -> f32 {
    let d = xx * yy;
    if d <= TINY_EPS {
        return 1.0;
    }
    xy / d.sqrt()
}

/// `corr_from_covar`.
fn corr_from_covar(corr: &mut [f32; 6], c: &[f32; 10], num_comps: u32) {
    corr[CORR_RG] = corr_pair(c[COV_RG], c[COV_RR], c[COV_GG]);
    corr[CORR_RB] = corr_pair(c[COV_RB], c[COV_RR], c[COV_BB]);
    corr[CORR_GB] = corr_pair(c[COV_GB], c[COV_GG], c[COV_BB]);
    corr[CORR_RA] = 1.0;
    corr[CORR_GA] = 1.0;
    corr[CORR_BA] = 1.0;
    if num_comps == 4 {
        corr[CORR_RA] = corr_pair(c[COV_RA], c[COV_RR], c[COV_AA]);
        corr[CORR_GA] = corr_pair(c[COV_GA], c[COV_GG], c[COV_AA]);
        corr[CORR_BA] = corr_pair(c[COV_BA], c[COV_BB], c[COV_AA]);
    }
}

/// `compute_covariance`.
fn compute_covariance(
    cov: &mut [f32; 10],
    mean: &mut [f32; 4],
    pb: &PixelBuf,
    num_comps: u32,
    zero_centered: bool,
) {
    let mut moments = [0.0f32; 10];
    *mean = [0.0; 4];
    compute_block_moments(&mut moments, mean, pb, num_comps, zero_centered);
    let total = pb.width * pb.height;
    calc_unnormalized_covariance(&moments, mean, cov, num_comps, total, zero_centered);
    if !zero_centered {
        let oo = 1.0f32 / total as f32;
        for c in 0..num_comps as usize {
            mean[c] *= oo;
        }
    }
}

/// `covar_mul_vec4`.
fn covar_mul_vec4(c: &[f32; 10], s: &[f32; 4]) -> [f32; 4] {
    let (x, y, z, w) = (s[0], s[1], s[2], s[3]);
    [
        c[COV_RR] * x + c[COV_RG] * y + c[COV_RB] * z + c[COV_RA] * w,
        c[COV_RG] * x + c[COV_GG] * y + c[COV_GB] * z + c[COV_GA] * w,
        c[COV_RB] * x + c[COV_GB] * y + c[COV_BB] * z + c[COV_BA] * w,
        c[COV_RA] * x + c[COV_GA] * y + c[COV_BA] * z + c[COV_AA] * w,
    ]
}

/// `covar_mul_vec3`.
fn covar_mul_vec3(c: &[f32; 10], s: &[f32; 4]) -> [f32; 4] {
    let (x, y, z) = (s[0], s[1], s[2]);
    [
        c[COV_RR] * x + c[COV_RG] * y + c[COV_RB] * z,
        c[COV_RG] * x + c[COV_GG] * y + c[COV_GB] * z,
        c[COV_RB] * x + c[COV_GB] * y + c[COV_BB] * z,
        0.0,
    ]
}

/// `get_initial_axis_from_largest_diag`.
fn get_initial_axis_from_largest_diag(c: &[f32; 10], num_comps: u32) -> [f32; 4] {
    let mut best_col = 0;
    let mut best_diag = c[COV_RR];
    if c[COV_GG] > best_diag {
        best_diag = c[COV_GG];
        best_col = 1;
    }
    if c[COV_BB] > best_diag {
        best_diag = c[COV_BB];
        best_col = 2;
    }
    if num_comps == 4 && c[COV_AA] > best_diag {
        best_col = 3;
    }
    let col = &COVAR_COL_INDICES[best_col];
    [
        c[col[0]],
        c[col[1]],
        c[col[2]],
        if num_comps == 4 { c[col[3]] } else { 0.0 },
    ]
}

/// `compute_principle_axis`.
fn compute_principle_axis(cov: &[f32; 10], max_pair_iters: u32, num_comps: u32) -> [f32; 4] {
    let mut axis = get_initial_axis_from_largest_diag(cov, num_comps);
    let s = fmax(
        fmax(fmax(axis[0].abs(), axis[1].abs()), axis[2].abs()),
        axis[3].abs(),
    );
    if s > TINY_EPS {
        let k = 1.0 / s;
        for a in axis.iter_mut() {
            *a *= k;
        }
    }
    for i in 0..max_pair_iters {
        let prev = axis;
        if num_comps == 3 {
            let t = covar_mul_vec3(cov, &axis);
            axis = covar_mul_vec3(cov, &t);
        } else {
            let t = covar_mul_vec4(cov, &axis);
            axis = covar_mul_vec4(cov, &t);
        }
        let total_sq = vec_normalize(&mut axis, num_comps as usize);
        if total_sq < TINY_EPS {
            axis = [1.0, 1.0, 1.0, if num_comps == 4 { 1.0 } else { 0.0 }];
            vec_normalize(&mut axis, num_comps as usize);
            break;
        }
        if i != 0 {
            let d = vec4_dot(&axis, &prev);
            if d >= 0.9997 {
                break;
            }
        }
    }
    axis
}

/// `block_stats`.
#[derive(Clone, Copy, Debug, Default)]
struct BlockStats {
    covar: [f32; 10],
    corr: [f32; 6],
    mean: [f32; 4],
    axis: [f32; 4],
}

/// `calc_initial_cem_endpoints`: returns endpoints in ASTC direct order (LR HR LG HG LB HB LA
/// HA).
fn calc_initial_cem_endpoints(
    block: &PixelBuf,
    num_comps: u32,
    out_stats: Option<&mut BlockStats>,
    zero_centered: bool,
) -> [f32; 8] {
    let mut cov = [0.0f32; 10];
    let mut mean = [0.0f32; 4];
    compute_covariance(&mut cov, &mut mean, block, num_comps, zero_centered);
    let mut stats = out_stats;
    if let Some(s) = stats.as_deref_mut() {
        s.covar = cov;
        corr_from_covar(&mut s.corr, &cov, num_comps);
        s.mean = mean;
    }
    cov[COV_RR] += SMALL_EPS;
    cov[COV_GG] += SMALL_EPS;
    cov[COV_BB] += SMALL_EPS;
    if num_comps == 4 {
        cov[COV_AA] += SMALL_EPS;
    }
    let axis = compute_principle_axis(&cov, 5, num_comps);
    if let Some(s) = stats {
        s.axis = axis;
    }
    let mut span = [1e30f32, -1e30f32];
    for y in 0..block.height {
        for x in 0..block.width {
            let mut d = 0.0f32;
            for c in 0..num_comps {
                d += (block.get(x, y, c) - mean[c as usize]) * axis[c as usize];
            }
            span[0] = fmin(span[0], d);
            span[1] = fmax(span[1], d);
        }
    }
    if (span[0] + 1.0) > span[1] {
        span[0] -= 0.5;
        span[1] += 0.5;
    }
    let mut vals = [0.0f32; 8];
    for e in 0..2 {
        for c in 0..num_comps as usize {
            vals[c * 2 + e] = fclamp(axis[c] * span[e] + mean[c], 0.0, 255.0);
        }
        if num_comps == 3 {
            vals[6] = 0.0;
            vals[7] = 255.0;
        }
    }
    vals
}

/// `fit_and_measure`.
fn fit_and_measure(
    block: &PixelBuf,
    num_comps: u32,
    zero_centered: bool,
    out_span: &mut [f32; 4],
    out_ortho_error: &mut f32,
    out_stats: Option<&mut BlockStats>,
    out_ortho_error_rgba: Option<&mut [f32; 4]>,
) {
    let endpoints = calc_initial_cem_endpoints(block, num_comps, out_stats, zero_centered);
    out_span[3] = 0.0;
    let mut org = [0.0f32; 4];
    let mut dir = [0.0f32; 4];
    let nc = num_comps as usize;
    for c in 0..nc {
        org[c] = endpoints[c * 2];
        dir[c] = endpoints[c * 2 + 1] - endpoints[c * 2];
        out_span[c] = dir[c];
    }
    let mut len = 0.0f32;
    for c in 0..nc {
        len += dir[c] * dir[c];
    }
    let inv_dir_sq_len = 1.0 / (len + TINY_EPS);
    let mut err_c = [0.0f32; 4];
    for y in 0..block.height {
        for x in 0..block.width {
            let mut d = 0.0f32;
            for c in 0..nc {
                d += (block.get(x, y, c as u32) - org[c]) * dir[c];
            }
            let dist_along = d * inv_dir_sq_len;
            for c in 0..nc {
                let v = block.get(x, y, c as u32);
                let nearest = dir[c] * dist_along + org[c];
                err_c[c] += sq(v - nearest);
            }
        }
    }
    let mut total = 0.0f32;
    for c in 0..nc {
        total += err_c[c];
    }
    *out_ortho_error = total;
    if let Some(o) = out_ortho_error_rgba {
        *o = err_c;
    }
}

/// `fit_and_measure_2D`.
fn fit_and_measure_2d(
    block: &PixelBuf,
    chan_to_zero: u32,
    zero_centered: bool,
    out_span: &mut [f32; 4],
    out_ortho_error: &mut f32,
) {
    let mut temp = block.clone();
    temp.set_comp_to_val(chan_to_zero, 0.0);
    fit_and_measure(
        &temp,
        3,
        zero_centered,
        out_span,
        out_ortho_error,
        None,
        None,
    );
}

/// `fit_and_measure_1D`: returns (span, slam to 255 SSE).
fn fit_and_measure_1d(block: &PixelBuf, comp: u32) -> (f32, f32) {
    let (mut lo, mut hi, mut a_err) = (1e30f32, -1e30f32, 0.0f32);
    for y in 0..block.height {
        for x in 0..block.width {
            let v = block.get(x, y, comp);
            lo = fmin(lo, v);
            hi = fmax(hi, v);
            a_err += sq(v - 255.0);
        }
    }
    (hi - lo, a_err)
}

// ------------------------------------------------------------------------------------------
// Shortlist generation.

/// `astc_unpacked_config`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AstcUnpackedConfig {
    pub cem: u8,
    pub grid_width: u8,
    pub grid_height: u8,
    pub endpoint_range: u8,
    pub weight_range: u8,
    pub dual_plane: u8,
    pub ccs_index: u8,
    pub unused: u8,
}

/// `single_subset_shortlist_state`.
#[derive(Clone)]
struct ShortlistState {
    block_width: u32,
    block_height: u32,
    max_candidates: u32,
    num_src_block_comps: u32,
    src_is_luma_only: bool,
    should_include_dual_plane: bool,
    /// `m_pbuf` over `m_block_pixels`.
    pbuf: PixelBuf,
    downsample_sse: [[f32; 11]; 11],
    sp_spans: [[f32; 4]; CEM_TOTAL_INDICES],
    sp_slam_to_line_error: [f32; CEM_TOTAL_INDICES],
    sp_valid: [bool; CEM_TOTAL_INDICES],
    dp_spans: [[[f32; 4]; 4]; CEM_TOTAL_INDICES],
    dp_slam_to_line_error: [[f32; 4]; CEM_TOTAL_INDICES],
    dp_valid: [[bool; 4]; CEM_TOTAL_INDICES],
    stats: BlockStats,
    best_sse: Vec<f32>,
    best_configs: Vec<AstcUnpackedConfig>,
}

impl ShortlistState {
    fn new() -> Self {
        Self {
            block_width: 0,
            block_height: 0,
            max_candidates: 0,
            num_src_block_comps: 0,
            src_is_luma_only: false,
            should_include_dual_plane: false,
            pbuf: PixelBuf::new(0, 0),
            downsample_sse: [[0.0; 11]; 11],
            sp_spans: [[0.0; 4]; CEM_TOTAL_INDICES],
            sp_slam_to_line_error: [0.0; CEM_TOTAL_INDICES],
            sp_valid: [false; CEM_TOTAL_INDICES],
            dp_spans: [[[0.0; 4]; 4]; CEM_TOTAL_INDICES],
            dp_slam_to_line_error: [[0.0; 4]; CEM_TOTAL_INDICES],
            dp_valid: [[false; 4]; CEM_TOTAL_INDICES],
            stats: BlockStats::default(),
            best_sse: vec![0.0; MAX_CANDIDATES as usize],
            best_configs: vec![AstcUnpackedConfig::default(); MAX_CANDIDATES as usize],
        }
    }
}

/// `should_include_dual_plane`.
fn should_include_dual_plane(state: &ShortlistState, has_a: bool) -> bool {
    let s = &state.stats;
    let has_r = s.covar[COV_RR] != 0.0;
    let has_g = s.covar[COV_GG] != 0.0;
    let has_b = s.covar[COV_BB] != 0.0;
    let total = u32::from(has_r) + u32::from(has_g) + u32::from(has_b) + u32::from(has_a);
    if total <= 1 {
        return false;
    }
    const MIN_A_DP_CORR: f32 = 0.9999;
    const MIN_RGB_DP_CORR: f32 = 0.9999;
    if has_a {
        let m = fmin(
            fmin(s.corr[CORR_RA].abs(), s.corr[CORR_GA].abs()),
            s.corr[CORR_BA].abs(),
        );
        if m < MIN_A_DP_CORR {
            return true;
        }
    }
    let min_p = fmin(
        fmin(s.corr[CORR_RG].abs(), s.corr[CORR_RB].abs()),
        s.corr[CORR_GB].abs(),
    );
    min_p < MIN_RGB_DP_CORR
}

/// `compute_block_metrics`.
fn compute_block_metrics(state: &mut ShortlistState, num_comps: u32, ctx: &SingleSubsetEncContext) {
    let mut temp = state.pbuf.clone();
    temp.width = state.block_width;
    temp.height = state.block_height;

    state.sp_spans = [[0.0; 4]; CEM_TOTAL_INDICES];
    state.sp_slam_to_line_error = [0.0; CEM_TOTAL_INDICES];
    state.sp_valid = [false; CEM_TOTAL_INDICES];
    state.dp_spans = [[[0.0; 4]; 4]; CEM_TOTAL_INDICES];
    state.dp_slam_to_line_error = [[0.0; 4]; CEM_TOTAL_INDICES];
    state.dp_valid = [[false; 4]; CEM_TOTAL_INDICES];

    let mut chan_ranges = [0.0f32; 4];
    let mut a_slam_to_255_sse = 0.0f32;
    for c in 0..4 {
        let (span, slam) = fit_and_measure_1d(&temp, c);
        chan_ranges[c as usize] = span;
        if c == 3 {
            a_slam_to_255_sse = slam;
        }
    }

    let mut cem6_ortho_rgba = [0.0f32; 4];
    let mut cem10_ortho_rgba = [0.0f32; 4];

    {
        let (sp, err) = (
            &mut state.sp_spans[CEM6],
            &mut state.sp_slam_to_line_error[CEM6],
        );
        fit_and_measure(&temp, 3, true, sp, err, None, Some(&mut cem6_ortho_rgba));
    }
    state.sp_valid[CEM6] = true;

    if num_comps == 4 {
        {
            let (sp, err) = (
                &mut state.sp_spans[CEM8],
                &mut state.sp_slam_to_line_error[CEM8],
            );
            fit_and_measure(&temp, 3, false, sp, err, None, None);
        }
        state.sp_valid[CEM8] = true;
        {
            let (sp, err) = (
                &mut state.sp_spans[CEM10],
                &mut state.sp_slam_to_line_error[CEM10],
            );
            fit_and_measure(&temp, 4, true, sp, err, None, Some(&mut cem10_ortho_rgba));
        }
        state.sp_valid[CEM10] = true;
        {
            let (sp, err) = (
                &mut state.sp_spans[CEM12],
                &mut state.sp_slam_to_line_error[CEM12],
            );
            fit_and_measure(&temp, 4, false, sp, err, Some(&mut state.stats), None);
        }
        state.sp_valid[CEM12] = true;
    } else {
        {
            let (sp, err) = (
                &mut state.sp_spans[CEM8],
                &mut state.sp_slam_to_line_error[CEM8],
            );
            fit_and_measure(&temp, 3, false, sp, err, Some(&mut state.stats), None);
        }
        state.sp_valid[CEM8] = true;
    }

    const CEM6_10_DP_RGB_CHAN_ORTHO_WEIGHT: f32 = 0.25;

    for comp in 0..num_comps {
        let cu = comp as usize;
        temp.swap_comp_with_alpha(comp);
        if comp < 3 {
            state.dp_spans[CEM6][cu][0] = state.sp_spans[CEM6][0];
            state.dp_spans[CEM6][cu][1] = state.sp_spans[CEM6][1];
            state.dp_spans[CEM6][cu][2] = state.sp_spans[CEM6][2];
            state.dp_slam_to_line_error[CEM6][cu] = a_slam_to_255_sse;
            for c in 0..3 {
                let w = if c == cu {
                    CEM6_10_DP_RGB_CHAN_ORTHO_WEIGHT
                } else {
                    1.0
                };
                state.dp_slam_to_line_error[CEM6][cu] += cem6_ortho_rgba[c] * w;
            }
            state.dp_valid[CEM6][cu] = true;

            {
                let (sp, err) = (
                    &mut state.dp_spans[CEM8][cu],
                    &mut state.dp_slam_to_line_error[CEM8][cu],
                );
                fit_and_measure_2d(&temp, comp, false, sp, err);
            }
            state.dp_spans[CEM8][cu][cu] = 0.0;
            state.dp_spans[CEM8][cu][3] = chan_ranges[cu];
            state.dp_spans[CEM8][cu].swap(3, cu);
            state.dp_slam_to_line_error[CEM8][cu] += a_slam_to_255_sse;
            state.dp_valid[CEM8][cu] = true;
        }

        if num_comps == 4 {
            if comp == 3 {
                state.dp_spans[CEM10][cu][0] = state.sp_spans[CEM6][0];
                state.dp_spans[CEM10][cu][1] = state.sp_spans[CEM6][1];
                state.dp_spans[CEM10][cu][2] = state.sp_spans[CEM6][2];
                state.dp_spans[CEM10][cu][3] = chan_ranges[3];
                state.dp_slam_to_line_error[CEM10][cu] = state.sp_slam_to_line_error[CEM6];
            } else {
                state.dp_spans[CEM10][cu] = state.sp_spans[CEM10];
                state.dp_slam_to_line_error[CEM10][cu] = 0.0;
                for c in 0..4 {
                    let w = if c == cu {
                        CEM6_10_DP_RGB_CHAN_ORTHO_WEIGHT
                    } else {
                        1.0
                    };
                    state.dp_slam_to_line_error[CEM10][cu] += cem10_ortho_rgba[c] * w;
                }
            }
            state.dp_valid[CEM10][cu] = true;

            {
                let (sp, err) = (
                    &mut state.dp_spans[CEM12][cu],
                    &mut state.dp_slam_to_line_error[CEM12][cu],
                );
                fit_and_measure(&temp, 3, false, sp, err, None, None);
            }
            state.dp_spans[CEM12][cu][3] = chan_ranges[cu];
            state.dp_spans[CEM12][cu].swap(3, cu);
            state.dp_valid[CEM12][cu] = true;
        }

        temp.swap_comp_with_alpha(comp);
    }

    state.sp_slam_to_line_error[CEM6] += a_slam_to_255_sse;
    state.sp_slam_to_line_error[CEM8] += a_slam_to_255_sse;

    // Parseval: estimate the SSE lost to weight grid downsampling.
    let mut dct_temp = [0.0f32; 144];
    for c in 0..num_comps as usize {
        ctx.dct
            .forward_in_place(&mut temp.buf[PB_COMP * c..], PB_ROW, &mut dct_temp);
    }

    state.downsample_sse = [[0.0; 11]; 11];

    let (bw, bh) = (state.block_width, state.block_height);
    for y in 0..bh {
        for x in 0..bw {
            let mut e = 0.0f32;
            for c in 0..num_comps {
                e += sq(temp.get(x, y, c));
            }
            temp.set(x, y, 0, e);
        }
    }

    // prepare_dct_energy_prefix_table
    let mut prefix = [0.0f64; PB_COMP];
    temp.set(0, 0, 0, 0.0);
    let mut total_ac_energy = 0.0f64;
    for y in 0..bh {
        let mut row_sum = 0.0f64;
        for x in 0..bw {
            let ac = temp.get(x, y, 0);
            total_ac_energy += f64::from(ac);
            row_sum += f64::from(ac);
            let above = if y != 0 {
                prefix[x as usize + (y as usize - 1) * PB_ROW]
            } else {
                0.0
            };
            prefix[x as usize + y as usize * PB_ROW] = row_sum + above;
        }
    }

    for gh in 2..=bh {
        for gw in 2..=bw {
            if gw * gh > 64 {
                continue;
            }
            let kept = prefix[(gw - 1) as usize + (gh - 1) as usize * PB_ROW];
            let lost = total_ac_energy - kept;
            let sse = if lost > 0.0 { lost } else { 0.0 };
            state.downsample_sse[(gh - 2) as usize][(gw - 2) as usize] = sse as f32;
        }
    }

    state.should_include_dual_plane = false;
    if !ctx.disable_dual_plane {
        state.should_include_dual_plane = should_include_dual_plane(state, num_comps == 4);
    }
}

/// `analytical_quant_est_sse`.
#[inline]
fn analytical_quant_est_sse(
    astc_cem: u32,
    e_levels: u32,
    w_levels: u32,
    spans: &[f32; 4],
    num_pixels: u32,
) -> f32 {
    let dep = INTERVALS_RECIP[(e_levels as usize).min(256)];
    let dw = INTERVALS_RECIP[(w_levels as usize + 1).min(256)];
    let ab_sum = AB_SUM_TAB[(w_levels as usize).min(32)];
    let num_chans = get_num_cem_chans(astc_cem);
    let mut pixel_sse = if e_levels == 256 {
        0.0
    } else {
        (dep * dep) * ab_sum * 5418.75 * num_chans as f32
    };
    let k = (dw * dw) * (1.0f32 / 12.0);
    let mut t = spans[0] * spans[0] + spans[1] * spans[1] + spans[2] * spans[2];
    if num_chans == 4 {
        t += spans[3] * spans[3];
    }
    pixel_sse += k * t;
    pixel_sse * num_pixels as f32
}

fn cem_to_shortlist_index(astc_cem: u32) -> usize {
    if astc_cem >= 6 {
        ((astc_cem - 6) >> 1) as usize
    } else if astc_cem == 0 {
        CEM8
    } else {
        CEM12
    }
}

/// `estimate_base_config_sse` (does not include the downsample SSE).
#[inline]
fn estimate_base_config_sse(state: &ShortlistState, cfg: &AstcUnpackedConfig) -> f32 {
    let astc_cem = u32::from(cfg.cem);
    let ci = cem_to_shortlist_index(astc_cem).min(CEM_TOTAL_INDICES - 1);
    let nel = ah::get_ise_levels(u32::from(cfg.endpoint_range));
    let nwl = ah::get_ise_levels(u32::from(cfg.weight_range));
    let ccs = (cfg.ccs_index & 3) as usize;
    let num_block_pixels = state.block_height * state.block_width;
    let dual = cfg.dual_plane != 0;
    let spans = if dual {
        &state.dp_spans[ci][ccs]
    } else {
        &state.sp_spans[ci]
    };
    let quant_error = analytical_quant_est_sse(astc_cem, nel, nwl, spans, num_block_pixels);
    let ortho_error = if dual {
        state.dp_slam_to_line_error[ci][ccs]
    } else {
        state.sp_slam_to_line_error[ci]
    };
    quant_error * DEF_QUANT_WEIGHT + ortho_error
}

/// `estimate_full_config_sse(state, cfg, base_sse, scale_weight)`.
#[inline]
fn estimate_full_config_sse(
    state: &ShortlistState,
    cfg: &AstcUnpackedConfig,
    base_sse: f32,
    scale_weight: f32,
) -> f32 {
    let gh = (cfg.grid_height as usize).saturating_sub(2).min(10);
    let gw = (cfg.grid_width as usize).saturating_sub(2).min(10);
    state.downsample_sse[gh][gw] * scale_weight + base_sse
}

/// `estimate_and_add_config`.
fn estimate_and_add_config(
    state: &mut ShortlistState,
    sse: f32,
    cfg: &AstcUnpackedConfig,
    max_candidate_sse: &mut f32,
    num_candidates: &mut u32,
    max_candidates: u32,
) {
    if *num_candidates < max_candidates {
        let n = *num_candidates as usize;
        state.best_configs[n] = *cfg;
        state.best_sse[n] = sse;
        *max_candidate_sse = fmax(*max_candidate_sse, sse);
        *num_candidates += 1;
        return;
    }
    if sse >= *max_candidate_sse {
        return;
    }
    let mut worst_cfg = *cfg;
    let mut worst_sse = sse;
    *max_candidate_sse = 0.0;
    for i in 0..*num_candidates as usize {
        if state.best_sse[i] > worst_sse {
            std::mem::swap(&mut state.best_sse[i], &mut worst_sse);
            std::mem::swap(&mut state.best_configs[i], &mut worst_cfg);
        }
        *max_candidate_sse = fmax(*max_candidate_sse, state.best_sse[i]);
    }
}

/// `init_single_subset_shortlist_state`.
fn init_single_subset_shortlist_state(
    pixels: &[[u8; 4]],
    ctx: &SingleSubsetEncContext,
    state: &mut ShortlistState,
    src_is_luma_only: bool,
    num_src_block_comps: u32,
) {
    state.block_width = ctx.block_width;
    state.block_height = ctx.block_height;
    state.max_candidates = ctx.max_candidates;
    state.num_src_block_comps = num_src_block_comps;
    state.src_is_luma_only = src_is_luma_only;
    state.pbuf.width = ctx.block_width;
    state.pbuf.height = ctx.block_height;
    state.pbuf.load_block(pixels, num_src_block_comps);
    compute_block_metrics(state, num_src_block_comps, ctx);
}

/// `generate_single_subset_shortlist`.
fn generate_single_subset_shortlist(
    packed_configs: &[u32],
    ctx: &SingleSubsetEncContext,
    num_src_block_comps: u32,
    state: &mut ShortlistState,
    scale_weight: f32,
    max_candidates: u32,
) -> u32 {
    let has_alpha = num_src_block_comps == 4;
    let num_actual_block_chans = if has_alpha { 4 } else { 3 };
    let max_candidates = max_candidates.min(MAX_CANDIDATES);
    let mut num_candidates = 0u32;
    let mut max_candidate_sse = 0.0f32;
    let mut cfg = AstcUnpackedConfig::default();

    for &packed in packed_configs {
        cfg.cem = extract_bits(packed, 0, 4) as u8;
        if num_actual_block_chans < 4 && does_cem_have_alpha(u32::from(cfg.cem)) {
            continue;
        }
        cfg.grid_width = extract_bits(packed, 4, 4) as u8;
        cfg.grid_height = extract_bits(packed, 8, 4) as u8;
        let (gw, gh) = (u32::from(cfg.grid_width), u32::from(cfg.grid_height));
        let wh_flag = gw <= ctx.block_width && gh <= ctx.block_height;
        if !wh_flag {
            continue;
        }
        let hw_flag = gw != gh && gh <= ctx.block_width && gw <= ctx.block_height;
        cfg.endpoint_range = extract_bits(packed, 12, 5) as u8;
        cfg.weight_range = extract_bits(packed, 17, 4) as u8;
        cfg.dual_plane = u8::from(extract_bits(packed, 21, 1) != 0);

        if cfg.dual_plane != 0 {
            if !state.should_include_dual_plane {
                continue;
            }
            let (mut ccs_first, mut ccs_last) = (0u32, num_actual_block_chans - 1);
            if !does_cem_have_alpha(u32::from(cfg.cem)) {
                ccs_last = 2;
            } else if cfg.cem == 4 {
                ccs_first = 3;
            }
            if hw_flag {
                let mut alt = cfg;
                std::mem::swap(&mut alt.grid_width, &mut alt.grid_height);
                for ccs in ccs_first..=ccs_last {
                    cfg.ccs_index = ccs as u8;
                    let base_sse = estimate_base_config_sse(state, &cfg);
                    let wh_sse = estimate_full_config_sse(state, &cfg, base_sse, scale_weight);
                    estimate_and_add_config(
                        state,
                        wh_sse,
                        &cfg,
                        &mut max_candidate_sse,
                        &mut num_candidates,
                        max_candidates,
                    );
                    alt.ccs_index = ccs as u8;
                    let hw_sse = estimate_full_config_sse(state, &alt, base_sse, scale_weight);
                    estimate_and_add_config(
                        state,
                        hw_sse,
                        &alt,
                        &mut max_candidate_sse,
                        &mut num_candidates,
                        max_candidates,
                    );
                }
            } else {
                for ccs in ccs_first..=ccs_last {
                    cfg.ccs_index = ccs as u8;
                    let base_sse = estimate_base_config_sse(state, &cfg);
                    let sse = estimate_full_config_sse(state, &cfg, base_sse, scale_weight);
                    estimate_and_add_config(
                        state,
                        sse,
                        &cfg,
                        &mut max_candidate_sse,
                        &mut num_candidates,
                        max_candidates,
                    );
                }
            }
        } else {
            cfg.ccs_index = 0;
            if hw_flag {
                let base_sse = estimate_base_config_sse(state, &cfg);
                let wh_sse = estimate_full_config_sse(state, &cfg, base_sse, scale_weight);
                estimate_and_add_config(
                    state,
                    wh_sse,
                    &cfg,
                    &mut max_candidate_sse,
                    &mut num_candidates,
                    max_candidates,
                );
                std::mem::swap(&mut cfg.grid_width, &mut cfg.grid_height);
                let hw_sse = estimate_full_config_sse(state, &cfg, base_sse, scale_weight);
                estimate_and_add_config(
                    state,
                    hw_sse,
                    &cfg,
                    &mut max_candidate_sse,
                    &mut num_candidates,
                    max_candidates,
                );
            } else {
                let base_sse = estimate_base_config_sse(state, &cfg);
                let sse = estimate_full_config_sse(state, &cfg, base_sse, scale_weight);
                estimate_and_add_config(
                    state,
                    sse,
                    &cfg,
                    &mut max_candidate_sse,
                    &mut num_candidates,
                    max_candidates,
                );
            }
        }
    }
    num_candidates
}

// ------------------------------------------------------------------------------------------
// Block to weight grid downsampling.

/// `g_pseudoinverse_coeff_offsets[block_size - 4][grid_size - 2]`.
const PSEUDOINVERSE_COEFF_OFFSETS: [[i16; 11]; 9] = [
    [0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [20, 30, 45, -1, -1, -1, -1, -1, -1, -1, -1],
    [65, 77, 95, 119, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [149, 165, 189, 221, 261, 309, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [365, 385, 415, 455, 505, 565, 635, 715, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [805, 829, 865, 913, 973, 1045, 1129, 1225, 1333, 1453, -1],
];

fn pseudoinverse_offset(src: u32, dst: u32) -> Option<usize> {
    let o = *PSEUDOINVERSE_COEFF_OFFSETS
        .get((src as usize).wrapping_sub(4))?
        .get((dst as usize).wrapping_sub(2))?;
    if o < 0 { None } else { Some(o as usize) }
}

/// `pseudoinverse_block_to_grid`.
fn pseudoinverse_block_to_grid(src: &PixelBuf, dst: &mut PixelBuf, num_comps: u32) {
    let (sw, sh) = (src.width, src.height);
    let (dw, dh) = (dst.width, dst.height);
    if sw == dw && sh == dh {
        dst.buf = src.buf;
        return;
    }
    let h_ofs = pseudoinverse_offset(sw, dw);
    let v_ofs = pseudoinverse_offset(sh, dh);
    let coeffs = &PSEUDOINVERSE_COEFFS[..];
    let mut temp = [[0.0f32; 4]; 12];
    let nc = num_comps as usize;

    for dst_y in 0..dh {
        if sh == dh || v_ofs.is_none() {
            for sx in 0..sw {
                for c in 0..nc {
                    temp[sx as usize][c] = src.get(sx, dst_y, c as u32);
                }
            }
        } else {
            let vo = v_ofs.unwrap_or(0);
            for sx in 0..sw {
                let (mut r, mut g, mut b, mut a) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for sy in 0..sh {
                    let w = coeffs[vo + (sy * dh + dst_y) as usize];
                    r += src.get(sx, sy, 0) * w;
                    if nc != 1 {
                        g += src.get(sx, sy, 1) * w;
                        b += src.get(sx, sy, 2) * w;
                        if nc == 4 {
                            a += src.get(sx, sy, 3) * w;
                        }
                    }
                }
                let t = &mut temp[sx as usize];
                t[0] = r;
                if nc != 1 {
                    t[1] = g;
                    t[2] = b;
                    if nc == 4 {
                        t[3] = a;
                    }
                }
            }
        }

        if sw == dw || h_ofs.is_none() {
            for dx in 0..dw {
                for c in 0..nc {
                    dst.set(dx, dst_y, c as u32, temp[dx as usize][c]);
                }
            }
        } else {
            let ho = h_ofs.unwrap_or(0);
            for dx in 0..dw {
                let (mut r, mut g, mut b, mut a) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for sx in 0..sw {
                    let w = coeffs[ho + (sx * dw + dx) as usize];
                    let t = &temp[sx as usize];
                    r += t[0] * w;
                    if nc != 1 {
                        g += t[1] * w;
                        b += t[2] * w;
                        if nc == 4 {
                            a += t[3] * w;
                        }
                    }
                }
                dst.set(dx, dst_y, 0, r);
                if nc != 1 {
                    dst.set(dx, dst_y, 1, g);
                    dst.set(dx, dst_y, 2, b);
                    if nc == 4 {
                        dst.set(dx, dst_y, 3, a);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// CEM encoding / decoding.

/// `cem_decode`: rank space CEM values to decoded endpoints `[LR LG LB LA HR HG HB HA]`.
fn cem_decode(
    cem: u32,
    vals_q: &[u8],
    endpoint_range: u32,
    out: &mut [f32; 8],
    actual_scale: Option<&mut f32>,
) {
    let mut dq = [0u8; 8];
    let n = (ah::get_num_cem_values(cem) as usize).min(8);
    if endpoint_range < BISE_256_LEVELS {
        for i in 0..n {
            dq[i] = ep_rank_to_val(endpoint_range, u32::from(vals_q[i])) as u8;
        }
    } else {
        dq[..n].copy_from_slice(&vals_q[..n]);
    }
    let v = |i: usize| i32::from(dq[i]);
    let (v0, v1, v2, v3) = (v(0), v(1), v(2), v(3));
    match cem {
        0 | 4 => {
            *out = [
                v0 as f32, v0 as f32, v0 as f32, 255.0, v1 as f32, v1 as f32, v1 as f32, 255.0,
            ];
            if cem == 4 {
                out[3] = v2 as f32;
                out[7] = v3 as f32;
            }
        }
        6 | 10 => {
            out[0] = ((v0 * v3) >> 8) as f32;
            out[1] = ((v1 * v3) >> 8) as f32;
            out[2] = ((v2 * v3) >> 8) as f32;
            out[3] = 255.0;
            out[4] = v0 as f32;
            out[5] = v1 as f32;
            out[6] = v2 as f32;
            out[7] = 255.0;
            if cem == 10 {
                out[3] = f32::from(dq[4]);
                out[7] = f32::from(dq[5]);
            }
            if let Some(s) = actual_scale {
                *s = v3 as f32 * (1.0 / 256.0);
            }
        }
        8 | 12 => {
            let (mut v0, mut v1, mut v2, mut v3) = (v0, v1, v2, v3);
            let (v4, v5) = (v(4), v(5));
            let (mut v6, mut v7) = (255, 255);
            if cem == 12 {
                v6 = v(6);
                v7 = v(7);
            }
            if (v1 + v3 + v5) >= (v0 + v2 + v4) {
                *out = [
                    v0 as f32, v2 as f32, v4 as f32, v6 as f32, v1 as f32, v3 as f32, v5 as f32,
                    v7 as f32,
                ];
            } else {
                // blue_contract(r, g, b)
                v0 = (v0 + v4) >> 1;
                v2 = (v2 + v4) >> 1;
                v1 = (v1 + v5) >> 1;
                v3 = (v3 + v5) >> 1;
                *out = [
                    v1 as f32, v3 as f32, v5 as f32, v7 as f32, v0 as f32, v2 as f32, v4 as f32,
                    v6 as f32,
                ];
            }
            if let Some(s) = actual_scale {
                *s = 0.0;
            }
        }
        _ => {
            let mut e = [[0i32; 2]; 4];
            ah::decode_endpoint(cem, &mut e, &dq);
            for c in 0..4 {
                out[c] = e[c][0] as f32;
                out[4 + c] = e[c][1] as f32;
            }
            if let Some(s) = actual_scale {
                *s = 0.0;
            }
        }
    }
}

/// `calc_shortest_endpoint_dist` (decoded L RGBA / H RGBA order).
fn calc_shortest_endpoint_dist(a: &[f32; 8], b: &[f32; 8], num_comps: u32) -> f32 {
    let (d0, d1) = if num_comps == 3 {
        (
            vec3_squared_dist(&a[0..], &b[0..]) + vec3_squared_dist(&a[4..], &b[4..]),
            vec3_squared_dist(&a[0..], &b[4..]) + vec3_squared_dist(&a[4..], &b[0..]),
        )
    } else {
        (
            vec4_squared_dist(&a[0..], &b[0..]) + vec4_squared_dist(&a[4..], &b[4..]),
            vec4_squared_dist(&a[0..], &b[4..]) + vec4_squared_dist(&a[4..], &b[0..]),
        )
    };
    fmin(d0, d1)
}

/// `quant_endpoint_val_to_rank`.
#[inline]
fn quant_endpoint_val_to_rank(value: f32, range: u32, num_levels: u32) -> i32 {
    let nl = num_levels as i32;
    if num_levels != 0 && (num_levels & (num_levels - 1)) == 0 {
        f2i(value * (1.0f32 / 255.0) * (num_levels - 1) as f32 + 0.5).clamp(0, nl - 1)
    } else {
        let value = fclamp(value, 0.0, 255.0);
        let v = f2i(value).clamp(0, 255);
        let r0 = ep_val_to_rank(range, v as u32) as i32;
        let v0 = (ep_rank_to_val(range, r0 as u32) as f32 - value).abs();
        let rp = (r0 + 1).min(nl - 1);
        let vp = (ep_rank_to_val(range, rp as u32) as f32 - value).abs();
        if v0 < vp { r0 } else { rp }
    }
}

/// `compute_endpoint_sum`.
#[inline]
fn compute_endpoint_sum(vals: &[u8], range: u32) -> i32 {
    let mut sum = 0;
    for i in 0..6 {
        let mut v = ep_rank_to_val(range, u32::from(vals[i])) as i32;
        if i & 1 == 0 {
            v = -v;
        }
        sum += v;
    }
    sum
}

/// `cem_bc_encode`.
fn cem_bc_encode(cem: u32, range: u32, vals: &mut [u8], num_levels: u32, use_bc: bool) {
    let mut cur_sum = compute_endpoint_sum(vals, range);
    if cur_sum == 0 {
        for i in 0..6 {
            let dir = if i & 1 != 0 { 1 } else { -1 };
            let cur_r = i32::from(vals[i]);
            let new_r = (cur_r + dir).clamp(0, num_levels as i32 - 1);
            if new_r == cur_r {
                continue;
            }
            vals[i] = new_r as u8;
            cur_sum = compute_endpoint_sum(vals, range);
            if cur_sum != 0 {
                break;
            }
        }
    }
    let cur_bc = cur_sum < 0;
    if cur_bc != use_bc {
        let num_comps = if cem >= 12 { 4 } else { 3 };
        for i in 0..num_comps {
            vals.swap(2 * i, 2 * i + 1);
        }
    }
}

/// `encode_cem_8_12`: true if blue contraction was used.
fn encode_cem_8_12(cem: u32, vals: &mut [u8], range: u32, ep: &[f32; 8], allow_bc: bool) -> bool {
    let nl = ah::get_ise_levels(range);
    let n = if cem >= 12 { 8 } else { 6 };
    let mut use_bc = allow_bc && nl < 256;
    let mut enc = [0.0f32; 8];
    if use_bc {
        for i in 0..2 {
            let (mut r, mut g, b) = (ep[i], ep[2 + i], ep[4 + i]);
            r = r * 2.0 - b;
            g = g * 2.0 - b;
            let cr = fclamp(r, 0.0, 255.0);
            let cg = fclamp(g, 0.0, 255.0);
            if r != cr || g != cg {
                use_bc = false;
                break;
            }
            enc[i] = cr;
            enc[2 + i] = cg;
            enc[4 + i] = b;
        }
    }
    if !use_bc {
        enc[..6].copy_from_slice(&ep[..6]);
    }
    if cem == 12 {
        enc[6] = ep[6];
        enc[7] = ep[7];
    }
    for i in 0..n {
        vals[i] = quant_endpoint_val_to_rank(enc[i], range, nl) as u8;
    }
    cem_bc_encode(cem, range, vals, nl, use_bc);
    use_bc
}

/// `encode_cem_6_10`.
fn encode_cem_6_10(cem: u32, vals: &mut [u8], range: u32, ep: &[f32; 8]) {
    let nl = ah::get_ise_levels(range);
    let mut h = [0.0f32; 3];
    for c in 0..3 {
        vals[c] = quant_endpoint_val_to_rank(ep[c * 2 + 1], range, nl) as u8;
        h[c] = ep_rank_to_val(range, u32::from(vals[c])) as f32;
    }
    let l = [ep[0], ep[2], ep[4]];
    let hh_dot = vec3_dot(&h, &h);
    let lh_dot = vec3_dot(&l, &h);
    let scale = (256.0 * lh_dot) / (hh_dot + TINY_EPS);
    vals[3] = quant_endpoint_val_to_rank(scale, range, nl) as u8;
    if cem == 10 {
        vals[4] = quant_endpoint_val_to_rank(ep[6], range, nl) as u8;
        vals[5] = quant_endpoint_val_to_rank(ep[7], range, nl) as u8;
    }
}

/// `encode_cem_0_4`.
fn encode_cem_0_4(cem: u32, vals: &mut [u8], range: u32, ep: &[f32; 8]) {
    let nl = ah::get_ise_levels(range);
    vals[0] = quant_endpoint_val_to_rank(ep[0], range, nl) as u8;
    vals[1] = quant_endpoint_val_to_rank(ep[1], range, nl) as u8;
    if cem == 4 {
        vals[2] = quant_endpoint_val_to_rank(ep[6], range, nl) as u8;
        vals[3] = quant_endpoint_val_to_rank(ep[7], range, nl) as u8;
    }
}

/// `basist::astc_ldr_t::pack_base_offset` (CEM 9 / 13): returns (status, base+offset
/// clamped). Output values are ISE symbols of `dst_range`.
fn pack_base_offset(
    cem: u32,
    dst_range: u32,
    out: &mut [u8],
    l: [u8; 4],
    h: [u8; 4],
    mut use_bc: bool,
    auto_disable_bc_if_clamped: bool,
) -> (bool, bool) {
    let mut bc_clamped = false;
    let mut base_ofs_clamped = false;
    let (mut pack_l, mut pack_h) = (l, h);
    if use_bc {
        let enc_l = blue_contract_enc(pack_l, &mut bc_clamped, i32::from(pack_l[2]));
        let enc_h = blue_contract_enc(pack_h, &mut bc_clamped, i32::from(pack_h[2]));
        if bc_clamped && auto_disable_bc_if_clamped {
            use_bc = false;
        } else {
            pack_h = enc_l;
            pack_l = enc_h;
        }
    }
    let (mut dr, mut dg, mut db, mut da) = (0i32, 0i32, 0i32, 0i32);
    let mut low_clamp = -32;
    for pass in 0..4 {
        let d = |c: usize| i32::from(pack_h[c]) - i32::from(pack_l[c]);
        base_ofs_clamped = false;
        let mut cl = |v: i32| {
            let c = v.clamp(low_clamp, 31);
            if c != v {
                base_ofs_clamped = true;
            }
            c
        };
        dr = cl(d(0));
        dg = cl(d(1));
        db = cl(d(2));
        da = cl(d(3));
        let s = dr + dg + db;
        if (s < 0) == use_bc {
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
    }
    let (mut v0, mut v2, mut v4) = (
        i32::from(pack_l[0]),
        i32::from(pack_l[1]),
        i32::from(pack_l[2]),
    );
    let (mut v1, mut v3, mut v5) = (dr, dg, db);
    bit_transfer_signed_enc(&mut v1, &mut v0);
    bit_transfer_signed_enc(&mut v3, &mut v2);
    bit_transfer_signed_enc(&mut v5, &mut v4);
    let (mut v6, mut v7) = (0i32, 0i32);
    if does_cem_have_alpha(cem) {
        v6 = i32::from(pack_l[3]);
        v7 = da;
        bit_transfer_signed_enc(&mut v7, &mut v6);
    }
    let new8 = [
        v0 as u8, v1 as u8, v2 as u8, v3 as u8, v4 as u8, v5 as u8, v6 as u8, v7 as u8,
    ];
    let status = requantize_from_256(cem, &new8, dst_range, out);
    (status, base_ofs_clamped)
}

/// `encode_cem_9_13`: false if the base+offset encoding clamped. Output is in ISE space.
fn encode_cem_9_13(cem: u32, vals: &mut [u8], range: u32, ep: &[f32; 8], allow_bc: bool) -> bool {
    let nc = get_num_cem_chans(cem) as usize;
    let mut e = [[0u8, 0, 0, 255], [0, 0, 0, 255]];
    for c in 0..nc {
        e[0][c] = fast_roundf_int(ep[c * 2]).clamp(0, 255) as u8;
        e[1][c] = fast_roundf_int(ep[c * 2 + 1]).clamp(0, 255) as u8;
    }
    let (_status, clamped) = pack_base_offset(cem, range, vals, e[0], e[1], allow_bc, true);
    !clamped
}

/// `astc_ldrf::cem_encode`: `endpoints` in ASTC direct order (LR HR LG HG LB HB LA HA),
/// output rank space CEM values (ISE for 9/13). For 9/13, true if base+offset didn't clamp.
pub(crate) fn cem_encode(
    cem: u32,
    endpoints: &[f32; 8],
    range: u32,
    vals: &mut [u8],
    allow_bc: bool,
    high_effort: bool,
) -> bool {
    match cem {
        0 | 4 => {
            encode_cem_0_4(cem, vals, range, endpoints);
            true
        }
        6 | 10 => {
            encode_cem_6_10(cem, vals, range, endpoints);
            true
        }
        8 | 12 => {
            let used_bc = encode_cem_8_12(cem, vals, range, endpoints, allow_bc);
            if high_effort && used_bc {
                let mut ep_bc = [0.0f32; 8];
                cem_decode(cem, vals, range, &mut ep_bc, None);
                let mut plain = [0u8; 8];
                encode_cem_8_12(cem, &mut plain, range, endpoints, false);
                let mut ep_plain = [0.0f32; 8];
                cem_decode(cem, &plain, range, &mut ep_plain, None);
                let nc = get_num_cem_chans(cem) as usize;
                let mut desired = [0.0f32; 8];
                for c in 0..nc {
                    desired[c] = endpoints[c * 2];
                    desired[c + 4] = endpoints[c * 2 + 1];
                }
                let dist_bc = calc_shortest_endpoint_dist(&desired, &ep_bc, nc as u32);
                let dist_plain = calc_shortest_endpoint_dist(&desired, &ep_plain, nc as u32);
                if dist_plain < dist_bc {
                    let n = ah::get_num_cem_values(cem) as usize;
                    vals[..n].copy_from_slice(&plain[..n]);
                }
            }
            true
        }
        9 | 13 => encode_cem_9_13(cem, vals, range, endpoints, allow_bc),
        _ => {
            let n = (ah::get_num_cem_values(cem) as usize).min(vals.len());
            vals[..n].fill(0);
            false
        }
    }
}

// ------------------------------------------------------------------------------------------
// Weight evaluation and endpoint refinement.

/// `eval_weights_first_plane`: rank space weights (`[0,64]` for `BISE_64_LEVELS`).
fn eval_weights_first_plane(
    pixels: &PixelBuf,
    weights: &mut [u8],
    weight_range: u32,
    cem: u32,
    cem_vals: &[u8],
    endpoint_range: u32,
    num_comps: u32,
    chan_to_swap: i32,
) {
    let mut e = [0.0f32; 8];
    cem_decode(cem, cem_vals, endpoint_range, &mut e, None);
    if chan_to_swap >= 0 {
        let c = (chan_to_swap & 3) as usize;
        e.swap(c, 3);
        e.swap(4 + c, 7);
    }
    let mut dir = [
        e[4] - e[0],
        e[5] - e[1],
        e[6] - e[2],
        if num_comps == 4 { e[7] - e[3] } else { 0.0 },
    ];
    let mut nwl = ah::get_ise_levels(weight_range);
    if nwl == 64 {
        nwl = 65;
    }
    let nm1 = nwl - 1;
    let dir_len2 = vec4_dot(&dir, &dir);
    let total = (pixels.width * pixels.height) as usize;
    if dir_len2 < TINY_EPS {
        weights[..total].fill(0);
        return;
    }
    let k = nm1 as f32 / dir_len2;
    for d in dir.iter_mut().take(num_comps as usize) {
        *d *= k;
    }
    let w_bias = 0.5 - vec4_dot(&e[..4], &dir);
    let mut o = 0;
    for y in 0..pixels.height {
        for x in 0..pixels.width {
            let v = pixels.pixel4(x, y);
            let w = if num_comps == 4 {
                (v[0] * dir[0]) + (v[1] * dir[1]) + (v[2] * dir[2]) + (v[3] * dir[3]) + w_bias
            } else {
                (v[0] * dir[0]) + (v[1] * dir[1]) + (v[2] * dir[2]) + w_bias
            };
            let mut qw = f2i(w);
            if qw as u32 > nm1 {
                qw = ((!qw) >> 31) & nm1 as i32;
            }
            weights[o] = qw as u8;
            o += 1;
        }
    }
}

/// `eval_weights_for_plane`.
fn eval_weights_for_plane(
    pixels: &PixelBuf,
    weight_stride: usize,
    weights: &mut [u8],
    weight_range: u32,
    cem: u32,
    cem_vals: &[u8],
    endpoint_range: u32,
    active_chans_mask: u32,
    chan_to_swap_with_alpha: i32,
) {
    let mut e = [0.0f32; 8];
    cem_decode(cem, cem_vals, endpoint_range, &mut e, None);
    let mut dir = [e[4] - e[0], e[5] - e[1], e[6] - e[2], e[7] - e[3]];
    for i in 0..4 {
        if active_chans_mask & (1 << i) == 0 {
            dir[i] = 0.0;
        }
    }
    let mut nwl = ah::get_ise_levels(weight_range);
    if nwl == 64 {
        nwl = 65;
    }
    let nm1 = nwl - 1;
    let dir_len2 = vec4_dot(&dir, &dir);
    let total = (pixels.width * pixels.height) as usize;
    if dir_len2 < TINY_EPS {
        for i in 0..total {
            weights[i * weight_stride] = 0;
        }
        return;
    }
    let k = nm1 as f32 / dir_len2;
    for d in dir.iter_mut() {
        *d *= k;
    }
    let w_bias = 0.5 - vec4_dot(&e[..4], &dir);
    let mut o = 0;
    for y in 0..pixels.height {
        for x in 0..pixels.width {
            let mut v = pixels.pixel4(x, y);
            if chan_to_swap_with_alpha >= 0 {
                v.swap(3, (chan_to_swap_with_alpha & 3) as usize);
            }
            let w = (v[0] * dir[0]) + (v[1] * dir[1]) + (v[2] * dir[2]) + (v[3] * dir[3]) + w_bias;
            let mut qw = f2i(w);
            if qw as u32 > nm1 {
                qw = ((!qw) >> 31) & nm1 as i32;
            }
            weights[o] = qw as u8;
            o += weight_stride;
        }
    }
}

/// Dequantized `[0,64]` weight of rank `qw` for `refine_endpoints_given_weights_*`.
#[inline]
fn refine_dw(weight_range: u32, qw: u8) -> i32 {
    if weight_range <= BISE_32_LEVELS {
        w_rank_to_val(weight_range, u32::from(qw)) as i32
    } else {
        i32::from(qw)
    }
}

/// Least squares alpha (or single channel) endpoints for `refine_*` (`z00`..`det` part).
fn refine_alpha(
    pixels: &PixelBuf,
    weights: &[u8],
    weight_range: u32,
    did_clamp: &mut bool,
) -> (f32, f32) {
    let (mut z00, mut z10, mut z11) = (0.0f32, 0.0f32, 0.0f32);
    let (mut q00_a, mut t_a) = (0.0f32, 0.0f32);
    let mut i = 0;
    for y in 0..pixels.height {
        for x in 0..pixels.width {
            let a = pixels.get(x, y, 3);
            let w = refine_dw(weight_range, weights[i]) as f32 * (1.0 / 64.0);
            i += 1;
            z00 += w * w;
            z10 += (1.0 - w) * w;
            z11 += (1.0 - w) * (1.0 - w);
            q00_a += w * a;
            t_a += a;
        }
    }
    let q10_a = t_a - q00_a;
    let z01 = z10;
    let mut det = z00 * z11 - z01 * z10;
    if det >= TINY_EPS {
        det = 1.0 / det;
        let iz00 = z11 * det;
        let iz01 = -z01 * det;
        let iz10 = -z10 * det;
        let iz11 = z00 * det;
        let xl = clamp_flag(iz10 * q00_a + iz11 * q10_a, 0.0, 255.0, did_clamp);
        let xh = clamp_flag(iz00 * q00_a + iz01 * q10_a, 0.0, 255.0, did_clamp);
        (xl, xh)
    } else {
        let xl = t_a / (pixels.width * pixels.height) as f32;
        (xl, xl)
    }
}

/// `refine_endpoints_given_weights_cem_6_or_10_method3`.
fn refine_endpoints_cem_6_or_10(
    pixels: &PixelBuf,
    weights: &[u8],
    weight_range: u32,
    out: &mut [f32; 8],
    num_comps: u32,
) -> bool {
    let pixel_count_f = (pixels.width * pixels.height) as f32;
    let mut did_clamp = false;
    let (mut sum_w, mut sum_w2) = (0.0f32, 0.0f32);
    let mut rgb_sum = [0.0f32; 3];
    let mut weighted_rgb_sum = [0.0f32; 3];
    let (mut min_dw, mut max_dw) = (256i32, 0i32);
    let mut i = 0;
    for y in 0..pixels.height {
        for x in 0..pixels.width {
            let p = [
                pixels.get(x, y, 0),
                pixels.get(x, y, 1),
                pixels.get(x, y, 2),
            ];
            let dw = refine_dw(weight_range, weights[i]);
            i += 1;
            min_dw = min_dw.min(dw);
            max_dw = max_dw.max(dw);
            let w = dw as f32 * (1.0 / 64.0);
            sum_w += w;
            sum_w2 += w * w;
            for c in 0..3 {
                rgb_sum[c] += p[c];
            }
            for c in 0..3 {
                weighted_rgb_sum[c] += p[c] * w;
            }
        }
    }
    let rgb_mean = rgb_sum.map(|v| v / pixel_count_f);
    let a_ = pixel_count_f - 2.0 * sum_w + sum_w2;
    let b_ = sum_w - sum_w2;
    let c_ = sum_w2;
    let pb = weighted_rgb_sum;
    let pa = [
        rgb_sum[0] - weighted_rgb_sum[0],
        rgb_sum[1] - weighted_rgb_sum[1],
        rgb_sum[2] - weighted_rgb_sum[2],
    ];
    const K_MAX_SCALE01: f32 = 255.0 / 256.0;
    // (best_obj, best_scale01)
    let mut best = (-1.0f32, K_MAX_SCALE01);
    let try_scale = |best: &mut (f32, f32), s: f32| {
        if !(0.0..=K_MAX_SCALE01).contains(&s) {
            return;
        }
        let den = a_ * s * s + 2.0 * b_ * s + c_;
        if den < TINY_EPS {
            return;
        }
        let v = [pa[0] * s + pb[0], pa[1] * s + pb[1], pa[2] * s + pb[2]];
        let obj = vec3_dot(&v, &v) / den;
        if obj > best.0 {
            *best = (obj, s);
        }
    };

    let mut out_high = rgb_mean;
    let mut out_scale = K_MAX_SCALE01;

    if max_dw == min_dw {
        let w = min_dw as f32 * (1.0 / 64.0);
        let scale01 = K_MAX_SCALE01;
        let f = scale01 + w * (1.0 - scale01);
        let max_mean = rgb_mean[0].max(rgb_mean[1].max(rgb_mean[2]));
        if f > TINY_EPS && max_mean <= 255.0 * f {
            for v in out_high.iter_mut() {
                *v /= f;
            }
        }
        for v in out_high.iter_mut() {
            *v = clamp_flag(*v, 0.0, 255.0, &mut did_clamp);
        }
        out_scale = scale01;
    } else {
        let n2 = vec3_dot(&pa, &pa);
        let n1 = vec3_dot(&pa, &pb);
        let n0 = vec3_dot(&pb, &pb);
        let q1 = n2 * c_ - n0 * a_;
        let q0 = n1 * c_ - n0 * b_;
        let q2 = n2 * b_ - n1 * a_;
        let (aq0, aq1, aq2) = (q0.abs(), q1.abs(), q2.abs());
        let m = aq0.max(aq1.max(aq2));
        if m > 0.0 {
            let s = 1.0 / m;
            let a = q2 * s;
            let b = q1 * s;
            let c = q0 * s;
            if a.abs() > TINY_EPS {
                let disc = b * b - 4.0 * a * c;
                if disc >= 0.0 {
                    let sqrt_disc = disc.sqrt();
                    let t = -0.5 * (b + sqrt_disc.copysign(b));
                    if t != 0.0 {
                        try_scale(&mut best, t / a);
                        try_scale(&mut best, c / t);
                    } else {
                        try_scale(&mut best, -b * (0.5 / a));
                    }
                }
            } else if b.abs() > TINY_EPS {
                try_scale(&mut best, -c / b);
            }
        }
        if best.0 < 0.0 {
            for s in [K_MAX_SCALE01, 0.875, 0.75, 0.5, 0.25] {
                try_scale(&mut best, s);
            }
        }
        if best.0 >= 0.0 {
            let scale01 = best.1;
            let high_den = a_ * scale01 * scale01 + 2.0 * b_ * scale01 + c_;
            if high_den > TINY_EPS {
                for c in 0..3 {
                    out_high[c] = pa[c] * scale01 + pb[c];
                }
                for v in out_high.iter_mut() {
                    *v /= high_den;
                }
            }
            for v in out_high.iter_mut() {
                *v = clamp_flag(*v, 0.0, 255.0, &mut did_clamp);
            }
            out_scale = clamp_flag(scale01, 0.0, K_MAX_SCALE01, &mut did_clamp);
        }
    }

    for c in 0..3 {
        out[c * 2] = out_scale * out_high[c];
        out[c * 2 + 1] = out_high[c];
    }
    out[6] = 255.0;
    out[7] = 255.0;
    if num_comps == 4 {
        let (xl, xh) = refine_alpha(pixels, weights, weight_range, &mut did_clamp);
        out[6] = xl;
        out[7] = xh;
    }
    did_clamp
}

/// `refine_endpoints_given_weights_cem_8_9_12_or_13`.
fn refine_endpoints_cem_8_9_12_13(
    pixels: &PixelBuf,
    weights: &[u8],
    weight_range: u32,
    out: &mut [f32; 8],
    num_comps: u32,
) -> bool {
    let nc = num_comps as usize;
    let (mut z00, mut z10, mut z11) = (0.0f32, 0.0f32, 0.0f32);
    let mut q00_a = [0.0f32; 4];
    let mut t_a = [0.0f32; 4];
    let mut did_clamp = false;
    let mut i = 0;
    for y in 0..pixels.height {
        for x in 0..pixels.width {
            let w = refine_dw(weight_range, weights[i]) as f32 * (1.0 / 64.0);
            i += 1;
            z00 += w * w;
            z10 += (1.0 - w) * w;
            z11 += (1.0 - w) * (1.0 - w);
            for c in 0..nc {
                let a = pixels.get(x, y, c as u32);
                q00_a[c] += w * a;
                t_a[c] += a;
            }
        }
    }
    let mut q10_a = [0.0f32; 4];
    for c in 0..nc {
        q10_a[c] = t_a[c] - q00_a[c];
    }
    let z01 = z10;
    let mut det = z00 * z11 - z01 * z10;
    if det >= TINY_EPS {
        det = 1.0 / det;
        let iz00 = z11 * det;
        let iz01 = -z01 * det;
        let iz10 = -z10 * det;
        let iz11 = z00 * det;
        for c in 0..nc {
            out[c * 2] = clamp_flag(
                iz10 * q00_a[c] + iz11 * q10_a[c],
                0.0,
                255.0,
                &mut did_clamp,
            );
            out[c * 2 + 1] = clamp_flag(
                iz00 * q00_a[c] + iz01 * q10_a[c],
                0.0,
                255.0,
                &mut did_clamp,
            );
        }
    } else {
        let oo = 1.0f32 / (pixels.width * pixels.height) as f32;
        for c in 0..nc {
            out[c * 2] = t_a[c] * oo;
            out[c * 2 + 1] = out[c * 2];
        }
    }
    if num_comps == 3 {
        out[6] = 255.0;
        out[7] = 255.0;
    }
    did_clamp
}

/// `refine_endpoints_given_weights_cem_0_or_4`.
fn refine_endpoints_cem_0_or_4(
    pixels: &PixelBuf,
    cem: u32,
    weights: &[u8],
    weight_range: u32,
    out: &mut [f32; 8],
) -> bool {
    let (mut z00, mut z10, mut z11) = (0.0f32, 0.0f32, 0.0f32);
    let mut q00_a = [0.0f32; 2];
    let mut t_a = [0.0f32; 2];
    let mut did_clamp = false;
    let nac = if cem == 4 { 2 } else { 1 };
    let mut i = 0;
    for y in 0..pixels.height {
        for x in 0..pixels.width {
            let w = refine_dw(weight_range, weights[i]) as f32 * (1.0 / 64.0);
            i += 1;
            z00 += w * w;
            z10 += (1.0 - w) * w;
            z11 += (1.0 - w) * (1.0 - w);
            for c in 0..nac {
                let a = pixels.get(x, y, if c != 0 { 3 } else { 0 });
                q00_a[c] += w * a;
                t_a[c] += a;
            }
        }
    }
    let mut q10_a = [0.0f32; 2];
    for c in 0..nac {
        q10_a[c] = t_a[c] - q00_a[c];
    }
    let z01 = z10;
    out[6] = 255.0;
    out[7] = 255.0;
    let mut det = z00 * z11 - z01 * z10;
    if det >= TINY_EPS {
        det = 1.0 / det;
        let iz00 = z11 * det;
        let iz01 = -z01 * det;
        let iz10 = -z10 * det;
        let iz11 = z00 * det;
        for c in 0..nac {
            let l = clamp_flag(
                iz10 * q00_a[c] + iz11 * q10_a[c],
                0.0,
                255.0,
                &mut did_clamp,
            );
            let h = clamp_flag(
                iz00 * q00_a[c] + iz01 * q10_a[c],
                0.0,
                255.0,
                &mut did_clamp,
            );
            if c == 0 {
                out[0] = l;
                out[1] = h;
            } else {
                out[6] = l;
                out[7] = h;
            }
        }
    } else {
        let oo = 1.0f32 / (pixels.width * pixels.height) as f32;
        for c in 0..nac {
            let l = t_a[c] * oo;
            if c == 0 {
                out[0] = l;
                out[1] = l;
            } else {
                out[6] = l;
                out[7] = l;
            }
        }
    }
    out[2] = out[0];
    out[3] = out[1];
    out[4] = out[0];
    out[5] = out[1];
    did_clamp
}

/// `refine_endpoints_given_weights`: true if clamping occured.
fn refine_endpoints_given_weights(
    pixels: &PixelBuf,
    cem: u32,
    weights: &[u8],
    weight_range: u32,
    out: &mut [f32; 8],
    num_comps: u32,
) -> bool {
    match cem {
        0 | 4 => refine_endpoints_cem_0_or_4(pixels, cem, weights, weight_range, out),
        6 | 10 => refine_endpoints_cem_6_or_10(pixels, weights, weight_range, out, num_comps),
        8 | 9 | 12 | 13 => {
            refine_endpoints_cem_8_9_12_13(pixels, weights, weight_range, out, num_comps)
        }
        _ => false,
    }
}

/// `eval_weights_second_plane` (in place on `cem_vals`; the CCS channel was rotated into A
/// when `ccs_chan_rotated_flag`).
fn eval_weights_second_plane(
    pixels: &PixelBuf,
    cem: u32,
    ccs_index: u32,
    weights0: &mut [u8],
    weights1: &mut [u8],
    weight_range: u32,
    desired: &mut [f32; 8],
    cem_vals: &mut [u8],
    endpoint_range: u32,
    ccs_chan_rotated_flag: bool,
    ctx: &SingleSubsetEncContext,
) {
    let (mut lo_v, mut hi_v) = (1e30f32, -1e30f32);
    let src_ccs = if ccs_chan_rotated_flag { 3 } else { ccs_index };
    for y in 0..pixels.height {
        for x in 0..pixels.width {
            let v = pixels.get(x, y, src_ccs);
            lo_v = fmin(lo_v, v);
            hi_v = fmax(hi_v, v);
        }
    }
    let nel = ah::get_ise_levels(endpoint_range);
    if ccs_index == 3 {
        let lo_q = quant_endpoint_val_to_rank(lo_v, endpoint_range, nel) as u8;
        let hi_q = quant_endpoint_val_to_rank(hi_v, endpoint_range, nel) as u8;
        match cem {
            10 => {
                cem_vals[4] = lo_q;
                cem_vals[5] = hi_q;
                desired[4] = lo_v;
                desired[5] = hi_v;
            }
            12 => {
                cem_vals[6] = lo_q;
                cem_vals[7] = hi_q;
                desired[6] = lo_v;
                desired[7] = hi_v;
            }
            4 => {
                cem_vals[2] = lo_q;
                cem_vals[3] = hi_q;
                desired[2] = lo_v;
                desired[3] = hi_v;
            }
            _ => {}
        }
    } else if is_cem_8_or_12(cem) {
        desired[6] = lo_v;
        desired[7] = hi_v;
        let ci = (ccs_index & 3) as usize;
        desired.swap(6, ci * 2);
        desired.swap(7, ci * 2 + 1);
        cem_encode(
            cem,
            desired,
            endpoint_range,
            cem_vals,
            true,
            ctx.higher_effort_bc,
        );
    }

    let mut fe = [0.0f32; 8];
    cem_decode(cem, cem_vals, endpoint_range, &mut fe, None);
    let nwl = ah::get_ise_levels(weight_range);
    let ci = (ccs_index & 3) as usize;
    let l = fe[ci];
    let h = fe[4 + ci];
    let scale = (nwl - 1) as f32 / ((h - l) + TINY_EPS);
    let mut o = 0;
    for y in 0..pixels.height {
        for x in 0..pixels.width {
            let v = pixels.get(x, y, src_ccs);
            weights1[o] = f2i((v - l) * scale + 0.5).clamp(0, nwl as i32 - 1) as u8;
            o += 1;
        }
    }
    if ccs_index < 3 && is_cem_8_or_12(cem) {
        eval_weights_first_plane(
            pixels,
            weights0,
            weight_range,
            cem,
            cem_vals,
            endpoint_range,
            3,
            ccs_index as i32,
        );
    }
}

/// `try_base_ofs_cem`.
fn try_base_ofs_cem(
    cem_index: u32,
    block: &PixelBuf,
    log_blk: &mut LogAstcBlock,
    desired: &[f32; 8],
    ccs_chan_rotated_flag: bool,
) {
    if is_cem_9_or_13(u32::from(log_blk.color_endpoint_modes[0])) {
        return;
    }
    let erange = u32::from(log_blk.endpoint_ise_range);
    if erange == BISE_256_LEVELS {
        return;
    }
    let mut bo = [0u8; 8];
    let cem_bo = if cem_index == 8 { 9 } else { 13 };
    let nc = get_num_cem_chans(cem_index);
    let n_bo = ah::get_num_cem_values(cem_bo) as usize;
    if !encode_cem_9_13(cem_bo, &mut bo, erange, desired, true) {
        return;
    }
    let ise_to_rank = &ah::dequant_tables().endpoint_tab(erange).ise_to_rank;
    for v in bo.iter_mut().take(n_bo) {
        *v = ise_to_rank.get(*v as usize).copied().unwrap_or(0);
    }
    let mut ep_bo = [0.0f32; 8];
    cem_decode(cem_bo, &bo, erange, &mut ep_bo, None);
    let mut ep_direct = [0.0f32; 8];
    cem_decode(cem_index, &log_blk.endpoints, erange, &mut ep_direct, None);
    let mut d = [0.0f32; 8];
    for c in 0..nc as usize {
        d[c] = desired[c * 2];
        d[c + 4] = desired[c * 2 + 1];
    }
    let dist_bo = calc_shortest_endpoint_dist(&d, &ep_bo, nc);
    let dist_direct = calc_shortest_endpoint_dist(&d, &ep_direct, nc);
    if dist_bo < dist_direct {
        log_blk.color_endpoint_modes[0] = cem_bo as u8;
        log_blk.endpoints[..n_bo].copy_from_slice(&bo[..n_bo]);
        let w1_mask = if log_blk.dual_plane {
            1u32 << log_blk.color_component_selector
        } else {
            0
        };
        let w0_mask = ((1u32 << nc) - 1) ^ w1_mask;
        let swap = if ccs_chan_rotated_flag {
            i32::from(log_blk.color_component_selector)
        } else {
            -1
        };
        let stride = if log_blk.dual_plane { 2 } else { 1 };
        let endpoints = log_blk.endpoints;
        let wrange = u32::from(log_blk.weight_ise_range);
        eval_weights_for_plane(
            block,
            stride,
            &mut log_blk.weights,
            wrange,
            cem_bo,
            &endpoints,
            erange,
            w0_mask,
            swap,
        );
        if log_blk.dual_plane {
            eval_weights_for_plane(
                block,
                2,
                &mut log_blk.weights[1..],
                wrange,
                cem_bo,
                &endpoints,
                erange,
                w1_mask,
                swap,
            );
        }
    }
}

/// `get_pca_config`: (num_pca_comps, ccs_chan_rotated_flag).
fn get_pca_config(cfg: &AstcUnpackedConfig) -> (u32, bool) {
    if cfg.dual_plane != 0 {
        if cfg.cem != 4 && cfg.ccs_index < 3 {
            let n = if cfg.cem == 10 { 4 } else { 3 };
            return (n, is_cem_8_or_12(u32::from(cfg.cem)));
        }
        (3, false)
    } else {
        (if cfg.cem == 4 || cfg.cem >= 10 { 4 } else { 3 }, false)
    }
}

/// `create_single_subset_block`.
fn create_single_subset_block(
    block: &PixelBuf,
    log_blk: &mut LogAstcBlock,
    ctx: &SingleSubsetEncContext,
    num_pca_comps: u32,
    ccs_chan_rotated_flag: bool,
    try_base_ofs: bool,
) {
    let cem = u32::from(log_blk.color_endpoint_modes[0]);
    let erange = u32::from(log_blk.endpoint_ise_range);
    let wrange = u32::from(log_blk.weight_ise_range);
    let mut desired = calc_initial_cem_endpoints(block, num_pca_comps, None, is_cem_6_or_10(cem));
    cem_encode(
        cem,
        &desired,
        erange,
        &mut log_blk.endpoints,
        true,
        ctx.higher_effort_bc,
    );
    eval_weights_first_plane(
        block,
        &mut log_blk.weights,
        wrange,
        cem,
        &log_blk.endpoints,
        erange,
        num_pca_comps,
        -1,
    );
    for _ in 0..ctx.num_ls_iterations {
        refine_endpoints_given_weights(
            block,
            cem,
            &log_blk.weights,
            wrange,
            &mut desired,
            num_pca_comps,
        );
        cem_encode(
            cem,
            &desired,
            erange,
            &mut log_blk.endpoints,
            true,
            ctx.higher_effort_bc,
        );
        eval_weights_first_plane(
            block,
            &mut log_blk.weights,
            wrange,
            cem,
            &log_blk.endpoints,
            erange,
            num_pca_comps,
            -1,
        );
    }
    if log_blk.dual_plane {
        let mut w1 = [0u8; MAX_GRID_WEIGHTS as usize];
        let ccs = u32::from(log_blk.color_component_selector);
        eval_weights_second_plane(
            block,
            cem,
            ccs,
            &mut log_blk.weights,
            &mut w1,
            wrange,
            &mut desired,
            &mut log_blk.endpoints,
            erange,
            ccs_chan_rotated_flag,
            ctx,
        );
        let n = usize::from(log_blk.grid_width) * usize::from(log_blk.grid_height);
        let mut w0 = [0u8; MAX_GRID_WEIGHTS as usize];
        w0[..n].copy_from_slice(&log_blk.weights[..n]);
        for i in 0..n.min(32) {
            log_blk.weights[i * 2] = w0[i];
            log_blk.weights[i * 2 + 1] = w1[i];
        }
    }
    if try_base_ofs && (cem == 8 || cem == 12) {
        try_base_ofs_cem(cem, block, log_blk, &desired, ccs_chan_rotated_flag);
    }
}

/// `encode_single_subset_block`: result in rank space.
fn encode_single_subset_block(
    ctx: &SingleSubsetEncContext,
    src_block: &PixelBuf,
    num_src_block_comps: u32,
    cfg: &AstcUnpackedConfig,
    log_blk: &mut LogAstcBlock,
    try_base_ofs: bool,
) {
    log_blk.clear();
    log_blk.grid_width = cfg.grid_width;
    log_blk.grid_height = cfg.grid_height;
    log_blk.endpoint_ise_range = cfg.endpoint_range;
    log_blk.weight_ise_range = cfg.weight_range;
    log_blk.dual_plane = cfg.dual_plane != 0;
    log_blk.color_component_selector = cfg.ccs_index;
    log_blk.num_partitions = 1;
    log_blk.color_endpoint_modes[0] = cfg.cem;
    log_blk.user_mode = USER_MODE_RANK_VALUES;

    let (num_pca_comps, rotated) = get_pca_config(cfg);
    let mut grid = PixelBuf::new(u32::from(cfg.grid_width), u32::from(cfg.grid_height));
    let nchans = get_num_cem_chans(u32::from(cfg.cem)).min(num_src_block_comps);
    pseudoinverse_block_to_grid(src_block, &mut grid, nchans);
    if nchans == 3 {
        grid.set_comp_to_val(3, 255.0);
    }
    if rotated {
        grid.swap_comp_with_alpha(u32::from(cfg.ccs_index));
    }
    create_single_subset_block(&grid, log_blk, ctx, num_pca_comps, rotated, try_base_ofs);
}

/// `weight_polish` (1-3 subsets): returns true if `lblock_to_refine` changed.
fn weight_polish(
    src_block: &PixelBuf,
    lblock_to_refine: &mut LogAstcBlock,
    ctx: &SingleSubsetEncContext,
    grid_data: &AstcBlockGridData,
    best_lblock_error: &mut f64,
    best_lblock: &mut LogAstcBlock,
    all_candidates: Option<&mut Vec<LogAstcBlock>>,
) -> bool {
    let (bw, bh) = (ctx.block_width, ctx.block_height);
    let wrange = u32::from(lblock_to_refine.weight_ise_range);
    let wt = ah::dequant_tables().weight_tab(wrange);
    let nwl = ah::get_ise_levels(wrange) as i32;

    let mut refined = *lblock_to_refine;
    convert_rank_lblock_to_ise(&mut refined);
    let mut changed = false;
    let (gw, gh) = (
        u32::from(refined.grid_width),
        u32::from(refined.grid_height),
    );
    let num_planes = if refined.dual_plane { 2 } else { 1 };

    let Some(dec) = XuastcLdrBlockDecoder::new(
        &refined,
        bw,
        bh,
        ctx.astc_decode_mode_srgb,
        &grid_data.upsample_weights,
    ) else {
        return false;
    };

    for y in 0..gh {
        for x in 0..gw {
            let idx = (x + y * gw) as usize;
            let Some(influenced) = grid_data.grid_to_texel_influence_list.get(idx) else {
                continue;
            };
            let cur_rank = i32::from(
                wt.ise_to_rank
                    .get(refined.weights[idx] as usize)
                    .copied()
                    .unwrap_or(0),
            );
            let widx = idx * num_planes;
            let orig_weight_ise = refined.weights[widx];

            let texel_err = |weights: &[u8; 64]| -> f64 {
                let mut e = 0.0f64;
                for &packed in influenced {
                    let (tx, ty) = (u32::from(packed & 0xFF), u32::from(packed >> 8));
                    let c = dec.decode_texel(weights, tx, ty);
                    e += f64::from(calc_weighted_error(c, src_block.pixel4(tx, ty), ctx));
                }
                e
            };

            let mut best_err = texel_err(&refined.weights);
            if best_err == 0.0 {
                continue;
            }
            let mut best_weight_ise = orig_weight_ise;
            for idir in [-1i32, 1] {
                let new_rank = (cur_rank + idir).clamp(0, nwl - 1);
                let new_ise = wt.rank_to_ise.get(new_rank as usize).copied().unwrap_or(0);
                refined.weights[widx] = new_ise;
                let trial_err = texel_err(&refined.weights);
                refined.weights[widx] = orig_weight_ise;
                if trial_err < best_err {
                    best_err = trial_err;
                    best_weight_ise = new_ise;
                    changed = true;
                    if best_err == 0.0 {
                        break;
                    }
                }
            }
            refined.weights[widx] = best_weight_ise;
        }
    }

    convert_ise_lblock_to_rank(&mut refined);
    if changed {
        let refined_error = compute_block_error(&refined, src_block, ctx);
        if refined_error < *best_lblock_error {
            *best_lblock_error = refined_error;
            *best_lblock = refined;
        }
        if let Some(all) = all_candidates {
            all.push(refined);
        }
        *lblock_to_refine = refined;
    }
    changed
}

/// `compress_single_subset_internal`.
fn compress_single_subset_internal(
    ctx: &SingleSubsetEncContext,
    pixels: &[[u8; 4]],
    best_lblock: &mut LogAstcBlock,
    mut all_candidates: Option<&mut Vec<LogAstcBlock>>,
    state: &mut ShortlistState,
    always_compute_error: bool,
    scale_weight: f32,
) -> f64 {
    let total = ctx.total_block_pixels as usize;
    let first = pixels[0];
    let last = total - 1;
    if pixels[last] == first {
        let mut i = 1;
        while i < last {
            if pixels[i] != first {
                break;
            }
            i += 1;
        }
        if i == last {
            set_ldr_solid_block(
                best_lblock,
                u32::from(first[0]),
                u32::from(first[1]),
                u32::from(first[2]),
                u32::from(first[3]),
            );
            if let Some(all) = all_candidates {
                all.push(*best_lblock);
            }
            return 0.0;
        }
    }

    let mut num_src_block_comps = 3;
    let mut src_is_luma_only = true;
    if ctx.has_alpha {
        for p in &pixels[..total] {
            if p[0] != p[1] || p[0] != p[2] {
                src_is_luma_only = false;
            }
            if p[3] != 255 {
                num_src_block_comps = 4;
            }
        }
    } else {
        for p in &pixels[..total] {
            if p[0] != p[1] || p[0] != p[2] {
                src_is_luma_only = false;
                break;
            }
        }
    }

    let configs: &[u32] = if src_is_luma_only {
        &SINGLE_SUBSET_CONFIGS_LA
    } else {
        &SINGLE_SUBSET_CONFIGS_RGBA
    };

    init_single_subset_shortlist_state(pixels, ctx, state, src_is_luma_only, num_src_block_comps);
    let total_candidates = generate_single_subset_shortlist(
        configs,
        ctx,
        num_src_block_comps,
        state,
        scale_weight,
        ctx.max_candidates,
    );

    if total_candidates == 0 {
        set_ldr_solid_block(best_lblock, 0xFF, 0, 0xFF, 0xFF);
        if let Some(all) = all_candidates {
            all.push(*best_lblock);
        }
        return f64::MAX;
    }

    let mut best_err = f64::MAX;
    for cand_index in 0..total_candidates as usize {
        let mut trial = LogAstcBlock::default();
        let cfg = state.best_configs[cand_index];
        encode_single_subset_block(
            ctx,
            &state.pbuf,
            num_src_block_comps,
            &cfg,
            &mut trial,
            ctx.try_base_ofs,
        );
        if let Some(all) = all_candidates.as_deref_mut() {
            all.push(trial);
        }
        if always_compute_error || total_candidates > 1 {
            let err = compute_block_error(&trial, &state.pbuf, ctx);
            if err < best_err {
                best_err = err;
                *best_lblock = trial;
            }
        } else {
            best_err = -1.0;
            *best_lblock = trial;
        }
    }

    if ctx.weight_polishing
        && let Some(gd) = find_astc_block_grid_data(
            ctx.block_width,
            ctx.block_height,
            u32::from(best_lblock.grid_width),
            u32::from(best_lblock.grid_height),
        )
    {
        // C++ passes best_lblock as both the block to refine and the best block (aliased).
        let mut to_refine = *best_lblock;
        weight_polish(
            &state.pbuf,
            &mut to_refine,
            ctx,
            gd,
            &mut best_err,
            best_lblock,
            all_candidates,
        );
        *best_lblock = to_refine;
    }

    convert_rank_lblock_to_ise(best_lblock);
    best_err
}

/// `astc_ldrf::compress_single_subset`: `best_lblock` is in ISE space, candidates in rank
/// space.
pub(crate) fn compress_single_subset(
    ctx: &SingleSubsetEncContext,
    pixels: &[[u8; 4]],
    best_lblock: &mut LogAstcBlock,
    all_candidates: Option<&mut Vec<LogAstcBlock>>,
    always_compute_error: bool,
) -> f64 {
    let mut state = ShortlistState::new();
    compress_single_subset_internal(
        ctx,
        pixels,
        best_lblock,
        all_candidates,
        &mut state,
        always_compute_error,
        DEF_SCALE_WEIGHT,
    )
}

// ------------------------------------------------------------------------------------------
// Subsets.

type SubsetIndices = [[u32; 256]; 3];

/// `weight_gradient_descent`: refines the weights given the current 2/3 subset endpoints.
fn weight_gradient_descent(
    src_block: &PixelBuf,
    lblock_to_refine: &mut LogAstcBlock,
    subset_pbuf: &[PixelBuf],
    subset_pixel_indices: &SubsetIndices,
    cfg: &AstcUnpackedConfig,
    ctx: &SingleSubsetEncContext,
    grid_data: &AstcBlockGridData,
    best_lblock_error: &mut f64,
    best_lblock: &mut LogAstcBlock,
) -> bool {
    let (bw, bh) = (ctx.block_width, ctx.block_height);
    let total_block_pixels = (bw * bh) as usize;
    let cem = u32::from(cfg.cem);
    let n_cem = ah::get_num_cem_values(cem) as usize;
    let (gw, gh) = (u32::from(cfg.grid_width), u32::from(cfg.grid_height));
    let total_grid = (gw * gh) as usize;
    let wrange = u32::from(cfg.weight_range);
    let erange = u32::from(cfg.endpoint_range);
    let num_partitions = usize::from(lblock_to_refine.num_partitions).min(3);

    let mut ideal = [0u8; MAX_BLOCK_PIXELS];
    for s in 0..num_partitions {
        let mut w = [0u8; MAX_BLOCK_PIXELS];
        eval_weights_first_plane(
            &subset_pbuf[s],
            &mut w,
            BISE_64_LEVELS,
            cem,
            &lblock_to_refine.endpoints[s * n_cem..],
            erange,
            get_num_cem_chans(cem),
            -1,
        );
        for j in 0..subset_pbuf[s].width as usize {
            let pi = subset_pixel_indices[s][j] as usize;
            if pi < total_block_pixels {
                ideal[pi] = w[j];
            }
        }
    }

    let mut dq_grid = [0u8; MAX_GRID_WEIGHTS as usize];
    for i in 0..total_grid {
        dq_grid[i] = w_rank_to_val(wrange, u32::from(lblock_to_refine.weights[i])) as u8;
    }
    let mut dq_block = [0u8; MAX_BLOCK_PIXELS];
    if u32::from(lblock_to_refine.grid_width) == bw && u32::from(lblock_to_refine.grid_height) == bh
    {
        dq_block[..total_block_pixels].copy_from_slice(&dq_grid[..total_block_pixels]);
    } else {
        upsample_weight_grid_xuastc_ldr(
            bw,
            bh,
            u32::from(lblock_to_refine.grid_width),
            u32::from(lblock_to_refine.grid_height),
            &dq_grid,
            &mut dq_block,
        );
    }

    let mut residuals = [0i32; MAX_BLOCK_PIXELS];
    for i in 0..total_block_pixels {
        residuals[i] = i32::from(ideal[i]) - i32::from(dq_block[i]);
    }

    // downsample_weight_residual_grid
    let at = &grid_data.unweighted_downsample_matrix;
    let mut down = [0.0f32; MAX_GRID_WEIGHTS as usize];
    for g in 0..total_grid {
        let row = &at[g * total_block_pixels..(g + 1) * total_block_pixels];
        let mut total = 0.0f32;
        for i in 0..total_block_pixels {
            if row[i] != 0.0 {
                total += row[i] * residuals[i] as f32;
            }
        }
        down[g] = total;
    }
    for i in 0..total_grid {
        down[i] *= grid_data.one_over_diag_ata[i];
    }

    const Q: f32 = 1.0;
    let mut refined = *lblock_to_refine;
    let mut changed = false;
    for i in 0..total_grid {
        let v = w_rank_to_val(wrange, u32::from(lblock_to_refine.weights[i])) as f32 + down[i] * Q;
        let iv = f2i(v.round()).clamp(0, 64);
        let new_weight = w_val_to_rank(wrange, iv as u32) as u8;
        if refined.weights[i] != new_weight {
            refined.weights[i] = new_weight;
            changed = true;
        }
    }

    if changed {
        let refined_error = compute_block_error(&refined, src_block, ctx);
        if refined_error < *best_lblock_error {
            *best_lblock_error = refined_error;
            *best_lblock = refined;
        }
        *lblock_to_refine = refined;
    }
    changed
}

/// `refine_endpoints_with_current_weights`: returns true if the error improved.
fn refine_endpoints_with_current_weights(
    src_block: &PixelBuf,
    lblock_to_refine: &mut LogAstcBlock,
    subset_pbuf: &[PixelBuf],
    subset_pixel_indices: &SubsetIndices,
    cfg: &AstcUnpackedConfig,
    ctx: &SingleSubsetEncContext,
    best_lblock_error: &mut f64,
    best_lblock: &mut LogAstcBlock,
) -> bool {
    let (bw, bh) = (ctx.block_width, ctx.block_height);
    let total_block_pixels = (bw * bh) as usize;
    let cem = u32::from(cfg.cem);
    let n_cem = ah::get_num_cem_values(cem) as usize;
    let total_grid = usize::from(cfg.grid_width) * usize::from(cfg.grid_height);
    let num_partitions = usize::from(lblock_to_refine.num_partitions).min(3);
    let wrange = u32::from(cfg.weight_range);
    let erange = u32::from(cfg.endpoint_range);

    let mut trial = *lblock_to_refine;
    let mut dq_grid = [0u8; MAX_GRID_WEIGHTS as usize];
    for i in 0..total_grid {
        dq_grid[i] = w_rank_to_val(wrange, u32::from(trial.weights[i])) as u8;
    }
    let mut dq_block = [0u8; MAX_BLOCK_PIXELS];
    if u32::from(trial.grid_width) == bw && u32::from(trial.grid_height) == bh {
        dq_block[..total_block_pixels].copy_from_slice(&dq_grid[..total_block_pixels]);
    } else {
        upsample_weight_grid_xuastc_ldr(
            bw,
            bh,
            u32::from(trial.grid_width),
            u32::from(trial.grid_height),
            &dq_grid,
            &mut dq_block,
        );
    }
    let nchans = get_num_cem_chans(cem);
    for s in 0..num_partitions {
        let n = subset_pbuf[s].width as usize;
        let mut sw = [0u8; MAX_BLOCK_PIXELS];
        for i in 0..n {
            sw[i] = dq_block[(subset_pixel_indices[s][i] as usize).min(MAX_BLOCK_PIXELS - 1)];
        }
        let mut nf = [0.0f32; 8];
        refine_endpoints_given_weights(&subset_pbuf[s], cem, &sw, BISE_64_LEVELS, &mut nf, nchans);
        let mut nv = [0u8; 8];
        cem_encode(cem, &nf, erange, &mut nv, true, ctx.higher_effort_bc);
        trial.endpoints[n_cem * s..n_cem * (s + 1)].copy_from_slice(&nv[..n_cem]);
    }

    let trial_error = compute_block_error(&trial, src_block, ctx);
    let mut status = false;
    if trial_error < *best_lblock_error {
        *best_lblock_error = trial_error;
        *best_lblock = trial;
        status = true;
    }
    *lblock_to_refine = trial;
    status
}

/// Shared method 1 / method 2 subset encoding of one (carrier config, partition) pair
/// (the common body of `compress_block_2subsets_internal` / `compress_block_3subsets_internal`).
fn encode_subset_pattern(
    ctx: &SubsetEncContext,
    state: &ShortlistState,
    cfg: &AstcUnpackedConfig,
    carrier: &LogAstcBlock,
    grid_data: &AstcBlockGridData,
    num_subsets: usize,
    seed_id: u32,
    subset_pbuf: &[PixelBuf],
    subset_pixel_indices: &SubsetIndices,
    best_lblock_error: &mut f64,
    best_lblock: &mut LogAstcBlock,
    all_candidates: &mut Option<&mut Vec<LogAstcBlock>>,
) {
    let ss = &ctx.ss;
    let (bw, bh) = (ss.block_width, ss.block_height);
    let total_block_pixels = (bw * bh) as usize;
    let cem = u32::from(cfg.cem);
    let num_cem_comps = get_num_cem_chans(cem);
    let wrange = u32::from(cfg.weight_range);
    let erange = u32::from(cfg.endpoint_range);

    let mut fl = LogAstcBlock {
        user_mode: USER_MODE_RANK_VALUES,
        grid_width: carrier.grid_width,
        grid_height: carrier.grid_height,
        endpoint_ise_range: cfg.endpoint_range,
        weight_ise_range: cfg.weight_range,
        num_partitions: num_subsets as u8,
        partition_id: seed_id as u16,
        ..Default::default()
    };
    for s in 0..num_subsets {
        fl.color_endpoint_modes[s] = cfg.cem;
    }

    let n_cem = ah::get_num_cem_values(cem) as usize;
    let total_grid = usize::from(carrier.grid_width) * usize::from(carrier.grid_height);
    let pbuf = &state.pbuf;

    if ctx.use_method1 {
        fl.weights[..total_grid].copy_from_slice(&carrier.weights[..total_grid]);
        for s in 0..num_subsets {
            fl.endpoints[s * n_cem..(s + 1) * n_cem].copy_from_slice(&carrier.endpoints[..n_cem]);
        }
        const NUM_M1_PASSES: u32 = 2;
        for pass in 0..NUM_M1_PASSES {
            refine_endpoints_with_current_weights(
                pbuf,
                &mut fl,
                subset_pbuf,
                subset_pixel_indices,
                cfg,
                ss,
                best_lblock_error,
                best_lblock,
            );
            let changed = if pass == NUM_M1_PASSES - 1 {
                weight_polish(
                    pbuf,
                    &mut fl,
                    ss,
                    grid_data,
                    best_lblock_error,
                    best_lblock,
                    None,
                )
            } else {
                weight_gradient_descent(
                    pbuf,
                    &mut fl,
                    subset_pbuf,
                    subset_pixel_indices,
                    cfg,
                    ss,
                    grid_data,
                    best_lblock_error,
                    best_lblock,
                )
            };
            if !changed {
                break;
            }
        }
        if let Some(all) = all_candidates.as_deref_mut() {
            all.push(fl);
        }
    }

    if ctx.use_method2 {
        let mut best_pat_err = f64::MAX;
        let mut best_pat = LogAstcBlock::default();
        let mut desired = [0u8; MAX_BLOCK_PIXELS];
        for s in 0..num_subsets {
            let initial = calc_initial_cem_endpoints(
                &subset_pbuf[s],
                num_cem_comps,
                None,
                is_cem_6_or_10(cem),
            );
            let vals = &mut fl.endpoints[s * n_cem..];
            cem_encode(cem, &initial, erange, vals, true, ss.higher_effort_bc);
            let mut sw = [0u8; MAX_BLOCK_PIXELS];
            eval_weights_first_plane(
                &subset_pbuf[s],
                &mut sw,
                wrange,
                cem,
                vals,
                erange,
                num_cem_comps,
                -1,
            );
            for _ in 0..ss.num_ls_iterations {
                let mut refined = [0.0f32; 8];
                refine_endpoints_given_weights(
                    &subset_pbuf[s],
                    cem,
                    &sw,
                    wrange,
                    &mut refined,
                    num_cem_comps,
                );
                let vals = &mut fl.endpoints[s * n_cem..];
                cem_encode(cem, &refined, erange, vals, true, ss.higher_effort_bc);
                eval_weights_first_plane(
                    &subset_pbuf[s],
                    &mut sw,
                    wrange,
                    cem,
                    vals,
                    erange,
                    num_cem_comps,
                    -1,
                );
            }
            for j in 0..subset_pbuf[s].width as usize {
                let pi = subset_pixel_indices[s][j] as usize;
                if pi < total_block_pixels {
                    desired[pi] = sw[j];
                }
            }
        }

        if bw == u32::from(cfg.grid_width) && bh == u32::from(cfg.grid_height) {
            fl.weights[..total_block_pixels].copy_from_slice(&desired[..total_block_pixels]);
        } else {
            for v in desired.iter_mut().take(total_block_pixels) {
                *v = w_rank_to_val(wrange, u32::from(*v)) as u8;
            }
            let mut src = PixelBuf::new(bw, bh);
            for y in 0..bh {
                for x in 0..bw {
                    src.set(x, y, 0, f32::from(desired[(x + y * bw) as usize]));
                }
            }
            let (gw, gh) = (u32::from(cfg.grid_width), u32::from(cfg.grid_height));
            let mut dst = PixelBuf::new(gw, gh);
            pseudoinverse_block_to_grid(&src, &mut dst, 1);
            for y in 0..gh {
                for x in 0..gw {
                    let d = f2i(dst.get(x, y, 0).round()).clamp(0, 64);
                    fl.weights[(x + y * gw) as usize] = w_val_to_rank(wrange, d as u32) as u8;
                }
            }
        }

        let err = compute_block_error(&fl, pbuf, ss);
        if err < best_pat_err {
            best_pat_err = err;
            best_pat = fl;
        }

        const NUM_M2_PASSES: u32 = 2;
        for pass in 0..NUM_M2_PASSES {
            refine_endpoints_with_current_weights(
                pbuf,
                &mut fl,
                subset_pbuf,
                subset_pixel_indices,
                cfg,
                ss,
                &mut best_pat_err,
                &mut best_pat,
            );
            let changed = if pass == NUM_M2_PASSES - 1 {
                weight_polish(
                    pbuf,
                    &mut fl,
                    ss,
                    grid_data,
                    &mut best_pat_err,
                    &mut best_pat,
                    None,
                )
            } else {
                weight_gradient_descent(
                    pbuf,
                    &mut fl,
                    subset_pbuf,
                    subset_pixel_indices,
                    cfg,
                    ss,
                    grid_data,
                    &mut best_pat_err,
                    &mut best_pat,
                )
            };
            if !changed {
                break;
            }
        }

        if best_pat_err < *best_lblock_error {
            *best_lblock_error = best_pat_err;
            *best_lblock = best_pat;
        }
        if let Some(all) = all_candidates.as_deref_mut() {
            all.push(best_pat);
        }
    }
}

const NUM_DOT_THRESH_FRACS: [u8; 6] = [1, 3, 5, 9, 11, 15];
const DOT_THRESH_FRACS15: [f32; 15] = [
    -0.55, -0.45, -0.35, -0.25, -0.15, -0.10, -0.075, 0.0, 0.075, 0.10, 0.15, 0.25, 0.35, 0.45,
    0.55,
];
const DOT_THRESH_FRACS11: [f32; 11] = [
    -0.45, -0.35, -0.25, -0.15, -0.075, 0.0, 0.075, 0.15, 0.25, 0.35, 0.45,
];
const DOT_THRESH_FRACS9: [f32; 9] = [-0.35, -0.25, -0.15, -0.075, 0.0, 0.075, 0.15, 0.25, 0.35];
const DOT_THRESH_FRACS5: [f32; 5] = [-0.2, -0.1, 0.0, 0.1, 0.2];
const DOT_THRESH_FRACS3: [f32; 3] = [-0.1, 0.0, 0.1];
const DOT_THRESH_FRACS1: [f32; 1] = [0.0];

fn dot_thresh_fracts(index: usize) -> &'static [f32] {
    match index {
        0 => &DOT_THRESH_FRACS1,
        1 => &DOT_THRESH_FRACS3,
        2 => &DOT_THRESH_FRACS5,
        3 => &DOT_THRESH_FRACS9,
        4 => &DOT_THRESH_FRACS11,
        _ => &DOT_THRESH_FRACS15,
    }
}

/// Encodes the shortlisted carrier configuration `cfg` as a single subset block (rank space).
fn encode_carrier(
    ss: &SingleSubsetEncContext,
    state: &ShortlistState,
    cfg: &AstcUnpackedConfig,
) -> LogAstcBlock {
    let mut carrier = LogAstcBlock::default();
    encode_single_subset_block(
        ss,
        &state.pbuf,
        state.num_src_block_comps,
        cfg,
        &mut carrier,
        false,
    );
    convert_ise_lblock_to_rank(&mut carrier);
    carrier
}

/// `compress_block_2subsets_internal`.
fn compress_block_2subsets_internal(
    ctx: &SubsetEncContext,
    pixels: &[[u8; 4]],
    best_lblock_error: &mut f64,
    best_lblock: &mut LogAstcBlock,
    mut all_candidates: Option<&mut Vec<LogAstcBlock>>,
    state: &mut ShortlistState,
) {
    let ss = &ctx.ss;
    let (bw, bh) = (ss.block_width, ss.block_height);
    let total_block_pixels = ss.total_block_pixels;
    const SUBSET_SCALE_WEIGHT: f32 = 2.0;

    let total_candidates = generate_single_subset_shortlist(
        &TWO_SUBSET_CONFIGS_RGBA,
        ss,
        state.num_src_block_comps,
        state,
        SUBSET_SCALE_WEIGHT,
        ctx.num_carrier_candidates,
    );
    if total_candidates == 0 {
        return;
    }

    let dt_index = (NUM_DOT_THRESH_FRACTS - 1).min(ctx.two_subset_dot_thresh_fract_index) as usize;
    let num_dot_thresh_fracts = NUM_DOT_THRESH_FRACS[dt_index] as usize;
    let fracts = dot_thresh_fracts(dt_index);

    let stats = state.stats;
    let dot_of = |p: &[u8; 4]| {
        let r = f32::from(p[0]) - stats.mean[0];
        let g = f32::from(p[1]) - stats.mean[1];
        let b = f32::from(p[2]) - stats.mean[2];
        let a = f32::from(p[3]) - stats.mean[3];
        (r * stats.axis[0]) + (g * stats.axis[1]) + (b * stats.axis[2]) + (a * stats.axis[3])
    };

    let mut dot_range = 0.0f32;
    if num_dot_thresh_fracts > 1 {
        let (mut min_dot, mut max_dot) = (f32::MAX, -f32::MAX);
        for p in &pixels[..total_block_pixels as usize] {
            let d = dot_of(p);
            min_dot = fmin(min_dot, d);
            max_dot = fmax(max_dot, d);
        }
        dot_range = max_dot - min_dot;
    }

    let n_unique = (ctx.num_unique_two_subset_pats as usize)
        .min(ctx.two_subset_pat_bitmask.len())
        .min(MAX_UNIQUE_2SUBSET_PATS);
    let mut best_diffs = vec![0u32; MAX_UNIQUE_2SUBSET_PATS];

    for cci in 0..total_candidates as usize {
        let cfg = state.best_configs[cci];
        let cem_has_a = does_cem_have_alpha(u32::from(cfg.cem));
        let Some(gd) = find_astc_block_grid_data(
            bw,
            bh,
            u32::from(cfg.grid_width),
            u32::from(cfg.grid_height),
        ) else {
            continue;
        };
        let carrier = encode_carrier(ss, state, &cfg);

        for &frac in fracts.iter().take(num_dot_thresh_fracts) {
            let mut desired = Bitmask192::zero();
            let thresh = frac * dot_range;
            for (bit, p) in pixels[..total_block_pixels as usize].iter().enumerate() {
                let s = u32::from(dot_of(p) > thresh);
                desired.set_bit(bit as u32, s);
            }
            let active = Bitmask192::lsb_mask(total_block_pixels);
            let desired_inv = desired.not().and(&active);

            if ctx.num_pattern_candidates == 1 {
                let (mut best_diff, mut best_diff_inv) = (u32::MAX, u32::MAX);
                for i in 0..n_unique {
                    let pat = &ctx.two_subset_pat_bitmask[i];
                    let diff = (pat.xor(&desired).popcount() << 16) | i as u32;
                    let diff_inv = (pat.xor(&desired_inv).popcount() << 16) | i as u32;
                    best_diff = diff.min(best_diff);
                    best_diff_inv = diff_inv.min(best_diff_inv);
                }
                best_diffs[0] = best_diff.min(best_diff_inv);
            } else {
                for i in 0..n_unique {
                    let pat = &ctx.two_subset_pat_bitmask[i];
                    let diff = pat.xor(&desired).popcount();
                    let diff_inv = pat.xor(&desired_inv).popcount();
                    best_diffs[i] = (diff.min(diff_inv) << 16) + i as u32;
                }
                // Unique keys: equivalent to std::sort.
                best_diffs[..n_unique].sort_unstable();
            }

            for pci in 0..ctx.num_pattern_candidates as usize {
                let Some(&bd) = best_diffs.get(pci) else {
                    break;
                };
                let upi = (bd & 1023) as usize;
                let Some(pat) = ctx.two_subset_pat_bitmask.get(upi) else {
                    continue;
                };
                let seed_id = u32::from(ctx.unique_two_subset_pats.get(upi).copied().unwrap_or(0));

                let mut sp = [PixelBuf::new(0, 1), PixelBuf::new(0, 1)];
                let mut spi: SubsetIndices = [[0; 256]; 3];
                let mut bit = 0;
                for y in 0..bh {
                    for x in 0..bw {
                        let s = usize::from(pat.is_bit_set(bit));
                        let p = pixels[bit as usize];
                        bit += 1;
                        let a = if cem_has_a { f32::from(p[3]) } else { 255.0 };
                        let cw = sp[s].width;
                        spi[s][cw as usize] = x + y * bw;
                        sp[s].set(cw, 0, 0, f32::from(p[0]));
                        sp[s].set(cw, 0, 1, f32::from(p[1]));
                        sp[s].set(cw, 0, 2, f32::from(p[2]));
                        sp[s].set(cw, 0, 3, a);
                        sp[s].width = cw + 1;
                    }
                }

                encode_subset_pattern(
                    ctx,
                    state,
                    &cfg,
                    &carrier,
                    gd,
                    2,
                    seed_id,
                    &sp,
                    &spi,
                    best_lblock_error,
                    best_lblock,
                    &mut all_candidates,
                );
            }
        }
    }

    convert_rank_lblock_to_ise(best_lblock);
}

/// `create_desired_partitions_3subsets`.
fn create_desired_partitions_3subsets(
    stats: &BlockStats,
    total_block_pixels: usize,
    pixels: &[[u8; 4]],
    desired: &mut [u8],
) -> bool {
    desired[..total_block_pixels].fill(0);
    let mut cc = [[0.0f32; 4]; 3];
    let (mut brightest, mut darkest) = (-BIG_FLOAT_VAL, BIG_FLOAT_VAL);
    for p in &pixels[..total_block_pixels] {
        let v = p.map(f32::from);
        let inten = ((v[0] - stats.mean[0]) * stats.axis[0])
            + ((v[1] - stats.mean[1]) * stats.axis[1])
            + ((v[2] - stats.mean[2]) * stats.axis[2])
            + ((v[3] - stats.mean[3]) * stats.axis[3]);
        if inten < darkest {
            darkest = inten;
            cc[0] = v;
        }
        if inten > brightest {
            brightest = inten;
            cc[1] = v;
        }
    }
    let mut furthest = 0.0f32;
    cc[2] = cc[0];
    for p in &pixels[..total_block_pixels] {
        let v = p.map(f32::from);
        let da = vec4_squared_dist(&v, &cc[0]);
        if da == 0.0 {
            continue;
        }
        let db = vec4_squared_dist(&v, &cc[1]);
        if db == 0.0 {
            continue;
        }
        let d2 = da + db;
        if d2 > furthest {
            furthest = d2;
            cc[2] = v;
        }
    }
    let same = |a: &[f32; 4], b: &[f32; 4]| a.map(f32::to_bits) == b.map(f32::to_bits);
    if same(&cc[0], &cc[1]) || same(&cc[0], &cc[2]) || same(&cc[1], &cc[2]) {
        return false;
    }
    const NUM_ITERS: u32 = 4;
    for s in 0..NUM_ITERS {
        let mut n = [0u32; 3];
        let mut means = [[0.0f32; 4]; 3];
        let mut changed = false;
        for (i, p) in pixels[..total_block_pixels].iter().enumerate() {
            let v = p.map(f32::from);
            let d0 = vec4_squared_dist(&v, &cc[0]);
            let d1 = vec4_squared_dist(&v, &cc[1]);
            let d2 = vec4_squared_dist(&v, &cc[2]);
            let mut idx = 0;
            let mut md = d0;
            if d1 < md {
                md = d1;
                idx = 1;
            }
            if d2 < md {
                idx = 2;
            }
            if idx != desired[i] as usize {
                changed = true;
            }
            desired[i] = idx as u8;
            for c in 0..4 {
                means[idx][c] += v[c];
            }
            n[idx] += 1;
        }
        if n[0] == 0 || n[1] == 0 || n[2] == 0 {
            return false;
        }
        if !changed {
            break;
        }
        if s < NUM_ITERS - 1 {
            for j in 0..3 {
                let k = 1.0f32 / n[j] as f32;
                for c in 0..4 {
                    cc[j][c] = means[j][c] * k;
                }
            }
        }
    }
    true
}

/// `compress_block_3subsets_internal`.
fn compress_block_3subsets_internal(
    ctx: &SubsetEncContext,
    thread_ctx: &mut SubsetEncThreadContext,
    pixels: &[[u8; 4]],
    best_lblock_error: &mut f64,
    best_lblock: &mut LogAstcBlock,
    mut all_candidates: Option<&mut Vec<LogAstcBlock>>,
    state: &mut ShortlistState,
) {
    let ss = &ctx.ss;
    let (bw, bh) = (ss.block_width, ss.block_height);
    let total_block_pixels = ss.total_block_pixels as usize;
    const SUBSET_SCALE_WEIGHT: f32 = 2.0;

    let total_candidates = generate_single_subset_shortlist(
        &THREE_SUBSET_CONFIGS_RGBA,
        ss,
        state.num_src_block_comps,
        state,
        SUBSET_SCALE_WEIGHT,
        ctx.num_carrier_candidates,
    );
    if total_candidates == 0 {
        return;
    }

    thread_ctx.pat_vec.init(bw, bh, None);
    if !create_desired_partitions_3subsets(
        &state.stats,
        total_block_pixels,
        pixels,
        &mut thread_ctx.pat_vec.parts,
    ) {
        return;
    }

    let Some(pd) = ctx.part_data_p3.as_deref() else {
        return;
    };
    let mut cand_patterns = [0u32; MAX_UNIQUE_3SUBSET_PATS];
    let num_pat_candidates = pd.part_lhs_map.find(
        &thread_ctx.pat_vec,
        &mut cand_patterns,
        ctx.num_pattern_candidates,
        false,
    );
    if num_pat_candidates == 0 {
        return;
    }

    for cci in 0..total_candidates as usize {
        let cfg = state.best_configs[cci];
        let cem_has_a = does_cem_have_alpha(u32::from(cfg.cem));
        let Some(gd) = find_astc_block_grid_data(
            bw,
            bh,
            u32::from(cfg.grid_width),
            u32::from(cfg.grid_height),
        ) else {
            continue;
        };
        let carrier = encode_carrier(ss, state, &cfg);

        for &upi in cand_patterns.iter().take(num_pat_candidates as usize) {
            let Some(pat) = pd.partition_pats.get(upi as usize) else {
                continue;
            };
            let seed_id = pd.unique_index_to_part_seed[upi as usize & 1023].max(0) as u32;

            let mut sp = [
                PixelBuf::new(0, 1),
                PixelBuf::new(0, 1),
                PixelBuf::new(0, 1),
            ];
            let mut spi: SubsetIndices = [[0; 256]; 3];
            let mut i = 0;
            for y in 0..bh {
                for x in 0..bw {
                    let s = usize::from(pat.at(x, y)).min(2);
                    let p = pixels[i];
                    i += 1;
                    let a = if cem_has_a { f32::from(p[3]) } else { 255.0 };
                    let cw = sp[s].width;
                    spi[s][cw as usize] = x + y * bw;
                    sp[s].set(cw, 0, 0, f32::from(p[0]));
                    sp[s].set(cw, 0, 1, f32::from(p[1]));
                    sp[s].set(cw, 0, 2, f32::from(p[2]));
                    sp[s].set(cw, 0, 3, a);
                    sp[s].width = cw + 1;
                }
            }

            encode_subset_pattern(
                ctx,
                state,
                &cfg,
                &carrier,
                gd,
                3,
                seed_id,
                &sp,
                &spi,
                best_lblock_error,
                best_lblock,
                &mut all_candidates,
            );
        }
    }

    convert_rank_lblock_to_ise(best_lblock);
}

/// `astc_ldrf::compress_block_subsets`: 1-3 subsets. `best_lblock` is in ISE space,
/// candidates in rank space.
pub(crate) fn compress_block_subsets(
    ctx: &SubsetEncContext,
    thread_ctx: &mut SubsetEncThreadContext,
    pixels: &[[u8; 4]],
    best_lblock: &mut LogAstcBlock,
    mut all_candidates: Option<&mut Vec<LogAstcBlock>>,
) -> f64 {
    let mut state = ShortlistState::new();
    let mut best_err = compress_single_subset_internal(
        &ctx.ss,
        pixels,
        best_lblock,
        all_candidates.as_deref_mut(),
        &mut state,
        true,
        DEF_SCALE_WEIGHT,
    );
    if best_lblock.solid_color_flag_ldr || ctx.max_subsets == 1 {
        return best_err;
    }
    let c = &state.stats.covar;
    let max_chan_var = fmax(fmax(fmax(c[COV_RR], c[COV_GG]), c[COV_BB]), c[COV_AA])
        / ctx.ss.total_block_pixels as f32;

    if max_chan_var >= ctx.two_subset_var_thresh {
        compress_block_2subsets_internal(
            ctx,
            pixels,
            &mut best_err,
            best_lblock,
            all_candidates.as_deref_mut(),
            &mut state,
        );
    }
    if ctx.max_subsets == 3 && max_chan_var >= ctx.three_subset_var_thresh {
        compress_block_3subsets_internal(
            ctx,
            thread_ctx,
            pixels,
            &mut best_err,
            best_lblock,
            all_candidates,
            &mut state,
        );
    }
    best_err
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitmask192_ops() {
        let m = Bitmask192::lsb_mask(100);
        assert_eq!(m.popcount(), 100);
        let mut b = Bitmask192::zero();
        b.set_bit(130, 1);
        assert!(b.is_bit_set(130));
        b.set_bit(130, 0);
        assert!(b.is_zero());
        assert!(Bitmask192::new(0, 0, 1).less(&Bitmask192::new(0, 1, 1)));
    }
}
