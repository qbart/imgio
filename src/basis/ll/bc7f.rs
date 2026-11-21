// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of basisu `bc7f` (`basisu_transcoder.cpp`): the real-time analytical BC7 encoder used
//! when transcoding ASTC LDR / XUASTC LDR to BC7.
//!
//! Only the scalar code path is ported (`BASISU_BC7F_USE_SSE41` is 0 upstream). Float math
//! follows the C++ operation order in `f32`; float to int casts emulate x86 `cvttss2si`.
//! The direct ASTC -> BC7 packers (`pack_from_astc_*`) live in `bc7f_xuastc.rs`; perf stats
//! are not ported.

use super::uastc::uastc_tables;
use super::uastc_tables::{
    ASTC_BC7_COMMON_PARTITIONS2, ASTC_BC7_COMMON_PARTITIONS3, BC7_PARTITION2, BC7_PARTITION3,
    BC7_TABLE_ANCHOR_INDEX_SECOND_SUBSET, BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_1,
    BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_2, BC7_WEIGHTS2, BC7_WEIGHTS3, BC7_WEIGHTS4,
};
use std::sync::OnceLock;

type Px = [u8; 4];
type Block = [u8; 16];

// Encoder configuration flags.
pub const PACK_BC7_FLAG_USE_2_SUBSETS_RGB: u32 = 1;
pub const PACK_BC7_FLAG_USE_2_SUBSETS_RGBA: u32 = 2;
pub const PACK_BC7_FLAG_USE_3_SUBSETS_RGB: u32 = 4;
pub const PACK_BC7_FLAG_USE_DUAL_PLANE_RGB: u32 = 8;
pub const PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA: u32 = 16;
pub const PACK_BC7_FLAG_PBIT_OPT: u32 = 32;
pub const PACK_BC7_FLAG_PBIT_OPT_MODE6: u32 = 64;
pub const PACK_BC7_FLAG_USE_TRIVIAL_MODE6: u32 = 128;
pub const PACK_BC7_FLAG_PARTIALLY_ANALYTICAL_RGB: u32 = 256;
pub const PACK_BC7_FLAG_PARTIALLY_ANALYTICAL_RGBA: u32 = 512;
pub const PACK_BC7_FLAG_NON_ANALYTICAL_RGB: u32 = 1024;
pub const PACK_BC7_FLAG_NON_ANALYTICAL_RGBA: u32 = 2048;
pub const PACK_BC7_FLAG_ASTC_COMPATIBLE: u32 = 4096;
pub const PACK_BC7_FLAG_DISABLE_RGB_DUAL_PLANE: u32 = 8192;

pub const PACK_BC7_FLAG_DEFAULT_FASTEST: u32 = PACK_BC7_FLAG_USE_TRIVIAL_MODE6;
pub const PACK_BC7_FLAG_DEFAULT_FASTER: u32 =
    PACK_BC7_FLAG_PBIT_OPT | PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA | PACK_BC7_FLAG_USE_TRIVIAL_MODE6;
pub const PACK_BC7_FLAG_DEFAULT_FAST: u32 = PACK_BC7_FLAG_USE_2_SUBSETS_RGB
    | PACK_BC7_FLAG_USE_2_SUBSETS_RGBA
    | PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA
    | PACK_BC7_FLAG_PBIT_OPT
    | PACK_BC7_FLAG_USE_TRIVIAL_MODE6;
/// `cPackBC7FlagDefault`.
pub const PACK_BC7_FLAG_DEFAULT: u32 = PACK_BC7_FLAG_USE_2_SUBSETS_RGB
    | PACK_BC7_FLAG_USE_2_SUBSETS_RGBA
    | PACK_BC7_FLAG_USE_3_SUBSETS_RGB
    | PACK_BC7_FLAG_USE_DUAL_PLANE_RGB
    | PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA
    | PACK_BC7_FLAG_PBIT_OPT
    | PACK_BC7_FLAG_PBIT_OPT_MODE6
    | PACK_BC7_FLAG_USE_TRIVIAL_MODE6;
/// `cPackBC7FlagDefaultPartiallyAnalytical` (used by the transcoder in high quality mode).
pub const PACK_BC7_FLAG_DEFAULT_PARTIALLY_ANALYTICAL: u32 = PACK_BC7_FLAG_DEFAULT
    | PACK_BC7_FLAG_PARTIALLY_ANALYTICAL_RGB
    | PACK_BC7_FLAG_PARTIALLY_ANALYTICAL_RGBA;
pub const PACK_BC7_FLAG_DEFAULT_NON_ANALYTICAL: u32 = (PACK_BC7_FLAG_DEFAULT_PARTIALLY_ANALYTICAL
    | PACK_BC7_FLAG_NON_ANALYTICAL_RGB
    | PACK_BC7_FLAG_NON_ANALYTICAL_RGBA)
    & !PACK_BC7_FLAG_USE_TRIVIAL_MODE6;

const MAX_PATTERNS2_TO_CHECK: usize = 64;
const MAX_PATTERNS3_TO_CHECK: usize = 64;
const UNIQUE_PBIT_DISCOUNT: f32 = 0.85;
const SHARED_PBIT_DISCOUNT: f32 = 0.95;
const SMALL_FLOAT_VAL: f32 = 0.0000125;

const TRIVIAL_BLOCK_THRESH_RGB: i32 = 20 * 16;
const TRIVIAL_BLOCK_THRESH_RGBA: i32 = 2 * 16;
const DP_BLOCK_VAR_THRESH: i32 = 2 * 16;
const STRONG_CORR_THRESH: f32 = 0.85;
const MIN_BLOCK_MAX_VAR_23SUBSETS: i32 = 100 * 16;
const HIGH_ORTHO_ENERGY_THRESH: f32 = 1.0 * 16.0;
const ORTHO_RATIO_23SUBSET_RATIO_THRESH: f32 = 0.004;
const DP_BLOCK_VAR_THRESH_RGBA: i32 = 16;
const ALPHA_DECORR_THRESHOLD: f32 = 0.995;
const STRONG_DECORR_THRESH_RGBA: f32 = 0.85;
const MIN_BLOCK_MAX_VAR_3SUBSETS: i32 = 500 * 16;
const MIN_BLOCK_MAX_VAR_23SUBSETS_RGBA: i32 = 100 * 16;
const HIGH_ORTHO_ENERGY_THRESH_RGBA: f32 = 1.0 * 16.0;
const ORTHO_RATIO_23SUBSET_RATIO_THRESH_RGBA: f32 = 0.004;

/// Tables built by `bc7f::init`.
struct Tables {
    /// `g_bc7_{2,3,4}bit_ls_tab`: (w*w, (1-w)*w, (1-w)*(1-w), w).
    ls2: [[f32; 4]; 4],
    ls3: [[f32; 4]; 8],
    ls4: [[f32; 4]; 16],
    part2_bitmasks: [u16; 64],
    /// Low 16 bits: subset 0 pixels, high 16 bits: subset 1 pixels.
    part3_bitmasks: [u32; 64],
}

fn ls_tab<const N: usize>(weights: &[u32; N]) -> [[f32; 4]; N] {
    let mut t = [[0.0f32; 4]; N];
    for (o, &wi) in t.iter_mut().zip(weights.iter()) {
        let w = wi as f32 * (1.0f32 / 64.0f32);
        *o = [w * w, (1.0 - w) * w, (1.0 - w) * (1.0 - w), w];
    }
    t
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut part2_bitmasks = [0u16; 64];
        for (i, m) in part2_bitmasks.iter_mut().enumerate() {
            for x in 0..16 {
                *m |= (BC7_PARTITION2[i * 16 + x] as u16) << x;
            }
        }
        let mut part3_bitmasks = [0u32; 64];
        for (i, m) in part3_bitmasks.iter_mut().enumerate() {
            for j in 0..16 {
                match BC7_PARTITION3[i * 16 + j] {
                    0 => *m |= 1 << j,
                    1 => *m |= 0x10000 << j,
                    _ => {}
                }
            }
        }
        Tables {
            ls2: ls_tab(&BC7_WEIGHTS2),
            ls3: ls_tab(&BC7_WEIGHTS3),
            ls4: ls_tab(&BC7_WEIGHTS4),
            part2_bitmasks,
            part3_bitmasks,
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Scalar helpers (C++ semantics).

/// `(int)x` as compiled for x86-64 (`cvttss2si`): out of range / NaN gives `INT_MIN`.
#[inline]
pub(super) fn f2i(x: f32) -> i32 {
    if x >= -2147483648.0 && x < 2147483648.0 {
        x as i32
    } else {
        i32::MIN
    }
}

#[inline]
fn fast_roundf_pos_int(x: f32) -> i32 {
    f2i(x + 0.5)
}

#[inline]
fn fast_roundf_int(x: f32) -> i32 {
    if x >= 0.0 { f2i(x + 0.5) } else { f2i(x - 0.5) }
}

/// `basisu::clamp` (`(v < lo) ? lo : ((v > hi) ? hi : v)`, NaN passes through).
#[inline]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `basisu::maximum` for floats (`(a > b) ? a : b`).
#[inline]
fn maxf(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[inline]
fn minf(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

#[inline]
fn sq(v: i32) -> i32 {
    v * v
}

#[inline]
fn from_7(v: i32) -> i32 {
    (v << 1) | (v >> 6)
}

#[inline]
fn from_7p(v: i32, p: i32) -> i32 {
    (v << 1) | p
}

#[inline]
fn to_7p(c8: i32, pbit: i32) -> i32 {
    let e = (c8 as u32).wrapping_add((pbit ^ 1) as u32) >> 1;
    e.min(127) as i32
}

#[inline]
fn to_7(c8: i32) -> i32 {
    (c8 * 127 + 127) / 255
}

#[inline]
fn to_7f(c: f32, pbit: i32) -> i32 {
    to_7p(fast_roundf_pos_int(c), pbit)
}

#[inline]
fn to_7_clamp(c: f32, pbit: i32) -> i32 {
    to_7p(clampi(fast_roundf_int(c), 0, 255), pbit)
}

#[inline]
fn to_5(c8: i32) -> i32 {
    (c8 * 31 + 127) / 255
}

#[inline]
fn to_5_clamp(c: f32) -> i32 {
    clampi(fast_roundf_int(c * (31.0f32 / 255.0f32)), 0, 31)
}

#[inline]
fn to_6(c8: i32) -> i32 {
    (c8 * 63 + 127) / 255
}

/// Shared helper of `to_6(c8, pbit)` / `to_5(c8, pbit)`: quantize to `scale` levels, then step
/// to the nearest level with the requested p-bit.
#[inline]
fn to_np(c8: i32, pbit: i32, scale: i32) -> i32 {
    let mut q = (c8 * scale + 127) / 255;
    if (q & 1) != pbit {
        if c8 * scale >= 255 * q {
            q = if q < scale { q + 1 } else { q - 1 };
        } else {
            q = if q > 0 { q - 1 } else { q + 1 };
        }
    }
    q >> 1
}

#[inline]
fn to_6p(c8: i32, pbit: i32) -> i32 {
    to_np(c8, pbit, 127)
}

#[inline]
fn to_6_clamp(c: f32, pbit: i32) -> i32 {
    to_6p(clampi(fast_roundf_int(c), 0, 255), pbit)
}

#[inline]
fn to_5p(c8: i32, pbit: i32) -> i32 {
    to_np(c8, pbit, 63)
}

#[inline]
fn to_5_clamp_p(c: f32, pbit: i32) -> i32 {
    to_5p(clampi(fast_roundf_int(c), 0, 255), pbit)
}

#[inline]
fn from_6p(v: i32, p: i32) -> i32 {
    let v = (v << 1) | p;
    (v << 1) | (v >> 6)
}

#[inline]
fn from_4p(v: i32, p: i32) -> i32 {
    let v = (v << 1) | p;
    (v << 3) | (v >> 2)
}

#[inline]
fn from_5(v: i32) -> i32 {
    (v << 3) | (v >> 2)
}

#[inline]
fn from_5p(v: i32, p: i32) -> i32 {
    let v = (v << 1) | p;
    (v << 2) | (v >> 4)
}

#[inline]
fn from_6(v: i32) -> i32 {
    (v << 2) | (v >> 4)
}

#[inline]
fn put_le(dst: &mut [u8], v: u64) {
    for (i, d) in dst.iter_mut().enumerate() {
        *d = (v >> (8 * i)) as u8;
    }
}

// ---------------------------------------------------------------------------------------------
// Physical block encoders.

/// `encode_mode0_rgb_block`: 3 subsets, 4-bit endpoints, 6 unique p-bits, 3-bit weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mode0_rgb_block(
    block: &mut Block,
    part_id: u32,
    mut lr: [u32; 3],
    mut lg: [u32; 3],
    mut lb: [u32; 3],
    mut hr: [u32; 3],
    mut hg: [u32; 3],
    mut hb: [u32; 3],
    mut p: [u32; 6],
    w: &[u8; 16],
) {
    let pid = part_id as usize;
    let part = &BC7_PARTITION3[pid * 16..pid * 16 + 16];
    let a0 = BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_1[pid] as usize;
    let a1 = BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_2[pid] as usize;
    let mut inv = [0u32; 3];
    for (s, anchor) in [0, a0, a1].into_iter().enumerate() {
        if w[anchor] & 4 != 0 {
            std::mem::swap(&mut lr[s], &mut hr[s]);
            std::mem::swap(&mut lg[s], &mut hg[s]);
            std::mem::swap(&mut lb[s], &mut hb[s]);
            p.swap(s * 2, s * 2 + 1);
            inv[s] = 7;
        }
    }
    let low: u64 = 1
        | ((part_id as u64) << 1)
        | ((lr[0] as u64) << 5)
        | ((hr[0] as u64) << 9)
        | ((lr[1] as u64) << 13)
        | ((hr[1] as u64) << 17)
        | ((lr[2] as u64) << 21)
        | ((hr[2] as u64) << 25)
        | ((lg[0] as u64) << 29)
        | ((hg[0] as u64) << 33)
        | ((lg[1] as u64) << 37)
        | ((hg[1] as u64) << 41)
        | ((lg[2] as u64) << 45)
        | ((hg[2] as u64) << 49)
        | ((lb[0] as u64) << 53)
        | ((hb[0] as u64) << 57)
        | ((lb[1] as u64) << 61);
    put_le(&mut block[0..8], low);
    let mut high: u64 = ((lb[1] >> 3) as u64)
        | ((hb[1] as u64) << 1)
        | ((lb[2] as u64) << 5)
        | ((hb[2] as u64) << 9)
        | ((p[0] as u64) << 13)
        | ((p[1] as u64) << 14)
        | ((p[2] as u64) << 15)
        | ((p[3] as u64) << 16)
        | ((p[4] as u64) << 17)
        | ((p[5] as u64) << 18);
    let mut ofs = 19u32;
    for i in 0..16 {
        let wv = (w[i] as u32 ^ inv[part[i] as usize]) as u64;
        high |= wv << ofs;
        ofs += 3 - (i == 0 || i == a0 || i == a1) as u32;
    }
    put_le(&mut block[8..16], high);
}

/// `encode_mode1_rgb_block`: 2 subsets, 6-bit endpoints, 2 shared p-bits, 3-bit weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mode1_rgb_block(
    block: &mut Block,
    part_id: u32,
    mut lr: [u32; 2],
    mut lg: [u32; 2],
    mut lb: [u32; 2],
    mut hr: [u32; 2],
    mut hg: [u32; 2],
    mut hb: [u32; 2],
    p0: u32,
    p1: u32,
    w: &[u8; 16],
) {
    let pid = part_id as usize;
    let part = &BC7_PARTITION2[pid * 16..pid * 16 + 16];
    let anchor = BC7_TABLE_ANCHOR_INDEX_SECOND_SUBSET[pid] as usize;
    let mut inv = [0u32; 2];
    for (s, a) in [0, anchor].into_iter().enumerate() {
        if w[a] & 4 != 0 {
            std::mem::swap(&mut lr[s], &mut hr[s]);
            std::mem::swap(&mut lg[s], &mut hg[s]);
            std::mem::swap(&mut lb[s], &mut hb[s]);
            inv[s] = 7;
        }
    }
    block[0] = (0b10 | (part_id << 2)) as u8;
    let x: u64 = (lr[0] as u64)
        | ((hr[0] as u64) << 6)
        | ((lr[1] as u64) << 12)
        | ((hr[1] as u64) << 18)
        | ((lg[0] as u64) << 24)
        | ((hg[0] as u64) << 30)
        | ((lg[1] as u64) << 36)
        | ((hg[1] as u64) << 42)
        | ((lb[0] as u64) << 48)
        | ((hb[0] as u64) << 54)
        | ((lb[1] as u64) << 60);
    put_le(&mut block[1..9], x);
    block[9] = ((lb[1] >> 4) | (hb[1] << 2)) as u8;
    let mut y: u64 = (p0 | (p1 << 1)) as u64;
    let mut ofs = 2u32;
    for i in 0..16 {
        let wv = (w[i] as u32 ^ inv[part[i] as usize]) as u64;
        y |= wv << ofs;
        ofs += 3 - (i == 0 || i == anchor) as u32;
    }
    put_le(&mut block[10..16], y);
}

/// `encode_mode2_rgb_block`: 3 subsets, 5-bit endpoints, no p-bits, 2-bit weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mode2_rgb_block(
    block: &mut Block,
    part_id: u32,
    mut lr: [u32; 3],
    mut lg: [u32; 3],
    mut lb: [u32; 3],
    mut hr: [u32; 3],
    mut hg: [u32; 3],
    mut hb: [u32; 3],
    w: &[u8; 16],
) {
    let pid = part_id as usize;
    let part = &BC7_PARTITION3[pid * 16..pid * 16 + 16];
    let a0 = BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_1[pid] as usize;
    let a1 = BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_2[pid] as usize;
    let mut inv = [0u32; 3];
    for (s, anchor) in [0, a0, a1].into_iter().enumerate() {
        if w[anchor] & 2 != 0 {
            std::mem::swap(&mut lr[s], &mut hr[s]);
            std::mem::swap(&mut lg[s], &mut hg[s]);
            std::mem::swap(&mut lb[s], &mut hb[s]);
            inv[s] = 3;
        }
    }
    let v: u64 = 0b100
        | ((part_id as u64) << 3)
        | ((lr[0] as u64) << 9)
        | ((hr[0] as u64) << 14)
        | ((lr[1] as u64) << 19)
        | ((hr[1] as u64) << 24)
        | ((lr[2] as u64) << 29)
        | ((hr[2] as u64) << 34)
        | ((lg[0] as u64) << 39)
        | ((hg[0] as u64) << 44)
        | ((lg[1] as u64) << 49)
        | ((hg[1] as u64) << 54)
        | ((lg[2] as u64) << 59);
    put_le(&mut block[0..8], v);
    let mut v1: u64 = (hg[2] as u64)
        | ((lb[0] as u64) << 5)
        | ((hb[0] as u64) << 10)
        | ((lb[1] as u64) << 15)
        | ((hb[1] as u64) << 20)
        | ((lb[2] as u64) << 25)
        | ((hb[2] as u64) << 30);
    put_le(&mut block[8..12], v1);
    v1 >>= 32;
    let mut ofs = 3u32;
    for i in 0..16 {
        let wv = (w[i] as u32 ^ inv[part[i] as usize]) as u64;
        v1 |= wv << ofs;
        ofs += 2 - (i == 0 || i == a0 || i == a1) as u32;
    }
    put_le(&mut block[12..16], v1);
}

/// `encode_mode3_rgb_block`: 2 subsets, 7-bit endpoints, 4 unique p-bits, 2-bit weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mode3_rgb_block(
    block: &mut Block,
    part_id: u32,
    mut lr: [u32; 2],
    mut lg: [u32; 2],
    mut lb: [u32; 2],
    mut hr: [u32; 2],
    mut hg: [u32; 2],
    mut hb: [u32; 2],
    mut p: [u32; 4],
    w: &[u8; 16],
) {
    let pid = part_id as usize;
    let part = &BC7_PARTITION2[pid * 16..pid * 16 + 16];
    let anchor = BC7_TABLE_ANCHOR_INDEX_SECOND_SUBSET[pid] as usize;
    let mut inv = [0u32; 2];
    for (s, a) in [0, anchor].into_iter().enumerate() {
        if w[a] & 2 != 0 {
            std::mem::swap(&mut lr[s], &mut hr[s]);
            std::mem::swap(&mut lg[s], &mut hg[s]);
            std::mem::swap(&mut lb[s], &mut hb[s]);
            p.swap(s * 2, s * 2 + 1);
            inv[s] = 3;
        }
    }
    let x: u64 = 0b1000
        | ((part_id as u64) << 4)
        | ((lr[0] as u64) << 10)
        | ((hr[0] as u64) << 17)
        | ((lr[1] as u64) << 24)
        | ((hr[1] as u64) << 31)
        | ((lg[0] as u64) << 38)
        | ((hg[0] as u64) << 45)
        | ((lg[1] as u64) << 52)
        | ((hg[1] as u64) << 59);
    put_le(&mut block[0..8], x);
    let mut y: u64 = ((hg[1] >> 5) as u64)
        | ((lb[0] as u64) << 2)
        | ((hb[0] as u64) << 9)
        | ((lb[1] as u64) << 16)
        | ((hb[1] as u64) << 23)
        | ((p[0] as u64) << 30)
        | ((p[1] as u64) << 31)
        | ((p[2] as u64) << 32)
        | ((p[3] as u64) << 33);
    let mut ofs = 34u32;
    for i in 0..16 {
        let wv = (w[i] as u32 ^ inv[part[i] as usize]) as u64;
        y |= wv << ofs;
        ofs += 2 - (i == 0 || i == anchor) as u32;
    }
    put_le(&mut block[8..16], y);
}

/// `encode_mode4_rgba_block`: 5-bit RGB / 6-bit A endpoints, rotation, index selector.
/// `w0` are the RGB weights, `w1` the alpha weights; `index_flag` 1 means RGB uses 3 bits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mode4_rgba_block(
    block: &mut Block,
    mut lr: u32,
    mut lg: u32,
    mut lb: u32,
    mut la: u32,
    mut hr: u32,
    mut hg: u32,
    mut hb: u32,
    mut ha: u32,
    w0: &[u8; 16],
    w1: &[u8; 16],
    rot_index: u32,
    index_flag: u32,
) {
    let mut inv = [0u32; 2];
    let (w2bit, w3bit) = if index_flag != 0 { (w1, w0) } else { (w0, w1) };
    if w3bit[0] & 4 != 0 {
        inv[0] = 7;
        if index_flag != 0 {
            std::mem::swap(&mut lr, &mut hr);
            std::mem::swap(&mut lg, &mut hg);
            std::mem::swap(&mut lb, &mut hb);
        } else {
            std::mem::swap(&mut la, &mut ha);
        }
    }
    if w2bit[0] & 2 != 0 {
        inv[1] = 3;
        if index_flag != 0 {
            std::mem::swap(&mut la, &mut ha);
        } else {
            std::mem::swap(&mut lr, &mut hr);
            std::mem::swap(&mut lg, &mut hg);
            std::mem::swap(&mut lb, &mut hb);
        }
    }
    block[0] = (0b10000 | (rot_index << 5) | (index_flag << 7)) as u8;
    let mut x: u64 = (lr as u64)
        | ((hr as u64) << 5)
        | ((lg as u64) << 10)
        | ((hg as u64) << 15)
        | ((lb as u64) << 20)
        | ((hb as u64) << 25)
        | ((la as u64) << 30)
        | ((ha as u64) << 36);
    put_le(&mut block[1..6], x);
    x >>= 40;
    let mut ofs0 = 2u32;
    for (i, &wv) in w2bit.iter().enumerate() {
        x |= ((wv as u32 ^ inv[1]) as u64) << ofs0;
        ofs0 += 2 - (i == 0) as u32;
    }
    put_le(&mut block[6..10], x);
    x >>= 32;
    let mut ofs1 = 1u32;
    for (i, &wv) in w3bit.iter().enumerate() {
        x |= ((wv as u32 ^ inv[0]) as u64) << ofs1;
        ofs1 += 3 - (i == 0) as u32;
    }
    put_le(&mut block[10..16], x);
}

/// `pack_mode5_solid`: lossless solid RGBA block.
pub(crate) fn pack_mode5_solid(block: &mut Block, c: &Px) {
    let m5 = &uastc_tables().bc7_mode_5_optimal_endpoints;
    block[0] = 0b0010_0000;
    let (lr, hr) = (m5[c[0] as usize].lo as u64, m5[c[0] as usize].hi as u64);
    let (lg, hg) = (m5[c[1] as usize].lo as u64, m5[c[1] as usize].hi as u64);
    let (lb, hb) = (m5[c[2] as usize].lo as u64, m5[c[2] as usize].hi as u64);
    let a = c[3] as u64;
    let mut x = lr | (hr << 7) | (lg << 14) | (hg << 21) | (lb << 28) | (hb << 35);
    x |= (a << 42) | (a << 50);
    put_le(&mut block[1..8], x);
    x >>= 56;
    block[8..16].copy_from_slice(&[0xac, 0xaa, 0xaa, 0xaa, 0, 0, 0, 0]);
    block[8] |= x as u8;
}

/// `encode_mode5_rgba_block`: 7-bit RGB / 8-bit A endpoints, 2-bit weights for both planes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mode5_rgba_block(
    block: &mut Block,
    mut lr: u32,
    mut lg: u32,
    mut lb: u32,
    mut la: u32,
    mut hr: u32,
    mut hg: u32,
    mut hb: u32,
    mut ha: u32,
    color_w: &[u8; 16],
    alpha_w: &[u8; 16],
    rot_index: u32,
) {
    let (mut color_inv, mut alpha_inv) = (0u32, 0u32);
    if color_w[0] & 2 != 0 {
        std::mem::swap(&mut lr, &mut hr);
        std::mem::swap(&mut lg, &mut hg);
        std::mem::swap(&mut lb, &mut hb);
        color_inv = 3;
    }
    if alpha_w[0] & 2 != 0 {
        std::mem::swap(&mut la, &mut ha);
        alpha_inv = 3;
    }
    let low: u64 = (1u64 << 5)
        | ((rot_index as u64) << 6)
        | ((lr as u64) << 8)
        | ((hr as u64) << 15)
        | ((lg as u64) << 22)
        | ((hg as u64) << 29)
        | ((lb as u64) << 36)
        | ((hb as u64) << 43)
        | ((la as u64) << 50)
        | ((ha as u64) << 58);
    put_le(&mut block[0..8], low);
    let mut high: u64 = ((ha >> 6) & 3) as u64;
    let mut ofs = 2u32;
    for (i, &wv) in color_w.iter().enumerate() {
        high |= ((wv as u32 ^ color_inv) as u64) << ofs;
        ofs += 2 - (i == 0) as u32;
    }
    for (i, &wv) in alpha_w.iter().enumerate() {
        high |= ((wv as u32 ^ alpha_inv) as u64) << ofs;
        ofs += 2 - (i == 0) as u32;
    }
    put_le(&mut block[8..16], high);
}

/// `encode_mode6_rgba_block`: 7-bit endpoints, 2 unique p-bits, 4-bit weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mode6_rgba_block(
    block: &mut Block,
    mut lr: u32,
    mut lg: u32,
    mut lb: u32,
    mut la: u32,
    mut p0: u32,
    mut hr: u32,
    mut hg: u32,
    mut hb: u32,
    mut ha: u32,
    mut p1: u32,
    w: &[u8; 16],
) {
    let mut inv = 0u32;
    if w[0] & 8 != 0 {
        std::mem::swap(&mut lr, &mut hr);
        std::mem::swap(&mut lg, &mut hg);
        std::mem::swap(&mut lb, &mut hb);
        std::mem::swap(&mut la, &mut ha);
        std::mem::swap(&mut p0, &mut p1);
        inv = 15;
    }
    let mut x: u64 = 0b100_0000
        | ((lr as u64) << 7)
        | ((hr as u64) << 14)
        | ((lg as u64) << 21)
        | ((hg as u64) << 28)
        | ((lb as u64) << 35)
        | ((hb as u64) << 42)
        | ((la as u64) << 49)
        | ((ha as u64) << 56);
    put_le(&mut block[0..7], x);
    x >>= 56;
    x |= (p0 as u64) << 7;
    block[7] = x as u8;
    let mut y: u64 = p1 as u64;
    let mut ofs = 1u32;
    for (i, &wv) in w.iter().enumerate() {
        y |= ((wv as u32 ^ inv) as u64) << ofs;
        ofs += 3 + (i > 0) as u32;
    }
    put_le(&mut block[8..16], y);
}

/// `encode_mode7_rgba_block`: 2 subsets, 5-bit RGBA endpoints, 4 unique p-bits, 2-bit weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mode7_rgba_block(
    block: &mut Block,
    part_id: u32,
    mut lr: [u32; 2],
    mut lg: [u32; 2],
    mut lb: [u32; 2],
    mut la: [u32; 2],
    mut hr: [u32; 2],
    mut hg: [u32; 2],
    mut hb: [u32; 2],
    mut ha: [u32; 2],
    mut p: [u32; 4],
    w: &[u8; 16],
) {
    let pid = part_id as usize;
    let part = &BC7_PARTITION2[pid * 16..pid * 16 + 16];
    let anchor = BC7_TABLE_ANCHOR_INDEX_SECOND_SUBSET[pid] as usize;
    let mut inv = [0u32; 2];
    for (s, a) in [0, anchor].into_iter().enumerate() {
        if w[a] & 2 != 0 {
            std::mem::swap(&mut lr[s], &mut hr[s]);
            std::mem::swap(&mut lg[s], &mut hg[s]);
            std::mem::swap(&mut lb[s], &mut hb[s]);
            std::mem::swap(&mut la[s], &mut ha[s]);
            p.swap(s * 2, s * 2 + 1);
            inv[s] = 3;
        }
    }
    let x: u64 = 0x80
        | ((part_id as u64) << 8)
        | ((lr[0] as u64) << 14)
        | ((hr[0] as u64) << 19)
        | ((lr[1] as u64) << 24)
        | ((hr[1] as u64) << 29)
        | ((lg[0] as u64) << 34)
        | ((hg[0] as u64) << 39)
        | ((lg[1] as u64) << 44)
        | ((hg[1] as u64) << 49)
        | ((lb[0] as u64) << 54)
        | ((hb[0] as u64) << 59);
    put_le(&mut block[0..8], x);
    let mut y: u64 = (lb[1] as u64)
        | ((hb[1] as u64) << 5)
        | ((la[0] as u64) << 10)
        | ((ha[0] as u64) << 15)
        | ((la[1] as u64) << 20)
        | ((ha[1] as u64) << 25)
        | ((p[0] as u64) << 30)
        | ((p[1] as u64) << 31)
        | ((p[2] as u64) << 32)
        | ((p[3] as u64) << 33);
    let mut ofs = 34u32;
    for i in 0..16 {
        let wv = (w[i] as u32 ^ inv[part[i] as usize]) as u64;
        y |= wv << ofs;
        ofs += 2 - (i == 0 || i == anchor) as u32;
    }
    put_le(&mut block[8..16], y);
}

// ---------------------------------------------------------------------------------------------
// Least squares.

/// 2x2 normal-equation solve shared by the LS helpers. Returns (iz00, iz01, iz10, iz11).
#[inline]
fn ls_inverse(z00: f32, z10: f32, z11: f32) -> Option<(f32, f32, f32, f32)> {
    let z01 = z10;
    let det = z00 * z11 - z01 * z10;
    if det.abs() < 1e-8f32 {
        return None;
    }
    let det = 1.0 / det;
    Some((z11 * det, -z01 * det, -z10 * det, z00 * det))
}

/// `compute_least_squares_endpoints_1D` for channel `comp`: returns (xl, xh).
fn ls_endpoints_1d(
    w: &[u8; 16],
    tab: &[[f32; 4]],
    px: &[Px; 16],
    comp: usize,
    t: f32,
) -> Option<(f32, f32)> {
    let (mut z00, mut z10, mut z11, mut q00) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..16 {
        let s = &tab[w[i] as usize];
        z00 += s[0];
        z10 += s[1];
        z11 += s[2];
        q00 += s[3] * px[i][comp] as f32;
    }
    let q10 = t - q00;
    let (iz00, iz01, iz10, iz11) = ls_inverse(z00, z10, z11)?;
    let xh = clampf(iz00 * q00 + iz01 * q10, 0.0, 255.0);
    let xl = clampf(iz10 * q00 + iz11 * q10, 0.0, 255.0);
    Some((xl, xh))
}

/// `compute_least_squares_endpoints_3D` / `_4D` (`N` channels): returns (xl, xh).
fn ls_endpoints<const N: usize>(
    w: &[u8; 16],
    tab: &[[f32; 4]],
    px: &[Px; 16],
    t: [f32; N],
) -> Option<([f32; 4], [f32; 4])> {
    let (mut z00, mut z10, mut z11) = (0.0f32, 0.0f32, 0.0f32);
    let mut q00 = [0.0f32; N];
    for i in 0..16 {
        let s = &tab[w[i] as usize];
        z00 += s[0];
        z10 += s[1];
        z11 += s[2];
        for c in 0..N {
            q00[c] += s[3] * px[i][c] as f32;
        }
    }
    let (iz00, iz01, iz10, iz11) = ls_inverse(z00, z10, z11)?;
    let (mut xl, mut xh) = ([0.0f32; 4], [0.0f32; 4]);
    for c in 0..N {
        let q10 = t[c] - q00[c];
        xh[c] = clampf(iz00 * q00[c] + iz01 * q10, 0.0, 255.0);
        xl[c] = clampf(iz10 * q00[c] + iz11 * q10, 0.0, 255.0);
    }
    Some((xl, xh))
}

/// Per-subset LS accumulators used by the multi-subset packers.
#[derive(Clone, Copy, Default)]
struct SubsetLs {
    z00: f32,
    z10: f32,
    z11: f32,
    q00: [f32; 4],
}

/// Accumulates LS sums per subset. `subset_of(i)` maps a pixel to its subset.
fn accum_subsets<const S: usize>(
    w: &[u8; 16],
    tab: &[[f32; 4]],
    px: &[Px; 16],
    chans: usize,
    subset_of: impl Fn(usize) -> usize,
) -> [SubsetLs; S] {
    let mut acc = [SubsetLs::default(); S];
    for i in 0..16 {
        let a = &mut acc[subset_of(i)];
        let s = &tab[w[i] as usize];
        a.z00 += s[0];
        a.z10 += s[1];
        a.z11 += s[2];
        let wt = s[3];
        for c in 0..chans {
            a.q00[c] += wt * px[i][c] as f32;
        }
    }
    acc
}

/// Solves one subset: returns (low, high) per channel, unclamped.
fn solve_subset(a: &SubsetLs, totals: [i32; 4], chans: usize) -> Option<([f32; 4], [f32; 4])> {
    let mut q10 = [0.0f32; 4];
    for c in 0..chans {
        q10[c] = totals[c] as f32 - a.q00[c];
    }
    let (iz00, iz01, iz10, iz11) = ls_inverse(a.z00, a.z10, a.z11)?;
    let (mut l, mut h) = ([0.0f32; 4], [0.0f32; 4]);
    for c in 0..chans {
        h[c] = iz00 * a.q00[c] + iz01 * q10[c];
        l[c] = iz10 * a.q00[c] + iz11 * q10[c];
    }
    Some((l, h))
}

// ---------------------------------------------------------------------------------------------
// Weight evaluation (the `_sse` variants return the same weights plus the SSE).

#[inline]
fn bc7_sse1(p: i32, l: i32, d: i32, w: u32) -> u32 {
    let e = p - (l + ((d * w as i32 + 32) >> 6));
    (e * e) as u32
}

/// Projects `dot` onto a `max+1` level weight, clamping like the C++ code.
#[inline]
fn quant_sel(dot: i32, f: f32, max: i32) -> i32 {
    let mut sel = f2i(dot as f32 * f + 0.5);
    if (sel as u32) > max as u32 {
        sel = (!sel >> 31) & max;
    }
    sel
}

/// Dequantized single-subset endpoints (`N` channels) for weight evaluation.
struct Line<const N: usize> {
    l: [i32; N],
    d: [i32; N],
    f: f32,
    sofs: i32,
}

impl<const N: usize> Line<N> {
    fn new(l: [i32; N], h: [i32; N], levels: f32) -> Self {
        let mut d = [0i32; N];
        let mut dd = 0i32;
        let mut sofs = 0i32;
        for c in 0..N {
            d[c] = h[c] - l[c];
            dd += sq(d[c]);
            sofs += l[c] * d[c];
        }
        Line {
            l,
            d,
            f: levels / (dd as f32 + 0.00000125f32),
            sofs,
        }
    }

    #[inline]
    fn sel(&self, p: &Px, max: i32) -> i32 {
        let mut dot = 0i32;
        for c in 0..N {
            dot += p[c] as i32 * self.d[c];
        }
        quant_sel(dot - self.sofs, self.f, max)
    }

    #[inline]
    fn sse(&self, p: &Px, w: u32) -> u32 {
        let mut e = 0u32;
        for c in 0..N {
            e += bc7_sse1(p[c] as i32, self.l[c], self.d[c], w);
        }
        e
    }
}

/// `eval_weights_mode6_rgb[_sse]`: alpha is assumed to be 255 (packed A=127 with the p-bits).
fn eval_weights_mode6_rgb(
    px: &[Px; 16],
    w: &mut [u8; 16],
    l: [i32; 3],
    h: [i32; 3],
    p0: i32,
    p1: i32,
) -> u32 {
    let lq = [from_7p(l[0], p0), from_7p(l[1], p0), from_7p(l[2], p0)];
    let hq = [from_7p(h[0], p1), from_7p(h[1], p1), from_7p(h[2], p1)];
    let line = Line::new(lq, hq, 15.0);
    let la = from_7p(127, p0);
    let da = from_7p(127, p1) - la;
    let mut sse = 0u32;
    for i in 0..16 {
        let sel = line.sel(&px[i], 15);
        w[i] = sel as u8;
        let wt = BC7_WEIGHTS4[sel as usize];
        sse += line.sse(&px[i], wt) + bc7_sse1(px[i][3] as i32, la, da, wt);
    }
    sse
}

/// `eval_weights_mode6_rgba[_sse]`.
fn eval_weights_mode6_rgba(
    px: &[Px; 16],
    w: &mut [u8; 16],
    l: [i32; 4],
    p0: i32,
    h: [i32; 4],
    p1: i32,
) -> u32 {
    let lq = l.map(|v| from_7p(v, p0));
    let hq = h.map(|v| from_7p(v, p1));
    let line = Line::new(lq, hq, 15.0);
    let mut sse = 0u32;
    for i in 0..16 {
        let sel = line.sel(&px[i], 15);
        w[i] = sel as u8;
        sse += line.sse(&px[i], BC7_WEIGHTS4[sel as usize]);
    }
    sse
}

/// Two-subset weight evaluation over `subset_bitmask` (bit i set: pixel i in subset 1).
fn eval_weights_2subsets<const N: usize>(
    px: &[Px; 16],
    w: &mut [u8; 16],
    lines: &[Line<N>; 2],
    subset_bitmask: u32,
    weights: &[u32],
) -> u32 {
    let max = weights.len() as i32 - 1;
    let mut sse = 0u32;
    for i in 0..16 {
        let line = &lines[((subset_bitmask >> i) & 1) as usize];
        let sel = line.sel(&px[i], max);
        w[i] = sel as u8;
        sse += line.sse(&px[i], weights[sel as usize]);
    }
    sse
}

/// Three-subset weight evaluation over BC7 partition `pat_index`.
fn eval_weights_3subsets(
    px: &[Px; 16],
    w: &mut [u8; 16],
    lines: &[Line<3>; 3],
    pat_index: u32,
    weights: &[u32],
) -> u32 {
    let max = weights.len() as i32 - 1;
    let part = &BC7_PARTITION3[pat_index as usize * 16..pat_index as usize * 16 + 16];
    let mut sse = 0u32;
    for i in 0..16 {
        let line = &lines[part[i] as usize];
        let sel = line.sel(&px[i], max);
        w[i] = sel as u8;
        sse += line.sse(&px[i], weights[sel as usize]);
    }
    sse
}

/// `eval_weights_mode1_rgb[_sse]`.
fn eval_weights_mode1_rgb(
    px: &[Px; 16],
    w: &mut [u8; 16],
    e: &Ep<2>,
    pbits: &[u32; 2],
    mask: u32,
) -> u32 {
    let lines = [0, 1].map(|s| {
        let p = pbits[s] as i32;
        Line::new(
            [
                from_6p(e.lr[s] as i32, p),
                from_6p(e.lg[s] as i32, p),
                from_6p(e.lb[s] as i32, p),
            ],
            [
                from_6p(e.hr[s] as i32, p),
                from_6p(e.hg[s] as i32, p),
                from_6p(e.hb[s] as i32, p),
            ],
            7.0,
        )
    });
    eval_weights_2subsets(px, w, &lines, mask, &BC7_WEIGHTS3)
}

/// `eval_weights_mode3_rgb[_sse]`.
fn eval_weights_mode3_rgb(
    px: &[Px; 16],
    w: &mut [u8; 16],
    e: &Ep<2>,
    pbits: &[u32; 4],
    mask: u32,
) -> u32 {
    let lines = [0, 1].map(|s| {
        let (pl, ph) = (pbits[s * 2] as i32, pbits[s * 2 + 1] as i32);
        Line::new(
            [
                from_7p(e.lr[s] as i32, pl),
                from_7p(e.lg[s] as i32, pl),
                from_7p(e.lb[s] as i32, pl),
            ],
            [
                from_7p(e.hr[s] as i32, ph),
                from_7p(e.hg[s] as i32, ph),
                from_7p(e.hb[s] as i32, ph),
            ],
            3.0,
        )
    });
    eval_weights_2subsets(px, w, &lines, mask, &BC7_WEIGHTS2)
}

/// `eval_weights_mode7_rgba[_sse]`.
fn eval_weights_mode7_rgba(
    px: &[Px; 16],
    w: &mut [u8; 16],
    e: &Ep<2>,
    la: &[u32; 2],
    ha: &[u32; 2],
    pbits: &[u32; 4],
    mask: u32,
) -> u32 {
    let lines = [0, 1].map(|s| {
        let (pl, ph) = (pbits[s * 2] as i32, pbits[s * 2 + 1] as i32);
        Line::new(
            [
                from_5p(e.lr[s] as i32, pl),
                from_5p(e.lg[s] as i32, pl),
                from_5p(e.lb[s] as i32, pl),
                from_5p(la[s] as i32, pl),
            ],
            [
                from_5p(e.hr[s] as i32, ph),
                from_5p(e.hg[s] as i32, ph),
                from_5p(e.hb[s] as i32, ph),
                from_5p(ha[s] as i32, ph),
            ],
            3.0,
        )
    });
    eval_weights_2subsets(px, w, &lines, mask, &BC7_WEIGHTS2)
}

/// `eval_weights_mode0_rgb[_sse]`.
fn eval_weights_mode0_rgb(
    px: &[Px; 16],
    w: &mut [u8; 16],
    e: &Ep<3>,
    pbits: &[u32; 6],
    pat: u32,
) -> u32 {
    let lines = [0, 1, 2].map(|s| {
        let (pl, ph) = (pbits[s * 2] as i32, pbits[s * 2 + 1] as i32);
        Line::new(
            [
                from_4p(e.lr[s] as i32, pl),
                from_4p(e.lg[s] as i32, pl),
                from_4p(e.lb[s] as i32, pl),
            ],
            [
                from_4p(e.hr[s] as i32, ph),
                from_4p(e.hg[s] as i32, ph),
                from_4p(e.hb[s] as i32, ph),
            ],
            7.0,
        )
    });
    eval_weights_3subsets(px, w, &lines, pat, &BC7_WEIGHTS3)
}

/// `eval_weights_mode2_rgb[_sse]`.
fn eval_weights_mode2_rgb(px: &[Px; 16], w: &mut [u8; 16], e: &Ep<3>, pat: u32) -> u32 {
    let lines = [0, 1, 2].map(|s| {
        Line::new(
            [
                from_5(e.lr[s] as i32),
                from_5(e.lg[s] as i32),
                from_5(e.lb[s] as i32),
            ],
            [
                from_5(e.hr[s] as i32),
                from_5(e.hg[s] as i32),
                from_5(e.hb[s] as i32),
            ],
            3.0,
        )
    });
    eval_weights_3subsets(px, w, &lines, pat, &BC7_WEIGHTS2)
}

/// `eval_weights_mode4_{3,2}bit_rgb[_sse]` / `eval_weights_mode5_2bit_rgb[_sse]`: `deq` expands
/// the endpoints, `weights` selects the index precision.
fn eval_weights_rgb_1subset(
    px: &[Px; 16],
    w: &mut [u8; 16],
    l: [i32; 3],
    h: [i32; 3],
    deq: fn(i32) -> i32,
    weights: &[u32],
) -> u32 {
    let max = weights.len() as i32 - 1;
    let line = Line::new(l.map(deq), h.map(deq), max as f32);
    let mut sse = 0u32;
    for i in 0..16 {
        let sel = line.sel(&px[i], max);
        w[i] = sel as u8;
        sse += line.sse(&px[i], weights[sel as usize]);
    }
    sse
}

/// `eval_weights_mode4_{2,3}bit_a[_sse]` (`la`/`ha` already dequantized for mode 5).
fn eval_weights_a(px: &[Px; 16], w: &mut [u8; 16], la: i32, ha: i32, weights: &[u32]) -> u32 {
    let max = weights.len() as i32 - 1;
    let da = ha - la;
    let f = max as f32 / (da as f32 + 0.00000125f32);
    let mut sse = 0u32;
    for i in 0..16 {
        let a = px[i][3] as i32;
        let mut sel = f2i((a - la) as f32 * f + 0.5);
        if (sel as u32) > max as u32 {
            sel = (!sel >> 31) & max;
        }
        w[i] = sel as u8;
        sse += bc7_sse1(a, la, da, weights[sel as usize]);
    }
    sse
}

// ---------------------------------------------------------------------------------------------
// P-bit selection.

/// Endpoint quantization for one p-bit (`clampi((int)((x*scale-p)/2+.5)*2+p, ...)`).
#[inline]
fn quant_with_pbit(x: f32, scalep: f32, p: i32, iscalep: i32) -> u8 {
    let t = f2i((x * scalep - p as f32) * 0.5 + 0.5);
    clampi(t.wrapping_mul(2).wrapping_add(p), p, iscalep - 1 + p) as u8
}

#[inline]
fn scale_to_8(v: u8, total_bits: u32) -> u8 {
    let s = ((v as u32) << (8 - total_bits)) as u8;
    // C++ promotes to int: a shift by 8 (8-bit components) gives 0, not an overflow.
    s | ((u32::from(s) >> total_bits) as u8)
}

/// `determine_unique_pbits`: `xl`/`xh` are in [0,1]. Returns (min color, max color, pbits).
pub(super) fn determine_unique_pbits(
    total_comps: usize,
    comp_bits: u32,
    xl: &[f32; 4],
    xh: &[f32; 4],
) -> (Px, Px, [u32; 2]) {
    let total_bits = comp_bits + 1;
    let iscalep = (1i32 << total_bits) - 1;
    let scalep = iscalep as f32;
    let (mut best_err0, mut best_err1) = (1e9f32, 1e9f32);
    let (mut best_min, mut best_max, mut best_pbits) = ([0u8; 4], [0u8; 4], [0u32; 2]);
    for p in 0..2i32 {
        let mut xmin = [0u8; 4];
        let mut xmax = [0u8; 4];
        for c in 0..4 {
            xmin[c] = quant_with_pbit(xl[c], scalep, p, iscalep);
            xmax[c] = quant_with_pbit(xh[c], scalep, p, iscalep);
        }
        let (mut err0, mut err1) = (0.0f32, 0.0f32);
        for i in 0..total_comps {
            let sl = scale_to_8(xmin[i], total_bits) as f32;
            let sh = scale_to_8(xmax[i], total_bits) as f32;
            let d0 = sl - xl[i] * 255.0;
            let d1 = sh - xh[i] * 255.0;
            err0 += d0 * d0;
            err1 += d1 * d1;
        }
        if err0 < best_err0 {
            best_err0 = err0;
            best_pbits[0] = p as u32;
            best_min = xmin.map(|v| v >> 1);
        }
        if err1 < best_err1 {
            best_err1 = err1;
            best_pbits[1] = p as u32;
            best_max = xmax.map(|v| v >> 1);
        }
    }
    (best_min, best_max, best_pbits)
}

/// `determine_shared_pbits`: both endpoints use the same p-bit.
pub(super) fn determine_shared_pbits(
    total_comps: usize,
    comp_bits: u32,
    xl: &[f32; 4],
    xh: &[f32; 4],
) -> (Px, Px, [u32; 2]) {
    let total_bits = comp_bits + 1;
    let iscalep = (1i32 << total_bits) - 1;
    let scalep = iscalep as f32;
    let mut best_err = 1e9f32;
    let (mut best_min, mut best_max, mut best_pbits) = ([0u8; 4], [0u8; 4], [0u32; 2]);
    for p in 0..2i32 {
        let mut xmin = [0u8; 4];
        let mut xmax = [0u8; 4];
        for c in 0..4 {
            xmin[c] = quant_with_pbit(xl[c], scalep, p, iscalep);
            xmax[c] = quant_with_pbit(xh[c], scalep, p, iscalep);
        }
        let mut err = 0.0f32;
        for i in 0..total_comps {
            let sl = scale_to_8(xmin[i], total_bits) as i32 as f32;
            let sh = scale_to_8(xmax[i], total_bits) as i32 as f32;
            let d0 = sl * (1.0f32 / 255.0f32) - xl[i];
            let d1 = sh * (1.0f32 / 255.0f32) - xh[i];
            err += d0 * d0 + d1 * d1;
        }
        if err < best_err {
            best_err = err;
            best_pbits = [p as u32; 2];
            best_min = xmin.map(|v| v >> 1);
            best_max = xmax.map(|v| v >> 1);
        }
    }
    (best_min, best_max, best_pbits)
}

// ---------------------------------------------------------------------------------------------
// Analytical estimates.

/// Multi-channel `analytical_quant_est_sse` (span weights always null in bc7f).
fn analytical_quant_est_sse(
    e_levels: i32,
    w_levels: i32,
    num_chans: usize,
    spans: &[i32; 4],
    endpoint_weight_scale: f32,
    num_pixels: i32,
) -> f32 {
    let dep = 1.0f32 / (e_levels - 1) as f32;
    let dw = 1.0f32 / (w_levels - 1) as f32;
    let n = w_levels as f32;
    let ab_sum = (2.0f32 * n - 1.0) / (3.0f32 * (n - 1.0));
    let mut pixel_sse = if e_levels == 256 {
        0.0f32
    } else {
        (dep * dep)
            * ((1.0f32 / 12.0f32) * ab_sum * (255.0f32 * 255.0f32))
            * num_chans as f32
            * endpoint_weight_scale
    };
    let k = (dw * dw) * (1.0f32 / 12.0f32);
    for &s in spans.iter().take(num_chans) {
        pixel_sse += k * (s * s) as f32 * 1.0f32;
    }
    pixel_sse * num_pixels as f32
}

/// Single-channel `analytical_quant_est_sse`.
fn analytical_quant_est_sse1(
    e_levels: i32,
    w_levels: i32,
    span: i32,
    span_weight: f32,
    endpoint_weight_scale: f32,
    num_pixels: i32,
) -> f32 {
    let dep = 1.0f32 / (e_levels - 1) as f32;
    let dw = 1.0f32 / (w_levels - 1) as f32;
    let n = w_levels as f32;
    let ab_sum = (2.0f32 * n - 1.0) / (3.0f32 * (n - 1.0));
    let mut pixel_sse = if e_levels == 256 {
        0.0f32
    } else {
        (dep * dep) * ((1.0f32 / 12.0f32) * ab_sum * (255.0f32 * 255.0f32)) * endpoint_weight_scale
    };
    pixel_sse += (dw * dw) * (1.0f32 / 12.0f32) * (span * span) as f32 * span_weight;
    pixel_sse * num_pixels as f32
}

/// `estimate_slam_to_line_sse_3D`: returns (ortho var, ortho ratio).
fn estimate_slam_to_line_sse_3d(
    cov: &[f32; 6],
    mut xr: f32,
    mut yr: f32,
    mut zr: f32,
) -> (f32, f32) {
    let total_var = cov[0] + cov[3] + cov[5];
    let l = (xr * xr + yr * yr + zr * zr).sqrt();
    if l < SMALL_FLOAT_VAL {
        xr = 0.577350269;
        yr = 0.577350269;
        zr = 0.577350269;
    } else {
        let l = 1.0 / l;
        xr *= l;
        yr *= l;
        zr *= l;
    }
    let xr2 = cov[0] * xr + cov[1] * yr + cov[2] * zr;
    let xg2 = cov[1] * xr + cov[3] * yr + cov[4] * zr;
    let xb2 = cov[2] * xr + cov[4] * yr + cov[5] * zr;
    let principal_axis_var = xr2 * xr + xg2 * yr + xb2 * zr;
    let ortho_var = maxf(0.0, total_var - principal_axis_var);
    let ratio = if total_var > SMALL_FLOAT_VAL {
        ortho_var / total_var
    } else {
        0.0
    };
    (ortho_var, ratio)
}

/// `estimate_slam_to_line_sse_4D`: returns (ortho var, ortho ratio).
fn estimate_slam_to_line_sse_4d(
    cov: &[f32; 10],
    mut xr: f32,
    mut yr: f32,
    mut zr: f32,
    mut wr: f32,
) -> (f32, f32) {
    let total_var = cov[0] + cov[4] + cov[7] + cov[9];
    let l = (xr * xr + yr * yr + zr * zr + wr * wr).sqrt();
    if l < SMALL_FLOAT_VAL {
        xr = 0.5;
        yr = 0.5;
        zr = 0.5;
        wr = 0.5;
    } else {
        let l = 1.0 / l;
        xr *= l;
        yr *= l;
        zr *= l;
        wr *= l;
    }
    let xr2 = cov[0] * xr + cov[1] * yr + cov[2] * zr + cov[3] * wr;
    let xg2 = cov[1] * xr + cov[4] * yr + cov[5] * zr + cov[6] * wr;
    let xb2 = cov[2] * xr + cov[5] * yr + cov[7] * zr + cov[8] * wr;
    let xa2 = cov[3] * xr + cov[6] * yr + cov[8] * zr + cov[9] * wr;
    let principal_axis_var = xr2 * xr + xg2 * yr + xb2 * zr + xa2 * wr;
    let ortho_var = maxf(0.0, total_var - principal_axis_var);
    let ratio = if total_var > SMALL_FLOAT_VAL {
        ortho_var / total_var
    } else {
        0.0
    };
    (ortho_var, ratio)
}

/// `calc_sse`: squared RGBA error of a packed BC7 block vs `px` (`u32::MAX` if invalid).
pub fn calc_sse(block: &Block, px: &[Px; 16]) -> u32 {
    let mut unpacked = [[0u8; 4]; 16];
    if !crate::bc::bc7::unpack_bc7(block, &mut unpacked) {
        return u32::MAX;
    }
    let mut sse = 0u32;
    for i in 0..16 {
        for c in 0..4 {
            sse += sq(px[i][c] as i32 - unpacked[i][c] as i32) as u32;
        }
    }
    sse
}

// ---------------------------------------------------------------------------------------------
// Shared analysis helpers.

/// RGB endpoints for `S` subsets.
#[derive(Clone, Copy)]
struct Ep<const S: usize> {
    lr: [u32; S],
    lg: [u32; S],
    lb: [u32; S],
    hr: [u32; S],
    hg: [u32; S],
    hb: [u32; S],
}

impl<const S: usize> Default for Ep<S> {
    fn default() -> Self {
        Ep {
            lr: [0; S],
            lg: [0; S],
            lb: [0; S],
            hr: [0; S],
            hg: [0; S],
            hb: [0; S],
        }
    }
}

impl<const S: usize> Ep<S> {
    fn set(&mut self, s: usize, l: &Px, h: &Px) {
        self.lr[s] = l[0] as u32;
        self.lg[s] = l[1] as u32;
        self.lb[s] = l[2] as u32;
        self.hr[s] = h[0] as u32;
        self.hg[s] = h[1] as u32;
        self.hb[s] = h[2] as u32;
    }
}

/// Integer 3x3 covariance (upper triangle: rr, rg, rb, gg, gb, bb) of `px` vs `mean`.
#[inline]
fn add_cov3(icov: &mut [i32; 6], p: &Px, mean: [i32; 3]) {
    let r = p[0] as i32 - mean[0];
    let g = p[1] as i32 - mean[1];
    let b = p[2] as i32 - mean[2];
    icov[0] += r * r;
    icov[1] += r * g;
    icov[2] += r * b;
    icov[3] += g * g;
    icov[4] += g * b;
    icov[5] += b * b;
}

/// Integer 4x4 covariance (rr, rg, rb, ra, gg, gb, ga, bb, ba, aa).
#[inline]
fn add_cov4(icov: &mut [i32; 10], p: &Px, mean: [i32; 4]) {
    let r = p[0] as i32 - mean[0];
    let g = p[1] as i32 - mean[1];
    let b = p[2] as i32 - mean[2];
    let a = p[3] as i32 - mean[3];
    icov[0] += r * r;
    icov[1] += r * g;
    icov[2] += r * b;
    icov[3] += r * a;
    icov[4] += g * g;
    icov[5] += g * b;
    icov[6] += g * a;
    icov[7] += b * b;
    icov[8] += b * a;
    icov[9] += a * a;
}

/// One power iteration step on the 3x3 covariance seeded with its diagonal.
fn principal_axis_3d(cov: &[f32; 6], sc: f32) -> [f32; 3] {
    let (wx, wy, wz) = (sc * cov[0], sc * cov[3], sc * cov[5]);
    [
        cov[0] * wx + cov[1] * wy + cov[2] * wz,
        cov[1] * wx + cov[3] * wy + cov[4] * wz,
        cov[2] * wx + cov[4] * wy + cov[5] * wz,
    ]
}

#[inline]
fn mul_cov4(cov: &[f32; 10], v: [f32; 4]) -> [f32; 4] {
    [
        cov[0] * v[0] + cov[1] * v[1] + cov[2] * v[2] + cov[3] * v[3],
        cov[1] * v[0] + cov[4] * v[1] + cov[5] * v[2] + cov[6] * v[3],
        cov[2] * v[0] + cov[5] * v[1] + cov[7] * v[2] + cov[8] * v[3],
        cov[3] * v[0] + cov[6] * v[1] + cov[8] * v[2] + cov[9] * v[3],
    ]
}

/// Integer projection axis (scaled by 16) from a float axis, `def` when the axis is tiny.
fn int_axis<const N: usize>(axis: [f32; N], def: [i32; N]) -> [i32; N] {
    let mut k = axis[0].abs();
    for v in axis.iter().skip(1) {
        k = maxf(k, v.abs());
    }
    let mut s = def;
    if k.abs() >= SMALL_FLOAT_VAL {
        let m = 2048.0f32 / k;
        for c in 0..N {
            s[c] = f2i(axis[c] * m);
        }
    }
    s.map(|v| ((v as u32) << 4) as i32)
}

#[inline]
fn dot_axis<const N: usize>(p: &Px, axis: &[i32; N], i: usize) -> i32 {
    let mut d = 0i32;
    for c in 0..N {
        d = d.wrapping_add((p[c] as i32).wrapping_mul(axis[c]));
    }
    d.wrapping_add(i as i32)
}

/// Pixel indices with the lowest/highest projection on `axis` (ties: lowest index).
fn minmax_on_axis<const N: usize>(px: &[Px; 16], axis: &[i32; N]) -> (usize, usize) {
    let (mut low, mut high) = (i32::MAX, i32::MIN);
    for (i, p) in px.iter().enumerate() {
        let d = dot_axis(p, axis, i);
        low = low.min(d);
        high = high.max(d);
    }
    ((low & 15) as usize, (high & 15) as usize)
}

/// Best 2-subset pattern for `desired` (bit i: pixel i's subset). Returns (index, bits).
fn pick_2subset_pattern(desired: u32, flags: u32) -> (u32, u32) {
    let t = tables();
    let mut best_diff = u32::MAX;
    let mut check = |p: u32| {
        let diff = (t.part2_bitmasks[p as usize] as u32 ^ desired).count_ones() as i32;
        let min_diff = ((diff.min(16 - diff) as u32) << 8) | p;
        if min_diff < best_diff {
            best_diff = min_diff;
        }
    };
    if flags & PACK_BC7_FLAG_ASTC_COMPATIBLE != 0 {
        for e in ASTC_BC7_COMMON_PARTITIONS2.iter() {
            check(e.0 as u32);
        }
    } else {
        for p in 0..MAX_PATTERNS2_TO_CHECK as u32 {
            check(p);
        }
    }
    let idx = best_diff & 0xFF;
    (idx, t.part2_bitmasks[idx as usize] as u32)
}

/// Subset means (rounded) and totals; `subset_of` maps pixel index to subset.
fn subset_stats<const S: usize>(
    px: &[Px; 16],
    chans: usize,
    subset_of: impl Fn(usize) -> usize,
) -> ([[i32; 4]; S], [i32; S], [[i32; 4]; S]) {
    let mut totals = [[0i32; 4]; S];
    let mut counts = [0i32; S];
    for i in 0..16 {
        let s = subset_of(i);
        for c in 0..chans {
            totals[s][c] += px[i][c] as i32;
        }
        counts[s] += 1;
    }
    let mut means = [[0i32; 4]; S];
    for s in 0..S {
        let t = counts[s] as u32;
        let h = t >> 1;
        for c in 0..chans {
            means[s][c] = if t == 0 {
                0
            } else {
                ((totals[s][c] as u32).wrapping_add(h) / t) as i32
            };
        }
    }
    (totals, counts, means)
}

// ---------------------------------------------------------------------------------------------
// Mode packers.

/// `pack_mode1_or_3_rgb`. Writes the estimate to `final_est` and the actual SSE to `actual`.
#[allow(clippy::too_many_arguments)]
fn pack_mode1_or_3_rgb(
    block: &mut Block,
    px: &[Px; 16],
    axis: [f32; 3],
    mean: [i32; 3],
    sse_est_to_beat: f32,
    flags: u32,
    final_est: &mut f32,
    actual: Option<&mut u32>,
) -> bool {
    let tabs = tables();
    let mut desired = 0u32;
    for i in 0..16 {
        let r = (px[i][0] as i32 - mean[0]) as f32;
        let g = (px[i][1] as i32 - mean[1]) as f32;
        let b = (px[i][2] as i32 - mean[2]) as f32;
        let subset = (r * axis[0] + g * axis[1] + b * axis[2] > 0.0) as u32;
        desired |= subset << i;
    }
    let (pat_index, pat_bits) = pick_2subset_pattern(desired, flags);
    let subset_of = |i: usize| ((pat_bits >> i) & 1) as usize;
    let (totals, counts, means) = subset_stats::<2>(px, 3, subset_of);

    let mut icov = [[0i32; 6]; 2];
    for i in 0..16 {
        let s = subset_of(i);
        add_cov3(
            &mut icov[s],
            &px[i],
            [means[s][0], means[s][1], means[s][2]],
        );
    }

    let mut axes = [[0i32; 3]; 2];
    let mut slam = 0.0f32;
    for s in 0..2 {
        let max_var = icov[s][0].max(icov[s][3]).max(icov[s][5]);
        let cov = icov[s].map(|v| v as f32);
        let sc = 1.0f32 / (max_var as f32 + 0.0000125f32);
        let alt = principal_axis_3d(&cov, sc);
        slam += estimate_slam_to_line_sse_3d(&cov, alt[0], alt[1], alt[2]).0;
        axes[s] = int_axis(alt, [306, 601, 117]);
    }

    let (mut low_dot, mut high_dot) = ([i32::MAX; 2], [i32::MIN; 2]);
    for i in 0..16 {
        let s = subset_of(i);
        let d = dot_axis(&px[i], &axes[s], i);
        low_dot[s] = low_dot[s].min(d);
        high_dot[s] = high_dot[s].max(d);
    }
    let low_c = low_dot.map(|v| (v & 15) as usize);
    let high_c = high_dot.map(|v| (v & 15) as usize);

    let pbit_opt = flags & PACK_BC7_FLAG_PBIT_OPT != 0;
    let mut quant = [0.0f32; 2];
    for s in 0..2 {
        let mut spans = [0i32; 4];
        for c in 0..3 {
            spans[c] = px[high_c[s]][c] as i32 - px[low_c[s]][c] as i32;
        }
        let d1 = if pbit_opt { UNIQUE_PBIT_DISCOUNT } else { 1.0 };
        let d3 = if pbit_opt { SHARED_PBIT_DISCOUNT } else { 1.0 };
        quant[0] += analytical_quant_est_sse(64, 8, 3, &spans, d1, counts[s]);
        quant[1] += analytical_quant_est_sse(128, 4, 3, &spans, d3, counts[s]);
    }
    let mode1_est = slam + quant[0];
    let mode3_est = slam + quant[1];
    let q = 1.0f32 / 255.0f32;
    let pxf = |p: &Px| [p[0] as f32 * q, p[1] as f32 * q, p[2] as f32 * q, 0.0];

    if mode1_est < mode3_est {
        *final_est = mode1_est;
        if mode1_est >= sse_est_to_beat {
            return false;
        }
        let mut e = Ep::<2>::default();
        let mut pbits = [0u32; 2];
        for s in 0..2 {
            let (lc, hc) = (&px[low_c[s]], &px[high_c[s]]);
            if pbit_opt {
                let (mn, mx, pb) = determine_shared_pbits(3, 6, &pxf(lc), &pxf(hc));
                pbits[s] = pb[0];
                e.set(s, &mn, &mx);
            } else {
                let l = lc[0] as i32 + lc[1] as i32 + lc[2] as i32;
                let h = hc[0] as i32 + hc[1] as i32 + hc[2] as i32;
                if l.max(h) >= 129 * 3 {
                    pbits[s] = 1;
                }
                let p = pbits[s] as i32;
                e.lr[s] = to_6p(lc[0] as i32, p) as u32;
                e.lg[s] = to_6p(lc[1] as i32, p) as u32;
                e.lb[s] = to_6p(lc[2] as i32, p) as u32;
                e.hr[s] = to_6p(hc[0] as i32, p) as u32;
                e.hg[s] = to_6p(hc[1] as i32, p) as u32;
                e.hb[s] = to_6p(hc[2] as i32, p) as u32;
            }
        }
        let mut w = [0u8; 16];
        eval_weights_mode1_rgb(px, &mut w, &e, &pbits, pat_bits);
        let acc = accum_subsets::<2>(&w, &tabs.ls3, px, 3, subset_of);
        for s in 0..2 {
            let t = [totals[s][0], totals[s][1], totals[s][2], 0];
            let Some((sl, sh)) = solve_subset(&acc[s], t, 3) else {
                continue;
            };
            if pbit_opt {
                let xl = [
                    clampf(sl[0] * q, 0.0, 1.0),
                    clampf(sl[1] * q, 0.0, 1.0),
                    clampf(sl[2] * q, 0.0, 1.0),
                    0.0,
                ];
                let xh = [
                    clampf(sh[0] * q, 0.0, 1.0),
                    clampf(sh[1] * q, 0.0, 1.0),
                    clampf(sh[2] * q, 0.0, 1.0),
                    0.0,
                ];
                let (mn, mx, pb) = determine_shared_pbits(3, 6, &xl, &xh);
                pbits[s] = pb[0];
                e.set(s, &mn, &mx);
            } else {
                let l = sl[0] + sl[1] + sl[2];
                let h = sh[0] + sh[1] + sh[2];
                pbits[s] = (maxf(l, h) >= 129.0f32 * 3.0f32) as u32;
                let p = pbits[s] as i32;
                e.lr[s] = to_6_clamp(sl[0], p) as u32;
                e.hr[s] = to_6_clamp(sh[0], p) as u32;
                e.lg[s] = to_6_clamp(sl[1], p) as u32;
                e.hg[s] = to_6_clamp(sh[1], p) as u32;
                e.lb[s] = to_6_clamp(sl[2], p) as u32;
                e.hb[s] = to_6_clamp(sh[2], p) as u32;
            }
        }
        let sse = eval_weights_mode1_rgb(px, &mut w, &e, &pbits, pat_bits);
        if let Some(a) = actual {
            *a = sse;
        }
        encode_mode1_rgb_block(
            block, pat_index, e.lr, e.lg, e.lb, e.hr, e.hg, e.hb, pbits[0], pbits[1], &w,
        );
    } else {
        *final_est = mode3_est;
        if mode3_est >= sse_est_to_beat {
            return false;
        }
        let mut e = Ep::<2>::default();
        let mut pbits = [0u32; 4];
        for s in 0..2 {
            let (lc, hc) = (&px[low_c[s]], &px[high_c[s]]);
            if pbit_opt {
                let (mn, mx, pb) = determine_unique_pbits(3, 7, &pxf(lc), &pxf(hc));
                pbits[s * 2] = pb[0];
                pbits[s * 2 + 1] = pb[1];
                e.set(s, &mn, &mx);
            } else {
                let l = lc[0] as i32 + lc[1] as i32 + lc[2] as i32;
                let lp = (l >= 129) as i32;
                pbits[s * 2] = lp as u32;
                e.lr[s] = to_7p(lc[0] as i32, lp) as u32;
                e.lg[s] = to_7p(lc[1] as i32, lp) as u32;
                e.lb[s] = to_7p(lc[2] as i32, lp) as u32;
                let h = hc[0] as i32 + hc[1] as i32 + hc[2] as i32;
                let hp = (h >= 129) as i32;
                pbits[s * 2 + 1] = hp as u32;
                e.hr[s] = to_7p(hc[0] as i32, hp) as u32;
                e.hg[s] = to_7p(hc[1] as i32, hp) as u32;
                e.hb[s] = to_7p(hc[2] as i32, hp) as u32;
            }
        }
        let mut w = [0u8; 16];
        eval_weights_mode3_rgb(px, &mut w, &e, &pbits, pat_bits);
        let acc = accum_subsets::<2>(&w, &tabs.ls2, px, 3, subset_of);
        for s in 0..2 {
            let t = [totals[s][0], totals[s][1], totals[s][2], 0];
            let Some((sl, sh)) = solve_subset(&acc[s], t, 3) else {
                continue;
            };
            if pbit_opt {
                let xl = [
                    clampf(sl[0] * q, 0.0, 1.0),
                    clampf(sl[1] * q, 0.0, 1.0),
                    clampf(sl[2] * q, 0.0, 1.0),
                    0.0,
                ];
                let xh = [
                    clampf(sh[0] * q, 0.0, 1.0),
                    clampf(sh[1] * q, 0.0, 1.0),
                    clampf(sh[2] * q, 0.0, 1.0),
                    0.0,
                ];
                let (mn, mx, pb) = determine_unique_pbits(3, 7, &xl, &xh);
                pbits[s * 2] = pb[0];
                pbits[s * 2 + 1] = pb[1];
                e.set(s, &mn, &mx);
            } else {
                let l = sl[0] + sl[1] + sl[2];
                let lp = (l >= 129.0f32 * 3.0f32) as i32;
                pbits[s * 2] = lp as u32;
                e.lr[s] = to_7_clamp(sl[0], lp) as u32;
                e.lg[s] = to_7_clamp(sl[1], lp) as u32;
                e.lb[s] = to_7_clamp(sl[2], lp) as u32;
                let h = sh[0] + sh[1] + sh[2];
                let hp = (h >= 129.0f32 * 3.0f32) as i32;
                pbits[s * 2 + 1] = hp as u32;
                e.hr[s] = to_7_clamp(sh[0], hp) as u32;
                e.hg[s] = to_7_clamp(sh[1], hp) as u32;
                e.hb[s] = to_7_clamp(sh[2], hp) as u32;
            }
        }
        let sse = eval_weights_mode3_rgb(px, &mut w, &e, &pbits, pat_bits);
        if let Some(a) = actual {
            *a = sse;
        }
        encode_mode3_rgb_block(
            block, pat_index, e.lr, e.lg, e.lb, e.hr, e.hg, e.hb, pbits, &w,
        );
    }
    true
}

#[inline]
fn dist3(a: &Px, b: &Px) -> i32 {
    sq(b[0] as i32 - a[0] as i32) + sq(b[1] as i32 - a[1] as i32) + sq(b[2] as i32 - a[2] as i32)
}

/// `determine_3subsets`: splits the block along the axis, then splits the worse half.
fn determine_3subsets(px: &[Px; 16], axis: [f32; 3], mean: [i32; 3]) -> Option<[u8; 16]> {
    let mut idx = [0usize; 16];
    let mut means = [[0i32; 3]; 2];
    let mut total = [0i32; 2];
    for i in 0..16 {
        let rd = px[i][0] as i32 - mean[0];
        let gd = px[i][1] as i32 - mean[1];
        let bd = px[i][2] as i32 - mean[2];
        let s = (rd as f32 * axis[0] + gd as f32 * axis[1] + bd as f32 * axis[2] > 0.0) as usize;
        idx[i] = s;
        for c in 0..3 {
            means[s][c] += px[i][c] as i32;
        }
        total[s] += 1;
    }
    for s in 0..2 {
        let t = total[s] as u32;
        if t == 0 {
            return None;
        }
        for c in 0..3 {
            means[s][c] = ((means[s][c] as u32).wrapping_add(t >> 1) / t) as i32;
        }
    }
    let mut sses = [0i32; 2];
    for i in 0..16 {
        let s = idx[i];
        let m = [means[s][0] as u8, means[s][1] as u8, means[s][2] as u8, 0];
        sses[s] += dist3(&px[i], &m);
    }
    let split = (sses[1] > sses[0]) as usize;
    if total[split] < 2 {
        return None;
    }
    let (mut lo_y, mut hi_y) = (i32::MAX, 0i32);
    for i in 0..16 {
        if idx[i] != split {
            continue;
        }
        let y = ((px[i][0] as i32 + px[i][1] as i32 + px[i][2] as i32) << 4) + i as i32;
        lo_y = lo_y.min(y);
        hi_y = hi_y.max(y);
    }
    let (lo, hi) = ((lo_y & 15) as usize, (hi_y & 15) as usize);
    if lo == hi {
        return None;
    }
    let mut out = [2u8; 16];
    for i in 0..16 {
        if idx[i] == split {
            let d0 = dist3(&px[lo], &px[i]);
            let d1 = dist3(&px[hi], &px[i]);
            out[i] = (d1 > d0) as u8;
        }
    }
    Some(out)
}

#[inline]
fn pop16(x: u32) -> i32 {
    (x & 0xFFFF).count_ones() as i32
}

/// `pick_3subset_pat_index`: returns (best pattern, best pattern among the first 16).
fn pick_3subset_pat_index(desired: &[u8; 16], flags: u32) -> (u32, u32) {
    let t = tables();
    let mut first16 = 0u32;
    let mut m = [0u32; 3];
    for (i, &s) in desired.iter().enumerate() {
        m[s as usize] |= 1 << i;
    }
    let n0 = pop16(m[0]);
    let n1 = pop16(m[1]);
    let n2 = 16 - n0 - n1;
    let mut best_score = -1i32;
    let mut best_pat = 0u32;

    let score = |p: usize| -> i32 {
        let s0m = t.part3_bitmasks[p] & 0xFFFF;
        let s1m = t.part3_bitmasks[p] >> 16;
        let c00 = pop16(m[0] & s0m);
        let c01 = pop16(m[0] & s1m);
        let c02 = n0 - c00 - c01;
        let c10 = pop16(m[1] & s0m);
        let c11 = pop16(m[1] & s1m);
        let c12 = n1 - c10 - c11;
        let c20 = pop16(m[2] & s0m);
        let c21 = pop16(m[2] & s1m);
        let c22 = n2 - c20 - c21;
        let cands = [
            c00 + c11 + c22,
            c00 + c12 + c21,
            c01 + c10 + c22,
            c01 + c12 + c20,
            c02 + c10 + c21,
            c02 + c11 + c20,
        ];
        let mut s = cands[0];
        for &c in &cands[1..] {
            if c > s {
                s = c;
            }
        }
        s
    };

    if flags & PACK_BC7_FLAG_ASTC_COMPATIBLE != 0 {
        for (ti, e) in ASTC_BC7_COMMON_PARTITIONS3.iter().enumerate() {
            let p = e.0 as u32;
            let s = score(p as usize);
            if s > best_score {
                best_score = s;
                best_pat = p;
                if s == 16 {
                    if p <= 15 {
                        first16 = best_pat;
                    }
                    break;
                }
            }
            if ti == 6 {
                first16 = best_pat;
            }
        }
    } else {
        for p in 0..MAX_PATTERNS3_TO_CHECK as u32 {
            let s = score(p as usize);
            if s > best_score {
                best_score = s;
                best_pat = p;
                if s == 16 {
                    if p <= 15 {
                        first16 = best_pat;
                    }
                    break;
                }
            }
            if p == 15 {
                first16 = best_pat;
            }
        }
    }
    (best_pat, first16)
}

/// `pack_mode0_or_2_rgb`: false if packing failed or the estimate doesn't beat the target.
#[allow(clippy::too_many_arguments)]
fn pack_mode0_or_2_rgb(
    block: &mut Block,
    px: &[Px; 16],
    axis: [f32; 3],
    mean: [i32; 3],
    sse_est_to_beat: f32,
    flags: u32,
    final_est: &mut f32,
    actual: Option<&mut u32>,
) -> bool {
    let tabs = tables();
    let Some(desired) = determine_3subsets(px, axis, mean) else {
        *final_est = 1e9;
        return false;
    };
    let (pat2, pat0) = pick_3subset_pat_index(&desired, flags);
    let pats = [pat0, pat2];

    let mut quant_sse = [0.0f32; 2];
    let mut slam_sse = [0.0f32; 2];
    let mut mode_count = [[0i32; 3]; 2];
    let mut mode_low = [[0usize; 3]; 2];
    let mut mode_high = [[0usize; 3]; 2];
    let mut mode_totals = [[[0i32; 4]; 3]; 2];

    for mi in 0..2 {
        if mi == 1 && pats[0] == pats[1] {
            mode_count[1] = mode_count[0];
            mode_low[1] = mode_low[0];
            mode_high[1] = mode_high[0];
            mode_totals[1] = mode_totals[0];
            slam_sse[1] = slam_sse[0];
        } else {
            let pi = pats[mi] as usize;
            let part = &BC7_PARTITION3[pi * 16..pi * 16 + 16];
            let subset_of = |i: usize| part[i] as usize;
            let (totals, counts, means) = subset_stats::<3>(px, 3, subset_of);
            mode_totals[mi] = totals;
            mode_count[mi] = counts;

            let mut icov = [[0i32; 6]; 3];
            for i in 0..16 {
                let s = subset_of(i);
                add_cov3(
                    &mut icov[s],
                    &px[i],
                    [means[s][0], means[s][1], means[s][2]],
                );
            }
            let mut axes = [[0i32; 3]; 3];
            let mut slam = 0.0f32;
            for s in 0..3 {
                let max_var = icov[s][0].max(icov[s][3]).max(icov[s][5]);
                let cov = icov[s].map(|v| v as f32);
                let sc = 1.0f32 / (max_var as f32 + 0.0000125f32);
                let alt = principal_axis_3d(&cov, sc);
                slam += estimate_slam_to_line_sse_3d(&cov, alt[0], alt[1], alt[2]).0;
                axes[s] = int_axis(alt, [306, 601, 117]);
            }
            slam_sse[mi] = slam;

            let (mut low_dot, mut high_dot) = ([i32::MAX; 3], [i32::MIN; 3]);
            for i in 0..16 {
                let s = subset_of(i);
                let d = dot_axis(&px[i], &axes[s], i);
                low_dot[s] = low_dot[s].min(d);
                high_dot[s] = high_dot[s].max(d);
            }
            mode_low[mi] = low_dot.map(|v| (v & 15) as usize);
            mode_high[mi] = high_dot.map(|v| (v & 15) as usize);
        }

        for s in 0..3 {
            let mut spans = [0i32; 4];
            for c in 0..3 {
                spans[c] = px[mode_high[mi][s]][c] as i32 - px[mode_low[mi][s]][c] as i32;
            }
            quant_sse[mi] += if mi == 0 {
                analytical_quant_est_sse(16, 8, 3, &spans, UNIQUE_PBIT_DISCOUNT, mode_count[mi][s])
            } else {
                analytical_quant_est_sse(32, 4, 3, &spans, 1.0, mode_count[mi][s])
            };
        }
    }

    let est0 = quant_sse[0] + slam_sse[0];
    let est2 = quant_sse[1] + slam_sse[1];
    let q = 1.0f32 / 255.0f32;

    if est0 < est2 {
        *final_est = est0;
        if est0 >= sse_est_to_beat {
            return false;
        }
        let pi = pats[0];
        let part = &BC7_PARTITION3[pi as usize * 16..pi as usize * 16 + 16];
        let mut xl = [[0.0f32; 4]; 3];
        let mut xh = [[0.0f32; 4]; 3];
        for s in 0..3 {
            let (lc, hc) = (&px[mode_low[0][s]], &px[mode_high[0][s]]);
            for c in 0..3 {
                xl[s][c] = lc[c] as f32 * q;
                xh[s][c] = hc[c] as f32 * q;
            }
        }
        let mut e = Ep::<3>::default();
        let mut pbits = [0u32; 6];
        for s in 0..3 {
            let (el, eh, pb) = determine_unique_pbits(3, 4, &xl[s], &xh[s]);
            pbits[s * 2] = pb[0];
            pbits[s * 2 + 1] = pb[1];
            e.set(s, &el, &eh);
        }
        let mut w = [0u8; 16];
        eval_weights_mode0_rgb(px, &mut w, &e, &pbits, pi);
        let acc = accum_subsets::<3>(&w, &tabs.ls3, px, 3, |i| part[i] as usize);
        for s in 0..3 {
            let t = mode_totals[0][s];
            let Some((l, h)) = solve_subset(&acc[s], [t[0], t[1], t[2], 0], 3) else {
                continue;
            };
            for c in 0..3 {
                xl[s][c] = clampf(q * l[c], 0.0, 1.0);
                xh[s][c] = clampf(q * h[c], 0.0, 1.0);
            }
        }
        for s in 0..3 {
            let (el, eh, pb) = determine_unique_pbits(3, 4, &xl[s], &xh[s]);
            pbits[s * 2] = pb[0];
            pbits[s * 2 + 1] = pb[1];
            e.set(s, &el, &eh);
        }
        let sse = eval_weights_mode0_rgb(px, &mut w, &e, &pbits, pi);
        if let Some(a) = actual {
            *a = sse;
        }
        encode_mode0_rgb_block(block, pi, e.lr, e.lg, e.lb, e.hr, e.hg, e.hb, pbits, &w);
    } else {
        *final_est = est2;
        if est2 >= sse_est_to_beat {
            return false;
        }
        let pi = pats[1];
        let part = &BC7_PARTITION3[pi as usize * 16..pi as usize * 16 + 16];
        let mut e = Ep::<3>::default();
        for s in 0..3 {
            let (lc, hc) = (&px[mode_low[1][s]], &px[mode_high[1][s]]);
            e.lr[s] = to_5(lc[0] as i32) as u32;
            e.lg[s] = to_5(lc[1] as i32) as u32;
            e.lb[s] = to_5(lc[2] as i32) as u32;
            e.hr[s] = to_5(hc[0] as i32) as u32;
            e.hg[s] = to_5(hc[1] as i32) as u32;
            e.hb[s] = to_5(hc[2] as i32) as u32;
        }
        let mut w = [0u8; 16];
        eval_weights_mode2_rgb(px, &mut w, &e, pi);
        let acc = accum_subsets::<3>(&w, &tabs.ls2, px, 3, |i| part[i] as usize);
        for s in 0..3 {
            let t = mode_totals[1][s];
            let Some((l, h)) = solve_subset(&acc[s], [t[0], t[1], t[2], 0], 3) else {
                continue;
            };
            e.hr[s] = to_5_clamp(h[0]) as u32;
            e.lr[s] = to_5_clamp(l[0]) as u32;
            e.hg[s] = to_5_clamp(h[1]) as u32;
            e.lg[s] = to_5_clamp(l[1]) as u32;
            e.hb[s] = to_5_clamp(h[2]) as u32;
            e.lb[s] = to_5_clamp(l[2]) as u32;
        }
        let sse = eval_weights_mode2_rgb(px, &mut w, &e, pi);
        if let Some(a) = actual {
            *a = sse;
        }
        encode_mode2_rgb_block(block, pi, e.lr, e.lg, e.lb, e.hr, e.hg, e.hb, &w);
    }
    true
}

/// `pack_mode4_or_5`: dual plane with channel `dp_chan` (0-3) rotated into alpha.
fn pack_mode4_or_5(
    block: &mut Block,
    orig: &[Px; 16],
    dp_chan: usize,
    sse_est_to_beat: f32,
    final_est: &mut f32,
    actual: Option<&mut u32>,
) -> bool {
    let tabs = tables();
    let mut px = *orig;
    if dp_chan != 3 {
        for p in px.iter_mut() {
            p.swap(dp_chan, 3);
        }
    }
    let px = &px;

    let mut total = [0i32; 4];
    let (mut min_a, mut max_a) = (255i32, 0i32);
    for p in px.iter() {
        for c in 0..4 {
            total[c] += p[c] as i32;
        }
        min_a = min_a.min(p[3] as i32);
        max_a = max_a.max(p[3] as i32);
    }
    let mean = [
        (total[0] + 8) >> 4,
        (total[1] + 8) >> 4,
        (total[2] + 8) >> 4,
    ];
    let mut icov = [0i32; 6];
    for p in px.iter() {
        add_cov3(&mut icov, p, mean);
    }
    let cov3 = icov.map(|v| v as f32);
    let max_var3 = icov[0].max(icov[3]).max(icov[5]);
    let sc3 = if max_var3 != 0 {
        1.0f32 / max_var3 as f32
    } else {
        0.0
    };
    let alt = principal_axis_3d(&cov3, sc3);
    let rgb_slam = estimate_slam_to_line_sse_3d(&cov3, alt[0], alt[1], alt[2]).0;
    let axis = int_axis(alt, [306, 601, 117]);
    let (low_c, high_c) = minmax_on_axis(px, &axis);

    let (lo, hi) = (&px[low_c], &px[high_c]);
    let rgb_spans = [
        hi[0] as i32 - lo[0] as i32,
        hi[1] as i32 - lo[1] as i32,
        hi[2] as i32 - lo[2] as i32,
        0,
    ];
    let a_span = max_a - min_a;
    let spw = 1.0f32;

    let m4_rgb3 = analytical_quant_est_sse(32, 8, 3, &rgb_spans, 1.0, 16);
    let m4_a2 = analytical_quant_est_sse1(64, 4, a_span, spw, 1.0, 16);
    let m4_rgb2 = analytical_quant_est_sse(32, 4, 3, &rgb_spans, 1.0, 16);
    let m4_a3 = analytical_quant_est_sse1(64, 8, a_span, spw, 1.0, 16);
    let total_m4_rgb3_a2 = rgb_slam + m4_rgb3 + m4_a2;
    let total_m4_rgb2_a3 = rgb_slam + m4_rgb2 + m4_a3;
    let m5_rgb = analytical_quant_est_sse(128, 4, 3, &rgb_spans, 1.0, 16);
    let m5_a = analytical_quant_est_sse1(256, 4, a_span, spw, 1.0, 16);
    let total_m5 = rgb_slam + m5_rgb + m5_a;
    let rot = (dp_chan as u32 + 1) & 3;
    let rgb_t = [total[0] as f32, total[1] as f32, total[2] as f32];

    if total_m5 < minf(total_m4_rgb3_a2, total_m4_rgb2_a3) {
        *final_est = total_m5;
        if total_m5 >= sse_est_to_beat {
            return false;
        }
        let mut l = [to_7(lo[0] as i32), to_7(lo[1] as i32), to_7(lo[2] as i32)];
        let mut h = [to_7(hi[0] as i32), to_7(hi[1] as i32), to_7(hi[2] as i32)];
        let (mut la, mut ha) = (min_a, max_a);
        let mut w0 = [0u8; 16];
        let mut sse = eval_weights_rgb_1subset(px, &mut w0, l, h, from_7, &BC7_WEIGHTS2);
        if let Some((xl, xh)) = ls_endpoints(&w0, &tabs.ls2, px, rgb_t) {
            for c in 0..3 {
                l[c] = fast_roundf_int(xl[c] * (127.0f32 / 255.0f32));
                h[c] = fast_roundf_int(xh[c] * (127.0f32 / 255.0f32));
            }
            sse = eval_weights_rgb_1subset(px, &mut w0, l, h, from_7, &BC7_WEIGHTS2);
        }
        let mut w1 = [0u8; 16];
        let mut a_sse = eval_weights_a(px, &mut w1, la, ha, &BC7_WEIGHTS2);
        if let Some((nal, nah)) = ls_endpoints_1d(&w1, &tabs.ls2, px, 3, total[3] as f32) {
            la = fast_roundf_int(nal);
            ha = fast_roundf_int(nah);
            a_sse = eval_weights_a(px, &mut w1, la, ha, &BC7_WEIGHTS2);
        }
        if let Some(a) = actual {
            *a = sse.wrapping_add(a_sse);
        }
        encode_mode5_rgba_block(
            block,
            l[0] as u32,
            l[1] as u32,
            l[2] as u32,
            la as u32,
            h[0] as u32,
            h[1] as u32,
            h[2] as u32,
            ha as u32,
            &w0,
            &w1,
            rot,
        );
    } else {
        // Mode 4: index_flag 1 = 3-bit RGB / 2-bit A, 0 = 2-bit RGB / 3-bit A.
        let rgb3 = total_m4_rgb3_a2 < total_m4_rgb2_a3;
        let est = if rgb3 {
            total_m4_rgb3_a2
        } else {
            total_m4_rgb2_a3
        };
        *final_est = est;
        if est >= sse_est_to_beat {
            return false;
        }
        let (rgb_w, rgb_tab, a_w, a_tab): (&[u32], &[[f32; 4]], &[u32], &[[f32; 4]]) = if rgb3 {
            (&BC7_WEIGHTS3, &tabs.ls3, &BC7_WEIGHTS2, &tabs.ls2)
        } else {
            (&BC7_WEIGHTS2, &tabs.ls2, &BC7_WEIGHTS3, &tabs.ls3)
        };
        let mut l = [to_5(lo[0] as i32), to_5(lo[1] as i32), to_5(lo[2] as i32)];
        let mut h = [to_5(hi[0] as i32), to_5(hi[1] as i32), to_5(hi[2] as i32)];
        let (mut la, mut ha) = (to_6(min_a), to_6(max_a));
        let mut w0 = [0u8; 16];
        let mut sse = eval_weights_rgb_1subset(px, &mut w0, l, h, from_5, rgb_w);
        if let Some((xl, xh)) = ls_endpoints(&w0, rgb_tab, px, rgb_t) {
            for c in 0..3 {
                l[c] = fast_roundf_int(xl[c] * (31.0f32 / 255.0f32));
                h[c] = fast_roundf_int(xh[c] * (31.0f32 / 255.0f32));
            }
            sse = eval_weights_rgb_1subset(px, &mut w0, l, h, from_5, rgb_w);
        }
        let mut w1 = [0u8; 16];
        let mut a_sse = eval_weights_a(px, &mut w1, from_6(la), from_6(ha), a_w);
        if let Some((nal, nah)) = ls_endpoints_1d(&w1, a_tab, px, 3, total[3] as f32) {
            la = fast_roundf_int(nal * (63.0f32 / 255.0f32));
            ha = fast_roundf_int(nah * (63.0f32 / 255.0f32));
            a_sse = eval_weights_a(px, &mut w1, from_6(la), from_6(ha), a_w);
        }
        if let Some(a) = actual {
            *a = sse.wrapping_add(a_sse);
        }
        encode_mode4_rgba_block(
            block,
            l[0] as u32,
            l[1] as u32,
            l[2] as u32,
            la as u32,
            h[0] as u32,
            h[1] as u32,
            h[2] as u32,
            ha as u32,
            &w0,
            &w1,
            rot,
            rgb3 as u32,
        );
    }
    true
}

/// `pack_mode7_rgba`: 2 subsets, RGBA.
#[allow(clippy::too_many_arguments)]
fn pack_mode7_rgba(
    block: &mut Block,
    px: &[Px; 16],
    axis: [f32; 4],
    mean: [i32; 4],
    sse_est_to_beat: f32,
    flags: u32,
    final_est: &mut f32,
    actual: Option<&mut u32>,
) -> bool {
    let tabs = tables();
    let mut desired = 0u32;
    for i in 0..16 {
        let r = (px[i][0] as i32 - mean[0]) as f32;
        let g = (px[i][1] as i32 - mean[1]) as f32;
        let b = (px[i][2] as i32 - mean[2]) as f32;
        let a = (px[i][3] as i32 - mean[3]) as f32;
        let subset = (r * axis[0] + g * axis[1] + b * axis[2] + a * axis[3] > 0.0) as u32;
        desired |= subset << i;
    }
    let (pat_index, pat_bits) = pick_2subset_pattern(desired, flags);
    let subset_of = |i: usize| ((pat_bits >> i) & 1) as usize;
    let (totals, counts, means) = subset_stats::<2>(px, 4, subset_of);

    let mut icov4 = [[0i32; 10]; 2];
    for i in 0..16 {
        let s = subset_of(i);
        add_cov4(&mut icov4[s], &px[i], means[s]);
    }
    let mut axes = [[0i32; 4]; 2];
    let mut slam = 0.0f32;
    for s in 0..2 {
        let max_var = icov4[s][0]
            .max(icov4[s][4])
            .max(icov4[s][7])
            .max(icov4[s][9]);
        let cov4 = icov4[s].map(|v| v as f32);
        let sc4 = if max_var != 0 {
            1.0f32 / max_var as f32
        } else {
            0.0
        };
        let v0 = mul_cov4(
            &cov4,
            [sc4 * cov4[0], sc4 * cov4[4], sc4 * cov4[7], sc4 * cov4[9]],
        );
        let v1 = mul_cov4(&cov4, v0);
        slam += estimate_slam_to_line_sse_4d(&cov4, v1[0], v1[1], v1[2], v1[3]).0;
        axes[s] = int_axis(v1, [256; 4]);
    }
    let (mut low_dot, mut high_dot) = ([i32::MAX; 2], [i32::MIN; 2]);
    for i in 0..16 {
        let s = subset_of(i);
        let d = dot_axis(&px[i], &axes[s], i);
        low_dot[s] = low_dot[s].min(d);
        high_dot[s] = high_dot[s].max(d);
    }
    let low_c = low_dot.map(|v| (v & 15) as usize);
    let high_c = high_dot.map(|v| (v & 15) as usize);

    let pbit_opt = flags & PACK_BC7_FLAG_PBIT_OPT != 0;
    let mut quant = 0.0f32;
    for s in 0..2 {
        let mut spans = [0i32; 4];
        for c in 0..4 {
            spans[c] = px[high_c[s]][c] as i32 - px[low_c[s]][c] as i32;
        }
        let d = if pbit_opt { UNIQUE_PBIT_DISCOUNT } else { 1.0 };
        quant += analytical_quant_est_sse(32, 4, 4, &spans, d, counts[s]);
    }
    let est = slam + quant;
    *final_est = est;
    if est >= sse_est_to_beat {
        return false;
    }

    let q = 1.0f32 / 255.0f32;
    let mut e = Ep::<2>::default();
    let (mut la, mut ha) = ([0u32; 2], [0u32; 2]);
    let mut pbits = [0u32; 4];
    for s in 0..2 {
        let (lc, hc) = (&px[low_c[s]], &px[high_c[s]]);
        if pbit_opt {
            let sxl = lc.map(|v| v as f32 * q);
            let sxh = hc.map(|v| v as f32 * q);
            let (mn, mx, pb) = determine_unique_pbits(4, 5, &sxl, &sxh);
            pbits[s * 2] = pb[0];
            pbits[s * 2 + 1] = pb[1];
            e.set(s, &mn, &mx);
            la[s] = mn[3] as u32;
            ha[s] = mx[3] as u32;
        } else {
            let lp = (lc[3] >= 129) as i32;
            let hp = (hc[3] >= 129) as i32;
            pbits[s * 2] = lp as u32;
            pbits[s * 2 + 1] = hp as u32;
            e.lr[s] = to_5p(lc[0] as i32, lp) as u32;
            e.lg[s] = to_5p(lc[1] as i32, lp) as u32;
            e.lb[s] = to_5p(lc[2] as i32, lp) as u32;
            la[s] = to_5p(lc[3] as i32, lp) as u32;
            e.hr[s] = to_5p(hc[0] as i32, hp) as u32;
            e.hg[s] = to_5p(hc[1] as i32, hp) as u32;
            e.hb[s] = to_5p(hc[2] as i32, hp) as u32;
            ha[s] = to_5p(hc[3] as i32, hp) as u32;
        }
    }
    let mut w = [0u8; 16];
    eval_weights_mode7_rgba(px, &mut w, &e, &la, &ha, &pbits, pat_bits);
    let acc = accum_subsets::<2>(&w, &tabs.ls2, px, 4, subset_of);
    for s in 0..2 {
        let Some((sl, sh)) = solve_subset(&acc[s], totals[s], 4) else {
            continue;
        };
        if pbit_opt {
            let sxl = sl.map(|v| clampf(v * q, 0.0, 1.0));
            let sxh = sh.map(|v| clampf(v * q, 0.0, 1.0));
            let (mn, mx, pb) = determine_unique_pbits(4, 5, &sxl, &sxh);
            pbits[s * 2] = pb[0];
            pbits[s * 2 + 1] = pb[1];
            e.set(s, &mn, &mx);
            la[s] = mn[3] as u32;
            ha[s] = mx[3] as u32;
        } else {
            let lp = (sl[3] >= 129.0) as i32;
            let hp = (sh[3] >= 129.0) as i32;
            pbits[s * 2] = lp as u32;
            pbits[s * 2 + 1] = hp as u32;
            e.lr[s] = to_5_clamp_p(sl[0], lp) as u32;
            e.lg[s] = to_5_clamp_p(sl[1], lp) as u32;
            e.lb[s] = to_5_clamp_p(sl[2], lp) as u32;
            la[s] = to_5_clamp_p(sl[3], lp) as u32;
            e.hr[s] = to_5_clamp_p(sh[0], hp) as u32;
            e.hg[s] = to_5_clamp_p(sh[1], hp) as u32;
            e.hb[s] = to_5_clamp_p(sh[2], hp) as u32;
            ha[s] = to_5_clamp_p(sh[3], hp) as u32;
        }
    }
    let sse = eval_weights_mode7_rgba(px, &mut w, &e, &la, &ha, &pbits, pat_bits);
    if let Some(a) = actual {
        *a = sse;
    }
    encode_mode7_rgba_block(
        block, pat_index, e.lr, e.lg, e.lb, la, e.hr, e.hg, e.hb, ha, pbits, &w,
    );
    true
}

// ---------------------------------------------------------------------------------------------
// Top level encoders.

fn is_solid(px: &[Px; 16]) -> bool {
    px[0] == px[15] && px[1..15].iter().all(|p| *p == px[0])
}

/// RGB block statistics shared by the RGB encoders.
struct RgbStats {
    total: [i32; 3],
    spans: [i32; 4],
    mean: [i32; 3],
    icov: [i32; 6],
    max_var: i32,
}

fn rgb_stats(px: &[Px; 16]) -> RgbStats {
    let mut total = [0i32; 3];
    let (mut mn, mut mx) = ([255i32; 3], [0i32; 3]);
    for p in px.iter() {
        for c in 0..3 {
            let v = p[c] as i32;
            total[c] += v;
            mn[c] = mn[c].min(v);
            mx[c] = mx[c].max(v);
        }
    }
    let mean = total.map(|t| (t + 8) >> 4);
    let mut icov = [0i32; 6];
    for p in px.iter() {
        add_cov3(&mut icov, p, mean);
    }
    RgbStats {
        total,
        spans: [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 0],
        mean,
        icov,
        max_var: icov[0].max(icov[3]).max(icov[5]),
    }
}

/// Picks the most decorrelated RGB channel (indices of the rr/rg/rb/gg/gb/bb entries given).
fn rgb_dual_plane_chan(icov: &[i32], idx: [usize; 6], thresh: f32) -> i32 {
    let (rr, rg, rb, gg, gb, bb) = (
        icov[idx[0]],
        icov[idx[1]],
        icov[idx[2]],
        icov[idx[3]],
        icov[idx[4]],
        icov[idx[5]],
    );
    let (has_r, has_g, has_b) = (rr > 16, gg > 16, bb > 16);
    let active = has_r as u32 + has_g as u32 + has_b as u32;
    if active < 2 {
        return -1;
    }
    let (r_var, g_var, b_var) = (rr as f32, gg as f32, bb as f32);
    let rg_corr = if has_r && has_g {
        (rg as f32 / (r_var * g_var).sqrt()).abs()
    } else {
        1.0
    };
    let rb_corr = if has_r && has_b {
        (rb as f32 / (r_var * b_var).sqrt()).abs()
    } else {
        1.0
    };
    let gb_corr = if has_g && has_b {
        (gb as f32 / (g_var * b_var).sqrt()).abs()
    } else {
        1.0
    };
    let min_p = minf(minf(rg_corr, rb_corr), gb_corr);
    if min_p >= thresh {
        return -1;
    }
    if active == 2 {
        if !has_r { 1 } else { 0 }
    } else if rg_corr < gb_corr && rb_corr < gb_corr {
        0
    } else if rg_corr < rb_corr && gb_corr < rb_corr {
        1
    } else {
        2
    }
}

/// Mode 6 endpoints for RGB blocks from two pixels (p-bit optimized or forced to 1).
fn mode6_rgb_endpoints(lo: &Px, hi: &Px, flags: u32) -> ([i32; 3], [i32; 3], i32, i32) {
    if flags & PACK_BC7_FLAG_PBIT_OPT_MODE6 != 0 {
        let q = 1.0f32 / 255.0f32;
        let sxl = [lo[0] as f32 * q, lo[1] as f32 * q, lo[2] as f32 * q, 0.0];
        let sxh = [hi[0] as f32 * q, hi[1] as f32 * q, hi[2] as f32 * q, 0.0];
        let (mn, mx, pb) = determine_unique_pbits(3, 7, &sxl, &sxh);
        (
            [mn[0] as i32, mn[1] as i32, mn[2] as i32],
            [mx[0] as i32, mx[1] as i32, mx[2] as i32],
            pb[0] as i32,
            pb[1] as i32,
        )
    } else {
        let l = [
            to_7p(lo[0] as i32, 1),
            to_7p(lo[1] as i32, 1),
            to_7p(lo[2] as i32, 1),
        ];
        let h = [
            to_7p(hi[0] as i32, 1),
            to_7p(hi[1] as i32, 1),
            to_7p(hi[2] as i32, 1),
        ];
        (l, h, 1, 1)
    }
}

/// Mode 6 endpoints for RGB blocks from LS results (0-255 floats).
fn mode6_rgb_endpoints_ls(
    xl: &[f32; 4],
    xh: &[f32; 4],
    flags: u32,
) -> ([i32; 3], [i32; 3], i32, i32) {
    if flags & PACK_BC7_FLAG_PBIT_OPT_MODE6 != 0 {
        let q = 1.0f32 / 255.0f32;
        let sxl = [xl[0] * q, xl[1] * q, xl[2] * q, 0.0];
        let sxh = [xh[0] * q, xh[1] * q, xh[2] * q, 0.0];
        let (mn, mx, pb) = determine_unique_pbits(3, 7, &sxl, &sxh);
        (
            [mn[0] as i32, mn[1] as i32, mn[2] as i32],
            [mx[0] as i32, mx[1] as i32, mx[2] as i32],
            pb[0] as i32,
            pb[1] as i32,
        )
    } else {
        (
            [to_7f(xl[0], 1), to_7f(xl[1], 1), to_7f(xl[2], 1)],
            [to_7f(xh[0], 1), to_7f(xh[1], 1), to_7f(xh[2], 1)],
            1,
            1,
        )
    }
}

/// Trivial mode 6 endpoint pixels: min/max luma-ish projection (ties: lowest index).
fn trivial_minmax(px: &[Px; 16], use_alpha: bool) -> (usize, usize) {
    let (mut low, mut high) = (i32::MAX, 0i32);
    for (i, p) in px.iter().enumerate() {
        let mut y = 32 * p[0] as i32 + 64 * p[1] as i32 + 16 * p[2] as i32;
        if use_alpha {
            y += 64 * p[3] as i32;
        }
        y += i as i32;
        low = low.min(y);
        high = high.max(y);
    }
    ((low & 0xF) as usize, (high & 0xF) as usize)
}

#[allow(clippy::too_many_arguments)]
fn encode_mode6_rgb(block: &mut Block, l: [i32; 3], h: [i32; 3], p0: i32, p1: i32, w: &[u8; 16]) {
    encode_mode6_rgba_block(
        block,
        l[0] as u32,
        l[1] as u32,
        l[2] as u32,
        127,
        p0 as u32,
        h[0] as u32,
        h[1] as u32,
        h[2] as u32,
        127,
        p1 as u32,
        w,
    );
}

/// `fast_pack_bc7_rgb_analytical`: assumes the block has no alpha.
pub fn fast_pack_bc7_rgb_analytical(block: &mut Block, px: &[Px; 16], flags: u32) {
    if is_solid(px) {
        pack_mode5_solid(block, &px[0]);
        return;
    }
    let st = rgb_stats(px);
    if st.max_var == 0 {
        pack_mode5_solid(block, &px[0]);
        return;
    }
    let mut dp_chan = -1i32;
    if flags & PACK_BC7_FLAG_USE_DUAL_PLANE_RGB != 0 && st.max_var >= DP_BLOCK_VAR_THRESH {
        dp_chan = rgb_dual_plane_chan(&st.icov, [0, 1, 2, 3, 4, 5], STRONG_CORR_THRESH);
    }

    if flags & PACK_BC7_FLAG_USE_TRIVIAL_MODE6 != 0
        && dp_chan == -1
        && st.max_var < TRIVIAL_BLOCK_THRESH_RGB
    {
        let (lo, hi) = trivial_minmax(px, false);
        let (l, h, p0, p1) = mode6_rgb_endpoints(&px[lo], &px[hi], flags);
        let mut w = [0u8; 16];
        eval_weights_mode6_rgb(px, &mut w, l, h, p0, p1);
        encode_mode6_rgb(block, l, h, p0, p1, &w);
        return;
    }

    let cov = st.icov.map(|v| v as f32);
    let sc = if st.max_var != 0 {
        1.0f32 / st.max_var as f32
    } else {
        0.0
    };
    let alt = principal_axis_3d(&cov, sc);

    let need_sse_estimates = flags & PACK_BC7_FLAG_USE_2_SUBSETS_RGB != 0 || dp_chan >= 0;
    let (mode6_slam, mode6_ortho_ratio) = if need_sse_estimates {
        estimate_slam_to_line_sse_3d(&cov, alt[0], alt[1], alt[2])
    } else {
        (0.0, 0.0)
    };
    let mode6_sse_est = if need_sse_estimates {
        mode6_slam + analytical_quant_est_sse(128, 16, 3, &st.spans, 1.0, 16)
    } else {
        0.0
    };

    if flags & PACK_BC7_FLAG_USE_2_SUBSETS_RGB != 0
        && st.max_var >= MIN_BLOCK_MAX_VAR_23SUBSETS
        && mode6_ortho_ratio > ORTHO_RATIO_23SUBSET_RATIO_THRESH
        && mode6_slam >= HIGH_ORTHO_ENERGY_THRESH
    {
        if flags & PACK_BC7_FLAG_USE_3_SUBSETS_RGB != 0 && st.max_var >= MIN_BLOCK_MAX_VAR_3SUBSETS
        {
            let mut est02 = 1e9f32;
            if pack_mode0_or_2_rgb(
                block,
                px,
                alt,
                st.mean,
                mode6_sse_est,
                flags,
                &mut est02,
                None,
            ) {
                let mut est13 = 1e9f32;
                let mut temp = [0u8; 16];
                if pack_mode1_or_3_rgb(&mut temp, px, alt, st.mean, est02, flags, &mut est13, None)
                {
                    *block = temp;
                }
                return;
            }
        }
        let mut dummy = 0.0f32;
        if pack_mode1_or_3_rgb(
            block,
            px,
            alt,
            st.mean,
            mode6_sse_est,
            flags,
            &mut dummy,
            None,
        ) {
            return;
        }
    }

    if dp_chan >= 0 {
        let mut dummy = 0.0f32;
        if pack_mode4_or_5(block, px, dp_chan as usize, mode6_sse_est, &mut dummy, None) {
            return;
        }
    }

    let tabs = tables();
    let axis = int_axis(alt, [306, 601, 117]);
    let (lo, hi) = minmax_on_axis(px, &axis);
    let (mut l, mut h, mut p0, mut p1) = mode6_rgb_endpoints(&px[lo], &px[hi], flags);
    let mut w = [0u8; 16];
    eval_weights_mode6_rgb(px, &mut w, l, h, p0, p1);
    let t = st.total.map(|v| v as f32);
    if let Some((xl, xh)) = ls_endpoints(&w, &tabs.ls4, px, t) {
        (l, h, p0, p1) = mode6_rgb_endpoints_ls(&xl, &xh, flags);
        eval_weights_mode6_rgb(px, &mut w, l, h, p0, p1);
    }
    encode_mode6_rgb(block, l, h, p0, p1, &w);
}

/// `fast_pack_bc7_rgb_partial_analytical`: like the analytical encoder but resolves the mode
/// by actual SSE. Returns the block's SSE.
pub fn fast_pack_bc7_rgb_partial_analytical(block: &mut Block, px: &[Px; 16], flags: u32) -> u32 {
    if is_solid(px) {
        pack_mode5_solid(block, &px[0]);
        return 0;
    }
    let st = rgb_stats(px);
    if st.max_var == 0 {
        pack_mode5_solid(block, &px[0]);
        return 0;
    }
    let mut dp_chan = -1i32;
    let non_analytical = flags & PACK_BC7_FLAG_NON_ANALYTICAL_RGB != 0;
    if flags & PACK_BC7_FLAG_USE_DUAL_PLANE_RGB != 0
        && ((!non_analytical && st.max_var >= DP_BLOCK_VAR_THRESH)
            || (non_analytical && st.max_var >= DP_BLOCK_VAR_THRESH / 2))
    {
        let thresh = if non_analytical {
            0.999
        } else {
            STRONG_CORR_THRESH
        };
        dp_chan = rgb_dual_plane_chan(&st.icov, [0, 1, 2, 3, 4, 5], thresh);
    }

    if flags & PACK_BC7_FLAG_USE_TRIVIAL_MODE6 != 0
        && dp_chan == -1
        && st.max_var < TRIVIAL_BLOCK_THRESH_RGB
    {
        let (lo, hi) = trivial_minmax(px, false);
        let (l, h, p0, p1) = mode6_rgb_endpoints(&px[lo], &px[hi], flags);
        let mut w = [0u8; 16];
        let sse = eval_weights_mode6_rgb(px, &mut w, l, h, p0, p1);
        encode_mode6_rgb(block, l, h, p0, p1, &w);
        return sse;
    }

    let tabs = tables();
    let cov = st.icov.map(|v| v as f32);
    let sc = if st.max_var != 0 {
        1.0f32 / st.max_var as f32
    } else {
        0.0
    };
    let alt = principal_axis_3d(&cov, sc);
    let axis = int_axis(alt, [306, 601, 117]);
    let (lo, hi) = minmax_on_axis(px, &axis);
    let (mut l, mut h, mut p0, mut p1) = mode6_rgb_endpoints(&px[lo], &px[hi], flags);
    let mut w = [0u8; 16];
    let mut mode6_sse = eval_weights_mode6_rgb(px, &mut w, l, h, p0, p1);
    if mode6_sse != 0 {
        let t = st.total.map(|v| v as f32);
        if let Some((xl, xh)) = ls_endpoints(&w, &tabs.ls4, px, t) {
            let (tl, th, tp0, tp1) = mode6_rgb_endpoints_ls(&xl, &xh, flags);
            let mut tw = [0u8; 16];
            let ls_sse = eval_weights_mode6_rgb(px, &mut tw, tl, th, tp0, tp1);
            if ls_sse < mode6_sse {
                mode6_sse = ls_sse;
                w = tw;
                (l, h, p0, p1) = (tl, th, tp0, tp1);
            }
        }
    }

    let (mut sse02, mut sse13, mut sse45) = (u32::MAX, u32::MAX, u32::MAX);
    let (mut blk02, mut blk13, mut blk45) = ([0u8; 16], [0u8; 16], [0u8; 16]);
    let mut dummy = 0.0f32;
    if mode6_sse != 0 {
        if non_analytical {
            if flags & PACK_BC7_FLAG_USE_2_SUBSETS_RGB != 0 {
                if flags & PACK_BC7_FLAG_USE_3_SUBSETS_RGB != 0 {
                    pack_mode0_or_2_rgb(
                        &mut blk02,
                        px,
                        alt,
                        st.mean,
                        1e9,
                        flags,
                        &mut dummy,
                        Some(&mut sse02),
                    );
                }
                pack_mode1_or_3_rgb(
                    &mut blk13,
                    px,
                    alt,
                    st.mean,
                    1e9,
                    flags,
                    &mut dummy,
                    Some(&mut sse13),
                );
            }
            if flags & PACK_BC7_FLAG_USE_DUAL_PLANE_RGB != 0 {
                let ch = if dp_chan >= 0 { dp_chan as usize } else { 1 };
                pack_mode4_or_5(&mut blk45, px, ch, 1e9, &mut dummy, Some(&mut sse45));
            }
        } else {
            let (slam, ratio) = estimate_slam_to_line_sse_3d(&cov, alt[0], alt[1], alt[2]);
            if flags & PACK_BC7_FLAG_USE_2_SUBSETS_RGB != 0
                && st.max_var >= MIN_BLOCK_MAX_VAR_23SUBSETS
                && ratio > ORTHO_RATIO_23SUBSET_RATIO_THRESH
                && slam >= HIGH_ORTHO_ENERGY_THRESH
            {
                if flags & PACK_BC7_FLAG_USE_3_SUBSETS_RGB != 0
                    && st.max_var >= MIN_BLOCK_MAX_VAR_3SUBSETS
                {
                    pack_mode0_or_2_rgb(
                        &mut blk02,
                        px,
                        alt,
                        st.mean,
                        1e9,
                        flags,
                        &mut dummy,
                        Some(&mut sse02),
                    );
                }
                pack_mode1_or_3_rgb(
                    &mut blk13,
                    px,
                    alt,
                    st.mean,
                    1e9,
                    flags,
                    &mut dummy,
                    Some(&mut sse13),
                );
            }
            if dp_chan >= 0 {
                pack_mode4_or_5(
                    &mut blk45,
                    px,
                    dp_chan as usize,
                    1e9,
                    &mut dummy,
                    Some(&mut sse45),
                );
            }
        }
    }

    let best = mode6_sse.min(sse02).min(sse13).min(sse45);
    if mode6_sse == best {
        encode_mode6_rgb(block, l, h, p0, p1, &w);
    } else if sse45 != u32::MAX && best == sse45 {
        *block = blk45;
    } else if sse02 != u32::MAX && best == sse02 {
        *block = blk02;
    } else if sse13 != u32::MAX && best == sse13 {
        *block = blk13;
    }
    best
}

/// RGBA block statistics shared by the RGBA encoders.
struct RgbaStats {
    total: [i32; 4],
    spans: [i32; 4],
    mean: [i32; 4],
    icov4: [i32; 10],
    max_var4: i32,
}

fn rgba_stats(px: &[Px; 16]) -> RgbaStats {
    let mut total = [0i32; 4];
    let (mut mn, mut mx) = ([255i32; 4], [0i32; 4]);
    for p in px.iter() {
        for c in 0..4 {
            let v = p[c] as i32;
            total[c] += v;
            mn[c] = mn[c].min(v);
            mx[c] = mx[c].max(v);
        }
    }
    let mean = total.map(|t| (t + 8) >> 4);
    let mut icov4 = [0i32; 10];
    for p in px.iter() {
        add_cov4(&mut icov4, p, mean);
    }
    RgbaStats {
        total,
        spans: [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], mx[3] - mn[3]],
        mean,
        icov4,
        max_var4: icov4[0].max(icov4[4]).max(icov4[7]).max(icov4[9]),
    }
}

/// Dual plane channel for RGBA blocks: alpha first, then the RGB decorrelation test.
fn rgba_dual_plane_chan(icov4: &[i32; 10], rgb_allowed: bool, rgb_thresh: f32) -> i32 {
    let (r_var, g_var, b_var, a_var) = (
        icov4[0] as f32,
        icov4[4] as f32,
        icov4[7] as f32,
        icov4[9] as f32,
    );
    if icov4[9] > 0 {
        let p03 = if icov4[0] != 0 {
            (icov4[3] as f32 / (r_var * a_var).sqrt()).abs()
        } else {
            1.0
        };
        let p13 = if icov4[4] != 0 {
            (icov4[6] as f32 / (g_var * a_var).sqrt()).abs()
        } else {
            1.0
        };
        let p23 = if icov4[7] != 0 {
            (icov4[8] as f32 / (b_var * a_var).sqrt()).abs()
        } else {
            1.0
        };
        if minf(minf(p03, p13), p23) < ALPHA_DECORR_THRESHOLD {
            return 3;
        }
    }
    if !rgb_allowed {
        return -1;
    }
    rgb_dual_plane_chan(icov4, [0, 1, 2, 4, 5, 7], rgb_thresh)
}

/// Power iteration (4 steps) for the RGBA principal axis.
fn principal_axis_4d(cov4: &[f32; 10], max_var4: i32) -> [f32; 4] {
    let sc4 = if max_var4 != 0 {
        1.0f32 / max_var4 as f32
    } else {
        0.0
    };
    let mut v = [sc4 * cov4[0], sc4 * cov4[4], sc4 * cov4[7], sc4 * cov4[9]];
    for _ in 0..4 {
        let mut x = mul_cov4(cov4, v);
        let t = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]).sqrt();
        if t > SMALL_FLOAT_VAL {
            let t = 1.0 / t;
            for c in x.iter_mut() {
                *c *= t;
            }
        } else {
            x = [0.25; 4];
        }
        v = x;
    }
    v
}

/// Mode 6 RGBA endpoints from two 0-255 colors (`opt`: p-bit optimizer, else threshold).
fn mode6_rgba_endpoints(
    lo: [f32; 4],
    hi: [f32; 4],
    flags: u32,
    from_pixels: bool,
) -> ([i32; 4], [i32; 4], i32, i32) {
    if flags & PACK_BC7_FLAG_PBIT_OPT_MODE6 != 0 {
        let q = 1.0f32 / 255.0f32;
        let (mn, mx, pb) = determine_unique_pbits(4, 7, &lo.map(|v| v * q), &hi.map(|v| v * q));
        (
            mn.map(|v| v as i32),
            mx.map(|v| v as i32),
            pb[0] as i32,
            pb[1] as i32,
        )
    } else if from_pixels {
        let p0 = (lo[3] > 128.0) as i32;
        let p1 = (hi[3] > 128.0) as i32;
        (
            lo.map(|v| to_7p(v as i32, p0)),
            hi.map(|v| to_7p(v as i32, p1)),
            p0,
            p1,
        )
    } else {
        let p0 = (lo[3] >= 129.0) as i32;
        let p1 = (hi[3] >= 129.0) as i32;
        (lo.map(|v| to_7f(v, p0)), hi.map(|v| to_7f(v, p1)), p0, p1)
    }
}

#[inline]
fn pxf(p: &Px) -> [f32; 4] {
    p.map(|v| v as f32)
}

fn encode_mode6_rgba(block: &mut Block, l: [i32; 4], h: [i32; 4], p0: i32, p1: i32, w: &[u8; 16]) {
    encode_mode6_rgba_block(
        block,
        l[0] as u32,
        l[1] as u32,
        l[2] as u32,
        l[3] as u32,
        p0 as u32,
        h[0] as u32,
        h[1] as u32,
        h[2] as u32,
        h[3] as u32,
        p1 as u32,
        w,
    );
}

/// `fast_pack_bc7_rgba_analytical`: assumes the block has alpha.
pub fn fast_pack_bc7_rgba_analytical(block: &mut Block, px: &[Px; 16], flags: u32) {
    if is_solid(px) {
        pack_mode5_solid(block, &px[0]);
        return;
    }
    let st = rgba_stats(px);
    let mut dp_chan = -1i32;
    if flags & PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA != 0 && st.max_var4 >= DP_BLOCK_VAR_THRESH_RGBA {
        let rgb_allowed = flags & PACK_BC7_FLAG_DISABLE_RGB_DUAL_PLANE == 0;
        dp_chan = rgba_dual_plane_chan(&st.icov4, rgb_allowed, STRONG_DECORR_THRESH_RGBA);
    }

    if flags & PACK_BC7_FLAG_USE_TRIVIAL_MODE6 != 0
        && dp_chan == -1
        && st.max_var4 < TRIVIAL_BLOCK_THRESH_RGBA
    {
        let (lo, hi) = trivial_minmax(px, true);
        let (l, h, p0, p1) = mode6_rgba_endpoints(pxf(&px[lo]), pxf(&px[hi]), flags, true);
        let mut w = [0u8; 16];
        eval_weights_mode6_rgba(px, &mut w, l, p0, h, p1);
        encode_mode6_rgba(block, l, h, p0, p1, &w);
        return;
    }

    let cov4 = st.icov4.map(|v| v as f32);
    let v = principal_axis_4d(&cov4, st.max_var4);
    let (mode6_slam, mode6_ortho_ratio) =
        estimate_slam_to_line_sse_4d(&cov4, v[0], v[1], v[2], v[3]);
    let mode6_sse_est = mode6_slam + analytical_quant_est_sse(128, 16, 4, &st.spans, 1.0, 16);

    let (mut est45, mut est7) = (1e9f32, 1e9f32);
    let mut blk45 = [0u8; 16];
    if dp_chan >= 0 {
        pack_mode4_or_5(
            &mut blk45,
            px,
            dp_chan as usize,
            mode6_sse_est,
            &mut est45,
            None,
        );
    }
    let mut blk7 = [0u8; 16];
    if flags & PACK_BC7_FLAG_USE_2_SUBSETS_RGBA != 0
        && st.max_var4 >= MIN_BLOCK_MAX_VAR_23SUBSETS_RGBA
        && mode6_ortho_ratio > ORTHO_RATIO_23SUBSET_RATIO_THRESH_RGBA
        && mode6_slam >= HIGH_ORTHO_ENERGY_THRESH_RGBA
    {
        pack_mode7_rgba(
            &mut blk7,
            px,
            v,
            st.mean,
            mode6_sse_est,
            flags,
            &mut est7,
            None,
        );
    }

    if est45 < est7 && est45 < mode6_sse_est {
        *block = blk45;
        return;
    } else if est7 < est45 && est7 < mode6_sse_est {
        *block = blk7;
        return;
    }

    let tabs = tables();
    let axis = int_axis(v, [256; 4]);
    let (lo, hi) = minmax_on_axis(px, &axis);
    let (mut l, mut h, mut p0, mut p1) =
        mode6_rgba_endpoints(pxf(&px[lo]), pxf(&px[hi]), flags, true);
    let mut w = [0u8; 16];
    eval_weights_mode6_rgba(px, &mut w, l, p0, h, p1);
    let t = st.total.map(|v| v as f32);
    if let Some((xl, xh)) = ls_endpoints(&w, &tabs.ls4, px, t) {
        (l, h, p0, p1) = mode6_rgba_endpoints(xl, xh, flags, false);
        eval_weights_mode6_rgba(px, &mut w, l, p0, h, p1);
    }
    encode_mode6_rgba(block, l, h, p0, p1, &w);
}

/// `fast_pack_bc7_rgba_partial_analytical`: returns the block's SSE.
pub fn fast_pack_bc7_rgba_partial_analytical(block: &mut Block, px: &[Px; 16], flags: u32) -> u32 {
    if is_solid(px) {
        pack_mode5_solid(block, &px[0]);
        return 0;
    }
    let st = rgba_stats(px);
    let mut dp_chan = -1i32;
    let non_analytical = flags & PACK_BC7_FLAG_NON_ANALYTICAL_RGBA != 0;
    if flags & PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA != 0
        && ((!non_analytical && st.max_var4 >= DP_BLOCK_VAR_THRESH_RGBA)
            || (non_analytical && st.max_var4 > 16))
    {
        let thresh = if non_analytical {
            0.999
        } else {
            STRONG_DECORR_THRESH_RGBA
        };
        dp_chan = rgba_dual_plane_chan(&st.icov4, true, thresh);
    }

    if flags & PACK_BC7_FLAG_USE_TRIVIAL_MODE6 != 0
        && dp_chan == -1
        && st.max_var4 < TRIVIAL_BLOCK_THRESH_RGBA
    {
        let (lo, hi) = trivial_minmax(px, true);
        let (l, h, p0, p1) = mode6_rgba_endpoints(pxf(&px[lo]), pxf(&px[hi]), flags, true);
        let mut w = [0u8; 16];
        let sse = eval_weights_mode6_rgba(px, &mut w, l, p0, h, p1);
        encode_mode6_rgba(block, l, h, p0, p1, &w);
        return sse;
    }

    let tabs = tables();
    let cov4 = st.icov4.map(|v| v as f32);
    let v = principal_axis_4d(&cov4, st.max_var4);
    let axis = int_axis(v, [256; 4]);
    let (lo, hi) = minmax_on_axis(px, &axis);
    let (mut l, mut h, mut p0, mut p1) =
        mode6_rgba_endpoints(pxf(&px[lo]), pxf(&px[hi]), flags, true);
    let mut w = [0u8; 16];
    let mut mode6_sse = eval_weights_mode6_rgba(px, &mut w, l, p0, h, p1);
    if mode6_sse != 0 {
        let t = st.total.map(|v| v as f32);
        if let Some((xl, xh)) = ls_endpoints(&w, &tabs.ls4, px, t) {
            let (tl, th, tp0, tp1) = mode6_rgba_endpoints(xl, xh, flags, false);
            let mut tw = [0u8; 16];
            let ls_sse = eval_weights_mode6_rgba(px, &mut tw, tl, tp0, th, tp1);
            if ls_sse < mode6_sse {
                mode6_sse = ls_sse;
                w = tw;
                (l, h, p0, p1) = (tl, th, tp0, tp1);
            }
        }
    }

    let (mut sse7, mut sse45) = (u32::MAX, u32::MAX);
    let (mut blk7, mut blk45) = ([0u8; 16], [0u8; 16]);
    let mut dummy = 0.0f32;
    if mode6_sse != 0 {
        if non_analytical {
            if flags & PACK_BC7_FLAG_USE_2_SUBSETS_RGBA != 0 {
                pack_mode7_rgba(
                    &mut blk7,
                    px,
                    v,
                    st.mean,
                    1e9,
                    flags,
                    &mut dummy,
                    Some(&mut sse7),
                );
            }
            if flags & PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA != 0 {
                let ch = if dp_chan >= 0 { dp_chan as usize } else { 3 };
                pack_mode4_or_5(&mut blk45, px, ch, 1e9, &mut dummy, Some(&mut sse45));
            }
        } else {
            let (slam, ratio) = estimate_slam_to_line_sse_4d(&cov4, v[0], v[1], v[2], v[3]);
            if flags & PACK_BC7_FLAG_USE_2_SUBSETS_RGBA != 0
                && st.max_var4 >= MIN_BLOCK_MAX_VAR_23SUBSETS_RGBA
                && ratio > ORTHO_RATIO_23SUBSET_RATIO_THRESH_RGBA
                && slam >= HIGH_ORTHO_ENERGY_THRESH_RGBA
            {
                pack_mode7_rgba(
                    &mut blk7,
                    px,
                    v,
                    st.mean,
                    1e9,
                    flags,
                    &mut dummy,
                    Some(&mut sse7),
                );
            }
            if dp_chan >= 0 {
                pack_mode4_or_5(
                    &mut blk45,
                    px,
                    dp_chan as usize,
                    1e9,
                    &mut dummy,
                    Some(&mut sse45),
                );
            }
        }
    }

    let best = mode6_sse.min(sse45).min(sse7);
    if sse45 != u32::MAX && best == sse45 {
        *block = blk45;
    } else if sse7 != u32::MAX && best == sse7 {
        *block = blk7;
    } else {
        encode_mode6_rgba(block, l, h, p0, p1, &w);
    }
    best
}

/// `fast_pack_bc7_auto_rgba`: dispatches to the RGB or RGBA encoder depending on alpha.
pub fn fast_pack_bc7_auto_rgba(block: &mut Block, px: &[Px; 16], flags: u32) -> u32 {
    if px.iter().any(|p| p[3] < 255) {
        if flags & PACK_BC7_FLAG_PARTIALLY_ANALYTICAL_RGBA != 0 {
            return fast_pack_bc7_rgba_partial_analytical(block, px, flags);
        }
        fast_pack_bc7_rgba_analytical(block, px, flags);
        return 0;
    }
    fast_pack_bc7_auto_rgb(block, px, flags)
}

/// `fast_pack_bc7_auto_rgb`: the source block must not have alpha.
pub fn fast_pack_bc7_auto_rgb(block: &mut Block, px: &[Px; 16], flags: u32) -> u32 {
    if flags & PACK_BC7_FLAG_PARTIALLY_ANALYTICAL_RGB != 0 {
        return fast_pack_bc7_rgb_partial_analytical(block, px, flags);
    }
    fast_pack_bc7_rgb_analytical(block, px, flags);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u32) -> u8 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (*seed >> 24) as u8
    }

    /// Random-ish blocks: mixes of gradients, noise and partial alpha.
    fn blocks() -> Vec<[Px; 16]> {
        let mut seed = 1u32;
        let mut out = Vec::new();
        for k in 0..2000u32 {
            let mut px = [[0u8; 4]; 16];
            let kind = k % 5;
            let (a, b) = (
                [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)],
                [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)],
            );
            for (i, p) in px.iter_mut().enumerate() {
                let t = i as u32 * 17;
                for c in 0..3 {
                    p[c] = match kind {
                        0 => lcg(&mut seed),
                        1 => ((a[c] as u32 * (255 - t) + b[c] as u32 * t) / 255) as u8,
                        2 => {
                            if (i ^ (i >> 2)) & 1 == 0 {
                                a[c]
                            } else {
                                b[c].wrapping_add(lcg(&mut seed) & 7)
                            }
                        }
                        _ => a[c].wrapping_add(lcg(&mut seed) & 15),
                    };
                }
                p[3] = match k % 3 {
                    0 => 255,
                    1 => lcg(&mut seed),
                    _ => {
                        if i < 8 {
                            255
                        } else {
                            128 + (lcg(&mut seed) & 63)
                        }
                    }
                };
            }
            out.push(px);
        }
        out
    }

    #[test]
    fn partial_analytical_reports_actual_sse() {
        for flags in [
            PACK_BC7_FLAG_DEFAULT_PARTIALLY_ANALYTICAL,
            PACK_BC7_FLAG_DEFAULT_NON_ANALYTICAL,
        ] {
            for px in blocks() {
                let mut b = [0u8; 16];
                let sse = fast_pack_bc7_auto_rgba(&mut b, &px, flags);
                assert_eq!(sse, calc_sse(&b, &px));
                let mut opaque = px;
                for p in opaque.iter_mut() {
                    p[3] = 255;
                }
                let sse =
                    fast_pack_bc7_auto_rgb(&mut b, &opaque, flags | PACK_BC7_FLAG_ASTC_COMPATIBLE);
                assert_eq!(sse, calc_sse(&b, &opaque));
            }
        }
    }

    #[test]
    fn analytical_blocks_decode() {
        for flags in [
            PACK_BC7_FLAG_DEFAULT,
            PACK_BC7_FLAG_DEFAULT_FAST,
            PACK_BC7_FLAG_DEFAULT & !PACK_BC7_FLAG_PBIT_OPT,
        ] {
            for px in blocks() {
                let mut b = [0u8; 16];
                fast_pack_bc7_auto_rgba(&mut b, &px, flags);
                assert_ne!(calc_sse(&b, &px), u32::MAX);
            }
        }
    }
}
