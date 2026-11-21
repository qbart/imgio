// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of basisu `bc7u` logical BC7 blocks (`basisu_transcoder.cpp`): `log_bc7_block`, physical
//! pack/unpack, pixel decode and the integer endpoint DPCM used by the XUBC7 decoder.

use super::bc7f::{
    encode_mode0_rgb_block, encode_mode1_rgb_block, encode_mode2_rgb_block, encode_mode3_rgb_block,
    encode_mode4_rgba_block, encode_mode5_rgba_block, encode_mode6_rgba_block,
    encode_mode7_rgba_block,
};
use super::uastc::uastc_tables;
use super::uastc_tables::{
    BC7_PARTITION2, BC7_PARTITION3, BC7_TABLE_ANCHOR_INDEX_SECOND_SUBSET,
    BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_1, BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_2, BC7_WEIGHTS2,
    BC7_WEIGHTS3, BC7_WEIGHTS4,
};

type Px = [u8; 4];

pub(crate) const BC7_NUM_SUBSETS: [u8; 8] = [3, 2, 3, 2, 1, 1, 1, 2];
pub(crate) const BC7_PARTITION_BITS: [u8; 8] = [4, 6, 6, 6, 0, 0, 0, 6];
pub(crate) const BC7_COLOR_INDEX_BITCOUNT: [u8; 8] = [3, 3, 2, 2, 2, 2, 4, 2];
pub(crate) const BC7_ALPHA_INDEX_BITCOUNT: [u8; 8] = [0, 0, 0, 0, 3, 2, 4, 2];

/// `endpoint_format`: (rgb bits, alpha bits, p-bits per subset).
pub(crate) const ENDPOINT_FORMATS: [(u8, u8, u8); 8] = [
    (4, 0, 2),
    (6, 0, 1),
    (5, 0, 0),
    (7, 0, 2),
    (5, 6, 0),
    (7, 8, 0),
    (7, 7, 2),
    (5, 5, 2),
];

/// `log_bc7_block`: a BC7 block's fields, unpacked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LogBc7Block {
    /// 0-7, -1 invalid.
    pub mode: i8,
    pub num_partitions: u8,
    pub pattern_bits: u8,
    pub pattern_index: u8,
    pub num_planes: u8,
    pub dp_rotation_index: u8,
    pub mode4_index_selector: u8,
    /// [rgb, a] bits.
    pub endpoint_bits: [u8; 2],
    /// [subset][l or h][component].
    pub endpoints: [[[u8; 4]; 2]; 3],
    pub weight_bits: [u8; 2],
    /// [plane][texel].
    pub weights: [[u8; 16]; 2],
    pub num_pbits: u8,
    pub shared_pbits: bool,
    pub pbits: [u8; 6],
}

impl LogBc7Block {
    #[inline]
    pub fn is_dual_plane(&self) -> bool {
        self.num_planes == 2
    }

    /// Channel interpolated on the scalar plane (dual plane only).
    #[inline]
    pub fn color_component_selector(&self) -> usize {
        ((self.dp_rotation_index as usize) + 3) & 3
    }

    /// Weight plane used by endpoint channel `c` (before the mode 4/5 rotation).
    #[inline]
    pub fn endpoint_channel_weight_plane(&self, c: usize) -> usize {
        if !self.is_dual_plane() {
            return 0;
        }
        let sel = (self.mode4_index_selector & 1) as usize;
        if c == 3 { 1 - sel } else { sel }
    }

    /// Number of endpoint components (3 or 4).
    #[inline]
    pub fn num_comps(&self) -> usize {
        if self.endpoint_bits[1] != 0 { 4 } else { 3 }
    }
}

/// `dequant_weight`: BC7 weight index to [0,64].
#[inline]
pub(crate) fn dequant_weight(w: u32, bits: u32) -> i32 {
    let w = w as usize;
    (match bits {
        2 => BC7_WEIGHTS2.get(w),
        3 => BC7_WEIGHTS3.get(w),
        4 => BC7_WEIGHTS4.get(w),
        _ => None,
    })
    .map_or(0, |&v| v as i32)
}

const fn build_weight_quant() -> [[u8; 65]; 3] {
    let mut t = [[0u8; 65]; 3];
    let mut b = 0;
    while b < 3 {
        let n = 1usize << (b + 2);
        let mut de = 0;
        while de <= 64 {
            let (mut best_idx, mut best_err) = (0usize, u32::MAX);
            let mut idx = 0;
            while idx < n {
                let dq = match b {
                    0 => BC7_WEIGHTS2[idx],
                    1 => BC7_WEIGHTS3[idx],
                    _ => BC7_WEIGHTS4[idx],
                } as i32;
                let err = (de as i32 - dq).unsigned_abs();
                if err < best_err {
                    best_err = err;
                    best_idx = idx;
                }
                idx += 1;
            }
            t[b][de] = best_idx as u8;
            de += 1;
        }
        b += 1;
    }
    t
}

/// `g_weight_quant`.
const WEIGHT_QUANT: [[u8; 65]; 3] = build_weight_quant();

/// `quant_weight`: [0,64] (clamped) to the nearest BC7 weight index, `bits` in [2,4].
#[inline]
pub(crate) fn quant_weight(val: i32, bits: u32) -> u8 {
    let v = val.clamp(0, 64) as usize;
    WEIGHT_QUANT
        .get((bits as usize).wrapping_sub(2))
        .map_or(0, |t| t[v])
}

#[inline]
fn bc7_dequant_p(val: u32, pbit: u32, val_bits: u32) -> u32 {
    let total_bits = val_bits + 1;
    let mut v = (val << 1) | pbit;
    v <<= 8 - total_bits;
    v |= v >> total_bits;
    v & 0xFF
}

#[inline]
fn bc7_dequant(val: u32, val_bits: u32) -> u32 {
    let mut v = val << (8 - val_bits);
    v |= v >> val_bits;
    v & 0xFF
}

#[inline]
fn bc7_interp(l: u32, h: u32, w: u32, bits: u32) -> u32 {
    let wt = dequant_weight(w, bits) as u32;
    match bits {
        2..=4 => (l * (64 - wt) + h * wt + 32) >> 6,
        _ => 0,
    }
}

/// `fetch_bits`: up to 8 bits from a physical block.
#[inline]
fn fetch_bits(p: &[u8; 16], num_bits: u32, bit_ofs: &mut u32) -> u8 {
    if num_bits == 0 {
        return 0;
    }
    if *bit_ofs + num_bits > 128 {
        return 0;
    }
    let byte_ofs = (*bit_ofs >> 3) as usize;
    let b0 = p[byte_ofs] as u32;
    let b1 = p[(byte_ofs + 1).min(15)] as u32;
    let b = (b0 | (b1 << 8)) >> (*bit_ofs & 7);
    *bit_ofs += num_bits;
    (b & ((1u32 << num_bits) - 1)) as u8
}

/// `unpack_bc7`: physical to logical block. False for an invalid mode byte.
pub(crate) fn unpack_phys(p: &[u8; 16], log: &mut LogBc7Block) -> bool {
    *log = LogBc7Block::default();
    if p[0] == 0 {
        log.mode = -1;
        return false;
    }
    let mode = p[0].trailing_zeros();
    log.mode = mode as i8;
    let mut ofs = mode + 1;
    match mode {
        0 | 2 => {
            log.num_partitions = 3;
            log.pattern_bits = if mode == 0 { 4 } else { 6 };
            log.num_planes = 1;
            log.endpoint_bits[0] = if mode == 0 { 4 } else { 5 };
            log.num_pbits = if mode == 0 { 6 } else { 0 };
            log.weight_bits[0] = if mode == 0 { 3 } else { 2 };
            log.pattern_index = fetch_bits(p, log.pattern_bits as u32, &mut ofs);
            for c in 0..3 {
                for s in 0..3 {
                    for e in 0..2 {
                        log.endpoints[s][e][c] =
                            fetch_bits(p, log.endpoint_bits[0] as u32, &mut ofs);
                    }
                }
            }
            for i in 0..log.num_pbits as usize {
                log.pbits[i] = fetch_bits(p, 1, &mut ofs);
            }
            let pi = log.pattern_index as usize;
            let (a1, a2) = (
                BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_1[pi] as usize,
                BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_2[pi] as usize,
            );
            let wb = log.weight_bits[0] as u32;
            for i in 0..16 {
                let n = if i == 0 || i == a1 || i == a2 {
                    wb - 1
                } else {
                    wb
                };
                log.weights[0][i] = fetch_bits(p, n, &mut ofs);
            }
        }
        1 | 3 | 7 => {
            log.num_partitions = 2;
            log.pattern_bits = 6;
            log.num_planes = 1;
            log.endpoint_bits[0] = match mode {
                7 => 5,
                1 => 6,
                _ => 7,
            };
            log.endpoint_bits[1] = if mode == 7 { 5 } else { 0 };
            log.num_pbits = if mode == 1 { 2 } else { 4 };
            log.shared_pbits = mode == 1;
            log.weight_bits[0] = if mode == 1 { 3 } else { 2 };
            log.pattern_index = fetch_bits(p, 6, &mut ofs);
            let num_comps = if mode == 7 { 4 } else { 3 };
            for c in 0..num_comps {
                for s in 0..2 {
                    for e in 0..2 {
                        log.endpoints[s][e][c] =
                            fetch_bits(p, log.endpoint_bits[0] as u32, &mut ofs);
                    }
                }
            }
            for i in 0..log.num_pbits as usize {
                log.pbits[i] = fetch_bits(p, 1, &mut ofs);
            }
            let anchor = BC7_TABLE_ANCHOR_INDEX_SECOND_SUBSET[log.pattern_index as usize] as usize;
            let wb = log.weight_bits[0] as u32;
            for i in 0..16 {
                let n = if i == 0 || i == anchor { wb - 1 } else { wb };
                log.weights[0][i] = fetch_bits(p, n, &mut ofs);
            }
        }
        4 | 5 => {
            log.num_partitions = 1;
            log.num_planes = 2;
            log.endpoint_bits = if mode == 4 { [5, 6] } else { [7, 8] };
            log.weight_bits = [2, if mode == 4 { 3 } else { 2 }];
            log.dp_rotation_index = fetch_bits(p, 2, &mut ofs);
            log.mode4_index_selector = if mode == 4 {
                fetch_bits(p, 1, &mut ofs)
            } else {
                0
            };
            for c in 0..4 {
                for e in 0..2 {
                    let n = log.endpoint_bits[(c == 3) as usize] as u32;
                    log.endpoints[0][e][c] = fetch_bits(p, n, &mut ofs);
                }
            }
            for plane in 0..2 {
                let wb = log.weight_bits[plane] as u32;
                for i in 0..16 {
                    let n = if i == 0 { wb - 1 } else { wb };
                    log.weights[plane][i] = fetch_bits(p, n, &mut ofs);
                }
            }
        }
        _ => {
            // mode 6
            log.num_partitions = 1;
            log.num_planes = 1;
            log.endpoint_bits = [7, 7];
            log.num_pbits = 2;
            log.weight_bits[0] = 4;
            for c in 0..4 {
                log.endpoints[0][0][c] = fetch_bits(p, 7, &mut ofs);
                log.endpoints[0][1][c] = fetch_bits(p, 7, &mut ofs);
            }
            for i in 0..2 {
                log.pbits[i] = fetch_bits(p, 1, &mut ofs);
            }
            for w in 0..16 {
                log.weights[0][w] = fetch_bits(p, if w == 0 { 3 } else { 4 }, &mut ofs);
            }
        }
    }
    true
}

/// `unpack_endpoints`: the dequantized 8-bit endpoints of `subset` (alpha 255 when absent).
pub(crate) fn unpack_endpoints(log: &LogBc7Block, subset: usize) -> [Px; 2] {
    let mut ep = [[0, 0, 0, 255], [0, 0, 0, 255]];
    let num_comps = log.num_comps();
    let subset = subset.min(2);
    for (e, out) in ep.iter_mut().enumerate() {
        for c in 0..num_comps {
            let bits = log.endpoint_bits[(c == 3) as usize] as u32;
            let v = log.endpoints[subset][e][c] as u32;
            out[c] = if log.num_pbits != 0 {
                let pi = if log.shared_pbits {
                    subset
                } else {
                    subset * 2 + e
                };
                bc7_dequant_p(v, log.pbits[pi.min(5)] as u32 & 1, bits)
            } else {
                bc7_dequant(v, bits)
            } as u8;
        }
    }
    ep
}

/// Quantizes an 8-bit value to a (bits+1)-bit lattice point with parity `p`.
#[inline]
fn quant_endpoint_pbit(v8: u32, p: i32, iscalep: i32) -> i32 {
    let k = (v8 as i32 * iscalep + 255 - 255 * p) / 510;
    (k * 2 + p).clamp(p, iscalep - 1 + p)
}

#[inline]
fn quant_endpoint(v8: u32, num_bits: u32) -> u32 {
    let maxv = (1u32 << num_bits) - 1;
    (v8 * maxv * 2 + 255) / 510
}

/// `determine_unique_pbits_int` / `determine_shared_pbits_int`.
fn determine_pbits_int(
    shared: bool,
    total_comps: usize,
    comp_bits: u32,
    xl: &Px,
    xh: &Px,
) -> ([Px; 2], [u8; 2]) {
    let total_bits = comp_bits + 1;
    let iscalep = (1i32 << total_bits) - 1;
    let (mut best_err0, mut best_err1) = (u64::MAX, u64::MAX);
    let mut best = [[0u8; 4]; 2];
    let mut best_pbits = [0u8; 2];
    for p in 0..2i32 {
        let (mut xmin, mut xmax, mut sl, mut sh) = ([0i32; 4], [0i32; 4], [0i32; 4], [0i32; 4]);
        for c in 0..4 {
            xmin[c] = quant_endpoint_pbit(xl[c] as u32, p, iscalep);
            xmax[c] = quant_endpoint_pbit(xh[c] as u32, p, iscalep);
            sl[c] = xmin[c] << (8 - total_bits);
            sl[c] |= sl[c] >> total_bits;
            sh[c] = xmax[c] << (8 - total_bits);
            sh[c] |= sh[c] >> total_bits;
        }
        let (mut err0, mut err1) = (0u64, 0u64);
        for i in 0..total_comps {
            let d0 = (sl[i] - xl[i] as i32) as i64;
            let d1 = (sh[i] - xh[i] as i32) as i64;
            err0 += (d0 * d0) as u64;
            err1 += (d1 * d1) as u64;
        }
        if shared {
            let err = err0 + err1;
            if err < best_err0 {
                best_err0 = err;
                best_pbits = [p as u8; 2];
                for j in 0..4 {
                    best[0][j] = (xmin[j] >> 1) as u8;
                    best[1][j] = (xmax[j] >> 1) as u8;
                }
            }
        } else {
            if err0 < best_err0 {
                best_err0 = err0;
                best_pbits[0] = p as u8;
                for j in 0..4 {
                    best[0][j] = (xmin[j] >> 1) as u8;
                }
            }
            if err1 < best_err1 {
                best_err1 = err1;
                best_pbits[1] = p as u8;
                for j in 0..4 {
                    best[1][j] = (xmax[j] >> 1) as u8;
                }
            }
        }
    }
    (best, best_pbits)
}

/// `pack_endpoints_int`: 8-bit endpoints into `mode`'s endpoint format (integer only).
fn pack_endpoints_int(mode: usize, src: &[Px; 2]) -> ([Px; 2], [u8; 2]) {
    let (rgb_bits, a_bits, pbits) = ENDPOINT_FORMATS[mode & 7];
    let num_comps = if a_bits != 0 { 4 } else { 3 };
    if pbits == 0 {
        let mut dst = [[0u8; 4]; 2];
        for e in 0..2 {
            for c in 0..4 {
                let nbits = if c == 3 { a_bits } else { rgb_bits } as u32;
                dst[e][c] = if nbits != 0 {
                    quant_endpoint(src[e][c] as u32, nbits) as u8
                } else {
                    0
                };
            }
        }
        (dst, [0, 0])
    } else {
        determine_pbits_int(pbits == 1, num_comps, rgb_bits as u32, &src[0], &src[1])
    }
}

/// `endpoint_dpcm` (decode direction): reconstructs `log`'s `subset` endpoints and p-bits from
/// residuals against `pred`'s `pred_subset` endpoints. `residuals` in RR GG BB AA order.
pub(crate) fn endpoint_dpcm_decode(
    pred: &LogBc7Block,
    pred_subset: usize,
    log: &mut LogBc7Block,
    subset: usize,
    residuals: &[u8; 8],
    res_pbits: &[u8; 2],
) {
    let mode = (log.mode as usize) & 7;
    let num_pbits = ENDPOINT_FORMATS[mode].2 as usize;
    let mut pe = unpack_endpoints(pred, pred_subset);
    if pred.is_dual_plane() {
        let ccs = pred.color_component_selector();
        for e in pe.iter_mut() {
            e.swap(ccs, 3);
        }
    }
    let ccs = log.color_component_selector();
    if log.is_dual_plane() {
        for e in pe.iter_mut() {
            e.swap(ccs, 3);
        }
    }
    let (packed, packed_pbits) = pack_endpoints_int(mode, &pe);
    let num_comps = log.num_comps();
    let (mut g_channel, mut a_channel) = (1usize, 3usize);
    if log.is_dual_plane() {
        a_channel = ccs;
        if ccs == 1 {
            g_channel = 3;
        }
    }
    let mut t = *residuals;
    for c in 0..num_comps {
        if c == g_channel || c == a_channel {
            continue;
        }
        t[c * 2] = t[c * 2].wrapping_add(t[g_channel * 2]);
        t[c * 2 + 1] = t[c * 2 + 1].wrapping_add(t[g_channel * 2 + 1]);
    }
    let subset = subset.min(2);
    for c in 0..num_comps {
        let bits = log.endpoint_bits[(c == 3) as usize] as u32;
        let mask = ((1u32 << bits) - 1) as u8;
        log.endpoints[subset][0][c] = t[c * 2].wrapping_add(packed[0][c]) & mask;
        log.endpoints[subset][1][c] = t[c * 2 + 1].wrapping_add(packed[1][c]) & mask;
    }
    for p in 0..num_pbits {
        if let Some(slot) = log.pbits.get_mut(subset * num_pbits + p) {
            *slot = (res_pbits[p].wrapping_add(packed_pbits[p])) & 1;
        }
    }
}

/// `init_log_blk`.
pub(crate) fn init_log_blk(log: &mut LogBc7Block, mode: u32) {
    let m = (mode & 7) as usize;
    let (rgb_bits, a_bits, pbits) = ENDPOINT_FORMATS[m];
    *log = LogBc7Block::default();
    log.mode = m as i8;
    log.num_partitions = BC7_NUM_SUBSETS[m];
    log.num_planes = if m == 4 || m == 5 { 2 } else { 1 };
    log.num_pbits = log.num_partitions * pbits;
    log.pattern_bits = BC7_PARTITION_BITS[m];
    log.endpoint_bits = [rgb_bits, a_bits];
    log.weight_bits[0] = BC7_COLOR_INDEX_BITCOUNT[m];
    log.weight_bits[1] = if log.num_planes == 2 {
        BC7_ALPHA_INDEX_BITCOUNT[m]
    } else {
        0
    };
    log.shared_pbits = pbits == 1;
}

/// `create_solid_blk`: mode 5 block decoding to `c` exactly.
pub(crate) fn create_solid_blk(log: &mut LogBc7Block, c: &Px) {
    init_log_blk(log, 5);
    let m5 = &uastc_tables().bc7_mode_5_optimal_endpoints;
    for ch in 0..3 {
        let e = &m5[c[ch] as usize];
        log.endpoints[0][0][ch] = e.lo;
        log.endpoints[0][1][ch] = e.hi;
    }
    log.endpoints[0][0][3] = c[3];
    log.endpoints[0][1][3] = c[3];
    log.weights[0] = [1; 16];
}

#[inline]
fn texel_subset(log: &LogBc7Block, i: usize) -> usize {
    let pi = log.pattern_index as usize * 16 + i;
    match log.num_partitions {
        2 => BC7_PARTITION2.get(pi).map_or(0, |&s| s as usize),
        3 => BC7_PARTITION3.get(pi).map_or(0, |&s| s as usize),
        _ => 0,
    }
}

/// Decodes texel `i` given the subset's unpacked endpoints.
#[inline]
fn decode_texel(log: &LogBc7Block, ep: &[Px; 2], i: usize) -> Px {
    let sel = (log.mode4_index_selector & 1) as usize;
    let mut res = [0u8, 0, 0, 255];
    for c in 0..3 {
        res[c] = bc7_interp(
            ep[0][c] as u32,
            ep[1][c] as u32,
            log.weights[sel][i] as u32,
            log.weight_bits[sel] as u32,
        ) as u8;
    }
    if log.num_comps() == 4 {
        let plane = if log.num_planes == 2 { 1 - sel } else { 0 };
        res[3] = bc7_interp(
            ep[0][3] as u32,
            ep[1][3] as u32,
            log.weights[plane][i] as u32,
            log.weight_bits[plane] as u32,
        ) as u8;
    }
    if log.dp_rotation_index != 0 {
        res.swap(3, ((log.dp_rotation_index - 1) & 3) as usize);
    }
    res
}

/// `unpack_bc7` (logical): decodes all 16 texels. False for an invalid block.
pub(crate) fn unpack_pixels(log: &LogBc7Block, px: &mut [Px; 16]) -> bool {
    if log.mode < 0 {
        *px = [[0; 4]; 16];
        return false;
    }
    let mut eps = [[[0u8; 4]; 2]; 3];
    for (s, e) in eps.iter_mut().enumerate().take(log.num_partitions as usize) {
        *e = unpack_endpoints(log, s);
    }
    for (i, p) in px.iter_mut().enumerate() {
        *p = decode_texel(log, &eps[texel_subset(log, i).min(2)], i);
    }
    true
}

/// `unpack_bc7_texel`: decodes texel (`x`, `y`).
pub(crate) fn unpack_texel(log: &LogBc7Block, x: usize, y: usize) -> Option<Px> {
    if log.mode < 0 {
        return None;
    }
    let i = (x + y * 4) & 15;
    let ep = unpack_endpoints(log, texel_subset(log, i));
    Some(decode_texel(log, &ep, i))
}

fn split3(log: &LogBc7Block, e: usize, c: usize) -> [u32; 3] {
    [0, 1, 2].map(|s| log.endpoints[s][e][c] as u32)
}

fn split2(log: &LogBc7Block, e: usize, c: usize) -> [u32; 2] {
    [0, 1].map(|s| log.endpoints[s][e][c] as u32)
}

/// `pack_bc7`: logical to physical (may swap endpoints / invert weights, lossless in pixels).
pub(crate) fn pack_bc7(log: &LogBc7Block, out: &mut [u8; 16]) -> bool {
    let pi = log.pattern_index as u32;
    let p = log.pbits.map(u32::from);
    let ep = |e: usize, c: usize| log.endpoints[0][e][c] as u32;
    match log.mode {
        0 => encode_mode0_rgb_block(
            out,
            pi & 15,
            split3(log, 0, 0),
            split3(log, 0, 1),
            split3(log, 0, 2),
            split3(log, 1, 0),
            split3(log, 1, 1),
            split3(log, 1, 2),
            p,
            &log.weights[0],
        ),
        2 => encode_mode2_rgb_block(
            out,
            pi & 63,
            split3(log, 0, 0),
            split3(log, 0, 1),
            split3(log, 0, 2),
            split3(log, 1, 0),
            split3(log, 1, 1),
            split3(log, 1, 2),
            &log.weights[0],
        ),
        1 => encode_mode1_rgb_block(
            out,
            pi & 63,
            split2(log, 0, 0),
            split2(log, 0, 1),
            split2(log, 0, 2),
            split2(log, 1, 0),
            split2(log, 1, 1),
            split2(log, 1, 2),
            p[0],
            p[1],
            &log.weights[0],
        ),
        3 => encode_mode3_rgb_block(
            out,
            pi & 63,
            split2(log, 0, 0),
            split2(log, 0, 1),
            split2(log, 0, 2),
            split2(log, 1, 0),
            split2(log, 1, 1),
            split2(log, 1, 2),
            [p[0], p[1], p[2], p[3]],
            &log.weights[0],
        ),
        4 => {
            let sel = (log.mode4_index_selector & 1) as usize;
            encode_mode4_rgba_block(
                out,
                ep(0, 0),
                ep(0, 1),
                ep(0, 2),
                ep(0, 3),
                ep(1, 0),
                ep(1, 1),
                ep(1, 2),
                ep(1, 3),
                &log.weights[sel],
                &log.weights[1 - sel],
                log.dp_rotation_index as u32,
                sel as u32,
            )
        }
        5 => encode_mode5_rgba_block(
            out,
            ep(0, 0),
            ep(0, 1),
            ep(0, 2),
            ep(0, 3),
            ep(1, 0),
            ep(1, 1),
            ep(1, 2),
            ep(1, 3),
            &log.weights[0],
            &log.weights[1],
            log.dp_rotation_index as u32,
        ),
        6 => encode_mode6_rgba_block(
            out,
            ep(0, 0),
            ep(0, 1),
            ep(0, 2),
            ep(0, 3),
            p[0],
            ep(1, 0),
            ep(1, 1),
            ep(1, 2),
            ep(1, 3),
            p[1],
            &log.weights[0],
        ),
        7 => encode_mode7_rgba_block(
            out,
            pi & 63,
            split2(log, 0, 0),
            split2(log, 0, 1),
            split2(log, 0, 2),
            split2(log, 0, 3),
            split2(log, 1, 0),
            split2(log, 1, 1),
            split2(log, 1, 2),
            split2(log, 1, 3),
            [p[0], p[1], p[2], p[3]],
            &log.weights[0],
        ),
        _ => {
            *out = [0; 16];
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phys_roundtrip_matches_pixels() {
        // A few hand-made blocks: unpack -> pack -> unpack is pixel exact.
        let blocks: [[u8; 16]; 3] = [
            [
                0x40, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55,
                0x66, 0x77,
            ],
            [
                0x02, 0x81, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55,
                0x66, 0x77,
            ],
            [
                0x10, 0xe1, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55,
                0x66, 0x77,
            ],
        ];
        for b in &blocks {
            let mut log = LogBc7Block::default();
            assert!(unpack_phys(b, &mut log));
            let (mut a, mut c) = ([[0u8; 4]; 16], [[0u8; 4]; 16]);
            assert!(unpack_pixels(&log, &mut a));
            let mut packed = [0u8; 16];
            assert!(pack_bc7(&log, &mut packed));
            assert!(crate::bc::bc7::unpack_bc7(&packed, &mut c));
            assert_eq!(a, c);
            let mut d = [[0u8; 4]; 16];
            assert!(crate::bc::bc7::unpack_bc7(b, &mut d));
            assert_eq!(a, d);
        }
    }
}
