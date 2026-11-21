// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of the basisu XBC7 stream decoder (`basisu_xbc7_decoder.h` / `.inl`): the tagged blob
//! container (Zstandard via `ruzstd`), per-stripe block decoding (commands, endpoint DPCM,
//! weight predictors, residual DPCM and the fixed point 4x4 DCT), and tiny-mip streams.
//!
//! The decoder is total: any malformed stream is `Error::InvalidData`. Stripes reference only
//! their own block rows, so they are decoded in parallel.

use crate::basis::ll::bc7u::{
    self, ENDPOINT_FORMATS, LogBc7Block, create_solid_blk, dequant_weight, init_log_blk,
    quant_weight,
};
use crate::buffer::Buffer;
use crate::{Error, Result};
use std::sync::Mutex;

// ---------------------------------------------------------------------------------------------
// fixed16_16 (Q15.16), release-build semantics: narrowing to i32 wraps.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Fx(i32);

const FX_ONE: i32 = 1 << 16;

/// `(x + half) >> bits`, rounding half away from zero.
#[inline]
fn rounded_rshift(x: i64, bits: u32) -> i64 {
    let half = 1i64 << (bits - 1);
    if x >= 0 {
        (x + half) >> bits
    } else {
        -((x.wrapping_neg() + half) >> bits)
    }
}

impl Fx {
    #[inline]
    fn from_int(x: i32) -> Fx {
        Fx((i64::from(x) * i64::from(FX_ONE)) as i32)
    }

    #[inline]
    fn add(self, b: Fx) -> Fx {
        Fx((i64::from(self.0) + i64::from(b.0)) as i32)
    }

    #[inline]
    fn sub(self, b: Fx) -> Fx {
        Fx((i64::from(self.0) - i64::from(b.0)) as i32)
    }

    #[inline]
    fn mul(self, b: Fx) -> Fx {
        Fx(rounded_rshift(i64::from(self.0) * i64::from(b.0), 16) as i32)
    }

    #[inline]
    fn div(self, b: Fx) -> Fx {
        let q = (i64::from(self.0) * (1i64 << 17))
            .checked_div(i64::from(b.0))
            .unwrap_or(0);
        Fx(rounded_rshift(q, 1) as i32)
    }

    #[inline]
    fn mul_int(self, s: i32) -> Fx {
        Fx((i64::from(self.0) * i64::from(s)) as i32)
    }

    /// `operator/(int32_t)`: rounded half away from zero.
    #[inline]
    fn div_int(self, s: i32) -> Fx {
        let (num, den) = (i64::from(self.0), i64::from(s));
        if den == 0 {
            return Fx(0);
        }
        let half = if den >= 0 { den / 2 } else { -den / 2 };
        let neg = (num < 0) != (den < 0);
        let q = (num.abs() + half) / den.abs();
        Fx((if neg { -q } else { q }) as i32)
    }

    #[inline]
    fn round_to_int(self) -> i32 {
        let w = i64::from(self.0);
        if w >= 0 {
            ((w + 32768) >> 16) as i32
        } else {
            -(((-w + 32768) >> 16) as i32)
        }
    }

    #[inline]
    fn mul_round_to_int(self, b: Fx) -> i32 {
        rounded_rshift(i64::from(self.0) * i64::from(b.0), 32) as i32
    }

    #[inline]
    fn mul_wide(self, b: Fx) -> i64 {
        i64::from(self.0) * i64::from(b.0)
    }

    #[inline]
    fn from_sum(s: i64) -> Fx {
        Fx(rounded_rshift(s, 16) as i32)
    }
}

/// Exact `floor(sqrt(x))` (`fixed_detail::isqrt_floor`).
fn isqrt_floor(mut x: u64) -> u64 {
    if x == 0 {
        return 0;
    }
    let mut r = 0u64;
    let mut bit = 1u64 << ((63 - x.leading_zeros()) & !1);
    while bit != 0 {
        let t = r + bit;
        let mask = 0u64.wrapping_sub(u64::from(x >= t));
        x -= t & mask;
        r = (r >> 1) + (bit & mask);
        bit >>= 2;
    }
    r
}

// ---------------------------------------------------------------------------------------------
// Weight grid DCT (decoder side of `xbc7_weight_grid_dct_fixed`).

/// `g_zigzag4x4_xy`: [index] = (x, y).
const ZIGZAG: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 1),
    (3, 2),
    (2, 3),
    (3, 3),
];

/// `g_base_4x4_quant` (raw Q15.16).
const BASE_4X4_QUANT: [i32; 16] = [
    65536, 229376, 1572864, 3342336, 229376, 786432, 2621440, 5111808, 1572864, 2621440, 4456448,
    6750208, 3342336, 5111808, 6750208, 7864320,
];

/// `g_scale_quant_steps_fixed` for 2/3/4-bit weights.
const SCALE_QUANT_STEPS: [i32; 3] = [88859, 81640, 75649];

/// `DEADZONE_ALPHA_FIXED` (0.5).
const DEADZONE_ALPHA: i64 = 32768;

/// Butterfly constants of the 4-point orthonormal DCT (Q15.16).
const DCT_HALF: Fx = Fx(32768);
const DCT_C1: Fx = Fx(42813);
const DCT_C3: Fx = Fx(17734);

fn inverse_ortho(y: [Fx; 4]) -> [Fx; 4] {
    let b0 = y[0].mul_wide(DCT_HALF) + y[2].mul_wide(DCT_HALF);
    let b1 = y[0].mul_wide(DCT_HALF) - y[2].mul_wide(DCT_HALF);
    let t0 = y[1].mul_wide(DCT_C1) + y[3].mul_wide(DCT_C3);
    let t1 = y[1].mul_wide(DCT_C3) - y[3].mul_wide(DCT_C1);
    [
        Fx::from_sum(b0 + t0),
        Fx::from_sum(b1 + t1),
        Fx::from_sum(b1 - t1),
        Fx::from_sum(b0 - t0),
    ]
}

/// `dct2fx::inverse_4x4`.
fn inverse_4x4(src: &[Fx; 16]) -> [Fx; 16] {
    let mut t = [Fx(0); 16];
    for v in 0..4 {
        let out = inverse_ortho([src[v], src[4 + v], src[8 + v], src[12 + v]]);
        for (x, o) in out.into_iter().enumerate() {
            t[x * 4 + v] = o;
        }
    }
    let mut dst = [Fx(0); 16];
    for x in 0..4 {
        let out = inverse_ortho([t[x * 4], t[x * 4 + 1], t[x * 4 + 2], t[x * 4 + 3]]);
        dst[x * 4..x * 4 + 4].copy_from_slice(&out);
    }
    dst
}

fn compute_quant_table(q: Fx, level_scale: Fx) -> [i32; 16] {
    let mut tab = [1i32; 16];
    if q >= Fx::from_int(100) {
        return tab;
    }
    for y in 0..4 {
        for x in (if y > 0 { y } else { 1 })..4 {
            let mut qs = Fx(BASE_4X4_QUANT[x + y * 4])
                .mul_round_to_int(level_scale)
                .max(1);
            if x + y == 1 {
                qs = qs.min(73);
            }
            tab[x + y * 4] = qs;
            tab[y + x * 4] = qs;
        }
    }
    tab
}

fn compute_level_scale(q: Fx, span_len: Fx, num_weight_bits: u32) -> Fx {
    let idx = match num_weight_bits {
        3 => 1,
        4 => 2,
        _ => 0,
    };
    let q = if q < Fx::from_int(1) {
        Fx::from_int(1)
    } else if q > Fx::from_int(100) {
        Fx::from_int(100)
    } else {
        q
    };
    let level_scale = if q < Fx::from_int(50) {
        Fx::from_int(5000).div(q)
    } else {
        Fx::from_int(200).sub(q.mul_int(2))
    };
    let level_scale = level_scale.div_int(100);
    let floor = Fx::from_int(14);
    let adaptive = Fx::from_int(64).div(if span_len > floor { span_len } else { floor });
    let adaptive = adaptive.mul(Fx(SCALE_QUANT_STEPS[idx]));
    level_scale.mul(adaptive)
}

/// `get_max_span_len`: largest endpoint span (Euclidean) of the plane, Q15.16.
fn max_span_len(log: &LogBc7Block, plane: usize) -> Fx {
    let mut max_ssq = 0u32;
    if log.is_dual_plane() {
        let [l, h] = bc7u::unpack_endpoints(log, 0);
        for c in 0..4 {
            if log.endpoint_channel_weight_plane(c) == plane {
                let d = h[c] as i32 - l[c] as i32;
                max_ssq += (d * d) as u32;
            }
        }
    } else {
        for s in 0..(log.num_partitions as usize).min(3) {
            let [l, h] = bc7u::unpack_endpoints(log, s);
            let ssq: u32 = (0..4)
                .map(|c| {
                    let d = h[c] as i32 - l[c] as i32;
                    (d * d) as u32
                })
                .sum();
            max_ssq = max_ssq.max(ssq);
        }
    }
    let x = u64::from(max_ssq) << 32;
    let mut f = isqrt_floor(x);
    f += u64::from(x - f * f > f);
    Fx(f as i32)
}

#[inline]
fn sat_raw(raw: i64) -> Fx {
    let lim = 2048i64 * i64::from(FX_ONE);
    Fx(raw.clamp(-lim, lim) as i32)
}

fn dequant_deadzone(q: i32, l: i32, x: usize, y: usize) -> Fx {
    if (x == 1 && y == 0) || (x == 0 && y == 1) {
        return sat_raw(i64::from(q) * i64::from(l) * i64::from(FX_ONE));
    }
    if q == 0 || l <= 0 {
        return Fx(0);
    }
    let aq = i64::from(q).abs();
    let mag = DEADZONE_ALPHA * i64::from(l) + aq * i64::from(l) * i64::from(FX_ONE);
    sat_raw(if q < 0 { -mag } else { mag })
}

/// One AC symbol: (zero run, coefficient or `i16::MAX` for end of block).
type Ac = (u32, i32);

/// `xbc7_weight_grid_dct_fixed::inverse`: reconstructs `plane`'s weights.
fn dct_inverse(
    global_q: Fx,
    plane: usize,
    preds: Option<&[i32; 16]>,
    dc: i32,
    acs: &[Ac],
    log: &mut LogBc7Block,
) -> bool {
    let bits = u32::from(log.weight_bits[plane]);
    if !(2..=4).contains(&bits) {
        return false;
    }
    let span = max_span_len(log, plane);
    let level = compute_level_scale(global_q, span, bits);
    let tab = compute_quant_table(global_q, level);
    let mut dw = [Fx(0); 16];
    // Weight-grid DC is uniformly quantized by 2^(6 - weight bits).
    dw[0] = Fx::from_int(dc * (1 << (6 - bits)));
    let mut zig = 1u32;
    for &(run, coeff) in acs {
        if run + zig > 16 {
            return false;
        }
        zig += run;
        if zig >= 16 {
            break;
        }
        if coeff == i32::from(i16::MAX) {
            return false;
        }
        let (x, y) = ZIGZAG[zig as usize];
        let idx = x + y * 4;
        dw[idx] = dequant_deadzone(coeff, tab[idx], x, y);
        zig += 1;
    }
    let idct = inverse_4x4(&dw);
    for i in 0..16 {
        let pred = preds.map_or(0, |p| p[i]);
        let v = idct[i].add(Fx::from_int(pred)).round_to_int().clamp(0, 64);
        log.weights[plane][i] = quant_weight(v, bits);
    }
    true
}

// ---------------------------------------------------------------------------------------------
// Causal block access and weight predictors.

/// Blocks of one stripe (rows `first_row..end_row`), the reference tile.
struct Tile<'a> {
    blks: &'a [LogBc7Block],
    nbx: i32,
    first_row: i32,
    end_row: i32,
}

impl Tile<'_> {
    #[inline]
    fn get(&self, x: i32, y: i32) -> Option<&LogBc7Block> {
        if x < 0 || x >= self.nbx || y < self.first_row || y >= self.end_row {
            return None;
        }
        self.blks
            .get(((y - self.first_row) * self.nbx + x) as usize)
    }
}

const CAND_ABSOLUTE: u32 = 0;
const CAND_LEFT_EDGE: u32 = 1;
const CAND_UPPER_EDGE: u32 = 2;
const CAND_LU_BLEND: u32 = 3;
const CAND_REFLECT_LEFT: u32 = 4;
const CAND_REFLECT_UPPER: u32 = 5;
const CAND_LU_AVG: u32 = 6;
const CAND_LU_BLEND_STRONG: u32 = 7;
const CAND_GRADIENT: u32 = 8;
const CAND_GRADIENT_DAMPED: u32 = 9;
const CAND_DIAG_AVG: u32 = 10;
const CAND_DIAG_EDGE_BLEND: u32 = 11;
const CAND_UPPER_DIAG_EDGE_BLEND: u32 = 12;
const CAND_MED: u32 = 13;
const CAND_GAB: u32 = 14;
const CAND_PLANE_FIT: u32 = 15;
const CAND_DDL: u32 = 16;
const CAND_DDR: u32 = 17;
const CAND_FIRST_XY_DELTA: u32 = 18;
const TOTAL_CANDIDATES: u32 = CAND_FIRST_XY_DELTA + 32;

/// `g_xbc7_xy_deltas`: causal block references.
const XY_DELTAS: [(i8, i8); 32] = [
    (-1, 0),
    (-2, 0),
    (-3, 0),
    (-4, 0),
    (3, -1),
    (2, -1),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-2, -1),
    (-3, -1),
    (-4, -1),
    (3, -2),
    (2, -2),
    (1, -2),
    (0, -2),
    (-1, -2),
    (-2, -2),
    (-3, -2),
    (-4, -2),
    (3, -3),
    (2, -3),
    (1, -3),
    (0, -3),
    (-1, -3),
    (-2, -3),
    (-3, -3),
    (-4, -3),
    (3, -4),
    (2, -4),
    (1, -4),
    (0, -4),
];

#[inline]
fn ixy(x: usize, y: usize) -> usize {
    x + y * 4
}

/// Dequantized weight `w` of plane `plane` (plane 0 for single plane blocks).
#[inline]
fn fetch_w(b: &LogBc7Block, plane: usize, w: usize) -> i32 {
    let sp = if b.is_dual_plane() { plane } else { 0 };
    dequant_weight(u32::from(b.weights[sp][w]), u32::from(b.weight_bits[sp]))
}

/// `eval_weight_predictor`: the 16 weight predictions of candidate `cand` with amplitude code
/// `amp` at block (`bx`, `by`). None if a needed block is unavailable.
fn eval_weight_predictor(
    cand: u32,
    amp: u32,
    bx: i32,
    by: i32,
    tile: &Tile,
    p: usize,
) -> Option<[i32; 16]> {
    let ld = tile.get(bx - 1, by - 1);
    let rd = tile.get(bx + 1, by - 1);
    let up = tile.get(bx, by - 1);
    let left = tile.get(bx - 1, by);

    let cand_blk = if cand >= CAND_FIRST_XY_DELTA {
        let (dx, dy) = *XY_DELTAS.get((cand - CAND_FIRST_XY_DELTA) as usize)?;
        tile.get(bx + i32::from(dx), by + i32::from(dy))
    } else {
        match cand {
            CAND_LEFT_EDGE | CAND_REFLECT_LEFT => left,
            CAND_UPPER_EDGE | CAND_REFLECT_UPPER => up,
            CAND_LU_BLEND | CAND_LU_AVG | CAND_LU_BLEND_STRONG | CAND_PLANE_FIT => {
                both2(up, left).and(left)
            }
            CAND_GRADIENT | CAND_GRADIENT_DAMPED | CAND_MED | CAND_GAB | CAND_DDR => {
                both2(both2(up, ld), left).and(left)
            }
            CAND_DIAG_AVG | CAND_DIAG_EDGE_BLEND => both2(rd, ld).and(ld),
            CAND_UPPER_DIAG_EDGE_BLEND => both2(both2(up, rd), ld).and(ld),
            CAND_DDL => both2(rd, up).and(up),
            _ => None,
        }
    }?;

    let orig: [i32; 16] = std::array::from_fn(|w| fetch_w(cand_blk, p, w));
    let mut wp = orig;
    // Neighbor edges (only fetched by candidates that verified the block exists).
    let upper_edge = |b: Option<&LogBc7Block>| -> Option<[i32; 4]> {
        let b = b?;
        Some(std::array::from_fn(|x| fetch_w(b, p, ixy(x, 3))))
    };

    match cand {
        CAND_LEFT_EDGE => {
            for y in 0..4 {
                for x in 0..4 {
                    wp[ixy(x, y)] = orig[ixy(3, y)];
                }
            }
        }
        CAND_UPPER_EDGE => {
            for y in 0..4 {
                for x in 0..4 {
                    wp[ixy(x, y)] = orig[ixy(x, 3)];
                }
            }
        }
        CAND_LU_BLEND | CAND_LU_AVG | CAND_LU_BLEND_STRONG => {
            let ue = upper_edge(up)?;
            for y in 0..4 {
                let lv = orig[ixy(3, y)];
                for x in 0..4 {
                    let uv = ue[x];
                    wp[ixy(x, y)] = match cand {
                        CAND_LU_BLEND => {
                            let (wl, wu) = (4 - x as i32, 4 - y as i32);
                            let den = wl + wu;
                            (wl * lv + wu * uv + (den >> 1)) / den
                        }
                        CAND_LU_AVG => (lv + uv + 1) >> 1,
                        _ => {
                            let (dx, dy) = (4 - x as i32, 4 - y as i32);
                            let (wl, wu) = (dx * dx, dy * dy);
                            let den = wl + wu;
                            (wl * lv + wu * uv + (den >> 1)) / den
                        }
                    };
                }
            }
        }
        CAND_REFLECT_LEFT => {
            for y in 0..4 {
                for x in 0..4 {
                    wp[ixy(x, y)] = orig[ixy(3 - x, y)];
                }
            }
        }
        CAND_REFLECT_UPPER => {
            for y in 0..4 {
                for x in 0..4 {
                    wp[ixy(x, y)] = orig[ixy(x, 3 - y)];
                }
            }
        }
        CAND_GRADIENT | CAND_GRADIENT_DAMPED | CAND_MED | CAND_GAB => {
            let ue = upper_edge(up)?;
            let corner = fetch_w(ld?, p, ixy(3, 3));
            for y in 0..4 {
                let lv = orig[ixy(3, y)];
                for x in 0..4 {
                    let uv = ue[x];
                    wp[ixy(x, y)] = match cand {
                        CAND_GRADIENT => (lv + uv - corner).clamp(0, 64),
                        CAND_GRADIENT_DAMPED => {
                            let grad = (lv + uv - corner).clamp(0, 64);
                            let (wl, wu) = (4 - x as i32, 4 - y as i32);
                            let den = wl + wu;
                            let blend7 = (wl * lv + wu * uv + (den >> 1)) / den;
                            (grad + blend7 + 1) >> 1
                        }
                        CAND_MED => {
                            let (mn, mx) = (lv.min(uv), lv.max(uv));
                            let pred = if corner >= mx {
                                mn
                            } else if corner <= mn {
                                mx
                            } else {
                                lv + uv - corner
                            };
                            pred.clamp(0, 64)
                        }
                        _ => {
                            let wl = (lv - corner).abs() + 1;
                            let wu = (uv - corner).abs() + 1;
                            let den = wl + wu;
                            (wl * lv + wu * uv + (den >> 1)) / den
                        }
                    };
                }
            }
        }
        CAND_DIAG_AVG => {
            let r = rd?;
            for w in 0..16 {
                wp[w] = (orig[w] + fetch_w(r, p, w) + 1) >> 1;
            }
        }
        CAND_DIAG_EDGE_BLEND => {
            let r = rd?;
            for y in 0..4 {
                let lv = orig[ixy(3, y)];
                let rv = fetch_w(r, p, ixy(0, y));
                for x in 0..4 {
                    let xi = x as i32;
                    wp[ixy(x, y)] = ((3 - xi) * lv + xi * rv + 1) / 3;
                }
            }
        }
        CAND_UPPER_DIAG_EDGE_BLEND => {
            let ue = upper_edge(up)?;
            let r = rd?;
            for y in 0..4 {
                let ldr = orig[ixy(3, y)];
                let rdl = fetch_w(r, p, ixy(0, y));
                for x in 0..4 {
                    let xi = x as i32;
                    let diag = ((3 - xi) * ldr + xi * rdl + 1) / 3;
                    let (wu, wd) = (4 - y as i32, 1 + y as i32);
                    let den = wu + wd;
                    wp[ixy(x, y)] = (wu * ue[x] + wd * diag + (den >> 1)) / den;
                }
            }
        }
        CAND_PLANE_FIT => {
            let ue = upper_edge(up)?;
            let le: [i32; 4] = std::array::from_fn(|y| orig[ixy(3, y)]);
            let sum_u: i32 = ue.iter().sum();
            let sum_l: i32 = le.iter().sum();
            let gx10 = -3 * ue[0] - ue[1] + ue[2] + 3 * ue[3];
            let gy10 = -3 * le[0] - le[1] + le[2] + 3 * le[3];
            let base = 5 * (sum_u + sum_l);
            for y in 0..4 {
                for x in 0..4 {
                    let num = base + gx10 * (4 * x as i32 - 1) + gy10 * (4 * y as i32 - 1);
                    let t = num + 20;
                    let pred = if t >= 0 { t / 40 } else { -((-t + 39) / 40) };
                    wp[ixy(x, y)] = pred.clamp(0, 64);
                }
            }
        }
        CAND_DDL => {
            let u = up?;
            let r = rd?;
            let mut t = [0i32; 8];
            for x in 0..4 {
                t[x] = fetch_w(u, p, ixy(x, 3));
                t[4 + x] = fetch_w(r, p, ixy(x, 3));
            }
            for y in 0..4 {
                for x in 0..4 {
                    let d = x + y;
                    wp[ixy(x, y)] = if d == 6 {
                        (t[6] + 3 * t[7] + 2) >> 2
                    } else {
                        (t[d] + 2 * t[d + 1] + t[d + 2] + 2) >> 2
                    };
                }
            }
        }
        CAND_DDR => {
            let mut a = [0i32; 9];
            for y in 0..4 {
                a[3 - y] = orig[ixy(3, y)];
            }
            a[4] = fetch_w(ld?, p, ixy(3, 3));
            let ue = upper_edge(up)?;
            a[5..9].copy_from_slice(&ue);
            for y in 0..4 {
                for x in 0..4 {
                    let d = 4 + x - y;
                    wp[ixy(x, y)] = (a[d - 1] + 2 * a[d] + a[d + 1] + 2) >> 2;
                }
            }
        }
        _ => {}
    }

    if amp != 0 {
        let mean = (wp.iter().sum::<i32>() + 8) >> 4;
        for w in wp.iter_mut() {
            *w = match amp {
                1 => (2 * mean - *w).clamp(0, 64),
                2 => (*w + mean + 1) >> 1,
                _ => ((2 * mean - *w).clamp(0, 64) + mean + 1) >> 1,
            };
        }
    }
    Some(wp)
}

// ---------------------------------------------------------------------------------------------
// Blob container.

/// Every malformed stream error.
const BAD: Error = Error::InvalidData;

const MAGIC_BEGIN: u8 = 0xB7;
const MAGIC_END: u8 = 0x6A;
const MAX_IDS: usize = 128;
const MAX_TOTAL_UNCOMP: u64 = 1 << 30;

const BLOB_HEADER: usize = 0;
const BLOB_COMMANDS: usize = 1;
const BLOB_CONFIG: usize = 2;
const BLOB_PARTITION2: usize = 3;
const BLOB_PARTITION3: usize = 4;
const BLOB_WEIGHT_PREDICTORS: usize = 5;
const BLOB_DC_COEFFS_SMALL: usize = 6;
const BLOB_AC_COEFFS: usize = 8;
const BLOB_COEFF_SIGNS: usize = 9;
const BLOB_PBITS: usize = 10;
const BLOB_EP_DELTA_FINE_R: usize = 11;
const BLOB_EP_DELTA_COARSE_R: usize = 15;
const BLOB_EP_RAW: usize = 19;
const BLOB_EP_BLOCK_INDEX: usize = 20;
const BLOB_RAW_WEIGHT_BITS: usize = 21;
const BLOB_SOLID_RGBA_DELTAS: usize = 22;
const BLOB_DPCM_WEIGHT_RESID2: usize = 23;
const BLOB_STRIPE_SEEK_TABLE: usize = 26;

const MAX_STRIPES: usize = 16;
const MAX_DIM: u32 = 16384;

/// Where a blob's bytes live.
#[derive(Clone, Copy, Default)]
enum Loc {
    #[default]
    None,
    Raw(usize, usize),
    Arena(usize, usize),
}

/// `blob_stream_reader`: validated directory, compressed blobs inflated into one arena.
struct Blobs<'a> {
    data: &'a [u8],
    arena: Buffer,
    loc: [Loc; MAX_IDS],
}

/// Bounds-checked LEB128 (at most 5 bytes / 32 bits).
fn read_varint(data: &[u8], ofs: &mut usize) -> Option<u32> {
    let mut v = 0u32;
    let mut shift = 0;
    while shift < 35 {
        let b = *data.get(*ofs)?;
        *ofs += 1;
        if shift == 28 && b > 0x0F {
            return None;
        }
        v |= u32::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return Some(v);
        }
        shift += 7;
    }
    None
}

impl<'a> Blobs<'a> {
    fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() < 3 || data[0] != MAGIC_BEGIN {
            return Err(BAD);
        }
        let num_blobs = data[1] as usize;
        let mut ofs = 2usize;
        // (id, uncompressed size, stored size, data offset)
        let mut entries = [(0usize, 0usize, 0usize, 0usize); 255];
        let mut seen = [false; MAX_IDS];
        let mut total_arena = 0u64;
        for e in entries.iter_mut().take(num_blobs) {
            let id_flag = *data.get(ofs).ok_or(BAD)?;
            ofs += 1;
            let id = (id_flag & 0x7F) as usize;
            let compressed = id_flag & 0x80 != 0;
            let uncomp = read_varint(data, &mut ofs).ok_or(BAD)? as usize;
            let mut stored = 0usize;
            if compressed {
                stored = read_varint(data, &mut ofs).ok_or(BAD)? as usize;
                if stored == 0 {
                    return Err(BAD);
                }
            }
            let data_ofs = ofs;
            if uncomp == 0 || seen[id] {
                return Err(BAD);
            }
            let stored_bytes = if stored != 0 { stored } else { uncomp };
            if ofs as u64 + stored_bytes as u64 > data.len() as u64 {
                return Err(BAD);
            }
            ofs += stored_bytes;
            if stored != 0 {
                if stored >= uncomp {
                    return Err(BAD);
                }
                total_arena += uncomp as u64;
                if total_arena > MAX_TOTAL_UNCOMP {
                    return Err(BAD);
                }
            }
            seen[id] = true;
            *e = (id, uncomp, stored, data_ofs);
        }
        if ofs + 1 != data.len() || data[ofs] != MAGIC_END {
            return Err(BAD);
        }
        let mut arena = crate::buffer::zeroed(total_arena as usize)?;
        let mut loc = [Loc::None; MAX_IDS];
        let mut arena_ofs = 0usize;
        for &(id, uncomp, stored, data_ofs) in entries.iter().take(num_blobs) {
            if stored == 0 {
                loc[id] = Loc::Raw(data_ofs, uncomp);
            } else {
                let src = &data[data_ofs..data_ofs + stored];
                let dst = &mut arena[arena_ofs..arena_ofs + uncomp];
                let mut dec = ruzstd::decoding::FrameDecoder::new();
                match dec.decode_all(src, dst) {
                    Ok(n) if n == uncomp => {}
                    _ => return Err(BAD),
                }
                loc[id] = Loc::Arena(arena_ofs, uncomp);
                arena_ofs += uncomp;
            }
        }
        Ok(Blobs { data, arena, loc })
    }

    fn get(&self, id: usize) -> &[u8] {
        match self.loc.get(id).copied().unwrap_or_default() {
            Loc::None => &[],
            Loc::Raw(o, n) => self.data.get(o..o + n).unwrap_or(&[]),
            Loc::Arena(o, n) => self.arena.get(o..o + n).unwrap_or(&[]),
        }
    }
}

/// Forward byte cursor over `[start, end)` of a blob.
struct ByteCursor<'a> {
    data: &'a [u8],
    ofs: usize,
    end: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(data: &'a [u8], start: u64, end: u64) -> Self {
        ByteCursor {
            data,
            ofs: start as usize,
            end: end as usize,
        }
    }

    #[inline]
    fn get(&mut self) -> Option<u8> {
        if self.ofs >= self.end {
            return None;
        }
        let b = *self.data.get(self.ofs)?;
        self.ofs += 1;
        Some(b)
    }

    fn consumed(&self) -> bool {
        self.ofs == self.end
    }
}

/// LSB-first bit reader over the bit range `[start, end)` of a blob.
struct BitCursor<'a> {
    data: &'a [u8],
    ofs: u64,
    end: u64,
}

impl<'a> BitCursor<'a> {
    fn new(data: &'a [u8], start: u64, end: u64) -> Self {
        BitCursor {
            data,
            ofs: start,
            end,
        }
    }

    #[inline]
    fn get_bits(&mut self, n: u32) -> Option<u32> {
        if u64::from(n) > self.end.saturating_sub(self.ofs) {
            return None;
        }
        if n == 0 {
            return Some(0);
        }
        let byte_idx = (self.ofs >> 3) as usize;
        let bit_idx = (self.ofs & 7) as u32;
        let num_bytes = ((bit_idx + n + 7) >> 3) as usize;
        let mut bits = 0u64;
        for i in 0..num_bytes {
            bits |= u64::from(*self.data.get(byte_idx + i)?) << (i * 8);
        }
        self.ofs += u64::from(n);
        Some(((bits >> bit_idx) & ((1u64 << n) - 1)) as u32)
    }

    fn consumed(&self) -> bool {
        self.end.saturating_sub(self.ofs) < 8
    }
}

// ---------------------------------------------------------------------------------------------
// Image decoder.

/// Decoded XBC7 image: logical blocks, row major.
pub(crate) struct Decoded {
    pub blocks: vas::Vector<LogBc7Block, vas::storage::Dynamic>,
}

/// Parsed regular (blob) stream, shared by all stripe decoders.
struct Stream<'a> {
    blobs: Blobs<'a>,
    /// `seek[id][stripe]`: start offset (bytes, or bits for bit blobs); `[id][num_stripes]` = end.
    seek: [[u64; MAX_STRIPES + 1]; BLOB_STRIPE_SEEK_TABLE],
    nbx: u32,
    global_q: u32,
    has_alpha: bool,
}

/// All cursors of one stripe.
struct Cursors<'a> {
    commands: ByteCursor<'a>,
    configs: ByteCursor<'a>,
    partitions2: ByteCursor<'a>,
    partitions3: ByteCursor<'a>,
    predictors: ByteCursor<'a>,
    dc_coeffs: ByteCursor<'a>,
    ac_coeffs: ByteCursor<'a>,
    solid_deltas: ByteCursor<'a>,
    ep_block_index: ByteCursor<'a>,
    raw_weights: ByteCursor<'a>,
    wt_resid: [ByteCursor<'a>; 3],
    /// [0..4] fine R,G,B,A; [4..8] coarse R,G,B,A.
    ep_deltas: [ByteCursor<'a>; 8],
    coeff_signs: BitCursor<'a>,
    pbits: BitCursor<'a>,
    ep_raw: BitCursor<'a>,
}

impl<'a> Cursors<'a> {
    fn new(st: &'a Stream, s: usize) -> Self {
        let b = |id: usize| ByteCursor::new(st.blobs.get(id), st.seek[id][s], st.seek[id][s + 1]);
        let bits = |id: usize| BitCursor::new(st.blobs.get(id), st.seek[id][s], st.seek[id][s + 1]);
        Cursors {
            commands: b(BLOB_COMMANDS),
            configs: b(BLOB_CONFIG),
            partitions2: b(BLOB_PARTITION2),
            partitions3: b(BLOB_PARTITION3),
            predictors: b(BLOB_WEIGHT_PREDICTORS),
            dc_coeffs: b(BLOB_DC_COEFFS_SMALL),
            ac_coeffs: b(BLOB_AC_COEFFS),
            solid_deltas: b(BLOB_SOLID_RGBA_DELTAS),
            ep_block_index: b(BLOB_EP_BLOCK_INDEX),
            raw_weights: b(BLOB_RAW_WEIGHT_BITS),
            wt_resid: std::array::from_fn(|i| b(BLOB_DPCM_WEIGHT_RESID2 + i)),
            ep_deltas: std::array::from_fn(|i| {
                if i < 4 {
                    b(BLOB_EP_DELTA_FINE_R + i)
                } else {
                    b(BLOB_EP_DELTA_COARSE_R + i - 4)
                }
            }),
            coeff_signs: bits(BLOB_COEFF_SIGNS),
            pbits: bits(BLOB_PBITS),
            ep_raw: bits(BLOB_EP_RAW),
        }
    }

    /// Per-stripe desync tripwires: every stream consumed exactly its range.
    fn consumed(&self) -> bool {
        self.commands.consumed()
            && self.configs.consumed()
            && self.partitions2.consumed()
            && self.partitions3.consumed()
            && self.predictors.consumed()
            && self.dc_coeffs.consumed()
            && self.ac_coeffs.consumed()
            && self.solid_deltas.consumed()
            && self.ep_block_index.consumed()
            && self.raw_weights.consumed()
            && self.wt_resid.iter().all(ByteCursor::consumed)
            && self.ep_deltas.iter().all(ByteCursor::consumed)
            && self.coeff_signs.consumed()
            && self.pbits.consumed()
            && self.ep_raw.consumed()
    }
}

/// Decodes one block at (`bx`, `by`) (the body of `image_unpacker::decode_stripe`).
fn decode_block(
    st: &Stream,
    cur: &mut Cursors,
    tile: &Tile,
    bx: i32,
    by: i32,
) -> Option<LogBc7Block> {
    let left = tile.get(bx - 1, by);
    let up = tile.get(bx, by - 1);
    let ld = tile.get(bx - 1, by - 1);
    let rd = tile.get(bx + 1, by - 1);
    let has_alpha = st.has_alpha;
    let mut log = LogBc7Block::default();

    let cmd_byte = cur.commands.get()?;
    let cmd = cmd_byte & 7;
    let ep_mode = (cmd_byte >> 3) & 7;
    let wt_mode = (cmd_byte >> 6) & 1;
    if cmd_byte & 0x80 != 0 {
        return None; // P-frame flag: reserved
    }

    if cmd <= 2 {
        if cmd_byte != cmd {
            return None;
        }
        match cmd {
            0 => log = *left?,
            1 => log = *up?,
            _ => {
                // Solid: DPCM vs the neighbor edge average in pixel space.
                let mut preds = [0i32; 4];
                let mut num = 0;
                if let Some(l) = left {
                    for y in 0..4 {
                        let px = bc7u::unpack_texel(l, 3, y)?;
                        for c in 0..4 {
                            preds[c] += i32::from(px[c]);
                        }
                    }
                    num += 4;
                }
                if let Some(u) = up {
                    for x in 0..4 {
                        let px = bc7u::unpack_texel(u, x, 3)?;
                        for c in 0..4 {
                            preds[c] += i32::from(px[c]);
                        }
                    }
                    num += 4;
                }
                if num != 0 {
                    for p in preds.iter_mut() {
                        *p = (*p + num / 2) / num;
                    }
                }
                let mut solid = [0u8, 0, 0, 255];
                for c in 0..(if has_alpha { 4 } else { 3 }) {
                    let delta = cur.solid_deltas.get()?;
                    solid[c] = delta.wrapping_add(preds[c] as u8);
                }
                create_solid_blk(&mut log, &solid);
            }
        }
        return Some(log);
    }

    // Config.
    if cmd == 3 {
        let config = cur.configs.get()?;
        if config & 0xC0 != 0 {
            return None;
        }
        let mode = u32::from(config & 7);
        let rot = (config >> 3) & 3;
        let sel = (config >> 5) & 1;
        init_log_blk(&mut log, mode);
        if log.num_planes == 2 {
            log.dp_rotation_index = rot;
        } else if rot != 0 {
            return None;
        }
        if mode == 4 {
            log.mode4_index_selector = sel;
        } else if sel != 0 {
            return None;
        }
    } else {
        let src = match cmd {
            4 => left,
            5 => up,
            6 => ld,
            _ => rd,
        }?;
        init_log_blk(&mut log, src.mode as u32);
        log.dp_rotation_index = src.dp_rotation_index;
        log.mode4_index_selector = src.mode4_index_selector;
    }

    // Partition index.
    if log.num_partitions == 2 {
        let pat = cur.partitions2.get()?;
        if pat >= 64 {
            return None;
        }
        log.pattern_index = pat;
    } else if log.num_partitions == 3 {
        let pat = cur.partitions3.get()?;
        if u32::from(pat) >= (1u32 << log.pattern_bits) {
            return None;
        }
        log.pattern_index = pat;
    }

    let mode = (log.mode as usize) & 7;
    let fmt_pbits = ENDPOINT_FORMATS[mode].2 as usize;
    let num_comps = log.num_comps();
    let num_subsets = (log.num_partitions as usize).min(3);

    // Endpoints.
    if ep_mode == 0 {
        for subset in 0..num_subsets {
            for c in 0..num_comps {
                for e in 0..2 {
                    let n = u32::from(log.endpoint_bits[(c == 3) as usize]);
                    log.endpoints[subset][e][c] = cur.ep_raw.get_bits(n)? as u8;
                }
            }
        }
        for pb in 0..log.num_pbits as usize {
            let v = cur.ep_raw.get_bits(1)? as u8;
            if let Some(slot) = log.pbits.get_mut(pb) {
                *slot = v;
            }
        }
    } else {
        let (pred, pred_subset) = match ep_mode {
            1 => (left, 0),
            2 => (up, 0),
            3 => (ld, 0),
            4 => (rd, 0),
            6 => (left, 1),
            7 => (up, 1),
            _ => {
                let di = cur.ep_block_index.get()?;
                let (dx, dy) = *XY_DELTAS.get(di as usize)?;
                (tile.get(bx + i32::from(dx), by + i32::from(dy)), 0)
            }
        };
        let pred = pred?;
        if pred_subset != 0 && pred.num_partitions < 2 {
            return None;
        }
        let fine = log.endpoint_bits[0] >= 6;
        for subset in 0..num_subsets {
            let mut residuals = [0u8; 8];
            let mut num_residuals = num_comps * 2;
            if !has_alpha && mode == 6 {
                num_residuals = 6;
            }
            for i in (0..num_residuals).step_by(2) {
                let strm = &mut cur.ep_deltas[(if fine { 0 } else { 4 }) + (i >> 1)];
                residuals[i] = strm.get()?;
                residuals[i + 1] = strm.get()?;
            }
            let mut rp = [0u8; 2];
            for p in rp.iter_mut().take(fmt_pbits) {
                *p = cur.pbits.get_bits(1)? as u8;
            }
            bc7u::endpoint_dpcm_decode(pred, pred_subset, &mut log, subset, &residuals, &rp);
            if !has_alpha && mode == 6 {
                log.endpoints[0][0][3] = 127;
                log.endpoints[0][1][3] = 127;
            }
        }
    }

    // Weights: residual DCT or lossless residual DPCM.
    let pred_byte = u32::from(cur.predictors.get()?);
    if pred_byte >= TOTAL_CANDIDATES * 4 {
        return None;
    }
    let cand = pred_byte % TOTAL_CANDIDATES;
    let amp = pred_byte / TOTAL_CANDIDATES;
    if amp != 0 && cand == CAND_ABSOLUTE {
        return None;
    }
    let global_q = Fx::from_int(st.global_q as i32);
    for p in 0..(log.num_planes as usize).min(2) {
        let preds = if cand != CAND_ABSOLUTE {
            Some(eval_weight_predictor(cand, amp, bx, by, tile, p)?)
        } else {
            None
        };
        if wt_mode == 0 {
            let num_bits = u32::from(log.weight_bits[p]);
            if !(2..=4).contains(&num_bits) {
                return None;
            }
            let mask = ((1u32 << num_bits) - 1) as u8;
            let strm = if cand == CAND_ABSOLUTE {
                &mut cur.raw_weights
            } else {
                &mut cur.wt_resid[(num_bits - 2) as usize]
            };
            let mut syms = [0u8; 16];
            if num_bits == 2 {
                for i in (0..16).step_by(4) {
                    let b = strm.get()?;
                    syms[i] = b & 3;
                    syms[i + 1] = (b >> 2) & 3;
                    syms[i + 2] = (b >> 4) & 3;
                    syms[i + 3] = b >> 6;
                }
            } else {
                for i in (0..16).step_by(2) {
                    let b = strm.get()?;
                    let (lo, hi) = (b & 0xF, b >> 4);
                    if num_bits == 3 && (lo > 7 || hi > 7) {
                        return None;
                    }
                    syms[i] = lo;
                    syms[i + 1] = hi;
                }
            }
            for i in 0..16 {
                let pred_index = preds.map_or(0, |pr| quant_weight(pr[i], num_bits));
                log.weights[p][i] = syms[i].wrapping_add(pred_index) & mask;
            }
            continue;
        }

        // DC.
        let mut dc = i32::from(cur.dc_coeffs.get()?);
        if pred_byte != CAND_ABSOLUTE && cur.coeff_signs.get_bits(1)? != 0 {
            dc = -dc;
        }
        // ACs.
        let mut acs = [(0u32, 0i32); 16];
        let mut num_acs = 0;
        let mut zig = 1u32;
        while zig < 16 {
            let b = cur.ac_coeffs.get()?;
            if b == 0xFF {
                acs[num_acs] = (16 - zig, i32::from(i16::MAX));
                num_acs += 1;
                break;
            }
            let run = u32::from(b);
            if zig + run > 15 {
                return None;
            }
            let mag = i32::from(cur.ac_coeffs.get()?);
            if mag == 0 {
                return None;
            }
            let sign = cur.coeff_signs.get_bits(1)?;
            acs[num_acs] = (run, if sign != 0 { -mag } else { mag });
            num_acs += 1;
            zig += run + 1;
        }
        if !dct_inverse(global_q, p, preds.as_ref(), dc, &acs[..num_acs], &mut log) {
            return None;
        }
    }
    Some(log)
}

/// Decodes stripe `s` (block rows `first_row..end_row`) into `rows`.
fn decode_stripe(
    st: &Stream,
    s: usize,
    first_row: u32,
    end_row: u32,
    rows: &mut [LogBc7Block],
) -> Result<()> {
    let mut cur = Cursors::new(st, s);
    let nbx = st.nbx as usize;
    for by in first_row..end_row {
        for bx in 0..nbx {
            let blk = {
                let tile = Tile {
                    blks: rows,
                    nbx: st.nbx as i32,
                    first_row: first_row as i32,
                    end_row: end_row as i32,
                };
                decode_block(st, &mut cur, &tile, bx as i32, by as i32)
            }
            .ok_or(Error::InvalidData)?;
            let i = (by - first_row) as usize * nbx + bx;
            *rows.get_mut(i).ok_or(Error::InvalidData)? = blk;
        }
    }
    if cur.consumed() {
        Ok(())
    } else {
        Err(Error::InvalidData)
    }
}

/// Decodes an XBC7 stream whose block grid must be `blocks_x` x `blocks_y`.
pub(crate) fn decode(comp: &[u8], blocks_x: u32, blocks_y: u32) -> Result<Decoded> {
    let first = *comp.first().ok_or(BAD)?;
    let mut blocks = vas::Vector::<LogBc7Block, vas::storage::Dynamic>::default();

    if first == 0xB8 || first == 0xB9 {
        // Tiny mip: [marker][nbx][nby] then raw BC7 blocks.
        if comp.len() < 3 {
            return Err(BAD);
        }
        let (nbx, nby) = (u32::from(comp[1]), u32::from(comp[2]));
        if nbx == 0 || nby == 0 || comp.len() as u64 != 3 + u64::from(nbx) * u64::from(nby) * 16 {
            return Err(BAD);
        }
        if nbx != blocks_x || nby != blocks_y {
            return Err(BAD);
        }
        let n = (nbx * nby) as usize;
        blocks
            .resize(n, LogBc7Block::default())
            .map_err(|_| Error::OutOfMemory)?;
        for (b, out) in comp[3..]
            .as_chunks::<16>()
            .0
            .iter()
            .zip(blocks.as_mut_slice())
        {
            if !bc7u::unpack_phys(b, out) {
                return Err(BAD);
            }
        }
        return Ok(Decoded { blocks });
    }
    if first != MAGIC_BEGIN {
        return Err(BAD);
    }
    let blobs = Blobs::parse(comp)?;

    // Header: u16 width, u16 height, u8 dct_q, u8 flags, u8 num_stripes.
    let hdr = blobs.get(BLOB_HEADER);
    if hdr.len() != 7 {
        return Err(BAD);
    }
    let width = u32::from(u16::from_le_bytes([hdr[0], hdr[1]]));
    let height = u32::from(u16::from_le_bytes([hdr[2], hdr[3]]));
    let global_q = u32::from(hdr[4]);
    let flags = hdr[5];
    if width == 0 || height == 0 || width > MAX_DIM || height > MAX_DIM {
        return Err(BAD);
    }
    if !(1..=100).contains(&global_q) || flags & !1 != 0 {
        return Err(BAD);
    }
    let has_alpha = flags & 1 != 0;
    let (nbx, nby) = (width.div_ceil(4), height.div_ceil(4));
    let total_blocks = (nbx * nby) as usize;
    let num_stripes = u32::from(hdr[6]);
    if num_stripes == 0 || num_stripes > nby || num_stripes as usize > MAX_STRIPES {
        return Err(BAD);
    }
    if blobs.get(BLOB_COMMANDS).len() != total_blocks {
        return Err(BAD);
    }
    let ns = num_stripes as usize;

    // Per-stripe seek table (deltas, byte-plane transposed).
    let mut seek = [[0u64; MAX_STRIPES + 1]; BLOB_STRIPE_SEEK_TABLE];
    for (id, row) in seek.iter_mut().enumerate().skip(1) {
        let size = blobs.get(id).len() as u64;
        let bit_blob = id == BLOB_COEFF_SIGNS || id == BLOB_PBITS || id == BLOB_EP_RAW;
        row[ns] = if bit_blob { size * 8 } else { size };
    }
    if ns > 1 {
        let num_streams = BLOB_STRIPE_SEEK_TABLE - 1;
        let t = blobs.get(BLOB_STRIPE_SEEK_TABLE);
        let num_entries = ns * num_streams;
        if t.len() != num_entries * 4 {
            return Err(BAD);
        }
        for (id, row) in seek.iter_mut().enumerate().skip(1) {
            let blob_end = row[ns];
            let mut running = 0u64;
            for (st, slot) in row.iter_mut().enumerate().take(ns) {
                let e = st * num_streams + (id - 1);
                let delta = u64::from(t[e])
                    | (u64::from(t[num_entries + e]) << 8)
                    | (u64::from(t[2 * num_entries + e]) << 16)
                    | (u64::from(t[3 * num_entries + e]) << 24);
                if st == 0 && delta != 0 {
                    return Err(BAD);
                }
                running += delta;
                if running > blob_end {
                    return Err(BAD);
                }
                *slot = running;
            }
        }
    }

    // Init callback: geometry must match the container.
    if nbx != blocks_x || nby != blocks_y {
        return Err(BAD);
    }
    blocks
        .resize(total_blocks, LogBc7Block::default())
        .map_err(|_| Error::OutOfMemory)?;

    let stream = Stream {
        blobs,
        seek,
        nbx,
        global_q,
        has_alpha,
    };

    // `compute_stripe_ranges`: the first nby % ns stripes carry one extra row.
    let (base_rows, extra) = (nby / num_stripes, nby % num_stripes);
    let mut ranges = [(0u32, 0u32); MAX_STRIPES];
    let mut row = 0;
    for (i, r) in ranges.iter_mut().enumerate().take(ns) {
        let n = base_rows + u32::from((i as u32) < extra);
        *r = (row, row + n);
        row += n;
    }

    // Split the block buffer by stripe and decode stripes in parallel.
    let mut parts: [&mut [LogBc7Block]; MAX_STRIPES] = Default::default();
    let mut rest = blocks.as_mut_slice();
    for (i, &(r0, r1)) in ranges.iter().enumerate().take(ns) {
        let (head, tail) = rest.split_at_mut(((r1 - r0) * nbx) as usize);
        parts[i] = head;
        rest = tail;
    }
    let stream = &stream;
    // Stripes are handed out from a shared queue; the calling thread works too, so a failed
    // spawn only means less parallelism. A stripe whose result never arrives counts as failed.
    let queue = Mutex::new(parts.iter_mut().take(ns).enumerate());
    let results: Mutex<[bool; MAX_STRIPES]> = Mutex::new([false; MAX_STRIPES]);
    let run = || {
        loop {
            let next = match queue.lock() {
                Ok(mut it) => it.next(),
                Err(_) => None,
            };
            let Some((i, part)) = next else { break };
            let (r0, r1) = ranges[i];
            let ok = decode_stripe(stream, i, r0, r1, std::mem::take(part)).is_ok();
            if let Ok(mut r) = results.lock() {
                r[i] = ok;
            }
        }
    };
    let workers = if total_blocks * 16 >= crate::parallel::MIN_PARALLEL_PIXELS {
        std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(ns)
    } else {
        1
    };
    std::thread::scope(|scope| {
        for _ in 1..workers {
            let _ = std::thread::Builder::new().spawn_scoped(scope, run);
        }
        run();
    });
    let results = results.into_inner().map_err(|_| BAD)?;
    if results.iter().take(ns).all(|&ok| ok) {
        Ok(Decoded { blocks })
    } else {
        Err(BAD)
    }
}

/// `a && b`: `b` when both are present.
#[inline]
fn both2<'a>(a: Option<&'a LogBc7Block>, b: Option<&'a LogBc7Block>) -> Option<&'a LogBc7Block> {
    a.and(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isqrt_exact() {
        for x in [
            0u64,
            1,
            2,
            3,
            4,
            15,
            16,
            17,
            1 << 40,
            (1 << 63) + 12345,
            u64::MAX,
        ] {
            let r = isqrt_floor(x);
            assert!(u128::from(r) * u128::from(r) <= u128::from(x));
            assert!(u128::from(r + 1) * u128::from(r + 1) > u128::from(x));
        }
    }

    #[test]
    fn fixed_rounding() {
        assert_eq!(Fx(98304).round_to_int(), 2); // 1.5 -> 2
        assert_eq!(Fx(-98304).round_to_int(), -2);
        assert_eq!(Fx::from_int(7).div_int(2), Fx(229376)); // 3.5
        assert_eq!(Fx::from_int(1).div(Fx::from_int(3)), Fx(21845));
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn block(&mut self, mode: u32) -> [u8; 16] {
            let mut b = [0u8; 16];
            for v in b.iter_mut() {
                *v = self.next() as u8;
            }
            b[0] = (u32::from(b[0]) & !((2u32 << mode) - 1) | (1 << mode)) as u8;
            b
        }
        fn log(&mut self) -> LogBc7Block {
            let mode = (self.next() & 7) as u32;
            let mut l = LogBc7Block::default();
            assert!(bc7u::unpack_phys(&self.block(mode), &mut l));
            l
        }
    }

    struct Fnv(u64);
    impl Fnv {
        fn bytes(&mut self, b: &[u8]) {
            for &x in b {
                self.0 = (self.0 ^ u64::from(x)).wrapping_mul(0x100000001b3);
            }
        }
        fn log(&mut self, l: &LogBc7Block) {
            self.bytes(l.endpoints.as_flattened().as_flattened());
            self.bytes(l.weights.as_flattened());
            self.bytes(&l.pbits);
        }
    }

    /// Hashes from tools/basisu-golden/xubc7_check.cpp (C++ transcoder, same PRNG streams).
    #[test]
    fn matches_cpp_internals() {
        const N: u32 = 100000;
        let astc = [
            0xc4a541375baaaf02u64,
            0x72a1c69644afb63e,
            0x2757e7e2fb76b08c,
            0xab49189b1ec2c149,
            0xb37a8be40360ffa4,
            0xbe045e0a8a2c6723,
            0x65bd91d41fe8ad6b,
            0x228cee54db364a59,
        ];
        for (mode, &want) in astc.iter().enumerate() {
            let mut r = Rng(0x1234567 + mode as u64);
            let mut h = Fnv(0xcbf29ce484222325);
            for _ in 0..N {
                let b = r.block(mode as u32);
                let mut out = [0u8; 16];
                let ok = crate::basis::ll::bc7f_astc::fast_pack_astc(
                    &mut out,
                    &b,
                    crate::basis::ll::bc7f::PACK_BC7_FLAG_DEFAULT,
                );
                h.bytes(&[ok as u8]);
                h.bytes(&out);
            }
            assert_eq!(h.0, want, "fast_pack_astc mode {mode}");
        }

        // Logical unpack / pack / pixels.
        let mut r = Rng(0xabcdef);
        let mut h = Fnv(0xcbf29ce484222325);
        for i in 0..N {
            let b = r.block(i & 7);
            let mut l = LogBc7Block::default();
            bc7u::unpack_phys(&b, &mut l);
            let mut out = [0u8; 16];
            bc7u::pack_bc7(&l, &mut out);
            let mut px = [[0u8; 4]; 16];
            bc7u::unpack_pixels(&l, &mut px);
            h.bytes(&out);
            h.bytes(px.as_flattened());
            let t = bc7u::unpack_texel(&l, (i & 3) as usize, ((i >> 2) & 3) as usize);
            h.bytes(&t.unwrap_or_default());
        }
        assert_eq!(h.0, 0x3efe620638ca1b79, "log");

        // Endpoint DPCM.
        let mut r = Rng(0x5eed);
        let mut h = Fnv(0xcbf29ce484222325);
        for _ in 0..N {
            let pred = r.log();
            let mut cur = r.log();
            let ps = (r.next() % u64::from(pred.num_partitions)) as usize;
            for s in 0..cur.num_partitions as usize {
                let mut res = [0u8; 8];
                for v in res.iter_mut() {
                    *v = r.next() as u8;
                }
                let pb = [(r.next() & 1) as u8, (r.next() & 1) as u8];
                bc7u::endpoint_dpcm_decode(&pred, ps, &mut cur, s, &res, &pb);
            }
            h.log(&cur);
        }
        assert_eq!(h.0, 0xca736b7d929edb89, "dpcm");

        // Weight grid DCT.
        let mut r = Rng(0xdc7);
        let mut h = Fnv(0xcbf29ce484222325);
        for _ in 0..N {
            let mut l = r.log();
            let p = (r.next() % u64::from(l.num_planes)) as usize;
            let preds: [i32; 16] = std::array::from_fn(|_| (r.next() % 65) as i32);
            let use_preds = r.next() & 1 != 0;
            let q = 1 + (r.next() % 100) as i32;
            let dc = (r.next() % 511) as i32 - 255;
            let mut acs = Vec::new();
            let mut zig = 1u32;
            while zig < 16 {
                if r.next().is_multiple_of(4) {
                    acs.push((16 - zig, i32::from(i16::MAX)));
                    break;
                }
                let run = (r.next() % u64::from(16 - zig)) as u32;
                let mag = 1 + (r.next() % 255) as i32;
                acs.push((run, if r.next() & 1 != 0 { -mag } else { mag }));
                zig += run + 1;
            }
            let pr = if use_preds { Some(&preds) } else { None };
            let ok = dct_inverse(Fx::from_int(q), p, pr, dc, &acs, &mut l);
            h.bytes(&[ok as u8]);
            h.bytes(l.weights.as_flattened());
        }
        assert_eq!(h.0, 0xdc43939225b9d55a, "dct");

        // Weight predictors.
        let mut r = Rng(0x9e7);
        let mut h = Fnv(0xcbf29ce484222325);
        let mut grid = [LogBc7Block::default(); 36];
        for _ in 0..N / 10 {
            for g in grid.iter_mut() {
                *g = r.log();
            }
            let tile = Tile {
                blks: &grid,
                nbx: 6,
                first_row: 0,
                end_row: 6,
            };
            for _ in 0..20 {
                let bx = (r.next() % 6) as i32;
                let by = (r.next() % 6) as i32;
                let cand = 1 + (r.next() % u64::from(TOTAL_CANDIDATES - 1)) as u32;
                let amp = (r.next() & 3) as u32;
                let p = (r.next() & 1) as usize;
                let out = eval_weight_predictor(cand, amp, bx, by, &tile, p);
                h.bytes(&[out.is_some() as u8]);
                if let Some(o) = out {
                    for v in o {
                        h.bytes(&v.to_le_bytes());
                    }
                }
            }
        }
        assert_eq!(h.0, 0x1fcf91471cc6c915, "pred");
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode(&[], 1, 1).is_err());
        assert!(decode(&[0xB7, 0, 0x6A], 1, 1).is_err());
        assert!(decode(&[0xB8, 1, 1], 1, 1).is_err());
        assert!(decode(&[0x12; 40], 1, 1).is_err());
    }
}
