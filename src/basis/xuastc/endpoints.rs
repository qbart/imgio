// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of the `astc_ldr_t` endpoint helpers used by XUASTC LDR endpoint prediction:
//! `decode_endpoints`, `requantize_ise_endpoints`, `pack_base_offset` and
//! `convert_endpoints_across_cems` (integer only, part of the XUASTC specification).

use crate::astc::helpers::{
    self as ah, BISE_256_LEVELS, CEM_LDR_LUM_ALPHA_DIRECT, CEM_LDR_LUM_DIRECT,
    CEM_LDR_RGB_BASE_PLUS_OFFSET, CEM_LDR_RGB_BASE_SCALE, CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A,
    CEM_LDR_RGB_DIRECT, CEM_LDR_RGBA_BASE_PLUS_OFFSET, CEM_LDR_RGBA_DIRECT,
    FIRST_VALID_ENDPOINT_ISE_RANGE, LAST_VALID_ENDPOINT_ISE_RANGE,
};
use crate::basis::ll::astc_6x6_hdr::quant_preserve2;
use std::sync::OnceLock;

/// `MAX_CEM_ENDPOINT_VALS`.
pub(crate) const MAX_CEM_VALS: usize = 8;

type Rgba = [u8; 4];

/// `ISE_to_val` of an endpoint range (out of range symbols read as 0).
fn dequant(range: u32, ise: u8) -> u8 {
    if range == BISE_256_LEVELS {
        return ise;
    }
    ah::dequant_tables()
        .endpoint_tab(range)
        .ise_to_val
        .get(usize::from(ise))
        .copied()
        .unwrap_or(0)
}

fn quant(range: u32, v: u8) -> u8 {
    ah::dequant_tables()
        .endpoint_tab(range)
        .val_to_ise
        .get(usize::from(v))
        .copied()
        .unwrap_or(0)
}

fn levels(range: u32) -> u32 {
    ah::get_ise_levels(range)
}

/// `bit_transfer_signed_dec`.
fn bit_transfer_dec(a: &mut i32, b: &mut i32) {
    *b >>= 1;
    *b |= *a & 0x80;
    *a >>= 1;
    *a &= 0x3F;
    if *a & 0x20 != 0 {
        *a -= 0x40;
    }
}

/// `bit_transfer_signed_enc`.
fn bit_transfer_enc(a: &mut i32, b: &mut i32) {
    let bit = *b & 0x80 != 0;
    *b = (*b << 1) & 0xFF;
    *a &= 0x3F;
    *a <<= 1;
    if bit {
        *a |= 0x80;
    }
}

fn has_alpha(cem: u32) -> bool {
    matches!(
        cem,
        CEM_LDR_LUM_ALPHA_DIRECT
            | ah::CEM_LDR_LUM_ALPHA_BASE_PLUS_OFS
            | CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A
            | CEM_LDR_RGBA_DIRECT
            | CEM_LDR_RGBA_BASE_PLUS_OFFSET
    )
}

/// `cem_supports_bc`.
pub(crate) fn cem_supports_bc(cem: u32) -> bool {
    matches!(
        cem,
        CEM_LDR_RGB_DIRECT
            | CEM_LDR_RGBA_DIRECT
            | CEM_LDR_RGB_BASE_PLUS_OFFSET
            | CEM_LDR_RGBA_BASE_PLUS_OFFSET
    )
}

fn base_cem_without_alpha(cem: u32) -> u32 {
    match cem {
        CEM_LDR_LUM_ALPHA_DIRECT => CEM_LDR_LUM_DIRECT,
        CEM_LDR_RGBA_DIRECT => CEM_LDR_RGB_DIRECT,
        CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => CEM_LDR_RGB_BASE_SCALE,
        CEM_LDR_RGBA_BASE_PLUS_OFFSET => CEM_LDR_RGB_BASE_PLUS_OFFSET,
        c => c,
    }
}

/// `used_blue_contraction`.
pub(crate) fn used_blue_contraction(cem: u32, vals: &[u8], range: u32) -> bool {
    let d = |i: usize| i32::from(dequant(range, vals.get(i).copied().unwrap_or(0)));
    match cem {
        CEM_LDR_RGB_DIRECT | CEM_LDR_RGBA_DIRECT => d(1) + d(3) + d(5) < d(0) + d(2) + d(4),
        CEM_LDR_RGB_BASE_PLUS_OFFSET | CEM_LDR_RGBA_BASE_PLUS_OFFSET => {
            let mut e = [d(0), d(1), d(2), d(3), d(4), d(5)];
            for i in 0..3 {
                let (lo, hi) = e.split_at_mut(2 * i + 1);
                bit_transfer_dec(&mut hi[0], &mut lo[2 * i]);
            }
            e[1] + e[3] + e[5] < 0
        }
        _ => false,
    }
}

/// `apply_delta_to_bise_endpoint_val`.
fn apply_delta(range: u32, ise_val: u8, delta: i32) -> u8 {
    if delta == 0 {
        return ise_val;
    }
    let t = ah::dequant_tables().endpoint_tab(range);
    let cur = i32::from(
        t.ise_to_rank
            .get(usize::from(ise_val))
            .copied()
            .unwrap_or(0),
    );
    let new = (cur + delta).clamp(0, levels(range) as i32 - 1);
    t.rank_to_ise.get(new as usize).copied().unwrap_or(0)
}

/// `decode_endpoints`: low and high RGBA8 endpoints of one partition.
pub(crate) fn decode_endpoints(cem: u32, vals: &[u8], range: u32) -> (Rgba, Rgba) {
    let n = (ah::get_num_cem_values(cem) as usize).min(MAX_CEM_VALS);
    let mut dq = [0u8; MAX_CEM_VALS];
    for (d, &v) in dq.iter_mut().zip(vals).take(n) {
        *d = dequant(range, v);
    }
    let mut e = [[0i32; 2]; 4];
    ah::decode_endpoint(cem, &mut e, &dq[..n]);
    (e.map(|c| c[0] as u8), e.map(|c| c[1] as u8))
}

/// `g_base_ofs_nudges[range][pos=0, neg=1][ise]`.
fn base_ofs_nudges() -> &'static [[[u8; 256]; 2]; 21] {
    static T: OnceLock<Box<[[[u8; 256]; 2]; 21]>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Box::new([[[0u8; 256]; 2]; 21]);
        for range in FIRST_VALID_ENDPOINT_ISE_RANGE..=LAST_VALID_ENDPOINT_ISE_RANGE {
            let n = levels(range);
            let split = |ise: u32| {
                let (mut a, mut b) = (i32::from(dequant(range, ise as u8)), 0);
                bit_transfer_dec(&mut a, &mut b);
                (a, b)
            };
            for neg in 0..2 {
                for cur in 0..n {
                    let (cur_a, cur_b) = split(cur);
                    let (mut best_err, mut best) = (i32::MAX, 0u32);
                    for trial in 0..n {
                        let (ta, tb) = split(trial);
                        if tb != cur_b || ta == cur_a {
                            continue;
                        }
                        if (neg == 1 && ta > cur_a) || (neg == 0 && ta < cur_a) {
                            continue;
                        }
                        let e = (ta - cur_a).abs();
                        if e < best_err {
                            best_err = e;
                            best = trial;
                        }
                    }
                    if best_err == i32::MAX {
                        best = cur;
                    }
                    t[range as usize][neg][cur as usize] = best as u8;
                }
            }
        }
        t
    })
}

/// Dequantized, bit transferred base+offset deltas (v1 + v3 + v5).
fn base_ofs_delta_sum(dst: &[u8; MAX_CEM_VALS], range: u32) -> i32 {
    let mut v = [0i32; 6];
    for (o, &s) in v.iter_mut().zip(dst) {
        *o = i32::from(dequant(range, s));
    }
    for i in 0..3 {
        let (lo, hi) = v.split_at_mut(2 * i + 1);
        bit_transfer_dec(&mut hi[0], &mut lo[2 * i]);
    }
    v[1] + v[3] + v[5]
}

/// `requantize_ise_endpoints`: preserves blue contraction and base+offset MSBs.
pub(crate) fn requantize_ise_endpoints(
    cem: u32,
    src_range: u32,
    src: &[u8; MAX_CEM_VALS],
    dst_range: u32,
    dst: &mut [u8; MAX_CEM_VALS],
) -> bool {
    if !ah::is_cem_ldr(cem) {
        return false;
    }
    let n = (ah::get_num_cem_values(cem) as usize).min(MAX_CEM_VALS);
    if src_range == dst_range {
        dst[..n].copy_from_slice(&src[..n]);
        return true;
    }
    let mut dq = [0u8; MAX_CEM_VALS];
    for (d, &s) in dq.iter_mut().zip(src).take(n) {
        *d = dequant(src_range, s);
    }
    if dst_range == BISE_256_LEVELS {
        dst[..n].copy_from_slice(&dq[..n]);
        return true;
    }
    let dd = |v: u8| i32::from(dequant(dst_range, v));
    match cem {
        CEM_LDR_RGB_BASE_PLUS_OFFSET | CEM_LDR_RGBA_BASE_PLUS_OFFSET => {
            for i in 0..n {
                dst[i] = if i & 1 != 0 {
                    quant_preserve2(dst_range, dq[i])
                } else {
                    quant(dst_range, dq[i])
                };
            }
            let src_bc = used_blue_contraction(cem, src, src_range);
            let mut quant_bc = base_ofs_delta_sum(dst, dst_range) < 0;
            const MAX_TRIES: usize = 5;
            if src_bc != quant_bc {
                // nudge_delta = quant_bc ? 1 : -1; table index 1 = negative nudge.
                let neg = usize::from(!quant_bc);
                let Some(nudges) = base_ofs_nudges().get(dst_range as usize) else {
                    return false;
                };
                // The first component tried rotates, starting with blue.
                for rover in (2usize..).take(MAX_TRIES) {
                    for j in 0..3 {
                        let i = 1 + ((rover + j) % 3) * 2;
                        let new = nudges[neg][usize::from(dst[i])];
                        if new != dst[i] {
                            dst[i] = new;
                            break;
                        }
                    }
                    quant_bc = base_ofs_delta_sum(dst, dst_range) < 0;
                    if src_bc == quant_bc {
                        break;
                    }
                }
            }
        }
        CEM_LDR_RGB_DIRECT | CEM_LDR_RGBA_DIRECT => {
            let s0 = u32::from(dq[0]) + u32::from(dq[2]) + u32::from(dq[4]);
            let s1 = u32::from(dq[1]) + u32::from(dq[3]) + u32::from(dq[5]);
            let orig_bc = s1 < s0;
            for i in 0..n {
                dst[i] = quant(dst_range, dq[i]);
            }
            let q0 = dd(dst[0]) + dd(dst[2]) + dd(dst[4]);
            let q1 = dd(dst[1]) + dd(dst[3]) + dd(dst[5]);
            let quant_bc = q1 < q0;
            if orig_bc != quant_bc {
                if q0 == q1 {
                    if q1 != 0 {
                        for i in 0..3 {
                            let new = apply_delta(dst_range, dst[1 + i * 2], -1);
                            if new != dst[1 + i * 2] {
                                dst[1 + i * 2] = new;
                                break;
                            }
                        }
                    } else {
                        for i in 0..3 {
                            let new = apply_delta(dst_range, dst[i * 2], 1);
                            if new != dst[i * 2] {
                                dst[i * 2] = new;
                                break;
                            }
                        }
                    }
                } else {
                    let pairs = if cem == CEM_LDR_RGBA_DIRECT { 4 } else { 3 };
                    for i in 0..pairs {
                        dst.swap(2 * i, 2 * i + 1);
                    }
                }
            }
        }
        _ => {
            for i in 0..n {
                dst[i] = quant(dst_range, dq[i]);
            }
        }
    }
    true
}

/// `blue_contract_enc`.
fn blue_contract_enc(orig: Rgba, did_clamp: &mut bool, encoded_b: i32) -> Rgba {
    let tr = i32::from(orig[0]) * 2 - encoded_b;
    let tg = i32::from(orig[1]) * 2 - encoded_b;
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

/// `pack_base_offset` (with `auto_disable_blue_contraction_if_clamped` = false).
fn pack_base_offset(
    cem: u32,
    dst_range: u32,
    dst: &mut [u8; MAX_CEM_VALS],
    l: Rgba,
    h: Rgba,
    use_bc: bool,
) -> bool {
    if cem != CEM_LDR_RGB_BASE_PLUS_OFFSET && cem != CEM_LDR_RGBA_BASE_PLUS_OFFSET {
        return false;
    }
    let (mut pack_l, mut pack_h) = (l, h);
    if use_bc {
        let mut clamped = false;
        let enc_l = blue_contract_enc(pack_l, &mut clamped, i32::from(pack_l[2]));
        let enc_h = blue_contract_enc(pack_h, &mut clamped, i32::from(pack_h[2]));
        pack_h = enc_l;
        pack_l = enc_h;
    }
    let (mut dr, mut dg, mut db, mut da) = (0i32, 0i32, 0i32, 0i32);
    let mut low_clamp = -32;
    for pass in 0..4 {
        let d = |c: usize| i32::from(pack_h[c]) - i32::from(pack_l[c]);
        dr = d(0).clamp(low_clamp, 31);
        dg = d(1).clamp(low_clamp, 31);
        db = d(2).clamp(low_clamp, 31);
        da = d(3).clamp(low_clamp, 31);
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
    bit_transfer_enc(&mut v1, &mut v0);
    bit_transfer_enc(&mut v3, &mut v2);
    bit_transfer_enc(&mut v5, &mut v4);
    let mut e = [
        v0 as u8, v1 as u8, v2 as u8, v3 as u8, v4 as u8, v5 as u8, 0, 0,
    ];
    if has_alpha(cem) {
        let (mut v6, mut v7) = (i32::from(pack_l[3]), da);
        bit_transfer_enc(&mut v7, &mut v6);
        e[6] = v6 as u8;
        e[7] = v7 as u8;
    }
    requantize_ise_endpoints(cem, BISE_256_LEVELS, &e, dst_range, dst)
}

/// `convert_endpoints_across_cems` with `always_repack` and
/// `auto_disable_blue_contraction_if_clamped` false (the decoder's prediction path).
pub(crate) fn convert_endpoints_across_cems(
    prev_cem: u32,
    prev_range: u32,
    prev: &[u8; MAX_CEM_VALS],
    dst_cem: u32,
    dst_range: u32,
    dst: &mut [u8; MAX_CEM_VALS],
    use_bc: bool,
) -> bool {
    if !ah::is_cem_ldr(prev_cem) || !ah::is_cem_ldr(dst_cem) {
        return false;
    }
    if prev_cem == dst_cem {
        return requantize_ise_endpoints(prev_cem, prev_range, prev, dst_range, dst);
    }
    let prev_base = base_cem_without_alpha(prev_cem);
    let dst_base = base_cem_without_alpha(dst_cem);
    if prev_base == dst_base {
        if !has_alpha(dst_cem) {
            // Alpha stripped.
            return requantize_ise_endpoints(prev_base, prev_range, prev, dst_range, dst);
        }
        // Alpha added: plug in 255 for both alphas.
        if !requantize_ise_endpoints(prev_base, prev_range, prev, dst_range, dst) {
            return false;
        }
        let a = quant(dst_range, 255);
        match dst_cem {
            CEM_LDR_LUM_ALPHA_DIRECT => {
                dst[2] = a;
                dst[3] = a;
            }
            CEM_LDR_RGBA_DIRECT => {
                dst[6] = a;
                dst[7] = a;
            }
            CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => {
                dst[4] = a;
                dst[5] = a;
            }
            CEM_LDR_RGBA_BASE_PLUS_OFFSET => {
                dst[6] = a;
                dst[7] = quant(dst_range, 128);
            }
            _ => {}
        }
        return true;
    }

    // Different CEM classes: fully decode, then repack.
    let (pl, ph) = decode_endpoints(prev_cem, prev, prev_range);
    let mut e = [0u8; MAX_CEM_VALS];
    let sum3 = |c: Rgba| u32::from(c[0]) + u32::from(c[1]) + u32::from(c[2]);
    match dst_cem {
        CEM_LDR_LUM_DIRECT | CEM_LDR_LUM_ALPHA_DIRECT => {
            e[0] = ((sum3(pl) + 1) / 3) as u8;
            e[1] = ((sum3(ph) + 1) / 3) as u8;
            if dst_cem == CEM_LDR_LUM_ALPHA_DIRECT {
                e[2] = pl[3];
                e[3] = ph[3];
            }
            if prev_cem != CEM_LDR_LUM_DIRECT && prev_cem != CEM_LDR_LUM_ALPHA_DIRECT && e[0] > e[1]
            {
                e.swap(0, 1);
                e.swap(2, 3);
            }
            requantize_ise_endpoints(dst_cem, BISE_256_LEVELS, &e, dst_range, dst)
        }
        CEM_LDR_RGB_DIRECT | CEM_LDR_RGBA_DIRECT => {
            let rgba = dst_cem == CEM_LDR_RGBA_DIRECT;
            for c in 0..if rgba { 4 } else { 3 } {
                e[2 * c] = pl[c];
                e[2 * c + 1] = ph[c];
            }
            if use_bc {
                let mut clamped = false;
                let bl = i32::from(dequant(dst_range, quant(dst_range, pl[2])));
                let bh = i32::from(dequant(dst_range, quant(dst_range, ph[2])));
                let enc_l = blue_contract_enc(pl, &mut clamped, bl);
                let enc_h = blue_contract_enc(ph, &mut clamped, bh);
                for c in 0..3 {
                    e[2 * c] = enc_h[c];
                    e[2 * c + 1] = enc_l[c];
                }
                if rgba {
                    e[6] = ph[3];
                    e[7] = pl[3];
                }
            }
            let s0 = u32::from(e[0]) + u32::from(e[2]) + u32::from(e[4]);
            let s1 = u32::from(e[1]) + u32::from(e[3]) + u32::from(e[5]);
            if (s1 < s0) != use_bc {
                if s0 == s1 {
                    if s1 != 0 {
                        for i in 0..3 {
                            let new = apply_delta(BISE_256_LEVELS, e[1 + i * 2], -1);
                            if new != e[1 + i * 2] {
                                e[1 + i * 2] = new;
                                break;
                            }
                        }
                    } else {
                        for i in 0..3 {
                            let new = apply_delta(BISE_256_LEVELS, e[i * 2], 1);
                            if new != e[i * 2] {
                                e[i * 2] = new;
                                break;
                            }
                        }
                    }
                } else {
                    let n = ah::get_num_cem_values(dst_cem) as usize;
                    for i in (0..n.min(MAX_CEM_VALS)).step_by(2) {
                        e.swap(i, i + 1);
                    }
                }
            }
            requantize_ise_endpoints(dst_cem, BISE_256_LEVELS, &e, dst_range, dst)
        }
        CEM_LDR_RGB_BASE_SCALE | CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => {
            let (mut lc, mut hc) = (pl, ph);
            let from_scale =
                prev_cem == CEM_LDR_RGB_BASE_SCALE || prev_cem == CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A;
            if !from_scale && sum3(lc) > sum3(hc) {
                std::mem::swap(&mut lc, &mut hc);
            }
            e[0] = hc[0];
            e[1] = hc[1];
            e[2] = hc[2];
            let dot = |a: Rgba, b: Rgba| {
                (0..3)
                    .map(|c| i32::from(a[c]) * i32::from(b[c]))
                    .sum::<i32>()
            };
            let id = dot(lc, hc);
            let inrm = dot(hc, hc);
            const IMAX_S: i32 = (1024 * 255) / 256;
            let mut iscale = IMAX_S;
            if inrm > 0 {
                iscale = (id * 1024) / inrm;
            }
            iscale = iscale.clamp(0, IMAX_S);
            iscale = ((iscale + 2) >> 2).clamp(0, 255);
            e[3] = iscale as u8;
            if dst_cem == CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A {
                e[4] = lc[3];
                e[5] = hc[3];
                if !from_scale && e[4] > e[5] {
                    e.swap(4, 5);
                }
            }
            requantize_ise_endpoints(dst_cem, BISE_256_LEVELS, &e, dst_range, dst)
        }
        CEM_LDR_RGB_BASE_PLUS_OFFSET | CEM_LDR_RGBA_BASE_PLUS_OFFSET => {
            pack_base_offset(dst_cem, dst_range, dst, pl, ph, use_bc)
        }
        _ => false,
    }
}
