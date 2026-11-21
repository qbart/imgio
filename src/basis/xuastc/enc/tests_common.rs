// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! `common*.rs` vs. the C++ `astc_ldr` common API: regenerates the inputs of
//! `tools/basisu-golden/astc_ldr_common.cpp` and checks every hash in
//! `testdata/basisu_enc/basisu_astc_ldr_common.txt`.
#![allow(clippy::panic, clippy::needless_range_loop)]

use std::collections::BTreeMap;

use super::common::*;
use crate::astc::helpers::{self as ah, BISE_64_LEVELS};
use crate::basis::ll::enc::ColorRgba;

const GOLDEN: &str = include_str!("../../../../testdata/basisu_enc/basisu_astc_ldr_common.txt");

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
    fn range(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

struct Group {
    h: u64,
    count: u32,
}

impl Default for Group {
    fn default() -> Self {
        Self {
            h: 0xcbf29ce484222325,
            count: 0,
        }
    }
}

impl Group {
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.h ^= x as u64;
            self.h = self.h.wrapping_mul(0x100000001b3);
        }
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
}

const SIZES: [u32; 16] = [
    16, 144, 36, 64, 20, 25, 30, 48, 80, 100, 120, 1, 2, 5, 9, 12,
];
const LDR_CEMS: [u32; 8] = [0, 4, 6, 8, 9, 10, 12, 13];

fn make_block(b: u32, px: &mut [ColorRgba; 144]) -> u32 {
    let mut r = Lcg(b.wrapping_mul(7919).wrapping_add(17));
    let n = SIZES[(b % 16) as usize];
    let kind = (b / 16) % 10;
    let mut base = [0i32; 4];
    let mut base2 = [0i32; 4];
    for c in 0..4 {
        base[c] = r.range(256) as i32;
    }
    for c in 0..4 {
        base2[c] = r.range(256) as i32;
    }
    let alpha = r.next() & 1 != 0;
    let dx = r.range(41) as i32 - 20;
    let dy = r.range(41) as i32 - 20;
    for i in 0..n {
        let (x, y) = ((i % 12) as i32, (i / 12) as i32);
        let mut v = [0i32; 4];
        for c in 0..4 {
            v[c] = match kind {
                0 => r.range(256) as i32,
                1 => base[c] + r.range(9) as i32 - 4,
                2 => base[c] + (x * dx + y * dy) * (c as i32 + 1) / 4 + r.range(5) as i32 - 2,
                3 => base[c],
                4 => {
                    if r.next() & 1 != 0 {
                        base[c]
                    } else {
                        base2[c]
                    }
                }
                5 => {
                    let t = ((i * 255) / n) as i32;
                    base[c] + ((base2[c] - base[c]) * t) / 255
                }
                6 => {
                    let g = base[0] + x * dx / 2 + y * dy / 2;
                    if c == 3 { base[3] } else { g }
                }
                7 => r.range(2) as i32 * 255,
                8 => {
                    if c == 3 {
                        r.range(256) as i32
                    } else {
                        base[c] + r.range(33) as i32 - 16
                    }
                }
                _ => {
                    if (x + y) & 1 != 0 {
                        base[c]
                    } else {
                        base2[c] + r.range(3) as i32
                    }
                }
            };
        }
        if !alpha && kind != 8 {
            v[3] = 255;
        }
        px[i as usize] = v.map(|x| x.clamp(0, 255) as u8);
    }
    n
}

fn hash_stats(g: &mut Group, ps: &PixelStats) {
    g.u32(ps.num_pixels);
    g.bytes(&ps.min);
    g.bytes(&ps.max);
    for c in 0..4 {
        g.f32(ps.min_f[c]);
        g.f32(ps.max_f[c]);
        g.f32(ps.mean_f[c]);
        g.f32(ps.mean_rel_axis4[c]);
    }
    for c in 0..3 {
        g.f32(ps.mean_rel_axis3[c]);
        g.f32(ps.zero_rel_axis3[c]);
    }
    g.u32(ps.has_alpha as u32);
    for s in &ps.rgba_stats {
        g.u32(s.n);
        for v in [s.total, s.avg, s.var, s.std_dev, s.min, s.max, s.range] {
            g.f32(v);
        }
    }
    for i in 0..ps.num_pixels as usize {
        for c in 0..4 {
            g.f32(ps.pixels_f[i][c]);
        }
    }
    g.count += 1;
}

fn pick_ccs(r: &mut Lcg, cem: u32) -> i32 {
    match cem {
        0 => -1,
        4 => {
            if r.next() & 1 != 0 {
                3
            } else {
                -1
            }
        }
        6 | 8 | 9 => r.range(4) as i32 - 1,
        _ => r.range(5) as i32 - 1,
    }
}

fn pick_weight_range(r: &mut Lcg) -> u32 {
    let v = r.range(13);
    if v == 12 { BISE_64_LEVELS } else { v }
}

fn num_weight_levels(wr: u32) -> u32 {
    if wr == BISE_64_LEVELS {
        65
    } else {
        ah::get_ise_levels(wr)
    }
}

fn comp_weights(r: &mut Lcg) -> [u32; 4] {
    if r.next() & 1 != 0 {
        [9, 11, 1, 11]
    } else {
        [1, 1, 1, 1]
    }
}

fn run() -> BTreeMap<String, Group> {
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    global_init();

    const NUM_BLOCKS: u32 = 480;
    let mut stats = vec![PixelStats::default(); NUM_BLOCKS as usize];
    {
        let g = groups.entry("pixel_stats".into()).or_default();
        for b in 0..NUM_BLOCKS {
            let mut px = [[0u8; 4]; 144];
            let n = make_block(b, &mut px);
            let ps = &mut stats[b as usize];
            ps.clear();
            ps.init(n, &px[..n as usize]);
            hash_stats(g, ps);
        }
    }

    // blue contraction, weight ISE deltas
    {
        let mut r = Lcg(1);
        let g = groups.entry("blue_contract".into()).or_default();
        for _ in 0..2000 {
            let mut c = [0u8; 4];
            for k in 0..4 {
                c[k] = r.range(256) as u8;
            }
            let eb = r.range(256) as i32;
            let mut clamped = false;
            let e = blue_contract_enc(c, &mut clamped, eb);
            let d = blue_contract_dec(e[0] as i32, e[1] as i32, e[2] as i32, e[3] as i32);
            g.bytes(&e);
            g.bytes(&d);
            g.u32(clamped as u32);
            g.count += 1;
        }
        let gw = groups.entry("apply_delta_weight".into()).or_default();
        for wr in 0..=11 {
            for v in 0..ah::get_ise_levels(wr) {
                for d in -3..=3 {
                    gw.u32(apply_delta_to_bise_weight_val(wr, v as i32, d) as u32);
                    gw.count += 1;
                }
            }
        }
    }

    // get_colors / get_colors_raw_weights / decode_endpoints
    {
        let mut r = Lcg(2);
        let mut gc = Group::default();
        let mut gd = Group::default();
        for _ in 0..3000 {
            let cem = LDR_CEMS[r.range(8) as usize];
            let er = 4 + r.range(17);
            let levels = ah::get_ise_levels(er);
            let mut ev = [0u8; 8];
            for k in 0..8 {
                ev[k] = r.range(levels) as u8;
            }
            let srgb = r.next() & 1 != 0;
            let (mut l, mut h) = ([0u8; 4], [0u8; 4]);
            let mut scale = -1.0f32;
            decode_endpoints(cem, &ev, er, &mut l, &mut h, Some(&mut scale));
            gd.bytes(&l);
            gd.bytes(&h);
            gd.f32(scale);
            gd.count += 1;

            let mut cols = [[0u8; 4]; 65];
            let wr = r.range(12);
            let n = get_colors_cem(cem, &ev, er, wr, &mut cols, srgb);
            gc.u32(n);
            gc.bytes(cols[..n as usize].as_flattened());
            let n = get_colors_raw_weights_cem(cem, &ev, er, &mut cols, srgb);
            gc.u32(n);
            gc.bytes(cols[..n as usize].as_flattened());
            let mut l8 = [0u8; 4];
            let mut h8 = [0u8; 4];
            for k in 0..3 {
                l8[k] = r.range(256) as u8;
            }
            l8[3] = if r.next() & 1 != 0 {
                255
            } else {
                r.range(256) as u8
            };
            for k in 0..3 {
                h8[k] = r.range(256) as u8;
            }
            h8[3] = if l8[3] == 255 {
                255
            } else {
                r.range(256) as u8
            };
            let n = get_colors(&l8, &h8, wr, &mut cols, srgb);
            gc.u32(n);
            gc.bytes(cols[..n as usize].as_flattened());
            gc.count += 1;

            let mut ev20 = [0u8; 8];
            for k in 0..8 {
                ev20[k] = r.range(256) as u8;
            }
            decode_endpoints_ise20(cem, &ev20, &mut l, &mut h);
            gd.bytes(&l);
            gd.bytes(&h);
        }
        groups.insert("get_colors".into(), gc);
        groups.insert("decode_endpoints".into(), gd);
    }

    // eval_solution / eval_solution_dp
    {
        let mut r = Lcg(3);
        let mut gs = Group::default();
        let mut gp = Group::default();
        for _ in 0..3000 {
            let ps = &stats[r.range(NUM_BLOCKS) as usize];
            let cem = LDR_CEMS[r.range(8) as usize];
            let er = 4 + r.range(17);
            let levels = ah::get_ise_levels(er);
            let mut ev = [0u8; 8];
            for k in 0..8 {
                ev[k] = r.range(levels) as u8;
            }
            let wr = pick_weight_range(&mut r);
            let decode_mode_srgb = r.next() & 1 != 0;
            let cw = comp_weights(&mut r);
            let exhaustive = r.range(4) == 0;
            let mut forced0 = [0u8; 144];
            let mut forced1 = [0u8; 144];
            let nwl = num_weight_levels(wr);
            for k in 0..144 {
                forced0[k] = r.range(nwl) as u8;
                forced1[k] = r.range(nwl) as u8;
            }
            let fmode = r.range(4);
            let p = CemEncodeParams {
                comp_weights: cw,
                decode_mode_srgb,
                use_exhaustive_weight_eval: exhaustive,
                forced_weight_vals0: if fmode == 1 || fmode == 2 {
                    Some(&forced0)
                } else {
                    None
                },
                forced_weight_vals1: if fmode >= 2 { Some(&forced1) } else { None },
                ..Default::default()
            };
            let mut w0 = [0xCDu8; 144];
            let mut w1 = [0xCDu8; 144];
            let e = eval_solution(ps, cem, &ev, er, &mut w0, wr, &p);
            gs.u64(e);
            gs.bytes(&w0);
            gs.count += 1;

            let ccs = r.range(4);
            w0 = [0xCD; 144];
            w1.fill(0xCD);
            let e = eval_solution_dp(ps, cem, ccs, &ev, er, &mut w0, &mut w1, wr, &p);
            gp.u64(e);
            gp.bytes(&w0);
            gp.bytes(&w1);
            gp.count += 1;
        }
        groups.insert("eval_solution".into(), gs);
        groups.insert("eval_solution_dp".into(), gp);
    }

    // cem_encode_pixels
    {
        let mut r = Lcg(4);
        for cem in LDR_CEMS {
            let mut g = Group::default();
            let mut gf = Group::default();
            for _ in 0..4000 {
                let ps = &stats[r.range(NUM_BLOCKS) as usize];
                let ccs = pick_ccs(&mut r, cem);
                let er = 4 + r.range(17);
                let wr = pick_weight_range(&mut r);
                let decode_mode_srgb = r.next() & 1 != 0;
                let cw = comp_weights(&mut r);
                let exhaustive = r.range(4) == 0;
                let max_ls_passes = [0, 1, 3, 5][r.range(4) as usize];
                let refine = [0, 0, 1, 5, 17, 20][r.range(6) as usize];
                let nudging = r.next() & 1 != 0;
                let ep_refine = r.next() & 1 != 0;
                let use_bc = r.next() & 1 != 0;
                let cm = r.range(8);
                let cur_err = if cm == 0 {
                    r.range(2000) as u64
                } else {
                    u64::MAX
                };
                let forced = r.range(5) == 0;
                let mut forced0 = [0u8; 144];
                let mut forced1 = [0u8; 144];
                let nwl = num_weight_levels(wr);
                for k in 0..144 {
                    forced0[k] = r.range(nwl) as u8;
                    forced1[k] = r.range(nwl) as u8;
                }
                let p = CemEncodeParams {
                    comp_weights: cw,
                    decode_mode_srgb,
                    use_exhaustive_weight_eval: exhaustive,
                    max_ls_passes,
                    total_weight_refine_passes: refine,
                    worst_weight_nudging_flag: nudging,
                    endpoint_refinement_flag: ep_refine,
                    forced_weight_vals0: if forced { Some(&forced0) } else { None },
                    forced_weight_vals1: if forced && ccs >= 0 {
                        Some(&forced1)
                    } else {
                        None
                    },
                };
                let mut ev = [0xCDu8; 8];
                let mut w0 = [0xCDu8; 144];
                let mut w1 = [0xCDu8; 144];
                if cm == 0 {
                    // The encoders start refinement from the caller's buffers when not
                    // improved: keep them valid.
                    let el = ah::get_ise_levels(er);
                    for k in 0..8 {
                        ev[k] = (forced1[k] as u32 % el) as u8;
                    }
                    w0 = forced0;
                    w1 = forced1;
                }
                let mut try_direct = true;
                let e = cem_encode_pixels(
                    cem,
                    ccs,
                    ps,
                    &p,
                    er,
                    wr,
                    &mut ev,
                    &mut w0,
                    if ccs >= 0 { Some(&mut w1) } else { None },
                    cur_err,
                    use_bc,
                    Some(&mut try_direct),
                );
                let gg = if forced { &mut gf } else { &mut g };
                gg.u64(e);
                gg.bytes(&ev);
                gg.bytes(&w0);
                gg.bytes(&w1);
                gg.u32(try_direct as u32);
                gg.count += 1;
            }
            groups.insert(format!("cem_encode_pixels_{cem}"), g);
            groups.insert(format!("cem_encode_pixels_forced_{cem}"), gf);
        }
    }

    // cem_surrogate_encode_pixels
    {
        let mut r = Lcg(5);
        for cem in [0u32, 4, 6, 8, 10, 12, 9] {
            let mut g = Group::default();
            for _ in 0..3000 {
                let ps = &stats[r.range(NUM_BLOCKS) as usize];
                let mut ccs = pick_ccs(&mut r, cem);
                if cem == 4 && ccs >= 0 {
                    ccs = 3;
                }
                let er = 4 + r.range(17);
                let wr = pick_weight_range(&mut r);
                let p = CemEncodeParams {
                    comp_weights: comp_weights(&mut r),
                    ..Default::default()
                };
                let flags = r.range(4);
                let (mut l, mut h) = ([0f32; 4], [0f32; 4]);
                let mut s = 0f32;
                let mut w0 = [-1f32; 144];
                let mut w1 = [-1f32; 144];
                let e = cem_surrogate_encode_pixels(
                    cem, ccs, ps, &p, er, wr, &mut l, &mut h, &mut s, &mut w0, &mut w1, flags,
                );
                g.f32(e);
                g.f32(s);
                for c in 0..4 {
                    g.f32(l[c]);
                    g.f32(h[c]);
                }
                for k in 0..144 {
                    g.f32(w0[k]);
                    g.f32(w1[k]);
                }
                g.count += 1;
            }
            groups.insert(format!("cem_surrogate_encode_pixels_{cem}"), g);
        }

        let mut gq = Group::default();
        for _ in 0..2000 {
            let e = r.range(1000001) as f32 * (1.0f32 / 1000000.0);
            let levels = ah::get_ise_levels(4 + r.range(17));
            let flags = r.range(2);
            gq.f32(surrogate_quant_endpoint_val(e, levels, flags));
            let v = [e, 1.0 - e, e * 0.5, e * e];
            let q = surrogate_quant_endpoint(&v, levels, flags);
            for c in 0..4 {
                gq.f32(q[c]);
            }
            gq.count += 1;
        }
        groups.insert("surrogate_quant".into(), gq);
    }
    groups
}

#[test]
fn astc_ldr_common_golden() {
    let groups = run();
    let mut checked = 0;
    let mut bad = Vec::new();
    for line in GOLDEN.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 3 {
            continue;
        }
        let Some(g) = groups.get(f[0]) else {
            bad.push(format!("{}: missing", f[0]));
            continue;
        };
        let got = format!("{:016x}", g.h);
        let count = g.count.to_string();
        if got != f[1] || count != f[2] {
            bad.push(format!(
                "{}: got {} {} want {} {}",
                f[0], got, count, f[1], f[2]
            ));
        }
        checked += 1;
    }
    assert!(bad.is_empty(), "mismatches:\n{}", bad.join("\n"));
    assert_eq!(checked, groups.len());
}
