// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of basisu `etc1f` (`basisu_transcoder.cpp`): the real-time ETC1 encoder used when
//! transcoding ASTC LDR / XUASTC LDR to ETC1 and ETC2 RGBA.
//!
//! The C++ `pack_etc1_state` only memoizes the last solid block, so it is omitted (same output).

use super::etc::{ETC1_INTEN_TABLES, SELECTOR_INDEX_TO_ETC1};
use super::etc1f_tables::ETC1_MOD_TABS;
use std::sync::OnceLock;

const NUM_SOLID_MODS: usize = 4;

/// Tables built by `etc1f::init`.
struct Tables {
    nearest5: [u8; 256],
    nearest4: [u8; 256],
    /// `[desired8][mod][sel]`
    solid5_base: [[[u8; 4]; NUM_SOLID_MODS]; 256],
    solid5_err: [[[u8; 4]; NUM_SOLID_MODS]; 256],
    solid4_base: [[[u8; 4]; NUM_SOLID_MODS]; 256],
    solid4_err: [[[u8; 4]; NUM_SOLID_MODS]; 256],
    solid_gray: [[u8; 8]; 256],
}

fn dequant4(v: u32) -> i32 {
    ((v << 4) | v) as i32
}

fn dequant5(v: u32) -> i32 {
    ((v << 3) | (v >> 2)) as i32
}

fn expand4(v: i32) -> i32 {
    (v << 4) | v
}

fn expand5(v: i32) -> i32 {
    (v << 3) | (v >> 2)
}

fn clamp255(v: i32) -> i32 {
    v.clamp(0, 255)
}

/// Best `bits`-bit base per (desired value, modifier table, selector) and its squared error.
fn solid_tables(
    bits: u32,
) -> (
    [[[u8; 4]; NUM_SOLID_MODS]; 256],
    [[[u8; 4]; NUM_SOLID_MODS]; 256],
) {
    let mut base = [[[0u8; 4]; NUM_SOLID_MODS]; 256];
    let mut err = [[[0u8; 4]; NUM_SOLID_MODS]; 256];
    for desired in 0..256i32 {
        for m in 0..NUM_SOLID_MODS {
            for sel in 0..4 {
                let (mut best_err, mut best_base) = (i32::MAX, 0u32);
                for b in 0..(1u32 << bits) {
                    let d = if bits == 5 { dequant5(b) } else { dequant4(b) };
                    let e = (clamp255(d + ETC1_INTEN_TABLES[m][sel]) - desired).abs();
                    if e < best_err {
                        best_err = e;
                        best_base = b;
                        if best_err == 0 {
                            break;
                        }
                    }
                }
                base[desired as usize][m][sel] = best_base as u8;
                err[desired as usize][m][sel] = (best_err * best_err).min(255) as u8;
            }
        }
    }
    (base, err)
}

fn nearest(bits: u32) -> [u8; 256] {
    let mut out = [0u8; 256];
    for (i, o) in out.iter_mut().enumerate() {
        let (mut best_e, mut best) = (i32::MAX, 0);
        for s in 0..(1i32 << bits) {
            let rec = if bits == 5 { expand5(s) } else { expand4(s) };
            let e = (rec - i as i32).abs();
            if e < best_e {
                best_e = e;
                best = s;
            }
        }
        *o = best as u8;
    }
    out
}

fn tables() -> &'static Tables {
    static T: OnceLock<Box<Tables>> = OnceLock::new();
    T.get_or_init(|| {
        let (solid5_base, solid5_err) = solid_tables(5);
        let (solid4_base, solid4_err) = solid_tables(4);
        let mut t = Box::new(Tables {
            nearest5: nearest(5),
            nearest4: nearest(4),
            solid5_base,
            solid5_err,
            solid4_base,
            solid4_err,
            solid_gray: [[0; 8]; 256],
        });
        for i in 0..256 {
            let mut b = [0u8; 8];
            pack_solid_with(&t, &mut b, [i as u8, i as u8, i as u8, 255]);
            t.solid_gray[i] = b;
        }
        t
    })
}

fn dequant_d3(v: u8) -> i32 {
    (((v << 5) as i8) >> 5) as i32
}

/// Base colors of the two subblocks: `(base8 per subblock, intensity table per subblock)`.
fn block_bases(block: &[u8; 8], channels: usize) -> ([[i32; 3]; 2], [usize; 2]) {
    let b3 = block[3];
    let mut base = [[0i32; 3]; 2];
    for c in 0..channels {
        let v = block[c];
        if b3 & 2 != 0 {
            base[0][c] = dequant5(u32::from(v >> 3));
            base[1][c] = dequant5(((v >> 3) as i32 + dequant_d3(v & 7)).clamp(0, 31) as u32);
        } else {
            base[0][c] = dequant4(u32::from(v >> 4));
            base[1][c] = dequant4(u32::from(v & 15));
        }
    }
    (base, [(b3 >> 5) as usize, ((b3 >> 2) & 7) as usize])
}

/// `get_block_colors`: the 4 RGB colors of each subblock.
fn block_colors(block: &[u8; 8]) -> [[[i32; 3]; 4]; 2] {
    let (base, tab) = block_bases(block, 3);
    let mut out = [[[0i32; 3]; 4]; 2];
    for s in 0..2 {
        for i in 0..4 {
            let d = ETC1_INTEN_TABLES[tab[s]][i];
            for c in 0..3 {
                out[s][i][c] = clamp255(base[s][c] + d);
            }
        }
    }
    out
}

/// `get_block_colors_y`: the 4 gray levels of each subblock.
fn block_colors_y(block: &[u8; 8]) -> [[i32; 4]; 2] {
    let (base, tab) = block_bases(block, 1);
    let mut out = [[0i32; 4]; 2];
    for s in 0..2 {
        for i in 0..4 {
            out[s][i] = clamp255(base[s][0] + ETC1_INTEN_TABLES[tab[s]][i]);
        }
    }
    out
}

/// `corr_round_555` / `corr_round_444`: per-channel floor or ceil minimizing the spread of the
/// rounding errors between channels.
fn corr_round(rgb: [i32; 3], bits: u32) -> [i32; 3] {
    let max = (1i32 << bits) - 1;
    let expand = |v: i32| if bits == 5 { expand5(v) } else { expand4(v) };
    let lo = rgb.map(|v| (v * max) / 255);
    let hi = lo.map(|l| if l < max { l + 1 } else { max });
    let q8 = [0, 1, 2].map(|c| [expand(lo[c]), expand(hi[c])]);
    let cost = |m: usize| {
        let t = [q8[0][m & 1], q8[1][(m >> 1) & 1], q8[2][(m >> 2) & 1]];
        let e = [rgb[0] - t[0], rgb[1] - t[1], rgb[2] - t[2]];
        let j = (e[0] - e[1]).pow(2) + (e[1] - e[2]).pow(2) + (e[2] - e[0]).pow(2);
        (j, t)
    };
    let (mut best_j, mut best) = cost(0);
    for m in 1..8 {
        let (j, t) = cost(m);
        if j < best_j {
            best_j = j;
            best = t;
        }
    }
    best.map(|v| v >> (8 - bits))
}

fn pack_solid_with(t: &Tables, block: &mut [u8; 8], color: [u8; 4]) {
    let [r8, g8, b8, _] = color.map(usize::from);
    const RW: u32 = 2;
    const GW: u32 = 4;
    let (mut best_err, mut best_mod, mut best_sel, mut best4) = (u32::MAX, 0usize, 0usize, false);
    'search: for m in 0..NUM_SOLID_MODS {
        for sel in 0..4 {
            let e5 = RW * u32::from(t.solid5_err[r8][m][sel])
                + GW * u32::from(t.solid5_err[g8][m][sel])
                + u32::from(t.solid5_err[b8][m][sel]);
            if e5 < best_err {
                (best_err, best_mod, best_sel, best4) = (e5, m, sel, false);
                if best_err == 0 {
                    break 'search;
                }
            }
            let e4 = RW * u32::from(t.solid4_err[r8][m][sel])
                + GW * u32::from(t.solid4_err[g8][m][sel])
                + u32::from(t.solid4_err[b8][m][sel]);
            if e4 < best_err {
                (best_err, best_mod, best_sel, best4) = (e4, m, sel, true);
            }
        }
    }
    for (c, v) in [r8, g8, b8].into_iter().enumerate() {
        block[c] = if best4 {
            let b = t.solid4_base[v][best_mod][best_sel];
            b | (b << 4)
        } else {
            t.solid5_base[v][best_mod][best_sel] << 3
        };
    }
    let diff = u8::from(!best4);
    block[3] = (diff << 1) | ((best_mod as u8) << 5) | ((best_mod as u8) << 2);
    let sels = SELECTOR_INDEX_TO_ETC1[best_sel];
    let lb = if sels & 2 != 0 { 0xFF } else { 0 };
    let hb = if sels & 1 != 0 { 0xFF } else { 0 };
    block[4..8].copy_from_slice(&[lb, lb, hb, hb]);
}

const VI: [usize; 16] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1];
const HI: [usize; 16] = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];
const SUBSETS: [[usize; 16]; 2] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
];
/// `[flip][subblock][sel] -> (low, high)` selector bit masks.
const SEL_BITMASKS: [[u16; 2]; 16] = [
    [0xff, 0xff],
    [0x0, 0xff],
    [0x0, 0x0],
    [0xff, 0x0],
    [0xff00, 0xff00],
    [0x0, 0xff00],
    [0x0, 0x0],
    [0xff00, 0x0],
    [0x3333, 0x3333],
    [0x0, 0x3333],
    [0x0, 0x0],
    [0x3333, 0x0],
    [0xcccc, 0xcccc],
    [0x0, 0xcccc],
    [0x0, 0x0],
    [0xcccc, 0x0],
];
const TRAN: [u16; 4] = [1, 0, 2, 3];

fn write_sel_masks(block: &mut [u8; 8], l: u16, h: u16) {
    block[7] = l as u8;
    block[6] = (l >> 8) as u8;
    block[5] = h as u8;
    block[4] = (h >> 8) as u8;
}

/// Best (error << 5 | mod << 2 | sel) over the solid tables, per subblock: `err_of(table)`
/// yields the 4 per-selector table errors for a modifier.
fn best_solid(errs: impl Fn(usize) -> [u32; 4]) -> (u32, usize, usize) {
    let mut best = u32::MAX;
    for m in 0..NUM_SOLID_MODS {
        let e = errs(m);
        let m4 = (m << 2) as u32;
        let c = [0, 1, 2, 3].map(|s| (e[s] << 5) + (m4 + s as u32));
        best = best.min(c[0].min(c[1]).min(c[2].min(c[3])));
    }
    (best >> 5, ((best >> 2) & 7) as usize, (best & 3) as usize)
}

/// `pack_etc1_solid_subblocks`.
fn pack_solid_subblocks(t: &Tables, block: &mut [u8; 8], means: [[u8; 3]; 2], flip: usize) {
    const RW: u32 = 2;
    const GW: u32 = 4;
    let mut r5 = [(0u32, 0usize, 0usize); 2];
    let mut r4 = [(0u32, 0usize, 0usize); 2];
    let mut base5 = [[0u8; 3]; 2];
    let mut base4 = [[0u8; 3]; 2];
    for s in 0..2 {
        let [r8, g8, b8] = means[s].map(usize::from);
        let err = |tab: &[[[u8; 4]; NUM_SOLID_MODS]; 256], m: usize| {
            [0, 1, 2, 3].map(|sel| {
                RW * u32::from(tab[r8][m][sel])
                    + GW * u32::from(tab[g8][m][sel])
                    + u32::from(tab[b8][m][sel])
            })
        };
        r5[s] = best_solid(|m| err(&t.solid5_err, m));
        r4[s] = best_solid(|m| err(&t.solid4_err, m));
        for (c, v) in [r8, g8, b8].into_iter().enumerate() {
            base5[s][c] = t.solid5_base[v][r5[s].1][r5[s].2];
            base4[s][c] = t.solid4_base[v][r4[s].1][r4[s].2];
        }
    }
    let use_abs = r4[0].0 + r4[1].0 < r5[0].0 + r5[1].0
        || (0..3).any(|c| {
            let d = i32::from(base5[1][c]) - i32::from(base5[0][c]);
            !(-4..=3).contains(&d)
        });
    let flip8 = flip as u8;
    let sels = if use_abs {
        for c in 0..3 {
            block[c] = base4[1][c] | (base4[0][c] << 4);
        }
        block[3] = flip8 | ((r4[0].1 as u8) << 5) | ((r4[1].1 as u8) << 2);
        [r4[0].2, r4[1].2]
    } else {
        for c in 0..3 {
            let d = (i32::from(base5[1][c]) - i32::from(base5[0][c])) & 7;
            block[c] = d as u8 | (base5[0][c] << 3);
        }
        block[3] = flip8 | 2 | ((r5[0].1 as u8) << 5) | ((r5[1].1 as u8) << 2);
        [r5[0].2, r5[1].2]
    };
    let (mut l, mut h) = (0u16, 0u16);
    for s in 0..2 {
        let m = SEL_BITMASKS[flip * 8 + s * 4 + sels[s]];
        l |= m[0];
        h |= m[1];
    }
    write_sel_masks(block, l, h);
}

/// `pack_etc1_grayscale_solid_subblocks`.
fn pack_gray_solid_subblocks(t: &Tables, block: &mut [u8; 8], means: [u8; 2], flip: usize) {
    let mut r5 = [(0u32, 0usize, 0usize); 2];
    let mut r4 = [(0u32, 0usize, 0usize); 2];
    let mut base5 = [0u8; 2];
    let mut base4 = [0u8; 2];
    for s in 0..2 {
        let y8 = usize::from(means[s]);
        r5[s] = best_solid(|m| t.solid5_err[y8][m].map(u32::from));
        r4[s] = best_solid(|m| t.solid4_err[y8][m].map(u32::from));
        base5[s] = t.solid5_base[y8][r5[s].1][r5[s].2];
        base4[s] = t.solid4_base[y8][r4[s].1][r4[s].2];
    }
    let use_abs = r4[0].0 + r4[1].0 < r5[0].0 + r5[1].0
        || !(-4..=3).contains(&(i32::from(base5[1]) - i32::from(base5[0])));
    let flip8 = flip as u8;
    let sels = if use_abs {
        let v = base4[1] | (base4[0] << 4);
        block[..3].fill(v);
        block[3] = flip8 | ((r4[0].1 as u8) << 5) | ((r4[1].1 as u8) << 2);
        [r4[0].2, r4[1].2]
    } else {
        let d = (i32::from(base5[1]) - i32::from(base5[0])) & 7;
        block[..3].fill(d as u8 | (base5[0] << 3));
        block[3] = flip8 | 2 | ((r5[0].1 as u8) << 5) | ((r5[1].1 as u8) << 2);
        [r5[0].2, r5[1].2]
    };
    let (mut l, mut h) = (0u16, 0u16);
    for s in 0..2 {
        let m = SEL_BITMASKS[flip * 8 + s * 4 + sels[s]];
        l |= m[0];
        h |= m[1];
    }
    write_sel_masks(block, l, h);
}

/// Selector bits from per-pixel luma `lum(i)` against each subblock's 4 levels `levels[s]`
/// (pairwise sums are compared, so `lum` is in the doubled scale).
fn selectors(block: &mut [u8; 8], flip: usize, levels: [[u32; 4]; 2], lum: impl Fn(usize) -> u32) {
    let (mut l, mut h) = (0u16, 0u16);
    for s in 0..2 {
        let y = levels[s];
        let (y01, y12, y23) = (y[0] + y[1], y[1] + y[2], y[2] + y[3]);
        let mut put = |i: usize, ofs: usize| {
            let v = lum(i);
            let t = TRAN[usize::from(v < y01) + usize::from(v < y12) + usize::from(v < y23)];
            l |= (t & 1) << ofs;
            h |= (t >> 1) << ofs;
        };
        if flip != 0 {
            for yy in 0..2 {
                for x in 0..4 {
                    put(x + (s * 2 + yy) * 4, s * 2 + yy + x * 4);
                }
            }
        } else {
            for x in 0..2 {
                for yy in 0..4 {
                    put(s * 2 + x + yy * 4, (s * 2 + x) * 4 + yy);
                }
            }
        }
    }
    write_sel_masks(block, l, h);
}

/// Subblock stats for the non-solid path: mean, half span and modifier table per subblock.
fn mod_tables(var8: [i32; 2], sum8: [i32; 2], min: [i32; 2], max: [i32; 2]) -> ([i32; 2], [u8; 2]) {
    let mut mean = [0i32; 2];
    let mut tab = [0u8; 2];
    for i in 0..2 {
        let var = ((var8[i] << 3) - sum8[i] * sum8[i]).max(0);
        let stddev_y = (var as f32).sqrt() * (1.0f32 / 8.0f32);
        mean[i] = (sum8[i] + 4) >> 3;
        let half_span = (max[i] - mean[i]).max(mean[i] - min[i]);
        let sd = ((9.0f32 * (stddev_y / half_span.max(1) as f32)).ceil() as i32 - 1).clamp(0, 7);
        tab[i] = ETC1_MOD_TABS[half_span.clamp(1, 255) as usize][sd as usize];
    }
    (mean, tab)
}

/// `etc1f::pack_etc1_grayscale`: ETC1 block for 16 gray values.
pub fn pack_etc1_grayscale(block: &mut [u8; 8], px: &[u8; 16]) {
    let t = tables();
    if px.iter().all(|&p| p == px[0]) {
        *block = t.solid_gray[usize::from(px[0])];
        return;
    }
    let (mut ay, mut ay2) = ([0i32; 4], [0i32; 4]);
    for i in 0..16 {
        let y = i32::from(px[i]);
        for k in [VI[i], HI[i]] {
            ay[k] += y;
            ay2[k] += y * y;
        }
    }
    let std = [0, 1, 2, 3]
        .map(|i| (((ay2[i] << 3) - ay[i] * ay[i]).max(0) as f32 * (1.0f32 / 64.0f32)).sqrt());
    let flip = usize::from(std[2] + std[3] < std[0] + std[1]);
    let (mut var8, mut sum8) = ([0i32; 2], [0i32; 2]);
    let (mut min, mut max) = ([i32::MAX; 2], [i32::MIN; 2]);
    for i in 0..16 {
        let y = i32::from(px[i]);
        let s = SUBSETS[flip][i];
        var8[s] += y * y;
        sum8[s] += y;
        min[s] = min[s].min(y);
        max[s] = max[s].max(y);
    }
    if max[0] - min[0] < 8 && max[1] - min[1] < 8 {
        let means = [((sum8[0] + 4) / 8) as u8, ((sum8[1] + 4) / 8) as u8];
        if means[0] == means[1] {
            *block = t.solid_gray[usize::from(means[0])];
        } else {
            pack_gray_solid_subblocks(t, block, means, flip);
        }
        return;
    }
    let (mean, tab) = mod_tables(var8, sum8, min, max);
    let m5 = mean.map(|m| i32::from(t.nearest5[m as usize]));
    let d5 = m5[1] - m5[0];
    let flip8 = flip as u8;
    if !(-4..=3).contains(&d5) {
        let m4 = mean.map(|m| t.nearest4[m as usize]);
        block[..3].fill(m4[1] | (m4[0] << 4));
        block[3] = flip8 | (tab[0] << 5) | (tab[1] << 2);
    } else {
        block[..3].fill((d5 & 7) as u8 | ((m5[0] as u8) << 3));
        block[3] = flip8 | 2 | (tab[0] << 5) | (tab[1] << 2);
    }
    let levels = block_colors_y(block).map(|s| s.map(|v| v as u32));
    selectors(block, flip, levels, |i| u32::from(px[i]) * 2);
}

/// `color_rgba::get_709_luma`.
fn luma_709(p: [u8; 4]) -> u8 {
    ((13938 * u32::from(p[0]) + 46869 * u32::from(p[1]) + 4729 * u32::from(p[2]) + 32768) >> 16)
        as u8
}

/// `etc1f::pack_etc1`: ETC1 block for 16 RGBA pixels (alpha ignored).
pub fn pack_etc1(block: &mut [u8; 8], px: &[[u8; 4]; 16]) {
    let t = tables();
    let rgb = |p: [u8; 4]| [p[0], p[1], p[2]];
    if px.iter().all(|&p| rgb(p) == rgb(px[0])) {
        pack_solid(block, px[0]);
        return;
    }
    let (mut ay, mut ay2, mut ac2) = ([0i32; 4], [0i32; 4], [0i32; 4]);
    let (mut total_c2, mut max_c2) = (0i32, 0i32);
    for i in 0..16 {
        let [r, g, b, _] = px[i].map(i32::from);
        let (rg, bg) = (r - g, b - g);
        let y = (r + g + b + 1) / 3;
        let c2 = rg * rg + bg * bg;
        total_c2 += c2;
        max_c2 = max_c2.max(c2);
        for k in [VI[i], HI[i]] {
            ay[k] += y;
            ay2[k] += y * y;
            ac2[k] += c2;
        }
    }
    // Low chroma: encode as grayscale.
    if total_c2 < 300 && max_c2 < 32 {
        let y = if total_c2 == 0 {
            px.map(|p| p[0])
        } else {
            px.map(luma_709)
        };
        pack_etc1_grayscale(block, &y);
        return;
    }
    let std_luma = [0, 1, 2, 3]
        .map(|i| (((ay2[i] << 3) - ay[i] * ay[i]).max(0) as f32 * (1.0f32 / 64.0f32)).sqrt());
    let std_chroma = [0, 1, 2, 3].map(|i| (ac2[i] as f32 * (1.0f32 / 8.0f32)).sqrt());
    let flip0 = (std_luma[0] + std_luma[1]) * 2.0 + (std_chroma[0] + std_chroma[1]);
    let flip1 = (std_luma[2] + std_luma[3]) * 2.0 + (std_chroma[2] + std_chroma[3]);
    let flip = usize::from(flip1 < flip0);

    let (mut var8, mut sum8, mut sum_rgb) = ([0i32; 2], [0i32; 2], [[0i32; 3]; 2]);
    let (mut min, mut max) = ([i32::MAX; 2], [i32::MIN; 2]);
    for i in 0..16 {
        let [r, g, b, _] = px[i].map(i32::from);
        let y = (r + g + b + 1) / 3;
        let s = SUBSETS[flip][i];
        var8[s] += y * y;
        sum8[s] += y;
        sum_rgb[s][0] += r;
        sum_rgb[s][1] += g;
        sum_rgb[s][2] += b;
        min[s] = min[s].min(y);
        max[s] = max[s].max(y);
    }
    if max[0] - min[0] < 8 && max[1] - min[1] < 8 {
        let means = sum_rgb.map(|s| s.map(|v| ((v + 4) / 8) as u8));
        if means[0] == means[1] {
            let [r, g, b] = means[0];
            pack_solid(block, [r, g, b, 255]);
        } else {
            pack_solid_subblocks(t, block, means, flip);
        }
        return;
    }
    let (_, tab) = mod_tables(var8, sum8, min, max);
    let mean_rgb = sum_rgb.map(|s| s.map(|v| (v + 4) >> 3));
    let m5 = mean_rgb.map(|m| corr_round(m, 5));
    let d5 = [0, 1, 2].map(|c| m5[1][c] - m5[0][c]);
    // Unsigned in C++: a negative delta + 4 also selects the 4-bit absolute mode.
    let z = ((d5[0] + 4) | (d5[1] + 4) | (d5[2] + 4)) as u32;
    let flip8 = flip as u8;
    if z > 7 {
        let m4 = mean_rgb.map(|m| corr_round(m, 4));
        for c in 0..3 {
            block[c] = (m4[1][c] | (m4[0][c] << 4)) as u8;
        }
        block[3] = flip8 | (tab[0] << 5) | (tab[1] << 2);
    } else {
        for c in 0..3 {
            block[c] = ((d5[c] & 7) | (m5[0][c] << 3)) as u8;
        }
        block[3] = flip8 | 2 | (tab[0] << 5) | (tab[1] << 2);
    }
    let levels =
        block_colors(block).map(|s| s.map(|c| (c[0] * 54 + c[1] * 183 + c[2] * 19) as u32));
    selectors(block, flip, levels, |i| {
        let [r, g, b, _] = px[i].map(u32::from);
        r * 108 + g * 366 + b * 38
    });
}

/// `etc1f::pack_etc1_solid` (outside of `init`).
fn pack_solid(block: &mut [u8; 8], color: [u8; 4]) {
    let t = tables();
    if color[0] == color[1] && color[0] == color[2] {
        *block = t.solid_gray[usize::from(color[0])];
        return;
    }
    pack_solid_with(t, block, color);
}
