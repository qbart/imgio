// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of basisu `bc7f::fast_pack_astc` (`basisu_transcoder.cpp`): latent to latent BC7 to
//! ASTC LDR 4x4 transcoding (used by XUBC7 -> ASTC 4x4). Blocks whose 2/3-subset BC7 pattern has
//! no ASTC equivalent are re-encoded with the `bc7f` encoder in ASTC compatible mode first.
//!
//! Also holds the parts of `astc_ldr_t` it needs (`convert_endpoints_across_cems` from RGB
//! direct to base+scale / base+offset, `pack_base_offset`, `requantize_ise_endpoints` for
//! those CEMs and the base+offset nudge tables). Float math follows the C++ in `f32`.

use super::astc_6x6_hdr::quantize_preserve2;
use super::astc_helpers::{
    self as ah, BISE_4_LEVELS, BISE_5_LEVELS, BISE_6_LEVELS, BISE_8_LEVELS, BISE_12_LEVELS,
    BISE_16_LEVELS, BISE_24_LEVELS, BISE_32_LEVELS, BISE_40_LEVELS, BISE_48_LEVELS, BISE_64_LEVELS,
    BISE_96_LEVELS, BISE_160_LEVELS, BISE_192_LEVELS, BISE_256_LEVELS,
    CEM_LDR_RGB_BASE_PLUS_OFFSET, CEM_LDR_RGB_BASE_SCALE, CEM_LDR_RGB_DIRECT, CEM_LDR_RGBA_DIRECT,
    LogAstcBlock,
};
use super::bc7f::{
    PACK_BC7_FLAG_ASTC_COMPATIBLE, PACK_BC7_FLAG_PBIT_OPT, PACK_BC7_FLAG_PBIT_OPT_MODE6,
    PACK_BC7_FLAG_USE_2_SUBSETS_RGB, PACK_BC7_FLAG_USE_2_SUBSETS_RGBA,
    PACK_BC7_FLAG_USE_3_SUBSETS_RGB, PACK_BC7_FLAG_USE_DUAL_PLANE_RGB,
    PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA, PACK_BC7_FLAG_USE_TRIVIAL_MODE6, fast_pack_bc7_auto_rgba,
};
use super::bc7u::{self, LogBc7Block};
use super::uastc_tables::{
    ASTC_BC7_COMMON_PARTITIONS2, ASTC_BC7_COMMON_PARTITIONS3,
    ASTC_TO_BC7_PARTITION_INDEX_PERM_TABLES,
};
use std::sync::OnceLock;

type Px = [u8; 4];

/// `(int)x` as compiled for x86-64 (`cvttss2si`).
#[inline]
fn f2i(x: f32) -> i32 {
    if (-2147483648.0..2147483648.0).contains(&x) {
        x as i32
    } else {
        i32::MIN
    }
}

#[inline]
fn sq(v: i32) -> i32 {
    v * v
}

/// Endpoint quantization tables of one ISE range.
#[derive(Clone, Copy)]
struct Tabs {
    to: &'static [u8],
    from: &'static [u8],
}

impl Tabs {
    fn endpoint(range: u32) -> Self {
        let t = ah::dequant_tables().endpoint_tab(range);
        Tabs {
            to: &t.val_to_ise,
            from: &t.ise_to_val,
        }
    }

    #[inline]
    fn q(&self, v: u8) -> u8 {
        self.to.get(v as usize).copied().unwrap_or(0)
    }

    #[inline]
    fn dq(&self, i: u8) -> u8 {
        self.from.get(i as usize).copied().unwrap_or(0)
    }
}

#[inline]
fn weight_to_ise(range: u32) -> &'static [u8] {
    &ah::dequant_tables().weight_tab(range).val_to_ise
}

#[inline]
fn lookup(t: &[u8], i: i32) -> u8 {
    t.get(i as usize).copied().unwrap_or(0)
}

/// `bc7u::read_bits32` over a physical block.
#[inline]
fn read_bits(block: &[u8; 16], ofs: &mut u32, n: u32) -> u32 {
    let v = u128::from_le_bytes(*block);
    let r = ((v >> (*ofs & 127)) as u32) & ((1u64 << n) - 1) as u32;
    *ofs += n;
    r
}

#[inline]
fn dequant_p(val: u32, pbit: u32, val_bits: u32) -> u8 {
    let total = val_bits + 1;
    let mut v = (val << 1) | pbit;
    v <<= 8 - total;
    v |= v >> total;
    v as u8
}

#[inline]
fn dequant(val: u32, val_bits: u32) -> u8 {
    let mut v = val << (8 - val_bits);
    v |= v >> val_bits;
    v as u8
}

#[inline]
fn texel_partition(seed: u16, i: usize, subsets: i32) -> usize {
    ah::compute_texel_partition(
        seed as u32,
        (i & 3) as u32,
        (i >> 2) as u32,
        0,
        subsets,
        true,
    ) as usize
}

// ---------------------------------------------------------------------------------------------
// Weight grid selection.

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Grid {
    G4x4 = 0,
    G4x3 = 1,
    G3x4 = 2,
}

#[inline]
fn line_sse(l: [u8; 4]) -> u32 {
    let c0 = l[0] as i32;
    let c1 = (l[1] as i32 + l[2] as i32 + 1) >> 1;
    let c2 = l[3] as i32;
    let p1 = (5 * c0 + 11 * c1 + 8) >> 4;
    let p2 = (11 * c1 + 5 * c2 + 8) >> 4;
    let d1 = p1 - l[1] as i32;
    let d2 = p2 - l[2] as i32;
    (d1 * d1 + d2 * d2) as u32
}

fn sse_4x3(w: &[u8; 16]) -> u32 {
    (0..4)
        .map(|x| line_sse([w[x], w[x + 4], w[x + 8], w[x + 12]]))
        .sum()
}

fn sse_3x4(w: &[u8; 16]) -> u32 {
    (0..4)
        .map(|y| line_sse([w[y * 4], w[y * 4 + 1], w[y * 4 + 2], w[y * 4 + 3]]))
        .sum()
}

fn choose_grid(w: &[u8; 16], max_sse: u32) -> Grid {
    let (e43, e34) = (sse_4x3(w), sse_3x4(w));
    let (ok43, ok34) = (e43 <= max_sse, e34 <= max_sse);
    match (ok43, ok34) {
        (true, true) => {
            if e43 <= e34 {
                Grid::G4x3
            } else {
                Grid::G3x4
            }
        }
        (true, false) => Grid::G4x3,
        (false, true) => Grid::G3x4,
        _ => Grid::G4x4,
    }
}

#[inline]
fn avg(a: u8, b: u8) -> u8 {
    ((a as u32 + b as u32 + 1) >> 1) as u8
}

fn to_4x3(s: &[u8; 16], d: &mut [u8], inv: u8) {
    for x in 0..4 {
        d[x] = s[x] ^ inv;
        d[4 + x] = avg(s[4 + x], s[8 + x]) ^ inv;
        d[8 + x] = s[12 + x] ^ inv;
    }
}

fn to_3x4(s: &[u8; 16], d: &mut [u8], inv: u8) {
    for y in 0..4 {
        d[y * 3] = s[y * 4] ^ inv;
        d[y * 3 + 1] = avg(s[y * 4 + 1], s[y * 4 + 2]) ^ inv;
        d[y * 3 + 2] = s[y * 4 + 3] ^ inv;
    }
}

/// Applies a reduced grid choice (4x3 / 3x4) to `log`; false for 4x4.
fn apply_grid(log: &mut LogAstcBlock, grid: Grid, w: &[u8; 16], inv: u8) -> bool {
    match grid {
        Grid::G3x4 => {
            log.grid_width = 3;
            to_3x4(w, &mut log.weights, inv);
            true
        }
        Grid::G4x3 => {
            log.grid_height = 3;
            to_4x3(w, &mut log.weights, inv);
            true
        }
        Grid::G4x4 => false,
    }
}

const BC7_2SUBSETS_TO_COMMON_PATTERN: [i16; 53] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, -1, -1, 26, -1, -1, 27, 28, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 29,
];

fn set_solid(log: &mut LogAstcBlock, c: Px) {
    log.solid_color_flag_ldr = true;
    for (i, &v) in c.iter().enumerate() {
        log.set_solid_color(i, (v as u16) | ((v as u16) << 8));
    }
}

// ---------------------------------------------------------------------------------------------
// astc_ldr_t helpers.

fn blue_contract_enc(orig: Px, did_clamp: &mut bool, encoded_b: i32) -> Px {
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

fn bit_transfer_signed_enc(a: &mut i32, b: &mut i32) {
    let bit = (*b & 0x80) != 0;
    *b = (*b << 1) & 0xFF;
    *a &= 0x3F;
    *a <<= 1;
    if bit {
        *a |= 0x80;
    }
}

/// `g_base_ofs_nudges[range][pos=0, neg=1][ise]`.
fn base_ofs_nudges() -> &'static [[[u8; 256]; 2]; 21] {
    static T: OnceLock<[[[u8; 256]; 2]; 21]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[[0u8; 256]; 2]; 21];
        for range in ah::FIRST_VALID_ENDPOINT_ISE_RANGE..=ah::LAST_VALID_ENDPOINT_ISE_RANGE {
            let from = &ah::dequant_tables().endpoint_tab(range).ise_to_val;
            let levels = (ah::get_ise_levels(range) as usize)
                .min(from.len())
                .min(256);
            let dec = |i: usize| {
                let (mut a, mut b) = (from[i] as i32, 0);
                ah::bit_transfer_signed(&mut a, &mut b);
                (a, b)
            };
            for pn in 0..2 {
                let delta = if pn == 1 { -1 } else { 1 };
                for cur in 0..levels {
                    let (cur_a, cur_b) = dec(cur);
                    let (mut best_err, mut best) = (i32::MAX, 0usize);
                    for trial in 0..levels {
                        let (ta, tb) = dec(trial);
                        if cur_b != tb || ta == cur_a {
                            continue;
                        }
                        if (delta < 0 && ta > cur_a) || (delta > 0 && ta < cur_a) {
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
                    t[range as usize][pn][cur] = best as u8;
                }
            }
        }
        t
    })
}

/// `cem9_or_13_used_blue_contraction`.
fn base_ofs_used_bc(vals: &[u8], t: Tabs) -> bool {
    let mut d = [0i32; 6];
    for (o, &v) in d.iter_mut().zip(vals) {
        *o = t.dq(v) as i32;
    }
    let (a, b) = d.split_at_mut(1);
    ah::bit_transfer_signed(&mut b[0], &mut a[0]);
    let (a, b) = d.split_at_mut(3);
    ah::bit_transfer_signed(&mut b[0], &mut a[2]);
    let (a, b) = d.split_at_mut(5);
    ah::bit_transfer_signed(&mut b[0], &mut a[4]);
    d[1] + d[3] + d[5] < 0
}

/// `requantize_ise_endpoints` from 256 levels for CEM 6 (base+scale) or 9 (base+offset).
fn requantize_from_256(cem: u32, src: &[u8; 8], dst_range: u32, dst: &mut [u8]) -> bool {
    let n = ah::get_num_cem_values(cem) as usize;
    if dst.len() < n {
        return false;
    }
    if dst_range == BISE_256_LEVELS {
        dst[..n].copy_from_slice(&src[..n]);
        return true;
    }
    let t = Tabs::endpoint(dst_range);
    if cem == CEM_LDR_RGB_BASE_PLUS_OFFSET {
        for i in 0..n {
            dst[i] = if i & 1 != 0 {
                quantize_preserve2(dst_range, src[i])
            } else {
                t.q(src[i])
            };
        }
        let src_used_bc = base_ofs_used_bc(src, Tabs::endpoint(BISE_256_LEVELS));
        let mut quant_used_bc = base_ofs_used_bc(dst, t);
        if src_used_bc != quant_used_bc {
            let neg = !quant_used_bc;
            let nudges = &base_ofs_nudges()[(dst_range as usize).min(20)][neg as usize];
            let mut rover = 2usize;
            for _ in 0..5 {
                for j in 0..3 {
                    let i = (rover + j) % 3;
                    let cur = dst[1 + i * 2];
                    let new = nudges[cur as usize];
                    if new != cur {
                        dst[1 + i * 2] = new;
                        break;
                    }
                }
                quant_used_bc = base_ofs_used_bc(dst, t);
                if src_used_bc == quant_used_bc {
                    break;
                }
                rover += 1;
            }
        }
    } else {
        for i in 0..n {
            dst[i] = t.q(src[i]);
        }
    }
    true
}

/// `pack_base_offset` for CEM 9. Returns (status, base+offset clamped).
fn pack_base_offset(
    dst_range: u32,
    dst: &mut [u8],
    l: Px,
    h: Px,
    mut use_bc: bool,
) -> (bool, bool) {
    let mut bc_clamped = false;
    let mut base_ofs_clamped = false;
    let (mut pack_l, mut pack_h) = (l, h);
    if use_bc {
        let enc_l = blue_contract_enc(pack_l, &mut bc_clamped, pack_l[2] as i32);
        let enc_h = blue_contract_enc(pack_h, &mut bc_clamped, pack_h[2] as i32);
        if bc_clamped {
            use_bc = false;
        } else {
            pack_h = enc_l;
            pack_l = enc_h;
        }
    }
    let (mut dr, mut dg, mut db) = (0i32, 0i32, 0i32);
    let mut low_clamp = -32;
    for pass in 0..4 {
        let d = |c: usize| pack_h[c] as i32 - pack_l[c] as i32;
        base_ofs_clamped = false;
        let mut clampc = |v: i32| {
            let c = v.clamp(low_clamp, 31);
            if c != v {
                base_ofs_clamped = true;
            }
            c
        };
        dr = clampc(d(0));
        dg = clampc(d(1));
        db = clampc(d(2));
        let _da = clampc(d(3));
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
    let (mut v0, mut v2, mut v4) = (pack_l[0] as i32, pack_l[1] as i32, pack_l[2] as i32);
    let (mut v1, mut v3, mut v5) = (dr, dg, db);
    bit_transfer_signed_enc(&mut v1, &mut v0);
    bit_transfer_signed_enc(&mut v3, &mut v2);
    bit_transfer_signed_enc(&mut v5, &mut v4);
    let new8 = [
        v0 as u8, v1 as u8, v2 as u8, v3 as u8, v4 as u8, v5 as u8, 0, 0,
    ];
    let status = requantize_from_256(CEM_LDR_RGB_BASE_PLUS_OFFSET, &new8, dst_range, dst);
    (status, base_ofs_clamped)
}

/// `convert_endpoints_across_cems` from 256-level RGB direct endpoints to `dst_cem` (6 or 9),
/// `always_repack` and `auto_disable_blue_contraction_if_clamped` set. Returns (status,
/// base+offset clamped).
fn convert_from_rgb_direct(
    src: &[u8; 6],
    dst_cem: u32,
    dst_range: u32,
    dst: &mut [u8],
    use_bc: bool,
) -> (bool, bool) {
    let t256 = Tabs::endpoint(BISE_256_LEVELS);
    let dq = src.map(|v| t256.dq(v));
    let mut de = [[0i32; 2]; 4];
    ah::decode_endpoint(CEM_LDR_RGB_DIRECT, &mut de, &dq);
    let l: Px = [0, 1, 2, 3].map(|c| de[c][0] as u8);
    let h: Px = [0, 1, 2, 3].map(|c| de[c][1] as u8);
    match dst_cem {
        CEM_LDR_RGB_BASE_SCALE => {
            let (mut lc, mut hc) = (l, h);
            let sum = |p: &Px| p[0] as i32 + p[1] as i32 + p[2] as i32;
            if sum(&lc) > sum(&hc) {
                std::mem::swap(&mut lc, &mut hc);
            }
            let mut new8 = [0u8; 8];
            new8[..3].copy_from_slice(&hc[..3]);
            let id: i32 = (0..3).map(|c| lc[c] as i32 * hc[c] as i32).sum();
            let inrm: i32 = (0..3).map(|c| sq(hc[c] as i32)).sum();
            const IMAX_S: i32 = (1024 * 255) / 256;
            let mut iscale = IMAX_S;
            if inrm > 0 {
                iscale = (id * 1024) / inrm;
            }
            iscale = iscale.clamp(0, IMAX_S);
            iscale = ((iscale + 2) >> 2).clamp(0, 255);
            new8[3] = iscale as u8;
            (requantize_from_256(dst_cem, &new8, dst_range, dst), false)
        }
        CEM_LDR_RGB_BASE_PLUS_OFFSET => pack_base_offset(dst_range, dst, l, h, use_bc),
        _ => (false, false),
    }
}

/// `cem_encode_bc` (CEM 8 or 12): prefers blue contraction; returns true if the weights must
/// be inverted.
fn cem_encode_bc(cem: u32, dst: &mut [u8], src: &[Px], t: Tabs) -> bool {
    let (l, h) = (src[0], src[1]);
    let enc_lb = t.dq(t.q(l[2])) as i32;
    let enc_hb = t.dq(t.q(h[2])) as i32;
    let mut did_clamp = false;
    let mut enc_l = blue_contract_enc(l, &mut did_clamp, enc_lb);
    let mut enc_h = blue_contract_enc(h, &mut did_clamp, enc_hb);
    let mut desired_bc = true;
    if did_clamp {
        desired_bc = false;
        enc_l = l;
        enc_h = h;
    }
    let set = |dst: &mut [u8], a: &Px, b: &Px| {
        for c in 0..3 {
            dst[c * 2] = t.q(a[c]);
            dst[c * 2 + 1] = t.q(b[c]);
        }
        if cem == CEM_LDR_RGBA_DIRECT {
            dst[6] = t.q(a[3]);
            dst[7] = t.q(b[3]);
        }
    };
    let swap = |dst: &mut [u8]| {
        dst.swap(0, 1);
        dst.swap(2, 3);
        dst.swap(4, 5);
        if cem == CEM_LDR_RGBA_DIRECT {
            dst.swap(6, 7);
        }
    };
    let sums = |dst: &[u8]| {
        (
            t.dq(dst[0]) as u32 + t.dq(dst[2]) as u32 + t.dq(dst[4]) as u32,
            t.dq(dst[1]) as u32 + t.dq(dst[3]) as u32 + t.dq(dst[5]) as u32,
        )
    };
    set(dst, &enc_l, &enc_h);
    let (s0, s1) = sums(dst);
    let mut invert = false;
    let actual_bc = s1 < s0;
    if actual_bc != desired_bc {
        if desired_bc && s0 == s1 {
            // Disable BC, swapping makes no difference.
            set(dst, &l, &h);
            let (s0, s1) = sums(dst);
            if s1 < s0 {
                invert = true;
                swap(dst);
            }
        } else {
            swap(dst);
            if !desired_bc {
                invert = true;
            }
        }
    } else if desired_bc {
        invert = true;
    }
    invert
}

/// `cem_encode_no_bc` (CEM 8 or 12): never blue contraction; true if weights must be inverted.
fn cem_encode_no_bc(cem: u32, dst: &mut [u8], src: &[Px], t: Tabs) -> bool {
    let (l, h) = (src[0], src[1]);
    for c in 0..3 {
        dst[c * 2] = t.q(l[c]);
        dst[c * 2 + 1] = t.q(h[c]);
    }
    if cem == CEM_LDR_RGBA_DIRECT {
        dst[6] = t.q(l[3]);
        dst[7] = t.q(h[3]);
    }
    let s0 = t.dq(dst[0]) as u32 + t.dq(dst[2]) as u32 + t.dq(dst[4]) as u32;
    let s1 = t.dq(dst[1]) as u32 + t.dq(dst[3]) as u32 + t.dq(dst[5]) as u32;
    if s1 < s0 {
        dst.swap(0, 1);
        dst.swap(2, 3);
        dst.swap(4, 5);
        if cem == CEM_LDR_RGBA_DIRECT {
            dst.swap(6, 7);
        }
        return true;
    }
    false
}

/// `map_astc_block_weights_rgb`: 1-3 subsets with equal CEMs; per-texel projection weights.
fn map_weights_rgb(px: &[Px; 16], log: &LogAstcBlock, t: Tabs) -> [u8; 16] {
    let num_subsets = (log.num_partitions as usize).clamp(1, 3);
    let cem = log.color_endpoint_modes[0] as u32;
    let nv = ah::get_num_cem_values(cem) as usize;
    let mut dq = [0u8; 24];
    for i in 0..(nv * num_subsets).min(18) {
        dq[i] = t.dq(log.endpoints[i]);
    }
    let (mut dr, mut dg, mut db, mut sofs) = ([0i32; 3], [0i32; 3], [0i32; 3], [0i32; 3]);
    let mut f = [0f32; 3];
    for s in 0..num_subsets {
        let mut de = [[0i32; 2]; 4];
        ah::decode_endpoint(cem, &mut de, &dq[s * nv..]);
        dr[s] = de[0][1] - de[0][0];
        dg[s] = de[1][1] - de[1][0];
        db[s] = de[2][1] - de[2][0];
        f[s] = 64.0f32 / ((sq(dr[s]) + sq(dg[s]) + sq(db[s])) as f32 + 0.00000125f32);
        sofs[s] = -(de[0][0] * dr[s] + de[1][0] * dg[s] + de[2][0] * db[s]);
    }
    let wtab = weight_to_ise(log.weight_ise_range as u32);
    let mut w = [0u8; 16];
    for (i, o) in w.iter_mut().enumerate() {
        let s = if num_subsets > 1 {
            texel_partition(log.partition_id, i, num_subsets as i32).min(2)
        } else {
            0
        };
        let p = &px[i];
        let dot = p[0] as i32 * dr[s] + p[1] as i32 * dg[s] + p[2] as i32 * db[s] + sofs[s];
        let sel = f2i(dot as f32 * f[s] + 0.5f32).clamp(0, 64);
        *o = lookup(wtab, sel);
    }
    w
}

fn set_cems(log: &mut LogAstcBlock, n: usize, cem: u32) {
    for c in log.color_endpoint_modes.iter_mut().take(n) {
        *c = cem as u8;
    }
}

fn set_ranges(log: &mut LogAstcBlock, endpoint: u32, weight: u32) {
    log.endpoint_ise_range = endpoint as u8;
    log.weight_ise_range = weight as u8;
}

/// `cem_encode_no_bc` of a subset's endpoints at 256 levels (the CEM conversion source).
fn src_rgb_direct(ep: &[Px]) -> [u8; 6] {
    let mut s = [0u8; 8];
    cem_encode_no_bc(
        CEM_LDR_RGB_DIRECT,
        &mut s,
        ep,
        Tabs::endpoint(BISE_256_LEVELS),
    );
    [s[0], s[1], s[2], s[3], s[4], s[5]]
}

/// Tries base+offset for every subset (`perm[astc_subset]` -> BC7 subset); false if clamped.
fn try_base_ofs(log: &mut LogAstcBlock, endpoints: &[Px], subsets: usize, perm: &[u8; 3]) -> bool {
    let range = log.endpoint_ise_range as u32;
    for s in 0..subsets {
        let b = perm[s] as usize;
        let src = src_rgb_direct(&endpoints[b * 2..b * 2 + 2]);
        let (ok, clamped) = convert_from_rgb_direct(
            &src,
            CEM_LDR_RGB_BASE_PLUS_OFFSET,
            range,
            &mut log.endpoints[s * 6..s * 6 + 6],
            true,
        );
        if !ok || clamped {
            return false;
        }
    }
    true
}

fn fast_pack_astc_3subsets(
    mode: u32,
    log: &mut LogAstcBlock,
    bc7: &[u8; 16],
    px: &[Px; 16],
) -> bool {
    let endpoint_bits = if mode == 0 { 4 } else { 5 };
    let num_pbits = if mode == 0 { 6 } else { 0 };
    let mut ofs = mode + 1;
    let part = read_bits(bc7, &mut ofs, if mode == 0 { 4 } else { 6 });
    let Some(&(_, astc_part, perm_index)) = ASTC_BC7_COMMON_PARTITIONS3
        .iter()
        .find(|e| e.0 as u32 == part)
    else {
        log.error_flag = true;
        return false;
    };
    log.partition_id = astc_part;
    let perm = &ASTC_TO_BC7_PARTITION_INDEX_PERM_TABLES[(perm_index as usize).min(5)];

    let mut endpoints = [[0u8; 4]; 6];
    for c in 0..3 {
        for e in endpoints.iter_mut() {
            e[c] = read_bits(bc7, &mut ofs, endpoint_bits) as u8;
        }
    }
    let mut pbits = [0u32; 6];
    for p in pbits.iter_mut().take(num_pbits) {
        *p = read_bits(bc7, &mut ofs, 1);
    }
    for (e, ep) in endpoints.iter_mut().enumerate() {
        for c in 0..4 {
            ep[c] = if c == 3 {
                255
            } else if num_pbits != 0 {
                dequant_p(ep[c] as u32, pbits[e], endpoint_bits)
            } else {
                dequant(ep[c] as u32, endpoint_bits)
            };
        }
    }
    log.num_partitions = 3;

    // First try base+scale, 3x4/4x3 or 4x4.
    let mut used_base_scale = true;
    set_cems(log, 3, CEM_LDR_RGB_BASE_SCALE);
    for pass in 0..2 {
        if pass == 0 {
            set_ranges(log, BISE_32_LEVELS, BISE_8_LEVELS);
        } else {
            set_ranges(log, BISE_40_LEVELS, BISE_4_LEVELS);
        }
        let t = Tabs::endpoint(log.endpoint_ise_range as u32);
        used_base_scale = true;
        for s in 0..3 {
            let b = perm[s] as usize;
            let src = src_rgb_direct(&endpoints[b * 2..b * 2 + 2]);
            let (ok, _) = convert_from_rgb_direct(
                &src,
                CEM_LDR_RGB_BASE_SCALE,
                log.endpoint_ise_range as u32,
                &mut log.endpoints[s * 4..s * 4 + 4],
                false,
            );
            if !ok {
                used_base_scale = false;
                break;
            }
            let (pl, ph) = (&endpoints[b * 2], &endpoints[b * 2 + 1]);
            let sum = |p: &Px| p[0] as i32 + p[1] as i32 + p[2] as i32;
            let pl = if sum(pl) > sum(ph) { ph } else { pl };
            let mut dq = [0u8; 4];
            for i in 0..4 {
                dq[i] = t.dq(log.endpoints[s * 4 + i]);
            }
            let mut de = [[0i32; 2]; 4];
            ah::decode_endpoint(CEM_LDR_RGB_BASE_SCALE, &mut de, &dq);
            let dist2 = sq(pl[0] as i32 - de[0][0])
                + sq(pl[1] as i32 - de[1][0])
                + sq(pl[2] as i32 - de[2][0]);
            if dist2 > 256 * 3 {
                used_base_scale = false;
                break;
            }
        }
        if !used_base_scale {
            break;
        }
        let w = map_weights_rgb(px, log, t);
        if pass == 1 {
            log.weights[..16].copy_from_slice(&w);
        } else if apply_grid(log, choose_grid(&w, 6), &w, 0) {
            break;
        }
    }

    if !used_base_scale {
        // Try base+offset 3x4 or 4x3.
        set_cems(log, 3, CEM_LDR_RGB_BASE_PLUS_OFFSET);
        set_ranges(log, BISE_16_LEVELS, BISE_4_LEVELS);
        let t = Tabs::endpoint(log.endpoint_ise_range as u32);
        let used_base_ofs = try_base_ofs(log, &endpoints, 3, perm);
        let w = if used_base_ofs {
            map_weights_rgb(px, log, t)
        } else {
            // RGB direct 3x4 or 4x3.
            set_cems(log, 3, CEM_LDR_RGB_DIRECT);
            set_ranges(log, BISE_16_LEVELS, BISE_4_LEVELS);
            let t = Tabs::endpoint(log.endpoint_ise_range as u32);
            for s in 0..3 {
                let b = perm[s] as usize;
                cem_encode_bc(
                    CEM_LDR_RGB_DIRECT,
                    &mut log.endpoints[s * 6..s * 6 + 6],
                    &endpoints[b * 2..b * 2 + 2],
                    t,
                );
            }
            map_weights_rgb(px, log, t)
        };
        if !apply_grid(log, choose_grid(&w, 3), &w, 0) {
            // 4x4: base+offset, else RGB direct.
            set_cems(log, 3, CEM_LDR_RGB_BASE_PLUS_OFFSET);
            set_ranges(log, BISE_12_LEVELS, BISE_4_LEVELS);
            let t = Tabs::endpoint(log.endpoint_ise_range as u32);
            if try_base_ofs(log, &endpoints, 3, perm) {
                let w = map_weights_rgb(px, log, t);
                log.weights[..16].copy_from_slice(&w);
            } else {
                set_cems(log, 3, CEM_LDR_RGB_DIRECT);
                set_ranges(log, BISE_12_LEVELS, BISE_4_LEVELS);
                let t = Tabs::endpoint(log.endpoint_ise_range as u32);
                for s in 0..3 {
                    let b = perm[s] as usize;
                    cem_encode_bc(
                        CEM_LDR_RGB_DIRECT,
                        &mut log.endpoints[s * 6..s * 6 + 6],
                        &endpoints[b * 2..b * 2 + 2],
                        t,
                    );
                }
                let w = map_weights_rgb(px, log, t);
                log.weights[..16].copy_from_slice(&w);
            }
        }
    }
    true
}

/// Looks up the ASTC partition for a BC7 2-subset pattern: (astc seed, invert subsets).
fn common_partition2(part: u32) -> Option<(u16, bool)> {
    let r = *BC7_2SUBSETS_TO_COMMON_PATTERN.get(part as usize)?;
    let e = ASTC_BC7_COMMON_PARTITIONS2.get(usize::try_from(r).ok()?)?;
    Some((e.1, e.2))
}

fn swap_subsets(log: &mut LogAstcBlock, n: usize) {
    for i in 0..n {
        log.endpoints.swap(i, i + n);
    }
}

fn fast_pack_astc_2subsets(
    mode: u32,
    log: &mut LogAstcBlock,
    bc7: &[u8; 16],
    px: &[Px; 16],
) -> bool {
    let comps = if mode == 7 { 4 } else { 3 };
    let weight_bits = if mode == 1 { 3 } else { 2 };
    let endpoint_bits = match mode {
        7 => 5,
        1 => 6,
        _ => 7,
    };
    let num_pbits = if mode == 1 { 2 } else { 4 };
    let shared = mode == 1;
    let mut ofs = mode + 1;
    let part = read_bits(bc7, &mut ofs, 6);
    let Some((astc_part, invert)) = common_partition2(part) else {
        log.error_flag = true;
        return false;
    };
    log.partition_id = astc_part;

    let mut endpoints = [[0u8; 4]; 4];
    for c in 0..comps {
        for e in endpoints.iter_mut() {
            e[c] = read_bits(bc7, &mut ofs, endpoint_bits) as u8;
        }
    }
    let mut pbits = [0u32; 4];
    for p in pbits.iter_mut().take(num_pbits) {
        *p = read_bits(bc7, &mut ofs, 1);
    }
    let alpha_c = if mode == 7 { 4 } else { 3 };
    for (e, ep) in endpoints.iter_mut().enumerate() {
        for c in 0..4 {
            ep[c] = if c == alpha_c {
                255
            } else {
                dequant_p(
                    ep[c] as u32,
                    pbits[if shared { e >> 1 } else { e }],
                    endpoint_bits,
                )
            };
        }
    }
    log.num_partitions = 2;
    set_cems(log, 2, CEM_LDR_RGB_DIRECT);
    {
        if weight_bits == 2 {
            set_ranges(log, BISE_64_LEVELS, BISE_4_LEVELS);
        } else {
            set_ranges(log, BISE_32_LEVELS, BISE_8_LEVELS);
        }
        let t = Tabs::endpoint(log.endpoint_ise_range as u32);
        cem_encode_bc(
            CEM_LDR_RGB_DIRECT,
            &mut log.endpoints[0..6],
            &endpoints[0..2],
            t,
        );
        cem_encode_bc(
            CEM_LDR_RGB_DIRECT,
            &mut log.endpoints[6..12],
            &endpoints[2..4],
            t,
        );
        if invert {
            swap_subsets(log, 6);
        }
        let w = map_weights_rgb(px, log, t);
        if apply_grid(log, choose_grid(&w, 4), &w, 0) {
            return true;
        }
    }

    // RGB direct / base+offset, 4x4.
    if weight_bits == 2 {
        set_ranges(log, BISE_40_LEVELS, BISE_4_LEVELS);
    } else {
        set_ranges(log, BISE_32_LEVELS, BISE_5_LEVELS);
    }
    let t = Tabs::endpoint(log.endpoint_ise_range as u32);
    let use_base_ofs = try_base_ofs(log, &endpoints, 2, &[0, 1, 2]);
    if use_base_ofs {
        set_cems(log, 2, CEM_LDR_RGB_BASE_PLUS_OFFSET);
    } else {
        cem_encode_bc(
            CEM_LDR_RGB_DIRECT,
            &mut log.endpoints[0..6],
            &endpoints[0..2],
            t,
        );
        cem_encode_bc(
            CEM_LDR_RGB_DIRECT,
            &mut log.endpoints[6..12],
            &endpoints[2..4],
            t,
        );
    }
    if invert {
        swap_subsets(log, 6);
    }
    let w = map_weights_rgb(px, log, t);
    log.weights[..16].copy_from_slice(&w);
    true
}

/// Mode 7 endpoints (CEM 12, both subsets) at `range`; returns the per-subset weight inversion.
fn mode7_endpoints(
    log: &mut LogAstcBlock,
    endpoints: &[Px; 4],
    range: u32,
    invert: bool,
) -> [u8; 2] {
    log.endpoint_ise_range = range as u8;
    let t = Tabs::endpoint(range);
    for s in 0..2 {
        for c in 0..4 {
            log.endpoints[s * 8 + c * 2] = t.q(endpoints[s * 2][c]);
            log.endpoints[s * 8 + c * 2 + 1] = t.q(endpoints[s * 2 + 1][c]);
        }
    }
    let sum =
        |e: &[u8; 18], o: usize| t.dq(e[o]) as i32 + t.dq(e[o + 2]) as i32 + t.dq(e[o + 4]) as i32;
    let mut invw = [0u8; 2];
    for s in 0..2 {
        if sum(&log.endpoints, s * 8 + 1) < sum(&log.endpoints, s * 8) {
            for k in 0..4 {
                log.endpoints.swap(s * 8 + k * 2, s * 8 + k * 2 + 1);
            }
            invw[s] = 3;
        }
    }
    if invert {
        swap_subsets(log, 8);
        invw.swap(0, 1);
    }
    invw
}

fn pack_mode7(log: &mut LogAstcBlock, bc7: &[u8; 16]) -> bool {
    let mut ofs = 8;
    let part = read_bits(bc7, &mut ofs, 6);
    let Some((astc_part, invert)) = common_partition2(part) else {
        log.error_flag = true;
        return false;
    };
    log.partition_id = astc_part;
    let mut endpoints = [[0u8; 4]; 4];
    for c in 0..4 {
        for e in endpoints.iter_mut() {
            e[c] = read_bits(bc7, &mut ofs, 5) as u8;
        }
    }
    let mut pbits = [0u32; 4];
    for p in pbits.iter_mut() {
        *p = read_bits(bc7, &mut ofs, 1);
    }
    let anchor = super::uastc_tables::BC7_TABLE_ANCHOR_INDEX_SECOND_SUBSET[part as usize & 63];
    let mut weights = [0u8; 16];
    for (i, w) in weights.iter_mut().enumerate() {
        let n = if i == 0 || i == anchor as usize { 1 } else { 2 };
        *w = read_bits(bc7, &mut ofs, n) as u8;
    }
    for (e, ep) in endpoints.iter_mut().enumerate() {
        for c in 0..4 {
            ep[c] = dequant_p(ep[c] as u32, pbits[e], 5);
        }
    }
    log.num_partitions = 2;
    set_cems(log, 2, CEM_LDR_RGBA_DIRECT);
    log.weight_ise_range = BISE_4_LEVELS as u8;

    let invw = mode7_endpoints(log, &endpoints, BISE_24_LEVELS, invert);
    let mut temp = [0u8; 16];
    for (i, t) in temp.iter_mut().enumerate() {
        *t = weights[i] ^ invw[texel_partition(log.partition_id, i, 2) & 1];
    }
    if !apply_grid(log, choose_grid(&temp, 4), &temp, 0) {
        let invw = mode7_endpoints(log, &endpoints, BISE_16_LEVELS, invert);
        for i in 0..16 {
            log.weights[i] = weights[i] ^ invw[texel_partition(log.partition_id, i, 2) & 1];
        }
    }
    true
}

/// Dual plane endpoints (mode 4/5): CEM 8 if opaque else 12. True if weights must be inverted.
fn dual_plane_endpoints(log: &mut LogAstcBlock, endpoints: &[Px; 2], cem: u32) -> bool {
    log.color_endpoint_modes[0] = cem as u8;
    let t = Tabs::endpoint(log.endpoint_ise_range as u32);
    let comps = if cem == CEM_LDR_RGBA_DIRECT { 4 } else { 3 };
    for c in 0..comps {
        log.endpoints[c * 2] = t.q(endpoints[0][c]);
        log.endpoints[c * 2 + 1] = t.q(endpoints[1][c]);
    }
    let s0 = t.dq(log.endpoints[0]) as i32
        + t.dq(log.endpoints[2]) as i32
        + t.dq(log.endpoints[4]) as i32;
    let s1 = t.dq(log.endpoints[1]) as i32
        + t.dq(log.endpoints[3]) as i32
        + t.dq(log.endpoints[5]) as i32;
    if s1 < s0 {
        for c in 0..comps {
            log.endpoints.swap(c * 2, c * 2 + 1);
        }
        return true;
    }
    false
}

fn pack_mode45(mode: u32, log: &mut LogAstcBlock, bc7: &[u8; 16]) -> bool {
    let (wb, a_wb) = (2u32, if mode == 4 { 3u32 } else { 2 });
    let (eb, a_eb) = if mode == 4 { (5u32, 6u32) } else { (7, 8) };
    let mut ofs = mode + 1;
    let comp_rot = read_bits(bc7, &mut ofs, 2);
    let index_mode = if mode == 4 {
        read_bits(bc7, &mut ofs, 1)
    } else {
        0
    };
    let mut endpoints = [[0u8; 4]; 2];
    for c in 0..4 {
        for e in endpoints.iter_mut() {
            e[c] = read_bits(bc7, &mut ofs, if c == 3 { a_eb } else { eb }) as u8;
        }
    }
    let weight_bits = if index_mode != 0 {
        [a_wb, wb]
    } else {
        [wb, a_wb]
    };
    let (mut weights, mut a_weights) = ([0u8; 16], [0u8; 16]);
    {
        let (first, second) = if index_mode != 0 {
            (&mut a_weights, &mut weights)
        } else {
            (&mut weights, &mut a_weights)
        };
        let im = index_mode as usize;
        for (i, w) in first.iter_mut().enumerate() {
            *w = read_bits(bc7, &mut ofs, weight_bits[im] - (i == 0) as u32) as u8;
        }
        for (i, w) in second.iter_mut().enumerate() {
            *w = read_bits(bc7, &mut ofs, weight_bits[1 - im] - (i == 0) as u32) as u8;
        }
    }
    if weights[0] == weights[15] && a_weights[0] == a_weights[15] {
        let solid = (1..15).all(|k| weights[0] == weights[k] && a_weights[0] == a_weights[k]);
        if solid {
            let mut lb = LogBc7Block::default();
            if !bc7u::unpack_phys(bc7, &mut lb) {
                return false;
            }
            let c = bc7u::unpack_texel(&lb, 0, 0).unwrap_or([0; 4]);
            set_solid(log, c);
            return true;
        }
    }
    for e in endpoints.iter_mut() {
        for c in 0..4 {
            e[c] = dequant(e[c] as u32, if c == 3 { a_eb } else { eb });
        }
        if comp_rot >= 1 {
            e.swap(3, (comp_rot - 1) as usize & 3);
        }
    }
    log.num_partitions = 1;
    log.dual_plane = true;
    log.color_component_selector = (comp_rot.wrapping_sub(1) & 3) as u8;

    let err_3x4 = sse_3x4(&weights) + sse_3x4(&a_weights);
    let err_4x3 = sse_4x3(&weights) + sse_4x3(&a_weights);
    let w_thresh =
        (if weight_bits[0] > 4 { 6 } else { 4 }) + (if weight_bits[1] > 4 { 6 } else { 4 });
    let opaque = endpoints[0][3] == 255 && endpoints[1][3] == 255;
    let (w0_max, w1_max) = ((1i32 << weight_bits[0]) - 1, (1i32 << weight_bits[1]) - 1);
    let (w0_half, w1_half) = (w0_max >> 1, w1_max >> 1);
    let (n, w0s, w1s, invw);
    let (mut g0, mut g1) = ([0u8; 16], [0u8; 16]);
    if err_3x4 <= w_thresh || err_4x3 <= w_thresh {
        if err_3x4 <= err_4x3 {
            log.grid_width = 3;
            to_3x4(&weights, &mut g0, 0);
            to_3x4(&a_weights, &mut g1, 0);
        } else {
            log.grid_height = 3;
            to_4x3(&weights, &mut g0, 0);
            to_4x3(&a_weights, &mut g1, 0);
        }
        let uses_3bit = weight_bits[0] == 3 || weight_bits[1] == 3;
        if opaque {
            if uses_3bit {
                set_ranges(log, BISE_64_LEVELS, BISE_8_LEVELS);
            } else {
                set_ranges(log, BISE_256_LEVELS, BISE_4_LEVELS);
            }
            invw = dual_plane_endpoints(log, &endpoints, CEM_LDR_RGB_DIRECT);
        } else {
            if uses_3bit {
                set_ranges(log, BISE_48_LEVELS, BISE_6_LEVELS);
            } else {
                set_ranges(log, BISE_192_LEVELS, BISE_4_LEVELS);
            }
            invw = dual_plane_endpoints(log, &endpoints, CEM_LDR_RGBA_DIRECT);
        }
        n = 12;
        w0s = g0;
        w1s = g1;
    } else {
        if opaque {
            set_ranges(log, BISE_160_LEVELS, BISE_4_LEVELS);
            invw = dual_plane_endpoints(log, &endpoints, CEM_LDR_RGB_DIRECT);
        } else {
            set_ranges(log, BISE_48_LEVELS, BISE_4_LEVELS);
            invw = dual_plane_endpoints(log, &endpoints, CEM_LDR_RGBA_DIRECT);
        }
        n = 16;
        w0s = weights;
        w1s = a_weights;
    }
    let wtab = weight_to_ise(log.weight_ise_range as u32);
    for w in 0..n {
        let mut v0 = (w0s[w] as i32 * 64 + w0_half) / w0_max;
        let mut v1 = (w1s[w] as i32 * 64 + w1_half) / w1_max;
        if invw {
            v0 = 64 - v0;
            v1 = 64 - v1;
        }
        log.weights[w * 2] = lookup(wtab, v0);
        log.weights[w * 2 + 1] = lookup(wtab, v1);
    }
    true
}

fn pack_mode6(log: &mut LogAstcBlock, bc7: &[u8; 16]) -> bool {
    let bits = u128::from_le_bytes(*bc7);
    let (lo, hi) = (bits as u64, (bits >> 64) as u64);
    let f = |ofs: u32| ((lo >> ofs) & 127) as u8;
    let (r0, r1, g0, g1, b0, b1, a0, a1) = (f(7), f(14), f(21), f(28), f(35), f(42), f(49), f(56));
    let p0 = (lo >> 63) as u8;
    let p1 = (hi & 1) as u8;
    let mut weights = [0u8; 16];
    weights[0] = ((hi >> 1) & 7) as u8;
    for (i, w) in weights.iter_mut().enumerate().skip(1) {
        *w = ((hi >> (4 + (i - 1) * 4)) & 15) as u8;
    }
    let from_7 = |v: u8, p: u8| (v << 1) | p;
    log.num_partitions = 1;
    if a0 < 127 || a1 < 127 {
        log.color_endpoint_modes[0] = CEM_LDR_RGBA_DIRECT as u8;
        set_ranges(log, BISE_96_LEVELS, BISE_12_LEVELS);
        let t = Tabs::endpoint(log.endpoint_ise_range as u32);
        let endpoints = [
            [
                from_7(r0, p0),
                from_7(g0, p0),
                from_7(b0, p0),
                from_7(a0, p0),
            ],
            [
                from_7(r1, p1),
                from_7(g1, p1),
                from_7(b1, p1),
                from_7(a1, p1),
            ],
        ];
        let invw = if cem_encode_bc(CEM_LDR_RGBA_DIRECT, &mut log.endpoints[0..8], &endpoints, t) {
            15
        } else {
            0
        };
        const W2ISE: [u8; 16] = [0, 4, 8, 8, 2, 6, 10, 10, 11, 11, 7, 3, 9, 9, 5, 1];
        for w in 0..16 {
            log.weights[w] = W2ISE[((weights[w] ^ invw) & 15) as usize];
        }
        return true;
    }
    let endpoints = [
        [from_7(r0, p0), from_7(g0, p0), from_7(b0, p0), 255],
        [from_7(r1, p1), from_7(g1, p1), from_7(b1, p1), 255],
    ];
    log.color_endpoint_modes[0] = CEM_LDR_RGB_DIRECT as u8;
    let grid = choose_grid(&weights, 6);
    if grid != Grid::G4x4 {
        set_ranges(log, BISE_256_LEVELS, BISE_16_LEVELS);
        let t = Tabs::endpoint(log.endpoint_ise_range as u32);
        let invw = if cem_encode_no_bc(CEM_LDR_RGB_DIRECT, &mut log.endpoints[0..6], &endpoints, t)
        {
            15
        } else {
            0
        };
        apply_grid(log, grid, &weights, invw);
    } else {
        set_ranges(log, BISE_192_LEVELS, BISE_16_LEVELS);
        let t = Tabs::endpoint(log.endpoint_ise_range as u32);
        let invw = if cem_encode_bc(CEM_LDR_RGB_DIRECT, &mut log.endpoints[0..6], &endpoints, t) {
            15
        } else {
            0
        };
        for w in 0..16 {
            log.weights[w] = weights[w] ^ invw;
        }
    }
    true
}

/// `fast_pack_astc(const bc7_block&, log_astc_block&, const color_rgba*)`: latent to latent.
/// False if the block can't be mapped (unsupported 2/3-subset pattern) or is invalid.
fn pack_from_bc7(bc7: &[u8; 16], log: &mut LogAstcBlock, px: Option<&[Px; 16]>) -> bool {
    log.clear();
    log.grid_width = 4;
    log.grid_height = 4;
    if bc7[0] == 0 {
        log.error_flag = true;
        return false;
    }
    let mode = bc7[0].trailing_zeros();
    let texels = |out: &mut [Px; 16]| -> bool {
        match px {
            Some(p) => {
                *out = *p;
                true
            }
            None => crate::bc::bc7::unpack_bc7(bc7, out),
        }
    };
    match mode {
        7 => pack_mode7(log, bc7),
        0 | 2 | 1 | 3 => {
            let mut t = [[0u8; 4]; 16];
            if !texels(&mut t) {
                return false;
            }
            if mode == 0 || mode == 2 {
                fast_pack_astc_3subsets(mode, log, bc7, &t)
            } else {
                fast_pack_astc_2subsets(mode, log, bc7, &t)
            }
        }
        4 | 5 => pack_mode45(mode, log, bc7),
        _ => pack_mode6(log, bc7),
    }
}

/// `fast_pack_astc(log_astc_block&, const color_rgba*, flags)`: encodes pixels via `bc7f`.
fn pack_from_pixels(log: &mut LogAstcBlock, px: &[Px; 16], override_flags: u32) -> bool {
    log.clear();
    if px.iter().all(|p| *p == px[0]) {
        set_solid(log, px[0]);
        return true;
    }
    log.grid_width = 4;
    log.grid_height = 4;
    let flags = if override_flags == 0 {
        PACK_BC7_FLAG_PBIT_OPT
            | PACK_BC7_FLAG_PBIT_OPT_MODE6
            | PACK_BC7_FLAG_USE_TRIVIAL_MODE6
            | PACK_BC7_FLAG_USE_2_SUBSETS_RGB
            | PACK_BC7_FLAG_ASTC_COMPATIBLE
            | PACK_BC7_FLAG_USE_DUAL_PLANE_RGB
            | PACK_BC7_FLAG_USE_DUAL_PLANE_RGBA
            | PACK_BC7_FLAG_USE_3_SUBSETS_RGB
            | PACK_BC7_FLAG_USE_2_SUBSETS_RGBA
    } else {
        override_flags | PACK_BC7_FLAG_ASTC_COMPATIBLE
    };
    let mut b = [0u8; 16];
    fast_pack_bc7_auto_rgba(&mut b, px, flags);
    pack_from_bc7(&b, log, Some(px))
}

/// `bc7f::fast_pack_astc(void*, const bc7_block&, flags)`: BC7 block to a physical ASTC LDR
/// 4x4 block. False (and a zero block) on failure.
pub(crate) fn fast_pack_astc(out: &mut [u8; 16], bc7: &[u8; 16], flags: u32) -> bool {
    let mut log = LogAstcBlock::default();
    let mut ok = pack_from_bc7(bc7, &mut log, None);
    if !ok {
        let mut px = [[0u8; 4]; 16];
        ok = crate::bc::bc7::unpack_bc7(bc7, &mut px)
            && pack_from_pixels(&mut log, &px, flags | PACK_BC7_FLAG_ASTC_COMPATIBLE);
    }
    if !ok || !ah::pack_astc_block(out, &log) {
        *out = [0; 16];
        return false;
    }
    true
}
