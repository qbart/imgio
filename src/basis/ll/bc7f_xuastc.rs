// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of the basisu `bc7f::pack_from_astc_*` family (`basisu_transcoder.cpp`): direct
//! ASTC logical block -> BC7 packers used by the XUASTC LDR 4x4 / 6x6 / 8x6 -> BC7 fast paths,
//! plus the block helpers of `transcode_slice` (`blocks_same_solid_colors`, ...).
//!
//! Inputs are single-subset, single-plane (or solid) blocks; weights are upsampled to block
//! resolution in [0,64]. Float math follows the C++ operation order in `f32`.

use super::bc7f::{
    PACK_BC7_FLAG_PBIT_OPT_MODE6, PACK_BC7_FLAG_USE_TRIVIAL_MODE6, determine_shared_pbits,
    determine_unique_pbits, encode_mode1_rgb_block, encode_mode6_rgba_block, f2i,
    fast_pack_bc7_rgb_analytical,
};
use crate::astc::helpers::{self as ah, LogAstcBlock};

type Px = [u8; 4];
type Block = [u8; 16];

/// `color_rgba` of a solid LDR block (`m_solid_color[c] >> 8`).
pub(crate) fn solid_px(b: &LogAstcBlock) -> Px {
    [0, 1, 2, 3].map(|c| (b.solid_color(c) >> 8) as u8)
}

/// `astc_ldr_t::decode_endpoints` for partition 0: low and high RGBA8 endpoints.
pub(crate) fn decode_endpoints(b: &LogAstcBlock) -> (Px, Px) {
    let cem = u32::from(b.color_endpoint_modes[0]);
    let n = (ah::get_num_cem_values(cem) as usize).min(b.endpoints.len());
    let tab = ah::dequant_tables().endpoint_ise_to_val(u32::from(b.endpoint_ise_range));
    let mut dq = [0u8; 8];
    for (d, &v) in dq.iter_mut().zip(&b.endpoints[..n]) {
        *d = tab.get(usize::from(v)).copied().unwrap_or(0);
    }
    let mut e = [[0i32; 2]; 4];
    ah::decode_endpoint(cem, &mut e, &dq[..n.min(8)]);
    (e.map(|c| c[0] as u8), e.map(|c| c[1] as u8))
}

/// Endpoints of a solid (l = h = color) or single-subset block.
fn endpoints_or_solid(b: &LogAstcBlock) -> (Px, Px) {
    if b.solid_color_flag_ldr {
        let c = solid_px(b);
        (c, c)
    } else {
        decode_endpoints(b)
    }
}

/// `astc_helpers::does_cem_have_alpha`.
fn does_cem_have_alpha(cem: u32) -> bool {
    matches!(
        cem,
        ah::CEM_LDR_LUM_ALPHA_DIRECT
            | ah::CEM_LDR_LUM_ALPHA_BASE_PLUS_OFS
            | ah::CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A
            | ah::CEM_LDR_RGBA_DIRECT
            | ah::CEM_LDR_RGBA_BASE_PLUS_OFFSET
            | ah::CEM_HDR_RGB_LDR_ALPHA
            | ah::CEM_HDR_RGB_HDR_ALPHA
    )
}

/// `blocks_same_solid_colors`.
pub(crate) fn blocks_same_solid_colors(a: &LogAstcBlock, b: &LogAstcBlock, tol: u32) -> bool {
    if !a.solid_color_flag_ldr || !b.solid_color_flag_ldr {
        return false;
    }
    if tol == 0 {
        return (0..4).all(|i| a.solid_color(i) == b.solid_color(i));
    }
    (0..4).all(|i| {
        let ac = i32::from(a.solid_color(i) >> 8);
        let bc = i32::from(b.solid_color(i) >> 8);
        (ac - bc).abs() <= tol as i32
    })
}

/// `blocks_same_single_subset_endpoints`.
pub(crate) fn blocks_same_single_subset_endpoints(
    a: &LogAstcBlock,
    b: &LogAstcBlock,
    tol: u32,
) -> bool {
    if a.solid_color_flag_ldr || b.solid_color_flag_ldr {
        return false;
    }
    if a.dual_plane || b.dual_plane {
        return false;
    }
    if a.num_partitions > 1 || b.num_partitions > 1 {
        return false;
    }
    if a.color_endpoint_modes[0] != b.color_endpoint_modes[0] {
        return false;
    }
    if a.endpoint_ise_range != b.endpoint_ise_range {
        return false;
    }
    if tol > 0 {
        let (al, ah) = decode_endpoints(a);
        let (bl, bh) = decode_endpoints(b);
        (0..4).all(|i| {
            (i32::from(al[i]) - i32::from(bl[i])).abs() <= tol as i32
                && (i32::from(ah[i]) - i32::from(bh[i])).abs() <= tol as i32
        })
    } else {
        let n = (ah::get_num_cem_values(u32::from(a.color_endpoint_modes[0])) as usize)
            .min(a.endpoints.len());
        a.endpoints[..n] == b.endpoints[..n]
    }
}

/// `block_has_alpha` (solid or single-subset blocks).
pub(crate) fn block_has_alpha(a: &LogAstcBlock) -> bool {
    if a.solid_color_flag_ldr {
        return (a.solid_color(3) >> 8) != 255;
    }
    does_cem_have_alpha(u32::from(a.color_endpoint_modes[0]))
}

/// `astc_upsample_grid_weights`: dequantized plane 0 weights upsampled to `bw` x `bh` ([0,64]).
/// Solid blocks leave `dst` untouched. Returns false on an invalid weight grid.
pub(crate) fn astc_upsample_grid_weights(
    b: &LogAstcBlock,
    dst: &mut [u8],
    bw: u32,
    bh: u32,
) -> bool {
    if b.solid_color_flag_ldr {
        return true;
    }
    let (gw, gh) = (u32::from(b.grid_width), u32::from(b.grid_height));
    let n = (gw * gh) as usize;
    if gw < 2
        || gh < 2
        || gw > bw
        || gh > bh
        || n > b.weights.len()
        || dst.len() < (bw * bh) as usize
    {
        return false;
    }
    let tab = ah::dequant_tables().weight_ise_to_val(u32::from(b.weight_ise_range));
    let mut dq = [0u8; ah::MAX_GRID_WEIGHTS as usize];
    for (d, &w) in dq.iter_mut().zip(&b.weights[..n]) {
        match tab.get(usize::from(w)) {
            Some(&v) => *d = v,
            None => return false,
        }
    }
    // upsample_weight_grid copies when the grid matches the block size.
    ah::upsample_weight_grid(bw, bh, gw, gh, &dq[..n], dst);
    true
}

/// ASTC [0,64] weight to a BC7 4-bit weight.
#[inline]
fn w4(w: u8) -> u8 {
    ((u32::from(w) * 15 + 32) >> 6) as u8
}

/// ASTC [0,64] weight to a BC7 3-bit weight.
#[inline]
fn w3(w: u8) -> u8 {
    ((u32::from(w) * 7 + 32) >> 6) as u8
}

fn unit(c: &Px) -> [f32; 4] {
    let q = 1.0f32 / 255.0f32;
    c.map(|v| f32::from(v) * q)
}

/// Mode 6 block from RGBA endpoints (unique p-bits) and 4-bit weights.
fn mode6(block: &mut Block, l: &Px, h: &Px, w: &[u8; 16]) {
    let (mn, mx, p) = determine_unique_pbits(4, 7, &unit(l), &unit(h));
    let (mn, mx) = (mn.map(u32::from), mx.map(u32::from));
    encode_mode6_rgba_block(
        block, mn[0], mn[1], mn[2], mn[3], p[0], mx[0], mx[1], mx[2], mx[3], p[1], w,
    );
}

/// Mode 1 block from two subsets' RGB endpoints (shared p-bits) and 3-bit weights.
fn mode1(block: &mut Block, part_id: u32, l: &[Px; 2], h: &[Px; 2], w: &[u8; 16]) {
    let (mn0, mx0, p0) = determine_shared_pbits(3, 6, &unit(&l[0]), &unit(&h[0]));
    let (mn1, mx1, p1) = determine_shared_pbits(3, 6, &unit(&l[1]), &unit(&h[1]));
    let c = |a: &Px, b: &Px, i: usize| [u32::from(a[i]), u32::from(b[i])];
    encode_mode1_rgb_block(
        block,
        part_id,
        c(&mn0, &mn1, 0),
        c(&mn0, &mn1, 1),
        c(&mn0, &mn1, 2),
        c(&mx0, &mx1, 0),
        c(&mx0, &mx1, 1),
        c(&mx0, &mx1, 2),
        p0[0],
        p1[0],
        w,
    );
}

#[inline]
fn wt(w: &[u8], i: usize) -> u8 {
    w.get(i).copied().unwrap_or(0)
}

/// `pack_from_astc_4x4_single_subset`: 4x4 single-subset, single-plane block to mode 6.
pub(crate) fn pack_from_astc_4x4_single_subset(block: &mut Block, b: &LogAstcBlock) -> bool {
    let mut up = [0u8; 16];
    if !astc_upsample_grid_weights(b, &mut up, 4, 4) {
        return false;
    }
    let (l, h) = decode_endpoints(b);
    mode6(block, &l, &h, &up.map(w4));
    true
}

/// `pack_from_astc_single_subset`: the 4x4 window at (`ofs_x`, `ofs_y`) of one single-subset
/// block (`bw` wide upsampled weights) to mode 6.
pub(crate) fn pack_from_astc_single_subset(
    block: &mut Block,
    b: &LogAstcBlock,
    up: &[u8],
    ofs_x: usize,
    ofs_y: usize,
    bw: usize,
) {
    let (l, h) = decode_endpoints(b);
    let w: [u8; 16] =
        std::array::from_fn(|i| w4(wt(up, (ofs_x + (i & 3)) + (ofs_y + (i >> 2)) * bw)));
    mode6(block, &l, &h, &w);
}

/// Upsampled weight of BC7 texel (x, y) for a block straddling two 6x6 ASTC blocks (b0 left
/// or top, b1 right or bottom); `None` selects b1.
fn weight_6x6_pair(w0: &[u8], w1: &[u8], x: usize, y: usize, dx: usize, dy: usize) -> (bool, u8) {
    if dy == 0 || dy == 2 {
        let oy = if dy == 2 { 2 } else { 0 };
        if x < 2 {
            (false, wt(w0, (x + 4) + (y + oy) * 6))
        } else {
            (true, wt(w1, (x - 2) + (y + oy) * 6))
        }
    } else {
        let ox = if dx == 2 { 2 } else { 0 };
        if y < 2 {
            (false, wt(w0, (x + ox) + (y + 4) * 6))
        } else {
            (true, wt(w1, (x + ox) + (y - 2) * 6))
        }
    }
}

/// Upsampled weight of BC7 texel (x, y) for a block straddling two vertically adjacent 8x6
/// ASTC blocks; the bool selects b1.
fn weight_8x6_pair(w0: &[u8], w1: &[u8], x: usize, y: usize, dx: usize) -> (bool, u8) {
    if y < 2 {
        (false, wt(w0, (dx * 4 + x) + (y + 4) * 8))
    } else {
        (true, wt(w1, (dx * 4 + x) + (y - 2) * 8))
    }
}

/// `pack_from_astc_to_single_subset_same_endpoints` (6x6 or 8x6): averaged endpoints, mode 6.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_from_astc_to_single_subset_same_endpoints(
    block: &mut Block,
    b0: &LogAstcBlock,
    w0: &[u8],
    b1: &LogAstcBlock,
    w1: &[u8],
    dx: usize,
    dy: usize,
    is_6x6: bool,
) {
    let (mut l, mut h) = decode_endpoints(b0);
    let (al, ah) = decode_endpoints(b1);
    for c in 0..4 {
        l[c] = ((u32::from(l[c]) + u32::from(al[c]) + 1) >> 1) as u8;
        h[c] = ((u32::from(h[c]) + u32::from(ah[c]) + 1) >> 1) as u8;
    }
    let w: [u8; 16] = std::array::from_fn(|i| {
        let (x, y) = (i & 3, i >> 2);
        let (_, v) = if is_6x6 {
            weight_6x6_pair(w0, w1, x, y, dx, dy)
        } else {
            weight_8x6_pair(w0, w1, x, y, dx)
        };
        w4(v)
    });
    mode6(block, &l, &h, &w);
}

/// `pack_from_astc_6x6_to_two_subsets_different_endpoints`: mode 1, one subset per 6x6 block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_from_astc_6x6_to_two_subsets_different_endpoints(
    block: &mut Block,
    b0: &LogAstcBlock,
    w0: &[u8],
    b1: &LogAstcBlock,
    w1: &[u8],
    dx: usize,
    dy: usize,
) {
    let (l0, h0) = endpoints_or_solid(b0);
    let (l1, h1) = endpoints_or_solid(b1);
    let solid = [b0.solid_color_flag_ldr, b1.solid_color_flag_ldr];
    let part_id = if dx == 0 || dx == 2 { 13 } else { 0 };
    let w: [u8; 16] = std::array::from_fn(|i| {
        let (s, v) = weight_6x6_pair(w0, w1, i & 3, i >> 2, dx, dy);
        if solid[usize::from(s)] { 0 } else { w3(v) }
    });
    mode1(block, part_id, &[l0, l1], &[h0, h1], &w);
}

/// `astc_helpers::channel_interpolate`.
fn channel_interpolate(le: u8, he: u8, w: u32, srgb: bool) -> u32 {
    let (le, he) = expand_ep(le, he, srgb);
    ah::weight_interpolate(le, he, w) >> 8
}

/// 8-bit endpoints to the 16-bit ASTC LDR8 / sRGB8 interpolation endpoints.
fn expand_ep(le: u8, he: u8, srgb: bool) -> (u32, u32) {
    let (le, he) = (u32::from(le), u32::from(he));
    if srgb {
        ((le << 8) | 0x80, (he << 8) | 0x80)
    } else {
        ((le << 8) | le, (he << 8) | he)
    }
}

/// `pack_from_astc_8x6_to_two_subsets_different_endpoints_hq`: mode 1 with endpoints fit to
/// the used weight range, or (both subsets low contrast) an analytical encode of the decoded
/// texels. Returns `fallback_encode_flag`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_from_astc_8x6_to_two_subsets_different_endpoints_hq(
    block: &mut Block,
    b0: &LogAstcBlock,
    w0: &[u8],
    b1: &LogAstcBlock,
    w1: &[u8],
    dx: usize,
    srgb: bool,
) -> bool {
    let solid = [b0.solid_color_flag_ldr, b1.solid_color_flag_ldr];
    let (l0, h0) = endpoints_or_solid(b0);
    let (l1, h1) = endpoints_or_solid(b1);
    let (mut l, mut h) = ([l0, l1], [h0, h1]);
    let texel = |i: usize| -> (usize, u32) {
        let (s, v) = weight_8x6_pair(w0, w1, i & 3, i >> 2, dx);
        let s = usize::from(s);
        (s, if solid[s] { 0 } else { u32::from(v) })
    };
    let (mut low_w, mut high_w) = ([u32::MAX; 2], [0u32; 2]);
    for i in 0..16 {
        let (s, w) = texel(i);
        low_w[s] = low_w[s].min(w);
        high_w[s] = high_w[s].max(w);
    }
    let (orig_l, orig_h) = (l, h);
    let mut num_low_stddev = 0;
    for s in 0..2 {
        if solid[s] {
            continue;
        }
        if low_w[s] > 0 || high_w[s] < 64 {
            for c in 0..3 {
                l[s][c] = channel_interpolate(orig_l[s][c], orig_h[s][c], low_w[s], srgb) as u8;
                h[s][c] = channel_interpolate(orig_l[s][c], orig_h[s][c], high_w[s], srgb) as u8;
            }
        }
        let d = |c: usize| {
            let v = i32::from(h[s][c]) - i32::from(l[s][c]);
            (v * v) as u32
        };
        const E_DELTA_THRESH: u32 = 60;
        num_low_stddev += u32::from(d(0) + d(1) + d(2) < E_DELTA_THRESH);
    }
    if num_low_stddev == 2 {
        // Both subsets are non-solid here.
        let ep: [[(u32, u32); 3]; 2] =
            std::array::from_fn(|s| std::array::from_fn(|c| expand_ep(l[s][c], h[s][c], srgb)));
        let px: [Px; 16] = std::array::from_fn(|i| {
            let (s, w) = texel(i);
            let f = |c: usize| (ah::weight_interpolate(ep[s][c].0, ep[s][c].1, w) >> 8) as u8;
            [f(0), f(1), f(2), 255]
        });
        let flags = PACK_BC7_FLAG_USE_TRIVIAL_MODE6 | PACK_BC7_FLAG_PBIT_OPT_MODE6;
        fast_pack_bc7_rgb_analytical(block, &px, flags);
        return true;
    }
    let scale: [f32; 2] = std::array::from_fn(|s| {
        if low_w[s] == high_w[s] {
            0.0
        } else {
            7.0f32 / (high_w[s] - low_w[s]) as f32
        }
    });
    let w: [u8; 16] = std::array::from_fn(|i| {
        let (s, w) = texel(i);
        if solid[s] || low_w[s] == high_w[s] {
            return 0;
        }
        let f = (w as f32 - low_w[s] as f32) * scale[s];
        f2i(f + 0.5).clamp(0, 7) as u8
    });
    mode1(block, 13, &l, &h, &w);
    false
}

/// `pack_astc_6x6_to_two_subsets_middle_block`: the BC7 block at the center of 2x2 6x6 ASTC
/// blocks (`blocks[x][y]`), as mode 1 with top/bottom (or left/right) subsets whose endpoints
/// average the two blocks of each half.
pub(crate) fn pack_astc_6x6_to_two_subsets_middle_block(
    block: &mut Block,
    blocks: &[[&LogAstcBlock; 2]; 2],
    weights: &[[[u8; 36]; 2]; 2],
    do_left_right: bool,
) {
    let (p, q) = if do_left_right {
        ([blocks[0][0], blocks[1][0]], [blocks[0][1], blocks[1][1]])
    } else {
        ([blocks[0][0], blocks[0][1]], [blocks[1][0], blocks[1][1]])
    };
    let mut l = [[0u8; 4]; 2];
    let mut h = [[0u8; 4]; 2];
    for i in 0..2 {
        let (el, eh) = endpoints_or_solid(p[i]);
        let (el2, eh2) = endpoints_or_solid(q[i]);
        l[i] = el;
        h[i] = eh;
        for c in 0..3 {
            l[i][c] = ((u32::from(el[c]) + u32::from(el2[c]) + 1) >> 1) as u8;
            h[i][c] = ((u32::from(eh[c]) + u32::from(eh2[c]) + 1) >> 1) as u8;
        }
    }
    let w: [u8; 16] = std::array::from_fn(|i| {
        let (x, y) = (i & 3, i >> 2);
        let (bx, by) = (usize::from(x >= 2), usize::from(y >= 2));
        let sx = if x < 2 { x + 4 } else { x - 2 };
        let sy = if y < 2 { y + 4 } else { y - 2 };
        if blocks[bx][by].solid_color_flag_ldr {
            0
        } else {
            w3(weights[bx][by][sx + sy * 6])
        }
    });
    mode1(block, if do_left_right { 0 } else { 13 }, &l, &h, &w);
}
