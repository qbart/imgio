// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of `basisu_astc_ldr_encode.cpp` lines ~45-3560 (namespace `astc_ldr`): deblocking
//! filter helpers, weight grid requantization / downsampling, block error evaluation, the
//! QCD weight solver, weight polishing, the full trial encoders (`encode_trial*`), the float
//! "surrogate" encoders, partition estimation, Sobel filtering and DCT energy helpers.
//! `code_block_weights` lives in `dct_enc`.
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::excessive_precision,
    clippy::neg_cmp_op_on_partial_ord
)]

use super::common::{
    CemEncodeParams, PixelStats, Vec4F, apply_delta_to_bise_weight_val, cem_encode_pixels,
    cem_surrogate_encode_pixels, eval_solution, eval_solution_dp,
};
use super::grid_data::{AstcBlockGridData, find_astc_block_grid_data};
use super::partitions::{
    NUM_PART3_MAPPINGS, PartitionPatternVec, PartitionsData, texel_partitions,
};
use crate::astc::helpers::{
    BISE_64_LEVELS, LogAstcBlock, MAX_BLOCK_PIXELS, MAX_PARTITIONS, WeightedSample,
    decode_endpoint, dequant_tables, get_ise_levels, get_num_cem_values, upsample_weight_grid,
    weight_interpolate,
};
use crate::basis::ll::astc_hdr_enc::downsample_weight_grid;
use crate::basis::ll::comp::Image;
use crate::basis::ll::enc::ColorRgba;

const MAXP: usize = MAX_BLOCK_PIXELS;
const MAX_CEM_ENDPOINT_VALS: usize = 8;

/// `basisu::SMALL_FLOAT_VAL`.
pub(crate) const SMALL_FLOAT_VAL: f32 = 0.0000125;
/// `basisu::BIG_FLOAT_VAL`.
pub(crate) const BIG_FLOAT_VAL: f32 = 1e30;

/// `basisu::fast_roundf_pos_int`.
#[inline]
pub(crate) fn fast_roundf_pos_int(x: f32) -> i32 {
    (x + 0.5) as i32
}

/// `basisu::fast_roundf_int`.
#[inline]
pub(crate) fn fast_roundf_int(x: f32) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    }
}

#[inline]
fn sub4(a: &Vec4F, b: &Vec4F) -> Vec4F {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// `vec4F::dot`.
#[inline]
fn dot4(a: &Vec4F, b: &Vec4F) -> f32 {
    let mut r = a[0] * b[0];
    r += a[1] * b[1];
    r += a[2] * b[2];
    r += a[3] * b[3];
    r
}

/// `vec4F::squared_distance`.
#[inline]
fn squared_distance4(a: &Vec4F, b: &Vec4F) -> f32 {
    let mut d2 = 0.0f32;
    for i in 0..4 {
        let d = a[i] - b[i];
        d2 += d * d;
    }
    d2
}

// ---------------------------------------------------------------------------------------------
// Deblocking filter helpers

#[inline]
fn apply_kernel(a: u32, b: u32, c: u32) -> u32 {
    (a + b + c + 1) / 3
}

#[inline]
fn filter_horiz(src: &Image, x: i32, y: i32) -> ColorRgba {
    let (l, c, r) = (
        src.get_clamped(x - 1, y),
        src.get_clamped(x, y),
        src.get_clamped(x + 1, y),
    );
    std::array::from_fn(|ch| apply_kernel(l[ch] as u32, c[ch] as u32, r[ch] as u32) as u8)
}

#[inline]
fn filter_vert(src: &Image, x: i32, y: i32) -> ColorRgba {
    let (u, c, d) = (
        src.get_clamped(x, y - 1),
        src.get_clamped(x, y),
        src.get_clamped(x, y + 1),
    );
    std::array::from_fn(|ch| apply_kernel(u[ch] as u32, c[ch] as u32, d[ch] as u32) as u8)
}

/// `BASISU_CORNER`.
#[inline]
fn corner(src: &Image, sx: i32, sy: i32) -> ColorRgba {
    let h_l = src.get_clamped(sx - 1, sy);
    let h_c = src.get_clamped(sx, sy);
    let h_r = src.get_clamped(sx + 1, sy);
    let v_u = src.get_clamped(sx, sy - 1);
    let v_d = src.get_clamped(sx, sy + 1);
    std::array::from_fn(|c| {
        let s = h_l[c] as i32 + 2 * h_c[c] as i32 + h_r[c] as i32 + v_u[c] as i32 + v_d[c] as i32;
        fast_roundf_pos_int(s as f32 * (1.0f32 / 6.0)).min(255) as u8
    })
}

/// `deblock_block_region`: `bx`,`by` = texel offsets; `dst_tile` is `(fbw + 2) x (fbh + 2)`.
pub(crate) fn deblock_block_region(
    fbw: i32,
    fbh: i32,
    src_img: &Image,
    bx: i32,
    by: i32,
    dst_tile: &mut Image,
) {
    for ty in 0..(fbh + 2) {
        let on_horiz_edge = ty <= 1 || ty >= fbh;
        let sy = by - 1 + ty;
        for tx in 0..(fbw + 2) {
            let on_vert_edge = tx <= 1 || tx >= fbw;
            let sx = bx - 1 + tx;
            let out = if on_vert_edge && on_horiz_edge {
                corner(src_img, sx, sy)
            } else if on_vert_edge {
                filter_horiz(src_img, sx, sy)
            } else if on_horiz_edge {
                filter_vert(src_img, sx, sy)
            } else {
                src_img.get_clamped(sx, sy)
            };
            dst_tile.set_clipped(tx, ty, out);
        }
    }
}

/// `deblock_block_region_interior`.
pub(crate) fn deblock_block_region_interior(
    fbw: i32,
    fbh: i32,
    src_img: &Image,
    bx: i32,
    by: i32,
    dst_tile: &mut Image,
    dst_x: i32,
    dst_y: i32,
) {
    let x_left = bx;
    let x_right = bx + fbw - 1;
    let y_top = by;
    let y_bottom = by + fbh - 1;

    dst_tile.set_clipped(dst_x, dst_y, corner(src_img, x_left, y_top));
    dst_tile.set_clipped(dst_x + fbw - 1, dst_y, corner(src_img, x_right, y_top));
    dst_tile.set_clipped(dst_x, dst_y + fbh - 1, corner(src_img, x_left, y_bottom));
    dst_tile.set_clipped(
        dst_x + fbw - 1,
        dst_y + fbh - 1,
        corner(src_img, x_right, y_bottom),
    );

    // Top and bottom edges.
    let mut sy = y_top;
    while sy <= y_bottom {
        let ty = dst_y + (sy - by);
        for tx_offset in 1..fbw - 1 {
            let sx = bx + tx_offset;
            let out = filter_vert(src_img, sx, sy);
            dst_tile.set_clipped(dst_x + tx_offset, ty, out);
        }
        sy += fbh - 1;
    }

    // Left and right edges.
    let mut sx = x_left;
    while sx <= x_right {
        let tx = dst_x + (sx - bx);
        for ty_offset in 1..fbh - 1 {
            let sy = by + ty_offset;
            let out = filter_horiz(src_img, sx, sy);
            dst_tile.set_clipped(tx, dst_y + ty_offset, out);
        }
        sx += fbw - 1;
    }

    // Interior: pass-through copy (the C++ memcpy fast path gives the same result).
    for ty_offset in 1..(fbh - 1) {
        let sy = by + ty_offset;
        for tx_offset in 1..(fbw - 1) {
            let sx = bx + tx_offset;
            dst_tile.set_clipped(
                dst_x + tx_offset,
                dst_y + ty_offset,
                src_img.get_clamped(sx, sy),
            );
        }
    }
}

/// `deblock_image`.
pub(crate) fn deblock_image(fbw: i32, fbh: i32, src_img: &Image, dst_img: &mut Image) {
    dst_img.resize(src_img.get_width(), src_img.get_height());
    let num_blocks_x = src_img.get_block_width(fbw as u32);
    let num_blocks_y = src_img.get_block_height(fbh as u32);
    for by in 0..num_blocks_y {
        for bx in 0..num_blocks_x {
            let (x, y) = ((bx * fbw as u32) as i32, (by * fbh as u32) as i32);
            deblock_block_region_interior(fbw, fbh, src_img, x, y, dst_img, x, y);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Weight requantization / resampling

/// `astc_ldr_requantize_astc_weights`.
pub(crate) fn astc_ldr_requantize_astc_weights(
    n: u32,
    src_ise_vals: &[u8],
    from_ise_range: u32,
    dst_ise_vals: &mut [u8],
    to_ise_range: u32,
) {
    let n = n as usize;
    let dt = dequant_tables();
    if from_ise_range == to_ise_range {
        dst_ise_vals[..n].copy_from_slice(&src_ise_vals[..n]);
    } else if from_ise_range == BISE_64_LEVELS {
        let quant_tab = &dt.weight_tab(to_ise_range).val_to_ise;
        for i in 0..n {
            dst_ise_vals[i] = quant_tab[src_ise_vals[i] as usize];
        }
    } else if to_ise_range == BISE_64_LEVELS {
        let dequant_tab = &dt.weight_tab(from_ise_range).ise_to_val;
        for i in 0..n {
            dst_ise_vals[i] = dequant_tab[src_ise_vals[i] as usize];
        }
    } else {
        let dequant_tab = &dt.weight_tab(from_ise_range).ise_to_val;
        let quant_tab = &dt.weight_tab(to_ise_range).val_to_ise;
        for i in 0..n {
            dst_ise_vals[i] = quant_tab[dequant_tab[src_ise_vals[i] as usize] as usize];
        }
    }
}

/// `astc_ldr_downsample_ise_weights`.
pub(crate) fn astc_ldr_downsample_ise_weights(
    dequant_weight_ise_range: u32,
    quant_weight_ise_range: u32,
    block_w: u32,
    block_h: u32,
    grid_w: u32,
    grid_h: u32,
    src_weights: &[u8],
    dst_weights: &mut [u8],
    downsample_matrix: &[f32],
) {
    let dt = dequant_tables();
    let nb = (block_w * block_h) as usize;
    if block_w == grid_w && block_h == grid_h {
        if dequant_weight_ise_range != quant_weight_ise_range {
            astc_ldr_requantize_astc_weights(
                nb as u32,
                src_weights,
                dequant_weight_ise_range,
                dst_weights,
                quant_weight_ise_range,
            );
        } else {
            dst_weights[..nb].copy_from_slice(&src_weights[..nb]);
        }
        return;
    }

    let mut desired_weights = [0u8; MAXP];
    if dequant_weight_ise_range == BISE_64_LEVELS {
        desired_weights[..nb].copy_from_slice(&src_weights[..nb]);
    } else {
        let dequant_tab = &dt.weight_tab(dequant_weight_ise_range).ise_to_val;
        for i in 0..nb {
            desired_weights[i] = dequant_tab[src_weights[i] as usize];
        }
    }

    if quant_weight_ise_range == BISE_64_LEVELS {
        downsample_weight_grid(
            downsample_matrix,
            block_w,
            block_h,
            grid_w,
            grid_h,
            &desired_weights,
            dst_weights,
        );
    } else {
        let mut raw = [0u8; MAXP];
        downsample_weight_grid(
            downsample_matrix,
            block_w,
            block_h,
            grid_w,
            grid_h,
            &desired_weights,
            &mut raw,
        );
        let quant_tab = &dt.weight_tab(quant_weight_ise_range).val_to_ise;
        for i in 0..(grid_w * grid_h) as usize {
            dst_weights[i] = quant_tab[raw[i] as usize];
        }
    }
}

/// `downsample_weight_residual_grid`.
pub(crate) fn downsample_weight_residual_grid(
    matrix_weights: &[f32],
    bx: u32,
    by: u32,
    wx: u32,
    wy: u32,
    src_weights: &[i32],
    dst_weights: &mut [f32],
) {
    let n = (bx * by) as usize;
    let mut m = 0usize;
    for y in 0..wy {
        for x in 0..wx {
            let row = &matrix_weights[m..m + n];
            let mut total = 0.0f32;
            for i in 0..n {
                if row[i] != 0.0 {
                    total += row[i] * src_weights[i] as f32;
                }
            }
            dst_weights[(x + y * wx) as usize] = total;
            m += n;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// astc_helpers misc. weight / endpoint accessors

/// `astc_helpers::get_weight`.
#[inline]
pub(crate) fn get_weight(log_block: &LogAstcBlock, plane_index: u32, i: u32) -> u8 {
    let num_planes = if log_block.dual_plane { 2 } else { 1 };
    log_block.weights[(i * num_planes + plane_index) as usize]
}

/// `astc_helpers::get_weight` (mutable).
#[inline]
pub(crate) fn get_weight_mut(log_block: &mut LogAstcBlock, plane_index: u32, i: u32) -> &mut u8 {
    let num_planes = if log_block.dual_plane { 2 } else { 1 };
    &mut log_block.weights[(i * num_planes + plane_index) as usize]
}

/// `astc_helpers::extract_weights`.
pub(crate) fn extract_weights(log_block: &LogAstcBlock, weights: &mut [u8], plane_index: u32) {
    let num_planes = if log_block.dual_plane { 2 } else { 1 };
    let n = log_block.grid_width as u32 * log_block.grid_height as u32;
    for i in 0..n {
        weights[i as usize] = log_block.weights[(i * num_planes + plane_index) as usize];
    }
}

/// `astc_helpers::set_weights`.
pub(crate) fn set_weights(log_block: &mut LogAstcBlock, weights: &[u8], plane_index: u32) {
    let num_planes = if log_block.dual_plane { 2 } else { 1 };
    let n = log_block.grid_width as u32 * log_block.grid_height as u32;
    for i in 0..n {
        log_block.weights[(i * num_planes + plane_index) as usize] = weights[i as usize];
    }
}

/// `astc_helpers::get_endpoints` offset (start of a partition's endpoint values).
pub(crate) fn get_endpoints_ofs(log_block: &LogAstcBlock, partition_index: u32) -> usize {
    let mut ofs = 0;
    for i in 0..partition_index as usize {
        ofs += get_num_cem_values(log_block.color_endpoint_modes[i] as u32) as usize;
    }
    ofs
}

// ---------------------------------------------------------------------------------------------
// XUASTC LDR block decoding

/// `astc_helpers::xuastc_ldr_block_decoder` (texel decoder with cached endpoints; weights are
/// read from the log block passed to [`Self::decode_texel`]). Also used for whole block
/// decodes (`decode_block_xuastc_ldr` produces identical texels).
pub(crate) struct XuastcLdrBlockDecoder<'a> {
    blk_width: u32,
    upsample_weights: &'a [WeightedSample],
    solid_color: Option<ColorRgba>,
    weight_dequant: &'static [u8],
    /// [subset][comp][l/h], expanded to 16 bits.
    endpoints: [[[i32; 2]; 4]; 3],
    part: &'static [u8; 144],
    part_shift: u32,
    is_upsampling: bool,
    num_partitions: u32,
    dual_plane: bool,
    ccs: usize,
    grid_width: u32,
}

impl<'a> XuastcLdrBlockDecoder<'a> {
    /// `init`. `upsample_weights` = the grid data's `m_upsample_weights` (may be empty when
    /// the grid is not upsampled).
    pub fn new(
        log_blk: &LogAstcBlock,
        blk_width: u32,
        blk_height: u32,
        decode_mode_srgb: bool,
        upsample_weights: &'a [WeightedSample],
    ) -> Self {
        let tp = texel_partitions();
        let mut d = Self {
            blk_width,
            upsample_weights,
            solid_color: None,
            weight_dequant: &[],
            endpoints: [[[0; 2]; 4]; 3],
            part: &tp[(log_blk.partition_id & 1023) as usize],
            part_shift: 0,
            is_upsampling: false,
            num_partitions: log_blk.num_partitions as u32,
            dual_plane: log_blk.dual_plane,
            ccs: log_blk.color_component_selector as usize & 3,
            grid_width: log_blk.grid_width as u32,
        };
        if log_blk.solid_color_flag_ldr {
            d.solid_color = Some(std::array::from_fn(|c| (log_blk.solid_color(c) >> 8) as u8));
            return d;
        }
        let dt = dequant_tables();
        d.weight_dequant = dt.weight_ise_to_val(log_blk.weight_ise_range as u32);
        let ep_dequant = dt.endpoint_ise_to_val(log_blk.endpoint_ise_range as u32);
        let cem_index = log_blk.color_endpoint_modes[0] as u32;
        let num_cem_vals = get_num_cem_values(cem_index) as usize;
        let num_parts = (log_blk.num_partitions as usize).min(3);
        let mut dq = [0u8; 18];
        for i in 0..(num_cem_vals * num_parts).min(18) {
            dq[i] = ep_dequant
                .get(log_blk.endpoints[i] as usize)
                .copied()
                .unwrap_or(0);
        }
        for subset in 0..num_parts {
            let mut e = [[0i32; 2]; 4];
            decode_endpoint(cem_index, &mut e, &dq[subset * num_cem_vals..]);
            for c in 0..4 {
                for lh in 0..2 {
                    let v = e[c][lh];
                    e[c][lh] = if decode_mode_srgb {
                        (v << 8) | 0x80
                    } else {
                        (v << 8) | v
                    };
                }
            }
            d.endpoints[subset] = e;
        }
        let large_block = blk_width * blk_height >= 31;
        d.part_shift = if log_blk.num_partitions == 3 { 2 } else { 0 } + (large_block as u32) * 4;
        d.is_upsampling =
            log_blk.grid_width as u32 != blk_width || log_blk.grid_height as u32 != blk_height;
        d
    }

    /// `decode_texel`.
    #[inline]
    pub fn decode_texel(&self, log_blk: &LogAstcBlock, x: u32, y: u32) -> ColorRgba {
        if let Some(c) = self.solid_color {
            return c;
        }
        let texel_index = (x + y * self.blk_width) as usize;
        let w = &log_blk.weights;
        let dq = |i: usize| self.weight_dequant[w[i] as usize] as u32;
        let (w0, w1);
        if !self.is_upsampling {
            if self.dual_plane {
                w0 = dq(texel_index * 2);
                w1 = dq(texel_index * 2 + 1);
            } else {
                w0 = dq(texel_index);
                w1 = 0;
            }
        } else {
            let s = &self.upsample_weights[texel_index];
            let gw = self.grid_width as usize;
            let [[w00, w01], [w10, w11]] = s.weights.map(|r| r.map(u32::from));
            if !self.dual_plane {
                let idx0 = s.src_x as usize + s.src_y as usize * gw;
                let mut a = 8 + w00 * dq(idx0);
                if w01 != 0 {
                    a += w01 * dq(idx0 + 1);
                }
                if w10 != 0 {
                    a += w10 * dq(idx0 + gw);
                }
                if w11 != 0 {
                    a += w11 * dq(idx0 + 1 + gw);
                }
                w0 = a >> 4;
                w1 = 0;
            } else {
                let idx0 = (s.src_x as usize + s.src_y as usize * gw) * 2;
                let mut a = 8 + w00 * dq(idx0);
                let mut b = 8 + w00 * dq(idx0 + 1);
                if w01 != 0 {
                    a += w01 * dq(idx0 + 2);
                    b += w01 * dq(idx0 + 3);
                }
                if w10 != 0 {
                    let i2 = idx0 + gw * 2;
                    a += w10 * dq(i2);
                    b += w10 * dq(i2 + 1);
                }
                if w11 != 0 {
                    let i3 = idx0 + 2 + gw * 2;
                    a += w11 * dq(i3);
                    b += w11 * dq(i3 + 1);
                }
                w0 = a >> 4;
                w1 = b >> 4;
            }
        }
        let mut cw = [w0; 4];
        let mut subset = 0usize;
        if self.dual_plane {
            cw[self.ccs] = w1;
        } else if self.num_partitions > 1 {
            let v = self.part[(x + y * 12) as usize] as u32;
            subset = ((v >> self.part_shift) & 3) as usize;
        }
        let e = &self.endpoints[subset.min(2)];
        std::array::from_fn(|c| {
            (weight_interpolate(e[c][0] as u32, e[c][1] as u32, cw[c]) >> 8) as u8
        })
    }
}

/// Upsample weights of a block's grid (empty if the grid is not upsampled or invalid).
fn grid_upsample_weights(bw: u32, bh: u32, gw: u32, gh: u32) -> &'static [WeightedSample] {
    if gw == bw && gh == bh {
        return &[];
    }
    find_astc_block_grid_data(bw, bh, gw, gh).map_or(&[], |g| g.upsample_weights.as_slice())
}

/// `decode_block_xuastc_ldr` into `pixels` (block texels, row major).
pub(crate) fn decode_block_xuastc_ldr(
    log_blk: &LogAstcBlock,
    pixels: &mut [ColorRgba],
    bw: u32,
    bh: u32,
    decode_mode_srgb: bool,
) {
    let up = if log_blk.solid_color_flag_ldr {
        &[][..]
    } else {
        grid_upsample_weights(
            bw,
            bh,
            log_blk.grid_width as u32,
            log_blk.grid_height as u32,
        )
    };
    let dec = XuastcLdrBlockDecoder::new(log_blk, bw, bh, decode_mode_srgb, up);
    for y in 0..bh {
        for x in 0..bw {
            pixels[(x + y * bw) as usize] = dec.decode_texel(log_blk, x, y);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Error evaluation

/// `weighted_color_error`.
#[inline]
pub(crate) fn weighted_color_error(a: &ColorRgba, b: &ColorRgba, p: &CemEncodeParams) -> u32 {
    let sq = |c: usize| {
        let d = a[c] as i32 - b[c] as i32;
        (d * d) as u32
    };
    (sq(0) * p.comp_weights[0])
        .wrapping_add(sq(1) * p.comp_weights[1])
        .wrapping_add(sq(2) * p.comp_weights[2])
        .wrapping_add(sq(3) * p.comp_weights[3])
}

/// `eval_error(block_width, block_height, enc_log_block, pixel_stats, params)`.
pub(crate) fn eval_error(
    block_width: u32,
    block_height: u32,
    enc_log_block: &LogAstcBlock,
    pixel_stats: &PixelStats,
    params: &CemEncodeParams,
) -> u64 {
    let mut dec = [[0u8; 4]; MAXP];
    decode_block_xuastc_ldr(
        enc_log_block,
        &mut dec,
        block_width,
        block_height,
        params.decode_mode_srgb,
    );
    let n = (block_width * block_height) as usize;
    let mut total_err = 0u64;
    for i in 0..n {
        total_err += weighted_color_error(&dec[i], &pixel_stats.pixels[i], params) as u64;
    }
    total_err
}

/// `eval_error(..., cem_index, dual_plane_flag, ccs_index, ..., pEndpoint_vals, ...)` (the raw
/// parameter overload).
pub(crate) fn eval_error_raw(
    block_width: u32,
    block_height: u32,
    pixel_stats: &PixelStats,
    cem_index: u32,
    dual_plane_flag: bool,
    ccs_index: i32,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    grid_width: u32,
    grid_height: u32,
    endpoint_vals: &[u8],
    weight_grid_vals0: &[u8],
    weight_grid_vals1: &[u8],
    params: &CemEncodeParams,
) -> u64 {
    let total_grid_pixels = (grid_width * grid_height) as usize;
    let mut b = LogAstcBlock {
        grid_width: grid_width as u8,
        grid_height: grid_height as u8,
        weight_ise_range: weight_ise_range as u8,
        endpoint_ise_range: endpoint_ise_range as u8,
        num_partitions: 1,
        ..Default::default()
    };
    b.color_endpoint_modes[0] = cem_index as u8;
    let nv = get_num_cem_values(cem_index) as usize;
    b.endpoints[..nv].copy_from_slice(&endpoint_vals[..nv]);
    if dual_plane_flag {
        b.dual_plane = true;
        b.color_component_selector = ccs_index as u8;
        for i in 0..total_grid_pixels {
            b.weights[i * 2] = weight_grid_vals0[i];
            b.weights[i * 2 + 1] = weight_grid_vals1[i];
        }
    } else {
        b.weights[..total_grid_pixels].copy_from_slice(&weight_grid_vals0[..total_grid_pixels]);
    }
    eval_error(block_width, block_height, &b, pixel_stats, params)
}

/// `compute_psnr_from_wsse`.
pub(crate) fn compute_psnr_from_wsse(
    block_width: u32,
    block_height: u32,
    sse: u64,
    total_comp_weights: f32,
) -> f32 {
    let total_block_pixels = block_width * block_height;
    let wmse = sse as f32 / (total_comp_weights * total_block_pixels as f32);
    if wmse > 1e-5 {
        20.0 * (255.0f32 / wmse.sqrt()).log10()
    } else {
        10000.0
    }
}

// ---------------------------------------------------------------------------------------------
// Quantized coordinate descent

/// `qcd::qcd_min_solver`: quantized coordinate descent, quadratic objective.
#[derive(Default)]
pub(crate) struct QcdMinSolver {
    n: usize,
    k: usize,
    q: usize,
    /// N*K, column k at `[k*N]`.
    ucols: Vec<f32>,
    /// K, `||u_k||^2` (>= eps).
    alpha: Vec<f32>,
    /// Q sorted unique labels (raw [0,64] weights).
    labels: Vec<f32>,
}

impl QcdMinSolver {
    /// `init`: `u_rowmajor` is N x K.
    pub fn init(&mut self, u_rowmajor: &[f32], n: usize, k: usize, labels_u: &[i32]) -> bool {
        if n == 0 || k == 0 || labels_u.is_empty() {
            return false;
        }
        self.n = n;
        self.k = k;
        self.ucols.clear();
        self.ucols.resize(n * k, 0.0);
        for kk in 0..k {
            for t in 0..n {
                self.ucols[kk * n + t] = u_rowmajor[t * k + kk];
            }
        }
        self.alpha.clear();
        for kk in 0..k {
            let uk = &self.ucols[kk * n..kk * n + n];
            let mut a = 0.0f32;
            for t in 0..n {
                a += uk[t] * uk[t];
            }
            if !(a > 0.0) {
                a = 1e-8;
            }
            self.alpha.push(a);
        }
        self.labels.clear();
        self.labels.extend(labels_u.iter().map(|&l| l as f32));
        self.q = self.labels.len();
        true
    }

    /// `build_residual`: r = U*g - w*.
    pub fn build_residual(&self, g_idx: &[i32], w_star: &[f32], r_out: &mut [f32]) {
        let n = self.n;
        r_out[..n].iter_mut().for_each(|r| *r = 0.0);
        for kk in 0..self.k {
            let uk = &self.ucols[kk * n..kk * n + n];
            let s = self.labels[g_idx[kk] as usize];
            for t in 0..n {
                r_out[t] += s * uk[t];
            }
        }
        for t in 0..n {
            r_out[t] -= w_star[t];
        }
    }

    /// `sweep` (accept_eps = 1e-6): returns the number of moves accepted.
    pub fn sweep(&self, g_idx: &mut [i32], r_io: &mut [f32]) -> i32 {
        let accept_eps = 1e-6f32;
        let n = self.n;
        let q = self.q as i32;
        let mut num_moved = 0;
        for kk in 0..self.k {
            let uk = &self.ucols[kk * n..kk * n + n];
            let mut beta = 0.0f32;
            for t in 0..n {
                beta += r_io[t] * uk[t];
            }
            let a = self.alpha[kk];
            let cur_u = self.labels[g_idx[kk] as usize];
            let s_star = cur_u - beta / a;
            let j0 = self.nearest_label_idx(s_star);
            let cand = [
                j0,
                if j0 + 1 < q { j0 + 1 } else { j0 },
                if j0 > 0 { j0 - 1 } else { j0 },
            ];
            let mut best_j = g_idx[kk];
            let mut best_de = 0.0f32;
            for &j in &cand {
                if j == g_idx[kk] {
                    continue;
                }
                let s = self.labels[j as usize];
                let d = s - cur_u;
                let de = 2.0 * d * beta + d * d * a;
                if best_j == g_idx[kk] || de < best_de {
                    best_de = de;
                    best_j = j;
                }
            }
            if best_j != g_idx[kk] && best_de < -accept_eps {
                let d = self.labels[best_j as usize] - cur_u;
                for t in 0..n {
                    r_io[t] += d * uk[t];
                }
                g_idx[kk] = best_j;
                num_moved += 1;
            }
        }
        num_moved
    }

    /// `residual_energy`.
    pub fn residual_energy(&self, r: &[f32]) -> f32 {
        let mut e = 0.0f32;
        for t in 0..self.n {
            e += r[t] * r[t];
        }
        e
    }

    fn nearest_label_idx(&self, x: f32) -> i32 {
        let q = self.q as i32;
        if q <= 1 {
            return 0;
        }
        if x <= self.labels[0] {
            return 0;
        }
        if x >= self.labels[(q - 1) as usize] {
            return q - 1;
        }
        let (mut lo, mut hi) = (0i32, q - 1);
        while hi - lo > 1 {
            let mid = (lo + hi) >> 1;
            if x >= self.labels[mid as usize] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let dlo = (x - self.labels[lo as usize]).abs();
        let dhi = (x - self.labels[hi as usize]).abs();
        if dlo <= dhi { lo } else { hi }
    }
}

// ---------------------------------------------------------------------------------------------
// Weight polishing

const NUM_WEIGHT_POLISH_PASSES: u32 = 1;

/// `polish_block_weights_final_fast`: true if improved.
pub(crate) fn polish_block_weights_final_fast(
    enc_log_block: &LogAstcBlock,
    weights0: &mut [u8],
    weights1: &mut [u8],
    block_width: u32,
    block_height: u32,
    grid_width: u32,
    grid_height: u32,
    pixel_stats: &PixelStats,
    params: &CemEncodeParams,
    grid_data: &AstcBlockGridData,
    cur_err: &mut u64,
) -> bool {
    if *cur_err == 0 {
        return false;
    }
    let dual_plane_flag = enc_log_block.dual_plane;
    let weight_ise_range = enc_log_block.weight_ise_range as u32;
    let mut improved_flag = false;

    let mut cur = *enc_log_block;
    set_weights(&mut cur, weights0, 0);
    if dual_plane_flag {
        set_weights(&mut cur, weights1, 1);
    }
    let dec = XuastcLdrBlockDecoder::new(
        &cur,
        block_width,
        block_height,
        params.decode_mode_srgb,
        &grid_data.upsample_weights,
    );

    for _ in 0..NUM_WEIGHT_POLISH_PASSES {
        for y in 0..grid_height {
            for x in 0..grid_width {
                let gi = x + y * grid_width;
                let influenced = &grid_data.grid_to_texel_influence_list[gi as usize];
                for plane_iter in 0..(if dual_plane_flag { 2 } else { 1 }) {
                    let base_weight = get_weight(&cur, plane_iter, gi);
                    for delta in [-1i32, 1] {
                        let new_weight = apply_delta_to_bise_weight_val(
                            weight_ise_range,
                            base_weight as i32,
                            delta,
                        ) as u8;
                        if new_weight == base_weight {
                            continue;
                        }
                        let mut trial_err = *cur_err;
                        for &p in influenced {
                            let (tx, ty) = ((p & 0xFF) as u32, (p >> 8) as u32);
                            let dc = dec.decode_texel(&cur, tx, ty);
                            let ti = (tx + ty * block_width) as usize;
                            trial_err = trial_err.wrapping_sub(weighted_color_error(
                                &dc,
                                &pixel_stats.pixels[ti],
                                params,
                            )
                                as u64);
                        }
                        let saved_weight = get_weight(&cur, plane_iter, gi);
                        *get_weight_mut(&mut cur, plane_iter, gi) = new_weight;
                        for &p in influenced {
                            let (tx, ty) = ((p & 0xFF) as u32, (p >> 8) as u32);
                            let dc = dec.decode_texel(&cur, tx, ty);
                            let ti = (tx + ty * block_width) as usize;
                            trial_err = trial_err.wrapping_add(weighted_color_error(
                                &dc,
                                &pixel_stats.pixels[ti],
                                params,
                            )
                                as u64);
                        }
                        if trial_err < *cur_err {
                            *cur_err = trial_err;
                            improved_flag = true;
                        } else {
                            *get_weight_mut(&mut cur, plane_iter, gi) = saved_weight;
                        }
                    }
                }
            }
        }
    }

    if improved_flag {
        extract_weights(&cur, weights0, 0);
        if dual_plane_flag {
            extract_weights(&cur, weights1, 1);
        }
    }
    improved_flag
}

/// Splits the block's pixels into per-subset lists (`part_pixels`, `num_part_pixels`).
fn split_subset_pixels(
    block_width: u32,
    block_height: u32,
    pixels: &[ColorRgba],
    pat: &PartitionPatternVec,
    part_pixels: &mut [[ColorRgba; MAXP]; 4],
    num_part_pixels: &mut [u32; 4],
) {
    *num_part_pixels = [0; 4];
    for y in 0..block_height {
        for x in 0..block_width {
            let px = pixels[(x + y * block_width) as usize];
            let pi = pat.parts[(x + y * block_width) as usize] as usize & 3;
            part_pixels[pi][num_part_pixels[pi] as usize] = px;
            num_part_pixels[pi] += 1;
        }
    }
}

fn new_pixel_stats_array() -> [PixelStats; 4] {
    std::array::from_fn(|_| PixelStats::default())
}

/// `polish_block_weights` (1-3 subsets, requires initial weights). Always returns true.
pub(crate) fn polish_block_weights(
    block_width: u32,
    block_height: u32,
    pixel_stats: &PixelStats,
    enc_log_block: &mut LogAstcBlock,
    params: &CemEncodeParams,
    pat: Option<&PartitionPatternVec>,
    improved_flag: &mut bool,
    gradient_descent_flag: bool,
    polish_weights_flag: bool,
    qcd_enabled_flag: bool,
) -> bool {
    *improved_flag = false;
    if !gradient_descent_flag && !polish_weights_flag && !qcd_enabled_flag {
        return true;
    }

    let grid_width = enc_log_block.grid_width as u32;
    let grid_height = enc_log_block.grid_height as u32;
    let cem_index = enc_log_block.color_endpoint_modes[0] as u32;
    let num_subsets = enc_log_block.num_partitions as u32;
    let dual_plane_flag = enc_log_block.dual_plane;
    let ccs_index = enc_log_block.color_component_selector as u32;
    let endpoint_ise_range = enc_log_block.endpoint_ise_range as u32;
    let weight_ise_range = enc_log_block.weight_ise_range as u32;

    let dt = dequant_tables();
    let wtab = dt.weight_tab(weight_ise_range);
    let dequant_tab = &wtab.ise_to_val;
    let quant_tab = &wtab.val_to_ise;

    let Some(grid_data) =
        find_astc_block_grid_data(block_width, block_height, grid_width, grid_height)
    else {
        return true;
    };

    let total_block_pixels = (block_width * block_height) as usize;
    let total_grid_pixels = (grid_width * grid_height) as usize;

    let mut cur_err = eval_error(
        block_width,
        block_height,
        enc_log_block,
        pixel_stats,
        params,
    );

    let mut weights0 = [0u8; MAXP];
    let mut weights1 = [0u8; MAXP];
    extract_weights(enc_log_block, &mut weights0, 0);
    if dual_plane_flag {
        extract_weights(enc_log_block, &mut weights1, 1);
    }

    if gradient_descent_flag {
        let mut ideal0 = [0u8; MAXP];
        let mut ideal1 = [0u8; MAXP];

        if num_subsets == 1 {
            if dual_plane_flag {
                eval_solution_dp(
                    pixel_stats,
                    cem_index,
                    ccs_index,
                    &enc_log_block.endpoints,
                    endpoint_ise_range,
                    &mut ideal0,
                    &mut ideal1,
                    BISE_64_LEVELS,
                    params,
                );
            } else {
                eval_solution(
                    pixel_stats,
                    cem_index,
                    &enc_log_block.endpoints,
                    endpoint_ise_range,
                    &mut ideal0,
                    BISE_64_LEVELS,
                    params,
                );
            }
        } else if let Some(pat) = pat {
            let mut part_pixels = [[[0u8; 4]; MAXP]; 4];
            let mut num_part_pixels = [0u32; 4];
            split_subset_pixels(
                block_width,
                block_height,
                &pixel_stats.pixels,
                pat,
                &mut part_pixels,
                &mut num_part_pixels,
            );
            let mut part_stats = new_pixel_stats_array();
            let mut part_raw_weights = [[0u8; MAXP]; 4];
            for pi in 0..num_subsets as usize {
                part_stats[pi].init(num_part_pixels[pi], &part_pixels[pi]);
                let ofs = get_endpoints_ofs(enc_log_block, pi as u32);
                eval_solution(
                    &part_stats[pi],
                    cem_index,
                    &enc_log_block.endpoints[ofs..],
                    endpoint_ise_range,
                    &mut part_raw_weights[pi],
                    BISE_64_LEVELS,
                    params,
                );
            }
            num_part_pixels = [0; 4];
            for i in 0..total_block_pixels {
                let pi = pat.parts[i] as usize & 3;
                ideal0[i] = part_raw_weights[pi][num_part_pixels[pi] as usize];
                num_part_pixels[pi] += 1;
            }
        }

        // Current upsampled raw [0,64] weights.
        let mut dq0 = [0u8; MAXP];
        let mut dq1 = [0u8; MAXP];
        let mut up0 = [0u8; MAXP];
        let mut up1 = [0u8; MAXP];
        astc_ldr_requantize_astc_weights(
            total_grid_pixels as u32,
            &weights0,
            weight_ise_range,
            &mut dq0,
            BISE_64_LEVELS,
        );
        if dual_plane_flag {
            astc_ldr_requantize_astc_weights(
                total_grid_pixels as u32,
                &weights1,
                weight_ise_range,
                &mut dq1,
                BISE_64_LEVELS,
            );
        }
        upsample_weight_grid(
            block_width,
            block_height,
            grid_width,
            grid_height,
            &dq0,
            &mut up0,
        );
        if dual_plane_flag {
            upsample_weight_grid(
                block_width,
                block_height,
                grid_width,
                grid_height,
                &dq1,
                &mut up1,
            );
        }

        let mut resid0 = [0i32; MAXP];
        let mut resid1 = [0i32; MAXP];
        for i in 0..total_block_pixels {
            resid0[i] = ideal0[i] as i32 - up0[i] as i32;
        }
        if dual_plane_flag {
            for i in 0..total_block_pixels {
                resid1[i] = ideal1[i] as i32 - up1[i] as i32;
            }
        }

        let mut gr0 = [0f32; MAXP];
        let mut gr1 = [0f32; MAXP];
        let udm = &grid_data.unweighted_downsample_matrix;
        let inv_diag = &grid_data.one_over_diag_ata;
        downsample_weight_residual_grid(
            udm,
            block_width,
            block_height,
            grid_width,
            grid_height,
            &resid0,
            &mut gr0,
        );
        for i in 0..total_grid_pixels {
            gr0[i] *= inv_diag[i];
        }
        if dual_plane_flag {
            downsample_weight_residual_grid(
                udm,
                block_width,
                block_height,
                grid_width,
                grid_height,
                &resid1,
                &mut gr1,
            );
            for i in 0..total_grid_pixels {
                gr1[i] *= inv_diag[i];
            }
        }

        let q = 1.0f32;
        let mut refined0 = [0u8; MAXP];
        let mut refined1 = [0u8; MAXP];
        for i in 0..total_grid_pixels {
            let v = dequant_tab[weights0[i] as usize] as f32 + gr0[i] * q;
            let iv = (v.round() as i32).clamp(0, 64);
            refined0[i] = quant_tab[iv as usize];
        }
        if dual_plane_flag {
            for i in 0..total_grid_pixels {
                let v = dequant_tab[weights1[i] as usize] as f32 + gr1[i] * q;
                let iv = (v.round() as i32).clamp(0, 64);
                refined1[i] = quant_tab[iv as usize];
            }
        }

        let mut refined_log_block = *enc_log_block;
        set_weights(&mut refined_log_block, &refined0, 0);
        if dual_plane_flag {
            set_weights(&mut refined_log_block, &refined1, 1);
        }
        let mut refined_err = eval_error(
            block_width,
            block_height,
            &refined_log_block,
            pixel_stats,
            params,
        );
        if refined_err < cur_err {
            cur_err = refined_err;
            weights0[..total_grid_pixels].copy_from_slice(&refined0[..total_grid_pixels]);
            if dual_plane_flag {
                weights1[..total_grid_pixels].copy_from_slice(&refined1[..total_grid_pixels]);
            }
            *improved_flag = true;
        }

        if qcd_enabled_flag {
            let mut ideal_f0 = [0f32; MAXP];
            let mut ideal_f1 = [0f32; MAXP];
            for i in 0..total_block_pixels {
                ideal_f0[i] = ideal0[i] as f32;
                if dual_plane_flag {
                    ideal_f1[i] = ideal1[i] as f32;
                }
            }

            let mut solver = QcdMinSolver::default();
            let num_weight_levels = get_ise_levels(weight_ise_range) as usize;
            let mut labels = [0i32; 33];
            for i in 0..num_weight_levels {
                labels[i] = dequant_tab[wtab.rank_to_ise[i] as usize] as i32;
            }
            solver.init(
                &grid_data.upsample_matrix,
                total_block_pixels,
                total_grid_pixels,
                &labels[..num_weight_levels],
            );

            let mut grid_idx0 = [0i32; MAXP];
            let mut grid_idx1 = [0i32; MAXP];
            let ise_to_rank = &wtab.ise_to_rank;
            for i in 0..total_grid_pixels {
                grid_idx0[i] = ise_to_rank[refined0[i] as usize] as i32;
                if dual_plane_flag {
                    grid_idx1[i] = ise_to_rank[refined1[i] as usize] as i32;
                }
            }

            let mut r0 = [0f32; MAXP];
            let mut r1 = [0f32; MAXP];
            const MAX_QCD_SWEEPS: u32 = 5;
            solver.build_residual(&grid_idx0, &ideal_f0, &mut r0);
            for _ in 0..MAX_QCD_SWEEPS {
                if solver.sweep(&mut grid_idx0, &mut r0) == 0 {
                    break;
                }
            }
            if dual_plane_flag {
                solver.build_residual(&grid_idx1, &ideal_f1, &mut r1);
                for _ in 0..MAX_QCD_SWEEPS {
                    if solver.sweep(&mut grid_idx1, &mut r1) == 0 {
                        break;
                    }
                }
            }

            let rank_to_ise = &wtab.rank_to_ise;
            for i in 0..total_grid_pixels {
                refined0[i] = rank_to_ise[grid_idx0[i] as usize];
                if dual_plane_flag {
                    refined1[i] = rank_to_ise[grid_idx1[i] as usize];
                }
            }

            refined_log_block = *enc_log_block;
            set_weights(&mut refined_log_block, &refined0, 0);
            if dual_plane_flag {
                set_weights(&mut refined_log_block, &refined1, 1);
            }
            refined_err = eval_error(
                block_width,
                block_height,
                &refined_log_block,
                pixel_stats,
                params,
            );
            if refined_err < cur_err {
                cur_err = refined_err;
                weights0[..total_grid_pixels].copy_from_slice(&refined0[..total_grid_pixels]);
                if dual_plane_flag {
                    weights1[..total_grid_pixels].copy_from_slice(&refined1[..total_grid_pixels]);
                }
                *improved_flag = true;
            }
        }
    }

    if polish_weights_flag
        && polish_block_weights_final_fast(
            enc_log_block,
            &mut weights0,
            &mut weights1,
            block_width,
            block_height,
            grid_width,
            grid_height,
            pixel_stats,
            params,
            grid_data,
            &mut cur_err,
        )
    {
        *improved_flag = true;
    }

    if *improved_flag {
        set_weights(enc_log_block, &weights0, 0);
        if dual_plane_flag {
            set_weights(enc_log_block, &weights1, 1);
        }
    }
    true
}

// ---------------------------------------------------------------------------------------------
// Full trial encoders

/// Options shared by the trial encoders (the C++ default arguments).
#[derive(Clone, Copy, Debug)]
pub(crate) struct TrialFlags {
    pub gradient_descent_flag: bool,
    pub polish_weights_flag: bool,
    pub qcd_enabled_flag: bool,
    pub use_blue_contraction: bool,
}

impl Default for TrialFlags {
    fn default() -> Self {
        Self {
            gradient_descent_flag: true,
            polish_weights_flag: true,
            qcd_enabled_flag: true,
            use_blue_contraction: true,
        }
    }
}

#[inline]
fn early_out(early_out_thresh: f32, pass: u32, prev_cur_err: u64, cur_err: u64) -> bool {
    if early_out_thresh != 0.0 && pass != 0 && prev_cur_err != 0 {
        let pct = prev_cur_err.wrapping_sub(cur_err) as f64 / prev_cur_err as f64;
        if pct < early_out_thresh as f64 {
            return true;
        }
    }
    false
}

/// `encode_trial_subsets` (2-3 subsets). `pat_seed_index` is the ASTC partition seed;
/// `try_direct_encoding_flag` receives `*pTry_direct_encoding_flag`.
pub(crate) fn encode_trial_subsets(
    block_width: u32,
    block_height: u32,
    pixel_stats: &PixelStats,
    cem_index: u32,
    num_parts: u32,
    pat_seed_index: u32,
    pat: &PartitionPatternVec,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    grid_width: u32,
    grid_height: u32,
    early_out_thresh: f32,
    enc_log_block: &mut LogAstcBlock,
    params: &CemEncodeParams,
    refine_only_flag: bool,
    flags: TrialFlags,
    mut try_direct_encoding_flag: Option<&mut bool>,
) -> bool {
    if let Some(f) = try_direct_encoding_flag.as_deref_mut() {
        *f = false;
    }
    let is_downsampling = grid_width < block_width || grid_height < block_height;
    let total_grid_pixels = (grid_width * grid_height) as usize;
    let np = num_parts as usize;

    let mut part_pixels = [[[0u8; 4]; MAXP]; 4];
    let mut num_part_pixels = [0u32; 4];
    split_subset_pixels(
        block_width,
        block_height,
        &pixel_stats.pixels,
        pat,
        &mut part_pixels,
        &mut num_part_pixels,
    );

    let mut part_stats = new_pixel_stats_array();
    let mut part_endpoints = [[0u8; MAX_CEM_ENDPOINT_VALS]; 4];
    let mut part_weights = [[0u8; MAXP]; 4];

    for pi in 0..np {
        part_stats[pi].init(num_part_pixels[pi], &part_pixels[pi]);
        if !refine_only_flag {
            let mut tdef = false;
            let block_err = cem_encode_pixels(
                cem_index,
                -1,
                &part_stats[pi],
                params,
                endpoint_ise_range,
                weight_ise_range,
                &mut part_endpoints[pi],
                &mut part_weights[pi],
                None,
                u64::MAX,
                flags.use_blue_contraction,
                Some(&mut tdef),
            );
            if block_err == u64::MAX {
                return false;
            }
            if tdef && let Some(f) = try_direct_encoding_flag.as_deref_mut() {
                *f = true;
            }
        }
    }

    let num_endpoint_vals = get_num_cem_values(cem_index) as usize;

    if !refine_only_flag {
        let mut block_weights = [0u8; MAXP];
        num_part_pixels = [0; 4];
        for i in 0..(block_width * block_height) as usize {
            let pi = pat.parts[i] as usize & 3;
            block_weights[i] = part_weights[pi][num_part_pixels[pi] as usize];
            num_part_pixels[pi] += 1;
        }

        enc_log_block.clear();
        enc_log_block.grid_width = grid_width as u8;
        enc_log_block.grid_height = grid_height as u8;
        enc_log_block.weight_ise_range = weight_ise_range as u8;
        enc_log_block.endpoint_ise_range = endpoint_ise_range as u8;
        enc_log_block.num_partitions = num_parts as u8;
        for i in 0..np {
            enc_log_block.color_endpoint_modes[i] = cem_index as u8;
        }
        enc_log_block.partition_id = pat_seed_index as u16;

        if is_downsampling {
            let Some(gd) =
                find_astc_block_grid_data(block_width, block_height, grid_width, grid_height)
            else {
                return false;
            };
            astc_ldr_downsample_ise_weights(
                weight_ise_range,
                weight_ise_range,
                block_width,
                block_height,
                grid_width,
                grid_height,
                &block_weights,
                &mut enc_log_block.weights,
                &gd.downsample_matrix,
            );
        } else {
            enc_log_block.weights[..total_grid_pixels]
                .copy_from_slice(&block_weights[..total_grid_pixels]);
        }

        for p in 0..np {
            enc_log_block.endpoints[num_endpoint_vals * p..num_endpoint_vals * (p + 1)]
                .copy_from_slice(&part_endpoints[p][..num_endpoint_vals]);
        }
    }

    let wdq = dequant_tables().weight_ise_to_val(weight_ise_range);
    let mut prev_cur_err = u64::MAX;
    const NUM_REFINEMENT_PASSES: u32 = 3;
    for refine_pass in 0..NUM_REFINEMENT_PASSES {
        let cur_err = eval_error(
            block_width,
            block_height,
            enc_log_block,
            pixel_stats,
            params,
        );
        if cur_err == 0 {
            break;
        }
        if early_out(early_out_thresh, refine_pass, prev_cur_err, cur_err) {
            break;
        }
        prev_cur_err = cur_err;

        let mut dq0 = [0u8; MAXP];
        let mut up0 = [0u8; MAXP];
        for i in 0..total_grid_pixels {
            dq0[i] = wdq[enc_log_block.weights[i] as usize];
        }
        upsample_weight_grid(
            block_width,
            block_height,
            grid_width,
            grid_height,
            &dq0,
            &mut up0,
        );

        let mut alt = *enc_log_block;

        let mut raw_part_weights = [[0u8; MAXP]; 4];
        num_part_pixels = [0; 4];
        for i in 0..(block_width * block_height) as usize {
            let pi = pat.parts[i] as usize & 3;
            raw_part_weights[pi][num_part_pixels[pi] as usize] = up0[i];
            num_part_pixels[pi] += 1;
        }

        for pi in 0..np {
            let mut temp_params = *params;
            temp_params.forced_weight_vals0 = Some(&raw_part_weights[pi]);
            let mut temp_weights = [0u8; MAXP];
            let mut tdef = false;
            let block_err = cem_encode_pixels(
                cem_index,
                -1,
                &part_stats[pi],
                &temp_params,
                endpoint_ise_range,
                BISE_64_LEVELS,
                &mut alt.endpoints[num_endpoint_vals * pi..],
                &mut temp_weights,
                None,
                u64::MAX,
                flags.use_blue_contraction,
                Some(&mut tdef),
            );
            if block_err == u64::MAX {
                return false;
            }
            if tdef && let Some(f) = try_direct_encoding_flag.as_deref_mut() {
                *f = true;
            }
        }

        let ref_err = eval_error(block_width, block_height, &alt, pixel_stats, params);
        if ref_err < cur_err {
            *enc_log_block = alt;
        }

        if refine_pass == NUM_REFINEMENT_PASSES - 1 {
            break;
        }

        if is_downsampling && (flags.gradient_descent_flag || flags.polish_weights_flag) {
            let mut improved_flag = false;
            polish_block_weights(
                block_width,
                block_height,
                pixel_stats,
                enc_log_block,
                params,
                Some(pat),
                &mut improved_flag,
                flags.gradient_descent_flag,
                flags.polish_weights_flag,
                flags.qcd_enabled_flag,
            );
            if !improved_flag {
                break;
            }
        } else {
            break;
        }
    }
    true
}

/// `encode_trial` (1 subset, single or dual plane; `ccs_index` = -1 for single plane).
pub(crate) fn encode_trial(
    block_width: u32,
    block_height: u32,
    pixel_stats: &PixelStats,
    cem_index: u32,
    dual_plane_flag: bool,
    ccs_index: i32,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    grid_width: u32,
    grid_height: u32,
    early_out_thresh: f32,
    enc_log_block: &mut LogAstcBlock,
    params: &CemEncodeParams,
    flags: TrialFlags,
    mut try_direct_encoding_flag: Option<&mut bool>,
) -> bool {
    if let Some(f) = try_direct_encoding_flag.as_deref_mut() {
        *f = false;
    }
    let is_downsampling = grid_width < block_width || grid_height < block_height;
    let total_grid_pixels = (grid_width * grid_height) as usize;
    let dequant_tab = dequant_tables().weight_ise_to_val(weight_ise_range);

    enc_log_block.clear();
    enc_log_block.grid_width = grid_width as u8;
    enc_log_block.grid_height = grid_height as u8;
    enc_log_block.weight_ise_range = weight_ise_range as u8;
    enc_log_block.endpoint_ise_range = endpoint_ise_range as u8;
    enc_log_block.dual_plane = dual_plane_flag;
    if dual_plane_flag {
        enc_log_block.color_component_selector = ccs_index as u8;
    }
    enc_log_block.num_partitions = 1;
    enc_log_block.color_endpoint_modes[0] = cem_index as u8;

    let nev = get_num_cem_values(cem_index) as usize;
    let mut fullres_endpoints = [0u8; MAX_CEM_ENDPOINT_VALS];
    let mut weights0 = [0u8; MAXP];
    let mut weights1 = [0u8; MAXP];

    if !is_downsampling {
        let mut tdef = false;
        let block_err = cem_encode_pixels(
            cem_index,
            ccs_index,
            pixel_stats,
            params,
            endpoint_ise_range,
            weight_ise_range,
            &mut fullres_endpoints,
            &mut weights0,
            Some(&mut weights1),
            u64::MAX,
            flags.use_blue_contraction,
            Some(&mut tdef),
        );
        if block_err == u64::MAX {
            return false;
        }
        if tdef && let Some(f) = try_direct_encoding_flag.as_deref_mut() {
            *f = tdef;
        }
        if dual_plane_flag {
            for i in 0..total_grid_pixels {
                enc_log_block.weights[i * 2] = weights0[i];
                enc_log_block.weights[i * 2 + 1] = weights1[i];
            }
        } else {
            enc_log_block.weights[..total_grid_pixels]
                .copy_from_slice(&weights0[..total_grid_pixels]);
        }
        enc_log_block.endpoints[..nev].copy_from_slice(&fullres_endpoints[..nev]);
        return true;
    }

    let Some(gd) = find_astc_block_grid_data(block_width, block_height, grid_width, grid_height)
    else {
        return false;
    };
    let downsample_matrix = &gd.downsample_matrix;

    let mut fr0 = [0u8; MAXP];
    let mut fr1 = [0u8; MAXP];
    let mut tdef = false;
    let block_err = cem_encode_pixels(
        cem_index,
        ccs_index,
        pixel_stats,
        params,
        endpoint_ise_range,
        weight_ise_range,
        &mut fullres_endpoints,
        &mut fr0,
        Some(&mut fr1),
        u64::MAX,
        flags.use_blue_contraction,
        Some(&mut tdef),
    );
    if block_err == u64::MAX {
        return false;
    }
    if tdef && let Some(f) = try_direct_encoding_flag.as_deref_mut() {
        *f = tdef;
    }

    astc_ldr_downsample_ise_weights(
        weight_ise_range,
        weight_ise_range,
        block_width,
        block_height,
        grid_width,
        grid_height,
        &fr0,
        &mut weights0,
        downsample_matrix,
    );
    set_weights(enc_log_block, &weights0, 0);
    if dual_plane_flag {
        astc_ldr_downsample_ise_weights(
            weight_ise_range,
            weight_ise_range,
            block_width,
            block_height,
            grid_width,
            grid_height,
            &fr1,
            &mut weights1,
            downsample_matrix,
        );
        set_weights(enc_log_block, &weights1, 1);
    }
    enc_log_block.endpoints[..nev].copy_from_slice(&fullres_endpoints[..nev]);

    let mut prev_cur_err = u64::MAX;
    const NUM_OUTER_PASSES: u32 = 3;
    for outer_pass in 0..NUM_OUTER_PASSES {
        let cur_err = eval_error_raw(
            block_width,
            block_height,
            pixel_stats,
            cem_index,
            dual_plane_flag,
            ccs_index,
            endpoint_ise_range,
            weight_ise_range,
            grid_width,
            grid_height,
            &enc_log_block.endpoints,
            &weights0,
            &weights1,
            params,
        );
        if cur_err == 0 {
            break;
        }
        if early_out(early_out_thresh, outer_pass, prev_cur_err, cur_err) {
            break;
        }
        prev_cur_err = cur_err;

        // Endpoint refinement given the current upsampled weights.
        {
            extract_weights(enc_log_block, &mut weights0, 0);
            if dual_plane_flag {
                extract_weights(enc_log_block, &mut weights1, 1);
            }
            let mut dq0 = [0u8; MAXP];
            let mut up0 = [0u8; MAXP];
            for i in 0..total_grid_pixels {
                dq0[i] = dequant_tab[weights0[i] as usize];
            }
            upsample_weight_grid(
                block_width,
                block_height,
                grid_width,
                grid_height,
                &dq0,
                &mut up0,
            );
            let mut dq1 = [0u8; MAXP];
            let mut up1 = [0u8; MAXP];
            if dual_plane_flag {
                for i in 0..total_grid_pixels {
                    dq1[i] = dequant_tab[weights1[i] as usize];
                }
                upsample_weight_grid(
                    block_width,
                    block_height,
                    grid_width,
                    grid_height,
                    &dq1,
                    &mut up1,
                );
            }

            let mut refine_params = *params;
            refine_params.forced_weight_vals0 = Some(&up0);
            if dual_plane_flag {
                refine_params.forced_weight_vals1 = Some(&up1);
            }

            let mut refined_endpoints = [0u8; MAX_CEM_ENDPOINT_VALS];
            let mut rw0 = [0u8; MAXP];
            let mut rw1 = [0u8; MAXP];
            let refined_block_err = cem_encode_pixels(
                cem_index,
                ccs_index,
                pixel_stats,
                &refine_params,
                endpoint_ise_range,
                BISE_64_LEVELS,
                &mut refined_endpoints,
                &mut rw0,
                Some(&mut rw1),
                u64::MAX,
                flags.use_blue_contraction,
                Some(&mut tdef),
            );
            if refined_block_err != u64::MAX && refined_block_err < cur_err {
                enc_log_block.endpoints[..nev].copy_from_slice(&refined_endpoints[..nev]);
                if tdef && let Some(f) = try_direct_encoding_flag.as_deref_mut() {
                    *f = tdef;
                }
            }
        }

        if outer_pass == NUM_OUTER_PASSES - 1
            || (!flags.gradient_descent_flag && !flags.polish_weights_flag)
        {
            break;
        }

        let mut improved_flag = false;
        polish_block_weights(
            block_width,
            block_height,
            pixel_stats,
            enc_log_block,
            params,
            None,
            &mut improved_flag,
            flags.gradient_descent_flag,
            flags.polish_weights_flag,
            flags.qcd_enabled_flag,
        );
        if !improved_flag {
            break;
        }
    }
    true
}

/// `encode_trial_refine_only` (1 subset only): refines the endpoints given the current
/// weights.
pub(crate) fn encode_trial_refine_only(
    block_width: u32,
    block_height: u32,
    pixel_stats: &PixelStats,
    enc_log_block: &mut LogAstcBlock,
    params: &CemEncodeParams,
    use_blue_contraction: bool,
    try_direct_encoding_flag: Option<&mut bool>,
) -> bool {
    let mut try_direct_encoding_flag = try_direct_encoding_flag;
    if let Some(f) = try_direct_encoding_flag.as_deref_mut() {
        *f = false;
    }
    let cem_index = enc_log_block.color_endpoint_modes[0] as u32;
    let dual_plane_flag = enc_log_block.dual_plane;
    let ccs_index = if dual_plane_flag {
        enc_log_block.color_component_selector as i32
    } else {
        -1
    };
    let endpoint_ise_range = enc_log_block.endpoint_ise_range as u32;
    let weight_ise_range = enc_log_block.weight_ise_range as u32;
    let grid_width = enc_log_block.grid_width as u32;
    let grid_height = enc_log_block.grid_height as u32;
    let total_grid_pixels = grid_width * grid_height;
    let wdq = dequant_tables().weight_ise_to_val(weight_ise_range);

    let mut dq0 = [0u8; MAXP];
    let mut up0 = [0u8; MAXP];
    for i in 0..total_grid_pixels {
        dq0[i as usize] = wdq[get_weight(enc_log_block, 0, i) as usize];
    }
    upsample_weight_grid(
        block_width,
        block_height,
        grid_width,
        grid_height,
        &dq0,
        &mut up0,
    );
    let mut dq1 = [0u8; MAXP];
    let mut up1 = [0u8; MAXP];
    if dual_plane_flag {
        for i in 0..total_grid_pixels {
            dq1[i as usize] = wdq[get_weight(enc_log_block, 1, i) as usize];
        }
        upsample_weight_grid(
            block_width,
            block_height,
            grid_width,
            grid_height,
            &dq1,
            &mut up1,
        );
    }

    let mut refine_params = *params;
    refine_params.forced_weight_vals0 = Some(&up0);
    if dual_plane_flag {
        refine_params.forced_weight_vals1 = Some(&up1);
    }

    let mut refined_endpoints = [0u8; MAX_CEM_ENDPOINT_VALS];
    let mut rw0 = [0u8; MAXP];
    let mut rw1 = [0u8; MAXP];
    let mut tdef = false;
    let refined_block_err = cem_encode_pixels(
        cem_index,
        ccs_index,
        pixel_stats,
        &refine_params,
        endpoint_ise_range,
        BISE_64_LEVELS,
        &mut refined_endpoints,
        &mut rw0,
        Some(&mut rw1),
        u64::MAX,
        use_blue_contraction,
        Some(&mut tdef),
    );
    if tdef && let Some(f) = try_direct_encoding_flag {
        *f = tdef;
    }
    if refined_block_err != u64::MAX {
        let cur_err = eval_error(
            block_width,
            block_height,
            enc_log_block,
            pixel_stats,
            params,
        );
        if refined_block_err < cur_err {
            let nev = get_num_cem_values(cem_index) as usize;
            enc_log_block.endpoints[..nev].copy_from_slice(&refined_endpoints[..nev]);
        }
    }
    true
}

// ---------------------------------------------------------------------------------------------
// Surrogate (float) encoding

/// `log_surrogate_astc_blk`. If not downsampling, the grid size may equal the block size
/// (and so may not be valid ASTC).
#[derive(Clone, Copy)]
pub(crate) struct LogSurrogateAstcBlk {
    pub grid_width: i32,
    pub grid_height: i32,
    /// Base+scale or direct variants only.
    pub cem_index: u32,
    /// -1 for single plane.
    pub ccs_index: i32,
    pub num_endpoint_levels: u32,
    pub num_weight_levels: u32,
    /// 1-3.
    pub num_parts: u32,
    /// ASTC seed index (10 bits) if `num_parts > 1`.
    pub seed_index: u32,
    /// [subset][l/h].
    pub endpoints: [[Vec4F; 2]; MAX_PARTITIONS as usize],
    pub scales: [f32; MAX_PARTITIONS as usize],
    pub weights0: [f32; MAXP],
    pub weights1: [f32; MAXP],
}

impl Default for LogSurrogateAstcBlk {
    fn default() -> Self {
        Self {
            grid_width: 0,
            grid_height: 0,
            cem_index: 0,
            ccs_index: 0,
            num_endpoint_levels: 0,
            num_weight_levels: 0,
            num_parts: 0,
            seed_index: 0,
            endpoints: [[[0.0; 4]; 2]; 4],
            scales: [0.0; 4],
            weights0: [0.0; MAXP],
            weights1: [0.0; MAXP],
        }
    }
}

/// `upsample_surrogate_weights` (all `block_width * block_height` texels in order).
fn upsample_surrogate_weights(
    weighted_samples: &[WeightedSample],
    src_weights: &[f32],
    dst_weights: &mut [f32],
    num_texels: usize,
    wx: u32,
    num_weight_levels: u32,
) {
    let weight_levels_minus_1 = (num_weight_levels - 1) as f32 * (1.0f32 / 16.0);
    let inv_weight_levels = 1.0f32 / (num_weight_levels - 1) as f32;
    let wx = wx as usize;
    for i in 0..num_texels {
        let s = &weighted_samples[i];
        let [[w00, w01], [w10, w11]] = s.weights.map(|r| r.map(u32::from));
        let (sx, sy) = (s.src_x as usize, s.src_y as usize);
        let mut total = 0.0f32;
        if w00 != 0 {
            total += src_weights[sx + sy * wx] * w00 as f32;
        }
        if w01 != 0 {
            total += src_weights[sx + 1 + sy * wx] * w01 as f32;
        }
        if w10 != 0 {
            total += src_weights[sx + (sy + 1) * wx] * w10 as f32;
        }
        if w11 != 0 {
            total += src_weights[sx + 1 + (sy + 1) * wx] * w11 as f32;
        }
        dst_weights[i] =
            fast_roundf_pos_int(total * weight_levels_minus_1) as f32 * inv_weight_levels;
    }
}

impl LogSurrogateAstcBlk {
    /// `clear`.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// `decode(block_width, block_height, pPixels, pPat)`.
    pub fn decode(
        &self,
        block_width: u32,
        block_height: u32,
        pixels: &mut [Vec4F],
        pat: Option<&PartitionPatternVec>,
    ) {
        let dual_plane = self.ccs_index >= 0;
        let total_block_pixels = (block_width * block_height) as usize;
        let total_grid_pixels = (self.grid_width * self.grid_height) as usize;
        let mut up0 = [0f32; MAXP];
        let mut up1 = [0f32; MAXP];
        let mut w0: &[f32] = &self.weights0;
        let mut w1: &[f32] = &self.weights1;
        if total_grid_pixels < total_block_pixels
            && let Some(gd) = find_astc_block_grid_data(
                block_width,
                block_height,
                self.grid_width as u32,
                self.grid_height as u32,
            )
        {
            upsample_surrogate_weights(
                &gd.upsample_weights,
                &self.weights0,
                &mut up0,
                total_block_pixels,
                self.grid_width as u32,
                self.num_weight_levels,
            );
            w0 = &up0;
            if dual_plane {
                upsample_surrogate_weights(
                    &gd.upsample_weights,
                    &self.weights1,
                    &mut up1,
                    total_block_pixels,
                    self.grid_width as u32,
                    self.num_weight_levels,
                );
                w1 = &up1;
            }
        }
        for i in 0..total_block_pixels {
            let mut part_index = 0usize;
            if self.num_parts > 1
                && let Some(pat) = pat
            {
                part_index = pat.parts[i] as usize & 3;
            }
            let l = &self.endpoints[part_index][0];
            let h = &self.endpoints[part_index][1];
            let dst = &mut pixels[i];
            for c in 0..4 {
                let w = if c as i32 == self.ccs_index {
                    w1[i]
                } else {
                    w0[i]
                };
                let one_minus_w = 1.0 - w;
                dst[c] = l[c] * one_minus_w + h[c] * w;
            }
        }
    }

    /// `decode(block_width, block_height, pPixels, pPat_data)`.
    pub fn decode_with_part_data(
        &self,
        block_width: u32,
        block_height: u32,
        pixels: &mut [Vec4F],
        part_data: &PartitionsData,
    ) {
        if self.num_parts == 1 {
            return self.decode(block_width, block_height, pixels, None);
        }
        let unique = part_data.part_seed_to_unique_index[(self.seed_index & 1023) as usize];
        let pat = part_data.partition_pats.get(unique.max(0) as usize);
        self.decode(block_width, block_height, pixels, pat)
    }
}

/// `downsample_float_weight_grid`.
fn downsample_float_weight_grid(
    matrix_weights: &[f32],
    bx: u32,
    by: u32,
    wx: u32,
    wy: u32,
    src_weights: &[f32],
    dst_weights: &mut [f32],
    num_weight_levels: u32,
) {
    let n = (bx * by) as usize;
    let weight_levels_minus_1 = (num_weight_levels - 1) as f32;
    let inv_weight_levels = 1.0f32 / (num_weight_levels - 1) as f32;
    let mut m = 0usize;
    for y in 0..wy {
        for x in 0..wx {
            let row = &matrix_weights[m..m + n];
            let mut total = 0.0f32;
            for i in 0..n {
                if row[i] != 0.0 {
                    total += row[i] * src_weights[i];
                }
            }
            dst_weights[(x + y * wx) as usize] =
                fast_roundf_pos_int(total * weight_levels_minus_1) as f32 * inv_weight_levels;
            m += n;
        }
    }
}

/// `decode_surrogate_and_compute_error`.
fn decode_surrogate_and_compute_error(
    block_width: u32,
    block_height: u32,
    pixel_stats: &PixelStats,
    log_block: &LogSurrogateAstcBlk,
    pat: Option<&PartitionPatternVec>,
    params: &CemEncodeParams,
) -> f32 {
    let mut dec = [[0f32; 4]; MAXP];
    log_block.decode(block_width, block_height, &mut dec, pat);
    let wr = params.comp_weights[0] as f32;
    let wg = params.comp_weights[1] as f32;
    let wb = params.comp_weights[2] as f32;
    let wa = params.comp_weights[3] as f32;
    let total_texels = (block_width * block_height) as usize;
    let mut total_err = 0.0f32;
    if wr == 1.0 && wg == 1.0 && wb == 1.0 && wa == 1.0 {
        for i in 0..total_texels {
            let s = &pixel_stats.pixels_f[i];
            let d = &dec[i];
            let (dr, dg, db, da) = (s[0] - d[0], s[1] - d[1], s[2] - d[2], s[3] - d[3]);
            total_err += (dr * dr) + (dg * dg) + (db * db) + (da * da);
        }
    } else {
        for i in 0..total_texels {
            let s = &pixel_stats.pixels_f[i];
            let d = &dec[i];
            let (dr, dg, db, da) = (s[0] - d[0], s[1] - d[1], s[2] - d[2], s[3] - d[3]);
            total_err += (wr * dr * dr) + (wg * dg * dg) + (wb * db * db) + (wa * da * da);
        }
    }
    total_err
}

/// `encode_surrogate_trial`: returns the WSSE.
pub(crate) fn encode_surrogate_trial(
    block_width: u32,
    block_height: u32,
    pixel_stats: &PixelStats,
    cem_index: u32,
    ccs_index: i32,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    grid_width: u32,
    grid_height: u32,
    log_block: &mut LogSurrogateAstcBlk,
    params: &CemEncodeParams,
    flags: u32,
) -> f32 {
    let is_downsampling = grid_width < block_width || grid_height < block_height;
    let dual_plane_flag = ccs_index >= 0;

    log_block.cem_index = cem_index;
    log_block.ccs_index = ccs_index;
    log_block.grid_width = grid_width as i32;
    log_block.grid_height = grid_height as i32;
    log_block.num_parts = 1;
    log_block.seed_index = 0;
    log_block.scales = [0.0; 4];
    log_block.num_endpoint_levels = get_ise_levels(endpoint_ise_range);
    log_block.num_weight_levels = get_ise_levels(weight_ise_range);

    if is_downsampling {
        let Some(gd) =
            find_astc_block_grid_data(block_width, block_height, grid_width, grid_height)
        else {
            return BIG_FLOAT_VAL;
        };
        let mut t0 = [0f32; MAXP];
        let mut t1 = [0f32; MAXP];
        let [l, h] = &mut log_block.endpoints[0];
        cem_surrogate_encode_pixels(
            cem_index,
            ccs_index,
            pixel_stats,
            params,
            endpoint_ise_range,
            weight_ise_range,
            l,
            h,
            &mut log_block.scales[0],
            &mut t0,
            &mut t1,
            flags,
        );
        downsample_float_weight_grid(
            &gd.downsample_matrix,
            block_width,
            block_height,
            grid_width,
            grid_height,
            &t0,
            &mut log_block.weights0,
            log_block.num_weight_levels,
        );
        if dual_plane_flag {
            downsample_float_weight_grid(
                &gd.downsample_matrix,
                block_width,
                block_height,
                grid_width,
                grid_height,
                &t1,
                &mut log_block.weights1,
                log_block.num_weight_levels,
            );
        }
        decode_surrogate_and_compute_error(
            block_width,
            block_height,
            pixel_stats,
            log_block,
            None,
            params,
        )
    } else {
        let lb = &mut *log_block;
        let [l, h] = &mut lb.endpoints[0];
        cem_surrogate_encode_pixels(
            cem_index,
            ccs_index,
            pixel_stats,
            params,
            endpoint_ise_range,
            weight_ise_range,
            l,
            h,
            &mut lb.scales[0],
            &mut lb.weights0,
            &mut lb.weights1,
            flags,
        )
    }
}

/// `encode_surrogate_trial_subsets`: returns the WSSE.
pub(crate) fn encode_surrogate_trial_subsets(
    block_width: u32,
    block_height: u32,
    pixel_stats: &PixelStats,
    cem_index: u32,
    num_subsets: u32,
    pat_seed_index: u32,
    pat: &PartitionPatternVec,
    endpoint_ise_range: u32,
    weight_ise_range: u32,
    grid_width: u32,
    grid_height: u32,
    log_block: &mut LogSurrogateAstcBlk,
    params: &CemEncodeParams,
    flags: u32,
) -> f32 {
    let is_downsampling = grid_width < block_width || grid_height < block_height;
    let num_weight_levels = get_ise_levels(weight_ise_range);
    let num_endpoint_levels = get_ise_levels(endpoint_ise_range);
    let ns = num_subsets as usize;

    let mut part_pixels = [[[0u8; 4]; MAXP]; 4];
    let mut num_part_pixels = [0u32; 4];
    split_subset_pixels(
        block_width,
        block_height,
        &pixel_stats.pixels,
        pat,
        &mut part_pixels,
        &mut num_part_pixels,
    );

    let mut part_stats = new_pixel_stats_array();
    let mut part_weights = [[0f32; MAXP]; 4];
    let mut temp_block_weights = [0f32; MAXP];

    let mut total_subset_err = 0.0f64;
    for pi in 0..ns {
        part_stats[pi].init(num_part_pixels[pi], &part_pixels[pi]);
        let [l, h] = &mut log_block.endpoints[pi];
        let subset_err = cem_surrogate_encode_pixels(
            cem_index,
            -1,
            &part_stats[pi],
            params,
            endpoint_ise_range,
            weight_ise_range,
            l,
            h,
            &mut log_block.scales[pi],
            &mut part_weights[pi],
            &mut temp_block_weights,
            flags,
        );
        total_subset_err += subset_err as f64;
    }

    {
        let dst: &mut [f32] = if is_downsampling {
            &mut temp_block_weights
        } else {
            &mut log_block.weights0
        };
        num_part_pixels = [0; 4];
        for i in 0..(block_width * block_height) as usize {
            let pi = pat.parts[i] as usize & 3;
            dst[i] = part_weights[pi][num_part_pixels[pi] as usize];
            num_part_pixels[pi] += 1;
        }
    }

    log_block.cem_index = cem_index;
    log_block.ccs_index = -1;
    log_block.num_endpoint_levels = num_endpoint_levels;
    log_block.num_weight_levels = num_weight_levels;
    log_block.grid_width = grid_width as i32;
    log_block.grid_height = grid_height as i32;
    log_block.num_parts = num_subsets;
    log_block.seed_index = pat_seed_index;

    if is_downsampling {
        let Some(gd) =
            find_astc_block_grid_data(block_width, block_height, grid_width, grid_height)
        else {
            return BIG_FLOAT_VAL;
        };
        downsample_float_weight_grid(
            &gd.downsample_matrix,
            block_width,
            block_height,
            grid_width,
            grid_height,
            &temp_block_weights,
            &mut log_block.weights0,
            num_weight_levels,
        );
        total_subset_err = decode_surrogate_and_compute_error(
            block_width,
            block_height,
            pixel_stats,
            log_block,
            Some(pat),
            params,
        ) as f64;
    }
    total_subset_err as f32
}

// ---------------------------------------------------------------------------------------------
// Partition estimation

/// `estimate_slam_to_line_sse_3D`.
#[inline]
fn estimate_slam_to_line_sse_3d(cov: &[f32; 6], mut xr: f32, mut yr: f32, mut zr: f32) -> f32 {
    let total_var = cov[0] + cov[3] + cov[5];
    let mut l = (xr * xr + yr * yr + zr * zr).sqrt();
    if l < SMALL_FLOAT_VAL {
        xr = 0.577350269;
        yr = 0.577350269;
        zr = 0.577350269;
    } else {
        l = 1.0 / l;
        xr *= l;
        yr *= l;
        zr *= l;
    }
    let xr2 = cov[0] * xr + cov[1] * yr + cov[2] * zr;
    let xg2 = cov[1] * xr + cov[3] * yr + cov[4] * zr;
    let xb2 = cov[2] * xr + cov[4] * yr + cov[5] * zr;
    let principal_axis_var = xr2 * xr + xg2 * yr + xb2 * zr;
    max_f(0.0, total_var - principal_axis_var)
}

/// `estimate_slam_to_line_sse_4D`.
#[inline]
fn estimate_slam_to_line_sse_4d(
    cov: &[f32; 10],
    mut xr: f32,
    mut yr: f32,
    mut zr: f32,
    mut wr: f32,
) -> f32 {
    let total_var = cov[0] + cov[4] + cov[7] + cov[9];
    let mut l = (xr * xr + yr * yr + zr * zr + wr * wr).sqrt();
    if l < SMALL_FLOAT_VAL {
        xr = 0.5;
        yr = 0.5;
        zr = 0.5;
        wr = 0.5;
    } else {
        l = 1.0 / l;
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
    max_f(0.0, total_var - principal_axis_var)
}

/// `basisu::maximum(a, b)` for floats (`(a < b) ? b : a`).
#[inline]
fn max_f(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

/// `basisu::minimum(a, b)` for floats (`(a < b) ? a : b`).
#[inline]
pub(crate) fn min_f(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// Per-subset integer means (rounded) for the direct CEMs, 0 for base+scale.
fn subset_means<const N: usize>(
    pixels: &[ColorRgba],
    n: usize,
    base_scale_flag: bool,
    num_subsets: usize,
    pat: &PartitionPatternVec,
    part_pixels: &mut [[ColorRgba; MAXP]; 4],
    num_part_pixels: &mut [u32; 4],
) -> [[i32; N]; 4] {
    let mut means = [[0i32; N]; 4];
    *num_part_pixels = [0; 4];
    for i in 0..n {
        let px = pixels[i];
        let pi = pat.parts[i] as usize & 3;
        part_pixels[pi][num_part_pixels[pi] as usize] = px;
        if !base_scale_flag {
            for c in 0..N {
                means[pi][c] += px[c] as i32;
            }
        }
        num_part_pixels[pi] += 1;
    }
    if !base_scale_flag {
        for i in 0..num_subsets {
            let n = num_part_pixels[i] as i32;
            if n == 0 {
                continue;
            }
            let r = n >> 1;
            for c in 0..N {
                means[i][c] = (means[i][c] + r) / n;
            }
        }
    }
    means
}

/// `estimate_partition_rgb_sse`.
fn estimate_partition_rgb_sse(
    pixel_stats: &PixelStats,
    base_scale_flag: bool,
    block_width: u32,
    block_height: u32,
    num_subsets: u32,
    pat: &PartitionPatternVec,
) -> f32 {
    let ns = num_subsets as usize;
    let mut part_pixels = [[[0u8; 4]; MAXP]; 4];
    let mut num_part_pixels = [0u32; 4];
    let means = subset_means::<3>(
        &pixel_stats.pixels,
        (block_width * block_height) as usize,
        base_scale_flag,
        ns,
        pat,
        &mut part_pixels,
        &mut num_part_pixels,
    );

    let mut icov = [[0i32; 6]; 4];
    for p in 0..ns {
        let [mr, mg, mb] = means[p];
        let c = &mut icov[p];
        for i in 0..num_part_pixels[p] as usize {
            let px = &part_pixels[p][i];
            let r = px[0] as i32 - mr;
            let g = px[1] as i32 - mg;
            let b = px[2] as i32 - mb;
            c[0] += r * r;
            c[1] += r * g;
            c[2] += r * b;
            c[3] += g * g;
            c[4] += g * b;
            c[5] += b * b;
        }
    }

    let mut est = 0.0f32;
    for p in 0..ns {
        let block_max_var = icov[p][0].max(icov[p][3]).max(icov[p][5]);
        let cov: [f32; 6] = std::array::from_fn(|i| icov[p][i] as f32);
        let sc = if block_max_var != 0 {
            1.0 / block_max_var as f32
        } else {
            0.0
        };
        let (wx, wy, wz) = (sc * cov[0], sc * cov[3], sc * cov[5]);
        let alt_xr = cov[0] * wx + cov[1] * wy + cov[2] * wz;
        let alt_xg = cov[1] * wx + cov[3] * wy + cov[4] * wz;
        let alt_xb = cov[2] * wx + cov[4] * wy + cov[5] * wz;
        est += estimate_slam_to_line_sse_3d(&cov, alt_xr, alt_xg, alt_xb);
    }
    est
}

/// `estimate_partition_rgba_sse`.
fn estimate_partition_rgba_sse(
    pixel_stats: &PixelStats,
    base_scale_flag: bool,
    block_width: u32,
    block_height: u32,
    num_subsets: u32,
    pat: &PartitionPatternVec,
) -> f32 {
    let ns = num_subsets as usize;
    let mut part_pixels = [[[0u8; 4]; MAXP]; 4];
    let mut num_part_pixels = [0u32; 4];
    let means = subset_means::<4>(
        &pixel_stats.pixels,
        (block_width * block_height) as usize,
        base_scale_flag,
        ns,
        pat,
        &mut part_pixels,
        &mut num_part_pixels,
    );

    let mut icov = [[0i32; 10]; 4];
    for p in 0..ns {
        let [mr, mg, mb, ma] = means[p];
        let c = &mut icov[p];
        for i in 0..num_part_pixels[p] as usize {
            let px = &part_pixels[p][i];
            let r = px[0] as i32 - mr;
            let g = px[1] as i32 - mg;
            let b = px[2] as i32 - mb;
            let a = px[3] as i32 - ma;
            c[0] += r * r;
            c[1] += r * g;
            c[2] += r * b;
            c[3] += r * a;
            c[4] += g * g;
            c[5] += g * b;
            c[6] += g * a;
            c[7] += b * b;
            c[8] += b * a;
            c[9] += a * a;
        }
    }

    let mut est = 0.0f32;
    for s in 0..ns {
        let m4 = icov[s][0].max(icov[s][4]).max(icov[s][7]).max(icov[s][9]);
        let c: [f32; 10] = std::array::from_fn(|i| icov[s][i] as f32);
        let sc4 = if m4 != 0 { 1.0 / m4 as f32 } else { 0.0 };
        let (wx, wy, wz, wa) = (sc4 * c[0], sc4 * c[4], sc4 * c[7], sc4 * c[9]);
        let x0 = c[0] * wx + c[1] * wy + c[2] * wz + c[3] * wa;
        let y0 = c[1] * wx + c[4] * wy + c[5] * wz + c[6] * wa;
        let z0 = c[2] * wx + c[5] * wy + c[7] * wz + c[8] * wa;
        let w0 = c[3] * wx + c[6] * wy + c[8] * wz + c[9] * wa;
        let x1 = c[0] * x0 + c[1] * y0 + c[2] * z0 + c[3] * w0;
        let y1 = c[1] * x0 + c[4] * y0 + c[5] * z0 + c[6] * w0;
        let z1 = c[2] * x0 + c[5] * y0 + c[7] * z0 + c[8] * w0;
        let w1 = c[3] * x0 + c[6] * y0 + c[8] * z0 + c[9] * w0;
        est += estimate_slam_to_line_sse_4d(&c, x1, y1, z1, w1);
    }
    est
}

/// `estimate_partition_sse`.
pub(crate) fn estimate_partition_sse(
    cem_index: u32,
    pixel_stats: &PixelStats,
    block_width: u32,
    block_height: u32,
    num_subsets: u32,
    pat: &PartitionPatternVec,
) -> f32 {
    use crate::astc::helpers::{
        CEM_LDR_RGB_BASE_SCALE, CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A, CEM_LDR_RGB_DIRECT,
        CEM_LDR_RGBA_DIRECT,
    };
    match cem_index {
        CEM_LDR_RGB_DIRECT => estimate_partition_rgb_sse(
            pixel_stats,
            false,
            block_width,
            block_height,
            num_subsets,
            pat,
        ),
        CEM_LDR_RGB_BASE_SCALE => estimate_partition_rgb_sse(
            pixel_stats,
            true,
            block_width,
            block_height,
            num_subsets,
            pat,
        ),
        CEM_LDR_RGBA_DIRECT => estimate_partition_rgba_sse(
            pixel_stats,
            false,
            block_width,
            block_height,
            num_subsets,
            pat,
        ),
        CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => estimate_partition_rgba_sse(
            pixel_stats,
            true,
            block_width,
            block_height,
            num_subsets,
            pat,
        ),
        _ => 0.0,
    }
}

/// `BASISU_LSH_FILTERING`.
const LSH_FILTERING: bool = true;

/// `estimate_partition2`: fills `best_parts` (unique pattern indices) and updates
/// `num_best_parts` with the actual number of results (LSH path; the VP tree path is disabled
/// in the C++ build).
pub(crate) fn estimate_partition2(
    block_width: u32,
    block_height: u32,
    pixels: &PixelStats,
    best_parts: &mut [i32],
    num_best_parts: &mut i32,
    part_data: &PartitionsData,
    brute_force_flag: bool,
) -> bool {
    let num_block_pixels = (block_width * block_height) as usize;
    let mut desired = PartitionPatternVec::new(block_width, block_height, None);
    let mut desired_alt = PartitionPatternVec::new(block_width, block_height, None);
    for i in 0..num_block_pixels {
        let proj = dot4(
            &sub4(&pixels.pixels_f[i], &pixels.mean_f),
            &pixels.mean_rel_axis4,
        );
        desired.parts[i] = (proj < 0.0) as u8;
        desired_alt.parts[i] = 1 - desired.parts[i];
    }

    let total_unique = part_data.total_unique_patterns as usize;
    if brute_force_flag {
        let mut sim = Vec::with_capacity(total_unique);
        for part_index in 0..total_unique {
            let pat_vec = &part_data.partition_pats[part_index];
            let (mut non_inv, mut inv) = (0i32, 0i32);
            for i in 0..num_block_pixels {
                let part = pat_vec.parts[i];
                if part == desired.parts[i] {
                    non_inv += 1;
                }
                if (part ^ 1) == desired.parts[i] {
                    inv += 1;
                }
            }
            let total_sim = non_inv.max(inv) as u32;
            sim.push((total_sim << 16) | part_index as u32);
        }
        sim.sort_unstable();
        for i in 0..(*num_best_parts).max(0) as usize {
            best_parts[i] = (sim[(total_unique - 1) - i] & 0xFFFF) as i32;
        }
    } else {
        let mut results = [0u32; 1024];
        let total_results = part_data
            .part_lhs_map
            .find(&desired, &mut results, 1024, LSH_FILTERING) as usize;
        if total_results == 0 {
            *num_best_parts = 0;
            return false;
        }
        let mut sim = Vec::with_capacity(total_results);
        for &part_index in &results[..total_results] {
            let pat_vec = &part_data.partition_pats[part_index as usize];
            let a = desired.get_squared_distance_2subsets(pat_vec);
            let b = desired_alt.get_squared_distance_2subsets(pat_vec);
            let dist2 = a.min(b) as u32;
            sim.push((dist2 << 16) | part_index);
        }
        sim.sort_unstable();
        *num_best_parts = (*num_best_parts).min(total_results as i32);
        for i in 0..*num_best_parts as usize {
            best_parts[i] = (sim[i] & 0xFFFF) as i32;
        }
    }
    true
}

/// `estimate_partition3` (k-means into 3 clusters, then LSH lookup).
pub(crate) fn estimate_partition3(
    block_width: u32,
    block_height: u32,
    pixels: &PixelStats,
    best_parts: &mut [i32],
    num_best_parts: &mut i32,
    part_data: &PartitionsData,
    brute_force_flag: bool,
) -> bool {
    const NUM_SUBSETS: usize = 3;
    let num_block_pixels = (block_width * block_height) as usize;
    let training_vecs = &pixels.pixels_f;

    let mut brightest_inten = -BIG_FLOAT_VAL;
    let mut darkest_inten = BIG_FLOAT_VAL;
    let mut centroids = [[0f32; 4]; NUM_SUBSETS];

    for i in 0..num_block_pixels {
        let v = &training_vecs[i];
        let inten = dot4(&sub4(v, &pixels.mean_f), &pixels.mean_rel_axis4);
        if inten < darkest_inten {
            darkest_inten = inten;
            centroids[0] = *v;
        }
        if inten > brightest_inten {
            brightest_inten = inten;
            centroids[1] = *v;
        }
    }
    if centroids[0] == centroids[1] {
        return false;
    }

    let mut furthest_dist2 = 0.0f32;
    for i in 0..num_block_pixels {
        let v = &training_vecs[i];
        let dist_a = squared_distance4(v, &centroids[0]);
        if dist_a == 0.0 {
            continue;
        }
        let dist_b = squared_distance4(v, &centroids[1]);
        if dist_b == 0.0 {
            continue;
        }
        let dist2 = dist_a + dist_b;
        if dist2 > furthest_dist2 {
            furthest_dist2 = dist2;
            centroids[2] = *v;
        }
    }
    if centroids[0] == centroids[2] || centroids[1] == centroids[2] {
        return false;
    }

    let mut cluster_pixels = [[0u32; MAXP]; NUM_SUBSETS];
    let mut num_cluster_pixels = [0u32; NUM_SUBSETS];
    const NUM_ITERS: u32 = 4;
    for s in 0..NUM_ITERS {
        num_cluster_pixels = [0; NUM_SUBSETS];
        let mut new_means = [[0f32; 4]; NUM_SUBSETS];
        for i in 0..num_block_pixels {
            let d = [
                squared_distance4(&training_vecs[i], &centroids[0]),
                squared_distance4(&training_vecs[i], &centroids[1]),
                squared_distance4(&training_vecs[i], &centroids[2]),
            ];
            let mut min_d = d[0];
            let mut min_idx = 0usize;
            for j in 1..NUM_SUBSETS {
                if d[j] < min_d {
                    min_d = d[j];
                    min_idx = j;
                }
            }
            cluster_pixels[min_idx][num_cluster_pixels[min_idx] as usize] = i as u32;
            for c in 0..4 {
                new_means[min_idx][c] += training_vecs[i][c];
            }
            num_cluster_pixels[min_idx] += 1;
        }
        for j in 0..NUM_SUBSETS {
            if num_cluster_pixels[j] == 0 {
                return false;
            }
            if s != NUM_ITERS - 1 {
                let n = num_cluster_pixels[j] as f32;
                centroids[j] = std::array::from_fn(|c| new_means[j][c] / n);
            }
        }
    }

    let mut desired = PartitionPatternVec::new(block_width, block_height, None);
    for p in 0..NUM_SUBSETS {
        for i in 0..num_cluster_pixels[p] as usize {
            desired.parts[cluster_pixels[p][i] as usize] = p as u8;
        }
    }

    let desired_parts: [PartitionPatternVec; NUM_PART3_MAPPINGS] =
        std::array::from_fn(|j| desired.get_permuted3(j as u32));
    let lowest_dist = |pat: &PartitionPatternVec| {
        let mut lowest = u32::MAX;
        for dp in &desired_parts {
            let dist = pat.get_squared_distance(dp) as u32;
            if dist < lowest {
                lowest = dist;
            }
        }
        lowest
    };

    if brute_force_flag {
        let total_unique = part_data.total_unique_patterns as usize;
        let mut sim = Vec::with_capacity(total_unique);
        for part_index in 0..total_unique {
            let d = lowest_dist(&part_data.partition_pats[part_index]);
            sim.push(d.wrapping_shl(16) | part_index as u32);
        }
        sim.sort_unstable();
        for i in 0..(*num_best_parts).max(0) as usize {
            best_parts[i] = (sim[i] & 0xFFFF) as i32;
        }
    } else {
        let mut results = [0u32; 1024];
        let total_results = part_data
            .part_lhs_map
            .find(&desired, &mut results, 1024, LSH_FILTERING) as usize;
        if total_results == 0 {
            *num_best_parts = 0;
            return false;
        }
        let mut sim = Vec::with_capacity(total_results);
        for &part_index in &results[..total_results] {
            let d = lowest_dist(&part_data.partition_pats[part_index as usize]);
            sim.push(d.wrapping_shl(16) | part_index);
        }
        sim.sort_unstable();
        *num_best_parts = (*num_best_parts).min(total_results as i32);
        for i in 0..*num_best_parts as usize {
            best_parts[i] = (sim[i] & 0xFFFF) as i32;
        }
    }
    true
}

// ---------------------------------------------------------------------------------------------
// Sobel / DCT energy

/// `g_sobel_x` ([y][x], row major).
pub(crate) const SOBEL_X: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
/// `g_sobel_y` ([y][x], row major).
pub(crate) const SOBEL_Y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

/// `compute_sobel`.
pub(crate) fn compute_sobel(orig: &Image, dest: &mut Image, matrix_3x3: &[f32; 9]) {
    let width = orig.get_width();
    let height = orig.get_height();
    dest.resize(width, height);
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut d = [128.0f32; 4];
            for my in -1..=1i32 {
                for mx in -1..=1i32 {
                    let w = matrix_3x3[((my + 1) * 3 + (mx + 1)) as usize];
                    if w == 0.0 {
                        continue;
                    }
                    let s = orig.get_clamped(x + mx, y + my);
                    for c in 0..4 {
                        d[c] += w * s[c] as f32;
                    }
                }
            }
            *dest.get_mut(x as u32, y as u32) = d.map(|v| fast_roundf_int(v).clamp(0, 255) as u8);
        }
    }
}

/// `compute_ac_energy_from_dct`: squares the coefficients in place (DC zeroed), returns the
/// total energy excluding DC.
pub(crate) fn compute_ac_energy_from_dct(
    block_width: u32,
    block_height: u32,
    dct: &mut [f32],
) -> f32 {
    let n = (block_width * block_height) as usize;
    let mut total = 0.0f32;
    for i in 1..n {
        let v = dct[i] * dct[i];
        dct[i] = v;
        total += v;
    }
    dct[0] = 0.0;
    total
}

/// `compute_preserved_dct_energy`.
pub(crate) fn compute_preserved_dct_energy(
    block_width: u32,
    block_height: u32,
    energy: &[f32],
    grid_w: u32,
    grid_h: u32,
) -> f32 {
    let mut tot = 0.0f32;
    for y in 0..block_height {
        for x in 0..block_width {
            if x < grid_w && y < grid_h {
                tot += energy[(x + y * block_width) as usize];
            }
        }
    }
    tot
}

/// `compute_lost_dct_energy_orig`.
pub(crate) fn compute_lost_dct_energy_orig(
    block_width: u32,
    block_height: u32,
    energy: &[f32],
    grid_w: u32,
    grid_h: u32,
) -> f32 {
    let mut tot = 0.0f32;
    for y in 0..block_height {
        for x in 0..block_width {
            if x < grid_w && y < grid_h {
                continue;
            }
            tot += energy[(x + y * block_width) as usize];
        }
    }
    tot
}

/// `compute_lost_dct_energy`.
pub(crate) fn compute_lost_dct_energy(
    block_width: u32,
    block_height: u32,
    energy: &[f32],
    grid_w: u32,
    grid_h: u32,
) -> f32 {
    let (mut t0, mut t1, mut t2, mut t3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let bw = block_width as usize;
    for y in 0..block_height {
        let row = &energy[y as usize * bw..y as usize * bw + bw];
        let mut x = if y < grid_h { grid_w as usize } else { 0 };
        while x + 4 <= bw {
            t0 += row[x];
            t1 += row[x + 1];
            t2 += row[x + 2];
            t3 += row[x + 3];
            x += 4;
        }
        while x < bw {
            t0 += row[x];
            x += 1;
        }
    }
    t0 + t1 + t2 + t3
}

/// `prepare_dct_energy_prefix_table`.
pub(crate) fn prepare_dct_energy_prefix_table(
    block_width: u32,
    block_height: u32,
    energy: &[f32],
    prefix: &mut [f32],
) {
    let bw = block_width as usize;
    for y in 0..block_height as usize {
        let mut row_sum = 0.0f32;
        for x in 0..bw {
            row_sum += energy[x + y * bw];
            let above = if y > 0 { prefix[x + (y - 1) * bw] } else { 0.0 };
            prefix[x + y * bw] = row_sum + above;
        }
    }
}

/// `query_dct_energy_prefix_sum`.
#[inline]
pub(crate) fn query_dct_energy_prefix_sum(
    block_width: u32,
    prefix: &[f32],
    grid_w: u32,
    grid_h: u32,
) -> f32 {
    prefix[((grid_w - 1) + (grid_h - 1) * block_width) as usize]
}

/// `compute_lost_dct_energy_prefix_sum`.
#[inline]
pub(crate) fn compute_lost_dct_energy_prefix_sum(
    block_width: u32,
    prefix: &[f32],
    grid_w: u32,
    grid_h: u32,
    total_ac_energy: f32,
) -> f32 {
    let kept = query_dct_energy_prefix_sum(block_width, prefix, grid_w, grid_h);
    max_f(total_ac_energy - kept, 0.0)
}
