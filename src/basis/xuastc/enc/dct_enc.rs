// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Encoder side of the XUASTC LDR weight grid DCT: `basist::astc_ldr_t::dct2f` (forward
//! transform), the encode parts of `grid_weight_dct` and `code_block_weights`
//! (`basisu_astc_ldr_encode.cpp`). Decoding (dequantization + IDCT) reuses the decoder's
//! [`crate::basis::xuastc::dct`].
#![allow(clippy::needless_range_loop)]

use super::{DctCoeff, DctSyms};
use crate::astc::helpers::{self as ah, LogAstcBlock, MAX_GRID_WEIGHTS};
use crate::basis::xuastc::dct as xdct;

/// `dct2f::cMaxSize`.
const DCT_MAX_SIZE: u32 = 12;

/// `basist::astc_ldr_t::dct2f`: separable float DCT-II of a `rows` x `cols` block.
#[derive(Clone, Debug, Default)]
pub(crate) struct Dct2f {
    rows: u32,
    cols: u32,
    /// `[u * rows + x]`.
    c_col: Vec<f32>,
    /// `[v * cols + y]`.
    c_row: Vec<f32>,
    a_col: Vec<f32>,
    a_row: Vec<f32>,
}

impl Dct2f {
    /// `init(rows, cols)` (call with grid height / grid width). `None` if out of range.
    pub fn new(rows: u32, cols: u32) -> Option<Self> {
        if !(2..=DCT_MAX_SIZE).contains(&rows) || !(2..=DCT_MAX_SIZE).contains(&cols) {
            return None;
        }
        let (m, n) = (rows as usize, cols as usize);
        let mut d = Self {
            rows,
            cols,
            c_col: vec![0.0; m * m],
            c_row: vec![0.0; n * n],
            a_col: vec![0.0; m],
            a_row: vec![0.0; n],
        };
        let pi = std::f32::consts::PI;
        let inv_m = 1.0f32 / rows as f32;
        d.a_col[0] = inv_m.sqrt();
        for u in 1..m {
            d.a_col[u] = (2.0 * inv_m).sqrt();
        }
        let inv_n = 1.0f32 / cols as f32;
        d.a_row[0] = inv_n.sqrt();
        for v in 1..n {
            d.a_row[v] = (2.0 * inv_n).sqrt();
        }
        for u in 0..m {
            for x in 0..m {
                let angle = (pi * ((2 * x + 1) * u) as f32) / (2.0 * rows as f32);
                d.c_col[u * m + x] = angle.cos();
            }
        }
        for v in 0..n {
            for y in 0..n {
                let angle = (pi * ((2 * y + 1) * v) as f32) / (2.0 * cols as f32);
                d.c_row[v * n + y] = angle.cos();
            }
        }
        Some(d)
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// `forward(pSrc, src_stride, pDst, dst_stride, pWork)`, in place on `buf` (source and
    /// destination share the same buffer/stride, as all encoder callers do). `work` must
    /// hold `rows * cols` values.
    pub fn forward_in_place(&self, buf: &mut [f32], stride: usize, work: &mut [f32]) {
        let (m, n) = (self.rows as usize, self.cols as usize);
        // horizontal
        for x in 0..m {
            let row_in = &buf[x * stride..x * stride + n];
            for v in 0..n {
                let cv = &self.c_row[v * n..v * n + n];
                let mut s = 0.0f32;
                for y in 0..n {
                    s += row_in[y] * cv[y];
                }
                work[x * n + v] = s * self.a_row[v];
            }
        }
        // vertical
        for v in 0..n {
            for u in 0..m {
                let cu = &self.c_col[u * m..u * m + m];
                let mut s = 0.0f32;
                for x in 0..m {
                    s += work[x * n + v] * cu[x];
                }
                buf[u * stride + v] = s * self.a_col[u];
            }
        }
    }

    /// `forward(pSrc, pDst, work)` (contiguous `rows` x `cols`).
    pub fn forward(&self, src: &[f32], dst: &mut [f32]) {
        let n = (self.rows * self.cols) as usize;
        dst[..n].copy_from_slice(&src[..n]);
        let mut work = [0.0f32; 144];
        self.forward_in_place(dst, self.cols as usize, &mut work);
    }
}

/// `grid_dim_value`.
#[derive(Clone, Debug)]
struct GridDimValue {
    zigzag: [usize; 64],
    dct: Dct2f,
}

/// `basist::astc_ldr_t::grid_weight_dct` (encoder side).
#[derive(Clone, Debug)]
pub(crate) struct GridWeightDct {
    pub block_width: u32,
    pub block_height: u32,
    /// `[(gh - 2) * 11 + (gw - 2)]`.
    grid_dims: Vec<Option<GridDimValue>>,
}

impl GridWeightDct {
    /// `init(block_width, block_height)`.
    pub fn new(block_width: u32, block_height: u32) -> Self {
        let mut grid_dims = vec![None; 11 * 11];
        for gh in 2..=block_height.min(12) {
            for gw in 2..=block_width.min(12) {
                if gw * gh > MAX_GRID_WEIGHTS {
                    continue;
                }
                let Some(dct) = Dct2f::new(gh, gw) else {
                    continue;
                };
                let mut zigzag = [0usize; 64];
                xdct::zigzag(gw as usize, gh as usize, &mut zigzag);
                grid_dims[((gh - 2) * 11 + (gw - 2)) as usize] = Some(GridDimValue { zigzag, dct });
            }
        }
        Self {
            block_width,
            block_height,
            grid_dims,
        }
    }

    fn find(&self, gw: u32, gh: u32) -> Option<&GridDimValue> {
        if !(2..=12).contains(&gw) || !(2..=12).contains(&gh) {
            return None;
        }
        self.grid_dims
            .get(((gh - 2) * 11 + (gw - 2)) as usize)
            .and_then(|v| v.as_ref())
    }

    /// `get_num_weight_dc_levels`.
    pub fn get_num_weight_dc_levels(weight_ise_range: u32) -> u32 {
        xdct::num_weight_dc_levels(weight_ise_range)
    }

    /// `compute_quant_table` (index 0 = DC = 1).
    pub fn compute_quant_table(
        &self,
        q: f32,
        grid_width: u32,
        grid_height: u32,
        level_scale: f32,
        tab: &mut [i32; 64],
    ) {
        tab[0] = 1;
        let state = xdct::QuantState {
            q,
            sx: 8.0 / self.block_width as f32,
            sy: 8.0 / self.block_height as f32,
            level_scale,
        };
        for y in 0..grid_height {
            for x in (if y != 0 { 0 } else { 1 })..grid_width {
                // `sample_jpeg_quant` == `sample_quant_table` for non-negative coordinates.
                tab[(x + y * grid_width) as usize] = xdct::sample_quant_table(&state, x, y);
            }
        }
    }

    /// `quantize_deadzone`.
    #[inline]
    pub fn quantize_deadzone(d: f32, l: i32, alpha: f32, x: u32, y: u32) -> i32 {
        if (x == 1 && y == 0) || (x == 0 && y == 1) {
            return (d / l as f32).round() as i32;
        }
        if l <= 0 {
            return 0;
        }
        let s = d.abs();
        let tau = alpha * l as f32;
        if s <= tau {
            return 0;
        }
        let qf = (s - tau) / l as f32;
        let q = (qf + 0.5).floor() as i32;
        if d < 0.0 { -q } else { q }
    }

    /// `decode_block_weights(q, plane_index, log_blk, nullptr, nullptr, work, &syms)`.
    pub fn decode_block_weights(
        &self,
        q: f32,
        plane_index: u32,
        log_blk: &mut LogAstcBlock,
        syms: &DctSyms,
    ) -> bool {
        if self
            .find(
                u32::from(log_blk.grid_width),
                u32::from(log_blk.grid_height),
            )
            .is_none()
        {
            return false;
        }
        xdct::decode_block_weights_coeffs(
            self.block_width,
            self.block_height,
            q,
            plane_index,
            log_blk,
            syms.dc_sym,
            syms.coeffs.iter().map(|c| (c.num_zeros, c.coeff)),
        )
    }
}

/// `code_block_weights(gw_dct, q, plane_index, log_blk, syms, dct_work)`: forward DCT +
/// deadzone quantization of one weight plane into run/coefficient symbols.
pub(crate) fn code_block_weights(
    gw_dct: &GridWeightDct,
    q: f32,
    plane_index: u32,
    log_blk: &LogAstcBlock,
    syms: &mut DctSyms,
) -> bool {
    syms.clear();

    let (grid_width, grid_height) = (
        u32::from(log_blk.grid_width),
        u32::from(log_blk.grid_height),
    );
    let total_grid_samples = (grid_width * grid_height) as usize;
    let num_planes = if log_blk.dual_plane { 2 } else { 1 };
    let wrange = u32::from(log_blk.weight_ise_range);

    let Some(grid_dim_vals) = gw_dct.find(grid_width, grid_height) else {
        return false;
    };
    let ise_to_val = &ah::dequant_tables().weight_tab(wrange).ise_to_val;

    let mut orig_weights = [0.0f32; 144];
    let mut weight_sum = 0.0f32;
    for i in 0..total_grid_samples {
        let w = log_blk.weights[i * num_planes + plane_index as usize];
        let v = f32::from(ise_to_val.get(w as usize).copied().unwrap_or(0));
        orig_weights[i] = v;
        weight_sum += v;
    }

    let scale = xdct::coding_scale(wrange);
    let mut scaled_mean_weight = (scale * (weight_sum / total_grid_samples as f32)).round();
    scaled_mean_weight = scaled_mean_weight.clamp(0.0, 64.0 * scale);
    let mean_weight = scaled_mean_weight / scale;

    for w in orig_weights.iter_mut().take(total_grid_samples) {
        *w -= mean_weight;
    }

    let span_len = xdct::max_span_len(log_blk, plane_index);

    let mut dct_weights = [0.0f32; 144];
    grid_dim_vals.dct.forward(&orig_weights, &mut dct_weights);

    let level_scale = xdct::compute_level_scale(q, span_len, wrange);

    let mut quant_tab = [0i32; 64];
    gw_dct.compute_quant_table(q, grid_width, grid_height, level_scale, &mut quant_tab);

    syms.dc_sym = scaled_mean_weight as i32 as u32;
    syms.num_dc_levels = (64.0 * scale) as u32 + 1;

    let mut dct_coeffs = [0i32; 64];
    for y in 0..grid_height {
        for x in 0..grid_width {
            if x == 0 && y == 0 {
                dct_coeffs[0] = 0;
                continue;
            }
            let i = (x + y * grid_width) as usize;
            dct_coeffs[i] = GridWeightDct::quantize_deadzone(
                dct_weights[i],
                quant_tab[i],
                xdct::DEADZONE_ALPHA,
                x,
                y,
            );
        }
    }

    let mut total_zeros = 0u32;
    for i in 0..total_grid_samples {
        let dct_idx = grid_dim_vals.zigzag[i];
        if dct_idx == 0 {
            continue;
        }
        let coeff = dct_coeffs[dct_idx];
        if coeff == 0 {
            total_zeros += 1;
            continue;
        }
        syms.coeffs.push(DctCoeff {
            num_zeros: total_zeros as u16,
            coeff: coeff as i16,
        });
        syms.max_coeff_mag = syms.max_coeff_mag.max(coeff.unsigned_abs());
        syms.max_zigzag_index = syms.max_zigzag_index.max(i as u32);
        total_zeros = 0;
    }
    if total_zeros != 0 {
        syms.coeffs.push(DctCoeff {
            num_zeros: total_zeros as u16,
            coeff: i16::MAX,
        });
    }
    true
}

/// `apply_weight_grid_dct`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_weight_grid_dct(
    block_width: u32,
    block_height: u32,
    grid_coder: &GridWeightDct,
    out_block: &mut super::EncodeBlockOutput,
    pixel_stats: &super::common::PixelStats,
    recalc_block_wsse: bool,
    enc_cfg: &super::effort::HighLevelConfig,
    p2: &super::partitions::PartitionsData,
    p3: &super::partitions::PartitionsData,
    try_refining_endpoints: bool,
) -> bool {
    use super::block_helpers::{
        TrialFlags, encode_trial_refine_only, encode_trial_subsets, eval_error,
    };
    if out_block.trial_mode_index < 0 {
        return true;
    }
    let num_planes = if out_block.log_blk.dual_plane { 2 } else { 1 };
    let total = out_block.log_blk.grid_width as usize * out_block.log_blk.grid_height as usize;
    for plane_index in 0..num_planes {
        let log = &mut out_block.log_blk;
        let syms = &mut out_block.packed_dct_plane_data[plane_index];
        if !code_block_weights(grid_coder, enc_cfg.base_q, plane_index as u32, log, syms) {
            return false;
        }
        for i in 0..total {
            log.weights[i * num_planes + plane_index] = 0;
        }
        if !grid_coder.decode_block_weights(enc_cfg.base_q, plane_index as u32, log, syms) {
            return false;
        }
    }
    let p = &enc_cfg.cem_enc_params;
    if !try_refining_endpoints {
        if recalc_block_wsse {
            out_block.sse = eval_error(
                block_width,
                block_height,
                &out_block.log_blk,
                pixel_stats,
                p,
            );
        }
        return true;
    }
    let mut cur_err = eval_error(
        block_width,
        block_height,
        &out_block.log_blk,
        pixel_stats,
        p,
    );
    let mut nb = out_block.log_blk;
    let status = if nb.num_partitions == 1 {
        encode_trial_refine_only(
            block_width,
            block_height,
            pixel_stats,
            &mut nb,
            p,
            true,
            None,
        )
    } else {
        let pd = if nb.num_partitions == 2 { p2 } else { p3 };
        let u = pd.part_seed_to_unique_index[nb.partition_id as usize];
        let Some(pat) = pd.partition_pats.get(u as usize) else {
            return false;
        };
        let (cem, np, pid, er, wr, gw, gh) = (
            nb.color_endpoint_modes[0] as u32,
            nb.num_partitions as u32,
            nb.partition_id as u32,
            nb.endpoint_ise_range as u32,
            nb.weight_ise_range as u32,
            nb.grid_width as u32,
            nb.grid_height as u32,
        );
        encode_trial_subsets(
            block_width,
            block_height,
            pixel_stats,
            cem,
            np,
            pid,
            pat,
            er,
            wr,
            gw,
            gh,
            enc_cfg.encode_trial_subsets_early_out_thresh,
            &mut nb,
            p,
            true,
            TrialFlags {
                gradient_descent_flag: enc_cfg.gradient_descent_flag,
                polish_weights_flag: enc_cfg.polish_weights_flag,
                qcd_enabled_flag: enc_cfg.qcd_enabled_flag,
                use_blue_contraction: enc_cfg.use_blue_contraction,
            },
            None,
        )
    };
    if status {
        let n = ah::get_num_cem_values(nb.color_endpoint_modes[0] as u32) as usize
            * nb.num_partitions as usize;
        if out_block.log_blk.endpoints[..n] != nb.endpoints[..n] {
            for plane_index in 0..num_planes {
                if !grid_coder.decode_block_weights(
                    enc_cfg.base_q,
                    plane_index as u32,
                    &mut nb,
                    &out_block.packed_dct_plane_data[plane_index],
                ) {
                    return false;
                }
            }
            let new_err = eval_error(block_width, block_height, &nb, pixel_stats, p);
            if new_err < cur_err {
                cur_err = new_err;
                out_block.log_blk = nb;
            }
        }
    }
    if recalc_block_wsse {
        out_block.sse = cur_err;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_roundtrip_dc() {
        let d = Dct2f::new(4, 6).unwrap_or_default();
        let src = [1.0f32; 24];
        let mut dst = [0.0f32; 24];
        d.forward(&src, &mut dst);
        assert!((dst[0] - 24f32.sqrt()).abs() < 1e-4);
        assert!(dst[1..].iter().all(|v| v.abs() < 1e-4));
    }
}
