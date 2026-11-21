// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of `basist::astc_ldr_t::astc_block_grid_data` and its global table
//! (`init_astc_block_grid_data_hash` in the transcoder plus the encoder-only fields filled by
//! `astc_ldr::encoder_init`): per block size / weight grid size up- and downsampling data.
//! Entries are built lazily on first use.

use crate::astc::helpers::{self as ah, MAX_GRID_WEIGHTS, WeightedSample};
use std::sync::OnceLock;

/// `astc_block_grid_data`.
#[derive(Clone, Debug, Default)]
pub(crate) struct AstcBlockGridData {
    pub weight_gamma: f32,
    /// rows = block texels, cols = grid samples (`[r * num_grid + c]`).
    pub upsample_matrix: Vec<f32>,
    /// rows = grid samples, cols = block texels.
    pub downsample_matrix: Vec<f32>,
    /// For each grid sample: packed `x | (y << 8)` of the texels it influences.
    pub grid_to_texel_influence_list: Vec<Vec<u16>>,
    pub upsample_weights: Vec<WeightedSample>,
    /// `At` (rows = grid samples, cols = block texels), for gradient descent.
    pub unweighted_downsample_matrix: Vec<f32>,
    /// `1 / diag(AtA)`.
    pub one_over_diag_ata: Vec<f32>,
}

/// `compute_upsample_matrix`: `[texel * num_grid + grid]`.
pub(crate) fn compute_upsample_matrix(bw: u32, bh: u32, gw: u32, gh: u32) -> Vec<f32> {
    let num_block = (bw * bh) as usize;
    let num_grid = (gw * gh) as usize;
    let mut samples = vec![WeightedSample::default(); num_block];
    ah::compute_upsample_weights(bw, bh, gw, gh, &mut samples);
    let mut m = vec![0.0f32; num_block * num_grid];
    for (d, ws) in samples.iter().enumerate() {
        let row = &mut m[d * num_grid..(d + 1) * num_grid];
        for y in 0..2u32 {
            for x in 0..2u32 {
                let w = f32::from(ws.weights[y as usize][x as usize]) * (1.0 / 16.0);
                if w == 0.0 {
                    continue;
                }
                let idx = (u32::from(ws.src_x) + x) + (u32::from(ws.src_y) + y) * gw;
                if let Some(r) = row.get_mut(idx as usize) {
                    *r = w;
                }
            }
        }
    }
    m
}

/// `compute_adjoint_downsample_matrix`.
fn compute_adjoint_downsample_matrix(up: &[f32], num_block: usize, num_grid: usize) -> Vec<f32> {
    let mut dinv = vec![0.0f32; num_grid];
    for (j, dj) in dinv.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for i in 0..num_block {
            sum += up[i * num_grid + j];
        }
        if sum > 0.0 {
            *dj = 1.0 / sum;
        }
    }
    let mut d = vec![0.0f32; num_grid * num_block];
    for j in 0..num_grid {
        for i in 0..num_block {
            d[j * num_block + i] = dinv[j] * up[i * num_grid + j];
        }
    }
    d
}

fn build(bw: u32, bh: u32, gw: u32, gh: u32) -> AstcBlockGridData {
    let num_block = (bw * bh) as usize;
    let num_grid = (gw * gh) as usize;
    let up = compute_upsample_matrix(bw, bh, gw, gh);

    let mut accum = 0.0f32;
    for t in 0..num_block {
        let mut row_sum_sq = 0.0f32;
        for &w in &up[t * num_grid..(t + 1) * num_grid] {
            row_sum_sq += w * w;
        }
        accum += row_sum_sq;
    }
    let weight_gamma = accum / num_block as f32;

    let down = compute_adjoint_downsample_matrix(&up, num_block, num_grid);

    let mut upsample_weights = vec![WeightedSample::default(); num_block];
    ah::compute_upsample_weights(bw, bh, gw, gh, &mut upsample_weights);

    let mut influence = vec![Vec::new(); num_grid];
    for (g, list) in influence.iter_mut().enumerate() {
        for b in 0..num_block {
            if down[g * num_block + b] == 0.0 {
                continue;
            }
            let (x, y) = (b as u32 % bw, b as u32 / bw);
            list.push((x | (y << 8)) as u16);
        }
    }

    // compute_upsample_matrix_transposed
    let mut at = vec![0.0f32; num_grid * num_block];
    for j in 0..num_grid {
        for i in 0..num_block {
            at[j * num_block + i] = up[i * num_grid + j];
        }
    }

    // compute_diag_AtA_vector
    let mut diag = vec![0.0f32; num_grid];
    for r in 0..num_block {
        for (c, d) in diag.iter_mut().enumerate() {
            let arc = up[r * num_grid + c];
            *d += arc * arc;
        }
    }
    for d in diag.iter_mut() {
        *d = 1.0 / *d;
    }

    AstcBlockGridData {
        weight_gamma,
        upsample_matrix: up,
        downsample_matrix: down,
        grid_to_texel_influence_list: influence,
        upsample_weights,
        unweighted_downsample_matrix: at,
        one_over_diag_ata: diag,
    }
}

const BWH_COUNT: usize = 9;
const GWH_COUNT: usize = 11;
const LUT_SIZE: usize = BWH_COUNT * BWH_COUNT * GWH_COUNT * GWH_COUNT;

/// `find_astc_block_grid_data(block_width, block_height, grid_width, grid_height)`. `None`
/// for an invalid configuration.
pub(crate) fn find_astc_block_grid_data(
    bw: u32,
    bh: u32,
    gw: u32,
    gh: u32,
) -> Option<&'static AstcBlockGridData> {
    static T: OnceLock<Vec<OnceLock<AstcBlockGridData>>> = OnceLock::new();
    if !(4..=12).contains(&bw)
        || !(4..=12).contains(&bh)
        || gw < 2
        || gh < 2
        || gw > bw
        || gh > bh
        || gw * gh > MAX_GRID_WEIGHTS
    {
        return None;
    }
    let idx = (gh as usize - 2)
        + (gw as usize - 2) * GWH_COUNT
        + (bh as usize - 4) * GWH_COUNT * GWH_COUNT
        + (bw as usize - 4) * GWH_COUNT * GWH_COUNT * BWH_COUNT;
    let t = T.get_or_init(|| (0..LUT_SIZE).map(|_| OnceLock::new()).collect());
    t.get(idx)
        .map(|slot| slot.get_or_init(|| build(bw, bh, gw, gh)))
}
