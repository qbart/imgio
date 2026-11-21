// SPDX-FileCopyrightText: 2011-2026 Arm Limited
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from astc-encoder and modified; see NOTICE.md.
//! Port of `astcenc_compute_variance.cpp` (`init_compute_averages`,
//! `compute_pixel_region_variance`) and the `compute_averages` worker of `astcenc_entry.cpp`:
//! the alpha averages over a `(2 * radius + 1)` texel kernel (`a_scale_radius`) that the
//! compressor uses to skip blocks with negligible alpha.
//!
//! The summed-area tables use the Brent-Kung parallel prefix order of the C++ code. Only the
//! alpha lane of the C++ `N` table reaches the output (the `N^2` table and the RGB lanes are
//! unused), so only that lane is computed; per lane float math is identical.

use super::vecmath::{Vint4, float16_to_float_v};
use super::{AstcencImage, AstcencImageData, AstcencSwizzle, AstcencSwz, ParallelManager};

/// `avg_args`: the region averaging setup shared by the worker threads.
#[derive(Debug, Clone, Copy)]
pub struct AvgArgs {
    /// `alpha_kernel_radius`.
    pub radius: usize,
    /// `have_z`: the image has more than one slice (3D kernel).
    pub have_z: bool,
    pub img_size_x: usize,
    pub img_size_y: usize,
    pub img_size_z: usize,
    /// `blk_size_xy`: the maximum working region size in X and Y.
    pub blk_size_xy: usize,
    /// `blk_size_z`: the maximum working region size in Z.
    pub blk_size_z: usize,
    /// `work_memory_size` (alpha lane only: one float per padded texel).
    pub work_memory_size: usize,
}

impl AvgArgs {
    /// `init_compute_averages`: the region setup and the parallel task count.
    pub fn new(img: &AstcencImage, alpha_kernel_radius: usize) -> (AvgArgs, usize) {
        let size_x = img.dim_x as usize;
        let size_y = img.dim_y as usize;
        let size_z = img.dim_z as usize;

        // Compute maximum block size and from that the working memory buffer size
        let kerneldim = 2 * alpha_kernel_radius + 1;

        let have_z = size_z > 1;
        let max_blk_size_xy = if have_z { 16 } else { 32 };

        let max_blk_size_z_lim = if have_z { 16 } else { 1 };
        let max_blk_size_z = size_z.min(max_blk_size_z_lim);

        let max_padsize_xy = max_blk_size_xy + kerneldim;
        let max_padsize_z = max_blk_size_z + if have_z { kerneldim } else { 0 };

        let args = AvgArgs {
            radius: alpha_kernel_radius,
            have_z,
            img_size_x: size_x,
            img_size_y: size_y,
            img_size_z: size_z,
            blk_size_xy: max_blk_size_xy,
            blk_size_z: max_blk_size_z,
            work_memory_size: max_padsize_xy * max_padsize_xy * max_padsize_z,
        };

        // The parallel task count
        let tasks_z = size_z.div_ceil(max_blk_size_z.max(1));
        let tasks_y = size_y.div_ceil(max_blk_size_xy);
        (args, tasks_z * tasks_y)
    }
}

/// One `pixel_region_args` region: size and offset in X, Y and Z.
#[derive(Debug, Clone, Copy)]
struct Region {
    size: [usize; 3],
    offset: [usize; 3],
}

/// `brent_kung_prefix_sum` over `items` values of `d` starting at `base`, `stride` apart.
fn brent_kung_prefix_sum(d: &mut [f32], base: usize, items: usize, stride: usize) {
    if items < 2 {
        return;
    }
    let mut add = |dst: usize, src: usize| {
        if let Some(&s) = d.get(src)
            && let Some(o) = d.get_mut(dst)
        {
            *o += s;
        }
    };

    let mut lc_stride = 2usize;
    let mut log2_stride = 1usize;

    // The reduction-tree loop
    loop {
        let step = lc_stride >> 1;
        let start = lc_stride - 1;
        let iters = items >> log2_stride;

        let mut da = base + start * stride;
        let ofs = step * stride;
        let ofs_stride = stride << log2_stride;
        for _ in 0..iters {
            add(da, da - ofs);
            da += ofs_stride;
        }

        log2_stride += 1;
        lc_stride <<= 1;
        if lc_stride > items {
            break;
        }
    }

    // The expansion-tree loop
    loop {
        log2_stride -= 1;
        lc_stride >>= 1;

        let step = lc_stride >> 1;
        let start = step + lc_stride - 1;
        let iters = (items - step) >> log2_stride;

        let mut da = base + start * stride;
        let ofs = step * stride;
        let ofs_stride = stride << log2_stride;
        for _ in 0..iters {
            add(da, da - ofs);
            da += ofs_stride;
        }
        if lc_stride <= 2 {
            break;
        }
    }
}

/// The source coordinate of padded work buffer coordinate `p` (1-based in the padded
/// region): clamped to the image edge.
fn src_coord(p: usize, offset: usize, radius: usize, dim: usize) -> usize {
    (p + offset)
        .saturating_sub(radius)
        .min(dim.saturating_sub(1))
}

/// The swizzled, clamped alpha of texel `(x, y)` of slice `z`, as the C++ loaders of
/// `compute_pixel_region_variance` produce it.
fn load_alpha(img: &AstcencImage, swz: &AstcencSwizzle, z: usize, y: usize, x: usize) -> f32 {
    let base = 4 * (img.dim_x as usize * y + x);
    match &img.data {
        AstcencImageData::U8(p) => {
            let d = p.get(z).and_then(|d| d.get(base..base + 4));
            let c = |s: AstcencSwz| match s {
                AstcencSwz::Zero => 0u8,
                AstcencSwz::One => 255,
                s => d.and_then(|d| d.get(s as usize)).copied().unwrap_or(0),
            };
            f32::from(c(swz.a)) * (1.0 / 255.0)
        }
        AstcencImageData::F16(p) => {
            let d = p.get(z).and_then(|d| d.get(base..base + 4));
            let c = |s: AstcencSwz| match s {
                AstcencSwz::Zero => 0u16,
                AstcencSwz::One => 0x3C00,
                s => d.and_then(|d| d.get(s as usize)).copied().unwrap_or(0),
            };
            let a = float16_to_float_v(Vint4::new(0, 0, 0, i32::from(c(swz.a)))).lane(3);
            // Match the block loader's alpha clamping before building the summed-area table
            clamp1f(a)
        }
        AstcencImageData::F32(p) => {
            let d = p.get(z).and_then(|d| d.get(base..base + 4));
            let c = |s: AstcencSwz| match s {
                AstcencSwz::Zero => 0.0f32,
                AstcencSwz::One => 1.0,
                s => d.and_then(|d| d.get(s as usize)).copied().unwrap_or(0.0),
            };
            // Match the block loader's alpha clamping before building the summed-area table
            clamp1f(c(swz.a))
        }
    }
}

/// `astc::clamp1f` (NaN clamps to 0).
fn clamp1f(v: f32) -> f32 {
    if v > 1.0 {
        1.0
    } else if v > 0.0 {
        v
    } else {
        0.0
    }
}

/// `compute_pixel_region_variance`: the alpha averages of region `arg`, written to `out`
/// (`size_x * size_y * size_z` values in x, y, z order) using `work` as the summed-area table.
fn compute_pixel_region_variance(
    img: &AstcencImage,
    swz: &AstcencSwizzle,
    ag: &AvgArgs,
    arg: Region,
    work: &mut [f32],
    out: &mut [f32],
) {
    let have_z = ag.have_z;
    let [size_x, size_y, size_z] = arg.size;
    let [offset_x, offset_y, offset_z] = arg.offset;
    let alpha_kernel_radius = ag.radius;

    // Compute buffer sizes and dimensions
    let kernel_radius = alpha_kernel_radius;
    let kerneldim = 2 * kernel_radius + 1;
    let kernel_radius_xy = kernel_radius;
    let kernel_radius_z = if have_z { kernel_radius } else { 0 };

    let padsize_x = size_x + kerneldim;
    let padsize_y = size_y + kerneldim;
    let padsize_z = size_z + if have_z { kerneldim } else { 0 };

    let zd_start = usize::from(have_z);

    // Scaling factors to apply to Y and Z for accesses into the work buffers
    let yst = padsize_x;
    let zst = padsize_x * padsize_y;
    let idx = |z: usize, y: usize, x: usize| z * zst + y * yst + x;

    let dim_x = img.dim_x as usize;
    let dim_y = img.dim_y as usize;
    let dim_z = img.dim_z as usize;

    // Load N values into the work buffer
    for z in zd_start..padsize_z {
        let z_src = src_coord(z - zd_start, offset_z, kernel_radius_z, dim_z);
        for y in 1..padsize_y {
            let y_src = src_coord(y - 1, offset_y, kernel_radius_xy, dim_y);
            for x in 1..padsize_x {
                let x_src = src_coord(x - 1, offset_x, kernel_radius_xy, dim_x);
                if let Some(w) = work.get_mut(idx(z, y, x)) {
                    *w = load_alpha(img, swz, z_src, y_src, x_src);
                }
            }
        }
    }

    // Pad with an extra layer of 0s; this forms the edge of the SAT tables
    let mut zero = |i: usize| {
        if let Some(w) = work.get_mut(i) {
            *w = 0.0;
        }
    };
    for z in 0..padsize_z {
        for y in 0..padsize_y {
            zero(idx(z, y, 0));
        }
        for x in 0..padsize_x {
            zero(idx(z, 0, x));
        }
    }
    if have_z {
        for y in 0..padsize_y {
            for x in 0..padsize_x {
                zero(idx(0, y, x));
            }
        }
    }

    // Generate the summed-area table; this is done in-place, using a Brent-Kung parallel-prefix
    // based algorithm to minimize precision loss
    for z in zd_start..padsize_z {
        for y in 1..padsize_y {
            brent_kung_prefix_sum(work, idx(z, y, 1), padsize_x - 1, 1);
        }
    }
    for z in zd_start..padsize_z {
        for x in 1..padsize_x {
            brent_kung_prefix_sum(work, idx(z, 1, x), padsize_y - 1, yst);
        }
    }
    if have_z {
        for y in 1..padsize_y {
            for x in 1..padsize_x {
                brent_kung_prefix_sum(work, idx(1, y, x), padsize_z - 1, zst);
            }
        }
    }

    // Compute constants used in the variance-calculation
    let alpha_kdim = (2 * alpha_kernel_radius + 1) as f32;
    let alpha_rsamples = if have_z {
        1.0 / (alpha_kdim * alpha_kdim * alpha_kdim)
    } else {
        1.0 / (alpha_kdim * alpha_kdim)
    };

    let w = |z: usize, y: usize, x: usize| work.get(idx(z, y, x)).copied().unwrap_or(0.0);
    let mut out_index = 0;
    // Use the summed-area tables to compute variance for each neighborhood
    for z in 0..size_z {
        let z_src = z + kernel_radius_z;
        let z_low = z_src.wrapping_sub(alpha_kernel_radius);
        let z_high = z_src + alpha_kernel_radius + 1;
        for y in 0..size_y {
            let y_src = y + kernel_radius_xy;
            let y_low = y_src - alpha_kernel_radius;
            let y_high = y_src + alpha_kernel_radius + 1;
            for x in 0..size_x {
                let x_src = x + kernel_radius_xy;
                let x_low = x_src - alpha_kernel_radius;
                let x_high = x_src + alpha_kernel_radius + 1;

                // Summed-area table lookups for alpha average
                let vasum = if have_z {
                    (w(z_high, y_low, x_low) - w(z_high, y_low, x_high) - w(z_high, y_high, x_low)
                        + w(z_high, y_high, x_high))
                        - (w(z_low, y_low, x_low)
                            - w(z_low, y_low, x_high)
                            - w(z_low, y_high, x_low)
                            + w(z_low, y_high, x_high))
                } else {
                    w(0, y_low, x_low) - w(0, y_low, x_high) - w(0, y_high, x_low)
                        + w(0, y_high, x_high)
                };

                if let Some(o) = out.get_mut(out_index) {
                    *o = vasum * alpha_rsamples;
                }
                out_index += 1;
            }
        }
    }
}

/// `compute_averages`: computes the regions the task manager hands to this thread and writes
/// their alpha averages into `averages` (`dim_x * dim_y * dim_z` values). Returns `false` when
/// a working buffer could not be allocated (the task manager is then cancelled).
pub fn compute_averages(
    manage_avg: &ParallelManager,
    ag: &AvgArgs,
    img: &AstcencImage,
    swz: &AstcencSwizzle,
    averages: &std::sync::RwLock<Vec<f32>>,
) -> bool {
    let size_x = ag.img_size_x;
    let size_y = ag.img_size_y;
    let size_z = ag.img_size_z;

    let step_xy = ag.blk_size_xy;
    let step_z = ag.blk_size_z;

    let tasks_y = size_y.div_ceil(step_xy);
    let region_len = step_xy * step_xy * step_z;

    let mut work = Vec::new();
    let mut region = Vec::new();
    if work.try_reserve_exact(ag.work_memory_size).is_err()
        || region.try_reserve_exact(region_len).is_err()
    {
        manage_avg.cancel();
        return false;
    }
    work.resize(ag.work_memory_size, 0.0f32);
    region.resize(region_len, 0.0f32);

    let mut ok = true;
    // Each thread claims tasks until none remain
    loop {
        let (base, count) = manage_avg.get_task_assignment(16);
        if count == 0 {
            break;
        }

        for i in base..base + count {
            let z_task = i / tasks_y;
            let y_task = i - (z_task * tasks_y);

            let z = z_task * step_z;
            let y = y_task * step_xy;

            let rsize_z = step_z.min(size_z - z);
            let rsize_y = step_xy.min(size_y - y);

            let mut x = 0;
            while x < size_x {
                let rsize_x = step_xy.min(size_x - x);
                let arg = Region {
                    size: [rsize_x, rsize_y, rsize_z],
                    offset: [x, y, z],
                };
                compute_pixel_region_variance(img, swz, ag, arg, &mut work, &mut region);

                // Copy the region into the shared table (regions are disjoint)
                match averages.write() {
                    Ok(mut out) => {
                        for rz in 0..rsize_z {
                            for ry in 0..rsize_y {
                                let src = (rz * rsize_y + ry) * rsize_x;
                                let dst = ((z + rz) * size_y + (y + ry)) * size_x + x;
                                match (
                                    region.get(src..src + rsize_x),
                                    out.get_mut(dst..dst + rsize_x),
                                ) {
                                    (Some(s), Some(d)) => d.copy_from_slice(s),
                                    _ => ok = false,
                                }
                            }
                        }
                    }
                    Err(_) => ok = false,
                }
                x += step_xy;
            }
        }

        if !ok {
            manage_avg.cancel();
        }
        manage_avg.complete_task_assignment(count);
        if !ok {
            break;
        }
    }
    ok
}
