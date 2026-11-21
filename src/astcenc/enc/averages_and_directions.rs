// SPDX-FileCopyrightText: 2011-2026 Arm Limited
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from astc-encoder and modified; see NOTICE.md.
//! Port of `astcenc_averages_and_directions.cpp`: partition averages, dominant
//! directions and the squared error against the uncorrelated / same chroma lines.
//! Loops run 8 lanes wide with 4 lane accumulators like the AVX2 build.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use super::internal::*;
use super::vecmath::*;

isa_fn! {
/// Masked 8-wide accumulation of `data[i..i + 8]` into `acc` (`haccumulate(acc, loada(..), m)`).
#[inline]
fn acc_masked(acc: &mut Vfloat4, data: &[f32], i: usize, m: Vmask8) {
    Vfloat8::haccumulate_masked(acc, Vfloat8::loada(&data[i..]), m);
}
}

isa_fn! {
/// `compute_partition_averages_rgb` / `_rgba` for `P` partitions (2..4) and `C` components:
/// the first `P - 1` partitions are accumulated, the last one is the block total minus them.
#[inline]
fn partition_averages_n<const P: usize, const C: usize>(
    pi: &PartitionInfo,
    blk: &ImageBlock,
    mean: Vfloat4,
    averages: &mut [Vfloat4; 4],
) {
    let texel_count = blk.texel_count as usize;
    let mut pp = [[Vfloat4::zero(); C]; 3];
    let planes = [&blk.data_r, &blk.data_g, &blk.data_b, &blk.data_a];
    let mut lane_id = Vint8::lane_id();
    let mut i = 0;
    while i < texel_count {
        let texel_partition = Vint8::from_u8x8(&pi.partition_of_texel[i..]);
        let lane_mask = lane_id.lt(Vint8::splat(texel_count as i32));
        lane_id += Vint8::splat(SIMD_WIDTH as i32);
        let mut masks = [Vmask8::default(); 3];
        for p in 0..P - 1 {
            masks[p] = lane_mask & texel_partition.eq(Vint8::splat(p as i32));
        }
        for c in 0..C {
            let data = Vfloat8::loada(&planes[c][i..]);
            for p in 0..P - 1 {
                Vfloat8::haccumulate_masked(&mut pp[p][c], data, masks[p]);
            }
        }
        i += SIMD_WIDTH;
    }

    let block_total = mean * blk.texel_count as f32;
    let mut last = block_total;
    for p in 0..P - 1 {
        let s = |c: usize| pp[p][c].hadd_s();
        let t = if C == 3 {
            Vfloat4::new3(s(0), s(1), s(2))
        } else {
            Vfloat4::new(s(0), s(1), s(2), s(3))
        };
        last = last - t;
        averages[p] = t / pi.partition_texel_count[p] as f32;
    }
    averages[P - 1] = last / pi.partition_texel_count[P - 1] as f32;
}
}

isa_fn! {
/// `compute_partition_averages_rgb` / `_rgba` with `comps` components (3 or 4).
fn compute_partition_averages(
    pi: &PartitionInfo,
    blk: &ImageBlock,
    comps: usize,
    averages: &mut [Vfloat4; 4],
) {
    let mean = if comps == 3 {
        blk.data_mean.swz3(0, 1, 2)
    } else {
        blk.data_mean
    };
    match (pi.partition_count, comps) {
        (2, 3) => partition_averages_n::<2, 3>(pi, blk, mean, averages),
        (3, 3) => partition_averages_n::<3, 3>(pi, blk, mean, averages),
        (4, 3) => partition_averages_n::<4, 3>(pi, blk, mean, averages),
        (2, _) => partition_averages_n::<2, 4>(pi, blk, mean, averages),
        (3, _) => partition_averages_n::<3, 4>(pi, blk, mean, averages),
        (4, _) => partition_averages_n::<4, 4>(pi, blk, mean, averages),
        _ => averages[0] = mean,
    }
}
}

isa_fn! {
/// `compute_partition_averages_rgba_grayscale`: accumulates only R and A (G and B equal R),
/// keeping the general RGBA path's accumulation order and final-partition subtraction.
fn compute_partition_averages_rgba_grayscale(
    pi: &PartitionInfo,
    blk: &ImageBlock,
    averages: &mut [Vfloat4; 4],
) {
    let partition_count = pi.partition_count as usize;
    let texel_count = blk.texel_count as usize;
    if partition_count == 1 {
        averages[0] = blk.data_mean;
        return;
    }

    let mut pp = [[Vfloat4::zero(); 2]; 3];
    let mut lane_id = Vint8::lane_id();
    let mut i = 0;
    while i < texel_count {
        let texel_partition = Vint8::from_u8x8(&pi.partition_of_texel[i..]);
        let lane_mask = lane_id.lt(Vint8::splat(texel_count as i32));
        lane_id += Vint8::splat(SIMD_WIDTH as i32);
        let mut masks = [Vmask8::default(); 3];
        for p in 0..partition_count - 1 {
            masks[p] = lane_mask & texel_partition.eq(Vint8::splat(p as i32));
        }
        for p in 0..partition_count - 1 {
            acc_masked(&mut pp[p][0], &blk.data_r, i, masks[p]);
            acc_masked(&mut pp[p][1], &blk.data_a, i, masks[p]);
        }
        i += SIMD_WIDTH;
    }

    let block_total = blk.data_mean * blk.texel_count as f32;
    let mut last = block_total;
    for p in 0..partition_count - 1 {
        let r = pp[p][0].hadd_s();
        let t = Vfloat4::new(r, r, r, pp[p][1].hadd_s());
        last = last - t;
        averages[p] = t / pi.partition_texel_count[p] as f32;
    }
    averages[partition_count - 1] = last / pi.partition_texel_count[partition_count - 1] as f32;
}
}

isa_fn! {
/// Shared direction search: sum of the positive half spaces per axis, pick the longest.
fn best_direction(texels: impl Iterator<Item = Vfloat4>, average: Vfloat4, axes: usize) -> Vfloat4 {
    let mut sums = [Vfloat4::zero(); 4];
    let zero = Vfloat4::zero();
    for t in texels {
        let texel_datum = t - average;
        for (a, sum) in sums.iter_mut().enumerate().take(axes) {
            let tdm = texel_datum.swz4(a, a, a, a).gt(zero);
            *sum += Vfloat4::select(zero, texel_datum, tdm);
        }
    }
    let mut best_vector = sums[0];
    let mut best_sum = sums[0].dot(sums[0]);
    for a in 1..axes {
        let prod = sums[a].dot(sums[a]);
        let mask = prod.gt(best_sum);
        best_vector = Vfloat4::select(best_vector, sums[a], mask);
        if a + 1 < axes {
            best_sum = Vfloat4::select(best_sum, prod, mask);
        }
    }
    best_vector
}
}

isa_fn! {
/// `compute_avgs_and_dirs_4_comp_grayscale`: G and B equal R, so only the R and A half
/// spaces are summed. `rgb_metrics` receives the RGB-only direction.
fn compute_avgs_and_dirs_4_comp_grayscale(
    pi: &PartitionInfo,
    blk: &ImageBlock,
    pm: &mut [PartitionMetrics; 4],
    mut rgb_metrics: Option<&mut [PartitionMetrics; 4]>,
) {
    let partition_count = pi.partition_count as usize;
    let mut partition_averages = [Vfloat4::zero(); 4];
    compute_partition_averages_rgba_grayscale(pi, blk, &mut partition_averages);
    let zero = Vfloat4::zero();
    for partition in 0..partition_count {
        let texel_indexes = &pi.texels_of_partition[partition];
        let texel_count = pi.partition_texel_count[partition] as usize;
        let average = partition_averages[partition];
        pm[partition].avg = average;

        let mut sum_xp = zero;
        let mut sum_wp = zero;
        for &iwt in &texel_indexes[..texel_count] {
            let iwt = iwt as usize;
            let r = blk.data_r[iwt];
            let texel_datum = Vfloat4::new(r, r, r, blk.data_a[iwt]) - average;
            let tdm0 = texel_datum.swz4(0, 0, 0, 0).gt(zero);
            sum_xp += Vfloat4::select(zero, texel_datum, tdm0);
            let tdm3 = texel_datum.swz4(3, 3, 3, 3).gt(zero);
            sum_wp += Vfloat4::select(zero, texel_datum, tdm3);
        }

        // Preserve the RGB-only direction before alpha can change the RGBA winner
        if let Some(rm) = rgb_metrics.as_deref_mut() {
            rm[partition].avg = average.swz3(0, 1, 2);
            rm[partition].dir = sum_xp.swz3(0, 1, 2);
        }

        let prod_xp = sum_xp.dot(sum_xp);
        let prod_wp = sum_wp.dot(sum_wp);
        pm[partition].dir = Vfloat4::select(sum_xp, sum_wp, prod_wp.gt(prod_xp));
    }
}
}

isa_fn! {
/// `compute_avgs_and_dirs_4_comp`. `rgb_metrics` (grayscale blocks only) receives the
/// RGB partition metrics computed on the way.
pub fn compute_avgs_and_dirs_4_comp(
    pi: &PartitionInfo,
    blk: &ImageBlock,
    pm: &mut [PartitionMetrics; 4],
    rgb_metrics: Option<&mut [PartitionMetrics; 4]>,
) {
    if blk.grayscale {
        compute_avgs_and_dirs_4_comp_grayscale(pi, blk, pm, rgb_metrics);
        return;
    }
    let partition_count = pi.partition_count as usize;
    let mut partition_averages = [Vfloat4::zero(); 4];
    compute_partition_averages(pi, blk, 4, &mut partition_averages);
    for partition in 0..partition_count {
        let texel_indexes = &pi.texels_of_partition[partition];
        let texel_count = pi.partition_texel_count[partition] as usize;
        let average = partition_averages[partition];
        pm[partition].avg = average;
        pm[partition].dir = best_direction(
            texel_indexes[..texel_count]
                .iter()
                .map(|&i| blk.texel(i as usize)),
            average,
            4,
        );
    }
}
}

isa_fn! {
/// `compute_avgs_and_dirs_3_comp`.
pub fn compute_avgs_and_dirs_3_comp(
    pi: &PartitionInfo,
    blk: &ImageBlock,
    omitted_component: u32,
    pm: &mut [PartitionMetrics; 4],
) {
    let mut partition_averages = [Vfloat4::zero(); 4];
    compute_partition_averages(pi, blk, 4, &mut partition_averages);
    let (lanes, planes): ([usize; 3], [usize; 3]) = match omitted_component {
        0 => ([1, 2, 3], [1, 2, 3]),
        1 => ([0, 2, 3], [0, 2, 3]),
        2 => ([0, 1, 3], [0, 1, 3]),
        _ => ([0, 1, 2], [0, 1, 2]),
    };
    for avg in partition_averages.iter_mut() {
        *avg = avg.swz3(lanes[0], lanes[1], lanes[2]);
    }
    let (vr, vg, vb) = (
        blk.data(planes[0]),
        blk.data(planes[1]),
        blk.data(planes[2]),
    );
    let partition_count = pi.partition_count as usize;
    for partition in 0..partition_count {
        let texel_indexes = &pi.texels_of_partition[partition];
        let texel_count = pi.partition_texel_count[partition] as usize;
        let average = partition_averages[partition];
        pm[partition].avg = average;
        let texels = texel_indexes[..texel_count].iter().map(|&i| {
            let i = i as usize;
            Vfloat4::new3(vr[i], vg[i], vb[i])
        });
        pm[partition].dir = best_direction(texels, average, 3);
    }
}
}

isa_fn! {
/// `compute_avgs_and_dirs_3_comp_rgb`.
pub fn compute_avgs_and_dirs_3_comp_rgb(
    pi: &PartitionInfo,
    blk: &ImageBlock,
    pm: &mut [PartitionMetrics; 4],
) {
    let partition_count = pi.partition_count as usize;
    let mut partition_averages = [Vfloat4::zero(); 4];
    compute_partition_averages(pi, blk, 3, &mut partition_averages);
    for partition in 0..partition_count {
        let texel_indexes = &pi.texels_of_partition[partition];
        let texel_count = pi.partition_texel_count[partition] as usize;
        let average = partition_averages[partition];
        pm[partition].avg = average;
        pm[partition].dir = best_direction(
            texel_indexes[..texel_count]
                .iter()
                .map(|&i| blk.texel3(i as usize)),
            average,
            3,
        );
    }
}
}

isa_fn! {
/// `compute_avgs_and_dirs_2_comp`.
pub fn compute_avgs_and_dirs_2_comp(
    pt: &PartitionInfo,
    blk: &ImageBlock,
    component1: u32,
    component2: u32,
    pm: &mut [PartitionMetrics; 4],
) {
    let (mut average, vr, vg) = if component1 == 0 && component2 == 1 {
        (blk.data_mean.swz2(0, 1), blk.data(0), blk.data(1))
    } else if component1 == 0 && component2 == 2 {
        (blk.data_mean.swz2(0, 2), blk.data(0), blk.data(2))
    } else {
        (blk.data_mean.swz2(1, 2), blk.data(1), blk.data(2))
    };
    let partition_count = pt.partition_count as usize;
    for partition in 0..partition_count {
        let texel_indexes = &pt.texels_of_partition[partition];
        let texel_count = pt.partition_texel_count[partition] as usize;
        let idx = &texel_indexes[..texel_count];
        // Only compute a partition mean if more than one partition
        if partition_count > 1 {
            average = Vfloat4::zero();
            for &i in idx {
                average += Vfloat4::new(vr[i as usize], vg[i as usize], 0.0, 0.0);
            }
            average = average / texel_count as f32;
        }
        pm[partition].avg = average;
        let texels = idx
            .iter()
            .map(|&i| Vfloat4::new(vr[i as usize], vg[i as usize], 0.0, 0.0));
        pm[partition].dir = best_direction(texels, average, 2);
    }
}
}

isa_fn! {
/// Texel indices `texels_of_partition[p][min(i, last) ..][..4]` of the two batch candidates,
/// packed one per quad (`pack_quads` of `vint4(ptr)` loads); exhausted candidates repeat their
/// final padded quad.
#[inline]
fn batch_quad_indices(candidates: [&PartitionInfo; 2], p: usize, i: usize) -> Vint8 {
    let mut m = [0i32; 8];
    for c in 0..2 {
        let pi = candidates[c];
        let last = (pi.partition_texel_count[p] as usize).wrapping_sub(1) & !3;
        let base = i.min(last);
        if let Some(q) = pi.texels_of_partition[p]
            .get(base..)
            .and_then(|t| t.first_chunk::<4>())
        {
            for l in 0..4 {
                m[c * 4 + l] = i32::from(q[l]);
            }
        }
    }
    Vint8 { m }
}
}

isa_fn! {
/// `compute_partition_errors_luminance_batch`: luminance texels lie on the line (L, L, L)
/// through the origin which both endpoint models fit exactly, so only the range dependent
/// weight quantization error remains.
pub fn compute_partition_errors_luminance_batch(
    candidates: [&PartitionInfo; 2],
    blk: &ImageBlock,
    weight_imprecision: f32,
    uncor_errors: &mut [f32; 2],
    samec_errors: &mut [f32; 2],
) {
    let partition_count = candidates[0].partition_count as usize;
    let mut error_sum = Vfloat8::zero();
    let ew_r = Vfloat8::splat(blk.channel_weight.lane(0));
    let ew_g = Vfloat8::splat(blk.channel_weight.lane(1));
    let ew_b = Vfloat8::splat(blk.channel_weight.lane(2));

    // Process partition p from every candidate together, preserving each candidate's
    // partition order
    for p in 0..partition_count {
        let mut counts = [Vfloat4::zero(); 2];
        let mut max_count = 0usize;
        for c in 0..2 {
            let count = candidates[c].partition_texel_count[p] as usize;
            max_count = max_count.max(count);
            counts[c] = Vfloat4::splat(count as f32);
        }

        let mut low = Vfloat8::splat(1e10);
        let mut high = Vfloat8::splat(-1e10);
        let mut i = 0;
        while i < max_count {
            let r = Vfloat8::gather(&blk.data_r, batch_quad_indices(candidates, p, i));
            low = r.min(low);
            high = r.max(high);
            i += 4;
        }

        // Keep the channel weights separate and retain the RGB addition order
        let length = high.max_quads() - low.min_quads();
        let length_squared = length * length;
        let error_weight = Vfloat8::pack_quads(counts) * Vfloat8::splat(weight_imprecision);
        let error = (length_squared * (ew_r * error_weight))
            + (length_squared * (ew_g * error_weight))
            + (length_squared * (ew_b * error_weight));
        error_sum += error;
    }

    *uncor_errors = error_sum.lane0_quads();
    *samec_errors = error_sum.lane0_quads();
}
}

/// Which data path a candidate batch uses (`compute_partition_errors_batch::<KIND>`).
#[allow(non_snake_case, non_upper_case_globals)]
mod BatchKind {
    /// Grayscale RGB with alpha: RRRA direction lanes.
    pub const LuminanceAlpha: u8 = 0;
    /// RGB, alpha constant.
    pub const Rgb: u8 = 1;
    /// General RGBA.
    pub const Rgba: u8 = 2;
}

isa_fn! {
/// The shared body of `compute_partition_errors_{luminance_alpha,rgb,rgba}_batch`: direction
/// finding with one candidate per quad, then error scoring of four texels per candidate quad,
/// keeping each candidate's arithmetic order of the single candidate code.
#[inline]
fn compute_partition_errors_batch<const KIND: u8>(
    candidates: [&PartitionInfo; 2],
    blk: &ImageBlock,
    weight_imprecision: f32,
    uncor_errors: &mut [f32; 2],
    samec_errors: &mut [f32; 2],
) {
    let kind = KIND;
    let partition_count = candidates[0].partition_count as usize;

    // Reuse the original per-channel texel accumulation and reduction for partition averages
    let mut averages = [[Vfloat4::zero(); 4]; 2];
    for c in 0..2 {
        match kind {
            BatchKind::LuminanceAlpha => {
                compute_partition_averages_rgba_grayscale(candidates[c], blk, &mut averages[c])
            }
            BatchKind::Rgb => compute_partition_averages(candidates[c], blk, 3, &mut averages[c]),
            _ => compute_partition_averages(candidates[c], blk, 4, &mut averages[c]),
        }
    }

    let safe = Vfloat8::broadcast_quad(if kind == BatchKind::Rgb {
        #[allow(clippy::excessive_precision)]
        let v = 0.577350258827209473f32;
        Vfloat4::new(v, v, v, 0.0)
    } else {
        Vfloat4::splat(0.5)
    });
    let channel_weight = Vfloat8::broadcast_quad(blk.channel_weight);
    let zero = Vfloat8::zero();

    // Preserve four running texel-error sums per candidate across partition boundaries
    let mut uncor_sum = zero;
    let mut samec_sum = zero;
    let mut uncor_quant = [zero; 4];
    let mut samec_quant = [zero; 4];

    let ew_r = Vfloat8::splat(blk.channel_weight.lane(0));
    let ew_g = Vfloat8::splat(blk.channel_weight.lane(1));
    let ew_b = Vfloat8::splat(blk.channel_weight.lane(2));
    let ew_a = Vfloat8::splat(blk.channel_weight.lane(3));

    for p in 0..partition_count {
        let mut colors = [Vfloat4::zero(); 2];
        let mut counts = [Vfloat4::zero(); 2];
        let mut max_count = 0usize;
        for c in 0..2 {
            let count = candidates[c].partition_texel_count[p] as usize;
            max_count = max_count.max(count);
            colors[c] = averages[c][p];
            counts[c] = Vfloat4::splat(count as f32);
        }

        // Direction estimation, one candidate per quad
        let mean = Vfloat8::pack_quads(colors);
        let countv = Vfloat8::pack_quads(counts);
        let mut sum_xp = zero;
        let mut sum_yp = zero;
        let mut sum_zp = zero;
        let mut sum_wp = zero;
        let mut index = zero;
        for i in 0..max_count {
            for c in 0..2 {
                let pi = candidates[c];
                let last = (pi.partition_texel_count[p] as usize).wrapping_sub(1);
                let texel = pi.texels_of_partition[p][i.min(last).min(BLOCK_MAX_TEXELS - 1)] as usize;
                colors[c] = match kind {
                    BatchKind::LuminanceAlpha => {
                        let r = blk.data_r[texel];
                        Vfloat4::new(r, r, r, blk.data_a[texel])
                    }
                    BatchKind::Rgb => blk.texel3(texel),
                    _ => blk.texel(texel),
                };
            }

            // Preserve texel order and mask exhausted candidates
            let datum = Vfloat8::select(zero, Vfloat8::pack_quads(colors) - mean, index.lt(countv));
            index += Vfloat8::splat(1.0);

            sum_xp += Vfloat8::select(zero, datum, datum.quad_lane(0).gt(zero));
            if kind != BatchKind::LuminanceAlpha {
                sum_yp += Vfloat8::select(zero, datum, datum.quad_lane(1).gt(zero));
                sum_zp += Vfloat8::select(zero, datum, datum.quad_lane(2).gt(zero));
            }
            if kind != BatchKind::Rgb {
                sum_wp += Vfloat8::select(zero, datum, datum.quad_lane(3).gt(zero));
            }
        }

        // Select the dominant direction independently for each candidate
        let best = match kind {
            BatchKind::LuminanceAlpha => {
                let prod_xp = sum_xp.dot_quads(sum_xp);
                let prod_wp = sum_wp.dot_quads(sum_wp);
                Vfloat8::select(sum_xp, sum_wp, prod_wp.gt(prod_xp))
            }
            _ => {
                let prod_xp = sum_xp.dot_quads(sum_xp);
                let prod_yp = sum_yp.dot_quads(sum_yp);
                let prod_zp = sum_zp.dot_quads(sum_zp);
                let mut best = sum_xp;
                let mut best_sum = prod_xp;
                let mask = prod_yp.gt(best_sum);
                best = Vfloat8::select(best, sum_yp, mask);
                best_sum = Vfloat8::select(best_sum, prod_yp, mask);
                let mask = prod_zp.gt(best_sum);
                best = Vfloat8::select(best, sum_zp, mask);
                if kind == BatchKind::Rgba {
                    best_sum = Vfloat8::select(best_sum, prod_zp, mask);
                    let prod_wp = sum_wp.dot_quads(sum_wp);
                    best = Vfloat8::select(best, sum_wp, prod_wp.gt(best_sum));
                }
                best
            }
        };

        // Keep direction selection, normalization, and line setup in the same candidate groups
        let bs = best.normalize_safe_quads(safe);
        let samec_bs = mean.normalize_safe_quads(safe);
        let amod = if kind == BatchKind::Rgb {
            mean - bs * (mean * bs).sum_rgb_quads()
        } else {
            mean - bs * mean.dot_quads(bs)
        };

        let (bs0, bs1, bs2, bs3) = (
            bs.quad_lane(0),
            bs.quad_lane(1),
            bs.quad_lane(2),
            bs.quad_lane(3),
        );
        let (amod0, amod1, amod2, amod3) = (
            amod.quad_lane(0),
            amod.quad_lane(1),
            amod.quad_lane(2),
            amod.quad_lane(3),
        );
        let (samec0, samec1, samec2, samec3) = (
            samec_bs.quad_lane(0),
            samec_bs.quad_lane(1),
            samec_bs.quad_lane(2),
            samec_bs.quad_lane(3),
        );
        let mut lane_ids = (Vint8::lane_id() & 3).int_to_float();

        let mut low = Vfloat8::splat(1e10);
        let mut high = Vfloat8::splat(-1e10);
        let mut i = 0;
        while i < max_count {
            let texels = batch_quad_indices(candidates, p, i);
            let active = lane_ids.lt(countv);
            lane_ids += Vfloat8::splat(4.0);

            let r = Vfloat8::gather(&blk.data_r, texels);
            match kind {
                BatchKind::LuminanceAlpha => {
                    let a = Vfloat8::gather(&blk.data_a, texels);
                    // Retain the three RGB additions and separately weighted channel errors
                    let rgb_param = r * bs0;
                    let param = rgb_param + rgb_param + rgb_param + (a * bs3);
                    low = param.min(low);
                    high = param.max(high);
                    let dr = (amod0 - r) + (param * bs0);
                    let da = (amod3 - a) + (param * bs3);
                    let error = (ew_r * dr * dr) + (ew_g * dr * dr) + (ew_b * dr * dr)
                        + (ew_a * da * da);
                    uncor_sum += Vfloat8::select(zero, error, active);

                    let rgb_param = r * samec0;
                    let param = rgb_param + rgb_param + rgb_param + (a * samec3);
                    let dr = param * samec0 - r;
                    let da = param * samec3 - a;
                    let error = (ew_r * dr * dr) + (ew_g * dr * dr) + (ew_b * dr * dr)
                        + (ew_a * da * da);
                    samec_sum += Vfloat8::select(zero, error, active);
                }
                BatchKind::Rgb => {
                    let g = Vfloat8::gather(&blk.data_g, texels);
                    let b = Vfloat8::gather(&blk.data_b, texels);
                    let param = (r * bs0) + (g * bs1) + (b * bs2);
                    low = param.min(low);
                    high = param.max(high);
                    let d0 = (amod0 - r) + (param * bs0);
                    let d1 = (amod1 - g) + (param * bs1);
                    let d2 = (amod2 - b) + (param * bs2);
                    let error = (ew_r * d0 * d0) + (ew_g * d1 * d1) + (ew_b * d2 * d2);
                    uncor_sum += Vfloat8::select(zero, error, active);

                    let param = (r * samec0) + (g * samec1) + (b * samec2);
                    let d0 = param * samec0 - r;
                    let d1 = param * samec1 - g;
                    let d2 = param * samec2 - b;
                    let error = (ew_r * d0 * d0) + (ew_g * d1 * d1) + (ew_b * d2 * d2);
                    samec_sum += Vfloat8::select(zero, error, active);
                }
                _ => {
                    let g = Vfloat8::gather(&blk.data_g, texels);
                    let b = Vfloat8::gather(&blk.data_b, texels);
                    let a = Vfloat8::gather(&blk.data_a, texels);
                    let param = (r * bs0) + (g * bs1) + (b * bs2) + (a * bs3);
                    low = param.min(low);
                    high = param.max(high);
                    let d0 = (amod0 - r) + (param * bs0);
                    let d1 = (amod1 - g) + (param * bs1);
                    let d2 = (amod2 - b) + (param * bs2);
                    let d3 = (amod3 - a) + (param * bs3);
                    let error = (ew_r * d0 * d0) + (ew_g * d1 * d1) + (ew_b * d2 * d2)
                        + (ew_a * d3 * d3);
                    uncor_sum += Vfloat8::select(zero, error, active);

                    let param = (r * samec0) + (g * samec1) + (b * samec2) + (a * samec3);
                    let d0 = param * samec0 - r;
                    let d1 = param * samec1 - g;
                    let d2 = param * samec2 - b;
                    let d3 = param * samec3 - a;
                    let error = (ew_r * d0 * d0) + (ew_g * d1 * d1) + (ew_b * d2 * d2)
                        + (ew_a * d3 * d3);
                    samec_sum += Vfloat8::select(zero, error, active);
                }
            }
            i += 4;
        }

        // Defer adding quantization errors until after the texel sums are reduced
        let length = (high.max_quads() - low.min_quads()).max_s(1e-7);
        let error_weights = channel_weight * (countv * Vfloat8::splat(weight_imprecision));
        let uncor_vector = bs * length;
        let samec_vector = samec_bs * length;
        let uq = (uncor_vector * uncor_vector) * error_weights;
        let sq = (samec_vector * samec_vector) * error_weights;
        if kind == BatchKind::Rgb {
            uncor_quant[p] = uq.sum_rgb_quads();
            samec_quant[p] = sq.sum_rgb_quads();
        } else {
            uncor_quant[p] = uq.sum_quads();
            samec_quant[p] = sq.sum_quads();
        }
    }

    // Reduce each candidate's texel sums as (s0 + s2) + (s1 + s3), matching hadd_s(vfloat4),
    // then add quantization errors in the original partition order
    uncor_sum = uncor_sum.sum_quads();
    samec_sum = samec_sum.sum_quads();
    for p in 0..partition_count {
        uncor_sum += uncor_quant[p];
        samec_sum += samec_quant[p];
    }
    *uncor_errors = uncor_sum.lane0_quads();
    *samec_errors = samec_sum.lane0_quads();
}
}

isa_fn! {
/// `compute_partition_errors_luminance_alpha_batch`.
pub fn compute_partition_errors_luminance_alpha_batch(
    candidates: [&PartitionInfo; 2],
    blk: &ImageBlock,
    weight_imprecision: f32,
    uncor_errors: &mut [f32; 2],
    samec_errors: &mut [f32; 2],
) {
    compute_partition_errors_batch::<{ BatchKind::LuminanceAlpha }>(
        candidates,
        blk,
        weight_imprecision,
        uncor_errors,
        samec_errors,
    );
}
}

isa_fn! {
/// `compute_partition_errors_rgb_batch`.
pub fn compute_partition_errors_rgb_batch(
    candidates: [&PartitionInfo; 2],
    blk: &ImageBlock,
    weight_imprecision: f32,
    uncor_errors: &mut [f32; 2],
    samec_errors: &mut [f32; 2],
) {
    compute_partition_errors_batch::<{ BatchKind::Rgb }>(
        candidates,
        blk,
        weight_imprecision,
        uncor_errors,
        samec_errors,
    );
}
}

isa_fn! {
/// `compute_partition_errors_rgba_batch`.
pub fn compute_partition_errors_rgba_batch(
    candidates: [&PartitionInfo; 2],
    blk: &ImageBlock,
    weight_imprecision: f32,
    uncor_errors: &mut [f32; 2],
    samec_errors: &mut [f32; 2],
) {
    compute_partition_errors_batch::<{ BatchKind::Rgba }>(
        candidates,
        blk,
        weight_imprecision,
        uncor_errors,
        samec_errors,
    );
}
}
