// SPDX-FileCopyrightText: 2011-2026 Arm Limited
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from astc-encoder and modified; see NOTICE.md.
//! Port of `astcenc_compress_symbolic.cpp`: the per block search over partition counts,
//! planes and block modes, with weight realignment and refinement.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use super::color_quantize::pack_color_endpoints;
use super::decompress_symbolic::*;
use super::find_best_partitioning::find_best_partition_candidates;
use super::ideal_endpoints_and_weights::*;
use super::internal::*;
use super::mathlib;
use super::pick_best_endpoint_format::compute_ideal_endpoint_formats;
use super::symbolic_physical::symbolic_to_physical;
use super::tables::QUANT_AND_XFER_TABLES;
use super::vecmath::*;
use super::weight_align::{compute_angular_endpoints_1plane, compute_angular_endpoints_2planes};
use super::{ASTCENC_FLG_MAP_RGBM, AstcencConfig, AstcencProfile};

isa_fn! {
/// `merge_endpoints`: merges two planes of endpoints into a single vector.
fn merge_endpoints(
    ep_plane1: &Endpoints,
    ep_plane2: &Endpoints,
    component_plane2: u32,
    result: &mut Endpoints,
) {
    let partition_count = ep_plane1.partition_count;
    let sep_mask = Vint4::lane_id().eq(Vint4::splat(component_plane2 as i32));

    result.partition_count = partition_count;
    result.endpt0[0] = Vfloat4::select(ep_plane1.endpt0[0], ep_plane2.endpt0[0], sep_mask);
    result.endpt1[0] = Vfloat4::select(ep_plane1.endpt1[0], ep_plane2.endpt1[0], sep_mask);
}
}

isa_fn! {
/// Unpacked endpoints of every partition of `scb`.
fn unpack_endpoints(
    decode_mode: AstcencProfile,
    scb: &SymbolicCompressedBlock,
) -> ([Vint4; BLOCK_MAX_PARTITIONS], [Vint4; BLOCK_MAX_PARTITIONS]) {
    let mut endpnt0 = [Vint4::zero(); BLOCK_MAX_PARTITIONS];
    let mut endpnt1 = [Vint4::zero(); BLOCK_MAX_PARTITIONS];
    for pa_idx in 0..scb.partition_count as usize {
        let (e0, e1, _, _) = unpack_color_endpoints(
            decode_mode,
            scb.color_formats[pa_idx],
            &scb.color_values[pa_idx],
        );
        endpnt0[pa_idx] = e0;
        endpnt1[pa_idx] = e1;
    }
    (endpnt0, endpnt1)
}
}

isa_fn! {
/// Per partition endpoint base and 1/64 step of a plane (`plane_mask` lanes zeroed).
fn plane_offsets(
    partition_count: usize,
    endpnt0: &[Vint4; BLOCK_MAX_PARTITIONS],
    endpnt1: &[Vint4; BLOCK_MAX_PARTITIONS],
    plane_mask: Vmask4,
) -> (
    [Vfloat4; BLOCK_MAX_PARTITIONS],
    [Vfloat4; BLOCK_MAX_PARTITIONS],
) {
    let mut endpnt0f = [Vfloat4::zero(); BLOCK_MAX_PARTITIONS];
    let mut offset = [Vfloat4::zero(); BLOCK_MAX_PARTITIONS];
    for pa_idx in 0..partition_count {
        let mut epd = endpnt1[pa_idx] - endpnt0[pa_idx];
        epd = Vint4::select(epd, Vint4::zero(), plane_mask);
        endpnt0f[pa_idx] = endpnt0[pa_idx].int_to_float();
        offset[pa_idx] = epd.int_to_float() * (1.0 / 64.0);
    }
    (endpnt0f, offset)
}
}

isa_fn! {
/// `realign_weights_undecimated_1partition`: moves each weight of an undecimated single
/// partition grid one quantization step up or down when that lowers the error. The squared
/// error is quadratic in the weight change d (`E(d) - E(0) = 2 d A + d^2 B`), so only the
/// linear and quadratic coefficients are computed, 8 texels at a time.
fn realign_weights_undecimated_1partition(
    decode_mode: AstcencProfile,
    bsd: &BlockSizeDescriptor,
    blk: &ImageBlock,
    scb: &mut SymbolicCompressedBlock,
) -> bool {
    let bm = bsd.get_block_mode(scb.block_mode as usize);
    let qat = &QUANT_AND_XFER_TABLES[bm.quant_mode as usize];

    let (endpnt0, endpnt1, _, _) =
        unpack_color_endpoints(decode_mode, scb.color_formats[0], &scb.color_values[0]);
    let color_base = endpnt0.int_to_float();
    let endpoint_delta = (endpnt1 - endpnt0).int_to_float() * (1.0 / 64.0);
    let error_weight = blk.channel_weight;
    let texel_count = bsd.texel_count as usize;

    let max_plane = usize::from(bm.is_dual_plane);
    let mut plane_mask = Vint4::lane_id().eq(Vint4::splat(i32::from(scb.plane2_component)));
    let mut is_adjusted = false;

    for pl_idx in 0..=max_plane {
        let color_offset = Vfloat4::select(endpoint_delta, Vfloat4::zero(), plane_mask);
        let base = pl_idx * WEIGHTS_PLANE2_OFFSET;

        let sp = |v: Vfloat4, k: usize| Vfloat8::splat(v.lane(k));
        let (base_r, base_g, base_b, base_a) = (
            sp(color_base, 0),
            sp(color_base, 1),
            sp(color_base, 2),
            sp(color_base, 3),
        );
        let (offset_r, offset_g, offset_b, offset_a) = (
            sp(color_offset, 0),
            sp(color_offset, 1),
            sp(color_offset, 2),
            sp(color_offset, 3),
        );
        let (weight_r, weight_g, weight_b, weight_a) = (
            sp(error_weight, 0),
            sp(error_weight, 1),
            sp(error_weight, 2),
            sp(error_weight, 3),
        );
        let error_quadratic = Vfloat8::splat((color_offset * color_offset).dot_s(error_weight));
        let mut is_adjusted_batch = Vmask8::splat(false);

        let mut texel = 0;
        while texel < texel_count {
            let uqw = Vint8::from_u8x8(scb.weights.get(base + texel..).unwrap_or_default());
            let uqw_base = uqw.int_to_float();
            let ld = |d: &[f32; BLOCK_DATA_LEN]| Vfloat8::loada(&d[texel..]);
            let residual_r = (base_r + offset_r * uqw_base) - ld(&blk.data_r);
            let mut linear_rb = (residual_r * offset_r) * weight_r;
            let residual_b = (base_b + offset_b * uqw_base) - ld(&blk.data_b);
            linear_rb += (residual_b * offset_b) * weight_b;
            let residual_g = (base_g + offset_g * uqw_base) - ld(&blk.data_g);
            let mut linear_ga = (residual_g * offset_g) * weight_g;
            let residual_a = (base_a + offset_a * uqw_base) - ld(&blk.data_a);
            linear_ga += (residual_a * offset_a) * weight_a;
            let error_linear = linear_rb + linear_ga;

            let mut uqw_down = Vint8::zero();
            let mut uqw_up = Vint8::zero();
            for k in 0..SIMD_WIDTH {
                let pn = i32::from(qat.prev_next_values[(uqw.m[k] as usize).min(64)]);
                uqw_down.m[k] = pn & 0xFF;
                uqw_up.m[k] = pn >> 8;
            }
            let diff_down = uqw_down.int_to_float() - uqw_base;
            let diff_up = uqw_up.int_to_float() - uqw_base;
            let two = Vfloat8::splat(2.0);
            let error_down =
                two * diff_down * error_linear + diff_down * diff_down * error_quadratic;
            let error_up = two * diff_up * error_linear + diff_up * diff_up * error_quadratic;

            let zero = Vfloat8::zero();
            let is_up = error_up.lt(zero)
                & error_up.lt(error_down)
                & uqw_base.lt(Vfloat8::splat(64.0));
            let is_down = error_down.lt(zero) & uqw_base.gt(zero);
            let mut result = Vint8::select(uqw, uqw_down, is_down);
            result = Vint8::select(result, uqw_up, is_up);
            let bytes = result.pack_bytes8();
            if let Some(w) = scb
                .weights
                .get_mut(base + texel..)
                .and_then(|w| w.first_chunk_mut::<8>())
            {
                *w = bytes;
            } else {
                for k in 0..SIMD_WIDTH {
                    if let Some(w) = scb.weights.get_mut(base + texel + k) {
                        *w = bytes[k];
                    }
                }
            }
            let is_active = (Vint8::splat(texel as i32) + Vint8::lane_id())
                .lt(Vint8::splat(texel_count as i32));
            is_adjusted_batch = is_adjusted_batch | ((is_up | is_down) & is_active);
            texel += SIMD_WIDTH;
        }
        is_adjusted |= is_adjusted_batch.any();
        plane_mask = !plane_mask;
    }
    is_adjusted
}
}

isa_fn! {
/// `realign_weights_undecimated_multipartition`: the per texel form of the analytic
/// realignment for undecimated multi partition grids (single plane).
fn realign_weights_undecimated_multipartition(
    decode_mode: AstcencProfile,
    bsd: &BlockSizeDescriptor,
    blk: &ImageBlock,
    scb: &mut SymbolicCompressedBlock,
) -> bool {
    let partition_count = scb.partition_count as usize;
    let pi = bsd.get_partition_info(partition_count as u32, u32::from(scb.partition_index));
    let bm = bsd.get_block_mode(scb.block_mode as usize);
    let qat = &QUANT_AND_XFER_TABLES[bm.quant_mode as usize];

    let mut endpnt0f = [Vfloat4::zero(); BLOCK_MAX_PARTITIONS];
    let mut offset = [Vfloat4::zero(); BLOCK_MAX_PARTITIONS];
    for pa_idx in 0..partition_count {
        let (e0, e1, _, _) = unpack_color_endpoints(
            decode_mode,
            scb.color_formats[pa_idx],
            &scb.color_values[pa_idx],
        );
        endpnt0f[pa_idx] = e0.int_to_float();
        offset[pa_idx] = (e1 - e0).int_to_float() * (1.0 / 64.0);
    }

    let mut is_adjusted = false;
    let error_weight = blk.channel_weight;
    for texel in 0..bsd.texel_count as usize {
        let uqw = i32::from(scb.weights[texel]);
        let prev_and_next = u32::from(qat.prev_next_values[uqw as usize]);
        let uqw_base = uqw as f32;
        let uqw_down = (prev_and_next & 0xFF) as f32;
        let uqw_up = ((prev_and_next >> 8) & 0xFF) as f32;
        let uqw_diff_down = uqw_down - uqw_base;
        let uqw_diff_up = uqw_up - uqw_base;

        let partition = pi.partition_of_texel[texel] as usize;
        let color_offset = offset[partition];
        let color_base = endpnt0f[partition];
        let color = color_base + color_offset * uqw_base;
        let color_diff = color - blk.texel(texel);
        let color_delta = color_offset;
        let error_linear = (color_diff * color_delta).dot_s(error_weight);
        let error_quadratic = (color_delta * color_delta).dot_s(error_weight);
        let error_down = 2.0 * uqw_diff_down * error_linear
            + uqw_diff_down * uqw_diff_down * error_quadratic;
        let error_up =
            2.0 * uqw_diff_up * error_linear + uqw_diff_up * uqw_diff_up * error_quadratic;

        if error_up < 0.0 && error_up < error_down && uqw < 64 {
            scb.weights[texel] = uqw_up as u8;
            is_adjusted = true;
        } else if error_down < 0.0 && uqw > 0 {
            scb.weights[texel] = uqw_down as u8;
            is_adjusted = true;
        }
    }
    is_adjusted
}
}

isa_fn! {
/// `infill_weight`: the bilinear infill of one texel from a decimated weight grid.
#[inline]
fn infill_weight(di: &DecimationInfo, weights: &[f32; BLOCK_MAX_WEIGHTS], texel: usize) -> f32 {
    let w = |j: usize| weights[di.texel_weights_tr[j][texel] as usize % BLOCK_MAX_WEIGHTS];
    let c = |j: usize| di.texel_weight_contribs_float_tr[j][texel];
    (w(0) * c(0) + w(1) * c(1)) + (w(2) * c(2) + w(3) * c(3))
}
}

isa_fn! {
/// `realign_weights_decimated`: tries moving each weight of a decimated grid one
/// quantization step up or down, using the analytic error deltas and an infilled weight set
/// that is only rebuilt around a changed weight.
fn realign_weights_decimated(
    decode_mode: AstcencProfile,
    bsd: &BlockSizeDescriptor,
    blk: &ImageBlock,
    scb: &mut SymbolicCompressedBlock,
) -> bool {
    let partition_count = scb.partition_count as usize;
    let pi = bsd.get_partition_info(partition_count as u32, u32::from(scb.partition_index));

    let bm = bsd.get_block_mode(scb.block_mode as usize);
    let qat = &QUANT_AND_XFER_TABLES[bm.quant_mode as usize];

    let di = bsd.get_decimation_info(bm.decimation_mode as usize);
    let weight_count = di.weight_count as usize;

    let max_plane = usize::from(bm.is_dual_plane);
    let mut plane_mask = Vint4::lane_id().eq(Vint4::splat(i32::from(scb.plane2_component)));

    // Decode the color endpoints
    let (endpnt0, endpnt1) = unpack_endpoints(decode_mode, scb);

    let mut adjustments = false;
    let error_weight = blk.channel_weight;

    // For each plane and partition ...
    for pl_idx in 0..=max_plane {
        let (endpnt0f, offset) = plane_offsets(partition_count, &endpnt0, &endpnt1, plane_mask);
        let base = pl_idx * WEIGHTS_PLANE2_OFFSET;

        // Create an unquantized weight grid for this decimation level (in groups of 8)
        let mut uq_weightsf = [0.0f32; BLOCK_MAX_WEIGHTS];
        let mut we_idx = 0;
        while we_idx < weight_count {
            for k in we_idx..(we_idx + SIMD_WIDTH).min(BLOCK_MAX_WEIGHTS) {
                uq_weightsf[k] = f32::from(scb.weights.get(base + k).copied().unwrap_or(0));
            }
            we_idx += SIMD_WIDTH;
        }

        let mut infilled_weights = [0.0f32; BLOCK_MAX_TEXELS];
        for texel in 0..bsd.texel_count as usize {
            infilled_weights[texel] = infill_weight(di, &uq_weightsf, texel);
        }

        // For each weight compute the error delta for the previous and next values
        for we_idx in 0..weight_count {
            let uqw = i32::from(scb.weights[base + we_idx]);
            let prev_and_next = u32::from(qat.prev_next_values[uqw as usize]);

            let uqw_base = uq_weightsf[we_idx];
            let uqw_down = (prev_and_next & 0xFF) as f32;
            let uqw_up = ((prev_and_next >> 8) & 0xFF) as f32;

            let uqw_diff_down = uqw_down - uqw_base;
            let uqw_diff_up = uqw_up - uqw_base;

            let mut error_linearv = Vfloat4::zero();
            let mut error_quadraticv = Vfloat4::zero();

            let texels_to_evaluate = di.weight_texel_count[we_idx] as usize;
            for te_idx in 0..texels_to_evaluate {
                let texel = di.weight_texels_tr[te_idx][we_idx] as usize;
                let tw_base = di.texel_contrib_for_weight[te_idx][we_idx];
                let weight_base = infilled_weights[texel];

                let partition = pi.partition_of_texel[texel] as usize;
                let color_offset = offset[partition];
                let color_base = endpnt0f[partition];

                let color = color_base + color_offset * weight_base;
                let color_diff = color - blk.texel(texel);
                let color_delta = color_offset * tw_base;

                error_linearv += color_diff * color_delta;
                error_quadraticv += color_delta * color_delta;
            }

            let error_linear = (error_linearv * error_weight).hadd_s();
            let error_quadratic = (error_quadraticv * error_weight).hadd_s();
            let error_down = 2.0 * uqw_diff_down * error_linear
                + uqw_diff_down * uqw_diff_down * error_quadratic;
            let error_up =
                2.0 * uqw_diff_up * error_linear + uqw_diff_up * uqw_diff_up * error_quadratic;

            // Check if the prev or next error is better, and if so use it
            let mut weight_adjusted = false;
            if error_up < 0.0 && error_up < error_down && uqw < 64 {
                uq_weightsf[we_idx] = uqw_up;
                scb.weights[base + we_idx] = uqw_up as u8;
                weight_adjusted = true;
            } else if error_down < 0.0 && uqw > 0 {
                uq_weightsf[we_idx] = uqw_down;
                scb.weights[base + we_idx] = uqw_down as u8;
                weight_adjusted = true;
            }
            if weight_adjusted {
                adjustments = true;
                for te_idx in 0..texels_to_evaluate {
                    let texel = di.weight_texels_tr[te_idx][we_idx] as usize;
                    infilled_weights[texel] = infill_weight(di, &uq_weightsf, texel);
                }
            }
        }

        // Prepare iteration for plane 2
        plane_mask = !plane_mask;
    }

    adjustments
}
}

/// `compute_symbolic_block_difference_*` selector.
type DifferenceFn =
    fn(&AstcencConfig, &BlockSizeDescriptor, &SymbolicCompressedBlock, &ImageBlock) -> f32;

isa_fn! {
/// Smallest weight per lane that exceeds the color range (endpoints use the 0-65535 scale).
fn min_weight_cutoff(ep: &Endpoints, i: usize, min_ep: Vfloat4) -> Vfloat4 {
    let e = (Vfloat4::splat(65535.0) - ep.endpt0[i]) / (ep.endpt1[i] - ep.endpt0[i]);
    let use_ep = e.gt(Vfloat4::splat(0.5)) & e.lt(min_ep);
    Vfloat4::select(min_ep, e, use_ep)
}
}

/// Outcome of one refinement trial: whether to continue refining, stop this candidate,
/// or stop the candidate search.
enum Trial {
    Continue,
    NextCandidate,
    Done,
}

isa_fn! {
/// Shared pre / post realignment test of the candidate loops.
fn trial_test(
    errorval: f32,
    threshold_base: f32,
    iters_remaining: u32,
    workscb: &mut SymbolicCompressedBlock,
    scb: &mut SymbolicCompressedBlock,
    best_errorval_in_mode: &mut f32,
    best_errorval_in_scb: &mut f32,
    tune_errorval_threshold: f32,
) -> Trial {
    let mut errorval = errorval;
    if errorval == -ERROR_CALC_DEFAULT {
        errorval = -errorval;
        workscb.block_type = SYM_BTYPE_ERROR;
    }

    *best_errorval_in_mode = mathlib::min(errorval, *best_errorval_in_mode);

    // Average refinement improvement is 3.5% per iteration, so skip blocks that are
    // unlikely to catch up with the best block we have already
    let threshold = (0.045 * iters_remaining as f32) + threshold_base;
    if errorval > (threshold * *best_errorval_in_scb) {
        return Trial::NextCandidate;
    }

    if errorval < *best_errorval_in_scb {
        *best_errorval_in_scb = errorval;
        workscb.errorval = errorval;
        *scb = *workscb;

        if errorval < tune_errorval_threshold {
            // Skip remaining candidates - this is "good enough"
            return Trial::Done;
        }
    }
    Trial::Continue
}
}

isa_fn! {
/// `compress_symbolic_block_for_partition_1plane`: compresses a block using a chosen
/// partitioning and 1 plane of weights.
fn compress_symbolic_block_for_partition_1plane(
    config: &AstcencConfig,
    bsd: &BlockSizeDescriptor,
    blk: &ImageBlock,
    only_always: bool,
    tune_errorval_threshold: f32,
    partition_count: u32,
    partition_index: u32,
    scb: &mut SymbolicCompressedBlock,
    tmpbuf: &mut CompressionWorkingBuffers,
    quant_limit: i32,
) -> f32 {
    let max_weight_quant = (QUANT_32 as i32).min(quant_limit);

    let mut compute_difference: DifferenceFn = compute_symbolic_block_difference_1plane;
    if partition_count == 1 && config.flags & ASTCENC_FLG_MAP_RGBM == 0 {
        compute_difference = compute_symbolic_block_difference_1plane_1partition;
    }

    let pi = bsd.get_partition_info(partition_count, partition_index);

    // Compute ideal weights and endpoint colors, with no quantization or decimation
    let mut rgb_metrics = [PartitionMetrics::default(); BLOCK_MAX_PARTITIONS];
    let rgb_metrics_valid =
        compute_ideal_colors_and_weights_1plane(blk, pi, &mut tmpbuf.ei1, &mut rgb_metrics);
    let endpoint_rgb_metrics = if rgb_metrics_valid {
        Some(&rgb_metrics)
    } else {
        None
    };
    let ei = &tmpbuf.ei1;

    // For each decimation mode, compute an ideal set of weights with no quantization
    let max_decimation_modes = if only_always {
        bsd.decimation_mode_count_always
    } else {
        bsd.decimation_mode_count_selected
    } as usize;
    for i in 0..max_decimation_modes {
        let dm = bsd.get_decimation_mode(i);
        if !dm.is_ref_1plane(max_weight_quant as u32) {
            continue;
        }

        let di = bsd.get_decimation_info(i);
        compute_ideal_weights_for_decimation(
            ei,
            di,
            &mut tmpbuf.dec_weights_ideal[i * BLOCK_MAX_WEIGHTS..],
        );
    }

    // Compute maximum colors for the endpoints and ideal weights, then for each endpoint and
    // ideal weight pair, compute the smallest weight that will result in a color value > 1
    let mut min_ep = Vfloat4::splat(10.0);
    for i in 0..partition_count as usize {
        min_ep = min_weight_cutoff(&ei.ep, i, min_ep);
    }

    let min_wt_cutoff = min_ep.hmin_s();

    // For each mode, use the angular method to compute a shift
    compute_angular_endpoints_1plane(
        only_always,
        bsd,
        &tmpbuf.dec_weights_ideal,
        max_weight_quant as u32,
        &mut tmpbuf.angular,
    );

    // For each mode (which specifies a decimation and a quantization):
    //     * Compute number of bits needed for the quantized weights
    //     * Generate an optimized set of quantized weights
    //     * Compute quantization errors for the mode
    const FREE_BITS_FOR_PARTITION_COUNT: [i32; 4] = [
        115 - 4,
        111 - 4 - PARTITION_INDEX_BITS as i32,
        108 - 4 - PARTITION_INDEX_BITS as i32,
        105 - 4 - PARTITION_INDEX_BITS as i32,
    ];

    let max_block_modes = if only_always {
        bsd.block_mode_count_1plane_always
    } else {
        bsd.block_mode_count_1plane_selected
    } as usize;
    // Reused for every mode; each mode writes the entries it reads (rounded up to SIMD width)
    let mut dec_weights_uquantf = [0.0f32; BLOCK_MAX_WEIGHTS];
    for i in 0..max_block_modes {
        let bm = bsd.block_modes[i];

        if i32::from(bm.quant_mode) > max_weight_quant {
            tmpbuf.qwt_errors[i] = 1e38;
            continue;
        }

        let bitcount =
            FREE_BITS_FOR_PARTITION_COUNT[partition_count as usize - 1] - i32::from(bm.weight_bits);
        if bitcount <= 0 {
            tmpbuf.qwt_errors[i] = 1e38;
            continue;
        }

        if tmpbuf.angular.weight_high_value1[i] > 1.02 * min_wt_cutoff {
            tmpbuf.angular.weight_high_value1[i] = 1.0;
        }

        let decimation_mode = bm.decimation_mode as usize;
        let di = bsd.get_decimation_info(decimation_mode);

        tmpbuf.qwt_bitcounts[i] = bitcount as i8;

        // Generate the optimized set of weights for the weight mode
        compute_quantized_weights_for_decimation(
            di,
            tmpbuf.angular.weight_low_value1[i],
            tmpbuf.angular.weight_high_value1[i],
            &tmpbuf.dec_weights_ideal[BLOCK_MAX_WEIGHTS * decimation_mode..],
            &mut dec_weights_uquantf,
            &mut tmpbuf.dec_weights_uquant[BLOCK_MAX_WEIGHTS * i..],
            bm.get_weight_quant_mode(),
        );

        // Compute weight quantization errors for the block mode
        tmpbuf.qwt_errors[i] = compute_error_of_weight_set_1plane(ei, di, &dec_weights_uquantf);
    }

    // Decide the optimal combination of color endpoint encodings and weight encodings
    let mut partition_format_specifiers = [[0u8; BLOCK_MAX_PARTITIONS]; TUNE_MAX_TRIAL_CANDIDATES];
    let mut block_mode_index = [0i32; TUNE_MAX_TRIAL_CANDIDATES];

    let mut color_quant_level = [0u32; TUNE_MAX_TRIAL_CANDIDATES];
    let mut color_quant_level_mod = [0u32; TUNE_MAX_TRIAL_CANDIDATES];

    let candidate_count = compute_ideal_endpoint_formats(
        pi,
        blk,
        &ei.ep,
        endpoint_rgb_metrics,
        &tmpbuf.qwt_bitcounts,
        &tmpbuf.qwt_errors,
        config.tune_candidate_limit,
        0,
        max_block_modes as u32,
        &mut partition_format_specifiers,
        &mut block_mode_index,
        &mut color_quant_level,
        &mut color_quant_level_mod,
        &mut tmpbuf.formats,
    );

    // Iterate over the N believed-to-be-best modes to find out which one is actually best
    let mut best_errorval_in_mode = ERROR_CALC_DEFAULT;
    let mut best_errorval_in_scb = scb.errorval;

    'candidates: for i in 0..candidate_count as usize {
        let bm_packed_index = block_mode_index[i] as usize;
        let qw_bm = bsd.block_modes[bm_packed_index];

        let decimation_mode = qw_bm.decimation_mode as usize;
        let di = bsd.get_decimation_info(decimation_mode);

        // Recompute the ideal color endpoints before storing them
        let mut rgbs_colors = [Vfloat4::zero(); BLOCK_MAX_PARTITIONS];
        let mut rgbo_colors = [Vfloat4::zero(); BLOCK_MAX_PARTITIONS];

        let mut workscb = SymbolicCompressedBlock::default();
        let mut workep = ei.ep;

        // Preserve the quantized SIMD padding for undecimated weight refinement
        let src = BLOCK_MAX_WEIGHTS * bm_packed_index;
        let weight_count_simd = round_up_to_simd_multiple_vla(di.weight_count as usize);
        for j in 0..weight_count_simd.min(BLOCK_MAX_WEIGHTS) {
            workscb.weights[j] = tmpbuf.dec_weights_uquant[src + j];
        }

        for l in 0..config.tune_refinement_limit {
            recompute_ideal_colors_1plane(
                blk,
                pi,
                di,
                &workscb.weights,
                &partition_format_specifiers[i],
                &mut workep,
                &mut rgbs_colors,
                &mut rgbo_colors,
            );

            // Delta choices can change with precision, so use the base formats to enable the
            // mod trial
            let mut all_same = color_quant_level[i] != color_quant_level_mod[i];
            for j in 0..partition_count as usize {
                workscb.color_formats[j] = pack_color_endpoints(
                    workep.endpt0[j],
                    workep.endpt1[j],
                    rgbs_colors[j],
                    rgbo_colors[j],
                    partition_format_specifiers[i][j],
                    &mut workscb.color_values[j],
                    color_quant_level[i],
                );

                all_same = all_same
                    && partition_format_specifiers[i][j] == partition_format_specifiers[i][0];
            }

            // If all the color endpoint modes are the same, we get a few more bits to store
            // colors; requantize all the colors and see if the endpoint modes remain the same.
            workscb.color_formats_matched = 0;
            if partition_count >= 2 && all_same {
                let mut colorvals = [[0u8; 8]; BLOCK_MAX_PARTITIONS];
                let mut color_formats_mod = [0u8; BLOCK_MAX_PARTITIONS];
                let mut all_same_mod = true;
                for j in 0..partition_count as usize {
                    color_formats_mod[j] = pack_color_endpoints(
                        workep.endpt0[j],
                        workep.endpt1[j],
                        rgbs_colors[j],
                        rgbo_colors[j],
                        partition_format_specifiers[i][j],
                        &mut colorvals[j],
                        color_quant_level_mod[i],
                    );

                    // Early out as soon as it's no longer possible to use mod
                    if color_formats_mod[j] != color_formats_mod[0] {
                        all_same_mod = false;
                        break;
                    }
                }

                if all_same_mod {
                    workscb.color_formats_matched = 1;
                    workscb.color_values = colorvals;
                    workscb.color_formats = color_formats_mod;
                }
            }

            // Store header fields
            workscb.partition_count = partition_count as u8;
            workscb.partition_index = partition_index as u16;
            workscb.plane2_component = -1;
            workscb.quant_mode = if workscb.color_formats_matched != 0 {
                color_quant_level_mod[i]
            } else {
                color_quant_level[i]
            };
            workscb.block_mode = qw_bm.mode_index;
            workscb.block_type = SYM_BTYPE_NONCONST;

            // Pre-realign test
            if l == 0 {
                let errorval = compute_difference(config, bsd, &workscb, blk);
                match trial_test(
                    errorval,
                    1.08,
                    config.tune_refinement_limit - l,
                    &mut workscb,
                    scb,
                    &mut best_errorval_in_mode,
                    &mut best_errorval_in_scb,
                    tune_errorval_threshold,
                ) {
                    Trial::Continue => {}
                    Trial::NextCandidate => continue 'candidates,
                    Trial::Done => break 'candidates,
                }
            }

            // Perform a final pass over the weights to try to improve them.
            let adjustments = if di.weight_count != bsd.texel_count {
                realign_weights_decimated(config.profile, bsd, blk, &mut workscb)
            } else if partition_count == 1 {
                realign_weights_undecimated_1partition(config.profile, bsd, blk, &mut workscb)
            } else {
                realign_weights_undecimated_multipartition(config.profile, bsd, blk, &mut workscb)
            };

            // The unchanged first-pass candidate was already scored before weight refinement
            if l == 0 && !adjustments {
                break;
            }

            // Post-realign test
            let errorval = compute_difference(config, bsd, &workscb, blk);
            match trial_test(
                errorval,
                1.0,
                config.tune_refinement_limit - 1 - l,
                &mut workscb,
                scb,
                &mut best_errorval_in_mode,
                &mut best_errorval_in_scb,
                tune_errorval_threshold,
            ) {
                Trial::Continue => {}
                Trial::NextCandidate => continue 'candidates,
                Trial::Done => break 'candidates,
            }

            if !adjustments {
                break;
            }
        }
    }

    best_errorval_in_mode
}
}

isa_fn! {
/// `compress_symbolic_block_for_partition_2planes`: compresses a block using a single
/// partition and 2 planes of weights.
fn compress_symbolic_block_for_partition_2planes(
    config: &AstcencConfig,
    bsd: &BlockSizeDescriptor,
    blk: &ImageBlock,
    tune_errorval_threshold: f32,
    plane2_component: u32,
    scb: &mut SymbolicCompressedBlock,
    tmpbuf: &mut CompressionWorkingBuffers,
    quant_limit: i32,
) -> f32 {
    let max_weight_quant = (QUANT_32 as i32).min(quant_limit);

    // Compute ideal weights and endpoint colors, with no quantization or decimation
    compute_ideal_colors_and_weights_2planes(
        bsd,
        blk,
        plane2_component,
        &mut tmpbuf.ei1,
        &mut tmpbuf.ei2,
    );
    let ei1 = &tmpbuf.ei1;
    let ei2 = &tmpbuf.ei2;

    // For each decimation mode, compute an ideal set of weights with no quantization
    for i in 0..bsd.decimation_mode_count_selected as usize {
        let dm = bsd.get_decimation_mode(i);
        if !dm.is_ref_2plane(max_weight_quant as u32) {
            continue;
        }

        let di = bsd.get_decimation_info(i);

        compute_ideal_weights_for_decimation(
            ei1,
            di,
            &mut tmpbuf.dec_weights_ideal[i * BLOCK_MAX_WEIGHTS..],
        );

        compute_ideal_weights_for_decimation(
            ei2,
            di,
            &mut tmpbuf.dec_weights_ideal[i * BLOCK_MAX_WEIGHTS + WEIGHTS_PLANE2_OFFSET..],
        );
    }

    // Compute maximum colors for the endpoints and ideal weights, then for each endpoint and
    // ideal weight pair, compute the smallest weight that will result in a color value > 1
    let mut min_ep1 = min_weight_cutoff(&ei1.ep, 0, Vfloat4::splat(10.0));
    let min_ep2 = min_weight_cutoff(&ei2.ep, 0, Vfloat4::splat(10.0));

    let err_max = Vfloat4::splat(ERROR_CALC_DEFAULT);
    let err_mask = Vint4::lane_id().eq(Vint4::splat(plane2_component as i32));

    // Set the plane2 component to max error in ep1
    min_ep1 = Vfloat4::select(min_ep1, err_max, err_mask);

    let min_wt_cutoff1 = min_ep1.hmin_s();

    // Set the minwt2 to the plane2 component min in ep2
    let min_wt_cutoff2 = Vfloat4::select(err_max, min_ep2, err_mask).hmin_s();

    compute_angular_endpoints_2planes(
        bsd,
        &tmpbuf.dec_weights_ideal,
        max_weight_quant as u32,
        &mut tmpbuf.angular,
    );

    // For each mode (which specifies a decimation and a quantization):
    //     * Compute number of bits needed for the quantized weights
    //     * Generate an optimized set of quantized weights
    //     * Compute quantization errors for the mode
    let start_2plane = bsd.block_mode_count_1plane_selected as usize;
    let end_2plane = bsd.block_mode_count_1plane_2plane_selected as usize;

    // Reused for every mode; each mode writes the entries it reads (rounded up to SIMD width)
    let mut dec_weights_uquantf = [0.0f32; BLOCK_MAX_WEIGHTS];
    for i in start_2plane..end_2plane {
        let bm = bsd.block_modes[i];

        if i32::from(bm.quant_mode) > max_weight_quant {
            tmpbuf.qwt_errors[i] = 1e38;
            continue;
        }

        tmpbuf.qwt_bitcounts[i] = (109 - i32::from(bm.weight_bits)) as i8;

        if tmpbuf.angular.weight_high_value1[i] > 1.02 * min_wt_cutoff1 {
            tmpbuf.angular.weight_high_value1[i] = 1.0;
        }

        if tmpbuf.angular.weight_high_value2[i] > 1.02 * min_wt_cutoff2 {
            tmpbuf.angular.weight_high_value2[i] = 1.0;
        }

        let decimation_mode = bm.decimation_mode as usize;
        let di = bsd.get_decimation_info(decimation_mode);

        // Generate the optimized set of weights for the mode
        let dec_weights_ideal = &tmpbuf.dec_weights_ideal;
        let dec_weights_uquant = &mut tmpbuf.dec_weights_uquant;
        let AngularBuffers {
            weight_low_value1,
            weight_high_value1,
            weight_low_value2,
            weight_high_value2,
            ..
        } = &tmpbuf.angular;
        let ideal = BLOCK_MAX_WEIGHTS * decimation_mode;
        let uquant = BLOCK_MAX_WEIGHTS * i;
        compute_quantized_weights_for_decimation(
            di,
            weight_low_value1[i],
            weight_high_value1[i],
            &dec_weights_ideal[ideal..],
            &mut dec_weights_uquantf,
            &mut dec_weights_uquant[uquant..],
            bm.get_weight_quant_mode(),
        );

        compute_quantized_weights_for_decimation(
            di,
            weight_low_value2[i],
            weight_high_value2[i],
            &dec_weights_ideal[ideal + WEIGHTS_PLANE2_OFFSET..],
            &mut dec_weights_uquantf[WEIGHTS_PLANE2_OFFSET..],
            &mut dec_weights_uquant[uquant + WEIGHTS_PLANE2_OFFSET..],
            bm.get_weight_quant_mode(),
        );

        // Compute weight quantization errors for the block mode
        tmpbuf.qwt_errors[i] = compute_error_of_weight_set_2planes(
            ei1,
            ei2,
            di,
            &dec_weights_uquantf,
            &dec_weights_uquantf[WEIGHTS_PLANE2_OFFSET..],
        );
    }

    // Decide the optimal combination of color endpoint encodings and weight encodings
    let mut partition_format_specifiers = [[0u8; BLOCK_MAX_PARTITIONS]; TUNE_MAX_TRIAL_CANDIDATES];
    let mut block_mode_index = [0i32; TUNE_MAX_TRIAL_CANDIDATES];

    let mut color_quant_level = [0u32; TUNE_MAX_TRIAL_CANDIDATES];
    let mut color_quant_level_mod = [0u32; TUNE_MAX_TRIAL_CANDIDATES];

    let mut epm = Endpoints::default();
    merge_endpoints(&ei1.ep, &ei2.ep, plane2_component, &mut epm);

    let pi = bsd.get_partition_info(1, 0);
    let candidate_count = compute_ideal_endpoint_formats(
        pi,
        blk,
        &epm,
        None,
        &tmpbuf.qwt_bitcounts,
        &tmpbuf.qwt_errors,
        config.tune_candidate_limit,
        bsd.block_mode_count_1plane_selected,
        bsd.block_mode_count_1plane_2plane_selected,
        &mut partition_format_specifiers,
        &mut block_mode_index,
        &mut color_quant_level,
        &mut color_quant_level_mod,
        &mut tmpbuf.formats,
    );

    // Iterate over the N believed-to-be-best modes to find out which one is actually best
    let mut best_errorval_in_mode = ERROR_CALC_DEFAULT;
    let mut best_errorval_in_scb = scb.errorval;

    'candidates: for i in 0..candidate_count as usize {
        let bm_packed_index = block_mode_index[i] as usize;
        let qw_bm = bsd.block_modes[bm_packed_index];

        let decimation_mode = qw_bm.decimation_mode as usize;
        let di = bsd.get_decimation_info(decimation_mode);

        let mut rgbs_color = Vfloat4::zero();
        let mut rgbo_color = Vfloat4::zero();

        let mut workscb = SymbolicCompressedBlock::default();
        let mut workep = epm;

        // Preserve the quantized SIMD padding for undecimated weight refinement
        let src = BLOCK_MAX_WEIGHTS * bm_packed_index;
        let weight_count_simd = round_up_to_simd_multiple_vla(di.weight_count as usize);
        for j in 0..weight_count_simd.min(WEIGHTS_PLANE2_OFFSET) {
            workscb.weights[j] = tmpbuf.dec_weights_uquant[src + j];
            workscb.weights[j + WEIGHTS_PLANE2_OFFSET] =
                tmpbuf.dec_weights_uquant[src + WEIGHTS_PLANE2_OFFSET + j];
        }

        for l in 0..config.tune_refinement_limit {
            let (w1, w2) = workscb.weights.split_at(WEIGHTS_PLANE2_OFFSET);
            recompute_ideal_colors_2planes(
                blk,
                bsd,
                di,
                w1,
                w2,
                partition_format_specifiers[i][0],
                &mut workep,
                &mut rgbs_color,
                &mut rgbo_color,
                plane2_component as i32,
            );

            // Quantize the chosen color
            workscb.color_formats[0] = pack_color_endpoints(
                workep.endpt0[0],
                workep.endpt1[0],
                rgbs_color,
                rgbo_color,
                partition_format_specifiers[i][0],
                &mut workscb.color_values[0],
                color_quant_level[i],
            );

            // Store header fields
            workscb.partition_count = 1;
            workscb.partition_index = 0;
            workscb.quant_mode = color_quant_level[i];
            workscb.color_formats_matched = 0;
            workscb.block_mode = qw_bm.mode_index;
            workscb.plane2_component = plane2_component as i8;
            workscb.block_type = SYM_BTYPE_NONCONST;

            // Pre-realign test
            if l == 0 {
                let errorval = compute_symbolic_block_difference_2plane(config, bsd, &workscb, blk);
                match trial_test(
                    errorval,
                    1.08,
                    config.tune_refinement_limit - l,
                    &mut workscb,
                    scb,
                    &mut best_errorval_in_mode,
                    &mut best_errorval_in_scb,
                    tune_errorval_threshold,
                ) {
                    Trial::Continue => {}
                    Trial::NextCandidate => continue 'candidates,
                    Trial::Done => break 'candidates,
                }
            }

            // Perform a final pass over the weights to try to improve them.
            let adjustments = if di.weight_count != bsd.texel_count {
                realign_weights_decimated(config.profile, bsd, blk, &mut workscb)
            } else {
                realign_weights_undecimated_1partition(config.profile, bsd, blk, &mut workscb)
            };

            // The unchanged first-pass candidate was already scored before weight refinement
            if l == 0 && !adjustments {
                break;
            }

            // Post-realign test
            let errorval = compute_symbolic_block_difference_2plane(config, bsd, &workscb, blk);
            match trial_test(
                errorval,
                1.0,
                config.tune_refinement_limit - 1 - l,
                &mut workscb,
                scb,
                &mut best_errorval_in_mode,
                &mut best_errorval_in_scb,
                tune_errorval_threshold,
            ) {
                Trial::Continue => {}
                Trial::NextCandidate => continue 'candidates,
                Trial::Done => break 'candidates,
            }

            if !adjustments {
                break;
            }
        }
    }

    best_errorval_in_mode
}
}

isa_fn! {
/// `prepare_block_statistics`: the lowest absolute channel correlation of the block;
/// `covariance` receives the (rr, gg, bb, aa) variances and the (rg, rb, ra, gb), (ga, ba)
/// covariances.
fn prepare_block_statistics(
    texels_per_block: usize,
    blk: &ImageBlock,
    covariance: &mut [Vfloat4; 3],
) -> f32 {
    // Compute covariance matrix, as a collection of 10 scalars that form the upper-triangular
    // row of the matrix. The matrix is symmetric, so this is all we need for this use case.
    let (mut rs, mut gs, mut bs, mut as_) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut rr_var, mut gg_var, mut bb_var, mut aa_var) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut rg_cov, mut rb_cov, mut ra_cov) = (0.0f32, 0.0f32, 0.0f32);
    let (mut gb_cov, mut ga_cov, mut ba_cov) = (0.0f32, 0.0f32, 0.0f32);
    let mut weight_sum = 0.0f32;

    for i in 0..texels_per_block {
        let weight = blk.channel_weight.hadd_s() / 4.0;
        weight_sum += weight;

        let r = blk.data_r[i];
        let g = blk.data_g[i];
        let b = blk.data_b[i];
        let a = blk.data_a[i];

        let rw = r * weight;
        rs += rw;
        rr_var += r * rw;
        rg_cov += g * rw;
        rb_cov += b * rw;
        ra_cov += a * rw;

        let gw = g * weight;
        gs += gw;
        gg_var += g * gw;
        gb_cov += b * gw;
        ga_cov += a * gw;

        let bw = b * weight;
        bs += bw;
        bb_var += b * bw;
        ba_cov += a * bw;

        let aw = a * weight;
        as_ += aw;
        aa_var += a * aw;
    }

    let rpt = 1.0 / mathlib::max(weight_sum, 1e-7);

    rr_var -= rs * (rs * rpt);
    rg_cov -= gs * (rs * rpt);
    rb_cov -= bs * (rs * rpt);
    ra_cov -= as_ * (rs * rpt);

    gg_var -= gs * (gs * rpt);
    gb_cov -= bs * (gs * rpt);
    ga_cov -= as_ * (gs * rpt);

    bb_var -= bs * (bs * rpt);
    ba_cov -= as_ * (bs * rpt);

    aa_var -= as_ * (as_ * rpt);

    // Retain the covariance magnitudes before converting to correlation coefficients
    covariance[0] = Vfloat4::new(rr_var, gg_var, bb_var, aa_var).max(Vfloat4::zero());
    covariance[1] = Vfloat4::new(rg_cov, rb_cov, ra_cov, gb_cov);
    covariance[2] = Vfloat4::new(ga_cov, ba_cov, 0.0, 0.0);

    // These will give a NaN if a channel is constant - these are fixed up in the next step
    rg_cov *= mathlib::rsqrt(rr_var * gg_var);
    rb_cov *= mathlib::rsqrt(rr_var * bb_var);
    ra_cov *= mathlib::rsqrt(rr_var * aa_var);
    gb_cov *= mathlib::rsqrt(gg_var * bb_var);
    ga_cov *= mathlib::rsqrt(gg_var * aa_var);
    ba_cov *= mathlib::rsqrt(bb_var * aa_var);

    let fix = |v: f32| if v.is_nan() { 1.0 } else { v };
    let (rg_cov, rb_cov, ra_cov) = (fix(rg_cov), fix(rb_cov), fix(ra_cov));
    let (gb_cov, ga_cov, ba_cov) = (fix(gb_cov), fix(ga_cov), fix(ba_cov));

    let mut lowest_correlation = mathlib::min(rg_cov.abs(), rb_cov.abs());
    lowest_correlation = mathlib::min(lowest_correlation, ra_cov.abs());
    lowest_correlation = mathlib::min(lowest_correlation, gb_cov.abs());
    lowest_correlation = mathlib::min(lowest_correlation, ga_cov.abs());
    lowest_correlation = mathlib::min(lowest_correlation, ba_cov.abs());

    lowest_correlation
}
}

isa_fn! {
/// `estimate_3comp_line_error`: per lane, the error of the best line fit of a 3x3 covariance
/// matrix (power iteration from three seeds, Rayleigh quotient).
fn estimate_3comp_line_error(
    diag0: Vfloat4,
    diag1: Vfloat4,
    diag2: Vfloat4,
    cov01: Vfloat4,
    cov02: Vfloat4,
    cov12: Vfloat4,
) -> Vfloat4 {
    // Scale each matrix independently to keep power iteration in a useful numeric range
    let cov_scale = cov01.abs().max(cov02.abs().max(cov12.abs()));
    let scale = diag0
        .max(diag1)
        .max(diag2.max(cov_scale.max(Vfloat4::splat(1e-20))));
    let rscale = Vfloat4::splat(1.0) / scale;
    let diag0 = diag0 * rscale;
    let diag1 = diag1 * rscale;
    let diag2 = diag2 * rscale;
    let cov01 = cov01 * rscale;
    let cov02 = cov02 * rscale;
    let cov12 = cov12 * rscale;

    let seeds0 = [diag0, cov01, cov02];
    let seeds1 = [cov01, diag1, cov12];
    let seeds2 = [cov02, cov12, diag2];
    let mut best_variance = Vfloat4::zero();
    for i in 0..3 {
        let (mut x, mut y, mut z) = (seeds0[i], seeds1[i], seeds2[i]);
        for _ in 0..3 {
            let next_x = (diag0 * x + cov01 * y) + cov02 * z;
            let next_y = (cov01 * x + diag1 * y) + cov12 * z;
            let next_z = (cov02 * x + cov12 * y) + diag2 * z;
            x = next_x;
            y = next_y;
            z = next_z;
        }
        // The Rayleigh quotient measures variance captured by this direction
        let cx = (diag0 * x + cov01 * y) + cov02 * z;
        let cy = (cov01 * x + diag1 * y) + cov12 * z;
        let cz = (cov02 * x + cov12 * y) + diag2 * z;
        let norm = (x * x + y * y) + z * z;
        let variance = ((x * cx + y * cy) + z * cz) / norm.max(Vfloat4::splat(1e-20));
        best_variance = best_variance.max(variance);
    }
    ((diag0 + diag1) + diag2 - best_variance).max(Vfloat4::zero()) * scale
}
}

isa_fn! {
/// `select_dual_plane_components`: orders `plane_components` by the predicted error of
/// separating each component into the second weight plane and returns how many are worth
/// a trial (constant components and ones gaining less than `min_gain_factor` of the best
/// predicted gain are dropped).
fn select_dual_plane_components(
    blk: &ImageBlock,
    covariance: &[Vfloat4; 3],
    ideal_ep: &Endpoints,
    min_gain_factor: f32,
    plane_components: &mut [usize; BLOCK_MAX_COMPONENTS],
) -> usize {
    let diag = covariance[0] * blk.channel_weight;
    let cov0 = covariance[1];
    let cov1 = covariance[2];
    let channel_scale = blk.channel_weight.sqrt();
    let (r, g, b, a) = (diag.lane(0), diag.lane(1), diag.lane(2), diag.lane(3));
    let (sr, sg, sb, sa) = (
        channel_scale.lane(0),
        channel_scale.lane(1),
        channel_scale.lane(2),
        channel_scale.lane(3),
    );

    // Lane C holds the covariance of the channels remaining when C uses the second plane
    let diag0 = Vfloat4::new(g, r, r, r);
    let diag1 = Vfloat4::new(b, b, g, g);
    let cov01 = cov0.swz4(3, 1, 0, 0) * Vfloat4::new(sg * sb, sr * sb, sr * sg, sr * sg);
    let errors = if blk.is_constant_channel(3) {
        // With constant alpha, the remaining two-channel line error has a closed form
        let diff = diag0 - diag1;
        let root = (diff * diff + Vfloat4::splat(4.0) * cov01 * cov01).sqrt();
        (Vfloat4::splat(0.5) * (diag0 + diag1 - root)).max(Vfloat4::zero())
    } else {
        let diag2 = Vfloat4::new(a, a, a, b);
        let cov02 = Vfloat4::new(
            cov1.lane(0) * sg * sa,
            cov0.lane(2) * sr * sa,
            cov0.lane(2) * sr * sa,
            cov0.lane(1) * sr * sb,
        );
        let cov12 = Vfloat4::new(
            cov1.lane(1) * sb * sa,
            cov1.lane(1) * sb * sa,
            cov1.lane(0) * sg * sa,
            cov0.lane(3) * sg * sb,
        );
        estimate_3comp_line_error(diag0, diag1, diag2, cov01, cov02, cov12)
    };

    let mut scores = errors.m;
    let mut component_count = BLOCK_MAX_COMPONENTS;
    for i in 0..BLOCK_MAX_COMPONENTS {
        if blk.is_constant_channel(i) {
            scores[i] = ERROR_CALC_DEFAULT;
            component_count -= 1;
        }
    }

    // The single-plane error is common to every split, so rank by remaining fit error
    for i in 1..BLOCK_MAX_COMPONENTS {
        let component = plane_components[i];
        let mut j = i;
        while j > 0 && scores[component] < scores[plane_components[j - 1]] {
            plane_components[j] = plane_components[j - 1];
            j -= 1;
        }
        plane_components[j] = component;
    }

    // Reuse the single-plane fit direction to estimate its error in the same weighted metric
    let rg = cov0.lane(0) * sr * sg;
    let rb = cov0.lane(1) * sr * sb;
    let ra = cov0.lane(2) * sr * sa;
    let gb = cov0.lane(3) * sg * sb;
    let ga = cov1.lane(0) * sg * sa;
    let ba = cov1.lane(1) * sb * sa;
    let row0 = Vfloat4::new(r, rg, rb, ra);
    let row1 = Vfloat4::new(rg, g, gb, ga);
    let row2 = Vfloat4::new(rb, gb, b, ba);
    let row3 = Vfloat4::new(ra, ga, ba, a);
    let dir = (ideal_ep.endpt1[0] - ideal_ep.endpt0[0]) * channel_scale;
    let projected = (row0 * dir.swz4(0, 0, 0, 0) + row1 * dir.swz4(1, 1, 1, 1))
        + (row2 * dir.swz4(2, 2, 2, 2) + row3 * dir.swz4(3, 3, 3, 3));
    let variance = dir.dot_s(projected) / mathlib::max(dir.dot_s(dir), 1e-20);
    let total_variance = diag.hadd_s();
    let single_error = mathlib::max(total_variance - variance, 0.0);
    let best_gain = single_error - scores[plane_components[0]];

    // Keep ambiguous choices; only prune when the predicted benefit is above numeric noise
    if min_gain_factor > 0.0 && best_gain > (total_variance * 1e-4) {
        let error_limit = single_error - min_gain_factor * best_gain;
        while component_count > 1 && scores[plane_components[component_count - 1]] > error_limit
        {
            component_count -= 1;
        }
    }
    component_count
}
}

isa_fn! {
/// UNORM16 constant color of the block origin texel.
fn constant_u16(blk: &ImageBlock) -> [i32; 4] {
    let color_f32 = Vfloat4::clamp(0.0, 1.0, blk.origin_texel) * 65535.0;
    color_f32.float_to_int_rtn().m
}
}

isa_fn! {
/// `compress_block`: compresses one image block into a 16 byte physical ASTC block, and with
/// `restricted_pcb` (LZ-RDO) also into its restricted 1 partition, 1 plane encoding.
pub fn compress_block(
    config: &AstcencConfig,
    bsd: &BlockSizeDescriptor,
    blk: &ImageBlock,
    pcb: &mut [u8; 16],
    restricted_pcb: Option<&mut [u8; 16]>,
    tmpbuf: &mut CompressionWorkingBuffers,
) {
    let decode_mode = config.profile;
    let mut scb = SymbolicCompressedBlock::default();
    let mut plane_components: [usize; BLOCK_MAX_COMPONENTS] = [3, 2, 1, 0];
    let mut plane_component_count = 0usize;
    let mut covariance = [Vfloat4::zero(); 3];

    // Set stricter block targets for luminance data as we have more bits to play with
    let block_is_l = blk.is_luminance();
    let block_is_l_scale = if block_is_l { 1.0 / 1.5 } else { 1.0 };

    // Set slightly stricter block targets for lumalpha data as we have more bits to play with
    let block_is_la = blk.is_luminancealpha();
    let block_is_la_scale = if block_is_la { 1.0 / 1.05 } else { 1.0 };

    // Grayscale blocks can only separate alpha, so constant alpha needs no dual-plane search
    let mut block_skip_two_plane = blk.grayscale && blk.is_constant_channel(3);

    let max_partitions = config.tune_partition_count_limit;

    let requested_partition_indices = [
        config.tune_2partition_index_limit,
        config.tune_3partition_index_limit,
        config.tune_4partition_index_limit,
    ];

    let requested_partition_trials = [
        config.tune_2partitioning_candidate_limit,
        config.tune_3partitioning_candidate_limit,
        config.tune_4partitioning_candidate_limit,
    ];

    // Detected a constant-color block
    if blk.data_min.eq(blk.data_max).all() {
        scb.partition_count = 0;

        // Encode as FP16 if using HDR
        if decode_mode == AstcencProfile::Hdr || decode_mode == AstcencProfile::HdrRgbLdrA {
            scb.block_type = SYM_BTYPE_CONST_F16;
            scb.constant_color = float_to_float16_v(blk.origin_texel).m;
        } else {
            // Encode as UNORM16 if NOT using HDR
            scb.block_type = SYM_BTYPE_CONST_U16;
            scb.constant_color = constant_u16(blk);
        }

        symbolic_to_physical(bsd, &scb, pcb);
        if let Some(restricted_pcb) = restricted_pcb {
            *restricted_pcb = *pcb;
        }
        return;
    }

    let error_weight_sum = blk.channel_weight.hadd_s() * f32::from(bsd.texel_count);
    let error_threshold =
        config.tune_db_limit * error_weight_sum * block_is_l_scale * block_is_la_scale;

    // Set SCB and mode errors to a very high error value
    scb.errorval = ERROR_CALC_DEFAULT;
    scb.block_type = SYM_BTYPE_ERROR;

    let mut best_errorvals_for_pcount = [ERROR_CALC_DEFAULT; BLOCK_MAX_PARTITIONS];

    let exit_thresholds_for_pcount = [
        0.0,
        config.tune_2partition_early_out_limit_factor,
        config.tune_3partition_early_out_limit_factor,
        0.0,
    ];

    // Trial using 1 plane of weights and 1 partition. Most of the time we test it twice, first
    // with a mode cutoff of 0 and then with the specified mode cutoff. This causes an early-out
    // that speeds up encoding of easy blocks.
    let errorval_mult = [1.0 / config.tune_mse_overshoot, 1.0];

    let errorval_overshoot = 1.0 / config.tune_mse_overshoot;

    // If restricted encoding was requested, do an extra encode with the most likely candidate
    // mode, and write it to restricted_pcb. This is basically the same as the restricted MODE0
    // fast path trial, but with slightly different parameters and a different purpose.
    if let Some(restricted_pcb) = restricted_pcb {
        let mut workscb = scb;

        // Can't enable "only_always" for 3D blocks since no "always" block modes are available
        compress_symbolic_block_for_partition_1plane(
            config,
            bsd,
            blk,
            bsd.dim_z == 1,
            error_threshold * errorval_mult[1] * errorval_overshoot,
            1,
            0,
            &mut workscb,
            tmpbuf,
            QUANT_32 as i32,
        );
        if workscb.block_type == SYM_BTYPE_ERROR {
            // No encoding passed (possible with the RGBM error metric). Upstream encodes the
            // uninitialized fields of the error block (undefined behaviour); use the mean color
            // constant block of the main search instead (LZ-RDO is LDR only).
            workscb.block_type = SYM_BTYPE_CONST_U16;
            workscb.partition_count = 0;
            workscb.constant_color = Vfloat4::clamp(0.0, 65535.0, blk.data_mean)
                .float_to_int_rtn()
                .m;
        }
        symbolic_to_physical(bsd, &workscb, restricted_pcb);
    }

    // Only enable MODE0 fast path if enabled
    // Never enable for 3D blocks as no "always" block modes are available
    let mut start_trial = 1;
    if config.tune_search_mode0_enable >= TUNE_MIN_SEARCH_MODE0 && bsd.dim_z == 1 {
        start_trial = 0;
    }

    'tests: {
        let mut quant_limit = QUANT_32 as i32;
        for i in start_trial..2 {
            let errorval = compress_symbolic_block_for_partition_1plane(
                config,
                bsd,
                blk,
                i == 0,
                error_threshold * errorval_mult[i] * errorval_overshoot,
                1,
                0,
                &mut scb,
                tmpbuf,
                QUANT_32 as i32,
            );

            // Record the quant level so we can use the filter later searches
            if scb.block_type != SYM_BTYPE_ERROR {
                let bm = bsd.get_block_mode(scb.block_mode as usize);
                quant_limit = bm.get_weight_quant_mode() as i32;
            }

            best_errorvals_for_pcount[0] = mathlib::min(best_errorvals_for_pcount[0], errorval);
            if errorval < (error_threshold * errorval_mult[i]) {
                break 'tests;
            }
        }

        let mut lowest_correl = 0.0f32;
        if !block_skip_two_plane {
            lowest_correl =
                prepare_block_statistics(bsd.texel_count as usize, blk, &mut covariance);
        }

        block_skip_two_plane = block_skip_two_plane
            || lowest_correl >= config.tune_2plane_early_out_limit_correlation;
        if !block_skip_two_plane {
            plane_component_count = if blk.grayscale {
                1
            } else {
                select_dual_plane_components(
                    blk,
                    &covariance,
                    &tmpbuf.ei1.ep,
                    config.tune_2plane_component_limit_factor,
                    &mut plane_components,
                )
            };
        }

        // Test only components with competitive predicted error reductions
        for &i in &plane_components[..plane_component_count] {
            if blk.grayscale && i != 3 {
                continue;
            }

            if blk.is_constant_channel(i) {
                continue;
            }

            let errorval = compress_symbolic_block_for_partition_2planes(
                config,
                bsd,
                blk,
                error_threshold * errorval_overshoot,
                i as u32,
                &mut scb,
                tmpbuf,
                quant_limit,
            );

            // If attempting two planes is much worse than the best one plane result
            // then further two plane searches are unlikely to help so move on ...
            if errorval > (best_errorvals_for_pcount[0] * 1.85) {
                break;
            }

            if errorval < error_threshold {
                break 'tests;
            }
        }

        // Find best blocks for 2, 3 and 4 partitions
        for partition_count in 2..=max_partitions as usize {
            let mut partition_indices = [0u32; TUNE_MAX_PARTITIONING_CANDIDATES];

            let requested_indices = requested_partition_indices[partition_count - 2];

            let mut requested_trials = requested_partition_trials[partition_count - 2];
            requested_trials = requested_trials.min(requested_indices);

            let actual_trials = find_best_partition_candidates(
                bsd,
                blk,
                partition_count as u32,
                requested_indices,
                &mut partition_indices,
                requested_trials,
            );

            let best_error_in_prev = best_errorvals_for_pcount[partition_count - 2];

            for i in 0..actual_trials as usize {
                let errorval = compress_symbolic_block_for_partition_1plane(
                    config,
                    bsd,
                    blk,
                    false,
                    error_threshold * errorval_overshoot,
                    partition_count as u32,
                    partition_indices[i],
                    &mut scb,
                    tmpbuf,
                    quant_limit,
                );

                best_errorvals_for_pcount[partition_count - 1] =
                    mathlib::min(best_errorvals_for_pcount[partition_count - 1], errorval);

                // If using N partitions doesn't improve much over using N-1 partitions then skip
                // trying N+1. Error can dramatically improve if the data is correlated or
                // non-correlated and aligns with a partitioning that suits that encoding, so for
                // this inner loop check add a large error scale because the "other" trial could
                // be a lot better. No four-partition pruning threshold is defined.
                let best_error = best_errorvals_for_pcount[partition_count - 1];
                let best_error_scale = exit_thresholds_for_pcount[partition_count - 1] * 1.85;
                if partition_count < 4 && best_error > (best_error_in_prev * best_error_scale) {
                    break 'tests;
                }

                if errorval < error_threshold {
                    break 'tests;
                }
            }

            // If using N partitions doesn't improve much over using N-1 partitions then skip
            // trying N+1
            let best_error = best_errorvals_for_pcount[partition_count - 1];
            let best_error_scale = exit_thresholds_for_pcount[partition_count - 1];
            if partition_count < 4 && best_error > (best_error_in_prev * best_error_scale) {
                break 'tests;
            }
        }
    }

    // Replace any remaining error block with the mean color in the compression color space
    if scb.block_type == SYM_BTYPE_ERROR {
        let color = Vfloat4::clamp(0.0, 65535.0, blk.data_mean);
        if decode_mode == AstcencProfile::Hdr || decode_mode == AstcencProfile::HdrRgbLdrA {
            // Convert the LNS mean back to FP16, keeping LDR alpha in linear space
            let mut color_f16 = lns_to_sf16(color.float_to_int());
            if decode_mode == AstcencProfile::HdrRgbLdrA {
                let alpha_f16 = float_to_float16_v(color / 65535.0);
                color_f16.set_lane(3, alpha_f16.lane(3));
            }
            scb.block_type = SYM_BTYPE_CONST_F16;
            scb.constant_color = color_f16.m;
        } else {
            scb.block_type = SYM_BTYPE_CONST_U16;
            scb.constant_color = color.float_to_int_rtn().m;
        }
    }

    // Compress to a physical block
    symbolic_to_physical(bsd, &scb, pcb);
}
}
