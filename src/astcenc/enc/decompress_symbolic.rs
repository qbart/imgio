// SPDX-FileCopyrightText: 2011-2026 Arm Limited
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from astc-encoder and modified; see NOTICE.md.
//! Compressor side of `astcenc_decompress_symbolic.cpp`: weight unpacking with the
//! compressor's block size descriptor and the symbolic block error metrics.

#![allow(clippy::needless_range_loop)]

use super::internal::*;
use super::vecmath::*;
use super::{ASTCENC_FLG_MAP_RGBM, AstcencConfig, AstcencProfile};

isa_fn! {
/// `get_u8_component_mask`.
pub fn get_u8_component_mask(decode_mode: AstcencProfile, blk: &ImageBlock) -> Vmask4 {
    // Decode mode or sRGB forces writing to unorm8 output value
    Vmask4::splat(blk.decode_unorm8 || decode_mode == AstcencProfile::LdrSrgb)
}
}

isa_fn! {
/// `lerp_color_int`.
fn lerp_color_int(u8_mask: Vmask4, color0: Vint4, color1: Vint4, weights: Vint4) -> Vint4 {
    let color_delta = color1 - color0;
    let color = color0 + (color_delta * weights + Vint4::splat(32)).asr(6);

    // For decode_unorm8 values force the codec to bit replicate
    let color_u8 = color.asr(8) * Vint4::splat(257);
    Vint4::select(color, color_u8, u8_mask)
}
}

isa_fn! {
/// `unpack_color_endpoints` returning `vint4` endpoints.
pub fn unpack_color_endpoints(
    decode_mode: AstcencProfile,
    format: u8,
    input: &[u8; 8],
) -> (Vint4, Vint4, bool, bool) {
    let (e0, e1, rgb_hdr, alpha_hdr) = super::unpack_color_endpoints(decode_mode, format, input);
    (Vint4 { m: e0 }, Vint4 { m: e1 }, rgb_hdr, alpha_hdr)
}
}

/// `unpack_weights` lookup tables of one symbolic block: the plane weights as `i32` (the
/// vtable lookups only use the low index bits: 64 entries, or 32 per plane).
struct WeightTables {
    tab1: [i32; BLOCK_MAX_WEIGHTS],
    tab2: [i32; WEIGHTS_PLANE2_OFFSET],
    size: usize,
    direct: bool,
}

isa_fn! {
#[inline]
fn weight_tables(
    bsd: &BlockSizeDescriptor,
    scb: &SymbolicCompressedBlock,
    di: &DecimationInfo,
    is_dual_plane: bool,
) -> WeightTables {
    let size = if is_dual_plane {
        WEIGHTS_PLANE2_OFFSET
    } else {
        BLOCK_MAX_WEIGHTS
    };
    let mut t = WeightTables {
        tab1: [0; BLOCK_MAX_WEIGHTS],
        tab2: [0; WEIGHTS_PLANE2_OFFSET],
        size,
        direct: di.weight_count == bsd.texel_count,
    };
    for k in 0..size {
        t.tab1[k] = i32::from(scb.weights[k]);
    }
    if is_dual_plane {
        for k in 0..WEIGHTS_PLANE2_OFFSET {
            t.tab2[k] = i32::from(scb.weights[WEIGHTS_PLANE2_OFFSET + k]);
        }
    }
    t
}
}

isa_fn! {
/// The unquantized, undecimated weights (0..64) of texels `i..i + 8` of plane 1 and (dual
/// plane) plane 2, as `unpack_weights` computes them. Undecimated grids are copied directly
/// (the infill of a single full weight is exact).
#[inline]
fn unpack_weights_at(
    t: &WeightTables,
    di: &DecimationInfo,
    i: usize,
    is_dual_plane: bool,
) -> (Vint8, Vint8) {
    if t.direct {
        let w = |tab: &[i32]| Vint8::load(tab.get(i..).unwrap_or_default());
        let w2 = if is_dual_plane {
            w(&t.tab2)
        } else {
            Vint8::zero()
        };
        return (w(&t.tab1[..t.size]), w2);
    }
    let index_mask = t.size as i32 - 1;
    let max_weight_count =
        Vint8::from_u8x8(di.texel_weight_count.get(i..).unwrap_or_default()).hmax_s();
    let mut sum1 = Vint8::splat(8);
    let mut sum2 = Vint8::splat(8);
    for j in 0..max_weight_count.max(0) as usize {
        let idx =
            Vint8::from_u8x8(di.texel_weights_tr[j].get(i..).unwrap_or_default()) & index_mask;
        let contrib =
            Vint8::from_u8x8(di.texel_weight_contribs_int_tr[j].get(i..).unwrap_or_default());
        sum1 += Vint8 {
            m: super::sse::gather_epi32(&t.tab1[..t.size], idx.m),
        } * contrib;
        if is_dual_plane {
            sum2 += Vint8 {
                m: super::sse::gather_epi32(&t.tab2, idx.m),
            } * contrib;
        }
    }
    (sum1.lsr(4), sum2.lsr(4))
}
}

isa_fn! {
/// Perceptual RGBM error colors; `None` for a zero weight M decode.
fn rgbm_colors(config: &AstcencConfig, color: Vfloat4, old: Vfloat4) -> Option<(Vfloat4, Vfloat4)> {
    // Fail encodings that result in zero weight M pixels
    if color.lane(3) == 0.0 {
        return None;
    }
    let s = config.rgbm_m_scale;
    let c = Vfloat4::new(
        color.lane(0) * color.lane(3) * s,
        color.lane(1) * color.lane(3) * s,
        color.lane(2) * color.lane(3) * s,
        1.0,
    );
    let o = Vfloat4::new(
        old.lane(0) * old.lane(3) * s,
        old.lane(1) * old.lane(3) * s,
        old.lane(2) * old.lane(3) * s,
        1.0,
    );
    Some((c, o))
}
}

isa_fn! {
/// One texel's error term of `compute_symbolic_block_difference_{1,2}plane`.
fn texel_error(old_color: Vfloat4, color: Vfloat4, channel_weight: Vfloat4) -> Vfloat4 {
    let mut error = old_color - color;
    error = error.abs().min_s(1e15);
    error = error * error;
    error.dot(channel_weight).min_s(ERROR_CALC_DEFAULT)
}
}

isa_fn! {
/// `compute_symbolic_block_difference_2plane`.
pub fn compute_symbolic_block_difference_2plane(
    config: &AstcencConfig,
    bsd: &BlockSizeDescriptor,
    scb: &SymbolicCompressedBlock,
    blk: &ImageBlock,
) -> f32 {
    // If we detected an error-block, blow up immediately.
    if scb.block_type == SYM_BTYPE_ERROR {
        return ERROR_CALC_DEFAULT;
    }

    let bm = bsd.get_block_mode(scb.block_mode as usize);
    let di = bsd.get_decimation_info(bm.decimation_mode as usize);

    // Unquantize and undecimate the weights (8 texels at a time)
    let tables = weight_tables(bsd, scb, di, true);

    let plane2_mask = Vint4::lane_id().eq(Vint4::splat(i32::from(scb.plane2_component)));

    let mut summa = Vfloat4::zero();

    // Decode the color endpoints for this partition
    let (ep0, ep1, _, _) =
        unpack_color_endpoints(config.profile, scb.color_formats[0], &scb.color_values[0]);

    let u8_mask = get_u8_component_mask(config.profile, blk);

    // Unpack and compute error for each texel in the partition
    let mut chunk = (Vint8::zero(), Vint8::zero());
    for i in 0..bsd.texel_count as usize {
        if i % SIMD_WIDTH == 0 {
            chunk = unpack_weights_at(&tables, di, i, true);
        }
        let weight = Vint4::select(
            Vint4::splat(chunk.0.m[i % SIMD_WIDTH]),
            Vint4::splat(chunk.1.m[i % SIMD_WIDTH]),
            plane2_mask,
        );
        let colori = lerp_color_int(u8_mask, ep0, ep1, weight);

        let mut color = colori.int_to_float();
        let mut old_color = blk.texel(i);

        // Compare error using a perceptual decode metric for RGBM textures
        if config.flags & ASTCENC_FLG_MAP_RGBM != 0 {
            let Some((c, o)) = rgbm_colors(config, color, old_color) else {
                return -ERROR_CALC_DEFAULT;
            };
            color = c;
            old_color = o;
        }

        summa += texel_error(old_color, color, blk.channel_weight);
    }

    summa.lane(0)
}
}

isa_fn! {
/// `compute_symbolic_block_difference_1plane`.
pub fn compute_symbolic_block_difference_1plane(
    config: &AstcencConfig,
    bsd: &BlockSizeDescriptor,
    scb: &SymbolicCompressedBlock,
    blk: &ImageBlock,
) -> f32 {
    // If we detected an error-block, blow up immediately.
    if scb.block_type == SYM_BTYPE_ERROR {
        return ERROR_CALC_DEFAULT;
    }

    // Get the appropriate partition-table entry
    let partition_count = u32::from(scb.partition_count);
    let pi = bsd.get_partition_info(partition_count, u32::from(scb.partition_index));

    // Get the appropriate block descriptor
    let bm = bsd.get_block_mode(scb.block_mode as usize);
    let di = bsd.get_decimation_info(bm.decimation_mode as usize);

    // Unquantize and undecimate the weights
    let tables = weight_tables(bsd, scb, di, false);
    let mut plane1_weights = [0u8; BLOCK_MAX_TEXELS + SIMD_WIDTH];
    let mut i = 0;
    while i < bsd.texel_count as usize {
        let w = unpack_weights_at(&tables, di, i, false).0;
        if let Some(d) = plane1_weights.get_mut(i..).and_then(|d| d.first_chunk_mut::<8>()) {
            *d = w.pack_bytes8();
        }
        i += SIMD_WIDTH;
    }

    let u8_mask = get_u8_component_mask(config.profile, blk);

    let mut summa = Vfloat4::zero();
    for i in 0..partition_count as usize {
        // Decode the color endpoints for this partition
        let (ep0, ep1, _, _) =
            unpack_color_endpoints(config.profile, scb.color_formats[i], &scb.color_values[i]);

        // Unpack and compute error for each texel in the partition
        let texel_count = pi.partition_texel_count[i] as usize;
        for j in 0..texel_count {
            let tix = pi.texels_of_partition[i][j] as usize;
            let colori =
                lerp_color_int(u8_mask, ep0, ep1, Vint4::splat(i32::from(plane1_weights[tix])));

            let mut color = colori.int_to_float();
            let mut old_color = blk.texel(tix);

            // Compare error using a perceptual decode metric for RGBM textures
            if config.flags & ASTCENC_FLG_MAP_RGBM != 0 {
                let Some((c, o)) = rgbm_colors(config, color, old_color) else {
                    return -ERROR_CALC_DEFAULT;
                };
                color = c;
                old_color = o;
            }

            summa += texel_error(old_color, color, blk.channel_weight);
        }
    }

    summa.lane(0)
}
}

isa_fn! {
/// `compute_symbolic_block_difference_1plane_1partition`.
pub fn compute_symbolic_block_difference_1plane_1partition(
    config: &AstcencConfig,
    bsd: &BlockSizeDescriptor,
    scb: &SymbolicCompressedBlock,
    blk: &ImageBlock,
) -> f32 {
    // If we detected an error-block, blow up immediately.
    if scb.block_type == SYM_BTYPE_ERROR {
        return ERROR_CALC_DEFAULT;
    }

    // Get the appropriate block descriptor
    let bm = bsd.get_block_mode(scb.block_mode as usize);
    let di = bsd.get_decimation_info(bm.decimation_mode as usize);

    // Unquantize and undecimate the weights (8 texels at a time)
    let tables = weight_tables(bsd, scb, di, false);

    // Decode the color endpoints for this partition
    let (ep0, ep1, _, _) =
        unpack_color_endpoints(config.profile, scb.color_formats[0], &scb.color_values[0]);

    let u8_mask = get_u8_component_mask(config.profile, blk);

    // Unpack and compute error for each texel in the partition
    let mut summav = Vfloat4::zero();
    let mut lane_id = Vint8::lane_id();
    let texel_count = bsd.texel_count as usize;
    let cw = blk.channel_weight;
    let mut i = 0;
    while i < texel_count {
        let weight = unpack_weights_at(&tables, di, i, false).0;

        let mut err = [Vfloat8::zero(); 4];
        for c in 0..4 {
            // Interpolate the endpoint delta (`lerp_color_int`)
            let ep0_c = Vint8::splat(ep0.lane(c));
            let ep_delta = Vint8::splat(ep1.lane(c)) - ep0_c;
            let mut colori = ep0_c + (ep_delta * weight + Vint8::splat(32)).asr(6);

            // If using a U8 decode mode bit replicate top 8 bits
            let colori8 = colori.asr(8) * Vint8::splat(257);
            colori = Vint8::select(colori, colori8, Vmask8::splat(u8_mask.lane(c)));

            // Compute color diff
            let color = colori.int_to_float();
            let orig = Vfloat8::loada(blk.data(c).get(i..).unwrap_or_default());
            let e = (orig - color).abs().min(Vfloat8::splat(1e15));
            err[c] = e * e;
        }

        // Compute squared error metric
        let metric =
            err[0] * cw.lane(0) + err[1] * cw.lane(1) + err[2] * cw.lane(2) + err[3] * cw.lane(3);

        // Mask off bad lanes
        let mask = lane_id.lt(Vint8::splat(texel_count as i32));
        lane_id += Vint8::splat(SIMD_WIDTH as i32);
        Vfloat8::haccumulate_masked(&mut summav, metric, mask);
        i += SIMD_WIDTH;
    }

    summav.hadd_s()
}
}
