// SPDX-FileCopyrightText: 2026 Epic Games, Inc.
// SPDX-FileCopyrightText: 2026 Arm Limited
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from astc-encoder and modified; see NOTICE.md.
//! Port of `astcenc_lz_optimize.cpp`: LZ backend rate-distortion optimization.
//!
//! Move-to-front lists of recently used endpoints (with the block header) and weight bits are
//! kept while the blocks are revisited, and blocks are rebuilt from them where a combined
//! rate-distortion score (`error + lambda * estimated bits`) favours it: the reused bit
//! patterns let a backend LZ coder (Deflate, Zstd, LZ4, ...) find matches. The error is
//! weighted per texel by an activity measure (high-pass energy), so the extra error goes where
//! it is least visible.
//!
//! The work is split into independent tiles (setup) and chunks of 4096 blocks (optimization,
//! seeded by the chunk start), so the output does not depend on the thread count.

#![allow(clippy::needless_range_loop)]

use std::sync::RwLock;

use super::compressor::Compressor;
use super::decompress_symbolic::{
    compute_symbolic_block_difference_1plane_1partition_weighted, unpack_color_endpoints,
    unpack_weights,
};
use super::ideal_endpoints_and_weights::{
    compute_ideal_weights_for_decimation, compute_quantized_weights_for_decimation,
};
use super::image::load_image_block;
use super::internal::*;
use super::mathlib;
use super::symbolic_physical::{physical_to_symbolic, symbolic_to_physical};
use super::vecmath::*;
use super::{
    ASTCENC_FLG_USE_DECODE_UNORM8, AstcencConfig, AstcencError, AstcencImage, AstcencOutput,
    AstcencProfile, AstcencSwizzle, LzRdo,
};

/// Size of an ASTC block in bytes.
const ASTC_BLOCK_BYTES: usize = 16;

/// Number of best candidates to keep for each block during optimization.
const BEST_CANDIDATES_COUNT: usize = 8;

/// Maximum number of blocks to process per worker thread item.
const MAX_BLOCKS_PER_ITEM: usize = 4096;

/// Cache size for block decompression results, must be a power of 2.
const CACHE_SIZE: usize = MAX_BLOCKS_PER_ITEM;

/// `physical_block`: the 16 block bytes as a little-endian 128-bit value (bit `i` of the
/// block is bit `i` of the integer).
type PhysicalBlock = u128;

/// `physical_block::top_bits_mask`: a mask of the `count` most significant bits.
#[inline]
fn top_bits_mask(count: i32) -> PhysicalBlock {
    if count <= 0 {
        0
    } else if count >= 128 {
        !0
    } else {
        !0u128 << (128 - count)
    }
}

/// `physical_block::hash`.
#[inline]
fn block_hash(block: PhysicalBlock) -> u32 {
    // FNV-1a inspired constants for 64-bit operations
    const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
    const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;

    let mut h1 = block as u64;
    let mut h2 = (block >> 64) as u64;

    // Mix the words
    h1 = h1
        .wrapping_mul(PRIME64_1)
        .rotate_left(31)
        .wrapping_mul(PRIME64_2);
    h2 = h2
        .wrapping_mul(PRIME64_2)
        .rotate_left(29)
        .wrapping_mul(PRIME64_1);

    // Combine the results
    let mut result = (h1 ^ h2) as u32;

    // Final avalanche
    result ^= result >> 15;
    result = result.wrapping_mul(0x85eb_ca6b);
    result ^= result >> 13;
    result = result.wrapping_mul(0xc2b2_ae35);
    result ^= result >> 16;
    result
}

/// `decode_index_to_coords`: raster-scan index to `(x, y, z)`.
#[inline]
fn decode_index_to_coords(index: usize, xdim: usize, ydim: usize) -> (usize, usize, usize) {
    let quot = index / xdim;
    (index % xdim, quot % ydim, quot / ydim)
}

/// `log2_approx`: the scalar `log2` approximation of `astcenc_vecmathlib.h`.
#[allow(clippy::excessive_precision)]
#[inline]
fn log2_approx(v: f32) -> f32 {
    let i = v.to_bits() as i32;
    let e = ((i >> 23) - 127) as f32;
    let m = f32::from_bits(((i & 0x007F_FFFF) | 0x3F80_0000) as u32);

    // Polynomial fit of log2(x)/(x - 1), for x in range [1, 2)
    let c0 = 2.888_270_454_816_477_620_1_f32;
    let c1 = -2.520_749_625_778_070_066_63_f32;
    let c2 = 1.481_166_475_212_131_716_41_f32;
    let c3 = -0.465_725_644_288_844_778_798_f32;
    let c4 = 0.059_651_548_267_457_496_953_3_f32;
    let p = (((c4 * m + c3) * m + c2) * m + c1) * m + c0;

    // Increases the polynomial degree, but ensures that log2(1) == 0
    let p = p * (m - 1.0);
    p + e
}

/// `mode_byte_histogram`: histograms of the first block bytes (which select the mode).
struct ModeByteHistogram {
    counts: [[u16; 256]; Self::NHISTO],
    totals: [u16; Self::NHISTO],
    counter: u32,
}

impl ModeByteHistogram {
    /// Number of initial bytes to histogram.
    const NHISTO: usize = 2;
    /// Number of updates between frequency decays.
    const DECAY_PERIOD: u32 = 256;
    /// Implicit count in every bucket so nothing has 0 frequency.
    const BIAS: f32 = 0.25;

    fn new() -> Self {
        ModeByteHistogram {
            counts: [[0; 256]; Self::NHISTO],
            totals: [0; Self::NHISTO],
            counter: 0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    /// Adds the stats of a block.
    fn update(&mut self, value: PhysicalBlock) {
        // Periodically decay the histogram
        self.counter += 1;
        if self.counter == Self::DECAY_PERIOD {
            self.counter = 0;
            for j in 0..Self::NHISTO {
                let mut new_total: u16 = 0;
                for c in self.counts[j].iter_mut() {
                    *c >>= 1;
                    new_total = new_total.wrapping_add(*c);
                }
                self.totals[j] = new_total;
            }
        }

        for j in 0..Self::NHISTO {
            let byte = (value >> (8 * j)) as u8 as usize;
            self.counts[j][byte] = self.counts[j][byte].wrapping_add(1);
            self.totals[j] = self.totals[j].wrapping_add(1);
        }
    }

    /// The estimated cost in bits of the histogrammed bytes of a block.
    fn cost(&self, value: PhysicalBlock) -> f32 {
        let mut cost_numer = 1.0f32;
        let mut cost_denom = 1.0f32;
        for i in 0..Self::NHISTO {
            let byte = (value >> (8 * i)) as u8 as usize;
            // Implicit extra count of BIAS in every bucket
            cost_numer *= f32::from(self.totals[i]) + 256.0 * Self::BIAS;
            cost_denom *= f32::from(self.counts[i][byte]) + Self::BIAS;
        }
        log2_approx(cost_numer / cost_denom)
    }
}

/// `mtf_list`: move-to-front list of recently used blocks.
struct MtfList {
    /// Most recent first.
    list: Vec<PhysicalBlock>,
    size: usize,
}

impl MtfList {
    fn new(capacity: usize) -> Self {
        MtfList {
            list: vec![0; capacity],
            size: 0,
        }
    }

    fn reset(&mut self) {
        self.size = 0;
    }

    /// The first index whose masked bits match, or -1.
    #[inline]
    fn search(&self, value: PhysicalBlock, mask: PhysicalBlock) -> i32 {
        let masked_value = value & mask;
        self.list[..self.size]
            .iter()
            .position(|&b| b & mask == masked_value)
            .map_or(-1, |i| i as i32)
    }

    /// Moves `value` to the front (`pos` is the search result); returns its old position.
    fn update(&mut self, value: PhysicalBlock, pos: i32) -> i32 {
        let pos = if pos < 0 {
            // If not found, insert at the end, growing up to capacity
            if self.size < self.list.len() {
                self.size += 1;
            }
            self.size as i32 - 1
        } else {
            pos
        };
        if pos >= 0 {
            let p = pos as usize;
            self.list.copy_within(0..p, 1);
            if let Some(first) = self.list.first_mut() {
                *first = value;
            }
        }
        pos
    }

    fn encode(&mut self, value: PhysicalBlock, mask: PhysicalBlock) -> i32 {
        let pos = self.search(value, mask);
        self.update(value, pos)
    }
}

/// `log2_approx(pos + 32)` of the MTF positions `pos` below `count`.
fn mtf_log2_table(count: usize) -> Vec<f32> {
    (0..count).map(|i| log2_approx(i as f32 + 32.0)).collect()
}

/// `log2_approx(pos + 32)` from `table`.
#[inline]
fn mtf_log2(table: &[f32], pos: i32) -> f32 {
    usize::try_from(pos)
        .ok()
        .and_then(|p| table.get(p).copied())
        .unwrap_or_else(|| log2_approx(pos as f32 + 32.0))
}

/// `calculate_bit_cost_simple`: approximate cost in bits of a block given the MTF positions
/// of its endpoints and weights (-1 when not found); `log2_table` is `mtf_log2_table`.
fn calculate_bit_cost_simple(
    mtf_pos_endpoints: i32,
    mtf_pos_weights: i32,
    block: PhysicalBlock,
    weight_bits: i32,
    mode_histo: &ModeByteHistogram,
    log2_table: &[f32],
) -> f32 {
    let endpoint_bits = 128 - weight_bits;
    let mut total_cost;

    if mtf_pos_endpoints == -1 {
        // Assume bytes 2 and onwards are random, but score first 2 using histo
        total_cost = endpoint_bits as f32 - 16.0 + mode_histo.cost(block);
    } else {
        // Heuristic match cost, don't over-tune this
        total_cost = 10.0 + mtf_log2(log2_table, mtf_pos_endpoints);

        // If both matches are from the same block, it's just one big match
        if mtf_pos_weights == mtf_pos_endpoints {
            return total_cost;
        }
    }

    // If endpoints ended in the middle of a byte and we don't have a full match treat the
    // remaining bits in that byte as literals
    if endpoint_bits & 7 != 0 {
        total_cost += 8.0 - (endpoint_bits & 7) as f32;
    }

    if mtf_pos_weights == -1 {
        // Bill the remaining weight bits (the first byte, if partial, is already accounted for)
        total_cost += (weight_bits & !7) as f32;
    } else {
        // Heuristic match cost, don't over-tune this
        total_cost += 10.0 + mtf_log2(log2_table, mtf_pos_weights);
    }

    total_cost
}

isa_fn! {
/// `calculate_ssd_weighted`: per-texel weighted squared error of two interleaved RGBA blocks.
#[inline]
fn calculate_ssd_weighted(
    img1: &[f32],
    img2: &[f32],
    float_count: usize,
    weights: &[f32],
    channel_weights: Vfloat4,
) -> f32 {
    let mut sum = Vfloat4::zero();
    let mut i = 0;
    while i < float_count {
        let diff = Vfloat8::load(img1.get(i..).unwrap_or_default())
            - Vfloat8::load(img2.get(i..).unwrap_or_default());
        // Fold RGBA groups in texel order to preserve the four-wide accumulation
        Vfloat8::haccumulate(
            &mut sum,
            diff * diff * Vfloat8::load(weights.get(i..).unwrap_or_default()),
        );
        i += SIMD_WIDTH;
    }
    sum.dot_s(channel_weights)
}
}

/// `get_block_mode`.
#[inline]
fn get_block_mode(block: PhysicalBlock) -> usize {
    (block as usize) & 0x7ff
}

/// `get_weight_bits`.
#[inline]
fn get_weight_bits(block: PhysicalBlock, bsd: &BlockSizeDescriptor) -> i32 {
    i32::from(bsd.weight_bits_for_mode[get_block_mode(block)])
}

/// `xorshift32`.
#[inline]
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

isa_fn! {
/// `decompress_symbolic_block_raw`: decodes a symbolic block into interleaved RGBA floats in
/// the decoder's 16-bit working range (before the final output conversion). Only the active
/// texels are written; the caller keeps the SIMD padding zero.
fn decompress_symbolic_block_raw(
    config: &AstcencConfig,
    bsd: &BlockSizeDescriptor,
    scb: &SymbolicCompressedBlock,
    out_rgba: &mut [f32],
) {
    let texel_count = bsd.texel_count as usize;

    // The unorm8 rounding for the decode
    let u8_mask = if config.flags & ASTCENC_FLG_USE_DECODE_UNORM8 != 0 {
        Vmask4::new(true, true, true, true)
    } else if config.profile == AstcencProfile::LdrSrgb {
        Vmask4::new(true, true, true, false)
    } else {
        Vmask4::new(false, false, false, false)
    };
    // The real decoder would just use the top 8 bits, but we rescale into a 16-bit value that
    // rounds correctly
    let apply_u8_rounding =
        |colori: Vint4| Vint4::select(colori, colori.asr(8) * Vint4::splat(257), u8_mask);

    if scb.block_type != SYM_BTYPE_NONCONST {
        // Error and constant blocks output the same value for every texel; error and FP16
        // constant blocks (not legal in the LDR profiles) produce values far outside the
        // nominal [0,65535] range
        let mut color = Vfloat4::splat(-65536.0);
        if scb.block_type == SYM_BTYPE_CONST_U16 {
            color = apply_u8_rounding(Vint4 {
                m: scb.constant_color,
            })
            .int_to_float();
        }
        for texel in out_rgba.as_chunks_mut::<4>().0.iter_mut().take(texel_count) {
            *texel = color.m;
        }
        return;
    }

    // Get the appropriate partition-table entry
    let partition_count = u32::from(scb.partition_count);
    let pi = bsd.get_partition_info(partition_count, u32::from(scb.partition_index));

    // Get the appropriate block descriptors
    let bm = bsd.get_block_mode(scb.block_mode as usize);
    let di = bsd.get_decimation_info(bm.decimation_mode as usize);
    let is_dual_plane = bm.is_dual_plane;

    // Unquantize and undecimate the weights
    let mut plane1_weights = [0i32; BLOCK_MAX_TEXELS];
    let mut plane2_weights = [0i32; BLOCK_MAX_TEXELS];
    unpack_weights(bsd, scb, di, is_dual_plane, &mut plane1_weights, &mut plane2_weights);

    // Now that we have endpoint colors and weights, we can unpack texel colors
    let plane2_mask = Vint4::lane_id().eq(Vint4::splat(i32::from(scb.plane2_component)));
    let texels = out_rgba.as_chunks_mut::<4>().0;

    for i in 0..partition_count as usize {
        // Decode the color endpoints for this partition
        let (ep0, ep1, _, _) =
            unpack_color_endpoints(config.profile, scb.color_formats[i], &scb.color_values[i]);
        let diff = ep1 - ep0;

        let texel_count = pi.partition_texel_count[i] as usize;
        for &tix in &pi.texels_of_partition[i][..texel_count] {
            let tix = tix as usize;
            let weight = Vint4::select(
                Vint4::splat(plane1_weights[tix]),
                Vint4::splat(plane2_weights[tix]),
                plane2_mask,
            );

            // Interpolate the color
            let colori = ep0 + (diff * weight + Vint4::splat(32)).asr(6);
            if let Some(texel) = texels.get_mut(tix) {
                *texel = apply_u8_rounding(colori).int_to_float().m;
            }
        }
    }
}
}

/// A cached error and the target block generation it was computed against.
#[derive(Clone, Copy, Default)]
struct CachedError {
    /// The target generation, or zero if no error has been computed.
    generation: u32,
    /// The weighted squared error for this target generation.
    value: f32,
}

/// `block_error_calculator`: block error from physical block bits, with a hash table of
/// decoded blocks (rows of 8 entries with FIFO eviction, tag bytes holding extra hash bits and
/// one FIFO counter per row) and their errors against the current target block.
struct BlockErrorCalculator {
    row_count: usize,
    floats_per_entry: usize,
    tags: Vec<u8>,
    entries: Vec<PhysicalBlock>,
    pixels: Vec<f32>,
    errors: Vec<CachedError>,
    error_generation: u32,
    channel_weights: Vfloat4,
    /// Activity weights replicated across RGBA channels, with zero SIMD padding.
    rgba_weights: Vec<f32>,
    /// Interleaved RGBA values of the current block's original pixels.
    original_rgba: Vec<f32>,
}

impl BlockErrorCalculator {
    const ROW_SHIFT: usize = 3;
    const ITEMS_PER_ROW: usize = 1 << Self::ROW_SHIFT;
    const TAGS_PER_ROW: usize = Self::ITEMS_PER_ROW + 1;

    fn new(num_entries: usize, bsd: &BlockSizeDescriptor) -> Self {
        let row_count = num_entries >> Self::ROW_SHIFT;
        let floats_per_entry = round_up_to_simd_multiple_vla(bsd.texel_count as usize * 4);
        let entry_count = row_count * Self::ITEMS_PER_ROW;
        BlockErrorCalculator {
            row_count,
            floats_per_entry,
            // All tags clear (entries invalid); the pixel padding is zero once and stays so
            tags: vec![0; row_count * Self::TAGS_PER_ROW],
            entries: vec![0; entry_count],
            pixels: vec![0.0; entry_count * floats_per_entry],
            errors: vec![CachedError::default(); entry_count],
            error_generation: 0,
            channel_weights: Vfloat4::zero(),
            rgba_weights: vec![0.0; floats_per_entry],
            original_rgba: vec![0.0; floats_per_entry],
        }
    }

    /// `set_current_block`: the source pixels and per-texel activity weights of the block
    /// being encoded.
    fn set_current_block(&mut self, blk: &ImageBlock, weights: &[f32]) {
        // Errors depend on the source pixels, activity weights, and channel weights
        self.error_generation = self.error_generation.wrapping_add(1);
        if self.error_generation == 0 {
            self.errors.fill(CachedError::default());
            self.error_generation = 1;
        }

        // Copy interleaved texels and replicate their activity weights
        let texel_count = blk.texel_count as usize;
        for i in 0..self.floats_per_entry {
            let texel = i / 4;
            let (pixel, weight) = if texel < texel_count {
                (
                    blk.data_rgba.get(texel).map_or(0.0, |t| t.m[i % 4]),
                    weights.get(texel).copied().unwrap_or(0.0),
                )
            } else {
                (0.0, 0.0)
            };
            self.original_rgba[i] = pixel;
            self.rgba_weights[i] = weight;
        }

        self.channel_weights = blk.channel_weight;
    }
}

isa_fn! {
/// `block_error_calculator::eval`: the weighted squared error of a block against the current
/// target block.
fn eval_error(
    ec: &mut BlockErrorCalculator,
    config: &AstcencConfig,
    bsd: &BlockSizeDescriptor,
    candidate_bits: PhysicalBlock,
) -> f32 {
    let hash = block_hash(candidate_bits);

    // Use low bits of hash to determine row index and high bits to determine tags
    let our_tag = ((hash >> 25) | 0x80) as u8;
    let row = hash as usize & (ec.row_count - 1);
    let fpe = ec.floats_per_entry;
    let tag_base = row * BlockErrorCalculator::TAGS_PER_ROW;
    let entry_base = row * BlockErrorCalculator::ITEMS_PER_ROW;

    // Check full blocks only for matching tags, preserving the first-match order
    let mut matched_entry = None;
    for i in 0..BlockErrorCalculator::ITEMS_PER_ROW {
        if ec.tags[tag_base + i] == our_tag && ec.entries[entry_base + i] == candidate_bits {
            let entry = entry_base + i;
            if ec.errors[entry].generation == ec.error_generation {
                return ec.errors[entry].value;
            }
            matched_entry = Some(entry);
            break;
        }
    }

    // If we got no match, insert a new entry
    let entry = match matched_entry {
        Some(entry) => entry,
        None => {
            let counter = &mut ec.tags[tag_base + BlockErrorCalculator::ITEMS_PER_ROW];
            let lru_slot = (*counter as usize) & (BlockErrorCalculator::ITEMS_PER_ROW - 1);
            *counter = counter.wrapping_add(1);

            ec.tags[tag_base + lru_slot] = our_tag;
            ec.entries[entry_base + lru_slot] = candidate_bits;
            let entry = entry_base + lru_slot;

            // Turn the physical encoding back to symbolic and decompress it in raw form
            // (without applying the final color transform)
            let mut scb = SymbolicCompressedBlock::default();
            physical_to_symbolic(bsd, &candidate_bits.to_le_bytes(), &mut scb);
            if let Some(pixels) = ec.pixels.get_mut(entry * fpe..entry * fpe + fpe) {
                decompress_symbolic_block_raw(config, bsd, &scb, pixels);
            }
            entry
        }
    };

    let pixels = ec.pixels.get(entry * fpe..).unwrap_or_default();
    let error =
        calculate_ssd_weighted(&ec.original_rgba, pixels, fpe, &ec.rgba_weights, ec.channel_weights);
    ec.errors[entry] = CachedError {
        generation: ec.error_generation,
        value: error,
    };
    error
}
}

/// One of the best candidate blocks in the running.
#[derive(Clone, Copy, Default)]
struct CandidateItem {
    bits: PhysicalBlock,
    rd_cost: f32,
    mtf_position: i32,
    weight_bits: i32,
}

/// `candidate_list`: the best few candidates, in ascending order of `rd_cost`.
#[derive(Default)]
struct CandidateList {
    list: [CandidateItem; BEST_CANDIDATES_COUNT],
    count: usize,
}

impl CandidateList {
    /// True if the list is full and `cost_lower_bound` can not beat its worst entry.
    #[inline]
    fn quick_reject(&self, cost_lower_bound: f32) -> bool {
        self.count == BEST_CANDIDATES_COUNT
            && cost_lower_bound >= self.list[BEST_CANDIDATES_COUNT - 1].rd_cost
    }

    /// Inserts a candidate in cost order if it is one of the best.
    fn add(
        &mut self,
        bits: PhysicalBlock,
        rd_cost: f32,
        mtf_position: i32,
        bsd: &BlockSizeDescriptor,
    ) {
        if self.quick_reject(rd_cost) {
            return;
        }

        let mut insert_pos = self.count.min(BEST_CANDIDATES_COUNT - 1);

        // Find the position to insert, shifting elements out of the way
        while insert_pos > 0 && rd_cost < self.list[insert_pos - 1].rd_cost {
            self.list[insert_pos] = self.list[insert_pos - 1];
            insert_pos -= 1;
        }

        self.list[insert_pos] = CandidateItem {
            bits,
            rd_cost,
            mtf_position,
            weight_bits: get_weight_bits(bits, bsd),
        };

        if self.count < BEST_CANDIDATES_COUNT {
            self.count += 1;
        }
    }

    fn items(&self) -> &[CandidateItem] {
        &self.list[..self.count]
    }
}

/// `compact_simple_endpoints`: mode and endpoints of a simple (non-constant, 1 plane,
/// 1 partition) block, without its weights.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct CompactSimpleEndpoints {
    block_mode: u16,
    quant_mode: u8,
    color_format: u8,
    color_values: [u8; 8],
}

impl CompactSimpleEndpoints {
    fn from_symbolic(blk: &SymbolicCompressedBlock) -> Self {
        let color_format = blk.color_formats[0];
        // Make the color values canonical so we can do equality compares on them (by setting
        // unused values to 0)
        let num_color_values = (color_format as usize >> 2) * 2 + 2;
        let mut color_values = [0u8; 8];
        for (i, v) in color_values.iter_mut().enumerate() {
            if i < num_color_values {
                *v = blk.color_values[0][i];
            }
        }
        CompactSimpleEndpoints {
            block_mode: blk.block_mode,
            quant_mode: blk.quant_mode as u8,
            color_format,
            color_values,
        }
    }

    /// `to_symbolic`: sets the mode and endpoints of `blk`, keeping its weights.
    fn store_symbolic(self, blk: &mut SymbolicCompressedBlock) {
        blk.block_type = SYM_BTYPE_NONCONST;
        blk.partition_count = 1;
        blk.color_formats_matched = 0;
        blk.plane2_component = -1;
        blk.block_mode = self.block_mode;
        blk.color_formats[0] = self.color_format;
        blk.quant_mode = u32::from(self.quant_mode);
        blk.color_values[0] = self.color_values;
    }
}

isa_fn! {
/// `compact_simple_endpoints::normalized_axis`: normalized endpoint axis direction (zero for
/// coincident endpoints), an early out to rule out endpoint pairs.
fn normalized_axis(ep: &CompactSimpleEndpoints) -> Vfloat4 {
    let v = &ep.color_values;
    let color0 = Vint4::new(
        i32::from(v[0]),
        i32::from(v[1]),
        i32::from(v[2]),
        i32::from(v[3]),
    );
    let color1 = Vint4::new(
        i32::from(v[4]),
        i32::from(v[5]),
        i32::from(v[6]),
        i32::from(v[7]),
    );
    normalize_safe((color0 - color1).int_to_float(), Vfloat4::zero())
}
}

/// An entry of [`EndpointSubstMtf`].
#[derive(Clone, Copy, Default)]
struct SubstEntry {
    endpoints: CompactSimpleEndpoints,
    normalized_axis: Vfloat4,
    endpoint0: Vfloat4,
    endpoint1: Vfloat4,
    /// The endpoint direction divided by its squared length, unless near zero.
    projection: Vfloat4,
    /// The squared endpoint distance used to scale weight errors.
    weight_error_scale: f32,
}

/// `endpoint_subst_mtf`: endpoint substitution move-to-front list (the entries stay put,
/// an index permutation is shuffled instead).
struct EndpointSubstMtf {
    profile: AstcencProfile,
    entries: Vec<SubstEntry>,
    order: Vec<usize>,
    size: usize,
}

impl EndpointSubstMtf {
    fn new(capacity: usize, profile: AstcencProfile) -> Self {
        EndpointSubstMtf {
            profile,
            entries: vec![SubstEntry::default(); capacity],
            order: (0..capacity).collect(),
            size: 0,
        }
    }

    fn reset(&mut self) {
        self.size = 0;
    }

    /// The entry at logical index `index`.
    #[inline]
    fn get(&self, index: usize) -> &SubstEntry {
        &self.entries[self.order[index]]
    }

    fn search(&self, endpoints: &CompactSimpleEndpoints) -> i32 {
        (0..self.size)
            .find(|&i| self.entries[self.order[i]].endpoints == *endpoints)
            .map_or(-1, |i| i as i32)
    }
}

isa_fn! {
/// `endpoint_subst_mtf::encode`: search and move to front (inserting a new entry, with its
/// projection data, when not found).
fn subst_mtf_encode(mtf: &mut EndpointSubstMtf, endpoints: &CompactSimpleEndpoints) {
    let mtf_index = mtf.search(endpoints);
    let capacity = mtf.entries.len();
    let (mtf_index, id) = if mtf_index < 0 {
        // New entry. If we're not yet at capacity, grow the list, else replace oldest.
        if mtf.size < capacity {
            mtf.size += 1;
        }
        let mtf_index = mtf.size.saturating_sub(1);
        let id = mtf.order[mtf_index];

        // Projection data depends only on the endpoints and is reused across source blocks
        let (ep0, ep1, _, _) =
            unpack_color_endpoints(mtf.profile, endpoints.color_format, &endpoints.color_values);
        let endpoint0 = ep0.int_to_float();
        let endpoint1 = ep1.int_to_float();
        let mut direction = endpoint1 - endpoint0;
        let length_squared = direction.dot_s(direction);
        if length_squared > 1e-7 {
            direction = direction * Vfloat4::splat(1.0 / length_squared);
        }
        mtf.entries[id] = SubstEntry {
            endpoints: *endpoints,
            normalized_axis: normalized_axis(endpoints),
            endpoint0,
            endpoint1,
            projection: direction,
            weight_error_scale: length_squared,
        };
        (mtf_index, id)
    } else {
        let mtf_index = mtf_index as usize;
        (mtf_index, mtf.order[mtf_index])
    };

    // Move accessed block to front
    mtf.order.copy_within(0..mtf_index, 1);
    mtf.order[0] = id;
}
}

isa_fn! {
/// `is_simple_block`: one partition and one weight plane.
fn is_simple_block(block: PhysicalBlock, bsd: &BlockSizeDescriptor) -> bool {
    let npart = (block >> 11) & 3;
    if npart != 0 {
        return false;
    }
    !bsd.get_block_mode(get_block_mode(block)).is_dual_plane
}
}

isa_fn! {
/// Encodes the simple endpoints of `block` into the substitution list.
fn subst_mtf_encode_block(
    mtf: &mut EndpointSubstMtf,
    bsd: &BlockSizeDescriptor,
    block: PhysicalBlock,
) {
    let mut scb = SymbolicCompressedBlock::default();
    physical_to_symbolic(bsd, &block.to_le_bytes(), &mut scb);
    subst_mtf_encode(mtf, &CompactSimpleEndpoints::from_symbolic(&scb));
}
}

isa_fn! {
/// `compute_known_endpoint_weights`: the ideal weights of a simple block given its mode and
/// endpoints.
fn compute_known_endpoint_weights(
    blk: &ImageBlock,
    entry: &SubstEntry,
    ei: &mut EndpointsAndWeights,
) {
    ei.ep.partition_count = 1;
    ei.ep.endpt0[0] = entry.endpoint0;
    ei.ep.endpt1[0] = entry.endpoint1;

    let basef = entry.endpoint0;
    let dirf = entry.projection;

    // This path normalizes channel weights so hadd_s(blk.channel_weight) == 4.0 always
    let error_weight = 1.0f32;
    let wes = entry.weight_error_scale * error_weight;

    let base = [0, 1, 2, 3].map(|c| Vfloat8::splat(basef.lane(c)));
    let dir = [0, 1, 2, 3].map(|c| Vfloat8::splat(dirf.lane(c)));
    let mut lane_id = Vint8::lane_id();
    let texel_count = blk.texel_count as usize;

    let mut i = 0;
    while i < texel_count {
        let param_r = (Vfloat8::loada(blk.data_r.get(i..).unwrap_or_default()) - base[0]) * dir[0];
        let param_g = (Vfloat8::loada(blk.data_g.get(i..).unwrap_or_default()) - base[1]) * dir[1];
        let param_b = (Vfloat8::loada(blk.data_b.get(i..).unwrap_or_default()) - base[2]) * dir[2];
        let param_a = (Vfloat8::loada(blk.data_a.get(i..).unwrap_or_default()) - base[3]) * dir[3];

        // Match the dot_s reduction order across color channels
        let param = (param_r + param_b) + (param_g + param_a);

        // Zero padded texels for subsequent SIMD over-fetch
        let mask = lane_id.lt(Vint8::splat(texel_count as i32));
        lane_id += Vint8::splat(SIMD_WIDTH as i32);

        Vfloat8::select(Vfloat8::zero(), param.clampzo(), mask)
            .store(ei.weights.get_mut(i..).unwrap_or_default());
        Vfloat8::select(Vfloat8::zero(), Vfloat8::splat(wes), mask)
            .store(ei.weight_error_scale.get_mut(i..).unwrap_or_default());
        i += SIMD_WIDTH;
    }

    ei.is_constant_weight_error_scale = true;
}
}

/// Thread-local state of the optimization passes.
struct PassState {
    mtf_weights: MtfList,
    mtf_endpoints: MtfList,
    mtf_simple_endpoints: EndpointSubstMtf,
    hist: ModeByteHistogram,
    err_calc: BlockErrorCalculator,
    /// `log2_approx(pos + 32)` of every MTF position.
    log2_table: Vec<f32>,
}

isa_fn! {
/// `lz_rdo_optimization_passes` for one chunk of blocks: `data` holds the chunk's current
/// encodings (on entry the original ones) and receives the optimized blocks; `restricted`
/// and `weights` are the chunk's restricted encodings and per-texel weights.
#[allow(clippy::too_many_arguments)]
fn optimize_chunk(
    config: &AstcencConfig,
    bsd: &BlockSizeDescriptor,
    st: &mut PassState,
    tmpbuf: &mut CompressionWorkingBuffers,
    blk: &mut ImageBlock,
    (block_start, block_count): (usize, usize),
    (blocks_x, blocks_y): (usize, usize),
    lambda: f32,
    img: &AstcencImage,
    swz: &AstcencSwizzle,
    data: &mut [u8],
    original_blocks: &mut [u8],
    restricted: &[u8],
    weights: &[f32],
) {
    let profile = config.profile;
    let texel_count = bsd.texel_count as usize;
    let block_at = |d: &[u8], i: usize| -> PhysicalBlock {
        d.get(i * ASTC_BLOCK_BYTES..i * ASTC_BLOCK_BYTES + ASTC_BLOCK_BYTES)
            .and_then(|b| <[u8; 16]>::try_from(b).ok())
            .map_or(0, u128::from_le_bytes)
    };

    // Save the original block encodings we had so far, and start with the restricted ones
    let chunk_bytes = block_count * ASTC_BLOCK_BYTES;
    original_blocks[..chunk_bytes].copy_from_slice(&data[..chunk_bytes]);
    data[..chunk_bytes].copy_from_slice(&restricted[..chunk_bytes]);
    let other_encodings: [&[u8]; 2] = [restricted, original_blocks];

    // Outer pass loop
    for pass in 0..config.tune_lz_rdo_num_passes {
        // Odd passes go over blocks in reverse order
        let is_forward = pass & 1 == 0;

        // Reset MTF lists and histogram
        st.mtf_weights.reset();
        st.mtf_endpoints.reset();
        st.mtf_simple_endpoints.reset();
        st.hist.reset();

        // Seed the structures with random blocks; the block start index and pass index are
        // part of the seed for variety between chunks (multiplied by a large prime for
        // better mixing)
        let mut rng_state = ((block_start as u32)
            .wrapping_mul(3_677_199_193)
            .wrapping_add(pass.wrapping_mul(7))
            .wrapping_add(23_857_527))
            | 1;
        const MAX_SAMPLES: usize = 64;
        let num_samples = block_count.min(MAX_SAMPLES);

        for i in 0..num_samples {
            let rng = xorshift32(&mut rng_state);

            // Draw a random block index within our group
            let block_idx = if num_samples == block_count {
                i
            } else {
                rng as usize % block_count
            };

            let block_bits = if pass == 0 {
                // Initial pass draws from seed blocks
                let which = ((rng >> 31) & 1) as usize;
                block_at(other_encodings[which], block_idx)
            } else {
                // Other passes draw from existing blocks
                block_at(data, block_idx)
            };

            let weights_mask = top_bits_mask(get_weight_bits(block_bits, bsd));

            // Update structures
            st.mtf_weights.encode(block_bits, weights_mask);
            st.mtf_endpoints.encode(block_bits, !weights_mask);
            st.hist.update(block_bits);

            if is_simple_block(block_bits, bsd) {
                subst_mtf_encode_block(&mut st.mtf_simple_endpoints, bsd, block_bits);
            }
        }

        // Process the blocks in either forward or reverse order
        for block_iter in 0..block_count {
            let local_index = if is_forward {
                block_iter
            } else {
                block_count - 1 - block_iter
            };
            let block_index = block_start + local_index;
            let (x, y, z) = decode_index_to_coords(block_index, blocks_x, blocks_y);

            // Get current block data and compute its weight bits
            let current_bits = block_at(data, local_index);
            let current_weight_bits = get_weight_bits(current_bits, bsd);
            let mut best = current_bits;

            // Don't process blocks with no weight bits, accept void-extent as is
            if current_weight_bits == 0 {
                st.mtf_weights.encode(current_bits, 0);
                st.mtf_endpoints.encode(current_bits, !0);
                st.hist.update(current_bits);
                continue;
            }

            // Get the source pixels
            load_image_block(
                profile,
                img,
                blk,
                bsd,
                x * bsd.dim_x as usize,
                y * bsd.dim_y as usize,
                z * bsd.dim_z as usize,
                swz,
            );

            // Set up error calculation for the current block
            let block_texel_weights = weights
                .get(local_index * texel_count..)
                .unwrap_or_default();
            st.err_calc.set_current_block(blk, block_texel_weights);

            // Decode the original block to compute initial error, as well as the error for the
            // restricted approximation
            let original_err = eval_error(&mut st.err_calc, config, bsd, current_bits);
            let mut restricted_err = ERROR_CALC_DEFAULT;

            // Calculate masks for weights and endpoints
            let current_weights_mask = top_bits_mask(current_weight_bits);
            let mtf_weights_pos = st.mtf_weights.search(current_bits, current_weights_mask);
            let mtf_endpoints_pos = st.mtf_endpoints.search(current_bits, !current_weights_mask);

            // Figure out rate to determine initial RD cost
            let original_rate = calculate_bit_cost_simple(
                mtf_endpoints_pos,
                mtf_weights_pos,
                current_bits,
                current_weight_bits,
                &st.hist,
                &st.log2_table,
            );
            let mut best_rd_cost = original_err + lambda * original_rate;

            // Candidate structure for storing best candidates
            let mut best_endpoints = CandidateList::default();

            // Add the current block to the candidates
            best_endpoints.add(current_bits, best_rd_cost, mtf_endpoints_pos, bsd);

            // Try the other candidate encodings
            for other_encoding in other_encodings {
                let other_bits = block_at(other_encoding, local_index);
                let other_weight_bits = get_weight_bits(other_bits, bsd);
                let other_weight_mask = top_bits_mask(other_weight_bits);

                let mtf_weights_pos_other = st.mtf_weights.search(other_bits, other_weight_mask);
                let mtf_endpoints_pos_other =
                    st.mtf_endpoints.search(other_bits, !other_weight_mask);

                let other_err = eval_error(&mut st.err_calc, config, bsd, other_bits);
                if restricted_err == ERROR_CALC_DEFAULT {
                    restricted_err = other_err;
                }

                let other_rate = calculate_bit_cost_simple(
                    mtf_endpoints_pos_other,
                    mtf_weights_pos_other,
                    other_bits,
                    other_weight_bits,
                    &st.hist,
                    &st.log2_table,
                );
                let other_rd_cost = other_err + lambda * other_rate;

                if other_rd_cost < best_rd_cost {
                    best = other_bits;
                    best_rd_cost = other_rd_cost;
                }

                best_endpoints.add(other_bits, other_rd_cost, mtf_endpoints_pos_other, bsd);
            }

            // Find best simple endpoint candidates
            let simple_ref_block = block_at(other_encodings[0], local_index);
            if is_simple_block(simple_ref_block, bsd) {
                let mut scb = SymbolicCompressedBlock::default();

                // Decode reference block to determine target normalized_axis
                physical_to_symbolic(bsd, &simple_ref_block.to_le_bytes(), &mut scb);
                let ref_compact = CompactSimpleEndpoints::from_symbolic(&scb);

                let target_axis = normalized_axis(&ref_compact);
                let mut best_subst_err = ERROR_CALC_DEFAULT;

                for k in 0..st.mtf_simple_endpoints.size {
                    let entry = *st.mtf_simple_endpoints.get(k);

                    // Don't bother with endpoint substitution if the candidate's endpoint axis
                    // isn't at least somewhat aligned with our reference block. restricted_err
                    // (the error of the unconstrained 1-plane 1-subset encoding) is a proxy for
                    // how good these blocks can get: perfectly aligned axes are billed as 1x
                    // restricted_err, orthogonal ones as 2x; such candidates are interesting if
                    // not much worse than the best substitution found so far.
                    let quick_error_estimate =
                        (2.0 - entry.normalized_axis.dot_s(target_axis).abs()) * restricted_err;
                    if quick_error_estimate >= best_subst_err * 1.25 {
                        continue;
                    }

                    entry.endpoints.store_symbolic(&mut scb);
                    let bm = *bsd.get_block_mode(scb.block_mode as usize);

                    // Compute weights for our block using those endpoints
                    compute_known_endpoint_weights(blk, &entry, &mut tmpbuf.ei1);

                    // Compute ideal weights for our decimation mode
                    let di = bsd.get_decimation_info(bm.decimation_mode as usize);
                    let (dec_weights_ideal, dec_weights_ignored) =
                        tmpbuf.dec_weights_ideal.split_at_mut(BLOCK_MAX_WEIGHTS);
                    compute_ideal_weights_for_decimation(&tmpbuf.ei1, di, dec_weights_ideal);

                    // Quantize them
                    compute_quantized_weights_for_decimation(
                        di,
                        0.0,
                        1.0,
                        dec_weights_ideal,
                        dec_weights_ignored,
                        &mut tmpbuf.dec_weights_uquant,
                        bm.get_weight_quant_mode(),
                    );

                    // Copy quantized weights into scb
                    let weight_count = di.weight_count as usize;
                    scb.weights[..weight_count]
                        .copy_from_slice(&tmpbuf.dec_weights_uquant[..weight_count]);

                    // Determine the corresponding error
                    let mse = compute_symbolic_block_difference_1plane_1partition_weighted(
                        config,
                        bsd,
                        &scb,
                        blk,
                        block_texel_weights,
                    );

                    // No need to perform RD scoring if we blew past our target on the
                    // distortion term alone
                    if best_endpoints.quick_reject(mse) {
                        continue;
                    }

                    // Keep track of best substitution we've found so far
                    best_subst_err = mathlib::min(best_subst_err, mse);

                    // Emit the corresponding block so we have the bits for the rate calc
                    let mut candidate_bytes = [0u8; 16];
                    symbolic_to_physical(bsd, &scb, &mut candidate_bytes);
                    let candidate = u128::from_le_bytes(candidate_bytes);

                    // Calculate the rate
                    let endpoints_weight_bits = get_weight_bits(candidate, bsd);
                    let weights_mask = top_bits_mask(endpoints_weight_bits);

                    // Find the corresponding position in the MTF lists. If it's not actually
                    // in the endpoint MTF list anymore, skip (possible if e.g. a long run of
                    // distinct constant-color blocks crowds out the matched endpoints).
                    let endpoint_pos = st.mtf_endpoints.search(candidate, !weights_mask);
                    if endpoint_pos == -1 {
                        continue;
                    }

                    let weight_pos = st.mtf_weights.search(candidate, weights_mask);

                    let bit_cost = calculate_bit_cost_simple(
                        endpoint_pos,
                        weight_pos,
                        candidate,
                        endpoints_weight_bits,
                        &st.hist,
                        &st.log2_table,
                    );
                    let rd_cost = mse + lambda * bit_cost;
                    if rd_cost < best_rd_cost {
                        best = candidate;
                        best_rd_cost = rd_cost;
                    }

                    // Insert into best_endpoints if it's one of the best candidates
                    best_endpoints.add(candidate, rd_cost, endpoint_pos, bsd);
                }
            }

            // Find best weight candidates
            for k in 0..st.mtf_weights.size {
                let candidate_weights = st.mtf_weights.list[k];
                let weights_weight_bits = get_weight_bits(candidate_weights, bsd);
                if weights_weight_bits == 0 {
                    continue;
                }

                let weights_mask = top_bits_mask(weights_weight_bits);
                let just_weight_bits = candidate_weights & weights_mask;
                let endpoints_mask = !weights_mask;

                // Try every endpoint candidate that matches in weight bits
                for endpoints in best_endpoints.items() {
                    if weights_weight_bits == endpoints.weight_bits {
                        let combined_bits = just_weight_bits | (endpoints.bits & endpoints_mask);

                        let err = eval_error(&mut st.err_calc, config, bsd, combined_bits);
                        let bit_cost = calculate_bit_cost_simple(
                            endpoints.mtf_position,
                            k as i32,
                            combined_bits,
                            weights_weight_bits,
                            &st.hist,
                            &st.log2_table,
                        );
                        let rd_cost = err + lambda * bit_cost;
                        if rd_cost < best_rd_cost {
                            best = combined_bits;
                            best_rd_cost = rd_cost;
                        }
                    }
                }
            }

            // Write back best candidate
            if let Some(dst) = data.get_mut(
                local_index * ASTC_BLOCK_BYTES..local_index * ASTC_BLOCK_BYTES + ASTC_BLOCK_BYTES,
            ) {
                dst.copy_from_slice(&best.to_le_bytes());
            }

            // Recalculate masks for the best match
            let best_weights_mask = top_bits_mask(get_weight_bits(best, bsd));

            // Update statistics
            let best_mtf_weights_pos = st.mtf_weights.search(best, best_weights_mask);
            let best_mtf_endpoints_pos = st.mtf_endpoints.search(best, !best_weights_mask);
            st.mtf_weights.update(best, best_mtf_weights_pos);
            st.mtf_endpoints.update(best, best_mtf_endpoints_pos);
            st.hist.update(best);

            if is_simple_block(best, bsd) {
                // Simple blocks can be compactly encoded
                subst_mtf_encode_block(&mut st.mtf_simple_endpoints, bsd, best);
            }
        }
    }
}
}

isa_fn! {
/// Optimizes one chunk: copies its blocks out of `data_out`, runs [`optimize_chunk`] with
/// the shared restricted encodings and weights, and writes the blocks back.
#[allow(clippy::too_many_arguments)]
fn process_chunk(
    config: &AstcencConfig,
    bsd: &BlockSizeDescriptor,
    lz: &LzRdo,
    st: &mut PassState,
    tmpbuf: &mut CompressionWorkingBuffers,
    blk: &mut ImageBlock,
    (block_start, block_count): (usize, usize),
    blocks_xy: (usize, usize),
    lambda: f32,
    img: &AstcencImage,
    swz: &AstcencSwizzle,
    data_out: &AstcencOutput,
    (data, original_blocks): (&mut [u8], &mut [u8]),
) -> Result<(), AstcencError> {
    let texel_count = bsd.texel_count as usize;
    let range = block_start * ASTC_BLOCK_BYTES..(block_start + block_count) * ASTC_BLOCK_BYTES;
    let chunk_bytes = block_count * ASTC_BLOCK_BYTES;

    // The chunk's current encodings
    {
        let Ok(out) = data_out.lock() else {
            return Err(AstcencError::BadContext);
        };
        let (Some(src), Some(dst)) = (out.get(range.clone()), data.get_mut(..chunk_bytes)) else {
            return Err(AstcencError::BadContext);
        };
        dst.copy_from_slice(src);
    }

    {
        let (Ok(restricted), Ok(weights)) =
            (lz.restricted_blocks.read(), lz.per_texel_weights.read())
        else {
            return Err(AstcencError::BadContext);
        };
        let (Some(restricted), Some(weights)) = (
            restricted.get(range.clone()),
            weights.get(block_start * texel_count..),
        ) else {
            return Err(AstcencError::BadContext);
        };
        optimize_chunk(
            config,
            bsd,
            st,
            tmpbuf,
            blk,
            (block_start, block_count),
            blocks_xy,
            lambda,
            img,
            swz,
            data,
            original_blocks,
            restricted,
            weights,
        );
    }

    let Ok(mut out) = data_out.lock() else {
        return Err(AstcencError::BadContext);
    };
    let (Some(dst), Some(src)) = (out.get_mut(range), data.get(..chunk_bytes)) else {
        return Err(AstcencError::BadContext);
    };
    dst.copy_from_slice(src);
    Ok(())
}
}

isa_fn! {
/// `lz_rdo_optimize` (with `lz_rdo_optimization_passes`): every thread takes chunks of up to
/// 4096 blocks and revisits them, reusing previous endpoints and weights where the
/// rate-distortion score favours it. Requires the compression and the setup pass to be
/// complete; returns once all chunks are done. The progress callback reports `init_pct` to
/// 100% over the chunks.
#[allow(clippy::too_many_arguments)]
pub fn lz_rdo_optimize(
    compressor: &Compressor,
    config: &AstcencConfig,
    lz: &LzRdo,
    thread_index: usize,
    img: &AstcencImage,
    swz: &AstcencSwizzle,
    init_pct: f32,
    data_out: &AstcencOutput,
) -> Result<(), AstcencError> {
    let bsd = &*compressor.bsd;

    // Global lambda scaling fudge factor
    let mut lambda = config.lz_rdo_lambda;
    lambda *= 0.75;

    // Squared error in blocks scales with texel count; this makes lambda mostly independent
    // of block size
    lambda *= f32::from(bsd.texel_count) / 16.0;

    let result = optimization_passes(
        compressor,
        config,
        lz,
        (thread_index, lambda, init_pct),
        img,
        swz,
        data_out,
    );

    // Every chunk is complete here (or the passes failed and cancelled): release the buffers
    lz.manage_optimize.term(|| {
        if let Ok(mut v) = lz.restricted_blocks.write() {
            *v = Vec::new();
        }
        if let Ok(mut v) = lz.per_texel_weights.write() {
            *v = Vec::new();
        }
    });
    result
}
}

isa_fn! {
/// The per-thread chunk loop of `lz_rdo_optimization_passes`.
fn optimization_passes(
    compressor: &Compressor,
    config: &AstcencConfig,
    lz: &LzRdo,
    (thread_index, lambda, init_pct): (usize, f32, f32),
    img: &AstcencImage,
    swz: &AstcencSwizzle,
    data_out: &AstcencOutput,
) -> Result<(), AstcencError> {
    let bsd = &*compressor.bsd;

    let blocks_x = (img.dim_x as usize).div_ceil(bsd.dim_x as usize);
    let blocks_y = (img.dim_y as usize).div_ceil(bsd.dim_y as usize);
    let blocks_z = (img.dim_z as usize).div_ceil(bsd.dim_z as usize);

    let Some(tmpbuf) = compressor.working_buffers.get(thread_index) else {
        return Err(AstcencError::BadContext);
    };
    let Ok(mut tmpbuf) = tmpbuf.lock() else {
        return Err(AstcencError::BadContext);
    };
    let tmpbuf = &mut *tmpbuf;

    // Set up block re-encoding parameters
    let mut blk = ImageBlock {
        texel_count: bsd.texel_count,
        decode_unorm8: config.flags & ASTCENC_FLG_USE_DECODE_UNORM8 != 0,
        ..Default::default()
    };

    // Unlike normal encoding, we care about the absolute scale of errors, not just their
    // relative values, so normalize channel_weight to always have a sum of 4 (which is what
    // we get with the default of all-1s)
    let cw = Vfloat4::new(
        config.cw_r_weight,
        config.cw_g_weight,
        config.cw_b_weight,
        config.cw_a_weight,
    );
    blk.channel_weight = cw * Vfloat4::splat(4.0 / cw.hadd_s());

    // Allocate thread-local resources
    let mut data = vec![0u8; MAX_BLOCKS_PER_ITEM * ASTC_BLOCK_BYTES];
    let mut original_blocks = vec![0u8; MAX_BLOCKS_PER_ITEM * ASTC_BLOCK_BYTES];
    let endpoint_history = config.tune_lz_rdo_endpoint_history_size as usize;
    let mut st = PassState {
        mtf_weights: MtfList::new(config.tune_lz_rdo_weight_history_size as usize),
        // Endpoints and constant block bits
        mtf_endpoints: MtfList::new(endpoint_history * 2),
        mtf_simple_endpoints: EndpointSubstMtf::new(endpoint_history, config.profile),
        hist: ModeByteHistogram::new(),
        err_calc: BlockErrorCalculator::new(CACHE_SIZE, bsd),
        log2_table: mtf_log2_table(
            (endpoint_history * 2).max(config.tune_lz_rdo_weight_history_size as usize),
        ),
    };

    lz.manage_optimize.init_progress(
        blocks_x * blocks_y * blocks_z,
        config.progress_callback.as_ref(),
        init_pct,
        100.0,
    );

    // Main thread loop to process work items from the queue
    let mut result = Ok(());
    loop {
        let (block_start, block_count) =
            lz.manage_optimize.get_task_assignment(MAX_BLOCKS_PER_ITEM);
        if block_count == 0 {
            break;
        }
        let done = process_chunk(
            config,
            bsd,
            lz,
            &mut st,
            tmpbuf,
            &mut blk,
            (block_start, block_count),
            (blocks_x, blocks_y),
            lambda,
            img,
            swz,
            data_out,
            (&mut data, &mut original_blocks),
        );
        if done.is_err() {
            lz.manage_optimize.cancel();
            result = done;
        }
        lz.manage_optimize.complete_task_assignment(block_count);
        if result.is_err() {
            break;
        }
    }

    lz.manage_optimize.wait();
    result
}
}

/// A convolution filter kernel: `2 * radius + 1` taps.
struct FilterKernel {
    radius: usize,
    coeffs: [f32; 15],
}

/// Gaussian kernel with sigma=2.20.
#[allow(clippy::excessive_precision)]
static KERNEL_INITIAL_LPF: FilterKernel = FilterKernel {
    radius: 7,
    coeffs: [
        1.14907966e-03,
        4.40146100e-03,
        1.37123950e-02,
        3.47455753e-02,
        7.16069925e-02,
        1.20027593e-01,
        1.63635110e-01,
        1.81443588e-01,
        1.63635110e-01,
        1.20027593e-01,
        7.16069925e-02,
        3.47455753e-02,
        1.37123950e-02,
        4.40146100e-03,
        1.14907966e-03,
    ],
};

/// Gaussian kernel with sigma=1.25.
#[allow(clippy::excessive_precision)]
static KERNEL_SPREAD: FilterKernel = FilterKernel {
    radius: 4,
    coeffs: [
        1.90769133e-03,
        1.79195767e-02,
        8.87562444e-02,
        2.31804370e-01,
        3.19224234e-01,
        2.31804370e-01,
        8.87562444e-02,
        1.79195767e-02,
        1.90769133e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
};

isa_fn! {
/// `apply_1d_convolution_3d`: a convolution along X (0), Y (1) or Z (2) of a
/// `xdim * ydim * zdim` array (with SIMD padding after it). Along X, spans that fit the
/// unclipped kernel are filtered 8 at a time without renormalization and the other texels one
/// at a time, normalized by the kernel weights inside the line, as in the AVX2 build.
fn apply_1d_convolution_3d(
    input: &mut [f32],
    output: &mut [f32],
    (xdim, ydim, zdim): (usize, usize, usize),
    filter: &FilterKernel,
    direction: u32,
) {
    let radius = filter.radius;
    let coeff = |k: isize| filter.coeffs[(k + radius as isize) as usize];
    let r = radius as isize;

    if direction == 0 {
        // The first and last radius pixels of each line include border pixels; the safe
        // region shrinks by SIMD_WIDTH - 1 so the last pixel of a span stays inside it
        let excluded_region_width = 2 * radius + SIMD_WIDTH - 1;
        let safe_width = xdim - excluded_region_width.min(xdim);

        for yz in 0..ydim * zdim {
            let mut x = 0;
            while x < xdim {
                let pixel_index = yz * xdim + x;

                // Unsigned subtract makes one-sided test work
                if x.wrapping_sub(radius) < safe_width {
                    let mut sum = Vfloat8::zero();
                    for k in -r..=r {
                        let src = pixel_index.wrapping_add_signed(k);
                        sum += Vfloat8::splat(coeff(k)) * Vfloat8::load(input.get(src..).unwrap_or_default());
                    }
                    sum.store(output.get_mut(pixel_index..).unwrap_or_default());
                    x += SIMD_WIDTH;
                } else {
                    // Near boundary, work one pixel at a time
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;
                    for k in -r..=r {
                        // Negatives wrap around, which shows up as >= width
                        let sx = x.wrapping_add_signed(k);
                        if sx < xdim {
                            let weight = coeff(k);
                            weight_sum += weight;
                            sum += weight
                                * input.get(pixel_index.wrapping_add_signed(k)).copied().unwrap_or(0.0);
                        }
                    }

                    // Normalize by the actual sum of kernel weights used
                    sum /= weight_sum;
                    if let Some(o) = output.get_mut(pixel_index) {
                        *o = sum;
                    }
                    x += 1;
                }
            }
        }
    } else {
        // Direction is Y or Z
        let stride = if direction == 1 { xdim } else { xdim * ydim };
        let max_in_dir = if direction == 1 { ydim - 1 } else { zdim - 1 };

        // Clear input padding
        let total = xdim * ydim * zdim;
        Vfloat8::zero().store(input.get_mut(total..).unwrap_or_default());

        for z in 0..zdim {
            for y in 0..ydim {
                // The kernel clamping and sum are the same for the whole scan line
                let pos_in_dir = if direction == 1 { y } else { z };

                // Clamp kernel to active bounds
                let k0 = -(pos_in_dir.min(radius) as isize);
                let k1 = (max_in_dir - pos_in_dir).min(radius) as isize;

                // If we cut off pixels, determine normalization factor
                let mut overall_scale = Vfloat8::splat(1.0);
                if k0 != -r || k1 != r {
                    let mut kernel_sum = 0.0f32;
                    for k in k0..=k1 {
                        kernel_sum += coeff(k);
                    }
                    overall_scale = Vfloat8::splat(1.0 / kernel_sum);
                }

                // Process the pixels; this runs off the ends of scan lines (into the next line,
                // which is written later, or the padding)
                let row_index = (z * ydim + y) * xdim;
                let mut x = 0;
                while x < xdim {
                    let pixel_index = row_index + x;
                    let mut sum = Vfloat8::zero();
                    for k in k0..=k1 {
                        let src = pixel_index.wrapping_add_signed(k * stride as isize);
                        sum += Vfloat8::splat(coeff(k)) * Vfloat8::load(input.get(src..).unwrap_or_default());
                    }
                    (overall_scale * sum).store(output.get_mut(pixel_index..).unwrap_or_default());
                    x += SIMD_WIDTH;
                }
            }
        }
    }
}
}

isa_fn! {
/// `separable_convolve_3d`: X, Y (and Z) passes of `filter`.
fn separable_convolve_3d(
    input: &mut [f32],
    output: &mut [f32],
    workspace: &mut [f32],
    dims: (usize, usize, usize),
    filter: &FilterKernel,
) {
    if dims.2 > 1 {
        // 3D: X, Y, then Z
        apply_1d_convolution_3d(input, output, dims, filter, 0);
        apply_1d_convolution_3d(output, workspace, dims, filter, 1);
        apply_1d_convolution_3d(workspace, output, dims, filter, 2);
    } else {
        // 2D only: X then Y
        apply_1d_convolution_3d(input, workspace, dims, filter, 0);
        apply_1d_convolution_3d(workspace, output, dims, filter, 1);
    }
}
}

/// Maximum number of texels in a tile.
const WEIGHTS_TILE_MAX_TEXELS: usize = 256 * 256;

/// Tile buffer size, including padding for SIMD over-fetch (and, upstream, against
/// power-of-two aliasing).
const WEIGHTS_TILE_BUFFER_TEXELS: usize = WEIGHTS_TILE_MAX_TEXELS + 128 + SIMD_WIDTH;

/// `weights_tile_buffers`: tile buffers of the per-texel weight calculation.
struct WeightsTileBuffers {
    /// Input pixels by color channel.
    chan: [Vec<f32>; 4],
    /// Filter work space.
    work: Vec<f32>,
    /// Filter result space.
    filtered: Vec<f32>,
}

isa_fn! {
/// `lz_rdo_setup`: allocates the shared LZ-RDO buffers (first thread), then every thread
/// takes independent tiles (about 256x256 texels, block aligned) and computes per-texel
/// weights that weight down regions with a lot of high-frequency energy: quantization errors
/// are much more visible in smooth regions (flat areas or slow gradients) than where the
/// image is noisy or has sharp edges. An allocation failure sets `lz.alloc_failed`.
pub fn lz_rdo_setup(
    compressor: &Compressor,
    config: &AstcencConfig,
    lz: &LzRdo,
    img: &AstcencImage,
    swz: &AstcencSwizzle,
) {
    let bsd = &*compressor.bsd;
    let (dim_x, dim_y, dim_z) = (bsd.dim_x as usize, bsd.dim_y as usize, bsd.dim_z as usize);
    let texel_count = bsd.texel_count as usize;
    let (img_x, img_y, img_z) = (img.dim_x as usize, img.dim_y as usize, img.dim_z as usize);

    // Choose tile size to be an integer multiple of block size
    let (tile_texels_x, tile_texels_y, tile_texels_z) = if dim_z == 1 {
        // 2D blocks - use approximately 256x256 tiles
        ((256 / dim_x) * dim_x, (256 / dim_y) * dim_y, 1)
    } else {
        // 3D blocks - use approximately 32x32x32 tiles
        ((32 / dim_x) * dim_x, (32 / dim_y) * dim_y, (32 / dim_z) * dim_z)
    };

    // Calculate number of tiles needed in each dimension
    let tiles_x = img_x.div_ceil(tile_texels_x);
    let tiles_y = img_y.div_ceil(tile_texels_y);
    let tiles_z = img_z.div_ceil(tile_texels_z);
    let total_tile_count = tiles_x * tiles_y * tiles_z;

    // Image dimensions in blocks
    let img_blocks_x = img_x.div_ceil(dim_x);
    let img_blocks_y = img_y.div_ceil(dim_y);
    let num_blocks = img_blocks_x * img_blocks_y * img_z.div_ceil(dim_z);

    // Only the first thread actually runs the initializer
    lz.manage_setup.init_with(|| {
        let restricted = num_blocks.checked_mul(ASTC_BLOCK_BYTES);
        let weights = num_blocks
            .checked_mul(texel_count)
            .and_then(|n| n.checked_add(SIMD_WIDTH));
        let ok = match (restricted, weights, lz.restricted_blocks.write(), lz.per_texel_weights.write()) {
            (Some(rn), Some(wn), Ok(mut r), Ok(mut w)) => {
                r.clear();
                w.clear();
                let ok = r.try_reserve_exact(rn).is_ok() && w.try_reserve_exact(wn).is_ok();
                if ok {
                    r.resize(rn, 0);
                    w.resize(wn, 0.0);
                }
                ok
            }
            _ => false,
        };
        lz.alloc_failed.store(!ok, std::sync::atomic::Ordering::Relaxed);
        if ok { total_tile_count } else { 0 }
    });

    // Thread-local buffers for tile data. Allocated on demand, in case there are more workers
    // than tiles to work on.
    let mut buffers: Option<WeightsTileBuffers> = None;
    let mut tile_weights = Vec::new();

    // Set up image block
    let mut blk = ImageBlock {
        texel_count: bsd.texel_count,
        decode_unorm8: config.flags & ASTCENC_FLG_USE_DECODE_UNORM8 != 0,
        ..Default::default()
    };

    loop {
        // Grab next tile
        let (current_tile_index, tile_count) = lz.manage_setup.get_task_assignment(1);
        if tile_count == 0 {
            break;
        }

        let buffers = buffers.get_or_insert_with(|| WeightsTileBuffers {
            chan: [
                vec![0.0; WEIGHTS_TILE_BUFFER_TEXELS],
                vec![0.0; WEIGHTS_TILE_BUFFER_TEXELS],
                vec![0.0; WEIGHTS_TILE_BUFFER_TEXELS],
                vec![0.0; WEIGHTS_TILE_BUFFER_TEXELS],
            ],
            work: vec![0.0; WEIGHTS_TILE_BUFFER_TEXELS],
            filtered: vec![0.0; WEIGHTS_TILE_BUFFER_TEXELS],
        });

        // Determine tile coordinates and bounds
        let (tx, ty, tz) = decode_index_to_coords(current_tile_index, tiles_x, tiles_y);
        let tile_xpos = tx * tile_texels_x;
        let tile_ypos = ty * tile_texels_y;
        let tile_zpos = tz * tile_texels_z;

        // Number of active texels in a tile, rounded up to an integer block multiple
        let tile_xdim = tile_texels_x.min(img_x - tile_xpos).div_ceil(dim_x) * dim_x;
        let tile_ydim = tile_texels_y.min(img_y - tile_ypos).div_ceil(dim_y) * dim_y;
        let tile_zdim = tile_texels_z.min(img_z - tile_zpos).div_ceil(dim_z) * dim_z;
        let dims = (tile_xdim, tile_ydim, tile_zdim);

        // Load input pixels
        for bz in (0..tile_zdim).step_by(dim_z) {
            for by in (0..tile_ydim).step_by(dim_y) {
                for bx in (0..tile_xdim).step_by(dim_x) {
                    load_image_block(
                        config.profile,
                        img,
                        &mut blk,
                        bsd,
                        tile_xpos + bx,
                        tile_ypos + by,
                        tile_zpos + bz,
                        swz,
                    );

                    // Copy complete blocks; the loader extrudes texels beyond the image edge
                    let mut src_index = 0;
                    for z in 0..dim_z {
                        for y in 0..dim_y {
                            let dst_index = ((bz + z) * tile_ydim + (by + y)) * tile_xdim + bx;
                            for c in 0..4 {
                                let src = &blk.data(c)[src_index..src_index + dim_x];
                                buffers.chan[c][dst_index..dst_index + dim_x].copy_from_slice(src);
                            }
                            src_index += dim_x;
                        }
                    }
                }
            }
        }

        let tile_texels = tile_xdim * tile_ydim * tile_zdim;
        let padded_texels = round_up_to_simd_multiple_vla(tile_texels);

        // Compute the low-pass filtered input and, from there, the high-pass energy
        let WeightsTileBuffers { chan, work, filtered } = buffers;
        for c in 0..4 {
            separable_convolve_3d(&mut chan[c], filtered, work, dims, &KERNEL_INITIAL_LPF);

            // Clear padding
            chan[c][tile_texels..tile_texels + SIMD_WIDTH].fill(0.0);
            filtered[tile_texels..tile_texels + SIMD_WIDTH].fill(0.0);

            // Compute highpass energy (squared difference between original and low-pass
            // filtered pixels), stored in the space for channel 0 which is freed up as we're
            // going along
            if c == 0 {
                for i in 0..padded_texels {
                    let diff = chan[0][i] - filtered[i];
                    chan[0][i] = 0.0 + diff * diff;
                }
            } else {
                let (hp_energy, rest) = chan.split_at_mut(1);
                let channel = &rest[c - 1];
                for i in 0..padded_texels {
                    let diff = channel[i] - filtered[i];
                    hp_energy[0][i] += diff * diff;
                }
            }
        }

        // Apply second blur to spread the high-pass energy around a bit
        separable_convolve_3d(&mut chan[0], filtered, work, dims, &KERNEL_SPREAD);

        // Clear padding
        filtered[tile_texels..tile_texels + SIMD_WIDTH].fill(0.0);

        // Transform into final activity weights
        let c1 = 1.0f32 / 257.0;
        let c2 = 257.0f32 / 256.0;
        let activity_scale = 4.0f32 / 255.0;
        for v in filtered[..padded_texels].iter_mut() {
            *v = c1 / (c2 + activity_scale * v.sqrt());
        }

        // Copy to output in block layout; texels outside image bounds get zero weight
        let tile_bx = tile_xpos / dim_x;
        let tile_by = tile_ypos / dim_y;
        let tile_bz = tile_zpos / dim_z;
        let tile_blocks_x = tile_xdim / dim_x;
        let tile_blocks_y = tile_ydim / dim_y;
        let tile_blocks_z = tile_zdim / dim_z;

        tile_weights.clear();
        for bz in 0..tile_blocks_z {
            for by in 0..tile_blocks_y {
                for bx in 0..tile_blocks_x {
                    for z in 0..dim_z {
                        let zt = bz * dim_z + z;
                        for y in 0..dim_y {
                            let yt = by * dim_y + y;
                            let src_row_idx = (zt * tile_ydim + yt) * tile_xdim;
                            for x in 0..dim_x {
                                let xt = bx * dim_x + x;
                                let inside = tile_xpos + xt < img_x
                                    && tile_ypos + yt < img_y
                                    && tile_zpos + zt < img_z;
                                tile_weights.push(if inside {
                                    filtered[src_row_idx + xt]
                                } else {
                                    0.0
                                });
                            }
                        }
                    }
                }
            }
        }

        if let Ok(mut block_texel_weights) = lz.per_texel_weights.write() {
            let mut src = tile_weights.chunks_exact(texel_count);
            for bz in 0..tile_blocks_z {
                for by in 0..tile_blocks_y {
                    for bx in 0..tile_blocks_x {
                        let block_index = ((bz + tile_bz) * img_blocks_y + (by + tile_by))
                            * img_blocks_x
                            + (bx + tile_bx);
                        let start = block_index * texel_count;
                        if let (Some(dst), Some(s)) =
                            (block_texel_weights.get_mut(start..start + texel_count), src.next())
                        {
                            dst.copy_from_slice(s);
                        }
                    }
                }
            }
        }

        lz.manage_setup.complete_task_assignment(tile_count);
    }
}
}
