// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of the Basis Universal v2.50 XBC7 encoder (`encoder/basisu_xbc7_encode.cpp`,
//! `xbc7::pack_image`): the XUBC7 slice stream for one RGBA8 image, byte-identical to C++.
//!
//! Pipeline: BC7 base pack (`bc7f` or `bc7e_scalar`, rows in parallel) + endpoint
//! canonicalization (+ optional weight re-optimization and alt-pack RDO), then per stripe
//! (stripes in parallel; each only references its own block rows): block-reuse RDO, endpoint
//! RDO and the coding pass (commands, configs, endpoint DPCM sweep, weight DCT vs lossless DPCM
//! with the predictor bank shared with the decoder). Stripe streams are merged in order with a
//! seek table, every blob is Zstandard compressed (`zstd-pure-rs`, libzstd 1.5.7 output) where
//! it pays, and a raw "tiny mip" stream replaces the blob container when smaller. The stream is
//! decoded again and checked block for block (`m_self_validate`).
//!
//! Integer and Q15.16 fixed point everywhere except the block PSNR of the RDO passes, which
//! follows the C++ `f32` operation order (`log10f` / `sqrtf` from the platform libm, as C++).
//! Statistics, debug output and debug images are not ported (they never affect the output).

#![allow(clippy::needless_range_loop)]

use super::decoder::{
    self, Ac, BLOB_AC_COEFFS, BLOB_COEFF_SIGNS, BLOB_COMMANDS, BLOB_CONFIG, BLOB_DC_COEFFS_SMALL,
    BLOB_DPCM_WEIGHT_RESID2, BLOB_EP_BLOCK_INDEX, BLOB_EP_DELTA_COARSE_R, BLOB_EP_DELTA_FINE_R,
    BLOB_EP_RAW, BLOB_HEADER, BLOB_PARTITION2, BLOB_PARTITION3, BLOB_PBITS, BLOB_RAW_WEIGHT_BITS,
    BLOB_SOLID_RGBA_DELTAS, BLOB_STRIPE_SEEK_TABLE, BLOB_WEIGHT_PREDICTORS, CAND_ABSOLUTE,
    CAND_DDL, CAND_DDR, CAND_DIAG_AVG, CAND_DIAG_EDGE_BLEND, CAND_FIRST_XY_DELTA, CAND_GAB,
    CAND_GRADIENT, CAND_GRADIENT_DAMPED, CAND_LEFT_EDGE, CAND_LU_AVG, CAND_LU_BLEND,
    CAND_LU_BLEND_STRONG, CAND_MED, CAND_PLANE_FIT, CAND_REFLECT_LEFT, CAND_REFLECT_UPPER,
    CAND_UPPER_DIAG_EDGE_BLEND, CAND_UPPER_EDGE, DCT_C1, DCT_C3, DCT_HALF, DEADZONE_ALPHA, Fx,
    MAGIC_BEGIN, MAGIC_END, MAX_STRIPES, TOTAL_CANDIDATES, Tile, XY_DELTAS, ZIGZAG,
    compute_level_scale, compute_quant_table, dct_inverse, eval_weight_predictor, max_span_len,
};
use crate::basis::ll::bc7f;
use crate::basis::ll::bc7u::{
    self, ENDPOINT_FORMATS, LogBc7Block, create_solid_blk, dequant_weight, quant_weight,
};
use crate::basis::ll::uastc_tables::{BC7_PARTITION2, BC7_PARTITION3};
use crate::buffer::Buffer;
use crate::{Error, Result};
use zstd_pure_rs::prelude::{ZSTD_compress, ZSTD_compressBound, ZSTD_isError};

type Px = [u8; 4];
type Blocks = vas::Vector<LogBc7Block, vas::storage::Dynamic>;

/// `xbc7::DEFAULT_EFFORT_LEVEL`.
pub(crate) const DEFAULT_EFFORT_LEVEL: u32 = 3;
/// `xbc7::DEFAULT_BC7E_SCALAR_LEVEL`.
pub(crate) const DEFAULT_BC7E_LEVEL: u32 = 2;
/// `xbc7::BC7E_SCALAR_MAX_LEVEL`.
pub(crate) const BC7E_MAX_LEVEL: u32 = 6;
/// Zstandard level of `blob_stream_writer::serialize` (hardcoded 19 in C++).
pub(crate) const DEFAULT_ZSTD_LEVEL: i32 = 19;

const MIN_IMAGE_TEXEL_ROWS_TO_STRIPE: u32 = 128;
const MIN_STRIPE_BLOCK_ROWS: u32 = 16;
const SWEEP_SWITCH_MARGIN_PERMILLE: u64 = 30;
const ERR_SWITCH_MARGIN_PERMILLE: u64 = 20;
const NUM_BLOBS: usize = BLOB_STRIPE_SEEK_TABLE + 1;
/// `INT16_MAX` end-of-block AC marker.
const AC_EOB: i32 = i16::MAX as i32;
const TOTAL_CANDS: usize = TOTAL_CANDIDATES as usize;

/// `g_dpcm_resid_cost_obits`: [weight bits - 2][min(|centered residual|, 8)], 1/8 bits.
const DPCM_RESID_COST_OBITS: [[u64; 9]; 3] = [
    [8, 17, 47, 47, 47, 47, 47, 47, 47],
    [12, 19, 27, 38, 63, 63, 63, 63, 63],
    [12, 25, 29, 33, 38, 44, 52, 61, 81],
];
const DCT_DC_BYTE_COST_OBITS: u64 = 38;
const DCT_AC_BYTE_COST_OBITS: u64 = 28;

/// BC7 base encoder (`xbc7::bc7_encoder_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bc7Encoder {
    /// Real-time analytical `bc7f` packer (`cBC7F`).
    Bc7f,
    /// `bc7e_scalar` at [`EncodeParams::bc7e_level`] (`cBC7E_Scalar`).
    Bc7eScalar,
}

/// Encoder stripe count (decode parallelism vs. size).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stripes {
    /// `m_num_stripes = 0`: derived from the image height (`compute_encoder_stripes`).
    Auto,
    /// `set_num_stripes_for_image(desired)` (the basisu compressor): at least 16 block rows per
    /// stripe, one stripe below 128 texel rows.
    Desired(u32),
    /// `m_num_stripes = n` (0 = `Auto`): clamped to [1, min(block rows, 16)] only.
    #[cfg_attr(not(test), allow(dead_code))] // used by the golden tests
    Exact(u32),
}

/// `xbc7::pack_options` (debug / stats knobs omitted). [`Default`] is the basisu CLI `-xubc7`
/// default (see [`BasisuParams`]); [`EncodeParams::pack_options`] the plain struct defaults.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EncodeParams {
    /// Weight DCT quality [1,100]; 100 = lossless weights (always residual DPCM).
    pub dct_q: u32,
    /// Per-channel RGBA error weights of every candidate evaluation (and linear bc7e).
    pub weights: [u32; 4],
    /// Effort [0,10]: weight predictor / amp code pruning of the per-block sweeps.
    pub effort_level: u32,
    /// Exhaustively re-derive every base block's weights for its endpoints.
    pub optimize_weights_after_bc7f: bool,
    /// `bc7f` flags of the primary base pack.
    pub bc7_pack_flags: u32,
    pub bc7_encoder: Bc7Encoder,
    /// `bc7e_scalar` preset [0,6] (ultrafast .. slowest).
    pub bc7e_level: u32,
    /// sRGB source: bc7e runs perceptual (ignoring `weights`), else linear with `weights`.
    pub perceptual: bool,
    /// Decode the produced stream and verify every block round-trips.
    pub self_validate: bool,
    /// Alt-pack RDO: also pack with `bc7_pack_flags_alt`, keep it within the PSNR drop.
    pub bc7_alt_pack_enabled: bool,
    pub bc7_pack_flags_alt: u32,
    pub bc7_alt_max_psnr_drop: f32,
    /// Block-reuse RDO: Repeat of the left / upper neighbor.
    pub repeat_rdo_enabled: bool,
    pub repeat_rdo_max_psnr_drop: f32,
    /// Block-reuse RDO: the block's mean color.
    pub solid_rdo_enabled: bool,
    pub solid_rdo_max_psnr_drop: f32,
    /// Endpoint RDO: zero-residual endpoint DPCM from a causal neighbor.
    pub endpoint_rdo_enabled: bool,
    pub endpoint_rdo_max_psnr_drop: f32,
    /// Weight DCT AC truncation RDO (0 = off).
    pub ac_trunc_rdo_max_psnr_drop: f32,
    /// Absolute block PSNR floor of every RDO candidate.
    pub rdo_min_block_psnr: f32,
    /// DPCM wins when `dpcm_cost * 100 <= dct_cost * alpha`.
    pub wt_dpcm_alpha_pct: u32,
    pub stripes: Stripes,
    /// Zstandard level of every blob.
    pub zstd_level: i32,
}

impl EncodeParams {
    /// `xbc7::pack_options` member defaults.
    pub(crate) fn pack_options() -> Self {
        EncodeParams {
            dct_q: 100,
            weights: [1; 4],
            effort_level: DEFAULT_EFFORT_LEVEL,
            optimize_weights_after_bc7f: false,
            bc7_pack_flags: bc7f::PACK_BC7_FLAG_DEFAULT_PARTIALLY_ANALYTICAL,
            bc7_encoder: Bc7Encoder::Bc7f,
            bc7e_level: DEFAULT_BC7E_LEVEL,
            perceptual: false,
            self_validate: true,
            bc7_alt_pack_enabled: false,
            bc7_pack_flags_alt: bc7f::PACK_BC7_FLAG_PBIT_OPT
                | bc7f::PACK_BC7_FLAG_USE_TRIVIAL_MODE6,
            bc7_alt_max_psnr_drop: 0.5,
            repeat_rdo_enabled: false,
            repeat_rdo_max_psnr_drop: 0.5,
            solid_rdo_enabled: false,
            solid_rdo_max_psnr_drop: 0.5,
            endpoint_rdo_enabled: false,
            endpoint_rdo_max_psnr_drop: 0.5,
            ac_trunc_rdo_max_psnr_drop: 0.0,
            rdo_min_block_psnr: 33.0,
            wt_dpcm_alpha_pct: 100,
            stripes: Stripes::Auto,
            zstd_level: DEFAULT_ZSTD_LEVEL,
        }
    }

    /// `pack_options::set_rdo_level`: all RDO knobs from one level [0,100] (0 = off).
    pub(crate) fn set_rdo_level(&mut self, rdo_level: u32) {
        let rdo_level = rdo_level.min(100);
        if rdo_level == 0 {
            self.bc7_alt_pack_enabled = false;
            self.endpoint_rdo_enabled = false;
            self.repeat_rdo_enabled = false;
            self.solid_rdo_enabled = false;
            self.bc7_alt_max_psnr_drop = 0.0;
            self.endpoint_rdo_max_psnr_drop = 0.0;
            self.repeat_rdo_max_psnr_drop = 0.0;
            self.solid_rdo_max_psnr_drop = 0.0;
            self.ac_trunc_rdo_max_psnr_drop = 0.0;
            return;
        }
        let frac = rdo_level as f32 / 100.0;
        let general_drop = 10.0 * frac;
        let reuse_drop = 4.0 * frac;
        self.bc7_alt_pack_enabled = true;
        self.bc7_alt_max_psnr_drop = general_drop;
        self.endpoint_rdo_enabled = true;
        self.endpoint_rdo_max_psnr_drop = general_drop;
        self.ac_trunc_rdo_max_psnr_drop = general_drop;
        self.repeat_rdo_enabled = true;
        self.repeat_rdo_max_psnr_drop = reuse_drop;
        self.solid_rdo_enabled = true;
        self.solid_rdo_max_psnr_drop = reuse_drop;
    }

    /// `basis_compressor::encode_slices_to_xubc7` option setup.
    pub(crate) fn from_basisu(b: &BasisuParams) -> Self {
        let mut o = Self::pack_options();
        o.dct_q = if b.quality < 0 {
            100
        } else {
            b.quality.clamp(1, 100) as u32
        };
        o.set_rdo_level(b.rdo_level);
        o.weights = b.weights;
        let effort = b.effort_level.min(10);
        if o.dct_q >= 90 && effort >= 3 {
            o.optimize_weights_after_bc7f = true;
        }
        if effort <= 1 {
            o.bc7_pack_flags = bc7f::PACK_BC7_FLAG_DEFAULT;
        } else if effort == 10 {
            o.bc7_pack_flags = bc7f::PACK_BC7_FLAG_DEFAULT_NON_ANALYTICAL;
        }
        o.effort_level = effort;
        o.bc7_encoder = b.encoder;
        o.bc7e_level = b.bc7e_level.min(BC7E_MAX_LEVEL);
        o.perceptual = b.perceptual;
        o.stripes = Stripes::Desired(b.num_stripes);
        o
    }
}

impl Default for EncodeParams {
    /// The basisu CLI `-xubc7` default (`BasisuParams::default()`).
    fn default() -> Self {
        Self::from_basisu(&BasisuParams::default())
    }
}

/// The XUBC7 subset of `basis_compressor_params` (CLI defaults in [`Default`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BasisuParams {
    /// `m_quality_level` / `-quality` [1,100]; negative = unset (lossless, Q 100).
    pub quality: i32,
    /// `m_xubc7_effort_level` / `-effort` [0,10].
    pub effort_level: u32,
    /// `m_xubc7_rdo_level` / `-xubc7_rdo` [0,100].
    pub rdo_level: u32,
    /// `m_xubc7_num_stripes` / `-xubc7_stripes` [1,16] (desired; clamped per image).
    pub num_stripes: u32,
    /// `m_xubc7_encoder` / `-xubc7_bc7f`, `-xubc7_bc7e`.
    pub encoder: Bc7Encoder,
    /// `m_xubc7_bc7e_scalar_level` / `-xubc7_bc7e_scalar_level` [0,6].
    pub bc7e_level: u32,
    /// `m_ldr_channel_weights` (9,11,1,11 sRGB / 1,1,1,1 linear).
    pub weights: [u32; 4],
    /// `m_perceptual` (`-srgb` / `-linear`).
    pub perceptual: bool,
}

impl BasisuParams {
    /// `-srgb` (true, the default) or `-linear` channel weights and perceptual flag.
    #[cfg_attr(not(test), allow(dead_code))] // used by the golden tests
    pub(crate) fn set_srgb(&mut self, srgb: bool) {
        self.weights = if srgb { [9, 11, 1, 11] } else { [1; 4] };
        self.perceptual = srgb;
    }
}

impl Default for BasisuParams {
    fn default() -> Self {
        BasisuParams {
            quality: -1,
            effort_level: DEFAULT_EFFORT_LEVEL,
            rdo_level: 0,
            num_stripes: 8,
            encoder: Bc7Encoder::Bc7f,
            bc7e_level: DEFAULT_BC7E_LEVEL,
            weights: [9, 11, 1, 11],
            perceptual: true,
        }
    }
}

/// An encoded XBC7 stream plus the logical BC7 blocks it decodes to (row major).
pub(crate) struct Encoded {
    pub data: Buffer,
    pub blocks: Blocks,
}

/// Encodes a tightly packed RGBA8 image to an XBC7 slice stream (`xbc7::pack_image`).
#[cfg_attr(not(test), allow(dead_code))] // used by the golden tests
pub(crate) fn encode(
    rgba: &[u8],
    width: u32,
    height: u32,
    params: &EncodeParams,
) -> Result<Buffer> {
    Ok(encode_blocks(rgba, width, height, params)?.data)
}

// ---------------------------------------------------------------------------------------------
// Small helpers.

/// `basisu::clamp` on u32 (never panics, unlike `Ord::clamp` with `lo > hi`).
#[inline]
fn clampu(v: u32, lo: u32, hi: u32) -> u32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline]
fn push(buf: &mut Buffer, b: u8) -> Result<()> {
    buf.push(b).map_err(|_| Error::OutOfMemory)
}

#[inline]
fn extend(buf: &mut Buffer, b: &[u8]) -> Result<()> {
    buf.extend_from_slice(b).map_err(|_| Error::OutOfMemory)
}

/// `color_rgba::get_weighted_dist2` (int math, result as uint32).
#[inline]
fn weighted_dist2(a: &Px, b: &Px, w: &[u32; 4]) -> u32 {
    let mut s = 0i32;
    for c in 0..4 {
        let d = i32::from(a[c]) - i32::from(b[c]);
        s = s.wrapping_add((d * d).wrapping_mul(w[c] as i32));
    }
    s as u32
}

/// `xbc7_block_wsse_psnr`: weighted RGBA PSNR of a 4x4 block.
fn block_wsse_psnr(orig: &[Px; 16], dec: &[Px; 16], w: &[u32; 4]) -> f32 {
    let wsse: u64 = orig
        .iter()
        .zip(dec)
        .map(|(o, d)| u64::from(weighted_dist2(d, o, w)))
        .sum();
    let total = w[0]
        .wrapping_add(w[1])
        .wrapping_add(w[2])
        .wrapping_add(w[3]) as f32;
    let wmse = wsse as f32 / (total * 16.0);
    if wmse > 1e-5 {
        20.0 * (255.0 / wmse.sqrt()).log10()
    } else {
        10000.0
    }
}

#[inline]
fn unpack(log: &LogBc7Block) -> Result<[Px; 16]> {
    let mut px = [[0u8; 4]; 16];
    if bc7u::unpack_pixels(log, &mut px) {
        Ok(px)
    } else {
        Err(Error::InvalidData)
    }
}

/// `get_decoded_channel_weight_plane`.
#[inline]
fn decoded_channel_weight_plane(log: &LogBc7Block, c: usize) -> usize {
    if !log.is_dual_plane() {
        return 0;
    }
    let sel = (log.mode4_index_selector & 1) as usize;
    if c != log.color_component_selector() {
        sel
    } else {
        1 - sel
    }
}

/// `compare_block_configs`.
fn compare_configs(a: &LogBc7Block, b: &LogBc7Block, compare_partition_index: bool) -> bool {
    a.mode >= 0
        && b.mode >= 0
        && a.mode == b.mode
        && a.num_partitions == b.num_partitions
        && a.pattern_bits == b.pattern_bits
        && (!compare_partition_index || a.pattern_index == b.pattern_index)
        && a.num_planes == b.num_planes
        && a.dp_rotation_index == b.dp_rotation_index
        && a.mode4_index_selector == b.mode4_index_selector
        && a.endpoint_bits == b.endpoint_bits
        && a.weight_bits == b.weight_bits
        && a.num_pbits == b.num_pbits
        && a.shared_pbits == b.shared_pbits
}

/// `compare_block_full`: config, used endpoints, p-bits and weight planes.
fn compare_full(a: &LogBc7Block, b: &LogBc7Block) -> bool {
    if !compare_configs(a, b, true) {
        return false;
    }
    let ns = (a.num_partitions as usize).min(3);
    let np = (a.num_pbits as usize).min(6);
    let nw = (a.num_planes as usize).min(2);
    a.endpoints[..ns] == b.endpoints[..ns]
        && a.pbits[..np] == b.pbits[..np]
        && a.weights[..nw] == b.weights[..nw]
}

/// `is_solid_blk`: one subset, constant weights.
fn is_solid(log: &LogBc7Block) -> bool {
    if log.num_partitions > 1 {
        return false;
    }
    log.weights
        .iter()
        .take((log.num_planes as usize).min(2))
        .all(|w| w.iter().all(|&v| v == w[0]))
}

#[inline]
fn texel_subset(log: &LogBc7Block, i: usize) -> usize {
    let pi = log.pattern_index as usize * 16 + i;
    match log.num_partitions {
        2 => BC7_PARTITION2.get(pi).map_or(0, |&s| s as usize),
        3 => BC7_PARTITION3.get(pi).map_or(0, |&s| s as usize),
        _ => 0,
    }
}

/// `canonicalize_endpoints`: per subset (or dual plane channel group) endpoint 0 gets the
/// smaller channel sum, weights are inverted to match (pixel exact).
fn canonicalize_endpoints(log: &mut LogBc7Block) {
    if log.num_planes > 1 {
        let e = bc7u::unpack_endpoints(log, 0);
        for p in 0..(log.num_planes as usize).min(2) {
            let (mut l0, mut l1) = (0u32, 0u32);
            for c in 0..4 {
                if log.endpoint_channel_weight_plane(c) == p {
                    l0 += u32::from(e[0][c]);
                    l1 += u32::from(e[1][c]);
                }
            }
            if l0 <= l1 {
                continue;
            }
            for c in 0..4 {
                if log.endpoint_channel_weight_plane(c) == p {
                    let t = log.endpoints[0][0][c];
                    log.endpoints[0][0][c] = log.endpoints[0][1][c];
                    log.endpoints[0][1][c] = t;
                }
            }
            let max_w = ((1u32 << log.weight_bits[p]) - 1) as u8;
            for w in log.weights[p].iter_mut() {
                *w = max_w.wrapping_sub(*w);
            }
        }
    } else {
        let num_comps = log.num_comps();
        for s in 0..(log.num_partitions as usize).min(3) {
            let e = bc7u::unpack_endpoints(log, s);
            let (mut l0, mut l1) = (0u32, 0u32);
            for c in 0..num_comps {
                l0 += u32::from(e[0][c]);
                l1 += u32::from(e[1][c]);
            }
            if l0 <= l1 {
                continue;
            }
            for c in 0..num_comps {
                let t = log.endpoints[s][0][c];
                log.endpoints[s][0][c] = log.endpoints[s][1][c];
                log.endpoints[s][1][c] = t;
            }
            if log.num_pbits != 0 && !log.shared_pbits {
                log.pbits.swap(s * 2, s * 2 + 1);
            }
            let max_w = ((1u32 << log.weight_bits[0]) - 1) as u8;
            let single = log.num_partitions == 1;
            for i in 0..16 {
                if single || texel_subset(log, i) == s {
                    log.weights[0][i] = max_w.wrapping_sub(log.weights[0][i]);
                }
            }
        }
    }
}

/// `optimize_block_weights`: per texel optimal weights for fixed config + endpoints.
fn optimize_block_weights(
    blk: &LogBc7Block,
    orig: &[Px; 16],
    cw: &[u32; 4],
) -> Result<LogBc7Block> {
    let mut out = *blk;
    for p in 0..(blk.num_planes as usize).min(2) {
        let num_vals = 1u32 << blk.weight_bits[p];
        let mut scratch = *blk;
        if blk.is_dual_plane() {
            scratch.weights[1 - p] = [0; 16];
        }
        let mut best_err = [u64::MAX; 16];
        let mut best_w = [0u8; 16];
        let chans: [bool; 4] = std::array::from_fn(|c| decoded_channel_weight_plane(blk, c) == p);
        for w in 0..num_vals {
            scratch.weights[p] = [w as u8; 16];
            let dec = unpack(&scratch)?;
            for i in 0..16 {
                let mut err = 0u64;
                for c in 0..4 {
                    if !chans[c] {
                        continue;
                    }
                    let d = i32::from(dec[i][c]) - i32::from(orig[i][c]);
                    err += u64::from((d * d) as u32) * u64::from(cw[c]);
                }
                if err < best_err[i] {
                    best_err[i] = err;
                    best_w[i] = w as u8;
                }
            }
        }
        out.weights[p] = best_w;
    }
    Ok(out)
}

/// `endpoint_dpcm` (encode direction): residuals (RR GG BB AA, G decorrelated) and p-bit
/// residuals of `log`'s `subset` vs `pred`'s `pred_subset` packed into `log`'s format.
fn endpoint_dpcm_encode(
    pred: &LogBc7Block,
    pred_subset: usize,
    log: &LogBc7Block,
    subset: usize,
) -> ([u8; 8], usize, [u8; 2], usize) {
    let mode = (log.mode as usize) & 7;
    let num_pbits = ENDPOINT_FORMATS[mode].2 as usize;
    // Zero residuals decode to the packed prediction (always within the endpoint mask).
    let mut packed = *log;
    bc7u::endpoint_dpcm_decode(pred, pred_subset, &mut packed, subset, &[0; 8], &[0; 2]);
    let s = subset.min(2);
    let num_comps = log.num_comps();
    let mut res = [0u8; 8];
    for c in 0..num_comps {
        res[c * 2] = log.endpoints[s][0][c].wrapping_sub(packed.endpoints[s][0][c]);
        res[c * 2 + 1] = log.endpoints[s][1][c].wrapping_sub(packed.endpoints[s][1][c]);
    }
    let (mut g_channel, mut a_channel) = (1usize, 3usize);
    if log.is_dual_plane() {
        a_channel = log.color_component_selector();
        if a_channel == 1 {
            g_channel = 3;
        }
    }
    let (g0, g1) = (res[g_channel * 2], res[g_channel * 2 + 1]);
    for c in 0..num_comps {
        if c == g_channel || c == a_channel {
            continue;
        }
        res[c * 2] = res[c * 2].wrapping_sub(g0);
        res[c * 2 + 1] = res[c * 2 + 1].wrapping_sub(g1);
    }
    let mut pb = [0u8; 2];
    for (p, slot) in pb.iter_mut().enumerate().take(num_pbits) {
        let i = (subset * num_pbits + p).min(5);
        *slot = log.pbits[i].wrapping_sub(packed.pbits[i]) & 1;
    }
    (res, num_comps * 2, pb, num_pbits)
}

// ---------------------------------------------------------------------------------------------
// Output streams.

/// `basisu::bitwise_coder` (LSB first; `append` keeps the unflushed bits, no padding).
#[derive(Default)]
struct BitWriter {
    bytes: Buffer,
    buf: u32,
    size: u32,
    total: u64,
}

impl BitWriter {
    fn put(&mut self, bits: u32, n: u32) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.total += u64::from(n);
        let mut v = (u64::from(bits) << self.size) | u64::from(self.buf);
        self.size += n;
        while self.size >= 8 {
            push(&mut self.bytes, v as u8)?;
            v >>= 8;
            self.size -= 8;
        }
        self.buf = u32::from(v as u8);
        Ok(())
    }

    fn append(&mut self, o: &BitWriter) -> Result<()> {
        for &b in o.bytes.iter() {
            self.put(u32::from(b), 8)?;
        }
        if o.size != 0 {
            self.put(o.buf, o.size)?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.size != 0 {
            self.total += u64::from(8 - (self.size & 7));
            push(&mut self.bytes, self.buf as u8)?;
            self.buf = 0;
            self.size = 0;
        }
        Ok(())
    }
}

/// Byte blobs by id (`blob_stream_writer`).
struct Blobs([Buffer; NUM_BLOBS]);

impl Default for Blobs {
    fn default() -> Self {
        Blobs(std::array::from_fn(|_| Buffer::default()))
    }
}

impl Blobs {
    #[inline]
    fn put(&mut self, id: usize, b: u8) -> Result<()> {
        push(self.0.get_mut(id).ok_or(Error::InvalidData)?, b)
    }
}

#[inline]
fn push_varint(out: &mut Buffer, mut x: u32) -> Result<()> {
    while x >= 0x80 {
        push(out, (x | 0x80) as u8)?;
        x >>= 7;
    }
    push(out, x as u8)
}

/// `blob_stream_writer::serialize`: Zstandard where it pays (any win up to 128 bytes, else at
/// least 1/400 of the size), raw otherwise.
fn serialize(blobs: &Blobs, zstd_level: i32) -> Result<Buffer> {
    let num_stored = blobs.0.iter().filter(|b| !b.is_empty()).count();
    if num_stored > 255 {
        return Err(Error::InvalidArgument);
    }
    let mut out = Buffer::default();
    push(&mut out, MAGIC_BEGIN)?;
    push(&mut out, num_stored as u8)?;
    for (id, blob) in blobs.0.iter().enumerate() {
        if blob.is_empty() {
            continue;
        }
        let uncomp = u32::try_from(blob.len()).map_err(|_| Error::InvalidArgument)?;
        let bound = ZSTD_compressBound(blob.len());
        let mut comp = crate::buffer::zeroed(bound)?;
        let res = ZSTD_compress(&mut comp, blob, zstd_level);
        let mut stored = 0usize;
        if !ZSTD_isError(res) && res < blob.len() {
            let saved = (blob.len() - res) as u64;
            if blob.len() <= 128 || saved * 400 >= blob.len() as u64 {
                stored = res;
            }
        }
        push(&mut out, (id as u8) | if stored != 0 { 0x80 } else { 0 })?;
        push_varint(&mut out, uncomp)?;
        if stored != 0 {
            push_varint(&mut out, stored as u32)?;
            extend(&mut out, comp.get(..stored).ok_or(Error::InvalidData)?)?;
        } else {
            extend(&mut out, blob)?;
        }
    }
    push(&mut out, MAGIC_END)?;
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Weight grid DCT (encoder side of `xbc7_weight_grid_dct_fixed`).

/// Quantized DCT symbols of one plane: DC plus (zero run, coefficient | `AC_EOB`) pairs.
#[derive(Clone, Copy, Default)]
struct DctSyms {
    dc: i32,
    acs: [Ac; 16],
    n: usize,
}

impl DctSyms {
    #[inline]
    fn acs(&self) -> &[Ac] {
        &self.acs[..self.n.min(16)]
    }

    #[inline]
    fn push(&mut self, run: u32, coeff: i32) {
        if let Some(slot) = self.acs.get_mut(self.n) {
            *slot = (run, coeff);
            self.n += 1;
        }
    }

    /// Number of coded ACs (entries minus a trailing end-of-block).
    fn num_acs(&self) -> u32 {
        let a = self.acs();
        let eob = a.last().is_some_and(|&(_, c)| c == AC_EOB);
        (a.len() - usize::from(eob)) as u32
    }

    /// `xbc7_flat_to_syms`: canonical run length form of a natural order 4x4 grid.
    fn from_flat(flat: &[i32; 16]) -> DctSyms {
        let mut s = DctSyms {
            dc: flat[0],
            ..DctSyms::default()
        };
        let mut zeros = 0u32;
        for &(x, y) in &ZIGZAG[1..] {
            let ac = flat[x + y * 4];
            if ac == 0 {
                zeros += 1;
                continue;
            }
            s.push(zeros, ac);
            zeros = 0;
        }
        if zeros != 0 {
            s.push(zeros, AC_EOB);
        }
        s
    }

    /// `xbc7_syms_to_flat`.
    fn flat(self) -> [i32; 16] {
        let mut flat = [0i32; 16];
        flat[0] = self.dc;
        let mut zig = 1u32;
        for &(run, coeff) in self.acs() {
            zig += run;
            if zig >= 16 || coeff == AC_EOB {
                break;
            }
            let (x, y) = ZIGZAG[zig as usize];
            flat[x + y * 4] = coeff;
            zig += 1;
        }
        flat
    }
}

/// `dct4::forward_ortho`.
fn forward_ortho(x: [Fx; 4]) -> [Fx; 4] {
    let a0 = x[0].add(x[3]);
    let a1 = x[1].add(x[2]);
    let a2 = x[1].sub(x[2]);
    let a3 = x[0].sub(x[3]);
    [
        Fx::from_sum(a0.mul_wide(DCT_HALF) + a1.mul_wide(DCT_HALF)),
        Fx::from_sum(a3.mul_wide(DCT_C1) + a2.mul_wide(DCT_C3)),
        Fx::from_sum(a0.mul_wide(DCT_HALF) - a1.mul_wide(DCT_HALF)),
        Fx::from_sum(a3.mul_wide(DCT_C3) - a2.mul_wide(DCT_C1)),
    ]
}

/// `dct2fx::forward_4x4`: rows, then columns.
fn forward_4x4(src: &[Fx; 16]) -> [Fx; 16] {
    let mut t = [Fx(0); 16];
    for x in 0..4 {
        let out = forward_ortho([src[x * 4], src[x * 4 + 1], src[x * 4 + 2], src[x * 4 + 3]]);
        t[x * 4..x * 4 + 4].copy_from_slice(&out);
    }
    let mut dst = [Fx(0); 16];
    for v in 0..4 {
        let out = forward_ortho([t[v], t[4 + v], t[8 + v], t[12 + v]]);
        for (u, o) in out.into_iter().enumerate() {
            dst[u * 4 + v] = o;
        }
    }
    dst
}

/// `quantize_deadzone`.
fn quantize_deadzone(d: Fx, l: i32, x: usize, y: usize) -> i32 {
    if (x == 1 && y == 0) || (x == 0 && y == 1) {
        return if l == 0 {
            0
        } else {
            d.div_int(l).round_to_int()
        };
    }
    if l <= 0 {
        return 0;
    }
    let s = Fx(d.0.wrapping_abs());
    let tau = Fx(DEADZONE_ALPHA as i32).mul_int(l);
    if s <= tau {
        return 0;
    }
    let q = s.sub(tau).div_int(l).round_to_int();
    if d < Fx(0) { -q } else { q }
}

/// `xbc7_weight_grid_dct_fixed::forward`.
fn dct_forward(
    global_q: Fx,
    plane: usize,
    preds: Option<&[i32; 16]>,
    log: &LogBc7Block,
) -> DctSyms {
    let bits = u32::from(log.weight_bits[plane]);
    let w: [Fx; 16] = std::array::from_fn(|i| {
        let pred = preds.map_or(0, |p| p[i]);
        Fx::from_int(dequant_weight(u32::from(log.weights[plane][i]), bits) - pred)
    });
    let d = forward_4x4(&w);
    let span = max_span_len(log, plane);
    let level = compute_level_scale(global_q, span, bits);
    let tab = compute_quant_table(global_q, level);

    let mut coeffs = [0i32; 16];
    for y in 0..4 {
        for x in 0..4 {
            if x == 0 && y == 0 {
                let mut dc = d[0].round_to_int().clamp(-255, 255);
                // Uniform DC quantizer, step 2^(6 - weight bits), round half away from zero.
                let q = 1i32 << (6 - bits.clamp(2, 4));
                let max_mag = 256 / q;
                dc = if dc >= 0 {
                    (dc + q / 2) / q
                } else {
                    -((-dc + q / 2) / q)
                };
                coeffs[0] = dc.clamp(-max_mag, max_mag);
                continue;
            }
            let idx = x + y * 4;
            coeffs[idx] = quantize_deadzone(d[idx], tab[idx], x, y).clamp(-255, 255);
        }
    }
    let mut syms = DctSyms::from_flat(&coeffs);
    syms.dc = coeffs[0];
    syms
}

// ---------------------------------------------------------------------------------------------
// Predictor sweeps.

/// `g_xbc7_pred_priority`: most valuable weight predictors first.
const PRED_PRIORITY: [u32; 27] = [
    CAND_ABSOLUTE,
    CAND_LEFT_EDGE,
    CAND_UPPER_EDGE,
    CAND_FIRST_XY_DELTA,
    CAND_FIRST_XY_DELTA + 1,
    CAND_DIAG_AVG,
    CAND_FIRST_XY_DELTA + 6,
    CAND_FIRST_XY_DELTA + 2,
    CAND_FIRST_XY_DELTA + 7,
    CAND_FIRST_XY_DELTA + 8,
    CAND_REFLECT_LEFT,
    CAND_FIRST_XY_DELTA + 4,
    CAND_FIRST_XY_DELTA + 5,
    CAND_REFLECT_UPPER,
    CAND_PLANE_FIT,
    CAND_FIRST_XY_DELTA + 3,
    CAND_LU_BLEND_STRONG,
    CAND_DDR,
    CAND_DDL,
    CAND_GRADIENT,
    CAND_GRADIENT_DAMPED,
    CAND_LU_BLEND,
    CAND_LU_AVG,
    CAND_UPPER_DIAG_EDGE_BLEND,
    CAND_DIAG_EDGE_BLEND,
    CAND_MED,
    CAND_GAB,
];

/// `xbc7_build_pred_mask`: enabled candidates for an effort level (6 at 0, all at 10).
fn build_pred_mask(effort: u32) -> u64 {
    let effort = effort.min(10) as usize;
    let count = (6 + ((TOTAL_CANDS - 6) * effort + 5) / 10).min(TOTAL_CANDS);
    let mut used = [false; TOTAL_CANDS];
    let mut order = [0u32; TOTAL_CANDS];
    let mut n = 0;
    for &c in &PRED_PRIORITY {
        used[c as usize] = true;
        order[n] = c;
        n += 1;
    }
    for c in 0..TOTAL_CANDS {
        if !used[c] {
            order[n] = c as u32;
            n += 1;
        }
    }
    order[..count].iter().fold(0u64, |m, &c| m | (1u64 << c))
}

/// `xbc7_amp_codes_for_effort`.
fn amp_codes_for_effort(effort: u32) -> u32 {
    const T: [u32; 11] = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4];
    T[effort.min(10) as usize]
}

/// Shared per-image coding state.
struct Ctx<'a> {
    rgba: &'a [u8],
    width: usize,
    height: usize,
    nbx: u32,
    has_alpha: bool,
    p: &'a EncodeParams,
    q: Fx,
    pred_mask: u64,
    num_amp: u32,
}

impl Ctx<'_> {
    /// `image::extract_block_clamped`.
    fn block(&self, bx: u32, by: u32) -> [Px; 16] {
        let mut px = [[0u8; 4]; 16];
        for (i, p) in px.iter_mut().enumerate() {
            let x = (bx as usize * 4 + (i & 3)).min(self.width - 1);
            let y = (by as usize * 4 + (i >> 2)).min(self.height - 1);
            let o = (y * self.width + x) * 4;
            if let Some(s) = self.rgba.get(o..o + 4) {
                p.copy_from_slice(s);
            }
        }
        px
    }

    #[inline]
    fn psnr(&self, orig: &[Px; 16], dec: &[Px; 16]) -> f32 {
        block_wsse_psnr(orig, dec, &self.p.weights)
    }
}

/// Weight predictions of candidate `cand` / `amp` for every plane (None if unavailable).
#[inline]
fn eval_preds(cand: u32, amp: u32, bx: i32, by: i32, tile: &Tile, p: usize) -> Option<[i32; 16]> {
    eval_weight_predictor(cand, amp, bx, by, tile, p)
}

/// Winner of the weight DCT sweep.
struct DctChoice {
    syms: [DctSyms; 2],
    blk: LogBc7Block,
    pred_index: u32,
}

/// `pack_weights`: weight DCT predictor sweep, fewest ACs first, then error (2% margin).
fn pack_weights(
    ctx: &Ctx,
    tile: &Tile,
    bx: i32,
    by: i32,
    orig: &[Px; 16],
    log: &LogBc7Block,
) -> Result<DctChoice> {
    let mut best = DctChoice {
        syms: [DctSyms::default(); 2],
        blk: *log,
        pred_index: 0,
    };
    let mut best_err = u64::MAX;
    let mut best_num_acs = u32::MAX;
    let planes = (log.num_planes as usize).min(2);
    for amp in 0..ctx.num_amp {
        for cand in 0..TOTAL_CANDIDATES {
            if (amp != 0 && cand == CAND_ABSOLUTE) || (ctx.pred_mask >> cand) & 1 == 0 {
                continue;
            }
            let mut cl = *log;
            let mut syms = [DctSyms::default(); 2];
            let mut total_acs = 0u32;
            let mut valid = true;
            for p in 0..planes {
                let preds = if cand != CAND_ABSOLUTE {
                    match eval_preds(cand, amp, bx, by, tile, p) {
                        Some(w) => Some(w),
                        None => {
                            valid = false;
                            break;
                        }
                    }
                } else {
                    None
                };
                let s = dct_forward(ctx.q, p, preds.as_ref(), &cl);
                cl.weights[p] = [0; 16];
                if !dct_inverse(ctx.q, p, preds.as_ref(), s.dc, s.acs(), &mut cl) {
                    return Err(Error::InvalidData);
                }
                total_acs += s.num_acs();
                syms[p] = s;
            }
            if !valid {
                continue;
            }
            let px = unpack(&cl)?;
            let err: u64 = px
                .iter()
                .zip(orig)
                .map(|(a, b)| u64::from(weighted_dist2(a, b, &ctx.p.weights)))
                .sum();
            if total_acs < best_num_acs
                || (total_acs == best_num_acs
                    && err.wrapping_mul(1000)
                        < best_err.wrapping_mul(1000 - ERR_SWITCH_MARGIN_PERMILLE))
            {
                best = DctChoice {
                    syms,
                    blk: cl,
                    pred_index: cand + amp * TOTAL_CANDIDATES,
                };
                best_num_acs = total_acs;
                best_err = err;
            }
        }
    }
    Ok(best)
}

/// Winner of the lossless residual DPCM sweep.
struct DpcmChoice {
    pred_index: u32,
    syms: [[u8; 16]; 2],
    cost: u64,
}

/// `pack_weights_dpcm`: predictor minimizing the estimated residual cost (3% switch margin).
fn pack_weights_dpcm(ctx: &Ctx, tile: &Tile, bx: i32, by: i32, log: &LogBc7Block) -> DpcmChoice {
    let mut best = DpcmChoice {
        pred_index: 0,
        syms: [[0; 16]; 2],
        cost: u64::MAX,
    };
    let planes = (log.num_planes as usize).min(2);
    for amp in 0..ctx.num_amp {
        for cand in 0..TOTAL_CANDIDATES {
            if (amp != 0 && cand == CAND_ABSOLUTE) || (ctx.pred_mask >> cand) & 1 == 0 {
                continue;
            }
            let mut cost = 0u64;
            let mut syms = [[0u8; 16]; 2];
            let mut valid = true;
            for p in 0..planes {
                let bits = u32::from(log.weight_bits[p]).clamp(2, 4);
                let mask = (1u32 << bits) - 1;
                let half = 1u32 << (bits - 1);
                let preds = if cand != CAND_ABSOLUTE {
                    match eval_preds(cand, amp, bx, by, tile, p) {
                        Some(w) => Some(w),
                        None => {
                            valid = false;
                            break;
                        }
                    }
                } else {
                    None
                };
                for i in 0..16 {
                    let pred_index = preds.map_or(0, |pr| u32::from(quant_weight(pr[i], bits)));
                    let sym = u32::from(log.weights[p][i]).wrapping_sub(pred_index) & mask;
                    syms[p][i] = sym as u8;
                    if preds.is_some() {
                        let v = if sym >= half {
                            sym as i32 - (mask + 1) as i32
                        } else {
                            sym as i32
                        };
                        cost += DPCM_RESID_COST_OBITS[(bits - 2) as usize]
                            [(v.unsigned_abs() as usize).min(8)];
                    } else {
                        cost += 8 * if bits == 2 { 2 } else { 4 };
                    }
                }
            }
            if !valid {
                continue;
            }
            if best.cost == u64::MAX
                || cost.wrapping_mul(1000)
                    < best.cost.wrapping_mul(1000 - SWEEP_SWITCH_MARGIN_PERMILLE)
            {
                best = DpcmChoice {
                    pred_index: cand + amp * TOTAL_CANDIDATES,
                    syms,
                    cost,
                };
            }
        }
    }
    best
}

/// `ac_truncate_rdo`: zero the highest-frequency ACs (reverse zigzag, the 2x2 low corner kept)
/// while the block stays within the PSNR drop.
fn ac_truncate_rdo(
    ctx: &Ctx,
    tile: &Tile,
    bx: i32,
    by: i32,
    orig: &[Px; 16],
    best: &mut DctChoice,
) -> Result<()> {
    let planes = (best.blk.num_planes as usize).min(2);
    let cand = best.pred_index % TOTAL_CANDIDATES;
    let amp = best.pred_index / TOTAL_CANDIDATES;
    let mut preds: [Option<[i32; 16]>; 2] = [None; 2];
    if cand != CAND_ABSOLUTE {
        for (p, slot) in preds.iter_mut().enumerate().take(planes) {
            *slot = Some(eval_preds(cand, amp, bx, by, tile, p).ok_or(Error::InvalidData)?);
        }
    }
    let orig_psnr = ctx.psnr(orig, &unpack(&best.blk)?);
    let mut flat = [best.syms[0].flat(), best.syms[1].flat()];
    let drop = ctx.p.ac_trunc_rdo_max_psnr_drop;
    let floor = ctx.p.rdo_min_block_psnr;
    for zz in (1..16usize).rev() {
        if zz == 1 || zz == 2 || zz == 4 {
            continue;
        }
        let (x, y) = ZIGZAG[zz];
        let nat = x + y * 4;
        if !flat.iter().take(planes).any(|f| f[nat] != 0) {
            continue;
        }
        let mut saved = [0i32; 2];
        for p in 0..planes {
            saved[p] = flat[p][nat];
            flat[p][nat] = 0;
        }
        let mut trial = best.blk;
        let mut trial_syms = [DctSyms::default(); 2];
        for p in 0..planes {
            trial_syms[p] = DctSyms::from_flat(&flat[p]);
            trial.weights[p] = [0; 16];
            let s = &trial_syms[p];
            if !dct_inverse(ctx.q, p, preds[p].as_ref(), s.dc, s.acs(), &mut trial) {
                return Err(Error::InvalidData);
            }
        }
        let psnr = ctx.psnr(orig, &unpack(&trial)?);
        if psnr >= floor && psnr >= orig_psnr - drop {
            best.blk = trial;
            best.syms[..planes].copy_from_slice(&trial_syms[..planes]);
        } else {
            for p in 0..planes {
                flat[p][nat] = saved[p];
            }
            break;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Stripe passes.

/// One stripe: its block rows and base pack LUT flags.
struct StripeJob<'a> {
    first_row: u32,
    rows: &'a mut [LogBc7Block],
    lut: &'a [bool],
}

impl StripeJob<'_> {
    fn end_row(&self, nbx: u32) -> u32 {
        self.first_row + (self.rows.len() / nbx.max(1) as usize) as u32
    }
}

/// `block_reuse_rdo_pass` on one stripe: Repeat of a neighbor, else the mean color.
fn block_reuse_rdo(ctx: &Ctx, job: &mut StripeJob) -> Result<()> {
    let p = ctx.p;
    let nbx = ctx.nbx as usize;
    let (first, end) = (job.first_row, job.end_row(ctx.nbx));
    for by in first..end {
        for bx in 0..nbx {
            let row = (by - first) as usize * nbx;
            let idx = row + bx;
            let cur = job.rows[idx];
            let src = ctx.block(bx as u32, by);
            let orig_psnr = ctx.psnr(&src, &unpack(&cur)?);
            if p.repeat_rdo_enabled {
                let preds = [
                    (bx >= 1).then(|| job.rows[idx - 1]),
                    (by > first).then(|| job.rows[idx - nbx]),
                ];
                let mut best: Option<(LogBc7Block, f32)> = None;
                for pr in preds.iter().flatten() {
                    let ps = ctx.psnr(&src, &unpack(pr)?);
                    if ps >= p.rdo_min_block_psnr
                        && ps >= orig_psnr - p.repeat_rdo_max_psnr_drop
                        && best.is_none_or(|(_, b)| ps > b)
                    {
                        best = Some((*pr, ps));
                    }
                }
                if let Some((b, _)) = best {
                    job.rows[idx] = b;
                    continue;
                }
            }
            if p.solid_rdo_enabled && !is_solid(&cur) {
                let mut sum = [0u32; 4];
                for px in &src {
                    for c in 0..4 {
                        sum[c] += u32::from(px[c]);
                    }
                }
                let mut mean = sum.map(|s| ((s + 8) >> 4) as u8);
                if !ctx.has_alpha {
                    mean[3] = 255;
                }
                let mut cand = LogBc7Block::default();
                create_solid_blk(&mut cand, &mean);
                let ps = ctx.psnr(&src, &unpack(&cand)?);
                if ps >= p.rdo_min_block_psnr && ps >= orig_psnr - p.solid_rdo_max_psnr_drop {
                    job.rows[idx] = cand;
                }
            }
        }
    }
    Ok(())
}

/// `endpoint_dpcm_rdo_pass` on one stripe: endpoints forced to a neighbor's prediction,
/// weights re-optimized, best kept within the PSNR drop.
fn endpoint_rdo(ctx: &Ctx, job: &mut StripeJob) -> Result<()> {
    let p = ctx.p;
    let nbx = ctx.nbx as usize;
    let (first, end) = (job.first_row, job.end_row(ctx.nbx));
    for by in first..end {
        for bx in 0..nbx {
            let idx = (by - first) as usize * nbx + bx;
            let cur = job.rows[idx];
            if is_solid(&cur) {
                continue;
            }
            if (bx >= 1 && compare_full(&cur, &job.rows[idx - 1]))
                || (by > first && compare_full(&cur, &job.rows[idx - nbx]))
            {
                continue;
            }
            let src = ctx.block(bx as u32, by);
            let orig_psnr = ctx.psnr(&src, &unpack(&cur)?);
            let up = by > first;
            let preds = [
                (bx >= 1).then(|| job.rows[idx - 1]),
                up.then(|| job.rows[idx - nbx]),
                (bx >= 1 && up).then(|| job.rows[idx - nbx - 1]),
                (bx + 1 < nbx && up).then(|| job.rows[idx - nbx + 1]),
            ];
            let mut best: Option<(LogBc7Block, f32)> = None;
            for pr in preds.iter().flatten() {
                let mut cand = cur;
                for s in 0..cand.num_partitions as usize {
                    bc7u::endpoint_dpcm_decode(pr, 0, &mut cand, s, &[0; 8], &[0; 2]);
                }
                let cand = optimize_block_weights(&cand, &src, &p.weights)?;
                let cp = ctx.psnr(&src, &unpack(&cand)?);
                if best.is_none_or(|(_, b)| cp > b) {
                    best = Some((cand, cp));
                }
            }
            if let Some((b, bp)) = best
                && bp >= p.rdo_min_block_psnr
                && bp >= orig_psnr - p.endpoint_rdo_max_psnr_drop
            {
                job.rows[idx] = b;
            }
        }
    }
    Ok(())
}

/// Endpoint predictor choice of one block.
struct EpChoice<'a> {
    blk: &'a LogBc7Block,
    mode: u8,
    delta_index: u8,
    subset: usize,
}

/// `pack_stripe` for one block: emits its streams, returns the coded block.
fn code_block(
    ctx: &Ctx,
    tile: &Tile,
    bx: i32,
    by: i32,
    log: &LogBc7Block,
    used_lut: bool,
    job: &mut StripeOut,
) -> Result<LogBc7Block> {
    let left = tile.get(bx - 1, by);
    let up = tile.get(bx, by - 1);
    let ld = tile.get(bx - 1, by - 1);
    let rd = tile.get(bx + 1, by - 1);
    let has_alpha = ctx.has_alpha;
    let blobs = &mut job.blobs;

    if up.is_some_and(|u| compare_full(log, u)) {
        blobs.put(BLOB_COMMANDS, 1)?;
        return Ok(*log);
    }
    if left.is_some_and(|l| compare_full(log, l)) {
        blobs.put(BLOB_COMMANDS, 0)?;
        return Ok(*log);
    }

    if is_solid(log) {
        let mut preds = [0i32; 4];
        let mut num = 0i32;
        if let Some(l) = left {
            for y in 0..4 {
                let px = bc7u::unpack_texel(l, 3, y).ok_or(Error::InvalidData)?;
                for c in 0..4 {
                    preds[c] += i32::from(px[c]);
                }
            }
            num += 4;
        }
        if let Some(u) = up {
            for x in 0..4 {
                let px = bc7u::unpack_texel(u, x, 3).ok_or(Error::InvalidData)?;
                for c in 0..4 {
                    preds[c] += i32::from(px[c]);
                }
            }
            num += 4;
        }
        if num != 0 {
            for p in preds.iter_mut() {
                *p = (*p + num / 2) / num;
            }
        }
        blobs.put(BLOB_COMMANDS, 2)?;
        let mut solid = bc7u::unpack_texel(log, 0, 0).ok_or(Error::InvalidData)?;
        if !has_alpha {
            solid[3] = 255;
        }
        for c in 0..(if has_alpha { 4 } else { 3 }) {
            blobs.put(
                BLOB_SOLID_RGBA_DELTAS,
                (i32::from(solid[c]) - preds[c]) as u8,
            )?;
        }
        let mut out = LogBc7Block::default();
        create_solid_blk(&mut out, &solid);
        return Ok(out);
    }

    let orig = ctx.block(bx as u32, by as u32);

    // Config.
    let mut command: u8 = if left.is_some_and(|l| compare_configs(log, l, false)) {
        4
    } else if up.is_some_and(|u| compare_configs(log, u, false)) {
        5
    } else if ld.is_some_and(|b| compare_configs(log, b, false)) {
        6
    } else if rd.is_some_and(|b| compare_configs(log, b, false)) {
        7
    } else {
        let config =
            (log.mode as u8 & 7) | (log.dp_rotation_index << 3) | (log.mode4_index_selector << 5);
        blobs.put(BLOB_CONFIG, config)?;
        3
    };
    if log.num_partitions == 2 {
        blobs.put(BLOB_PARTITION2, log.pattern_index)?;
    } else if log.num_partitions == 3 {
        blobs.put(BLOB_PARTITION3, log.pattern_index)?;
    }

    // Endpoint predictor sweep: residual byte SSE, later candidates need a 3% margin.
    let mode6_opaque = log.mode == 6 && !has_alpha;
    let num_subsets = (log.num_partitions as usize).min(3);
    let ep_cost = |pred: &LogBc7Block, pred_subset: usize| -> u64 {
        let mut cost = 0u64;
        for s in 0..num_subsets {
            let (res, n, _, _) = endpoint_dpcm_encode(pred, pred_subset, log, s);
            let n = if mode6_opaque { 6 } else { n };
            for &r in res.iter().take(n) {
                let v = i64::from(r as i8);
                cost += (v * v) as u64;
            }
        }
        cost
    };
    let mut ep: Option<EpChoice> = None;
    let mut best_cost = u64::MAX;
    let mut consider = |ep: &mut Option<EpChoice<'_>>, cost: u64| -> bool {
        if ep.is_none()
            || cost.wrapping_mul(1000) < best_cost.wrapping_mul(1000 - SWEEP_SWITCH_MARGIN_PERMILLE)
        {
            best_cost = cost;
            true
        } else {
            false
        }
    };
    for (i, nb) in [left, up, ld, rd].into_iter().enumerate() {
        if let Some(b) = nb {
            let cost = ep_cost(b, 0);
            if consider(&mut ep, cost) {
                ep = Some(EpChoice {
                    blk: b,
                    mode: 1 + i as u8,
                    delta_index: 0,
                    subset: 0,
                });
            }
        }
    }
    for (i, nb) in [left, up].into_iter().enumerate() {
        if let Some(b) = nb
            && b.num_partitions >= 2
        {
            let cost = ep_cost(b, 1);
            if consider(&mut ep, cost) {
                ep = Some(EpChoice {
                    blk: b,
                    mode: 6 + i as u8,
                    delta_index: 0,
                    subset: 1,
                });
            }
        }
    }
    for (i, &(dx, dy)) in XY_DELTAS.iter().enumerate() {
        if let Some(b) = tile.get(bx + i32::from(dx), by + i32::from(dy)) {
            let cost = ep_cost(b, 0);
            if consider(&mut ep, cost) {
                ep = Some(EpChoice {
                    blk: b,
                    mode: 5,
                    delta_index: i as u8,
                    subset: 0,
                });
            }
        }
    }

    if let Some(e) = &ep {
        command |= e.mode << 3;
        if e.mode == 5 {
            blobs.put(BLOB_EP_BLOCK_INDEX, e.delta_index)?;
        }
        let fine = log.endpoint_bits[0] >= 6;
        for s in 0..num_subsets {
            let (res, mut n, pb, npb) = endpoint_dpcm_encode(e.blk, e.subset, log, s);
            if mode6_opaque {
                if log.endpoints[0][0][3] != 127 || log.endpoints[0][1][3] != 127 {
                    return Err(Error::InvalidData);
                }
                n = 6;
            }
            let base = if fine {
                BLOB_EP_DELTA_FINE_R
            } else {
                BLOB_EP_DELTA_COARSE_R
            };
            for i in (0..n).step_by(2) {
                blobs.put(base + (i >> 1), res[i])?;
                blobs.put(base + (i >> 1), res[i + 1])?;
            }
            for &b in pb.iter().take(npb) {
                job.pbits.put(u32::from(b), 1)?;
            }
        }
    } else {
        for s in 0..num_subsets {
            for c in 0..log.num_comps() {
                for e in 0..2 {
                    let bits = u32::from(log.endpoint_bits[usize::from(c == 3)]);
                    job.ep_raw.put(u32::from(log.endpoints[s][e][c]), bits)?;
                }
            }
        }
        for &b in log.pbits.iter().take((log.num_pbits as usize).min(6)) {
            job.ep_raw.put(u32::from(b), 1)?;
        }
    }

    // Weights: residual DCT vs lossless residual DPCM by estimated size.
    let force_dpcm = ctx.p.dct_q >= 100 || used_lut;
    let planes = (log.num_planes as usize).min(2);
    let mut dct: Option<DctChoice> = None;
    let mut dct_est = 0u64;
    if !force_dpcm {
        let d = pack_weights(ctx, tile, bx, by, &orig, log)?;
        for s in d.syms.iter().take(planes) {
            let num_eobs = s.acs().iter().filter(|&&(_, c)| c == AC_EOB).count() as u64;
            let num_acs = s.acs().len() as u64 - num_eobs;
            dct_est += DCT_DC_BYTE_COST_OBITS
                + 2 * DCT_AC_BYTE_COST_OBITS * num_acs
                + DCT_AC_BYTE_COST_OBITS * num_eobs
                + 8 * (num_acs + u64::from(d.pred_index != CAND_ABSOLUTE));
        }
        dct = Some(d);
    }
    let dpcm = pack_weights_dpcm(ctx, tile, bx, by, log);
    let use_dpcm = force_dpcm
        || dpcm.cost.wrapping_mul(100) <= dct_est.wrapping_mul(u64::from(ctx.p.wt_dpcm_alpha_pct));

    let Some(mut d) = dct.filter(|_| !use_dpcm) else {
        blobs.put(BLOB_WEIGHT_PREDICTORS, dpcm.pred_index as u8)?;
        let absolute = dpcm.pred_index == CAND_ABSOLUTE;
        for p in 0..planes {
            let bits = u32::from(log.weight_bits[p]);
            let id = if absolute {
                BLOB_RAW_WEIGHT_BITS
            } else {
                BLOB_DPCM_WEIGHT_RESID2 + (bits.clamp(2, 4) - 2) as usize
            };
            let s = &dpcm.syms[p];
            if bits == 2 {
                for q in s.as_chunks::<4>().0 {
                    blobs.put(id, q[0] | (q[1] << 2) | (q[2] << 4) | (q[3] << 6))?;
                }
            } else {
                for q in s.as_chunks::<2>().0 {
                    blobs.put(id, q[0] | (q[1] << 4))?;
                }
            }
        }
        blobs.put(BLOB_COMMANDS, command)?;
        return Ok(*log);
    };

    command |= 1 << 6;
    if ctx.p.ac_trunc_rdo_max_psnr_drop > 0.0 {
        ac_truncate_rdo(ctx, tile, bx, by, &orig, &mut d)?;
    }
    blobs.put(BLOB_WEIGHT_PREDICTORS, d.pred_index as u8)?;
    for s in d.syms.iter().take(planes) {
        blobs.put(BLOB_DC_COEFFS_SMALL, s.dc.unsigned_abs() as u8)?;
        if d.pred_index != CAND_ABSOLUTE {
            job.signs.put(u32::from(s.dc < 0), 1)?;
        }
        for &(run, coeff) in s.acs() {
            if coeff == AC_EOB {
                blobs.put(BLOB_AC_COEFFS, 0xFF)?;
            } else {
                blobs.put(BLOB_AC_COEFFS, run as u8)?;
                blobs.put(BLOB_AC_COEFFS, coeff.unsigned_abs() as u8)?;
                job.signs.put(u32::from(coeff < 0), 1)?;
            }
        }
    }
    blobs.put(BLOB_COMMANDS, command)?;
    Ok(d.blk)
}

/// Per stripe output streams.
#[derive(Default)]
struct StripeOut {
    blobs: Blobs,
    signs: BitWriter,
    pbits: BitWriter,
    ep_raw: BitWriter,
}

/// RDO passes and the coding pass of one stripe (stripes never reference each other).
fn encode_stripe(ctx: &Ctx, job: &mut StripeJob, out: &mut StripeOut) -> Result<()> {
    if ctx.p.repeat_rdo_enabled || ctx.p.solid_rdo_enabled {
        block_reuse_rdo(ctx, job)?;
    }
    if ctx.p.endpoint_rdo_enabled {
        endpoint_rdo(ctx, job)?;
    }
    let nbx = ctx.nbx as usize;
    let (first, end) = (job.first_row, job.end_row(ctx.nbx));
    for by in first..end {
        for bx in 0..nbx {
            let idx = (by - first) as usize * nbx + bx;
            let cur = job.rows[idx];
            let used_lut = job.lut.get(idx).copied().unwrap_or(false);
            let coded = {
                let tile = Tile {
                    blks: job.rows,
                    nbx: nbx as i32,
                    first_row: first as i32,
                    end_row: end as i32,
                };
                code_block(ctx, &tile, bx as i32, by as i32, &cur, used_lut, out)?
            };
            job.rows[idx] = coded;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Image level.

/// `compute_encoder_stripes` / `set_num_stripes_for_image` / the `m_num_stripes` clamp.
fn stripe_count(stripes: Stripes, height: u32, nby: u32) -> u32 {
    let max = MAX_STRIPES as u32;
    let auto = || {
        let mut n = 1;
        if height >= MIN_IMAGE_TEXEL_ROWS_TO_STRIPE {
            n = clampu(nby / MIN_STRIPE_BLOCK_ROWS, 1, max);
        }
        n.min(nby.max(1))
    };
    match stripes {
        Stripes::Auto | Stripes::Exact(0) => auto(),
        Stripes::Desired(d) => {
            let limit = if height < MIN_IMAGE_TEXEL_ROWS_TO_STRIPE {
                1
            } else {
                clampu(nby / MIN_STRIPE_BLOCK_ROWS, 1, max)
            };
            clampu(clampu(d, 1, limit), 1, nby.min(max))
        }
        Stripes::Exact(n) => clampu(n, 1, nby.min(max)),
    }
}

/// `xbc7_pack_bc7_base_block` + alt-pack RDO + canonicalization (+ weight re-optimization).
fn base_pack_block(
    ctx: &Ctx,
    bc7e: Option<&crate::bc7e::Params>,
    px: &[Px; 16],
) -> Result<(LogBc7Block, bool)> {
    let p = ctx.p;
    let (mut phys, used_lut) = match bc7e {
        Some(params) => crate::bc7e::compress_block_lut(px, params),
        None => {
            let mut b = [0u8; 16];
            bc7f::fast_pack_bc7_auto_rgba(&mut b, px, p.bc7_pack_flags);
            (b, false)
        }
    };
    let phys_pixels = |b: &[u8; 16]| -> Result<[Px; 16]> {
        let mut l = LogBc7Block::default();
        if !bc7u::unpack_phys(b, &mut l) {
            return Err(Error::InvalidData);
        }
        unpack(&l)
    };
    if p.bc7_alt_pack_enabled {
        let mut alt = [0u8; 16];
        bc7f::fast_pack_bc7_auto_rgba(&mut alt, px, p.bc7_pack_flags_alt);
        let psnr_primary = ctx.psnr(px, &phys_pixels(&phys)?);
        let psnr_alt = ctx.psnr(px, &phys_pixels(&alt)?);
        if psnr_alt >= p.rdo_min_block_psnr && (psnr_primary - psnr_alt) < p.bc7_alt_max_psnr_drop {
            phys = alt;
        }
    }
    let mut log = LogBc7Block::default();
    if !bc7u::unpack_phys(&phys, &mut log) {
        return Err(Error::InvalidData);
    }
    canonicalize_endpoints(&mut log);
    if p.optimize_weights_after_bc7f {
        log = optimize_block_weights(&log, px, &p.weights)?;
    }
    Ok((log, used_lut))
}

/// `bc7e_scalar` preset for `bc7e_level` (LUTs only at lossless Q).
fn bc7e_params(p: &EncodeParams) -> crate::bc7e::Params {
    use crate::bc7e::Params;
    let perc = p.perceptual;
    let mut bp = match p.bc7e_level.min(BC7E_MAX_LEVEL) {
        0 => Params::ultrafast(perc),
        1 => Params::veryfast(perc),
        2 => Params::fast(perc),
        3 => Params::basic(perc),
        4 => Params::slow(perc),
        5 => Params::veryslow(perc),
        _ => Params::slowest(perc),
    };
    if !perc {
        bp.weights = p.weights;
    }
    bp.use_luts = p.dct_q >= 100;
    bp
}

fn new_blocks(n: usize) -> Result<Blocks> {
    let mut v = Blocks::default();
    v.resize(n, LogBc7Block::default())
        .map_err(|_| Error::OutOfMemory)?;
    Ok(v)
}

/// [`encode`], also returning the coded logical blocks (what the stream decodes to).
pub(crate) fn encode_blocks(
    rgba: &[u8],
    width: u32,
    height: u32,
    params: &EncodeParams,
) -> Result<Encoded> {
    if width == 0 || height == 0 || width > u32::from(u16::MAX) || height > u32::from(u16::MAX) {
        return Err(Error::InvalidArgument);
    }
    if rgba.len() as u64 != u64::from(width) * u64::from(height) * 4 {
        return Err(Error::InvalidArgument);
    }
    if !(1..=100).contains(&params.dct_q) {
        return Err(Error::InvalidArgument);
    }
    let (nbx, nby) = (width.div_ceil(4), height.div_ceil(4));
    let (nbxs, total) = (nbx as usize, (nbx * nby) as usize);
    let ctx = Ctx {
        rgba,
        width: width as usize,
        height: height as usize,
        nbx,
        has_alpha: rgba.as_chunks::<4>().0.iter().any(|p| p[3] < 255),
        p: params,
        q: Fx::from_int(params.dct_q as i32),
        pred_mask: build_pred_mask(params.effort_level),
        num_amp: amp_codes_for_effort(params.effort_level),
    };

    // BC7 base pack, rows in parallel.
    let mut blocks = new_blocks(total)?;
    let mut lut = vas::Vector::<bool, vas::storage::Dynamic>::default();
    lut.resize(total, false).map_err(|_| Error::OutOfMemory)?;
    {
        let bc7e = (params.bc7_encoder == Bc7Encoder::Bc7eScalar).then(|| bc7e_params(params));
        let mut rows: Vec<(u32, &mut [LogBc7Block], &mut [bool])> = blocks
            .as_mut_slice()
            .chunks_mut(nbxs)
            .zip(lut.as_mut_slice().chunks_mut(nbxs))
            .enumerate()
            .map(|(by, (r, l))| (by as u32, r, l))
            .collect();
        crate::parallel::try_for_each_item(&mut rows, |(by, row, l)| {
            for (bx, (b, u)) in row.iter_mut().zip(l.iter_mut()).enumerate() {
                let px = ctx.block(bx as u32, *by);
                (*b, *u) = base_pack_block(&ctx, bc7e.as_ref(), &px)?;
            }
            Ok(())
        })?;
    }

    // Stripes: RDO + coding, in parallel.
    let ns = stripe_count(params.stripes, height, nby);
    let (base_rows, extra) = (nby / ns, nby % ns);
    let mut jobs: Vec<(StripeJob, StripeOut)> = Vec::new();
    {
        let mut rest = blocks.as_mut_slice();
        let mut lut_rest = lut.as_slice();
        let mut row = 0u32;
        for i in 0..ns {
            let n = base_rows + u32::from(i < extra);
            let (head, tail) = rest.split_at_mut((n * nbx) as usize);
            let (lh, lt) = lut_rest.split_at((n * nbx) as usize);
            jobs.push((
                StripeJob {
                    first_row: row,
                    rows: head,
                    lut: lh,
                },
                StripeOut::default(),
            ));
            rest = tail;
            lut_rest = lt;
            row += n;
        }
        crate::parallel::try_for_each_item(&mut jobs, |(job, out)| encode_stripe(&ctx, job, out))?;
    }

    // Merge the stripe streams in order, with the seek table (deltas, byte planes).
    let mut blobs = Blobs::default();
    let (mut signs, mut pbits, mut ep_raw) = (
        BitWriter::default(),
        BitWriter::default(),
        BitWriter::default(),
    );
    let seek_streams = BLOB_STRIPE_SEEK_TABLE - 1;
    let mut seek = vas::Vector::<u32, vas::storage::Dynamic>::default();
    let mut prev = [0u64; BLOB_STRIPE_SEEK_TABLE];
    for (_, out) in &jobs {
        if ns > 1 {
            for (id, pv) in prev.iter_mut().enumerate().skip(1) {
                let ofs = match id {
                    BLOB_COEFF_SIGNS => signs.total,
                    BLOB_PBITS => pbits.total,
                    BLOB_EP_RAW => ep_raw.total,
                    _ => blobs.0[id].len() as u64,
                };
                seek.push((ofs - *pv) as u32)
                    .map_err(|_| Error::OutOfMemory)?;
                *pv = ofs;
            }
        }
        for (dst, src) in blobs.0.iter_mut().zip(out.blobs.0.iter()) {
            extend(dst, src)?;
        }
        signs.append(&out.signs)?;
        pbits.append(&out.pbits)?;
        ep_raw.append(&out.ep_raw)?;
    }
    drop(jobs);
    for (id, w) in [
        (BLOB_EP_RAW, &mut ep_raw),
        (BLOB_PBITS, &mut pbits),
        (BLOB_COEFF_SIGNS, &mut signs),
    ] {
        if w.total != 0 {
            w.flush()?;
            blobs.0[id] = std::mem::take(&mut w.bytes);
        }
    }
    if ns > 1 {
        debug_assert_eq!(seek.len(), ns as usize * seek_streams);
        let t = &mut blobs.0[BLOB_STRIPE_SEEK_TABLE];
        for plane in 0..4 {
            for &e in seek.iter() {
                push(t, (e >> (plane * 8)) as u8)?;
            }
        }
    }
    let hdr = [
        width as u8,
        (width >> 8) as u8,
        height as u8,
        (height >> 8) as u8,
        params.dct_q as u8,
        u8::from(ctx.has_alpha),
        ns as u8,
    ];
    extend(&mut blobs.0[BLOB_HEADER], &hdr)?;
    let mut data = serialize(&blobs, params.zstd_level)?;

    // Tiny mip: raw BC7 blocks when smaller than the blob container.
    if nbx <= 255 && nby <= 255 && 3 + total * 16 < data.len() {
        let mut tiny = Buffer::default();
        extend(
            &mut tiny,
            &[
                if ctx.has_alpha { 0xB9 } else { 0xB8 },
                nbx as u8,
                nby as u8,
            ],
        )?;
        for b in blocks.as_mut_slice() {
            let mut phys = [0u8; 16];
            if !bc7u::pack_bc7(b, &mut phys) {
                return Err(Error::InvalidData);
            }
            extend(&mut tiny, &phys)?;
            // The decoder yields the re-canonicalized unpacked form.
            if !bc7u::unpack_phys(&phys, b) {
                return Err(Error::InvalidData);
            }
        }
        data = tiny;
    }

    if params.self_validate {
        let dec = decoder::decode(&data, nbx, nby)?;
        let ok = dec.blocks.len() == blocks.len()
            && dec
                .blocks
                .iter()
                .zip(blocks.iter())
                .all(|(a, b)| compare_full(b, a));
        if !ok {
            return Err(Error::InvalidData);
        }
    }
    Ok(Encoded { data, blocks })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;

    type TestResult = std::result::Result<(), Box<dyn StdError>>;

    /// Parameter sets of tools/basisu-golden/xubc7_encode.cpp: (id, quality, effort, rdo,
    /// desired stripes, bc7e level or -1 for bc7f, sRGB, exact stripes or 0).
    type Set = (&'static str, i32, u32, u32, u32, i32, bool, u32);
    const SETS: [Set; 12] = [
        ("default", -1, 3, 0, 8, -1, true, 0),
        ("q80", 80, 3, 0, 8, -1, true, 0),
        ("q50e0", 50, 0, 0, 8, -1, true, 0),
        ("q20e1", 20, 1, 0, 8, -1, true, 0),
        ("q90e10", 90, 10, 0, 8, -1, true, 0),
        ("q70e5r50", 70, 5, 50, 8, -1, true, 0),
        ("q30e2r100lin", 30, 2, 100, 8, -1, false, 0),
        ("bc7e2", -1, 3, 0, 8, 2, true, 0),
        ("bc7e0q60lin", 60, 3, 0, 8, 0, false, 0),
        ("bc7e4q85e6r20", 85, 6, 20, 8, 4, true, 0),
        ("q75e4s3", 75, 4, 0, 8, -1, true, 3),
        ("q100r30s4lin", -1, 3, 30, 8, -1, false, 4),
    ];

    fn params(id: &str) -> Option<EncodeParams> {
        let &(_, quality, effort_level, rdo_level, num_stripes, bc7e, srgb, exact) =
            SETS.iter().find(|s| s.0 == id)?;
        let mut b = BasisuParams {
            quality,
            effort_level,
            rdo_level,
            num_stripes,
            ..BasisuParams::default()
        };
        if bc7e >= 0 {
            b.encoder = Bc7Encoder::Bc7eScalar;
            b.bc7e_level = bc7e as u32;
        }
        b.set_srgb(srgb);
        let mut p = EncodeParams::from_basisu(&b);
        if exact != 0 {
            p.stripes = Stripes::Exact(exact);
        }
        Some(p)
    }

    fn fnv(b: &[u8]) -> u64 {
        b.iter().fold(0xcbf29ce484222325u64, |h, &x| {
            (h ^ u64::from(x)).wrapping_mul(0x100000001b3)
        })
    }

    struct Img {
        w: u32,
        h: u32,
        px: Vec<u8>,
    }

    const RAW_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/images/raw");

    fn raw(dir: &str, name: &str) -> std::result::Result<Img, Box<dyn StdError>> {
        let dims = name
            .trim_end_matches(".rgba")
            .rsplit('_')
            .next()
            .ok_or("name")?;
        let (w, h) = dims.split_once('x').ok_or("dims")?;
        let path = format!("{dir}/{name}");
        Ok(Img {
            w: w.parse()?,
            h: h.parse()?,
            px: std::fs::read(path)?,
        })
    }

    /// The golden tool's image variants: `NAME`, `NAME:opaque`, `NAME:tiny`, `NAME:tall`.
    fn image(dir: &str, key: &str) -> std::result::Result<Img, Box<dyn StdError>> {
        let (name, variant) = key.split_once(':').unwrap_or((key, ""));
        let img = raw(dir, name)?;
        let (w, h) = (img.w as usize, img.h as usize);
        Ok(match variant {
            "" => img,
            "opaque" => Img {
                px: img
                    .px
                    .chunks(4)
                    .flat_map(|p| [p[0], p[1], p[2], 255])
                    .collect(),
                ..img
            },
            "tiny" => Img {
                w: 6,
                h: 5,
                px: (0..5)
                    .flat_map(|y| img.px[y * w * 4..y * w * 4 + 24].to_vec())
                    .collect(),
            },
            "tall" => Img {
                w: img.w,
                h: img.h * 4,
                px: (0..h * 4)
                    .flat_map(|y| img.px[(y % h) * w * 4..(y % h + 1) * w * 4].to_vec())
                    .collect(),
            },
            _ => return Err("variant".into()),
        })
    }

    /// Encodes every line of a golden file (tools/basisu-golden/xubc7_encode.cpp output),
    /// returns (lines, mismatches).
    fn check_golden(
        golden: &str,
        dir: &str,
    ) -> std::result::Result<(usize, Vec<String>), Box<dyn StdError>> {
        let golden = std::fs::read_to_string(golden)?;
        let mut bad = Vec::new();
        let mut n = 0;
        for line in golden.lines() {
            let f: Vec<&str> = line.split(' ').collect();
            let img = image(dir, f[0])?;
            let p = params(f[1]).ok_or("params")?;
            let out = encode(&img.px, img.w, img.h, &p)?;
            let got = format!("{} {} {} {:016x}", f[0], f[1], out.len(), fnv(&out));
            if got != line {
                bad.push(format!("want {line}\n got {got}"));
            }
            n += 1;
        }
        Ok((n, bad))
    }

    /// Every line of testdata/encode/xubc7_golden.txt (C++ basisu v2.50 `xbc7::pack_image`).
    #[test]
    fn matches_basisu() -> TestResult {
        let golden = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/encode/xubc7_golden.txt"
        );
        let (n, bad) = check_golden(golden, RAW_DIR)?;
        assert!(
            bad.is_empty(),
            "{} mismatches:\n{}",
            bad.len(),
            bad.join("\n")
        );
        assert_eq!(n, 84);
        Ok(())
    }

    /// Offline check on larger images: `XUBC7_GOLDEN=file XUBC7_RAW=dir cargo test --release
    /// -- --ignored xubc7_extra` (golden from xubc7_encode run on the dir's NAME_WxH.rgba).
    #[test]
    #[ignore]
    fn xubc7_extra_golden() -> TestResult {
        let golden = std::env::var("XUBC7_GOLDEN")?;
        let dir = std::env::var("XUBC7_RAW")?;
        let (n, bad) = check_golden(&golden, &dir)?;
        assert!(
            bad.is_empty(),
            "{} of {n} mismatches:\n{}",
            bad.len(),
            bad.join("\n")
        );
        assert!(n > 0);
        Ok(())
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let se: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum();
        let mse = se / a.len() as f64;
        if mse == 0.0 {
            100.0
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        }
    }

    /// Pixels of row major logical blocks cropped to w x h.
    fn pixels(
        blocks: &[LogBc7Block],
        w: u32,
        h: u32,
    ) -> std::result::Result<Vec<u8>, Box<dyn StdError>> {
        let nbx = w.div_ceil(4) as usize;
        let mut out = vec![0u8; (w * h * 4) as usize];
        for (i, b) in blocks.iter().enumerate() {
            let px = unpack(b)?;
            for (t, p) in px.iter().enumerate() {
                let (x, y) = ((i % nbx) * 4 + (t & 3), (i / nbx) * 4 + (t >> 2));
                if x < w as usize && y < h as usize {
                    let o = (y * w as usize + x) * 4;
                    out[o..o + 4].copy_from_slice(p);
                }
            }
        }
        Ok(out)
    }

    /// Round trip: the stream decodes (via the XUBC7 decoder) to the coded blocks; at Q 100 the
    /// pixels equal the direct BC7 base encoder output, lossy levels stay close to the source.
    #[test]
    fn round_trip() -> TestResult {
        for name in [
            "rgba_64x48.rgba",
            "rgb_37x29.rgba",
            "mosaic_rgba_50x38.rgba:tall",
        ] {
            let img = image(RAW_DIR, name)?;
            let (nbx, nby) = (img.w.div_ceil(4), img.h.div_ceil(4));
            for encoder in [Bc7Encoder::Bc7f, Bc7Encoder::Bc7eScalar] {
                // Lossless weights, no weight re-optimization: exactly the base encoder.
                let p = EncodeParams {
                    bc7_encoder: encoder,
                    stripes: Stripes::Exact(3),
                    ..EncodeParams::pack_options()
                };
                let enc = encode_blocks(&img.px, img.w, img.h, &p)?;
                let dec = decoder::decode(&enc.data, nbx, nby)?;
                assert!(
                    dec.blocks
                        .iter()
                        .zip(enc.blocks.iter())
                        .all(|(a, b)| compare_full(a, b))
                );
                let ctx_params = p.clone();
                let ctx = Ctx {
                    rgba: &img.px,
                    width: img.w as usize,
                    height: img.h as usize,
                    nbx,
                    has_alpha: true,
                    p: &ctx_params,
                    q: Fx::from_int(100),
                    pred_mask: 0,
                    num_amp: 0,
                };
                let bp = bc7e_params(&p);
                let mut direct = Vec::new();
                for by in 0..nby {
                    for bx in 0..nbx {
                        let px = ctx.block(bx, by);
                        let phys = match encoder {
                            Bc7Encoder::Bc7f => {
                                let mut b = [0u8; 16];
                                bc7f::fast_pack_bc7_auto_rgba(&mut b, &px, p.bc7_pack_flags);
                                b
                            }
                            Bc7Encoder::Bc7eScalar => crate::bc7e::compress_block(&px, &bp),
                        };
                        let mut l = LogBc7Block::default();
                        assert!(bc7u::unpack_phys(&phys, &mut l));
                        direct.push(l);
                    }
                }
                assert_eq!(
                    pixels(&dec.blocks, img.w, img.h)?,
                    pixels(&direct, img.w, img.h)?,
                    "{name} {encoder:?}"
                );

                // Lossy weight DCT + RDO.
                for (q, rdo) in [(80, 0), (40, 60)] {
                    let mut b = BasisuParams {
                        quality: q,
                        rdo_level: rdo,
                        encoder,
                        ..BasisuParams::default()
                    };
                    b.set_srgb(false);
                    let enc = encode_blocks(&img.px, img.w, img.h, &EncodeParams::from_basisu(&b))?;
                    let dec = decoder::decode(&enc.data, nbx, nby)?;
                    let got = psnr(&img.px, &pixels(&dec.blocks, img.w, img.h)?);
                    let base = psnr(&img.px, &pixels(&direct, img.w, img.h)?);
                    assert!(
                        got > 30.0 && got <= base + 0.5,
                        "{name} {encoder:?} q{q} r{rdo}: {got} vs {base}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_bad_input() {
        let p = EncodeParams::default();
        assert!(encode(&[0; 16], 0, 4, &p).is_err());
        assert!(encode(&[0; 15], 2, 2, &p).is_err());
        let q0 = EncodeParams {
            dct_q: 0,
            ..EncodeParams::default()
        };
        assert!(encode(&[0; 16], 2, 2, &q0).is_err());
    }

    #[test]
    fn stripe_counts() {
        assert_eq!(stripe_count(Stripes::Desired(8), 127, 32), 1);
        assert_eq!(stripe_count(Stripes::Desired(8), 152, 38), 2);
        assert_eq!(stripe_count(Stripes::Desired(8), 4096, 1024), 8);
        assert_eq!(stripe_count(Stripes::Auto, 4096, 1024), 16);
        assert_eq!(stripe_count(Stripes::Exact(40), 8, 2), 2);
        assert_eq!(build_pred_mask(10), (1u64 << TOTAL_CANDS) - 1);
        assert_eq!(build_pred_mask(0).count_ones(), 6);
    }
}
