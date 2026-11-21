// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! XUASTC LDR ("supercompressed ASTC LDR", basisu tex formats 5..18) decompression to ASTC
//! logical blocks: port of `astc_ldr_t::xuastc_ldr_decompress_image` (full arithmetic and
//! hybrid arithmetic + Zstd syntaxes) and `xuastc_ldr_decompress_image_full_zstd`, including
//! the DCT weight grid mode.
//!
//! The C++ decoder keeps an 8 row ring of logical blocks and reports each block through a
//! callback; this port keeps every block of the image (all references point backwards, so the
//! decoded blocks are the same).

mod arith;
mod dct;
mod endpoints;
mod tables;

use arith::{ArithDec, BitModel, DataModel, GammaContexts};
use dct::{Coeff, DCT_MEAN_LEVELS1, DctSyms, decode_block_weights, num_weight_dc_levels};
use endpoints::{
    MAX_CEM_VALS, cem_supports_bc, convert_endpoints_across_cems, decode_endpoints,
    used_blue_contraction,
};

use super::file::BASIS_ASTC_ORDER;
use super::ll::astc_6x6_hdr::decode_values;
use super::ll::bitreader::BitwiseDecoder;
use super::ll::hdr_tables::REUSE_XY_DELTAS;
use crate::astc::helpers::{
    self as ah, CEM_LDR_LUM_ALPHA_DIRECT, CEM_LDR_LUM_DIRECT, CEM_LDR_RGB_BASE_PLUS_OFFSET,
    CEM_LDR_RGB_BASE_SCALE, CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A, CEM_LDR_RGB_DIRECT,
    CEM_LDR_RGBA_BASE_PLUS_OFFSET, CEM_LDR_RGBA_DIRECT, FIRST_VALID_ENDPOINT_ISE_RANGE,
    FIRST_VALID_WEIGHT_ISE_RANGE, LogAstcBlock,
};
use crate::{AstcBlock, Error, Result};
use std::sync::OnceLock;
use vas::Vector;
use vas::storage::Dynamic;

const ARITH_HEADER_MARKER: u32 = 0x01;
const ARITH_HEADER_MARKER_BITS: u32 = 5;
const FULL_ZSTD_HEADER_MARKER: u32 = 0x01;
const FULL_ZSTD_HEADER_MARKER_BITS: u32 = 5;
const FINAL_SYNC_MARKER: u32 = 0xAF;
const FINAL_SYNC_MARKER_BITS: u32 = 8;
const MAX_CONFIG_REUSE_NEIGHBORS: u32 = 3;
const DCT_RUN_LEN_EOB_SYM_INDEX: u32 = 64;
const DCT_MEAN_LEVELS0: u32 = 9;
const PART_HASH_SIZE: usize = 64;
const TM_HASH_BITS: u32 = 7;
const TM_HASH_SIZE: usize = 1 << TM_HASH_BITS;
const PART_HASH_BITS: u32 = 6;

const OTM_NUM_CEMS: u32 = 14;
const OTM_NUM_SUBSETS: u32 = 3;
const OTM_NUM_CCS: u32 = 5;
const OTM_NUM_GRID_SIZES: u32 = 2;
const OTM_NUM_GRID_ANISOS: u32 = 3;
const NUM_GROUPS: usize =
    (OTM_NUM_CEMS * OTM_NUM_SUBSETS * OTM_NUM_CCS * OTM_NUM_GRID_SIZES * OTM_NUM_GRID_ANISOS)
        as usize;

/// `xuastc_ldr_syntax`.
const SYNTAX_HYBRID_ARITH_ZSTD: u8 = 1;
const SYNTAX_FULL_ZSTD: u8 = 2;

/// `xuastc_ldr_arith_header`: flags byte + 11 lengths.
const ARITH_HEADER_SIZE: usize = 1 + 11 * 4;
/// `xuastc_ldr_full_zstd_header`: flags byte + 21 lengths.
const FULL_ZSTD_HEADER_SIZE: usize = 1 + 21 * 4;

/// `xuastc_mode` (arithmetic syntaxes).
const MODE_SOLID: u32 = 0;
const MODE_RAW: u32 = 1;
const MODE_REUSE_CFG_ENDPOINTS_LEFT: u32 = 2;
const MODE_REUSE_CFG_ENDPOINTS_UP: u32 = 3;
const MODE_REUSE_CFG_ENDPOINTS_DIAG: u32 = 4;
const MODE_RUN: u32 = 5;
const MODE_TOTAL: u32 = 6;

/// `xuastc_zstd_mode` and mode byte flags (full Zstd syntax).
const ZMODE_RUN: u32 = 0b01;
const ZMODE_SOLID: u32 = 0b0011;
const ZMODE_REUSE_CFG_ENDPOINTS_LEFT: u32 = 0b0111;
const MODE_BYTE_IS_BASE_OFS_FLAG: u32 = 1 << 3;
const MODE_BYTE_PART_HASH_HIT: u32 = 1 << 4;
const MODE_BYTE_DPCM_ENDPOINTS_FLAG: u32 = 1 << 5;
const MODE_BYTE_TM_HASH_HIT_FLAG: u32 = 1 << 6;
const MODE_BYTE_USE_DCT: u32 = 1 << 7;

/// One decompressed XUASTC LDR image: ASTC logical blocks, row major. (The transcoder reads
/// only the blocks and flags; the geometry is checked through [`Expected`].)
#[allow(dead_code)]
pub(crate) struct XuastcImage {
    pub block: AstcBlock,
    pub width: u32,
    pub height: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub has_alpha: bool,
    /// ASTC sRGB decode profile (from the stream header).
    pub srgb: bool,
    pub blocks: Vector<LogAstcBlock, Dynamic>,
}

/// Block size and block counts the container expects (checked before any allocation, like
/// the C++ transcoder's init callback).
pub(crate) type Expected = (AstcBlock, u32, u32);

// ---------------------------------------------------------------------------------------
// Trial modes (the ASTC configurations a stream can reference)

/// `trial_mode`.
#[derive(Clone, Copy, Default)]
struct TrialMode {
    grid_w: u8,
    grid_h: u8,
    cem: u8,
    /// -1: single plane.
    ccs_index: i8,
    endpoint_range: u8,
    weight_range: u8,
    num_parts: u8,
}

/// `g_encoder_trial_modes[i]` and `g_grouped_encoder_trial_modes[i]` of one block size.
struct ModeTable {
    modes: Vec<TrialMode>,
    /// `(start, len)` into `group_modes` per group.
    groups: Vec<(u32, u32)>,
    /// Trial mode indices, grouped, ascending within a group.
    group_modes: Vec<u16>,
}

/// `s_unique_ldr_index_to_astc_cem`.
const UNIQUE_LDR_INDEX_TO_ASTC_CEM: [u32; 6] = [
    CEM_LDR_LUM_DIRECT,
    CEM_LDR_LUM_ALPHA_DIRECT,
    CEM_LDR_RGB_BASE_SCALE,
    CEM_LDR_RGB_DIRECT,
    CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A,
    CEM_LDR_RGBA_DIRECT,
];

/// `calc_grid_aniso_val`.
fn grid_aniso(gw: u32, gh: u32, bw: u32, bh: u32) -> u32 {
    let (lhs, rhs) = (gw * bh, gh * bw);
    if lhs == rhs {
        0
    } else if lhs >= rhs {
        1
    } else {
        2
    }
}

fn group_index(cem: u32, subset: u32, ccs: u32, size: u32, aniso: u32) -> Option<usize> {
    if cem >= OTM_NUM_CEMS
        || subset >= OTM_NUM_SUBSETS
        || ccs >= OTM_NUM_CCS
        || size >= OTM_NUM_GRID_SIZES
        || aniso >= OTM_NUM_GRID_ANISOS
    {
        return None;
    }
    let i = (((cem * OTM_NUM_SUBSETS + subset) * OTM_NUM_CCS + ccs) * OTM_NUM_GRID_SIZES + size)
        * OTM_NUM_GRID_ANISOS
        + aniso;
    Some(i as usize)
}

/// `create_encoder_trial_modes_table`.
fn build_mode_table(bw: u32, bh: u32) -> ModeTable {
    let mut modes = Vec::new();
    let mut keys = Vec::new();
    for c in tables::ASTC_CFG_TABLE
        .as_chunks::<3>()
        .0
        .iter()
        .take(tables::TOTAL_ASTC_CFGS)
    {
        let mut packed = u32::from(c[0]) | (u32::from(c[1]) << 8) | (u32::from(c[2]) << 16);
        let mut field = |bits: u32| {
            let v = packed & ((1 << bits) - 1);
            packed >>= bits;
            v
        };
        let endpoint_range = field(5);
        let weight_range = field(4);
        let ccs_index = field(3);
        let num_subsets = field(2);
        let unique_cem = field(3);
        let grid_wh = field(7);
        let grid_w = grid_wh / 11 + 2;
        // Sorted by grid width.
        if grid_w > bw {
            break;
        }
        let grid_h = grid_wh % 11 + 2;
        if grid_h > bh {
            continue;
        }
        let cem = UNIQUE_LDR_INDEX_TO_ASTC_CEM
            .get(unique_cem as usize)
            .copied()
            .unwrap_or(0);
        let tm = TrialMode {
            grid_w: grid_w as u8,
            grid_h: grid_h as u8,
            cem: cem as u8,
            ccs_index: ccs_index as i8 - 1,
            endpoint_range: (endpoint_range + FIRST_VALID_ENDPOINT_ISE_RANGE) as u8,
            weight_range: (weight_range + FIRST_VALID_WEIGHT_ISE_RANGE) as u8,
            num_parts: (num_subsets + 1) as u8,
        };
        let size = u32::from(grid_w >= bw - 1 && grid_h >= bh - 1);
        keys.push(group_index(
            cem,
            num_subsets,
            ccs_index,
            size,
            grid_aniso(grid_w, grid_h, bw, bh),
        ));
        modes.push(tm);
    }
    let mut groups = vec![(0u32, 0u32); NUM_GROUPS];
    for g in keys.iter().flatten() {
        groups[*g].1 += 1;
    }
    let mut start = 0;
    for g in groups.iter_mut() {
        g.0 = start;
        start += g.1;
    }
    let mut fill = groups.iter().map(|g| g.0).collect::<Vec<_>>();
    let mut group_modes = vec![0u16; start as usize];
    for (i, g) in keys.iter().enumerate() {
        if let Some(g) = g {
            group_modes[fill[*g] as usize] = i as u16;
            fill[*g] += 1;
        }
    }
    ModeTable {
        modes,
        groups,
        group_modes,
    }
}

fn mode_table(size_index: usize) -> Option<&'static ModeTable> {
    static T: [OnceLock<ModeTable>; 14] = [const { OnceLock::new() }; 14];
    let (bw, bh) = BASIS_ASTC_ORDER.get(size_index)?.dim();
    Some(T.get(size_index)?.get_or_init(|| build_mode_table(bw, bh)))
}

impl ModeTable {
    fn get(&self, tm_index: u32) -> Result<TrialMode> {
        self.modes
            .get(tm_index as usize)
            .copied()
            .ok_or(Error::InvalidData)
    }

    /// `get_tm_candidates`.
    fn candidates(&self, cem: u32, subset: u32, ccs: u32, size: u32, aniso: u32) -> &[u16] {
        group_index(cem, subset, ccs, size, aniso)
            .and_then(|g| self.groups.get(g))
            .and_then(|&(s, n)| self.group_modes.get(s as usize..(s + n) as usize))
            .unwrap_or(&[])
    }
}

/// `get_total_unique_patterns`.
fn total_unique_patterns(size_index: usize, num_parts: u32) -> u32 {
    tables::TOTAL_UNIQUE_PATTERNS
        .get(size_index)
        .and_then(|p| p.get((num_parts as usize).wrapping_sub(2)))
        .map_or(0, |&n| u32::from(n))
}

/// `unique_pat_index_to_part_seed`.
fn part_seed(size_index: usize, num_parts: u32, unique: u32) -> Result<u16> {
    let t = match num_parts {
        2 => &tables::UNIQUE_TO_SEED_P2,
        3 => &tables::UNIQUE_TO_SEED_P3,
        _ => return Err(Error::InvalidData),
    };
    t.get(size_index)
        .and_then(|s| s.get(unique as usize))
        .copied()
        .ok_or(Error::InvalidData)
}

/// `part_hash_index` / `tm_hash_index` (Fibonacci hashing).
fn fib_hash(x: u32, size: usize) -> usize {
    (x.wrapping_mul(2654435769) as usize) & (size - 1)
}

/// `cem_to_ldrcem_index`.
fn cem_to_ldrcem_index(cem: u32) -> usize {
    match cem {
        CEM_LDR_LUM_DIRECT => 0,
        CEM_LDR_LUM_ALPHA_DIRECT => 1,
        CEM_LDR_RGB_BASE_SCALE => 2,
        CEM_LDR_RGB_DIRECT => 3,
        CEM_LDR_RGB_BASE_PLUS_OFFSET => 4,
        CEM_LDR_RGB_BASE_SCALE_PLUS_TWO_A => 5,
        CEM_LDR_RGBA_DIRECT => 6,
        CEM_LDR_RGBA_BASE_PLUS_OFFSET => 7,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------------------
// Byte streams

/// `simplified_bitwise_decoder`: 1, 2 or 4 bit codes that never cross bytes, or whole bytes.
#[derive(Default)]
struct SimpleBits<'a> {
    buf: &'a [u8],
    pos: usize,
    bit_buf: u32,
}

impl<'a> SimpleBits<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            bit_buf: 1,
        }
    }

    fn byte(&mut self) -> u32 {
        match self.buf.get(self.pos) {
            Some(&b) => {
                self.pos += 1;
                u32::from(b)
            }
            None => 0,
        }
    }

    fn bits(&mut self, n: u32) -> u32 {
        if self.bit_buf <= 1 {
            self.bit_buf = 256 | self.byte();
        }
        let r = self.bit_buf & ((1 << n) - 1);
        self.bit_buf >>= n;
        r
    }

    fn bits1(&mut self) -> u32 {
        self.bits(1)
    }

    fn bits2(&mut self) -> u32 {
        self.bits(2)
    }

    fn bits4(&mut self) -> u32 {
        self.bits(4)
    }

    fn bits8(&mut self) -> u32 {
        self.byte()
    }

    fn at_end(&self) -> bool {
        self.pos >= self.buf.len()
    }
}

type Bytes = Vector<u8, Dynamic>;

/// `zstd_decompress`: one frame with a known content size (empty input: empty output).
fn zstd_decompress(src: &[u8]) -> Result<Bytes> {
    if src.is_empty() {
        return Ok(Bytes::default());
    }
    let mut dec = ruzstd::decoding::FrameDecoder::new();
    let mut header = src;
    dec.init(&mut header).map_err(|_| Error::InvalidData)?;
    let size = dec.content_size();
    if size > i32::MAX as u64 {
        return Err(Error::InvalidData);
    }
    let mut out = crate::buffer::zeroed(size as usize)?;
    if size == 0 {
        return Ok(out);
    }
    match dec.decode_all(src, &mut out) {
        Ok(n) if n as u64 == size => Ok(out),
        _ => Err(Error::InvalidData),
    }
}

/// Little endian `u32` field `i` of a packed header (after the flags byte).
fn header_len(data: &[u8], i: usize) -> Result<usize> {
    let o = 1 + i * 4;
    let b = data.get(o..o + 4).ok_or(Error::InvalidData)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
}

/// Splits `len` bytes off the front of `cur`.
fn take<'a>(cur: &mut &'a [u8], len: usize) -> Result<&'a [u8]> {
    if len > cur.len() {
        return Err(Error::InvalidData);
    }
    let (a, b) = cur.split_at(len);
    *cur = b;
    Ok(a)
}

/// The DCT weight streams shared by the full Zstd and hybrid syntaxes (decompressed).
struct WeightStreams {
    mean0: Bytes,
    mean1: Bytes,
    run: Bytes,
    coeff: Bytes,
    weight2: Bytes,
    weight3: Bytes,
    weight4: Bytes,
    weight8: Bytes,
}

/// Readers over [`WeightStreams`] (sign bits are stored uncompressed).
struct WeightReaders<'a> {
    mean0: SimpleBits<'a>,
    mean1: SimpleBits<'a>,
    run: SimpleBits<'a>,
    coeff: SimpleBits<'a>,
    sign: SimpleBits<'a>,
    weight2: SimpleBits<'a>,
    weight3: SimpleBits<'a>,
    weight4: SimpleBits<'a>,
    weight8: SimpleBits<'a>,
}

impl WeightStreams {
    fn readers<'a>(&'a self, sign: &'a [u8]) -> WeightReaders<'a> {
        WeightReaders {
            mean0: SimpleBits::new(&self.mean0),
            mean1: SimpleBits::new(&self.mean1),
            run: SimpleBits::new(&self.run),
            coeff: SimpleBits::new(&self.coeff),
            sign: SimpleBits::new(sign),
            weight2: SimpleBits::new(&self.weight2),
            weight3: SimpleBits::new(&self.weight3),
            weight4: SimpleBits::new(&self.weight4),
            weight8: SimpleBits::new(&self.weight8),
        }
    }
}

impl WeightReaders<'_> {
    /// DCT weight symbols of one plane from the byte streams.
    fn dct_syms(
        &mut self,
        num_dc_levels: u32,
        total_weights: u32,
        syms: &mut DctSyms,
    ) -> Result<()> {
        syms.num_coeffs = 0;
        syms.dc_sym = if num_dc_levels == DCT_MEAN_LEVELS1 {
            self.mean1.bits8()
        } else {
            self.mean0.bits4()
        };
        let mut zig = 1u32;
        while zig < total_weights {
            let run_len = self.run.bits8();
            if run_len == DCT_RUN_LEN_EOB_SYM_INDEX {
                break;
            }
            zig += run_len;
            if zig >= total_weights {
                return Err(Error::InvalidData);
            }
            let sign = self.sign.bits1();
            let mut coeff = self.coeff.bits8() as i32 + 1;
            if sign != 0 {
                coeff = -coeff;
            }
            push_coeff(syms, run_len, coeff)?;
            zig += 1;
        }
        Ok(())
    }

    /// Weight grid DPCM of one plane (`levels` weight levels).
    fn dpcm_weights(
        &mut self,
        levels: u32,
        rank_to_ise: &[u8],
        out: &mut [u8],
        plane: usize,
        planes: usize,
        total: usize,
    ) -> Result<()> {
        let mut prev_w = levels / 2;
        for i in 0..total {
            let r = if levels <= 4 {
                self.weight2.bits2()
            } else if levels <= 8 {
                self.weight3.bits4()
            } else if levels <= 16 {
                self.weight4.bits4()
            } else {
                self.weight8.bits8()
            };
            let w = (prev_w + r) % levels;
            prev_w = w;
            set_weight(out, plane + i * planes, rank_to_ise, w)?;
        }
        Ok(())
    }
}

fn push_coeff(syms: &mut DctSyms, run_len: u32, coeff: i32) -> Result<()> {
    let slot = syms
        .coeffs
        .get_mut(syms.num_coeffs)
        .ok_or(Error::InvalidData)?;
    *slot = Coeff {
        num_zeros: u16::try_from(run_len).map_err(|_| Error::InvalidData)?,
        coeff: i16::try_from(coeff).map_err(|_| Error::InvalidData)?,
    };
    syms.num_coeffs += 1;
    Ok(())
}

fn set_weight(out: &mut [u8], idx: usize, rank_to_ise: &[u8], rank: u32) -> Result<()> {
    let v = rank_to_ise
        .get(rank as usize)
        .copied()
        .ok_or(Error::InvalidData)?;
    *out.get_mut(idx).ok_or(Error::InvalidData)? = v;
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Shared block helpers

/// Block grid of the image being decoded.
struct Grid {
    size_index: usize,
    block: AstcBlock,
    bw: u32,
    bh: u32,
    blocks_x: u32,
    blocks_y: u32,
    blocks: Vector<LogAstcBlock, Dynamic>,
}

impl Grid {
    fn new(size_index: usize, width: u32, height: u32, expect: Option<Expected>) -> Result<Self> {
        let block = *BASIS_ASTC_ORDER.get(size_index).ok_or(Error::InvalidData)?;
        let (bw, bh) = block.dim();
        let (blocks_x, blocks_y) = (width.div_ceil(bw), height.div_ceil(bh));
        if expect.is_some_and(|e| e != (block, blocks_x, blocks_y)) {
            return Err(Error::InvalidData);
        }
        let n = (blocks_x as usize)
            .checked_mul(blocks_y as usize)
            .ok_or(Error::InvalidData)?;
        let mut blocks = Vector::default();
        blocks
            .resize(n, LogAstcBlock::default())
            .map_err(|_| Error::OutOfMemory)?;
        Ok(Self {
            size_index,
            block,
            bw,
            bh,
            blocks_x,
            blocks_y,
            blocks,
        })
    }

    fn at(&self, bx: u32, by: u32) -> Result<&LogAstcBlock> {
        self.blocks
            .get(by as usize * self.blocks_x as usize + bx as usize)
            .ok_or(Error::InvalidData)
    }

    fn set(&mut self, bx: u32, by: u32, b: LogAstcBlock) -> Result<()> {
        let i = by as usize * self.blocks_x as usize + bx as usize;
        *self.blocks.get_mut(i).ok_or(Error::InvalidData)? = b;
        Ok(())
    }

    /// Left block, else the upper one (runs and solid color prediction).
    fn prev(&self, bx: u32, by: u32) -> Option<&LogAstcBlock> {
        if bx > 0 {
            self.at(bx - 1, by).ok()
        } else if by > 0 {
            self.at(bx, by - 1).ok()
        } else {
            None
        }
    }

    /// Neighbor `index` (0 = left, 1 = upper, 2 = upper left) for configuration reuse.
    fn neighbor(&self, bx: u32, by: u32, index: u32) -> Result<&LogAstcBlock> {
        let (dx, dy) = match index {
            0 => (1, 0),
            1 => (0, 1),
            2 => (1, 1),
            _ => return Err(Error::InvalidData),
        };
        if bx < dx || by < dy {
            return Err(Error::InvalidData);
        }
        self.at(bx - dx, by - dy)
    }

    fn into_image(self, width: u32, height: u32, has_alpha: bool, srgb: bool) -> XuastcImage {
        XuastcImage {
            block: self.block,
            width,
            height,
            blocks_x: self.blocks_x,
            blocks_y: self.blocks_y,
            has_alpha,
            srgb,
            blocks: self.blocks,
        }
    }
}

/// Solid color predictor: the previous block's solid color, or the midpoint of its first
/// partition's endpoints.
fn prev_solid_color(prev: Option<&LogAstcBlock>) -> [u32; 4] {
    let Some(p) = prev else {
        return [0; 4];
    };
    if p.solid_color_flag_ldr {
        return [0, 1, 2, 3].map(|c| u32::from(p.solid_color(c) >> 8));
    }
    let (l, h) = decode_endpoints(
        u32::from(p.color_endpoint_modes[0]),
        &p.endpoints,
        u32::from(p.endpoint_ise_range),
    );
    [0, 1, 2, 3].map(|c| (u32::from(l[c]) + u32::from(h[c]) + 1) >> 1)
}

fn solid_block(rgba: [u32; 4]) -> LogAstcBlock {
    let mut b = LogAstcBlock {
        solid_color_flag_ldr: true,
        ..LogAstcBlock::default()
    };
    for (c, v) in rgba.iter().enumerate() {
        let v = (v & 0xFF) as u16;
        b.set_solid_color(c, v | (v << 8));
    }
    b
}

/// Copies configuration, partition id and endpoints of `src` (full reuse modes).
fn reuse_cfg_endpoints(blk: &mut LogAstcBlock, src: &LogAstcBlock) -> u32 {
    let cem = src.color_endpoint_modes[0];
    for i in 0..usize::from(src.num_partitions).min(4) {
        blk.color_endpoint_modes[i] = cem;
    }
    blk.dual_plane = src.dual_plane;
    blk.color_component_selector = src.color_component_selector;
    blk.num_partitions = src.num_partitions;
    blk.partition_id = src.partition_id;
    blk.endpoint_ise_range = src.endpoint_ise_range;
    blk.weight_ise_range = src.weight_ise_range;
    blk.grid_width = src.grid_width;
    blk.grid_height = src.grid_height;
    let n = (ah::get_num_cem_values(u32::from(cem)) as usize * usize::from(src.num_partitions))
        .min(blk.endpoints.len());
    blk.endpoints[..n].copy_from_slice(&src.endpoints[..n]);
    u32::from(cem)
}

/// Applies trial mode `tm` (with the actual CEM) to `blk`.
fn apply_trial_mode(blk: &mut LogAstcBlock, tm: &TrialMode, actual_cem: u32) {
    for i in 0..usize::from(tm.num_parts).min(4) {
        blk.color_endpoint_modes[i] = actual_cem as u8;
    }
    blk.num_partitions = tm.num_parts;
    blk.dual_plane = tm.ccs_index >= 0;
    if blk.dual_plane {
        blk.color_component_selector = tm.ccs_index as u8;
    }
    blk.weight_ise_range = tm.weight_range;
    blk.endpoint_ise_range = tm.endpoint_range;
    blk.grid_width = tm.grid_w;
    blk.grid_height = tm.grid_h;
}

/// RGB(A) direct trial modes code an "is base+offset" flag.
fn base_ofs_cem(cem: u32, is_base_ofs: bool) -> u32 {
    match (cem, is_base_ofs) {
        (CEM_LDR_RGB_DIRECT, true) => CEM_LDR_RGB_BASE_PLUS_OFFSET,
        (CEM_LDR_RGBA_DIRECT, true) => CEM_LDR_RGBA_BASE_PLUS_OFFSET,
        (c, _) => c,
    }
}

/// Endpoint DPCM predictor block at reuse delta `index` (must not be solid).
fn endpoint_pred_block(grid: &Grid, bx: u32, by: u32, index: u32) -> Result<LogAstcBlock> {
    let [dx, dy] = *REUSE_XY_DELTAS
        .get(index as usize)
        .ok_or(Error::InvalidData)?;
    let rx = bx as i32 + i32::from(dx);
    let ry = by as i32 + i32::from(dy);
    if rx < 0 || ry < 0 || rx >= grid.blocks_x as i32 || ry >= grid.blocks_y as i32 {
        return Err(Error::InvalidData);
    }
    let p = *grid.at(rx as u32, ry as u32)?;
    if p.solid_color_flag_ldr {
        return Err(Error::InvalidData);
    }
    Ok(p)
}

/// Predicts the endpoints of every partition of `blk` from `pred` (`convert_endpoints_across_cems`).
fn predict_endpoints(
    pred: &LogAstcBlock,
    blk: &LogAstcBlock,
    use_bc: &[bool; 4],
) -> Result<[[u8; MAX_CEM_VALS]; 4]> {
    let mut src = [0u8; MAX_CEM_VALS];
    src.copy_from_slice(&pred.endpoints[..MAX_CEM_VALS]);
    let mut out = [[0u8; MAX_CEM_VALS]; 4];
    for (p, o) in out
        .iter_mut()
        .enumerate()
        .take(usize::from(blk.num_partitions))
    {
        if !convert_endpoints_across_cems(
            u32::from(pred.color_endpoint_modes[0]),
            u32::from(pred.endpoint_ise_range),
            &src,
            u32::from(blk.color_endpoint_modes[0]),
            u32::from(blk.endpoint_ise_range),
            o,
            use_bc[p],
        ) {
            return Err(Error::InvalidData);
        }
    }
    Ok(out)
}

/// Endpoint DPCM: each value is `rank_to_ise[(delta() + rank(predicted)) % levels]`.
fn apply_endpoint_dpcm(
    blk: &mut LogAstcBlock,
    predicted: &[[u8; MAX_CEM_VALS]; 4],
    num_parts: usize,
    vals: usize,
    mut delta: impl FnMut() -> u32,
) -> Result<()> {
    let range = u32::from(blk.endpoint_ise_range);
    let levels = ah::get_ise_levels(range);
    let t = ah::dequant_tables().endpoint_tab(range);
    for (p, pred_vals) in predicted.iter().enumerate().take(num_parts) {
        for v in 0..vals {
            let d = delta();
            let pred = *pred_vals.get(v).ok_or(Error::InvalidData)?;
            let rank = u32::from(
                t.ise_to_rank
                    .get(usize::from(pred))
                    .copied()
                    .ok_or(Error::InvalidData)?,
            );
            let e = (d + rank) % levels.max(1);
            let ise = t
                .rank_to_ise
                .get(e as usize)
                .copied()
                .ok_or(Error::InvalidData)?;
            *blk.endpoints
                .get_mut(p * vals + v)
                .ok_or(Error::InvalidData)? = ise;
        }
    }
    Ok(())
}

fn weight_rank_to_ise(range: u32) -> &'static [u8] {
    &ah::dequant_tables().weight_tab(range).rank_to_ise
}

// ---------------------------------------------------------------------------------------
// Header

struct StreamHeader {
    size_index: usize,
    srgb: bool,
    width: u32,
    height: u32,
    has_alpha: bool,
    use_dct: bool,
    dct_q: f32,
}

/// `bits(n, single)`: `single` marks the fields the arithmetic decoder reads with `get_bit`.
fn read_header(mut bits: impl FnMut(u32, bool) -> u32, marker: (u32, u32)) -> Result<StreamHeader> {
    if bits(marker.1, false) != marker.0 {
        return Err(Error::InvalidData);
    }
    let size_index = bits(4, false) as usize;
    if size_index >= BASIS_ASTC_ORDER.len() {
        return Err(Error::InvalidData);
    }
    let srgb = bits(1, true) != 0;
    let width = bits(16, false);
    let height = bits(16, false);
    let has_alpha = bits(1, true) != 0;
    let use_dct = bits(1, false) != 0;
    let int_q = if use_dct { bits(8, false) } else { 0 };
    let dct_q = int_q as f32 / 2.0;
    if use_dct && (dct_q <= 0.0 || dct_q > 100.0) {
        return Err(Error::InvalidData);
    }
    Ok(StreamHeader {
        size_index,
        srgb,
        width,
        height,
        has_alpha,
        use_dct,
        dct_q,
    })
}

/// Decompresses one XUASTC LDR image (any syntax) to ASTC logical blocks.
pub(crate) fn decompress(data: &[u8], expect: Option<Expected>) -> Result<XuastcImage> {
    match data.first() {
        None => Err(Error::InvalidData),
        Some(&SYNTAX_FULL_ZSTD) => decompress_full_zstd(data, expect),
        Some(_) => decompress_arith(data, expect),
    }
}

// ---------------------------------------------------------------------------------------
// Full Zstd syntax

/// `prev_block_state_full_zstd`: trial mode index, -1 for solid blocks.
type ZstdPrevState = i32;

fn decompress_full_zstd(data: &[u8], expect: Option<Expected>) -> Result<XuastcImage> {
    if data.len() < FULL_ZSTD_HEADER_SIZE {
        return Err(Error::InvalidData);
    }
    let len = |i| header_len(data, i);
    let (raw_len, mode_len) = (len(0)?, len(1)?);
    if raw_len == 0 || mode_len == 0 {
        return Err(Error::InvalidData);
    }
    let mut cur = &data[FULL_ZSTD_HEADER_SIZE..];
    let raw = take(&mut cur, raw_len)?;
    let z = |cur: &mut &[u8], i: usize| -> Result<Bytes> { zstd_decompress(take(cur, len(i)?)?) };
    let mode_bytes = z(&mut cur, 1)?;
    let solid_dpcm = z(&mut cur, 2)?;
    let reuse_indices = z(&mut cur, 3)?;
    let use_bc_bits = z(&mut cur, 4)?;
    let mut dpcm: [Bytes; 6] = Default::default();
    for (i, d) in dpcm.iter_mut().enumerate() {
        *d = z(&mut cur, 5 + i)?;
    }
    let mean0 = z(&mut cur, 11)?;
    let mean1 = z(&mut cur, 12)?;
    let run = z(&mut cur, 13)?;
    let coeff = z(&mut cur, 14)?;
    let sign = take(&mut cur, len(15)?)?;
    let streams = WeightStreams {
        mean0,
        mean1,
        run,
        coeff,
        weight2: z(&mut cur, 16)?,
        weight3: z(&mut cur, 17)?,
        weight4: z(&mut cur, 18)?,
        weight8: z(&mut cur, 19)?,
    };

    let mut raw_bits = BitwiseDecoder::new(raw);
    let hdr = read_header(
        |n, _| raw_bits.get_bits(n),
        (FULL_ZSTD_HEADER_MARKER, FULL_ZSTD_HEADER_MARKER_BITS),
    )?;
    let mut grid = Grid::new(hdr.size_index, hdr.width, hdr.height, expect)?;
    let tms = mode_table(hdr.size_index).ok_or(Error::InvalidData)?;
    let (bxs, bys) = (grid.blocks_x, grid.blocks_y);

    let mut mode_dec = SimpleBits::new(&mode_bytes);
    let mut solid_dec = SimpleBits::new(&solid_dpcm);
    let mut reuse_dec = SimpleBits::new(&reuse_indices);
    let mut use_bc_dec = SimpleBits::new(&use_bc_bits);
    let mut dpcm_dec = dpcm.each_ref().map(|d| SimpleBits::new(d));
    let mut w = streams.readers(sign);

    let mut prev_states = [
        vec![0 as ZstdPrevState; bxs as usize],
        vec![0 as ZstdPrevState; bxs as usize],
    ];
    let mut part2_hash = [-1i32; PART_HASH_SIZE];
    let mut part3_hash = [-1i32; PART_HASH_SIZE];
    let mut tm_hash = [-1i32; TM_HASH_SIZE];
    let mut syms = DctSyms::default();
    let mut cur_run_len = 0u32;

    for by in 0..bys {
        for bx in 0..bxs {
            let (row, up_row) = ((by & 1) as usize, ((by + 1) & 1) as usize);
            let left = if bx > 0 {
                Some(prev_states[row][bx as usize - 1])
            } else {
                None
            };
            let upper = if by > 0 {
                Some(prev_states[up_row][bx as usize])
            } else {
                None
            };
            let diag = if bx > 0 && by > 0 {
                Some(prev_states[up_row][bx as usize - 1])
            } else {
                None
            };

            if cur_run_len > 0 {
                let p = *grid.prev(bx, by).ok_or(Error::InvalidData)?;
                grid.set(bx, by, p)?;
                prev_states[row][bx as usize] = left.or(upper).ok_or(Error::InvalidData)?;
                cur_run_len -= 1;
                continue;
            }

            let mode_byte = mode_dec.bits8();
            if mode_byte & 3 == ZMODE_RUN {
                cur_run_len = 1 + (mode_byte >> 2);
                if bx == 0 && by == 0 {
                    return Err(Error::InvalidData);
                }
                if cur_run_len > bxs - bx {
                    return Err(Error::InvalidData);
                }
                let p = *grid.prev(bx, by).ok_or(Error::InvalidData)?;
                grid.set(bx, by, p)?;
                prev_states[row][bx as usize] = left.or(upper).ok_or(Error::InvalidData)?;
                cur_run_len -= 1;
                continue;
            } else if mode_byte & 15 == ZMODE_SOLID {
                let prev = prev_solid_color(grid.prev(bx, by));
                let dr = solid_dec.bits8();
                let dg = solid_dec.bits8();
                let db = solid_dec.bits8();
                let da = if hdr.has_alpha { solid_dec.bits8() } else { 0 };
                let a = if hdr.has_alpha { prev[3] + da } else { 255 };
                grid.set(
                    bx,
                    by,
                    solid_block([prev[0] + dr, prev[1] + dg, prev[2] + db, a]),
                )?;
                prev_states[row][bx as usize] = -1;
                continue;
            }

            let mut blk = LogAstcBlock::default();
            let tm_index: u32;

            if mode_byte & 1 == 0 {
                // Raw
                let config_reuse_index = (mode_byte >> 1) & 3;
                let actual_cem;
                if config_reuse_index < 3 {
                    let state = [left, upper, diag][config_reuse_index as usize]
                        .ok_or(Error::InvalidData)?;
                    let cfg = grid.neighbor(bx, by, config_reuse_index)?;
                    if state < 0 {
                        return Err(Error::InvalidData);
                    }
                    tm_index = state as u32;
                    blk.partition_id = cfg.partition_id;
                    actual_cem = u32::from(cfg.color_endpoint_modes[0]);
                } else {
                    let idx = if mode_byte & MODE_BYTE_TM_HASH_HIT_FLAG != 0 {
                        tm_hash[raw_bits.get_bits(TM_HASH_BITS) as usize & (TM_HASH_SIZE - 1)]
                    } else {
                        let t = raw_bits.decode_truncated_binary(tms.modes.len() as u32);
                        tm_hash[fib_hash(t, TM_HASH_SIZE)] = t as i32;
                        t as i32
                    };
                    let tm = tms.get(u32::try_from(idx).map_err(|_| Error::InvalidData)?)?;
                    tm_index = idx as u32;
                    actual_cem = base_ofs_cem(
                        u32::from(tm.cem),
                        mode_byte & MODE_BYTE_IS_BASE_OFS_FLAG != 0,
                    );
                    if tm.num_parts > 1 {
                        let parts = u32::from(tm.num_parts);
                        let total = total_unique_patterns(grid.size_index, parts);
                        let hash = if parts == 2 {
                            &mut part2_hash
                        } else {
                            &mut part3_hash
                        };
                        let unique = if mode_byte & MODE_BYTE_PART_HASH_HIT != 0 {
                            hash[raw_bits.get_bits(PART_HASH_BITS) as usize & (PART_HASH_SIZE - 1)]
                        } else {
                            let u = raw_bits.decode_truncated_binary(total);
                            hash[fib_hash(u, PART_HASH_SIZE)] = u as i32;
                            u as i32
                        };
                        let unique = u32::try_from(unique).map_err(|_| Error::InvalidData)?;
                        if unique >= total {
                            return Err(Error::InvalidData);
                        }
                        blk.partition_id = part_seed(grid.size_index, parts, unique)?;
                    }
                }
                let tm = tms.get(tm_index)?;
                let parts = usize::from(tm.num_parts);
                let vals = ah::get_num_cem_values(actual_cem) as usize;
                apply_trial_mode(&mut blk, &tm, actual_cem);

                if mode_byte & MODE_BYTE_DPCM_ENDPOINTS_FLAG != 0 {
                    let levels = ah::get_ise_levels(u32::from(blk.endpoint_ise_range));
                    let reuse_index = reuse_dec.bits8();
                    let pred = endpoint_pred_block(&grid, bx, by, reuse_index)?;
                    let mut use_bc = [false; 4];
                    if cem_supports_bc(actual_cem) {
                        for u in use_bc.iter_mut().take(parts) {
                            *u = use_bc_dec.bits1() != 0;
                        }
                    }
                    let predicted = predict_endpoints(&pred, &blk, &use_bc)?;
                    let s = match levels {
                        0..=8 => 0,
                        9..=16 => 1,
                        17..=32 => 2,
                        33..=64 => 3,
                        65..=128 => 4,
                        _ => 5,
                    };
                    let d = &mut dpcm_dec[s];
                    apply_endpoint_dpcm(&mut blk, &predicted, parts, vals, || {
                        if s < 2 { d.bits4() } else { d.bits8() }
                    })?;
                } else {
                    let n = (parts * vals) as u32;
                    if n as usize > blk.endpoints.len()
                        || !decode_values(
                            &mut raw_bits,
                            n,
                            u32::from(blk.endpoint_ise_range),
                            &mut blk.endpoints,
                        )
                    {
                        return Err(Error::InvalidData);
                    }
                }
            } else if mode_byte & 15 >= ZMODE_REUSE_CFG_ENDPOINTS_LEFT {
                let reuse_index = ((mode_byte >> 2) & 3).wrapping_sub(1);
                let state = [left, upper, diag]
                    .get(reuse_index as usize)
                    .copied()
                    .flatten()
                    .ok_or(Error::InvalidData)?;
                let cfg = *grid.neighbor(bx, by, reuse_index)?;
                if state < 0 {
                    return Err(Error::InvalidData);
                }
                tm_index = state as u32;
                reuse_cfg_endpoints(&mut blk, &cfg);
            } else {
                return Err(Error::InvalidData);
            }
            prev_states[row][bx as usize] = tm_index as i32;

            // Weights
            let tm = tms.get(tm_index)?;
            let planes = if tm.ccs_index >= 0 { 2 } else { 1 };
            let total = u32::from(tm.grid_w) * u32::from(tm.grid_h);
            let wrange = u32::from(blk.weight_ise_range);
            if hdr.use_dct && mode_byte & MODE_BYTE_USE_DCT != 0 {
                let dc_levels = num_weight_dc_levels(wrange);
                for plane in 0..planes {
                    w.dct_syms(dc_levels, total, &mut syms)?;
                    if !decode_block_weights(grid.bw, grid.bh, hdr.dct_q, plane, &mut blk, &syms) {
                        return Err(Error::InvalidData);
                    }
                }
            } else {
                let levels = ah::get_ise_levels(wrange);
                let r2i = weight_rank_to_ise(wrange);
                for plane in 0..planes as usize {
                    w.dpcm_weights(
                        levels,
                        r2i,
                        &mut blk.weights,
                        plane,
                        planes as usize,
                        total as usize,
                    )?;
                }
            }
            grid.set(bx, by, blk)?;
        }
    }

    if raw_bits.get_bits(FINAL_SYNC_MARKER_BITS) != FINAL_SYNC_MARKER || !mode_dec.at_end() {
        return Err(Error::InvalidData);
    }
    Ok(grid.into_image(hdr.width, hdr.height, hdr.has_alpha, hdr.srgb))
}

// ---------------------------------------------------------------------------------------
// Arithmetic syntaxes (full and hybrid)

/// `prev_block_state`.
#[derive(Clone, Copy, Default)]
struct PrevState {
    used_weight_dct: bool,
    first_endpoint_uses_bc: bool,
    reused_full_cfg: bool,
    used_part_hash: bool,
    /// -1 for solid blocks.
    tm_index: i32,
    /// Not including base+offset.
    base_cem_index: u32,
    subset_index: u32,
    ccs_index: u32,
    grid_size: u32,
    grid_aniso: u32,
}

impl PrevState {
    /// The state of a run block: a copy of `p`, marked as reused.
    fn run(p: &PrevState) -> PrevState {
        PrevState {
            reused_full_cfg: true,
            ..*p
        }
    }

    /// The (cleared) state of a block reusing `p`'s configuration.
    fn cfg_reuse(p: &PrevState) -> PrevState {
        PrevState {
            tm_index: p.tm_index,
            base_cem_index: p.base_cem_index,
            subset_index: p.subset_index,
            ccs_index: p.ccs_index,
            grid_size: p.grid_size,
            grid_aniso: p.grid_aniso,
            used_part_hash: p.used_part_hash,
            reused_full_cfg: true,
            ..PrevState::default()
        }
    }
}

/// Model index from a flag of the left and upper neighbors (missing neighbors count as set).
fn ctx2(left: Option<&PrevState>, upper: Option<&PrevState>, f: fn(&PrevState) -> bool) -> usize {
    usize::from(left.is_none_or(f)) | (usize::from(upper.is_none_or(f)) << 1)
}

/// Adaptive models of the arithmetic syntaxes.
struct Models {
    mode: DataModel,
    solid_color_dpcm: [DataModel; 4],
    raw_endpoints: Vec<DataModel>,
    dpcm_endpoints: Vec<DataModel>,
    is_base_ofs: BitModel,
    use_dct: [BitModel; 4],
    use_dpcm_endpoints: BitModel,
    cem_index: [DataModel; 8],
    subset_index: [DataModel; 3],
    ccs_index: [DataModel; 5],
    grid_size: [DataModel; 2],
    grid_aniso: [DataModel; 3],
    dct_run_len: DataModel,
    dct_coeff_mag: DataModel,
    weight_mean: [DataModel; 2],
    raw_weights: Vec<DataModel>,
    submodes: Vec<DataModel>,
    endpoints_use_bc: [BitModel; 4],
    endpoint_reuse_delta: DataModel,
    config_reuse: [DataModel; 4],
    run_len: GammaContexts,
    use_part_hash: [BitModel; 4],
    part2_hash_index: DataModel,
    part3_hash_index: DataModel,
}

impl Models {
    fn new(fast: bool) -> Self {
        let n = |syms: u32| DataModel::new(syms, false);
        let endpoint_models = || {
            (0..17)
                .map(|i| n(ah::get_ise_levels(FIRST_VALID_ENDPOINT_ISE_RANGE + i)))
                .collect::<Vec<_>>()
        };
        let (dct_run_len, dct_coeff_mag, weight_mean, raw_weights) = if fast {
            (
                DataModel::default(),
                DataModel::default(),
                [DataModel::default(), DataModel::default()],
                Vec::new(),
            )
        } else {
            (
                n(65),
                n(255),
                [n(DCT_MEAN_LEVELS0), n(DCT_MEAN_LEVELS1)],
                (0..12)
                    .map(|i| n(ah::get_ise_levels(FIRST_VALID_WEIGHT_ISE_RANGE + i)))
                    .collect(),
            )
        };
        Self {
            mode: n(MODE_TOTAL),
            solid_color_dpcm: std::array::from_fn(|_| DataModel::new(256, true)),
            raw_endpoints: endpoint_models(),
            dpcm_endpoints: endpoint_models(),
            is_base_ofs: BitModel::default(),
            use_dct: [BitModel::default(); 4],
            use_dpcm_endpoints: BitModel::default(),
            cem_index: std::array::from_fn(|_| n(OTM_NUM_CEMS)),
            subset_index: std::array::from_fn(|_| n(OTM_NUM_SUBSETS)),
            ccs_index: std::array::from_fn(|_| n(OTM_NUM_CCS)),
            grid_size: std::array::from_fn(|_| n(OTM_NUM_GRID_SIZES)),
            grid_aniso: std::array::from_fn(|_| n(OTM_NUM_GRID_ANISOS)),
            dct_run_len,
            dct_coeff_mag,
            weight_mean,
            raw_weights,
            submodes: vec![DataModel::default(); NUM_GROUPS],
            endpoints_use_bc: [BitModel::default(); 4],
            endpoint_reuse_delta: n(REUSE_XY_DELTAS.len() as u32),
            config_reuse: std::array::from_fn(|_| n(4)),
            run_len: GammaContexts::default(),
            use_part_hash: [BitModel::default(); 4],
            part2_hash_index: DataModel::new(PART_HASH_SIZE as u32, true),
            part3_hash_index: DataModel::new(PART_HASH_SIZE as u32, true),
        }
    }
}

fn model(v: &mut [DataModel], i: u32) -> Result<&mut DataModel> {
    v.get_mut(i as usize).ok_or(Error::InvalidData)
}

fn decompress_arith(data: &[u8], expect: Option<Expected>) -> Result<XuastcImage> {
    let hybrid = data.first() == Some(&SYNTAX_HYBRID_ARITH_ZSTD);
    let mut sign: &[u8] = &[];
    let (arith_data, streams) = if hybrid {
        if data.len() < ARITH_HEADER_SIZE {
            return Err(Error::InvalidData);
        }
        let len = |i| header_len(data, i);
        let arith_len = len(0)?;
        if arith_len < arith::ARITH_MIN_EXPECTED_DATA_BUF_SIZE {
            return Err(Error::InvalidData);
        }
        let mut cur = &data[ARITH_HEADER_SIZE..];
        let arith_data = take(&mut cur, arith_len)?;
        let z =
            |cur: &mut &[u8], i: usize| -> Result<Bytes> { zstd_decompress(take(cur, len(i)?)?) };
        let mean0 = z(&mut cur, 1)?;
        let mean1 = z(&mut cur, 2)?;
        let run = z(&mut cur, 3)?;
        let coeff = z(&mut cur, 4)?;
        sign = take(&mut cur, len(5)?)?;
        let streams = WeightStreams {
            mean0,
            mean1,
            run,
            coeff,
            weight2: z(&mut cur, 6)?,
            weight3: z(&mut cur, 7)?,
            weight4: z(&mut cur, 8)?,
            weight8: z(&mut cur, 9)?,
        };
        (arith_data, Some(streams))
    } else {
        (&data[1..], None)
    };
    let mut fast = streams.as_ref().map(|s| s.readers(sign));

    let mut dec = ArithDec::new(arith_data)?;
    let hdr = read_header(
        |n, single| {
            if single {
                dec.get_bit()
            } else {
                dec.get_bits(n)
            }
        },
        (ARITH_HEADER_MARKER, ARITH_HEADER_MARKER_BITS),
    )?;
    if hdr.width < 1 || hdr.height < 1 {
        return Err(Error::InvalidData);
    }
    let mut grid = Grid::new(hdr.size_index, hdr.width, hdr.height, expect)?;
    let tms = mode_table(hdr.size_index).ok_or(Error::InvalidData)?;
    let (bxs, bys) = (grid.blocks_x, grid.blocks_y);
    let (bw, bh) = (grid.bw, grid.bh);

    let mut m = Models::new(fast.is_some());
    let mut prev_states = [
        vec![PrevState::default(); bxs as usize],
        vec![PrevState::default(); bxs as usize],
    ];
    let mut part2_hash = [-1i32; PART_HASH_SIZE];
    let mut part3_hash = [-1i32; PART_HASH_SIZE];
    let mut syms = DctSyms::default();
    let mut cur_run_len = 0u32;

    for by in 0..bys {
        for bx in 0..bxs {
            let (row, up_row) = ((by & 1) as usize, ((by + 1) & 1) as usize);
            let left = (bx > 0).then(|| prev_states[row][bx as usize - 1]);
            let upper = (by > 0).then(|| prev_states[up_row][bx as usize]);
            let diag = (bx > 0 && by > 0).then(|| prev_states[up_row][bx as usize - 1]);
            let (left, upper, diag) = (left.as_ref(), upper.as_ref(), diag.as_ref());
            let pred_state = left.or(upper);

            if cur_run_len > 0 {
                let p = *grid.prev(bx, by).ok_or(Error::InvalidData)?;
                grid.set(bx, by, p)?;
                prev_states[row][bx as usize] =
                    PrevState::run(pred_state.ok_or(Error::InvalidData)?);
                cur_run_len -= 1;
                continue;
            }

            let mut blk = LogAstcBlock::default();
            let mode_index = dec.decode_sym(&mut m.mode);
            match mode_index {
                MODE_SOLID => {
                    let prev = prev_solid_color(grid.prev(bx, by));
                    let r = prev[0] + dec.decode_sym(&mut m.solid_color_dpcm[0]);
                    let g = prev[1] + dec.decode_sym(&mut m.solid_color_dpcm[1]);
                    let b = prev[2] + dec.decode_sym(&mut m.solid_color_dpcm[2]);
                    let a = if hdr.has_alpha {
                        prev[3] + dec.decode_sym(&mut m.solid_color_dpcm[3])
                    } else {
                        255
                    };
                    grid.set(bx, by, solid_block([r, g, b, a]))?;
                    prev_states[row][bx as usize] = PrevState {
                        // Bias the statistics towards using DCT (most common case).
                        used_weight_dct: hdr.use_dct,
                        first_endpoint_uses_bc: true,
                        tm_index: -1,
                        base_cem_index: CEM_LDR_RGB_DIRECT,
                        reused_full_cfg: false,
                        used_part_hash: true,
                        ..PrevState::default()
                    };
                    continue;
                }
                MODE_RUN => {
                    if bx == 0 && by == 0 {
                        return Err(Error::InvalidData);
                    }
                    cur_run_len = dec.decode_gamma(&mut m.run_len);
                    if cur_run_len == 0 || cur_run_len > bxs - bx {
                        return Err(Error::InvalidData);
                    }
                    let p = *grid.prev(bx, by).ok_or(Error::InvalidData)?;
                    grid.set(bx, by, p)?;
                    prev_states[row][bx as usize] =
                        PrevState::run(pred_state.ok_or(Error::InvalidData)?);
                    cur_run_len -= 1;
                    continue;
                }
                MODE_RAW
                | MODE_REUSE_CFG_ENDPOINTS_LEFT
                | MODE_REUSE_CFG_ENDPOINTS_UP
                | MODE_REUSE_CFG_ENDPOINTS_DIAG => {}
                _ => return Err(Error::InvalidData),
            }

            let mut st = PrevState::default();
            let tm_index: u32;
            if mode_index != MODE_RAW {
                // Full configuration, partition id and endpoints reuse.
                let n = mode_index - MODE_REUSE_CFG_ENDPOINTS_LEFT;
                let cfg_state = *[left, upper, diag][n as usize].ok_or(Error::InvalidData)?;
                if cfg_state.tm_index < 0 {
                    return Err(Error::InvalidData);
                }
                let cfg = *grid.neighbor(bx, by, n)?;
                tm_index = cfg_state.tm_index as u32;
                let actual_cem = reuse_cfg_endpoints(&mut blk, &cfg);
                st = PrevState::cfg_reuse(&cfg_state);
                if cem_supports_bc(actual_cem) {
                    st.first_endpoint_uses_bc = used_blue_contraction(
                        actual_cem,
                        &blk.endpoints,
                        u32::from(blk.endpoint_ise_range),
                    );
                }
            } else {
                let reuse_ctx = ctx2(left, upper, |s| s.reused_full_cfg);
                let config_reuse_index = dec.decode_sym(&mut m.config_reuse[reuse_ctx]);
                let actual_cem;
                if config_reuse_index < MAX_CONFIG_REUSE_NEIGHBORS {
                    let cfg_state = *[left, upper, diag][config_reuse_index as usize]
                        .ok_or(Error::InvalidData)?;
                    if cfg_state.tm_index < 0 {
                        return Err(Error::InvalidData);
                    }
                    let cfg = grid.neighbor(bx, by, config_reuse_index)?;
                    tm_index = cfg_state.tm_index as u32;
                    blk.partition_id = cfg.partition_id;
                    actual_cem = u32::from(cfg.color_endpoint_modes[0]);
                    st = PrevState::cfg_reuse(&cfg_state);
                } else {
                    // Full ASTC configuration.
                    let p = pred_state.copied().unwrap_or(PrevState {
                        base_cem_index: CEM_LDR_RGB_DIRECT,
                        ..PrevState::default()
                    });
                    let cem_index =
                        dec.decode_sym(&mut m.cem_index[cem_to_ldrcem_index(p.base_cem_index)]);
                    let subset_index = dec.decode_sym(model(&mut m.subset_index, p.subset_index)?);
                    let ccs_index = dec.decode_sym(model(&mut m.ccs_index, p.ccs_index)?);
                    let grid_size = dec.decode_sym(model(&mut m.grid_size, p.grid_size)?);
                    let grid_aniso = dec.decode_sym(model(&mut m.grid_aniso, p.grid_aniso)?);
                    let modes =
                        tms.candidates(cem_index, subset_index, ccs_index, grid_size, grid_aniso);
                    let mut submode_index = 0;
                    if modes.len() > 1 {
                        let g =
                            group_index(cem_index, subset_index, ccs_index, grid_size, grid_aniso)
                                .ok_or(Error::InvalidData)?;
                        let sm = m.submodes.get_mut(g).ok_or(Error::InvalidData)?;
                        if sm.num_data_syms() == 0 {
                            sm.init(modes.len() as u32, true);
                        }
                        submode_index = dec.decode_sym(sm);
                    }
                    tm_index = u32::from(
                        *modes
                            .get(submode_index as usize)
                            .ok_or(Error::InvalidData)?,
                    );
                    st.tm_index = tm_index as i32;
                    st.base_cem_index = cem_index;
                    st.subset_index = subset_index;
                    st.ccs_index = ccs_index;
                    st.grid_size = grid_size;
                    st.grid_aniso = grid_aniso;
                    st.reused_full_cfg = false;

                    let tm = tms.get(tm_index)?;
                    let mut cem = u32::from(tm.cem);
                    if cem == CEM_LDR_RGB_DIRECT || cem == CEM_LDR_RGBA_DIRECT {
                        cem = base_ofs_cem(cem, dec.decode_bit(&mut m.is_base_ofs));
                    }
                    actual_cem = cem;

                    if tm.num_parts > 1 {
                        let parts = u32::from(tm.num_parts);
                        let total = total_unique_patterns(grid.size_index, parts);
                        let ctx = ctx2(left, upper, |s| s.used_part_hash);
                        let hash = if parts == 2 {
                            &mut part2_hash
                        } else {
                            &mut part3_hash
                        };
                        let unique;
                        if !dec.decode_bit(&mut m.use_part_hash[ctx]) {
                            let u = dec.decode_truncated_binary(total);
                            hash[fib_hash(u, PART_HASH_SIZE)] = u as i32;
                            unique = u;
                            st.used_part_hash = false;
                        } else {
                            let hm = if parts == 2 {
                                &mut m.part2_hash_index
                            } else {
                                &mut m.part3_hash_index
                            };
                            let h = dec.decode_sym(hm) as usize;
                            unique = u32::try_from(hash[h & (PART_HASH_SIZE - 1)])
                                .map_err(|_| Error::InvalidData)?;
                            st.used_part_hash = true;
                        }
                        if unique >= total {
                            return Err(Error::InvalidData);
                        }
                        blk.partition_id = part_seed(grid.size_index, parts, unique)?;
                    } else {
                        st.used_part_hash = true; // bias to true
                    }
                }

                let tm = tms.get(tm_index)?;
                let parts = usize::from(tm.num_parts);
                let vals = ah::get_num_cem_values(actual_cem) as usize;
                apply_trial_mode(&mut blk, &tm, actual_cem);
                let erange = u32::from(blk.endpoint_ise_range);
                let emodel = erange
                    .checked_sub(FIRST_VALID_ENDPOINT_ISE_RANGE)
                    .ok_or(Error::InvalidData)?;

                // Endpoints
                if !dec.decode_bit(&mut m.use_dpcm_endpoints) {
                    let rm = model(&mut m.raw_endpoints, emodel)?;
                    for i in 0..parts * vals {
                        *blk.endpoints.get_mut(i).ok_or(Error::InvalidData)? =
                            dec.decode_sym(rm) as u8;
                    }
                } else {
                    let reuse_delta = dec.decode_sym(&mut m.endpoint_reuse_delta);
                    let pred = endpoint_pred_block(&grid, bx, by, reuse_delta)?;
                    let bc_ctx = ctx2(left, upper, |s| s.first_endpoint_uses_bc);
                    let mut use_bc = [false; 4];
                    if cem_supports_bc(actual_cem) {
                        for u in use_bc.iter_mut().take(parts) {
                            *u = dec.decode_bit(&mut m.endpoints_use_bc[bc_ctx]);
                        }
                    }
                    let predicted = predict_endpoints(&pred, &blk, &use_bc)?;
                    let dm = model(&mut m.dpcm_endpoints, emodel)?;
                    apply_endpoint_dpcm(&mut blk, &predicted, parts, vals, || {
                        u32::from(dec.decode_sym(dm) as u8)
                    })?;
                }
                if cem_supports_bc(actual_cem) {
                    st.first_endpoint_uses_bc =
                        used_blue_contraction(actual_cem, &blk.endpoints, erange);
                }
            }

            // Weights
            let tm = tms.get(tm_index)?;
            let planes = if tm.ccs_index >= 0 { 2 } else { 1 };
            let total = u32::from(tm.grid_w) * u32::from(tm.grid_h);
            let wrange = u32::from(blk.weight_ise_range);
            let used_dct = hdr.use_dct && {
                let ctx = ctx2(left, upper, |s| s.used_weight_dct);
                dec.decode_bit(&mut m.use_dct[ctx])
            };
            if used_dct {
                st.used_weight_dct = true;
                let dc_levels = num_weight_dc_levels(wrange);
                for plane in 0..planes {
                    if let Some(f) = fast.as_mut() {
                        f.dct_syms(dc_levels, total, &mut syms)?;
                    } else {
                        arith_dct_syms(&mut dec, &mut m, dc_levels, total, &mut syms)?;
                    }
                    if !decode_block_weights(bw, bh, hdr.dct_q, plane, &mut blk, &syms) {
                        return Err(Error::InvalidData);
                    }
                }
            } else {
                let levels = ah::get_ise_levels(wrange);
                let r2i = weight_rank_to_ise(wrange);
                for plane in 0..planes as usize {
                    if let Some(f) = fast.as_mut() {
                        f.dpcm_weights(
                            levels,
                            r2i,
                            &mut blk.weights,
                            plane,
                            planes as usize,
                            total as usize,
                        )?;
                    } else {
                        let wm = model(&mut m.raw_weights, wrange - FIRST_VALID_WEIGHT_ISE_RANGE)?;
                        let mut prev_w = levels / 2;
                        for i in 0..total as usize {
                            let r = dec.decode_sym(wm);
                            let w = (prev_w + r) % levels.max(1);
                            prev_w = w;
                            set_weight(&mut blk.weights, plane + i * planes as usize, r2i, w)?;
                        }
                    }
                }
            }
            grid.set(bx, by, blk)?;
            prev_states[row][bx as usize] = st;
        }
    }

    if dec.get_bits(FINAL_SYNC_MARKER_BITS) != FINAL_SYNC_MARKER {
        return Err(Error::InvalidData);
    }
    Ok(grid.into_image(hdr.width, hdr.height, hdr.has_alpha, hdr.srgb))
}

/// DCT weight symbols of one plane from the arithmetic stream.
fn arith_dct_syms(
    dec: &mut ArithDec,
    m: &mut Models,
    num_dc_levels: u32,
    total_weights: u32,
    syms: &mut DctSyms,
) -> Result<()> {
    syms.num_coeffs = 0;
    syms.dc_sym =
        dec.decode_sym(&mut m.weight_mean[usize::from(num_dc_levels == DCT_MEAN_LEVELS1)]);
    let mut zig = 1u32;
    while zig < total_weights {
        let run_len = dec.decode_sym(&mut m.dct_run_len);
        if run_len == DCT_RUN_LEN_EOB_SYM_INDEX {
            break;
        }
        zig += run_len;
        if zig >= total_weights {
            return Err(Error::InvalidData);
        }
        let sign = dec.get_bit();
        let mut coeff = dec.decode_sym(&mut m.dct_coeff_mag) as i32 + 1;
        if sign != 0 {
            coeff = -coeff;
        }
        push_coeff(syms, run_len, coeff)?;
        zig += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::astc::helpers::{DecodePixels, decode_block, pack_astc_block};

    /// Truncated and bit flipped slices of every test file decode or fail, never panic.
    #[test]
    fn corrupt_streams_never_panic() -> Result<(), Box<dyn std::error::Error>> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/basis/");
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as usize
        };
        for name in [
            "xuastc4x4_rgba.basis",
            "xuastc5x4_dct.basis",
            "xuastc6x6_rgba_arith.basis",
            "xuastc4x4_hybrid_dct.basis",
            "xuastc6x5_dct_arith.basis",
            "xuastc12x12_pow2.basis",
        ] {
            let data = std::fs::read(format!("{dir}{name}"))?;
            let file = crate::basis::BasisFile::parse(&data)?;
            let c = file.slice(0, 0, false).ok_or("no slice")?;
            let ofs = c.file_ofs as usize;
            let slice = &data[ofs..ofs + c.file_size as usize];
            assert!(super::decompress(slice, None).is_ok());
            for it in 0..300 {
                let mut d = slice.to_vec();
                if it % 7 == 0 {
                    d.truncate(rnd() % d.len());
                } else {
                    for _ in 0..1 + rnd() % 4 {
                        let i = rnd() % d.len();
                        d[i] ^= 1 << (rnd() % 8);
                    }
                }
                let Ok(img) = super::decompress(&d, None) else {
                    continue;
                };
                let (bw, bh) = img.block.dim();
                let mut px = [[0u8; 4]; 144];
                for b in img.blocks.iter() {
                    let _ = pack_astc_block(&mut [0u8; 16], b);
                    let out = DecodePixels::Ldr8(&mut px[..(bw * bh) as usize]);
                    let _ = decode_block(b, &mut { out }, bw, bh);
                }
            }
        }
        Ok(())
    }
}
