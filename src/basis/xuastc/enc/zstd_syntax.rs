// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! XUASTC LDR full Zstd syntax writer: port of `compress_image_full_zstd`, `zstd_compress` and
//! `encode_values` (`basisu_astc_ldr_encode.cpp`).

use super::CompressOutput;
use super::supercomp::{
    Ctx, WeightStreams, compare_log_blocks_for_equality, endpoint_dpcm_syms, endpoints_use_bc,
    is_base_ofs, len_u32, part_seed_to_unique, tm_is_direct,
};
use crate::astc::helpers::{self as ah, ISE_RANGE_TABLE, LogAstcBlock};
use crate::basis::ll::huffman_enc::BitwiseCoder;
use crate::basis::xuastc::{
    FINAL_SYNC_MARKER, FINAL_SYNC_MARKER_BITS, FULL_ZSTD_HEADER_MARKER,
    FULL_ZSTD_HEADER_MARKER_BITS, MAX_CONFIG_REUSE_NEIGHBORS, MODE_BYTE_DPCM_ENDPOINTS_FLAG,
    MODE_BYTE_IS_BASE_OFS_FLAG, MODE_BYTE_PART_HASH_HIT, MODE_BYTE_TM_HASH_HIT_FLAG,
    MODE_BYTE_USE_DCT, PART_HASH_BITS, PART_HASH_SIZE, SYNTAX_FULL_ZSTD, TM_HASH_BITS,
    TM_HASH_SIZE, ZMODE_REUSE_CFG_ENDPOINTS_LEFT, ZMODE_RUN, ZMODE_SOLID, fib_hash,
};
use crate::{Error, Result};

/// `xuastc_zstd_mode::cMODE_RAW`.
const ZMODE_RAW: u32 = 0;
/// `FULL_ZSTD_MAX_RUN_LEN`.
const FULL_ZSTD_MAX_RUN_LEN: u32 = 64;

/// `zstd_compress`: one frame (`ZSTD_compress` at `level`); empty input gives empty output.
pub(super) fn zstd_compress(data: &[u8], level: i32) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let bound = zstd_pure_rs::prelude::ZSTD_compressBound(data.len());
    let mut buf = Vec::new();
    buf.try_reserve_exact(bound)
        .map_err(|_| Error::OutOfMemory)?;
    buf.resize(bound, 0);
    let n = zstd_pure_rs::prelude::ZSTD_compress(&mut buf, data, level);
    if zstd_pure_rs::prelude::ZSTD_isError(n) || n > bound || n > u32::MAX as usize {
        return Err(Error::InvalidArgument);
    }
    buf.truncate(n);
    Ok(buf)
}

/// `encode_values`: BISE-style packing of `vals` (trit/quint groups first, then the bits).
/// Returns the number of bits written.
pub(super) fn encode_values(coder: &mut BitwiseCoder, vals: &[u8], endpoint_range: u32) -> u32 {
    let r = ISE_RANGE_TABLE[endpoint_range as usize];
    let (ep_bits, ep_trits, ep_quints) = (u32::from(r[0]), r[1] != 0, r[2] != 0);
    let mut tq_values = [0u32; 22];
    let (mut total_tq_values, mut tq_accum, mut tq_mul) = (0usize, 0u32, 1u32);
    for &v in vals {
        let tq = u32::from(v) >> ep_bits;
        if ep_trits {
            tq_accum += tq * tq_mul;
            tq_mul *= 3;
            if tq_mul == 243 {
                tq_values[total_tq_values] = tq_accum;
                total_tq_values += 1;
                tq_accum = 0;
                tq_mul = 1;
            }
        } else if ep_quints {
            tq_accum += tq * tq_mul;
            tq_mul *= 5;
            if tq_mul == 125 {
                tq_values[total_tq_values] = tq_accum;
                total_tq_values += 1;
                tq_accum = 0;
                tq_mul = 1;
            }
        }
    }
    let mut total_bits = 0;
    for &t in &tq_values[..total_tq_values] {
        let num_bits = if ep_trits { 8 } else { 7 };
        coder.put_bits(t, num_bits);
        total_bits += num_bits;
    }
    if tq_mul > 1 {
        let num_bits = if ep_trits {
            match tq_mul {
                3 => 2,
                9 => 4,
                27 => 5,
                _ => 7,
            }
        } else if tq_mul == 5 {
            3
        } else {
            5
        };
        coder.put_bits(tq_accum, num_bits);
        total_bits += num_bits;
    }
    for &v in vals {
        coder.put_bits(u32::from(v) & ((1 << ep_bits) - 1), ep_bits);
        total_bits += ep_bits;
    }
    total_bits
}

fn push_run(mode_bytes: &mut Vec<u8>, run_len: u32) {
    mode_bytes.push((ZMODE_RUN | ((run_len - 1) << 2)) as u8);
}

/// `compress_image_full_zstd`.
pub(super) fn compress_image_full_zstd(ctx: &Ctx) -> Result<CompressOutput> {
    let (bxs, bys) = (ctx.bxs, ctx.bys);
    let mut mode_bytes: Vec<u8> = Vec::new();
    let mut raw_bits = BitwiseCoder::default();
    let mut solid_dpcm_bytes: Vec<u8> = Vec::new();
    let mut reuse_indices: Vec<u8> = Vec::new();
    let mut use_bc_bits = BitwiseCoder::default();
    let mut dpcm_3bit = BitwiseCoder::default();
    let mut dpcm_4bit = BitwiseCoder::default();
    let mut dpcm_bytes: [Vec<u8>; 4] = Default::default(); // 5, 6, 7, 8 bit
    let mut ws = WeightStreams::default();

    let mut states = vec![0i32; (bxs * bys) as usize];
    let mut coded = vec![LogAstcBlock::default(); (bxs * bys) as usize];
    let mut part2_hash = [-1i32; PART_HASH_SIZE];
    let mut part3_hash = [-1i32; PART_HASH_SIZE];
    let mut tm_hash = [-1i32; TM_HASH_SIZE];
    let mut cur_run_len = 0u32;

    raw_bits.put_bits(FULL_ZSTD_HEADER_MARKER, FULL_ZSTD_HEADER_MARKER_BITS);
    raw_bits.put_bits(ctx.size_index as u32, 4);
    raw_bits.put_bits(u32::from(ctx.srgb), 1);
    raw_bits.put_bits(ctx.width, 16);
    raw_bits.put_bits(ctx.height, 16);
    raw_bits.put_bits(u32::from(ctx.has_alpha), 1);
    raw_bits.put_bits(u32::from(ctx.use_dct), 1);
    if ctx.use_dct {
        raw_bits.put_bits(ctx.int_q, 8);
    }

    for by in 0..bys {
        for bx in 0..bxs {
            let idx = (bx + by * bxs) as usize;
            let left = (bx > 0).then(|| states[idx - 1]);
            let upper = (by > 0).then(|| states[idx - bxs as usize]);
            let diag = (bx > 0 && by > 0).then(|| states[idx - bxs as usize - 1]);
            let blk_info = ctx.info(bx, by)?;
            let packed = blk_info.packed_out_block_index as usize;

            // Runs
            if bx > 0 || by > 0 {
                let cur = &blk_info
                    .out_blocks
                    .get(packed)
                    .ok_or(Error::InvalidArgument)?
                    .log_blk;
                let prev = *ctx.prev(&coded, bx, by).ok_or(Error::InvalidArgument)?;
                if compare_log_blocks_for_equality(cur, &prev) {
                    cur_run_len += 1;
                    coded[idx] = prev;
                    states[idx] = left.or(upper).ok_or(Error::InvalidArgument)?;
                    if cur_run_len == FULL_ZSTD_MAX_RUN_LEN {
                        push_run(&mut mode_bytes, cur_run_len);
                        cur_run_len = 0;
                    }
                    continue;
                }
            }
            if cur_run_len > 0 {
                push_run(&mut mode_bytes, cur_run_len);
                cur_run_len = 0;
            }

            let best = ctx.select_candidate(blk_info, &coded, bx, by)?;
            let blk_out = blk_info
                .out_blocks
                .get(best)
                .ok_or(Error::InvalidArgument)?;
            let cur = blk_out.log_blk;
            coded[idx] = cur;

            if blk_out.trial_mode_index < 0 {
                if !cur.solid_color_flag_ldr {
                    return Err(Error::InvalidArgument);
                }
                mode_bytes.push(ZMODE_SOLID as u8);
                let (d, n) = ctx.solid_deltas(&coded, bx, by, &cur);
                solid_dpcm_bytes.extend(d.iter().take(n).map(|&v| v as u8));
                states[idx] = -1;
                continue;
            }
            if cur.solid_color_flag_ldr {
                return Err(Error::InvalidArgument);
            }
            let tmi = i32::from(blk_out.trial_mode_index);
            let tm = ctx.tm(blk_out)?;
            let num_parts = usize::from(tm.num_parts);
            let cur_cem = u32::from(cur.color_endpoint_modes[0]);

            if let Some(ri) = ctx.full_cfg_endpoint_reuse_index(&coded, bx, by, &cur) {
                mode_bytes.push((ZMODE_REUSE_CFG_ENDPOINTS_LEFT + (ri << 2)) as u8);
                [left, upper, diag][ri as usize].ok_or(Error::InvalidArgument)?;
                states[idx] = tmi;
            } else {
                // Mode byte: UD C ED HH BO I I M (see the C++ source).
                let mut mode = ZMODE_RAW;
                let neighbor_cfg_match = (0..MAX_CONFIG_REUSE_NEIGHBORS).find(|&i| {
                    let Some(ns) = [left, upper, diag][i as usize] else {
                        return false;
                    };
                    let Some(n) = ctx.neighbor(&coded, bx, by, i) else {
                        return false;
                    };
                    ns == tmi
                        && n.color_endpoint_modes[0] == cur.color_endpoint_modes[0]
                        && n.partition_id == cur.partition_id
                });
                if let Some(ni) = neighbor_cfg_match {
                    mode |= ni << 1;
                    states[idx] = tmi;
                } else {
                    mode |= MAX_CONFIG_REUSE_NEIGHBORS << 1;
                    let h = fib_hash(tmi as u32, TM_HASH_SIZE);
                    if tm_hash[h] == tmi {
                        mode |= MODE_BYTE_TM_HASH_HIT_FLAG;
                        raw_bits.put_bits(h as u32, TM_HASH_BITS);
                    } else {
                        raw_bits.put_truncated_binary(tmi as u32, ctx.tms.modes.len() as u32);
                        tm_hash[h] = tmi;
                    }
                    states[idx] = tmi;
                    if tm_is_direct(&tm) && is_base_ofs(&cur) {
                        mode |= MODE_BYTE_IS_BASE_OFS_FLAG;
                    }
                    if tm.num_parts > 1 {
                        let parts = u32::from(tm.num_parts);
                        let (unique, total) =
                            part_seed_to_unique(ctx.size_index, parts, cur.partition_id)?;
                        let hash = if parts == 2 {
                            &mut part2_hash
                        } else {
                            &mut part3_hash
                        };
                        let h = fib_hash(unique, PART_HASH_SIZE);
                        if hash[h] != unique as i32 {
                            raw_bits.put_truncated_binary(unique, total);
                        } else {
                            mode |= MODE_BYTE_PART_HASH_HIT;
                            raw_bits.put_bits(h as u32, PART_HASH_BITS);
                        }
                        hash[h] = unique as i32;
                    }
                }

                // Endpoints
                let use_bc = endpoints_use_bc(&cur, num_parts);
                let erange = u32::from(cur.endpoint_ise_range);
                let levels = ah::get_ise_levels(erange);
                if let Some((reuse_index, predicted)) =
                    ctx.find_endpoint_pred(&coded, bx, by, &cur, num_parts, &use_bc, None)
                {
                    mode |= MODE_BYTE_DPCM_ENDPOINTS_FLAG;
                    reuse_indices.push(reuse_index as u8);
                    if crate::basis::xuastc::endpoints::cem_supports_bc(cur_cem) {
                        for &u in use_bc.iter().take(num_parts) {
                            use_bc_bits.put_bits(u32::from(u), 1);
                        }
                    }
                    endpoint_dpcm_syms(&cur, &predicted, num_parts, |e| {
                        if levels <= 8 {
                            dpcm_3bit.put_bits(e, 4);
                        } else if levels <= 16 {
                            dpcm_4bit.put_bits(e, 4);
                        } else if levels <= 32 {
                            dpcm_bytes[0].push(e as u8);
                        } else if levels <= 64 {
                            dpcm_bytes[1].push(e as u8);
                        } else if levels <= 128 {
                            dpcm_bytes[2].push(e as u8);
                        } else if levels <= 256 {
                            dpcm_bytes[3].push(e as u8);
                        }
                    });
                } else {
                    let n = num_parts * ah::get_num_cem_values(cur_cem) as usize;
                    encode_values(
                        &mut raw_bits,
                        cur.endpoints.get(..n).ok_or(Error::InvalidArgument)?,
                        erange,
                    );
                }
                mode_bytes.push(mode as u8);
            }

            // Weights
            let use_dct = WeightStreams::use_dct(ctx.use_dct, &cur, blk_out)?;
            if use_dct {
                if let Some(b) = mode_bytes.last_mut() {
                    *b |= MODE_BYTE_USE_DCT as u8;
                }
                let planes = if cur.dual_plane { 2 } else { 1 };
                for syms in blk_out.packed_dct_plane_data.iter().take(planes) {
                    ws.put_dct(syms);
                }
            } else {
                ws.put_dpcm(&cur);
            }
        }
        if cur_run_len > 0 {
            push_run(&mut mode_bytes, cur_run_len);
            cur_run_len = 0;
        }
    }

    raw_bits.put_bits(FINAL_SYNC_MARKER, FINAL_SYNC_MARKER_BITS);
    raw_bits.flush();
    dpcm_3bit.flush();
    dpcm_4bit.flush();
    use_bc_bits.flush();
    ws.flush();

    let level = if ctx.effort_level >= 3 { 19 } else { 9 };
    let z = |d: &[u8]| zstd_compress(d, level);
    let parts = [
        raw_bits.get_bytes().clone(),
        z(&mode_bytes)?,
        z(&solid_dpcm_bytes)?,
        z(&reuse_indices)?,
        z(use_bc_bits.get_bytes())?,
        z(dpcm_3bit.get_bytes())?,
        z(dpcm_4bit.get_bytes())?,
        z(&dpcm_bytes[0])?,
        z(&dpcm_bytes[1])?,
        z(&dpcm_bytes[2])?,
        z(&dpcm_bytes[3])?,
        z(ws.mean0_bits.get_bytes())?,
        z(&ws.mean1_bytes)?,
        z(&ws.run_bytes)?,
        z(&ws.coeff_bytes)?,
        ws.sign_bits.get_bytes().clone(),
        z(ws.weight2_bits.get_bytes())?,
        z(ws.weight3_bits.get_bytes())?,
        z(ws.weight4_bits.get_bytes())?,
        z(&ws.weight8_bits)?,
    ];
    let mut comp_data = vec![SYNTAX_FULL_ZSTD];
    for p in &parts {
        comp_data.extend_from_slice(&len_u32(p.len())?.to_le_bytes());
    }
    comp_data.extend_from_slice(&0u32.to_le_bytes()); // m_unused
    for p in &parts {
        comp_data.extend_from_slice(p);
    }
    len_u32(comp_data.len())?;
    Ok(CompressOutput {
        comp_data,
        coded_blocks: coded,
    })
}
