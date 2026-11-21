// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of `ldr_astc_block_encode_image_astcf` and `find_tm_index`
//! (`basisu_astc_ldr_encode.cpp`): encodes every block of an image with the fast `astc_ldrf`
//! encoder ([`super::fencode`]), keeping all of its candidates (optionally DCT coded) with
//! their trial mode index and actual weighted SSE.
//!
//! Rows are encoded independently on up to `threads` threads; the output does not depend on
//! the thread count.
#![allow(clippy::too_many_arguments)]

use super::common::CemEncodeParams;
use super::dct_enc::{GridWeightDct, code_block_weights};
use super::fencode::{
    SubsetEncContext, SubsetEncThreadContext, compress_block_subsets, compress_single_subset,
    convert_rank_lblock_to_ise, decode_block_xuastc_ldr, find_astc_block_size_index,
    init_multi_subset_context, init_single_subset_context, is_block_xuastc_ldr,
    set_ldr_solid_block,
};
use super::partitions::PartitionsData;
use super::{AstcLdrEncodeConfig, BLUR_ID_ASTCF, BlockInfo, EncodeBlockOutput, EncodeImageOutput};
use crate::astc::helpers::{self as ah, LogAstcBlock, MAX_BLOCK_PIXELS};
use crate::basis::ll::comp::Image;
use crate::basis::ll::enc::std_sort_by;
use crate::basis::ll::job_pool::map_jobs;
use crate::basis::xuastc::{grid_aniso, mode_table};
use crate::{Error, Result};
use std::sync::Arc;

/// `weighted_color_error`.
#[inline]
pub(crate) fn weighted_color_error(a: &[u8; 4], b: &[u8; 4], p: &CemEncodeParams) -> u32 {
    let d = |c: usize| {
        let e = i32::from(a[c]) - i32::from(b[c]);
        ((e * e) as u32).wrapping_mul(p.comp_weights[c])
    };
    d(0).wrapping_add(d(1))
        .wrapping_add(d(2))
        .wrapping_add(d(3))
}

/// `calc_grid_size_val`.
#[inline]
fn calc_grid_size_val(gw: u32, gh: u32, bw: u32, bh: u32) -> u32 {
    u32::from(gw >= bw - 1 && gh >= bh - 1)
}

/// `find_tm_index`: `Ok(-1)` for solid blocks, the encoder trial mode index otherwise, `Err`
/// if the block matches no trial mode (C++ returns -2).
pub(crate) fn find_tm_index(
    block_width: u32,
    block_height: u32,
    block_size_index: u32,
    log_blk: &LogAstcBlock,
) -> Result<i32> {
    if log_blk.solid_color_flag_ldr {
        return Ok(-1);
    }
    let mut cem = u32::from(log_blk.color_endpoint_modes[0]);
    if cem == ah::CEM_LDR_RGB_BASE_PLUS_OFFSET {
        cem = ah::CEM_LDR_RGB_DIRECT;
    } else if cem == ah::CEM_LDR_RGBA_BASE_PLUS_OFFSET {
        cem = ah::CEM_LDR_RGBA_DIRECT;
    }
    let table = mode_table(block_size_index as usize).ok_or(Error::InvalidArgument)?;
    let (gw, gh) = (
        u32::from(log_blk.grid_width),
        u32::from(log_blk.grid_height),
    );
    let ccs = if log_blk.dual_plane {
        u32::from(log_blk.color_component_selector) + 1
    } else {
        0
    };
    let tms = table.candidates(
        cem,
        u32::from(log_blk.num_partitions).wrapping_sub(1),
        ccs,
        calc_grid_size_val(gw, gh, block_width, block_height),
        grid_aniso(gw, gh, block_width, block_height),
    );
    for &tm_index in tms {
        let tm = table.get(u32::from(tm_index))?;
        if tm.endpoint_range == log_blk.endpoint_ise_range
            && tm.weight_range == log_blk.weight_ise_range
            && tm.grid_w == log_blk.grid_width
            && tm.grid_h == log_blk.grid_height
        {
            return Ok(i32::from(tm_index));
        }
    }
    Err(Error::InvalidData)
}

/// The `compress_image` per-encoder candidate limit (`max_candidate_limit`).
pub(crate) fn max_candidate_limit(cfg: &AstcLdrEncodeConfig, total_blocks: u32) -> u32 {
    if !cfg.scd_enabled {
        return 16;
    }
    let merge = cfg.merge_basisu_into_output;
    if total_blocks >= 512 * 512 {
        if merge { 8 } else { 16 }
    } else if total_blocks >= 256 * 256 {
        if merge { 16 } else { 32 }
    } else if merge {
        32
    } else {
        64
    }
}

/// `enc_cfg.m_base_q` from the DCT quality (`compress_image`).
pub(crate) fn base_q_from_dct_quality(dct_quality: f32) -> f32 {
    let int_q = ((dct_quality * 2.0).round() as i32).clamp(0, 200);
    int_q as f32 / 2.0
}

/// Shared tail of the image block encoders: DCT codes (optionally), finds the trial mode and
/// measures the actual weighted SSE of one candidate (already in ISE space).
pub(crate) fn finish_candidate(
    out: &mut EncodeBlockOutput,
    block_pixels: &[[u8; 4]],
    block_width: u32,
    block_height: u32,
    block_size_index: u32,
    grid_coder: Option<&GridWeightDct>,
    base_q: f32,
    cem_params: &CemEncodeParams,
) -> Result<()> {
    if !is_block_xuastc_ldr(&out.log_blk) {
        return Err(Error::InvalidData);
    }
    let mut tm_index = find_tm_index(block_width, block_height, block_size_index, &out.log_blk)?;
    let total_block_pixels = (block_width * block_height) as usize;

    if let (true, Some(gc)) = (tm_index >= 0, grid_coder) {
        let log_blk = &mut out.log_blk;
        let num_planes = if log_blk.dual_plane { 2 } else { 1 };
        let total_grid = usize::from(log_blk.grid_width) * usize::from(log_blk.grid_height);
        let mut total_empty_planes = 0;
        for plane in 0..num_planes {
            let syms = &mut out.packed_dct_plane_data[plane];
            if !code_block_weights(gc, base_q, plane as u32, log_blk, syms) {
                return Err(Error::InvalidData);
            }
            for i in 0..total_grid {
                log_blk.weights[i * num_planes + plane] = 0;
            }
            if !gc.decode_block_weights(base_q, plane as u32, log_blk, syms) {
                return Err(Error::InvalidData);
            }
            if syms.coeffs.len() == 1 && 1 + usize::from(syms.coeffs[0].num_zeros) == total_grid {
                total_empty_planes += 1;
            }
        }
        if log_blk.num_partitions == 1 && total_empty_planes == num_planes {
            // DC only after quantization: switch to void-extent.
            let mut sum = [0u32; 4];
            for p in &block_pixels[..total_block_pixels] {
                for c in 0..4 {
                    sum[c] += u32::from(p[c]);
                }
            }
            let n = total_block_pixels as u32;
            let round = n >> 1;
            let s = sum.map(|v| (v + round) / n);
            set_ldr_solid_block(log_blk, s[0], s[1], s[2], s[3]);
            tm_index = -1;
        }
    }

    out.trial_mode_index = i16::try_from(tm_index).map_err(|_| Error::InvalidData)?;

    let mut dec = [[0u8; 4]; MAX_BLOCK_PIXELS];
    if !decode_block_xuastc_ldr(
        &out.log_blk,
        &mut dec,
        block_width,
        block_height,
        cem_params.decode_mode_srgb,
    ) {
        return Err(Error::InvalidData);
    }
    let mut total_err = 0u64;
    for i in 0..total_block_pixels {
        total_err += u64::from(weighted_color_error(&dec[i], &block_pixels[i], cem_params));
    }
    out.sse = total_err;
    Ok(())
}

/// Keeps the `max_candidate_limit` lowest SSE candidates of a block (libstdc++ `std::sort`
/// order) and returns the new best index (0), or `best` unchanged.
pub(crate) fn shrink_candidates(
    blk_info: &mut BlockInfo,
    best: u32,
    max_candidate_limit: u32,
) -> u32 {
    if blk_info.out_blocks.len() <= max_candidate_limit as usize {
        return best;
    }
    let mut sorted: Vec<u32> = (0..blk_info.out_blocks.len() as u32).collect();
    let ob = &blk_info.out_blocks;
    std_sort_by(&mut sorted, |&a, &b| {
        ob[a as usize].sse < ob[b as usize].sse
    });
    let shrunk: Vec<EncodeBlockOutput> = sorted
        .iter()
        .take(max_candidate_limit as usize)
        .map(|&i| ob[i as usize].clone())
        .collect();
    blk_info.out_blocks = shrunk;
    0
}

/// Per image settings of the astcf encoder derived from the effort level.
struct AstcfSettings {
    num_candidates: u32,
    use_subsets: bool,
    max_subsets: u32,
    num_subset_carriers: u32,
    num_subset_pats: u32,
    dot_thresh_fract_index_2subsets: u32,
    weight_polishing: bool,
}

fn lerp(a: f32, b: f32, c: f32) -> f32 {
    a + (b - a) * c
}

fn round_clamp(v: f32, lo: i32, hi: i32) -> u32 {
    (v.round() as i32).clamp(lo, hi) as u32
}

fn astcf_settings(cfg: &AstcLdrEncodeConfig, bw: u32, bh: u32, has_alpha: bool) -> AstcfSettings {
    let effort = cfg.effort_level;
    let fe = effort as f32 * (1.0 / 10.0);
    let num_candidates = if has_alpha {
        round_clamp(lerp(16.0, 48.0, fe), 1, 64)
    } else {
        round_clamp(lerp(1.0, 48.0, fe), 1, 64)
    };
    let (mut num_subset_carriers, mut num_subset_pats) = (1, 1);
    let mut dot_thresh_fract_index_2subsets = 0;
    let (mut use_subsets, mut max_subsets) = (false, 1);
    if effort != 0 {
        if bw * bh <= 25 {
            num_subset_carriers = round_clamp(lerp(1.0, 2.0, fe), 1, 3);
            num_subset_pats = round_clamp(lerp(1.0, 6.0, fe), 1, 16);
        } else {
            num_subset_carriers = round_clamp(lerp(1.0, 3.0, fe), 1, 3);
            num_subset_pats = round_clamp(lerp(1.0, 16.0, fe), 1, 16);
        }
        dot_thresh_fract_index_2subsets = match effort {
            8 => 1,
            9 => 2,
            10 => 3,
            _ => 0,
        };
        use_subsets = num_subset_carriers > 0 && num_subset_pats > 0;
        if use_subsets {
            max_subsets = if effort > 1 { 3 } else { 2 };
        }
    }
    if cfg.force_disable_subsets {
        use_subsets = false;
    }
    AstcfSettings {
        num_candidates,
        use_subsets,
        max_subsets,
        num_subset_carriers,
        num_subset_pats,
        dot_thresh_fract_index_2subsets,
        weight_polishing: effort >= 2,
    }
}

/// `ldr_astc_block_encode_image_astcf`.
///
/// `cfg` supplies the effort level and the force-disable flags (`global_cfg`); `use_dct` /
/// `base_q` / `cem_params` / `max_candidate_limit` are the `ldr_astc_block_encode_image_*`
/// high level config values `compress_image` derives (see [`base_q_from_dct_quality`],
/// [`max_candidate_limit`]). `part_data` = the 2 and 3 subset `partitions_data` (built with
/// `PartitionsData::new(n, bw, bh, true)`).
pub(crate) fn encode_image_astcf(
    img: &Image,
    block_width: u32,
    block_height: u32,
    cfg: &AstcLdrEncodeConfig,
    use_dct: bool,
    base_q: f32,
    cem_params: &CemEncodeParams,
    max_candidate_limit: u32,
    part_data: (Arc<PartitionsData>, Arc<PartitionsData>),
    threads: u32,
) -> Result<EncodeImageOutput> {
    let block_size_index =
        find_astc_block_size_index(block_width, block_height).ok_or(Error::InvalidArgument)?;
    let num_blocks_x = img.get_block_width(block_width);
    let num_blocks_y = img.get_block_height(block_height);
    let has_alpha = img.has_alpha(3);
    let st = astcf_settings(cfg, block_width, block_height, has_alpha);

    let ss = init_single_subset_context(
        block_width,
        block_height,
        cem_params.decode_mode_srgb,
        &cem_params.comp_weights,
        st.num_candidates,
        2,
        cfg.force_disable_rgb_dual_plane,
        has_alpha,
        st.weight_polishing,
    )
    .ok_or(Error::InvalidArgument)?;
    let mut ctx = SubsetEncContext::new(ss);
    if cfg.effort_level <= 2 {
        // Note: init_multi_subset_context() below re-enables method 2 (as in C++).
        ctx.use_method2 = false;
    }
    if cfg.effort_level <= 6 {
        ctx.ss.higher_effort_bc = false;
        ctx.ss.try_base_ofs = false;
    }
    if st.use_subsets
        && !init_multi_subset_context(
            &mut ctx,
            st.max_subsets,
            st.num_subset_carriers,
            st.num_subset_pats,
            6.0 * 6.0,
            st.dot_thresh_fract_index_2subsets,
            6.0 * 6.0,
            part_data.0,
            part_data.1,
        )
    {
        return Err(Error::InvalidArgument);
    }

    let grid_coder = use_dct.then(|| GridWeightDct::new(block_width, block_height));
    let total_block_pixels = (block_width * block_height) as usize;

    let rows = map_jobs(threads, num_blocks_y as usize, |by| -> Result<_> {
        let by = by as u32;
        let mut thread_ctx = SubsetEncThreadContext::default();
        let mut all_candidates: Vec<LogAstcBlock> = Vec::with_capacity(64);
        let mut row = Vec::with_capacity(num_blocks_x as usize);
        for bx in 0..num_blocks_x {
            let mut block_pixels = [[0u8; 4]; MAX_BLOCK_PIXELS];
            img.extract_block_clamped(
                &mut block_pixels,
                bx * block_width,
                by * block_height,
                block_width,
                block_height,
            );
            all_candidates.clear();
            let mut best = LogAstcBlock::default();
            if st.use_subsets {
                compress_block_subsets(
                    &ctx,
                    &mut thread_ctx,
                    &block_pixels[..total_block_pixels],
                    &mut best,
                    Some(&mut all_candidates),
                );
            } else {
                compress_single_subset(
                    &ctx.ss,
                    &block_pixels[..total_block_pixels],
                    &mut best,
                    Some(&mut all_candidates),
                    false,
                );
            }
            if all_candidates.is_empty() {
                return Err(Error::InvalidData);
            }

            let mut info = BlockInfo::default();
            let (mut best_err, mut best_index) = (u64::MAX, 0u32);
            for (ci, cand) in all_candidates.iter().enumerate() {
                let mut out = EncodeBlockOutput {
                    blur_id: BLUR_ID_ASTCF,
                    log_blk: *cand,
                    ..Default::default()
                };
                convert_rank_lblock_to_ise(&mut out.log_blk);
                finish_candidate(
                    &mut out,
                    &block_pixels,
                    block_width,
                    block_height,
                    block_size_index,
                    grid_coder.as_ref(),
                    base_q,
                    cem_params,
                )?;
                if out.sse < best_err {
                    best_err = out.sse;
                    best_index = ci as u32;
                }
                info.out_blocks.push(out);
            }
            best_index = shrink_candidates(&mut info, best_index, max_candidate_limit);
            info.packed_out_block_index = best_index;
            let mut phys = [0u8; 16];
            let best_blk = &info
                .out_blocks
                .get(best_index as usize)
                .ok_or(Error::InvalidData)?
                .log_blk;
            if !ah::pack_astc_block(&mut phys, best_blk) {
                return Err(Error::InvalidData);
            }
            row.push((info, phys));
        }
        Ok(row)
    })
    .ok_or(Error::InvalidData)?;

    let mut out = EncodeImageOutput::new(num_blocks_x, num_blocks_y);
    for (by, row) in rows.into_iter().enumerate() {
        for (bx, (info, phys)) in row?.into_iter().enumerate() {
            let i = bx + by * num_blocks_x as usize;
            out.block_info[i] = info;
            out.packed_phys_blocks[i] = phys;
        }
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use crate::basis::xuastc::enc::dump;

    /// One line of `astcf_manifest.txt` (`name bw bh effort srgb quality scd merge
    /// disable_subsets disable_rgb_dp`, see `tools/basisu-golden/astcf_dumps.sh`). Effort 0
    /// 4x4 cases run the fast 4x4 encoder.
    pub(crate) struct Case {
        pub name: String,
        pub bw: u32,
        pub bh: u32,
        pub effort: u32,
        pub srgb: bool,
        pub quality: u32,
        pub scd: bool,
        pub merge: bool,
        pub disable_subsets: bool,
        pub disable_rgb_dp: bool,
    }

    pub(crate) fn parse_case(line: &str) -> Option<Case> {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 8 {
            return None;
        }
        let n = |i: usize| f[i].parse::<u32>().ok();
        Some(Case {
            name: f[0].to_string(),
            bw: n(1)?,
            bh: n(2)?,
            effort: n(3)?,
            srgb: n(4)? != 0,
            quality: n(5)?,
            scd: n(6)? != 0,
            merge: n(7)? != 0,
            disable_subsets: n(8).unwrap_or(0) != 0,
            disable_rgb_dp: n(9).unwrap_or(0) != 0,
        })
    }

    /// Runs the astcf encoder with the C++ CLI configuration of `case`.
    pub(crate) fn run_case(case: &Case, img: &Image, threads: u32) -> EncodeImageOutput {
        let cfg = AstcLdrEncodeConfig {
            astc_block_width: case.bw,
            astc_block_height: case.bh,
            astc_decode_mode_srgb: case.srgb,
            effort_level: case.effort,
            dct_quality: case.quality.max(1) as f32,
            use_dct: case.quality != 0,
            comp_weights: if case.srgb {
                [9, 11, 1, 11]
            } else {
                [1, 1, 1, 1]
            },
            scd_enabled: case.scd,
            merge_basisu_into_output: case.merge,
            use_astcf: true,
            force_disable_subsets: case.disable_subsets,
            force_disable_rgb_dual_plane: case.disable_rgb_dp,
            ..Default::default()
        };
        let cem_params = CemEncodeParams {
            comp_weights: cfg.comp_weights,
            decode_mode_srgb: cfg.astc_decode_mode_srgb,
            ..Default::default()
        };
        let base_q = base_q_from_dct_quality(cfg.dct_quality);
        if case.effort == 0 && case.bw == 4 && case.bh == 4 {
            return super::super::fast4x4::encode_image_fast_4x4(
                img,
                &cfg,
                cfg.use_dct,
                base_q,
                &cem_params,
                threads,
            )
            .expect("fast 4x4 encode");
        }
        let bx = img.get_block_width(case.bw);
        let by = img.get_block_height(case.bh);
        let limit = max_candidate_limit(&cfg, bx * by);
        let pd = (
            Arc::new(PartitionsData::new(2, case.bw, case.bh, true)),
            Arc::new(PartitionsData::new(3, case.bw, case.bh, true)),
        );
        encode_image_astcf(
            img,
            case.bw,
            case.bh,
            &cfg,
            cfg.use_dct,
            base_q,
            &cem_params,
            limit,
            pd,
            threads,
        )
        .expect("astcf encode")
    }

    fn cand_eq(a: &EncodeBlockOutput, b: &EncodeBlockOutput) -> bool {
        a.trial_mode_index == b.trial_mode_index
            && a.blur_id == b.blur_id
            && a.sse == b.sse
            && dump::log_blk_eq(&a.log_blk, &b.log_blk)
            && a.packed_dct_plane_data == b.packed_dct_plane_data
    }

    pub(crate) fn block_eq(a: &BlockInfo, b: &BlockInfo) -> bool {
        a.packed_out_block_index == b.packed_out_block_index
            && a.out_blocks.len() == b.out_blocks.len()
            && a.out_blocks
                .iter()
                .zip(&b.out_blocks)
                .all(|(x, y)| cand_eq(x, y))
            && a.low_freq_block_flag == b.low_freq_block_flag
            && a.super_strong_edges == b.super_strong_edges
            && a.very_strong_edges == b.very_strong_edges
            && a.strong_edges == b.strong_edges
    }

    /// Compares every block's candidate list and chosen index with the C++ astcf output
    /// (`NAME.first.xud`) for each case of `$XU_DUMP_DIR/astcf_manifest.txt`.
    #[test]
    #[ignore = "needs XU_DUMP_DIR (C++ encoder dumps)"]
    fn astcf_matches_cpp_dumps() {
        let Ok(dir) = std::env::var("XU_DUMP_DIR") else {
            return;
        };
        let manifest = std::fs::read_to_string(format!("{dir}/astcf_manifest.txt")).unwrap();
        let filter = std::env::var("XU_CASE").unwrap_or_default();
        let (mut total_ok, mut total_cases) = (0, 0);
        let mut stats = [0usize; 8];
        for case in manifest.lines().filter_map(parse_case) {
            if !case.name.contains(&filter) {
                continue;
            }
            let Some(d) = dump::load(&format!("{}.first.xud", case.name)) else {
                println!("{}: missing dump", case.name);
                continue;
            };
            let out = run_case(&case, &d.image, 0);
            assert_eq!(
                out.block_info.len(),
                d.out.block_info.len(),
                "{}",
                case.name
            );
            let mut ok = 0;
            let mut first_bad = None;
            for (i, (a, b)) in out.block_info.iter().zip(&d.out.block_info).enumerate() {
                if block_eq(a, b) {
                    ok += 1;
                } else if first_bad.is_none() {
                    first_bad = Some(i);
                }
            }
            println!(
                "{}: {}/{} blocks match",
                case.name,
                ok,
                out.block_info.len()
            );
            if let Some(i) = first_bad {
                let (a, b) = (&out.block_info[i], &d.out.block_info[i]);
                println!(
                    "  first mismatch block {i}: rust {} cands (best {}), c++ {} cands (best {})",
                    a.out_blocks.len(),
                    a.packed_out_block_index,
                    b.out_blocks.len(),
                    b.packed_out_block_index
                );
                for (j, (x, y)) in a.out_blocks.iter().zip(&b.out_blocks).enumerate() {
                    if !cand_eq(x, y) {
                        println!("  cand {j}:\n    rust {x:?}\n    c++  {y:?}");
                        break;
                    }
                }
            }
            for b in &d.out.block_info {
                for c in &b.out_blocks {
                    let l = &c.log_blk;
                    stats[0] += 1;
                    stats[1] += usize::from(l.num_partitions == 2);
                    stats[2] += usize::from(l.num_partitions == 3);
                    stats[3] += usize::from(l.dual_plane);
                    stats[4] += usize::from(matches!(l.color_endpoint_modes[0], 9 | 13));
                    stats[5] += usize::from(!c.packed_dct_plane_data[0].coeffs.is_empty());
                    stats[6] += usize::from(l.solid_color_flag_ldr);
                }
                stats[7] += usize::from(b.out_blocks.len() >= 16);
            }
            total_cases += 1;
            if ok == out.block_info.len() {
                total_ok += 1;
            }
        }
        println!(
            "candidates: {} (2 subsets {}, 3 subsets {}, dual plane {}, base+ofs {}, dct {}, \
             solid {}), blocks with >= 16 candidates: {}",
            stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7]
        );
        println!("{total_ok}/{total_cases} cases fully match");
        assert_eq!(total_ok, total_cases);
    }

    /// Cases of the non-ignored golden test (`testdata/basisu_enc/basisu_astcf.txt`): dump
    /// names from `tools/basisu-golden/astcf_dumps.sh` whose images have raw copies in
    /// `testdata/images/raw`.
    const GOLDEN_CASES: [&str; 9] = [
        "a_rgb_37x29_6x6_e3",
        "a_mosaic_rgba_50x38_4x4_e1",
        "a_rgba_64x48_10x8_e2",
        "l_rgba_64x48_8x8_e2",
        "q_rgba_64x48_5x5_e1",
        "x_rgba_64x48_6x6_e3",
        "d_rgba_64x48_12x12_e3",
        "f_mosaic_rgba_50x38_4x4_e0",
        "fq_rgba_64x48_4x4_e0",
    ];

    /// FNV-1a 64 over every block's candidate list.
    pub(crate) fn hash_output(out: &EncodeImageOutput) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut put = |b: &[u8]| {
            for &x in b {
                h = (h ^ u64::from(x)).wrapping_mul(0x100_0000_01b3);
            }
        };
        put(&out.blocks_x.to_le_bytes());
        put(&out.blocks_y.to_le_bytes());
        for b in &out.block_info {
            put(&b.packed_out_block_index.to_le_bytes());
            put(&[
                u8::from(b.low_freq_block_flag),
                u8::from(b.super_strong_edges),
                u8::from(b.very_strong_edges),
                u8::from(b.strong_edges),
            ]);
            put(&(b.out_blocks.len() as u32).to_le_bytes());
            for c in &b.out_blocks {
                put(&c.trial_mode_index.to_le_bytes());
                put(&c.blur_id.to_le_bytes());
                put(&c.sse.to_le_bytes());
                let l = &c.log_blk;
                put(&[
                    u8::from(l.error_flag),
                    u8::from(l.solid_color_flag_ldr),
                    u8::from(l.solid_color_flag_hdr),
                    l.user_mode,
                    l.grid_width,
                    l.grid_height,
                    u8::from(l.dual_plane),
                    l.weight_ise_range,
                    l.endpoint_ise_range,
                    l.color_component_selector,
                    l.num_partitions,
                ]);
                put(&l.partition_id.to_le_bytes());
                put(&l.color_endpoint_modes);
                put(&l.weights);
                put(&l.endpoints);
                for p in &c.packed_dct_plane_data {
                    for v in [
                        p.dc_sym,
                        p.num_dc_levels,
                        p.max_coeff_mag,
                        p.max_zigzag_index,
                    ] {
                        put(&v.to_le_bytes());
                    }
                    put(&(p.coeffs.len() as u32).to_le_bytes());
                    for k in &p.coeffs {
                        put(&k.num_zeros.to_le_bytes());
                        put(&k.coeff.to_le_bytes());
                    }
                }
            }
        }
        h
    }

    fn raw_image(name: &str) -> Option<Image> {
        // `<name>_<w>x<h>`
        let dims = name.rsplit('_').next()?;
        let (w, h) = dims.split_once('x')?;
        let (w, h) = (w.parse::<u32>().ok()?, h.parse::<u32>().ok()?);
        let path = format!(
            "{}/testdata/images/raw/{name}.rgba",
            env!("CARGO_MANIFEST_DIR")
        );
        let data = std::fs::read(path).ok()?;
        let mut img = Image::with_size(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = ((x + y * w) * 4) as usize;
                *img.get_mut(x, y) = [data[i], data[i + 1], data[i + 2], data[i + 3]];
            }
        }
        Some(img)
    }

    /// `name` = `<set>_<image>_<bw>x<bh>_e<effort>`: the image name.
    fn case_image_name(name: &str) -> String {
        let parts: Vec<&str> = name.split('_').collect();
        parts[1..parts.len() - 2].join("_")
    }

    /// Prints the golden lines for `GOLDEN_CASES` from the C++ dumps (and checks that the
    /// dumped input image is the raw test image). Regenerate the golden file with
    /// `tools/basisu-golden/astcf_dumps.sh` (`GOLDEN=1`).
    #[test]
    #[ignore = "needs XU_DUMP_DIR (C++ encoder dumps)"]
    fn astcf_golden_from_dumps() {
        let Ok(dir) = std::env::var("XU_DUMP_DIR") else {
            return;
        };
        let manifest = std::fs::read_to_string(format!("{dir}/astcf_manifest.txt")).unwrap();
        let mut lines = Vec::new();
        for case in manifest.lines().filter_map(parse_case) {
            if !GOLDEN_CASES.contains(&case.name.as_str()) {
                continue;
            }
            let d = dump::load(&format!("{}.first.xud", case.name)).unwrap();
            let img = raw_image(&case_image_name(&case.name)).unwrap();
            assert_eq!(img.get_pixels(), d.image.get_pixels(), "{}", case.name);
            let line = manifest
                .lines()
                .find(|l| l.split_whitespace().next() == Some(case.name.as_str()))
                .unwrap();
            lines.push(format!("{line}\t{:016x}", hash_output(&d.out)));
        }
        let out = std::env::var("XU_GOLDEN_OUT").unwrap_or_default();
        if out.is_empty() {
            println!("{}", lines.join("\n"));
        } else {
            std::fs::write(out, lines.join("\n") + "\n").unwrap();
        }
    }

    /// The astcf (and fast 4x4) encoder output must hash to the C++ encoder's output
    /// (`testdata/basisu_enc/basisu_astcf.txt`, generated from basisu v2.50 dumps).
    #[test]
    fn astcf_matches_cpp_golden() {
        let golden = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/basisu_enc/basisu_astcf.txt"
        ));
        let mut n = 0;
        for line in golden.lines() {
            let Some((cfg, want)) = line.split_once('\t') else {
                continue;
            };
            let case = parse_case(cfg).unwrap();
            let img = raw_image(&case_image_name(&case.name)).unwrap();
            // Thread count must not matter: alternate between 1 and 4 threads.
            let out = run_case(&case, &img, if n % 2 == 0 { 1 } else { 4 });
            assert_eq!(format!("{:016x}", hash_output(&out)), want, "{}", case.name);
            n += 1;
        }
        assert_eq!(n, GOLDEN_CASES.len());
    }

    /// Throughput check: `cargo test --release --lib astcf_speed -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn astcf_speed() {
        let src = raw_image("rgba_64x48").unwrap();
        let (w, h) = (512u32, 512u32);
        let mut img = Image::with_size(w, h);
        for y in 0..h {
            for x in 0..w {
                let p = src.get(x * 64 / w, y * 48 / h);
                let n = ((x * 7 + y * 13) % 5) as u8;
                *img.get_mut(x, y) = [p[0].saturating_add(n), p[1], p[2].saturating_sub(n), p[3]];
            }
        }
        let mut opaque = img.clone();
        opaque.set_alpha(255);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get() as u32);
        for (label, im, bw, bh, e) in [
            ("rgb 6x6 e3", &opaque, 6, 6, 3),
            ("rgb 4x4 e1", &opaque, 4, 4, 1),
            ("rgba 6x6 e3", &img, 6, 6, 3),
            ("rgb 8x8 e2 q60", &opaque, 8, 8, 2),
        ] {
            let q = if label.contains("q60") { 60 } else { 0 };
            let case = Case {
                name: String::new(),
                bw,
                bh,
                effort: e,
                srgb: true,
                quality: q,
                scd: false,
                merge: false,
                disable_subsets: false,
                disable_rgb_dp: false,
            };
            for t in [1, threads] {
                let t0 = std::time::Instant::now();
                let out = run_case(&case, im, t);
                let secs = t0.elapsed().as_secs_f64();
                println!(
                    "{label}: {t} threads: {:.3} s, {:.2} MPix/s ({} blocks)",
                    secs,
                    (w * h) as f64 / secs / 1e6,
                    out.block_info.len()
                );
            }
        }
    }
}
