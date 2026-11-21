// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
//! Tests of the XUASTC LDR bitstream writers: bit exactness against C++ dumps (`#[ignore]`d,
//! needs `$XU_DUMP_DIR`), and round trips of the logical blocks of the test files.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::{compare_log_blocks_for_equality, supercompress, supercompress_dims};
use crate::astc::helpers::{
    CEM_LDR_RGB_BASE_PLUS_OFFSET, CEM_LDR_RGB_DIRECT, CEM_LDR_RGBA_BASE_PLUS_OFFSET,
    CEM_LDR_RGBA_DIRECT, LogAstcBlock,
};
use crate::basis::file::BASIS_ASTC_ORDER;
use crate::basis::xuastc::enc::{
    AstcLdrEncodeConfig, BlockInfo, EncodeBlockOutput, EncodeImageOutput, XuastcSyntax, dump,
};
use crate::basis::xuastc::{decompress, mode_table};

/// The single slice payload of a `.basis` file (level 0).
fn basis_payload(data: &[u8]) -> Vec<u8> {
    let file = crate::basis::BasisFile::parse(data).unwrap();
    let c = file.slice(0, 0, false).unwrap();
    let ofs = c.file_ofs as usize;
    data[ofs..ofs + c.file_size as usize].to_vec()
}

fn config(
    bw: u32,
    bh: u32,
    syntax: XuastcSyntax,
    quality: u32,
    linear: bool,
    scd: bool,
) -> AstcLdrEncodeConfig {
    AstcLdrEncodeConfig {
        astc_block_width: bw,
        astc_block_height: bh,
        astc_decode_mode_srgb: !linear,
        compressed_syntax: syntax,
        effort_level: 3,
        dct_quality: if quality < 100 { quality as f32 } else { 85.0 },
        use_dct: quality < 100,
        lossy_supercompression: quality < 100,
        comp_weights: if linear { [1, 1, 1, 1] } else { [9, 11, 1, 11] },
        scd_enabled: scd,
        ..AstcLdrEncodeConfig::default()
    }
}

/// Compares coded blocks with the blocks the decoder produces from `payload`.
fn check_decodes(payload: &[u8], coded: &[LogAstcBlock], what: &str) {
    let img = decompress(payload, None).unwrap_or_else(|e| panic!("{what}: decode failed {e:?}"));
    assert_eq!(img.blocks.len(), coded.len(), "{what}");
    for (i, (a, b)) in img.blocks.iter().zip(coded).enumerate() {
        assert!(
            compare_log_blocks_for_equality(a, b),
            "{what}: block {i} differs\n{a:?}\n{b:?}"
        );
    }
}

#[test]
#[ignore = "needs XU_DUMP_DIR (C++ encoder dumps)"]
fn cpp_dump_matrix() {
    let Ok(dir) = std::env::var("XU_DUMP_DIR") else {
        return;
    };
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            let n = p.file_name()?.to_str()?.to_string();
            n.strip_suffix(".cfg").map(str::to_string)
        })
        .collect();
    names.sort();
    let (mut ok, mut bad) = (0, Vec::new());
    let (mut syntaxes, mut replaced, mut dct) = ([0; 3], 0, 0);
    for name in &names {
        let c = std::fs::read_to_string(format!("{dir}/{name}.cfg")).unwrap();
        let f: Vec<&str> = c.split_whitespace().collect();
        let (bw, bh): (u32, u32) = (f[0].parse().unwrap(), f[1].parse().unwrap());
        let syntax = match f[2] {
            "arith" => XuastcSyntax::FullArith,
            "hybrid" => XuastcSyntax::HybridArithZstd,
            _ => XuastcSyntax::FullZstd,
        };
        let q: u32 = f[3].parse().unwrap();
        let linear = f[4] == "1";
        let deblock = f[5] == "1";
        let mut cfg = config(bw, bh, syntax, q, linear, deblock && bw * bh >= 80);
        cfg.effort_level = f.get(6).map_or(3, |e| e.parse().unwrap());
        let d = dump::load(&format!("{name}.final.xud")).unwrap();
        let expected = basis_payload(&std::fs::read(format!("{dir}/{name}.basis")).unwrap());
        let out = supercompress(&d.image, &cfg, &d.out).unwrap();
        check_decodes(&expected, &out.coded_blocks, name);
        syntaxes[usize::from(expected[0].min(2))] += 1;
        dct += usize::from(cfg.use_dct);
        for (info, c) in d.out.block_info.iter().zip(&out.coded_blocks) {
            let chosen = &info.out_blocks[info.packed_out_block_index as usize].log_blk;
            replaced += usize::from(!compare_log_blocks_for_equality(chosen, c));
        }
        if out.comp_data == expected {
            ok += 1;
        } else {
            let first = out
                .comp_data
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b);
            bad.push(format!(
                "{name}: len {} vs {}, first diff {first:?}",
                out.comp_data.len(),
                expected.len()
            ));
        }
    }
    eprintln!(
        "bit exact: {ok} / {} (arith/hybrid/zstd payloads {syntaxes:?}, {dct} with DCT, \
         {replaced} lossy replacements)",
        names.len()
    );
    assert!(bad.is_empty(), "mismatches:\n{}", bad.join("\n"));
}

/// The trial mode of a decoded (non solid) block.
fn find_tm(size_index: usize, b: &LogAstcBlock) -> i16 {
    let t = mode_table(size_index).unwrap();
    let mut cem = u32::from(b.color_endpoint_modes[0]);
    if cem == CEM_LDR_RGB_BASE_PLUS_OFFSET {
        cem = CEM_LDR_RGB_DIRECT;
    } else if cem == CEM_LDR_RGBA_BASE_PLUS_OFFSET {
        cem = CEM_LDR_RGBA_DIRECT;
    }
    let ccs = if b.dual_plane {
        b.color_component_selector as i8
    } else {
        -1
    };
    let found: Vec<usize> = t
        .modes
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m.grid_w == b.grid_width
                && m.grid_h == b.grid_height
                && u32::from(m.cem) == cem
                && m.ccs_index == ccs
                && m.endpoint_range == b.endpoint_ise_range
                && m.weight_range == b.weight_ise_range
                && m.num_parts == b.num_partitions
        })
        .map(|(i, _)| i)
        .collect();
    assert_eq!(found.len(), 1, "trial mode lookup");
    found[0] as i16
}

/// One candidate per block (the decoded blocks), like an encoder output without alternatives.
fn enc_out_from_blocks(
    size_index: usize,
    bxs: u32,
    bys: u32,
    blocks: &[LogAstcBlock],
) -> EncodeImageOutput {
    let mut out = EncodeImageOutput::new(bxs, bys);
    for (info, b) in out.block_info.iter_mut().zip(blocks) {
        let trial_mode_index = if b.solid_color_flag_ldr {
            -1
        } else {
            find_tm(size_index, b)
        };
        *info = BlockInfo {
            out_blocks: vec![EncodeBlockOutput {
                trial_mode_index,
                log_blk: *b,
                ..EncodeBlockOutput::default()
            }],
            ..BlockInfo::default()
        };
    }
    out
}

/// Re-encodes the decoded blocks of the non-DCT test files: the bitstream of the original syntax
/// is fully determined by the blocks and must equal the file's payload; every other syntax must
/// round trip through the decoder.
#[test]
fn reencode_test_files() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/basis/");
    for name in [
        "xuastc4x4_rgba.basis",
        "xuastc6x6_rgba_arith.basis",
        "xuastc8x6_rgb_hybrid.basis",
        "xuastc12x12_pow2.basis",
        "xuastc6x6_mosaic_rgba.basis",
        "xuastc6x6_mosaic_rgb.basis",
        "xuastc8x6_mosaic_rgba.basis",
        "xuastc8x6_mosaic_rgb.basis",
    ] {
        let payload = basis_payload(&std::fs::read(format!("{dir}{name}")).unwrap());
        let img = decompress(&payload, None).unwrap();
        let size_index = BASIS_ASTC_ORDER
            .iter()
            .position(|b| *b == img.block)
            .unwrap();
        let (bw, bh) = img.block.dim();
        let blocks: Vec<LogAstcBlock> = img.blocks.iter().copied().collect();
        let enc_out = enc_out_from_blocks(size_index, img.blocks_x, img.blocks_y, &blocks);
        let orig_syntax = match payload[0] {
            0 => XuastcSyntax::FullArith,
            1 => XuastcSyntax::HybridArithZstd,
            _ => XuastcSyntax::FullZstd,
        };
        for syntax in [
            XuastcSyntax::FullArith,
            XuastcSyntax::HybridArithZstd,
            XuastcSyntax::FullZstd,
        ] {
            let cfg = AstcLdrEncodeConfig {
                astc_block_width: bw,
                astc_block_height: bh,
                astc_decode_mode_srgb: img.srgb,
                compressed_syntax: syntax,
                ..AstcLdrEncodeConfig::default()
            };
            let out =
                supercompress_dims(img.width, img.height, img.has_alpha, &cfg, &enc_out).unwrap();
            let what = format!("{name} {syntax:?}");
            check_decodes(&out.comp_data, &out.coded_blocks, &what);
            for (a, b) in out.coded_blocks.iter().zip(&blocks) {
                assert!(compare_log_blocks_for_equality(a, b), "{what}");
            }
            if syntax == orig_syntax {
                assert!(out.comp_data == payload, "{what}: not bit exact");
            }
        }
    }
}

/// Lossy replacement picks a neighbor's configuration within the PSNR threshold.
#[test]
fn lossy_replacement_reuses_neighbor_config() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/basis/");
    let payload =
        basis_payload(&std::fs::read(format!("{dir}xuastc6x6_rgba_arith.basis")).unwrap());
    let img = decompress(&payload, None).unwrap();
    let blocks: Vec<LogAstcBlock> = img.blocks.iter().copied().collect();
    let mut enc_out = enc_out_from_blocks(4, img.blocks_x, img.blocks_y, &blocks);
    // Give block 1 an alternative identical to block 0 (when block 0 isn't solid) with slightly
    // worse error; the chosen candidate is block 1 itself, at high PSNR.
    if blocks[0].solid_color_flag_ldr || blocks[1].solid_color_flag_ldr {
        return;
    }
    let alt = enc_out.block_info[0].out_blocks[0].clone();
    let info = &mut enc_out.block_info[1];
    info.out_blocks[0].sse = 10;
    info.out_blocks.push(EncodeBlockOutput { sse: 11, ..alt });
    let cfg = AstcLdrEncodeConfig {
        astc_block_width: 6,
        astc_block_height: 6,
        lossy_supercompression: true,
        comp_weights: [9, 11, 1, 11],
        ..AstcLdrEncodeConfig::default()
    };
    let out = supercompress_dims(img.width, img.height, img.has_alpha, &cfg, &enc_out).unwrap();
    check_decodes(&out.comp_data, &out.coded_blocks, "lossy");
    assert!(compare_log_blocks_for_equality(
        &out.coded_blocks[1],
        &blocks[0]
    ));
}
