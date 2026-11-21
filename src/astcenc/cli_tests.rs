// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
//! astcenc API output vs. the C++ astc-encoder 6.0 CLI (`astcenc-avx2`) for what
//! `imgio::encode` does not expose: alpha scaling (`-a <radius>`, `a_scale_radius` plus
//! `ASTCENC_FLG_USE_ALPHA_WEIGHT`) and 3D blocks (`-zdim` slice stacks). The references in
//! `testdata/astcenc/cli/` come from `scripts/gen_astcenc_testdata.py`; every case runs with
//! 1 and 16 threads on every instruction set copy the CPU has.
#![allow(clippy::panic)]

use super::*;
use crate::astc_file::AstcFile;

const RAW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/images/raw/");
const CLI: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/astcenc/cli/");

/// The CLI `-cl` / `-cs` / `-cH` profile.
fn profile(mode: &str) -> AstcencProfile {
    match mode {
        "l" => AstcencProfile::Ldr,
        "s" => AstcencProfile::LdrSrgb,
        _ => AstcencProfile::Hdr,
    }
}

/// The CLI preset.
fn preset(name: &str) -> f32 {
    match name {
        "fastest" => ASTCENC_PRE_FASTEST,
        "fast" => ASTCENC_PRE_FAST,
        "medium" => ASTCENC_PRE_MEDIUM,
        "thorough" => ASTCENC_PRE_THOROUGH,
        _ => ASTCENC_PRE_EXHAUSTIVE,
    }
}

/// Compresses `img` as the CLI would with `mode`, `quality`, block `b`, `-a radius` (0: off)
/// and `-rdo-lambda lambda` (0: off) and compares with the CLI reference `name`.
fn check(
    name: &str,
    img: &AstcencImage,
    b: [u32; 3],
    mode: &str,
    quality: &str,
    radius: u32,
    lambda: f32,
) {
    let mut flags = 0;
    if radius != 0 {
        flags |= ASTCENC_FLG_USE_ALPHA_WEIGHT;
    }
    if lambda > 0.0 {
        flags |= ASTCENC_FLG_USE_LZ_RDO;
    }
    let Ok(mut config) =
        astcenc_config_init(profile(mode), b[0], b[1], b[2], preset(quality), flags)
    else {
        panic!("{name}: config_init")
    };
    config.a_scale_radius = radius;
    config.lz_rdo_lambda = lambda;

    let Ok(file) = std::fs::read(format!("{CLI}{name}.astc")) else {
        panic!("{name}: missing reference")
    };
    let Ok(reference) = AstcFile::parse(&file) else {
        panic!("{name}: bad reference")
    };
    assert_eq!(
        (
            reference.block(),
            reference.width(),
            reference.height(),
            reference.depth()
        ),
        ((b[0], b[1], b[2]), img.dim_x, img.dim_y, img.dim_z),
        "{name}: header"
    );

    for isa in [AstcencIsa::Baseline, AstcencIsa::Sse41, AstcencIsa::Avx2] {
        if isa > AstcencIsa::detect() {
            continue;
        }
        for threads in [1u32, 16] {
            let threads = if cfg!(target_family = "wasm") {
                1
            } else {
                threads
            };
            let Ok(ctx) = astcenc_context_alloc_isa(&config, threads, isa) else {
                panic!("{name}: context_alloc")
            };
            let mut out = vec![0u8; reference.data().len()];
            let shared = std::sync::Mutex::new(out.as_mut_slice());
            if threads == 1 {
                let r = astcenc_compress_image(&ctx, img, &AstcencSwizzle::RGBA, &shared, 0);
                assert_eq!(r, Ok(()), "{name}: compress");
            } else {
                std::thread::scope(|sc| {
                    let hs: Vec<_> = (0..threads)
                        .map(|t| {
                            let (ctx, img, shared) = (&ctx, img, &shared);
                            sc.spawn(move || {
                                astcenc_compress_image(ctx, img, &AstcencSwizzle::RGBA, shared, t)
                            })
                        })
                        .collect();
                    for h in hs {
                        match h.join() {
                            Ok(Ok(())) => {}
                            other => panic!("{name}: compress: {other:?}"),
                        }
                    }
                });
            }
            let blocks = out.len() / 16;
            let diff = (0..blocks)
                .filter(|&i| out[i * 16..i * 16 + 16] != reference.data()[i * 16..i * 16 + 16])
                .count();
            assert_eq!(
                diff, 0,
                "{name} ({isa:?}, {threads} threads): {diff} of {blocks} blocks differ from astcenc 6.0"
            );
        }
    }
}

/// `cutout_rgba_96x64`.
fn cutout() -> (Vec<u8>, u32, u32) {
    let Ok(data) = std::fs::read(format!("{RAW}cutout_rgba_96x64.rgba")) else {
        panic!("cutout_rgba_96x64.rgba")
    };
    (data, 96, 64)
}

/// `(block, mode, preset, radius, lambda)` from a reference name
/// `<source>_<w>x<h>[x<d>]_<block>_<mode>_<preset>[_a<radius>][_rdo<lambda>]`.
fn parse_name(name: &str) -> ([u32; 3], &str, &str, u32, f32) {
    let parts: Vec<&str> = name.split('_').collect();
    let dims: Vec<u32> = parts[2].split('x').filter_map(|v| v.parse().ok()).collect();
    let block = [dims[0], dims[1], dims.get(2).copied().unwrap_or(1)];
    let (mut radius, mut lambda) = (0, 0.0);
    for p in &parts[5..] {
        if let Some(r) = p.strip_prefix("rdo") {
            lambda = r.parse().unwrap_or(0.0);
        } else if let Some(r) = p.strip_prefix('a') {
            radius = r.parse().unwrap_or(0);
        }
    }
    (block, parts[3], parts[4], radius, lambda)
}

#[test]
fn alpha_scale_matches_astcenc_cli() {
    let (data, w, h) = cutout();
    let u8_img = AstcencImage {
        dim_x: w,
        dim_y: h,
        dim_z: 1,
        data: AstcencImageData::U8(vec![data.clone()]),
    };
    // The CLI loads the -cH source (a DDS) as FP16 data
    let f16_img = AstcencImage {
        dim_x: w,
        dim_y: h,
        dim_z: 1,
        data: AstcencImageData::F16(vec![
            data.iter()
                .map(|&c| crate::half::from_f32(f32::from(c) / 255.0))
                .collect(),
        ]),
    };
    // Keep in sync with ASCALE_CASES of the generator script
    let cases = [
        "cutout_96x64_4x4_l_fast_a1",
        "cutout_96x64_6x6_s_medium_a1",
        "cutout_96x64_8x8_l_thorough_a1",
        "cutout_96x64_4x4_s_medium_a2",
        "cutout_96x64_6x6_l_thorough_a2",
        "cutout_96x64_8x8_s_fast_a2",
        "cutout_96x64_4x4_l_thorough_a4",
        "cutout_96x64_6x6_s_fast_a4",
        "cutout_96x64_8x8_l_medium_a4",
        "cutout_96x64_4x4_s_fast_a8",
        "cutout_96x64_6x6_l_medium_a8",
        "cutout_96x64_8x8_s_thorough_a8",
        "cutout_96x64_4x4_l_medium_a1_rdo20",
        "cutout_96x64_6x6_s_fast_a2_rdo20",
        "cutout_96x64_8x8_l_thorough_a4_rdo20",
        "cutout_96x64_4x4_s_medium_a8_rdo20",
        "cutout_96x64_8x8_l_fast_a8_rdo20",
        "cutout_96x64_6x6_H_medium_a2",
    ];
    for name in cases {
        let (block, mode, quality, radius, lambda) = parse_name(name);
        let img = if mode == "H" { &f16_img } else { &u8_img };
        check(name, img, block, mode, quality, radius, lambda);
    }
}

/// The 26x22x14 volume of the generator's `volume`: voxel (x, y, z) is cutout pixel
/// ((x + 7 * z) % 96, (y + 5 * z) % 64).
fn volume() -> AstcencImage {
    let (data, w, h) = cutout();
    let (dx, dy, dz) = (26usize, 22usize, 14usize);
    let slices = (0..dz)
        .map(|z| {
            let mut s = Vec::with_capacity(dx * dy * 4);
            for y in 0..dy {
                for x in 0..dx {
                    let sx = (x + 7 * z) % w as usize;
                    let sy = (y + 5 * z) % h as usize;
                    s.extend_from_slice(&data[(sy * w as usize + sx) * 4..][..4]);
                }
            }
            s
        })
        .collect();
    AstcencImage {
        dim_x: dx as u32,
        dim_y: dy as u32,
        dim_z: dz as u32,
        data: AstcencImageData::U8(slices),
    }
}

#[test]
fn volume_matches_astcenc_cli() {
    let img = volume();
    // Keep in sync with VOLUME_CASES of the generator script
    let cases = [
        "vol_26x22x14_4x4x4_l_medium",
        "vol_26x22x14_4x4x4_l_medium_rdo20",
        "vol_26x22x14_6x6x6_s_medium",
        "vol_26x22x14_6x6x6_s_medium_rdo20",
        "vol_26x22x14_3x3x3_l_thorough",
        "vol_26x22x14_4x4x4_l_medium_a2",
        "vol_26x22x14_6x6x6_s_medium_a4_rdo20",
    ];
    for name in cases {
        let (block, mode, quality, radius, lambda) = parse_name(name);
        check(name, &img, block, mode, quality, radius, lambda);
    }
}
