// SPDX-FileCopyrightText: 2010-2026 Mark Callow, The Khronos Group Inc. and KTX-Software contributors
// SPDX-License-Identifier: Apache-2.0
// From ktx-rs tests/astcenc_compress.rs; see NOTICE.md.
//! astcenc compressor (`astcenc_compress_image`) vs. golden output of the C++ original
//! (ktx-rs `scripts/golden/gen_astcenc_compress_golden.sh`): whole images over block sizes,
//! profiles, qualities, flag sets, swizzles and data types, on every instruction set copy;
//! also with LZ-RDO (`ASTCENC_FLG_USE_LZ_RDO`) for the LDR profiles, and with alpha scaling
//! (`a_scale_radius`) on images with transparent regions.
#![allow(clippy::panic, clippy::needless_range_loop)]

use super::*;
use std::collections::BTreeMap;

fn fnv(p: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in p {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub struct Lcg(pub u32);

impl Lcg {
    pub fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() & 0xffff) as f32 / 65535.0
    }
}

/// Texel value in [0, 1] for content `kind` (the harness `texel`).
fn texel(l: &mut Lcg, kind: u32, x: u32, y: u32, z: u32, c: u32) -> f32 {
    let n = l.unit();
    match kind {
        0 => {
            if c == 3 {
                1.0
            } else {
                let a = (x * (c + 1) + y * 3 + z * 5) as f32 / 48.0;
                (a + n * 0.02) - (a as i32) as f32
            }
        }
        1 => n,
        2 => {
            if (x + 2 * y) % 11 < 5 {
                if c == 0 { 0.9 } else { 0.1 + n * 0.05 }
            } else if c == 2 {
                0.8
            } else {
                0.3 + n * 0.05
            }
        }
        3 => {
            if c == 3 {
                ((x + y) % 9) as f32 / 8.0
            } else {
                0.25 + 0.5 * ((x * 7 + y * 3) % 13) as f32 / 12.0
            }
        }
        _ => {
            if c == 1 {
                0.5
            } else {
                0.25
            }
        }
    }
}

/// The harness float to half conversion (round to nearest even, tiny values flushed).
fn to_half(f: f32) -> u16 {
    let u = f.to_bits();
    let e = (u >> 23) & 0xff;
    let m = u & 0x7fffff;
    if f == 0.0 || e < 113 {
        return 0;
    }
    let mut he = e - 112;
    let mut hm = m >> 13;
    let rest = m & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && hm & 1 == 1) {
        hm += 1;
    }
    if hm == 0x400 {
        hm = 0;
        he += 1;
    }
    ((he << 10) | hm) as u16
}

#[test]
fn compress_matches_astcenc() {
    check_golden(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/astcenc/compress_golden.txt"
        )),
        false,
    );
}

#[test]
fn compress_lz_rdo_matches_astcenc() {
    check_golden(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/astcenc/compress_lz_rdo_golden.txt"
        )),
        true,
    );
}

/// Compresses the generated cases (`rdo`: the LZ-RDO set) with every instruction set copy and
/// 1..=4 threads and compares their hashes with `golden` (`tools/astcenc-golden`).
fn check_golden(golden: &str, rdo: bool) {
    let golden: BTreeMap<String, String> = golden
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    const SIZES: [[u32; 3]; 8] = [
        [4, 4, 1],
        [5, 4, 1],
        [6, 6, 1],
        [8, 8, 1],
        [10, 5, 1],
        [12, 12, 1],
        [3, 3, 3],
        [4, 4, 4],
    ];
    let profiles = [
        AstcencProfile::Ldr,
        AstcencProfile::LdrSrgb,
        AstcencProfile::HdrRgbLdrA,
        AstcencProfile::Hdr,
    ];
    let qualities = [0.0f32, 60.0, 98.0];
    let flag_sets = [
        0,
        ASTCENC_FLG_MAP_NORMAL,
        ASTCENC_FLG_USE_ALPHA_WEIGHT | ASTCENC_FLG_USE_PERCEPTUAL,
        ASTCENC_FLG_USE_DECODE_UNORM8,
    ];
    let mut fails = Vec::new();
    let mut n = 0;
    for (si, s) in SIZES.iter().enumerate() {
        for (pi, &prof) in profiles.iter().enumerate().take(if rdo { 2 } else { 4 }) {
            for (qi, &q) in qualities.iter().enumerate() {
                for fi in 0..4usize {
                    let mut flags = flag_sets[fi];
                    let hdr = pi >= 2;
                    if hdr && fi == 3 {
                        continue;
                    }
                    if pi == 0 && fi == 3 {
                        // RGBM with LZ-RDO can make upstream encode an uninitialized error block
                        flags = if rdo {
                            ASTCENC_FLG_USE_DECODE_UNORM8
                        } else {
                            ASTCENC_FLG_MAP_RGBM
                        };
                    }
                    let dt = (si + pi + qi + fi) % 3;
                    let kind = ((si * 3 + pi + qi * 2 + fi) % 5) as u32;
                    if rdo {
                        flags |= ASTCENC_FLG_USE_LZ_RDO;
                    }
                    let Ok(mut c) = astcenc_config_init(prof, s[0], s[1], s[2], q, flags) else {
                        panic!("config_init")
                    };
                    if rdo {
                        c.lz_rdo_lambda = [10.0, 20.0, 40.0][(si + qi + fi) % 3];
                    }
                    // 1..=4 threads sharing the context; the output must not depend on it.
                    // wasm has no threads.
                    let threads = if cfg!(target_family = "wasm") {
                        1
                    } else {
                        (n % 4) as u32 + 1
                    };
                    let (mut dx, mut dy, mut dz) = (
                        s[0] * 2 + 1,
                        s[1] * 2 - 1,
                        if s[2] == 1 { 1 } else { s[2] + 1 },
                    );
                    if rdo {
                        // Larger images; the first case spans two 4096 block optimization chunks
                        (dx, dy, dz) = (
                            s[0] * 9 + 1,
                            s[1] * 7 - 1,
                            if s[2] == 1 { 1 } else { s[2] * 3 + 1 },
                        );
                        if (si, pi, qi, fi) == (0, 0, 1, 0) {
                            (dx, dy) = (264, 257);
                        }
                    }
                    let mut l = Lcg((si * 1000 + pi * 100 + qi * 10 + fi) as u32);
                    let mut p8 = vec![Vec::new(); dz as usize];
                    let mut p16 = vec![Vec::new(); dz as usize];
                    let mut p32 = vec![Vec::new(); dz as usize];
                    for z in 0..dz {
                        for y in 0..dy {
                            for x in 0..dx {
                                for ch in 0..4 {
                                    let v = texel(&mut l, kind, x, y, z, ch).min(1.0);
                                    let hv = if hdr && ch != 3 { v * 6.0 } else { v };
                                    p8[z as usize].push((v * 255.0 + 0.5) as u8);
                                    p16[z as usize].push(to_half(hv));
                                    p32[z as usize].push(hv);
                                }
                            }
                        }
                    }
                    let img = AstcencImage {
                        dim_x: dx,
                        dim_y: dy,
                        dim_z: dz,
                        data: match dt {
                            0 => AstcencImageData::U8(p8),
                            1 => AstcencImageData::F16(p16),
                            _ => AstcencImageData::F32(p32),
                        },
                    };
                    let swz = if fi == 1 {
                        AstcencSwizzle {
                            r: AstcencSwz::R,
                            g: AstcencSwz::R,
                            b: AstcencSwz::R,
                            a: AstcencSwz::G,
                        }
                    } else {
                        AstcencSwizzle::RGBA
                    };
                    // Every instruction set copy the CPU runs must give the same output.
                    for isa in [AstcencIsa::Baseline, AstcencIsa::Sse41, AstcencIsa::Avx2] {
                        if isa > AstcencIsa::detect() {
                            continue;
                        }
                        let Ok(ctx) = astcenc_context_alloc_isa(&c, threads, isa) else {
                            panic!("context_alloc")
                        };
                        assert_eq!(ctx.isa(), Some(isa));
                        let blocks = dx.div_ceil(s[0]) * dy.div_ceil(s[1]) * dz.div_ceil(s[2]);
                        let mut out = vec![0u8; blocks as usize * 16];
                        let shared = std::sync::Mutex::new(out.as_mut_slice());
                        if cfg!(target_family = "wasm") {
                            if let Err(e) = astcenc_compress_image(&ctx, &img, &swz, &shared, 0) {
                                panic!("compress: {e:?}");
                            }
                        } else {
                            std::thread::scope(|sc| {
                                let hs: Vec<_> = (0..threads)
                                    .map(|t| {
                                        let (ctx, img, swz, shared) = (&ctx, &img, &swz, &shared);
                                        sc.spawn(move || {
                                            astcenc_compress_image(ctx, img, swz, shared, t)
                                        })
                                    })
                                    .collect();
                                for h in hs {
                                    match h.join() {
                                        Ok(Ok(())) => {}
                                        other => panic!("compress: {other:?}"),
                                    }
                                }
                            });
                        }
                        let prefix = if rdo { "rdo " } else { "" };
                        let key = format!("{prefix}s{si} p{pi} q{qi} f{fi} {isa:?}");
                        let got = format!("{:016x}", fnv(&out));
                        match golden.get(&format!("{prefix}s{si} p{pi} q{qi} f{fi}")) {
                            Some(w) if *w == got => {}
                            Some(w) => fails.push(format!("{key}: want {w} got {got}")),
                            None => fails.push(format!("{key}: missing")),
                        }
                    }
                    n += 1;
                }
            }
        }
    }
    assert_eq!(n, golden.len());
    if !fails.is_empty() {
        panic!(
            "{} of {n} mismatches:\n{}",
            fails.len(),
            fails[..fails.len().min(40)].join("\n")
        );
    }
}

/// The harness `ascale_texel`: noise RGB; alpha 0 except for a disc (1.0, with a 0.5 rim) and
/// near-zero specks.
fn ascale_texel(l: &mut Lcg, (x, y, z): (u32, u32, u32), c: u32, (dx, dy): (u32, u32)) -> f32 {
    let n = l.unit();
    if c != 3 {
        return n;
    }
    let ddx = x as i32 - (dx * 3 / 10) as i32;
    let ddy = y as i32 - (dy / 2) as i32;
    let d2 = ddx * ddx + ddy * ddy + (z * z * 4) as i32;
    let r = (dy / 4) as i32 + 1;
    if d2 < r * r {
        1.0
    } else if d2 < (r + 2) * (r + 2) {
        0.5
    } else if (x * 7 + y * 13 + z * 3) % 97 == 0 {
        // Specks of 0.0005 .. 0.004: near the negligible alpha threshold for float data
        (((x * 7 + y * 13 + z * 3) / 97) % 8 + 1) as f32 * 0.0005
    } else {
        0.0
    }
}

/// Runs `astcenc_compress_image` on `threads` threads sharing `ctx`.
fn compress_threads(
    ctx: &AstcencContext,
    img: &AstcencImage,
    swz: &AstcencSwizzle,
    out: &mut [u8],
    threads: u32,
) {
    let shared = std::sync::Mutex::new(out);
    if threads == 1 {
        if let Err(e) = astcenc_compress_image(ctx, img, swz, &shared, 0) {
            panic!("compress: {e:?}");
        }
        return;
    }
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let shared = &shared;
                sc.spawn(move || astcenc_compress_image(ctx, img, swz, shared, t))
            })
            .collect();
        for h in hs {
            match h.join() {
                Ok(Ok(())) => {}
                other => panic!("compress: {other:?}"),
            }
        }
    });
}

/// `a_scale_radius` (alpha scaling) vs. the `ascale` cases of `tools/astcenc-golden`: block
/// sizes (2D and 3D) x radii x profiles with rotating data
/// types, qualities, swizzles, the alpha weight flag and LZ-RDO; 1..=4 threads, every
/// instruction set copy.
#[test]
fn compress_alpha_scale_matches_astcenc() {
    let golden: BTreeMap<String, String> = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/astcenc/compress_alpha_scale_golden.txt"
    ))
    .lines()
    .filter_map(|l| l.split_once('\t'))
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    const SIZES: [[u32; 3]; 9] = [
        [4, 4, 1],
        [5, 4, 1],
        [6, 6, 1],
        [8, 8, 1],
        [12, 12, 1],
        [3, 3, 3],
        [10, 5, 1],
        [6, 6, 6],
        [4, 4, 4],
    ];
    const RADII: [u32; 5] = [1, 2, 4, 8, 100];
    let profiles = [
        AstcencProfile::Ldr,
        AstcencProfile::LdrSrgb,
        AstcencProfile::HdrRgbLdrA,
        AstcencProfile::Hdr,
    ];
    let qualities = [0.0f32, 60.0, 98.0];
    let mut fails = Vec::new();
    let mut n = 0;
    for (si, s) in SIZES.iter().enumerate() {
        for (ri, &radius) in RADII.iter().enumerate() {
            for (pi, &prof) in profiles.iter().enumerate() {
                let hdr = pi >= 2;
                let dt = (si + ri + pi) % 3;
                let qi = (si * 2 + ri + pi) % 3;
                let sw = (si + ri * 2 + pi) % 3;
                let rdo = !hdr && (si + ri + pi) % 2 == 0;
                let mut flags = if (si + ri) % 4 == 3 {
                    0
                } else {
                    ASTCENC_FLG_USE_ALPHA_WEIGHT
                };
                if rdo {
                    flags |= ASTCENC_FLG_USE_LZ_RDO;
                }
                let Ok(mut c) = astcenc_config_init(prof, s[0], s[1], s[2], qualities[qi], flags)
                else {
                    panic!("config_init")
                };
                c.a_scale_radius = radius;
                if rdo {
                    c.lz_rdo_lambda = [10.0, 20.0, 40.0][(si + ri) % 3];
                }
                let (dx, dy) = (s[0] * 8 + 3, s[1] * 6 + 1);
                let dz = if s[2] == 1 { 1 } else { s[2] * 3 + 1 };
                let mut l = Lcg((50000 + si * 1000 + ri * 100 + pi * 10) as u32);
                let mut p8 = vec![Vec::new(); dz as usize];
                let mut p16 = vec![Vec::new(); dz as usize];
                let mut p32 = vec![Vec::new(); dz as usize];
                for z in 0..dz {
                    for y in 0..dy {
                        for x in 0..dx {
                            for ch in 0..4 {
                                let v = ascale_texel(&mut l, (x, y, z), ch, (dx, dy));
                                let hv = if hdr && ch != 3 { v * 6.0 } else { v };
                                p8[z as usize].push((v * 255.0 + 0.5) as u8);
                                p16[z as usize].push(to_half(hv));
                                p32[z as usize].push(hv);
                            }
                        }
                    }
                }
                let img = AstcencImage {
                    dim_x: dx,
                    dim_y: dy,
                    dim_z: dz,
                    data: match dt {
                        0 => AstcencImageData::U8(p8),
                        1 => AstcencImageData::F16(p16),
                        _ => AstcencImageData::F32(p32),
                    },
                };
                let swz = match sw {
                    0 => AstcencSwizzle::RGBA,
                    1 => AstcencSwizzle {
                        r: AstcencSwz::R,
                        g: AstcencSwz::B,
                        b: AstcencSwz::B,
                        a: AstcencSwz::G,
                    },
                    _ => AstcencSwizzle {
                        a: AstcencSwz::Zero,
                        ..AstcencSwizzle::RGBA
                    },
                };
                let threads = if cfg!(target_family = "wasm") {
                    1
                } else {
                    (n % 4) as u32 + 1
                };
                let key = format!("ascale s{si} r{ri} p{pi}");
                for isa in [AstcencIsa::Baseline, AstcencIsa::Sse41, AstcencIsa::Avx2] {
                    if isa > AstcencIsa::detect() {
                        continue;
                    }
                    let Ok(ctx) = astcenc_context_alloc_isa(&c, threads, isa) else {
                        panic!("context_alloc")
                    };
                    let blocks = dx.div_ceil(s[0]) * dy.div_ceil(s[1]) * dz.div_ceil(s[2]);
                    let mut out = vec![0u8; blocks as usize * 16];
                    compress_threads(&ctx, &img, &swz, &mut out, threads);
                    let got = format!("{:016x}", fnv(&out));
                    match golden.get(&key) {
                        Some(w) if *w == got => {}
                        Some(w) => fails.push(format!("{key} {isa:?}: want {w} got {got}")),
                        None => fails.push(format!("{key} {isa:?}: missing")),
                    }
                }
                n += 1;
            }
        }
    }
    assert_eq!(n, golden.len());
    if !fails.is_empty() {
        panic!(
            "{} of {n} mismatches:\n{}",
            fails.len(),
            fails[..fails.len().min(40)].join("\n")
        );
    }
}

/// `validate_image`: volumetric images require a three-dimensional block size.
#[test]
fn compress_rejects_3d_image_with_2d_blocks() {
    let Ok(c) = astcenc_config_init(AstcencProfile::Ldr, 4, 4, 1, 60.0, 0) else {
        panic!("config_init")
    };
    let Ok(ctx) = astcenc_context_alloc(&c, 1) else {
        panic!("context_alloc")
    };
    let img = AstcencImage {
        dim_x: 8,
        dim_y: 8,
        dim_z: 2,
        data: AstcencImageData::U8(vec![vec![0; 256]; 2]),
    };
    let mut out = vec![0u8; 4 * 2 * 16];
    let shared = std::sync::Mutex::new(out.as_mut_slice());
    let r = astcenc_compress_image(&ctx, &img, &AstcencSwizzle::RGBA, &shared, 0);
    assert_eq!(r, Err(AstcencError::BadParam));
}

/// Compresses a noise image with a progress callback on `threads` threads and returns the
/// reports (value, thread) and the worker thread ids.
fn progress_run(
    rdo: bool,
    threads: u32,
    (dx, dy): (u32, u32),
) -> (
    Vec<(f32, std::thread::ThreadId)>,
    Vec<std::thread::ThreadId>,
) {
    let flags = if rdo { ASTCENC_FLG_USE_LZ_RDO } else { 0 };
    let Ok(mut c) = astcenc_config_init(AstcencProfile::Ldr, 4, 4, 1, 0.0, flags) else {
        panic!("config_init")
    };
    c.lz_rdo_lambda = if rdo { 20.0 } else { 0.0 };
    let reports = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&reports);
    c.progress_callback = Some(AstcencProgressCallback::new(move |v| {
        if let Ok(mut r) = sink.lock() {
            r.push((v, std::thread::current().id()));
        }
    }));
    let Ok(ctx) = astcenc_context_alloc(&c, threads) else {
        panic!("context_alloc")
    };
    let mut l = Lcg(7);
    let data: Vec<u8> = (0..dx * dy * 4).map(|_| (l.next() & 0xff) as u8).collect();
    let img = AstcencImage {
        dim_x: dx,
        dim_y: dy,
        dim_z: 1,
        data: AstcencImageData::U8(vec![data]),
    };
    let mut out = vec![0u8; (dx / 4 * dy / 4 * 16) as usize];
    let shared = std::sync::Mutex::new(out.as_mut_slice());
    let workers = if threads == 1 {
        if let Err(e) = astcenc_compress_image(&ctx, &img, &AstcencSwizzle::RGBA, &shared, 0) {
            panic!("compress: {e:?}");
        }
        vec![std::thread::current().id()]
    } else {
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let (ctx, img, shared) = (&ctx, &img, &shared);
                    sc.spawn(move || {
                        let r = astcenc_compress_image(ctx, img, &AstcencSwizzle::RGBA, shared, t);
                        (r, std::thread::current().id())
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| match h.join() {
                    Ok((Ok(()), id)) => id,
                    other => panic!("compress: {other:?}"),
                })
                .collect()
        })
    };
    let Ok(r) = reports.lock() else {
        panic!("reports")
    };
    (r.clone(), workers)
}

/// `progress_callback`: single threaded, the reports are those of the C++ `ParallelManager`
/// (`tools/astcenc-golden` `progress`: 0..100% over the compression, or 0..33% and 33..100%
/// over the compression and the LZ-RDO optimization) and run on the compressing thread; with
/// several threads each stage rises to its end and the reports run on the worker threads.
#[test]
fn progress_callback_matches_parallel_manager() {
    let golden: BTreeMap<&str, Vec<f32>> = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/astcenc/compress_progress_golden.txt"
    ))
    .lines()
    .filter_map(|l| l.split_once('\t'))
    .map(|(k, v)| {
        (
            k,
            v.split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect(),
        )
    })
    .collect();
    let (dx, dy) = (1024u32, 512u32);

    for (rdo, key) in [(false, "progress plain"), (true, "progress rdo")] {
        let (reports, workers) = progress_run(rdo, 1, (dx, dy));
        let values: Vec<f32> = reports.iter().map(|r| r.0).collect();
        assert_eq!(Some(&values), golden.get(key), "{key}");
        assert!(reports.iter().all(|r| r.1 == workers[0]));
        assert_eq!(workers[0], std::thread::current().id());
    }

    if cfg!(target_family = "wasm") {
        return;
    }
    for rdo in [false, true] {
        let (reports, workers) = progress_run(rdo, 4, (dx, dy));
        let values: Vec<f32> = reports.iter().map(|r| r.0).collect();
        // Each stage reports rising values and its end once; with LZ-RDO a worker can start
        // the optimization before another reports the end of the compression (as in C++)
        let stages: Vec<Vec<f32>> = if rdo {
            vec![
                values.iter().copied().filter(|&v| v <= 33.0).collect(),
                values.iter().copied().filter(|&v| v > 33.0).collect(),
            ]
        } else {
            vec![values.clone()]
        };
        let ends: &[f32] = if rdo { &[33.0, 100.0] } else { &[100.0] };
        for (stage, &end) in stages.iter().zip(ends) {
            assert!(stage.windows(2).all(|w| w[0] < w[1]), "{values:?}");
            assert_eq!(stage.iter().copied().fold(0.0, f32::max), end, "{values:?}");
        }
        assert!(values.len() >= 4, "{values:?}");
        assert!(reports.iter().all(|r| workers.contains(&r.1)));
        assert!(!workers.contains(&std::thread::current().id()));
    }
}
