// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! `block_helpers.rs` / `lowlevel.rs` vs. the C++ block level full encoder: regenerates the
//! inputs of `tools/basisu-golden/astc_ldr_block.cpp` and checks every hash in
//! `testdata/basisu_enc/basisu_astc_ldr_block.txt`.
#![allow(clippy::panic, clippy::unwrap_used, clippy::needless_range_loop)]

use std::collections::BTreeMap;

use super::EncodeBlockOutput;
use super::block_helpers::*;
use super::common::{CemEncodeParams, PixelStats};
use super::dct_enc::Dct2f;
use super::lowlevel::{
    EncodeBlockStats, LdrAstcLowlevelBlockEncoder, LdrAstcLowlevelBlockEncoderParams,
};
use super::partitions::{PartitionPatternVec, PartitionsData};
use crate::astc::helpers::LogAstcBlock;
use crate::basis::ll::comp::Image;
use crate::basis::xuastc::{ModeTable, build_mode_table};

const GOLDEN: &str = include_str!("../../../../testdata/basisu_enc/basisu_astc_ldr_block.txt");
const NUM_CASES: u32 = 200;

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
    fn range(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

struct Group {
    h: u64,
    count: u32,
}

impl Default for Group {
    fn default() -> Self {
        Self {
            h: 0xcbf29ce484222325,
            count: 0,
        }
    }
}

impl Group {
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.h ^= x as u64;
            self.h = self.h.wrapping_mul(0x100000001b3);
        }
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    fn blk(&mut self, b: &LogAstcBlock) {
        self.bytes(&[
            b.error_flag as u8,
            b.solid_color_flag_ldr as u8,
            b.solid_color_flag_hdr as u8,
            b.user_mode,
            b.grid_width,
            b.grid_height,
            b.dual_plane as u8,
            b.weight_ise_range,
            b.endpoint_ise_range,
            b.color_component_selector,
            b.num_partitions,
        ]);
        self.bytes(&b.partition_id.to_le_bytes());
        self.bytes(&b.color_endpoint_modes);
        self.bytes(&b.weights);
        self.bytes(&b.endpoints);
    }
    fn sblk(&mut self, b: &LogSurrogateAstcBlk, n: usize) {
        self.u32(b.grid_width as u32);
        self.u32(b.grid_height as u32);
        self.u32(b.cem_index);
        self.u32(b.ccs_index as u32);
        self.u32(b.num_endpoint_levels);
        self.u32(b.num_weight_levels);
        self.u32(b.num_parts);
        self.u32(b.seed_index);
        for p in 0..4 {
            for c in 0..4 {
                self.f32(b.endpoints[p][0][c]);
                self.f32(b.endpoints[p][1][c]);
            }
            self.f32(b.scales[p]);
        }
        for i in 0..n {
            self.f32(b.weights0[i]);
            self.f32(b.weights1[i]);
        }
    }
}

#[derive(Default)]
struct Groups(BTreeMap<String, Group>);

impl Groups {
    fn g(&mut self, name: &str) -> &mut Group {
        let g = self.0.entry(name.to_string()).or_default();
        g.count += 1;
        g
    }
}

const BS: [[u32; 2]; 10] = [
    [4, 4],
    [5, 4],
    [6, 5],
    [6, 6],
    [8, 5],
    [8, 8],
    [10, 6],
    [10, 10],
    [12, 10],
    [12, 12],
];

fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn make_image(c: u32, w: u32, h: u32) -> Image {
    let mut r = Lcg(c.wrapping_mul(7919).wrapping_add(31));
    let kind = c % 8;
    let alpha_mode = (c / 8) % 3;
    let mut base = [0i32; 4];
    let mut base2 = [0i32; 4];
    let mut base3 = [0i32; 4];
    for v in base.iter_mut() {
        *v = r.range(256) as i32;
    }
    for v in base2.iter_mut() {
        *v = r.range(256) as i32;
    }
    for v in base3.iter_mut() {
        *v = r.range(256) as i32;
    }
    let dx = r.range(41) as i32 - 20;
    let dy = r.range(41) as i32 - 20;
    let mut img = Image::with_size(w, h);
    for y in 0..h {
        for x in 0..w {
            let (xi, yi, wi, hi) = (x as i32, y as i32, w as i32, h as i32);
            let mut v = [0i32; 4];
            for ch in 0..4usize {
                v[ch] = match kind {
                    0 => base[ch] + (dx * xi * (ch as i32 + 1)) / 4 + (dy * yi) / 3,
                    1 => {
                        if (xi * 3 + (yi * dx) / 10) > (wi * 3) / 2 {
                            base[ch]
                        } else {
                            base2[ch]
                        }
                    }
                    2 => base[ch] + r.range(61) as i32 - 30,
                    3 => base[ch],
                    4 => {
                        let t = ((xi * 2 + yi) * 3) / (wi * 2 + hi);
                        let b = if t == 0 {
                            base[ch]
                        } else if t == 1 {
                            base2[ch]
                        } else {
                            base3[ch]
                        };
                        b + r.range(9) as i32 - 4
                    }
                    5 => base[ch] + (dx * xi) / 3 + r.range(17) as i32 - 8,
                    6 => base[0] + (dx * xi) / 2 + (dy * yi) / 2,
                    _ => {
                        let t = (xi * 64) / (wi - 1);
                        (base[ch] * (64 - t) + base2[ch] * t) / 64
                    }
                };
            }
            if alpha_mode == 0 {
                v[3] = 255;
            } else if alpha_mode == 2 {
                v[3] = 255 - (xi * 255) / (wi - 1);
            }
            *img.get_mut(x, y) = v.map(clamp8);
        }
    }
    img
}

fn sobel_xy(img: &Image) -> Image {
    let mut sx = Image::new();
    let mut sy = Image::new();
    compute_sobel(img, &mut sx, &SOBEL_X);
    compute_sobel(img, &mut sy, &SOBEL_Y);
    let mut xy = Image::with_size(img.get_width(), img.get_height());
    for y in 0..img.get_height() {
        for x in 0..img.get_width() {
            let a = sx.get(x, y);
            let b = sy.get(x, y);
            *xy.get_mut(x, y) = std::array::from_fn(|c| {
                ((a[c] as i32 - 128).abs() + (b[c] as i32 - 128).abs()).clamp(0, 255) as u8
            });
        }
    }
    xy
}

struct SizeCtx {
    tms: ModeTable,
    p2: PartitionsData,
    p3: PartitionsData,
    dct: Dct2f,
}

#[test]
fn basisu_astc_ldr_block_golden() {
    let mut golden = BTreeMap::new();
    for line in GOLDEN.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        golden.insert(
            f[0].to_string(),
            (f[1].to_string(), f[2].parse::<u32>().unwrap()),
        );
    }

    let ctx: Vec<SizeCtx> = BS
        .iter()
        .map(|&[bw, bh]| SizeCtx {
            tms: build_mode_table(bw, bh),
            p2: PartitionsData::new(2, bw, bh, true),
            p3: PartitionsData::new(3, bw, bh, true),
            dct: Dct2f::new(bh, bw).unwrap(),
        })
        .collect();

    let mut enc = LdrAstcLowlevelBlockEncoder::new();
    let mut gr = Groups::default();

    for c in 0..NUM_CASES {
        let si = (c % 10) as usize;
        let [bw, bh] = BS[si];
        let n = (bw * bh) as usize;
        let sc = &ctx[si];

        let img = make_image(c, bw * 2, bh * 2);
        let sob = sobel_xy(&img);
        {
            let g = gr.g("sobel");
            for p in sob.get_pixels() {
                g.bytes(p);
            }
        }

        let (bx, by) = (c & 1, (c >> 1) & 1);
        let mut px = [[0u8; 4]; 144];
        for y in 0..bh {
            for x in 0..bw {
                px[(x + y * bw) as usize] = img.get(bx * bw + x, by * bh + y);
            }
        }
        let mut ps = PixelStats::default();
        ps.init(n as u32, &px);

        let mut cep = CemEncodeParams::default();
        let srgb = ((c / 3) & 1) != 0;
        cep.decode_mode_srgb = srgb;
        if srgb {
            cep.comp_weights = [9, 11, 1, 11];
        }

        let mut r = Lcg(c.wrapping_mul(104729).wrapping_add(7));

        let mut parts2 = [0i32; 512];
        let mut parts3 = [0i32; 512];
        let mut num2 = (32u32 << (c % 3)).min(sc.p2.total_unique_patterns) as i32;
        let mut num3 = (32u32 << (c % 3)).min(sc.p3.total_unique_patterns) as i32;
        {
            let ok2 = estimate_partition2(bw, bh, &ps, &mut parts2, &mut num2, &sc.p2, false);
            let g = gr.g("est_part2");
            g.u32(ok2 as u32);
            g.u32(if ok2 { num2 as u32 } else { 0 });
            if ok2 {
                for &p in &parts2[..num2 as usize] {
                    g.u32(p as u32);
                }
            } else {
                num2 = 0;
            }

            let ok3 = estimate_partition3(bw, bh, &ps, &mut parts3, &mut num3, &sc.p3, false);
            let g = gr.g("est_part3");
            g.u32(ok3 as u32);
            g.u32(if ok3 { num3 as u32 } else { 0 });
            if ok3 {
                for &p in &parts3[..num3 as usize] {
                    g.u32(p as u32);
                }
            } else {
                num3 = 0;
            }

            let mut bf = [0i32; 1024];
            let mut nbf = 16;
            let okb = estimate_partition2(bw, bh, &ps, &mut bf, &mut nbf, &sc.p2, true);
            let g = gr.g("est_part2_bf");
            g.u32(okb as u32);
            g.u32(nbf as u32);
            for &p in &bf[..nbf as usize] {
                g.u32(p as u32);
            }
            nbf = 16;
            let okb3 = estimate_partition3(bw, bh, &ps, &mut bf, &mut nbf, &sc.p3, true);
            let g = gr.g("est_part3_bf");
            g.u32(okb3 as u32);
            if okb3 {
                g.u32(nbf as u32);
                for &p in &bf[..nbf as usize] {
                    g.u32(p as u32);
                }
            }

            let g = gr.g("est_sse");
            let cems = [8u32, 6, 12, 10];
            for i in 0..num2.min(4) as usize {
                for &cem in &cems {
                    let pat = &sc.p2.partition_pats[parts2[i] as usize];
                    g.f32(estimate_partition_sse(cem, &ps, bw, bh, 2, pat));
                }
            }
            for i in 0..num3.min(4) as usize {
                for &cem in &cems {
                    let pat = &sc.p3.partition_pats[parts3[i] as usize];
                    g.f32(estimate_partition_sse(cem, &ps, bw, bh, 3, pat));
                }
            }
        }

        for _ in 0..6 {
            let tm_index = r.range(sc.tms.modes.len() as u32);
            let tm = sc.tms.modes[tm_index as usize];
            let pick = r.next();
            let flags = TrialFlags {
                gradient_descent_flag: pick & 1 == 0,
                polish_weights_flag: pick & 2 == 0,
                qcd_enabled_flag: pick & 4 == 0,
                use_blue_contraction: pick & 8 == 0,
            };
            let sflags = (pick >> 4) & 1;
            let eo = if (pick >> 5) & 1 != 0 { 0.1f32 } else { 0.0 };
            let (cem, ccs) = (tm.cem as u32, tm.ccs_index as i32);
            let (er, wr) = (tm.endpoint_range as u32, tm.weight_range as u32);
            let (gw, gh) = (tm.grid_w as u32, tm.grid_h as u32);
            let num_parts = tm.num_parts as u32;

            let pd = if num_parts == 3 { &sc.p3 } else { &sc.p2 };
            let num_est = if num_parts == 3 { num3 } else { num2 };
            if num_parts > 1 && num_est == 0 {
                gr.g("trial_skip").u32(tm_index);
                continue;
            }
            let mut seed = 0u32;
            let mut pat: Option<&PartitionPatternVec> = None;
            if num_parts > 1 {
                let list = if num_parts == 3 { &parts3 } else { &parts2 };
                let unique = list[((pick >> 8) % num_est as u32) as usize] as usize;
                seed = pd.unique_index_to_part_seed[unique] as u32;
                pat = Some(&pd.partition_pats[unique]);
            }

            {
                let mut lb = LogSurrogateAstcBlk::default();
                let e = match pat {
                    None => encode_surrogate_trial(
                        bw, bh, &ps, cem, ccs, er, wr, gw, gh, &mut lb, &cep, sflags,
                    ),
                    Some(pat) => encode_surrogate_trial_subsets(
                        bw, bh, &ps, cem, num_parts, seed, pat, er, wr, gw, gh, &mut lb, &cep,
                        sflags,
                    ),
                };
                let g = gr.g("surrogate");
                g.u32(tm_index);
                g.f32(e);
                g.sblk(&lb, (gw * gh) as usize);

                let mut dec = [[0f32; 4]; 144];
                lb.decode(bw, bh, &mut dec, pat);
                let g = gr.g("surrogate_decode");
                for d in &dec[..n] {
                    for &v in d {
                        g.f32(v);
                    }
                }
            }

            let mut lb = LogAstcBlock::default();
            let mut tdef = false;
            let ok = match pat {
                None => encode_trial(
                    bw,
                    bh,
                    &ps,
                    cem,
                    ccs >= 0,
                    ccs,
                    er,
                    wr,
                    gw,
                    gh,
                    eo,
                    &mut lb,
                    &cep,
                    flags,
                    Some(&mut tdef),
                ),
                Some(pat) => encode_trial_subsets(
                    bw,
                    bh,
                    &ps,
                    cem,
                    num_parts,
                    seed,
                    pat,
                    er,
                    wr,
                    gw,
                    gh,
                    eo,
                    &mut lb,
                    &cep,
                    false,
                    flags,
                    Some(&mut tdef),
                ),
            };
            {
                let g = gr.g("trial");
                g.u32(tm_index);
                g.u32(ok as u32);
                g.u32(tdef as u32);
                if ok {
                    g.blk(&lb);
                    g.u64(eval_error(bw, bh, &lb, &ps, &cep));
                }
            }
            if !ok {
                continue;
            }

            {
                let off = TrialFlags {
                    gradient_descent_flag: false,
                    polish_weights_flag: false,
                    qcd_enabled_flag: false,
                    use_blue_contraction: flags.use_blue_contraction,
                };
                let mut lb2 = LogAstcBlock::default();
                let ok2 = match pat {
                    None => encode_trial(
                        bw,
                        bh,
                        &ps,
                        cem,
                        ccs >= 0,
                        ccs,
                        er,
                        wr,
                        gw,
                        gh,
                        0.0,
                        &mut lb2,
                        &cep,
                        off,
                        None,
                    ),
                    Some(pat) => encode_trial_subsets(
                        bw, bh, &ps, cem, num_parts, seed, pat, er, wr, gw, gh, 0.0, &mut lb2,
                        &cep, false, off, None,
                    ),
                };
                let g = gr.g("polish");
                g.u32(ok2 as u32);
                if ok2 {
                    let mut improved = false;
                    polish_block_weights(
                        bw,
                        bh,
                        &ps,
                        &mut lb2,
                        &cep,
                        pat,
                        &mut improved,
                        true,
                        true,
                        pick & 16 == 0,
                    );
                    g.u32(improved as u32);
                    g.blk(&lb2);
                    g.u64(eval_error(bw, bh, &lb2, &ps, &cep));
                }
            }

            let mut lb3 = lb;
            let mut tdef3 = false;
            match pat {
                None => {
                    encode_trial_refine_only(
                        bw,
                        bh,
                        &ps,
                        &mut lb3,
                        &cep,
                        flags.use_blue_contraction,
                        Some(&mut tdef3),
                    );
                    let g = gr.g("refine_only");
                    g.u32(tdef3 as u32);
                    g.blk(&lb3);
                }
                Some(pat) => {
                    let ok3 = encode_trial_subsets(
                        bw,
                        bh,
                        &ps,
                        cem,
                        num_parts,
                        seed,
                        pat,
                        er,
                        wr,
                        gw,
                        gh,
                        eo,
                        &mut lb3,
                        &cep,
                        true,
                        flags,
                        Some(&mut tdef3),
                    );
                    let g = gr.g("refine_only_subsets");
                    g.u32(ok3 as u32);
                    g.u32(tdef3 as u32);
                    g.blk(&lb3);
                }
            }
        }

        {
            let mut p = LdrAstcLowlevelBlockEncoderParams::new(
                &sob,
                &sc.p2,
                &sc.p3,
                &cep,
                &sc.tms.modes,
                &sc.tms,
                &sc.dct,
            );
            p.bx = bx;
            p.by = by;
            p.block_width = bw;
            p.block_height = bh;
            p.total_block_pixels = n as u32;
            p.alpha_cems = ps.has_alpha;
            p.use_alpha_or_opaque_modes = ps.has_alpha;
            p.use_lum_direct_modes = c % 4 == 0;
            p.use_base_scale_modes = c % 3 != 2;
            p.use_direct_modes = c % 17 != 16 || !p.use_base_scale_modes;
            p.subsets_enabled = c % 5 != 3;
            p.use_dual_planes = c % 5 != 4;
            p.grid_hv_filtering = c % 7 == 3;
            p.filter_horizontally_flag = c & 1 != 0;
            p.use_small_grids_only = c % 11 == 5;
            p.early_stop_wpsnr = if c % 6 == 1 { 45.0 } else { 0.0 };
            p.early_stop2_wpsnr = if c % 6 == 2 { 40.0 } else { 0.0 };
            p.bucket_pruning_passes = c % 9 != 8;
            p.use_superbuckets = c % 13 != 12;
            p.final_encode_always_try_rgb_direct = c % 10 == 7;
            p.final_encode_try_base_ofs = c % 8 != 6;
            p.use_fast_part_est_stage2 = c % 12 != 11;
            if c % 4 == 1 {
                p.dp_active_chans[1] = false;
            }

            let mut out_blocks: Vec<EncodeBlockOutput> = Vec::new();
            let mut stats = EncodeBlockStats::default();
            let ok = enc.full_encode(&p, &ps, &mut out_blocks, c & 3, &mut stats);
            let g = gr.g("full_encode");
            g.u32(ok as u32);
            g.u32(out_blocks.len() as u32);
            for ob in &out_blocks {
                g.u32(ob.trial_mode_index as i32 as u32);
                g.u32(ob.blur_id as u32);
                g.blk(&ob.log_blk);
                g.u64(ob.sse);
            }
            if std::env::var("XU_BLOCK_DEBUG").is_ok() {
                for (i, ob) in out_blocks.iter().enumerate() {
                    let mut gg = Group::default();
                    gg.blk(&ob.log_blk);
                    eprintln!(
                        "{c} {i} tm={} sse={} blk={:016x}",
                        ob.trial_mode_index, ob.sse, gg.h
                    );
                    if std::env::var("XU_BLOCK_DEBUG2").is_ok_and(|v| v == c.to_string()) {
                        let b = &ob.log_blk;
                        eprintln!(
                            "  cem {} parts {} pid {} er {} wr {} g {}x{} dp {} ccs {}",
                            b.color_endpoint_modes[0],
                            b.num_partitions,
                            b.partition_id,
                            b.endpoint_ise_range,
                            b.weight_ise_range,
                            b.grid_width,
                            b.grid_height,
                            b.dual_plane as u32,
                            b.color_component_selector
                        );
                        eprintln!("  E: {:?}", b.endpoints);
                        eprintln!("  W: {:?}", b.weights);
                    }
                }
            }
            let g = gr.g("full_encode_stats");
            g.u32(stats.total_superbuckets_created);
            g.u32(stats.total_buckets_created);
            g.u32(stats.total_surrogate_encodes);
            g.u32(stats.total_shortlist_candidates);
            g.u32(stats.total_full_encodes);
            g.f32(enc.sobel_energy);
            g.f32(enc.max_std_dev);
            g.u32(enc.block_complexity_index);
        }
    }

    let mut bad = Vec::new();
    for (name, g) in &gr.0 {
        match golden.get(name) {
            Some((h, cnt)) if *h == format!("{:016x}", g.h) && *cnt == g.count => {}
            other => bad.push(format!(
                "{name}: got {:016x}/{} expected {:?}",
                g.h, g.count, other
            )),
        }
    }
    for name in golden.keys() {
        if !gr.0.contains_key(name) {
            bad.push(format!("{name}: missing"));
        }
    }
    assert!(bad.is_empty(), "mismatches:\n{}", bad.join("\n"));
}
