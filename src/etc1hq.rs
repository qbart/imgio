//! High-quality ETC1 block encoder: an exhaustive flip x (individual | differential) search,
//! every subblock fitted with Basis Universal's `etc1_optimizer` (`ll::etc_enc`, bit-exact
//! port; the second subblock of a differential block is constrained to the first's base color,
//! as in rg_etc1). Solid blocks use basisu `pack_etc1_block_solid_color`.

use crate::basis::ll::etc_enc::{
    BasisEtc1PackParams, BasisEtcQuality, Etc1Optimizer, Etc1OptimizerParams, Etc1OptimizerResults,
    EtcBlock, pack_etc1_block_solid_color,
};

/// Pixel indices (row major in the 4x4 block) of subblock `s` for `flip`.
fn subblock_pixels(flip: bool, s: usize) -> [usize; 8] {
    if flip {
        // Top / bottom 4x2 halves.
        core::array::from_fn(|i| s * 8 + i)
    } else {
        // Left / right 2x4 halves, column by column like rg_etc1.
        let c = s * 2;
        [c, c + 4, c + 8, c + 12, c + 1, c + 5, c + 9, c + 13]
    }
}

/// Encodes 16 RGBA8 pixels (row major) to an 8-byte ETC1 block (alpha ignored). `uber`
/// selects the optimizer's slowest search; `perceptual` its weighted error.
pub(crate) fn encode_block(px: &[[u8; 4]; 16], uber: bool, perceptual: bool) -> [u8; 8] {
    search(px, uber, perceptual, false)
}

/// The flip x mode search; `differential_only` skips individual mode (ETC2 punch-through,
/// where the diff bit is the opaque bit).
fn search(px: &[[u8; 4]; 16], uber: bool, perceptual: bool, differential_only: bool) -> [u8; 8] {
    let mut block = EtcBlock::default();
    if !differential_only && px.iter().all(|p| p[..3] == px[0][..3]) {
        pack_etc1_block_solid_color(&mut block, &px[0][..3]);
        return block.bytes;
    }
    let pack = BasisEtc1PackParams {
        quality: if uber {
            BasisEtcQuality::Uber
        } else {
            BasisEtcQuality::Slow
        },
        perceptual,
        ..BasisEtc1PackParams::default()
    };
    let mut optimizer = Etc1Optimizer::new();
    let mut best: Option<(u64, bool, bool, [Etc1OptimizerResults; 2])> = None;
    for flip in [false, true] {
        for color4 in [false, true] {
            if color4 && differential_only {
                continue;
            }
            let mut results: [Etc1OptimizerResults; 2] = Default::default();
            let mut total = 0u64;
            let mut ok = true;
            for s in 0..2 {
                let pixels = subblock_pixels(flip, s).map(|i| px[i]);
                let mut params = Etc1OptimizerParams::from_pack_params(&pack);
                params.num_src_pixels = 8;
                params.src_pixels = &pixels;
                params.use_color4 = color4;
                if !color4 && s == 1 {
                    params.constrain_against_base_color5 = true;
                    params.base_color5 = results[0].block_color_unscaled;
                }
                results[s].n = 8;
                results[s].selectors = vec![0; 8];
                optimizer.init(&params);
                if !optimizer.compute(&params, &mut results[s]) {
                    ok = false;
                    break;
                }
                total = total.saturating_add(results[s].error);
                if best.as_ref().is_some_and(|b| total >= b.0) {
                    ok = false;
                    break;
                }
            }
            if ok {
                best = Some((total, flip, color4, results));
            }
        }
    }
    let Some((_, flip, color4, results)) = best else {
        // The optimizer always succeeds for 8 pixels; fall back to the solid average.
        pack_etc1_block_solid_color(&mut block, &px[0][..3]);
        return block.bytes;
    };
    block.set_flip_bit(flip);
    block.set_diff_bit(!color4);
    let (c0, c1) = (
        results[0].block_color_unscaled,
        results[1].block_color_unscaled,
    );
    if color4 {
        block.set_block_color4(&c0, &c1);
    } else {
        block.set_block_color5(&c0, &c1);
    }
    for (s, r) in results.iter().enumerate() {
        block.set_inten_table(s as u32, r.block_inten_table);
        for (k, &i) in subblock_pixels(flip, s).iter().enumerate() {
            let sel = r.selectors.get(k).copied().unwrap_or(0);
            block.set_selector((i & 3) as u32, (i >> 2) as u32, u32::from(sel));
        }
    }
    block.bytes
}

/// A punch-through block choice: (error, flip, color5 per subblock, table per subblock).
type Choice = (u64, bool, [[i32; 3]; 2], [usize; 2]);

/// ETC2 modifier pairs (`a`, `b`) per table.
const MODIFIERS: [[i32; 2]; 8] = [
    [2, 8],
    [5, 17],
    [9, 29],
    [13, 42],
    [18, 60],
    [24, 80],
    [33, 106],
    [47, 183],
];

fn ext5(c: i32) -> i32 {
    (c << 3) | (c >> 2)
}

/// Encodes 16 RGBA8 pixels to an ETC2 RGB A1 (punch-through) block: alpha < 128 is
/// transparent. Opaque blocks use the differential-only search (opaque bit set); blocks with
/// transparent texels clear the opaque bit, where raw index 2 is transparent and 0 / 1 / 3
/// give base / base + b / base - b, and search 5-bit base colors around each subblock's
/// opaque average for every flip x table, keeping the best pair within the delta range.
pub(crate) fn encode_punchthrough_block(
    px: &[[u8; 4]; 16],
    uber: bool,
    perceptual: bool,
) -> [u8; 8] {
    if px.iter().all(|p| p[3] >= 128) {
        let mut b = search(px, uber, perceptual, true);
        b[3] |= 2; // opaque (diff) bit
        return b;
    }
    // Error of the opaque texels of `pixels` for base color `c` (5-bit) and table `t`.
    let err_of = |pixels: &[[i32; 3]], c: [i32; 3], t: usize| -> u64 {
        let base = c.map(ext5);
        let b = MODIFIERS[t][1];
        pixels
            .iter()
            .map(|p| {
                [0, b, -b]
                    .iter()
                    .map(|m| {
                        (0..3)
                            .map(|k| {
                                let v = (base[k] + m).clamp(0, 255);
                                ((p[k] - v) * (p[k] - v)) as u64
                            })
                            .sum::<u64>()
                    })
                    .min()
                    .unwrap_or(0)
            })
            .sum()
    };
    let keep = if uber { 32 } else { 8 };
    // (error, flip, [color5 per subblock], [table per subblock])
    let mut best: Option<Choice> = None;
    for flip in [false, true] {
        let opaque: [Vec<[i32; 3]>; 2] = core::array::from_fn(|s| {
            subblock_pixels(flip, s)
                .iter()
                .filter(|&&i| px[i][3] >= 128)
                .map(|&i| [0, 1, 2].map(|c| i32::from(px[i][c])))
                .collect()
        });
        // Per subblock: candidates (error, color5, table) around the average, best first.
        let mut cands: [Vec<(u64, [i32; 3], usize)>; 2] = [Vec::new(), Vec::new()];
        for (s, list) in cands.iter_mut().enumerate() {
            let pts = &opaque[s];
            let n = pts.len().max(1) as i32;
            let avg: [i32; 3] = [0, 1, 2].map(|c| pts.iter().map(|p| p[c]).sum::<i32>() / n);
            for (t, &[_, b]) in MODIFIERS.iter().enumerate() {
                for shift in [0, b, -b] {
                    let center = avg.map(|v| ((v - shift).clamp(0, 255) * 31 + 127) / 255);
                    for d in 0..27 {
                        let c = [0, 1, 2]
                            .map(|k| (center[k] + (d / [1, 3, 9][k]) % 3 - 1).clamp(0, 31));
                        if !list
                            .iter()
                            .any(|e: &(u64, [i32; 3], usize)| e.1 == c && e.2 == t)
                        {
                            list.push((err_of(pts, c, t), c, t));
                        }
                    }
                }
            }
            list.sort_by_key(|e| e.0);
            list.truncate(keep);
        }
        let mut consider = |err: u64, colors: [[i32; 3]; 2], tables: [usize; 2]| {
            if best.as_ref().is_none_or(|x| err < x.0) {
                best = Some((err, flip, colors, tables));
            }
        };
        // Unconstrained pairs that happen to fit the delta range.
        for a in &cands[0] {
            for b in &cands[1] {
                if (0..3).all(|k| (-4..=3).contains(&(b.1[k] - a.1[k]))) {
                    consider(a.0 + b.0, [a.1, b.1], [a.2, b.2]);
                }
            }
        }
        // One subblock fixed to a candidate, the other searched over its whole delta box.
        for (fixed, fixed_cands) in cands.iter().enumerate() {
            let other = 1 - fixed;
            for a in fixed_cands.iter().take(keep / 2) {
                let (mut e_best, mut c_best, mut t_best) = (u64::MAX, a.1, 0);
                for t in 0..8 {
                    for d in 0..512 {
                        let delta = [d % 8 - 4, d / 8 % 8 - 4, d / 64 - 4];
                        // Delta is other - fixed for fixed = 0, fixed - other for fixed = 1.
                        let c = [0, 1, 2].map(|k| {
                            if fixed == 0 {
                                a.1[k] + delta[k]
                            } else {
                                a.1[k] - delta[k]
                            }
                        });
                        if c.iter().any(|v| !(0..=31).contains(v)) {
                            continue;
                        }
                        let e = err_of(&opaque[other], c, t);
                        if e < e_best {
                            (e_best, c_best, t_best) = (e, c, t);
                        }
                    }
                }
                let mut colors = [a.1; 2];
                let mut tables = [a.2; 2];
                colors[other] = c_best;
                tables[other] = t_best;
                consider(a.0.saturating_add(e_best), colors, tables);
            }
        }
    }
    let mut block = EtcBlock::default();
    let Some((_, flip, colors, tables)) = best else {
        return block.bytes; // unreachable: the constrained search always finds a pair
    };
    block.set_flip_bit(flip);
    let c = colors.map(|c| [c[0] as u8, c[1] as u8, c[2] as u8, 255]);
    block.set_block_color5(&c[0], &c[1]);
    block.set_diff_bit(false); // the opaque bit (set_block_color5 sets it)
    for s in 0..2 {
        block.set_inten_table(s as u32, tables[s] as u32);
        let base = colors[s].map(ext5);
        let b = MODIFIERS[tables[s]][1];
        for i in subblock_pixels(flip, s) {
            let p = px[i];
            let raw = if p[3] < 128 {
                2
            } else {
                // raw 0: base, 1: +b, 3: -b
                [(0u32, 0), (1, b), (3, -b)]
                    .iter()
                    .min_by_key(|(_, m)| {
                        (0..3)
                            .map(|k| {
                                let v = (base[k] + m).clamp(0, 255);
                                (i32::from(p[k]) - v).pow(2)
                            })
                            .sum::<i32>()
                    })
                    .map_or(0, |r| r.0)
            };
            block.set_raw_selector((i & 3) as u32, (i >> 2) as u32, raw);
        }
    }
    block.bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decoded vs source squared error of one block.
    fn sse(px: &[[u8; 4]; 16], block: &[u8; 8]) -> u64 {
        let dec = crate::decode_to_buffer(crate::Format::Etc1Rgb, block, 4, 4).unwrap_or_default();
        px.iter()
            .zip(dec.as_chunks::<4>().0)
            .map(|(a, b)| {
                (0..3)
                    .map(|c| (i64::from(a[c]) - i64::from(b[c])).pow(2) as u64)
                    .sum::<u64>()
            })
            .sum()
    }

    #[test]
    fn beats_or_matches_the_real_time_encoder() {
        let raw = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/images/raw/rgba_64x48.rgba"
        ))
        .unwrap_or_default();
        let (mut hq, mut rt) = (0u64, 0u64);
        for by in 0..12 {
            for bx in 0..16 {
                let px: [[u8; 4]; 16] = core::array::from_fn(|i| {
                    let at = ((by * 4 + i / 4) * 64 + bx * 4 + i % 4) * 4;
                    [raw[at], raw[at + 1], raw[at + 2], 255]
                });
                hq += sse(&px, &encode_block(&px, false, false));
                let mut b = [0u8; 8];
                crate::basis::ll::etc1f::pack_etc1(&mut b, &px);
                rt += sse(&px, &b);
            }
        }
        assert!(hq < rt, "hq {hq} vs etc1f {rt}");
    }

    #[test]
    fn punchthrough_blocks_decode() {
        let raw = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/images/raw/mosaic_rgba_50x38.rgba"
        ))
        .unwrap_or_default();
        let mut se = 0u64;
        let mut n = 0u64;
        let (mut se_t, mut n_t) = (0u64, 0u64);
        for by in 0..9 {
            for bx in 0..12 {
                let mut px: [[u8; 4]; 16] = core::array::from_fn(|i| {
                    let at = ((by * 4 + i / 4) * 50 + bx * 4 + i % 4) * 4;
                    [raw[at], raw[at + 1], raw[at + 2], raw[at + 3]]
                });
                // Cut some holes.
                if (bx + by) % 3 == 0 {
                    for i in [0, 5, 10, 15] {
                        px[i][3] = 0;
                    }
                }
                let b = encode_punchthrough_block(&px, false, false);
                let dec =
                    crate::decode_to_buffer(crate::Format::Etc2Rgba1, &b, 4, 4).unwrap_or_default();
                for (p, d) in px.iter().zip(dec.as_chunks::<4>().0) {
                    assert_eq!(
                        p[3] >= 128,
                        d[3] == 255,
                        "alpha {p:?} -> {d:?} block {b:02x?} px {px:?}"
                    );
                    if p[3] >= 128 {
                        let e = (0..3)
                            .map(|c| (i64::from(p[c]) - i64::from(d[c])).pow(2) as u64)
                            .sum::<u64>();
                        if px.iter().all(|q| q[3] >= 128) {
                            se += e;
                            n += 3;
                        } else {
                            se_t += e;
                            n_t += 3;
                        }
                    }
                }
            }
        }
        let db = |se: u64, n: u64| 10.0 * (255.0f64 * 255.0 / (se as f64 / n as f64)).log10();
        println!(
            "opaque blocks {:.1} dB, blocks with holes {:.1} dB",
            db(se, n),
            db(se_t, n_t)
        );
        // No T / H / planar modes yet: two-color subblock edges stay hard (differential only).
        assert!(db(se, n) > 18.0 && db(se_t, n_t) > 18.0);
    }

    #[test]
    fn solid_and_two_color_blocks() {
        let px = [[10, 200, 30, 255]; 16];
        assert!(sse(&px, &encode_block(&px, false, true)) <= 16 * 3 * 16);
        // Left half black, right half white: one subblock each with flip off.
        let px: [[u8; 4]; 16] =
            core::array::from_fn(|i| if i % 4 < 2 { [0, 0, 0, 255] } else { [255; 4] });
        let b = encode_block(&px, true, false);
        assert!(sse(&px, &b) < 16 * 3 * 64, "{}", sse(&px, &b));
    }
}
