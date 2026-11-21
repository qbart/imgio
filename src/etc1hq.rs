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
    let mut block = EtcBlock::default();
    if px.iter().all(|p| p[..3] == px[0][..3]) {
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
