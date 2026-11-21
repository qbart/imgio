//! Signed BC6H (`BC6H_SF16`) block encoder (own implementation): mode 11 (one region, 10-bit
//! signed endpoints, 4-bit indices), packed with the basisu `pack_bc6h_block` port.
//!
//! Work happens in the decoder's signed "unquantized" domain: a half `h` maps to
//! `sign * round(|h| * 32 / 31)` (inverse of the final `* 31 >> 5`), endpoints are quantized
//! with the decoder's signed unquantize rule, indices pick the nearest interpolant
//! (`(e0 * (64 - w) + e1 * w + 32) >> 6`). Endpoints start at the correlation-aligned
//! bounding box; higher efforts alternate index assignment and per-channel endpoint searches.

#![allow(clippy::needless_range_loop)]

use crate::basis::ll::bc6h::{Bc6hLogicalBlock, pack_bc6h_block};

/// A mode choice: (error, basisu mode index, endpoints, indices).
type Choice = (i64, u32, [[i32; 3]; 2], [u8; 16]);

const WEIGHTS: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
/// Single-region modes (basisu logical index, base bits, delta bits; 0 = direct endpoints).
const MODES: [(u32, i32, i32); 4] = [(10, 10, 0), (11, 11, 9), (12, 12, 8), (13, 16, 4)];

/// Decoder rule: signed endpoint value of `bits` bits -> unquantized.
fn unquantize(x: i32, bits: i32) -> i32 {
    if bits >= 16 {
        return x;
    }
    let a = x.abs();
    let u = if a == 0 {
        0
    } else if a >= (1 << (bits - 1)) - 1 {
        0x7FFF
    } else {
        ((a << 15) + 0x4000) >> (bits - 1)
    };
    if x < 0 { -u } else { u }
}

/// Nearest `bits`-bit signed endpoint value for unquantized `t`.
fn quantize(t: i32, bits: i32) -> i32 {
    let lim = (1 << (bits - 1)) - 1;
    if bits >= 16 {
        return t.clamp(-lim, lim);
    }
    let a = t.abs();
    let guess = (i64::from(a) * i64::from(lim) / 0x7FFF) as i32;
    let best = (guess - 1..=guess + 1)
        .map(|x| x.clamp(0, lim))
        .min_by_key(|&x| (unquantize(x, bits) - a).abs())
        .unwrap_or(0);
    if t < 0 { -best } else { best }
}

/// Half bits -> signed unquantized domain.
fn to_domain(h: u16) -> i32 {
    let mag = i32::from(h & 0x7FFF).min(0x7BFF);
    let v = ((mag * 32 + 15) / 31).min(0x7FFF);
    if h & 0x8000 != 0 { -v } else { v }
}

fn interp(e0: i32, e1: i32, w: usize) -> i32 {
    (e0 * (64 - WEIGHTS[w]) + e1 * WEIGHTS[w] + 32) >> 6
}

/// Encodes 16 RGB half texels (row major, 48 halves) to a signed BC6H block. `effort` 0 =
/// bounding box per mode, 1-2 = refinement passes with wider endpoint searches.
pub(crate) fn encode_block(px: &[u16; 48], effort: u32) -> [u8; 16] {
    let t: [[i32; 3]; 16] = core::array::from_fn(|i| [0, 1, 2].map(|c| to_domain(px[i * 3 + c])));
    // Bounding box, low / high swapped on channels anti-correlated with the widest one.
    let range = |c: usize| {
        let v = t.iter().map(|p| p[c]);
        (v.clone().min().unwrap_or(0), v.max().unwrap_or(0))
    };
    let widest = (0..3).max_by_key(|&c| range(c).1 - range(c).0).unwrap_or(0);
    let mean = |c: usize| t.iter().map(|p| i64::from(p[c])).sum::<i64>() / 16;
    let cov = |c: usize| {
        t.iter()
            .map(|p| (i64::from(p[c]) - mean(c)) * (i64::from(p[widest]) - mean(widest)))
            .sum::<i64>()
    };
    let bbox: [[i32; 3]; 2] = {
        let mut e = [[0; 3]; 2];
        for c in 0..3 {
            let (lo, hi) = range(c);
            (e[0][c], e[1][c]) = if cov(c) < 0 { (hi, lo) } else { (lo, hi) };
        }
        e
    };
    // (error, mode, endpoints, weights)
    let mut best: Option<Choice> = None;
    for &(mode, bits, dbits) in &MODES {
        let lim = (1 << (bits - 1)) - 1;
        let fits = |e: &[[i32; 3]; 2]| {
            dbits == 0
                || (0..3).all(|c| {
                    let d = e[1][c] - e[0][c];
                    (-(1 << (dbits - 1))..(1 << (dbits - 1))).contains(&d)
                })
        };
        let assign = |e: &[[i32; 3]; 2]| -> ([u8; 16], i64) {
            let u = [0, 1].map(|k| [0, 1, 2].map(|c| unquantize(e[k][c], bits)));
            let mut w = [0u8; 16];
            let mut total = 0i64;
            for (i, p) in t.iter().enumerate() {
                let (k, err) = (0..16)
                    .map(|k| {
                        let err: i64 = (0..3)
                            .map(|c| {
                                let d = i64::from(interp(u[0][c], u[1][c], k) - p[c]);
                                d * d
                            })
                            .sum();
                        (k, err)
                    })
                    .min_by_key(|x| x.1)
                    .unwrap_or((0, 0));
                w[i] = k as u8;
                total += err;
            }
            (w, total)
        };
        let mut e = bbox.map(|p| p.map(|v| quantize(v, bits)));
        if !fits(&e) {
            // Pull the second endpoint into the delta range of the first.
            let half = 1 << (dbits - 1);
            for c in 0..3 {
                e[1][c] = e[1][c]
                    .clamp(e[0][c] - half, e[0][c] + half - 1)
                    .clamp(-lim, lim);
            }
        }
        let (mut w, mut err) = assign(&e);
        let radius = [0, 2, 6][effort.min(2) as usize];
        for _ in 0..effort.min(2) * 2 {
            for c in 0..3 {
                let err_c = |a: i32, b: i32| -> i64 {
                    let (ua, ub) = (unquantize(a, bits), unquantize(b, bits));
                    t.iter()
                        .zip(w)
                        .map(|(p, k)| {
                            let d = i64::from(interp(ua, ub, k as usize) - p[c]);
                            d * d
                        })
                        .sum()
                };
                let (mut ba, mut bb, mut be) = (e[0][c], e[1][c], err_c(e[0][c], e[1][c]));
                let step = if bits >= 16 { 1 << 5 } else { 1 };
                for da in -radius..=radius {
                    for db in -radius..=radius {
                        let (a, b) = (
                            (e[0][c] + da * step).clamp(-lim, lim),
                            (e[1][c] + db * step).clamp(-lim, lim),
                        );
                        if dbits != 0
                            && !(-(1 << (dbits - 1))..(1 << (dbits - 1))).contains(&(b - a))
                        {
                            continue;
                        }
                        let er = err_c(a, b);
                        if er < be {
                            (ba, bb, be) = (a, b, er);
                        }
                    }
                }
                e[0][c] = ba;
                e[1][c] = bb;
            }
            let (nw, total) = assign(&e);
            if total >= err {
                break;
            }
            (w, err) = (nw, total);
        }
        // Anchor index 0 must have its top bit clear (swapping keeps the delta in range only
        // if its negation fits).
        if w[0] >= 8 {
            let swapped = [e[1], e[0]];
            if !fits(&swapped) {
                continue;
            }
            e = swapped;
            for k in &mut w {
                *k = 15 - *k;
            }
        }
        if best.as_ref().is_none_or(|b| err < b.0) {
            best = Some((err, mode, e, w));
        }
    }
    let Some((_, mode, e, w)) = best else {
        return [0; 16]; // unreachable: mode 10 (direct endpoints) always fits
    };
    let (_, bits, dbits) = MODES
        .iter()
        .copied()
        .find(|m| m.0 == mode)
        .unwrap_or(MODES[0]);
    let mut log = Bc6hLogicalBlock {
        mode,
        ..Bc6hLogicalBlock::default()
    };
    for c in 0..3 {
        log.endpoints[c][0] = (e[0][c] & ((1 << bits) - 1)) as u32;
        log.endpoints[c][1] = if dbits == 0 {
            (e[1][c] & ((1 << bits) - 1)) as u32
        } else {
            ((e[1][c] - e[0][c]) & ((1 << dbits) - 1)) as u32
        };
    }
    log.weights = w;
    let mut out = [0u8; 16];
    pack_bc6h_block(&mut out, &log);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(block: &[u8; 16]) -> Vec<f32> {
        let d = crate::decode_to_buffer(crate::Format::Bc6hSfloat, block, 4, 4).unwrap_or_default();
        d.as_chunks::<2>()
            .0
            .iter()
            .map(|b| crate::half::to_f32(u16::from_le_bytes(*b)))
            .collect()
    }

    #[test]
    fn signed_gradients_round_trip() {
        for effort in 0..3 {
            // Negative and positive channels (BC6H interpolates in the log-like half domain, so
            // gradients that cross zero inside one block cannot be linear; tested below).
            let vals: [f32; 48] = core::array::from_fn(|i| {
                let (p, c) = (i / 3, i % 3);
                let x = (p & 3) as f32 / 3.0;
                [-3.0 + x * 2.0, -0.5 + x * 0.25, 3.0 - x * 2.0][c] * (1.0 + (p >> 2) as f32 * 0.01)
            });
            let px = vals.map(crate::half::from_f32);
            let d = decode(&encode_block(&px, effort));
            for p in 0..16 {
                for c in 0..3 {
                    let (want, got) = (vals[p * 3 + c], d[p * 4 + c]);
                    let tol = [0.12, 0.05, 0.05][effort as usize] * want.abs();
                    assert!(
                        (want - got).abs() <= tol,
                        "effort {effort} px {p} c {c}: {want} vs {got}"
                    );
                }
            }
        }
    }

    #[test]
    fn sign_crossing_keeps_signs_and_range() {
        let vals: [f32; 48] = core::array::from_fn(|i| ((i / 3) as f32 / 15.0) * 4.0 - 2.0);
        let px = vals.map(crate::half::from_f32);
        let d = decode(&encode_block(&px, 2));
        for p in 0..16 {
            let (want, got) = (vals[p * 3], d[p * 4]);
            assert!(
                (-2.1..=2.1).contains(&got) && (want.abs() < 0.3 || want.signum() == got.signum()),
                "{want} vs {got}"
            );
        }
    }

    #[test]
    fn domain_and_quantizer_invert_the_decoder() {
        for bits in [10, 11, 12, 16] {
            let lim = (1 << (bits - 1)) - 1;
            for x in -lim..=lim {
                assert_eq!(quantize(unquantize(x, bits), bits), x, "{bits} {x}");
            }
        }
        assert_eq!(to_domain(0x8000), 0);
    }
}
