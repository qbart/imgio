//! ETC2 T, H and planar mode block encoders (own implementation; bit layouts mirror the
//! decoder in `crate::etc`, which the tests use as the oracle).
//!
//! - Planar: per-channel least-squares origin / horizontal / vertical colors (RGB 676),
//!   refined by a +-1 search under the exact decode formula.
//! - T / H: the block's colors split in two clusters (2-means seeded along the widest
//!   channel), cluster means quantized to RGB444 with a +-1 search per channel, every
//!   distance; each texel takes its best paint color. With punch-through transparency, paint
//!   index 2 is the transparent texel and is never used for color.
//!
//! T, H and planar blocks are signalled by overflowing the differential R, G or B channel;
//! the free bits are chosen so the overflow always happens (both base / delta signs tried).

const DISTANCES: [i32; 8] = [3, 6, 11, 16, 23, 32, 41, 64];

type Px = [u8; 4];

fn ext4(v: i32) -> i32 {
    (v << 4) | v
}
fn ext6(v: i32) -> i32 {
    (v << 2) | (v >> 4)
}
fn ext7(v: i32) -> i32 {
    (v << 1) | (v >> 6)
}

fn put(block: &mut u64, size: u32, hi: u32, v: i32) {
    let shift = hi + 1 - size;
    let mask = ((1u64 << size) - 1) << shift;
    *block = (*block & !mask) | (((v as u64) << shift) & mask);
}

fn put_index(block: &mut u64, i: usize, index: u32) {
    let (x, y) = (i & 3, i >> 2);
    let b = x * 4 + y;
    *block |= u64::from(index & 1) << b;
    *block |= u64::from(index >> 1) << (16 + b);
}

/// Squared RGB error of `p` against `c`.
fn err(p: &Px, c: [i32; 3]) -> u64 {
    (0..3)
        .map(|k| {
            let d = i32::from(p[k]) - c[k];
            (d * d) as u64
        })
        .sum()
}

/// Makes `base` (5 bits at `hi`) + `delta` (3 bits at `hi - 5`) overflow, given the bits of
/// the lower 2 of base (`low2`) and lower 2 of delta (`dlow2`): bits above are free.
fn force_overflow(block: &mut u64, hi: u32, low2: i32, dlow2: i32) {
    if low2 + dlow2 >= 4 {
        // base = 28 + low2, delta = dlow2 (>= 0): sum >= 32.
        put(block, 3, hi, 0b111);
        put(block, 1, hi - 5, 0);
    } else {
        // base = low2, delta = -4 + dlow2: sum < 0.
        put(block, 3, hi, 0);
        put(block, 1, hi - 5, 1);
    }
}

/// Sets the free top bit of a 5-bit base whose low 4 bits are `low4` so that adding the
/// 3-bit signed delta `delta` stays in [0, 31].
fn avoid_overflow(block: &mut u64, hi: u32, low4: i32, delta: i32) {
    put(block, 1, hi, i32::from(low4 + delta < 0));
}

fn sext3(v: i32) -> i32 {
    if v >= 4 { v - 8 } else { v }
}

/// Planar mode block and its error (opaque texels only; planar has no transparency).
pub(crate) fn planar(px: &[Px; 16]) -> (u64, u64) {
    let mut q = [[0i32; 3]; 3]; // [channel][o, h, v]
    for c in 0..3 {
        // Least squares for p(x, y) = o + x (h - o) / 4 + y (v - o) / 4.
        // Basis: o * (1 - x/4 - y/4) + h * x/4 + v * y/4.
        let mut ata = [[0f64; 3]; 3];
        let mut atb = [0f64; 3];
        for (i, p) in px.iter().enumerate() {
            let (x, y) = ((i & 3) as f64, (i >> 2) as f64);
            let row = [1.0 - x / 4.0 - y / 4.0, x / 4.0, y / 4.0];
            for r in 0..3 {
                for s in 0..3 {
                    ata[r][s] += row[r] * row[s];
                }
                atb[r] += row[r] * f64::from(p[c]);
            }
        }
        let sol = solve3(ata, atb);
        let bits = if c == 1 { 7 } else { 6 };
        let max = (1 << bits) - 1;
        let quant =
            |v: f64| ((v.clamp(0.0, 255.0) / 255.0 * f64::from(max)).round() as i32).clamp(0, max);
        let ext = |v: i32| if c == 1 { ext7(v) } else { ext6(v) };
        let start = sol.map(quant);
        let mut best = (u64::MAX, start);
        for d in 0..27 {
            let cand = [0, 1, 2].map(|k| (start[k] + (d / [1, 3, 9][k]) % 3 - 1).clamp(0, max));
            let [o, h, v] = cand.map(ext);
            let e: u64 = px
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let (x, y) = ((i & 3) as i32, (i >> 2) as i32);
                    let val = ((x * (h - o) + y * (v - o) + 4 * o + 2) >> 2).clamp(0, 255);
                    let dd = i32::from(p[c]) - val;
                    (dd * dd) as u64
                })
                .sum();
            if e < best.0 {
                best = (e, cand);
            }
        }
        q[c] = best.1;
    }
    let [ro, rh, rv] = q[0];
    let [go, gh, gv] = q[1];
    let [bo, bh, bv] = q[2];
    let mut b = 0u64;
    put(&mut b, 6, 62, ro);
    put(&mut b, 1, 56, go >> 6);
    put(&mut b, 6, 54, go & 63);
    put(&mut b, 1, 48, bo >> 5);
    put(&mut b, 2, 44, (bo >> 3) & 3);
    put(&mut b, 3, 41, bo & 7);
    put(&mut b, 5, 38, rh >> 1);
    put(&mut b, 1, 32, rh & 1);
    put(&mut b, 7, 31, gh);
    put(&mut b, 6, 24, bh);
    put(&mut b, 6, 18, rv);
    put(&mut b, 7, 12, gv);
    put(&mut b, 6, 5, bv);
    put(&mut b, 1, 33, 1); // diff bit
    // R: base = [b63 | ro >> 2], delta = [ro & 3, go >> 6] must not overflow.
    avoid_overflow(&mut b, 63, ro >> 2, sext3(((ro & 3) << 1) | (go >> 6)));
    // G: base = [b55 | go bits 5..2], delta = [go & 3, bo >> 5] must not overflow.
    avoid_overflow(
        &mut b,
        55,
        (go >> 2) & 15,
        sext3(((go & 3) << 1) | (bo >> 5)),
    );
    // B: base = [b47 b46 b45 | bo bits 4..3], delta = [b42 | bo bits 2..1] must overflow.
    force_overflow(&mut b, 47, (bo >> 3) & 3, (bo >> 1) & 3);
    let e = (0..16)
        .map(|i| {
            let (x, y) = ((i & 3) as i32, (i >> 2) as i32);
            let ch = |o: i32, h: i32, v: i32| {
                ((x * (h - o) + y * (v - o) + 4 * o + 2) >> 2).clamp(0, 255)
            };
            err(
                &px[i],
                [
                    ch(ext6(ro), ext6(rh), ext6(rv)),
                    ch(ext7(go), ext7(gh), ext7(gv)),
                    ch(ext6(bo), ext6(bh), ext6(bv)),
                ],
            )
        })
        .sum();
    (b, e)
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    if d.abs() < 1e-12 {
        let avg = b.iter().sum::<f64>() / 3.0;
        return [avg; 3];
    }
    core::array::from_fn(|k| {
        let mut m = a;
        for r in 0..3 {
            m[r][k] = b[r];
        }
        det(m) / d
    })
}

/// Best T or H mode block for `px` (`transparent[i]`: texel `i` is punch-through
/// transparent, encoded as paint index 2) and its error over the other texels.
pub(crate) fn t_or_h(px: &[Px; 16], transparent: &[bool; 16], effort: u32) -> (u64, u64) {
    let opaque: Vec<usize> = (0..16).filter(|&i| !transparent[i]).collect();
    // 2-means on the opaque texels, seeded with the extremes of the widest channel.
    let ch = (0..3)
        .max_by_key(|&c| {
            let v = opaque.iter().map(|&i| px[i][c]);
            v.clone().max().unwrap_or(0) - v.min().unwrap_or(0)
        })
        .unwrap_or(0);
    let lo = opaque
        .iter()
        .copied()
        .min_by_key(|&i| px[i][ch])
        .unwrap_or(0);
    let hi = opaque
        .iter()
        .copied()
        .max_by_key(|&i| px[i][ch])
        .unwrap_or(0);
    let mut centers = [lo, hi].map(|i| [0, 1, 2].map(|k| i32::from(px[i][k])));
    for _ in 0..4 {
        let mut groups: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        for &i in &opaque {
            let g = usize::from(err(&px[i], centers[1]) < err(&px[i], centers[0]));
            groups[g].push(i);
        }
        for (g, c) in centers.iter_mut().enumerate() {
            if let Some(n) = (!groups[g].is_empty()).then_some(groups[g].len() as i32) {
                *c = [0, 1, 2]
                    .map(|k| groups[g].iter().map(|&i| i32::from(px[i][k])).sum::<i32>() / n);
            }
        }
    }
    let q4 = |v: i32| ((v.clamp(0, 255) * 15 + 127) / 255).clamp(0, 15);
    let around = |c: [i32; 3], wide: bool| -> Vec<[i32; 3]> {
        let s = c.map(q4);
        let r = if wide { 1 } else { 0 };
        let mut out = Vec::new();
        for dr in -r..=r {
            for dg in -r..=r {
                for db in -r..=r {
                    out.push([
                        (s[0] + dr).clamp(0, 15),
                        (s[1] + dg).clamp(0, 15),
                        (s[2] + db).clamp(0, 15),
                    ]);
                }
            }
        }
        out
    };
    let wide = effort > 0;
    let cost = |paint: &[[i32; 3]; 4]| -> (u64, [u32; 16]) {
        let mut idx = [0u32; 16];
        let mut e = 0u64;
        for i in 0..16 {
            if transparent[i] {
                idx[i] = 2;
                continue;
            }
            let (k, ek) = (0..4u32)
                .filter(|&k| !(k == 2 && transparent.iter().any(|&t| t)))
                .map(|k| (k, err(&px[i], paint[k as usize])))
                .min_by_key(|t| t.1)
                .unwrap_or((0, 0));
            idx[i] = k;
            e += ek;
        }
        (e, idx)
    };
    let add = |c: [i32; 3], d: i32| c.map(|v| (v + d).clamp(0, 255));
    let mut best: (u64, u64) = (u64::MAX, 0);
    // T mode: one group is the single color (paint 0), the other gets c2 and c2 +- d.
    for single in 0..2 {
        for c1 in around(centers[single], wide) {
            for c2 in around(centers[1 - single], wide) {
                for (d, &dist) in DISTANCES.iter().enumerate() {
                    let (e1, e2) = (c1.map(ext4), c2.map(ext4));
                    let paint = [e1, add(e2, dist), e2, add(e2, -dist)];
                    let (e, idx) = cost(&paint);
                    if e < best.0 {
                        let mut b = 0u64;
                        put(&mut b, 2, 60, c1[0] >> 2);
                        put(&mut b, 2, 57, c1[0] & 3);
                        put(&mut b, 4, 55, c1[1]);
                        put(&mut b, 4, 51, c1[2]);
                        put(&mut b, 4, 47, c2[0]);
                        put(&mut b, 4, 43, c2[1]);
                        put(&mut b, 4, 39, c2[2]);
                        put(&mut b, 2, 35, (d >> 1) as i32);
                        put(&mut b, 1, 32, (d & 1) as i32);
                        put(&mut b, 1, 33, 1);
                        force_overflow(&mut b, 63, c1[0] >> 2, c1[0] & 3);
                        for (i, &k) in idx.iter().enumerate() {
                            put_index(&mut b, i, k);
                        }
                        best = (e, b);
                    }
                }
            }
        }
    }
    // H mode: paint c1 +- d, c2 +- d; distance LSB = (c1 >= c2 as 12-bit values).
    for c1 in around(centers[0], wide) {
        for c2 in around(centers[1], wide) {
            let v = |c: [i32; 3]| (c[0] << 8) | (c[1] << 4) | c[2];
            for (d, &dist) in DISTANCES.iter().enumerate() {
                // Order the colors so the implied LSB matches `d`.
                let (a, b2) = if (v(c1) >= v(c2)) == (d & 1 == 1) {
                    (c1, c2)
                } else {
                    (c2, c1)
                };
                if (v(a) >= v(b2)) != (d & 1 == 1) {
                    continue; // equal colors can only encode odd distances
                }
                let (e1, e2) = (a.map(ext4), b2.map(ext4));
                let paint = [add(e1, dist), add(e1, -dist), add(e2, dist), add(e2, -dist)];
                let (e, idx) = cost(&paint);
                if e < best.0 {
                    let mut b = 0u64;
                    put(&mut b, 4, 62, a[0]);
                    put(&mut b, 3, 58, a[1] >> 1);
                    put(&mut b, 1, 52, a[1] & 1);
                    put(&mut b, 1, 51, a[2] >> 3);
                    put(&mut b, 3, 49, a[2] & 7);
                    put(&mut b, 4, 46, b2[0]);
                    put(&mut b, 4, 42, b2[1]);
                    put(&mut b, 4, 38, b2[2]);
                    put(&mut b, 1, 34, (d >> 2) as i32);
                    put(&mut b, 1, 32, ((d >> 1) & 1) as i32);
                    put(&mut b, 1, 33, 1);
                    // R: base = [b63 | a.r], delta = sext3(a.g >> 1) must not overflow.
                    avoid_overflow(&mut b, 63, a[0], sext3(a[1] >> 1));
                    // G: base = [b55 b54 b53 | a.g & 1, a.b >> 3], delta = [b50 | a.b bits 2..1].
                    force_overflow(&mut b, 55, ((a[1] & 1) << 1) | (a[2] >> 3), (a[2] >> 1) & 3);
                    for (i, &k) in idx.iter().enumerate() {
                        put_index(&mut b, i, k);
                    }
                    best = (e, b);
                }
            }
        }
    }
    (best.1, best.0)
}

/// ETC1-compatible `block` (8 bytes) error under the ETC2 decoder.
pub(crate) fn block_error(block: &[u8; 8], px: &[Px; 16], format: crate::Format) -> u64 {
    let dec = crate::decode_to_buffer(format, block, 4, 4).unwrap_or_default();
    px.iter()
        .zip(dec.as_chunks::<4>().0)
        .filter(|(p, _)| format != crate::Format::Etc2Rgba1 || p[3] >= 128)
        .map(|(p, d)| err(p, [0, 1, 2].map(|k| i32::from(d[k]))))
        .sum()
}

/// Improves an ETC1-style `block` with T, H and planar candidates (opaque `format`
/// `Etc2Rgb`, or `Etc2Rgba1` honoring punch-through alpha), keeping the lowest error.
pub(crate) fn improve(
    block: [u8; 8],
    px: &[Px; 16],
    format: crate::Format,
    effort: u32,
) -> [u8; 8] {
    let punch = format == crate::Format::Etc2Rgba1;
    let transparent: [bool; 16] = core::array::from_fn(|i| punch && px[i][3] < 128);
    let any_transparent = transparent.iter().any(|&t| t);
    let mut best = (block_error(&block, px, format), block);
    let mut consider = |b: u64| {
        let mut bytes = b.to_be_bytes();
        if punch && any_transparent {
            bytes[3] &= !2; // opaque bit off
        }
        let e = block_error(&bytes, px, format);
        if e < best.0 {
            best = (e, bytes);
        }
    };
    if !any_transparent {
        consider(planar(px).0);
    }
    consider(t_or_h(px, &transparent, effort).0);
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(b: u64, f: crate::Format) -> Vec<u8> {
        crate::decode_to_buffer(f, &b.to_be_bytes(), 4, 4)
            .unwrap_or_default()
            .to_vec()
    }

    #[test]
    fn planar_reproduces_gradients() {
        let px: [Px; 16] = core::array::from_fn(|i| {
            let (x, y) = ((i & 3) as u8, (i >> 2) as u8);
            [10 + x * 40, 200 - y * 30, 50 + x * 10 + y * 20, 255]
        });
        let (b, e) = planar(&px);
        let d = decode(b, crate::Format::Etc2Rgb);
        let real: u64 = px
            .iter()
            .zip(d.as_chunks::<4>().0)
            .map(|(p, q)| err(p, [0, 1, 2].map(|k| i32::from(q[k]))))
            .sum();
        assert_eq!(e, real);
        assert!(e < 16 * 3 * 9, "{e}");
    }

    #[test]
    fn t_and_h_two_color_blocks() {
        // Two flat colors split diagonally: T / H handle what ETC1 subblocks cannot.
        let px: [Px; 16] = core::array::from_fn(|i| {
            if (i & 3) + (i >> 2) < 3 {
                [230, 30, 30, 255]
            } else {
                [30, 60, 220, 255]
            }
        });
        let (b, e) = t_or_h(&px, &[false; 16], 1);
        let d = decode(b, crate::Format::Etc2Rgb);
        let real: u64 = px
            .iter()
            .zip(d.as_chunks::<4>().0)
            .map(|(p, q)| err(p, [0, 1, 2].map(|k| i32::from(q[k]))))
            .sum();
        assert_eq!(e, real, "error model must match the decoder");
        assert!(e < 16 * 3 * 64, "{e}");
        // With punch-through holes.
        let mut tr = [false; 16];
        tr[0] = true;
        tr[15] = true;
        let (b, _) = t_or_h(&px, &tr, 1);
        let mut bytes = b.to_be_bytes();
        bytes[3] &= !2;
        let d = crate::decode_to_buffer(crate::Format::Etc2Rgba1, &bytes, 4, 4).unwrap_or_default();
        for (i, q) in d.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(q[3] == 0, tr[i], "texel {i}");
        }
    }
}
