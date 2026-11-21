//! Encoders for the legacy mobile formats Basis Universal only produces by transcoding (own
//! implementations, checked against imgio's decoders):
//!
//! - ATC RGB (mode 0: color0 RGB555, color1 RGB565, interpolants 5/8 and 3/8) and ATC RGBA
//!   (BC4 alpha + ATC color).
//! - PVRTC2 4bpp in hard-transition, non-interpolated blocks (the mode imgio decodes): opaque
//!   (RGB554 / RGB555) or translucent (ARGB3443 / ARGB3444), interpolants 5/8 and 3/8.
//! - FXT1 CC_MIXED: each 4x4 half BC1-encoded then converted (basisu `convert_dxt1_to_fxt1`).
//!
//! Two-endpoint fit: principal axis through the texels (power iteration), extremes quantized
//! to the format's precision in both endpoint orders; indices pick the nearest decoded palette
//! entry; from effort 1 least-squares endpoint refits, at effort 2 a +-1 search per channel.

type Px = [u8; 4];

/// Weights (out of 8) of palette entries 0..3 between endpoint a (0) and b (3).
const W8: [i32; 4] = [0, 3, 5, 8];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Atc,
    Pvrtc2Opaque,
    Pvrtc2Translucent,
}

/// Bits per channel (r, g, b, a) of endpoint 0 and 1.
fn precision(k: Kind) -> [[u32; 4]; 2] {
    match k {
        Kind::Atc => [[5, 5, 5, 0], [5, 6, 5, 0]],
        Kind::Pvrtc2Opaque => [[5, 5, 4, 0], [5, 5, 5, 0]],
        Kind::Pvrtc2Translucent => [[4, 4, 3, 3], [4, 4, 4, 3]],
    }
}

fn ext(v: i32, from: u32, to: u32) -> i32 {
    // Bit replication from `from` to `to` bits.
    let mut out = 0;
    let mut have = 0;
    while have < to {
        let take = from.min(to - have);
        out = (out << take) | (v >> (from - take));
        have += take;
    }
    out
}

/// Decoded 8-bit palette (RGBA) of quantized endpoints `q` (as in imgio's decoders).
fn palette(k: Kind, q: [[i32; 4]; 2]) -> [[i32; 4]; 4] {
    let (a, b) = match k {
        Kind::Atc => (
            [
                ext(q[0][0], 5, 8),
                ext(q[0][1], 5, 8),
                ext(q[0][2], 5, 8),
                255,
            ],
            [
                ext(q[1][0], 5, 8),
                ext(q[1][1], 6, 8),
                ext(q[1][2], 5, 8),
                255,
            ],
        ),
        Kind::Pvrtc2Opaque => {
            let b4 = (q[0][2] << 1) | (q[0][2] >> 3);
            (
                [ext(q[0][0], 5, 8), ext(q[0][1], 5, 8), ext(b4, 5, 8), 255],
                [
                    ext(q[1][0], 5, 8),
                    ext(q[1][1], 5, 8),
                    ext(q[1][2], 5, 8),
                    255,
                ],
            )
        }
        Kind::Pvrtc2Translucent => {
            let x5 = |v: i32| (v << 1) | (v >> 3);
            let a5 = [x5(q[0][0]), x5(q[0][1]), (q[0][2] << 2) | (q[0][2] >> 1)];
            let b5 = [x5(q[1][0]), x5(q[1][1]), x5(q[1][2])];
            let (aa, ba) = (q[0][3] << 1, (q[1][3] << 1) | 1);
            (
                [
                    ext(a5[0], 5, 8),
                    ext(a5[1], 5, 8),
                    ext(a5[2], 5, 8),
                    (aa << 4) | aa,
                ],
                [
                    ext(b5[0], 5, 8),
                    ext(b5[1], 5, 8),
                    ext(b5[2], 5, 8),
                    (ba << 4) | ba,
                ],
            )
        }
    };
    let mix = |wa: i32, wb: i32| -> [i32; 4] {
        if k == Kind::Atc {
            let c = [0, 1, 2].map(|i| (a[i] * wa + b[i] * wb) >> 3);
            [c[0], c[1], c[2], 255]
        } else {
            [0, 1, 2, 3].map(|i| (a[i] * wa + b[i] * wb) / 8)
        }
    };
    [a, mix(5, 3), mix(3, 5), b]
}

fn dist(p: &Px, c: &[i32; 4], alpha: bool) -> i64 {
    (0..if alpha { 4 } else { 3 })
        .map(|i| {
            let d = i64::from(p[i]) - i64::from(c[i]);
            d * d
        })
        .sum()
}

/// Nearest palette indices and the total error.
fn assign(px: &[Px; 16], pal: &[[i32; 4]; 4], alpha: bool) -> ([u8; 16], i64) {
    let mut idx = [0u8; 16];
    let mut total = 0;
    for (i, p) in px.iter().enumerate() {
        let (k, e) = (0..4)
            .map(|k| (k, dist(p, &pal[k], alpha)))
            .min_by_key(|t| t.1)
            .unwrap_or((0, 0));
        idx[i] = k as u8;
        total += e;
    }
    (idx, total)
}

fn quantize(v: f32, bits: u32) -> i32 {
    if bits == 0 {
        return 0;
    }
    let max = ((1 << bits) - 1) as f32;
    (v.clamp(0.0, 255.0) / 255.0 * max).round() as i32
}

/// A fit: quantized endpoints, indices, error.
type Fit = ([[i32; 4]; 2], [u8; 16], i64);

/// Best quantized endpoints and indices for `px` (channels: RGB, or RGBA with `alpha`).
fn fit(px: &[Px; 16], k: Kind, effort: u32) -> ([[i32; 4]; 2], [u8; 16]) {
    let alpha = k == Kind::Pvrtc2Translucent;
    let n = if alpha { 4 } else { 3 };
    let f: Vec<[f32; 4]> = px.iter().map(|p| p.map(f32::from)).collect();
    let mean: [f32; 4] = core::array::from_fn(|c| f.iter().map(|p| p[c]).sum::<f32>() / 16.0);
    // Principal axis by power iteration on the covariance.
    let mut cov = [[0f32; 4]; 4];
    for p in &f {
        for r in 0..n {
            for s in 0..n {
                cov[r][s] += (p[r] - mean[r]) * (p[s] - mean[s]);
            }
        }
    }
    let mut axis = [1f32, 1.0, 1.0, if alpha { 1.0 } else { 0.0 }];
    for _ in 0..8 {
        let mut next = [0f32; 4];
        for r in 0..n {
            next[r] = (0..n).map(|s| cov[r][s] * axis[s]).sum();
        }
        let len = next.iter().map(|v| v * v).sum::<f32>().sqrt();
        if len < 1e-6 {
            break;
        }
        axis = next.map(|v| v / len);
    }
    let proj: Vec<f32> = f
        .iter()
        .map(|p| (0..n).map(|c| (p[c] - mean[c]) * axis[c]).sum())
        .collect();
    let (lo, hi) = proj
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let ends = [lo, hi].map(|t| core::array::from_fn::<f32, 4, _>(|c| mean[c] + axis[c] * t));
    let prec = precision(k);
    let quant = |e: [[f32; 4]; 2]| -> [[i32; 4]; 2] {
        core::array::from_fn(|j| core::array::from_fn(|c| quantize(e[j][c], prec[j][c])))
    };
    let mut best: Option<Fit> = None;
    let consider = |q: [[i32; 4]; 2], best: &mut Option<Fit>| {
        let (idx, e) = assign(px, &palette(k, q), alpha);
        if best.as_ref().is_none_or(|b| e < b.2) {
            *best = Some((q, idx, e));
        }
    };
    consider(quant(ends), &mut best);
    consider(quant([ends[1], ends[0]]), &mut best);
    // Least-squares refits from the current indices.
    for _ in 0..[0, 2, 3][effort.min(2) as usize] {
        let Some((_, idx, _)) = best else { break };
        let (mut aa, mut ab, mut bb) = (0f32, 0f32, 0f32);
        let (mut xa, mut xb) = ([0f32; 4], [0f32; 4]);
        for (i, p) in f.iter().enumerate() {
            let wb = W8[idx[i] as usize] as f32 / 8.0;
            let wa = 1.0 - wb;
            aa += wa * wa;
            ab += wa * wb;
            bb += wb * wb;
            for c in 0..4 {
                xa[c] += wa * p[c];
                xb[c] += wb * p[c];
            }
        }
        let det = aa * bb - ab * ab;
        if det.abs() < 1e-6 {
            break;
        }
        let e: [[f32; 4]; 2] = [
            core::array::from_fn(|c| (xa[c] * bb - xb[c] * ab) / det),
            core::array::from_fn(|c| (xb[c] * aa - xa[c] * ab) / det),
        ];
        consider(quant(e), &mut best);
    }
    // +-1 per channel and endpoint.
    if effort >= 2 {
        for _ in 0..2 {
            let Some((q0, _, _)) = best else { break };
            for j in 0..2 {
                for c in 0..n {
                    for d in [-1, 1] {
                        let mut q = q0;
                        let max = (1 << prec[j][c]) - 1;
                        q[j][c] = (q[j][c] + d).clamp(0, max);
                        consider(q, &mut best);
                    }
                }
            }
        }
    }
    best.map_or(([[0; 4]; 2], [0; 16]), |b| (b.0, b.1))
}

/// ATC RGB color block (8 bytes).
pub(crate) fn atc_block(px: &[Px; 16], effort: u32) -> [u8; 8] {
    let (q, idx) = fit(px, Kind::Atc, effort);
    let c0 = (q[0][0] << 10) | (q[0][1] << 5) | q[0][2]; // bit 15 clear: mode 0
    let c1 = (q[1][0] << 11) | (q[1][1] << 5) | q[1][2];
    let sels = idx
        .iter()
        .enumerate()
        .fold(0u32, |s, (i, &k)| s | (u32::from(k) << (i * 2)));
    let mut out = [0u8; 8];
    out[..2].copy_from_slice(&(c0 as u16).to_le_bytes());
    out[2..4].copy_from_slice(&(c1 as u16).to_le_bytes());
    out[4..].copy_from_slice(&sels.to_le_bytes());
    out
}

/// PVRTC2 4bpp hard-transition block (8 bytes); translucent when any alpha < 255.
pub(crate) fn pvrtc2_block(px: &[Px; 16], effort: u32) -> [u8; 8] {
    let translucent = px.iter().any(|p| p[3] < 255);
    let k = if translucent {
        Kind::Pvrtc2Translucent
    } else {
        Kind::Pvrtc2Opaque
    };
    let (q, idx) = fit(px, k, effort);
    let mut d = 1u32 << 15; // hard flag; modulation flag (bit 0) clear
    if translucent {
        d |= (q[0][0] as u32) << 8
            | (q[0][1] as u32) << 4
            | (q[0][2] as u32) << 1
            | (q[0][3] as u32) << 12;
        d |= (q[1][0] as u32) << 24
            | (q[1][1] as u32) << 20
            | (q[1][2] as u32) << 16
            | (q[1][3] as u32) << 28;
    } else {
        d |= 1 << 31;
        d |= (q[0][0] as u32) << 10 | (q[0][1] as u32) << 5 | (q[0][2] as u32) << 1;
        d |= (q[1][0] as u32) << 26 | (q[1][1] as u32) << 21 | (q[1][2] as u32) << 16;
    }
    let mut out = [0u8; 8];
    for (i, &k) in idx.iter().enumerate() {
        out[i >> 2] |= k << ((i & 3) * 2);
    }
    out[4..].copy_from_slice(&d.to_le_bytes());
    out
}

/// FXT1 CC_MIXED block (16 bytes) for an 8x4 texel block (`px` row major, 8 wide).
pub(crate) fn fxt1_block(px: &[Px; 32], bc1_flags: u32) -> [u8; 16] {
    use crate::basis::ll::dxt::Dxt1Block;
    use crate::basis::ll::etc::Color32;
    let mut out = [0u8; 16];
    for half in 0..2 {
        let colors: [Color32; 16] = core::array::from_fn(|i| Color32 {
            c: px[(i >> 2) * 8 + half * 4 + (i & 3)],
        });
        let mut d = Dxt1Block::default();
        crate::basis::ll::uastc_bc1::encode_bc1(&mut d, &colors, bc1_flags);
        crate::basis::ll::fxt1::convert_dxt1_to_fxt1(&mut out, &d, half as u32);
    }
    out
}

#[cfg(test)]
mod tests {
    fn psnr(a: &[u8], b: &[u8], channels: usize) -> f64 {
        let mut se = 0f64;
        let mut n = 0f64;
        for (p, q) in a.chunks(4).zip(b.chunks(4)) {
            for c in 0..channels {
                se += (f64::from(p[c]) - f64::from(q[c])).powi(2);
                n += 1.0;
            }
        }
        10.0 * (65025.0 / (se / n).max(1e-9)).log10()
    }

    fn image() -> (u32, u32, Vec<u8>) {
        let raw = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/images/raw/rgba_64x48.rgba"
        ))
        .unwrap_or_default();
        (64, 48, raw)
    }

    #[test]
    fn decode_round_trips() {
        let (w, h, raw) = image();
        let opaque: Vec<u8> = raw
            .chunks(4)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect();
        for (f, src, channels, min_db) in [
            (crate::Format::AtcRgb, &opaque, 3, 29.0),
            (crate::Format::AtcRgba, &raw, 4, 30.0),
            (crate::Format::Pvrtc2Rgb, &opaque, 3, 29.0),
            (crate::Format::Pvrtc2Rgba, &raw, 4, 27.0),
            (crate::Format::Fxt1Rgb, &opaque, 3, 29.0),
        ] {
            for q in [crate::Quality::Fastest, crate::Quality::Best] {
                let s = crate::Surface::new(crate::Format::Rgba8Unorm, w, h, src)
                    .unwrap_or_else(|_| std::process::abort());
                let enc = crate::encode_to_buffer(
                    &s,
                    f,
                    &crate::EncodeOptions {
                        quality: q,
                        ..Default::default()
                    },
                )
                .unwrap_or_default();
                let dec = crate::decode_to_buffer(f, &enc, w, h).unwrap_or_default();
                let db = psnr(src, &dec, channels);
                assert!(db > min_db, "{f:?} {q:?}: {db:.1} dB");
            }
        }
    }
}
