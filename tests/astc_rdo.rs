//! ASTC LZ rate-distortion optimization (`EncodeOptions::rdo_lambda`): the block data gets
//! more compressible by an LZ coder (measured with zstd, the coder KTX2 supercompression
//! uses) as lambda grows, at a bounded quality cost; bad lambdas are rejected and non-ASTC
//! formats ignore it. Byte identity with the C++ CLI is `tests/astcenc_cli.rs`.

use imgio::{AstcBlock, AstcKind, EncodeOptions, Error as ImgError, Format, Quality, Surface};
use std::error::Error;

const RAW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/images/raw/");

/// Integer hash of a lattice point to [0, 1].
fn hash(x: i64, y: i64, seed: i64) -> f64 {
    let mut v =
        (x.wrapping_mul(374_761_393) + y.wrapping_mul(668_265_263) + seed * 2_246_822_519) as u32;
    v = (v ^ (v >> 13)).wrapping_mul(1_274_126_177);
    f64::from((v ^ (v >> 16)) & 0xffff) / 65535.0
}

/// Smoothstep-interpolated value noise with lattice spacing `cell`.
fn value_noise(x: i64, y: i64, cell: i64, seed: i64) -> f64 {
    let (gx, gy) = (x / cell, y / cell);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let fx = s((x % cell) as f64 / cell as f64);
    let fy = s((y % cell) as f64 / cell as f64);
    let top = hash(gx, gy, seed) * (1.0 - fx) + hash(gx + 1, gy, seed) * fx;
    let bottom = hash(gx, gy + 1, seed) * (1.0 - fx) + hash(gx + 1, gy + 1, seed) * fx;
    top * (1.0 - fy) + bottom * fy
}

/// A procedural photo-like RGBA8 image: multi-octave luminance texture (shared by the
/// channels, as in natural images) times a slowly varying color tint.
fn textured(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 4);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let lum = [(0.6, 64), (0.25, 16), (0.1, 4), (0.05, 2)]
                .iter()
                .enumerate()
                .map(|(i, &(amp, cell))| amp * value_noise(x, y, cell, 7 + i as i64))
                .sum::<f64>();
            for c in 0..3 {
                let tint = 0.6 + 0.4 * value_noise(x, y, 96, c);
                out.push((lum * tint * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
            }
            out.push(255);
        }
    }
    out
}

/// zstd level 9 compressed size.
fn zstd_size(data: &[u8]) -> usize {
    use zstd_pure_rs::prelude::{ZSTD_compress, ZSTD_compressBound, ZSTD_isError};
    let mut buf = vec![0u8; ZSTD_compressBound(data.len())];
    let n = ZSTD_compress(&mut buf, data, 9);
    assert!(!ZSTD_isError(n), "zstd error");
    n
}

/// RGB PSNR in dB.
fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    let (mut se, mut n) = (0.0f64, 0usize);
    for (pa, pb) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        for c in 0..3 {
            let d = f64::from(pa[c]) - f64::from(pb[c]);
            se += d * d;
            n += 1;
        }
    }
    let mse = se / n as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

#[test]
fn rdo_lambda_trades_quality_for_lz_size() -> Result<(), Box<dyn Error>> {
    let (w, h) = (256, 256);
    let data = textured(w as usize, h as usize);
    let surface = Surface::new(Format::Rgba8Unorm, w, h, &data)?;
    for (block, kind) in [
        (AstcBlock::B4x4, AstcKind::Unorm),
        (AstcBlock::B6x6, AstcKind::Srgb),
    ] {
        let format = Format::Astc(block, kind);
        let mut prev: Option<(f32, usize, f64)> = None;
        let mut base_psnr = 0.0;
        for rdo_lambda in [0.0, 10.0, 20.0, 40.0] {
            let opts = EncodeOptions {
                quality: Quality::Default,
                rdo_lambda,
            };
            let enc = imgio::encode_to_buffer(&surface, format, &opts)?;
            assert_eq!(enc.len(), format.surface_bytes(w, h)?);
            let size = zstd_size(&enc);
            let dec = imgio::decode_to_buffer(format, &enc, w, h)?;
            let psnr = psnr_rgb(&data, &dec);
            println!("{block:?} {kind:?} lambda {rdo_lambda}: zstd {size} bytes, {psnr:.2} dB");
            if rdo_lambda == 0.0 {
                base_psnr = psnr;
            }
            if let Some((pl, ps, pp)) = prev {
                assert!(
                    size < ps,
                    "{block:?}: lambda {rdo_lambda} zstd size {size} >= {ps} at lambda {pl}"
                );
                assert!(
                    psnr <= pp + 0.05,
                    "{block:?}: lambda {rdo_lambda} PSNR {psnr} above {pp} at lambda {pl}"
                );
            }
            // The quality cost stays bounded
            assert!(
                psnr > base_psnr - 0.2 * f64::from(rdo_lambda) - 0.1,
                "{block:?}: lambda {rdo_lambda} PSNR {psnr:.2} too far below {base_psnr:.2}"
            );
            prev = Some((rdo_lambda, size, psnr));
        }
    }
    Ok(())
}

#[test]
fn rdo_lambda_validation() -> Result<(), Box<dyn Error>> {
    let data = std::fs::read(format!("{RAW}rgba_64x48.rgba"))?;
    let surface = Surface::new(Format::Rgba8Unorm, 64, 48, &data)?;
    let astc = Format::Astc(AstcBlock::B4x4, AstcKind::Unorm);
    for bad in [-1.0, f32::NAN] {
        let opts = EncodeOptions {
            rdo_lambda: bad,
            ..Default::default()
        };
        assert!(matches!(
            imgio::encode_to_buffer(&surface, astc, &opts),
            Err(ImgError::InvalidArgument)
        ));
    }
    // Other formats (ASTC HDR included) ignore the lambda
    let rdo = EncodeOptions {
        rdo_lambda: 20.0,
        ..Default::default()
    };
    for format in [
        Format::Bc7Unorm,
        Format::Bc1Unorm,
        Format::Astc(AstcBlock::B4x4, AstcKind::Float),
    ] {
        let plain = imgio::encode_to_buffer(&surface, format, &EncodeOptions::default())?;
        let with_rdo = imgio::encode_to_buffer(&surface, format, &rdo)?;
        assert_eq!(&with_rdo[..], &plain[..], "{format:?}");
    }
    // And LDR ASTC uses it
    let with_rdo = imgio::encode_to_buffer(&surface, astc, &rdo)?;
    let plain = imgio::encode_to_buffer(&surface, astc, &EncodeOptions::default())?;
    assert_ne!(&with_rdo[..], &plain[..]);
    Ok(())
}
