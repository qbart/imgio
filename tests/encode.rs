//! Block encoders vs the C++ Basis Universal encoders (testdata/encode/golden.txt from
//! tools/basisu-golden/encode.cpp on testdata/images/raw), plus round trips.

use imgio::{EncodeOptions, Format, Quality, Surface};
use std::error::Error;

const RAW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/images/raw/");

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| {
        (h ^ u64::from(x)).wrapping_mul(0x100000001b3)
    })
}

/// Raw image `name` (NAME_WxH.rgba) -> (width, height, RGBA8 pixels).
fn raw(name: &str) -> Result<(u32, u32, Vec<u8>), Box<dyn Error>> {
    let stem = name.split('.').next().ok_or("name")?;
    let dims = stem.rsplit('_').next().ok_or("name")?;
    let (w, h) = dims.split_once('x').ok_or("dims")?;
    Ok((
        w.parse()?,
        h.parse()?,
        std::fs::read(format!("{RAW}{name}"))?,
    ))
}

#[test]
fn matches_basisu() -> Result<(), Box<dyn Error>> {
    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/encode/golden.txt"
    ))?;
    let mut n = 0;
    for line in golden.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let (w, h, px) = raw(f[0])?;
        let format = match f[1] {
            "bc1" => Format::Bc1Unorm,
            "bc3" => Format::Bc3Unorm,
            "bc4" => Format::Bc4Unorm,
            "bc5" => Format::Bc5Unorm,
            "bc7" => Format::Bc7Unorm,
            "etc1" => Format::Etc1Rgb,
            "etc2rgba" | "etc2rgba_eac" => Format::Etc2Rgba,
            "eacr11" => Format::EacR11Unorm,
            "eacrg11" => Format::EacRg11Unorm,
            "pvrtc1rgb" => Format::Pvrtc1Rgb,
            "pvrtc1rgba" => Format::Pvrtc1Rgba,
            _ => Format::Bc6hUfloat,
        };
        let all = [
            Quality::Fastest,
            Quality::Fast,
            Quality::Default,
            Quality::High,
            Quality::Best,
        ];
        let quality = match (f[1], f[2].parse::<usize>()?) {
            // encode_bc1 flags 0 / 1 / 2.
            ("bc1" | "bc3", q) => [Quality::Fast, Quality::Default, Quality::High][q],
            // pack_eac / pack_eac_high_quality.
            ("etc1" | "etc2rgba" | "etc2rgba_eac" | "eacr11" | "eacrg11", q) => {
                [Quality::Default, Quality::High][q]
            }
            (_, q) => all[q],
        };
        let src_format = if f[0].ends_with(".rgba16f") {
            Format::Rgba16Float
        } else {
            Format::Rgba8Unorm
        };
        // PVRTC1: the top-left power-of-two crop.
        let (w, h, px) = if f[1].starts_with("pvrtc") {
            let (pw, ph) = (1 << w.ilog2(), 1 << h.ilog2());
            let crop: Vec<u8> = (0..ph as usize)
                .flat_map(|y| px[y * w as usize * 4..(y * w as usize + pw as usize) * 4].to_vec())
                .collect();
            (pw, ph, crop)
        } else {
            (w, h, px)
        };
        let src = Surface::new(src_format, w, h, &px)?;
        let mut out = imgio::encode_to_buffer(&src, format, &EncodeOptions { quality })?.to_vec();
        if f[1] == "etc2rgba_eac" {
            // High also changes the ETC1 color halves: compare the EAC alpha halves only.
            out = out
                .as_chunks::<16>()
                .0
                .iter()
                .flat_map(|b| b[..8].to_vec())
                .collect();
        }
        assert_eq!(format!("{:016x}", fnv(&out)), f[3], "{line}");
        n += 1;
    }
    assert_eq!(n, 65);
    Ok(())
}

/// BC7 High / Best = basisu bc7e_scalar slow / slowest (perceptual for sRGB), vs the bc7e
/// golden (testdata/encode/bc7e_golden.txt, tools/basisu-golden/bc7e.cpp).
#[test]
fn bc7_high_best_match_bc7e() -> Result<(), Box<dyn Error>> {
    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/encode/bc7e_golden.txt"
    ))?;
    let mut n = 0;
    for line in golden.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let quality = match f[1] {
            "slow" => Quality::High,
            "slowest" => Quality::Best,
            _ => continue,
        };
        let format = if f[2] == "1" {
            Format::Bc7Srgb
        } else {
            Format::Bc7Unorm
        };
        let (w, h, px) = raw(f[0])?;
        let src = Surface::new(Format::Rgba8Unorm, w, h, &px)?;
        let out = imgio::encode_to_buffer(&src, format, &EncodeOptions { quality })?;
        assert_eq!(format!("{:016x}", fnv(&out)), f[3], "{line}");
        n += 1;
    }
    assert_eq!(n, 12);
    Ok(())
}

/// Peak signal-to-noise ratio (dB) of `a` vs `b` over the given channels.
fn psnr(a: &[u8], b: &[u8], channels: &[usize]) -> f64 {
    let mut se = 0f64;
    let mut count = 0f64;
    for (p, q) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        for &c in channels {
            let d = f64::from(p[c]) - f64::from(q[c]);
            se += d * d;
            count += 1.0;
        }
    }
    10.0 * (255.0f64 * 255.0 / (se / count).max(1e-9)).log10()
}

#[test]
fn round_trips() -> Result<(), Box<dyn Error>> {
    let (w, h, px) = raw("rgba_64x48.rgba")?;
    let src = Surface::new(Format::Rgba8Unorm, w, h, &px)?;
    for (format, channels, min_db) in [
        (Format::Bc1Unorm, &[0, 1, 2][..], 30.0),
        (Format::Bc2Unorm, &[0, 1, 2, 3][..], 30.0),
        (Format::Bc3Unorm, &[0, 1, 2, 3][..], 30.0),
        (Format::Bc4Unorm, &[0][..], 40.0),
        (Format::Etc1Rgb, &[0, 1, 2][..], 26.0),
        (Format::Etc2Rgba, &[0, 1, 2, 3][..], 26.0),
    ] {
        let enc = imgio::encode_to_buffer(&src, format, &EncodeOptions::default())?;
        let dec = imgio::decode_to_buffer(format, &enc, w, h)?;
        let dec = if format == Format::Bc4Unorm {
            dec.iter()
                .flat_map(|&r| [r, 0, 0, 255])
                .collect::<Vec<u8>>()
        } else {
            dec.to_vec()
        };
        let db = psnr(&px, &dec, channels);
        assert!(db > min_db, "{format:?}: {db:.1} dB");
    }
    // ETC2 RGB with T / H / planar candidates (High) beats the best ETC1 block.
    let etc = |f: Format, q: Quality| -> Result<f64, Box<dyn Error>> {
        let enc = imgio::encode_to_buffer(&src, f, &EncodeOptions { quality: q })?;
        let dec = imgio::decode_to_buffer(f, &enc, w, h)?;
        Ok(psnr(&px, &dec, &[0, 1, 2]))
    };
    let (etc1_best, etc2_high) = (
        etc(Format::Etc1Rgb, Quality::Best)?,
        etc(Format::Etc2Rgb, Quality::High)?,
    );
    assert!(
        etc2_high > etc1_best + 2.0,
        "etc2 {etc2_high:.2} vs etc1 {etc1_best:.2}"
    );
    // ETC2 RGB A1: alpha < 128 decodes transparent, the rest opaque.
    for q in [Quality::Default, Quality::Best] {
        let enc = imgio::encode_to_buffer(&src, Format::Etc2Rgba1, &EncodeOptions { quality: q })?;
        let dec = imgio::decode_to_buffer(Format::Etc2Rgba1, &enc, w, h)?;
        for (p, d) in px.as_chunks::<4>().0.iter().zip(dec.as_chunks::<4>().0) {
            assert_eq!(d[3], if p[3] >= 128 { 255 } else { 0 });
        }
    }
    // Signed targets: a [-1, 1] ramp (RG16 snorm source) through BC4 / BC5 / EAC snorm.
    let ramp: Vec<u8> = (0..64 * 4)
        .flat_map(|i| {
            let (x, y) = (i % 64, i / 64);
            let r = (x as f32 / 63.0 * 2.0 - 1.0) * 32767.0;
            let g = (x + y * 3) as f32 / 72.0 * 2.0 - 1.0;
            let mut v = (r.round() as i16).to_le_bytes().to_vec();
            v.extend_from_slice(&((g * 32767.0).round() as i16).to_le_bytes());
            v
        })
        .collect();
    let ssrc = Surface::new(Format::Rg16Snorm, 64, 4, &ramp)?;
    for f in [
        Format::Bc4Snorm,
        Format::Bc5Snorm,
        Format::EacR11Snorm,
        Format::EacRg11Snorm,
    ] {
        let enc = imgio::encode_to_buffer(&ssrc, f, &EncodeOptions::default())?;
        let dec = imgio::decode_to_buffer(f, &enc, 64, 4)?;
        let out = Surface::new(f.decoded(), 64, 4, &dec)?;
        let a = imgio::convert::convert_to_buffer(&ssrc, Format::Rgba32Float)?;
        let b = imgio::convert::convert_to_buffer(&out, Format::Rgba32Float)?;
        let channels = if matches!(f, Format::Bc4Snorm | Format::EacR11Snorm) {
            1
        } else {
            2
        };
        let fl = |v: &[u8], i: usize| {
            f32::from_le_bytes([v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3]])
        };
        let mut max = 0f32;
        for p in 0..256 {
            for c in 0..channels {
                max = max.max((fl(&a, p * 4 + c) - fl(&b, p * 4 + c)).abs());
            }
        }
        assert!(max < 0.08, "{f:?}: max error {max}");
    }
    // ETC1: the optimizer search (High / Best) beats etc1f (Default).
    let mut prev = 0.0;
    for q in [Quality::Default, Quality::High, Quality::Best] {
        let enc = imgio::encode_to_buffer(&src, Format::Etc1Rgb, &EncodeOptions { quality: q })?;
        let dec = imgio::decode_to_buffer(Format::Etc1Rgb, &enc, w, h)?;
        let db = psnr(&px, &dec, &[0, 1, 2]);
        println!("etc1 {q:?}: {db:.2} dB");
        assert!(db >= prev, "etc1 {q:?}: {db:.2} dB < {prev:.2}");
        prev = db;
    }
    // BC7 at every quality.
    for (q, min_db) in [
        (Quality::Fastest, 30.0),
        (Quality::Default, 35.0),
        (Quality::Best, 35.0),
    ] {
        let enc = imgio::encode_to_buffer(&src, Format::Bc7Unorm, &EncodeOptions { quality: q })?;
        let dec = imgio::decode_to_buffer(Format::Bc7Unorm, &enc, w, h)?;
        let db = psnr(&px, &dec, &[0, 1, 2, 3]);
        assert!(db > min_db, "bc7 {q:?}: {db:.1} dB");
    }
    // BC6H: relative error of the decoded halves.
    let (hw, hh, hpx) = raw("hdr_40x24.rgba16f")?;
    let hsrc = Surface::new(Format::Rgba16Float, hw, hh, &hpx)?;
    let enc = imgio::encode_to_buffer(&hsrc, Format::Bc6hUfloat, &EncodeOptions::default())?;
    let dec = imgio::decode_to_buffer(Format::Bc6hUfloat, &enc, hw, hh)?;
    let half =
        |b: &[u8], i: usize| imgio::half::to_f32(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]));
    let mut err = 0f64;
    for p in 0..(hw * hh) as usize {
        for c in 0..3 {
            let (a, b) = (half(&hpx, p * 4 + c), half(&dec, p * 4 + c));
            err += f64::from((a - b).abs() / a.max(0.05));
        }
    }
    let mean = err / f64::from(hw * hh * 3);
    assert!(mean < 0.1, "bc6h mean relative error {mean}");
    // Signed BC6H: the HDR image mirrored to negative on the right half.
    let neg: Vec<u8> = hpx
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(i, b)| {
            let (x, c) = ((i / 4) as u32 % hw, i % 4);
            let v = u16::from_le_bytes(*b);
            let v = if x >= hw / 2 && c < 3 { v | 0x8000 } else { v };
            v.to_le_bytes()
        })
        .collect();
    let nsrc = Surface::new(Format::Rgba16Float, hw, hh, &neg)?;
    let enc = imgio::encode_to_buffer(&nsrc, Format::Bc6hSfloat, &EncodeOptions::default())?;
    let dec = imgio::decode_to_buffer(Format::Bc6hSfloat, &enc, hw, hh)?;
    let mut err = 0f64;
    for p in 0..(hw * hh) as usize {
        for c in 0..3 {
            let (a, b) = (half(&neg, p * 4 + c), half(&dec, p * 4 + c));
            assert!(
                a.abs() < 0.05 || a.signum() == b.signum(),
                "sign {a} vs {b}"
            );
            err += f64::from((a - b).abs() / a.abs().max(0.05));
        }
    }
    let mean = err / f64::from(hw * hh * 3);
    assert!(mean < 0.1, "bc6h signed mean relative error {mean}");
    // ASTC LDR / sRGB through astcenc (multithreaded on large images: tile the source).
    let big: Vec<u8> = (0..512 * 384)
        .flat_map(|i| {
            let at = ((i / 512 % 48) * 64 + i % 512 % 64) * 4;
            [px[at], px[at + 1], px[at + 2], px[at + 3]]
        })
        .collect();
    let bsrc = Surface::new(Format::Rgba8Unorm, 512, 384, &big)?;
    for (block, kind, min_db) in [
        (imgio::AstcBlock::B4x4, imgio::AstcKind::Unorm, 34.0),
        (imgio::AstcBlock::B6x5, imgio::AstcKind::Srgb, 28.0),
        (imgio::AstcBlock::B8x8, imgio::AstcKind::Unorm, 24.0),
    ] {
        let f = Format::Astc(block, kind);
        for q in [Quality::Fastest, Quality::Default] {
            let opts = EncodeOptions { quality: q };
            for (s, (sw, sh), want) in [(&src, (w, h), &px[..]), (&bsrc, (512, 384), &big[..])] {
                let enc = imgio::encode_to_buffer(s, f, &opts)?;
                let dec = imgio::decode_to_buffer(f, &enc, sw, sh)?;
                let db = psnr(want, &dec, &[0, 1, 2, 3]);
                assert!(db > min_db, "{f:?} {q:?} {sw}x{sh}: {db:.1} dB");
            }
        }
    }
    // ASTC HDR 6x6 from the half-float image.
    let f = Format::Astc(imgio::AstcBlock::B6x6, imgio::AstcKind::Float);
    let enc = imgio::encode_to_buffer(&hsrc, f, &EncodeOptions::default())?;
    let dec = imgio::decode_to_buffer(f, &enc, hw, hh)?;
    let mut err = 0f64;
    for p in 0..(hw * hh) as usize {
        for c in 0..3 {
            let (a, b) = (half(&hpx, p * 4 + c), half(&dec, p * 4 + c));
            err += f64::from((a - b).abs() / a.max(0.05));
        }
    }
    let mean = err / f64::from(hw * hh * 3);
    assert!(mean < 0.1, "astc hdr mean relative error {mean}");
    // Any uncompressed source: RGBA32F gives the same blocks as RGBA8.
    let f32px = imgio::convert::convert_to_buffer(&src, Format::Rgba32Float)?;
    let fsrc = Surface::new(Format::Rgba32Float, w, h, &f32px)?;
    let opts = EncodeOptions::default();
    assert_eq!(
        &imgio::encode_to_buffer(&fsrc, Format::Bc3Unorm, &opts)?[..],
        &imgio::encode_to_buffer(&src, Format::Bc3Unorm, &opts)?[..]
    );
    assert!(!imgio::can_encode(Format::Rgba8Unorm));
    // Every compressed format has an encoder.
    use Format::*;
    for f in [
        Bc1Unorm,
        Bc1Srgb,
        Bc2Unorm,
        Bc2Srgb,
        Bc3Unorm,
        Bc3Srgb,
        Bc4Unorm,
        Bc4Snorm,
        Bc5Unorm,
        Bc5Snorm,
        Bc6hUfloat,
        Bc6hSfloat,
        Bc7Unorm,
        Bc7Srgb,
        Etc1Rgb,
        Etc2Rgb,
        Etc2Srgb,
        Etc2Rgba1,
        Etc2Rgba1Srgb,
        Etc2Rgba,
        Etc2RgbaSrgb,
        EacR11Unorm,
        EacR11Snorm,
        EacRg11Unorm,
        EacRg11Snorm,
        Pvrtc1Rgb,
        Pvrtc1Rgba,
        Pvrtc2Rgb,
        Pvrtc2Rgba,
        AtcRgb,
        AtcRgba,
        Fxt1Rgb,
    ] {
        assert!(imgio::can_encode(f), "{f:?}");
    }
    Ok(())
}
