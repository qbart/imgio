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
            "etc2rgba" => Format::Etc2Rgba,
            "eacr11" => Format::EacR11Unorm,
            "eacrg11" => Format::EacRg11Unorm,
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
            ("etc1" | "etc2rgba" | "eacr11" | "eacrg11", q) => [Quality::Default, Quality::High][q],
            (_, q) => all[q],
        };
        let src_format = if f[0].ends_with(".rgba16f") {
            Format::Rgba16Float
        } else {
            Format::Rgba8Unorm
        };
        let src = Surface::new(src_format, w, h, &px)?;
        let out = imgio::encode_to_buffer(&src, format, &EncodeOptions { quality })?;
        assert_eq!(format!("{:016x}", fnv(&out)), f[3], "{line}");
        n += 1;
    }
    assert_eq!(n, 65);
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
    assert!(imgio::encode_to_buffer(&src, Format::AtcRgb, &opts).is_err());
    Ok(())
}
