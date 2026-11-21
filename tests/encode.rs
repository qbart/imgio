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
    let dims = name
        .trim_end_matches(".rgba")
        .rsplit('_')
        .next()
        .ok_or("name")?;
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
            _ => Format::Bc5Unorm,
        };
        let quality = match f[2] {
            "0" => Quality::Fast,
            "1" => Quality::Default,
            _ => Quality::High,
        };
        let src = Surface::new(Format::Rgba8Unorm, w, h, &px)?;
        let out = imgio::encode_to_buffer(&src, format, &EncodeOptions { quality })?;
        assert_eq!(format!("{:016x}", fnv(&out)), f[3], "{line}");
        n += 1;
    }
    assert_eq!(n, 24);
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
