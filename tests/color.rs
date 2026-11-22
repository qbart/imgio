//! Colour management against LittleCMS 2.19 (testdata/color, made by
//! scripts/gen_color_testdata.py): matrix / TRC and LUT profiles, gray profiles, intents.
#![cfg(feature = "color")]

use imgio::color::{self, Channels, ColorSpace, Intent, Profile};
use imgio::{Error, Format, Surface};
use std::error::Error as StdError;
use std::path::PathBuf;

type R<T = ()> = Result<T, Box<dyn StdError>>;

fn read(name: &str) -> R<Vec<u8>> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/color")
        .join(name);
    Ok(std::fs::read(&p).map_err(|e| format!("{name}: {e}"))?)
}

fn intent(v: u8) -> Intent {
    match v {
        0 => Intent::Perceptual,
        1 => Intent::RelativeColorimetric,
        2 => Intent::Saturation,
        _ => Intent::AbsoluteColorimetric,
    }
}

/// Largest allowed (max, mean) difference to LittleCMS 8-bit transforms (`NOOPTIMIZE`) per
/// case: matrix / TRC and gray profiles are exact; LUT profiles differ by LittleCMS's 16-bit
/// stage precision near black (one pixel at 2, the true value lies between).
fn bound(from: &str, to: &str) -> (u8, f64) {
    if from == "lut" || to == "lut" {
        (2, 0.02)
    } else {
        (0, 0.0)
    }
}

#[test]
fn matches_littlecms() -> R {
    let text = String::from_utf8(read("cases.txt")?)?;
    let mut n = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (from, to, it) = (f[0], f[1], f[2].parse::<u8>()?);
        let (w, h): (u32, u32) = (f[3].parse()?, f[4].parse()?);
        let (ci, co): (usize, usize) = (f[5].parse()?, f[6].parse()?);
        let src = read(if ci == 1 {
            "src_gray.raw"
        } else {
            "src_rgb.raw"
        })?;
        let pf = Profile::from_icc(&read(&format!("{from}.icc"))?)?;
        let pt = Profile::from_icc(&read(&format!("{to}.icc"))?)?;
        let fmt = |c: usize| {
            if c == 1 {
                Format::R8Unorm
            } else {
                Format::Rgb8Unorm
            }
        };
        let s = Surface::new(fmt(ci), w, h, &src)?;
        let img = color::convert(&s, &pf, &pt, intent(it), fmt(co))?;
        let want = read(&format!("{from}__{to}__{it}.raw"))?;
        assert_eq!(img.data.len(), want.len());
        let d: Vec<u8> = img
            .data
            .iter()
            .zip(&want)
            .map(|(a, b)| a.abs_diff(*b))
            .collect();
        let max = d.iter().copied().max().unwrap_or(0);
        let mean = d.iter().map(|&v| f64::from(v)).sum::<f64>() / d.len() as f64;
        println!("{from:>10} -> {to:<10} intent {it}: max {max} mean {mean:.4}");
        if let Some(i) = d.iter().position(|&v| v == max).filter(|_| max > 1) {
            let p = i / co * ci;
            println!(
                "   worst at {i}: src {:?} ours {} lcms {}",
                &src[p..p + ci],
                img.data[i],
                want[i]
            );
        }
        let (bm, bmean) = bound(from, to);
        assert!(
            max <= bm && mean <= bmean,
            "{from} -> {to} {it}: max {max} mean {mean}"
        );
        n += 1;
    }
    assert_eq!(n, 14);
    Ok(())
}

#[test]
fn builtins_cicp_and_round_trips() -> R {
    let src = read("src_rgb.raw")?;
    let s = Surface::new(Format::Rgb8Unorm, 64, 48, &src)?;
    // The built-in sRGB agrees with LittleCMS's sRGB profile within 1.
    let lcms = Profile::from_icc(&read("lcms_srgb.icc")?)?;
    let ours = Profile::srgb()?;
    let p3 = Profile::from_icc(&read("p3.icc")?)?;
    let a = color::convert(&s, &ours, &p3, Intent::Perceptual, Format::Rgb8Unorm)?;
    let b = color::convert(&s, &lcms, &p3, Intent::Perceptual, Format::Rgb8Unorm)?;
    assert!(
        a.data
            .iter()
            .zip(b.data.iter())
            .all(|(x, y)| x.abs_diff(*y) <= 1)
    );
    // Identity and float round trips through a wider space.
    let same = color::convert(
        &s,
        &ours,
        &ours,
        Intent::RelativeColorimetric,
        Format::Rgb8Unorm,
    )?;
    assert_eq!(&same.data[..], &src[..]);
    let wide = color::convert(
        &s,
        &ours,
        &Profile::new(ColorSpace::Rec2020)?,
        Intent::RelativeColorimetric,
        Format::Rgba32Float,
    )?;
    let back = color::convert(
        &wide.surface()?,
        &Profile::new(ColorSpace::Rec2020)?,
        &ours,
        Intent::RelativeColorimetric,
        Format::Rgb8Unorm,
    )?;
    assert!(
        back.data
            .iter()
            .zip(src.iter())
            .all(|(x, y)| x.abs_diff(*y) <= 1)
    );
    // sRGB -> linear sRGB is the sRGB EOTF.
    let lin = color::convert(
        &s,
        &ours,
        &Profile::new(ColorSpace::LinearSrgb)?,
        Intent::RelativeColorimetric,
        Format::Rgba32Float,
    )?;
    for i in [0usize, 7, 100, 1000] {
        let v = f64::from(src[i * 3]) / 255.0;
        let want = if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        };
        let k = i * 16;
        let got = f32::from_le_bytes([
            lin.data[k],
            lin.data[k + 1],
            lin.data[k + 2],
            lin.data[k + 3],
        ]);
        assert!((f64::from(got) - want).abs() < 2e-4, "{i}: {got} vs {want}");
    }
    // CICP: BT.709 primaries + sRGB transfer = sRGB; BT.2020 + PQ; unknown values.
    let c = Profile::cicp(1, 13, 0, true)?;
    let viac = color::convert(
        &s,
        &c,
        &ours,
        Intent::RelativeColorimetric,
        Format::Rgb8Unorm,
    )?;
    assert!(
        viac.data
            .iter()
            .zip(src.iter())
            .all(|(x, y)| x.abs_diff(*y) <= 1)
    );
    assert!(Profile::cicp(9, 16, 0, true).is_ok());
    assert_eq!(
        Profile::cicp(200, 13, 0, true).err(),
        Some(Error::Unsupported)
    );
    // Every built-in builds a transform to sRGB; ICC bytes round trip.
    for cs in [
        ColorSpace::Srgb,
        ColorSpace::LinearSrgb,
        ColorSpace::DisplayP3,
        ColorSpace::DciP3,
        ColorSpace::AdobeRgb,
        ColorSpace::ProPhoto,
        ColorSpace::Rec2020,
        ColorSpace::Rec2100Pq,
        ColorSpace::Rec2100Hlg,
        ColorSpace::AcesCg,
        ColorSpace::Aces2065,
    ] {
        let p = Profile::new(cs)?;
        assert_eq!(p.data_channels(), Channels::Rgb);
        color::convert(&s, &p, &ours, Intent::Perceptual, Format::Rgba16Float)?;
        let again = Profile::from_icc(&p.icc()?)?;
        let x = color::convert(
            &s,
            &p,
            &ours,
            Intent::RelativeColorimetric,
            Format::Rgb8Unorm,
        )?;
        let y = color::convert(
            &s,
            &again,
            &ours,
            Intent::RelativeColorimetric,
            Format::Rgb8Unorm,
        )?;
        assert!(
            x.data
                .iter()
                .zip(y.data.iter())
                .all(|(a, b)| a.abs_diff(*b) <= 1),
            "{cs:?}"
        );
    }
    assert_eq!(
        Profile::new(ColorSpace::Gray(2.2))?.data_channels(),
        Channels::Gray
    );
    Ok(())
}

#[test]
fn alpha_and_errors() -> R {
    let px: Vec<u8> = (0..16 * 4).map(|i| (i * 37) as u8).collect();
    let s = Surface::new(Format::Rgba8Unorm, 4, 4, &px)?;
    let srgb = Profile::srgb()?;
    let p3 = Profile::new(ColorSpace::DisplayP3)?;
    let out = color::convert(&s, &srgb, &p3, Intent::Perceptual, Format::Rgba8Unorm)?;
    for i in 0..16 {
        assert_eq!(out.data[i * 4 + 3], px[i * 4 + 3]);
    }
    let gray = Profile::new(ColorSpace::Gray(2.2))?;
    let ga: Vec<u8> = (0..32).map(|i| (i * 8) as u8).collect();
    let gs = Surface::new(Format::Rg8Unorm, 4, 4, &ga)?;
    let g = color::convert(
        &gs,
        &gray,
        &Profile::new(ColorSpace::Gray(1.0))?,
        Intent::Perceptual,
        Format::Rg8Unorm,
    )?;
    for i in 0..16 {
        assert_eq!(g.data[i * 2 + 1], ga[i * 2 + 1]);
    }
    // Gray data with an RGB profile, RGB output with a gray profile.
    assert_eq!(
        color::convert(&gs, &srgb, &p3, Intent::Perceptual, Format::Rgb8Unorm).err(),
        Some(Error::InvalidArgument)
    );
    assert_eq!(
        color::convert(&s, &srgb, &gray, Intent::Perceptual, Format::Rgb8Unorm).err(),
        Some(Error::InvalidArgument)
    );
    let bc = Surface::new(Format::Bc1Unorm, 4, 4, &px[..8])?;
    assert_eq!(
        color::convert(&bc, &srgb, &p3, Intent::Perceptual, Format::Rgb8Unorm).err(),
        Some(Error::Unsupported)
    );
    assert_eq!(
        Profile::from_icc(b"not a profile").err(),
        Some(Error::InvalidData)
    );
    // Truncated / corrupted profiles never panic.
    let icc = read("lut.icc")?;
    for len in (0..icc.len()).step_by(997) {
        let _ = Profile::from_icc(&icc[..len]);
    }
    let p3b = read("p3.icc")?;
    let mut rng = 0x1234_5678_9ABC_DEF0u64;
    for _ in 0..500 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let mut d = p3b.clone();
        let i = (rng as usize >> 8) % d.len();
        d[i] ^= 1 << ((rng >> 3) & 7);
        if let Ok(p) = Profile::from_icc(&d) {
            let _ = color::convert(&s, &p, &srgb, Intent::Perceptual, Format::Rgba8Unorm);
        }
    }
    Ok(())
}
