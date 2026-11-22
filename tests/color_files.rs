//! Colour descriptions read from files and colour managed decoding (testdata/color/files, made
//! by scripts/gen_color_testdata.py).
#![cfg(all(feature = "color", feature = "png"))]

use imgio::Format;
use imgio::color::{ColorSpace, Intent, Profile};
use imgio::file::{self, Colorimetry, Transfer};
use std::error::Error as StdError;
use std::path::PathBuf;

type R<T = ()> = Result<T, Box<dyn StdError>>;

fn read(name: &str) -> R<Vec<u8>> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/color")
        .join(name);
    Ok(std::fs::read(&p).map_err(|e| format!("{name}: {e}"))?)
}

fn icc_of(name: &str) -> R<Vec<u8>> {
    match file::colorimetry(&read(name)?)? {
        Colorimetry::Icc(b) => Ok(b.to_vec()),
        c => Err(format!("{name}: {c:?}").into()),
    }
}

#[test]
fn embedded_profiles_are_found() -> R {
    let p3 = read("p3.icc")?;
    for f in [
        "files/p3.png",
        "files/p3.jpg",
        "files/p3.webp",
        "files/p3.tif",
        "files/p3.avif",
    ] {
        assert_eq!(icc_of(f)?, p3, "{f}");
    }
    // 245 KB profile over several APP2 segments.
    assert_eq!(icc_of("files/lut.jpg")?, read("lut.icc")?);
    match file::colorimetry(&read("files/cicp.png")?)? {
        Colorimetry::Cicp {
            primaries,
            transfer,
            matrix,
            full_range,
        } => {
            assert_eq!((primaries, transfer, matrix, full_range), (9, 16, 0, true));
        }
        c => return Err(format!("{c:?}").into()),
    }
    match file::colorimetry(&read("files/gama_chrm.png")?)? {
        Colorimetry::Primaries { xy, transfer } => {
            assert_eq!(xy[3], [0.3457, 0.3585]);
            assert_eq!(xy[0], [0.64, 0.33]);
            assert_eq!(transfer, Transfer::Gamma(100_000.0 / 55556.0));
        }
        c => return Err(format!("{c:?}").into()),
    }
    match file::colorimetry(&read("files/p3.exr")?)? {
        Colorimetry::Primaries { xy, transfer } => {
            assert_eq!(
                xy,
                [[0.68, 0.32], [0.265, 0.69], [0.15, 0.06], [0.3127, 0.329]]
            );
            assert_eq!(transfer, Transfer::Linear);
        }
        c => return Err(format!("{c:?}").into()),
    }
    match file::colorimetry(&read("files/primaries.hdr")?)? {
        Colorimetry::Primaries { xy, .. } => assert_eq!(xy[0], [0.708, 0.292]),
        c => return Err(format!("{c:?}").into()),
    }
    assert!(matches!(
        file::colorimetry(b"BM nothing")?,
        Colorimetry::Unspecified
    ));
    Ok(())
}

#[test]
fn decode_to_matches_littlecms() -> R {
    let srgb = Profile::srgb()?;
    // PNG with a P3 profile: byte identical with LittleCMS (same target profile); imgio's
    // built-in sRGB differs from LittleCMS's by colorant rounding (within 1).
    let lcms_srgb = Profile::from_icc(&read("lcms_srgb.icc")?)?;
    let want = read("files/png_p3_to_srgb.raw")?;
    let img = file::decode_to(&read("files/p3.png")?, &lcms_srgb, Intent::Perceptual)?;
    assert_eq!(img.format, Format::Rgb8Unorm);
    assert_eq!(&img.data[..], &want[..]);
    let img = file::decode_to(&read("files/p3.png")?, &srgb, Intent::Perceptual)?;
    assert_eq!(img.format, Format::Rgb8Srgb);
    assert!(img.data.iter().zip(&want).all(|(a, b)| a.abs_diff(*b) <= 1));
    // gAMA + cHRM: the profile is built from the chunk values (LittleCMS reference profile
    // stores them as s15Fixed16).
    let img = file::decode_to(
        &read("files/gama_chrm.png")?,
        &lcms_srgb,
        Intent::Perceptual,
    )?;
    let want = read("files/png_gama_chrm_to_srgb.raw")?;
    assert!(img.data.iter().zip(&want).all(|(a, b)| a.abs_diff(*b) <= 1));
    // Plain sRGB files are returned as decoded.
    let plain = file::encode(file::FileFormat::Png, &img.surface()?)?;
    let back = file::decode_to(&plain, &srgb, Intent::Perceptual)?;
    assert_eq!(&back.data[..], &img.data[..]);
    // Other targets and gray output.
    let p3 = Profile::new(ColorSpace::DisplayP3)?;
    let in_p3 = file::decode_to(&read("files/p3.png")?, &p3, Intent::RelativeColorimetric)?;
    let raw = file::decode(&read("files/p3.png")?)?;
    assert!(
        in_p3
            .data
            .iter()
            .zip(raw.data.iter())
            .all(|(a, b)| a.abs_diff(*b) <= 1)
    );
    assert_eq!(in_p3.format, Format::Rgb8Unorm);
    let g = file::decode_to(
        &read("files/p3.png")?,
        &Profile::new(ColorSpace::GraySrgb)?,
        Intent::Perceptual,
    )?;
    assert_eq!(g.format, Format::R8Unorm);
    // HDR: PQ cICP and linear EXR to linear sRGB floats.
    let lin = Profile::new(ColorSpace::LinearSrgb)?;
    let pq = file::decode_to(&read("files/cicp.png")?, &lin, Intent::RelativeColorimetric)?;
    assert_eq!(pq.format, Format::Rgba32Float);
    #[cfg(feature = "exr")]
    {
        let e = file::decode_to(&read("files/p3.exr")?, &lin, Intent::RelativeColorimetric)?;
        assert_eq!(e.format, Format::Rgba32Float);
        // A neutral input stays neutral (same D65 white; the built-in's colorants are rounded
        // differently: within 0.1%).
        let k = 20 * 16;
        let f =
            |o: usize| f32::from_le_bytes([e.data[o], e.data[o + 1], e.data[o + 2], e.data[o + 3]]);
        assert!(
            (f(k) - f(k + 4)).abs() < 1e-3 * f(k) && (f(k) - f(k + 8)).abs() < 1e-3 * f(k),
            "{} {} {}",
            f(k),
            f(k + 4),
            f(k + 8)
        );
    }
    Ok(())
}
