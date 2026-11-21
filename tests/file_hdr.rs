//! Radiance HDR decoder vs ffmpeg / imagecodecs decodes (bit-exact) and hand-built files
//! (testdata/file/hdr, see cases.txt), encoder vs ffmpeg / imagecodecs (golden `enc/` files:
//! scanlines byte-identical to imagecodecs `rgbe_encode`), RGBE conversion edge cases and
//! corrupt input.

mod file_own_common;

use file_own_common::*;
use imgio::file::{FileFormat, hdr};
use imgio::{Format, Surface};

#[test]
fn decodes_reference_files() -> R {
    check_cases("hdr", FileFormat::Hdr, hdr::decode)
}

#[test]
fn header_values() -> R {
    let d = dir("hdr");
    let h = hdr::decode_hdr(&std::fs::read(format!("{d}/flat_exposure.hdr"))?)?;
    assert_eq!((h.exposure, h.xyz), (1.0, false)); // EXPOSURE=2.0 and EXPOSURE=0.5
    let h = hdr::decode_hdr(&std::fs::read(format!("{d}/xyze.hdr"))?)?;
    assert!(h.xyz);
    Ok(())
}

fn hdr_source() -> R<Vec<u8>> {
    // testdata/images/raw/hdr_40x24.rgba16f, the generator's float source.
    Ok(std::fs::read(format!(
        "{}/testdata/images/raw/hdr_40x24.rgba16f",
        env!("CARGO_MANIFEST_DIR")
    ))?)
}

#[test]
fn encoded_golden() -> R {
    let src = hdr_source()?;
    let s = Surface::new(Format::Rgba16Float, 40, 24, &src)?;
    for (name, rle) in [("hdr_rle.hdr", true), ("hdr_flat.hdr", false)] {
        let out = hdr::encode(
            &s,
            &hdr::EncodeOptions {
                rle,
                exposure: None,
            },
        )?;
        golden_encoded("hdr", name, &out)?;
        // Our decode equals the reference-decoded expected pixels of the imagecodecs file.
        let want = cases("hdr")?
            .into_iter()
            .find(|c| c.file == "ic_rle.hdr")
            .ok_or("ic_rle.hdr missing")?
            .expected;
        assert_eq!(&hdr::decode(&out)?.data[..], &want[..], "{name}");
    }
    Ok(())
}

#[test]
fn encode_options_and_inputs() -> R {
    // EXPOSURE line round trips; gray input is replicated to RGB.
    let g: Vec<u8> = [0.25f32, 1.0, 3.5, 1000.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = hdr::encode(
        &Surface::new(Format::R32Float, 2, 2, &g)?,
        &hdr::EncodeOptions {
            rle: true,
            exposure: Some(2.5),
        },
    )?;
    let d = hdr::decode_hdr(&out)?;
    assert_eq!(d.exposure, 2.5);
    let px: Vec<f32> = d
        .image
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(&px[..4], &[0.25, 0.25, 0.25, 1.0]);
    assert_eq!(&px[12..16], &[1000.0, 1000.0, 1000.0, 1.0]);
    Ok(())
}

#[test]
fn rgbe_conversion() {
    // ffmpeg float2rgbe / Walter rgbe2float.
    assert_eq!(hdr::f32_to_rgbe([1.0, 0.5, 0.25]), [128, 64, 32, 129]);
    assert_eq!(hdr::f32_to_rgbe([-1.0, 2.0, 300.7]), [0, 1, 150, 137]);
    assert_eq!(hdr::f32_to_rgbe([1e-33, 0.0, 0.0]), [0; 4]);
    assert_eq!(hdr::f32_to_rgbe([f32::MAX, 0.0, 0.0]), [255, 0, 0, 255]);
    assert_eq!(hdr::f32_to_rgbe([f32::NAN, 1.0, 0.0]), [0, 128, 0, 129]);
    assert_eq!(hdr::rgbe_to_f32([128, 64, 1, 129]), [1.0, 0.5, 0.0078125]);
    assert_eq!(
        hdr::rgbe_to_f32([255, 0, 3, 1]),
        [255.0 * 2f32.powi(-135), 0.0, 3.0 * 2f32.powi(-135)]
    );
    assert_eq!(hdr::rgbe_to_f32([9, 9, 9, 0]), [0.0; 3]);
    // Every RGBE value survives f32 -> RGBE (mantissa >= 128 for the largest channel).
    for e in 1..=255u8 {
        for m in 128..=255u8 {
            let f = hdr::rgbe_to_f32([m, m / 2, 0, e]);
            if f[0] >= 1e-32 {
                assert_eq!(hdr::f32_to_rgbe(f), [m, m / 2, 0, e]);
            }
        }
    }
}

#[test]
fn corrupt_input_never_panics() -> R {
    for c in cases("hdr")? {
        fuzz(&c.data, hdr::decode);
    }
    for bad in [
        &b"#?RADIANCE\n\n-Y 2 +X 2\n"[..],
        b"#?RADIANCE\n\n-Y 2 +Y 2\n\0\0\0\0",
        b"#?RADIANCE\n\n-Y 0 +X 2\n",
        b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n",
        b"#?RADIANCE\n\n-Y 1 +X 2\n\x01\x01\x01\x05\0\0\0\0",
    ] {
        assert!(hdr::decode(bad).is_err());
    }
    Ok(())
}
