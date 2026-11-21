//! BMP decoder vs Pillow / ffmpeg decodes and hand-built files (testdata/file/bmp, see
//! cases.txt for the reference of each file), encoder vs Pillow / ffmpeg (golden `enc/`
//! files), round trips and corrupt input.

mod file_own_common;

use file_own_common::*;
use imgio::file::{FileFormat, bmp};
use imgio::{Format, Surface};

#[test]
fn decodes_reference_files() -> R {
    check_cases("bmp", FileFormat::Bmp, bmp::decode)
}

#[test]
fn covers_header_and_depth_variants() -> R {
    // Spot checks that the case list keeps covering the ROADMAP / spec features.
    let files: Vec<String> = cases("bmp")?.into_iter().map(|c| c.file).collect();
    for f in [
        "os2v1_p4.bmp",       // BITMAPCOREHEADER
        "os2v2_p8.bmp",       // OS/2 2.x 64-byte header
        "os2v2_16_rgb24.bmp", // OS/2 2.x 16-byte header
        "info_p2.bmp",        // 2-bit
        "rle4_holes.bmp",     // RLE4 delta / EOL / EOB
        "rle8_holes.bmp",     // RLE8 delta / EOL / EOB
        "bf_rgb565.bmp",      // bit fields
        "alphabf_bgra32.bmp", // BI_ALPHABITFIELDS
        "v4_bgra32.bmp",
        "v5_a2rgb10.bmp",
        "topdown_rgb24.bmp",
    ] {
        assert!(files.iter().any(|x| x == f), "{f}");
    }
    Ok(())
}

#[test]
fn rle_holes_are_transparent() -> R {
    let data = std::fs::read(format!("{}/rle8_holes.bmp", dir("bmp")))?;
    let img = bmp::decode(&data)?;
    assert_eq!(img.format, Format::Rgba8Srgb);
    // The top row is never drawn (early end of bitmap), the bottom row starts drawn.
    assert!(img.data[..50 * 4].iter().all(|&v| v == 0));
    let bottom = &img.data[37 * 50 * 4..];
    assert_eq!(bottom[3], 255);
    Ok(())
}

#[test]
fn encoded_golden() -> R {
    // Files checked against Pillow and ffmpeg by scripts/gen_file_testdata_own.py.
    for kind in ["rgb", "rgba", "gray"] {
        let (f, w, h, px) = ldr_source(kind)?;
        let out = bmp::encode(&Surface::new(f, w, h, &px)?, &bmp::EncodeOptions::default())?;
        golden_encoded("bmp", &format!("{kind}.bmp"), &out)?;
        let back = bmp::decode(&out)?;
        assert_eq!((back.format, back.width, back.height), (f, w, h));
        assert_eq!(&back.data[..], &px[..], "{kind}");
    }
    Ok(())
}

#[test]
fn encode_round_trips_other_inputs() -> R {
    let rgba = raw_rgba("mosaic_rgba_50x38.rgba")?;
    // BGRA -> RGBA file.
    let bgra: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[2], p[1], p[0], p[3]])
        .collect();
    let out = bmp::encode(
        &Surface::new(Format::Bgra8Unorm, 50, 38, &bgra)?,
        &bmp::EncodeOptions::default(),
    )?;
    assert_eq!(&bmp::decode(&out)?.data[..], &rgba[..]);
    // Gray + alpha -> RGBA with gray replicated.
    let ga: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[1], p[3]])
        .collect();
    let out = imgio::file::encode(
        FileFormat::Bmp,
        &Surface::new(Format::Rg8Unorm, 50, 38, &ga)?,
    )?;
    let back = imgio::file::decode(&out)?;
    let want: Vec<u8> = ga
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[0], p[0], p[1]])
        .collect();
    assert_eq!(&back.data[..], &want[..]);
    // 1x1 and odd widths (row padding).
    for w in 1..6u32 {
        let px: Vec<u8> = (0..w * 3 * 2).map(|i| (i * 37) as u8).collect();
        let out = bmp::encode(
            &Surface::new(Format::Rgb8Unorm, w, 2, &px)?,
            &bmp::EncodeOptions::default(),
        )?;
        assert_eq!(&bmp::decode(&out)?.data[..], &px[..]);
    }
    // Deeper formats are not silently truncated.
    let f = vec![0u8; 16];
    assert!(
        bmp::encode(
            &Surface::new(Format::Rgba32Float, 1, 1, &f)?,
            &bmp::EncodeOptions::default()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn corrupt_input_never_panics() -> R {
    for c in cases("bmp")? {
        fuzz(&c.data, bmp::decode);
    }
    assert!(bmp::decode(b"BM").is_err());
    assert!(bmp::decode(&[]).is_err());
    Ok(())
}
