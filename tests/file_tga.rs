//! TGA decoder vs Pillow / ffmpeg decodes and hand-built files (testdata/file/tga, see
//! cases.txt for the reference of each file), encoder vs Pillow / ffmpeg (golden `enc/`
//! files), round trips and corrupt input.

mod file_own_common;

use file_own_common::*;
use imgio::file::{FileFormat, tga};
use imgio::{Format, Surface};

#[test]
fn decodes_reference_files() -> R {
    check_cases("tga", FileFormat::Tga, tga::decode)
}

#[test]
fn covers_types_and_origins() -> R {
    let files: Vec<String> = cases("tga")?.into_iter().map(|c| c.file).collect();
    for f in [
        "cmap15_i8.tga",
        "cmap16_i16_rle.tga",
        "cmap24_i8_rle.tga",
        "cmap32_i16.tga",
        "tc15.tga",
        "tc16_a1_rle_bottom.tga",
        "rgb24_bl.tga",
        "rgb24_br_rle.tga",
        "rgb24_tl.tga",
        "rgb24_tr.tga",
        "gray8_rle_br.tga",
        "pil_la16_rle_top.tga",
        "rgba32_attr0_rle.tga",
        "rgba32_attr3.tga",
    ] {
        assert!(files.iter().any(|x| x == f), "{f}");
    }
    Ok(())
}

#[test]
fn metadata() -> R {
    let d = dir("tga");
    let t = tga::decode_tga(&std::fs::read(format!("{d}/rgba32_attr3.tga"))?)?;
    assert_eq!(&t.image_id[..], b"ext attr 3");
    assert_eq!(t.attribute_type, Some(3));
    let t = tga::decode_tga(&std::fs::read(format!("{d}/cmap24_i8_rle.tga"))?)?;
    assert_eq!(&t.image_id[..], b"imgio colormapped test");
    assert_eq!(t.attribute_type, None);
    Ok(())
}

#[test]
fn encoded_golden() -> R {
    // Files checked against Pillow and ffmpeg by scripts/gen_file_testdata_own.py.
    for kind in ["rgb", "rgba", "gray"] {
        let (f, w, h, px) = ldr_source(kind)?;
        for rle in [false, true] {
            let out = tga::encode(&Surface::new(f, w, h, &px)?, &tga::EncodeOptions { rle })?;
            let name = format!("{kind}{}.tga", if rle { "_rle" } else { "" });
            golden_encoded("tga", &name, &out)?;
            assert_eq!(FileFormat::detect(&out), Some(FileFormat::Tga));
            let back = tga::decode(&out)?;
            assert_eq!((back.format, back.width, back.height), (f, w, h), "{name}");
            assert_eq!(&back.data[..], &px[..], "{name}");
        }
    }
    Ok(())
}

#[test]
fn rle_round_trips_edge_rows() -> R {
    // Runs of exactly 128 / 129, alternating pixels, single pixels at row ends.
    let w = 300u32;
    let mut px = Vec::new();
    for y in 0..4u32 {
        for x in 0..w {
            let v = match y {
                0 => 7,
                1 => (x / 129) as u8,
                2 => (x % 2) as u8,
                _ => (x * 13 % 251) as u8,
            };
            px.extend([v, v.wrapping_add(1), 3, if x == w - 1 { 0 } else { 255 }]);
        }
    }
    let out = tga::encode(
        &Surface::new(Format::Rgba8Unorm, w, 4, &px)?,
        &tga::EncodeOptions { rle: true },
    )?;
    assert_eq!(&tga::decode(&out)?.data[..], &px[..]);
    Ok(())
}

#[test]
fn corrupt_input_never_panics() -> R {
    for c in cases("tga")? {
        fuzz(&c.data, tga::decode);
    }
    assert!(tga::decode(&[0; 17]).is_err());
    Ok(())
}
