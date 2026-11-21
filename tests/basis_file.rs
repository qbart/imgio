//! `.basis` container parsing on files written by Basis Universal v2.50 (testdata/basis).

use imgio::basis::{BasisFile, BasisTexFormat, BasisTexType};
use std::error::Error;

fn read(name: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    Ok(std::fs::read(format!(
        "{}/testdata/basis/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))?)
}

#[test]
fn parses_all_formats() -> Result<(), Box<dyn Error>> {
    let cases = [
        ("etc1s_rgba.basis", BasisTexFormat::Etc1s, 64, 48, 7, true),
        ("etc1s_rgb.basis", BasisTexFormat::Etc1s, 37, 29, 1, false),
        (
            "uastc_rgba.basis",
            BasisTexFormat::UastcLdr4x4,
            64,
            48,
            7,
            true,
        ),
        (
            "uastc_rgb.basis",
            BasisTexFormat::UastcLdr4x4,
            37,
            29,
            1,
            false,
        ),
        (
            "uastc_hdr4x4.basis",
            BasisTexFormat::UastcHdr4x4,
            40,
            24,
            1,
            false,
        ),
        (
            "astc_hdr6x6.basis",
            BasisTexFormat::AstcHdr6x6,
            40,
            24,
            1,
            false,
        ),
        (
            "uastc_hdr6x6i.basis",
            BasisTexFormat::UastcHdr6x6Intermediate,
            40,
            24,
            1,
            false,
        ),
    ];
    for (name, fmt, w, h, levels, alpha) in cases {
        let data = read(name)?;
        let f = BasisFile::parse(&data)?;
        assert_eq!(f.tex_format(), fmt, "{name}");
        assert_eq!(f.tex_type(), BasisTexType::Tex2D);
        assert_eq!(f.image_count(), 1);
        assert_eq!(f.level_count(0), levels, "{name}");
        assert_eq!(f.has_alpha(), alpha, "{name}");
        let l0 = f.level_info(0, 0)?;
        assert_eq!((l0.width, l0.height), (w, h), "{name}");
        let (bw, bh) = fmt.block_dim();
        assert_eq!((l0.blocks_x, l0.blocks_y), (w.div_ceil(bw), h.div_ceil(bh)));
        let last = f.level_info(0, levels - 1)?;
        assert_eq!(
            (last.width, last.height),
            ((w >> (levels - 1)).max(1), (h >> (levels - 1)).max(1))
        );
        assert!(f.level_info(0, levels).is_err());
    }
    Ok(())
}

#[test]
fn rejects_corruption() -> Result<(), Box<dyn Error>> {
    let mut data = read("etc1s_rgb.basis")?;
    assert!(BasisFile::parse(&data[..data.len() - 1]).is_err());
    let last = data.len() - 1;
    data[last] ^= 1; // data CRC
    assert!(BasisFile::parse(&data).is_err());
    data[last] ^= 1;
    data[20] ^= 1; // header CRC
    assert!(BasisFile::parse(&data).is_err());
    Ok(())
}
