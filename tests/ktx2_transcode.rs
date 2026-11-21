//! KTX2 (Basis payloads) transcoding vs. the C++ Basis Universal `ktx2_transcoder`: FNV-1a
//! hashes of every level/layer/face transcoded to every supported target
//! (testdata/ktx2/golden.txt, tools/basisu-golden).

use imgio::basis::BasisTexFormat;
use imgio::ktx2::Ktx2File;
use imgio::{AstcBlock, AstcKind, Format};
use std::error::Error;

fn format(id: u32) -> Option<Format> {
    use Format::*;
    Some(match id {
        0 => Etc1Rgb,
        1 => Etc2Rgba,
        2 => Bc1Unorm,
        3 => Bc3Unorm,
        4 => Bc4Unorm,
        5 => Bc5Unorm,
        6 | 7 => Bc7Unorm,
        8 => Pvrtc1Rgb,
        9 => Pvrtc1Rgba,
        10 => Astc(AstcBlock::B4x4, AstcKind::Unorm),
        11 => AtcRgb,
        12 => AtcRgba,
        13 => Rgba8Unorm,
        14 => Rgb565,
        15 => Bgr565,
        16 => Rgba4444,
        18 => Pvrtc2Rgb,
        19 => Pvrtc2Rgba,
        20 => EacR11Unorm,
        21 => EacRg11Unorm,
        22 => Bc6hUfloat,
        23 => Astc(AstcBlock::B4x4, AstcKind::Float),
        24 => Rgb16Float,
        25 => Rgba16Float,
        26 => Rgb9e5Float,
        27 => Astc(AstcBlock::B6x6, AstcKind::Float),
        _ => return None, // 17 = FXT1, not ported yet
    })
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| {
        (h ^ u64::from(x)).wrapping_mul(0x100000001b3)
    })
}

#[test]
fn matches_basisu() -> Result<(), Box<dyn Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/ktx2/");
    let golden = std::fs::read_to_string(format!("{dir}golden.txt"))?;
    let (mut failures, mut checked) = (Vec::new(), 0);
    for line in golden.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let n = |i: usize| f[i].parse::<u32>();
        let (id, level, layer, face) = (n(1)?, n(2)?, n(3)?, n(4)?);
        let Some(fmt) = format(id) else { continue };
        let data = std::fs::read(format!("{dir}{}", f[0]))?;
        let file = Ktx2File::parse(&data)?;
        let got = file.transcode_to_buffer(level, layer, face, fmt);
        let ok = match (f[5], &got) {
            ("fail", Err(_)) => true,
            (size, Ok(buf)) => {
                buf.len() == size.parse::<usize>()? && format!("{:016x}", fnv(buf)) == f[6]
            }
            _ => false,
        };
        if !ok {
            failures.push(format!("{line} -> {:?}", got.map(|b| b.len())));
        }
        checked += 1;
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(checked, 349 - 9); // minus FXT1 lines
    Ok(())
}

#[test]
fn header_info() -> Result<(), Box<dyn Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/ktx2/");
    let cases = [
        ("etc1s_rgba.ktx2", BasisTexFormat::Etc1s, 7, 1, true),
        ("etc1s_array.ktx2", BasisTexFormat::Etc1s, 1, 2, false),
        (
            "uastc_rgba_zstd.ktx2",
            BasisTexFormat::UastcLdr4x4,
            7,
            1,
            true,
        ),
        (
            "uastc_rgb_raw.ktx2",
            BasisTexFormat::UastcLdr4x4,
            1,
            1,
            false,
        ),
        (
            "uastc_hdr4x4.ktx2",
            BasisTexFormat::UastcHdr4x4,
            1,
            1,
            false,
        ),
        ("astc_hdr6x6.ktx2", BasisTexFormat::AstcHdr6x6, 1, 1, false),
        (
            "uastc_hdr6x6i.ktx2",
            BasisTexFormat::UastcHdr6x6Intermediate,
            1,
            1,
            false,
        ),
    ];
    for (name, tex, levels, layers, alpha) in cases {
        let data = std::fs::read(format!("{dir}{name}"))?;
        let f = Ktx2File::parse(&data)?;
        assert_eq!(f.tex_format(), tex, "{name}");
        assert_eq!(f.level_count(), levels, "{name}");
        assert_eq!(f.layer_count(), layers, "{name}");
        assert_eq!(f.has_alpha(), alpha, "{name}");
        assert_eq!(f.face_count(), 1);
        assert!(
            f.transcode_to_buffer(levels, 0, 0, Format::Rgba16Float)
                .is_err()
        );
        let mut bad = data.clone();
        bad[0] = 0;
        assert!(Ktx2File::parse(&bad).is_err());
        assert!(Ktx2File::parse(&data[..100]).is_err());
    }
    Ok(())
}
