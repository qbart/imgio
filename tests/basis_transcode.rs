//! `.basis` transcoding vs. the C++ Basis Universal transcoder: FNV-1a hashes of every image
//! level transcoded to every supported target (testdata/basis/golden.txt, tools/basisu-golden).

use imgio::basis::BasisFile;
use imgio::{AstcBlock, AstcKind, Format};
use std::error::Error;

/// basisu `transcoder_texture_format` id -> imgio format.
fn format(id: u32) -> Option<Format> {
    use Format::*;
    Some(match id {
        0 => Etc1Rgb,
        1 => Etc2Rgba,
        2 => Bc1Unorm,
        3 => Bc3Unorm,
        4 => Bc4Unorm,
        5 => Bc5Unorm,
        6 | 7 => Bc7Unorm, // 7: deprecated BC7_ALT, same output
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
        // ASTC LDR 5x4 .. 12x12 (basisu order).
        28..=40 => Astc(
            [
                AstcBlock::B5x4,
                AstcBlock::B5x5,
                AstcBlock::B6x5,
                AstcBlock::B6x6,
                AstcBlock::B8x5,
                AstcBlock::B8x6,
                AstcBlock::B10x5,
                AstcBlock::B10x6,
                AstcBlock::B8x8,
                AstcBlock::B10x8,
                AstcBlock::B10x10,
                AstcBlock::B12x10,
                AstcBlock::B12x12,
            ][id as usize - 28],
            AstcKind::Unorm,
        ),
        _ => return None,
    })
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| {
        (h ^ u64::from(x)).wrapping_mul(0x100000001b3)
    })
}

/// Checks every golden line whose file name starts with `prefix`; returns the count.
fn check(prefix: &str) -> Result<usize, Box<dyn Error>> {
    check_with(prefix, false)
}

/// Like `check`, optionally skipping targets imgio cannot produce yet (work in progress).
fn check_with(prefix: &str, only_supported: bool) -> Result<usize, Box<dyn Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/basis/");
    let golden = std::fs::read_to_string(format!("{dir}golden.txt"))?;
    let mut failures = Vec::new();
    let mut n = 0;
    for line in golden.lines().filter(|l| l.starts_with(prefix)) {
        let f: Vec<&str> = line.split(' ').collect();
        let (name, id, level) = (f[0], f[1].parse::<u32>()?, f[2].parse::<u32>()?);
        let data = std::fs::read(format!("{dir}{name}"))?;
        let file = BasisFile::parse(&data)?;
        // 17 = FXT1: not ported yet (see ROADMAP.md).
        let Some(fmt) = format(id) else {
            assert_eq!(id, 17, "unknown format id");
            continue;
        };
        if only_supported && !file.can_transcode_to(fmt) {
            continue;
        }
        let got = file.transcode_to_buffer(0, level, fmt);
        let ok = match (f.get(3), &got) {
            (Some(&"fail"), Err(_)) => true,
            (Some(size), Ok(buf)) => {
                buf.len() == size.parse::<usize>()? && format!("{:016x}", fnv(buf)) == f[4]
            }
            _ => false,
        };
        if !ok {
            failures.push(format!(
                "{line} -> {:?}",
                got.as_ref().map(|b| (b.len(), format!("{:016x}", fnv(b))))
            ));
        }
        n += 1;
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(n)
}

#[test]
fn etc1s() -> Result<(), Box<dyn Error>> {
    assert_eq!(check("etc1s_")?, 168); // 176 minus 8 FXT1 lines
    Ok(())
}

#[test]
fn uastc_ldr() -> Result<(), Box<dyn Error>> {
    assert_eq!(check("uastc_rgb")?, 136);
    Ok(())
}

#[test]
fn uastc_hdr() -> Result<(), Box<dyn Error>> {
    assert_eq!(check("uastc_hdr")?, 10);
    Ok(())
}

#[test]
fn astc_hdr_6x6() -> Result<(), Box<dyn Error>> {
    assert_eq!(check("astc_hdr6x6")?, 5);
    Ok(())
}

#[test]
fn astc_ldr() -> Result<(), Box<dyn Error>> {
    assert_eq!(check("astc_ldr")?, 240); // every target, mips, deblocked 12x12, PVRTC1
    Ok(())
}
