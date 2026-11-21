//! `.basis` transcoding vs. the C++ Basis Universal transcoder: FNV-1a hashes of every image
//! level transcoded to every supported target (testdata/basis/golden.txt, tools/basisu-golden).

use imgio::basis::{BasisFile, TranscodeOptions};
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
        17 => Fxt1Rgb,
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
    check_where(|l| l.starts_with(prefix), only_supported)
}

/// Like `check_with`, for the golden lines matching `filter`.
fn check_where(
    filter: impl Fn(&str) -> bool,
    only_supported: bool,
) -> Result<usize, Box<dyn Error>> {
    check_file("golden.txt", filter, only_supported)
}

/// Checks the lines of `golden` (in testdata/basis) matching `filter`. golden_flags.txt lines
/// start with the basisu decode flags.
fn check_file(
    golden: &str,
    filter: impl Fn(&str) -> bool,
    only_supported: bool,
) -> Result<usize, Box<dyn Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/basis/");
    let flagged = golden == "golden_flags.txt";
    let golden = std::fs::read_to_string(format!("{dir}{golden}"))?;
    let mut failures = Vec::new();
    let mut n = 0;
    for line in golden.lines().filter(|l| filter(l)) {
        let mut f: Vec<&str> = line.split(' ').collect();
        let opts = if flagged {
            TranscodeOptions::from_flags(f.remove(0).parse()?)
        } else {
            TranscodeOptions::default()
        };
        let (name, id, level) = (f[0], f[1].parse::<u32>()?, f[2].parse::<u32>()?);
        let data = std::fs::read(format!("{dir}{name}"))?;
        let file = BasisFile::parse(&data)?;
        let fmt = format(id).ok_or("unknown format id")?;
        if only_supported && !file.can_transcode_to(fmt) {
            continue;
        }
        let got = file.transcode_to_buffer_with(0, level, fmt, &opts);
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
    assert_eq!(check("etc1s_")?, 176);
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

#[test]
fn xubc7() -> Result<(), Box<dyn Error>> {
    assert_eq!(check("xubc7")?, 128); // every target incl. ASTC 4x4, mips, PVRTC1 failures
    Ok(())
}

#[test]
fn xuastc_ldr() -> Result<(), Box<dyn Error>> {
    // xuastc4x4_rgba: full Zstd (small mips: full arithmetic); xuastc5x4_dct: Zstd + DCT;
    // xuastc6x6_rgba_arith: arithmetic; xuastc8x6_rgb_hybrid: <= 64 blocks, so arithmetic;
    // xuastc4x4_hybrid_dct: hybrid + DCT; xuastc6x5_dct_arith: arithmetic + DCT;
    // xuastc12x12_pow2: deblocked, PVRTC1; xuastc{6x6,8x6}_mosaic_*: block-structured images
    // reaching every branch of the 6x6 / 8x6 BC7 fast paths.
    // BC7 (id 6) from 4x4, 6x6 and 8x6 uses the direct ASTC -> BC7 fast paths.
    assert_eq!(check_where(|l| l.starts_with("xuastc"), false)?, 704);
    Ok(())
}

#[test]
fn decode_flags() -> Result<(), Box<dyn Error>> {
    assert_eq!(check_file("golden_flags.txt", |_| true, false)?, 2520);
    Ok(())
}

#[test]
fn options_flags_roundtrip() {
    for flags in [
        0,
        4,
        8,
        32,
        64,
        128,
        512,
        1024,
        1056,
        4 | 8 | 32 | 64 | 128 | 1024,
    ] {
        assert_eq!(TranscodeOptions::from_flags(flags).flags(), flags);
    }
    assert_eq!(TranscodeOptions::default().flags(), 0);
}
