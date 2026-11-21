//! KTX2 (Basis payloads) transcoding vs. the C++ Basis Universal `ktx2_transcoder`: FNV-1a
//! hashes of every level/layer/face transcoded to every supported target
//! (testdata/ktx2/golden.txt, tools/basisu-golden).

use imgio::basis::{BasisTexFormat, TranscodeOptions};
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

/// Checks every golden line whose file name passes `filter`; returns the count.
fn check(filter: impl Fn(&str) -> bool) -> Result<usize, Box<dyn Error>> {
    check_with(filter, false)
}

/// Like `check`, optionally skipping targets imgio cannot produce yet (work in progress).
fn check_with(
    filter: impl Fn(&str) -> bool,
    only_supported: bool,
) -> Result<usize, Box<dyn Error>> {
    check_file("golden.txt", filter, only_supported)
}

/// Checks the lines of `golden` (in testdata/ktx2) matching `filter`. golden_flags.txt lines
/// start with the basisu decode flags.
fn check_file(
    golden: &str,
    filter: impl Fn(&str) -> bool,
    only_supported: bool,
) -> Result<usize, Box<dyn Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/ktx2/");
    let flagged = golden == "golden_flags.txt";
    let golden = std::fs::read_to_string(format!("{dir}{golden}"))?;
    let (mut failures, mut checked) = (Vec::new(), 0);
    for line in golden.lines().filter(|l| filter(l)) {
        let mut f: Vec<&str> = line.split(' ').collect();
        let opts = if flagged {
            TranscodeOptions::from_flags(f.remove(0).parse()?)
        } else {
            TranscodeOptions::default()
        };
        let n = |i: usize| f[i].parse::<u32>();
        let (id, level, layer, face) = (n(1)?, n(2)?, n(3)?, n(4)?);
        let Some(fmt) = format(id) else { continue };
        let data = std::fs::read(format!("{dir}{}", f[0]))?;
        let file = Ktx2File::parse(&data)?;
        if only_supported && !file.can_transcode_to(fmt) {
            continue;
        }
        let got = file.transcode_to_buffer_with(level, layer, face, fmt, &opts);
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
    Ok(checked)
}

/// Payloads covered by their own tests.
fn pending(line: &str) -> bool {
    ["astc_ldr", "xuastc"].iter().any(|p| line.starts_with(p)) // own tests
}

#[test]
fn matches_basisu() -> Result<(), Box<dyn Error>> {
    assert_eq!(check(|l| !pending(l))?, 349 + 96); // + XUBC7
    Ok(())
}

#[test]
fn astc_ldr() -> Result<(), Box<dyn Error>> {
    // Every target; astc_ldr10x10_rgb.ktx2 carries DeblockFilterID = 1.
    assert_eq!(check(|l| l.starts_with("astc_ldr"))?, 128);
    Ok(())
}

#[test]
fn xubc7() -> Result<(), Box<dyn Error>> {
    // Every target incl. ASTC 4x4 and PVRTC1 (power-of-two), all mips.
    assert_eq!(check(|l| l.starts_with("xubc7"))?, 96);
    Ok(())
}

#[test]
fn xuastc_ldr() -> Result<(), Box<dyn Error>> {
    // xuastc10x8_dct_arith.ktx2 carries DeblockFilterID = 1.
    // BC7 (id 6) from 4x4 and 6x6 uses the direct ASTC -> BC7 fast paths.
    assert_eq!(check(|l| l.starts_with("xuastc"))?, 160);
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
        assert_eq!(f.tex_format(), Some(tex), "{name}");
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

#[test]
fn header_info_astc_ldr_xuastc_xubc7() -> Result<(), Box<dyn Error>> {
    use BasisTexFormat::{AstcLdr, XuastcLdr, Xubc7};
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/ktx2/");
    // (name, format, levels, layers, alpha, sRGB)
    let cases = [
        (
            "astc_ldr8x5_rgba.ktx2",
            AstcLdr(AstcBlock::B8x5),
            7,
            1,
            true,
            true,
        ),
        (
            "astc_ldr10x10_rgb.ktx2",
            AstcLdr(AstcBlock::B10x10),
            1,
            1,
            false,
            false,
        ),
        (
            "xuastc6x6_rgba.ktx2",
            XuastcLdr(AstcBlock::B6x6),
            7,
            1,
            true,
            true,
        ),
        (
            "xuastc10x8_dct_arith.ktx2",
            XuastcLdr(AstcBlock::B10x8),
            1,
            1,
            false,
            true,
        ),
        (
            "xuastc4x4_array.ktx2",
            XuastcLdr(AstcBlock::B4x4),
            1,
            2,
            false,
            true,
        ),
        ("xubc7_rgba.ktx2", Xubc7, 6, 1, true, true),
    ];
    for (name, tex, levels, layers, alpha, srgb) in cases {
        let data = std::fs::read(format!("{dir}{name}"))?;
        let f = Ktx2File::parse(&data)?;
        assert_eq!(f.tex_format(), Some(tex), "{name}");
        assert_eq!(f.level_count(), levels, "{name}");
        assert_eq!(f.layer_count(), layers, "{name}");
        assert_eq!(f.has_alpha(), alpha, "{name}");
        assert_eq!(f.is_srgb(), srgb, "{name}");
    }
    Ok(())
}

#[test]
fn decode_flags() -> Result<(), Box<dyn Error>> {
    assert_eq!(check_file("golden_flags.txt", |_| true, false)?, 625);
    Ok(())
}

/// `transcode_all` (levels / layers / faces in parallel) equals one-by-one `transcode`.
#[test]
fn transcode_all_matches_single() -> Result<(), Box<dyn Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/ktx2/");
    let mut checked = 0;
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_none_or(|e| e != "ktx2") {
            continue;
        }
        let data = std::fs::read(&path)?;
        let file = Ktx2File::parse(&data)?;
        let opts = TranscodeOptions::default();
        for fmt in [Format::Bc7Unorm, Format::Rgba8Unorm, Format::Bc6hUfloat] {
            if !file.can_transcode_to(fmt) {
                continue;
            }
            let all = file.transcode_all(fmt, &opts)?;
            let mut i = 0;
            for level in 0..file.level_count() {
                for layer in 0..file.layer_count().max(1) {
                    for face in 0..file.face_count() {
                        let t = &all[i];
                        assert_eq!((t.image, t.face, t.level), (layer, face, level));
                        let one = file.transcode_to_buffer(level, layer, face, fmt)?;
                        assert_eq!(&t.data[..], &one[..], "{path:?} {fmt:?} {level} {layer}");
                        i += 1;
                        checked += 1;
                    }
                }
            }
            assert_eq!(i, all.len());
        }
    }
    assert!(checked > 30, "{checked}");
    Ok(())
}

/// `ktx2::write_basis` vs basisu's own KTX2 writer: each testdata/ktx2/*.ktx2 (basisu `-ktx2`)
/// has a twin testdata/ktx2/basis/*.basis made with the same options; repackaging the twin
/// with the reference file's key/values (and zstd level) must give the same bytes.
#[test]
fn write_basis_matches_basisu() -> Result<(), Box<dyn Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/ktx2/");
    let mut n = 0;
    for entry in std::fs::read_dir(format!("{dir}basis"))? {
        let path = entry?.path();
        let name = path.file_stem().and_then(|s| s.to_str()).ok_or("name")?;
        let basis = std::fs::read(&path)?;
        let want = std::fs::read(format!("{dir}{name}.ktx2"))?;
        let reference = Ktx2File::parse(&want)?;
        // Keys as written by basisu (the alignment padding key is recreated by the writer).
        let kv: Vec<(String, Vec<u8>)> = reference
            .key_values()
            .filter(|(k, _)| k.first() != Some(&127))
            .map(|(k, v)| (String::from_utf8_lossy(k).into_owned(), v.to_vec()))
            .collect();
        let kv: Vec<(&str, &[u8])> = kv.iter().map(|(k, v)| (k.as_str(), &v[..])).collect();
        // basisu's default KTX2 Zstandard level.
        let zstd = (u32::from_le_bytes([want[44], want[45], want[46], want[47]]) == 2).then_some(6);
        let opts = imgio::ktx2::BasisKtx2Options {
            zstd,
            key_values: &kv,
        };
        let got = imgio::ktx2::write_basis(&basis, &opts)?;
        assert!(
            got[..] == want[..],
            "{name}: {} vs {} bytes",
            got.len(),
            want.len()
        );
        n += 1;
    }
    assert!(n >= 13, "{n}");
    Ok(())
}

/// Encode to KTX2 (every codec, with mips) and transcode back: same pixels as the `.basis`.
#[test]
fn encode_basis_ktx2_round_trip() -> Result<(), Box<dyn Error>> {
    use imgio::basis::{BasisCodec, BasisEncodeOptions, BasisFile};
    let raw = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/images/raw/");
    let ldr = std::fs::read(format!("{raw}rgba_64x48.rgba"))?;
    let hdr = std::fs::read(format!("{raw}hdr_40x24.rgba16f"))?;
    for (codec, src, fmt, target) in [
        (
            BasisCodec::Etc1s,
            imgio::Surface::new(Format::Rgba8Unorm, 64, 48, &ldr)?,
            0,
            Format::Rgba8Unorm,
        ),
        (
            BasisCodec::UastcLdr4x4,
            imgio::Surface::new(Format::Rgba8Unorm, 64, 48, &ldr)?,
            1,
            Format::Rgba8Unorm,
        ),
        (
            BasisCodec::UastcHdr4x4,
            imgio::Surface::new(Format::Rgba16Float, 40, 24, &hdr)?,
            0,
            Format::Rgba16Float,
        ),
        (
            BasisCodec::AstcHdr6x6,
            imgio::Surface::new(Format::Rgba16Float, 40, 24, &hdr)?,
            1,
            Format::Rgba16Float,
        ),
        (
            BasisCodec::UastcHdr6x6Intermediate,
            imgio::Surface::new(Format::Rgba16Float, 40, 24, &hdr)?,
            0,
            Format::Rgba16Float,
        ),
    ] {
        let opts = BasisEncodeOptions {
            codec,
            mipmaps: true,
            threads: 1,
            ..BasisEncodeOptions::default()
        };
        let basis = imgio::basis::encode_basis(&[src], &opts)?;
        let ktx2_opts = imgio::ktx2::BasisKtx2Options {
            zstd: (fmt == 1).then_some(3),
            key_values: &[],
        };
        let ktx2 = imgio::ktx2::write_basis(&basis, &ktx2_opts)?;
        let enc = imgio::ktx2::encode_basis(&[src], &opts, &ktx2_opts)?;
        let (b, k, e) = (
            BasisFile::parse(&basis)?,
            Ktx2File::parse(&ktx2)?,
            Ktx2File::parse(&enc)?,
        );
        assert!(
            e.key_values()
                .any(|(k, v)| k == b"KTXwriter" && v.starts_with(b"imgio "))
        );
        assert_eq!(k.level_count(), b.level_count(0));
        for level in 0..k.level_count() {
            let want = b.transcode_to_buffer(0, level, target)?;
            assert_eq!(
                &k.transcode_to_buffer(level, 0, 0, target)?[..],
                &want[..],
                "{codec:?} {level}"
            );
            assert_eq!(
                &e.transcode_to_buffer(level, 0, 0, target)?[..],
                &want[..],
                "{codec:?} {level}"
            );
        }
    }
    Ok(())
}
