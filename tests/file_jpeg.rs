//! JPEG decode against libjpeg-turbo 3.2 (testdata/file/jpeg, made by
//! scripts/gen_file_testdata_webjpgavif.py with Pillow / imagecodecs): lossy files within
//! measured bounds, lossless files exact, EXIF orientation as metadata.
#![cfg(feature = "jpeg")]

use imgio::file::{self, FileFormat, jpeg};
use imgio::{Format, Surface};
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;

type R<T = ()> = Result<T, Box<dyn Error>>;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/file/jpeg")
}

fn read(name: &str) -> R<Vec<u8>> {
    Ok(std::fs::read(dir().join(name)).map_err(|e| format!("{name}: {e}"))?)
}

struct Entry {
    name: String,
    w: u32,
    h: u32,
    ch: usize,
    bits: u32,
    kv: HashMap<String, String>,
}

/// `<file> <width> <height> <channels> <bits> [key=value...]` lines.
fn manifest() -> R<Vec<Entry>> {
    let text = String::from_utf8(read("manifest.txt")?)?;
    let mut out = Vec::new();
    for l in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let field = |i: usize| f.get(i).copied().ok_or_else(|| format!("bad line {l}"));
        out.push(Entry {
            name: field(0)?.to_string(),
            w: field(1)?.parse()?,
            h: field(2)?.parse()?,
            ch: field(3)?.parse()?,
            bits: field(4)?.parse()?,
            kv: f
                .iter()
                .skip(5)
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        });
    }
    Ok(out)
}

/// Expected output: the libjpeg-turbo decode; > 8 bits (lossless) scaled to 16-bit unorm like
/// imgio (`v * 65535 / max`, rounded), RGB > 8 bits with an opaque alpha.
fn expected(e: &Entry) -> R<Vec<u8>> {
    let r = read(&e.name.replace(".jpg", ".ref"))?;
    if e.bits == 8 {
        return Ok(r);
    }
    let max = (1u32 << e.bits) - 1;
    let v: Vec<u8> = r
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|b| {
            let v = u32::from(u16::from_le_bytes(*b));
            (((v * 65535 + max / 2) / max) as u16).to_le_bytes()
        })
        .collect();
    if e.ch == 3 {
        return Ok(v
            .as_chunks::<6>()
            .0
            .iter()
            .flat_map(|p| [&p[..], &[255, 255]].concat())
            .collect());
    }
    Ok(v)
}

fn format_of(e: &Entry) -> Format {
    match (e.ch, e.bits > 8) {
        (1, false) => Format::R8Unorm,
        (1, true) => Format::R16Unorm,
        (_, false) => Format::Rgb8Srgb,
        (_, true) => Format::Rgba16Unorm,
    }
}

/// (max per-channel difference, mean absolute difference) to libjpeg-turbo (Pillow: islow
/// IDCT, fancy upsampling), measured with jpeg-decoder 0.3.2. The differences come from its
/// IDCT (stb_image style, not jidctint) and its 20-bit YCbCr -> RGB fixed point; lossless
/// files are exact. See docs/spec/file_jpeg.md for the per-file numbers.
fn bound(e: &Entry) -> (u8, f64) {
    if e.name.starts_with("lossless") {
        return (0, 0.0);
    }
    match e.name.as_str() {
        "gray.jpg" | "gray_progressive.jpg" | "ycck_adobe.jpg" => (1, 0.05),
        "cmyk_adobe.jpg" | "baseline_420.jpg" | "baseline_411.jpg" => (2, 0.12),
        _ => (3, 0.25),
    }
}

#[test]
fn decode_matches_libjpeg_turbo_within_bounds() -> R {
    for e in manifest()?
        .iter()
        .filter(|e| !e.kv.contains_key("unsupported"))
    {
        let data = read(&e.name)?;
        assert_eq!(
            FileFormat::detect(&data),
            Some(FileFormat::Jpeg),
            "{}",
            e.name
        );
        let img = jpeg::decode(&data).map_err(|err| format!("{}: {err}", e.name))?;
        assert_eq!(
            (img.width, img.height, img.format),
            (e.w, e.h, format_of(e)),
            "{}",
            e.name
        );
        let want = expected(e)?;
        assert_eq!(img.data.len(), want.len());
        let diffs = img.data.iter().zip(&want).map(|(a, b)| a.abs_diff(*b));
        let max = diffs.clone().max().unwrap_or(0);
        let mean = diffs.map(f64::from).sum::<f64>() / want.len() as f64;
        println!("{:24} max {max:3} mean {mean:.4}", e.name);
        let (bmax, bmean) = bound(e);
        assert!(
            max <= bmax && mean <= bmean,
            "{}: max {max} mean {mean}",
            e.name
        );
        assert_eq!(&file::decode(&data)?.data[..], &img.data[..]);
        assert_eq!(file::decode_animation(&data)?.frames.len(), 1);
    }
    Ok(())
}

#[test]
fn exif_orientation_is_metadata_only() -> R {
    for e in manifest()? {
        let Ok((img, meta)) = jpeg::decode_with_metadata(&read(&e.name)?) else {
            continue;
        };
        let want: u8 = match e.kv.get("orientation") {
            Some(v) => v.parse()?,
            None => 1,
        };
        assert_eq!(meta.orientation, want, "{}", e.name);
        assert_eq!(meta.exif.is_some(), want != 1, "{}", e.name);
        if let Some(exif) = &meta.exif {
            assert_eq!(jpeg::exif_orientation(exif), Some(want));
            // Pixels are not rotated.
            assert_eq!((img.width, img.height), (e.w, e.h));
        }
    }
    // Parser edge cases: empty, no entries, LE / BE (with prefix), out-of-range value.
    assert_eq!(jpeg::exif_orientation(b""), None);
    assert_eq!(jpeg::exif_orientation(b"II*\0\x08\0\0\0"), None);
    assert_eq!(
        jpeg::exif_orientation(b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x08\0\0\0"),
        Some(8)
    );
    assert_eq!(
        jpeg::exif_orientation(b"Exif\0\0MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0\x05\0\0"),
        Some(5)
    );
    assert_eq!(
        jpeg::exif_orientation(b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x09\0\0\0"),
        None
    );
    Ok(())
}

#[test]
fn unsupported_and_no_encoder() -> R {
    for e in manifest()?
        .iter()
        .filter(|e| e.kv.contains_key("unsupported"))
    {
        assert_eq!(
            jpeg::decode(&read(&e.name)?).err(),
            Some(imgio::Error::Unsupported),
            "{}",
            e.name
        );
    }
    let px = [0u8; 12];
    let s = Surface::new(Format::Rgb8Srgb, 2, 2, &px)?;
    assert_eq!(
        jpeg::encode(&s, &jpeg::EncodeOptions::default()).err(),
        Some(imgio::Error::Unsupported)
    );
    assert_eq!(
        file::encode(FileFormat::Jpeg, &s).err(),
        Some(imgio::Error::Unsupported)
    );
    Ok(())
}

/// Truncated and bit-flipped files return `Err` or `Ok`, never panic.
#[test]
fn corrupt_input_never_panics() -> R {
    for e in manifest()? {
        let data = read(&e.name)?;
        let step = (data.len() / 97).max(1);
        for len in (0..data.len())
            .step_by(step)
            .chain([data.len() - 1, 2, 3, 4, 20])
        {
            let _ = jpeg::decode(&data[..len.min(data.len())]);
        }
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        for i in 0..300 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let mut d = data.clone();
            // The headers (first ~700 bytes) more often than the entropy-coded data.
            let span = if i % 2 == 0 {
                data.len().min(700)
            } else {
                data.len()
            };
            d[(rng as usize >> 8) % span] ^= 1 << ((rng >> 3) & 7);
            if i % 5 == 0 {
                d[(rng as usize >> 20) % data.len()] = (rng >> 40) as u8;
            }
            let _ = jpeg::decode(&d);
        }
    }
    assert!(jpeg::decode(&[0xFF, 0xD8, 0xFF]).is_err());
    assert!(jpeg::decode(&[0xFF, 0xD8, 0xFF, 0xC0, 0, 2, 0xFF, 0xD9]).is_err());
    Ok(())
}
