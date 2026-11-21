//! Shared helpers of the BMP / TGA / HDR / zfile tests (testdata/file/<fmt>, made by
//! scripts/gen_file_testdata_own.py).
#![allow(dead_code)]

use imgio::file::{FileFormat, Image};
use imgio::{Format, Surface};
use std::error::Error;

pub type R<T = ()> = Result<T, Box<dyn Error>>;

pub fn dir(fmt: &str) -> String {
    format!("{}/testdata/file/{fmt}", env!("CARGO_MANIFEST_DIR"))
}

pub fn format_by_name(s: &str) -> R<Format> {
    Ok(match s {
        "R8Unorm" => Format::R8Unorm,
        "Rg8Unorm" => Format::Rg8Unorm,
        "Rgb8Srgb" => Format::Rgb8Srgb,
        "Rgba8Srgb" => Format::Rgba8Srgb,
        "Rgba32Float" => Format::Rgba32Float,
        _ => return Err(format!("unknown format {s}").into()),
    })
}

/// One line of `cases.txt`.
pub struct Case {
    pub file: String,
    pub data: Vec<u8>,
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub expected: Vec<u8>,
    pub reference: String,
}

pub fn cases(fmt: &str) -> R<Vec<Case>> {
    let d = dir(fmt);
    let list = std::fs::read_to_string(format!("{d}/cases.txt"))?;
    let mut out = Vec::new();
    for l in list
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = l.split_whitespace().collect();
        let [file, format, w, h, exp, reference] = f[..] else {
            return Err(format!("bad line {l}").into());
        };
        out.push(Case {
            file: file.into(),
            data: std::fs::read(format!("{d}/{file}"))?,
            format: format_by_name(format)?,
            width: w.parse()?,
            height: h.parse()?,
            expected: std::fs::read(format!("{d}/expected/{exp}"))?,
            reference: reference.into(),
        });
    }
    assert!(!out.is_empty());
    Ok(out)
}

/// Every case decodes (through detection and the module decoder) to its expected pixels, and
/// `FileFormat::detect` names the format.
pub fn check_cases(fmt: &str, ff: FileFormat, decode: fn(&[u8]) -> imgio::Result<Image>) -> R {
    for c in cases(fmt)? {
        assert_eq!(FileFormat::detect(&c.data), Some(ff), "{}", c.file);
        for img in [decode(&c.data)?, imgio::file::decode(&c.data)?] {
            assert_eq!(
                (img.format, img.width, img.height),
                (c.format, c.width, c.height),
                "{} ({})",
                c.file,
                c.reference
            );
            let first = img.data.iter().zip(&c.expected).position(|(a, b)| a != b);
            assert!(
                img.data[..] == c.expected[..],
                "{} ({}): pixels differ at byte {first:?}",
                c.file,
                c.reference
            );
        }
    }
    Ok(())
}

/// Truncations and byte flips of `data` never panic (any `Ok` / `Err` is fine).
pub fn fuzz(data: &[u8], decode: fn(&[u8]) -> imgio::Result<Image>) {
    let n = data.len();
    let mut lens: Vec<usize> = (0..n.min(400)).collect();
    lens.extend((400..n).step_by((n / 97).max(1)));
    for l in lens {
        let _ = decode(&data[..l]);
    }
    let mut buf = data.to_vec();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ n as u64;
    for i in 0..400 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Every header byte, then random positions.
        let pos = if i < 160.min(n) {
            i
        } else {
            (seed >> 33) as usize % n
        };
        for x in [0xFFu8, 0x80, 0x01, (seed >> 20) as u8 | 1] {
            buf[pos] ^= x;
            let _ = decode(&buf);
            buf[pos] ^= x;
        }
    }
}

/// testdata/images/raw RGBA8 image.
pub fn raw_rgba(name: &str) -> R<Vec<u8>> {
    Ok(std::fs::read(format!(
        "{}/testdata/images/raw/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))?)
}

/// The encoder sources of the `enc/` golden files: `rgb` (rgb_37x29), `rgba`
/// (mosaic_rgba_50x38), `gray` (red channel of rgb_37x29).
pub fn ldr_source(kind: &str) -> R<(Format, u32, u32, Vec<u8>)> {
    let rgb = raw_rgba("rgb_37x29.rgba")?;
    Ok(match kind {
        "rgb" => (
            Format::Rgb8Srgb,
            37,
            29,
            rgb.as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect(),
        ),
        "gray" => (
            Format::R8Unorm,
            37,
            29,
            rgb.as_chunks::<4>().0.iter().map(|p| p[0]).collect(),
        ),
        _ => (
            Format::Rgba8Srgb,
            50,
            38,
            raw_rgba("mosaic_rgba_50x38.rgba")?,
        ),
    })
}

/// Compares encoder output with the golden file `<fmt>/enc/<name>` (checked by the reference
/// decoders when generated); with IMGIO_WRITE_ENCODED=1 writes it instead.
pub fn golden_encoded(fmt: &str, name: &str, data: &[u8]) -> R {
    let path = format!("{}/enc/{name}", dir(fmt));
    if std::env::var("IMGIO_WRITE_ENCODED").is_ok() {
        std::fs::create_dir_all(format!("{}/enc", dir(fmt)))?;
        std::fs::write(&path, data)?;
        return Ok(());
    }
    let want = std::fs::read(&path)?;
    assert!(want == data, "{fmt}/enc/{name}: encoder output changed");
    Ok(())
}

pub fn surface(img: &Image) -> R<Surface<'_>> {
    Ok(img.surface()?)
}
