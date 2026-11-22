//! AVIF decode / encode against dav1d and libavif 1.4 (testdata/file/avif, made by
//! scripts/gen_file_testdata_webjpgavif.py with imagecodecs and ffmpeg libaom-av1): YUV planes
//! exact vs dav1d, RGB exact vs libavif's built-in conversion (bounded where libavif used
//! libyuv), encoder output verified in libavif.
#![cfg(feature = "avif")]

use imgio::file::{self, FileFormat, avif};
use imgio::{Error as ImgError, Format, Surface};
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;

type R<T = ()> = Result<T, Box<dyn Error>>;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/file/avif")
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
fn manifest(file: &str) -> R<Vec<Entry>> {
    let text = String::from_utf8(read(file)?)?;
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

fn kv<'a>(e: &'a Entry, key: &str) -> R<&'a str> {
    Ok(e.kv
        .get(key)
        .ok_or_else(|| format!("{}: no {key}", e.name))?)
}

/// Samples of a raw buffer (u8, or u16 little endian above 8 bits).
fn samples(raw: &[u8], bits: u32) -> Vec<u32> {
    if bits > 8 {
        raw.as_chunks::<2>()
            .0
            .iter()
            .map(|b| u32::from(u16::from_le_bytes(*b)))
            .collect()
    } else {
        raw.iter().map(|&b| u32::from(b)).collect()
    }
}

/// imgio's output as samples at depth `bits` (16-bit output scaled back, exact for 10 / 12).
fn ours_at_depth(img: &file::Image, bits: u32) -> Vec<u32> {
    let max = (1u32 << bits) - 1;
    let wide = img.format == Format::Rgba16Unorm;
    samples(&img.data, if wide { 16 } else { 8 })
        .into_iter()
        .map(|v| if wide { (v * max + 32767) / 65535 } else { v })
        .collect()
}

#[test]
fn planes_match_dav1d_exactly() -> R {
    for e in manifest("manifest.txt")? {
        let data = read(&e.name)?;
        assert_eq!(
            FileFormat::detect(&data),
            Some(FileFormat::Avif),
            "{}",
            e.name
        );
        let p = avif::decode_planes(&data).map_err(|err| format!("{}: {err}", e.name))?;
        assert_eq!(
            (p.width, p.height, u32::from(p.depth)),
            (e.w, e.h, e.bits),
            "{}",
            e.name
        );
        let sub = match p.subsampling {
            avif::Subsampling::Yuv420 => "420",
            avif::Subsampling::Yuv422 => "422",
            avif::Subsampling::Yuv444 => "444",
            avif::Subsampling::Mono => "400",
        };
        assert_eq!(sub, kv(&e, "sub")?, "{}", e.name);
        assert_eq!(p.full_range, kv(&e, "range")? == "full", "{}", e.name);
        assert_eq!(p.matrix.to_string(), kv(&e, "matrix")?, "{}", e.name);
        let yuv = read(&e.name.replace(".avif", ".yuv"))?;
        assert!(
            [&p.y[..], &p.u[..], &p.v[..]].concat() == yuv,
            "{}: planes differ from dav1d",
            e.name
        );
        // Alpha: libavif's alpha channel (full range, same depth).
        match &p.alpha {
            Some(a) => {
                assert_eq!(e.ch, 4, "{}", e.name);
                let reference = samples(&read(&e.name.replace(".avif", ".ref"))?, e.bits);
                let want: Vec<u32> = reference
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|px| px[3])
                    .collect();
                assert_eq!(samples(a, e.bits), want, "{}: alpha", e.name);
            }
            None => assert_eq!(e.ch, 3, "{}", e.name),
        }
    }
    Ok(())
}

/// Max difference to libavif's RGB (imagecodecs) at the file's depth.
///
/// 0 where libavif uses its built-in float conversion (10 / 12 bits, YCgCo, identity,
/// monochrome): imgio reproduces it bit for bit. 8-bit YCbCr goes through libyuv in this
/// libavif build (integer arithmetic): up to 2 off, and up to 11 for limited-range BT.709,
/// where libyuv's result also disagrees with a float64 evaluation of the BT.709 formulas
/// (imgio matches that within rounding; the 10-bit limited-range files are exact). libyuv's
/// bilinear chroma filter also treats the last column of odd-width 4:2:x images differently,
/// so that column is skipped there.
fn bound(e: &Entry) -> R<u32> {
    let (sub, matrix) = (kv(e, "sub")?, kv(e, "matrix")?);
    let libyuv = e.bits == 8 && sub != "400" && !matches!(matrix, "0" | "8");
    Ok(match (libyuv, kv(e, "range")?, matrix) {
        (false, ..) => 0,
        (true, "limited", "1") => 11,
        _ => 2,
    })
}

#[test]
fn rgb_matches_libavif() -> R {
    for e in manifest("manifest.txt")? {
        let data = read(&e.name)?;
        let img = avif::decode(&data).map_err(|err| format!("{}: {err}", e.name))?;
        let fmt = match (e.bits, e.ch) {
            (8, 3) => Format::Rgb8Srgb,
            (8, _) => Format::Rgba8Srgb,
            _ => Format::Rgba16Unorm,
        };
        assert_eq!(
            (img.width, img.height, img.format),
            (e.w, e.h, fmt),
            "{}",
            e.name
        );
        let ours = ours_at_depth(&img, e.bits);
        let mut want = samples(&read(&e.name.replace(".avif", ".ref"))?, e.bits);
        if e.bits > 8 && e.ch == 3 {
            let max = (1u32 << e.bits) - 1;
            want = want
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2], max])
                .collect();
        }
        assert_eq!(ours.len(), want.len(), "{}", e.name);
        let bound = bound(&e)?;
        let ch = ours.len() / (e.w * e.h) as usize;
        let skip_last = bound > 0 && e.w % 2 == 1 && kv(&e, "sub")? != "444";
        let mut max = 0;
        for (i, (a, b)) in ours.iter().zip(&want).enumerate() {
            let x = (i / ch) as u32 % e.w;
            if !(skip_last && x == e.w - 1) {
                max = max.max(a.abs_diff(*b));
            }
        }
        println!("{:24} max {max}", e.name);
        assert!(max <= bound, "{}: max diff {max}", e.name);
        assert_eq!(&file::decode(&data)?.data[..], &img.data[..]);
        assert_eq!(file::decode_animation(&data)?.frames.len(), 1);
    }
    Ok(())
}

/// Sources encoded by `dump_encoded` / checked against `encoded.txt`: (name, channels, format).
const ENCODE_CASES: [(&str, usize, Format); 3] = [
    ("rgb_37x29", 3, Format::Rgb8Srgb),
    ("rgba_64x48", 4, Format::Rgba8Srgb),
    ("mosaic_rgb_50x38", 3, Format::Rgb8Unorm),
];

fn dims(name: &str) -> R<(u32, u32)> {
    let d = name.rsplit('_').next().ok_or("no size")?;
    let (w, h) = d.split_once('x').ok_or("no size")?;
    Ok((w.parse()?, h.parse()?))
}

/// testdata/images/raw/<name>.rgba reduced to `ch` channels.
fn source(name: &str, ch: usize) -> R<Vec<u8>> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("testdata/images/raw/{name}.rgba"));
    Ok(std::fs::read(p)?
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| p[..ch].to_vec())
        .collect())
}

/// Encodes a source with speed 8. ravif splits into tiles only when an image is larger than its
/// minimum tile size, so the output of these small images does not depend on the thread count.
fn encode_case(name: &str, ch: usize, fmt: Format) -> R<Vec<u8>> {
    let (w, h) = dims(name)?;
    let px = source(name, ch)?;
    let opts = avif::EncodeOptions {
        speed: 8,
        ..Default::default()
    };
    Ok(avif::encode(&Surface::new(fmt, w, h, &px)?, &opts)?.to_vec())
}

/// PSNR over the color channels of pixels with alpha > 0 (ravif cleans fully transparent
/// colors by default).
fn psnr(a: &[u8], b: &[u8], ch: usize) -> f64 {
    let (mut se, mut n) = (0.0, 0usize);
    for (p, q) in a.chunks_exact(ch).zip(b.chunks_exact(ch)) {
        if ch == 4 && q[3] == 0 {
            continue;
        }
        for c in 0..3 {
            se += (f64::from(p[c]) - f64::from(q[c])).powi(2);
            n += 1;
        }
    }
    let mse = se / n as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

#[test]
fn encode_round_trips() -> R {
    for (name, ch, fmt) in ENCODE_CASES {
        let data = encode_case(name, ch, fmt)?;
        assert_eq!(FileFormat::detect(&data), Some(FileFormat::Avif));
        let img = avif::decode(&data)?;
        let px = source(name, ch)?;
        let (w, h) = dims(name)?;
        assert_eq!(
            (img.width, img.height, img.format),
            (w, h, Format::Rgba16Unorm),
            "{name}"
        );
        let ours: Vec<u8> = ours_at_depth(&img, 8)
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| p[..ch].iter().map(|&v| v as u8).collect::<Vec<u8>>())
            .collect();
        // ravif's quality 80 on these noisy synthetic images: ~30-31 dB (libavif agrees).
        let q = psnr(&ours, &px, ch);
        println!("{name}: PSNR {q:.2}");
        assert!(q > 28.0, "{name}: PSNR {q:.2}");
        if ch == 4 {
            let alpha = |v: &[u8]| {
                v.as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| [p[3]; 3])
                    .collect::<Vec<u8>>()
            };
            let qa = psnr(&alpha(&ours), &alpha(&px), 3);
            println!("{name}: alpha PSNR {qa:.2}");
            assert!(qa > 35.0, "{name}: alpha PSNR {qa:.2}");
        }
    }
    // 8-bit files, gray inputs, BGRA, 1x1, options and errors.
    let px = source("rgba_64x48", 4)?;
    let rgba = px.as_chunks::<4>().0;
    let gray: Vec<u8> = rgba.iter().map(|p| p[1]).collect();
    let opts = avif::EncodeOptions {
        bit_depth: 8,
        quality: 90.0,
        speed: 10,
        ..Default::default()
    };
    let img = avif::decode(&avif::encode(
        &Surface::new(Format::R8Unorm, 64, 48, &gray)?,
        &opts,
    )?)?;
    assert_eq!(img.format, Format::Rgb8Srgb);
    let rgb = |v: &[u8]| v.iter().flat_map(|&v| [v, v, v]).collect::<Vec<u8>>();
    let g: Vec<u8> = img.data.as_chunks::<3>().0.iter().map(|p| p[1]).collect();
    let q = psnr(&rgb(&g), &rgb(&gray), 3);
    println!("gray 8-bit q90: PSNR {q:.2}");
    assert!(q > 28.0, "gray: PSNR {q:.2}");
    let bgra: Vec<u8> = rgba.iter().flat_map(|p| [p[2], p[1], p[0], p[3]]).collect();
    let img = avif::decode(&file::encode(
        FileFormat::Avif,
        &Surface::new(Format::Bgra8Unorm, 64, 48, &bgra)?,
    )?)?;
    assert_eq!(img.format, Format::Rgba16Unorm);
    let ga: Vec<u8> = rgba.iter().flat_map(|p| [p[0], p[3]]).collect();
    let img = avif::decode(&avif::encode(
        &Surface::new(Format::Rg8Unorm, 64, 48, &ga)?,
        &opts,
    )?)?;
    assert_eq!(img.format, Format::Rgba8Srgb);
    let one = [10u8, 200, 30];
    let img = avif::decode(&avif::encode(
        &Surface::new(Format::Rgb8Srgb, 1, 1, &one)?,
        &opts,
    )?)?;
    assert_eq!((img.width, img.height), (1, 1));
    let s = Surface::new(Format::Rgba8Srgb, 64, 48, &px)?;
    for bad in [
        avif::EncodeOptions {
            quality: 0.0,
            ..Default::default()
        },
        avif::EncodeOptions {
            alpha_quality: 101.0,
            ..Default::default()
        },
        avif::EncodeOptions {
            quality: f32::NAN,
            ..Default::default()
        },
        avif::EncodeOptions {
            speed: 0,
            ..Default::default()
        },
        avif::EncodeOptions {
            speed: 11,
            ..Default::default()
        },
        avif::EncodeOptions {
            bit_depth: 12,
            ..Default::default()
        },
    ] {
        assert_eq!(
            avif::encode(&s, &bad).err(),
            Some(ImgError::InvalidArgument),
            "{bad:?}"
        );
    }
    let f = vec![0u8; 4 * 4 * 8];
    let s = Surface::new(Format::Rgba16Unorm, 4, 4, &f)?;
    assert_eq!(
        avif::encode(&s, &avif::EncodeOptions::default()).err(),
        Some(ImgError::Unsupported)
    );
    Ok(())
}

/// The stored `enc_*` files were written by imgio's encoder and checked against libavif
/// (imagecodecs, PSNR) by `scripts/gen_file_testdata_webjpgavif.py --encoded`; imgio decodes
/// them exactly like libavif (10-bit: built-in conversion) and still writes the same bytes.
#[test]
fn encode_matches_libavif_verified_files() -> R {
    let list = manifest("encoded.txt")?;
    assert_eq!(list.len(), ENCODE_CASES.len());
    for (name, ch, fmt) in ENCODE_CASES {
        let file = format!("enc_{name}.avif");
        let e = list
            .iter()
            .find(|e| e.name == file)
            .ok_or_else(|| format!("{file} not listed"))?;
        let data = read(&file)?;
        assert!(
            encode_case(name, ch, fmt)? == data,
            "{file}: encoder output changed, rerun the generator with --encoded"
        );
        let ours = ours_at_depth(&avif::decode(&data)?, e.bits);
        let mut want = samples(&read(&file.replace(".avif", ".ref"))?, e.bits);
        if e.ch == 3 {
            let max = (1 << e.bits) - 1;
            want = want
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2], max])
                .collect();
        }
        assert!(ours == want, "{file}");
    }
    Ok(())
}

/// Writes imgio's encoder output to $IMGIO_DUMP_DIR (used by the generator script).
#[test]
#[ignore]
fn dump_encoded() -> R {
    let Ok(dir) = std::env::var("IMGIO_DUMP_DIR") else {
        return Ok(());
    };
    for (name, ch, fmt) in ENCODE_CASES {
        std::fs::write(
            PathBuf::from(&dir).join(format!("enc_{name}.avif")),
            encode_case(name, ch, fmt)?,
        )?;
    }
    Ok(())
}

/// Truncated and bit-flipped files return `Err` or `Ok`, never panic.
#[test]
fn corrupt_input_never_panics() -> R {
    for e in manifest("manifest.txt")? {
        let data = read(&e.name)?;
        let step = (data.len() / 61).max(1);
        for len in (0..data.len())
            .step_by(step)
            .chain([data.len() - 1, 12, 32])
        {
            let _ = avif::decode(&data[..len.min(data.len())]);
        }
        let mut rng = 0xD1B5_4A32_D192_ED03u64;
        for i in 0..120 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let mut d = data.clone();
            // The container (first ~400 bytes) more often than the AV1 payload.
            let span = if i % 2 == 0 {
                data.len().min(400)
            } else {
                data.len()
            };
            d[(rng as usize >> 8) % span] ^= 1 << ((rng >> 3) & 7);
            if i % 5 == 0 {
                d[(rng as usize >> 20) % data.len()] = (rng >> 40) as u8;
            }
            let _ = avif::decode(&d);
        }
    }
    assert!(avif::decode(b"\0\0\0\x1cftypavif\0\0\0\0mif1miaf").is_err());
    Ok(())
}
