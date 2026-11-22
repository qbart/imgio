//! WebP decode / encode against libwebp 1.6 (testdata/file/webp, made by
//! scripts/gen_file_testdata_webjpgavif.py): stills and alpha variants exact vs libwebp's
//! decoder, animations exact vs Pillow's composited frames (libwebp `WebPAnimDecoder`),
//! lossless encodes exact in imgio and in Pillow.
#![cfg(feature = "webp")]

use imgio::file::{self, FileFormat, webp};
use imgio::{Format, Surface};
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;

type R<T = ()> = Result<T, Box<dyn Error>>;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/file/webp")
}

fn read(name: &str) -> R<Vec<u8>> {
    Ok(std::fs::read(dir().join(name)).map_err(|e| format!("{name}: {e}"))?)
}

struct Entry {
    name: String,
    w: u32,
    h: u32,
    ch: usize,
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

/// testdata/images/raw/<name>.rgba reduced to `ch` channels.
fn source(name: &str, ch: usize) -> R<Vec<u8>> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("testdata/images/raw/{name}.rgba"));
    let rgba = std::fs::read(p)?;
    Ok(rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| p[..ch].to_vec())
        .collect())
}

fn reference(e: &Entry) -> R<Vec<u8>> {
    match e.kv.get("source") {
        Some(s) => source(s, e.ch),
        None => read(&e.name.replace(".webp", ".ref")),
    }
}

fn max_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

fn rgb_format(ch: usize) -> Format {
    if ch == 4 {
        Format::Rgba8Srgb
    } else {
        Format::Rgb8Srgb
    }
}

#[test]
fn stills_match_libwebp_exactly() -> R {
    for e in manifest("manifest.txt")?
        .iter()
        .filter(|e| !e.kv.contains_key("frames"))
    {
        let data = read(&e.name)?;
        assert_eq!(
            FileFormat::detect(&data),
            Some(FileFormat::WebP),
            "{}",
            e.name
        );
        let img = webp::decode(&data).map_err(|err| format!("{}: {err}", e.name))?;
        assert_eq!(
            (img.width, img.height, img.format),
            (e.w, e.h, rgb_format(e.ch)),
            "{}",
            e.name
        );
        let want = reference(e)?;
        // image-webp ports libwebp's VP8 reconstruction, fancy upsampling and YUV -> RGB.
        assert!(
            img.data[..] == want[..],
            "{}: max diff {}",
            e.name,
            max_diff(&img.data, &want)
        );
        // The generic API agrees.
        assert_eq!(&file::decode(&data)?.data[..], &img.data[..]);
        assert_eq!(file::decode_animation(&data)?.frames.len(), 1);
    }
    Ok(())
}

#[test]
fn animations_match_libwebp_anim_decoder() -> R {
    for e in manifest("manifest.txt")?
        .iter()
        .filter(|e| e.kv.contains_key("frames"))
    {
        let data = read(&e.name)?;
        let anim = webp::decode_animation(&data).map_err(|err| format!("{}: {err}", e.name))?;
        let frames: usize = kv(e, "frames")?.parse()?;
        let delays = kv(e, "delays")?
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<u32>, _>>()?;
        assert_eq!(anim.frames.len(), frames, "{}", e.name);
        assert_eq!(anim.loops, kv(e, "loops")?.parse::<u32>()?, "{}", e.name);
        let want = reference(e)?;
        let size = (e.w * e.h * 4) as usize;
        for (i, f) in anim.frames.iter().enumerate() {
            assert_eq!(
                (f.image.width, f.image.height, f.image.format),
                (e.w, e.h, Format::Rgba8Srgb)
            );
            assert_eq!(f.delay, (delays[i], 1000), "{} frame {i}", e.name);
            let w = &want[i * size..(i + 1) * size];
            assert!(
                &f.image.data[..] == w,
                "{} frame {i}: max diff {}",
                e.name,
                max_diff(&f.image.data, w)
            );
        }
        // The main image is the first frame.
        assert_eq!(&webp::decode(&data)?.data[..], &want[..size], "{}", e.name);
        assert_eq!(file::decode_animation(&data)?.frames.len(), frames);
    }
    Ok(())
}

/// Sources encoded by `dump_encoded` / checked against `encoded.txt` (name, channels, format).
const ENCODE_CASES: [(&str, usize, Format); 4] = [
    ("mosaic_rgb_50x38", 3, Format::Rgb8Srgb),
    ("cutout_rgba_96x64", 4, Format::Rgba8Srgb),
    ("rgb_37x29", 3, Format::Rgb8Unorm),
    ("rgba_64x48", 4, Format::Rgba8Unorm),
];

fn dims(name: &str) -> R<(u32, u32)> {
    let d = name.rsplit('_').next().ok_or("no size")?;
    let (w, h) = d.split_once('x').ok_or("no size")?;
    Ok((w.parse()?, h.parse()?))
}

fn encode_case(name: &str, ch: usize, fmt: Format) -> R<Vec<u8>> {
    let (w, h) = dims(name)?;
    let px = source(name, ch)?;
    let s = Surface::new(fmt, w, h, &px)?;
    Ok(webp::encode(&s, &webp::EncodeOptions::default())?.to_vec())
}

#[test]
fn encode_round_trips_exactly() -> R {
    for (name, ch, fmt) in ENCODE_CASES {
        let data = encode_case(name, ch, fmt)?;
        assert_eq!(FileFormat::detect(&data), Some(FileFormat::WebP));
        let img = webp::decode(&data)?;
        let (w, h) = dims(name)?;
        assert_eq!(
            (img.width, img.height, img.format),
            (w, h, rgb_format(ch)),
            "{name}"
        );
        assert!(img.data[..] == source(name, ch)?[..], "{name}");
    }
    // Gray, gray + alpha and BGRA inputs.
    let (w, h) = (37, 29);
    let rgba = source("rgb_37x29", 4)?;
    let px = rgba.as_chunks::<4>().0;
    let gray: Vec<u8> = px.iter().map(|p| p[1]).collect();
    let ga: Vec<u8> = px.iter().flat_map(|p| [p[0], p[2]]).collect();
    let bgra: Vec<u8> = px.iter().flat_map(|p| [p[2], p[1], p[0], 128]).collect();
    let cases: [(Format, &[u8], Vec<u8>); 3] = [
        (
            Format::R8Unorm,
            &gray,
            gray.iter().flat_map(|&g| [g, g, g]).collect(),
        ),
        (
            Format::Rg8Unorm,
            &ga,
            ga.as_chunks::<2>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[0], p[0], p[1]])
                .collect(),
        ),
        (
            Format::Bgra8Unorm,
            &bgra,
            px.iter().flat_map(|p| [p[0], p[1], p[2], 128]).collect(),
        ),
    ];
    for (fmt, input, expect) in cases {
        let data = file::encode(FileFormat::WebP, &Surface::new(fmt, w, h, input)?)?;
        assert!(file::decode(&data)?.data[..] == expect[..], "{fmt:?}");
    }
    let opts = webp::EncodeOptions {
        predictor_transform: false,
    };
    let px = source("rgba_64x48", 4)?;
    let data = webp::encode(&Surface::new(Format::Rgba8Srgb, 64, 48, &px)?, &opts)?;
    assert_eq!(&webp::decode(&data)?.data[..], &px[..]);
    // Formats without a lossless 8-bit mapping.
    let f = vec![0u8; 4 * 4 * 8];
    let s = Surface::new(Format::Rgba16Unorm, 4, 4, &f)?;
    assert_eq!(
        webp::encode(&s, &webp::EncodeOptions::default()).err(),
        Some(imgio::Error::Unsupported)
    );
    Ok(())
}

/// imgio's encoder output is byte-identical to the files verified with Pillow (libwebp) by
/// `scripts/gen_file_testdata_webjpgavif.py --encoded`, which decoded them exactly.
#[test]
fn encode_matches_libwebp_verified_files() -> R {
    let list = manifest("encoded.txt")?;
    assert_eq!(list.len(), ENCODE_CASES.len());
    for (name, ch, fmt) in ENCODE_CASES {
        let file = format!("enc_{name}.webp");
        let e = list
            .iter()
            .find(|e| e.name == file)
            .ok_or_else(|| format!("{file} not listed"))?;
        assert_eq!(e.ch, ch);
        assert!(
            encode_case(name, ch, fmt)? == read(&file)?,
            "{file}: encoder output changed, rerun the generator with --encoded"
        );
        assert_eq!(
            read(&file.replace(".webp", ".ref"))?,
            source(name, ch)?,
            "{file}"
        );
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
            PathBuf::from(&dir).join(format!("enc_{name}.webp")),
            encode_case(name, ch, fmt)?,
        )?;
    }
    Ok(())
}

/// Truncated and bit-flipped files return `Err` or `Ok`, never panic.
#[test]
fn corrupt_input_never_panics() -> R {
    let mut names: Vec<String> = manifest("manifest.txt")?
        .into_iter()
        .map(|e| e.name)
        .collect();
    names.push("enc_cutout_rgba_96x64.webp".into());
    for name in names {
        let data = read(&name)?;
        let step = (data.len() / 97).max(1);
        for len in (0..data.len())
            .step_by(step)
            .chain([data.len() - 1, 11, 12, 20, 30])
        {
            let d = &data[..len.min(data.len())];
            let _ = webp::decode(d);
            let _ = webp::decode_animation(d);
        }
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for i in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let mut d = data.clone();
            // Headers more often than payload.
            let span = if i % 2 == 0 {
                data.len().min(64)
            } else {
                data.len()
            };
            d[(rng as usize >> 8) % span] ^= 1 << ((rng >> 3) & 7);
            if i % 5 == 0 {
                d[(rng as usize >> 20) % data.len()] = (rng >> 40) as u8;
            }
            let _ = webp::decode(&d);
            let _ = webp::decode_animation(&d);
        }
    }
    assert!(webp::decode(b"RIFF\x04\0\0\0WEBP").is_err());
    assert!(webp::decode(b"RIFF\xff\xff\xff\xffWEBPVP8 ").is_err());
    Ok(())
}
