//! JPEG XL: decode vs libjxl (files and decodes from `scripts/gen_file_testdata_jxlj2kvdb.py`),
//! lossless encode round trips, hostile input.
#![cfg(feature = "jxl")]

use imgio::file::{self, FileFormat, jxl};
use imgio::{Error, Format, Surface};
use std::path::PathBuf;

type Res<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/file/jxl")
}

fn read(name: &str) -> Res<Vec<u8>> {
    Ok(std::fs::read(dir().join(name))?)
}

/// (name, check) from meta.txt; check is "exact" or "max N".
fn cases() -> Res<Vec<(String, Option<u32>)>> {
    let meta = std::fs::read_to_string(dir().join("meta.txt"))?;
    let mut out = Vec::new();
    for l in meta.lines() {
        let mut it = l.split_whitespace();
        let name = it.next().ok_or("name")?.to_string();
        let max = match it.next().ok_or("check")? {
            "exact" => None,
            "max" => Some(it.next().ok_or("bound")?.parse()?),
            c => return Err(format!("bad check {c}").into()),
        };
        out.push((name, max));
    }
    Ok(out)
}

fn expected_format(name: &str) -> Format {
    let n = name.trim_start_matches("lossy_");
    if n.starts_with("rgba8") || n.starts_with("anim_rgba8") {
        Format::Rgba8Srgb
    } else if n.starts_with("rgb8") {
        Format::Rgb8Srgb
    } else if n.starts_with("graya8") {
        Format::Rg8Unorm
    } else if n.starts_with("gray8") {
        Format::R8Unorm
    } else if n.starts_with("rgba16") || n.starts_with("rgb16") {
        Format::Rgba16Unorm
    } else if n.starts_with("graya16") {
        Format::Rg16Unorm
    } else if n.starts_with("gray16") {
        Format::R16Unorm
    } else if n.starts_with("gray_f32") {
        Format::R32Float
    } else {
        Format::Rgba32Float
    }
}

/// Max absolute difference in sample units (8/16-bit integers; floats must match exactly).
fn max_diff(format: Format, a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    match format {
        Format::R16Unorm | Format::Rg16Unorm | Format::Rgba16Unorm => a
            .as_chunks::<2>()
            .0
            .iter()
            .zip(b.as_chunks::<2>().0.iter())
            .map(|(x, y)| {
                u16::from_le_bytes([x[0], x[1]]).abs_diff(u16::from_le_bytes([y[0], y[1]])) as u32
            })
            .max()
            .unwrap_or(0),
        Format::R32Float | Format::Rgba32Float => {
            if a == b {
                0
            } else {
                u32::MAX
            }
        }
        _ => a
            .iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y) as u32)
            .max()
            .unwrap_or(0),
    }
}

#[test]
fn decode_matches_libjxl() -> Res {
    let cases = cases()?;
    assert_eq!(cases.len(), 17);
    for (name, max) in cases {
        let data = read(&format!("{name}.jxl"))?;
        let want = read(&format!("{name}.raw"))?;
        assert_eq!(
            FileFormat::detect(&data),
            Some(FileFormat::JpegXl),
            "{name}"
        );
        let img = file::decode(&data)?;
        assert_eq!(img.format, expected_format(&name), "{name}");
        let first = if name.starts_with("anim") {
            &want[..want.len() / 3]
        } else {
            &want[..]
        };
        let d = max_diff(img.format, &img.data, first);
        let bound = max.unwrap_or(0);
        println!("{name}: max diff {d} (bound {bound})");
        assert!(d <= bound, "{name}: max diff {d} > {bound}");
    }
    Ok(())
}

#[test]
fn info_flags() -> Res {
    let i = jxl::info(&read("rgb8_container.jxl")?)?;
    assert!(i.container && !i.xyb && !i.float && !i.alpha && !i.gray);
    assert_eq!((i.width, i.height, i.bits_per_sample), (37, 29, 8));
    let i = jxl::info(&read("rgba8.jxl")?)?;
    assert!(!i.container && i.alpha && !i.animated);
    let i = jxl::info(&read("lossy_rgba8.jxl")?)?;
    assert!(i.container && i.xyb && i.alpha);
    let i = jxl::info(&read("graya16.jxl")?)?;
    assert!(i.gray && i.alpha && i.bits_per_sample == 16);
    let i = jxl::info(&read("rgba_f32.jxl")?)?;
    assert!(i.float && i.bits_per_sample == 32);
    let i = jxl::info(&read("rgb_f16.jxl")?)?;
    assert!(i.float && i.bits_per_sample == 16);
    assert!(jxl::info(&read("anim_rgba8.jxl")?)?.animated);
    Ok(())
}

#[test]
fn animation() -> Res {
    let data = read("anim_rgba8.jxl")?;
    let want = read("anim_rgba8.raw")?;
    for anim in [
        file::decode_animation(&data)?,
        jxl::decode_animation(&data)?,
    ] {
        assert_eq!(anim.frames.len(), 3);
        assert_eq!(anim.loops, 0);
        for (i, f) in anim.frames.iter().enumerate() {
            assert_eq!((f.image.width, f.image.height), (24, 16));
            assert_eq!(f.image.format, Format::Rgba8Srgb);
            assert_eq!(
                &f.image.data[..],
                &want[i * 1536..(i + 1) * 1536],
                "frame {i}"
            );
            // Frame duration / ticks per second as libjxl (imagecodecs) writes them.
            assert_eq!(f.delay, (1, 10), "frame {i}");
        }
    }
    // A still image is one frame.
    let still = file::decode_animation(&read("rgba8.jxl")?)?;
    assert_eq!(still.frames.len(), 1);
    assert_eq!(still.frames[0].delay, (0, 1));
    Ok(())
}

fn raw_rgba(name: &str, w: usize, h: usize) -> Res<Vec<u8>> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/images/raw")
        .join(name);
    let v = std::fs::read(p)?;
    assert_eq!(v.len(), w * h * 4);
    Ok(v)
}

/// Source surfaces for the encoder: (format, w, h, bytes).
/// (format, width, height, pixels).
type Source = (Format, u32, u32, Vec<u8>);

fn encode_sources() -> Res<Vec<Source>> {
    let rgba = raw_rgba("rgba_64x48.rgba", 64, 48)?;
    let px = || rgba.as_chunks::<4>().0.iter();
    let wide = |v: u8, i: usize| (v as u16) << 8 | ((i * 37) & 0xFF) as u16;
    let le16 = |it: Vec<u16>| it.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let rgb: Vec<u8> = px().flat_map(|p| [p[0], p[1], p[2]]).collect();
    let gray: Vec<u8> = px().map(|p| p[1]).collect();
    let ga: Vec<u8> = px().flat_map(|p| [p[1], p[3]]).collect();
    let bgra: Vec<u8> = px().flat_map(|p| [p[2], p[1], p[0], p[3]]).collect();
    let g16 = le16(px().enumerate().map(|(i, p)| wide(p[1], i)).collect());
    let ga16 = le16(
        px().enumerate()
            .flat_map(|(i, p)| [wide(p[1], i), wide(p[3], i + 1)])
            .collect(),
    );
    let rgba16 = le16(rgba.iter().enumerate().map(|(i, &v)| wide(v, i)).collect());
    Ok(vec![
        (Format::R8Unorm, 64, 48, gray),
        (Format::Rg8Unorm, 64, 48, ga),
        (Format::Rgb8Srgb, 64, 48, rgb.clone()),
        (Format::Rgb8Unorm, 64, 48, rgb),
        (Format::Rgba8Srgb, 64, 48, rgba.clone()),
        (Format::Bgra8Unorm, 64, 48, bgra),
        (Format::R16Unorm, 64, 48, g16),
        (Format::Rg16Unorm, 64, 48, ga16),
        (Format::Rgba16Unorm, 64, 48, rgba16),
        // Odd size, several 256x256 groups would be too big for a test file; 2x2 minimum.
        (Format::Rgba8Srgb, 2, 2, rgba[..16].to_vec()),
    ])
}

/// What our decoder returns for an encoded `format`, and the expected bytes.
fn decoded_form(format: Format, src: &[u8]) -> (Format, Vec<u8>) {
    match format {
        Format::Rgb8Unorm | Format::Rgb8Srgb => (Format::Rgb8Srgb, src.to_vec()),
        Format::Rgba8Unorm | Format::Rgba8Srgb => (Format::Rgba8Srgb, src.to_vec()),
        Format::Bgra8Unorm => (
            Format::Rgba8Srgb,
            src.as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| [p[2], p[1], p[0], p[3]])
                .collect(),
        ),
        // Stored as RGBA (R = G = B = gray).
        Format::Rg16Unorm => (
            Format::Rgba16Unorm,
            src.as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[0], p[1], p[0], p[1], p[2], p[3]])
                .collect(),
        ),
        f => (f, src.to_vec()),
    }
}

#[test]
fn encode_round_trip() -> Res {
    let dump = std::env::var_os("IMGIO_JXL_ENC_DUMP").map(PathBuf::from);
    for (i, (format, w, h, src)) in encode_sources()?.into_iter().enumerate() {
        let s = Surface::new(format, w, h, &src)?;
        for opts in [
            jxl::EncodeOptions::default(),
            jxl::EncodeOptions {
                effort: 0,
                threads: 0,
            },
        ] {
            let enc = jxl::encode(&s, &opts)?;
            assert_eq!(FileFormat::detect(&enc), Some(FileFormat::JpegXl));
            let img = file::decode(&enc)?;
            let (want_format, want) = decoded_form(format, &src);
            assert_eq!(img.format, want_format, "{format:?}");
            assert_eq!((img.width, img.height), (w, h));
            assert!(img.data[..] == want[..], "{format:?} {w}x{h}");
            if let Some(d) = &dump {
                std::fs::create_dir_all(d)?;
                let stem = format!("{i:02}_{format:?}_e{}", opts.effort);
                std::fs::write(d.join(format!("{stem}.jxl")), &enc[..])?;
                // libjxl returns 8-bit interleaved / 16-bit samples without padding.
                std::fs::write(d.join(format!("{stem}.src")), &want)?;
            }
        }
    }
    let enc = file::encode(
        FileFormat::JpegXl,
        &Surface::new(Format::R8Unorm, 2, 2, &[1, 2, 3, 4])?,
    )?;
    assert_eq!(&file::decode(&enc)?.data[..], &[1, 2, 3, 4]);
    Ok(())
}

#[test]
fn encode_unsupported() -> Res {
    let f32px = [0u8; 16 * 4];
    let s = Surface::new(Format::Rgba32Float, 2, 2, &f32px)?;
    assert_eq!(
        jxl::encode(&s, &Default::default()).err(),
        Some(Error::Unsupported)
    );
    let s = Surface::new(Format::Rgba8Srgb, 1, 4, &f32px[..16])?;
    assert_eq!(
        jxl::encode(&s, &Default::default()).err(),
        Some(Error::Unsupported)
    );
    let s = Surface::new(Format::Bc1Unorm, 4, 4, &f32px[..8])?;
    assert_eq!(
        jxl::encode(&s, &Default::default()).err(),
        Some(Error::Unsupported)
    );
    Ok(())
}

#[test]
fn limits() -> Res {
    let data = read("rgba8.jxl")?;
    let opts = jxl::DecodeOptions {
        max_pixels: 64 * 48 - 1,
        ..Default::default()
    };
    assert_eq!(
        jxl::decode_with(&data, &opts).err(),
        Some(Error::Unsupported)
    );
    let opts = jxl::DecodeOptions {
        max_alloc_bytes: 1024,
        ..Default::default()
    };
    assert!(jxl::decode_with(&data, &opts).is_err());
    Ok(())
}

#[test]
fn hostile_input_never_panics() -> Res {
    // More iterations: IMGIO_JXL_FUZZ=<n>.
    let iters: usize = std::env::var("IMGIO_JXL_FUZZ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    for (name, _) in cases()? {
        let data = read(&format!("{name}.jxl"))?;
        let step = (data.len() / 61).max(1);
        for len in (0..data.len()).step_by(step) {
            assert!(
                jxl::decode(&data[..len]).is_err(),
                "{name} truncated at {len}"
            );
            let _ = jxl::decode_animation(&data[..len]);
        }
        let mut seed = 0x9E37_79B9u32 ^ data.len() as u32;
        for _ in 0..iters {
            let mut d = data.clone();
            for _ in 0..3 {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let at = seed as usize % d.len();
                d[at] ^= 1 << (seed >> 29);
            }
            let _ = jxl::decode(&d);
        }
    }
    for junk in [
        &b""[..],
        &[0xFF, 0x0A],
        &[0xFF, 0x0A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
    ] {
        assert!(jxl::decode(junk).is_err());
    }
    Ok(())
}
