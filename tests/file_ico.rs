//! ICO / CUR against Pillow (testdata/file/ico, made by scripts/gen_file_testdata_ico.py):
//! every entry decodes exactly like Pillow's `IcoFile.frame`, encoder output is decoded
//! exactly by Pillow (`encoded_verified.txt`, `--verify` mode of the script).
#![cfg(feature = "png")]

use imgio::file::{self, FileFormat, Image, ico};
use imgio::{Error, Format, Surface};
use std::error::Error as StdError;
use std::path::PathBuf;

type R<T = ()> = Result<T, Box<dyn StdError>>;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/file/ico")
}

fn read(name: &str) -> R<Vec<u8>> {
    Ok(std::fs::read(dir().join(name)).map_err(|e| format!("{name}: {e}"))?)
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, &x| {
        (h ^ u64::from(x)).wrapping_mul(0x100_0000_01B3)
    })
}

/// RGBA8 bytes of a decoded entry (PNG entries keep their native format).
fn rgba(img: &Image) -> R<Vec<u8>> {
    Ok(match img.format {
        Format::Rgba8Srgb | Format::Rgba8Unorm => img.data.to_vec(),
        Format::Rgb8Srgb => img
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        f => return Err(format!("format {f:?}").into()),
    })
}

#[test]
fn decode_matches_pillow() -> R {
    let text = String::from_utf8(read("cases.txt")?)?;
    let mut files = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let name = f[0];
        let n: usize = f[1].parse()?;
        let largest: usize = f[2].parse()?;
        let data = read(name)?;
        assert_eq!(FileFormat::detect(&data), Some(FileFormat::Ico), "{name}");
        let inf = ico::info(&data)?;
        assert_eq!(inf.entries.len(), n, "{name}");
        assert_eq!(inf.cursor, name.ends_with(".cur"));
        assert_eq!(inf.largest(), Some(largest), "{name}");
        for i in 0..n {
            let e = inf.entries[i];
            let v: Vec<u32> = f[3 + i * 6..9 + i * 6]
                .iter()
                .map(|s| s.parse())
                .collect::<Result<_, _>>()?;
            assert_eq!((e.width, e.height, u32::from(e.bits)), (v[0], v[1], v[2]));
            let hs = e.hotspot.map(|(x, y)| (u32::from(x), u32::from(y)));
            assert_eq!(hs, inf.cursor.then_some((v[3], v[4])), "{name} {i}");
            assert_eq!(e.png, v[5] == 1);
            let img = ico::decode_entry(&data, i)?;
            assert_eq!((img.width, img.height), (v[0], v[1]), "{name} {i}");
            let want = read(&format!("expected/{name}.{i}.rgba"))?;
            assert!(rgba(&img)? == want, "{name} entry {i} differs from Pillow");
        }
        let main = file::decode(&data)?;
        assert_eq!(&main.data[..], &ico::decode_entry(&data, largest)?.data[..]);
        assert_eq!(ico::decode_all(&data)?.len(), n);
        files += 1;
    }
    assert_eq!(files, 6);
    assert_eq!(
        ico::decode_entry(&read("mixed.ico")?, 3).err(),
        Some(Error::InvalidArgument)
    );
    Ok(())
}

fn source() -> R<(Vec<u8>, u32, u32)> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/images/raw/rgba_64x48.rgba");
    Ok((std::fs::read(p)?, 64, 48))
}

/// (name, encoded bytes) of the encoder cases.
fn encode_cases() -> R<Vec<(String, Vec<u8>)>> {
    let (px, w, h) = source()?;
    let s = Surface::new(Format::Rgba8Srgb, w, h, &px)?;
    let mut out = Vec::new();
    out.push((
        "default.ico".to_string(),
        ico::encode(&s, &ico::EncodeOptions::default())?.to_vec(),
    ));
    let o = ico::EncodeOptions {
        sizes: &[16, 40],
        png: ico::PngEntries::All,
        hotspot: None,
    };
    out.push(("png.ico".to_string(), ico::encode(&s, &o)?.to_vec()));
    let o = ico::EncodeOptions {
        sizes: &[32, 48, 24],
        png: ico::PngEntries::None,
        hotspot: Some((10, 30)),
    };
    out.push(("cursor.cur".to_string(), ico::encode(&s, &o)?.to_vec()));
    // Gray and RGB sources, odd sizes, an explicit 256 entry.
    let gray: Vec<u8> = px.iter().step_by(4).copied().collect();
    let g = Surface::new(Format::R8Unorm, 7, 5, &gray[..35])?;
    out.push((
        "gray_small.ico".to_string(),
        ico::encode(&g, &ico::EncodeOptions::default())?.to_vec(),
    ));
    let big: Vec<u8> = (0..256 * 256 * 3).map(|i| (i * 7 % 251) as u8).collect();
    let b = Surface::new(Format::Rgb8Srgb, 256, 256, &big)?;
    let t = Surface::new(Format::Rgba8Srgb, w, h, &px)?;
    out.push((
        "entries.ico".to_string(),
        ico::encode_entries(&[b, t], None, ico::PngEntries::Large)?.to_vec(),
    ));
    Ok(out)
}

#[test]
fn encoder_round_trips_and_is_verified() -> R {
    let text = String::from_utf8(read("encoded_verified.txt").unwrap_or_default())?;
    let verified: Vec<(&str, &str)> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once(' '))
        .collect();
    let write = std::env::var("IMGIO_WRITE_ENCODED").is_ok();
    let (px, w, h) = source()?;
    for (name, enc) in encode_cases()? {
        assert_eq!(FileFormat::detect(&enc), Some(FileFormat::Ico));
        let inf = ico::info(&enc)?;
        assert_eq!(inf.cursor, name.ends_with(".cur"));
        for i in 0..inf.entries.len() {
            let img = ico::decode_entry(&enc, i)?;
            if write {
                std::fs::create_dir_all(dir().join("enc"))?;
                std::fs::write(dir().join(format!("enc/{name}.{i}.rgba")), rgba(&img)?)?;
            }
        }
        if write {
            std::fs::write(dir().join(format!("enc/{name}")), &enc)?;
            continue;
        }
        let want = verified
            .iter()
            .find(|(n, _)| *n == name)
            .ok_or(format!(
                "{name}: re-run scripts/gen_file_testdata_ico.py --verify"
            ))?
            .1;
        assert_eq!(
            format!("{:016x}", fnv(&enc)),
            want,
            "{name}: encoder output changed, re-run scripts/gen_file_testdata_ico.py --verify"
        );
    }
    // Specific properties.
    let cases = encode_cases()?;
    let def = ico::info(&cases[0].1)?;
    let sizes: Vec<(u32, bool)> = def.entries.iter().map(|e| (e.width, e.png)).collect();
    assert_eq!(sizes, [(16, false), (24, false), (32, false), (48, false)]);
    // Same-size entry = source pixels exactly (aspect kept: 64x48 -> 48x36 etc.).
    let s = Surface::new(Format::Rgba8Srgb, w, h, &px)?;
    let e48 = ico::decode_entry(&cases[0].1, 3)?;
    assert_eq!((e48.width, e48.height), (48, 36));
    let cur = ico::info(&cases[2].1)?;
    let hs: Vec<_> = cur.entries.iter().map(|e| e.hotspot).collect();
    assert_eq!(hs, [Some((4, 11)), Some((5, 15)), Some((8, 23))]);
    let ent = ico::info(&cases[4].1)?;
    assert_eq!(
        ent.entries
            .iter()
            .map(|e| (e.width, e.png))
            .collect::<Vec<_>>(),
        [(256, true), (64, false)]
    );
    assert_eq!(&ico::decode_entry(&cases[4].1, 1)?.data[..], &px[..]);
    let small = ico::decode(&cases[3].1)?;
    assert_eq!((small.width, small.height), (7, 5));
    // Errors.
    let o = ico::EncodeOptions {
        sizes: &[300],
        ..Default::default()
    };
    assert_eq!(ico::encode(&s, &o).err(), Some(Error::InvalidArgument));
    assert_eq!(
        ico::encode_entries(&[], None, ico::PngEntries::Large).err(),
        Some(Error::InvalidArgument)
    );
    let f = [0u8; 8 * 8 * 4 * 2];
    let hdr = Surface::new(Format::Rgba16Unorm, 8, 8, &f)?;
    assert_eq!(
        ico::encode(&hdr, &ico::EncodeOptions::default()).err(),
        Some(Error::Unsupported)
    );
    assert!(!file::encode(FileFormat::Ico, &s)?.is_empty());
    Ok(())
}

/// Truncated and bit-flipped files return `Err` or `Ok`, never panic.
#[test]
fn corrupt_input_never_panics() -> R {
    for name in [
        "dib_depths.ico",
        "dib_odd.ico",
        "mixed.ico",
        "hotspot.cur",
        "pil_bmp.ico",
    ] {
        let data = read(name)?;
        let step = (data.len() / 97).max(1);
        for len in (0..data.len()).step_by(step) {
            let _ = ico::decode_all(&data[..len]);
            let _ = ico::decode(&data[..len]);
        }
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for i in 0..400 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let mut d = data.clone();
            let span = if i % 2 == 0 {
                data.len().min(200)
            } else {
                data.len()
            };
            d[(rng as usize >> 8) % span] ^= 1 << ((rng >> 3) & 7);
            if i % 3 == 0 {
                d[(rng as usize >> 20) % span] = (rng >> 40) as u8;
            }
            let _ = ico::decode_all(&d);
        }
    }
    Ok(())
}
