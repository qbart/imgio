//! OpenEXR decode / encode vs the OpenEXR reference (testdata/file/exr, written by the OpenEXR
//! 3.5 Python bindings and OpenImageIO; expected pixels decoded by OpenEXR, see
//! `scripts/gen_file_testdata_exrtiff.py`).
#![cfg(feature = "exr")]

use imgio::file::exr::{self, Compression, DecodeOptions, EncodeOptions, Levels, Window};
use imgio::file::{self, FileFormat};
use imgio::{Error, Format, Surface};
use std::error::Error as StdError;

type TestResult = Result<(), Box<dyn StdError>>;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/file/exr/");

fn fnv(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn read(name: &str) -> Result<Vec<u8>, Box<dyn StdError>> {
    Ok(std::fs::read(format!("{DIR}{name}"))?)
}

/// Every case of expected.txt: decode matches the OpenEXR decode under imgio's output policy.
#[test]
fn reference_decodes() -> TestResult {
    let list = std::fs::read_to_string(format!("{DIR}expected.txt"))?;
    let mut n = 0;
    for line in list
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let t: Vec<&str> = line.split_whitespace().collect();
        let data = read(t[0])?;
        assert_eq!(FileFormat::detect(&data), Some(FileFormat::Exr), "{line}");
        if t[1] == "unsupported" {
            assert_eq!(exr::decode(&data).err(), Some(Error::Unsupported), "{line}");
            continue;
        }
        let (fmt, w, h, hash) = (
            t[t.len() - 4],
            t[t.len() - 3],
            t[t.len() - 2],
            t[t.len() - 1],
        );
        let (w, h): (u32, u32) = (w.parse()?, h.parse()?);
        let hash = u64::from_str_radix(hash, 16)?;
        let level = |s: &str| -> Result<(u32, u32), Box<dyn StdError>> {
            let (x, y) = s.split_once(',').ok_or("level")?;
            Ok((x.parse()?, y.parse()?))
        };
        let (got_fmt, got_w, got_h, bytes) = match t[1] {
            "decode" => {
                let img = file::decode(&data)?;
                (
                    format!("{:?}", img.format),
                    img.width,
                    img.height,
                    img.data.to_vec(),
                )
            }
            "plane" => {
                let planes = exr::decode_planes(&data, t[2].parse()?, level(t[3])?, &[t[4]])?;
                let p = planes.first().ok_or("plane")?;
                assert_eq!(p.sample_type, exr::SampleType::U32);
                ("U32".to_string(), p.width, p.height, p.data.to_vec())
            }
            _ => {
                let arg = |k: &str| {
                    t.iter()
                        .find_map(|a| a.strip_prefix(k))
                        .ok_or(format!("{k} in {line}"))
                };
                let layer = arg("layer=")?;
                let opts = DecodeOptions {
                    part: arg("part=")?.parse()?,
                    layer: (layer != "-").then(|| layer.to_string()),
                    level: level(arg("level=")?)?,
                    window: if arg("window=")? == "data" {
                        Window::Data
                    } else {
                        Window::Display
                    },
                };
                let img = exr::decode_with(&data, &opts)?;
                (
                    format!("{:?}", img.format),
                    img.width,
                    img.height,
                    img.data.to_vec(),
                )
            }
        };
        assert_eq!((got_fmt.as_str(), got_w, got_h), (fmt, w, h), "{line}");
        assert_eq!(fnv(&bytes), hash, "{line}");
        n += 1;
    }
    assert!(n >= 60, "{n} cases");
    Ok(())
}

#[test]
fn part_info() -> TestResult {
    let parts = exr::parts(&read("multipart.exr")?)?;
    assert_eq!(parts.len(), 2);
    let (p0, p1) = (&parts[0], &parts[1]);
    assert_eq!(p0.name.as_deref(), Some("beauty"));
    assert_eq!(p1.name.as_deref(), Some("depth"));
    let layers: Vec<&str> = p0.layers.iter().map(|s| s.as_str()).collect();
    assert_eq!(layers, ["", "diffuse", "specular"]);
    assert_eq!(p0.channels.len(), 10);
    assert_eq!(p0.compression, Compression::Zip);
    assert_eq!(p0.tile_size, None);
    let spec = p0
        .channels
        .iter()
        .find(|c| c.name == "specular.G")
        .ok_or("ch")?;
    assert_eq!(spec.sample_type, exr::SampleType::F32);
    assert_eq!(p1.compression, Compression::Piz);
    assert_eq!(p1.tile_size, Some((8, 8)));
    assert_eq!(
        (
            p1.data_window.x,
            p1.data_window.y,
            p1.data_window.width,
            p1.data_window.height
        ),
        (2, 3, 30, 20)
    );
    assert_eq!(
        (p1.display_window.width, p1.display_window.height),
        (37, 29)
    );

    let mip = exr::parts(&read("mip16_zip.exr")?)?;
    assert_eq!(
        mip[0].levels,
        Levels::Mip {
            count: 6,
            round_up: false
        }
    );
    assert_eq!(mip[0].level_size((3, 3))?, (4, 3));
    assert!(mip[0].level_size((1, 2)).is_err());
    let rip = exr::parts(&read("rip16_none.exr")?)?;
    assert_eq!(
        rip[0].levels,
        Levels::Rip {
            x: 6,
            y: 5,
            round_up: false
        }
    );
    assert_eq!(rip[0].level_size((2, 1))?, (9, 10));
    let w = exr::parts(&read("window_zip.exr")?)?;
    assert_eq!((w[0].data_window.x, w[0].data_window.y), (5, -3));
    for (name, c) in [
        ("rgba16_none.exr", Compression::None),
        ("rgba16_rle.exr", Compression::Rle),
        ("rgba16_zips.exr", Compression::Zips),
        ("rgba16_pxr24.exr", Compression::Pxr24),
        ("rgba16_b44.exr", Compression::B44),
        ("rgba16_b44a.exr", Compression::B44a),
        ("rgba16_dwaa.exr", Compression::Dwaa),
        ("rgba16_dwab.exr", Compression::Dwab),
    ] {
        assert_eq!(exr::parts(&read(name)?)?[0].compression, c, "{name}");
    }
    // Raw planes: all channels in file order, f16 samples as stored.
    let planes = exr::decode_planes(&read("ya16_rle.exr")?, 0, (0, 0), &[])?;
    let names: Vec<&str> = planes.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["A", "Y"]);
    // Errors: missing part, layer, level, channel.
    let data = read("multipart.exr")?;
    let bad = |o: DecodeOptions| exr::decode_with(&data, &o).err();
    assert_eq!(
        bad(DecodeOptions {
            part: 2,
            ..Default::default()
        }),
        Some(Error::InvalidArgument)
    );
    assert_eq!(
        bad(DecodeOptions {
            layer: Some("nope".into()),
            ..Default::default()
        }),
        Some(Error::InvalidArgument)
    );
    assert_eq!(
        bad(DecodeOptions {
            level: (1, 1),
            ..Default::default()
        }),
        Some(Error::InvalidArgument)
    );
    assert!(exr::decode_planes(&data, 0, (0, 0), &["Q"]).is_err());
    Ok(())
}

/// Deterministic float pattern: wide range, negatives, zeros; `n` channels, alpha in 0..1.
fn pattern(w: u32, h: u32, n: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity((w * h) as usize * n);
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for y in 0..h {
        for x in 0..w {
            for c in 0..n {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let r = (s >> 8) as f32 / (1u32 << 24) as f32;
                let val = if c == 3 {
                    r
                } else {
                    let e = (x as f32 / w as f32) * 20.0 - 10.0 + r * 2.0;
                    let sign = if (x + y + c as u32).is_multiple_of(9) {
                        -1.0
                    } else {
                        1.0
                    };
                    if (x * 3 + y) % 17 == 0 {
                        0.0
                    } else {
                        sign * e.exp2()
                    }
                };
                v.push(val);
            }
        }
    }
    v
}

fn source(format: Format, w: u32, h: u32) -> Vec<u8> {
    let (n, half) = match format {
        Format::Rgba16Float => (4, true),
        Format::Rgb16Float => (3, true),
        Format::R16Float => (1, true),
        Format::Rgba32Float => (4, false),
        Format::R32Float => (1, false),
        _ => (4, false),
    };
    let v = pattern(w, h, n, w * 31 + h);
    if half {
        v.iter()
            .flat_map(|&f| imgio::half::from_f32(f).to_le_bytes())
            .collect()
    } else {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }
}

/// Max error of lossy encodes, per sample: relative for f32 (PXR24 keeps 15 of 23 mantissa
/// bits), absolute in half ULPs at the block's range for B44 (checked loosely: 6 bits of
/// precision).
fn close(a: f32, b: f32, rel: f32) -> bool {
    a == b || (a - b).abs() <= rel * a.abs().max(b.abs()) || (a.is_nan() && b.is_nan())
}

fn samples(img_fmt: Format, data: &[u8]) -> Vec<f32> {
    match img_fmt {
        Format::Rgba16Float | Format::Rgb16Float | Format::R16Float => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| imgio::half::to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        _ => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    }
}

fn comp_name(c: Compression) -> &'static str {
    match c {
        Compression::None => "none",
        Compression::Rle => "rle",
        Compression::Zips => "zips",
        Compression::Zip => "zip",
        Compression::Piz => "piz",
        Compression::Pxr24 => "pxr24",
        Compression::B44 => "b44",
        Compression::B44a => "b44a",
        _ => "other",
    }
}

/// Writes `IMGIO_FILE_DUMP/<name>` (encoder output for `gen_file_testdata_exrtiff.py verify`).
fn dump(name: &str, bytes: &[u8]) -> TestResult {
    if let Some(dir) = std::env::var_os("IMGIO_FILE_DUMP") {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(std::path::Path::new(&dir).join(name), bytes)?;
    }
    Ok(())
}

/// Encode -> our decode for every storable format, compression and block layout; lossless
/// compressions are exact. The encoded files of each case are locked by hash (verified with
/// OpenEXR by `scripts/gen_file_testdata_exrtiff.py verify`, see testdata/file/exr/encode.txt).
#[test]
fn encode_round_trip() -> TestResult {
    let (w, h) = (37, 29);
    let locked = std::fs::read_to_string(format!("{DIR}encode.txt"))?;
    let mut fresh = String::new();
    let comps = [
        Compression::None,
        Compression::Rle,
        Compression::Zips,
        Compression::Zip,
        Compression::Piz,
        Compression::Pxr24,
        Compression::B44,
        Compression::B44a,
    ];
    for format in [
        Format::Rgba16Float,
        Format::Rgb16Float,
        Format::R16Float,
        Format::Rgba32Float,
        Format::R32Float,
    ] {
        let src = source(format, w, h);
        let surf = Surface::new(format, w, h, &src)?;
        dump(&format!("exr_src_{format:?}_{w}x{h}.raw"), &src)?;
        for c in comps {
            for tiles in [None, Some((16, 8))] {
                let opts = EncodeOptions {
                    compression: c,
                    tiles,
                    ..Default::default()
                };
                let enc = exr::encode(&surf, &opts)?;
                let name = format!(
                    "exr_{format:?}_{}_{}.exr",
                    comp_name(c),
                    if tiles.is_some() { "tiled" } else { "scan" }
                );
                dump(&name, &enc)?;
                fresh += &format!("{name} {:016x}\n", fnv(&enc));
                assert_eq!(FileFormat::detect(&enc), Some(FileFormat::Exr));
                let img = exr::decode(&enc)?;
                assert_eq!(
                    (img.format, img.width, img.height),
                    (format, w, h),
                    "{name}"
                );
                dump(&format!("{name}.dec"), &img.data)?;
                let half = matches!(
                    format,
                    Format::Rgba16Float | Format::Rgb16Float | Format::R16Float
                );
                let lossy = match c {
                    Compression::Pxr24 => !half,
                    Compression::B44 | Compression::B44a => half,
                    _ => false,
                };
                if !lossy {
                    assert!(img.data[..] == src[..], "{name}: not lossless");
                } else {
                    let (a, b) = (samples(format, &img.data), samples(format, &src));
                    let rel = if half { 0.6 } else { 1.0 / 32768.0 };
                    let bad = a
                        .iter()
                        .zip(&b)
                        .filter(|(x, y)| !close(**x, **y, rel))
                        .count();
                    // B44 quantizes 4x4 blocks with a shared exponent: tiny values next to
                    // large ones lose all precision, so only most samples are close.
                    let allowed = if half { a.len() / 4 } else { 0 };
                    assert!(bad <= allowed, "{name}: {bad} samples off");
                }
            }
        }
    }
    // Mip chain (tiled, round down), generated with the default mip options.
    for format in [Format::Rgba16Float, Format::R32Float] {
        let src = source(format, w, h);
        let surf = Surface::new(format, w, h, &src)?;
        let opts = EncodeOptions {
            compression: Compression::Zip,
            mips: true,
            ..Default::default()
        };
        let enc = exr::encode(&surf, &opts)?;
        let name = format!("exr_{format:?}_zip_mips.exr");
        dump(&name, &enc)?;
        fresh += &format!("{name} {:016x}\n", fnv(&enc));
        let info = exr::parts(&enc)?;
        assert_eq!(
            info[0].levels,
            Levels::Mip {
                count: 6,
                round_up: false
            }
        );
        let mips = imgio::mip::generate_mips(&surf, 6, &opts.mip_options)?;
        for l in 0..6u32 {
            let img = exr::decode_with(
                &enc,
                &DecodeOptions {
                    level: (l, l),
                    ..Default::default()
                },
            )?;
            let want: &[u8] = if l == 0 { &src } else { &mips[l as usize - 1] };
            assert!(img.data[..] == want[..], "{name} level {l}");
            dump(&format!("exr_{format:?}_mip{l}.raw"), want)?;
        }
    }
    // 8-bit sRGB input is linearized to Rgba32Float.
    let px = [0u8, 128, 255, 255, 255, 64, 0, 128];
    let enc = exr::encode(
        &Surface::new(Format::Rgba8Srgb, 2, 1, &px)?,
        &EncodeOptions::default(),
    )?;
    let img = exr::decode(&enc)?;
    assert_eq!(img.format, Format::Rgba32Float);
    let v = samples(img.format, &img.data);
    assert_eq!(v[0], 0.0);
    assert!((v[1] - 0.2158605).abs() < 1e-6 && v[2] == 1.0 && (v[3] - 1.0).abs() < 1e-6);
    assert!((v[7] - 128.0 / 255.0).abs() < 1e-6);
    // Unsupported: DWA encode, compressed input formats.
    let s = Surface::new(Format::R32Float, 1, 1, &[0; 4])?;
    let dwa = EncodeOptions {
        compression: Compression::Dwaa,
        ..Default::default()
    };
    assert_eq!(exr::encode(&s, &dwa).err(), Some(Error::Unsupported));
    let bc = [0u8; 8];
    let s = Surface::new(Format::Bc1Unorm, 4, 4, &bc)?;
    assert_eq!(
        exr::encode(&s, &EncodeOptions::default()).err(),
        Some(Error::Unsupported)
    );
    // Default dispatch.
    let enc = file::encode(
        FileFormat::Exr,
        &Surface::new(Format::R32Float, 1, 1, &[0; 4])?,
    )?;
    assert_eq!(file::decode(&enc)?.format, Format::R32Float);

    if std::env::var_os("IMGIO_FILE_DUMP").is_some() {
        std::fs::write(
            std::path::Path::new(&std::env::var_os("IMGIO_FILE_DUMP").ok_or("dump")?)
                .join("exr_encode.txt"),
            &fresh,
        )?;
    }
    assert!(
        locked == fresh,
        "encoder output changed: re-verify with `IMGIO_FILE_DUMP=DIR cargo test --features \
         exr --test file_exr` + `scripts/gen_file_testdata_exrtiff.py verify DIR`, then copy \
         DIR/exr_encode.txt to testdata/file/exr/encode.txt"
    );
    Ok(())
}

/// Truncated and corrupted files return errors (or a valid image), never panic.
#[test]
fn corrupt_inputs() -> TestResult {
    let mut names: Vec<String> = std::fs::read_dir(DIR)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".exr"))
        .collect();
    names.sort();
    // IMGIO_FUZZ_ITERS raises the number of corrupted copies per file (offline soak runs).
    let iters: usize = std::env::var("IMGIO_FUZZ_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    for name in names {
        let data = read(&name)?;
        let step = (data.len() / 97).max(1);
        for len in (0..data.len()).step_by(step) {
            let _ = exr::decode(&data[..len]);
        }
        let mut s = 0x1234_5678u32 ^ data.len() as u32;
        for _ in 0..iters {
            let mut d = data.clone();
            for _ in 0..3 {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let i = s as usize % d.len();
                d[i] ^= (s >> 24) as u8 | 1;
            }
            let _ = exr::decode(&d);
            let _ = exr::parts(&d);
            let _ = exr::decode_planes(&d, 0, (0, 0), &[]);
        }
    }
    Ok(())
}
