//! TIFF decode / encode vs tifffile / libtiff (testdata/file/tiff, written by tifffile,
//! Pillow and a minimal writer in `scripts/gen_file_testdata_exrtiff.py`; expected pixels
//! decoded by tifffile or libtiff's RGBA reader).
#![cfg(feature = "tiff")]

use imgio::file::tiff::{self, Alpha, Compression, EncodeOptions};
use imgio::file::{self, FileFormat};
use imgio::{Error, Format, Surface};
use std::error::Error as StdError;

type TestResult = Result<(), Box<dyn StdError>>;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/file/tiff/");

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

/// Every case of expected.txt: exact for all but JPEG (zune-jpeg vs libjpeg: bounded).
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
        assert_eq!(FileFormat::detect(&data), Some(FileFormat::Tiff), "{line}");
        match t[1] {
            "unsupported" => {
                assert_eq!(
                    tiff::decode(&data).err(),
                    Some(Error::Unsupported),
                    "{line}"
                );
            }
            "jpeg" => {
                let img = file::decode(&data)?;
                let want = read(t[5])?;
                assert_eq!(
                    (format!("{:?}", img.format), img.width, img.height),
                    (t[2].to_string(), t[3].parse()?, t[4].parse()?),
                    "{line}"
                );
                assert_eq!(img.data.len(), want.len());
                let (mut max, mut sum) = (0i32, 0u64);
                for (a, b) in img.data.iter().zip(&want) {
                    let d = (*a as i32 - *b as i32).abs();
                    max = max.max(d);
                    sum += d as u64;
                }
                let mean = sum as f64 / want.len() as f64;
                // zune-jpeg and libjpeg-turbo differ in IDCT rounding and chroma upsampling.
                assert!(max <= 4 && mean < 0.5, "{line}: max {max}, mean {mean:.3}");
            }
            _ => {
                let page: usize = t[1].strip_prefix("page=").ok_or("page")?.parse()?;
                let img = tiff::decode_page(&data, page)?;
                assert_eq!(
                    (format!("{:?}", img.format), img.width, img.height),
                    (t[2].to_string(), t[3].parse()?, t[4].parse()?),
                    "{line}"
                );
                assert_eq!(fnv(&img.data), u64::from_str_radix(t[5], 16)?, "{line}");
            }
        }
        n += 1;
    }
    assert!(n >= 50, "{n} cases");
    Ok(())
}

#[test]
fn page_info() -> TestResult {
    let data = read("multipage.tif")?;
    assert_eq!(tiff::page_count(&data)?, 3);
    assert_eq!(
        tiff::decode_page(&data, 3).err(),
        Some(Error::InvalidArgument)
    );
    let p2 = tiff::page_info(&data, 2)?;
    assert_eq!(
        (p2.bits_per_sample, p2.sample_format, p2.tile),
        (32, 3, Some((16, 16)))
    );
    let big = tiff::page_info(&read("rgb8_planar_tiled_big.tif")?, 0)?;
    assert!(big.bigtiff && !big.little_endian);
    assert_eq!(
        (big.planar, big.tile, big.compression),
        (2, Some((32, 16)), 5)
    );
    let a = tiff::page_info(&read("rgba8_assoc_lzw.tif")?, 0)?;
    assert_eq!(a.alpha, Alpha::Associated);
    let a = tiff::page_info(&read("rgba8_bigtiff_packbits.tif")?, 0)?;
    assert_eq!(a.alpha, Alpha::Unassociated);
    let y = tiff::page_info(&read("ycbcr22.tif")?, 0)?;
    assert_eq!((y.photometric, y.samples_per_pixel), (6, 3));
    assert_eq!(tiff::page_count(&read("gray8_deflate.tif")?)?, 1);
    Ok(())
}

fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2_654_435_761) | 1;
    (0..len)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            // smooth-ish with noise so predictors matter
            ((i / 3) as u32 % 200 + (s >> 29)) as u8
        })
        .collect()
}

fn source(format: Format, w: u32, h: u32) -> Result<Vec<u8>, Box<dyn StdError>> {
    let len = format.surface_bytes(w, h)?;
    let mut v = pattern(len, w * 7 + h);
    match format {
        Format::R16Float | Format::Rgb16Float | Format::Rgba16Float => {
            for c in v.as_chunks_mut::<2>().0 {
                let f = (c[0] as f32 - 100.0) * 0.37 + c[1] as f32 / 256.0;
                c.copy_from_slice(&imgio::half::from_f32(f).to_le_bytes());
            }
        }
        Format::R32Float | Format::Rgba32Float => {
            for c in v.as_chunks_mut::<4>().0 {
                let f = (c[0] as f32 - 100.0) * 1.37e3 + c[1] as f32 / 7.0;
                c.copy_from_slice(&f.to_le_bytes());
            }
        }
        _ => {}
    }
    Ok(v)
}

fn dump(name: &str, bytes: &[u8]) -> TestResult {
    if let Some(dir) = std::env::var_os("IMGIO_FILE_DUMP") {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(std::path::Path::new(&dir).join(name), bytes)?;
    }
    Ok(())
}

/// Encode -> our decode is exact for every stored format, compression, predictor and BigTIFF;
/// the encoded files are locked by hash (verified with tifffile by
/// `scripts/gen_file_testdata_exrtiff.py verify`, see testdata/file/tiff/encode.txt).
#[test]
fn encode_round_trip() -> TestResult {
    let (w, h) = (37, 29);
    let locked = std::fs::read_to_string(format!("{DIR}encode.txt"))?;
    let mut fresh = String::new();
    let formats = [
        Format::R8Unorm,
        Format::Rg8Unorm,
        Format::Rgb8Srgb,
        Format::Rgba8Srgb,
        Format::R16Unorm,
        Format::Rg16Unorm,
        Format::Rgba16Unorm,
        Format::R16Float,
        Format::Rgb16Float,
        Format::Rgba16Float,
        Format::R32Float,
        Format::Rgba32Float,
    ];
    let comps = [
        (Compression::None, "none"),
        (Compression::Lzw, "lzw"),
        (Compression::Deflate, "deflate"),
        (Compression::PackBits, "packbits"),
    ];
    for format in formats {
        let src = source(format, w, h)?;
        let surf = Surface::new(format, w, h, &src)?;
        for (c, cn) in comps {
            for predictor in [false, true] {
                if predictor && matches!(c, Compression::None | Compression::PackBits) {
                    continue; // predictor ignored
                }
                for bigtiff in [false, true] {
                    if bigtiff && (predictor || c != Compression::Lzw) {
                        continue;
                    }
                    let opts = EncodeOptions {
                        compression: c,
                        predictor,
                        bigtiff,
                    };
                    let enc = tiff::encode(&surf, &opts)?;
                    let name = format!(
                        "tiff_{format:?}_{cn}{}{}.tif",
                        if predictor { "_pred" } else { "" },
                        if bigtiff { "_big" } else { "" }
                    );
                    dump(&name, &enc)?;
                    dump(&name.replace(".tif", ".src"), &src)?;
                    fresh += &format!("{name} {:016x}\n", fnv(&enc));
                    assert_eq!(FileFormat::detect(&enc), Some(FileFormat::Tiff));
                    let info = tiff::page_info(&enc, 0)?;
                    assert_eq!(info.bigtiff, bigtiff);
                    let img = tiff::decode(&enc)?;
                    let want_fmt = format;
                    assert_eq!(
                        (img.format, img.width, img.height),
                        (want_fmt, w, h),
                        "{name}"
                    );
                    assert!(img.data[..] == src[..], "{name}");
                }
            }
        }
    }
    // Multi-page.
    let a = source(Format::Rgba8Srgb, w, h)?;
    let b = source(Format::R16Unorm, 20, 9)?;
    let c = source(Format::Rgba32Float, 5, 3)?;
    let pages = [
        Surface::new(Format::Rgba8Srgb, w, h, &a)?,
        Surface::new(Format::R16Unorm, 20, 9, &b)?,
        Surface::new(Format::Rgba32Float, 5, 3, &c)?,
    ];
    let enc = tiff::encode_pages(&pages, &EncodeOptions::default())?;
    dump("tiff_multipage.tif", &enc)?;
    dump("tiff_multipage.src", &[&a[..], &b, &c].concat())?;
    fresh += &format!("tiff_multipage.tif {:016x}\n", fnv(&enc));
    assert_eq!(tiff::page_count(&enc)?, 3);
    for (i, (p, s)) in pages.iter().zip([&a, &b, &c]).enumerate() {
        let img = tiff::decode_page(&enc, i)?;
        assert_eq!((img.format, img.width), (p.format(), p.width()));
        assert!(img.data[..] == s[..], "page {i}");
    }
    // Converted inputs: BGRA swizzled, other formats as Rgba32Float.
    let bgra = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let img = tiff::decode(&tiff::encode(
        &Surface::new(Format::Bgra8Unorm, 2, 1, &bgra)?,
        &EncodeOptions::default(),
    )?)?;
    assert_eq!(
        (img.format, &img.data[..]),
        (Format::Rgba8Srgb, &[3u8, 2, 1, 4, 7, 6, 5, 8][..])
    );
    let px = 0xF800u16.to_le_bytes();
    let img = tiff::decode(&tiff::encode(
        &Surface::new(Format::Rgb565, 1, 1, &px)?,
        &EncodeOptions::default(),
    )?)?;
    assert_eq!(img.format, Format::Rgba32Float);
    assert_eq!(&img.data[..4], &1.0f32.to_le_bytes());
    let bc = [0u8; 8];
    let s = Surface::new(Format::Bc1Unorm, 4, 4, &bc)?;
    assert_eq!(
        tiff::encode(&s, &EncodeOptions::default()).err(),
        Some(Error::Unsupported)
    );
    let enc = file::encode(FileFormat::Tiff, &pages[0])?;
    assert!(file::decode(&enc)?.data[..] == a[..]);

    if let Some(dir) = std::env::var_os("IMGIO_FILE_DUMP") {
        std::fs::write(std::path::Path::new(&dir).join("tiff_encode.txt"), &fresh)?;
    }
    assert!(
        locked == fresh,
        "encoder output changed: re-verify with `IMGIO_FILE_DUMP=DIR cargo test --features \
         tiff --test file_tiff` + `scripts/gen_file_testdata_exrtiff.py verify DIR`, then copy \
         DIR/tiff_encode.txt to testdata/file/tiff/encode.txt"
    );
    Ok(())
}

/// Truncated and corrupted files return errors (or a valid image), never panic.
#[test]
fn corrupt_inputs() -> TestResult {
    let mut names: Vec<String> = std::fs::read_dir(DIR)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tif"))
        .collect();
    names.sort();
    // IMGIO_FUZZ_ITERS raises the number of corrupted copies per file (offline soak runs).
    let iters: usize = std::env::var("IMGIO_FUZZ_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    for name in names {
        let data = read(&name)?;
        let step = (data.len() / 97).max(1);
        for len in (0..data.len()).step_by(step) {
            let _ = tiff::decode(&data[..len]);
            let _ = tiff::page_count(&data[..len]);
        }
        let mut s = 0x9e37_79b9u32 ^ data.len() as u32;
        for _ in 0..iters {
            let mut d = data.clone();
            for _ in 0..3 {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                // Bias towards the header / IFD region.
                let i = if s & 1 == 0 {
                    (s >> 8) as usize % d.len().min(512)
                } else {
                    (s >> 8) as usize % d.len()
                };
                d[i] ^= (s >> 24) as u8 | 1;
            }
            let _ = tiff::decode(&d);
            let _ = tiff::page_info(&d, 0);
        }
    }
    // Hostile sizes: a 100000 x 100000 page with one tiny strip is rejected before allocating.
    let mut t = Vec::new();
    t.extend_from_slice(b"II*\0\x08\0\0\0");
    let entries: [(u16, u16, u32, u32); 6] = [
        (256, 4, 1, 100_000),
        (257, 4, 1, 100_000),
        (258, 3, 1, 8),
        (262, 3, 1, 1),
        (273, 4, 1, 0),
        (279, 4, 1, 1),
    ];
    t.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, typ, count, val) in entries {
        t.extend_from_slice(&tag.to_le_bytes());
        t.extend_from_slice(&typ.to_le_bytes());
        t.extend_from_slice(&count.to_le_bytes());
        t.extend_from_slice(&val.to_le_bytes());
    }
    t.extend_from_slice(&[0; 4]);
    assert_eq!(tiff::decode(&t).err(), Some(Error::InvalidData));
    Ok(())
}
