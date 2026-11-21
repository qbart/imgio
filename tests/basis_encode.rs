//! `encode_basis` round trips (encode the raw test images with every codec, parse the `.basis`
//! file, transcode it back and measure the error) and bit-exactness with the C++
//! `basis_compressor` on the ktx-rs comp golden cases (testdata/basisu_enc/basisu_comp.txt).

use imgio::basis::{
    BasisCodec, BasisEncodeOptions, BasisFile, BasisTexFormat, BasisTexType, encode_basis,
};
use imgio::{Format, Surface, half};
use std::collections::BTreeMap;
use std::error::Error;
use std::time::Instant;

type R<T> = Result<T, Box<dyn Error>>;

const RAW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/images/raw/");

/// Raw image `name` (NAME_WxH.ext) -> (width, height, bytes).
fn raw(name: &str) -> R<(u32, u32, Vec<u8>)> {
    let stem = name.split('.').next().ok_or("name")?;
    let dims = stem.rsplit('_').next().ok_or("name")?;
    let (w, h) = dims.split_once('x').ok_or("dims")?;
    Ok((
        w.parse()?,
        h.parse()?,
        std::fs::read(format!("{RAW}{name}"))?,
    ))
}

/// PSNR over RGBA8 channels `chans`.
fn psnr(a: &[u8], b: &[u8], chans: std::ops::Range<usize>) -> f64 {
    let (mut se, mut n) = (0f64, 0f64);
    for (pa, pb) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        for c in chans.clone() {
            let d = f64::from(pa[c]) - f64::from(pb[c]);
            se += d * d;
            n += 1.0;
        }
    }
    if se == 0.0 {
        return 99.0;
    }
    10.0 * (255.0f64 * 255.0 * n / se).log10()
}

fn halves(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// sum |a - b| / sum |a| over RGB of RGBA float pixels.
fn rel_err(src: &[f32], got: &[f32]) -> f64 {
    let (mut d, mut s) = (0f64, 0f64);
    for (ps, pg) in src.as_chunks::<4>().0.iter().zip(got.as_chunks::<4>().0) {
        for c in 0..3 {
            d += f64::from((ps[c] - pg[c]).abs());
            s += f64::from(ps[c].abs());
        }
    }
    d / s.max(1e-9)
}

fn ldr_opts(codec: BasisCodec) -> BasisEncodeOptions {
    BasisEncodeOptions {
        codec,
        ..Default::default()
    }
}

#[test]
fn ldr_round_trip() -> R<()> {
    // Minimum (RGB, alpha) PSNR. The images are noisy (BC7 / ASTC 4x4 reach ~37 dB on
    // rgba_64x48), ETC1S has one color per block and a shared codebook.
    let cases = [
        (
            "ETC1S",
            ldr_opts(BasisCodec::Etc1s),
            BasisTexFormat::Etc1s,
            (19.0, 40.0),
        ),
        (
            "ETC1S q255",
            BasisEncodeOptions {
                etc1s_quality: 255,
                ..ldr_opts(BasisCodec::Etc1s)
            },
            BasisTexFormat::Etc1s,
            (19.0, 40.0),
        ),
        (
            "UASTC",
            ldr_opts(BasisCodec::UastcLdr4x4),
            BasisTexFormat::UastcLdr4x4,
            (34.5, 37.0),
        ),
        (
            "UASTC RDO 1.0",
            BasisEncodeOptions {
                uastc_rdo_lambda: 1.0,
                ..ldr_opts(BasisCodec::UastcLdr4x4)
            },
            BasisTexFormat::UastcLdr4x4,
            (34.5, 37.0),
        ),
    ];
    for name in [
        "rgba_64x48.rgba",
        "rgb_37x29.rgba",
        "mosaic_rgba_50x38.rgba",
    ] {
        let (w, h, px) = raw(name)?;
        let src = Surface::new(Format::Rgba8Unorm, w, h, &px)?;
        let opaque = px.as_chunks::<4>().0.iter().all(|p| p[3] == 255);
        for (label, opts, tex, (min_rgb, min_a)) in &cases {
            let t = Instant::now();
            let file = encode_basis(&[src], opts)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let b = BasisFile::parse(&file)?;
            assert_eq!(b.tex_format(), *tex);
            assert_eq!(b.image_count(), 1);
            assert_eq!(b.level_count(0), 1);
            assert_eq!(b.has_alpha(), !opaque, "{name} {label}");
            assert!(b.is_srgb());
            let out = b.transcode_to_buffer(0, 0, Format::Rgba8Unorm)?;
            let rgb = psnr(&px, &out, 0..3);
            let a = psnr(&px, &out, 3..4);
            eprintln!(
                "{name} {label}: {} bytes, RGB {rgb:.2} dB, A {a:.2} dB, {ms:.1} ms",
                file.len()
            );
            assert!(rgb >= *min_rgb, "{name} {label}: RGB PSNR {rgb:.2}");
            assert!(a >= *min_a, "{name} {label}: alpha PSNR {a:.2}");
        }
    }
    Ok(())
}

#[test]
fn hdr_round_trip() -> R<()> {
    let (w, h, px) = raw("hdr_40x24.rgba16f")?;
    let src_f = halves(&px);
    let src = Surface::new(Format::Rgba16Float, w, h, &px)?;
    let cases = [
        (BasisCodec::UastcHdr4x4, BasisTexFormat::UastcHdr4x4, 0.05),
        (BasisCodec::AstcHdr6x6, BasisTexFormat::AstcHdr6x6, 0.08),
        (
            BasisCodec::UastcHdr6x6Intermediate,
            BasisTexFormat::UastcHdr6x6Intermediate,
            0.08,
        ),
    ];
    for (codec, tex, max_err) in cases {
        let opts = ldr_opts(codec);
        let t = Instant::now();
        let file = encode_basis(&[src], &opts)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let b = BasisFile::parse(&file)?;
        assert_eq!(b.tex_format(), tex);
        assert!(!b.has_alpha() && !b.is_srgb());
        let out = halves(&b.transcode_to_buffer(0, 0, Format::Rgba16Float)?);
        let e = rel_err(&src_f, &out);
        eprintln!(
            "hdr_40x24 {codec:?}: {} bytes, rel err {e:.4}, {ms:.1} ms",
            file.len()
        );
        assert!(e <= max_err, "{codec:?}: relative error {e:.4}");
        // Float input gives the same file.
        let f32px: Vec<u8> = src_f.iter().flat_map(|v| v.to_le_bytes()).collect();
        let src32 = Surface::new(Format::Rgba32Float, w, h, &f32px)?;
        assert_eq!(&encode_basis(&[src32], &opts)?[..], &file[..]);
    }
    // UASTC HDR 4x4 blocks are independent: the thread count doesn't change the file.
    let one = BasisEncodeOptions {
        threads: 1,
        ..ldr_opts(BasisCodec::UastcHdr4x4)
    };
    let many = BasisEncodeOptions { threads: 5, ..one };
    assert_eq!(
        &encode_basis(&[src], &one)?[..],
        &encode_basis(&[src], &many)?[..]
    );
    // RDO lambda shrinks the 6x6 intermediate file.
    let base = encode_basis(&[src], &ldr_opts(BasisCodec::UastcHdr6x6Intermediate))?;
    let rdo = encode_basis(
        &[src],
        &BasisEncodeOptions {
            hdr_6x6_lambda: 2000.0,
            ..ldr_opts(BasisCodec::UastcHdr6x6Intermediate)
        },
    )?;
    eprintln!("6x6i lambda 2000: {} -> {} bytes", base.len(), rdo.len());
    assert!(rdo.len() <= base.len());
    Ok(())
}

#[test]
fn mipmaps_and_arrays() -> R<()> {
    let (w, h, px) = raw("rgba_64x48.rgba")?;
    let src = Surface::new(Format::Rgba8Unorm, w, h, &px)?;
    for codec in [BasisCodec::Etc1s, BasisCodec::UastcLdr4x4] {
        let opts = BasisEncodeOptions {
            mipmaps: true,
            tex_type: BasisTexType::Tex2DArray,
            threads: 2,
            ..ldr_opts(codec)
        };
        let file = encode_basis(&[src, src], &opts)?;
        let b = BasisFile::parse(&file)?;
        assert_eq!(b.tex_type(), BasisTexType::Tex2DArray);
        assert_eq!(b.image_count(), 2);
        for image in 0..2 {
            assert_eq!(b.level_count(image), 7);
            for level in 0..7 {
                let li = b.level_info(image, level)?;
                assert_eq!((li.width, li.height), imgio::image::mip_dim(w, h, level));
                let out = b.transcode_to_buffer(image, level, Format::Rgba8Unorm)?;
                assert_eq!(out.len(), (li.width * li.height * 4) as usize);
            }
        }
        // Level 1 against a box downsample of the source (kaiser is close to it).
        let l1 = b.transcode_to_buffer(0, 1, Format::Rgba8Unorm)?;
        let mut boxed = vec![0u8; 32 * 24 * 4];
        for y in 0..24 {
            for x in 0..32 {
                for c in 0..4 {
                    let s: u32 = [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .iter()
                        .map(|(dx, dy)| u32::from(px[((y * 2 + dy) * 64 + x * 2 + dx) * 4 + c]))
                        .sum();
                    boxed[(y * 32 + x) * 4 + c] = ((s + 2) / 4) as u8;
                }
            }
        }
        let p = psnr(&boxed, &l1, 0..3);
        eprintln!("{codec:?} mip 1 vs box: {p:.2} dB");
        assert!(
            p > if codec == BasisCodec::Etc1s {
                20.0
            } else {
                26.0
            }
        );
    }
    // HDR mipmaps.
    let (w, h, px) = raw("hdr_40x24.rgba16f")?;
    let src = Surface::new(Format::Rgba16Float, w, h, &px)?;
    for codec in [BasisCodec::UastcHdr4x4, BasisCodec::AstcHdr6x6] {
        let file = encode_basis(
            &[src],
            &BasisEncodeOptions {
                mipmaps: true,
                ..ldr_opts(codec)
            },
        )?;
        let b = BasisFile::parse(&file)?;
        assert_eq!(b.level_count(0), 6);
        for level in 0..6 {
            b.transcode_to_buffer(0, level, Format::Rgba16Float)?;
        }
    }
    Ok(())
}

#[test]
fn y_flip() -> R<()> {
    let (w, h, px) = raw("rgb_37x29.rgba")?;
    let row = (w * 4) as usize;
    let flipped: Vec<u8> = px.chunks_exact(row).rev().flatten().copied().collect();
    let opts = BasisEncodeOptions {
        y_flip: true,
        ..ldr_opts(BasisCodec::UastcLdr4x4)
    };
    let a = encode_basis(&[Surface::new(Format::Rgba8Unorm, w, h, &px)?], &opts)?;
    let b = encode_basis(
        &[Surface::new(Format::Rgba8Unorm, w, h, &flipped)?],
        &ldr_opts(BasisCodec::UastcLdr4x4),
    )?;
    let (fa, fb) = (BasisFile::parse(&a)?, BasisFile::parse(&b)?);
    assert!(fa.y_flipped() && !fb.y_flipped());
    assert_eq!(
        &fa.transcode_to_buffer(0, 0, Format::Rgba8Unorm)?[..],
        &fb.transcode_to_buffer(0, 0, Format::Rgba8Unorm)?[..]
    );
    Ok(())
}

/// HDR files carry basisu 2.50's `BasisULibVers` + `KTXmapRange` (scale 1.0) key/value data:
/// the same bytes as the 2.50 tool's testdata/basis/*hdr*.basis (from the .hdr source).
#[test]
fn hdr_v2_keys() -> R<()> {
    let (w, h, px) = raw("hdr_40x24.rgba16f")?;
    let src = Surface::new(Format::Rgba16Float, w, h, &px)?;
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/basis/");
    let ext = |b: &[u8]| -> Vec<u8> {
        let le = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
        b[le(69)..le(69) + le(73)].to_vec()
    };
    for (codec, cpp) in [
        (BasisCodec::UastcHdr4x4, "uastc_hdr4x4.basis"),
        (BasisCodec::AstcHdr6x6, "astc_hdr6x6.basis"),
        (BasisCodec::UastcHdr6x6Intermediate, "uastc_hdr6x6i.basis"),
    ] {
        let ours = encode_basis(&[src], &ldr_opts(codec))?;
        let theirs = std::fs::read(format!("{dir}{cpp}"))?;
        assert_eq!(ext(&ours).len(), 55, "{cpp}");
        assert_eq!(ext(&ours), ext(&theirs), "{cpp}");
    }
    Ok(())
}

#[test]
fn bad_input() -> R<()> {
    let px = [0u8; 16];
    let s = Surface::new(Format::Rgba8Unorm, 2, 2, &px)?;
    assert!(encode_basis(&[], &BasisEncodeOptions::default()).is_err());
    let bad = BasisEncodeOptions {
        etc1s_quality: 0,
        ..Default::default()
    };
    assert!(encode_basis(&[s], &bad).is_err());
    let cube = BasisEncodeOptions {
        tex_type: BasisTexType::CubemapArray,
        ..Default::default()
    };
    assert!(encode_basis(&[s, s], &cube).is_err());
    // Compressed and other uncompressed inputs are accepted.
    let bc1 = imgio::encode_to_buffer(&s, Format::Bc1Unorm, &imgio::EncodeOptions::default())?;
    let s_bc1 = Surface::new(Format::Bc1Unorm, 2, 2, &bc1)?;
    let s_r8 = Surface::new(Format::R8Unorm, 2, 2, &px[..4])?;
    encode_basis(&[s_bc1, s_r8], &BasisEncodeOptions::default())?;
    Ok(())
}

// ---- Bit-exactness with the C++ basis_compressor (ktx-rs tests/basisu_comp.rs inputs). ----

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
}

/// ktx-rs `make_image` as RGBA8 bytes.
fn make_image(seed: u32, w: u32, h: u32, kind: u32, alpha: u32) -> Vec<u8> {
    let mut l = Lcg(seed.wrapping_mul(7919).wrapping_add(11));
    let mut img = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (xi, yi) = (x as i32, y as i32);
            let (mut r, mut g, mut b) = match kind {
                0 => (
                    (x * 255 / if w > 1 { w - 1 } else { 1 }) as i32,
                    (y * 255 / if h > 1 { h - 1 } else { 1 }) as i32,
                    128 + xi - yi,
                ),
                1 => {
                    let c = if ((x >> 2) + (y >> 2)) & 1 != 0 {
                        220
                    } else {
                        30
                    };
                    (c, (c + xi * 4) & 255, 255 - c)
                }
                2 => (
                    (l.next() & 255) as i32,
                    ((l.next() >> 4) & 255) as i32,
                    ((l.next() >> 8) & 255) as i32,
                ),
                _ => {
                    let c = (((x / 6) + (y / 5)) % 3) as i32 * 100 + ((x >> 1) & 1) as i32 * 9;
                    (c, 255 - c, (c * 2) & 255)
                }
            };
            if kind != 2 {
                r += (l.next() % 7) as i32 - 3;
                g += (l.next() % 7) as i32 - 3;
                b += (l.next() % 7) as i32 - 3;
            }
            let mut a = 255;
            if alpha == 1 {
                a = ((x * 37 + y * 11 + (l.next() & 15)) & 255) as i32;
            } else if alpha == 2 && x == w / 2 && y == h / 2 {
                a = 254;
            }
            img.extend([r, g, b, a].map(|v| v.clamp(0, 255) as u8));
        }
    }
    img
}

/// (tex_type, n, w, h, kind, alpha_mask, halve)
const INPUTS: [(u32, u32, u32, u32, u32, u32, bool); 13] = [
    (0, 1, 16, 16, 0, 0, false),
    (0, 1, 13, 7, 2, 1, false),
    (0, 1, 37, 29, 1, 0, false),
    (0, 3, 40, 24, 0, 0, true),
    (0, 3, 22, 18, 3, 2, true),
    (3, 3, 24, 16, 1, 0, false),
    (3, 3, 20, 12, 0, 7, false),
    (0, 1, 64, 64, 3, 0, false),
    (0, 1, 1, 1, 2, 0, false),
    (0, 1, 5, 3, 0, 4, false),
    (1, 2, 12, 12, 2, 0, false),
    (0, 1, 128, 96, 2, 0, false),
    (0, 1, 96, 80, 0, 1, false),
];

/// The golden configs `encode_basis` can express: (golden cfg index, options). The goldens
/// come from the pre-2.0 C++ `basis_compressor`, so without the v2 key/value data.
fn golden_cfgs() -> Vec<(usize, BasisEncodeOptions)> {
    let e = |quality: u32, level: u32, srgb: bool| BasisEncodeOptions {
        etc1s_quality: quality,
        etc1s_level: level,
        srgb,
        threads: 1,
        basisu_v2_keys: false,
        ..Default::default()
    };
    let u = |level: u32, lambda: f32| BasisEncodeOptions {
        codec: BasisCodec::UastcLdr4x4,
        uastc_level: level,
        uastc_rdo_lambda: lambda,
        threads: 1,
        basisu_v2_keys: false,
        ..Default::default()
    };
    vec![
        (0, e(128, 2, true)),
        (1, e(1, 1, true)),
        (2, e(64, 0, false)),
        (3, e(140, 3, true)),
        (7, e(255, 6, true)),
        (14, u(2, 0.0)),
        (15, u(0, 0.0)),
        (19, u(2, 1.0)),
        (23, e(223, 2, true)),
        (24, e(129, 2, true)),
        (25, e(100, 1, true)),
        (26, e(30, 2, false)),
        (27, e(223, 6, false)),
    ]
}

#[test]
fn matches_basisu_compressor() -> R<()> {
    let golden: BTreeMap<String, String> = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/basisu_enc/basisu_comp.txt"
    ))?
    .lines()
    .filter_map(|l| l.split_once('\t'))
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let mut n = 0;
    for (i, &(tex_type, count, mut w, mut h, kind, alpha_mask, halve)) in INPUTS.iter().enumerate()
    {
        let mut images = Vec::new();
        for k in 0..count {
            let alpha = if alpha_mask == 4 {
                2
            } else {
                (alpha_mask >> k) & 1
            };
            images.push((w, h, make_image(i as u32 * 17 + k, w, h, kind, alpha)));
            if halve {
                w = (w / 2).max(1);
                h = (h / 2).max(1);
            }
        }
        let surfaces = images
            .iter()
            .map(|(w, h, px)| Surface::new(Format::Rgba8Unorm, *w, *h, px))
            .collect::<Result<Vec<_>, _>>()?;
        for (c, mut opts) in golden_cfgs() {
            opts.tex_type = [
                BasisTexType::Tex2D,
                BasisTexType::Tex2DArray,
                BasisTexType::CubemapArray,
                BasisTexType::VideoFrames,
            ][tex_type as usize];
            let file = encode_basis(&surfaces, &opts)?;
            // ktx-rs case hash: FNV-1a over error code 0, file length and file bytes.
            let mut hash = 0xcbf29ce484222325u64;
            let len = file.len() as u32;
            for &b in [0u32.to_le_bytes(), len.to_le_bytes()]
                .iter()
                .flatten()
                .chain(file.iter())
            {
                hash = (hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
            let key = format!("{i}_{c}");
            assert_eq!(
                golden.get(&key).map(String::as_str),
                Some(format!("{hash:016x}").as_str()),
                "{key}"
            );
            n += 1;
        }
    }
    assert_eq!(n, INPUTS.len() * golden_cfgs().len());
    Ok(())
}

/// Mipmap generation and y flip vs. the C++ `basisu` v2.50 tool (`-basis -no_multithreading`
/// with `-uastc -mipmap`, `-etc1s -q 128 -mipmap`, `-uastc -y_flip` on testdata/images/*.png):
/// identical files (including the 31-byte `BasisULibVers` key/value header extension of 2.50),
/// and without that extension (`basisu_v2_keys: false`) every level transcodes the same.
#[test]
fn matches_basisu_tool() -> R<()> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/basisu_enc/");
    let cases = [
        (
            "cli_uastc_mipmap_rgba_64x48.basis",
            "rgba_64x48.rgba",
            BasisCodec::UastcLdr4x4,
            true,
            false,
        ),
        (
            "cli_etc1s_mipmap_rgba_64x48.basis",
            "rgba_64x48.rgba",
            BasisCodec::Etc1s,
            true,
            false,
        ),
        (
            "cli_uastc_yflip_rgb_37x29.basis",
            "rgb_37x29.rgba",
            BasisCodec::UastcLdr4x4,
            false,
            true,
        ),
    ];
    for (cpp, name, codec, mipmaps, y_flip) in cases {
        let (w, h, px) = raw(name)?;
        let ours = encode_basis(
            &[Surface::new(Format::Rgba8Unorm, w, h, &px)?],
            &BasisEncodeOptions {
                codec,
                mipmaps,
                y_flip,
                threads: 1,
                ..Default::default()
            },
        )?;
        let theirs = std::fs::read(format!("{dir}{cpp}"))?;
        assert!(ours[..] == theirs[..], "{cpp}");
        let ours = encode_basis(
            &[Surface::new(Format::Rgba8Unorm, w, h, &px)?],
            &BasisEncodeOptions {
                codec,
                mipmaps,
                y_flip,
                threads: 1,
                basisu_v2_keys: false,
                ..Default::default()
            },
        )?;
        assert_eq!(ours.len() + 31, theirs.len(), "{cpp}");
        let (a, b) = (BasisFile::parse(&ours)?, BasisFile::parse(&theirs)?);
        assert_eq!(a.level_count(0), b.level_count(0), "{cpp}");
        assert_eq!(a.y_flipped(), b.y_flipped(), "{cpp}");
        for level in 0..a.level_count(0) {
            assert!(
                a.transcode_to_buffer(0, level, Format::Rgba8Unorm)?[..]
                    == b.transcode_to_buffer(0, level, Format::Rgba8Unorm)?[..],
                "{cpp} level {level}"
            );
        }
    }
    Ok(())
}
