//! XUBC7 `encode_basis` / `ktx2::encode_basis` against the basisu v2.50 CLI goldens
//! (testdata/basisu_enc/cli_xubc7_*.{basis,ktx2}, `scripts/gen_basis_testdata.sh`): byte-exact
//! files, round trip quality, option validation. RGB sources are fed as RGBA8 with alpha 255
//! (as basisu reads them).

use imgio::basis::{BasisCodec, BasisEncodeOptions, BasisFile, BasisTexFormat, Xubc7Encoder};
use imgio::ktx2::{BasisKtx2Options, Ktx2File};
use imgio::{Error, Format, Surface};
use std::error::Error as StdError;

type R<T> = Result<T, Box<dyn StdError>>;

const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/");

/// Raw RGBA8 test image NAME_WxH -> (width, height, pixels).
fn raw(name: &str) -> R<(u32, u32, Vec<u8>)> {
    let dims = name.rsplit('_').next().ok_or("name")?;
    let (w, h) = dims.split_once('x').ok_or("dims")?;
    Ok((
        w.parse()?,
        h.parse()?,
        std::fs::read(format!("{TESTDATA}images/raw/{name}.rgba"))?,
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

struct Golden {
    /// File stem in testdata/basisu_enc.
    name: &'static str,
    /// Raw source image in testdata/images/raw.
    image: &'static str,
    opts: BasisEncodeOptions,
    /// Minimum level 0 (RGB, alpha) PSNR in dB (about 0.5 dB under the golden's).
    min_psnr: (f64, f64),
}

/// The CLI goldens: `-basis -xubc7` plus the options in the name.
fn goldens() -> Vec<Golden> {
    let x = BasisEncodeOptions {
        codec: BasisCodec::Xubc7,
        ..Default::default()
    };
    let g = |name, image, opts, min_psnr| Golden {
        name,
        image,
        opts,
        min_psnr,
    };
    vec![
        g(
            "cli_xubc7_mip_rgba_64x48",
            "rgba_64x48",
            BasisEncodeOptions { mipmaps: true, ..x },
            (36.0, 38.5),
        ),
        g("cli_xubc7_rgb_37x29", "rgb_37x29", x, (35.2, 72.0)),
        g(
            "cli_xubc7_mosaic_rgba_50x38",
            "mosaic_rgba_50x38",
            x,
            (56.2, 54.7),
        ),
        g(
            "cli_xubc7_q60_rgba_64x48",
            "rgba_64x48",
            BasisEncodeOptions {
                xubc7_quality: 60,
                ..x
            },
            (31.1, 38.7),
        ),
        g(
            "cli_xubc7_e0_mosaic_rgb_50x38",
            "mosaic_rgb_50x38",
            BasisEncodeOptions {
                xubc7_effort: 0,
                ..x
            },
            (33.8, 54.3),
        ),
        g(
            "cli_xubc7_e10_mip_mosaic_rgb_50x38",
            "mosaic_rgb_50x38",
            BasisEncodeOptions {
                xubc7_effort: 10,
                mipmaps: true,
                ..x
            },
            (33.6, 61.2),
        ),
        g(
            "cli_xubc7_rdo60_mosaic_rgba_50x38",
            "mosaic_rgba_50x38",
            BasisEncodeOptions {
                xubc7_rdo_level: 60,
                ..x
            },
            (53.6, 59.3),
        ),
        g(
            "cli_xubc7_bc7e4_linear_rgba_64x48",
            "rgba_64x48",
            BasisEncodeOptions {
                xubc7_encoder: Xubc7Encoder::Bc7e(4),
                srgb: false,
                ..x
            },
            (37.1, 38.7),
        ),
        g(
            "cli_xubc7_bc7e_q80_e5_mosaic_rgba_50x38",
            "mosaic_rgba_50x38",
            BasisEncodeOptions {
                xubc7_encoder: Xubc7Encoder::Bc7e(2),
                xubc7_quality: 80,
                xubc7_effort: 5,
                ..x
            },
            (55.2, 53.1),
        ),
    ]
}

fn golden_file(name: &str, ext: &str) -> R<Vec<u8>> {
    Ok(std::fs::read(format!("{TESTDATA}basisu_enc/{name}.{ext}"))?)
}

fn find(name: &str) -> R<Golden> {
    Ok(goldens()
        .into_iter()
        .find(|g| g.name == name)
        .ok_or("golden")?)
}

fn encode(g: &Golden, threads: u32) -> R<Vec<u8>> {
    let (w, h, px) = raw(g.image)?;
    Ok(imgio::basis::encode_basis(
        &[Surface::new(Format::Rgba8Unorm, w, h, &px)?],
        &BasisEncodeOptions { threads, ..g.opts },
    )?
    .to_vec())
}

/// Level 0 of `file` against its source: (RGB, alpha) PSNR, via RGBA8 and via BC7 (the XUBC7
/// native target), which must agree.
fn quality(g: &Golden, file: &[u8]) -> R<(f64, f64)> {
    let (w, h, px) = raw(g.image)?;
    let b = BasisFile::parse(file)?;
    assert_eq!(b.tex_format(), BasisTexFormat::Xubc7, "{}", g.name);
    assert_eq!(b.is_srgb(), g.opts.srgb, "{}", g.name);
    assert_eq!(
        b.has_alpha(),
        px.chunks(4).any(|p| p[3] != 255),
        "{}",
        g.name
    );
    let out = b.transcode_to_buffer(0, 0, Format::Rgba8Unorm)?;
    let bc7 = b.transcode_to_buffer(0, 0, Format::Bc7Unorm)?;
    let dec = imgio::decode_to_buffer(Format::Bc7Unorm, &bc7, w, h)?;
    assert!(dec[..] == out[..], "{}", g.name);
    Ok((psnr(&px, &out, 0..3), psnr(&px, &out, 3..4)))
}

/// `encode_basis` writes the CLI's files byte for byte (the output does not depend on the
/// thread count).
#[test]
fn matches_basisu_tool() -> R<()> {
    for g in goldens() {
        let golden = golden_file(g.name, "basis")?;
        for threads in [1, 4] {
            let ours = encode(&g, threads)?;
            assert!(
                ours == golden,
                "{} threads {threads}: {} vs {} bytes",
                g.name,
                ours.len(),
                golden.len()
            );
        }
    }
    Ok(())
}

/// `ktx2::encode_basis` with the CLI's key/values = the CLI's `-ktx2` file.
#[test]
fn ktx2_matches_basisu_tool() -> R<()> {
    let g = find("cli_xubc7_rdo60_mosaic_rgba_50x38")?;
    let want = golden_file(g.name, "ktx2")?;
    let reference = Ktx2File::parse(&want)?;
    assert_eq!(reference.tex_format(), Some(BasisTexFormat::Xubc7));
    let kv: Vec<(String, Vec<u8>)> = reference
        .key_values()
        .filter(|(k, _)| k.first() != Some(&127))
        .map(|(k, v)| (String::from_utf8_lossy(k).into_owned(), v.to_vec()))
        .collect();
    let kv: Vec<(&str, &[u8])> = kv.iter().map(|(k, v)| (k.as_str(), &v[..])).collect();
    let (w, h, px) = raw(g.image)?;
    let src = [Surface::new(Format::Rgba8Unorm, w, h, &px)?];
    let ktx2_opts = BasisKtx2Options {
        zstd: None,
        key_values: &kv,
    };
    let got = imgio::ktx2::encode_basis(&src, &g.opts, &ktx2_opts)?;
    assert!(got[..] == want[..], "{} vs {} bytes", got.len(), want.len());
    // Same pixels as the .basis.
    let f = Ktx2File::parse(&got)?;
    let basis = golden_file(g.name, "basis")?;
    let b = BasisFile::parse(&basis)?;
    assert_eq!(
        &f.transcode_to_buffer(0, 0, 0, Format::Rgba8Unorm)?[..],
        &b.transcode_to_buffer(0, 0, Format::Rgba8Unorm)?[..]
    );
    Ok(())
}

/// Round trip through `BasisFile::parse` + transcode, every level; option effects.
#[test]
fn round_trip() -> R<()> {
    for g in goldens() {
        let file = encode(&g, 0)?;
        let (rgb, a) = quality(&g, &file)?;
        eprintln!("{}: {} bytes, {rgb:.2} / {a:.2} dB", g.name, file.len());
        assert!(rgb >= g.min_psnr.0 && a >= g.min_psnr.1, "{}", g.name);
        let b = BasisFile::parse(&file)?;
        if g.opts.mipmaps {
            assert!(b.level_count(0) >= 6, "{}", g.name);
        }
        for level in 0..b.level_count(0) {
            let info = b.level_info(0, level)?;
            let px = b.transcode_to_buffer(0, level, Format::Rgba8Unorm)?;
            assert_eq!(px.len(), (info.width * info.height * 4) as usize);
        }
    }
    // Lower quality / RDO: smaller files (DCT pays off on the noisy image, RDO on the mosaic).
    let size = |image, q, rdo| -> R<usize> {
        let (w, h, px) = raw(image)?;
        let src = [Surface::new(Format::Rgba8Unorm, w, h, &px)?];
        Ok(imgio::basis::encode_basis(
            &src,
            &BasisEncodeOptions {
                codec: BasisCodec::Xubc7,
                xubc7_quality: q,
                xubc7_rdo_level: rdo,
                ..Default::default()
            },
        )?
        .len())
    };
    assert!(size("rgba_64x48", 30, 0)? < size("rgba_64x48", 100, 0)?);
    assert!(size("mosaic_rgba_50x38", 100, 80)? < size("mosaic_rgba_50x38", 100, 0)?);
    Ok(())
}

#[test]
fn bad_options() -> R<()> {
    let px = [7u8; 64];
    let s = Surface::new(Format::Rgba8Unorm, 4, 4, &px)?;
    let x = BasisEncodeOptions {
        codec: BasisCodec::Xubc7,
        ..Default::default()
    };
    assert_eq!(x.codec.tex_format(), BasisTexFormat::Xubc7);
    assert!(!x.codec.is_hdr());
    let bad = [
        BasisEncodeOptions {
            xubc7_quality: 0,
            ..x
        },
        BasisEncodeOptions {
            xubc7_quality: 101,
            ..x
        },
        BasisEncodeOptions {
            xubc7_effort: 11,
            ..x
        },
        BasisEncodeOptions {
            xubc7_rdo_level: 101,
            ..x
        },
        BasisEncodeOptions {
            xubc7_stripes: 0,
            ..x
        },
        BasisEncodeOptions {
            xubc7_stripes: 17,
            ..x
        },
        BasisEncodeOptions {
            xubc7_encoder: Xubc7Encoder::Bc7e(7),
            ..x
        },
        BasisEncodeOptions {
            ldr_channel_weights: Some([1, 0, 1, 1]),
            ..x
        },
    ];
    for opts in bad {
        assert!(
            matches!(
                imgio::basis::encode_basis(&[s], &opts),
                Err(Error::InvalidArgument)
            ),
            "{opts:?}"
        );
    }
    assert!(matches!(
        imgio::basis::encode_basis(&[], &x),
        Err(Error::InvalidArgument)
    ));
    // Valid extremes encode.
    for opts in [
        BasisEncodeOptions {
            xubc7_quality: 1,
            xubc7_effort: 10,
            xubc7_rdo_level: 100,
            xubc7_stripes: 16,
            xubc7_encoder: Xubc7Encoder::Bc7e(6),
            ldr_channel_weights: Some([256, 1, 1, 256]),
            ..x
        },
        BasisEncodeOptions {
            xubc7_effort: 0,
            xubc7_stripes: 1,
            xubc7_encoder: Xubc7Encoder::Bc7e(0),
            srgb: false,
            mipmaps: true,
            y_flip: true,
            ..x
        },
    ] {
        let r = imgio::basis::encode_basis(&[s], &opts);
        assert!(r.is_ok(), "{r:?}");
    }
    Ok(())
}
