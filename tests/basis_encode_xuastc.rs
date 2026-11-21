//! ASTC LDR / XUASTC LDR `encode_basis` against the basisu v2.50 CLI goldens
//! (testdata/basisu_enc/cli_{astc,xuastc}*.basis, `scripts/gen_basis_testdata.sh`): byte-exact
//! files, round trip quality, option validation.
//!
//! Everything around the block encoder (sources, mipmaps, config / key-value mapping, ASTC
//! packing, the `.basis` writer) is checked byte for byte against the same goldens by the unit
//! test `basis::encode::tests::writer_matches_basisu_goldens`, which injects the goldens' own
//! encoder outputs.

use imgio::basis::{
    BasisCodec, BasisEncodeOptions, BasisFile, BasisTexFormat, XuastcDeblocking, XuastcSyntax,
    encode_basis,
};
use imgio::{AstcBlock, Error, Format, Surface};
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
    /// Minimum level 0 (RGB, alpha) PSNR in dB (the golden reaches 0.5..0.6 dB more).
    min_psnr: (f64, f64),
}

/// The CLI goldens: `-basis -ldr_WxH[i]` plus the options in the name (see
/// scripts/gen_basis_testdata.sh).
fn goldens() -> Vec<Golden> {
    use AstcBlock::*;
    use BasisCodec::{AstcLdr, XuastcLdr};
    let o = |codec| BasisEncodeOptions {
        codec,
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
            "cli_astc4x4_rgba_64x48",
            "rgba_64x48",
            o(AstcLdr(B4x4)),
            (34.0, 37.3),
        ),
        g(
            "cli_astc4x4_e0_rgb_37x29",
            "rgb_37x29",
            BasisEncodeOptions {
                astc_ldr_effort: 0,
                ..o(AstcLdr(B4x4))
            },
            (33.2, 99.0),
        ),
        g(
            "cli_astc6x6_linear_mip_rgb_37x29",
            "rgb_37x29",
            BasisEncodeOptions {
                srgb: false,
                mipmaps: true,
                ..o(AstcLdr(B6x6))
            },
            (31.3, 99.0),
        ),
        g(
            "cli_astc8x6_e1_rgba_64x48",
            "rgba_64x48",
            BasisEncodeOptions {
                astc_ldr_effort: 1,
                ..o(AstcLdr(B8x6))
            },
            (28.0, 33.1),
        ),
        g(
            "cli_astc12x12_rgba_64x48",
            "rgba_64x48",
            o(AstcLdr(B12x12)),
            (22.0, 29.8),
        ),
        g(
            "cli_xuastc4x4_zstd_rgb_37x29",
            "rgb_37x29",
            o(XuastcLdr(B4x4)),
            (33.4, 99.0),
        ),
        g(
            "cli_xuastc5x5_e5_mip_rgb_37x29",
            "rgb_37x29",
            BasisEncodeOptions {
                astc_ldr_effort: 5,
                mipmaps: true,
                ..o(XuastcLdr(B5x5))
            },
            (28.4, 99.0),
        ),
        g(
            "cli_xuastc6x6_arith_mip_rgba_64x48",
            "rgba_64x48",
            BasisEncodeOptions {
                xuastc_syntax: XuastcSyntax::Arith,
                mipmaps: true,
                ..o(XuastcLdr(B6x6))
            },
            (32.5, 35.2),
        ),
        g(
            "cli_xuastc6x5_hybrid_linear_rgba_64x48",
            "rgba_64x48",
            BasisEncodeOptions {
                xuastc_syntax: XuastcSyntax::Hybrid,
                srgb: false,
                ..o(XuastcLdr(B6x5))
            },
            (31.5, 34.6),
        ),
        g(
            "cli_xuastc6x6_q60_rgba_64x48",
            "rgba_64x48",
            BasisEncodeOptions {
                xuastc_quality: 60,
                ..o(XuastcLdr(B6x6))
            },
            (30.6, 37.2),
        ),
        g(
            "cli_xuastc6x6_hybrid_q60_rgba_64x48",
            "rgba_64x48",
            BasisEncodeOptions {
                xuastc_syntax: XuastcSyntax::Hybrid,
                xuastc_quality: 60,
                ..o(XuastcLdr(B6x6))
            },
            (30.6, 37.2),
        ),
        g(
            "cli_xuastc8x6_arith_q60_mosaic_rgba_50x38",
            "mosaic_rgba_50x38",
            BasisEncodeOptions {
                xuastc_syntax: XuastcSyntax::Arith,
                xuastc_quality: 60,
                ..o(XuastcLdr(B8x6))
            },
            (40.2, 46.9),
        ),
        g(
            "cli_xuastc12x12_rgba_64x48",
            "rgba_64x48",
            o(XuastcLdr(B12x12)),
            (22.0, 29.8),
        ),
        g(
            "cli_xuastc12x12_nodeblock_rgba_64x48",
            "rgba_64x48",
            BasisEncodeOptions {
                xuastc_deblocking: XuastcDeblocking::Off,
                ..o(XuastcLdr(B12x12))
            },
            (21.8, 29.7),
        ),
    ]
}

fn golden_file(g: &Golden) -> R<Vec<u8>> {
    Ok(std::fs::read(format!(
        "{TESTDATA}basisu_enc/{}.basis",
        g.name
    ))?)
}

fn encode(g: &Golden, threads: u32) -> R<Vec<u8>> {
    let (w, h, px) = raw(g.image)?;
    Ok(encode_basis(
        &[Surface::new(Format::Rgba8Unorm, w, h, &px)?],
        &BasisEncodeOptions { threads, ..g.opts },
    )?
    .to_vec())
}

/// Level 0 of `file` against its source: (RGB, alpha) PSNR.
fn quality(g: &Golden, file: &[u8]) -> R<(f64, f64)> {
    let (_, _, px) = raw(g.image)?;
    let b = BasisFile::parse(file)?;
    assert_eq!(b.tex_format(), g.opts.codec.tex_format(), "{}", g.name);
    assert_eq!(b.is_srgb(), g.opts.srgb, "{}", g.name);
    assert_eq!(
        b.has_alpha(),
        px.chunks(4).any(|p| p[3] != 255),
        "{}",
        g.name
    );
    let out = b.transcode_to_buffer(0, 0, Format::Rgba8Unorm)?;
    Ok((psnr(&px, &out, 0..3), psnr(&px, &out, 3..4)))
}

/// The goldens themselves decode with the expected quality (calibrates `min_psnr`).
#[test]
fn goldens_decode() -> R<()> {
    for g in goldens() {
        let file = golden_file(&g)?;
        let (rgb, a) = quality(&g, &file)?;
        eprintln!("{}: {rgb:.2} / {a:.2} dB", g.name);
        assert!(rgb >= g.min_psnr.0 && a >= g.min_psnr.1, "{}", g.name);
        let b = BasisFile::parse(&file)?;
        let levels = if g.opts.mipmaps { 6 } else { 1 };
        assert!(b.level_count(0) >= levels, "{}", g.name);
    }
    Ok(())
}

/// `encode_basis` writes the CLI's files byte for byte (the output does not depend on the
/// thread count).
#[test]
fn matches_basisu_tool() -> R<()> {
    for g in goldens() {
        let golden = golden_file(&g)?;
        for threads in [1, 4] {
            let ours = encode(&g, threads)?;
            assert!(ours == golden, "{} threads {threads}", g.name);
        }
    }
    Ok(())
}

/// Round trip through `BasisFile::parse` + transcode to RGBA8, every level.
#[test]
fn round_trip() -> R<()> {
    for g in goldens() {
        let file = encode(&g, 0)?;
        let (rgb, a) = quality(&g, &file)?;
        eprintln!("{}: {} bytes, {rgb:.2} / {a:.2} dB", g.name, file.len());
        assert!(rgb >= g.min_psnr.0 && a >= g.min_psnr.1, "{}", g.name);
        let b = BasisFile::parse(&file)?;
        for level in 0..b.level_count(0) {
            let info = b.level_info(0, level)?;
            let px = b.transcode_to_buffer(0, level, Format::Rgba8Unorm)?;
            assert_eq!(px.len(), (info.width * info.height * 4) as usize);
        }
    }
    // Lower quality: smaller XUASTC files.
    let (w, h, px) = raw("rgba_64x48")?;
    let src = [Surface::new(Format::Rgba8Unorm, w, h, &px)?];
    let size = |q| -> R<usize> {
        Ok(encode_basis(
            &src,
            &BasisEncodeOptions {
                codec: BasisCodec::XuastcLdr(AstcBlock::B6x6),
                xuastc_quality: q,
                ..Default::default()
            },
        )?
        .len())
    };
    assert!(size(30)? < size(100)?);
    Ok(())
}

#[test]
fn bad_options() -> R<()> {
    let px = [7u8; 64];
    let s = Surface::new(Format::Rgba8Unorm, 4, 4, &px)?;
    let x = |b| BasisEncodeOptions {
        codec: BasisCodec::XuastcLdr(b),
        ..Default::default()
    };
    let bad = [
        BasisEncodeOptions {
            astc_ldr_effort: 11,
            ..x(AstcBlock::B6x6)
        },
        BasisEncodeOptions {
            xuastc_quality: 0,
            ..x(AstcBlock::B6x6)
        },
        BasisEncodeOptions {
            xuastc_quality: 101,
            ..x(AstcBlock::B4x4)
        },
        BasisEncodeOptions {
            ldr_channel_weights: Some([1, 0, 1, 1]),
            ..x(AstcBlock::B8x8)
        },
        BasisEncodeOptions {
            ldr_channel_weights: Some([1, 1, 1, 257]),
            codec: BasisCodec::AstcLdr(AstcBlock::B12x12),
            ..Default::default()
        },
    ];
    for opts in bad {
        assert!(
            matches!(encode_basis(&[s], &opts), Err(Error::InvalidArgument)),
            "{opts:?}"
        );
    }
    // Empty input, zero size.
    assert!(matches!(
        encode_basis(&[], &x(AstcBlock::B6x6)),
        Err(Error::InvalidArgument)
    ));
    // Every block size maps to its basisu tex format.
    for b in AstcBlock::ALL {
        assert_eq!(
            BasisCodec::XuastcLdr(b).tex_format(),
            BasisTexFormat::XuastcLdr(b)
        );
        assert_eq!(
            BasisCodec::AstcLdr(b).tex_format(),
            BasisTexFormat::AstcLdr(b)
        );
        assert!(!BasisCodec::AstcLdr(b).is_hdr() && !BasisCodec::XuastcLdr(b).is_hdr());
    }
    // Valid extremes pass validation and encode.
    for opts in [
        BasisEncodeOptions {
            astc_ldr_effort: 10,
            xuastc_quality: 1,
            ldr_channel_weights: Some([256, 1, 1, 256]),
            ..x(AstcBlock::B12x12)
        },
        BasisEncodeOptions {
            astc_ldr_effort: 0,
            ..x(AstcBlock::B4x4)
        },
    ] {
        let r = encode_basis(&[s], &opts);
        assert!(r.is_ok(), "{r:?}");
    }
    Ok(())
}
