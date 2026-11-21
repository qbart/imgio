//! `imgio::encode` to ASTC is byte identical with the C++ astc-encoder 6.0 CLI: the
//! `testdata/astcenc/cli/*.astc` references were written by `astcenc-avx2` from the raw
//! `testdata/images` copies (`scripts/gen_astcenc_testdata.py`; `-cl` / `-cs` / `-cH`, default
//! options). `Quality` maps to the CLI presets Fastest = `-fastest`, Fast = `-fast`, Default =
//! `-medium`, High = `-thorough`, Best = `-exhaustive`; unorm / sRGB / float formats to the LDR,
//! LDR sRGB and HDR profiles. `EncodeOptions::rdo_lambda` maps to `-rdo-lambda` (LZ-RDO).

use imgio::astc_file::AstcFile;
use imgio::{AstcBlock, AstcKind, EncodeOptions, Format, Quality, Surface};
use std::error::Error;

const RAW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/images/raw/");
const CLI: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/astcenc/cli/");

#[test]
fn matches_astcenc_cli() -> Result<(), Box<dyn Error>> {
    use AstcBlock::*;
    use AstcKind::{Float, Srgb, Unorm};
    // (reference, raw source, block, kind, quality); keep in sync with the generator script
    let cases = [
        (
            "rgba_64x48_4x4_l_medium",
            "rgba_64x48.rgba",
            B4x4,
            Unorm,
            Quality::Default,
        ),
        (
            "rgba_64x48_6x6_s_fast",
            "rgba_64x48.rgba",
            B6x6,
            Srgb,
            Quality::Fast,
        ),
        (
            "rgba_64x48_10x5_l_fastest",
            "rgba_64x48.rgba",
            B10x5,
            Unorm,
            Quality::Fastest,
        ),
        (
            "rgb_37x29_5x4_l_thorough",
            "rgb_37x29.rgba",
            B5x4,
            Unorm,
            Quality::High,
        ),
        (
            "rgb_37x29_8x5_s_exhaustive",
            "rgb_37x29.rgba",
            B8x5,
            Srgb,
            Quality::Best,
        ),
        (
            "mosaic_rgba_50x38_8x8_l_exhaustive",
            "mosaic_rgba_50x38.rgba",
            B8x8,
            Unorm,
            Quality::Best,
        ),
        (
            "mosaic_rgb_50x38_12x12_s_medium",
            "mosaic_rgb_50x38.rgba",
            B12x12,
            Srgb,
            Quality::Default,
        ),
        (
            "mosaic_rgba_50x38_4x4_s_thorough",
            "mosaic_rgba_50x38.rgba",
            B4x4,
            Srgb,
            Quality::High,
        ),
        (
            "hdr_40x24_6x6_H_medium",
            "hdr_40x24.rgba16f",
            B6x6,
            Float,
            Quality::Default,
        ),
        (
            "hdr_40x24_4x4_H_thorough",
            "hdr_40x24.rgba16f",
            B4x4,
            Float,
            Quality::High,
        ),
        (
            "hdr_40x24_12x12_H_fastest",
            "hdr_40x24.rgba16f",
            B12x12,
            Float,
            Quality::Fastest,
        ),
    ];
    for (name, src, block, kind, quality) in cases {
        check(name, src, block, kind, quality, 0.0)?;
    }
    Ok(())
}

#[test]
fn matches_astcenc_cli_lz_rdo() -> Result<(), Box<dyn Error>> {
    use AstcBlock::*;
    use AstcKind::{Srgb, Unorm};
    // (reference, raw source, block, kind, quality, lambda); keep in sync with the generator
    let cases = [
        (
            "rgba_64x48_4x4_l_medium_rdo20",
            "rgba_64x48.rgba",
            B4x4,
            Unorm,
            Quality::Default,
            20.0,
        ),
        (
            "rgba_64x48_6x6_s_fast_rdo10",
            "rgba_64x48.rgba",
            B6x6,
            Srgb,
            Quality::Fast,
            10.0,
        ),
        (
            "rgba_64x48_8x8_s_fast_rdo20",
            "rgba_64x48.rgba",
            B8x8,
            Srgb,
            Quality::Fast,
            20.0,
        ),
        (
            "rgb_37x29_4x4_s_thorough_rdo10",
            "rgb_37x29.rgba",
            B4x4,
            Srgb,
            Quality::High,
            10.0,
        ),
        (
            "mosaic_rgb_50x38_6x6_l_medium_rdo40",
            "mosaic_rgb_50x38.rgba",
            B6x6,
            Unorm,
            Quality::Default,
            40.0,
        ),
        (
            "mosaic_rgba_50x38_8x8_l_thorough_rdo40",
            "mosaic_rgba_50x38.rgba",
            B8x8,
            Unorm,
            Quality::High,
            40.0,
        ),
        // Two 4096 block optimization chunks
        (
            "tiled_rgba_320x240_4x4_l_medium_rdo20",
            "tiled_rgba_320x240.rgba",
            B4x4,
            Unorm,
            Quality::Default,
            20.0,
        ),
    ];
    for (name, src, block, kind, quality, rdo_lambda) in cases {
        check(name, src, block, kind, quality, rdo_lambda)?;
    }
    Ok(())
}

/// A `tw` x `th` RGBA8 image tiled from the `w` x `h` `rgba`; tile (tx, ty) is shifted by
/// (13 * ty, 7 * tx) pixels (wrapping), as `scripts/gen_astcenc_testdata.py` `tiled` makes it.
fn tiled(rgba: &[u8], (w, h): (usize, usize), (tw, th): (usize, usize)) -> Vec<u8> {
    let mut out = vec![0; tw * th * 4];
    for y in 0..th {
        for x in 0..tw {
            let sx = (x + 13 * (y / h)) % w;
            let sy = (y + 7 * (x / w)) % h;
            out[(y * tw + x) * 4..][..4].copy_from_slice(&rgba[(sy * w + sx) * 4..][..4]);
        }
    }
    out
}

/// Encodes raw source `src` (`tiled_rgba_320x240.rgba` is derived from `rgba_64x48.rgba`) and
/// compares the blocks with the CLI reference `name`.
fn check(
    name: &str,
    src: &str,
    block: AstcBlock,
    kind: AstcKind,
    quality: Quality,
    rdo_lambda: f32,
) -> Result<(), Box<dyn Error>> {
    let dims = src.split('.').next().and_then(|s| s.rsplit('_').next());
    let (w, h) = dims.and_then(|d| d.split_once('x')).ok_or("source dims")?;
    let (w, h): (u32, u32) = (w.parse()?, h.parse()?);
    let data = if src == "tiled_rgba_320x240.rgba" {
        let base = std::fs::read(format!("{RAW}rgba_64x48.rgba"))?;
        tiled(&base, (64, 48), (320, 240))
    } else {
        std::fs::read(format!("{RAW}{src}"))?
    };
    let src_format = if src.ends_with(".rgba16f") {
        Format::Rgba16Float
    } else {
        Format::Rgba8Unorm
    };
    let surface = Surface::new(src_format, w, h, &data)?;
    let format = Format::Astc(block, kind);
    let opts = EncodeOptions {
        quality,
        rdo_lambda,
    };
    let ours = imgio::encode_to_buffer(&surface, format, &opts)?;

    let file = std::fs::read(format!("{CLI}{name}.astc"))?;
    let reference = AstcFile::parse(&file)?;
    let (bx, by) = block.dim();
    assert_eq!(
        (reference.block(), reference.width(), reference.height()),
        ((bx, by, 1), w, h),
        "{name}: header"
    );
    let blocks = ours.len() / 16;
    let diff = (0..blocks)
        .filter(|&i| ours[i * 16..i * 16 + 16] != reference.data()[i * 16..i * 16 + 16])
        .count();
    assert!(
        ours.len() == reference.data().len() && diff == 0,
        "{name}: {diff} of {blocks} blocks differ from astcenc 6.0"
    );
    Ok(())
}
