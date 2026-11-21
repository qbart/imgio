//! DDS reader / writer vs files written by the Basis Universal v2.50 CLI (`basisu -dds`,
//! testdata/dds): every file parses, and rewriting its images reproduces it byte for byte.

use imgio::Format;
use imgio::dds::{self, DdsDesc, DdsFile, DdsWriteOptions};
use std::error::Error;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/dds/");

/// The images of `f` in the writer's input layout (per level: layers, then faces).
fn levels(f: &DdsFile) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let mut out = Vec::new();
    for level in 0..f.level_count() {
        let mut l = Vec::new();
        for layer in 0..f.layer_count() {
            for face in 0..f.face_count() {
                l.extend_from_slice(f.image_data(level, layer, face)?);
            }
        }
        out.push(l);
    }
    Ok(out)
}

#[test]
fn basisu_files_round_trip() -> Result<(), Box<dyn Error>> {
    // name, format, width, height, levels, layers, faces
    let cases = [
        ("bc7_mips", Format::Bc7Srgb, 64, 48, 7, 1, 1),
        ("bc1_srgb", Format::Bc1Srgb, 37, 29, 1, 1, 1),
        ("bc3_cube_mips", Format::Bc3Unorm, 32, 32, 6, 1, 6),
        ("bc4", Format::Bc4Unorm, 37, 29, 1, 1, 1),
        ("bc5_array", Format::Bc5Unorm, 64, 48, 1, 2, 1),
        ("bc2", Format::Bc2Srgb, 50, 38, 1, 1, 1),
        ("rgba8_mips", Format::Rgba8Srgb, 37, 29, 6, 1, 1),
        ("rgba8_srgb", Format::Rgba8Srgb, 32, 32, 1, 1, 1),
        ("r8", Format::R8Unorm, 37, 29, 1, 1, 1),
        ("rg8", Format::Rg8Unorm, 37, 29, 1, 1, 1),
        ("rgb565", Format::Rgb565, 37, 29, 1, 1, 1),
    ];
    for (name, format, w, h, mips, layers, faces) in cases {
        let data = std::fs::read(format!("{DIR}{name}.dds"))?;
        let f = DdsFile::parse(&data)?;
        assert_eq!(
            (
                f.format(),
                f.width(),
                f.height(),
                f.level_count(),
                f.layer_count(),
                f.face_count()
            ),
            (format, w, h, mips, layers, faces),
            "{name}"
        );
        assert!(f.is_dx10());
        let lv = levels(&f)?;
        let refs: Vec<&[u8]> = lv.iter().map(|l| &l[..]).collect();
        let desc = DdsDesc {
            format,
            width: w,
            height: h,
            layers,
            faces,
            levels: mips,
            depth: 0,
        };
        let out = dds::write(&desc, &refs, &DdsWriteOptions::default())?;
        assert!(out[..] == data[..], "{name}: rewrite differs");
    }
    // BGRA8 sRGB (DXGI 91): parses and rewrites byte-identically.
    let data = std::fs::read(format!("{DIR}bgra8.dds"))?;
    let f = DdsFile::parse(&data)?;
    assert_eq!(f.format(), Format::Bgra8Srgb);
    let lv: Vec<&[u8]> = (0..f.level_count())
        .map(|l| f.image_data(l, 0, 0))
        .collect::<Result<_, _>>()?;
    let desc = DdsDesc {
        format: f.format(),
        width: f.width(),
        height: f.height(),
        layers: 0,
        faces: 1,
        levels: f.level_count(),
        depth: 0,
    };
    let out = dds::write(
        &desc,
        &lv,
        &DdsWriteOptions {
            legacy: !f.is_dx10(),
        },
    )?;
    assert!(out[..] == data[..], "bgra8 rewrite differs");
    Ok(())
}

#[test]
fn legacy_headers() -> Result<(), Box<dyn Error>> {
    for format in [
        Format::Bc1Unorm,
        Format::Bc2Unorm,
        Format::Bc3Unorm,
        Format::Bc4Unorm,
        Format::Bc5Snorm,
        Format::Rgba8Unorm,
        Format::R8Unorm,
        Format::Rg8Unorm,
        Format::Rg16Unorm,
        Format::Rgb565,
        Format::Bgr565,
        Format::Rgba4444,
        Format::Rgba16Unorm,
        Format::Rgba16Float,
        Format::Rgba32Float,
    ] {
        for (faces, mips) in [(1, 3), (6, 1)] {
            let desc = DdsDesc {
                format,
                width: 8,
                height: 8,
                layers: 0,
                faces,
                levels: mips,
                depth: 0,
            };
            let lv: Vec<Vec<u8>> = (0..mips)
                .map(|l| {
                    let (w, h) = imgio::image::mip_dim(8, 8, l);
                    let n = format.surface_bytes(w, h).unwrap_or(0) * faces as usize;
                    (0..n).map(|i| (i * 7 + l as usize) as u8).collect()
                })
                .collect();
            let refs: Vec<&[u8]> = lv.iter().map(|l| &l[..]).collect();
            let out = dds::write(&desc, &refs, &DdsWriteOptions { legacy: true })?;
            let f = DdsFile::parse(&out)?;
            assert!(!f.is_dx10());
            assert_eq!(
                (f.format(), f.level_count(), f.face_count()),
                (format, mips, faces),
                "{format:?}"
            );
            assert_eq!(levels(&f)?, lv, "{format:?}");
        }
    }
    // No legacy header for sRGB / BC7 / arrays.
    let desc = DdsDesc {
        format: Format::Bc7Unorm,
        width: 4,
        height: 4,
        layers: 0,
        faces: 1,
        levels: 1,
        depth: 0,
    };
    let opts = DdsWriteOptions { legacy: true };
    assert!(dds::write(&desc, &[&[0u8; 16]], &opts).is_err());
    assert!(dds::write(&desc, &[&[0u8; 15]], &DdsWriteOptions::default()).is_err());
    assert!(DdsFile::parse(&[0u8; 200]).is_err());
    Ok(())
}

#[test]
fn volumes() -> Result<(), Box<dyn Error>> {
    // 8x4x6 volume, 3 levels (depth 6, 3, 1), DX10 and legacy headers.
    for (format, legacy) in [
        (Format::Rgba8Unorm, false),
        (Format::Bc1Unorm, true),
        (Format::R8Unorm, true),
    ] {
        let lv: Vec<Vec<u8>> = (0..3u32)
            .map(|l| {
                let (w, h) = imgio::image::mip_dim(8, 4, l);
                let slices = (6u32 >> l).max(1) as usize;
                let n = format.surface_bytes(w, h).unwrap_or(0) * slices;
                (0..n).map(|i| (i * 11 + l as usize * 3) as u8).collect()
            })
            .collect();
        let refs: Vec<&[u8]> = lv.iter().map(|l| &l[..]).collect();
        let desc = DdsDesc {
            format,
            width: 8,
            height: 4,
            layers: 0,
            faces: 1,
            levels: 3,
            depth: 6,
        };
        let out = dds::write(&desc, &refs, &DdsWriteOptions { legacy })?;
        let f = DdsFile::parse(&out)?;
        assert_eq!(
            (f.depth(), f.level_depth(1), f.level_depth(2), f.is_dx10()),
            (6, 3, 1, !legacy)
        );
        for (l, want) in lv.iter().enumerate() {
            assert_eq!(
                f.image_data(l as u32, 0, 0)?,
                &want[..],
                "{format:?} level {l}"
            );
        }
    }
    Ok(())
}
