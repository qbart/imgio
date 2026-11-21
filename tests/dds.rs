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
        };
        let out = dds::write(&desc, &refs, &DdsWriteOptions::default())?;
        assert!(out[..] == data[..], "{name}: rewrite differs");
    }
    // BGRA8 has no imgio format.
    let data = std::fs::read(format!("{DIR}bgra8.dds"))?;
    assert_eq!(DdsFile::parse(&data).err(), Some(imgio::Error::Unsupported));
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
    };
    let opts = DdsWriteOptions { legacy: true };
    assert!(dds::write(&desc, &[&[0u8; 16]], &opts).is_err());
    assert!(dds::write(&desc, &[&[0u8; 15]], &DdsWriteOptions::default()).is_err());
    assert!(DdsFile::parse(&[0u8; 200]).is_err());
    Ok(())
}
