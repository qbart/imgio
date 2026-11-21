//! KTX1 reader / writer vs files written by Basis Universal v2.50 (`basisu -export_ktx`,
//! testdata/ktx1, from KTX2 test files): every image equals imgio's transcode of the source
//! KTX2, and the writer reproduces each file byte for byte.

use imgio::ktx1::{self, Ktx1Desc, Ktx1File};
use imgio::ktx2::Ktx2File;
use imgio::{AstcBlock, AstcKind, Format};
use std::error::Error;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn basisu_export_ktx() -> Result<(), Box<dyn Error>> {
    let astc4 = Format::Astc(AstcBlock::B4x4, AstcKind::Unorm);
    // (file, source KTX2, transcode target, expected KTX1 format)
    let cases = [
        (
            "etc1s_rgba_BC1",
            "ktx2/etc1s_rgba",
            Format::Bc1Srgb,
            Format::Bc1Srgb,
        ),
        (
            "etc1s_rgba_BC3",
            "ktx2/etc1s_rgba",
            Format::Bc3Srgb,
            Format::Bc3Srgb,
        ),
        (
            "etc1s_rgba_BC7",
            "ktx2/etc1s_rgba",
            Format::Bc7Srgb,
            Format::Bc7Srgb,
        ),
        (
            "etc1s_rgba_ETC1",
            "ktx2/etc1s_rgba",
            Format::Etc1Rgb,
            Format::Etc1Rgb,
        ),
        (
            "etc1s_rgba_ETC2",
            "ktx2/etc1s_rgba",
            Format::Etc2RgbaSrgb,
            Format::Etc2RgbaSrgb,
        ),
        (
            "etc1s_rgba_ASTC",
            "ktx2/etc1s_rgba",
            astc4,
            Format::Astc(AstcBlock::B4x4, AstcKind::Srgb),
        ),
        (
            "etc1s_array_BC4",
            "ktx2/etc1s_array",
            Format::Bc4Unorm,
            Format::Bc4Unorm,
        ),
        (
            "etc1s_array_BC5",
            "ktx2/etc1s_array",
            Format::Bc5Unorm,
            Format::Bc5Unorm,
        ),
        (
            "etc1s_array_EAC_R11",
            "ktx2/etc1s_array",
            Format::EacR11Unorm,
            Format::EacR11Unorm,
        ),
        (
            "etc1s_array_EAC_RG11",
            "ktx2/etc1s_array",
            Format::EacRg11Unorm,
            Format::EacRg11Unorm,
        ),
        (
            "uastc_hdr4x4_BC6H",
            "ktx2/uastc_hdr4x4",
            Format::Bc6hUfloat,
            Format::Bc6hUfloat,
        ),
        (
            "uastc_hdr4x4_ASTC_HDR_4x4",
            "ktx2/uastc_hdr4x4",
            Format::Astc(AstcBlock::B4x4, AstcKind::Float),
            astc4,
        ),
        (
            "xubc7_rgba_PVRTC1",
            "ktx2/xubc7_rgba",
            Format::Pvrtc1Rgba,
            Format::Pvrtc1Rgba,
        ),
        (
            "cube_32_BC1",
            "ktx1/cube_32",
            Format::Bc1Srgb,
            Format::Bc1Srgb,
        ),
        (
            "cube_32_BC7",
            "ktx1/cube_32",
            Format::Bc7Srgb,
            Format::Bc7Srgb,
        ),
        (
            "cube_32_PVRTC2",
            "ktx1/cube_32",
            Format::Pvrtc2Rgba,
            Format::Pvrtc2Rgba,
        ),
    ];
    for (name, src, target, want) in cases {
        let data = std::fs::read(format!("{ROOT}/testdata/ktx1/{name}.ktx"))?;
        let f = Ktx1File::parse(&data)?;
        let k2data = std::fs::read(format!("{ROOT}/testdata/{src}.ktx2"))?;
        let k2 = Ktx2File::parse(&k2data)?;
        assert_eq!(f.format(), want, "{name}");
        assert_eq!(
            (
                f.width(),
                f.height(),
                f.level_count(),
                f.layer_count(),
                f.face_count()
            ),
            (
                k2.width(),
                k2.height(),
                k2.level_count(),
                k2.layer_count(),
                k2.face_count()
            ),
            "{name}"
        );
        let mut levels = Vec::new();
        for level in 0..f.level_count() {
            let mut l = Vec::new();
            for layer in 0..f.layer_count() {
                for face in 0..f.face_count() {
                    let img = f.image(level, layer, face)?;
                    let t = k2.transcode_to_buffer(level, layer, face, target)?;
                    assert!(img[..] == t[..], "{name} {level} {layer} {face}");
                    l.extend_from_slice(&img);
                }
            }
            levels.push(l);
        }
        let refs: Vec<&[u8]> = levels.iter().map(|l| &l[..]).collect();
        let desc = Ktx1Desc {
            format: f.format(),
            width: f.width(),
            height: f.height(),
            layers: if f.layer_count() > 1 {
                f.layer_count()
            } else {
                0
            },
            faces: f.face_count(),
            levels: f.level_count(),
            srgb: f.is_srgb(),
        };
        let out = ktx1::write(&desc, &refs, &[])?;
        assert!(
            out[..] == data[..],
            "{name}: rewrite differs ({} vs {} bytes)",
            out.len(),
            data.len()
        );
    }
    Ok(())
}

#[test]
fn uncompressed_round_trip() -> Result<(), Box<dyn Error>> {
    // Rows of 3 RGB565 / R8 texels need 4-byte row padding in the file.
    for format in [
        Format::R8Unorm,
        Format::Rgb565,
        Format::Rgba16Float,
        Format::Rgba8Unorm,
    ] {
        let (w, h) = (3, 5);
        let levels: Vec<Vec<u8>> = (0..3)
            .map(|l| {
                let (lw, lh) = imgio::image::mip_dim(w, h, l);
                let n = format.surface_bytes(lw, lh).unwrap_or(0);
                (0..n).map(|i| (i * 13 + l as usize) as u8).collect()
            })
            .collect();
        let refs: Vec<&[u8]> = levels.iter().map(|l| &l[..]).collect();
        let desc = Ktx1Desc {
            format,
            width: w,
            height: h,
            layers: 0,
            faces: 1,
            levels: 3,
            srgb: false,
        };
        let file = ktx1::write(&desc, &refs, &[("KTXorientation", b"S=r,T=d\0")])?;
        let f = Ktx1File::parse(&file)?;
        assert_eq!(f.format(), format);
        assert!(
            f.key_values()
                .any(|(k, v)| k == b"KTXorientation" && v == b"S=r,T=d\0")
        );
        for (l, want) in levels.iter().enumerate() {
            assert_eq!(
                &f.image(l as u32, 0, 0)?[..],
                &want[..],
                "{format:?} level {l}"
            );
        }
    }
    Ok(())
}
