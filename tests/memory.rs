//! GPU memory: computed sizes against the level data stored in every DDS / KTX1 / KTX2 test
//! file (written by KTX-Software, Microsoft texconv, basisu and imgio), known values, content
//! analysis and suggestions.

use imgio::memory::{self, Alpha, Layout, Target};
use imgio::{AstcBlock, AstcKind, Format, Surface};
use std::error::Error as StdError;
use std::path::{Path, PathBuf};

type R<T = ()> = Result<T, Box<dyn StdError>>;

fn files(dir: &Path, out: &mut Vec<PathBuf>) -> R {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            files(&p, out)?;
        } else if matches!(
            p.extension().and_then(|x| x.to_str()),
            Some("dds" | "ktx" | "ktx2" | "basis" | "astc")
        ) {
            out.push(p);
        }
    }
    Ok(())
}

/// Bytes of all images stored in a container (`None` when unknown / supercompressed).
fn stored(data: &[u8], m: &memory::ContainerMemory) -> R<Option<u64>> {
    let l = m.layout;
    let mut total = 0u64;
    if data.starts_with(b"DDS ") {
        let d = imgio::dds::DdsFile::parse(data)?;
        for lv in 0..l.levels {
            for la in 0..l.layers {
                for f in 0..l.faces {
                    total += d.image_data(lv, la, f)?.len() as u64;
                }
            }
        }
    } else if data.starts_with(&[0xAB, b'K', b'T', b'X', b' ', b'1', b'1']) {
        let k = imgio::ktx1::Ktx1File::parse(data)?;
        for lv in 0..l.levels {
            for la in 0..l.layers {
                for f in 0..l.faces {
                    total += k.image(lv, la, f)?.len() as u64;
                }
            }
        }
    } else if data.starts_with(&[0xAB, b'K', b'T', b'X', b' ', b'2', b'0']) {
        let k = imgio::ktx2::Ktx2File::parse(data)?;
        if m.basis.is_some() || m.format.is_none() {
            return Ok(None);
        }
        for lv in 0..l.levels {
            for la in 0..l.layers {
                for f in 0..l.faces {
                    match k.image_data(lv, la, f) {
                        Ok(b) => total += b.len() as u64,
                        // Zstd / zlib supercompressed levels.
                        Err(_) => return Ok(None),
                    }
                }
            }
        }
    } else {
        return Ok(None);
    }
    Ok(Some(total))
}

#[test]
fn matches_stored_level_data() -> R {
    let mut all = Vec::new();
    files(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata"),
        &mut all,
    )?;
    let (mut checked, mut basis) = (0, 0);
    for p in &all {
        let data = std::fs::read(p)?;
        let Ok(Some(m)) = memory::of_container(&data) else {
            continue;
        };
        if m.basis.is_some() {
            basis += 1;
            assert!(m.gpu_bytes.is_none());
            for t in [Target::Desktop, Target::Mobile] {
                let b = m.basis.ok_or("basis")?;
                let f = memory::transcode_target(b, m.has_alpha, m.srgb, t);
                assert!(memory::texture_bytes(f, &m.layout)? > 0);
            }
            continue;
        }
        let Some(want) = stored(&data, &m)? else {
            continue;
        };
        assert_eq!(m.gpu_bytes, Some(want), "{}: {m:?}", p.display());
        checked += 1;
    }
    assert!(checked >= 100, "{checked} containers checked");
    assert!(basis >= 20, "{basis} basis payloads");
    Ok(())
}

#[test]
fn known_sizes() -> R {
    let l = Layout::new_2d(1024, 1024, true);
    assert_eq!(l.levels, 11);
    assert_eq!(
        memory::texture_bytes(Format::Rgba8Unorm, &l)?,
        4 * 1_398_101
    );
    // 65536 + 16384 + ... + 4 + 1 + 1 + 1 blocks (2x2 and 1x1 levels still take a block).
    assert_eq!(memory::texture_bytes(Format::Bc7Srgb, &l)?, 16 * 87_383);
    assert_eq!(memory::texture_bytes(Format::Bc1Unorm, &l)?, 8 * 87_383);
    let astc = Format::Astc(AstcBlock::B6x6, AstcKind::Unorm);
    let l = Layout::new_2d(100, 60, false);
    assert_eq!(memory::texture_bytes(astc, &l)?, 17 * 10 * 16);
    let cube = Layout {
        faces: 6,
        layers: 2,
        ..Layout::new_2d(64, 64, false)
    };
    assert_eq!(
        memory::texture_bytes(Format::Rgba16Float, &cube)?,
        12 * 64 * 64 * 8
    );
    let vol = Layout {
        depth: 8,
        ..Layout::new_2d(8, 8, true)
    };
    // 8x8x8 + 4x4x4 + 2x2x2 + 1.
    assert_eq!(
        memory::texture_bytes(Format::R8Unorm, &vol)?,
        512 + 64 + 8 + 1
    );
    for bad in [
        Layout::new_2d(0, 4, false),
        Layout {
            faces: 2,
            ..Layout::new_2d(4, 4, false)
        },
        Layout {
            levels: 4,
            ..Layout::new_2d(4, 4, false)
        },
    ] {
        assert!(memory::texture_bytes(Format::Rgba8Unorm, &bad).is_err());
    }
    assert_eq!(memory::upload_format(Format::Rgb8Srgb), Format::Rgba8Srgb);
    assert!(memory::of_container(b"\x89PNG....")?.is_none());
    Ok(())
}

#[test]
fn content_and_suggestions() -> R {
    let mk = |px: &[[u8; 4]]| -> Vec<u8> { px.iter().flatten().copied().collect() };
    let opaque_color = mk(&[[10, 20, 30, 255], [1, 1, 1, 255]]);
    let cutout_gray = mk(&[[9, 9, 9, 0], [7, 7, 7, 255]]);
    let soft = mk(&[[9, 20, 9, 128], [7, 7, 7, 255]]);
    let a = |b: &[u8], f| memory::analyze(&Surface::new(f, 2, 1, b)?);
    let c = a(&opaque_color, Format::Rgba8Srgb)?;
    assert_eq!(
        (c.alpha, c.gray, c.hdr, c.srgb),
        (Alpha::Opaque, false, false, true)
    );
    assert_eq!(memory::suggest(&c, Target::Desktop), Format::Bc7Srgb);
    assert_eq!(
        memory::suggest(&c, Target::Mobile),
        Format::Astc(AstcBlock::B6x6, AstcKind::Srgb)
    );
    let c = a(&cutout_gray, Format::Rgba8Unorm)?;
    assert_eq!((c.alpha, c.gray), (Alpha::Binary, true));
    assert_eq!(memory::suggest(&c, Target::Desktop), Format::Bc5Unorm);
    let c = a(&soft, Format::Rgba8Unorm)?;
    assert_eq!(c.alpha, Alpha::Full);
    assert_eq!(
        memory::suggest(&c, Target::Mobile),
        Format::Astc(AstcBlock::B4x4, AstcKind::Unorm)
    );
    let g = [5u8, 6];
    let c = memory::analyze(&Surface::new(Format::R8Unorm, 2, 1, &g)?)?;
    assert_eq!(memory::suggest(&c, Target::Desktop), Format::Bc4Unorm);
    let h: Vec<u8> = [2.5f32; 8].iter().flat_map(|v| v.to_le_bytes()).collect();
    let c = memory::analyze(&Surface::new(Format::Rgba32Float, 2, 1, &h)?)?;
    assert_eq!(memory::suggest(&c, Target::Desktop), Format::Bc6hUfloat);
    assert_eq!(
        memory::suggest(&c, Target::Mobile),
        Format::Astc(AstcBlock::B6x6, AstcKind::Float)
    );
    Ok(())
}
