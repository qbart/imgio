//! KTX2 plain `VkFormat` files vs KTX-Software: testdata/ktx2/plain/*.ktx2 were written by
//! `ktx create --raw` (tools/ktx2-golden/gen.py) from deterministic images. The reader must
//! return every image and the writer must reproduce every file byte for byte.

use imgio::ktx2::{self, Ktx2Desc, Ktx2File, Supercompression, WriteOptions};
use std::error::Error;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/ktx2/plain/");
const WRITER: &[u8] = b"ktx create v4.4.2-2 / libktx v4.4.2-2\0";

struct Case {
    name: String,
    desc: Ktx2Desc,
    zstd: i32,
}

fn cases() -> Result<Vec<Case>, Box<dyn Error>> {
    let text = std::fs::read_to_string(format!("{DIR}cases.txt"))?;
    let mut out = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let n = |i: usize| f[i].parse::<u32>();
        out.push(Case {
            name: f[0].to_string(),
            desc: Ktx2Desc {
                format: ktx2::format_from_vk(n(1)?).ok_or("unknown vkFormat")?,
                width: n(2)?,
                height: n(3)?,
                layers: n(4)?,
                faces: n(5)?,
                levels: n(6)?,
                depth: n(8)?,
            },
            zstd: f[7].parse()?,
        });
    }
    Ok(out)
}

/// The generator's levels: image k has bytes (i * 7 + k * 13 + 3) & 255.
fn levels(d: &Ktx2Desc) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let mut k = 0usize;
    let mut out = Vec::new();
    for level in 0..d.levels {
        let (w, h) = imgio::image::mip_dim(d.width, d.height, level);
        let size = d.format.surface_bytes(w, h)?;
        let mut data = Vec::new();
        let slices = d.depth.checked_shr(level).unwrap_or(0).max(1);
        for _ in 0..d.layers.max(1) * d.faces * slices {
            data.extend((0..size).map(|i| ((i * 7 + k * 13 + 3) & 255) as u8));
            k += 1;
        }
        out.push(data);
    }
    Ok(out)
}

#[test]
fn reader_returns_every_image() -> Result<(), Box<dyn Error>> {
    let cases = cases()?;
    assert_eq!(cases.len(), 97);
    for c in &cases {
        let data = std::fs::read(format!("{DIR}{}.ktx2", c.name))?;
        let file = Ktx2File::parse(&data)?;
        let d = &c.desc;
        assert_eq!(file.format(), Some(d.format), "{}", c.name);
        // Plain ASTC LDR (and ASTC 6x6 HDR) blocks are also Basis payloads (transcodable).
        let basis = match d.format {
            imgio::Format::Astc(b, imgio::AstcKind::Unorm | imgio::AstcKind::Srgb) => {
                Some(imgio::basis::BasisTexFormat::AstcLdr(b))
            }
            imgio::Format::Astc(imgio::AstcBlock::B6x6, imgio::AstcKind::Float) => {
                Some(imgio::basis::BasisTexFormat::AstcHdr6x6)
            }
            _ => None,
        };
        assert_eq!(file.tex_format(), basis, "{}", c.name);
        assert_eq!((file.width(), file.height()), (d.width, d.height));
        assert_eq!(file.level_count(), d.levels);
        assert_eq!(file.layer_count(), d.layers.max(1));
        assert_eq!(file.face_count(), d.faces);
        assert_eq!(file.depth(), d.depth.max(1));
        let lv = levels(d)?;
        for (level, l) in lv.iter().enumerate() {
            let per = l.len() / (d.layers.max(1) * d.faces) as usize; // all depth slices
            for layer in 0..d.layers.max(1) {
                for face in 0..d.faces {
                    let i = (layer * d.faces + face) as usize;
                    let got = file.image_data(level as u32, layer, face)?;
                    assert_eq!(got, &l[i * per..(i + 1) * per], "{} {level}", c.name);
                }
            }
        }
        if basis.is_none() {
            assert!(!file.can_transcode_to(imgio::Format::Rgba8Unorm));
        }
    }
    Ok(())
}

#[test]
fn writer_matches_ktx_create() -> Result<(), Box<dyn Error>> {
    for c in cases()? {
        let lv = levels(&c.desc)?;
        let refs: Vec<&[u8]> = lv.iter().map(|l| &l[..]).collect();
        let sc = format!("--zstd {}\0", c.zstd);
        let mut kv: Vec<(&str, &[u8])> = vec![("KTXwriter", WRITER)];
        let supercompression = if c.zstd > 0 {
            kv.push(("KTXwriterScParams", sc.as_bytes()));
            Supercompression::Zstd(c.zstd)
        } else {
            Supercompression::None
        };
        let opts = WriteOptions {
            supercompression,
            key_values: &kv,
        };
        let got = ktx2::write(&c.desc, &refs, &opts)?;
        let want = std::fs::read(format!("{DIR}{}.ktx2", c.name))?;
        assert!(
            got[..] == want[..],
            "{}: {} vs {} bytes",
            c.name,
            got.len(),
            want.len()
        );
    }
    Ok(())
}

#[test]
fn writer_rejects_bad_input() {
    let d = Ktx2Desc {
        format: imgio::Format::Bc7Unorm,
        width: 8,
        height: 8,
        layers: 0,
        faces: 1,
        levels: 1,
        depth: 0,
    };
    let opts = WriteOptions::default();
    assert!(ktx2::write(&d, &[&[0u8; 63]], &opts).is_err());
    assert!(ktx2::write(&d, &[&[0u8; 64], &[0u8; 16]], &opts).is_err());
    assert!(
        ktx2::write(
            &Ktx2Desc {
                faces: 6,
                height: 4,
                ..d
            },
            &[&[0u8; 64]],
            &opts
        )
        .is_err()
    );
    let atc = Ktx2Desc {
        format: imgio::Format::AtcRgb,
        ..d
    };
    assert!(ktx2::write(&atc, &[&[0u8; 32]], &opts).is_err());
    let dup = WriteOptions {
        key_values: &[("a", b"1"), ("a", b"2")],
        ..opts
    };
    assert!(ktx2::write(&d, &[&[0u8; 64]], &dup).is_err());
    // Round trip without key/values.
    let file = ktx2::write(&d, &[&[7u8; 64]], &opts);
    let file = file.as_deref().unwrap_or(&[]);
    let parsed = Ktx2File::parse(file);
    assert!(parsed.is_ok_and(|p| p.image_data(0, 0, 0).is_ok_and(|b| b == [7u8; 64])));
}

#[test]
fn rgb8_and_bgra8_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    use imgio::Format;
    let px: Vec<u8> = (0..4 * 4 * 4).map(|i| (i * 7) as u8).collect();
    for f in [
        Format::Rgb8Unorm,
        Format::Rgb8Srgb,
        Format::Bgra8Unorm,
        Format::Bgra8Srgb,
    ] {
        let rgba = imgio::Surface::new(Format::Rgba8Unorm, 4, 4, &px)?;
        let data = imgio::convert::convert_to_buffer(&rgba, f)?;
        let desc = Ktx2Desc {
            format: f,
            width: 4,
            height: 4,
            layers: 0,
            faces: 1,
            levels: 1,
            depth: 0,
        };
        let file = imgio::ktx2::write(&desc, &[&data[..]], &WriteOptions::default())?;
        let k = Ktx2File::parse(&file)?;
        assert_eq!(k.format(), Some(f));
        assert_eq!(k.image_data(0, 0, 0)?, &data[..]);
        let back = imgio::convert::convert_to_buffer(
            &imgio::Surface::new(f, 4, 4, &data)?,
            Format::Rgba8Unorm,
        )?;
        for (a, b) in back.as_chunks::<4>().0.iter().zip(px.as_chunks::<4>().0) {
            assert_eq!(a[..3], b[..3]);
            assert_eq!(a[3], if f.block_bytes() == 3 { 255 } else { b[3] });
        }
        // DDS for BGRA (DXGI) and legacy RGB8 masks.
        let dd = imgio::dds::DdsDesc {
            format: f,
            width: 4,
            height: 4,
            layers: 0,
            faces: 1,
            levels: 1,
            depth: 0,
        };
        let legacy = imgio::dds::DdsWriteOptions { legacy: true };
        if let Ok(out) = imgio::dds::write(&dd, &[&data[..]], &legacy) {
            let d = imgio::dds::DdsFile::parse(&out)?;
            assert_eq!(d.image_data(0, 0, 0)?, &data[..], "{f:?}");
        }
    }
    Ok(())
}
