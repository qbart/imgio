//! Mip generation vs KTX-Software `ktx create --generate-mipmap` (testdata/ktx2/mips, made by
//! tools/ktx2-golden/gen.py from testdata/images): levels 1.. regenerated from the stored base
//! level must be byte-identical.

use imgio::ktx2::Ktx2File;
use imgio::mip::{self, Filter, MipOptions, Wrap};
use imgio::{Format, Surface};
use std::error::Error;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/ktx2/mips/");

#[test]
fn matches_ktx_create() -> Result<(), Box<dyn Error>> {
    let cases = std::fs::read_to_string(format!("{DIR}mips.txt"))?;
    let mut n = 0;
    for line in cases.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let opts = MipOptions {
            filter: Filter::from_name(f[1]).ok_or("filter")?,
            filter_scale: f[2].parse()?,
            wrap: match f[3] {
                "wrap" => Wrap::Wrap,
                "reflect" => Wrap::Reflect,
                _ => Wrap::Clamp,
            },
            srgb: None,
        };
        let data = std::fs::read(format!("{DIR}{}.ktx2", f[0]))?;
        let file = Ktx2File::parse(&data)?;
        let format = file.format().ok_or("plain format")?;
        let base = Surface::new(
            format,
            file.width(),
            file.height(),
            file.image_data(0, 0, 0)?,
        )?;
        let levels = mip::generate_mips(&base, file.level_count(), &opts)?;
        assert_eq!(levels.len() as u32 + 1, file.level_count());
        for (i, got) in levels.iter().enumerate() {
            let want = file.image_data(i as u32 + 1, 0, 0)?;
            assert!(&got[..] == want, "{} level {}", f[0], i + 1);
            n += 1;
        }
    }
    assert!(n > 40, "{n}");
    Ok(())
}

#[test]
fn formats_and_errors() -> Result<(), Box<dyn Error>> {
    // A flat image stays flat in every layout.
    for (format, px) in [
        (Format::Rgba8Unorm, vec![10u8, 20, 30, 40]),
        (Format::Rgba16Unorm, vec![1, 2, 3, 4, 5, 6, 7, 8]),
        (
            Format::Rgba32Float,
            [0.25f32, 2.0, -1.0, 1.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        (
            Format::Rgba16Float,
            vec![0, 0x3c, 0, 0x40, 0, 0xbc, 0, 0x3c],
        ),
    ] {
        let src: Vec<u8> = px.iter().copied().cycle().take(px.len() * 64).collect();
        let s = Surface::new(format, 8, 8, &src)?;
        let out = mip::resize(&s, 3, 5, &MipOptions::default())?;
        let want = &src[..px.len() * 15];
        if format == Format::Rgba32Float {
            // Filter weights sum to 1 within float rounding.
            let f = |b: &[u8]| -> Vec<f32> {
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect()
            };
            for (g, w) in f(&out).iter().zip(f(want)) {
                assert!((g - w).abs() <= 1e-6 * w.abs().max(1.0), "{g} vs {w}");
            }
        } else {
            assert_eq!(&out[..], want, "{format:?}");
        }
    }
    let s = Surface::new(Format::Bc7Unorm, 4, 4, &[0u8; 16])?;
    assert!(mip::generate_mips(&s, 2, &MipOptions::default()).is_err());
    let s = Surface::new(Format::R8Unorm, 4, 4, &[0u8; 16])?;
    assert!(mip::generate_mips(&s, 4, &MipOptions::default()).is_err());
    assert_eq!(mip::generate_mips(&s, 3, &MipOptions::default())?.len(), 2);
    Ok(())
}
