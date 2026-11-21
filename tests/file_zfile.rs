//! RenderMan zfile: no reference tool exists, so the files are built byte by byte here from
//! the layout (magic, u16 width / height, world-to-screen and world-to-camera matrices, f32
//! depths; byte order from the magic), plus encoder round trips and corrupt input.

mod file_own_common;

use file_own_common::*;
use imgio::file::{FileFormat, zfile};
use imgio::{Format, Surface};

/// A zfile from the spec: big or little endian.
fn build(be: bool, w: u16, h: u16, np: &[f32; 16], nl: &[f32; 16], z: &[f32]) -> Vec<u8> {
    let mut d = Vec::new();
    let u32b = |v: u32| if be { v.to_be_bytes() } else { v.to_le_bytes() };
    let u16b = |v: u16| if be { v.to_be_bytes() } else { v.to_le_bytes() };
    let f32b = |v: f32| if be { v.to_be_bytes() } else { v.to_le_bytes() };
    d.extend(u32b(0x2f08_67ab));
    d.extend(u16b(w));
    d.extend(u16b(h));
    for v in np.iter().chain(nl).chain(z) {
        d.extend(f32b(*v));
    }
    d
}

fn mats() -> ([f32; 16], [f32; 16]) {
    let np: [f32; 16] = std::array::from_fn(|i| i as f32 * 0.5 - 3.0);
    let nl: [f32; 16] = std::array::from_fn(|i| if i % 5 == 0 { 1.0 } else { i as f32 / 7.0 });
    (np, nl)
}

fn depths(w: usize, h: usize) -> Vec<f32> {
    (0..w * h)
        .map(|i| match i % 7 {
            0 => f32::INFINITY, // background
            1 => 1e30,
            _ => 0.1 + i as f32 * 0.37,
        })
        .collect()
}

#[test]
fn decodes_hand_built_files() -> R {
    let (np, nl) = mats();
    let (w, h) = (13usize, 7usize);
    let z = depths(w, h);
    for be in [false, true] {
        let d = build(be, w as u16, h as u16, &np, &nl, &z);
        assert_eq!(
            &d[..4],
            if be {
                &[0x2f, 0x08, 0x67, 0xab]
            } else {
                &[0xab, 0x67, 0x08, 0x2f]
            }
        );
        assert_eq!(FileFormat::detect(&d), Some(FileFormat::Zfile));
        let f = zfile::decode_zfile(&d)?;
        assert_eq!(f.big_endian, be);
        assert_eq!((f.world_to_screen, f.world_to_camera), (np, nl));
        assert_eq!(
            (f.image.format, f.image.width, f.image.height),
            (Format::R32Float, 13, 7)
        );
        let want: Vec<u8> = z.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(&f.image.data[..], &want[..]);
        assert_eq!(&imgio::file::decode(&d)?.data[..], &want[..]);
        // Trailing bytes are ignored, a missing depth is an error.
        let mut long = d.clone();
        long.push(0);
        assert!(zfile::decode(&long).is_ok());
        assert!(zfile::decode(&d[..d.len() - 1]).is_err());
    }
    Ok(())
}

#[test]
fn encode_matches_spec_bytes() -> R {
    let (np, nl) = mats();
    let z = depths(5, 3);
    let px: Vec<u8> = z.iter().flat_map(|v| v.to_le_bytes()).collect();
    let s = Surface::new(Format::R32Float, 5, 3, &px)?;
    for be in [false, true] {
        let opts = zfile::EncodeOptions {
            world_to_screen: np,
            world_to_camera: nl,
            big_endian: be,
        };
        let out = zfile::encode(&s, &opts)?;
        assert_eq!(&out[..], &build(be, 5, 3, &np, &nl, &z)[..]);
    }
    // Defaults: identity matrices, little endian.
    let out = imgio::file::encode(FileFormat::Zfile, &s)?;
    let f = zfile::decode_zfile(&out)?;
    let id: [f32; 16] = std::array::from_fn(|i| if i % 5 == 0 { 1.0 } else { 0.0 });
    assert_eq!(
        (f.world_to_screen, f.world_to_camera, f.big_endian),
        (id, id, false)
    );
    // Half floats widen exactly; multi-channel input is refused.
    let half: Vec<u8> = [1.5f32, -2.0, 65504.0, 0.0]
        .iter()
        .flat_map(|v| imgio::half::from_f32(*v).to_le_bytes())
        .collect();
    let out = zfile::encode(
        &Surface::new(Format::R16Float, 2, 2, &half)?,
        &zfile::EncodeOptions::default(),
    )?;
    let back = zfile::decode(&out)?;
    let v: Vec<f32> = back
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(v, [1.5, -2.0, 65504.0, 0.0]);
    let rgba = vec![0u8; 16];
    assert!(
        zfile::encode(
            &Surface::new(Format::Rgba8Unorm, 2, 2, &rgba)?,
            &zfile::EncodeOptions::default()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn corrupt_input_never_panics() {
    let (np, nl) = mats();
    for be in [false, true] {
        let d = build(be, 9, 4, &np, &nl, &depths(9, 4));
        fuzz(&d, zfile::decode);
    }
    // Bad magic, zero size.
    let mut d = build(false, 1, 1, &np, &nl, &[1.0]);
    d[0] ^= 1;
    assert!(zfile::decode(&d).is_err());
    assert!(zfile::decode(&build(false, 0, 1, &np, &nl, &[])).is_err());
}
