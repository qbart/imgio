//! Channel packing: exact results for same-size inputs, presets, remaps, sizes, split.
#![allow(clippy::needless_range_loop)]

use imgio::pack::{self, Channel, ChannelSpec, PackOptions, PackSize, Source};
use imgio::{Error, Format, Surface};
use std::error::Error as StdError;

type R<T = ()> = Result<T, Box<dyn StdError>>;

fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| f(x, y))
        .collect()
}

#[test]
fn orm_is_exact() -> R {
    let (w, h) = (17, 9);
    let ao = gray(w, h, |x, y| (x * 7 + y) as u8);
    let rough = gray(w, h, |x, y| (x * 3 + y * 11) as u8);
    let metal = gray(w, h, |x, _| if x % 2 == 0 { 255 } else { 0 });
    let inputs = [
        Surface::new(Format::R8Unorm, w, h, &ao)?,
        Surface::new(Format::R8Unorm, w, h, &rough)?,
        Surface::new(Format::R8Unorm, w, h, &metal)?,
    ];
    let img = pack::pack(&inputs, &PackOptions::orm())?;
    assert_eq!(
        (img.format, img.width, img.height),
        (Format::Rgba8Unorm, w, h)
    );
    for i in 0..(w * h) as usize {
        assert_eq!(
            &img.data[i * 4..i * 4 + 4],
            &[ao[i], rough[i], metal[i], 255]
        );
    }
    Ok(())
}

#[test]
fn mask_map_inverts_roughness() -> R {
    let (w, h) = (8, 8);
    let v = gray(w, h, |x, y| (x * 30 + y) as u8);
    let s = Surface::new(Format::R8Unorm, w, h, &v)?;
    let img = pack::pack(&[s, s, s, s], &PackOptions::mask_map())?;
    for i in 0..(w * h) as usize {
        assert_eq!(&img.data[i * 4..i * 4 + 4], &[v[i], v[i], v[i], 255 - v[i]]);
    }
    Ok(())
}

#[test]
fn any_channel_from_any_input() -> R {
    // Color from an RGBA image, alpha from the luma of a gray mask, constants, scale / bias.
    let (w, h) = (4, 2);
    let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i * 9) as u8).collect();
    let mask = gray(w, h, |x, y| (x * 60 + y * 10) as u8);
    let color = Surface::new(Format::Rgba8Unorm, w, h, &rgba)?;
    let mask3: Vec<u8> = mask.iter().flat_map(|&v| [v, v, v]).collect();
    let m = Surface::new(Format::Rgb8Unorm, w, h, &mask3)?;
    let mut o = PackOptions::new(Format::Rgba8Unorm)
        .with(0, ChannelSpec::input(0, Channel::B))
        .with(1, ChannelSpec::constant(0.5))
        .with(2, ChannelSpec::input(0, Channel::R).inverted())
        .with(3, ChannelSpec::input(1, Channel::Luma));
    let img = pack::pack(&[color, m], &o)?;
    for i in 0..(w * h) as usize {
        let p = &rgba[i * 4..i * 4 + 4];
        assert_eq!(
            &img.data[i * 4..i * 4 + 4],
            &[p[2], 128, 255 - p[0], mask[i]]
        );
    }
    // Scale / bias with clamping, then without clamping into a float format.
    o.channels = [
        Some(ChannelSpec {
            source: Source::Input {
                input: 0,
                channel: Channel::Max,
            },
            invert: false,
            scale: 2.0,
            bias: -0.25,
        }),
        Some(ChannelSpec::input(0, Channel::Average)),
        None,
        None,
    ];
    let img = pack::pack(&[color], &o)?;
    for i in 0..(w * h) as usize {
        let p = &rgba[i * 4..i * 4 + 4];
        let mx = f32::from(p[0].max(p[1]).max(p[2])) / 255.0;
        let r = ((mx * 2.0 - 0.25).clamp(0.0, 1.0) * 255.0).round() as u8;
        let avg = (f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2])) / 3.0 / 255.0;
        assert_eq!(img.data[i * 4], r);
        assert!(img.data[i * 4 + 1].abs_diff((avg * 255.0).round() as u8) <= 1);
        assert_eq!(&img.data[i * 4 + 2..i * 4 + 4], &[0, 255]);
    }
    o.format = Format::Rgba32Float;
    o.clamp = false;
    let img = pack::pack(&[color], &o)?;
    let r0 = f32::from_le_bytes([img.data[0], img.data[1], img.data[2], img.data[3]]);
    let mx = f32::from(rgba[0].max(rgba[1]).max(rgba[2])) / 255.0;
    assert_eq!(r0, -0.25 + 2.0 * mx);
    Ok(())
}

#[test]
fn sizes() -> R {
    let a = gray(8, 4, |x, _| x as u8 * 30);
    let b = gray(4, 2, |_, _| 77);
    let sa = Surface::new(Format::R8Unorm, 8, 4, &a)?;
    let sb = Surface::new(Format::R8Unorm, 4, 2, &b)?;
    let mut o = PackOptions::new(Format::Rg8Unorm)
        .with(0, ChannelSpec::input(0, Channel::R))
        .with(1, ChannelSpec::input(1, Channel::R));
    assert_eq!(
        pack::pack(&[sa, sb], &o).err(),
        Some(Error::InvalidArgument)
    );
    o.size = PackSize::Largest;
    let img = pack::pack(&[sa, sb], &o)?;
    assert_eq!((img.width, img.height), (8, 4));
    // The larger input is untouched, a constant input stays constant after resizing.
    for i in 0..32 {
        assert_eq!(&img.data[i * 2..i * 2 + 2], &[a[i], 77]);
    }
    o.size = PackSize::First;
    assert_eq!(pack::pack(&[sb, sa], &o)?.width, 4);
    o.size = PackSize::Fixed(3, 5);
    let img = pack::pack(&[sa, sb], &o)?;
    assert_eq!((img.width, img.height), (3, 5));
    o.size = PackSize::Fixed(0, 5);
    assert_eq!(
        pack::pack(&[sa, sb], &o).err(),
        Some(Error::InvalidArgument)
    );
    // Unused inputs of other sizes do not matter; no inputs at all gives a 1x1 constant.
    let o = PackOptions::new(Format::R8Unorm).with(0, ChannelSpec::input(1, Channel::R));
    assert_eq!(pack::pack(&[sa, sb], &o)?.width, 4);
    let o = PackOptions::new(Format::R8Unorm).with(0, ChannelSpec::constant(1.0));
    assert_eq!(&pack::pack(&[], &o)?.data[..], &[255]);
    Ok(())
}

#[test]
fn errors() -> R {
    let a = [0u8; 16];
    let s = Surface::new(Format::R8Unorm, 4, 4, &a)?;
    let o = PackOptions::new(Format::R8Unorm).with(0, ChannelSpec::input(1, Channel::R));
    assert_eq!(pack::pack(&[s], &o).err(), Some(Error::InvalidArgument));
    let o = PackOptions::new(Format::R8Unorm).with(0, ChannelSpec::input(64, Channel::R));
    let many = vec![s; 70];
    assert_eq!(pack::pack(&many, &o).err(), Some(Error::InvalidArgument));
    let o = PackOptions::new(Format::Bc1Unorm);
    assert_eq!(pack::pack(&[s], &o).err(), Some(Error::Unsupported));
    let bc = Surface::new(Format::Bc4Unorm, 4, 4, &a[..8])?;
    let o = PackOptions::new(Format::R8Unorm).with(0, ChannelSpec::input(0, Channel::R));
    assert_eq!(pack::pack(&[bc], &o).err(), Some(Error::Unsupported));
    Ok(())
}

#[test]
fn split_round_trips() -> R {
    let (w, h) = (5, 3);
    let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i * 13 + 1) as u8).collect();
    let s = Surface::new(Format::Rgba8Unorm, w, h, &rgba)?;
    let parts = pack::split(&s, Format::R8Unorm)?;
    for (c, p) in parts.iter().enumerate() {
        assert_eq!(p.format, Format::R8Unorm);
        for i in 0..(w * h) as usize {
            assert_eq!(p.data[i], rgba[i * 4 + c]);
        }
    }
    let surfaces: Vec<Surface> = parts
        .iter()
        .map(|p| p.surface())
        .collect::<Result<_, _>>()?;
    let mut o = PackOptions::new(Format::Rgba8Unorm);
    for c in 0..4 {
        o = o.with(c, ChannelSpec::input(c, Channel::R));
    }
    assert_eq!(&pack::pack(&surfaces, &o)?.data[..], &rgba[..]);
    // 16-bit and float split keep the precision.
    let p16 = pack::split(&s, Format::R16Unorm)?;
    assert_eq!(p16[1].data.len(), (w * h * 2) as usize);
    assert_eq!(
        u16::from_le_bytes([p16[1].data[0], p16[1].data[1]]),
        u16::from(rgba[1]) * 257
    );
    Ok(())
}
