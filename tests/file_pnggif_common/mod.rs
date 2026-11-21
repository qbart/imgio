//! Helpers shared by the PNG and GIF file tests: the expected-decode manifests written by
//! `scripts/gen_file_testdata_pnggif.py`, hashing, synthetic images, corrupt-input loops.
#![allow(dead_code)]

use imgio::file::{Animation, Frame, Image};
use imgio::{Format, Surface};
use std::collections::BTreeMap;
use std::error::Error;

pub type TestResult<T = ()> = Result<T, Box<dyn Error>>;

pub const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/");

pub fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| {
        (h ^ u64::from(x)).wrapping_mul(0x100000001b3)
    })
}

/// One expected decoded image.
#[derive(Debug, Clone)]
pub struct Expected {
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub delay: (u32, u32),
    pub hash: u64,
}

/// Expected decodes of one file: `main` image, composited `frames`, (`loops`, frame count).
#[derive(Debug, Default)]
pub struct Case {
    pub main: Option<Expected>,
    pub frames: Vec<Expected>,
    pub anim: Option<(u32, usize)>,
}

/// Parses `testdata/file/<fmt>/expected.txt`.
pub fn manifest(fmt: &str) -> TestResult<BTreeMap<String, Case>> {
    let text = std::fs::read_to_string(format!("{DIR}file/{fmt}/expected.txt"))?;
    let mut out: BTreeMap<String, Case> = BTreeMap::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        let case = out.entry(f[0].to_string()).or_default();
        if f[1] == "*" {
            case.anim = Some((f[3].parse()?, f[5].parse()?));
            continue;
        }
        let e = Expected {
            format: f[2].to_string(),
            width: f[3].parse()?,
            height: f[4].parse()?,
            delay: (f[5].parse()?, f[6].parse()?),
            hash: u64::from_str_radix(f[7], 16)?,
        };
        if f[1] == "main" {
            case.main = Some(e);
        } else {
            assert_eq!(f[1].parse::<usize>()?, case.frames.len(), "{line}");
            case.frames.push(e);
        }
    }
    Ok(out)
}

/// Asserts that `img` matches `e` (format label, size, pixels).
pub fn check_image(img: &Image, e: &Expected, what: &str) {
    assert_eq!(format!("{:?}", img.format), e.format, "{what}: format");
    assert_eq!((img.width, img.height), (e.width, e.height), "{what}: size");
    assert_eq!(fnv(&img.data), e.hash, "{what}: pixels");
}

/// Checks a decoded animation against the manifest (frames, delays, loops).
pub fn check_animation(anim: &Animation, case: &Case, name: &str) {
    let (loops, count) = case.anim.unwrap_or((0, 0));
    assert_eq!(anim.frames.len(), count, "{name}: frame count");
    assert_eq!(anim.loops, loops, "{name}: loops");
    for (i, (f, e)) in anim.frames.iter().zip(&case.frames).enumerate() {
        check_image(&f.image, e, &format!("{name} frame {i}"));
        assert_eq!(f.delay, e.delay, "{name} frame {i}: delay");
    }
}

/// Deterministic pseudo-random bytes (LCG).
pub fn pattern(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
        .collect()
}

/// A raw RGBA8 test image of `testdata/images/raw` (`name_WxH.rgba`).
pub fn raw_rgba(name: &str) -> TestResult<(Vec<u8>, u32, u32)> {
    let data = std::fs::read(format!("{DIR}images/raw/{name}.rgba"))?;
    let dims = name.rsplit('_').next().ok_or("name")?;
    let (w, h) = dims.split_once('x').ok_or("dims")?;
    Ok((data, w.parse()?, h.parse()?))
}

pub fn image(format: Format, width: u32, height: u32, data: &[u8]) -> TestResult<Image> {
    Ok(Image::from_surface(&Surface::new(
        format, width, height, data,
    )?)?)
}

pub fn animation(frames: Vec<(Image, (u32, u32))>, loops: u32) -> TestResult<Animation> {
    let mut v = vas::Vector::<Frame, vas::storage::Dynamic>::default();
    for (image, delay) in frames {
        v.push(Frame { image, delay }).map_err(|_| "push")?;
    }
    Ok(Animation { frames: v, loops })
}

/// Calls `f` on truncations of `data` at many lengths and on copies with flipped bytes
/// (`fix` repairs checksums after a flip so the parser gets past them). `f` must not panic.
pub fn corrupt_inputs(data: &[u8], fix: impl Fn(&mut [u8]), mut f: impl FnMut(&[u8])) {
    let step = (data.len() / 300).max(1);
    for len in (0..data.len()).step_by(step) {
        f(&data[..len]);
    }
    let mut seed = 0x1234_5678u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed >> 8
    };
    for i in (0..data.len()).step_by(step) {
        let mut d = data.to_vec();
        d[i] ^= 1 << (next() % 8);
        fix(&mut d);
        f(&d);
        let mut d = data.to_vec();
        d[i] = next() as u8;
        let j = next() as usize % data.len();
        d[j] = d[j].wrapping_add(next() as u8 | 1);
        fix(&mut d);
        f(&d);
    }
}
