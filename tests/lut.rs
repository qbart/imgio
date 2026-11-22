//! LUTs against colour-science 0.4.7 (testdata/lut, made by scripts/gen_lut_testdata.py).

use imgio::Format;
use imgio::Surface;
use imgio::lut::{self, Cube, Interpolation, Lut3d};
use std::error::Error as StdError;
use std::path::PathBuf;

type R<T = ()> = Result<T, Box<dyn StdError>>;

const W: u32 = 64;
const H: u32 = 32;

fn read(name: &str) -> R<Vec<u8>> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/lut")
        .join(name);
    Ok(std::fs::read(&p).map_err(|e| format!("{name}: {e}"))?)
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
fn parses_colour_science_files() -> R {
    let c = Cube::parse(&read("grade.cube")?)?;
    assert_eq!(c.title.as_deref(), Some("Grade test"));
    assert!(c.shaper.is_none());
    let l = c.lut.as_ref().ok_or("no 3D")?;
    assert_eq!((l.size, l.table.len()), (17, 17 * 17 * 17));
    assert_eq!((l.domain_min, l.domain_max), ([0.0; 3], [1.0; 3]));
    let c = Cube::parse(&read("domain.cube")?)?;
    let l = c.lut.as_ref().ok_or("no 3D")?;
    assert_eq!(l.domain_min, [-0.1, -0.2, 0.0]);
    assert_eq!(l.domain_max, [1.5, 2.0, 4.0]);
    let c = Cube::parse(&read("curve.cube")?)?;
    let s = c.shaper.as_ref().ok_or("no 1D")?;
    assert!(c.lut.is_none());
    assert_eq!((s.size, s.domain_max), (1024, [2.0; 3]));
    let c = Cube::parse(&read("shaper.cube")?)?;
    let (s, l) = (
        c.shaper.as_ref().ok_or("no 1D")?,
        c.lut.as_ref().ok_or("no 3D")?,
    );
    assert_eq!(
        (s.size, s.domain_min, s.domain_max),
        (256, [-0.25; 3], [4.0; 3])
    );
    assert_eq!(
        (l.size, l.domain_min, l.domain_max),
        (21, [0.0; 3], [1.0; 3])
    );
    assert_eq!(s.table[0], [-0.305; 3]);
    Ok(())
}

#[test]
fn apply_matches_colour_science() -> R {
    let input = read("input.f32")?;
    let src = Surface::new(Format::Rgba32Float, W, H, &input)?;
    let inp = floats(&input);
    let cases = [
        ("grade", "_tri", Interpolation::Trilinear),
        ("grade", "_tet", Interpolation::Tetrahedral),
        ("domain", "_tri", Interpolation::Trilinear),
        ("domain", "_tet", Interpolation::Tetrahedral),
        ("shaper", "_tri", Interpolation::Trilinear),
        ("shaper", "_tet", Interpolation::Tetrahedral),
        ("curve", "", Interpolation::Tetrahedral),
    ];
    for (name, tag, interp) in cases {
        let cube = Cube::parse(&read(&format!("{name}.cube"))?)?;
        let want = floats(&read(&format!("{name}{tag}.f32"))?);
        let img = lut::apply(&src, &cube, interp, Format::Rgba32Float)?;
        let got = floats(&img.data);
        let mut worst = 0.0f32;
        for p in 0..(W * H) as usize {
            for c in 0..3 {
                let (g, w) = (got[p * 4 + c], want[p * 3 + c]);
                worst = worst.max((g - w).abs() / w.abs().max(1.0));
            }
            // Alpha passes through.
            assert_eq!(got[p * 4 + 3], inp[p * 4 + 3]);
            // Single lookups agree with the image path.
            if p < 16 {
                let rgb = [inp[p * 4], inp[p * 4 + 1], inp[p * 4 + 2]];
                assert_eq!(cube.eval(rgb, interp)[..], got[p * 4..p * 4 + 3]);
            }
        }
        // Table entries are f32 here, f64 in colour-science.
        assert!(worst < 1e-6, "{name}{tag}: {worst}");
    }
    Ok(())
}

#[test]
fn write_round_trips() -> R {
    for name in ["grade", "domain", "curve", "shaper"] {
        let a = Cube::parse(&read(&format!("{name}.cube"))?)?;
        let text = a.to_bytes()?;
        let b = Cube::parse(&text)?;
        assert_eq!(a.title.as_deref(), b.title.as_deref());
        {
            match (&a.shaper, &b.shaper) {
                (Some(x), Some(y)) => {
                    assert_eq!(
                        (x.size, x.domain_min, x.domain_max),
                        (y.size, y.domain_min, y.domain_max)
                    );
                    assert_eq!(&x.table[..], &y.table[..], "{name}");
                }
                (None, None) => {}
                _ => return Err(format!("{name}: shaper").into()),
            }
        }
        match (&a.lut, &b.lut) {
            (Some(x), Some(y)) => {
                assert_eq!(
                    (x.size, x.domain_min, x.domain_max),
                    (y.size, y.domain_min, y.domain_max)
                );
                assert_eq!(&x.table[..], &y.table[..], "{name}");
            }
            (None, None) => {}
            _ => return Err(format!("{name}: 3D").into()),
        }
    }
    let text = Cube::new(Lut3d::identity(2)?).to_bytes()?;
    assert_eq!(
        std::str::from_utf8(&text)?,
        "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n"
    );
    Ok(())
}

#[test]
fn malformed_files_are_rejected() {
    let bad: [&[u8]; 9] = [
        b"",
        b"LUT_3D_SIZE 2\n0 0 0\n",
        b"LUT_3D_SIZE 1\n0 0 0\n",
        b"LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 x\n",
        b"LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\nTITLE \"x\"\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
        b"LUT_1D_SIZE 2\n0 0 0 0\n1 1 1\n",
        b"LUT_1D_SIZE 2\nDOMAIN_MIN 1 1 1\n0 0 0\n1 1 1\n",
        b"LUT_1D_SIZE 2\n0 0 nan\n1 1 1\n",
        b"\xff\xfe",
    ];
    for b in bad {
        assert!(Cube::parse(b).is_err(), "{:?}", String::from_utf8_lossy(b));
    }
    // Comments, CRLF, blank lines and unknown keywords are fine.
    let ok = b"# c\r\nTITLE \"t\"\r\nLUT_1D_SIZE 2\r\nLUT_IN_VIDEO_RANGE\r\n\r\n0 0 0\r\n1 1 1\r\n";
    assert!(Cube::parse(ok).is_ok());
}

#[test]
fn identity_and_images() -> R {
    let id = Lut3d::identity(5)?;
    for v in [[0.3f32, 0.71, 0.05], [0.0, 1.0, 0.5], [0.999, 0.25, 0.25]] {
        for i in [Interpolation::Trilinear, Interpolation::Tetrahedral] {
            let o = id.eval(v, i);
            assert!((0..3).all(|c| (o[c] - v[c]).abs() < 1e-6), "{v:?} {o:?}");
        }
    }
    // HALD level 4: 64 x 64, entry k at pixel k, red fastest.
    let h = id.to_hald(Format::Rgba32Float);
    assert!(h.is_err(), "5 is not a square");
    let id = Lut3d::identity(16)?;
    let h = id.to_hald(Format::Rgba8Unorm)?;
    assert_eq!((h.width, h.height), (64, 64));
    let k = 3 + 16 * (7 + 16 * 11);
    assert_eq!(&h.data[k * 4..k * 4 + 4], &[51, 119, 187, 255]);
    let g = Lut3d::from_fn(16, |v| [v[1], v[2] * 0.5, v[0]])?;
    let back = Lut3d::from_hald(&g.to_hald(Format::Rgba32Float)?.surface()?)?;
    assert_eq!(back.size, 16);
    assert_eq!(&back.table[..], &g.table[..]);
    assert_eq!(back.table[k][1], id.table[k][2] * 0.5);
    // Strip: pixel (r + n b, g).
    let s = id.to_strip(Format::Rgba8Unorm)?;
    assert_eq!((s.width, s.height), (256, 16));
    let p = 7 * 256 + 11 * 16 + 3;
    assert_eq!(&s.data[p * 4..p * 4 + 4], &[51, 119, 187, 255]);
    let back = Lut3d::from_strip(&g.to_strip(Format::Rgba32Float)?.surface()?)?;
    assert_eq!(&back.table[..], &g.table[..]);
    Ok(())
}

#[test]
fn bake_filters() -> R {
    use imgio::filter::{self, Adjust};
    let a = Adjust {
        contrast: 1.2,
        saturation: 0.7,
        gamma: 1.1,
        ..Adjust::default()
    };
    let l = Lut3d::bake(9, [0.0; 3], [1.0; 3], |s| filter::adjust(s, &a))?;
    // Lattice points hold the filter's output exactly.
    let px = [0.25f32, 0.5, 0.875, 1.0];
    let b: Vec<u8> = px.iter().flat_map(|v| v.to_le_bytes()).collect();
    let one = filter::adjust(&Surface::new(Format::Rgba32Float, 1, 1, &b)?, &a)?;
    let want = floats(&one.data);
    assert_eq!(l.table[2 + 9 * (4 + 9 * 7)][..], want[..3]);
    // A domain: lattice points spread over it.
    let l = Lut3d::bake(3, [-1.0, 0.0, 0.0], [1.0, 4.0, 2.0], |s| {
        imgio::file::Image::from_surface(s)
    })?;
    assert_eq!(l.table[2 + 3 * (1 + 3 * 2)], [1.0, 2.0, 2.0]);
    assert_eq!(
        l.eval([-0.5, 3.0, 1.5], Interpolation::Tetrahedral),
        [-0.5, 3.0, 1.5]
    );
    Ok(())
}

#[cfg(all(feature = "color", feature = "png"))]
#[test]
fn bake_profiles() -> R {
    use imgio::color::{self, Intent, Profile};
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/color");
    let p3 = Profile::from_icc(&std::fs::read(dir.join("p3.icc"))?)?;
    let srgb = Profile::from_icc(&std::fs::read(dir.join("lcms_srgb.icc"))?)?;
    let l = Lut3d::from_profiles(33, &p3, &srgb, Intent::Perceptual)?;
    // Lattice entries are the colour transform of the lattice points.
    let v = [8.0f32 / 32.0, 20.0 / 32.0, 1.0, 1.0];
    let b: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
    let one = color::convert(
        &Surface::new(Format::Rgba32Float, 1, 1, &b)?,
        &p3,
        &srgb,
        Intent::Perceptual,
        Format::Rgba32Float,
    )?;
    assert_eq!(l.table[8 + 33 * (20 + 33 * 32)][..], floats(&one.data)[..3]);
    // Applied to an 8-bit image: close to the exact transform (LittleCMS reference; the
    // lattice interpolates across the gamut clipping kinks).
    let img = imgio::file::decode(&std::fs::read(dir.join("files/p3.png"))?)?;
    let want = std::fs::read(dir.join("files/png_p3_to_srgb.raw"))?;
    let got = lut::apply(
        &img.surface()?,
        &Cube::new(l),
        Interpolation::Tetrahedral,
        Format::Rgb8Unorm,
    )?;
    let d: Vec<u8> = got
        .data
        .iter()
        .zip(&want)
        .map(|(a, b)| a.abs_diff(*b))
        .collect();
    let worst = d.iter().copied().max().ok_or("empty")?;
    let mean = d.iter().map(|&v| f64::from(v)).sum::<f64>() / d.len() as f64;
    assert!(worst <= 4 && mean < 0.25, "{worst} {mean}");
    Ok(())
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325, |h, &x| {
        (h ^ u64::from(x)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The first 1024 input pixels clamped to 0..1, as `channels` f32 (gray: red).
#[cfg(feature = "color")]
fn link_input(channels: usize) -> R<(Vec<u8>, Format)> {
    let inp = floats(&read("input.f32")?);
    let mut b = Vec::new();
    for p in 0..1024 {
        for c in 0..channels {
            b.extend_from_slice(&inp[p * 4 + c].clamp(0.0, 1.0).to_le_bytes());
        }
        if channels == 3 {
            b.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }
    let f = if channels == 1 {
        Format::R32Float
    } else {
        Format::Rgba32Float
    };
    Ok((b, f))
}

/// Largest difference between `got` (RGBA or R f32) and `want` (`n` per pixel).
#[cfg(feature = "color")]
fn worst(got: &[u8], want: &[f32], n: usize) -> f32 {
    let g = floats(got);
    let step = g.len() / 1024;
    (0..1024)
        .flat_map(|p| (0..n).map(move |c| (p, c)))
        .map(|(p, c)| (g[p * step + c] - want[p * n + c]).abs())
        .fold(0.0, f32::max)
}

/// Values of `got` differing from `want` by more than `tol`.
#[cfg(feature = "color")]
fn count_over(got: &[u8], want: &[f32], n: usize, tol: f32) -> usize {
    let g = floats(got);
    let step = g.len() / 1024;
    (0..1024)
        .flat_map(|p| (0..n).map(move |c| (p, c)))
        .filter(|&(p, c)| (g[p * step + c] - want[p * n + c]).abs() > tol)
        .count()
}

#[cfg(feature = "color")]
#[test]
fn device_links_match_littlecms() -> R {
    use imgio::color::{self, Intent, Profile};
    // Links made by LittleCMS linkicc (mAB tables), applied like transicc.
    for (name, ch) in [("p3_srgb", 3), ("gray", 1)] {
        let link = Profile::from_icc(&read(&format!("{name}.link.icc"))?)?;
        assert!(link.is_device_link());
        let (b, f) = link_input(ch)?;
        let src = Surface::new(f, 1024, 1, &b)?;
        let got = color::apply_link(&src, &link, f)?;
        let want = floats(&read(&format!("{name}_link.f32"))?);
        let e = worst(&got.data, &want, ch);
        // LittleCMS evaluates in f32: a matrix result on a 16-bit rounding edge of a sampled
        // curve can land one word away.
        let off = count_over(&got.data, &want, ch, 1e-6);
        assert!(
            e < 1.6e-5 && off <= 4,
            "{name}: {e}, {off} values over 1e-6"
        );
        // Not a source / target profile.
        assert!(color::convert(&src, &link, &Profile::srgb()?, Intent::Perceptual, f).is_err());
    }
    let (b, f) = link_input(3)?;
    let src = Surface::new(f, 1024, 1, &b)?;
    assert!(color::apply_link(&src, &Profile::srgb()?, f).is_err());
    // imgio's device link of grade.cube: LittleCMS applies it as the table says.
    let cube = Cube::parse(&read("grade.cube")?)?;
    let lut3 = cube.lut.as_ref().ok_or("no 3D")?;
    let icc = lut3.to_device_link("Grade test")?;
    if std::env::var("IMGIO_WRITE_ENCODED").is_ok() {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/lut/enc");
        std::fs::create_dir_all(&d)?;
        std::fs::write(d.join("grade_link.icc"), &icc[..])?;
        return Ok(());
    }
    let text = String::from_utf8(read("encoded_verified.txt")?)?;
    let want_hash = text
        .lines()
        .find_map(|l| l.strip_prefix("grade_link.icc "))
        .ok_or("re-run scripts/gen_lut_testdata.py --verify")?;
    assert_eq!(
        format!("{:016x}", fnv(&icc)),
        want_hash,
        "device link output changed, re-run scripts/gen_lut_testdata.py --verify"
    );
    let link = Profile::from_icc(&icc)?;
    assert!(link.is_device_link());
    let got = color::apply_link(&src, &link, f)?;
    let want = floats(&read("grade_link_lcms.f32")?);
    let e = worst(&got.data, &want, 3);
    let off = count_over(&got.data, &want, 3, 1e-6);
    assert!(
        e < 1.6e-5 && off <= 4,
        "grade link: {e}, {off} values over 1e-6"
    );
    // Close to applying the table in f64, clamped and rounded to 16 bits like the link (the
    // rest is LittleCMS's 16-bit interpolation).
    let mut q = Lut3d::identity(lut3.size)?;
    for (d, s) in q.table.iter_mut().zip(lut3.table.iter()) {
        *d = s.map(|v| ((f64::from(v).clamp(0.0, 1.0) * 65535.0).round() / 65535.0) as f32);
    }
    let direct = lut::apply(&src, &Cube::new(q), Interpolation::Tetrahedral, f)?;
    let e = worst(&direct.data, &want, 3);
    assert!(e < 3.1e-5, "{e}");
    // Only 0..1 domains.
    let dom = Cube::parse(&read("domain.cube")?)?;
    assert!(dom.lut.ok_or("no 3D")?.to_device_link("x").is_err());
    Ok(())
}
