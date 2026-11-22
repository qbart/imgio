//! Image based lighting: cube / equirect orientation (OpenGL face table), SH and convolved
//! irradiance against analytic results, specular prefiltering invariants and the split sum
//! table against brute force hemisphere integration (numpy, values below).

use imgio::file::Image;
use imgio::ibl::{self, Irradiance, SpecularOptions};
use imgio::{Error, Format, Surface};
use std::error::Error as StdError;

type R<T = ()> = Result<T, Box<dyn StdError>>;

fn px(img: &Image, x: u32, y: u32) -> [f32; 4] {
    let o = ((y * img.width + x) * 16) as usize;
    core::array::from_fn(|i| {
        let k = o + i * 4;
        f32::from_le_bytes([
            img.data[k],
            img.data[k + 1],
            img.data[k + 2],
            img.data[k + 3],
        ])
    })
}

fn rgba32f(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 4]) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 16) as usize);
    for y in 0..h {
        for x in 0..w {
            for c in f(x, y) {
                v.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    v
}

/// The direction of equirect pixel center (x, y) (independent of imgio's code).
fn eq_dir(x: u32, y: u32, w: u32, h: u32) -> [f32; 3] {
    let u = (x as f64 + 0.5) / w as f64;
    let v = (y as f64 + 0.5) / h as f64;
    let phi = (u - 0.5) * 2.0 * std::f64::consts::PI;
    let th = v * std::f64::consts::PI;
    [
        (th.sin() * phi.sin()) as f32,
        th.cos() as f32,
        (-th.sin() * phi.cos()) as f32,
    ]
}

/// OpenGL spec table: face texel center (x, y) of an n x n face.
fn gl_dir(f: usize, x: u32, y: u32, n: u32) -> [f32; 3] {
    let sc = 2.0 * (x as f32 + 0.5) / n as f32 - 1.0;
    let tc = 2.0 * (y as f32 + 0.5) / n as f32 - 1.0;
    let d = match f {
        0 => [1.0, -tc, -sc],
        1 => [-1.0, -tc, sc],
        2 => [sc, 1.0, tc],
        3 => [sc, -1.0, -tc],
        4 => [sc, -tc, 1.0],
        _ => [-sc, -tc, -1.0],
    };
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / l, d[1] / l, d[2] / l]
}

/// Cube faces of `n` x `n` RGBA32F with radiance `f(dir)`.
fn cube(n: u32, f: impl Fn([f32; 3]) -> [f32; 4]) -> Vec<Vec<u8>> {
    (0..6)
        .map(|face| rgba32f(n, n, |x, y| f(gl_dir(face, x, y, n))))
        .collect()
}

fn surfaces(c: &[Vec<u8>], n: u32) -> R<Vec<Surface<'_>>> {
    Ok(c.iter()
        .map(|d| Surface::new(Format::Rgba32Float, n, n, d))
        .collect::<Result<_, _>>()?)
}

#[test]
fn equirect_and_cube_orientation() -> R {
    // A panorama whose pixels are their own direction.
    let (w, h) = (512, 256);
    let eq = rgba32f(w, h, |x, y| {
        let d = eq_dir(x, y, w, h);
        [d[0], d[1], d[2], 1.0]
    });
    let s = Surface::new(Format::Rgba32Float, w, h, &eq)?;
    let n = 32;
    let faces = ibl::equirect_to_cube(&s, n, Format::Rgba32Float)?;
    let mut worst = 0.0f32;
    for (f, img) in faces.iter().enumerate() {
        assert_eq!((img.width, img.height), (n, n));
        for y in 0..n {
            for x in 0..n {
                let got = px(img, x, y);
                let want = gl_dir(f, x, y, n);
                // Texels at the poles average across the seam / pole: skip |y| > 0.98.
                if want[1].abs() > 0.98 || (want[2] > 0.0 && want[0].abs() < 0.02) {
                    continue;
                }
                for k in 0..3 {
                    worst = worst.max((got[k] - want[k]).abs());
                }
            }
        }
    }
    assert!(worst < 0.02, "equirect -> cube direction error {worst}");
    // Known orientation facts: +X face top-left looks to +Y +Z; +Z face center is +Z.
    let tl = px(&faces[0], 0, 0);
    assert!(tl[0] > 0.5 && tl[1] > 0.5 && tl[2] > 0.5, "{tl:?}");
    let c = px(&faces[4], n / 2, n / 2);
    assert!(c[2] > 0.99);
    // Back: cube of directions -> panorama.
    let dirs = cube(64, |d| [d[0], d[1], d[2], 1.0]);
    let ds = surfaces(&dirs, 64)?;
    let pano = ibl::cube_to_equirect(&ds, 128, 64, Format::Rgba32Float)?;
    let mut worst = 0.0f32;
    for y in 2..62 {
        for x in 0..128 {
            let got = px(&pano, x, y);
            let want = eq_dir(x, y, 128, 64);
            for k in 0..3 {
                worst = worst.max((got[k] - want[k]).abs());
            }
        }
    }
    assert!(worst < 0.03, "cube -> equirect direction error {worst}");
    // The face round trip direction <-> face is consistent.
    for f in 0..6 {
        for (x, y) in [(0, 0), (5, 9), (31, 31), (16, 3)] {
            let d = ibl::face_dir(
                f,
                2.0 * (x as f32 + 0.5) / 32.0 - 1.0,
                2.0 * (y as f32 + 0.5) / 32.0 - 1.0,
            );
            let (g, s, t) = ibl::dir_face(d);
            assert_eq!(g, f);
            assert!(
                (s * 32.0 - 0.5 - x as f32).abs() < 1e-3
                    && (t * 32.0 - 0.5 - y as f32).abs() < 1e-3
            );
        }
    }
    Ok(())
}

#[test]
fn irradiance_matches_analytic() -> R {
    // L = 1: E / pi = 1. L = 1 + z: E / pi = 1 + 2/3 n_z (band 1 is exact in SH).
    let n = 32;
    let one = cube(n, |_| [1.0, 2.0, 0.5, 1.0]);
    let sh = ibl::sh_project(&surfaces(&one, n)?)?;
    for d in [[0.0, 0.0, 1.0], [0.6, 0.0, -0.8], [0.0, -1.0, 0.0]] {
        let e = sh.irradiance(d);
        assert!(
            (e[0] - 1.0).abs() < 1e-4 && (e[1] - 2.0).abs() < 2e-4 && (e[2] - 0.5).abs() < 1e-4,
            "{e:?}"
        );
    }
    let lin = cube(n, |d| [1.0 + d[2], 1.0 - d[0], 1.0, 1.0]);
    let ls = surfaces(&lin, n)?;
    let sh = ibl::sh_project(&ls)?;
    let conv = ibl::irradiance(
        &ls,
        8,
        Irradiance::Convolve { source_size: 32 },
        Format::Rgba32Float,
    )?;
    let shc = ibl::irradiance(&ls, 8, Irradiance::Sh, Format::Rgba32Float)?;
    for f in 0..6 {
        for y in 0..8 {
            for x in 0..8 {
                let d = gl_dir(f, x, y, 8);
                let want = [1.0 + 2.0 / 3.0 * d[2], 1.0 - 2.0 / 3.0 * d[0], 1.0];
                let e = sh.irradiance(d);
                let a = px(&conv[f], x, y);
                let b = px(&shc[f], x, y);
                for k in 0..3 {
                    assert!(
                        (e[k] - want[k]).abs() < 1e-3,
                        "sh {f} {x} {y} {e:?} {want:?}"
                    );
                    assert!(
                        (a[k] - want[k]).abs() < 2e-3,
                        "conv {f} {x} {y} {a:?} {want:?}"
                    );
                    assert!((b[k] - want[k]).abs() < 1e-3);
                }
            }
        }
    }
    // A cosine lobe max(0, z): E/pi at +Z = 2/3 (convolution; SH is an approximation).
    let lobe = cube(32, |d| [d[2].max(0.0); 4]);
    let conv = ibl::irradiance(
        &surfaces(&lobe, 32)?,
        5,
        Irradiance::Convolve { source_size: 32 },
        Format::Rgba32Float,
    )?;
    // Texel (2, 2) of a 5x5 +Z face is the pole.
    let c = px(&conv[4], 2, 2);
    assert!((c[0] - 2.0 / 3.0).abs() < 2e-3, "{c:?}");
    Ok(())
}

#[test]
fn specular_prefilter() -> R {
    let n = 32;
    let one = cube(n, |_| [0.25, 1.0, 4.0, 1.0]);
    let opts = SpecularOptions {
        samples: 64,
        format: Format::Rgba32Float,
        ..Default::default()
    };
    let levels = ibl::prefilter_specular(&surfaces(&one, n)?, &opts)?;
    assert_eq!(levels.len(), 4); // 32, 16, 8, 4
    for (l, faces) in levels.iter().enumerate() {
        for img in faces.iter() {
            assert_eq!(img.width, n >> l);
            for y in 0..img.height {
                for x in 0..img.width {
                    let p = px(img, x, y);
                    assert!(
                        (p[0] - 0.25).abs() < 1e-5
                            && (p[1] - 1.0).abs() < 1e-5
                            && (p[2] - 4.0).abs() < 2e-5,
                        "{l} {p:?}"
                    );
                }
            }
        }
    }
    // Level 0 = the source; blur grows with roughness (L = 1 + z seen from +Z).
    let lin = cube(n, |d| [1.0 + d[2], 0.0, 0.0, 1.0]);
    let ls = surfaces(&lin, n)?;
    let opts = SpecularOptions {
        samples: 256,
        levels: 5,
        format: Format::Rgba32Float,
        ..Default::default()
    };
    let levels = ibl::prefilter_specular(&ls, &opts)?;
    assert_eq!(levels.len(), 5);
    for f in 0..6 {
        assert_eq!(&levels[0][f].data[..], &lin[f][..]);
    }
    let mut prev = 2.0f32;
    for faces in levels.iter() {
        let s = faces[4].width;
        let v = (px(&faces[4], s / 2 - 1, s / 2 - 1)[0] + px(&faces[4], s / 2, s / 2)[0]) / 2.0;
        assert!(v <= prev + 1e-4 && v > 1.0, "{v} after {prev}");
        prev = v;
    }
    assert!(prev < 1.9);
    // Deterministic.
    let again = ibl::prefilter_specular(&ls, &opts)?;
    assert_eq!(&again[3][2].data[..], &levels[3][2].data[..]);
    Ok(())
}

#[test]
fn brdf_lut_against_brute_force() -> R {
    // (N.V, roughness, scale, bias) by brute force integration of GGX D, Smith G (k = a/2) and
    // Schlick Fresnel over the hemisphere (numpy, 4000 x 8000 grid).
    let refs = [
        (0.9, 0.3, 0.977063, 0.000098),
        (0.5, 0.5, 0.728535, 0.018546),
        (0.2, 0.7, 0.612285, 0.030541),
        (0.7, 1.0, 0.360192, 0.000811),
        (0.1, 0.4, 0.516558, 0.149980),
    ];
    for (nv, r, a, b) in refs {
        let (ga, gb) = ibl::brdf_lut_value(nv, r, 4096);
        assert!(
            (ga - a).abs() < 0.01 && (gb - b).abs() < 0.01,
            "{nv} {r}: {ga} {gb} vs {a} {b}"
        );
    }
    let lut = ibl::brdf_lut(32, 256, Format::Rgba32Float)?;
    assert_eq!((lut.width, lut.height), (32, 32));
    let smooth = px(&lut, 31, 0);
    assert!(smooth[0] > 0.97 && smooth[1] < 0.01, "{smooth:?}");
    for y in 0..32 {
        for x in 0..32 {
            let p = px(&lut, x, y);
            assert!(
                p[0] >= 0.0 && p[1] >= 0.0 && p[0] + p[1] <= 1.0001,
                "{x} {y} {p:?}"
            );
        }
    }
    let u16lut = ibl::brdf_lut(16, 64, Format::Rg16Unorm)?;
    assert_eq!(u16lut.data.len(), 16 * 16 * 4);
    Ok(())
}

#[test]
fn errors() -> R {
    let d = vec![0u8; 4 * 4 * 16];
    let s = Surface::new(Format::Rgba32Float, 4, 4, &d)?;
    let r = Surface::new(Format::Rgba32Float, 4, 2, &d[..4 * 2 * 16])?;
    assert_eq!(
        ibl::equirect_to_cube(&s, 0, Format::Rgba16Float).err(),
        Some(Error::InvalidArgument)
    );
    assert_eq!(
        ibl::sh_project(&[s, s, s]).err(),
        Some(Error::InvalidArgument)
    );
    assert_eq!(
        ibl::sh_project(&[s, s, s, s, s, r]).err(),
        Some(Error::InvalidArgument)
    );
    assert_eq!(
        ibl::cube_to_equirect(&[s; 6], 0, 4, Format::Rgba16Float).err(),
        Some(Error::InvalidArgument)
    );
    let o = SpecularOptions {
        samples: 0,
        ..Default::default()
    };
    assert_eq!(
        ibl::prefilter_specular(&[s; 6], &o).err(),
        Some(Error::InvalidArgument)
    );
    assert_eq!(
        ibl::brdf_lut(0, 16, Format::Rg16Unorm).err(),
        Some(Error::InvalidArgument)
    );
    let bc = Surface::new(Format::Bc6hUfloat, 4, 4, &d[..16])?;
    assert_eq!(
        ibl::equirect_to_cube(&bc, 4, Format::Rgba16Float).err(),
        Some(Error::Unsupported)
    );
    Ok(())
}
