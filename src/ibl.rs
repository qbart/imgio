//! Image based lighting: environment map conversions and prefiltering.
//!
//! - [`equirect_to_cube`] / [`cube_to_equirect`]: latitude-longitude panoramas and cube maps
//!   (bilinear, supersampled when shrinking).
//! - [`sh_project`] / [`Sh9::irradiance`]: 3rd order spherical harmonics of the radiance
//!   (texel solid angle weighted) and the irradiance they give (Ramamoorthi and Hanrahan 2001).
//! - [`irradiance`]: diffuse irradiance cube from the SH or by brute force convolution.
//! - [`prefilter_specular`]: GGX prefiltered specular mip chain (split sum, Karis 2013;
//!   filtered importance sampling, Křivánek and Colbert 2007), level `i` of `n` at roughness
//!   `i / (n - 1)`.
//! - [`brdf_lut`]: the split sum scale / bias table (Karis 2013).
//!
//! Conventions: cube faces are +X, -X, +Y, -Y, +Z, -Z with the OpenGL / Vulkan / KTX
//! orientation (face texel `(s, t)`, `t` down the rows, maps to the major axis table of the
//! OpenGL specification, section 8.13). Equirect pixel `(u, v)` in 0..1 (rows top to bottom)
//! is the direction `(sin v pi sin phi, cos v pi, -sin v pi cos phi)` with `phi = (u - 0.5)
//! 2 pi`: the image center looks down -Z, the top row is +Y. Irradiance outputs are `E / pi`
//! (the outgoing radiance of a white Lambertian surface). Roughness is perceptual (GGX alpha =
//! roughness^2). Work is done on RGBA f32 copies (alpha carried through by the conversions,
//! ignored by the filters), outputs are written in any uncompressed format, rows in parallel.

use crate::file::Image;
use crate::{Error, Format, Result, Surface};
use core::f32::consts::PI;
use vas::Vector;
use vas::storage::Dynamic;

/// Six cube faces (+X, -X, +Y, -Y, +Z, -Z).
pub type Faces = [Image; 6];

type V3 = [f32; 3];

fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        a
    }
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Direction through face `f` at `(sc, tc)` in -1..1 (OpenGL major axis table inverted).
pub fn face_dir(f: usize, sc: f32, tc: f32) -> [f32; 3] {
    norm(match f {
        0 => [1.0, -tc, -sc],
        1 => [-1.0, -tc, sc],
        2 => [sc, 1.0, tc],
        3 => [sc, -1.0, -tc],
        4 => [sc, -tc, 1.0],
        _ => [-sc, -tc, -1.0],
    })
}

/// Face and `(s, t)` in 0..1 of a direction.
pub fn dir_face(d: [f32; 3]) -> (usize, f32, f32) {
    let [x, y, z] = d;
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    let (f, sc, tc, ma) = if ax >= ay && ax >= az {
        if x >= 0.0 {
            (0, -z, -y, ax)
        } else {
            (1, z, -y, ax)
        }
    } else if ay >= az {
        if y >= 0.0 {
            (2, x, z, ay)
        } else {
            (3, x, -z, ay)
        }
    } else if z >= 0.0 {
        (4, x, -y, az)
    } else {
        (5, -x, -y, az)
    };
    let ma = if ma > 0.0 { ma } else { 1.0 };
    (f, (sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5)
}

/// Equirect `(u, v)` (0..1) of a direction.
fn dir_uv(d: V3) -> (f32, f32) {
    let u = 0.5 + d[0].atan2(-d[2]) / (2.0 * PI);
    let v = d[1].clamp(-1.0, 1.0).acos() / PI;
    (u, v)
}

fn uv_dir(u: f32, v: f32) -> V3 {
    let phi = (u - 0.5) * 2.0 * PI;
    let th = v * PI;
    [th.sin() * phi.sin(), th.cos(), -th.sin() * phi.cos()]
}

/// An RGBA f32 image.
struct Plane {
    w: usize,
    h: usize,
    px: Vector<[f32; 4], Dynamic>,
}

impl Plane {
    fn new(w: usize, h: usize) -> Result<Plane> {
        let mut px = Vector::default();
        px.resize(w * h, [0.0; 4]).map_err(|_| Error::OutOfMemory)?;
        Ok(Plane { w, h, px })
    }

    fn from_surface(s: &Surface) -> Result<Plane> {
        if s.format().is_compressed() {
            return Err(Error::Unsupported);
        }
        let b = crate::convert::convert_to_buffer(s, Format::Rgba32Float)?;
        let (w, h) = (s.width() as usize, s.height() as usize);
        let mut p = Plane::new(w, h)?;
        for (o, c) in p.px.iter_mut().zip(b.as_chunks::<16>().0) {
            *o = core::array::from_fn(|i| {
                f32::from_le_bytes([c[i * 4], c[i * 4 + 1], c[i * 4 + 2], c[i * 4 + 3]])
            });
        }
        Ok(p)
    }

    fn get(&self, x: usize, y: usize) -> [f32; 4] {
        self.px[y * self.w + x]
    }

    /// Bilinear at pixel coordinates (centers at +0.5), clamped; `wrap_x` wraps columns.
    fn bilinear(&self, fx: f32, fy: f32, wrap_x: bool) -> [f32; 4] {
        let x = fx - 0.5;
        let y = (fy - 0.5).clamp(0.0, (self.h - 1) as f32);
        let (w, h) = (self.w as i64, self.h as i64);
        let x0 = x.floor();
        let y0 = y.floor();
        let (tx, ty) = (x - x0, y - y0);
        let col = |c: i64| -> usize {
            if wrap_x {
                c.rem_euclid(w) as usize
            } else {
                c.clamp(0, w - 1) as usize
            }
        };
        let row = |r: i64| r.clamp(0, h - 1) as usize;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let a = self.get(col(x0), row(y0));
        let b = self.get(col(x0 + 1), row(y0));
        let c = self.get(col(x0), row(y0 + 1));
        let d = self.get(col(x0 + 1), row(y0 + 1));
        core::array::from_fn(|i| {
            let top = a[i] + (b[i] - a[i]) * tx;
            let bot = c[i] + (d[i] - c[i]) * tx;
            top + (bot - top) * ty
        })
    }

    fn to_image(&self, format: Format) -> Result<Image> {
        let mut b = crate::buffer::zeroed(self.w * self.h * 16)?;
        for (o, p) in b.as_chunks_mut::<16>().0.iter_mut().zip(self.px.iter()) {
            for (i, v) in p.iter().enumerate() {
                o[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let s = Surface::new(Format::Rgba32Float, self.w as u32, self.h as u32, &b)?;
        let mut img = Image::new(format, self.w as u32, self.h as u32)?;
        crate::convert::convert(&s, format, &mut img.data)?;
        Ok(img)
    }

    /// 2x2 box average (odd sizes: the last row / column is averaged with itself).
    fn half(&self) -> Result<Plane> {
        let (w, h) = ((self.w / 2).max(1), (self.h / 2).max(1));
        let mut p = Plane::new(w, h)?;
        for y in 0..h {
            for x in 0..w {
                let xs = [(2 * x).min(self.w - 1), (2 * x + 1).min(self.w - 1)];
                let ys = [(2 * y).min(self.h - 1), (2 * y + 1).min(self.h - 1)];
                let mut s = [0.0f32; 4];
                for yy in ys {
                    for xx in xs {
                        let v = self.get(xx, yy);
                        for i in 0..4 {
                            s[i] += v[i];
                        }
                    }
                }
                p.px[y * w + x] = s.map(|v| v * 0.25);
            }
        }
        Ok(p)
    }
}

/// Fills a `w` x `h` plane with `f(x, y)`, rows in parallel.
fn render(w: usize, h: usize, f: impl Fn(usize, usize) -> [f32; 4] + Sync) -> Result<Plane> {
    let mut b = crate::buffer::zeroed(w * h * 16)?;
    crate::parallel::for_each_unit(&mut b, w * 16, w * h * 64, |y, row| {
        for (x, o) in row.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            let p = f(x, y);
            for (i, v) in p.iter().enumerate() {
                o[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    });
    let mut p = Plane::new(w, h)?;
    for (o, c) in p.px.iter_mut().zip(b.as_chunks::<16>().0) {
        *o = core::array::from_fn(|i| {
            f32::from_le_bytes([c[i * 4], c[i * 4 + 1], c[i * 4 + 2], c[i * 4 + 3]])
        });
    }
    Ok(p)
}

/// A cube map with a mip chain per face.
struct Cube {
    /// levels[l][face]
    levels: Vector<[Plane; 6], Dynamic>,
}

fn six<T>(mut f: impl FnMut(usize) -> Result<T>) -> Result<[T; 6]> {
    Ok([f(0)?, f(1)?, f(2)?, f(3)?, f(4)?, f(5)?])
}

impl Cube {
    fn from_faces(faces: &[Surface]) -> Result<Cube> {
        if faces.len() != 6 {
            return Err(Error::InvalidArgument);
        }
        let n = faces[0].width();
        if n == 0 || faces.iter().any(|f| f.width() != n || f.height() != n) {
            return Err(Error::InvalidArgument);
        }
        let base = six(|i| Plane::from_surface(&faces[i]))?;
        let mut levels = Vector::default();
        levels.push(base).map_err(|_| Error::OutOfMemory)?;
        Ok(Cube { levels })
    }

    fn size(&self) -> usize {
        self.levels.first().map_or(1, |l| l[0].w)
    }

    /// Builds the box-filtered mip chain down to 1x1.
    fn mips(&mut self) -> Result<()> {
        while let Some(last) = self.levels.last() {
            if last[0].w <= 1 {
                break;
            }
            let next = six(|i| last[i].half())?;
            self.levels.push(next).map_err(|_| Error::OutOfMemory)?;
        }
        Ok(())
    }

    fn sample_level(&self, l: usize, d: V3) -> [f32; 4] {
        let (f, s, t) = dir_face(d);
        let p = &self.levels[l.min(self.levels.len() - 1)][f];
        p.bilinear(s * p.w as f32, t * p.h as f32, false)
    }

    /// Trilinear sample at fractional `lod`.
    fn sample(&self, d: V3, lod: f32) -> [f32; 4] {
        let max = (self.levels.len() - 1) as f32;
        let lod = lod.clamp(0.0, max);
        let l0 = lod.floor() as usize;
        let t = lod - l0 as f32;
        let a = self.sample_level(l0, d);
        if t == 0.0 {
            return a;
        }
        let b = self.sample_level(l0 + 1, d);
        core::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
    }
}

/// Supersampling factor when `src` texels map to `dst` texels (1..=8).
fn supersample(src: usize, dst: usize) -> usize {
    src.div_ceil(dst.max(1)).clamp(1, 8)
}

/// Cube faces (`size` x `size`) from an equirectangular panorama (any aspect; 2:1 usual).
pub fn equirect_to_cube(src: &Surface, size: u32, format: Format) -> Result<Faces> {
    if size == 0 {
        return Err(Error::InvalidArgument);
    }
    let p = Plane::from_surface(src)?;
    let n = size as usize;
    let ss = supersample(p.w, 4 * n);
    six(|f| {
        render(n, n, |x, y| {
            let mut acc = [0.0f32; 4];
            for j in 0..ss {
                for i in 0..ss {
                    let sc = 2.0 * (x as f32 + (i as f32 + 0.5) / ss as f32) / n as f32 - 1.0;
                    let tc = 2.0 * (y as f32 + (j as f32 + 0.5) / ss as f32) / n as f32 - 1.0;
                    let (u, v) = dir_uv(face_dir(f, sc, tc));
                    let c = p.bilinear(u * p.w as f32, v * p.h as f32, true);
                    for k in 0..4 {
                        acc[k] += c[k];
                    }
                }
            }
            acc.map(|v| v / (ss * ss) as f32)
        })?
        .to_image(format)
    })
}

/// An equirectangular panorama (`width` x `height`) from six cube faces.
pub fn cube_to_equirect(
    faces: &[Surface],
    width: u32,
    height: u32,
    format: Format,
) -> Result<Image> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidArgument);
    }
    let cube = Cube::from_faces(faces)?;
    let (w, h) = (width as usize, height as usize);
    let ss = supersample(4 * cube.size(), w);
    render(w, h, |x, y| {
        let mut acc = [0.0f32; 4];
        for j in 0..ss {
            for i in 0..ss {
                let u = (x as f32 + (i as f32 + 0.5) / ss as f32) / w as f32;
                let v = (y as f32 + (j as f32 + 0.5) / ss as f32) / h as f32;
                let c = cube.sample_level(0, uv_dir(u, v));
                for k in 0..4 {
                    acc[k] += c[k];
                }
            }
        }
        acc.map(|v| v / (ss * ss) as f32)
    })?
    .to_image(format)
}

/// Solid angle of the face texel `(x, y)` of an `n` x `n` face.
fn texel_solid_angle(x: usize, y: usize, n: usize) -> f64 {
    let inv = 1.0 / n as f64;
    let x0 = 2.0 * x as f64 * inv - 1.0;
    let y0 = 2.0 * y as f64 * inv - 1.0;
    let (x1, y1) = (x0 + 2.0 * inv, y0 + 2.0 * inv);
    let a = |x: f64, y: f64| (x * y).atan2((x * x + y * y + 1.0).sqrt());
    a(x0, y0) - a(x0, y1) - a(x1, y0) + a(x1, y1)
}

/// SH basis (real, orthonormal) up to band 2 for a unit direction.
fn sh_basis(d: [f64; 3]) -> [f64; 9] {
    let [x, y, z] = d;
    [
        0.282_094_791_773_878_1,
        0.488_602_511_902_919_9 * y,
        0.488_602_511_902_919_9 * z,
        0.488_602_511_902_919_9 * x,
        1.092_548_430_592_079_2 * x * y,
        1.092_548_430_592_079_2 * y * z,
        0.315_391_565_252_520_05 * (3.0 * z * z - 1.0),
        1.092_548_430_592_079_2 * x * z,
        0.546_274_215_296_039_6 * (x * x - y * y),
    ]
}

/// Radiance projected on SH bands 0..2 (RGB per coefficient; order `1, y, z, x, xy, yz,
/// 3z^2 - 1, xz, x^2 - y^2`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sh9 {
    pub coeffs: [[f32; 3]; 9],
}

impl Sh9 {
    /// `E(n) / pi` for unit normal `n` (convolution with the clamped cosine: bands scaled by
    /// pi, 2 pi / 3, pi / 4).
    pub fn irradiance(&self, n: [f32; 3]) -> [f32; 3] {
        let b = sh_basis([f64::from(n[0]), f64::from(n[1]), f64::from(n[2])]);
        let band = [
            1.0,
            2.0 / 3.0,
            2.0 / 3.0,
            2.0 / 3.0,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
        ];
        let mut e = [0.0f64; 3];
        for i in 0..9 {
            for (c, ev) in e.iter_mut().enumerate() {
                *ev += f64::from(self.coeffs[i][c]) * b[i] * band[i];
            }
        }
        e.map(|v| v as f32)
    }
}

/// Projects the radiance of six cube faces on SH bands 0..2 (texel solid angles, f64 sums).
pub fn sh_project(faces: &[Surface]) -> Result<Sh9> {
    let cube = Cube::from_faces(faces)?;
    let n = cube.size();
    let mut acc = [[0.0f64; 3]; 9];
    for f in 0..6 {
        let p = &cube.levels[0][f];
        for y in 0..n {
            for x in 0..n {
                let sc = 2.0 * (x as f32 + 0.5) / n as f32 - 1.0;
                let tc = 2.0 * (y as f32 + 0.5) / n as f32 - 1.0;
                let d = face_dir(f, sc, tc);
                let b = sh_basis([f64::from(d[0]), f64::from(d[1]), f64::from(d[2])]);
                let w = texel_solid_angle(x, y, n);
                let c = p.get(x, y);
                for i in 0..9 {
                    for k in 0..3 {
                        acc[i][k] += f64::from(c[k]) * b[i] * w;
                    }
                }
            }
        }
    }
    Ok(Sh9 {
        coeffs: acc.map(|c| c.map(|v| v as f32)),
    })
}

/// Diffuse irradiance method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Irradiance {
    /// From [`sh_project`] (smooth, fast; ringing-free for most environments).
    Sh,
    /// Brute force cosine convolution over the source box-filtered to `source_size` (exact up
    /// to that resolution; cost `6 size^2 * 6 source_size^2`).
    Convolve { source_size: u32 },
}

/// Diffuse irradiance cube (`E / pi`), `size` x `size` faces.
pub fn irradiance(
    faces: &[Surface],
    size: u32,
    method: Irradiance,
    format: Format,
) -> Result<Faces> {
    if size == 0 {
        return Err(Error::InvalidArgument);
    }
    let n = size as usize;
    match method {
        Irradiance::Sh => {
            let sh = sh_project(faces)?;
            let a = |f: usize| {
                render(n, n, |x, y| {
                    let sc = 2.0 * (x as f32 + 0.5) / n as f32 - 1.0;
                    let tc = 2.0 * (y as f32 + 0.5) / n as f32 - 1.0;
                    let e = sh.irradiance(face_dir(f, sc, tc));
                    [e[0].max(0.0), e[1].max(0.0), e[2].max(0.0), 1.0]
                })?
                .to_image(format)
            };
            six(a)
        }
        Irradiance::Convolve { source_size } => {
            if source_size == 0 {
                return Err(Error::InvalidArgument);
            }
            let mut cube = Cube::from_faces(faces)?;
            cube.mips()?;
            // The level closest to (not below) source_size.
            let l = cube
                .levels
                .iter()
                .rposition(|l| l[0].w >= source_size as usize)
                .unwrap_or(0);
            let src = &cube.levels[l];
            let m = src[0].w;
            // Directions, solid angles and radiance of every source texel.
            let mut texels = Vector::<(V3, f32, [f32; 3]), Dynamic>::default();
            texels.reserve(6 * m * m).map_err(|_| Error::OutOfMemory)?;
            for (f, p) in src.iter().enumerate() {
                for y in 0..m {
                    for x in 0..m {
                        let sc = 2.0 * (x as f32 + 0.5) / m as f32 - 1.0;
                        let tc = 2.0 * (y as f32 + 0.5) / m as f32 - 1.0;
                        let c = p.get(x, y);
                        texels
                            .push((
                                face_dir(f, sc, tc),
                                texel_solid_angle(x, y, m) as f32,
                                [c[0], c[1], c[2]],
                            ))
                            .map_err(|_| Error::OutOfMemory)?;
                    }
                }
            }
            let texels = &texels[..];
            six(|f| {
                render(n, n, |x, y| {
                    let sc = 2.0 * (x as f32 + 0.5) / n as f32 - 1.0;
                    let tc = 2.0 * (y as f32 + 0.5) / n as f32 - 1.0;
                    let nd = face_dir(f, sc, tc);
                    let mut e = [0.0f64; 3];
                    for (d, w, c) in texels {
                        let cs = dot(nd, *d);
                        if cs > 0.0 {
                            let k = f64::from(cs * w);
                            for i in 0..3 {
                                e[i] += f64::from(c[i]) * k;
                            }
                        }
                    }
                    let inv = 1.0 / core::f64::consts::PI;
                    [
                        (e[0] * inv) as f32,
                        (e[1] * inv) as f32,
                        (e[2] * inv) as f32,
                        1.0,
                    ]
                })?
                .to_image(format)
            })
        }
    }
}

/// Van der Corput radical inverse (base 2).
fn radical_inverse(mut b: u32) -> f64 {
    b = b.rotate_right(16);
    b = ((b & 0x5555_5555) << 1) | ((b & 0xAAAA_AAAA) >> 1);
    b = ((b & 0x3333_3333) << 2) | ((b & 0xCCCC_CCCC) >> 2);
    b = ((b & 0x0F0F_0F0F) << 4) | ((b & 0xF0F0_F0F0) >> 4);
    b = ((b & 0x00FF_00FF) << 8) | ((b & 0xFF00_FF00) >> 8);
    f64::from(b) * 2.328_306_436_538_696_3e-10
}

/// GGX half vector around +Z for sample `(e1, e2)`, `a = alpha`.
fn ggx_half(e1: f64, e2: f64, a: f64) -> [f64; 3] {
    let phi = 2.0 * core::f64::consts::PI * e1;
    let cos_t = ((1.0 - e2) / (1.0 + (a * a - 1.0) * e2)).sqrt();
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    [sin_t * phi.cos(), sin_t * phi.sin(), cos_t]
}

/// Options of [`prefilter_specular`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecularOptions {
    /// Level 0 face size (0 = the source size).
    pub size: u32,
    /// Mip levels (0 = down to 4x4 faces, at least 1).
    pub levels: u32,
    /// GGX samples per texel (levels above 0).
    pub samples: u32,
    pub format: Format,
}

impl Default for SpecularOptions {
    fn default() -> Self {
        SpecularOptions {
            size: 0,
            levels: 0,
            samples: 512,
            format: Format::Rgba16Float,
        }
    }
}

/// GGX prefiltered specular cube: one [`Faces`] per level, level `i` at roughness
/// `i / (levels - 1)` and size `size >> i`. Level 0 is the source (copied, or resampled
/// when `size` differs). Samples: Hammersley points, GGX importance sampling with `N = V = R`, weighted
/// by `N.L`, each read from the source mip whose texel solid angle matches the sample's
/// (`0.5 log2(omega_s / omega_p)`).
pub fn prefilter_specular(
    faces: &[Surface],
    opts: &SpecularOptions,
) -> Result<Vector<Faces, Dynamic>> {
    let mut cube = Cube::from_faces(faces)?;
    let src = cube.size();
    let size = if opts.size == 0 {
        src
    } else {
        opts.size as usize
    };
    if opts.samples == 0 || size == 0 {
        return Err(Error::InvalidArgument);
    }
    let full = (usize::BITS - size.leading_zeros()) as usize; // log2(size) + 1
    let levels = if opts.levels == 0 {
        full.saturating_sub(2).max(1)
    } else {
        (opts.levels as usize).min(full)
    };
    cube.mips()?;
    let texel_sa = 4.0 * PI / (6.0 * (src * src) as f32);
    let ns = opts.samples as usize;
    let mut out = Vector::<Faces, Dynamic>::default();
    for l in 0..levels {
        let n = (size >> l).max(1);
        let rough = if levels > 1 {
            l as f32 / (levels - 1) as f32
        } else {
            0.0
        };
        let alpha = f64::from(rough * rough);
        if l == 0 && n == src {
            // Level 0 at the source size: the source itself.
            let faces = six(|f| cube.levels[0][f].to_image(opts.format))?;
            out.push(faces).map_err(|_| Error::OutOfMemory)?;
            continue;
        }
        let faces = six(|f| {
            render(n, n, |x, y| {
                let sc = 2.0 * (x as f32 + 0.5) / n as f32 - 1.0;
                let tc = 2.0 * (y as f32 + 0.5) / n as f32 - 1.0;
                let nd = face_dir(f, sc, tc);
                if l == 0 || rough == 0.0 {
                    let c = cube.sample(nd, 0.0);
                    return [c[0], c[1], c[2], 1.0];
                }
                let up = if nd[2].abs() < 0.999 {
                    [0.0, 0.0, 1.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let tx = norm(cross(up, nd));
                let ty = cross(nd, tx);
                let mut acc = [0.0f64; 3];
                let mut wsum = 0.0f64;
                for i in 0..ns {
                    let e1 = i as f64 / ns as f64;
                    let e2 = radical_inverse(i as u32);
                    let h = ggx_half(e1, e2, alpha);
                    let hw: V3 = core::array::from_fn(|k| {
                        (h[0] * f64::from(tx[k])
                            + h[1] * f64::from(ty[k])
                            + h[2] * f64::from(nd[k])) as f32
                    });
                    let vh = dot(nd, hw);
                    let ld: V3 = core::array::from_fn(|k| 2.0 * vh * hw[k] - nd[k]);
                    let nl = dot(nd, ld);
                    if nl <= 0.0 {
                        continue;
                    }
                    let nh = h[2];
                    let a2 = alpha * alpha;
                    let dd = nh * nh * (a2 - 1.0) + 1.0;
                    let d = a2 / (core::f64::consts::PI * dd * dd);
                    // pdf = D NdotH / (4 VdotH) with V = N: VdotH = NdotH.
                    let pdf = (d * 0.25).max(1e-12);
                    let sa = 1.0 / (ns as f64 * pdf);
                    let lod = (0.5 * (sa / f64::from(texel_sa)).log2()).max(0.0) as f32;
                    let c = cube.sample(norm(ld), lod);
                    let w = f64::from(nl);
                    for k in 0..3 {
                        acc[k] += f64::from(c[k]) * w;
                    }
                    wsum += w;
                }
                let inv = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
                [
                    (acc[0] * inv) as f32,
                    (acc[1] * inv) as f32,
                    (acc[2] * inv) as f32,
                    1.0,
                ]
            })?
            .to_image(opts.format)
        })?;
        out.push(faces).map_err(|_| Error::OutOfMemory)?;
    }
    Ok(out)
}

/// One texel of the split sum table: `(scale, bias)` for `n_dot_v` and perceptual
/// `roughness` with `samples` Hammersley points (Smith GGX, `k = alpha / 2`).
pub fn brdf_lut_value(n_dot_v: f64, roughness: f64, samples: u32) -> (f64, f64) {
    let v = [(1.0 - n_dot_v * n_dot_v).max(0.0).sqrt(), 0.0, n_dot_v];
    let alpha = roughness * roughness;
    let k = alpha / 2.0;
    let g1 = |x: f64| x / (x * (1.0 - k) + k);
    let (mut a, mut b) = (0.0f64, 0.0f64);
    let n = samples.max(1);
    for i in 0..n {
        let h = ggx_half(f64::from(i) / f64::from(n), radical_inverse(i), alpha);
        let vh = v[0] * h[0] + v[1] * h[1] + v[2] * h[2];
        let l = [
            2.0 * vh * h[0] - v[0],
            2.0 * vh * h[1] - v[1],
            2.0 * vh * h[2] - v[2],
        ];
        let nl = l[2].clamp(0.0, 1.0);
        let nh = h[2].clamp(0.0, 1.0);
        let vh = vh.clamp(0.0, 1.0);
        if nl > 0.0 && nh > 0.0 {
            let g = g1(n_dot_v) * g1(nl);
            let g_vis = g * vh / (nh * n_dot_v);
            let fc = (1.0 - vh).powi(5);
            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }
    (a / f64::from(n), b / f64::from(n))
}

/// The split sum BRDF table, `size` x `size`: column `x` is `N.V = (x + 0.5) / size`, row
/// `y` is roughness `(y + 0.5) / size` (row 0 smooth); R = scale, G = bias, B = 0, A = 1.
/// `Rg16Unorm` or `Rgba16Float` are typical formats.
pub fn brdf_lut(size: u32, samples: u32, format: Format) -> Result<Image> {
    if size == 0 || samples == 0 {
        return Err(Error::InvalidArgument);
    }
    let n = size as usize;
    render(n, n, |x, y| {
        let (a, b) = brdf_lut_value(
            (x as f64 + 0.5) / n as f64,
            (y as f64 + 0.5) / n as f64,
            samples,
        );
        [a as f32, b as f32, 0.0, 1.0]
    })?
    .to_image(format)
}

/// Faces as surfaces (for chaining the functions above).
pub fn surfaces(faces: &Faces) -> Result<[Surface<'_>; 6]> {
    six(|i| faces[i].surface())
}
