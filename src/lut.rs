//! Colour lookup tables: `.cube` files, 1D / 3D LUTs, applying and baking them.
//!
//! [`Lut3d`] is an `n x n x n` RGB lattice (red fastest), [`Lut1d`] three curves of `n` points;
//! both cover a per-channel input domain (default 0..1). [`Cube`] is a `.cube` file (Adobe
//! Cube LUT 1.0 and the Resolve variant: a 1D shaper followed by a 3D table), read with
//! [`Cube::parse`] and written with [`Cube::to_bytes`] (shortest round trip decimals).
//!
//! [`apply`] maps the RGB of an uncompressed surface (values as stored, like
//! [`crate::convert`]; alpha passed through) with trilinear or tetrahedral interpolation;
//! inputs are clamped to the domain. [`Lut3d::bake`] builds a table by running any per-pixel
//! operation over the lattice ([`crate::color::convert`] between profiles, [`crate::filter`]
//! colour adjustments, tone mapping, ...). [`Lut3d::to_hald`] / [`Lut3d::from_hald`] and
//! [`Lut3d::to_strip`] / [`Lut3d::from_strip`] store tables as images (HALD CLUT, and the
//! `n*n x n` strip of game engines).
//!
//! Checked against colour-science 0.4.7 (see docs/spec/lut.md).

use crate::file::Image;
use crate::{Buffer, Error, Format, Result, Surface};
use core::fmt::Write as _;
use vas::Vector;
use vas::storage::Dynamic;

/// An RGB triple.
pub type Rgb = [f32; 3];

/// A per-channel `(min, max)` domain.
type Range = (Rgb, Rgb);

/// Largest 3D table edge.
pub const MAX_3D_SIZE: usize = 256;
/// Largest 1D table length.
pub const MAX_1D_SIZE: usize = 65536;

/// Lattice interpolation of [`Lut3d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// 8 corners.
    Trilinear,
    /// 4 corners of the tetrahedron holding the point (neutral axis exact; what OCIO,
    /// Resolve and LittleCMS use).
    #[default]
    Tetrahedral,
}

/// Three curves of `size` points over `domain_min..domain_max`.
#[derive(Debug)]
pub struct Lut1d {
    pub size: usize,
    pub domain_min: Rgb,
    pub domain_max: Rgb,
    /// `size` entries.
    pub table: Vector<Rgb, Dynamic>,
}

/// An `size^3` lattice over `domain_min..domain_max`; entry `r + size * (g + size * b)`.
#[derive(Debug)]
pub struct Lut3d {
    pub size: usize,
    pub domain_min: Rgb,
    pub domain_max: Rgb,
    /// `size^3` entries.
    pub table: Vector<Rgb, Dynamic>,
}

/// A `.cube` file: an optional 1D shaper applied first, then an optional 3D table.
#[derive(Debug, Default)]
pub struct Cube {
    pub title: Option<vas::String<Dynamic>>,
    pub shaper: Option<Lut1d>,
    pub lut: Option<Lut3d>,
}

fn oom<T>(_: T) -> Error {
    Error::OutOfMemory
}

fn table_of(len: usize) -> Result<Vector<Rgb, Dynamic>> {
    let mut t = Vector::default();
    t.resize(len, [0.0; 3]).map_err(oom)?;
    Ok(t)
}

fn check_domain(min: Rgb, max: Rgb) -> Result<()> {
    let ok = (0..3).all(|c| min[c].is_finite() && max[c].is_finite() && max[c] > min[c]);
    if ok {
        Ok(())
    } else {
        Err(Error::InvalidArgument)
    }
}

/// `(x - min) / (max - min)` clamped to 0..1 (NaN -> 0), in f64.
fn norm(x: f32, min: f32, max: f32) -> f64 {
    let t = (f64::from(x) - f64::from(min)) / (f64::from(max) - f64::from(min));
    if t >= 0.0 { t.min(1.0) } else { 0.0 }
}

/// Lattice cell and fraction of a normalized coordinate.
fn cell(t: f64, n: usize) -> (usize, usize, f64) {
    let s = t * (n - 1) as f64;
    let i = (s as usize).min(n - 1);
    (i, (i + 1).min(n - 1), s - i as f64)
}

impl Lut1d {
    /// The identity over 0..1 (`size` 2..=[`MAX_1D_SIZE`]).
    pub fn identity(size: usize) -> Result<Lut1d> {
        Lut1d::from_fn(size, |v| v)
    }

    /// `f` sampled at `size` points per channel over 0..1.
    pub fn from_fn(size: usize, f: impl Fn(Rgb) -> Rgb) -> Result<Lut1d> {
        if !(2..=MAX_1D_SIZE).contains(&size) {
            return Err(Error::InvalidArgument);
        }
        let mut table = table_of(size)?;
        for (i, e) in table.iter_mut().enumerate() {
            let v = (i as f64 / (size - 1) as f64) as f32;
            *e = f([v; 3]);
        }
        Ok(Lut1d {
            size,
            domain_min: [0.0; 3],
            domain_max: [1.0; 3],
            table,
        })
    }

    fn validate(&self) -> Result<()> {
        if !(2..=MAX_1D_SIZE).contains(&self.size) || self.table.len() != self.size {
            return Err(Error::InvalidArgument);
        }
        check_domain(self.domain_min, self.domain_max)
    }

    fn eval64(&self, v: [f64; 3]) -> [f64; 3] {
        core::array::from_fn(|c| {
            let t = norm(v[c] as f32, self.domain_min[c], self.domain_max[c]);
            let (i0, i1, f) = cell(t, self.size);
            let (a, b) = (f64::from(self.table[i0][c]), f64::from(self.table[i1][c]));
            a + (b - a) * f
        })
    }

    /// Looks up `v` (clamped to the domain, linear interpolation per channel).
    pub fn eval(&self, v: Rgb) -> Rgb {
        let o = self.eval64(v.map(f64::from));
        o.map(|x| x as f32)
    }
}

impl Lut3d {
    /// The identity over 0..1 (`size` 2..=[`MAX_3D_SIZE`]).
    pub fn identity(size: usize) -> Result<Lut3d> {
        Lut3d::from_fn(size, |v| v)
    }

    /// `f` sampled on the lattice over 0..1.
    pub fn from_fn(size: usize, f: impl Fn(Rgb) -> Rgb) -> Result<Lut3d> {
        let mut lut = Lut3d::lattice(size)?;
        for e in lut.table.iter_mut() {
            *e = f(*e);
        }
        Ok(lut)
    }

    /// The identity table (entries are the lattice points).
    fn lattice(size: usize) -> Result<Lut3d> {
        if !(2..=MAX_3D_SIZE).contains(&size) {
            return Err(Error::InvalidArgument);
        }
        let mut table = table_of(size * size * size)?;
        let p = |i: usize| (i as f64 / (size - 1) as f64) as f32;
        for (k, e) in table.iter_mut().enumerate() {
            *e = [p(k % size), p(k / size % size), p(k / (size * size))];
        }
        Ok(Lut3d {
            size,
            domain_min: [0.0; 3],
            domain_max: [1.0; 3],
            table,
        })
    }

    /// Runs a per-pixel operation over the lattice: `op` gets an `Rgba32Float` surface of the
    /// lattice points over `domain_min..domain_max` (alpha 1, one pixel per entry in table
    /// order, `size^2` wide) and returns an image of the same size in any uncompressed format,
    /// whose RGB becomes the table (gray outputs fill all three). Spatial operations (blur,
    /// resize) do not make sense here.
    pub fn bake(
        size: usize,
        domain_min: Rgb,
        domain_max: Rgb,
        op: impl FnOnce(&Surface) -> Result<Image>,
    ) -> Result<Lut3d> {
        check_domain(domain_min, domain_max)?;
        let mut lut = Lut3d::lattice(size)?;
        let n = size * size * size;
        let mut px = crate::buffer::zeroed(n * 16)?;
        for (k, e) in lut.table.iter().enumerate() {
            let v: [f32; 4] = [
                lerp(domain_min[0], domain_max[0], e[0]),
                lerp(domain_min[1], domain_max[1], e[1]),
                lerp(domain_min[2], domain_max[2], e[2]),
                1.0,
            ];
            for (c, f) in v.iter().enumerate() {
                px[k * 16 + c * 4..k * 16 + c * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
        let (w, h) = ((size * size) as u32, size as u32);
        let s = Surface::new(Format::Rgba32Float, w, h, &px)?;
        let out = op(&s)?;
        if out.width != w || out.height != h {
            return Err(Error::InvalidArgument);
        }
        read_rgb(&out.surface()?, |k, rgb| lut.table[k] = rgb)?;
        lut.domain_min = domain_min;
        lut.domain_max = domain_max;
        Ok(lut)
    }

    /// A table converting between two colour profiles (RGB or gray; gray sources take the
    /// red channel), over 0..1.
    pub fn from_profiles(
        size: usize,
        from: &crate::color::Profile,
        to: &crate::color::Profile,
        intent: crate::color::Intent,
    ) -> Result<Lut3d> {
        use crate::color::Channels;
        let fmt = |p: &crate::color::Profile| match p.data_channels() {
            Channels::Gray => Ok(Format::R32Float),
            Channels::Rgb => Ok(Format::Rgba32Float),
            Channels::Other => Err(Error::Unsupported),
        };
        let (fi, fo) = (fmt(from)?, fmt(to)?);
        Lut3d::bake(size, [0.0; 3], [1.0; 3], |s| {
            if fi == Format::R32Float {
                let mut g = Image::new(Format::R32Float, s.width(), s.height())?;
                crate::convert::convert(s, Format::R32Float, &mut g.data)?;
                crate::color::convert(&g.surface()?, from, to, intent, fo)
            } else {
                crate::color::convert(s, from, to, intent, fo)
            }
        })
    }

    fn validate(&self) -> Result<()> {
        let n = self.size;
        if !(2..=MAX_3D_SIZE).contains(&n) || self.table.len() != n * n * n {
            return Err(Error::InvalidArgument);
        }
        check_domain(self.domain_min, self.domain_max)
    }

    fn at(&self, r: usize, g: usize, b: usize) -> [f64; 3] {
        let n = self.size;
        self.table[r + n * (g + n * b)].map(f64::from)
    }

    fn eval64(&self, v: [f64; 3], interp: Interpolation) -> [f64; 3] {
        let n = self.size;
        let t: [f64; 3] =
            core::array::from_fn(|c| norm(v[c] as f32, self.domain_min[c], self.domain_max[c]));
        let (r0, r1, x) = cell(t[0], n);
        let (g0, g1, y) = cell(t[1], n);
        let (b0, b1, z) = cell(t[2], n);
        let c000 = self.at(r0, g0, b0);
        let c111 = self.at(r1, g1, b1);
        let mix = |w: [(f64, [f64; 3]); 4]| -> [f64; 3] {
            core::array::from_fn(|c| w.iter().map(|(k, p)| k * p[c]).sum())
        };
        match interp {
            Interpolation::Tetrahedral => {
                if x > y && y > z {
                    mix([
                        (1.0 - x, c000),
                        (x - y, self.at(r1, g0, b0)),
                        (y - z, self.at(r1, g1, b0)),
                        (z, c111),
                    ])
                } else if x > z && z >= y {
                    mix([
                        (1.0 - x, c000),
                        (x - z, self.at(r1, g0, b0)),
                        (z - y, self.at(r1, g0, b1)),
                        (y, c111),
                    ])
                } else if z >= x && x > y {
                    mix([
                        (1.0 - z, c000),
                        (z - x, self.at(r0, g0, b1)),
                        (x - y, self.at(r1, g0, b1)),
                        (y, c111),
                    ])
                } else if y >= x && x > z {
                    mix([
                        (1.0 - y, c000),
                        (y - x, self.at(r0, g1, b0)),
                        (x - z, self.at(r1, g1, b0)),
                        (z, c111),
                    ])
                } else if y >= z && z >= x {
                    mix([
                        (1.0 - y, c000),
                        (y - z, self.at(r0, g1, b0)),
                        (z - x, self.at(r0, g1, b1)),
                        (x, c111),
                    ])
                } else {
                    mix([
                        (1.0 - z, c000),
                        (z - y, self.at(r0, g0, b1)),
                        (y - x, self.at(r0, g1, b1)),
                        (x, c111),
                    ])
                }
            }
            Interpolation::Trilinear => {
                let l = |a: [f64; 3], b: [f64; 3], f: f64| -> [f64; 3] {
                    core::array::from_fn(|c| a[c] + (b[c] - a[c]) * f)
                };
                let c00 = l(c000, self.at(r1, g0, b0), x);
                let c10 = l(self.at(r0, g1, b0), self.at(r1, g1, b0), x);
                let c01 = l(self.at(r0, g0, b1), self.at(r1, g0, b1), x);
                let c11 = l(self.at(r0, g1, b1), c111, x);
                l(l(c00, c10, y), l(c01, c11, y), z)
            }
        }
    }

    /// Looks up `v` (clamped to the domain).
    pub fn eval(&self, v: Rgb, interp: Interpolation) -> Rgb {
        self.eval64(v.map(f64::from), interp).map(|x| x as f32)
    }

    /// Stores the table as a HALD CLUT image (`size` must be a square `level^2`; the image is
    /// `level^3` square, entry `k` at pixel `k`). `format` is any uncompressed RGB(A) format.
    pub fn to_hald(&self, format: Format) -> Result<Image> {
        self.validate()?;
        let level = hald_level(self.size).ok_or(Error::InvalidArgument)?;
        let side = (level * level * level) as u32;
        self.to_image(side, side, format, |k| k)
    }

    /// Reads a HALD CLUT image (square, side `level^3`, 2 <= level <= 16 or so that the table
    /// fits [`MAX_3D_SIZE`]); domain 0..1.
    pub fn from_hald(src: &Surface) -> Result<Lut3d> {
        let side = src.width() as usize;
        if src.height() as usize != side {
            return Err(Error::InvalidArgument);
        }
        let level = (2..=16)
            .find(|l| l * l * l == side)
            .ok_or(Error::InvalidArgument)?;
        let mut lut = Lut3d::identity(level * level)?;
        read_rgb(src, |k, rgb| lut.table[k] = rgb)?;
        Ok(lut)
    }

    /// Stores the table as a strip of `size` slices along blue, `size * size` wide and `size`
    /// tall: pixel `(r + size * b, g)` (top row green 0), as Unreal / Unity colour grading
    /// LUT textures.
    pub fn to_strip(&self, format: Format) -> Result<Image> {
        self.validate()?;
        let n = self.size;
        self.to_image((n * n) as u32, n as u32, format, |k| {
            let (r, g, b) = (k % n, k / n % n, k / (n * n));
            g * n * n + b * n + r
        })
    }

    /// Reads a strip written like [`Lut3d::to_strip`] (`n * n` by `n`); domain 0..1.
    pub fn from_strip(src: &Surface) -> Result<Lut3d> {
        let n = src.height() as usize;
        if src.width() as usize != n * n {
            return Err(Error::InvalidArgument);
        }
        let mut lut = Lut3d::identity(n)?;
        read_rgb(src, |p, rgb| {
            let (x, g) = (p % (n * n), p / (n * n));
            lut.table[x % n + n * (g + n * (x / n))] = rgb;
        })?;
        Ok(lut)
    }

    /// Image `w x h` with table entry `k` at pixel `pos(k)`.
    fn to_image(
        &self,
        w: u32,
        h: u32,
        format: Format,
        pos: impl Fn(usize) -> usize,
    ) -> Result<Image> {
        if format.is_compressed() {
            return Err(Error::InvalidArgument);
        }
        let mut px = crate::buffer::zeroed(w as usize * h as usize * 16)?;
        for (k, e) in self.table.iter().enumerate() {
            let p = pos(k) * 16;
            for (c, f) in [e[0], e[1], e[2], 1.0].iter().enumerate() {
                px[p + c * 4..p + c * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
        let s = Surface::new(Format::Rgba32Float, w, h, &px)?;
        let mut img = Image::new(format, w, h)?;
        crate::convert::convert(&s, format, &mut img.data)?;
        Ok(img)
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (f64::from(a) + (f64::from(b) - f64::from(a)) * f64::from(t)) as f32
}

fn hald_level(size: usize) -> Option<usize> {
    (2..=16).find(|l| l * l == size)
}

/// Calls `f(pixel index, rgb)` for every pixel (gray formats replicated).
fn read_rgb(src: &Surface, mut f: impl FnMut(usize, Rgb)) -> Result<()> {
    let n = crate::resize::channels(src.format()).ok_or(Error::Unsupported)?;
    let px = crate::convert::convert_to_buffer(src, Format::Rgba32Float)?;
    for (k, p) in px.as_chunks::<16>().0.iter().enumerate() {
        let v = |c: usize| f32::from_le_bytes([p[c * 4], p[c * 4 + 1], p[c * 4 + 2], p[c * 4 + 3]]);
        f(
            k,
            if n <= 2 {
                [v(0); 3]
            } else {
                [v(0), v(1), v(2)]
            },
        );
    }
    Ok(())
}

impl Cube {
    /// A file holding only `lut`.
    pub fn new(lut: Lut3d) -> Cube {
        Cube {
            title: None,
            shaper: None,
            lut: Some(lut),
        }
    }

    /// Parses a `.cube` file: `TITLE`, `DOMAIN_MIN` / `DOMAIN_MAX` (both tables),
    /// `LUT_1D_SIZE`, `LUT_3D_SIZE`, Resolve's `LUT_1D_INPUT_RANGE` / `LUT_3D_INPUT_RANGE`,
    /// `#` comments, then the 1D entries followed by the 3D entries (red fastest). Other
    /// keywords are skipped. `InvalidData` for malformed files.
    pub fn parse(data: &[u8]) -> Result<Cube> {
        let text = core::str::from_utf8(data).map_err(|_| Error::InvalidData)?;
        let mut title: Option<vas::String<Dynamic>> = None;
        let (mut n1, mut n3) = (0usize, 0usize);
        let (mut dmin, mut dmax) = ([0.0f32; 3], [1.0f32; 3]);
        let (mut r1, mut r3): (Option<Range>, Option<Range>) = (None, None);
        let mut vals = Vector::<Rgb, Dynamic>::default();
        let num = |s: Option<&str>| -> Result<f32> {
            let v: f32 = s
                .ok_or(Error::InvalidData)?
                .parse()
                .map_err(|_| Error::InvalidData)?;
            if v.is_finite() {
                Ok(v)
            } else {
                Err(Error::InvalidData)
            }
        };
        let size = |s: Option<&str>, max: usize| -> Result<usize> {
            let v: usize = s
                .ok_or(Error::InvalidData)?
                .parse()
                .map_err(|_| Error::InvalidData)?;
            if (2..=max).contains(&v) {
                Ok(v)
            } else {
                Err(Error::InvalidData)
            }
        };
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let key = it.next().unwrap_or("");
            let starts_num = key
                .bytes()
                .next()
                .is_some_and(|b| b.is_ascii_digit() || b == b'-' || b == b'+' || b == b'.');
            if starts_num {
                let rgb = [num(Some(key))?, num(it.next())?, num(it.next())?];
                if it.next().is_some() {
                    return Err(Error::InvalidData);
                }
                vals.push(rgb).map_err(oom)?;
                continue;
            }
            if !vals.is_empty() {
                // Keywords come before the data.
                return Err(Error::InvalidData);
            }
            match key {
                "TITLE" => {
                    let rest = line.strip_prefix("TITLE").unwrap_or("").trim();
                    let t = rest
                        .strip_prefix('"')
                        .and_then(|r| r.strip_suffix('"'))
                        .unwrap_or(rest);
                    let mut s = vas::String::<Dynamic>::default();
                    s.push_str(t).map_err(oom)?;
                    title = Some(s);
                }
                "LUT_1D_SIZE" => n1 = size(it.next(), MAX_1D_SIZE)?,
                "LUT_3D_SIZE" => n3 = size(it.next(), MAX_3D_SIZE)?,
                "DOMAIN_MIN" => dmin = [num(it.next())?, num(it.next())?, num(it.next())?],
                "DOMAIN_MAX" => dmax = [num(it.next())?, num(it.next())?, num(it.next())?],
                "LUT_1D_INPUT_RANGE" | "LUT_3D_INPUT_RANGE" => {
                    let (a, b) = (num(it.next())?, num(it.next())?);
                    let r = Some(([a; 3], [b; 3]));
                    if key == "LUT_1D_INPUT_RANGE" {
                        r1 = r;
                    } else {
                        r3 = r;
                    }
                }
                _ => {}
            }
        }
        if (n1 == 0 && n3 == 0) || vals.len() != n1 + n3 * n3 * n3 {
            return Err(Error::InvalidData);
        }
        let mk = |n: usize,
                  off: usize,
                  (lo, hi): (Rgb, Rgb)|
         -> Result<(Vector<Rgb, Dynamic>, Rgb, Rgb)> {
            check_domain(lo, hi).map_err(|_| Error::InvalidData)?;
            let mut t = Vector::default();
            t.extend_from_slice(&vals[off..off + n]).map_err(oom)?;
            Ok((t, lo, hi))
        };
        let shaper = if n1 > 0 {
            let (table, lo, hi) = mk(n1, 0, r1.unwrap_or((dmin, dmax)))?;
            Some(Lut1d {
                size: n1,
                domain_min: lo,
                domain_max: hi,
                table,
            })
        } else {
            None
        };
        let lut = if n3 > 0 {
            let (table, lo, hi) = mk(n3 * n3 * n3, n1, r3.unwrap_or((dmin, dmax)))?;
            Some(Lut3d {
                size: n3,
                domain_min: lo,
                domain_max: hi,
                table,
            })
        } else {
            None
        };
        Ok(Cube { title, shaper, lut })
    }

    /// Writes the `.cube` text. A single table uses `DOMAIN_MIN` / `DOMAIN_MAX` (omitted for
    /// 0..1); a shaper with a 3D table uses Resolve's `LUT_1D_INPUT_RANGE` /
    /// `LUT_3D_INPUT_RANGE` (`InvalidArgument` unless both domains are the same on all
    /// channels).
    pub fn to_bytes(&self) -> Result<Buffer> {
        let mut s = vas::String::<Dynamic>::default();
        let w = |s: &mut vas::String<Dynamic>, a: core::fmt::Arguments| s.write_fmt(a).map_err(oom);
        if let Some(t) = &self.title {
            if t.contains('"') || t.contains('\n') || t.contains('\r') {
                return Err(Error::InvalidArgument);
            }
            w(&mut s, format_args!("TITLE \"{}\"\n", t.as_str()))?;
        }
        if let Some(l) = &self.shaper {
            l.validate()?;
        }
        if let Some(l) = &self.lut {
            l.validate()?;
        }
        let uniform =
            |lo: Rgb, hi: Rgb| lo[0] == lo[1] && lo[1] == lo[2] && hi[0] == hi[1] && hi[1] == hi[2];
        let unit = |lo: Rgb, hi: Rgb| lo == [0.0; 3] && hi == [1.0; 3];
        match (&self.shaper, &self.lut) {
            (None, None) => return Err(Error::InvalidArgument),
            (Some(a), Some(b)) => {
                if !uniform(a.domain_min, a.domain_max) || !uniform(b.domain_min, b.domain_max) {
                    return Err(Error::InvalidArgument);
                }
                w(&mut s, format_args!("LUT_1D_SIZE {}\n", a.size))?;
                w(
                    &mut s,
                    format_args!(
                        "LUT_1D_INPUT_RANGE {} {}\n",
                        a.domain_min[0], a.domain_max[0]
                    ),
                )?;
                w(&mut s, format_args!("LUT_3D_SIZE {}\n", b.size))?;
                w(
                    &mut s,
                    format_args!(
                        "LUT_3D_INPUT_RANGE {} {}\n",
                        b.domain_min[0], b.domain_max[0]
                    ),
                )?;
            }
            (one_d, three_d) => {
                let (key, n, lo, hi) = match (one_d, three_d) {
                    (Some(a), _) => ("LUT_1D_SIZE", a.size, a.domain_min, a.domain_max),
                    (_, Some(b)) => ("LUT_3D_SIZE", b.size, b.domain_min, b.domain_max),
                    _ => return Err(Error::InvalidArgument),
                };
                w(&mut s, format_args!("{key} {n}\n"))?;
                if !unit(lo, hi) {
                    w(
                        &mut s,
                        format_args!("DOMAIN_MIN {} {} {}\n", lo[0], lo[1], lo[2]),
                    )?;
                    w(
                        &mut s,
                        format_args!("DOMAIN_MAX {} {} {}\n", hi[0], hi[1], hi[2]),
                    )?;
                }
            }
        }
        let tables = self
            .shaper
            .iter()
            .map(|l| &l.table)
            .chain(self.lut.iter().map(|l| &l.table));
        for t in tables {
            for e in t.iter() {
                w(&mut s, format_args!("{} {} {}\n", e[0], e[1], e[2]))?;
            }
        }
        let mut out = Buffer::default();
        out.extend_from_slice(s.as_bytes()).map_err(oom)?;
        Ok(out)
    }

    fn validate(&self) -> Result<()> {
        if self.shaper.is_none() && self.lut.is_none() {
            return Err(Error::InvalidArgument);
        }
        if let Some(l) = &self.shaper {
            l.validate()?;
        }
        if let Some(l) = &self.lut {
            l.validate()?;
        }
        Ok(())
    }

    /// Looks up `v`: shaper, then the 3D table.
    pub fn eval(&self, v: Rgb, interp: Interpolation) -> Rgb {
        self.eval64(v.map(f64::from), interp).map(|x| x as f32)
    }

    fn eval64(&self, v: [f64; 3], interp: Interpolation) -> [f64; 3] {
        let v = match &self.shaper {
            Some(s) => s.eval64(v),
            None => v,
        };
        match &self.lut {
            Some(l) => l.eval64(v, interp),
            None => v,
        }
    }
}

/// Maps the RGB of `src` (gray, gray + alpha, RGB or RGBA; values as stored) through `cube`
/// and writes `format` (alpha copied, 1 when the source has none; gray + alpha outputs keep
/// alpha in green). Rows run in parallel; interpolation is done in f64.
///
/// Errors: `InvalidArgument` for an empty or malformed cube or a compressed output,
/// `Unsupported` for compressed / non-colour inputs.
pub fn apply(src: &Surface, cube: &Cube, interp: Interpolation, format: Format) -> Result<Image> {
    cube.validate()?;
    let n_in = crate::resize::channels(src.format()).ok_or(Error::Unsupported)?;
    let n_out = crate::resize::channels(format).ok_or(Error::InvalidArgument)?;
    let (w, h) = (src.width() as usize, src.height() as usize);
    let rgba = crate::convert::convert_to_buffer(src, Format::Rgba32Float)?;
    let mut out = crate::buffer::zeroed(w * h * 16)?;
    crate::parallel::for_each_unit(&mut out, w * 16, w * h * 64, |y, row| {
        for x in 0..w {
            let p = (y * w + x) * 16;
            let at = |c: usize| {
                let k = p + c * 4;
                f32::from_le_bytes([rgba[k], rgba[k + 1], rgba[k + 2], rgba[k + 3]])
            };
            let (rgb, a) = match n_in {
                1 => ([at(0); 3], 1.0),
                2 => ([at(0); 3], at(1)),
                3 => ([at(0), at(1), at(2)], 1.0),
                _ => ([at(0), at(1), at(2)], at(3)),
            };
            let o = cube.eval(rgb, interp);
            let v = if n_out <= 2 {
                [o[0], a, o[0], a]
            } else {
                [o[0], o[1], o[2], a]
            };
            for (c, f) in v.iter().enumerate() {
                row[x * 16 + c * 4..x * 16 + c * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
    });
    let s = Surface::new(Format::Rgba32Float, w as u32, h as u32, &out)?;
    let mut img = Image::new(format, w as u32, h as u32)?;
    crate::convert::convert(&s, format, &mut img.data)?;
    Ok(img)
}
