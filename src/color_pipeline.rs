// SPDX-FileCopyrightText: 1998-2022 Marti Maria Saguer (Little CMS: 16-bit curve and CLUT
// interpolation in `interp16`, `to_word`, `to_fixed_domain`, `linear_interp`, `Clut::eval`)
// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0 AND MIT
// The marked parts are translated to Rust from Little CMS 2.14 (cmsintrp.c, cmsgamma.c,
// cmslut.c) and modified; see NOTICE.md and LICENSES/MIT-LittleCMS.txt.
//! Per pixel ICC pipeline evaluation in f64 (the LittleCMS model without its optimizations):
//! device -> PCS (XYZ D50) -> device, through matrix / TRC shapers, gray TRCs or LUT tags
//! (`lut8`, `lut16`, `mAB`, `mBA`); sampled curves and CLUTs are evaluated in 16-bit fixed
//! point like LittleCMS does even in float pipelines.

use crate::{Error, Result};
use moxcms::{ColorProfile, DataColorSpace, LutStore, LutWarehouse, ToneReprCurve};
use vas::Vector;
use vas::storage::Dynamic;

type F = Vector<f64, Dynamic>;

/// ICC D50 illuminant.
pub(crate) const D50: [f64; 3] = [0.9642, 1.0, 0.8249];

fn vec_of(it: impl Iterator<Item = f64>) -> Result<F> {
    let mut v = F::default();
    for x in it {
        v.push(x).map_err(|_| Error::OutOfMemory)?;
    }
    Ok(v)
}

fn store(s: &LutStore) -> Result<F> {
    match s {
        LutStore::Store8(v) => vec_of(v.iter().map(|&x| f64::from(x) / 255.0)),
        LutStore::Store16(v) => vec_of(v.iter().map(|&x| f64::from(x) / 65535.0)),
    }
}

/// A tone curve.
pub(crate) enum Curve {
    Identity,
    Gamma(f64),
    /// Sampled at equal steps over 0..1.
    Table(F),
    /// ICC parametric types 0..4 with `g a b c d e f` (missing ones 0).
    Para(u8, [f64; 7]),
}

impl Curve {
    pub fn new(t: &ToneReprCurve) -> Result<Curve> {
        Ok(match t {
            ToneReprCurve::Lut(v) if v.is_empty() => Curve::Identity,
            ToneReprCurve::Lut(v) if v.len() == 1 => Curve::Gamma(f64::from(v[0]) / 256.0),
            ToneReprCurve::Lut(v) => {
                Curve::Table(vec_of(v.iter().map(|&x| f64::from(x) / 65535.0))?)
            }
            ToneReprCurve::Parametric(p) => {
                let kind = match p.len() {
                    1 => 0,
                    3 => 1,
                    4 => 2,
                    5 => 3,
                    7 => 4,
                    _ => return Err(Error::InvalidData),
                };
                let mut q = [0.0; 7];
                for (d, s) in q.iter_mut().zip(p) {
                    *d = f64::from(*s);
                }
                if kind == 0 {
                    Curve::Gamma(q[0])
                } else {
                    Curve::Para(kind, q)
                }
            }
        })
    }

    fn table(n: usize, src: &[f64], off: usize) -> Result<Curve> {
        vec_of((0..n).map(|i| src[off + i])).map(Curve::Table)
    }

    pub fn eval(&self, x: f64) -> f64 {
        match self {
            Curve::Identity => x,
            Curve::Gamma(g) => {
                if x <= 0.0 {
                    0.0
                } else {
                    x.powf(*g)
                }
            }
            Curve::Table(t) => interp16(t, x),
            Curve::Para(k, p) => {
                let [g, a, b, c, d, e, f] = *p;
                let pw = |v: f64| if v > 0.0 { v.powf(g) } else { 0.0 };
                match k {
                    1 => {
                        if x >= -b / a {
                            pw(a * x + b)
                        } else {
                            0.0
                        }
                    }
                    2 => {
                        if x >= -b / a {
                            pw(a * x + b) + c
                        } else {
                            c
                        }
                    }
                    3 => {
                        if x >= d {
                            pw(a * x + b)
                        } else {
                            c * x
                        }
                    }
                    _ => {
                        if x >= d {
                            pw(a * x + b) + e
                        } else {
                            c * x + f
                        }
                    }
                }
            }
        }
    }

    /// Inverse (tables: monotonic search with linear interpolation, clamped to 0..1).
    pub fn inv(&self, y: f64) -> f64 {
        let root = |v: f64, g: f64| if v > 0.0 { v.powf(1.0 / g) } else { 0.0 };
        match self {
            Curve::Identity => y,
            Curve::Gamma(g) => {
                if *g == 0.0 {
                    0.0
                } else {
                    root(y, *g)
                }
            }
            Curve::Table(t) => interp_inv(t, y),
            Curve::Para(k, p) => {
                let [g, a, b, c, d, e, f] = *p;
                match k {
                    1 => (root(y, g) - b) / a,
                    2 => {
                        if y >= c {
                            (root(y - c, g) - b) / a
                        } else {
                            -b / a
                        }
                    }
                    3 => {
                        let knee = if a * d + b > 0.0 {
                            (a * d + b).powf(g)
                        } else {
                            0.0
                        };
                        if y >= knee {
                            (root(y, g) - b) / a
                        } else if c != 0.0 {
                            y / c
                        } else {
                            0.0
                        }
                    }
                    _ => {
                        let knee = if a * d + b > 0.0 {
                            (a * d + b).powf(g)
                        } else {
                            0.0
                        } + e;
                        if y >= knee {
                            (root(y - e, g) - b) / a
                        } else if c != 0.0 {
                            (y - f) / c
                        } else {
                            0.0
                        }
                    }
                }
            }
        }
    }
}

/// Linear interpolation in a table over 0..1 (input clamped).
fn interp(t: &[f64], x: f64) -> f64 {
    let n = t.len();
    if n == 0 {
        return x;
    }
    if n == 1 {
        return t[0];
    }
    let p = x.clamp(0.0, 1.0) * (n - 1) as f64;
    let i = (p.floor() as usize).min(n - 2);
    let f = p - i as f64;
    t[i] + (t[i + 1] - t[i]) * f
}

/// A sampled curve the way LittleCMS evaluates it (`cmsEvalToneCurveFloat`: tables are 16-bit
/// curves even in float pipelines): input rounded to a 16-bit word, `LinLerp1D` fixed point
/// interpolation, 16-bit output.
fn interp16(t: &[f64], x: f64) -> f64 {
    let n = t.len();
    if n < 2 {
        return interp(t, x);
    }
    let d = x * 65535.0 + 0.5;
    let v: i64 = if d.is_nan() || d <= 0.0 {
        0
    } else if d >= 65535.0 {
        65535
    } else {
        d.floor() as i64
    };
    let word = |i: usize| (t[i] * 65535.0).round() as i64;
    if v == 0xffff {
        return word(n - 1) as f64 / 65535.0;
    }
    let val3 = (n - 1) as i64 * v;
    let val3 = val3 + (val3 + 0x7fff) / 0xffff;
    let (cell, rest) = ((val3 >> 16) as usize, val3 & 0xffff);
    let (y0, y1) = (word(cell), word((cell + 1).min(n - 1)));
    let out = ((((y1 - y0) * rest + 0x8000) >> 16) + y0) & 0xffff;
    out as f64 / 65535.0
}

/// Inverse of a monotonic table (either direction), clamped to 0..1.
fn interp_inv(t: &[f64], y: f64) -> f64 {
    let n = t.len();
    if n < 2 {
        return y;
    }
    let up = t[n - 1] >= t[0];
    let (lo, hi) = if up {
        (t[0], t[n - 1])
    } else {
        (t[n - 1], t[0])
    };
    if y <= lo {
        return if up { 0.0 } else { 1.0 };
    }
    if y >= hi {
        return if up { 1.0 } else { 0.0 };
    }
    // First segment containing y.
    for i in 0..n - 1 {
        let (a, b) = (t[i], t[i + 1]);
        let inside = if up {
            y >= a && y <= b
        } else {
            y <= a && y >= b
        };
        if inside {
            let f = if b != a { (y - a) / (b - a) } else { 0.0 };
            return (i as f64 + f) / (n - 1) as f64;
        }
    }
    0.0
}

/// A colour lookup table with `n_in` (1..=3) inputs.
struct Clut {
    grid: [usize; 3],
    n_in: usize,
    n_out: usize,
    data: F,
    /// LittleCMS uses trilinear interpolation for tables indexed by Lab (PCS -> device of Lab
    /// profiles), tetrahedral otherwise.
    trilinear: bool,
}

/// `_cmsQuickSaturateWord(v * 65535)`.
fn to_word(v: f64) -> i32 {
    let d = v * 65535.0 + 0.5;
    if d.is_nan() || d <= 0.0 {
        0
    } else if d >= 65535.0 {
        65535
    } else {
        d.floor() as i32
    }
}

/// `_cmsToFixedDomain`.
fn to_fixed_domain(a: i32) -> i32 {
    a + (a + 0x7fff) / 0xffff
}

/// LittleCMS `LinearInterp` (16.16 fixed point, u32 arithmetic).
fn linear_interp(a: i32, l: i32, h: i32) -> i32 {
    let dif = ((h - l) as u32).wrapping_mul(a as u32).wrapping_add(0x8000);
    i32::from(((dif >> 16).wrapping_add(l as u32)) as u16)
}

impl Clut {
    /// Evaluates like LittleCMS `EvaluateCLUTfloatIn16` (ICC CLUTs are 16-bit stages even in
    /// float pipelines): inputs rounded to 16-bit words, `Eval1Input` / `TetrahedralInterp16`
    /// / `TrilinearInterp16` (with their 32-bit wrap-around), outputs `word / 65535`. Two
    /// input tables use the float path.
    fn eval(&self, x: &[f64], out: &mut [f64]) {
        if !matches!(self.n_in, 1 | 3) {
            return self.eval_float(x, out);
        }
        let n = self.n_out;
        let word = |i: usize| -> i32 {
            (self.data.get(i).copied().unwrap_or(0.0) * 65535.0).round() as i32
        };
        let input: [i32; 3] =
            core::array::from_fn(|d| if d < self.n_in { to_word(x[d]) } else { 0 });
        // opta: stride of each input (the first input varies slowest).
        let mut opta = [0usize; 3];
        let mut stride = n;
        for d in (0..self.n_in).rev() {
            opta[d] = stride;
            stride *= self.grid[d];
        }
        let domain: [i32; 3] = core::array::from_fn(|d| self.grid[d] as i32 - 1);
        if self.n_in == 1 {
            if input[0] == 0xffff || domain[0] == 0 {
                let y0 = domain[0] as usize * opta[0];
                for (c, o) in out.iter_mut().enumerate().take(n) {
                    *o = f64::from(word(y0 + c)) / 65535.0;
                }
                return;
            }
            let fk = to_fixed_domain(input[0] * domain[0]);
            let (k0, rk) = ((fk >> 16) as usize, fk & 0xffff);
            let (b0, b1) = (k0 * opta[0], (k0 + 1) * opta[0]);
            for (c, o) in out.iter_mut().enumerate().take(n) {
                *o = f64::from(linear_interp(rk, word(b0 + c), word(b1 + c))) / 65535.0;
            }
            return;
        }
        let f: [i32; 3] = core::array::from_fn(|d| to_fixed_domain(input[d] * domain[d]));
        let base = (f[0] >> 16) as usize * opta[0]
            + (f[1] >> 16) as usize * opta[1]
            + (f[2] >> 16) as usize * opta[2];
        let (rx, ry, rz) = (f[0] & 0xffff, f[1] & 0xffff, f[2] & 0xffff);
        let step = |d: usize| if input[d] == 0xffff { 0 } else { opta[d] };
        let (x1, y1, z1) = (step(0), step(1), step(2));
        let t = |o: usize, c: usize| word(base + o + c);
        for (c, o) in out.iter_mut().enumerate().take(n) {
            let v = if self.trilinear {
                let lerp = |a: i32, l: i32, h: i32| -> i32 {
                    let p = (h - l).wrapping_mul(a).wrapping_add(0x8000) >> 16;
                    i32::from(l.wrapping_add(p) as u16)
                };
                let dx00 = lerp(rx, t(0, c), t(x1, c));
                let dx01 = lerp(rx, t(z1, c), t(x1 + z1, c));
                let dx10 = lerp(rx, t(y1, c), t(x1 + y1, c));
                let dx11 = lerp(rx, t(y1 + z1, c), t(x1 + y1 + z1, c));
                let dxy0 = lerp(ry, dx00, dx10);
                let dxy1 = lerp(ry, dx01, dx11);
                lerp(rz, dxy0, dxy1)
            } else {
                let c0 = t(0, c);
                let (c1, c2, c3) = if rx >= ry {
                    if ry >= rz {
                        (
                            t(x1, c) - c0,
                            t(x1 + y1, c) - t(x1, c),
                            t(x1 + y1 + z1, c) - t(x1 + y1, c),
                        )
                    } else if rz >= rx {
                        (
                            t(x1 + z1, c) - t(z1, c),
                            t(x1 + y1 + z1, c) - t(x1 + z1, c),
                            t(z1, c) - c0,
                        )
                    } else {
                        (
                            t(x1, c) - c0,
                            t(x1 + y1 + z1, c) - t(x1 + z1, c),
                            t(x1 + z1, c) - t(x1, c),
                        )
                    }
                } else if rx >= rz {
                    (
                        t(x1 + y1, c) - t(y1, c),
                        t(y1, c) - c0,
                        t(x1 + y1 + z1, c) - t(x1 + y1, c),
                    )
                } else if ry >= rz {
                    (
                        t(x1 + y1 + z1, c) - t(y1 + z1, c),
                        t(y1, c) - c0,
                        t(y1 + z1, c) - t(y1, c),
                    )
                } else {
                    (
                        t(x1 + y1 + z1, c) - t(y1 + z1, c),
                        t(y1 + z1, c) - t(z1, c),
                        t(z1, c) - c0,
                    )
                };
                let rest = c1
                    .wrapping_mul(rx)
                    .wrapping_add(c2.wrapping_mul(ry))
                    .wrapping_add(c3.wrapping_mul(rz))
                    .wrapping_add(0x8001);
                let r = rest.wrapping_add(rest >> 16) >> 16;
                i32::from(c0.wrapping_add(r) as u16)
            };
            *o = f64::from(v) / 65535.0;
        }
    }

    /// Bilinear (2 inputs) interpolation in f64.
    #[allow(clippy::needless_range_loop)]
    fn eval_float(&self, x: &[f64], out: &mut [f64]) {
        let n = self.n_out;
        let mut base = [0usize; 3];
        let mut frac = [0.0f64; 3];
        let mut step = [0usize; 3];
        let mut stride = n;
        for d in (0..self.n_in).rev() {
            let g = self.grid[d];
            let p = x[d].clamp(0.0, 1.0) * (g - 1) as f64;
            let i = (p.floor() as usize).min(g.saturating_sub(2));
            base[d] = i * stride;
            frac[d] = if g > 1 { p - i as f64 } else { 0.0 };
            step[d] = if g > 1 { stride } else { 0 };
            stride *= g;
        }
        let at = |o: usize, c: usize| self.data.get(o + c).copied().unwrap_or(0.0);
        let b0 = base[0] + base[1] + base[2];
        match self.n_in {
            1 => {
                for c in 0..n {
                    out[c] = at(b0, c) + (at(b0 + step[0], c) - at(b0, c)) * frac[0];
                }
            }
            2 => {
                for c in 0..n {
                    let v00 = at(b0, c);
                    let v10 = at(b0 + step[0], c);
                    let v01 = at(b0 + step[1], c);
                    let v11 = at(b0 + step[0] + step[1], c);
                    let a = v00 + (v10 - v00) * frac[0];
                    let b = v01 + (v11 - v01) * frac[0];
                    out[c] = a + (b - a) * frac[1];
                }
            }
            _ => {
                // LittleCMS Eval3Inputs tetrahedra.
                let (rx, ry, rz) = (frac[0], frac[1], frac[2]);
                let (x1, y1, z1) = (step[0], step[1], step[2]);
                for c in 0..n {
                    let c0 = at(b0, c);
                    let v = |o: usize| at(b0 + o, c);
                    let (c1, c2, c3);
                    if rx >= ry && ry >= rz {
                        c1 = v(x1) - c0;
                        c2 = v(x1 + y1) - v(x1);
                        c3 = v(x1 + y1 + z1) - v(x1 + y1);
                    } else if rx >= rz && rz >= ry {
                        c1 = v(x1) - c0;
                        c2 = v(x1 + y1 + z1) - v(x1 + z1);
                        c3 = v(x1 + z1) - v(x1);
                    } else if rz >= rx && rx >= ry {
                        c1 = v(x1 + z1) - v(z1);
                        c2 = v(x1 + y1 + z1) - v(x1 + z1);
                        c3 = v(z1) - c0;
                    } else if ry >= rx && rx >= rz {
                        c1 = v(x1 + y1) - v(y1);
                        c2 = v(y1) - c0;
                        c3 = v(x1 + y1 + z1) - v(x1 + y1);
                    } else if ry >= rz && rz >= rx {
                        c1 = v(x1 + y1 + z1) - v(y1 + z1);
                        c2 = v(y1) - c0;
                        c3 = v(y1 + z1) - v(y1);
                    } else {
                        c1 = v(x1 + y1 + z1) - v(y1 + z1);
                        c2 = v(y1 + z1) - v(z1);
                        c3 = v(z1) - c0;
                    }
                    out[c] = c0 + c1 * rx + c2 * ry + c3 * rz;
                }
            }
        }
    }
}

fn curves(cs: &[ToneReprCurve]) -> Result<Vector<Curve, Dynamic>> {
    let mut v = Vector::<Curve, Dynamic>::default();
    for c in cs {
        v.push(Curve::new(c)?).map_err(|_| Error::OutOfMemory)?;
    }
    Ok(v)
}

/// PCS value encodings of LUT tags.
#[derive(Clone, Copy, PartialEq)]
enum Pcs {
    Xyz,
    /// lut16: L 0..0xFF00, a / b with 0x8000 = 0.
    LabLegacy16,
    /// lut8, mAB / mBA: L 0..1 = 0..100, a / b 0..1 = -128..127.
    Lab,
}

/// A LUT tag ready to evaluate (3 or 1 channels in, 3 or 1 out).
pub(crate) struct Lut {
    n_in: usize,
    n_out: usize,
    /// lut8 / lut16 matrix, applied for XYZ input.
    matrix: Option<[[f64; 3]; 3]>,
    /// mAB / mBA matrix + offset.
    mab_matrix: Option<([[f64; 3]; 3], [f64; 3])>,
    first: Vector<Curve, Dynamic>,
    clut: Option<Clut>,
    /// mAB "M" / mBA "M" curves (around the matrix).
    m: Vector<Curve, Dynamic>,
    last: Vector<Curve, Dynamic>,
    /// mBA: B curves, matrix, M curves, CLUT, A curves.
    mba: bool,
    pcs: Pcs,
}

impl Lut {
    fn new(w: &LutWarehouse, pcs_lab: bool, input_is_pcs: bool) -> Result<Lut> {
        match w {
            LutWarehouse::Lut(l) => {
                let (n_in, n_out) = (
                    usize::from(l.num_input_channels),
                    usize::from(l.num_output_channels),
                );
                if !matches!(n_in, 1 | 3) || !matches!(n_out, 1 | 3) {
                    return Err(Error::Unsupported);
                }
                let it = store(&l.input_table)?;
                let ot = store(&l.output_table)?;
                let (ni, no) = (
                    usize::from(l.num_input_table_entries),
                    usize::from(l.num_output_table_entries),
                );
                let mut first = Vector::<Curve, Dynamic>::default();
                let mut last = Vector::<Curve, Dynamic>::default();
                for c in 0..n_in {
                    if it.len() < (c + 1) * ni {
                        return Err(Error::InvalidData);
                    }
                    first
                        .push(Curve::table(ni, &it, c * ni)?)
                        .map_err(|_| Error::OutOfMemory)?;
                }
                for c in 0..n_out {
                    if ot.len() < (c + 1) * no {
                        return Err(Error::InvalidData);
                    }
                    last.push(Curve::table(no, &ot, c * no)?)
                        .map_err(|_| Error::OutOfMemory)?;
                }
                let g = usize::from(l.num_clut_grid_points);
                let data = store(&l.clut_table)?;
                if g < 2 || data.len() < g.pow(n_in as u32) * n_out {
                    return Err(Error::InvalidData);
                }
                let lut16 = matches!(l.lut_type, moxcms::LutType::Lut16);
                Ok(Lut {
                    n_in,
                    n_out,
                    matrix: (input_is_pcs && !pcs_lab && n_in == 3).then_some(l.matrix.v),
                    mab_matrix: None,
                    first,
                    clut: Some(Clut {
                        grid: [
                            g,
                            if n_in > 1 { g } else { 1 },
                            if n_in > 2 { g } else { 1 },
                        ],
                        n_in,
                        n_out,
                        data,
                        trilinear: input_is_pcs && pcs_lab && n_in == 3,
                    }),
                    m: Vector::default(),
                    last,
                    mba: false,
                    pcs: if !pcs_lab {
                        Pcs::Xyz
                    } else if lut16 {
                        Pcs::LabLegacy16
                    } else {
                        Pcs::Lab
                    },
                })
            }
            LutWarehouse::Multidimensional(m) => {
                let (n_in, n_out) = (
                    usize::from(m.num_input_channels),
                    usize::from(m.num_output_channels),
                );
                if !matches!(n_in, 1 | 3) || !matches!(n_out, 1 | 3) {
                    return Err(Error::Unsupported);
                }
                let mba = input_is_pcs;
                // mAB: A (n_in) -> CLUT -> M -> matrix -> B (n_out).
                // mBA: B (n_in) -> matrix -> M -> CLUT -> A (n_out).
                let (first, last) = if mba {
                    (curves(&m.b_curves)?, curves(&m.a_curves)?)
                } else {
                    (curves(&m.a_curves)?, curves(&m.b_curves)?)
                };
                let clut = match &m.clut {
                    Some(s) => {
                        let (ci, co) = if mba {
                            (n_in.min(3), n_out)
                        } else {
                            (n_in, n_out.min(3))
                        };
                        let mut grid = [1usize; 3];
                        for (d, gp) in grid.iter_mut().zip(m.grid_points.iter()).take(ci) {
                            *d = usize::from(*gp).max(1);
                        }
                        let data = store(s)?;
                        if data.len() < grid.iter().product::<usize>() * co {
                            return Err(Error::InvalidData);
                        }
                        Some(Clut {
                            grid,
                            n_in: ci,
                            n_out: co,
                            data,
                            trilinear: mba && pcs_lab && ci == 3,
                        })
                    }
                    None => None,
                };
                let has_matrix = !m.m_curves.is_empty();
                Ok(Lut {
                    n_in,
                    n_out,
                    matrix: None,
                    mab_matrix: has_matrix.then_some((m.matrix.v, m.bias.v)),
                    first,
                    clut,
                    m: curves(&m.m_curves)?,
                    last,
                    mba,
                    pcs: if pcs_lab { Pcs::Lab } else { Pcs::Xyz },
                })
            }
        }
    }

    /// The A2B0 table of a device link: device values in and out (no PCS encoding).
    pub(crate) fn device_link(w: &LutWarehouse) -> Result<Lut> {
        Lut::new(w, false, false)
    }

    /// Input and output channels.
    pub(crate) fn channels(&self) -> (usize, usize) {
        (self.n_in, self.n_out)
    }

    /// Evaluates a device link table (normalized device values).
    pub(crate) fn eval_link(&self, v: &[f64; 3]) -> [f64; 3] {
        self.eval(v)
    }

    fn apply(cs: &[Curve], v: &mut [f64]) {
        for (x, c) in v.iter_mut().zip(cs) {
            *x = c.eval(*x);
        }
    }

    fn mat(m: &([[f64; 3]; 3], [f64; 3]), v: &mut [f64; 3]) {
        let (a, b) = m;
        let r: [f64; 3] =
            core::array::from_fn(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2] + b[i]);
        *v = r;
    }

    /// Evaluates normalized inputs (0..1) to normalized outputs.
    fn eval(&self, input: &[f64; 3]) -> [f64; 3] {
        let mut v = *input;
        if let Some(m) = self.matrix {
            Self::mat(&(m, [0.0; 3]), &mut v);
        }
        for x in v.iter_mut().take(self.n_in) {
            *x = x.clamp(0.0, 1.0);
        }
        Self::apply(&self.first, &mut v[..self.n_in]);
        if self.mba
            && let Some(m) = &self.mab_matrix
        {
            Self::mat(m, &mut v);
            Self::apply(&self.m, &mut v);
        }
        if let Some(c) = &self.clut {
            let mut o = [0.0; 3];
            let x = v;
            c.eval(&x[..c.n_in], &mut o[..c.n_out]);
            v = o;
        }
        if !self.mba
            && let Some(m) = &self.mab_matrix
        {
            Self::apply(&self.m, &mut v);
            Self::mat(m, &mut v);
        }
        for x in v.iter_mut().take(self.n_out) {
            *x = x.clamp(0.0, 1.0);
        }
        Self::apply(&self.last, &mut v[..self.n_out]);
        v
    }

    /// Normalized PCS of this tag -> XYZ.
    fn decode_pcs(&self, v: [f64; 3]) -> [f64; 3] {
        match self.pcs {
            Pcs::Xyz => v.map(|x| x * 65535.0 / 32768.0),
            Pcs::LabLegacy16 => {
                let e = v.map(|x| x * 65535.0);
                lab_to_xyz([e[0] / 652.8, e[1] / 256.0 - 128.0, e[2] / 256.0 - 128.0])
            }
            Pcs::Lab => lab_to_xyz([v[0] * 100.0, v[1] * 255.0 - 128.0, v[2] * 255.0 - 128.0]),
        }
    }

    /// XYZ -> normalized PCS of this tag.
    fn encode_pcs(&self, xyz: [f64; 3]) -> [f64; 3] {
        match self.pcs {
            Pcs::Xyz => xyz.map(|x| x * 32768.0 / 65535.0),
            Pcs::LabLegacy16 => {
                let l = xyz_to_lab(xyz);
                [
                    l[0] * 652.8 / 65535.0,
                    (l[1] + 128.0) * 256.0 / 65535.0,
                    (l[2] + 128.0) * 256.0 / 65535.0,
                ]
            }
            Pcs::Lab => {
                let l = xyz_to_lab(xyz);
                [l[0] / 100.0, (l[1] + 128.0) / 255.0, (l[2] + 128.0) / 255.0]
            }
        }
    }
}

fn lab_f(t: f64) -> f64 {
    if t > (6.0f64 / 29.0).powi(3) {
        t.cbrt()
    } else {
        t / (3.0 * (6.0f64 / 29.0).powi(2)) + 4.0 / 29.0
    }
}

fn lab_f_inv(t: f64) -> f64 {
    if t > 6.0 / 29.0 {
        t * t * t
    } else {
        3.0 * (6.0f64 / 29.0).powi(2) * (t - 4.0 / 29.0)
    }
}

pub(crate) fn xyz_to_lab(x: [f64; 3]) -> [f64; 3] {
    let f = [
        lab_f(x[0] / D50[0]),
        lab_f(x[1] / D50[1]),
        lab_f(x[2] / D50[2]),
    ];
    [
        116.0 * f[1] - 16.0,
        500.0 * (f[0] - f[1]),
        200.0 * (f[1] - f[2]),
    ]
}

pub(crate) fn lab_to_xyz(l: [f64; 3]) -> [f64; 3] {
    let fy = (l[0] + 16.0) / 116.0;
    let fx = fy + l[1] / 500.0;
    let fz = fy - l[2] / 200.0;
    [
        D50[0] * lab_f_inv(fx),
        D50[1] * lab_f_inv(fy),
        D50[2] * lab_f_inv(fz),
    ]
}

/// One side of a transform.
pub(crate) enum Side {
    Matrix {
        trc: [Curve; 3],
        m: [[f64; 3]; 3],
        inv: [[f64; 3]; 3],
    },
    Gray(Curve),
    Lut(Lut),
}

fn inverse(m: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let [[a, b, c], [d, e, f], [g, h, i]] = m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-12 {
        return Err(Error::InvalidData);
    }
    let k = 1.0 / det;
    Ok([
        [
            (e * i - f * h) * k,
            (c * h - b * i) * k,
            (b * f - c * e) * k,
        ],
        [
            (f * g - d * i) * k,
            (a * i - c * g) * k,
            (c * d - a * f) * k,
        ],
        [
            (d * h - e * g) * k,
            (b * g - a * h) * k,
            (a * e - b * d) * k,
        ],
    ])
}

/// Tables for intent 0 / 1 / 2 with the ICC fallback to 0.
fn pick(intent: u8, t: [&Option<LutWarehouse>; 3]) -> Option<&LutWarehouse> {
    let i = usize::from(intent.min(2));
    t[i].as_ref().or(t[0].as_ref())
}

impl Side {
    /// `input`: device -> PCS side; else PCS -> device. `intent` 0..3 (3 uses the
    /// colorimetric tables).
    pub fn new(p: &ColorProfile, intent: u8, input: bool) -> Result<Side> {
        let lab = p.pcs == DataColorSpace::Lab;
        let ti = if intent == 3 { 1 } else { intent };
        let lut = if input {
            pick(
                ti,
                [
                    &p.lut_a_to_b_perceptual,
                    &p.lut_a_to_b_colorimetric,
                    &p.lut_a_to_b_saturation,
                ],
            )
        } else {
            pick(
                ti,
                [
                    &p.lut_b_to_a_perceptual,
                    &p.lut_b_to_a_colorimetric,
                    &p.lut_b_to_a_saturation,
                ],
            )
        };
        if let Some(w) = lut {
            return Ok(Side::Lut(Lut::new(w, lab, !input)?));
        }
        match p.color_space {
            DataColorSpace::Gray => {
                let t = p.gray_trc.as_ref().ok_or(Error::Unsupported)?;
                Ok(Side::Gray(Curve::new(t)?))
            }
            DataColorSpace::Rgb => {
                let c = |t: &Option<ToneReprCurve>| {
                    t.as_ref().ok_or(Error::Unsupported).and_then(Curve::new)
                };
                let trc = [c(&p.red_trc)?, c(&p.green_trc)?, c(&p.blue_trc)?];
                let (r, g, b) = (p.red_colorant, p.green_colorant, p.blue_colorant);
                let m = [[r.x, g.x, b.x], [r.y, g.y, b.y], [r.z, g.z, b.z]];
                Ok(Side::Matrix {
                    trc,
                    m,
                    inv: inverse(m)?,
                })
            }
            _ => Err(Error::Unsupported),
        }
    }

    /// Device values (gray: `v[0]`) -> XYZ.
    pub fn device_to_xyz(&self, v: [f64; 3]) -> [f64; 3] {
        match self {
            Side::Matrix { trc, m, .. } => {
                let l = [trc[0].eval(v[0]), trc[1].eval(v[1]), trc[2].eval(v[2])];
                core::array::from_fn(|i| m[i][0] * l[0] + m[i][1] * l[1] + m[i][2] * l[2])
            }
            Side::Gray(t) => {
                let y = t.eval(v[0]);
                [D50[0] * y, y, D50[2] * y]
            }
            Side::Lut(l) => {
                let o = l.eval(&v);
                if l.n_out == 1 {
                    let y = l.decode_pcs([o[0], o[0], o[0]]);
                    [D50[0] * y[1], y[1], D50[2] * y[1]]
                } else {
                    l.decode_pcs(o)
                }
            }
        }
    }

    /// XYZ -> device values (gray: `[0]`).
    pub fn xyz_to_device(&self, x: [f64; 3]) -> [f64; 3] {
        match self {
            Side::Matrix { trc, inv, .. } => {
                let l: [f64; 3] = core::array::from_fn(|i| {
                    inv[i][0] * x[0] + inv[i][1] * x[1] + inv[i][2] * x[2]
                });
                core::array::from_fn(|i| {
                    let v = l[i];
                    // Sign-symmetric beyond the curve domain (extended range floats).
                    if v < 0.0 {
                        -trc[i].inv(-v)
                    } else {
                        trc[i].inv(v)
                    }
                })
            }
            Side::Gray(t) => {
                let y = x[1];
                let g = if y < 0.0 { -t.inv(-y) } else { t.inv(y) };
                [g, g, g]
            }
            Side::Lut(l) => l.eval(&l.encode_pcs(x)),
        }
    }
}

/// Media white point (wtpt) or D50.
pub(crate) fn media_white(p: &ColorProfile) -> [f64; 3] {
    p.media_white_point.map_or(D50, |w| [w.x, w.y, w.z])
}
