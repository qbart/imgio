//! Rectangle bin packers: MaxRects (Jylänki 2010, with its five heuristics), skyline
//! bottom-left and shelf best-height-fit. Deterministic: ties keep the first candidate in scan
//! order.

use crate::{Error, Result};
use vas::Vector;
use vas::storage::Dynamic;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    fn right(&self) -> u32 {
        self.x + self.w
    }
    fn bottom(&self) -> u32 {
        self.y + self.h
    }
    fn contains(&self, o: &Rect) -> bool {
        o.x >= self.x && o.y >= self.y && o.right() <= self.right() && o.bottom() <= self.bottom()
    }
    fn intersects(&self, o: &Rect) -> bool {
        o.x < self.right() && o.right() > self.x && o.y < self.bottom() && o.bottom() > self.y
    }
}

/// MaxRects free-rectangle choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaxRectsHeuristic {
    /// Smallest leftover on the shorter side (good general default).
    #[default]
    BestShortSideFit,
    /// Smallest leftover on the longer side.
    BestLongSideFit,
    /// Smallest free rectangle.
    BestAreaFit,
    /// Lowest, then leftmost position (Tetris-like).
    BottomLeft,
    /// Most edge contact with placed rectangles and the bin border.
    ContactPoint,
}

/// Packing algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// MaxRects: densest, slowest.
    MaxRects(MaxRectsHeuristic),
    /// Skyline bottom-left: fast, good for similar heights.
    Skyline,
    /// Shelves (rows), best height fit: fastest, simple layouts.
    Shelf,
}

impl Default for Algorithm {
    fn default() -> Self {
        Algorithm::MaxRects(MaxRectsHeuristic::BestShortSideFit)
    }
}

fn push<T>(v: &mut Vector<T, Dynamic>, x: T) -> Result<()> {
    v.push(x).map_err(|_| Error::OutOfMemory)
}

/// A bin being filled.
pub(crate) enum Bin {
    MaxRects {
        h: MaxRectsHeuristic,
        bw: u32,
        bh: u32,
        free: Vector<Rect, Dynamic>,
        used: Vector<Rect, Dynamic>,
    },
    Skyline {
        bw: u32,
        bh: u32,
        /// (x, y, width) segments, left to right.
        line: Vector<(u32, u32, u32), Dynamic>,
    },
    Shelf {
        bw: u32,
        bh: u32,
        /// (y, height, used width) shelves.
        shelves: Vector<(u32, u32, u32), Dynamic>,
        top: u32,
    },
}

impl Bin {
    pub fn new(alg: Algorithm, bw: u32, bh: u32) -> Result<Bin> {
        Ok(match alg {
            Algorithm::MaxRects(h) => {
                let mut free = Vector::default();
                push(
                    &mut free,
                    Rect {
                        x: 0,
                        y: 0,
                        w: bw,
                        h: bh,
                    },
                )?;
                Bin::MaxRects {
                    h,
                    bw,
                    bh,
                    free,
                    used: Vector::default(),
                }
            }
            Algorithm::Skyline => {
                let mut line = Vector::default();
                push(&mut line, (0, 0, bw))?;
                Bin::Skyline { bw, bh, line }
            }
            Algorithm::Shelf => Bin::Shelf {
                bw,
                bh,
                shelves: Vector::default(),
                top: 0,
            },
        })
    }

    /// Places a `w` x `h` rectangle (also tried as `h` x `w` when `rotate`); returns the
    /// position and whether it was rotated.
    pub fn insert(&mut self, w: u32, h: u32, rotate: bool) -> Result<Option<(u32, u32, bool)>> {
        match self {
            Bin::MaxRects {
                h: heur,
                bw,
                bh,
                free,
                used,
            } => maxrects_insert(*heur, *bw, *bh, free, used, w, h, rotate),
            Bin::Skyline { bw, bh, line } => skyline_insert(*bw, *bh, line, w, h, rotate),
            Bin::Shelf {
                bw,
                bh,
                shelves,
                top,
            } => shelf_insert(*bw, *bh, shelves, top, w, h, rotate),
        }
    }
}

fn common_interval(a1: u32, a2: u32, b1: u32, b2: u32) -> u32 {
    if a2 < b1 || b2 < a1 {
        0
    } else {
        a2.min(b2) - a1.max(b1)
    }
}

fn contact_score(bw: u32, bh: u32, used: &[Rect], x: u32, y: u32, w: u32, h: u32) -> u64 {
    let mut s = 0u64;
    if x == 0 || x + w == bw {
        s += u64::from(h);
    }
    if y == 0 || y + h == bh {
        s += u64::from(w);
    }
    for u in used {
        if u.x == x + w || u.right() == x {
            s += u64::from(common_interval(u.y, u.bottom(), y, y + h));
        }
        if u.y == y + h || u.bottom() == y {
            s += u64::from(common_interval(u.x, u.right(), x, x + w));
        }
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn maxrects_insert(
    heur: MaxRectsHeuristic,
    bw: u32,
    bh: u32,
    free: &mut Vector<Rect, Dynamic>,
    used: &mut Vector<Rect, Dynamic>,
    w: u32,
    h: u32,
    rotate: bool,
) -> Result<Option<(u32, u32, bool)>> {
    // Score: (primary, secondary), lower is better.
    let mut best: Option<((i64, i64), Rect, bool)> = None;
    for f in free.iter() {
        for (rw, rh, rot) in [(w, h, false), (h, w, true)] {
            if (rot && (!rotate || w == h)) || rw > f.w || rh > f.h {
                continue;
            }
            let lw = i64::from(f.w - rw);
            let lh = i64::from(f.h - rh);
            let score = match heur {
                MaxRectsHeuristic::BestShortSideFit => (lw.min(lh), lw.max(lh)),
                MaxRectsHeuristic::BestLongSideFit => (lw.max(lh), lw.min(lh)),
                MaxRectsHeuristic::BestAreaFit => (
                    i64::from(f.w) * i64::from(f.h) - i64::from(rw) * i64::from(rh),
                    lw.min(lh),
                ),
                MaxRectsHeuristic::BottomLeft => (i64::from(f.y + rh), i64::from(f.x)),
                MaxRectsHeuristic::ContactPoint => {
                    (-(contact_score(bw, bh, used, f.x, f.y, rw, rh) as i64), 0)
                }
            };
            if best.as_ref().is_none_or(|b| score < b.0) {
                best = Some((
                    score,
                    Rect {
                        x: f.x,
                        y: f.y,
                        w: rw,
                        h: rh,
                    },
                    rot,
                ));
            }
        }
    }
    let Some((_, r, rot)) = best else {
        return Ok(None);
    };
    // Split every free rectangle the new one overlaps.
    let mut added = Vector::<Rect, Dynamic>::default();
    let mut i = 0;
    while i < free.len() {
        let f = free[i];
        if !f.intersects(&r) {
            i += 1;
            continue;
        }
        if r.x > f.x {
            push(
                &mut added,
                Rect {
                    x: f.x,
                    y: f.y,
                    w: r.x - f.x,
                    h: f.h,
                },
            )?;
        }
        if r.right() < f.right() {
            push(
                &mut added,
                Rect {
                    x: r.right(),
                    y: f.y,
                    w: f.right() - r.right(),
                    h: f.h,
                },
            )?;
        }
        if r.y > f.y {
            push(
                &mut added,
                Rect {
                    x: f.x,
                    y: f.y,
                    w: f.w,
                    h: r.y - f.y,
                },
            )?;
        }
        if r.bottom() < f.bottom() {
            push(
                &mut added,
                Rect {
                    x: f.x,
                    y: r.bottom(),
                    w: f.w,
                    h: f.bottom() - r.bottom(),
                },
            )?;
        }
        free.remove(i);
    }
    let old = free.len();
    for a in added.iter() {
        push(free, *a)?;
    }
    // Prune free rectangles contained in another (keep the first of equal ones). The old
    // rectangles never contain each other, so only pairs with a new one are checked.
    let n = free.len();
    let mut keep = Vector::<bool, Dynamic>::default();
    keep.resize(n, true).map_err(|_| Error::OutOfMemory)?;
    for a in 0..n {
        if !keep[a] {
            continue;
        }
        let from = if a < old { old } else { 0 };
        for b in from..n {
            if a != b && keep[b] && free[b].contains(&free[a]) && (free[a] != free[b] || b < a) {
                keep[a] = false;
                break;
            }
        }
    }
    let mut k = 0;
    free.retain(|_| {
        let v = keep[k];
        k += 1;
        v
    });
    push(used, r)?;
    Ok(Some((r.x, r.y, rot)))
}

fn skyline_insert(
    bw: u32,
    bh: u32,
    line: &mut Vector<(u32, u32, u32), Dynamic>,
    w: u32,
    h: u32,
    rotate: bool,
) -> Result<Option<(u32, u32, bool)>> {
    // Bottom-left: lowest top edge, then narrowest segment width, then leftmost.
    // (score, segment, x, y, width, rotated)
    type Candidate = ((u32, u32), usize, u32, u32, u32, bool);
    let mut best: Option<Candidate> = None;
    for (rw, rh, rot) in [(w, h, false), (h, w, true)] {
        if rot && (!rotate || w == h) {
            continue;
        }
        for i in 0..line.len() {
            let x = line[i].0;
            if x + rw > bw {
                break;
            }
            // Height where the rectangle rests over segments i..
            let (mut y, mut left, mut j) = (0u32, rw, i);
            while left > 0 {
                let Some(&(_, sy, sw)) = line.get(j) else {
                    break;
                };
                y = y.max(sy);
                left = left.saturating_sub(sw);
                j += 1;
            }
            if left > 0 || y + rh > bh {
                continue;
            }
            let score = (y + rh, line[i].2);
            if best.as_ref().is_none_or(|b| score < b.0) {
                best = Some((score, i, x, y, rw, rot));
            }
        }
    }
    let Some((_, i, x, y, rw, rot)) = best else {
        return Ok(None);
    };
    let rh = if rot { w } else { h };
    line.insert(i, (x, y + rh, rw))
        .map_err(|_| Error::OutOfMemory)?;
    // Shrink / remove the segments now covered.
    let end = x + rw;
    let j = i + 1;
    while j < line.len() {
        let (sx, sy, sw) = line[j];
        if sx >= end {
            break;
        }
        if sx + sw <= end {
            line.remove(j);
        } else {
            line[j] = (end, sy, sx + sw - end);
            break;
        }
    }
    // Merge neighbours of equal height.
    let mut j = 0;
    while j + 1 < line.len() {
        if line[j].1 == line[j + 1].1 {
            line[j].2 += line[j + 1].2;
            line.remove(j + 1);
        } else {
            j += 1;
        }
    }
    Ok(Some((x, y, rot)))
}

fn shelf_insert(
    bw: u32,
    bh: u32,
    shelves: &mut Vector<(u32, u32, u32), Dynamic>,
    top: &mut u32,
    w: u32,
    h: u32,
    rotate: bool,
) -> Result<Option<(u32, u32, bool)>> {
    // Best height fit over existing shelves; a rotated fit only if it wastes less height.
    let mut best: Option<(u32, usize, u32, bool)> = None;
    for (rw, rh, rot) in [(w, h, false), (h, w, true)] {
        if rot && (!rotate || w == h) {
            continue;
        }
        for (i, &(_, sh, used)) in shelves.iter().enumerate() {
            if rh <= sh && used + rw <= bw {
                let waste = sh - rh;
                if best.as_ref().is_none_or(|b| waste < b.0) {
                    best = Some((waste, i, rw, rot));
                }
            }
        }
    }
    if let Some((_, i, rw, rot)) = best {
        let s = &mut shelves[i];
        let x = s.2;
        s.2 += rw;
        return Ok(Some((x, s.0, rot)));
    }
    // New shelf: the orientation with the smaller height (rotation lays tall items flat).
    let (rw, rh, rot) = if rotate && h > w {
        (h, w, true)
    } else {
        (w, h, false)
    };
    if rw > bw || *top + rh > bh {
        if rot && w <= bw && *top + h <= bh {
            push(shelves, (*top, h, w))?;
            let y = *top;
            *top += h;
            return Ok(Some((0, y, false)));
        }
        return Ok(None);
    }
    push(shelves, (*top, rh, rw))?;
    let y = *top;
    *top += rh;
    Ok(Some((0, y, rot)))
}
