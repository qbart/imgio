//! Texture atlases: packing invariants for every algorithm and option, pixel placement
//! (rotation, trimming, extrusion), duplicates, pages, determinism and the JSON layout.

use imgio::atlas::{
    self, Algorithm, AtlasOptions, MaxRectsHeuristic, PageSize, Placement, SortOrder, Trim,
};
use imgio::{Error, Format, Surface};
use std::error::Error as StdError;

type R<T = ()> = Result<T, Box<dyn StdError>>;

const ALGORITHMS: [Algorithm; 7] = [
    Algorithm::MaxRects(MaxRectsHeuristic::BestShortSideFit),
    Algorithm::MaxRects(MaxRectsHeuristic::BestLongSideFit),
    Algorithm::MaxRects(MaxRectsHeuristic::BestAreaFit),
    Algorithm::MaxRects(MaxRectsHeuristic::BottomLeft),
    Algorithm::MaxRects(MaxRectsHeuristic::ContactPoint),
    Algorithm::Skyline,
    Algorithm::Shelf,
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % u64::from(hi - lo + 1)) as u32
    }
}

/// RGBA8 sprites with unique pixels and a transparent margin of 0..3 pixels.
fn sprites(n: usize, seed: u64, max: u32) -> Vec<(u32, u32, Vec<u8>)> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|i| {
            let (w, h) = (rng.range(1, max), rng.range(1, max));
            let m = rng.range(0, 3);
            let mut px = vec![0u8; (w * h * 4) as usize];
            for y in 0..h {
                for x in 0..w {
                    let o = ((y * w + x) * 4) as usize;
                    let inside = x >= m && y >= m && x + m < w && y + m < h;
                    px[o] = i as u8;
                    px[o + 1] = x as u8;
                    px[o + 2] = y as u8;
                    px[o + 3] = if inside { 255 } else { 0 };
                }
            }
            (w, h, px)
        })
        .collect()
}

fn surfaces(s: &[(u32, u32, Vec<u8>)]) -> R<Vec<Surface<'_>>> {
    Ok(s.iter()
        .map(|(w, h, p)| Surface::new(Format::Rgba8Unorm, *w, *h, p))
        .collect::<Result<_, _>>()?)
}

/// Checks bounds, page constraints and that no two padded / extruded regions overlap.
fn check_layout(pages: &[(u32, u32)], ps: &[Placement], o: &AtlasOptions) {
    let (e, pad, b) = (o.extrude, o.padding, o.border);
    for &(w, h) in pages {
        assert!(w <= o.max_width && h <= o.max_height, "{w}x{h}");
        if o.power_of_two {
            assert!(w.is_power_of_two() && h.is_power_of_two(), "{w}x{h}");
        }
        if o.square {
            assert_eq!(w, h);
        }
        assert_eq!((w % o.multiple_of, h % o.multiple_of), (0, 0));
        if let PageSize::Fixed(fw, fh) = o.size {
            assert_eq!((w, h), (fw, fh));
        }
    }
    let owners: Vec<&Placement> = ps
        .iter()
        .enumerate()
        .filter(|(i, p)| p.alias_of == *i)
        .map(|(_, p)| p)
        .collect();
    for p in &owners {
        let (w, h) = pages[p.page];
        assert!(p.x >= b + e && p.y >= b + e, "{p:?}");
        assert!(
            p.x + p.w + e + b <= w && p.y + p.h + e + b <= h,
            "{p:?} in {w}x{h}"
        );
        let (tw, th) = if p.rotated {
            (p.trim_h, p.trim_w)
        } else {
            (p.trim_w, p.trim_h)
        };
        assert_eq!((p.w, p.h), (tw, th));
    }
    for (i, a) in owners.iter().enumerate() {
        for c in &owners[i + 1..] {
            if a.page != c.page {
                continue;
            }
            let r = |p: &Placement| (p.x - e, p.y - e, p.x + p.w + e + pad, p.y + p.h + e + pad);
            let (a0, a1, a2, a3) = r(a);
            let (c0, c1, c2, c3) = r(c);
            assert!(
                a2 <= c0 || c2 <= a0 || a3 <= c1 || c3 <= a1,
                "{a:?} overlaps {c:?}"
            );
        }
    }
}

/// Every region pixel (and its extrusion) is the right source pixel.
fn check_pixels(atlas: &atlas::Atlas, src: &[(u32, u32, Vec<u8>)], e: u32) {
    for p in atlas.placements.iter() {
        let page = &atlas.pages[p.page];
        let (sw, _, spx) = &src[p.alias_of];
        let e = e as i64;
        for ry in -e..i64::from(p.h) + e {
            for rx in -e..i64::from(p.w) + e {
                let cx = rx.clamp(0, i64::from(p.w) - 1) as u32;
                let cy = ry.clamp(0, i64::from(p.h) - 1) as u32;
                let (sx, sy) = if p.rotated {
                    (cy, p.trim_h - 1 - cx)
                } else {
                    (cx, cy)
                };
                let s = (((p.trim_y + sy) * sw + p.trim_x + sx) * 4) as usize;
                let dx = (i64::from(p.x) + rx) as u32;
                let dy = (i64::from(p.y) + ry) as u32;
                let d = ((dy * page.width + dx) * 4) as usize;
                assert_eq!(&page.data[d..d + 4], &spx[s..s + 4], "{p:?} at {rx},{ry}");
            }
        }
    }
}

#[test]
fn every_algorithm_and_option() -> R {
    let src = sprites(60, 0x1234_5678_9ABC_DEF1, 40);
    let s = surfaces(&src)?;
    let mut checked = 0;
    for alg in ALGORITHMS {
        for (k, (rot, pot, square, mult, pad, border, extrude)) in [
            (false, false, false, 1, 2, 0, 0),
            (true, false, false, 4, 1, 3, 1),
            (true, true, false, 1, 0, 0, 2),
            (false, false, true, 1, 3, 1, 0),
            (true, true, true, 1, 2, 2, 1),
        ]
        .into_iter()
        .enumerate()
        {
            let o = AtlasOptions {
                algorithm: alg,
                sort: [
                    SortOrder::MaxSide,
                    SortOrder::Area,
                    SortOrder::Height,
                    SortOrder::None,
                    SortOrder::Perimeter,
                ][k],
                rotation: rot,
                power_of_two: pot,
                square,
                multiple_of: mult,
                padding: pad,
                border,
                extrude,
                trim: if k % 2 == 1 {
                    Trim::Alpha(0.0)
                } else {
                    Trim::None
                },
                max_width: 1024,
                max_height: 1024,
                ..Default::default()
            };
            let a = atlas::build(&s, &o)?;
            assert_eq!(a.pages.len(), 1);
            let sizes: Vec<(u32, u32)> = a.pages.iter().map(|p| (p.width, p.height)).collect();
            check_layout(&sizes, &a.placements, &o);
            check_pixels(&a, &src, extrude);
            // Same input, same atlas.
            let b = atlas::build(&s, &o)?;
            assert_eq!(&a.pages[0].data[..], &b.pages[0].data[..]);
            assert_eq!(&a.placements[..], &b.placements[..]);
            // Occupancy of the MaxRects / skyline default runs.
            if k == 0 && alg != Algorithm::Shelf {
                let used: u64 = src
                    .iter()
                    .map(|(w, h, _)| u64::from((w + 2) * (h + 2)))
                    .sum();
                let area = u64::from(sizes[0].0 * sizes[0].1);
                assert!(used * 100 / area >= 70, "{alg:?}: {used} / {area}");
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 35);
    Ok(())
}

#[test]
fn trimming_and_duplicates() -> R {
    let mut src = sprites(10, 77, 24);
    // Duplicates of sprites 0 and 3, and a fully transparent sprite.
    let d0 = src[0].clone();
    let d3 = src[3].clone();
    src.push(d0);
    src.push(d3);
    src.push((5, 4, vec![0u8; 5 * 4 * 4]));
    let s = surfaces(&src)?;
    let o = AtlasOptions {
        trim: Trim::Alpha(0.0),
        dedupe: true,
        rotation: true,
        ..Default::default()
    };
    let a = atlas::build(&s, &o)?;
    let sizes: Vec<(u32, u32)> = a.pages.iter().map(|p| (p.width, p.height)).collect();
    check_layout(&sizes, &a.placements, &o);
    check_pixels(&a, &src, 0);
    assert_eq!(a.placements[10].alias_of, 0);
    assert_eq!(a.placements[11].alias_of, 3);
    let (p0, p10) = (a.placements[0], a.placements[10]);
    assert_eq!((p0.x, p0.y, p0.page), (p10.x, p10.y, p10.page));
    let t = a.placements[12];
    assert_eq!((t.trim_w, t.trim_h, t.source_w, t.source_h), (1, 1, 5, 4));
    // Trim rectangles are the opaque area.
    for (i, (w, h, px)) in src.iter().enumerate().take(10) {
        let p = a.placements[i];
        assert_eq!((p.source_w, p.source_h), (*w, *h));
        for y in 0..*h {
            for x in 0..*w {
                let opaque = px[((y * w + x) * 4 + 3) as usize] > 0;
                let inside = x >= p.trim_x
                    && y >= p.trim_y
                    && x < p.trim_x + p.trim_w
                    && y < p.trim_y + p.trim_h;
                assert!(!opaque || inside);
            }
        }
    }
    // Without dedupe every sprite has its own region.
    let o2 = AtlasOptions { dedupe: false, ..o };
    let b = atlas::build(&s, &o2)?;
    assert!(
        b.placements
            .iter()
            .enumerate()
            .all(|(i, p)| p.alias_of == i)
    );
    Ok(())
}

#[test]
fn pages_and_errors() -> R {
    let src = sprites(40, 99, 60);
    let s = surfaces(&src)?;
    let mut o = AtlasOptions {
        max_width: 128,
        max_height: 128,
        ..Default::default()
    };
    assert_eq!(atlas::build(&s, &o).err(), Some(Error::InvalidArgument));
    o.multipage = true;
    let a = atlas::build(&s, &o)?;
    assert!(a.pages.len() > 1);
    let sizes: Vec<(u32, u32)> = a.pages.iter().map(|p| (p.width, p.height)).collect();
    check_layout(&sizes, &a.placements, &o);
    check_pixels(&a, &src, 0);
    // Fixed size pages.
    o.size = PageSize::Fixed(100, 90);
    let a = atlas::build(&s, &o)?;
    let sizes: Vec<(u32, u32)> = a.pages.iter().map(|p| (p.width, p.height)).collect();
    assert!(sizes.iter().all(|&s| s == (100, 90)));
    check_layout(&sizes, &a.placements, &o);
    // A sprite larger than the page.
    let big = vec![0u8; 200 * 10 * 4];
    let b = Surface::new(Format::Rgba8Unorm, 200, 10, &big)?;
    assert_eq!(atlas::build(&[b], &o).err(), Some(Error::InvalidArgument));
    // Rotation lets it fit a tall page.
    let o = AtlasOptions {
        max_width: 16,
        max_height: 256,
        rotation: true,
        padding: 0,
        ..Default::default()
    };
    let a = atlas::build(&[b], &o)?;
    assert!(a.placements[0].rotated);
    assert_eq!((a.pages[0].width, a.pages[0].height), (10, 200));
    // Bad options, compressed input / output, empty sizes.
    let bad = AtlasOptions {
        multiple_of: 0,
        ..Default::default()
    };
    assert_eq!(atlas::build(&[b], &bad).err(), Some(Error::InvalidArgument));
    let bc = Surface::new(Format::Bc1Unorm, 4, 4, &big[..8])?;
    assert_eq!(
        atlas::build(&[bc], &AtlasOptions::default()).err(),
        Some(Error::Unsupported)
    );
    let o = AtlasOptions {
        format: Format::Bc7Unorm,
        ..Default::default()
    };
    assert_eq!(atlas::build(&[b], &o).err(), Some(Error::Unsupported));
    assert_eq!(
        atlas::layout(&[(0, 4)], &AtlasOptions::default()).err(),
        Some(Error::InvalidArgument)
    );
    Ok(())
}

#[test]
fn layout_formats_and_json() -> R {
    // Exact small layout: three rectangles, no padding, MaxRects BSSF.
    let o = AtlasOptions {
        padding: 0,
        ..Default::default()
    };
    let l = atlas::layout(&[(32, 16), (16, 16), (16, 16)], &o)?;
    assert_eq!(&l.pages[..], &[(32, 32)]);
    let pos: Vec<(u32, u32)> = l.placements.iter().map(|p| (p.x, p.y)).collect();
    assert_eq!(pos, [(0, 0), (0, 16), (16, 16)]);
    let json = l.to_json(&["wide", "a\"b", "c"], "atlas{}.png")?;
    let text = std::str::from_utf8(&json)?;
    assert!(text.starts_with("{\"frames\":{\n\"wide\":{\"frame\":{\"x\":0,\"y\":0,\"w\":32,\"h\":16},\"rotated\":false,\"trimmed\":false,"));
    assert!(text.contains("\"a\\\"b\":{\"frame\":{\"x\":0,\"y\":16,\"w\":16,\"h\":16}"));
    assert!(text.contains("\"pages\":[{\"image\":\"atlas0.png\",\"size\":{\"w\":32,\"h\":32}}]"));
    assert_eq!(
        l.to_json(&["x"], "a.png").err(),
        Some(Error::InvalidArgument)
    );
    // Mixed input formats into a 16-bit float atlas; UVs.
    let g = [10u8, 20, 30, 40];
    let rgb = [1u8, 2, 3, 4, 5, 6];
    let sg = Surface::new(Format::R8Unorm, 2, 2, &g)?;
    let sr = Surface::new(Format::Rgb8Unorm, 2, 1, &rgb)?;
    let o = AtlasOptions {
        format: Format::Rgba16Float,
        padding: 0,
        ..Default::default()
    };
    let a = atlas::build(&[sg, sr], &o)?;
    assert_eq!(a.pages[0].format, Format::Rgba16Float);
    let uv = a.uv(0).ok_or("uv")?;
    let p = a.placements[0];
    let (w, h) = (a.pages[0].width as f32, a.pages[0].height as f32);
    assert_eq!(
        uv,
        [
            p.x as f32 / w,
            p.y as f32 / h,
            (p.x + 2) as f32 / w,
            (p.y + 2) as f32 / h
        ]
    );
    assert!(a.uv(5).is_none());
    let json = a.to_json(&["g", "rgb"], "page.png")?;
    assert!(std::str::from_utf8(&json)?.contains("\"image\":\"page.png\""));
    Ok(())
}
