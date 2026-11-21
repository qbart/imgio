// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of basisu `xuastc_deblock_filter` (`basisu_transcoder.cpp`): smooths the borders of
//! every `fbw` x `fbh` block of an RGBA8 image whose size is a multiple of the block size.
//!
//! Per pixel, by its position inside its block: corners take a 5-tap cross (center weighted
//! twice, divided by 6 in f32 like the C++), top/bottom edges a vertical 3-tap average,
//! left/right edges a horizontal 3-tap average, the interior is copied. Neighbors are read
//! from the unfiltered image with coordinates clamped to the image.

use crate::parallel::for_each_unit;

/// Filters `src` (`width` x `height` RGBA8) into `dst` (same size), rows in parallel.
pub(crate) fn deblock(
    src: &[u8],
    width: usize,
    height: usize,
    fbw: usize,
    fbh: usize,
    dst: &mut [u8],
) {
    if width == 0 || height == 0 || fbw < 3 || fbh < 3 {
        dst[..src.len()].copy_from_slice(src);
        return;
    }
    let at = |x: isize, y: isize| -> [u32; 4] {
        let x = x.clamp(0, width as isize - 1) as usize;
        let y = y.clamp(0, height as isize - 1) as usize;
        let i = (y * width + x) * 4;
        [0, 1, 2, 3].map(|c| u32::from(src[i + c]))
    };
    let row = width * 4;
    for_each_unit(&mut dst[..row * height], row, width * height, |y, out| {
        let ly = y % fbh;
        let y_edge = ly == 0 || ly == fbh - 1;
        let yi = y as isize;
        for (x, o) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let lx = x % fbw;
            let x_edge = lx == 0 || lx == fbw - 1;
            let xi = x as isize;
            let v: [u32; 4] = match (x_edge, y_edge) {
                (true, true) => {
                    let (l, c, r) = (at(xi - 1, yi), at(xi, yi), at(xi + 1, yi));
                    let (u, d) = (at(xi, yi - 1), at(xi, yi + 1));
                    [0, 1, 2, 3].map(|k| {
                        let s = (l[k] + 2 * c[k] + r[k] + u[k] + d[k]) as f32 * (1.0f32 / 6.0f32);
                        ((s + 0.5) as i32).min(255) as u32
                    })
                }
                (false, true) => {
                    let (u, c, d) = (at(xi, yi - 1), at(xi, yi), at(xi, yi + 1));
                    [0, 1, 2, 3].map(|k| (u[k] + c[k] + d[k] + 1) / 3)
                }
                (true, false) => {
                    let (l, c, r) = (at(xi - 1, yi), at(xi, yi), at(xi + 1, yi));
                    [0, 1, 2, 3].map(|k| (l[k] + c[k] + r[k] + 1) / 3)
                }
                (false, false) => at(xi, yi),
            };
            *o = v.map(|c| c as u8);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_is_unchanged() {
        let src = [7u8; 12 * 10 * 4];
        let mut dst = [0u8; 12 * 10 * 4];
        deblock(&src, 12, 10, 6, 5, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn edges_average_across_the_seam() {
        // Two 4x4 blocks side by side: left 0, right 90 (alpha 255).
        let (w, h) = (8, 4);
        let mut src = [0u8; 8 * 4 * 4];
        for y in 0..h {
            for x in 0..w {
                let v = if x < 4 { 0 } else { 90 };
                src[(y * w + x) * 4..][..4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let mut dst = [0u8; 8 * 4 * 4];
        deblock(&src, w, h, 4, 4, &mut dst);
        let px = |x: usize, y: usize| dst[(y * w + x) * 4];
        assert_eq!(px(3, 1), 30); // (0 + 0 + 90 + 1) / 3
        assert_eq!(px(4, 1), 60); // (0 + 90 + 90 + 1) / 3
        assert_eq!(px(1, 1), 0); // interior
        assert_eq!(px(3, 0), 15); // corner: (0 + 0 + 90 + 0 + 0) / 6 = 15
    }
}
