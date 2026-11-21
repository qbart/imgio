// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal (encoder/basisu_pvrtc1_4.cpp/.h,
// encoder/basisu_gpu_texture.cpp) and modified; see NOTICE.md.
//! PVRTC decoding.
//!
//! PVRTC1 4bpp (after basisu `pvrtc4_image::get_pixel`): blocks are stored in twiddled
//! (Morton) order, block counts must be powers of two. Every pixel bilinearly blends the
//! RGBA5554 endpoints of the 2x2 blocks around it (wrapping at the edges, 16x precision,
//! then expanded to 8 bits) and applies the block's 2-bit modulation: weights 0, 3/8, 5/8, 1,
//! or with the block's punch-through flag 0, 1/2, 1/2 with alpha 0, 1.
//!
//! PVRTC2 4bpp (after basisu `unpack_pvrtc2`): only hard-transition, non-interpolated blocks
//! (modulation flag 0; what the Basis transcoder emits) are decoded, others are unsupported.

use crate::parallel::try_for_each_unit;
use crate::{Error, Result};

/// basisu `pvrtc4_swizzle_uv`: twiddled index of block (`x`, `y`) in a `w` x `h` block grid.
pub(crate) fn swizzle(w: usize, h: usize, x: usize, y: usize) -> usize {
    let (min_d, max_v) = if h < w { (h, x) } else { (w, y) };
    let (mut shift, mut out) = (0, 0);
    let mut bit = 1;
    while bit < min_d {
        if y & bit != 0 {
            out |= 1 << (2 * shift);
        }
        if x & bit != 0 {
            out |= 2 << (2 * shift);
        }
        bit <<= 1;
        shift += 1;
    }
    out | ((max_v >> shift) << (2 * shift))
}

/// basisu `pvrtc4_block::get_endpoint_5554`: RGB 5 bits, A 4 bits.
fn endpoint_5554(endpoints: u32, index: u32) -> [i32; 4] {
    let p = (endpoints >> (index * 16)) as i32;
    let (r, g, b, a);
    if p & 0x8000 != 0 {
        r = (p >> 10) & 31;
        g = (p >> 5) & 31;
        b = if index == 0 {
            let b = (p >> 1) & 15;
            (b << 1) | (b >> 3)
        } else {
            p & 31
        };
        a = 15;
    } else {
        a = ((p >> 12) & 7) << 1;
        let (r4, g4) = ((p >> 8) & 15, (p >> 4) & 15);
        r = (r4 << 1) | (r4 >> 3);
        g = (g4 << 1) | (g4 >> 3);
        b = if index == 0 {
            let b = (p >> 1) & 7;
            (b << 2) | (b >> 1)
        } else {
            let b = p & 15;
            (b << 1) | (b >> 3)
        };
    }
    [r, g, b, a]
}

/// basisu `pvrtc4_image::interpolate`: bilinear blend of 4 endpoints for pixel (`x`, `y`).
fn interpolate(x: usize, y: usize, p: [i32; 4], q: [i32; 4], r: [i32; 4], s: [i32; 4]) -> [i32; 4] {
    const W: [i32; 4] = [2, 3, 0, 1];
    let (u, v) = (W[x & 3], W[y & 3]);
    core::array::from_fn(|c| {
        let t = p[c] * 4 + u * (q[c] - p[c]);
        let b = r[c] * 4 + u * (s[c] - r[c]);
        let mut o = t * 4 + v * (b - t);
        if c < 3 {
            o >>= 1;
            o += o >> 5;
        } else {
            o += o >> 4;
        }
        o
    })
}

/// Decodes a PVRTC1 4bpp surface (`width` x `height`, blocks in twiddled order) into RGBA8
/// rows in `dst`. Errors: `InvalidArgument` for block counts that are not powers of two.
pub(crate) fn decode_pvrtc1(src: &[u8], width: u32, height: u32, dst: &mut [u8]) -> Result<()> {
    let (bw, bh) = (width.div_ceil(4) as usize, height.div_ceil(4) as usize);
    if !bw.is_power_of_two() || !bh.is_power_of_two() {
        return Err(Error::InvalidArgument);
    }
    let (w, h) = (width as usize, height as usize);
    let block = |x: usize, y: usize| -> (u32, u32) {
        let at = swizzle(bw, bh, x, y) * 8;
        let b = &src[at..at + 8];
        (
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        )
    };
    let row = w * 4;
    try_for_each_unit(&mut dst[..row * h], row, w * h, |y, out| {
        let y0 = (((y as isize - 2) >> 2).rem_euclid(bh as isize)) as usize;
        let y1 = (y0 + 1) % bh;
        for (x, o) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let x0 = (((x as isize - 2) >> 2).rem_euclid(bw as isize)) as usize;
            let x1 = (x0 + 1) % bw;
            let e = [
                block(x0, y0).1,
                block(x1, y0).1,
                block(x0, y1).1,
                block(x1, y1).1,
            ];
            let ends = |i: u32| {
                interpolate(
                    x,
                    y,
                    endpoint_5554(e[0], i),
                    endpoint_5554(e[1], i),
                    endpoint_5554(e[2], i),
                    endpoint_5554(e[3], i),
                )
            };
            let (modulation, endpoints) = block(x >> 2, y >> 2);
            let m = (modulation >> (((y & 3) * 4 + (x & 3)) * 2)) & 3;
            let c = match m {
                0 => ends(0),
                3 => ends(1),
                _ => {
                    let (l, h) = (ends(0), ends(1));
                    if endpoints & 1 != 0 {
                        let a = if m == 2 { 0 } else { (l[3] + h[3]) / 2 };
                        [(l[0] + h[0]) / 2, (l[1] + h[1]) / 2, (l[2] + h[2]) / 2, a]
                    } else if m == 2 {
                        core::array::from_fn(|k| (l[k] * 3 + h[k] * 5) / 8)
                    } else {
                        core::array::from_fn(|k| (l[k] * 5 + h[k] * 3) / 8)
                    }
                }
            };
            *o = c.map(|v| v as u8);
        }
        Ok(())
    })
}

/// Decodes one PVRTC2 4bpp block into 4x4 RGBA8 pixels (hard-transition blocks only).
pub(crate) fn decode_pvrtc2_block(block: &[u8], out: &mut [u8]) -> Result<()> {
    let b = block.first_chunk::<8>().ok_or(Error::InvalidData)?;
    let modulation = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let d = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let bits = |at: u32, n: u32| (d >> at) & ((1 << n) - 1);
    if bits(15, 1) == 0 || bits(0, 1) != 0 {
        return Err(Error::Unsupported);
    }
    let x5 = |v: u32| (v << 3) | (v >> 2);
    let (c0, c3) = if bits(31, 1) != 0 {
        // Opaque: color A = RGB554, color B = RGB555.
        let ba = bits(1, 4);
        let a = [bits(10, 5), bits(5, 5), (ba << 1) | (ba >> 3)].map(x5);
        let b = [bits(26, 5), bits(21, 5), bits(16, 5)].map(x5);
        ([a[0], a[1], a[2], 255], [b[0], b[1], b[2], 255])
    } else {
        // Translucent: color A = ARGB3443, color B = ARGB3444 (to RGBA5554, then 8888).
        let x4 = |v: u32| (v << 1) | (v >> 3);
        let ba = bits(1, 3);
        let a = [
            x4(bits(8, 4)),
            x4(bits(4, 4)),
            (ba << 2) | (ba >> 1),
            bits(12, 3) << 1,
        ];
        let b = [
            x4(bits(24, 4)),
            x4(bits(20, 4)),
            x4(bits(16, 4)),
            (bits(28, 3) << 1) | 1,
        ];
        let e = |c: [u32; 4]| [x5(c[0]), x5(c[1]), x5(c[2]), (c[3] << 4) | c[3]];
        (e(a), e(b))
    };
    let pal = [
        c0,
        core::array::from_fn::<u32, 4, _>(|k| (c0[k] * 5 + c3[k] * 3) / 8),
        core::array::from_fn::<u32, 4, _>(|k| (c0[k] * 3 + c3[k] * 5) / 8),
        c3,
    ];
    for (i, o) in out[..64].as_chunks_mut::<4>().0.iter_mut().enumerate() {
        *o = pal[((modulation >> (i * 2)) & 3) as usize].map(|v| v as u8);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swizzle_matches_morton() {
        assert_eq!(swizzle(4, 4, 1, 0), 2);
        assert_eq!(swizzle(4, 4, 0, 1), 1);
        assert_eq!(swizzle(4, 4, 3, 3), 15);
        // Rectangular: remaining high bits of the larger dimension go on top.
        assert_eq!(swizzle(4, 2, 2, 0), 4);
        assert_eq!(swizzle(2, 4, 0, 2), 4);
    }

    #[test]
    fn solid_opaque_surface() {
        // 8x8, every block: opaque white endpoints, modulation 0.
        let blk = [0, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff];
        let src: Vec<u8> = blk.iter().copied().cycle().take(4 * 8).collect();
        let mut dst = [0u8; 8 * 8 * 4];
        assert!(decode_pvrtc1(&src, 8, 8, &mut dst).is_ok());
        assert!(dst.iter().all(|&v| v == 255));
        assert!(decode_pvrtc1(&src, 12, 8, &mut dst).is_err());
    }
}
