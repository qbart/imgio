// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of basisu `encode_pvrtc1` (`basisu_transcoder.cpp`, XUASTC section): PVRTC1 4bpp from
//! an RGBA8 image. Endpoints are the per-4x4-block min/max colors (floor/ceil quantized);
//! modulation is then fitted against the bilinearly upsampled endpoints by the shared
//! `ll::pvrtc::fixup_pvrtc1_4_modulation`. Power-of-two block counts only.

use super::ll::pvrtc::{
    fixup_pvrtc1_4_modulation, get_endpoint_l8, get_opaque_endpoint_l0, get_opaque_endpoint_l1,
    set_endpoint_ceil, set_endpoint_floor, set_opaque_endpoint_ceil, set_opaque_endpoint_floor,
};
use crate::{Error, Result};
use vas::Vector;
use vas::storage::Dynamic;

/// The 4x4 block at block (`bx`, `by`) of `img` (`width` x `height` RGBA8), clamped reads.
fn block(img: &[u8], width: usize, height: usize, bx: usize, by: usize) -> [[u8; 4]; 16] {
    let mut out = [[0u8; 4]; 16];
    for (i, o) in out.iter_mut().enumerate() {
        let x = (bx * 4 + (i & 3)).min(width - 1);
        let y = (by * 4 + (i >> 2)).min(height - 1);
        let at = (y * width + x) * 4;
        o.copy_from_slice(&img[at..at + 4]);
    }
    out
}

/// Encodes `img` (`width` x `height` RGBA8, at least 1x1) to PVRTC1 4bpp RGB or RGBA with
/// `blocks_x` x `blocks_y` blocks into `dst` (8 bytes per block, twiddled order).
pub(crate) fn encode_pvrtc1(
    img: &[u8],
    width: usize,
    height: usize,
    rgba: bool,
    blocks_x: u32,
    blocks_y: u32,
    dst: &mut [u8],
) -> Result<()> {
    if !blocks_x.is_power_of_two() || !blocks_y.is_power_of_two() || width == 0 || height == 0 {
        return Err(Error::InvalidArgument);
    }
    let (bxs, bys) = (blocks_x as usize, blocks_y as usize);
    let mut endpoints = Vector::<u32, Dynamic>::default();
    endpoints
        .reserve(bxs * bys)
        .map_err(|_| Error::OutOfMemory)?;
    for by in 0..bys {
        for bx in 0..bxs {
            let px = block(img, width, height, bx, by);
            let (mut low, mut high) = ([255u8; 4], [0u8; 4]);
            for p in &px {
                for c in 0..4 {
                    low[c] = low[c].min(p[c]);
                    high[c] = high[c].max(p[c]);
                }
            }
            let mut e = 0u32;
            if rgba {
                set_endpoint_floor(&mut e, 0, low);
                set_endpoint_ceil(&mut e, 1, high);
            } else {
                set_opaque_endpoint_floor(&mut e, 0, low);
                set_opaque_endpoint_ceil(&mut e, 1, high);
            }
            endpoints.push(e).map_err(|_| Error::OutOfMemory)?;
        }
    }
    let mut cache: Option<(usize, [[u8; 4]; 16])> = None;
    let pixel = |bi: usize, lx: u32, ly: u32| -> Option<i32> {
        if cache.as_ref().map(|c| c.0) != Some(bi) {
            cache = Some((bi, block(img, width, height, bi % bxs, bi / bxs)));
        }
        let (_, px) = cache.as_ref()?;
        let p = px[(ly * 4 + lx) as usize].map(i32::from);
        Some(if rgba {
            16 * (p[0] + p[1] + p[2] + p[3])
        } else {
            (p[0] + p[1] + p[2]) * 16
        })
    };
    let ok = if rgba {
        let lum = |e: u32, idx: u32| get_endpoint_l8(e, idx) as i32;
        fixup_pvrtc1_4_modulation(&endpoints, dst, blocks_x, blocks_y, lum, pixel)
    } else {
        let lum = |e: u32, idx: u32| {
            let l = if idx == 0 {
                get_opaque_endpoint_l0(e)
            } else {
                get_opaque_endpoint_l1(e)
            };
            (l * 255 / 31) as i32
        };
        fixup_pvrtc1_4_modulation(&endpoints, dst, blocks_x, blocks_y, lum, pixel)
    };
    if ok { Ok(()) } else { Err(Error::InvalidData) }
}
