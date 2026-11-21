// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal (encoder/basisu_gpu_texture.cpp) and modified; see
// NOTICE.md.
//! ATC (AMD/Qualcomm ATI texture compression) decoding, after basisu `unpack_atc`.
//!
//! Color block (8 bytes): color0 = mode bit 15 + RGB555, color1 = RGB565, 2-bit indices.
//! Mode 0: c0, 5/8 c0 + 3/8 c1, 3/8 c0 + 5/8 c1, c1. Mode 1: black, c0 - c1/4, c0, c1.
//! ATC RGBA (interpolated alpha) is a BC4 alpha block followed by a color block.

use crate::Format;

/// Decodes an 8-byte ATC color block into 16 RGBA8 pixels (alpha 255).
fn color(b: &[u8; 8], out: &mut [[u8; 4]; 16]) {
    let c0 = u16::from_le_bytes([b[0], b[1]]) as i32;
    let c1 = u16::from_le_bytes([b[2], b[3]]) as i32;
    let sels = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let x5 = |v: i32| (v << 3) | (v >> 2);
    let a = [x5((c0 >> 10) & 31), x5((c0 >> 5) & 31), x5(c0 & 31)];
    let g6 = (c1 >> 5) & 63;
    let d = [x5((c1 >> 11) & 31), (g6 << 2) | (g6 >> 4), x5(c1 & 31)];
    let mut c = [[0i32; 3]; 4];
    c[3] = d;
    if c0 & 0x8000 != 0 {
        c[1] = [0, 1, 2].map(|k| (a[k] - (d[k] >> 2)).max(0));
        c[2] = a;
    } else {
        c[0] = a;
        c[1] = [0, 1, 2].map(|k| (a[k] * 5 + d[k] * 3) >> 3);
        c[2] = [0, 1, 2].map(|k| (a[k] * 3 + d[k] * 5) >> 3);
    }
    for (i, o) in out.iter_mut().enumerate() {
        let s = c[((sels >> (i * 2)) & 3) as usize];
        *o = [s[0] as u8, s[1] as u8, s[2] as u8, 255];
    }
}

/// Decodes one ATC RGB (8 bytes) or ATC RGBA (16 bytes) block into 4x4 RGBA8 pixels.
pub(crate) fn decode_block(format: Format, block: &[u8], out: &mut [u8]) {
    let mut px = [[0u8; 4]; 16];
    let (alpha, rgb) = if format == Format::AtcRgba {
        block.split_at(8)
    } else {
        block.split_at(0)
    };
    if let Some(rgb) = rgb.first_chunk::<8>() {
        color(rgb, &mut px);
    }
    if let Some(alpha) = alpha.first_chunk::<8>() {
        let mut a = [0u8; 16];
        crate::bc::bc1::bc4u(alpha, &mut a);
        for (p, a) in px.iter_mut().zip(a) {
            p[3] = a;
        }
    }
    out[..64].copy_from_slice(px.as_flattened());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes() {
        // Mode 0: c0 = white (555), c1 = black, selectors 0..3 in the first row.
        let blk = [0xff, 0x7f, 0, 0, 0b1110_0100, 0, 0, 0];
        let mut out = [0u8; 64];
        decode_block(Format::AtcRgb, &blk, &mut out);
        assert_eq!(
            &out[..16],
            &[
                255, 255, 255, 255, 159, 159, 159, 255, 95, 95, 95, 255, 0, 0, 0, 255
            ]
        );
        // Mode 1: selector 0 is black, 2 is c0.
        let blk = [0xff, 0xff, 0, 0, 0b0000_1000, 0, 0, 0];
        decode_block(Format::AtcRgb, &blk, &mut out);
        assert_eq!(&out[..8], &[0, 0, 0, 255, 255, 255, 255, 255]);
    }
}
