// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal (encoder/basisu_gpu_texture.cpp) and modified; see
// NOTICE.md.
//! FXT1 RGB decoding, after basisu `unpack_fxt1`: only CC_MIXED blocks without alpha (the mode
//! the Basis transcoder emits). Other modes (CC_HI, CC_CHROMA, CC_ALPHA, CC_MIXED with alpha)
//! are reported as unsupported.
//!
//! 16-byte block = 8x4 pixels: two 4x4 halves with 2-bit selectors (bytes 0..4 left, 4..8
//! right) and colors 0/1 (left) and 2/3 (right) as RGB555 plus a green LSB from `glsb` and
//! the first selector's high bit; palette c0, (2 c0 + c1) / 3, (c0 + 2 c1) / 3, c1.

use crate::{Error, Result};

/// Decodes one FXT1 block into 8x4 RGBA8 pixels (row major, 32 bytes per row).
pub(crate) fn decode_block(block: &[u8], out: &mut [u8]) -> Result<()> {
    let b = block.first_chunk::<16>().ok_or(Error::InvalidData)?;
    let lo = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
    let hi = u64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]);
    if hi >> 63 == 0 || (hi >> 60) & 1 != 0 {
        return Err(Error::Unsupported);
    }
    let glsb = ((hi >> 61) & 3) as u32;
    let field = |i: u32| ((hi >> i) & 31) as u32;
    let gbit = [
        ((lo & 3) as u32 >> 1) ^ (glsb & 1),
        glsb & 1,
        (((lo >> 32) & 3) as u32 >> 1) ^ (glsb >> 1),
        glsb >> 1,
    ];
    let colors: [[u32; 3]; 4] = core::array::from_fn(|k| {
        let s = k as u32 * 15;
        let (r, g, b) = (field(s + 10), (field(s + 5) << 1) | gbit[k], field(s));
        [
            (r << 3) | (r >> 2),
            (g << 2) | (g >> 4),
            (b << 3) | (b >> 2),
        ]
    });
    for half in 0..2 {
        let (c0, c1) = (colors[half * 2], colors[half * 2 + 1]);
        let pal = [
            c0,
            [0, 1, 2].map(|k| (c0[k] * 2 + c1[k] + 1) / 3),
            [0, 1, 2].map(|k| (c1[k] * 2 + c0[k] + 1) / 3),
            c1,
        ];
        for i in 0..16 {
            let sel = (b[half * 4 + (i >> 2)] >> ((i & 3) * 2)) & 3;
            let c = pal[sel as usize];
            let at = ((i >> 2) * 8 + half * 4 + (i & 3)) * 4;
            out[at..at + 4].copy_from_slice(&[c[0] as u8, c[1] as u8, c[2] as u8, 255]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_other_modes() {
        let mut out = [0u8; 128];
        assert_eq!(decode_block(&[0; 16], &mut out), Err(Error::Unsupported));
    }

    #[test]
    fn solid_mixed_block() {
        // CC_MIXED, every color = white (R/G/B 31, glsb 3), selectors 3 everywhere.
        let mut hi = 1u64 << 63 | 3u64 << 61;
        for k in 0..4 {
            hi |= 0x7fffu64 << (k * 15);
        }
        let mut blk = [0xffu8; 16];
        blk[8..].copy_from_slice(&hi.to_le_bytes());
        let mut out = [0u8; 128];
        assert_eq!(decode_block(&blk, &mut out), Ok(()));
        assert!(out.iter().all(|&v| v == 255));
    }
}
