// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of basisu `convert_etc1s_to_fxt1` (`basisu_transcoder.cpp`): ETC1S -> FXT1 CC_MIXED.
//!
//! One 16-byte FXT1 block covers 8x4 pixels = two ETC1S 4x4 blocks. Each half is transcoded
//! to BC1 first and then re-encoded (565 colors -> 555 + green LSB, remapped selectors).
//! Layout (little-endian): bytes 0..8 = 32 2-bit selectors; high 64 bits = colors 0..3
//! (B, G, R 5 bits each), then alpha (bit 60), glsb (61..62), mode (63).

use super::dxt::{Dxt1Block, convert_etc1s_to_dxt1};
use super::etc::{Endpoint, Selector};

fn conv_dxt1_to_fxt1_sels(sels: u8) -> u8 {
    const T: [u8; 16] = [0, 3, 1, 2, 12, 15, 13, 14, 4, 7, 5, 6, 8, 11, 9, 10];
    T[(sels & 15) as usize] | (T[(sels >> 4) as usize] << 4)
}

fn set_color(hi: &mut u64, slot: u32, c: [u32; 3]) {
    let v = u64::from(c[2] | (c[1] << 5) | (c[0] << 10));
    let shift = slot * 15;
    *hi = (*hi & !(0x7fffu64 << shift)) | (v << shift);
}

/// Writes half `subblock` (0 = left, 1 = right) of the FXT1 block `blk`. Subblock 0 writes the
/// whole block (right half duplicated from the left's border column); subblock 1 then
/// overwrites the right half and keeps glsb bit 0.
pub fn convert_etc1s_to_fxt1(blk: &mut [u8; 16], ep: &Endpoint, sel: &Selector, subblock: u32) {
    let mut d = Dxt1Block::default();
    convert_etc1s_to_dxt1(&mut d, ep, sel, false);
    let (l, h) = (d.get_low_color(), d.get_high_color());
    let mut c0 = [(l >> 11) & 31, (l >> 5) & 63, l & 31];
    let mut c1 = [(h >> 11) & 31, (h >> 5) & 63, h & 31];
    let (mut g0, mut g1) = (c0[1] & 1, c1[1] & 1);
    c0[1] >>= 1;
    c1[1] >>= 1;
    let mut sels = [0u8; 4];
    for (s, &b) in sels.iter_mut().zip(&d.bytes[4..8]) {
        *s = conv_dxt1_to_fxt1_sels(b);
    }
    if u32::from(sels[0] & 3) >> 1 != (g0 ^ g1) {
        std::mem::swap(&mut c0, &mut c1);
        std::mem::swap(&mut g0, &mut g1);
        for s in &mut sels {
            *s ^= 0xff;
        }
    }
    let mut hi = u64::from_le_bytes([
        blk[8], blk[9], blk[10], blk[11], blk[12], blk[13], blk[14], blk[15],
    ]);
    let g1 = u64::from(g1);
    if subblock == 0 {
        hi = (1u64 << 63) | ((g1 | (g1 << 1)) << 61);
        set_color(&mut hi, 0, c0);
        set_color(&mut hi, 1, c1);
        set_color(&mut hi, 2, c0);
        set_color(&mut hi, 3, c1);
        const BORDER_DUP: [u8; 4] = [0, 85, 170, 255];
        blk[..4].copy_from_slice(&sels);
        for i in 0..4 {
            blk[4 + i] = BORDER_DUP[(sels[i] >> 6) as usize];
        }
    } else {
        let glsb = ((hi >> 61) & 1) | (g1 << 1);
        hi = (hi & !(3u64 << 61)) | (glsb << 61);
        set_color(&mut hi, 2, c0);
        set_color(&mut hi, 3, c1);
        blk[4..8].copy_from_slice(&sels);
    }
    blk[8..].copy_from_slice(&hi.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_remap_is_a_permutation() {
        let mut seen = [false; 256];
        for s in 0..=255u8 {
            seen[conv_dxt1_to_fxt1_sels(s) as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }
}
