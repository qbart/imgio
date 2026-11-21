// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated from Basis Universal (basisu_gpu_texture.cpp) via ktx-rs src/basisu/gpu_texture.rs; modified.
//! BC6H block decoder.

#![allow(clippy::needless_range_loop)]

use super::tables::{BC6H_2SUBSET_PATTERNS, BC6H_BIT_LAYOUTS, BC6H_MODE_SIG_BITS};

const BC6H_MODE_LOOKUP: [i8; 32] = [
    0, 1, 2, 10, 0, 1, 3, 11, 0, 1, 4, 12, 0, 1, 5, 13, 0, 1, 6, -1, 0, 1, 7, -1, 0, 1, 8, -1, 0,
    1, 9, -1,
];
const BC6H_WEIGHT3: [u8; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const BC6H_WEIGHT4: [u8; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

#[inline]
fn bc6h_sign_extend(val: i32, bits: i32) -> i32 {
    (val << (32 - bits)) >> (32 - bits)
}

#[inline]
fn bc6h_apply_delta(base: i32, delta: i32, num_bits: i32, is_signed: bool) -> i32 {
    let bitmask = (1 << num_bits) - 1;
    let v = (base + delta) & bitmask;
    if is_signed {
        bc6h_sign_extend(v, num_bits)
    } else {
        v
    }
}

fn bc6h_dequantize(mut val: i32, bits: i32, is_signed: bool) -> i32 {
    if is_signed {
        if bits >= 16 {
            return val;
        }
        let s_flag = val < 0;
        if s_flag {
            val = -val;
        }
        let mut result = if val == 0 {
            0
        } else if val >= (1 << (bits - 1)) - 1 {
            0x7FFF
        } else {
            ((val << 15) + 0x4000) >> (bits - 1)
        };
        if s_flag {
            result = -result;
        }
        result
    } else if bits >= 15 {
        val
    } else if val == 0 {
        0
    } else if val == (1 << bits) - 1 {
        0xFFFF
    } else {
        ((val << 16) + 0x8000) >> bits
    }
}

#[inline]
fn bc6h_interpolate(a: i32, b: i32, weights: &[u8], index: usize) -> i32 {
    let w = weights.get(index).copied().unwrap_or(0) as i32;
    (a * (64 - w) + b * w + 32) >> 6
}

#[inline]
fn bc6h_convert_to_half(mut val: i32, is_signed: bool) -> u16 {
    if !is_signed {
        // scale by 31/64
        return ((val * 31) >> 6) as u16;
    }
    // scale by 31/32
    val = if val < 0 {
        -(((-val) * 31) >> 5)
    } else {
        (val * 31) >> 5
    };
    let mut s = 0;
    if val < 0 {
        s = 0x8000;
        val = -val;
    }
    (s | val) as u16
}

#[inline]
fn bc6h_get_bits(num_bits: u32, l: &mut u64, h: &mut u64) -> u32 {
    let v = (*l & ((1u64 << num_bits) - 1)) as u32;
    *l >>= num_bits;
    *l |= *h << (64 - num_bits);
    *h >>= num_bits;
    v
}

#[inline]
fn bc6h_reverse_bits(v: u32, num_bits: u32) -> u32 {
    let mut res = 0;
    for i in 0..num_bits {
        let bit = u32::from(v & (1 << i) != 0);
        res |= bit << (num_bits - 1 - i);
    }
    res
}

/// Port of `unpack_bc6h`: decodes a BC6H block to 16 RGB halves (row major). On an invalid
/// mode false is returned and, like the C++ (which clears 4 halves per 12 half row), only the
/// first 4 halves of each row are zeroed.
pub(crate) fn unpack_bc6h(block: &[u8; 16], dst: &mut [[u16; 3]; 16], is_signed: bool) -> bool {
    let mut lo = [0u8; 8];
    let mut hi = [0u8; 8];
    lo.copy_from_slice(&block[..8]);
    hi.copy_from_slice(&block[8..]);
    let mut blo = u64::from_le_bytes(lo);
    let mut bhi = u64::from_le_bytes(hi);

    // Unpack mode
    let mode = BC6H_MODE_LOOKUP[(blo & 31) as usize];
    if mode < 0 {
        for y in 0..4 {
            dst[y * 4] = [0; 3];
            dst[y * 4 + 1][0] = 0;
        }
        return false;
    }
    let mode = mode as usize;
    // Skip mode bits
    bc6h_get_bits(if mode < 2 { 2 } else { 5 }, &mut blo, &mut bhi);

    let num_subsets = if mode >= 10 { 1 } else { 2 };
    let is_mode_9_or_10 = mode == 9 || mode == 10;

    // Unpack endpoint components [subset][comp][l/h]
    let mut comps = [[[0i32; 2]; 3]; 2];
    let mut part_index = 0i32;
    for layout in BC6H_BIT_LAYOUTS[mode].iter() {
        let (comp, index, last_bit, first_bit) =
            (layout[0], layout[1], layout[2] as i32, layout[3] as i32);
        if comp < 0 {
            break;
        }
        let (subset, lh_index) = ((index >> 1).max(0) as usize, (index & 1) as usize);
        let res = if comp == 3 {
            &mut part_index
        } else {
            &mut comps[subset & 1][comp as usize % 3][lh_index]
        };
        if first_bit < 0 {
            *res |= (bc6h_get_bits(1, &mut blo, &mut bhi) << last_bit) as i32;
        } else {
            let total_bits = (last_bit - first_bit).unsigned_abs() + 1;
            let bit_shift = first_bit.min(last_bit);
            let mut b = bc6h_get_bits(total_bits, &mut blo, &mut bhi);
            if last_bit < first_bit {
                b = bc6h_reverse_bits(b, total_bits);
            }
            *res |= (b << bit_shift) as i32;
        }
    }

    // Sign extend/dequantize endpoints
    let sig = |i: usize| BC6H_MODE_SIG_BITS[mode][i] as i32;
    let num_sig_bits = sig(0);
    if is_signed {
        for comp in 0..3 {
            comps[0][comp][0] = bc6h_sign_extend(comps[0][comp][0], num_sig_bits);
        }
    }
    if is_signed || !is_mode_9_or_10 {
        for subset in 0..num_subsets {
            for comp in 0..3 {
                for lh in (if subset != 0 { 0 } else { 1 })..2 {
                    comps[subset][comp][lh] =
                        bc6h_sign_extend(comps[subset][comp][lh], sig(1 + comp));
                }
            }
        }
    }
    if !is_mode_9_or_10 {
        for subset in 0..num_subsets {
            for comp in 0..3 {
                for lh in (if subset != 0 { 0 } else { 1 })..2 {
                    comps[subset][comp][lh] = bc6h_apply_delta(
                        comps[0][comp][0],
                        comps[subset][comp][lh],
                        num_sig_bits,
                        is_signed,
                    );
                }
            }
        }
    }
    for subset in comps.iter_mut().take(num_subsets) {
        for comp in subset.iter_mut() {
            for v in comp.iter_mut() {
                *v = bc6h_dequantize(*v, num_sig_bits, is_signed);
            }
        }
    }

    // Now unpack weights and output texels
    let weight_bits: i32 = if mode >= 10 { 4 } else { 3 };
    let weights: &[u8] = if mode >= 10 {
        &BC6H_WEIGHT4
    } else {
        &BC6H_WEIGHT3
    };
    let pat = &BC6H_2SUBSET_PATTERNS[(part_index & 31) as usize];
    for (i, d) in dst.iter_mut().enumerate() {
        let mut subset = if num_subsets == 1 {
            if i != 0 { 0 } else { 0x80 }
        } else {
            pat[i] as i32
        };
        let num_bits = weight_bits + if subset & 0x80 != 0 { -1 } else { 0 };
        subset &= 1;
        let weight_index = bc6h_get_bits(num_bits as u32, &mut blo, &mut bhi) as usize;
        let s = subset as usize;
        for c in 0..3 {
            d[c] = bc6h_convert_to_half(
                bc6h_interpolate(comps[s][c][0], comps[s][c][1], weights, weight_index),
                is_signed,
            );
        }
    }
    true
}
