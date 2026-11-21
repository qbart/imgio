// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Encoder side of the basisu `arith` range coder (`basisu_transcoder_internal.h`):
//! `arith_enc`, model pricing (`get_price`) and `arith_fastbits_f32`. The adaptive models are
//! the decoder's ([`BitModel`], [`DataModel`], [`GammaContexts`]); both sides update them
//! identically.

use crate::basis::xuastc::arith::{
    ARITH_MAX_LEN, ARITH_MIN_EXPECTED_DATA_BUF_SIZE, ARITH_MIN_LEN, BM_LEN_SHIFT, BM_MAX_COUNT,
    BitModel, DM_LEN_SHIFT, DM_MAX_COUNT, DataModel, GammaContexts,
};
use std::sync::OnceLock;

/// `arith_fastbits_f32::TABLE_BITS`.
const FB_TABLE_BITS: u32 = 8;
const FB_TABLE_SIZE: usize = 1 << FB_TABLE_BITS;
const FB_MANT_BITS: u32 = 23;
const FB_FRAC_BITS: u32 = FB_MANT_BITS - FB_TABLE_BITS;
const FB_FRAC_MASK: u32 = (1 << FB_FRAC_BITS) - 1;

/// `arith_fastbits_f32::g_lut_edge` (log2 of `m` sampled at `1 + i / TABLE_SIZE`).
fn lut_edge() -> &'static [f32; FB_TABLE_SIZE + 1] {
    static T: OnceLock<[f32; FB_TABLE_SIZE + 1]> = OnceLock::new();
    T.get_or_init(|| {
        let inv_ln2 = std::f32::consts::LOG2_E; // 1.4426950408889634f
        let mut t = [0f32; FB_TABLE_SIZE + 1];
        for (i, v) in t.iter_mut().enumerate() {
            let m = 1.0f32 + i as f32 / FB_TABLE_SIZE as f32;
            *v = m.ln() * inv_ln2;
        }
        t
    })
}

/// `arith_fastbits_f32::bits_from_prob_linear`: approximates `-log2(p)`.
pub(crate) fn bits_from_prob_linear(p: f32) -> f32 {
    let p = if p < f32::MIN_POSITIVE { 0.0 } else { p };
    let u = p.to_bits();
    let e = ((u >> 23) & 0xFF) as i32 - 127;
    let mant = u & 0x7F_FFFF;
    let idx = (mant >> FB_FRAC_BITS) as usize;
    let frac = mant & FB_FRAC_MASK;
    let inv_scale = 1.0f32 / (1u32 << FB_FRAC_BITS) as f32;
    let t = frac as f32 * inv_scale;
    let lut = lut_edge();
    let y0 = lut[idx];
    let y1 = lut[idx + 1];
    let log2m = y0 + t * (y1 - y0);
    -(e as f32 + log2m)
}

impl BitModel {
    /// `arith_bit_model::get_price`.
    pub(crate) fn get_price(&self, bit: bool) -> f32 {
        let prob_0 = self.bit0_prob as f32 / BM_MAX_COUNT as f32;
        let prob = if bit { 1.0f32 - prob_0 } else { prob_0 };
        bits_from_prob_linear(prob)
    }
}

impl DataModel {
    /// `arith_data_model::get_price`.
    pub(crate) fn get_price(&self, sym_index: u32) -> f32 {
        if sym_index >= self.num_data_syms {
            return 0.0;
        }
        let s = sym_index as usize;
        let prob = (self.cum_sym_freqs[s + 1] - self.cum_sym_freqs[s]) as f32 / DM_MAX_COUNT as f32;
        bits_from_prob_linear(prob)
    }
}

/// `arith_enc`.
pub(crate) struct ArithEnc {
    data_buf: Vec<u8>,
    base: u32,
    length: u32,
}

impl Default for ArithEnc {
    fn default() -> Self {
        Self::new(0)
    }
}

impl ArithEnc {
    /// `init(reserve_size)`.
    pub(crate) fn new(reserve_size: usize) -> Self {
        Self {
            data_buf: Vec::with_capacity(reserve_size),
            base: 0,
            length: ARITH_MAX_LEN,
        }
    }

    /// `get_data_buf`.
    pub(crate) fn get_data_buf(&self) -> &[u8] {
        &self.data_buf
    }

    pub(crate) fn into_data_buf(self) -> Vec<u8> {
        self.data_buf
    }

    /// `put_bit`.
    pub(crate) fn put_bit(&mut self, bit: u32) {
        self.length >>= 1;
        if bit != 0 {
            let orig_base = self.base;
            self.base = self.base.wrapping_add(self.length);
            if orig_base > self.base {
                self.prop_carry();
            }
        }
        if self.length < ARITH_MIN_LEN {
            self.renorm();
        }
    }

    /// `put_bits` (1 to 20 bits).
    pub(crate) fn put_bits(&mut self, val: u32, num_bits: u32) {
        self.length >>= num_bits;
        let orig_base = self.base;
        self.base = self.base.wrapping_add(val.wrapping_mul(self.length));
        if orig_base > self.base {
            self.prop_carry();
        }
        if self.length < ARITH_MIN_LEN {
            self.renorm();
        }
    }

    /// `put_truncated_binary`: returns the number of bits written.
    pub(crate) fn put_truncated_binary(&mut self, v: u32, n: u32) -> u32 {
        let k = n.max(1).ilog2();
        let u = (1u32 << (k + 1)).wrapping_sub(n);
        if v < u {
            self.put_bits(v, k);
            return k;
        }
        let x = v + u;
        self.put_bits(x >> 1, k);
        self.put_bits(x & 1, 1);
        k + 1
    }

    /// `put_gamma`.
    pub(crate) fn put_gamma(&mut self, n: u32, ctxs: &mut GammaContexts) {
        let _ = self.put_gamma_and_return_price(n, ctxs);
    }

    /// `put_gamma_and_return_price`.
    pub(crate) fn put_gamma_and_return_price(&mut self, n: u32, ctxs: &mut GammaContexts) -> f32 {
        if n == 0 {
            return 0.0;
        }
        let k = n.ilog2() as usize;
        if k > 16 {
            return 0.0;
        }
        let mut total_price = 0.0f32;
        for i in 0..k {
            let m = &mut ctxs.prefix[i.min(2)];
            total_price += m.get_price(true);
            self.encode_bit(1, m);
        }
        let m = &mut ctxs.prefix[k.min(2)];
        total_price += m.get_price(false);
        self.encode_bit(0, m);
        for i in (0..k).rev() {
            let bit = (n >> i) & 1;
            let m = &mut ctxs.tail[i.min(3)];
            total_price += m.get_price(bit != 0);
            self.encode_bit(bit, m);
        }
        total_price
    }

    /// `encode(bit, arith_bit_model&)`.
    pub(crate) fn encode_bit(&mut self, bit: u32, dm: &mut BitModel) {
        let x = dm.bit0_prob.wrapping_mul(self.length >> BM_LEN_SHIFT);
        if bit == 0 {
            self.length = x;
            dm.bit0_count = dm.bit0_count.wrapping_add(1);
        } else {
            let orig_base = self.base;
            self.base = self.base.wrapping_add(x);
            self.length = self.length.wrapping_sub(x);
            if orig_base > self.base {
                self.prop_carry();
            }
        }
        dm.bit_count = dm.bit_count.wrapping_add(1);
        if self.length < ARITH_MIN_LEN {
            self.renorm();
        }
        dm.bits_until_update -= 1;
        if dm.bits_until_update <= 0 {
            dm.update();
        }
    }

    /// `encode_and_return_price(bit, arith_bit_model&)`.
    pub(crate) fn encode_bit_and_return_price(&mut self, bit: u32, dm: &mut BitModel) -> f32 {
        let price = dm.get_price(bit != 0);
        self.encode_bit(bit, dm);
        price
    }

    /// `encode(sym, arith_data_model&)`.
    pub(crate) fn encode(&mut self, sym: u32, dm: &mut DataModel) {
        let s = sym as usize;
        if sym >= dm.num_data_syms {
            return;
        }
        let orig_base = self.base;
        if sym == dm.num_data_syms - 1 {
            let x = dm.cum_sym_freqs[s].wrapping_mul(self.length >> DM_LEN_SHIFT);
            self.base = self.base.wrapping_add(x);
            self.length = self.length.wrapping_sub(x);
        } else {
            self.length >>= DM_LEN_SHIFT;
            let x = dm.cum_sym_freqs[s].wrapping_mul(self.length);
            self.base = self.base.wrapping_add(x);
            self.length = dm.cum_sym_freqs[s + 1]
                .wrapping_mul(self.length)
                .wrapping_sub(x);
        }
        if orig_base > self.base {
            self.prop_carry();
        }
        if self.length < ARITH_MIN_LEN {
            self.renorm();
        }
        dm.sym_freqs[s] = dm.sym_freqs[s].wrapping_add(1);
        dm.total_sym_freq = dm.total_sym_freq.wrapping_add(1);
        dm.num_syms_until_next_update -= 1;
        if dm.num_syms_until_next_update <= 0 {
            dm.update();
        }
    }

    /// `encode_and_return_price(sym, arith_data_model&)`.
    pub(crate) fn encode_and_return_price(&mut self, sym: u32, dm: &mut DataModel) -> f32 {
        let price = dm.get_price(sym);
        self.encode(sym, dm);
        price
    }

    /// `flush`.
    pub(crate) fn flush(&mut self) {
        let orig_base = self.base;
        if self.length <= 2 * ARITH_MIN_LEN {
            self.base = self.base.wrapping_add(ARITH_MIN_LEN >> 1);
            self.length = ARITH_MIN_LEN >> 9;
        } else {
            self.base = self.base.wrapping_add(ARITH_MIN_LEN);
            self.length = ARITH_MIN_LEN >> 1;
        }
        if orig_base > self.base {
            self.prop_carry();
        }
        self.renorm();
        if self.data_buf.len() < ARITH_MIN_EXPECTED_DATA_BUF_SIZE {
            self.data_buf.resize(ARITH_MIN_EXPECTED_DATA_BUF_SIZE, 0);
        }
    }

    fn prop_carry(&mut self) {
        for b in self.data_buf.iter_mut().rev() {
            if *b == 0xFF {
                *b = 0;
            } else {
                *b += 1;
                break;
            }
        }
    }

    fn renorm(&mut self) {
        loop {
            self.data_buf.push((self.base >> 24) as u8);
            self.base <<= 8;
            self.length <<= 8;
            if self.length >= ARITH_MIN_LEN {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::xuastc::arith::ArithDec;

    /// Random symbols through every coding primitive decode back identically.
    #[test]
    fn arith_round_trip() -> crate::Result<()> {
        let mut seed = 0x1234_5678u32;
        let mut rnd = |n: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed % n
        };
        let mut ops = Vec::new();
        for _ in 0..5000 {
            let op = rnd(6);
            let v = match op {
                0 => rnd(2),
                1 => rnd(1 << 12),
                2 => rnd(37),
                3 => rnd(7),
                4 => rnd(300),
                _ => 1 + rnd(1000),
            };
            ops.push((op, v));
        }
        let mut enc = ArithEnc::new(0);
        let (mut bm, mut dm, mut dm2, mut g) = (
            BitModel::default(),
            DataModel::new(7, false),
            DataModel::new(300, true),
            GammaContexts::default(),
        );
        for &(op, v) in &ops {
            match op {
                0 => enc.put_bit(v),
                1 => enc.put_bits(v, 12),
                2 => {
                    enc.put_truncated_binary(v, 37);
                }
                3 => {
                    enc.encode(v, &mut dm);
                    enc.encode_bit(v & 1, &mut bm);
                }
                4 => enc.encode(v, &mut dm2),
                _ => enc.put_gamma(v, &mut g),
            }
        }
        enc.put_bits(0xAF, 8);
        enc.flush();
        let buf = enc.into_data_buf();
        let mut dec = ArithDec::new(&buf)?;
        let (mut bm, mut dm, mut dm2, mut g) = (
            BitModel::default(),
            DataModel::new(7, false),
            DataModel::new(300, true),
            GammaContexts::default(),
        );
        for &(op, v) in &ops {
            match op {
                0 => assert_eq!(dec.get_bit(), v),
                1 => assert_eq!(dec.get_bits(12), v),
                2 => assert_eq!(dec.decode_truncated_binary(37), v),
                3 => {
                    assert_eq!(dec.decode_sym(&mut dm), v);
                    assert_eq!(u32::from(dec.decode_bit(&mut bm)), v & 1);
                }
                4 => assert_eq!(dec.decode_sym(&mut dm2), v),
                _ => assert_eq!(dec.decode_gamma(&mut g), v),
            }
        }
        assert_eq!(dec.get_bits(8), 0xAF);
        Ok(())
    }

    #[test]
    fn prices_match_log2() {
        for i in 1..1000u32 {
            let p = i as f32 / 1000.0;
            assert!((bits_from_prob_linear(p) - (-p.log2())).abs() < 0.00125);
        }
    }
}
