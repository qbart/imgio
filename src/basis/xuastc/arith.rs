// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of the basisu `arith` range decoder (basisu_transcoder_internal.h): adaptive bit and
//! multi-symbol models, plus the Elias-gamma contexts used by XUASTC LDR.
//!
//! Corrupt input never panics: unsigned arithmetic wraps like the C++ code, and every symbol
//! returned by [`ArithDec::decode_sym`] is below the model's symbol count.

use crate::{Error, Result};

const DM_LEN_SHIFT: u32 = 15;
const DM_MAX_COUNT: u32 = 1 << DM_LEN_SHIFT;
const BM_LEN_SHIFT: u32 = 13;
const BM_MAX_COUNT: u32 = 1 << BM_LEN_SHIFT;
const ARITH_MIN_LEN: u32 = 1 << 24;
const ARITH_MAX_LEN: u32 = u32::MAX;
/// `ArithMinExpectedDataBufSize`.
pub(super) const ARITH_MIN_EXPECTED_DATA_BUF_SIZE: usize = 5;
/// `ArithMaxSyms`.
const ARITH_MAX_SYMS: u32 = 2048;

/// `arith_bit_model`.
#[derive(Clone, Copy)]
pub(super) struct BitModel {
    bit0_prob: u32,
    bit0_count: u32,
    bit_count: u32,
    bits_until_update: i32,
    update_interval: u32,
}

impl Default for BitModel {
    fn default() -> Self {
        Self {
            bit0_count: 1,
            bit_count: 2,
            bit0_prob: 1 << (BM_LEN_SHIFT - 1),
            update_interval: 4,
            bits_until_update: 4,
        }
    }
}

impl BitModel {
    fn update(&mut self) {
        if self.bit_count >= BM_MAX_COUNT {
            self.bit_count = (self.bit_count + 1) >> 1;
            self.bit0_count = (self.bit0_count + 1) >> 1;
            if self.bit0_count == self.bit_count {
                self.bit_count += 1;
            }
        }
        let scale = 0x8000_0000u32 / self.bit_count.max(1);
        self.bit0_prob = self.bit0_count.wrapping_mul(scale) >> (31 - BM_LEN_SHIFT);
        self.update_interval = ((5 * self.update_interval) >> 2).clamp(4, 128);
        self.bits_until_update = self.update_interval as i32;
    }
}

/// `arith_gamma_contexts`.
#[derive(Default)]
pub(super) struct GammaContexts {
    prefix: [BitModel; 3],
    tail: [BitModel; 4],
}

/// `arith_data_model`. A model with no symbols (`Default`) is "uninitialized".
#[derive(Default, Clone)]
pub(super) struct DataModel {
    num_data_syms: u32,
    sym_freqs: Vec<u32>,
    total_sym_freq: u32,
    cum_sym_freqs: Vec<u32>,
    update_interval: u32,
    num_syms_until_next_update: i32,
}

impl DataModel {
    /// `arith_data_model(num_syms, faster_update)`.
    pub(super) fn new(num_syms: u32, faster_update: bool) -> Self {
        let mut m = Self::default();
        m.init(num_syms, faster_update);
        m
    }

    /// `init`: `num_syms` must be in [2, 2048].
    pub(super) fn init(&mut self, num_syms: u32, faster_update: bool) {
        let num_syms = num_syms.clamp(2, ARITH_MAX_SYMS);
        self.num_data_syms = num_syms;
        self.sym_freqs = vec![1; num_syms as usize];
        self.cum_sym_freqs = vec![0; num_syms as usize + 1];
        self.total_sym_freq = num_syms;
        self.update_interval = num_syms;
        self.num_syms_until_next_update = 0;
        self.update();
        if faster_update {
            self.update_interval = num_syms.div_ceil(8).max(4).min((num_syms + 6) << 3);
            self.num_syms_until_next_update = self.update_interval as i32;
        }
    }

    pub(super) fn num_data_syms(&self) -> u32 {
        self.num_data_syms
    }

    fn update(&mut self) {
        if self.num_data_syms == 0 {
            return;
        }
        while self.total_sym_freq >= DM_MAX_COUNT {
            self.total_sym_freq = 0;
            for f in self.sym_freqs.iter_mut() {
                *f = (*f + 1) >> 1;
                self.total_sym_freq += *f;
            }
        }
        let scale = 0x8000_0000u32 / self.total_sym_freq.max(1);
        let mut sum = 0u32;
        for (c, &f) in self.cum_sym_freqs.iter_mut().zip(&self.sym_freqs) {
            *c = scale.wrapping_mul(sum) >> (31 - DM_LEN_SHIFT);
            sum = sum.wrapping_add(f);
        }
        if let Some(last) = self.cum_sym_freqs.last_mut() {
            *last = DM_MAX_COUNT;
        }
        self.update_interval =
            ((5 * self.update_interval) >> 2).clamp(4, (self.num_data_syms + 6) << 3);
        self.num_syms_until_next_update = self.update_interval as i32;
    }
}

/// `arith_dec`.
pub(super) struct ArithDec<'a> {
    buf: &'a [u8],
    pos: usize,
    value: u32,
    length: u32,
}

impl<'a> ArithDec<'a> {
    /// `init`: needs at least [`ARITH_MIN_EXPECTED_DATA_BUF_SIZE`] bytes.
    pub(super) fn new(buf: &'a [u8]) -> Result<Self> {
        if buf.len() < ARITH_MIN_EXPECTED_DATA_BUF_SIZE {
            return Err(Error::InvalidData);
        }
        let value = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        Ok(Self {
            buf,
            pos: 4,
            value,
            length: ARITH_MAX_LEN,
        })
    }

    fn renorm(&mut self) {
        loop {
            let next = match self.buf.get(self.pos) {
                Some(&b) => {
                    self.pos += 1;
                    u32::from(b)
                }
                None => 0,
            };
            self.value = (self.value << 8) | next;
            self.length <<= 8;
            if self.length >= ARITH_MIN_LEN {
                break;
            }
        }
    }

    pub(super) fn get_bit(&mut self) -> u32 {
        self.length >>= 1;
        let bit = u32::from(self.value >= self.length);
        if bit != 0 {
            self.value = self.value.wrapping_sub(self.length);
        }
        if self.length < ARITH_MIN_LEN {
            self.renorm();
        }
        bit
    }

    /// `get_bits` (1 to 20 bits; other counts return 0 like the C++ code).
    pub(super) fn get_bits(&mut self, num_bits: u32) -> u32 {
        if !(1..=20).contains(&num_bits) {
            return 0;
        }
        self.length >>= num_bits;
        let v = self.value / self.length.max(1);
        self.value = self.value.wrapping_sub(self.length.wrapping_mul(v));
        if self.length < ARITH_MIN_LEN {
            self.renorm();
        }
        v
    }

    pub(super) fn decode_truncated_binary(&mut self, n: u32) -> u32 {
        let n = n.max(2);
        let k = 31 - n.leading_zeros();
        let u = (1u32 << (k + 1)) - n;
        let mut result = self.get_bits(k);
        if result >= u {
            result = ((result << 1) | self.get_bits(1)).wrapping_sub(u);
        }
        result
    }

    pub(super) fn decode_bit(&mut self, dm: &mut BitModel) -> bool {
        let x = dm.bit0_prob.wrapping_mul(self.length >> BM_LEN_SHIFT);
        let bit = self.value >= x;
        if !bit {
            self.length = x;
            dm.bit0_count = dm.bit0_count.wrapping_add(1);
        } else {
            self.value = self.value.wrapping_sub(x);
            self.length = self.length.wrapping_sub(x);
        }
        dm.bit_count = dm.bit_count.wrapping_add(1);
        if self.length < ARITH_MIN_LEN {
            self.renorm();
        }
        dm.bits_until_update -= 1;
        if dm.bits_until_update <= 0 {
            dm.update();
        }
        bit
    }

    /// `decode_gamma`: 0 when the prefix is impossibly long (corrupt data).
    pub(super) fn decode_gamma(&mut self, ctxs: &mut GammaContexts) -> u32 {
        let mut k = 0usize;
        while self.decode_bit(&mut ctxs.prefix[k.min(2)]) {
            k += 1;
            if k > 16 {
                return 0;
            }
        }
        let mut n = 1u32 << k;
        for i in (0..k).rev() {
            let bit = u32::from(self.decode_bit(&mut ctxs.tail[i.min(3)]));
            n |= bit << i;
        }
        n
    }

    pub(super) fn decode_sym(&mut self, dm: &mut DataModel) -> u32 {
        if dm.num_data_syms == 0 {
            return 0;
        }
        let (mut x, mut y) = (0u32, self.length);
        self.length >>= DM_LEN_SHIFT;
        let (mut low_idx, mut hi_idx) = (0u32, dm.num_data_syms);
        let mut mid_idx = hi_idx >> 1;
        loop {
            let z = self.length.wrapping_mul(dm.cum_sym_freqs[mid_idx as usize]);
            if z > self.value {
                hi_idx = mid_idx;
                y = z;
            } else {
                low_idx = mid_idx;
                x = z;
            }
            mid_idx = (low_idx + hi_idx) >> 1;
            if mid_idx == low_idx {
                break;
            }
        }
        self.value = self.value.wrapping_sub(x);
        self.length = y.wrapping_sub(x);
        if self.length < ARITH_MIN_LEN {
            self.renorm();
        }
        if let Some(f) = dm.sym_freqs.get_mut(low_idx as usize) {
            *f += 1;
        }
        dm.total_sym_freq += 1;
        dm.num_syms_until_next_update -= 1;
        if dm.num_syms_until_next_update <= 0 {
            dm.update();
        }
        low_idx
    }
}
