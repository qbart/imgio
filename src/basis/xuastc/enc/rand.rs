// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of `basisu::rand` (`basisu_enc.h`): `std::mt19937` plus the libstdc++ (GCC 13)
//! `uniform_int_distribution<int>`, `uniform_real_distribution<float>` and
//! `normal_distribution<float>` algorithms, so random sequences match the C++ encoder exactly.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// `std::mt19937`.
#[derive(Clone)]
pub(crate) struct Mt19937 {
    mt: [u32; N],
    idx: usize,
}

impl Mt19937 {
    /// Default seed (5489).
    pub fn new() -> Self {
        Self::with_seed(5489)
    }

    pub fn with_seed(s: u32) -> Self {
        let mut r = Self { mt: [0; N], idx: N };
        r.seed(s);
        r
    }

    /// `seed(s)`.
    pub fn seed(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.idx = N;
    }

    fn gen_rand(&mut self) {
        for k in 0..N - M {
            let y = (self.mt[k] & UPPER_MASK) | (self.mt[k + 1] & LOWER_MASK);
            self.mt[k] = self.mt[k + M] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
        }
        for k in N - M..N - 1 {
            let y = (self.mt[k] & UPPER_MASK) | (self.mt[k + 1] & LOWER_MASK);
            self.mt[k] = self.mt[k + M - N] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
        }
        let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
        self.idx = 0;
    }

    /// `operator()`.
    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= N {
            self.gen_rand();
        }
        let mut z = self.mt[self.idx];
        self.idx += 1;
        z ^= z >> 11;
        z ^= (z << 7) & 0x9d2c_5680;
        z ^= (z << 15) & 0xefc6_0000;
        z ^= z >> 18;
        z
    }

    /// `std::generate_canonical<float, 24>` (one 32-bit draw).
    fn canonical_f32(&mut self) -> f32 {
        let sum = self.next_u32() as f32;
        let ret = sum / 4_294_967_296.0f32;
        if ret >= 1.0 {
            // `std::nextafter(1.0f, 0.0f)`
            f32::from_bits(1.0f32.to_bits() - 1)
        } else {
            ret
        }
    }
}

impl Default for Mt19937 {
    fn default() -> Self {
        Self::new()
    }
}

/// `basisu::rand`.
#[derive(Clone, Default)]
pub(crate) struct BasisuRand {
    mt: Mt19937,
    /// `normal_distribution<float>` saved second value.
    norm_saved: f32,
    norm_saved_available: bool,
}

impl BasisuRand {
    pub fn new() -> Self {
        Self::default()
    }

    /// `rand(uint32_t s)`.
    pub fn with_seed(s: u32) -> Self {
        let mut r = Self::default();
        r.seed(s);
        r
    }

    /// `seed(s)` (also resets the normal distribution).
    pub fn seed(&mut self, s: u32) {
        self.mt.seed(s);
        self.norm_saved_available = false;
    }

    /// `irand(l, h)`: uniform in [l, h] (`uniform_int_distribution<int>`, Lemire downscaling).
    pub fn irand(&mut self, l: i32, h: i32) -> i32 {
        // `__urange` (64-bit `__uctype` arithmetic on LP64).
        let urange = (h as i64 as u64).wrapping_sub(l as i64 as u64);
        let ret: u64 = if urange < 0xffff_ffff {
            let range = (urange + 1) as u32;
            let mut product = u64::from(self.mt.next_u32()) * u64::from(range);
            let mut low = product as u32;
            if low < range {
                let threshold = range.wrapping_neg() % range;
                while low < threshold {
                    product = u64::from(self.mt.next_u32()) * u64::from(range);
                    low = product as u32;
                }
            }
            product >> 32
        } else {
            // `__urange == __urngrange`: the full 32-bit range (never wider for `int`).
            u64::from(self.mt.next_u32())
        };
        (ret as i64 as i32).wrapping_add(l)
    }

    /// `iprob(p)`: true with probability p/100.
    pub fn iprob(&mut self, p: i32) -> bool {
        self.irand(0, 99) < p
    }

    /// `bit()`.
    pub fn bit(&mut self) -> bool {
        self.irand(0, 1) == 1
    }

    /// `frand(l, h)`: uniform in [l, h) (`uniform_real_distribution<float>`).
    pub fn frand(&mut self, l: f32, h: f32) -> f32 {
        if l >= h {
            return l;
        }
        let mut x = (self.mt.canonical_f32() * (h - l)) + l;
        if x < l {
            x = l;
        } else if x >= h {
            // `std::nextafter(h, l)` with l < h
            x = next_toward_lower(h);
        }
        x
    }

    /// `gaussian(mean, stddev)` (`normal_distribution<float>`, Marsaglia polar method).
    pub fn gaussian(&mut self, mean: f32, stddev: f32) -> f32 {
        let n = if self.norm_saved_available {
            self.norm_saved_available = false;
            self.norm_saved
        } else {
            let (mut x, mut y, mut r2);
            loop {
                x = 2.0f32 * self.mt.canonical_f32() - 1.0;
                y = 2.0f32 * self.mt.canonical_f32() - 1.0;
                r2 = x * x + y * y;
                if !(r2 > 1.0 || r2 == 0.0) {
                    break;
                }
            }
            let mult = (-2.0f32 * r2.ln() / r2).sqrt();
            self.norm_saved = x * mult;
            self.norm_saved_available = true;
            y * mult
        };
        // `__ret * stddev() + mean()` with the default N(0, 1) parameters.
        let n = n * 1.0f32 + 0.0f32;
        mean + stddev * n
    }
}

/// `std::nextafter(v, -inf)` for finite `v`.
fn next_toward_lower(v: f32) -> f32 {
    if v == 0.0 {
        return -f32::from_bits(1);
    }
    let b = v.to_bits();
    if v > 0.0 {
        f32::from_bits(b - 1)
    } else {
        f32::from_bits(b + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_10000th() {
        // C++11 [rand.predef]: the 10000th invocation of a default-constructed mt19937.
        let mut r = Mt19937::new();
        let mut v = 0;
        for _ in 0..10000 {
            v = r.next_u32();
        }
        assert_eq!(v, 4_123_659_995);
    }

    /// FNV-1a over a mixed sequence of every `basisu::rand` method; the expected value was
    /// produced by the C++ `basisu::rand` (GCC 13 libstdc++).
    #[test]
    fn matches_cpp_sequence() {
        let mut h: u64 = 1_469_598_103_934_665_603;
        let mut mix = |v: u32| {
            h ^= u64::from(v);
            h = h.wrapping_mul(1_099_511_628_211);
        };
        for s in 0..64u32 {
            let mut r = BasisuRand::new();
            r.seed(s * 7919 + 3);
            for i in 0..200 {
                mix(r.irand(0, i % 17) as u32);
                mix(r.irand(-5, 1000 + i) as u32);
                mix(u32::from(r.iprob(10)));
                mix(u32::from(r.bit()));
                mix(r.frand(0.0, 0.3).to_bits());
                mix(r.gaussian(0.0, 1.7).to_bits());
                mix((3.0 + r.gaussian(0.0, 2.5)).round() as i32 as u32);
            }
        }
        assert_eq!(h, 9_230_473_086_785_962_005);
    }
}
