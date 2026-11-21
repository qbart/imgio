// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of basisu `bc7e_scalar` (`encoder/basisu_bc7e_scalar.cpp`, the scalar C++ translation
//! of `bc7e.ispc`): a high quality BC7 block encoder with seven speed presets.
//!
//! Bit-exact with the C++ built for x86-64 (SSE2 scalar float math, no fast-math). Float math
//! follows the C++ operation order in `f32`; `(int)` casts emulate `cvttss2si` / `cvttsd2si`
//! (out of range or NaN gives `INT_MIN`); `1.0f / sqrt(float)` is evaluated in `f64` because
//! the C++ resolves `sqrt` to the `double` overload. `min` / `max` / `clamp` follow the C++
//! templates (`a < b ? a : b`, `clamp = min(max(v, lo), hi)`). Unsigned arithmetic wraps.
//! `bc7e_compress_block_single_mode` (a testing API unused by the presets) is not ported.

// Crate-private encoder; it is wired into `imgio::encode` separately.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

type Cq = [i32; 4];
type V4 = [f32; 4];

const DENOM_BIAS: f32 = 0.0000125;
const CHECKERBOARD_PARTITION: u32 = 34;
const MAX_PARTITIONS0: u32 = 16;
const MAX_PARTITIONS1: u32 = 64;
const MAX_PARTITIONS2: u32 = 64;
const MAX_PARTITIONS3: u32 = 64;
const MAX_PARTITIONS7: u32 = 64;

const W2: [u32; 4] = [0, 21, 43, 64];
const W3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const W4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

// Least squares weight constants: w * w, (1 - w) * w, (1 - w) * (1 - w), w.
const W2X: [[f32; 4]; 4] = [
    [0.000000, 0.000000, 1.000000, 0.000000],
    [0.107666, 0.220459, 0.451416, 0.328125],
    [0.451416, 0.220459, 0.107666, 0.671875],
    [1.000000, 0.000000, 0.000000, 1.000000],
];
const W3X: [[f32; 4]; 8] = [
    [0.000000, 0.000000, 1.000000, 0.000000],
    [0.019775, 0.120850, 0.738525, 0.140625],
    [0.079102, 0.202148, 0.516602, 0.281250],
    [0.177979, 0.243896, 0.334229, 0.421875],
    [0.334229, 0.243896, 0.177979, 0.578125],
    [0.516602, 0.202148, 0.079102, 0.718750],
    [0.738525, 0.120850, 0.019775, 0.859375],
    [1.000000, 0.000000, 0.000000, 1.000000],
];
const W4X: [[f32; 4]; 16] = [
    [0.000000, 0.000000, 1.000000, 0.000000],
    [0.003906, 0.058594, 0.878906, 0.062500],
    [0.019775, 0.120850, 0.738525, 0.140625],
    [0.041260, 0.161865, 0.635010, 0.203125],
    [0.070557, 0.195068, 0.539307, 0.265625],
    [0.107666, 0.220459, 0.451416, 0.328125],
    [0.165039, 0.241211, 0.352539, 0.406250],
    [0.219727, 0.249023, 0.282227, 0.468750],
    [0.282227, 0.249023, 0.219727, 0.531250],
    [0.352539, 0.241211, 0.165039, 0.593750],
    [0.451416, 0.220459, 0.107666, 0.671875],
    [0.539307, 0.195068, 0.070557, 0.734375],
    [0.635010, 0.161865, 0.041260, 0.796875],
    [0.738525, 0.120850, 0.019775, 0.859375],
    [0.878906, 0.058594, 0.003906, 0.937500],
    [1.000000, 0.000000, 0.000000, 1.000000],
];
const PARTITION2: [[u8; 16]; 64] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
];
const PARTITION3: [[u8; 16]; 64] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2],
    [0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1],
    [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2],
    [0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2],
    [0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2],
    [0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2],
    [0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2],
    [0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2],
    [0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0],
    [0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
    [0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2],
    [0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2],
    [0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0],
    [0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2],
    [0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
    [0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2],
    [0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1],
    [0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2],
    [0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    [0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0],
    [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1],
    [0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1],
    [0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1],
    [0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2],
    [0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1],
    [0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2],
    [0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
    [0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2],
    [0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2],
    [0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2],
    [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1],
    [0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2],
    [0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0],
];
const ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];
const ANCHOR3_1: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5,
    15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5,
    10, 8, 13, 15, 12, 3, 3,
];
const ANCHOR3_2: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6,
    10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];
const PARTITION1: [u8; 16] = [0; 16];

const NUM_SUBSETS: [u32; 8] = [3, 2, 3, 2, 1, 1, 1, 2];
const PARTITION_BITS: [u32; 8] = [4, 6, 6, 6, 0, 0, 0, 6];
const COLOR_INDEX_BITCOUNT: [u32; 8] = [3, 3, 2, 2, 2, 2, 4, 2];
const ALPHA_INDEX_BITCOUNT: [u32; 8] = [0, 0, 0, 0, 3, 2, 4, 2];
const MODE_HAS_P_BITS: [bool; 8] = [true, true, false, true, false, false, true, true];
const MODE_HAS_SHARED_P_BITS: [bool; 8] = [false, true, false, false, false, false, false, false];
const COLOR_PRECISION: [u32; 8] = [4, 6, 5, 7, 5, 7, 7, 5];
const ALPHA_PRECISION: [u32; 8] = [0, 0, 0, 0, 6, 8, 7, 5];

const MODE_1_OPTIMAL_INDEX: usize = 2;
const MODE_7_OPTIMAL_INDEX: usize = 1;
const MODE_6_OPTIMAL_INDEX: usize = 5;
const MODE_4_OPTIMAL_INDEX3: usize = 2;
const MODE_4_OPTIMAL_INDEX2: usize = 1;
const MODE_5_OPTIMAL_INDEX: usize = 1;
const MODE_0_OPTIMAL_INDEX: usize = 2;

// ---------------------------------------------------------------------------------------------
// Parameters (`bc7e_compress_block_params`).

/// Opaque block settings (`m_opaque_settings`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OpaqueSettings {
    pub max_mode13_partitions_to_try: u32,
    pub max_mode0_partitions_to_try: u32,
    pub max_mode2_partitions_to_try: u32,
    pub use_mode: [bool; 7],
}

/// Alpha block settings (`m_alpha_settings`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AlphaSettings {
    pub max_mode7_partitions_to_try: u32,
    pub mode67_error_weight_mul: [u32; 4],
    pub use_mode4: bool,
    pub use_mode5: bool,
    pub use_mode6: bool,
    pub use_mode7: bool,
    pub use_mode4_rotation: bool,
    pub use_mode5_rotation: bool,
}

/// Encoder parameters (`bc7e_compress_block_params`). Build with one of the presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Params {
    pub max_partitions_mode: [u32; 8],
    pub weights: [u32; 4],
    pub uber_level: u32,
    pub refinement_passes: u32,
    pub mode4_rotation_mask: u32,
    pub mode4_index_mask: u32,
    pub mode5_rotation_mask: u32,
    pub uber1_mask: u32,
    pub perceptual: bool,
    pub pbit_search: bool,
    pub mode6_only: bool,
    /// Allow the precomputed one-color optimal endpoint tables (solid and average color paths).
    pub use_luts: bool,
    pub opaque: OpaqueSettings,
    pub alpha: AlphaSettings,
}

impl Params {
    /// `bc7e_compress_block_params_init`: the base settings every preset starts from.
    pub(crate) fn new(perceptual: bool) -> Self {
        Self {
            max_partitions_mode: [
                MAX_PARTITIONS0,
                MAX_PARTITIONS1,
                MAX_PARTITIONS2,
                MAX_PARTITIONS3,
                0,
                0,
                0,
                MAX_PARTITIONS7,
            ],
            weights: if perceptual {
                [128, 64, 16, 256]
            } else {
                [1, 1, 1, 1]
            },
            uber_level: 0,
            refinement_passes: 1,
            mode4_rotation_mask: 0xF,
            mode4_index_mask: 3,
            mode5_rotation_mask: 0xF,
            uber1_mask: 7,
            perceptual,
            pbit_search: false,
            mode6_only: false,
            use_luts: true,
            opaque: OpaqueSettings {
                max_mode13_partitions_to_try: 1,
                max_mode0_partitions_to_try: 1,
                max_mode2_partitions_to_try: 1,
                use_mode: [true; 7],
            },
            alpha: AlphaSettings {
                max_mode7_partitions_to_try: 1,
                mode67_error_weight_mul: [1; 4],
                use_mode4: true,
                use_mode5: true,
                use_mode6: true,
                use_mode7: true,
                use_mode4_rotation: true,
                use_mode5_rotation: true,
            },
        }
    }

    /// `bc7e_compress_block_params_init_ultrafast`: mode 6 only for opaque blocks.
    pub(crate) fn ultrafast(perceptual: bool) -> Self {
        let mut p = Self::new(perceptual);
        p.mode6_only = true;
        p.alpha.use_mode4 = true;
        p.alpha.use_mode5 = true;
        p.alpha.use_mode7 = false;
        p.mode4_rotation_mask = 1 + 4;
        p.mode4_index_mask = 3;
        p.mode5_rotation_mask = 1;
        p.pbit_search = false;
        p.uber_level = 0;
        p
    }

    /// `bc7e_compress_block_params_init_veryfast`.
    pub(crate) fn veryfast(perceptual: bool) -> Self {
        let mut p = Self::new(perceptual);
        if perceptual {
            p.opaque.use_mode[0] = false;
            p.opaque.use_mode[2] = false;
            p.opaque.use_mode[3] = false;
            p.opaque.use_mode[4] = false;
            p.opaque.use_mode[5] = false;
        } else {
            p.opaque.use_mode[2] = false;
            p.opaque.use_mode[4] = false;
            p.opaque.use_mode[5] = false;
        }
        p.alpha.use_mode5 = false;
        p.pbit_search = false;
        p.uber_level = 0;
        p
    }

    /// `bc7e_compress_block_params_init_fast`.
    pub(crate) fn fast(perceptual: bool) -> Self {
        let mut p = Self::new(perceptual);
        p.opaque.use_mode[0] = false;
        p.opaque.use_mode[2] = false;
        if perceptual {
            p.opaque.use_mode[3] = false;
        }
        p.opaque.use_mode[4] = false;
        p.opaque.use_mode[5] = false;
        p.alpha.use_mode5 = false;
        p.opaque.max_mode13_partitions_to_try = if perceptual { 1 } else { 2 };
        p.pbit_search = false;
        p.uber_level = 0;
        p
    }

    /// `bc7e_compress_block_params_init_basic`.
    pub(crate) fn basic(perceptual: bool) -> Self {
        let mut p = Self::new(perceptual);
        if perceptual {
            p.opaque.use_mode[0] = false;
            p.opaque.use_mode[2] = false;
            p.opaque.use_mode[3] = false;
            p.opaque.use_mode[4] = false;
            p.opaque.use_mode[5] = false;
        } else {
            p.max_partitions_mode[1] = 32;
            p.max_partitions_mode[2] = 32;
            p.max_partitions_mode[3] = 32;
            p.max_partitions_mode[7] = 32;
            p.opaque.use_mode[2] = false;
        }
        p.pbit_search = false;
        p.uber_level = 1;
        p
    }

    /// `bc7e_compress_block_params_init_slow`.
    pub(crate) fn slow(perceptual: bool) -> Self {
        let mut p = Self::new(perceptual);
        p.alpha.max_mode7_partitions_to_try = 2;
        p.pbit_search = true;
        p.uber_level = 0;
        p
    }

    /// `bc7e_compress_block_params_init_veryslow`.
    pub(crate) fn veryslow(perceptual: bool) -> Self {
        let mut p = Self::new(perceptual);
        p.opaque.max_mode13_partitions_to_try = 2;
        p.opaque.max_mode0_partitions_to_try = 2;
        p.opaque.max_mode2_partitions_to_try = 2;
        p.alpha.max_mode7_partitions_to_try = 2;
        p.pbit_search = true;
        p.uber_level = 2;
        p
    }

    /// `bc7e_compress_block_params_init_slowest`.
    pub(crate) fn slowest(perceptual: bool) -> Self {
        let mut p = Self::new(perceptual);
        p.opaque.max_mode13_partitions_to_try = 4;
        p.opaque.max_mode0_partitions_to_try = 4;
        p.opaque.max_mode2_partitions_to_try = 4;
        p.alpha.max_mode7_partitions_to_try = 4;
        p.pbit_search = true;
        p.uber_level = 4;
        p
    }
}

// ---------------------------------------------------------------------------------------------
// Scalar helpers (C++ semantics).

/// `(int)x` for a float (`cvttss2si`): out of range / NaN gives `INT_MIN`.
#[inline]
fn f2i(x: f32) -> i32 {
    if (-2147483648.0..2147483648.0).contains(&x) {
        x as i32
    } else {
        i32::MIN
    }
}

/// `(int)x` for a double (`cvttsd2si`): out of range / NaN gives `INT_MIN`.
#[inline]
fn d2i(x: f64) -> i32 {
    if x > -2147483649.0 && x < 2147483648.0 {
        x as i32
    } else {
        i32::MIN
    }
}

/// `(int64_t)x` for a float (`cvttss2si` 64-bit): out of range / NaN gives `INT64_MIN`.
#[inline]
fn f2i64(x: f32) -> i64 {
    if (-9223372036854775808.0..9223372036854775808.0).contains(&x) {
        x as i64
    } else {
        i64::MIN
    }
}

/// `(int)floor(x)` with `floor` resolving to the `double` overload.
#[inline]
fn floor_to_int(x: f32) -> i32 {
    d2i(f64::from(x).floor())
}

/// `1.0f / sqrt(s)`: the C++ calls `double sqrt(double)`, so the division happens in double.
#[inline]
fn recip_sqrt(s: f32) -> f32 {
    (1.0f64 / f64::from(s).sqrt()) as f32
}

#[inline]
fn minf(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

#[inline]
fn maxf(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[inline]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    minf(maxf(v, lo), hi)
}

#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    let v = if v > lo { v } else { lo };
    if v < hi { v } else { hi }
}

#[inline]
fn absf(v: f32) -> f32 {
    if v < 0.0 { -v } else { v }
}

#[inline]
fn saturate(v: f32) -> f32 {
    clampf(v, 0.0, 1.0)
}

#[inline]
fn iabs32(v: i32) -> i32 {
    v.wrapping_abs()
}

#[inline]
fn v4_from_color(c: &Cq) -> V4 {
    [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32]
}

#[inline]
fn v4_add(a: &V4, b: &V4) -> V4 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

#[inline]
fn v4_sub(a: &V4, b: &V4) -> V4 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[inline]
fn v4_mul(a: &V4, s: f32) -> V4 {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

#[inline]
fn v4_dot(a: &V4, b: &V4) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

#[inline]
fn v4_saturate(a: &V4) -> V4 {
    [
        saturate(a[0]),
        saturate(a[1]),
        saturate(a[2]),
        saturate(a[3]),
    ]
}

#[inline]
fn v4_normalize(v: &mut V4) {
    let s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    if s != 0.0 {
        let s = recip_sqrt(s);
        v[0] *= s;
        v[1] *= s;
        v[2] *= s;
        v[3] *= s;
    }
}

// ---------------------------------------------------------------------------------------------
// One-color optimal endpoint tables (`bc7e_compress_block_init`).

#[derive(Clone, Copy, Default)]
struct EndpointErr {
    err: u16,
    lo: u8,
    hi: u8,
}

struct Luts {
    /// `[c][hp][lp]`
    mode0: [[[EndpointErr; 2]; 2]; 256],
    /// `[c][pbit]`
    mode1: [[EndpointErr; 2]; 256],
    /// `[c][hp][lp]`
    mode6: [[[EndpointErr; 2]; 2]; 256],
    /// `[c][hp][lp]` (upstream keeps the best entry across the pbit loops)
    mode7: [[[EndpointErr; 2]; 2]; 256],
    mode4_3: [u32; 256],
    mode4_2: [u32; 256],
    mode5: [u32; 256],
}

/// Best `(lo, hi)` endpoint pair reproducing `c` at a fixed weight; `expand` maps a quantized
/// endpoint to 8 bits. Strict `<` keeps the first minimum like the C++.
fn best_pair(
    best: &mut EndpointErr,
    c: i32,
    levels: u32,
    weight: u32,
    lo_expand: impl Fn(u32) -> u32,
    hi_expand: impl Fn(u32) -> u32,
) {
    for l in 0..levels {
        let low = lo_expand(l);
        for h in 0..levels {
            let high = hi_expand(h);
            let k = ((low * (64 - weight) + high * weight + 32) >> 6) as i32;
            let err = (k - c) * (k - c);
            if err < i32::from(best.err) {
                *best = EndpointErr {
                    err: err as u16,
                    lo: l as u8,
                    hi: h as u8,
                };
            }
        }
    }
}

fn build_luts() -> Luts {
    let unset = EndpointErr {
        err: u16::MAX,
        lo: 0,
        hi: 0,
    };
    let mut t = Luts {
        mode0: [[[unset; 2]; 2]; 256],
        mode1: [[unset; 2]; 256],
        mode6: [[[unset; 2]; 2]; 256],
        mode7: [[[unset; 2]; 2]; 256],
        mode4_3: [0; 256],
        mode4_2: [0; 256],
        mode5: [0; 256],
    };
    for c in 0..256usize {
        let ci = c as i32;
        // Mode 0: 444.1
        for hp in 0..2u32 {
            for lp in 0..2u32 {
                let mut best = unset;
                let e = |v: u32, p: u32| {
                    let x = ((v << 1) | p) << 3;
                    x | (x >> 5)
                };
                best_pair(
                    &mut best,
                    ci,
                    16,
                    W3[MODE_0_OPTIMAL_INDEX],
                    |l| e(l, lp),
                    |h| e(h, hp),
                );
                t.mode0[c][hp as usize][lp as usize] = best;
            }
        }
        // Mode 1: 666.1
        for lp in 0..2u32 {
            let mut best = unset;
            let e = |v: u32| {
                let x = ((v << 1) | lp) << 1;
                x | (x >> 7)
            };
            best_pair(&mut best, ci, 64, W3[MODE_1_OPTIMAL_INDEX], e, e);
            t.mode1[c][lp as usize] = best;
        }
        // Mode 6: 777.1 4-bit indices
        for hp in 0..2u32 {
            for lp in 0..2u32 {
                let mut best = unset;
                best_pair(
                    &mut best,
                    ci,
                    128,
                    W4[MODE_6_OPTIMAL_INDEX],
                    |l| (l << 1) | lp,
                    |h| (h << 1) | hp,
                );
                t.mode6[c][hp as usize][lp as usize] = best;
            }
        }
        // Mode 5: 777 2-bit indices
        let mut best = unset;
        let e7 = |v: u32| {
            let x = v << 1;
            x | (x >> 7)
        };
        best_pair(&mut best, ci, 128, W2[MODE_5_OPTIMAL_INDEX], e7, e7);
        t.mode5[c] = u32::from(best.lo) | (u32::from(best.hi) << 8);
        // Mode 4: 555 3-bit and 2-bit indices
        let e5 = |v: u32| {
            let x = v << 3;
            x | (x >> 5)
        };
        let mut best = unset;
        best_pair(&mut best, ci, 32, W3[MODE_4_OPTIMAL_INDEX3], e5, e5);
        t.mode4_3[c] = u32::from(best.lo) | (u32::from(best.hi) << 8);
        let mut best = unset;
        best_pair(&mut best, ci, 32, W2[MODE_4_OPTIMAL_INDEX2], e5, e5);
        t.mode4_2[c] = u32::from(best.lo) | (u32::from(best.hi) << 8);
        // Mode 7: 555.1 2-bit indices; `best` is not reset between the pbit combinations.
        let mut best = unset;
        for hp in 0..2u32 {
            for lp in 0..2u32 {
                let e = |v: u32, p: u32| {
                    let x = ((v << 1) | p) << 2;
                    x | (x >> 6)
                };
                best_pair(
                    &mut best,
                    ci,
                    32,
                    W2[MODE_7_OPTIMAL_INDEX],
                    |l| e(l, lp),
                    |h| e(h, hp),
                );
                t.mode7[c][hp as usize][lp as usize] = best;
            }
        }
    }
    t
}

fn luts() -> &'static Luts {
    static LUTS: OnceLock<Box<Luts>> = OnceLock::new();
    LUTS.get_or_init(|| Box::new(build_luts()))
}

// ---------------------------------------------------------------------------------------------
// Least squares endpoint fits (normal equations).

/// Accumulated `z00, z10, z11` and per-channel `q00`, `t` sums.
#[inline]
fn lsq_sums(sels: &[i32], swx: &[[f32; 4]], px: &[Cq], nc: usize) -> ([f32; 3], V4, V4) {
    let (mut z00, mut z10, mut z11) = (0f32, 0f32, 0f32);
    let mut q00 = [0f32; 4];
    let mut t = [0f32; 4];
    for (&sel, c) in sels.iter().zip(px) {
        let w = &swx[sel as usize];
        z00 += w[0];
        z10 += w[1];
        z11 += w[2];
        for j in 0..nc {
            q00[j] += w[3] * c[j] as f32;
            t[j] += c[j] as f32;
        }
    }
    ([z00, z10, z11], q00, t)
}

/// Solves the 2x2 system for the channels `chans`; returns `(xl, xh)`.
#[inline]
fn lsq_solve(z: [f32; 3], q00: &V4, t: &V4, chans: std::ops::Range<usize>) -> (V4, V4) {
    let [z00, z10, z11] = z;
    let z01 = z10;
    let mut det = z00 * z11 - z01 * z10;
    if det != 0.0 {
        det = 1.0 / det;
    }
    let iz00 = z11 * det;
    let iz01 = -z01 * det;
    let iz10 = -z10 * det;
    let iz11 = z00 * det;
    let mut xl = [0f32; 4];
    let mut xh = [0f32; 4];
    for j in chans {
        let q10 = t[j] - q00[j];
        xl[j] = iz00 * q00[j] + iz01 * q10;
        xh[j] = iz10 * q00[j] + iz11 * q10;
    }
    (xl, xh)
}

/// `compute_least_squares_endpoints_rgba` / `_rgb` (alpha set to 255 for RGB) as the callers use
/// them: endpoints in 0..255 units.
fn lsq_endpoints(p: &CcParams, sels: &[i32], px: &[Cq]) -> (V4, V4) {
    if p.has_alpha {
        let (z, q00, t) = lsq_sums(sels, p.swx, px, 4);
        lsq_solve(z, &q00, &t, 0..4)
    } else {
        let (z, q00, t) = lsq_sums(sels, p.swx, px, 3);
        let (mut xl, mut xh) = lsq_solve(z, &q00, &t, 0..3);
        xl[3] = 255.0;
        xh[3] = 255.0;
        (xl, xh)
    }
}

/// `compute_least_squares_endpoints_a`.
fn lsq_endpoints_a(sels: &[i32], swx: &[[f32; 4]], px: &[Cq]) -> (f32, f32) {
    let (mut z00, mut z10, mut z11) = (0f32, 0f32, 0f32);
    let (mut q00, mut t) = (0f32, 0f32);
    for (&sel, c) in sels.iter().zip(px) {
        let w = &swx[sel as usize];
        z00 += w[0];
        z10 += w[1];
        z11 += w[2];
        q00 += w[3] * c[3] as f32;
        t += c[3] as f32;
    }
    let q00 = [0.0, 0.0, 0.0, q00];
    let t = [0.0, 0.0, 0.0, t];
    let (xl, xh) = lsq_solve([z00, z10, z11], &q00, &t, 3..4);
    (xl[3], xh[3])
}

// ---------------------------------------------------------------------------------------------
// Color cell compressor.

#[derive(Clone, Copy)]
struct CcParams {
    num_selector_weights: u32,
    sw: &'static [u32],
    swx: &'static [[f32; 4]],
    comp_bits: u32,
    weights: [u32; 4],
    has_alpha: bool,
    has_pbits: bool,
    endpoints_share_pbit: bool,
    perceptual: bool,
}

impl CcParams {
    /// `color_cell_compressor_params_clear`.
    fn cleared() -> Self {
        Self {
            num_selector_weights: 0,
            sw: &[],
            swx: &[],
            comp_bits: 0,
            weights: [1; 4],
            has_alpha: false,
            has_pbits: false,
            endpoints_share_pbit: false,
            perceptual: false,
        }
    }

    fn set_selector_weights(&mut self, n: u32) {
        self.num_selector_weights = n;
        match n {
            16 => {
                self.sw = &W4;
                self.swx = &W4X;
            }
            8 => {
                self.sw = &W3;
                self.swx = &W3X;
            }
            _ => {
                self.sw = &W2;
                self.swx = &W2X;
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct CcResults {
    best_overall_err: u64,
    low: Cq,
    high: Cq,
    pbits: [u32; 2],
    /// The result came from the one-color optimal endpoint tables.
    used_lut: bool,
}

#[inline]
fn scale_color(c: &Cq, p: &CcParams) -> Cq {
    let n = p.comp_bits + u32::from(p.has_pbits);
    let mut r = [0i32; 4];
    for i in 0..4 {
        let mut v = (c[i] as u32) << (8 - n);
        v |= v >> n;
        r[i] = v as i32;
    }
    r
}

const PR_WEIGHT: f32 = (0.5f32 / (1.0f32 - 0.2126f32)) * (0.5f32 / (1.0f32 - 0.2126f32));
const PB_WEIGHT: f32 = (0.5f32 / (1.0f32 - 0.0722f32)) * (0.5f32 / (1.0f32 - 0.0722f32));

#[inline]
fn color_distance_rgb(e1: &Cq, e2: &Cq, perceptual: bool, w: &[u32; 4]) -> u64 {
    let f = if perceptual {
        let l1 = e1[0] as f32 * 0.2126 + e1[1] as f32 * 0.7152 + e1[2] as f32 * 0.0722;
        let cr1 = e1[0] as f32 - l1;
        let cb1 = e1[2] as f32 - l1;
        let l2 = e2[0] as f32 * 0.2126 + e2[1] as f32 * 0.7152 + e2[2] as f32 * 0.0722;
        let cr2 = e2[0] as f32 - l2;
        let cb2 = e2[2] as f32 - l2;
        let dl = l1 - l2;
        let dcr = cr1 - cr2;
        let dcb = cb1 - cb2;
        w[0] as f32 * (dl * dl)
            + w[1] as f32 * PR_WEIGHT * (dcr * dcr)
            + w[2] as f32 * PB_WEIGHT * (dcb * dcb)
    } else {
        let dr = e1[0] as f32 - e2[0] as f32;
        let dg = e1[1] as f32 - e2[1] as f32;
        let db = e1[2] as f32 - e2[2] as f32;
        w[0] as f32 * dr * dr + w[1] as f32 * dg * dg + w[2] as f32 * db * db
    };
    f2i64(f) as u64
}

#[inline]
fn color_distance_rgba(e1: &Cq, e2: &Cq, perceptual: bool, w: &[u32; 4]) -> u64 {
    let da = e1[3] as f32 - e2[3] as f32;
    let a_err = w[3] as f32 * (da * da);
    let f = if perceptual {
        let l1 = e1[0] as f32 * 0.2126 + e1[1] as f32 * 0.7152 + e1[2] as f32 * 0.0722;
        let cr1 = e1[0] as f32 - l1;
        let cb1 = e1[2] as f32 - l1;
        let l2 = e2[0] as f32 * 0.2126 + e2[1] as f32 * 0.7152 + e2[2] as f32 * 0.0722;
        let cr2 = e2[0] as f32 - l2;
        let cb2 = e2[2] as f32 - l2;
        let dl = l1 - l2;
        let dcr = cr1 - cr2;
        let dcb = cb1 - cb2;
        w[0] as f32 * (dl * dl)
            + w[1] as f32 * PR_WEIGHT * (dcr * dcr)
            + w[2] as f32 * PB_WEIGHT * (dcb * dcb)
            + a_err
    } else {
        let dr = e1[0] as f32 - e2[0] as f32;
        let dg = e1[1] as f32 - e2[1] as f32;
        let db = e1[2] as f32 - e2[2] as f32;
        w[0] as f32 * dr * dr + w[1] as f32 * dg * dg + w[2] as f32 * db * db + a_err
    };
    f2i64(f) as u64
}

/// Total error of the interpolated color `c` against every pixel.
#[inline]
fn one_color_error(c: &Cq, p: &CcParams, px: &[Cq], rgba: bool) -> u64 {
    let mut total = 0u64;
    for q in px {
        let e = if rgba {
            color_distance_rgba(c, q, p.perceptual, &p.weights)
        } else {
            color_distance_rgb(c, q, p.perceptual, &p.weights)
        };
        total = total.wrapping_add(e);
    }
    total
}

/// `pack_mode1_to_one_color`.
fn pack_mode1_to_one_color(
    p: &CcParams,
    res: &mut CcResults,
    rgb: [usize; 3],
    sels: &mut [i32],
    px: &[Cq],
) -> u64 {
    let t = &luts().mode1;
    let mut best_err = u32::MAX;
    let mut best_p = 0usize;
    for pb in 0..2 {
        let err = u32::from(t[rgb[0]][pb].err)
            + u32::from(t[rgb[1]][pb].err)
            + u32::from(t[rgb[2]][pb].err);
        if err < best_err {
            best_err = err;
            best_p = pb;
        }
    }
    let e = [t[rgb[0]][best_p], t[rgb[1]][best_p], t[rgb[2]][best_p]];
    res.low = [e[0].lo.into(), e[1].lo.into(), e[2].lo.into(), 0];
    res.high = [e[0].hi.into(), e[1].hi.into(), e[2].hi.into(), 0];
    res.pbits = [best_p as u32, 0];
    for s in sels.iter_mut().take(px.len()) {
        *s = MODE_1_OPTIMAL_INDEX as i32;
    }
    let mut c = [0i32; 4];
    for i in 0..3 {
        let mut low = (((res.low[i] as u32) << 1) | res.pbits[0]) << 1;
        low |= low >> 7;
        let mut high = (((res.high[i] as u32) << 1) | res.pbits[0]) << 1;
        high |= high >> 7;
        let w = W3[MODE_1_OPTIMAL_INDEX];
        c[i] = ((low * (64 - w) + high * w + 32) >> 6) as i32;
    }
    c[3] = 255;
    let total = one_color_error(&c, p, px, false);
    res.best_overall_err = total;
    total
}

/// `pack_mode24_to_one_color` (modes 2 and 4: 555 endpoints, no pbits).
fn pack_mode24_to_one_color(
    p: &CcParams,
    res: &mut CcResults,
    rgb: [usize; 3],
    sels: &mut [i32],
    px: &[Cq],
) -> u64 {
    let l = luts();
    let three_bit = p.num_selector_weights == 8;
    let t = if three_bit { &l.mode4_3 } else { &l.mode4_2 };
    let e = [t[rgb[0]], t[rgb[1]], t[rgb[2]]];
    res.low = [
        (e[0] & 0xFF) as i32,
        (e[1] & 0xFF) as i32,
        (e[2] & 0xFF) as i32,
        0,
    ];
    res.high = [
        (e[0] >> 8) as i32,
        (e[1] >> 8) as i32,
        (e[2] >> 8) as i32,
        0,
    ];
    let (idx, w) = if three_bit {
        (MODE_4_OPTIMAL_INDEX3, W3[MODE_4_OPTIMAL_INDEX3])
    } else {
        (MODE_4_OPTIMAL_INDEX2, W2[MODE_4_OPTIMAL_INDEX2])
    };
    for s in sels.iter_mut().take(px.len()) {
        *s = idx as i32;
    }
    let mut c = [0i32; 4];
    for i in 0..3 {
        let mut low = (res.low[i] as u32) << 3;
        low |= low >> 5;
        let mut high = (res.high[i] as u32) << 3;
        high |= high >> 5;
        c[i] = ((low * (64 - w) + high * w + 32) >> 6) as i32;
    }
    c[3] = 255;
    let total = one_color_error(&c, p, px, false);
    res.best_overall_err = total;
    total
}

/// `pack_mode0_to_one_color`.
fn pack_mode0_to_one_color(
    p: &CcParams,
    res: &mut CcResults,
    rgb: [usize; 3],
    sels: &mut [i32],
    px: &[Cq],
) -> u64 {
    let t = &luts().mode0;
    let mut best_err = u32::MAX;
    let mut best_p = 0usize;
    for pb in 0..4usize {
        let (hp, lp) = (pb >> 1, pb & 1);
        let err = u32::from(t[rgb[0]][hp][lp].err)
            + u32::from(t[rgb[1]][hp][lp].err)
            + u32::from(t[rgb[2]][hp][lp].err);
        if err < best_err {
            best_err = err;
            best_p = pb;
        }
    }
    let (hp, lp) = (best_p >> 1, best_p & 1);
    let e = [t[rgb[0]][hp][lp], t[rgb[1]][hp][lp], t[rgb[2]][hp][lp]];
    res.low = [e[0].lo.into(), e[1].lo.into(), e[2].lo.into(), 0];
    res.high = [e[0].hi.into(), e[1].hi.into(), e[2].hi.into(), 0];
    res.pbits = [lp as u32, hp as u32];
    for s in sels.iter_mut().take(px.len()) {
        *s = MODE_0_OPTIMAL_INDEX as i32;
    }
    let mut c = [0i32; 4];
    for i in 0..3 {
        let mut low = (((res.low[i] as u32) << 1) | res.pbits[0]) << 3;
        low |= low >> 5;
        let mut high = (((res.high[i] as u32) << 1) | res.pbits[1]) << 3;
        high |= high >> 5;
        let w = W3[MODE_0_OPTIMAL_INDEX];
        c[i] = ((low * (64 - w) + high * w + 32) >> 6) as i32;
    }
    c[3] = 255;
    let total = one_color_error(&c, p, px, false);
    res.best_overall_err = total;
    total
}

/// `pack_mode6_to_one_color` / `pack_mode7_to_one_color` (RGBA, two unique pbits).
fn pack_mode67_to_one_color(
    mode: u32,
    p: &CcParams,
    res: &mut CcResults,
    rgba: [usize; 4],
    sels: &mut [i32],
    px: &[Cq],
) -> u64 {
    let l = luts();
    let (t, idx, w) = if mode == 6 {
        (&l.mode6, MODE_6_OPTIMAL_INDEX, W4[MODE_6_OPTIMAL_INDEX])
    } else {
        (&l.mode7, MODE_7_OPTIMAL_INDEX, W2[MODE_7_OPTIMAL_INDEX])
    };
    let mut best_err = u32::MAX;
    let mut best_p = 0usize;
    for pb in 0..4usize {
        let (hp, lp) = (pb >> 1, pb & 1);
        let err = u32::from(t[rgba[0]][hp][lp].err)
            + u32::from(t[rgba[1]][hp][lp].err)
            + u32::from(t[rgba[2]][hp][lp].err)
            + u32::from(t[rgba[3]][hp][lp].err);
        if err < best_err {
            best_err = err;
            best_p = pb;
        }
    }
    let (hp, lp) = (best_p >> 1, best_p & 1);
    let e = rgba.map(|c| t[c][hp][lp]);
    res.low = e.map(|x| i32::from(x.lo));
    res.high = e.map(|x| i32::from(x.hi));
    res.pbits = [lp as u32, hp as u32];
    for s in sels.iter_mut().take(px.len()) {
        *s = idx as i32;
    }
    let mut c = [0i32; 4];
    for i in 0..4 {
        let low = ((res.low[i] as u32) << 1) | res.pbits[0];
        let high = ((res.high[i] as u32) << 1) | res.pbits[1];
        c[i] = ((low * (64 - w) + high * w + 32) >> 6) as i32;
    }
    let total = one_color_error(&c, p, px, true);
    res.best_overall_err = total;
    total
}

/// Dispatches to the one-color packer of `mode`.
fn pack_one_color(
    mode: u32,
    p: &CcParams,
    res: &mut CcResults,
    rgba: [usize; 4],
    sels: &mut [i32],
    px: &[Cq],
) -> u64 {
    let rgb = [rgba[0], rgba[1], rgba[2]];
    match mode {
        0 => pack_mode0_to_one_color(p, res, rgb, sels, px),
        1 => pack_mode1_to_one_color(p, res, rgb, sels, px),
        6 | 7 => pack_mode67_to_one_color(mode, p, res, rgba, sels, px),
        _ => pack_mode24_to_one_color(p, res, rgb, sels, px),
    }
}

/// `evaluate_solution`: selects indices for the endpoints, keeps the result if it is better.
#[allow(clippy::too_many_arguments)]
fn evaluate_solution(
    low: &Cq,
    high: &Cq,
    pbits: [u32; 2],
    p: &CcParams,
    res: &mut CcResults,
    sels: &mut [i32; 16],
    tmp: &mut [i32; 16],
    px: &[Cq],
) -> u64 {
    let mut qmin = *low;
    let mut qmax = *high;
    if p.has_pbits {
        let (minp, maxp) = if p.endpoints_share_pbit {
            (pbits[0], pbits[0])
        } else {
            (pbits[0], pbits[1])
        };
        for c in 0..4 {
            qmin[c] = (((low[c] as u32) << 1) | minp) as i32;
            qmax[c] = (((high[c] as u32) << 1) | maxp) as i32;
        }
    }
    let amin = scale_color(&qmin, p);
    let amax = scale_color(&qmax, p);
    let n = p.num_selector_weights as usize;
    let nc = if p.has_alpha { 4 } else { 3 };

    let mut total = 0f32;
    let wr = p.weights[0] as f32;
    let mut wg = p.weights[1] as f32;
    let mut wb = p.weights[2] as f32;
    let wa = p.weights[3] as f32;

    let mut wc = [[0f32; 4]; 16];
    wc[0] = v4_from_color(&amin);
    wc[n - 1] = v4_from_color(&amax);
    for i in 1..n - 1 {
        let s = p.sw[i] as f32;
        for j in 0..nc {
            wc[i][j] = ((wc[0][j] * (64.0 - s) + wc[n - 1][j] * s + 32.0) * (1.0 / 64.0)).floor();
        }
    }

    if !p.perceptual {
        if n == 16 {
            // Project onto the endpoint line, then pick the better of two neighbors.
            let mut lr = amin[0] as f32;
            let mut lg = amin[1] as f32;
            let mut lb = amin[2] as f32;
            let mut la = amin[3] as f32;
            let dr = amax[0] as f32 - lr;
            let dg = amax[1] as f32 - lg;
            let db = amax[2] as f32 - lb;
            let da = amax[3] as f32 - la;
            let f = if p.has_alpha {
                n as f32 / (dr * dr + dg * dg + db * db + da * da + DENOM_BIAS)
            } else {
                n as f32 / (dr * dr + dg * dg + db * db + DENOM_BIAS)
            };
            lr *= -dr;
            lg *= -dg;
            lb *= -db;
            la *= -da;
            for (i, c) in px.iter().enumerate() {
                let r = c[0] as f32;
                let g = c[1] as f32;
                let b = c[2] as f32;
                let a = c[3] as f32;
                let d = if p.has_alpha {
                    (r * dr + lr) + (g * dg + lg) + (b * db + lb) + (a * da + la)
                } else {
                    (r * dr + lr) + (g * dg + lg) + (b * db + lb)
                };
                let best_sel = clampf((d * f + 0.5).floor(), 1.0, (n - 1) as f32);
                let best_sel0 = best_sel - 1.0;
                let w0 = &wc[f2i(best_sel0) as usize];
                let w1 = &wc[f2i(best_sel) as usize];
                let (dr0, dg0, db0) = (w0[0] - r, w0[1] - g, w0[2] - b);
                let (dr1, dg1, db1) = (w1[0] - r, w1[1] - g, w1[2] - b);
                let (err0, err1) = if p.has_alpha {
                    let da0 = w0[3] - a;
                    let da1 = w1[3] - a;
                    (
                        (wr * dr0 * dr0) + (wg * dg0 * dg0) + (wb * db0 * db0) + (wa * da0 * da0),
                        (wr * dr1 * dr1) + (wg * dg1 * dg1) + (wb * db1 * db1) + (wa * da1 * da1),
                    )
                } else {
                    (
                        wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0,
                        wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1,
                    )
                };
                let min_err = minf(err0, err1);
                total += min_err;
                tmp[i] = f2i(if min_err == err0 { best_sel0 } else { best_sel });
            }
        } else {
            // N == 8 / 4: exhaustive; ties pick the highest index (the C++ select chains).
            for (i, c) in px.iter().enumerate() {
                let pr = c[0] as f32;
                let pg = c[1] as f32;
                let pb = c[2] as f32;
                let pa = c[3] as f32;
                let mut best_err = f32::INFINITY;
                let mut best_sel = 0i32;
                for (j, w) in wc.iter().enumerate().take(n) {
                    let dr = w[0] - pr;
                    let dg = w[1] - pg;
                    let db = w[2] - pb;
                    let err = if p.has_alpha {
                        let da = w[3] - pa;
                        wr * dr * dr + wg * dg * dg + wb * db * db + wa * da * da
                    } else {
                        wr * dr * dr + wg * dg * dg + wb * db * db
                    };
                    if err <= best_err {
                        best_err = err;
                        best_sel = j as i32;
                    }
                }
                total += best_err;
                tmp[i] = best_sel;
            }
        }
    } else {
        wg *= PR_WEIGHT;
        wb *= PB_WEIGHT;
        let mut wy = [0f32; 16];
        let mut wcr = [0f32; 16];
        let mut wcb = [0f32; 16];
        for i in 0..n {
            let (r, g, b) = (wc[i][0], wc[i][1], wc[i][2]);
            let y = r * 0.2126 + g * 0.7152 + b * 0.0722;
            wy[i] = y;
            wcr[i] = r - y;
            wcb[i] = b - y;
        }
        for (i, c) in px.iter().enumerate() {
            let r = c[0] as f32;
            let g = c[1] as f32;
            let b = c[2] as f32;
            let a = c[3] as f32;
            let y = r * 0.2126 + g * 0.7152 + b * 0.0722;
            let cr = r - y;
            let cb = b - y;
            let mut best_err = 1e10f32;
            let mut best_sel = 0i32;
            for j in 0..n {
                let dl = y - wy[j];
                let dcr = cr - wcr[j];
                let dcb = cb - wcb[j];
                let err = if p.has_alpha {
                    let da = a - wc[j][3];
                    (wr * dl * dl) + (wg * dcr * dcr) + (wb * dcb * dcb) + (wa * da * da)
                } else {
                    (wr * dl * dl) + (wg * dcr * dcr) + (wb * dcb * dcb)
                };
                if err < best_err {
                    best_err = err;
                    best_sel = j as i32;
                }
            }
            total += best_err;
            tmp[i] = best_sel;
        }
    }

    let total_err = f2i64(total) as u64;
    if total_err < res.best_overall_err {
        res.best_overall_err = total_err;
        res.low = *low;
        res.high = *high;
        res.pbits = pbits;
        let np = px.len();
        sels[..np].copy_from_slice(&tmp[..np]);
    }
    total_err
}

/// `fixDegenerateEndpoints` (modes 1 and 4).
fn fix_degenerate_endpoints(
    mode: u32,
    tmin: &mut Cq,
    tmax: &mut Cq,
    xl: &V4,
    xh: &V4,
    iscale: u32,
) {
    if mode != 1 && mode != 4 {
        return;
    }
    let half = (iscale >> 1) as i32;
    let isc = iscale as i32;
    for i in 0..3 {
        if tmin[i] == tmax[i] && absf(xl[i] - xh[i]) > 0.0 {
            if tmin[i] > half {
                if tmin[i] > 0 {
                    tmin[i] -= 1;
                } else if tmax[i] < isc {
                    tmax[i] += 1;
                }
            } else if tmax[i] < isc {
                tmax[i] += 1;
            } else if tmin[i] > 0 {
                tmin[i] -= 1;
            }
            if mode == 4 {
                if tmin[i] > half {
                    if tmax[i] < isc {
                        tmax[i] += 1;
                    } else if tmin[i] > 0 {
                        tmin[i] -= 1;
                    }
                } else if tmin[i] > 0 {
                    tmin[i] -= 1;
                } else if tmax[i] < isc {
                    tmax[i] += 1;
                }
            }
        }
    }
}

/// Quantizes `x` (0..1) to `comp_bits + 1` bits with the pbit forced to `pb`.
#[inline]
fn quant_pbit(x: f32, scalep: f32, iscalep: i32, pb: i32) -> i32 {
    let v = f2i((x * scalep - pb as f32) / 2.0 + 0.5)
        .wrapping_mul(2)
        .wrapping_add(pb);
    clampi(v, pb, iscalep - 1 + pb)
}

/// `find_optimal_solution`: quantizes the float endpoints and evaluates the candidates.
#[allow(clippy::too_many_arguments)]
fn find_optimal_solution(
    mode: u32,
    xl_in: &V4,
    xh_in: &V4,
    p: &CcParams,
    res: &mut CcResults,
    sels: &mut [i32; 16],
    tmp: &mut [i32; 16],
    pbit_search: bool,
    px: &[Cq],
) -> u64 {
    let xl = v4_saturate(xl_in);
    let xh = v4_saturate(xh_in);

    if p.has_pbits {
        let iscalep = (1i32 << (p.comp_bits + 1)) - 1;
        let scalep = iscalep as f32;
        if pbit_search {
            // Compensated rounding + pbit search.
            let mut lo = [[0i32; 4]; 2];
            let mut hi = [[0i32; 4]; 2];
            for pb in 0..2i32 {
                for c in 0..4 {
                    lo[pb as usize][c] = quant_pbit(xl[c], scalep, iscalep, pb) >> 1;
                    hi[pb as usize][c] = quant_pbit(xh[c], scalep, iscalep, pb) >> 1;
                }
            }
            let half = (iscalep >> 1) as u32;
            let [lo0, lo1] = &mut lo;
            let [hi0, hi1] = &mut hi;
            fix_degenerate_endpoints(mode, lo0, hi0, &xl, &xh, half);
            fix_degenerate_endpoints(mode, lo1, hi1, &xl, &xh, half);
            if !p.endpoints_share_pbit {
                evaluate_solution(lo0, hi0, [0, 0], p, res, sels, tmp, px);
                evaluate_solution(lo0, hi1, [0, 1], p, res, sels, tmp, px);
                evaluate_solution(lo1, hi0, [1, 0], p, res, sels, tmp, px);
                evaluate_solution(lo1, hi1, [1, 1], p, res, sels, tmp, px);
            } else {
                evaluate_solution(lo0, hi0, [0, 0], p, res, sels, tmp, px);
                evaluate_solution(lo1, hi1, [1, 1], p, res, sels, tmp, px);
            }
        } else {
            // Compensated rounding.
            let total_comps = if p.has_alpha { 4 } else { 3 };
            let mut best_pbits = [0u32; 2];
            let mut best_min = [0i32; 4];
            let mut best_max = [0i32; 4];
            if !p.endpoints_share_pbit {
                let mut best_err0 = 1e9f32;
                let mut best_err1 = 1e9f32;
                for pb in 0..2i32 {
                    let mut xmin = [0i32; 4];
                    let mut xmax = [0i32; 4];
                    for c in 0..4 {
                        xmin[c] = quant_pbit(xl[c], scalep, iscalep, pb);
                        xmax[c] = quant_pbit(xh[c], scalep, iscalep, pb);
                    }
                    let sl = scale_color(&xmin, p);
                    let sh = scale_color(&xmax, p);
                    let mut err0 = 0f32;
                    let mut err1 = 0f32;
                    for i in 0..total_comps {
                        let d0 = sl[i] as f32 - xl[i] * 255.0;
                        let d1 = sh[i] as f32 - xh[i] * 255.0;
                        err0 += d0 * d0;
                        err1 += d1 * d1;
                    }
                    if err0 < best_err0 {
                        best_err0 = err0;
                        best_pbits[0] = pb as u32;
                        best_min = xmin.map(|v| v >> 1);
                    }
                    if err1 < best_err1 {
                        best_err1 = err1;
                        best_pbits[1] = pb as u32;
                        best_max = xmax.map(|v| v >> 1);
                    }
                }
            } else {
                let mut best_err = 1e9f32;
                for pb in 0..2i32 {
                    let mut xmin = [0i32; 4];
                    let mut xmax = [0i32; 4];
                    for c in 0..4 {
                        xmin[c] = quant_pbit(xl[c], scalep, iscalep, pb);
                        xmax[c] = quant_pbit(xh[c], scalep, iscalep, pb);
                    }
                    let sl = scale_color(&xmin, p);
                    let sh = scale_color(&xmax, p);
                    let mut err = 0f32;
                    for i in 0..total_comps {
                        let d0 = sl[i] as f32 / 255.0 - xl[i];
                        let d1 = sh[i] as f32 / 255.0 - xh[i];
                        err += d0 * d0 + d1 * d1;
                    }
                    if err < best_err {
                        best_err = err;
                        best_pbits = [pb as u32, pb as u32];
                        best_min = xmin.map(|v| v >> 1);
                        best_max = xmax.map(|v| v >> 1);
                    }
                }
            }
            fix_degenerate_endpoints(
                mode,
                &mut best_min,
                &mut best_max,
                &xl,
                &xh,
                (iscalep >> 1) as u32,
            );
            if res.best_overall_err == u64::MAX
                || best_min != res.low
                || best_max != res.high
                || best_pbits != res.pbits
            {
                evaluate_solution(&best_min, &best_max, best_pbits, p, res, sels, tmp, px);
            }
        }
    } else {
        let iscale = (1i32 << p.comp_bits) - 1;
        let scale = iscale as f32;
        let q = |x: f32| clampi(f2i(x * scale + 0.5), 0, 255);
        let mut tmin = xl.map(q);
        let mut tmax = xh.map(q);
        fix_degenerate_endpoints(mode, &mut tmin, &mut tmax, &xl, &xh, iscale as u32);
        if res.best_overall_err == u64::MAX || tmin != res.low || tmax != res.high {
            evaluate_solution(&tmin, &tmax, [0, 0], p, res, sels, tmp, px);
        }
    }
    res.best_overall_err
}

/// `color_cell_compression`: encodes one subset (`px`) with `mode`. In mode 6 `has_alpha` is
/// only set for transparent blocks.
fn color_cell_compression(
    mode: u32,
    p: &CcParams,
    res: &mut CcResults,
    sels: &mut [i32; 16],
    comp: &Params,
    px: &[Cq],
    refinement: bool,
) -> u64 {
    let mut tmp = [0i32; 16];
    res.best_overall_err = u64::MAX;
    res.used_lut = false;
    let np = px.len();
    let lut_mode = mode <= 2 || mode == 4 || mode >= 6;

    if lut_mode && comp.use_luts {
        let c0 = px[0];
        if px[1..].iter().all(|c| *c == c0) {
            res.used_lut = true;
            let rgba = c0.map(|v| v as usize);
            return pack_one_color(mode, p, res, rgba, sels, px);
        }
    }

    let mut sum = [0f32; 4];
    for c in px {
        sum = v4_add(&sum, &v4_from_color(c));
    }
    let mean_scaled = v4_mul(&sum, 1.0 / (np as i32 as f32));
    let mean = v4_saturate(&v4_mul(&sum, 1.0 / (np as f32 * 255.0)));

    let mut axis;
    if p.has_alpha {
        let mut v = [0f32; 4];
        for (i, c) in px.iter().enumerate() {
            let color = v4_sub(&v4_from_color(c), &mean_scaled);
            let a = v4_mul(&color, color[0]);
            let b = v4_mul(&color, color[1]);
            let cc = v4_mul(&color, color[2]);
            let d = v4_mul(&color, color[3]);
            let mut n = if i != 0 { v } else { color };
            v4_normalize(&mut n);
            v[0] += v4_dot(&a, &n);
            v[1] += v4_dot(&b, &n);
            v[2] += v4_dot(&cc, &n);
            v[3] += v4_dot(&d, &n);
        }
        axis = v;
        v4_normalize(&mut axis);
    } else {
        let mut cov = [0f32; 6];
        for c in px {
            let r = c[0] as f32 - mean_scaled[0];
            let g = c[1] as f32 - mean_scaled[1];
            let b = c[2] as f32 - mean_scaled[2];
            cov[0] += r * r;
            cov[1] += r * g;
            cov[2] += r * b;
            cov[3] += g * g;
            cov[4] += g * b;
            cov[5] += b * b;
        }
        let (mut vfr, mut vfg, mut vfb) = (0.9f32, 1.0f32, 0.7f32);
        for _ in 0..3 {
            let mut r = vfr * cov[0] + vfg * cov[1] + vfb * cov[2];
            let mut g = vfr * cov[1] + vfg * cov[3] + vfb * cov[4];
            let mut b = vfr * cov[2] + vfg * cov[4] + vfb * cov[5];
            let mut m = maxf(maxf(absf(r), absf(g)), absf(b));
            if m > 1e-10 {
                m = 1.0 / m;
                r *= m;
                g *= m;
                b *= m;
            }
            vfr = r;
            vfg = g;
            vfb = b;
        }
        let len = vfr * vfr + vfg * vfg + vfb * vfb;
        if len < 1e-10 {
            axis = [0.0; 4];
        } else {
            let len = recip_sqrt(len);
            axis = [vfr * len, vfg * len, vfb * len, 0.0];
        }
    }

    if v4_dot(&axis, &axis) < 0.5 {
        axis = if p.perceptual {
            [0.213, 0.715, 0.072, if p.has_alpha { 0.715 } else { 0.0 }]
        } else {
            [1.0, 1.0, 1.0, if p.has_alpha { 1.0 } else { 0.0 }]
        };
        v4_normalize(&mut axis);
    }

    let mut l = 1e9f32;
    let mut h = -1e9f32;
    for c in px {
        let q = v4_sub(&v4_from_color(c), &mean_scaled);
        let d = v4_dot(&q, &axis);
        l = minf(l, d);
        h = maxf(h, d);
    }
    l *= 1.0 / 255.0;
    h *= 1.0 / 255.0;

    let mut min_color = v4_saturate(&v4_add(&mean, &v4_mul(&axis, l)));
    let mut max_color = v4_saturate(&v4_add(&mean, &v4_mul(&axis, h)));
    let white = [1.0f32; 4];
    if v4_dot(&min_color, &white) > v4_dot(&max_color, &white) {
        std::mem::swap(&mut min_color, &mut max_color);
    }

    let pbit_search = comp.pbit_search;
    if find_optimal_solution(
        mode,
        &min_color,
        &max_color,
        p,
        res,
        sels,
        &mut tmp,
        pbit_search,
        px,
    ) == 0
    {
        return 0;
    }
    if !refinement {
        return res.best_overall_err;
    }

    for _ in 0..comp.refinement_passes {
        let (xl, xh) = lsq_endpoints(p, &sels[..np], px);
        let xl = v4_mul(&xl, 1.0 / 255.0);
        let xh = v4_mul(&xh, 1.0 / 255.0);
        if find_optimal_solution(mode, &xl, &xh, p, res, sels, &mut tmp, pbit_search, px) == 0 {
            return 0;
        }
    }

    if comp.uber_level > 0 {
        let mut sel0 = [0i32; 16];
        sel0[..np].copy_from_slice(&sels[..np]);
        let mut sel1 = [0i32; 16];
        let nsw = p.num_selector_weights;
        let max_selector = nsw as i32 - 1;

        let mut min_sel = 16u32;
        let mut max_sel = 0u32;
        for &s in &sel0[..np] {
            min_sel = min_sel.min(s as u32);
            max_sel = max_sel.max(s as u32);
        }

        for (bit, step) in [(1u32, 0u32), (2, 1), (4, 2)] {
            if comp.uber1_mask & bit == 0 {
                continue;
            }
            for i in 0..np {
                let mut s = sel0[i] as u32;
                let inc = s == min_sel && s < nsw - 1;
                let dec = s == max_sel && s > 0;
                match step {
                    0 if inc => s += 1,
                    1 if dec => s -= 1,
                    2 if inc => s += 1,
                    2 if dec => s -= 1,
                    _ => {}
                }
                sel1[i] = s as i32;
            }
            let (xl, xh) = lsq_endpoints(p, &sel1[..np], px);
            let xl = v4_mul(&xl, 1.0 / 255.0);
            let xh = v4_mul(&xh, 1.0 / 255.0);
            if find_optimal_solution(mode, &xl, &xh, p, res, sels, &mut tmp, pbit_search, px) == 0 {
                return 0;
            }
        }

        let uber_err_thresh = ((np as u32 * 56) >> 4) as u64;
        if comp.uber_level >= 2 && res.best_overall_err > uber_err_thresh {
            let q = if comp.uber_level >= 4 {
                comp.uber_level as i32 - 2
            } else {
                1
            };
            for ly in -q..=1 {
                for hy in max_selector - 1..=max_selector + q {
                    if ly == 0 && hy == max_selector {
                        continue;
                    }
                    for i in 0..np {
                        let v = (max_selector as f32 * (sel0[i] as f32 - ly as f32)
                            / (hy as f32 - ly as f32)
                            + 0.5)
                            .floor();
                        sel1[i] = f2i(clampf(v, 0.0, max_selector as f32));
                    }
                    let (xl, xh) = lsq_endpoints(p, &sel1[..np], px);
                    let xl = v4_mul(&xl, 1.0 / 255.0);
                    let xh = v4_mul(&xh, 1.0 / 255.0);
                    let search = pbit_search && comp.uber_level >= 2;
                    if find_optimal_solution(mode, &xl, &xh, p, res, sels, &mut tmp, search, px)
                        == 0
                    {
                        return 0;
                    }
                }
            }
        }
    }

    if lut_mode && comp.use_luts {
        let mut avg = CcResults {
            best_overall_err: res.best_overall_err,
            ..CcResults::default()
        };
        let rgba = mean.map(|m| f2i(0.5 + m * 255.0) as u32 as usize);
        let avg_err = pack_one_color(mode, p, &mut avg, rgba, &mut tmp, px);
        if avg_err < res.best_overall_err {
            res.best_overall_err = avg_err;
            res.low = avg.low;
            res.high = avg.high;
            res.pbits = avg.pbits;
            sels[..np].copy_from_slice(&tmp[..np]);
            res.used_lut = true;
        }
    }

    res.best_overall_err
}

// ---------------------------------------------------------------------------------------------
// Partition estimation.

/// `color_cell_compression_est` (modes 0..3) and `color_cell_compression_est_mode7`.
fn color_cell_compression_est(mode: u32, p: &CcParams, px: &[Cq]) -> u64 {
    let alpha = mode == 7;
    let nc = if alpha { 4 } else { 3 };
    let mut lo = [255f32; 4];
    let mut hi = [0f32; 4];
    for c in px {
        for j in 0..nc {
            lo[j] = minf(lo[j], c[j] as f32);
            hi[j] = maxf(hi[j], c[j] as f32);
        }
    }
    let n: u32 = if alpha {
        4
    } else {
        1 << COLOR_INDEX_BITCOUNT[mode as usize]
    };

    let s = lo;
    let mut di = [0f32; 4];
    for j in 0..nc {
        di[j] = hi[j] - lo[j];
    }
    let fa = di;
    let (low, high) = if alpha {
        (
            fa[0] * s[0] + fa[1] * s[1] + fa[2] * s[2] + fa[3] * s[3],
            fa[0] * hi[0] + fa[1] * hi[1] + fa[2] * hi[2] + fa[3] * hi[3],
        )
    } else {
        (
            fa[0] * s[0] + fa[1] * s[1] + fa[2] * s[2],
            fa[0] * hi[0] + fa[1] * hi[1] + fa[2] * hi[2],
        )
    };
    let scale = (n as f32 - 1.0) / ((high - low) + DENOM_BIAS);
    let inv_n = 1.0 / (n as f32 - 1.0);

    let w = &p.weights;
    let weighted = if alpha {
        !p.perceptual && (w[0] != 1 || w[1] != 1 || w[2] != 1 || w[3] != 1)
    } else {
        w[0] != 1 || w[1] != 1 || w[2] != 1
    };
    let wf = w.map(|x| x as f32);

    let mut total = 0f32;
    for c in px {
        let cf = v4_from_color(c);
        let d = if alpha {
            fa[0] * cf[0] + fa[1] * cf[1] + fa[2] * cf[2] + fa[3] * cf[3]
        } else {
            fa[0] * cf[0] + fa[1] * cf[1] + fa[2] * cf[2]
        };
        let t = clampf(((d - low) * scale + 0.5).floor() * inv_n, 0.0, 1.0);
        let dr = (s[0] + di[0] * t) - cf[0];
        let dg = (s[1] + di[1] * t) - cf[1];
        let db = (s[2] + di[2] * t) - cf[2];
        let da = (s[3] + di[3] * t) - cf[3];
        total += match (weighted, alpha) {
            (true, false) => wf[0] * dr * dr + wf[1] * dg * dg + wf[2] * db * db,
            (true, true) => wf[0] * dr * dr + wf[1] * dg * dg + wf[2] * db * db + wf[3] * da * da,
            (false, false) => dr * dr + dg * dg + db * db,
            (false, true) => dr * dr + dg * dg + db * db + da * da,
        };
    }
    f2i64(total) as u64
}

/// Splits the 16 pixels into subsets: `(colors, pixel indices, counts)`.
#[inline]
fn split_subsets(px: &[Cq; 16], part: &[u8; 16]) -> ([[Cq; 16]; 3], [[u8; 16]; 3], [usize; 3]) {
    let mut colors = [[[0i32; 4]; 16]; 3];
    let mut index = [[0u8; 16]; 3];
    let mut count = [0usize; 3];
    for (i, &s) in part.iter().enumerate() {
        let s = s as usize;
        colors[s][count[s]] = px[i];
        index[s][count[s]] = i as u8;
        count[s] += 1;
    }
    (colors, index, count)
}

/// Parameters for the partition estimators.
fn estimate_params(mode: u32, comp: &Params) -> CcParams {
    let mut p = CcParams::cleared();
    let bits = COLOR_INDEX_BITCOUNT[mode as usize];
    p.set_selector_weights(1 << bits);
    p.sw = if bits == 2 { &W2 } else { &W3 };
    p.swx = &[];
    p.weights = comp.weights;
    if mode >= 6 {
        for (w, m) in p.weights.iter_mut().zip(comp.alpha.mode67_error_weight_mul) {
            *w = w.wrapping_mul(m);
        }
    }
    p.perceptual = comp.perceptual;
    p
}

/// Estimated error of `partition` for `mode`.
fn estimate_partition_err(mode: u32, p: &CcParams, px: &[Cq; 16], partition: u32) -> u64 {
    let nsub = NUM_SUBSETS[mode as usize] as usize;
    let part = if nsub == 3 {
        &PARTITION3[partition as usize]
    } else {
        &PARTITION2[partition as usize]
    };
    let (colors, _, count) = split_subsets(px, part);
    let mut total = 0u64;
    for s in 0..nsub {
        total = total.wrapping_add(color_cell_compression_est(mode, p, &colors[s][..count[s]]));
    }
    total
}

fn total_partitions(mode: u32, comp: &Params) -> u32 {
    comp.max_partitions_mode[mode as usize].min(1u32 << PARTITION_BITS[mode as usize])
}

/// `estimate_partition`: the single best partition.
fn estimate_partition(mode: u32, px: &[Cq; 16], comp: &Params) -> u32 {
    let total = total_partitions(mode, comp);
    if total <= 1 {
        return 0;
    }
    let p = estimate_params(mode, comp);
    let mut best_err = u64::MAX;
    let mut best_partition = 0;
    for partition in 0..total {
        let err = estimate_partition_err(mode, &p, px, partition);
        if err < best_err {
            best_err = err;
            best_partition = partition;
            if best_err == 0 {
                break;
            }
        }
        if NUM_SUBSETS[mode as usize] == 2
            && partition == CHECKERBOARD_PARTITION
            && best_partition != CHECKERBOARD_PARTITION
        {
            break;
        }
    }
    best_partition
}

#[derive(Clone, Copy, Default)]
struct Solution {
    index: u32,
    err: u64,
}

/// `estimate_partition_list`: up to `max_solutions` partitions sorted by estimated error.
fn estimate_partition_list(
    mode: u32,
    px: &[Cq; 16],
    comp: &Params,
    sol: &mut [Solution; 64],
    max_solutions: u32,
) -> usize {
    let orig_max = max_solutions as i32;
    let total = total_partitions(mode, comp);
    if total <= 1 {
        sol[0] = Solution { index: 0, err: 0 };
        return 1;
    } else if orig_max >= total as i32 {
        for i in 0..total {
            sol[i as usize] = Solution {
                index: i,
                err: u64::from(i),
            };
        }
        return total as usize;
    }
    let mut max_solutions = orig_max;
    const HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD: i32 = 4;
    if NUM_SUBSETS[mode as usize] == 2 && max_solutions < HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD
    {
        max_solutions = HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD;
    }

    let p = estimate_params(mode, comp);
    let mut num = 0i32;
    for partition in 0..total {
        let err = estimate_partition_err(mode, &p, px, partition);
        let mut i = 0i32;
        while i < num {
            if err < sol[i as usize].err {
                break;
            }
            i += 1;
        }
        if i < num {
            let to_move = ((max_solutions - 1) - i).min(num - i);
            for j in (0..to_move).rev() {
                sol[(i + j + 1) as usize] = sol[(i + j) as usize];
            }
        }
        if num < max_solutions {
            num += 1;
        }
        if i < num {
            sol[i as usize] = Solution {
                index: partition,
                err,
            };
        }
    }
    num.min(orig_max).max(0) as usize
}

// ---------------------------------------------------------------------------------------------
// Block packing.

#[derive(Clone, Copy, Default)]
struct OptResults {
    mode: u32,
    partition: u32,
    selectors: [i32; 16],
    alpha_selectors: [i32; 16],
    low: [Cq; 3],
    high: [Cq; 3],
    pbits: [[u32; 2]; 3],
    rotation: u32,
    index_selector: u32,
    /// The winning encoding used the one-color tables on any subset.
    used_lut: bool,
}

#[inline]
fn set_block_bits(bytes: &mut [u8; 16], mut val: u32, mut num_bits: u32, ofs: &mut u32) {
    while num_bits > 0 {
        let n = (8 - (*ofs & 7)).min(num_bits);
        bytes[(*ofs >> 3) as usize] |= (val << (*ofs & 7)) as u8;
        val >>= n;
        num_bits -= n;
        *ofs += n;
    }
}

/// `encode_bc7_block`.
fn encode_bc7_block(r: &OptResults) -> [u8; 16] {
    let mode = r.mode as usize;
    let nsub = NUM_SUBSETS[mode];
    let total_partitions = 1u32 << PARTITION_BITS[mode];
    let part = match nsub {
        1 => &PARTITION1,
        2 => &PARTITION2[r.partition as usize],
        _ => &PARTITION3[r.partition as usize],
    };
    let separate_alpha = mode == 4 || mode == 5;

    let mut csel = r.selectors;
    let mut asel = r.alpha_selectors;
    let mut low = r.low;
    let mut high = r.high;
    let mut pbits = r.pbits;
    let mut anchor = [-1i32; 3];

    let color_index_bits = COLOR_INDEX_BITCOUNT[mode] + r.index_selector;
    let alpha_index_bits = ALPHA_INDEX_BITCOUNT[mode].wrapping_sub(r.index_selector);
    for k in 0..nsub as usize {
        let anchor_index = if k == 0 {
            0
        } else if nsub == 3 && k == 1 {
            ANCHOR3_1[r.partition as usize] as usize
        } else if nsub == 3 && k == 2 {
            ANCHOR3_2[r.partition as usize] as usize
        } else {
            ANCHOR2[r.partition as usize] as usize
        };
        anchor[k] = anchor_index as i32;

        let nci = 1i32 << color_index_bits;
        if csel[anchor_index] & (nci >> 1) != 0 {
            for i in 0..16 {
                if part[i] as usize == k {
                    csel[i] = (nci - 1) - csel[i];
                }
            }
            if separate_alpha {
                for q in 0..3 {
                    std::mem::swap(&mut low[k][q], &mut high[k][q]);
                }
            } else {
                std::mem::swap(&mut low[k], &mut high[k]);
            }
            if !MODE_HAS_SHARED_P_BITS[mode] {
                pbits[k].swap(0, 1);
            }
        }

        if separate_alpha {
            let nai = 1i32 << alpha_index_bits;
            if asel[anchor_index] & (nai >> 1) != 0 {
                for i in 0..16 {
                    if part[i] as usize == k {
                        asel[i] = (nai - 1) - asel[i];
                    }
                }
                std::mem::swap(&mut low[k][3], &mut high[k][3]);
            }
        }
    }

    let mut b = [0u8; 16];
    let mut ofs = 0u32;
    set_block_bits(&mut b, 1 << mode, mode as u32 + 1, &mut ofs);
    if separate_alpha {
        set_block_bits(&mut b, r.rotation, 2, &mut ofs);
    }
    if mode == 4 {
        set_block_bits(&mut b, r.index_selector, 1, &mut ofs);
    }
    if total_partitions > 1 {
        let bits = if total_partitions == 64 { 6 } else { 4 };
        set_block_bits(&mut b, r.partition, bits, &mut ofs);
    }
    let total_comps = if mode >= 4 { 4 } else { 3 };
    for comp in 0..total_comps {
        let bits = if comp == 3 {
            ALPHA_PRECISION[mode]
        } else {
            COLOR_PRECISION[mode]
        };
        for s in 0..nsub as usize {
            set_block_bits(&mut b, low[s][comp] as u32, bits, &mut ofs);
            set_block_bits(&mut b, high[s][comp] as u32, bits, &mut ofs);
        }
    }
    if MODE_HAS_P_BITS[mode] {
        for s in 0..nsub as usize {
            set_block_bits(&mut b, pbits[s][0], 1, &mut ofs);
            if !MODE_HAS_SHARED_P_BITS[mode] {
                set_block_bits(&mut b, pbits[s][1], 1, &mut ofs);
            }
        }
    }
    let is_anchor = |idx: i32| idx == anchor[0] || idx == anchor[1] || idx == anchor[2];
    let (first_bits, first, second_bits, second) = if r.index_selector != 0 {
        (alpha_index_bits, &asel, color_index_bits, &csel)
    } else {
        (color_index_bits, &csel, alpha_index_bits, &asel)
    };
    for idx in 0..16 {
        let n = first_bits - u32::from(is_anchor(idx as i32));
        set_block_bits(&mut b, first[idx] as u32, n, &mut ofs);
    }
    if separate_alpha {
        for idx in 0..16 {
            let n = second_bits - u32::from(is_anchor(idx as i32));
            set_block_bits(&mut b, second[idx] as u32, n, &mut ofs);
        }
    }
    b
}

/// `encode_bc7_block_mode6`.
fn encode_bc7_block_mode6(r: &OptResults) -> [u8; 16] {
    let (low, high, pbits, invert) = if r.selectors[0] & 8 != 0 {
        (r.high[0], r.low[0], [r.pbits[0][1], r.pbits[0][0]], 15u64)
    } else {
        (r.low[0], r.high[0], r.pbits[0], 0u64)
    };
    let mut l = 1u64 << 6;
    for c in 0..4 {
        l |= (low[c] as u64) << (7 + c * 14);
        l |= (high[c] as u64) << (14 + c * 14);
    }
    l |= u64::from(pbits[0]) << 63;
    let mut h = u64::from(pbits[1]);
    h |= (invert ^ r.selectors[0] as u64) << 1;
    for i in 1..16 {
        h |= (invert ^ r.selectors[i] as u64) << (i * 4);
    }
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&l.to_le_bytes());
    out[8..].copy_from_slice(&h.to_le_bytes());
    out
}

// ---------------------------------------------------------------------------------------------
// Mode handlers.

/// One partition trial of a multi-subset mode.
struct PartitionTrial {
    err: u64,
    results: [CcResults; 3],
    sels: [[i32; 16]; 3],
    index: [[u8; 16]; 3],
    count: [usize; 3],
}

/// Encodes every subset of `partition`, stopping once the running error exceeds `best_err`.
#[allow(clippy::too_many_arguments)]
fn try_partition(
    mode: u32,
    p: &CcParams,
    comp: &Params,
    px: &[Cq; 16],
    partition: u32,
    refinement: bool,
    best_err: u64,
) -> PartitionTrial {
    let nsub = NUM_SUBSETS[mode as usize] as usize;
    let part = if nsub == 3 {
        &PARTITION3[partition as usize]
    } else {
        &PARTITION2[partition as usize]
    };
    let (colors, index, count) = split_subsets(px, part);
    let mut t = PartitionTrial {
        err: 0,
        results: [CcResults::default(); 3],
        sels: [[0; 16]; 3],
        index,
        count,
    };
    for s in 0..nsub {
        let err = color_cell_compression(
            mode,
            p,
            &mut t.results[s],
            &mut t.sels[s],
            comp,
            &colors[s][..count[s]],
            refinement,
        );
        t.err = t.err.wrapping_add(err);
        if t.err > best_err {
            break;
        }
    }
    t
}

/// Copies a winning partition trial into `opt`; `pbits` = how many pbits per subset to copy.
fn apply_partition(opt: &mut OptResults, t: &PartitionTrial, nsub: usize, pbits: usize) {
    for s in 0..nsub {
        for i in 0..t.count[s] {
            opt.selectors[t.index[s][i] as usize] = t.sels[s][i];
        }
        opt.low[s] = t.results[s].low;
        opt.high[s] = t.results[s].high;
        opt.pbits[s][..pbits].copy_from_slice(&t.results[s].pbits[..pbits]);
    }
    opt.used_lut = t.results[..nsub].iter().any(|r| r.used_lut);
}

/// Tries the candidate partitions of a 2/3-subset mode; with more than two candidates the
/// winner is re-encoded with refinement (`disable_faster_part_selection` is false upstream).
#[allow(clippy::too_many_arguments)]
fn search_partitions(
    mode: u32,
    p: &CcParams,
    comp: &Params,
    px: &[Cq; 16],
    sol: &[Solution],
    always_refine: bool,
    pbits: usize,
    opt: &mut OptResults,
    best_err: &mut u64,
) {
    let nsub = NUM_SUBSETS[mode as usize] as usize;
    let refinement = always_refine || sol.len() <= 2;
    for s in sol {
        let t = try_partition(mode, p, comp, px, s.index, refinement, *best_err);
        if t.err < *best_err {
            *best_err = t.err;
            opt.mode = mode;
            opt.index_selector = 0;
            opt.rotation = 0;
            opt.partition = s.index;
            apply_partition(opt, &t, nsub, pbits);
        }
    }
    if !always_refine && sol.len() > 2 && opt.mode == mode {
        let t = try_partition(mode, p, comp, px, opt.partition, true, *best_err);
        if t.err < *best_err {
            *best_err = t.err;
            apply_partition(opt, &t, nsub, pbits);
        }
    }
}

/// Alpha index values for 6-bit (mode 4) alpha endpoints.
#[inline]
fn mode4_alpha_vals(la: u32, ha: u32, index_selector: u32) -> [i32; 8] {
    let mut vals = [0i32; 8];
    if index_selector == 0 {
        vals[0] = ((la << 2) | (la >> 4)) as i32;
        vals[7] = ((ha << 2) | (ha >> 4)) as i32;
        for i in 1..7 {
            vals[i] = ((vals[0] as u32)
                .wrapping_mul(64 - W3[i])
                .wrapping_add((vals[7] as u32).wrapping_mul(W3[i]))
                .wrapping_add(32)
                >> 6) as i32;
        }
    } else {
        vals[0] = ((la << 2) | (la >> 4)) as i32;
        vals[3] = ((ha << 2) | (ha >> 4)) as i32;
        vals[1] = (vals[0] * (64 - 21) + vals[3] * 21 + 32) >> 6;
        vals[2] = (vals[0] * (64 - 43) + vals[3] * 43 + 32) >> 6;
    }
    vals
}

/// Nearest alpha index per pixel (strict `<`, first wins) and the weighted error.
#[inline]
fn alpha_selectors(px: &[Cq; 16], vals: &[i32], weight: u32) -> ([i32; 16], u64) {
    let mut sels = [0i32; 16];
    let mut err = 0u64;
    for (i, c) in px.iter().enumerate() {
        let a = c[3];
        let mut s = 0;
        let mut be = iabs32(a - vals[0]);
        for (j, &v) in vals.iter().enumerate().skip(1) {
            let e = iabs32(a - v);
            if e < be {
                be = e;
                s = j as i32;
            }
        }
        err = err.wrapping_add(u64::from((be.wrapping_mul(be) as u32).wrapping_mul(weight)));
        sels[i] = s;
    }
    (sels, err)
}

/// `handle_alpha_block_mode4`: both index selectors of a (possibly rotated) block.
fn handle_alpha_block_mode4(
    px: &[Cq; 16],
    comp: &Params,
    p: &mut CcParams,
    lo_a: u32,
    hi_a: u32,
    opt4: &mut OptResults,
    mode4_err: &mut u64,
) {
    p.has_alpha = false;
    p.comp_bits = 5;
    p.has_pbits = false;
    p.endpoints_share_pbit = false;
    p.perceptual = comp.perceptual;

    for index_selector in 0..2u32 {
        if comp.mode4_index_mask & (1 << index_selector) == 0 {
            continue;
        }
        p.set_selector_weights(if index_selector != 0 { 8 } else { 4 });
        let nvals = if index_selector == 0 { 8 } else { 4 };

        let mut results = CcResults::default();
        let mut selectors = [0i32; 16];
        let mut trial_err =
            color_cell_compression(4, p, &mut results, &mut selectors, comp, px, true);

        let mut la = ((lo_a + 2) >> 2).min(63);
        let mut ha = ((hi_a + 2) >> 2).min(63);
        if la == ha && lo_a != hi_a {
            if ha != 63 {
                ha += 1;
            } else {
                la = la.saturating_sub(1);
            }
        }

        let mut best_alpha_err = u64::MAX;
        let mut best_la = 0u32;
        let mut best_ha = 0u32;
        let mut best_alpha_sels = [0i32; 16];
        for pass in 0..2 {
            let vals = mode4_alpha_vals(la, ha, index_selector);
            let (sels, err) = alpha_selectors(px, &vals[..nvals], p.weights[3]);
            if err < best_alpha_err {
                best_alpha_err = err;
                best_la = la;
                best_ha = ha;
                best_alpha_sels = sels;
            }
            if pass == 0 {
                let swx: &[[f32; 4]] = if index_selector != 0 { &W2X } else { &W3X };
                let (mut xl, mut xh) = lsq_endpoints_a(&sels, swx, px);
                if xl > xh {
                    std::mem::swap(&mut xl, &mut xh);
                }
                la = clampi(floor_to_int(xl * (63.0 / 255.0) + 0.5), 0, 63) as u32;
                ha = clampi(floor_to_int(xh * (63.0 / 255.0) + 0.5), 0, 63) as u32;
            }
        }

        if comp.uber_level > 0 {
            let d = (comp.uber_level as i32).min(3);
            for ld in -d..=d {
                for hd in -d..=d {
                    la = clampi(best_la as i32 + ld, 0, 63) as u32;
                    ha = clampi(best_ha as i32 + hd, 0, 63) as u32;
                    let vals = mode4_alpha_vals(la, ha, index_selector);
                    let (sels, err) = alpha_selectors(px, &vals[..nvals], p.weights[3]);
                    if err < best_alpha_err {
                        best_alpha_err = err;
                        best_la = la;
                        best_ha = ha;
                        best_alpha_sels = sels;
                    }
                }
            }
        }

        trial_err = trial_err.wrapping_add(best_alpha_err);
        if trial_err < *mode4_err {
            *mode4_err = trial_err;
            opt4.mode = 4;
            opt4.index_selector = index_selector;
            opt4.rotation = 0;
            opt4.partition = 0;
            opt4.used_lut = results.used_lut;
            opt4.low[0] = results.low;
            opt4.high[0] = results.high;
            opt4.low[0][3] = best_la as i32;
            opt4.high[0][3] = best_ha as i32;
            opt4.selectors = selectors;
            opt4.alpha_selectors = best_alpha_sels;
        }
    }
}

/// `handle_alpha_block_mode5` for a (possibly rotated) block.
fn handle_alpha_block_mode5(
    px: &[Cq; 16],
    comp: &Params,
    p: &mut CcParams,
    mut lo_a: u32,
    mut hi_a: u32,
    opt5: &mut OptResults,
    mode5_err: &mut u64,
) {
    p.set_selector_weights(4);
    p.comp_bits = 7;
    p.has_alpha = false;
    p.has_pbits = false;
    p.endpoints_share_pbit = false;
    p.perceptual = comp.perceptual;

    let mut results5 = CcResults::default();
    *mode5_err = color_cell_compression(5, p, &mut results5, &mut opt5.selectors, comp, px, true);
    opt5.low[0] = results5.low;
    opt5.high[0] = results5.high;

    if lo_a == hi_a {
        opt5.low[0][3] = lo_a as i32;
        opt5.high[0][3] = hi_a as i32;
        opt5.alpha_selectors = [0; 16];
    } else {
        let mut mode5_alpha_err = u64::MAX;
        let vals_of = |lo: u32, hi: u32| {
            let (v0, v3) = (lo as i32, hi as i32);
            [
                v0,
                (v0 * (64 - 21) + v3 * 21 + 32) >> 6,
                (v0 * (64 - 43) + v3 * 43 + 32) >> 6,
                v3,
            ]
        };
        for pass in 0..2 {
            let (sels, err) = alpha_selectors(px, &vals_of(lo_a, hi_a), p.weights[3]);
            if err < mode5_alpha_err {
                mode5_alpha_err = err;
                opt5.low[0][3] = lo_a as i32;
                opt5.high[0][3] = hi_a as i32;
                opt5.alpha_selectors = sels;
            }
            if pass == 0 {
                let (xl, xh) = lsq_endpoints_a(&sels, &W2X, px);
                let mut new_lo = clampi(floor_to_int(xl + 0.5), 0, 255) as u32;
                let mut new_hi = clampi(floor_to_int(xh + 0.5), 0, 255) as u32;
                if new_lo > new_hi {
                    std::mem::swap(&mut new_lo, &mut new_hi);
                }
                if new_lo == lo_a && new_hi == hi_a {
                    break;
                }
                lo_a = new_lo;
                hi_a = new_hi;
            }
        }

        if comp.uber_level > 0 {
            let d = (comp.uber_level as i32).min(3);
            for ld in -d..=d {
                for hd in -d..=d {
                    lo_a = clampi(opt5.low[0][3] + ld, 0, 255) as u32;
                    hi_a = clampi(opt5.high[0][3] + hd, 0, 255) as u32;
                    let (sels, err) = alpha_selectors(px, &vals_of(lo_a, hi_a), p.weights[3]);
                    if err < mode5_alpha_err {
                        mode5_alpha_err = err;
                        opt5.low[0][3] = lo_a as i32;
                        opt5.high[0][3] = hi_a as i32;
                        opt5.alpha_selectors = sels;
                    }
                }
            }
        }
        *mode5_err = mode5_err.wrapping_add(mode5_alpha_err);
    }

    opt5.mode = 5;
    opt5.index_selector = 0;
    opt5.rotation = 0;
    opt5.partition = 0;
    opt5.used_lut = results5.used_lut;
}

/// Pixels with alpha swapped into channel `rotation - 1` and their alpha range.
#[inline]
fn rotate(px: &[Cq; 16], rotation: u32) -> ([Cq; 16], u32, u32) {
    let mut out = *px;
    let mut lo = 255u32;
    let mut hi = 0u32;
    for c in out.iter_mut() {
        c.swap(3, rotation as usize - 1);
        lo = lo.min(c[3] as u32);
        hi = hi.max(c[3] as u32);
    }
    (out, lo, hi)
}

/// Mode 4 over the enabled rotations (`num_rotations` = 1 or 4).
#[allow(clippy::too_many_arguments)]
fn try_mode4(
    px: &[Cq; 16],
    comp: &Params,
    base: &CcParams,
    lo_a: u32,
    hi_a: u32,
    num_rotations: u32,
    opt: &mut OptResults,
    best_err: &mut u64,
) {
    let mut p4 = *base;
    for rotation in 0..num_rotations {
        if comp.mode4_rotation_mask & (1 << rotation) == 0 {
            continue;
        }
        p4.weights = base.weights;
        let (rot_px, tlo, thi);
        let (trial_px, trial_lo, trial_hi) = if rotation != 0 {
            p4.weights.swap(rotation as usize - 1, 3);
            (rot_px, tlo, thi) = rotate(px, rotation);
            (&rot_px, tlo, thi)
        } else {
            (px, lo_a, hi_a)
        };
        let mut trial = OptResults::default();
        let mut trial_err = *best_err;
        handle_alpha_block_mode4(
            trial_px,
            comp,
            &mut p4,
            trial_lo,
            trial_hi,
            &mut trial,
            &mut trial_err,
        );
        if trial_err < *best_err {
            *best_err = trial_err;
            opt.mode = 4;
            opt.index_selector = trial.index_selector;
            opt.rotation = rotation;
            opt.partition = 0;
            opt.used_lut = trial.used_lut;
            opt.low[0] = trial.low[0];
            opt.high[0] = trial.high[0];
            opt.selectors = trial.selectors;
            opt.alpha_selectors = trial.alpha_selectors;
        }
    }
}

/// Mode 5 over the enabled rotations (`num_rotations` = 1 or 4).
#[allow(clippy::too_many_arguments)]
fn try_mode5(
    px: &[Cq; 16],
    comp: &Params,
    base: &CcParams,
    lo_a: u32,
    hi_a: u32,
    num_rotations: u32,
    opt: &mut OptResults,
    best_err: &mut u64,
) {
    let mut p5 = *base;
    for rotation in 0..num_rotations {
        if comp.mode5_rotation_mask & (1 << rotation) == 0 {
            continue;
        }
        p5.weights = base.weights;
        let (rot_px, tlo, thi);
        let (trial_px, trial_lo, trial_hi) = if rotation != 0 {
            p5.weights.swap(rotation as usize - 1, 3);
            (rot_px, tlo, thi) = rotate(px, rotation);
            (&rot_px, tlo, thi)
        } else {
            (px, lo_a, hi_a)
        };
        let mut trial = OptResults::default();
        let mut trial_err = 0u64;
        handle_alpha_block_mode5(
            trial_px,
            comp,
            &mut p5,
            trial_lo,
            trial_hi,
            &mut trial,
            &mut trial_err,
        );
        if trial_err < *best_err {
            *best_err = trial_err;
            *opt = trial;
            opt.rotation = rotation;
        }
    }
}

/// `handle_alpha_block`: modes 4, 6, 5, 7 for blocks with alpha < 255.
fn handle_alpha_block(
    px: &[Cq; 16],
    comp: &Params,
    base: &CcParams,
    lo_a: u32,
    hi_a: u32,
) -> ([u8; 16], bool) {
    let mut base = *base;
    base.perceptual = comp.perceptual;
    let mut opt = OptResults::default();
    let mut best_err = u64::MAX;

    let mul = |p: &mut CcParams| {
        for (w, m) in p.weights.iter_mut().zip(comp.alpha.mode67_error_weight_mul) {
            *w = w.wrapping_mul(m);
        }
    };

    if comp.alpha.use_mode4 {
        let rots = if comp.perceptual || !comp.alpha.use_mode4_rotation {
            1
        } else {
            4
        };
        try_mode4(px, comp, &base, lo_a, hi_a, rots, &mut opt, &mut best_err);
    }

    if comp.alpha.use_mode6 {
        let mut p6 = base;
        mul(&mut p6);
        p6.set_selector_weights(16);
        p6.comp_bits = 7;
        p6.has_pbits = true;
        p6.endpoints_share_pbit = false;
        p6.has_alpha = true;
        let mut r6 = CcResults::default();
        let mut sels = [0i32; 16];
        let err = color_cell_compression(6, &p6, &mut r6, &mut sels, comp, px, true);
        if err < best_err {
            best_err = err;
            opt.mode = 6;
            opt.index_selector = 0;
            opt.rotation = 0;
            opt.partition = 0;
            opt.low[0] = r6.low;
            opt.high[0] = r6.high;
            opt.pbits[0] = r6.pbits;
            opt.used_lut = r6.used_lut;
            opt.selectors = sels;
        }
    }

    if comp.alpha.use_mode5 {
        let rots = if comp.perceptual || !comp.alpha.use_mode5_rotation {
            1
        } else {
            4
        };
        try_mode5(px, comp, &base, lo_a, hi_a, rots, &mut opt, &mut best_err);
    }

    if comp.alpha.use_mode7 {
        let mut sol = [Solution::default(); 64];
        let n = estimate_partition_list(
            7,
            px,
            comp,
            &mut sol,
            comp.alpha.max_mode7_partitions_to_try,
        );
        let mut p7 = base;
        mul(&mut p7);
        p7.set_selector_weights(4);
        p7.comp_bits = 5;
        p7.has_pbits = true;
        p7.endpoints_share_pbit = false;
        p7.has_alpha = true;
        search_partitions(
            7,
            &p7,
            comp,
            px,
            &sol[..n],
            false,
            2,
            &mut opt,
            &mut best_err,
        );
    }

    (encode_bc7_block(&opt), opt.used_lut)
}

/// Candidate partitions for `mode` (estimate_partition when only one is tried).
fn partition_candidates(
    mode: u32,
    px: &[Cq; 16],
    comp: &Params,
    max_to_try: u32,
    sol: &mut [Solution; 64],
) -> usize {
    if max_to_try == 1 {
        sol[0].index = estimate_partition(mode, px, comp);
        1
    } else {
        estimate_partition_list(mode, px, comp, sol, max_to_try)
    }
}

/// `handle_opaque_block`: modes 6, 1, 0, 3, 5, 2, 4 for opaque blocks.
fn handle_opaque_block(px: &[Cq; 16], comp: &Params, base: &CcParams) -> ([u8; 16], bool) {
    let mut p = *base;
    let mut opt = OptResults::default();
    let mut best_err = u64::MAX;
    let use_mode = &comp.opaque.use_mode;

    if use_mode[6] {
        p.set_selector_weights(16);
        p.comp_bits = 7;
        p.has_pbits = true;
        p.endpoints_share_pbit = false;
        p.perceptual = comp.perceptual;
        let mut r6 = CcResults::default();
        best_err = color_cell_compression(6, &p, &mut r6, &mut opt.selectors, comp, px, true);
        opt.mode = 6;
        opt.index_selector = 0;
        opt.rotation = 0;
        opt.partition = 0;
        opt.used_lut = r6.used_lut;
        opt.low[0] = r6.low;
        opt.high[0] = r6.high;
        opt.pbits[0] = r6.pbits;
    }

    let mut sol2 = [Solution::default(); 64];
    let mut n2 = 0;
    if use_mode[1] || use_mode[3] {
        n2 = partition_candidates(
            1,
            px,
            comp,
            comp.opaque.max_mode13_partitions_to_try,
            &mut sol2,
        );
    }

    if use_mode[1] {
        p.set_selector_weights(8);
        p.comp_bits = 6;
        p.has_pbits = true;
        p.endpoints_share_pbit = true;
        p.perceptual = comp.perceptual;
        search_partitions(
            1,
            &p,
            comp,
            px,
            &sol2[..n2],
            false,
            1,
            &mut opt,
            &mut best_err,
        );
    }

    if use_mode[0] {
        let mut sol3 = [Solution::default(); 64];
        let n3 = partition_candidates(
            0,
            px,
            comp,
            comp.opaque.max_mode0_partitions_to_try,
            &mut sol3,
        );
        p.set_selector_weights(8);
        p.comp_bits = 4;
        p.has_pbits = true;
        p.endpoints_share_pbit = false;
        p.perceptual = comp.perceptual;
        search_partitions(
            0,
            &p,
            comp,
            px,
            &sol3[..n3],
            true,
            2,
            &mut opt,
            &mut best_err,
        );
    }

    if use_mode[3] {
        p.set_selector_weights(4);
        p.comp_bits = 7;
        p.has_pbits = true;
        p.endpoints_share_pbit = false;
        p.perceptual = comp.perceptual;
        search_partitions(
            3,
            &p,
            comp,
            px,
            &sol2[..n2],
            false,
            2,
            &mut opt,
            &mut best_err,
        );
    }

    if !comp.perceptual && use_mode[5] {
        try_mode5(px, comp, &p, 255, 255, 4, &mut opt, &mut best_err);
    }

    if use_mode[2] {
        let mut sol3 = [Solution::default(); 64];
        let n3 = partition_candidates(
            2,
            px,
            comp,
            comp.opaque.max_mode2_partitions_to_try,
            &mut sol3,
        );
        p.set_selector_weights(4);
        p.comp_bits = 5;
        p.has_pbits = false;
        p.endpoints_share_pbit = false;
        p.perceptual = comp.perceptual;
        search_partitions(
            2,
            &p,
            comp,
            px,
            &sol3[..n3],
            true,
            0,
            &mut opt,
            &mut best_err,
        );
    }

    if !comp.perceptual && use_mode[4] {
        try_mode4(px, comp, &p, 255, 255, 4, &mut opt, &mut best_err);
    }

    (encode_bc7_block(&opt), opt.used_lut)
}

/// `handle_block_solid`: every solid block is encoded exactly with mode 5.
fn handle_block_solid(r: usize, g: usize, b: usize, a: i32) -> [u8; 16] {
    let t = &luts().mode5;
    let (er, eg, eb) = (t[r], t[g], t[b]);
    let opt = OptResults {
        mode: 5,
        low: [
            [
                (er & 0xFF) as i32,
                (eg & 0xFF) as i32,
                (eb & 0xFF) as i32,
                a,
            ],
            [0; 4],
            [0; 4],
        ],
        high: [
            [(er >> 8) as i32, (eg >> 8) as i32, (eb >> 8) as i32, a],
            [0; 4],
            [0; 4],
        ],
        selectors: [MODE_5_OPTIMAL_INDEX as i32; 16],
        ..OptResults::default()
    };
    encode_bc7_block(&opt)
}

/// `handle_opaque_block_mode6`.
fn handle_opaque_block_mode6(px: &[Cq; 16], comp: &Params, base: &CcParams) -> ([u8; 16], bool) {
    let mut p = *base;
    p.set_selector_weights(16);
    p.comp_bits = 7;
    p.has_pbits = true;
    p.endpoints_share_pbit = false;
    p.perceptual = comp.perceptual;
    let mut opt = OptResults::default();
    let mut r6 = CcResults::default();
    color_cell_compression(6, &p, &mut r6, &mut opt.selectors, comp, px, true);
    opt.mode = 6;
    opt.low[0] = r6.low;
    opt.high[0] = r6.high;
    opt.pbits[0] = r6.pbits;
    (encode_bc7_block_mode6(&opt), r6.used_lut)
}

// ---------------------------------------------------------------------------------------------
// Public (crate) API.

/// Compresses one 4x4 RGBA8 block (row-major) to BC7 (`bc7e_compress_blocks` for one block).
pub(crate) fn compress_block(px: &[[u8; 4]; 16], params: &Params) -> [u8; 16] {
    compress_block_lut(px, params).0
}

/// [`compress_block`], also returning the `pUsed_lut` flag: whether the winning encoding used
/// the one-color optimal endpoint tables (solid blocks always do).
pub(crate) fn compress_block_lut(px: &[[u8; 4]; 16], params: &Params) -> ([u8; 16], bool) {
    let mut t = [[0i32; 4]; 16];
    let (mut lo_r, mut hi_r) = (255i32, 0i32);
    let (mut lo_g, mut hi_g) = (255i32, 0i32);
    let (mut lo_b, mut hi_b) = (255i32, 0i32);
    let (mut lo_a, mut hi_a) = (255f32, 0f32);
    for (d, s) in t.iter_mut().zip(px) {
        *d = s.map(i32::from);
        lo_r = lo_r.min(d[0]);
        hi_r = hi_r.max(d[0]);
        lo_g = lo_g.min(d[1]);
        hi_g = hi_g.max(d[1]);
        lo_b = lo_b.min(d[2]);
        hi_b = hi_b.max(d[2]);
        let fa = d[3] as f32;
        lo_a = minf(lo_a, fa);
        hi_a = maxf(hi_a, fa);
    }

    if lo_r == hi_r && lo_g == hi_g && lo_b == hi_b && lo_a == hi_a {
        let block = handle_block_solid(lo_r as usize, lo_g as usize, lo_b as usize, lo_a as i32);
        return (block, true);
    }

    let mut cc = CcParams::cleared();
    cc.weights = params.weights;
    if lo_a < 255.0 {
        handle_alpha_block(&t, params, &cc, lo_a as i32 as u32, hi_a as i32 as u32)
    } else if params.mode6_only {
        handle_opaque_block_mode6(&t, params, &cc)
    } else {
        handle_opaque_block(&t, params, &cc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Format;
    use std::error::Error;

    type Preset = fn(bool) -> Params;

    const PRESETS: [(&str, Preset); 7] = [
        ("ultrafast", Params::ultrafast),
        ("veryfast", Params::veryfast),
        ("fast", Params::fast),
        ("basic", Params::basic),
        ("slow", Params::slow),
        ("veryslow", Params::veryslow),
        ("slowest", Params::slowest),
    ];

    fn fnv(b: &[u8]) -> u64 {
        b.iter().fold(0xcbf29ce484222325u64, |h, &x| {
            (h ^ u64::from(x)).wrapping_mul(0x100000001b3)
        })
    }

    /// Raw image NAME_WxH.rgba -> (width, height, pixels).
    fn raw(name: &str) -> Result<(usize, usize, Vec<u8>), Box<dyn Error>> {
        let dims = name
            .trim_end_matches(".rgba")
            .rsplit('_')
            .next()
            .ok_or("name")?;
        let (w, h) = dims.split_once('x').ok_or("dims")?;
        let path = format!("{}/testdata/images/raw/{name}", env!("CARGO_MANIFEST_DIR"));
        Ok((w.parse()?, h.parse()?, std::fs::read(path)?))
    }

    /// Encodes a whole image with edge clamping like `imgio::encode`.
    fn encode(w: usize, h: usize, img: &[u8], params: &Params) -> Vec<u8> {
        let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
        let mut out = Vec::with_capacity(bw * bh * 16);
        for by in 0..bh {
            for bx in 0..bw {
                let mut px = [[0u8; 4]; 16];
                for (i, p) in px.iter_mut().enumerate() {
                    let x = (bx * 4 + (i & 3)).min(w - 1);
                    let y = (by * 4 + (i >> 2)).min(h - 1);
                    let o = (y * w + x) * 4;
                    p.copy_from_slice(&img[o..o + 4]);
                }
                out.extend_from_slice(&compress_block(&px, params));
            }
        }
        out
    }

    #[test]
    fn matches_basisu() -> Result<(), Box<dyn Error>> {
        let golden = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/encode/bc7e_golden.txt"
        ))?;
        let mut n = 0;
        let mut bad = Vec::new();
        for line in golden.lines() {
            let f: Vec<&str> = line.split(' ').collect();
            let (w, h, img) = raw(f[0])?;
            let ctor = PRESETS.iter().find(|p| p.0 == f[1]).ok_or("preset")?.1;
            let out = encode(w, h, &img, &ctor(f[2] == "1"));
            if format!("{:016x}", fnv(&out)) != f[3] {
                bad.push(line.to_string());
            }
            n += 1;
        }
        assert!(bad.is_empty(), "mismatches: {bad:#?}");
        assert_eq!(n, 42);
        Ok(())
    }

    /// Mean squared error over RGBA.
    fn mse(a: &[u8], b: &[u8]) -> f64 {
        let se: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum();
        se / a.len() as f64
    }

    #[test]
    fn decodes_close_to_source() -> Result<(), Box<dyn Error>> {
        for name in [
            "rgba_64x48.rgba",
            "rgb_37x29.rgba",
            "mosaic_rgba_50x38.rgba",
        ] {
            let (w, h, img) = raw(name)?;
            for (preset, ctor) in PRESETS {
                for perceptual in [false, true] {
                    let bc = encode(w, h, &img, &ctor(perceptual));
                    let dec = crate::decode_to_buffer(Format::Bc7Unorm, &bc, w as u32, h as u32)?;
                    let e = mse(&img, &dec);
                    // ultrafast (mode 6 only) is the weakest; everything else is far below.
                    assert!(e < 100.0, "{name} {preset} {perceptual}: mse {e}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn solid_blocks_are_exact() -> Result<(), Box<dyn Error>> {
        for c in [
            [0u8, 0, 0, 255],
            [17, 200, 99, 255],
            [255, 1, 128, 7],
            [3, 3, 3, 0],
        ] {
            let bc = compress_block(&[c; 16], &Params::slowest(false));
            let dec = crate::decode_to_buffer(Format::Bc7Unorm, &bc, 4, 4)?;
            assert!(dec.chunks(4).all(|p| p == c), "{c:?}");
        }
        Ok(())
    }
}
