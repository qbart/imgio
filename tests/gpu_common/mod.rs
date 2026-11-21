// SPDX-FileCopyrightText: 2010-2026 Mark Callow, The Khronos Group Inc. and KTX-Software contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from ktx-rs `tests/vkloader.rs`, `tests/glloader.rs`; see NOTICE.md.
//! Helpers shared by the GPU upload tests.
#![allow(dead_code)]

use imgio::Format;
use imgio::gpu::{TextureData, TextureSource};
use std::error::Error;

pub type TestResult<T = ()> = Result<T, Box<dyn Error>>;

pub const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/");

/// Deterministic pseudo-random bytes (LCG, as the ktx-rs loader tests).
pub fn pattern(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
        .collect()
}

/// The tightly packed size of one image of `level` of `src` (all depth slices).
pub fn image_size(src: &dyn TextureSource, level: u32) -> TestResult<usize> {
    let shr = |v: u32| (v >> level).max(1);
    let f = src.format()?;
    Ok(f.surface_bytes(shr(src.width()), shr(src.height()))? * shr(src.depth()) as usize)
}

/// Every image of each level of `src`, tightly packed, layer after layer and face after face.
pub fn levels(src: &dyn TextureSource) -> TestResult<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    for level in 0..src.level_count() {
        let size = image_size(src, level)?;
        let mut l = Vec::new();
        for layer in 0..src.layer_count() {
            for face in 0..src.face_count() {
                l.extend_from_slice(&src.image_data(level, layer, face)?[..size]);
            }
        }
        out.push(l);
    }
    Ok(out)
}

/// Level data of a texture with `images` images per level, filled with `pattern`.
pub fn level_data(
    format: Format,
    (w, h, d): (u32, u32, u32),
    levels: u32,
    images: u32,
    seed: u32,
) -> TestResult<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    for level in 0..levels {
        let shr = |v: u32| (v >> level).max(1);
        let size = format.surface_bytes(shr(w), shr(h))? * (shr(d) * images) as usize;
        out.push(pattern(size, seed + level));
    }
    Ok(out)
}

/// A texture borrowing `data` (one entry per level).
pub fn texture<'a>(
    format: Format,
    (w, h, d): (u32, u32, u32),
    layers: u32,
    cube: bool,
    data: &'a [Vec<u8>],
) -> TestResult<TextureData<'a>> {
    let mut t = TextureData::new(format, w, h);
    if layers > 1 {
        t = t.with_layers(layers);
    }
    if cube {
        t = t.with_cube();
    }
    if d > 1 {
        t = t.with_depth(d);
    }
    for l in data {
        t.push_level(l)?;
    }
    Ok(t)
}

/// `rows` rows of `row` bytes, each padded with junk to a multiple of 4 bytes.
pub fn pad_rows(tight: &[u8], row: usize) -> Vec<u8> {
    let pitch = row.div_ceil(4) * 4;
    let mut out = Vec::new();
    for r in tight.chunks(row) {
        out.extend_from_slice(r);
        out.extend(std::iter::repeat_n(0xA5, pitch - row));
    }
    out
}

#[cfg(feature = "opengl")]
pub mod egl;
