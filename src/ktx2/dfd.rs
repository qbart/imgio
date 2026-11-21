// SPDX-FileCopyrightText: 2019-2020 The Khronos Group Inc.
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from KTX-Software dfdutils (createdfd.c, vk2dfd.inl) via ktx-rs
// src/dfd and modified; see NOTICE.md.
//! Khronos data format descriptors for the formats imgio writes, byte-identical with
//! KTX-Software's `vk2dfd`: one basic descriptor block, little-endian samples.

use crate::{AstcKind, Format};

const SAMPLE_START: usize = 6;
const MODEL_RGBSDA: u32 = 1;
const PRIMARIES_BT709: u32 = 1;
const TRANSFER_LINEAR: u32 = 1;
const TRANSFER_SRGB: u32 = 2;
const ALPHA: u32 = 15;
const LINEAR: u32 = 0x10;
const EXPONENT: u32 = 0x20;
const SIGNED: u32 = 0x40;
const FLOAT: u32 = 0x80;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Suffix {
    Unorm,
    Snorm,
    Srgb,
    Sfloat,
    Ufloat,
}

/// A data format descriptor: `words[0]` is the total size in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dfd {
    words: [u32; 32],
    len: usize,
}

impl Dfd {
    /// The DFD as words (including the leading total size).
    pub fn words(&self) -> &[u32] {
        &self.words[..self.len]
    }

    fn new(samples: usize) -> Self {
        let len = 1 + SAMPLE_START + samples * 4;
        let mut words = [0u32; 32];
        words[0] = 4 * len as u32;
        // Khronos vendor, basic descriptor type; version 2 (1.3) and block size.
        words[2] = 2 | ((4 * (SAMPLE_START + samples * 4) as u32) << 16);
        Dfd { words, len }
    }

    fn sample(&mut self, i: usize, word0: u32, lower: u32, upper: u32) {
        let at = 1 + SAMPLE_START + i * 4;
        self.words[at..at + 4].copy_from_slice(&[word0, 0, lower, upper]);
    }
}

fn transfer(s: Suffix) -> u32 {
    if s == Suffix::Srgb {
        TRANSFER_SRGB
    } else {
        TRANSFER_LINEAR
    }
}

/// `setChannelFlags`.
fn channel_flags(channel: u32, s: Suffix) -> u32 {
    match s {
        Suffix::Unorm => channel,
        Suffix::Snorm => channel | SIGNED,
        Suffix::Sfloat => channel | FLOAT | SIGNED,
        Suffix::Ufloat => channel | FLOAT,
        Suffix::Srgb if channel == ALPHA => channel | LINEAR,
        Suffix::Srgb => channel,
    }
}

fn low_mask(bits: u32) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1 << bits) - 1
    }
}

/// `writeHeader` for uncompressed formats.
fn header(samples: usize, bytes: u32, s: Suffix) -> Dfd {
    let mut d = Dfd::new(samples);
    d.words[3] = MODEL_RGBSDA | (PRIMARIES_BT709 << 8) | (transfer(s) << 16);
    d.words[5] = bytes;
    d
}

/// `writeSample` (little-endian: every sample is both top and bottom).
fn sample(d: &mut Dfd, i: usize, channel: u32, bits: u32, offset: u32, s: Suffix) {
    let channel = channel_flags(if channel == 3 { ALPHA } else { channel }, s);
    let word0 = offset | ((bits - 1) << 16) | (channel << 24);
    let (lower, upper) = match s {
        Suffix::Unorm | Suffix::Srgb => (0, low_mask(bits)),
        Suffix::Snorm => {
            let upper = low_mask(bits - 1);
            ((!upper).wrapping_add(1), upper)
        }
        Suffix::Sfloat => ((-1.0f32).to_bits(), 1.0f32.to_bits()),
        Suffix::Ufloat => (0.0f32.to_bits(), 1.0f32.to_bits()),
    };
    d.sample(i, word0, lower, upper);
}

/// `createDFDUnpacked` (little-endian, RGBA channel order).
fn unpacked(channels: u32, bytes: u32, s: Suffix) -> Dfd {
    unpacked_swap(channels, bytes, false, s)
}

/// `createDFDUnpacked` with `redBlueSwap` (BGR(A) order).
fn unpacked_swap(channels: u32, bytes: u32, swap: bool, s: Suffix) -> Dfd {
    let mut d = header(channels as usize, channels * bytes, s);
    for i in 0..channels {
        let c = if swap && (i == 0 || i == 2) { i ^ 2 } else { i };
        sample(&mut d, i as usize, c, 8 * bytes, 8 * i * bytes, s);
    }
    d
}

/// `createDFDPacked` (little-endian, no shifts): `bits` / `channels` from bit 0 up.
fn packed(bits: &[u32], channels: &[u32]) -> Dfd {
    let mut d = header(bits.len(), bits.iter().sum::<u32>() / 8, Suffix::Unorm);
    let mut offset = 0;
    for (i, (&b, &c)) in bits.iter().zip(channels).enumerate() {
        sample(&mut d, i, c, b, offset, Suffix::Unorm);
        offset += b;
    }
    d
}

/// `createDFDPackedShifted` special case for E5B9G9R9.
fn rgb9e5() -> Dfd {
    let mut d = header(6, 4, Suffix::Ufloat);
    for c in 0..3u32 {
        let i = 2 * c as usize;
        sample(&mut d, i, c, 9, 9 * c, Suffix::Unorm);
        d.words[1 + SAMPLE_START + i * 4 + 3] = 8448;
        sample(&mut d, i + 1, c | EXPONENT, 5, 27, Suffix::Unorm);
        let at = 1 + SAMPLE_START + (i + 1) * 4;
        d.words[at + 2] = 15;
        d.words[at + 3] = 31;
    }
    d
}

/// `createDFDCompressed`: model, samples (channel, bit offset) of `channel_bits` each, bytes.
fn compressed(
    model: u32,
    samples: &[(u32, u32)],
    channel_bits: u32,
    bytes: u32,
    block: (u32, u32),
    s: Suffix,
) -> Dfd {
    let mut d = Dfd::new(samples.len());
    d.words[3] = model | (PRIMARIES_BT709 << 8) | (transfer(s) << 16);
    d.words[4] = (block.0 - 1) | ((block.1 - 1) << 8);
    d.words[5] = bytes;
    let (lower, upper) = match s {
        Suffix::Snorm => (!0x7FFF_FFFFu32, 0x7FFF_FFFF),
        Suffix::Sfloat => ((-1.0f32).to_bits(), 1.0f32.to_bits()),
        Suffix::Ufloat => (0.0f32.to_bits(), 1.0f32.to_bits()),
        Suffix::Unorm | Suffix::Srgb => (0, u32::MAX),
    };
    for (i, &(channel, offset)) in samples.iter().enumerate() {
        let word0 = offset | ((channel_bits - 1) << 16) | (channel_flags(channel, s) << 24);
        d.sample(i, word0, lower, upper);
    }
    d
}

/// The DFD KTX-Software writes for `format` (`None`: no `VkFormat`, see `vk::vk_format`).
pub fn dfd(format: Format) -> Option<Dfd> {
    use Format::*;
    use Suffix::*;
    let suffix = |srgb: bool| if srgb { Srgb } else { Unorm };
    let bc8 =
        |model: u32, channel: u32, s: Suffix| compressed(model, &[(channel, 0)], 64, 8, (4, 4), s);
    let bc16 = |model: u32, ch: (u32, u32), s: Suffix| {
        compressed(model, &[(ch.0, 0), (ch.1, 64)], 64, 16, (4, 4), s)
    };
    let bc128 = |model: u32, s: Suffix| compressed(model, &[(0, 0)], 128, 16, (4, 4), s);
    Some(match format {
        R8Unorm => unpacked(1, 1, Unorm),
        R8Snorm => unpacked(1, 1, Snorm),
        Rg8Unorm => unpacked(2, 1, Unorm),
        Rg8Snorm => unpacked(2, 1, Snorm),
        Rgba8Unorm => unpacked(4, 1, Unorm),
        Rgba8Srgb => unpacked(4, 1, Srgb),
        Rgb8Unorm => unpacked(3, 1, Unorm),
        Rgb8Srgb => unpacked(3, 1, Srgb),
        Bgra8Unorm => unpacked_swap(4, 1, true, Unorm),
        Bgra8Srgb => unpacked_swap(4, 1, true, Srgb),
        R16Unorm => unpacked(1, 2, Unorm),
        R16Snorm => unpacked(1, 2, Snorm),
        Rg16Unorm => unpacked(2, 2, Unorm),
        Rg16Snorm => unpacked(2, 2, Snorm),
        Rgb16Float => unpacked(3, 2, Sfloat),
        Rgba16Float => unpacked(4, 2, Sfloat),
        Rgba16Unorm => unpacked(4, 2, Unorm),
        Rgba32Float => unpacked(4, 4, Sfloat),
        Rgb9e5Float => rgb9e5(),
        Rgba4444 => packed(&[4, 4, 4, 4], &[3, 2, 1, 0]),
        Rgb565 => packed(&[5, 6, 5], &[2, 1, 0]),
        Bgr565 => packed(&[5, 6, 5], &[0, 1, 2]),
        // BC1 is written as BC1_RGBA: channel "alpha present".
        Bc1Unorm | Bc1Srgb => bc8(0x80, 1, suffix(format == Bc1Srgb)),
        Bc2Unorm | Bc2Srgb => bc16(0x81, (ALPHA, 0), suffix(format == Bc2Srgb)),
        Bc3Unorm | Bc3Srgb => bc16(0x82, (ALPHA, 0), suffix(format == Bc3Srgb)),
        Bc4Unorm => bc8(0x83, 0, Unorm),
        Bc4Snorm => bc8(0x83, 0, Snorm),
        Bc5Unorm => bc16(0x84, (0, 1), Unorm),
        Bc5Snorm => bc16(0x84, (0, 1), Snorm),
        Bc6hUfloat => bc128(0x85, Ufloat),
        Bc6hSfloat => bc128(0x85, Sfloat),
        Bc7Unorm | Bc7Srgb => bc128(0x86, suffix(format == Bc7Srgb)),
        Etc1Rgb | Etc2Rgb | Etc2Srgb => {
            compressed(0xa1, &[(2, 0)], 64, 8, (4, 4), suffix(format == Etc2Srgb))
        }
        Etc2Rgba1 | Etc2Rgba1Srgb => compressed(
            0xa1,
            &[(2, 0), (ALPHA, 0)],
            64,
            8,
            (4, 4),
            suffix(format == Etc2Rgba1Srgb),
        ),
        Etc2Rgba | Etc2RgbaSrgb => compressed(
            0xa1,
            &[(ALPHA, 0), (2, 64)],
            64,
            16,
            (4, 4),
            suffix(format == Etc2RgbaSrgb),
        ),
        EacR11Unorm => compressed(0xa1, &[(0, 0)], 64, 8, (4, 4), Unorm),
        EacR11Snorm => compressed(0xa1, &[(0, 0)], 64, 8, (4, 4), Snorm),
        EacRg11Unorm => compressed(0xa1, &[(0, 0), (1, 64)], 64, 16, (4, 4), Unorm),
        EacRg11Snorm => compressed(0xa1, &[(0, 0), (1, 64)], 64, 16, (4, 4), Snorm),
        Astc(b, kind) => {
            let s = match kind {
                AstcKind::Unorm => Unorm,
                AstcKind::Srgb => Srgb,
                AstcKind::Float => Sfloat,
            };
            compressed(0xa2, &[(0, 0)], 128, 16, b.dim(), s)
        }
        Pvrtc1Rgb | Pvrtc1Rgba => compressed(0xa4, &[(0, 0)], 64, 8, (4, 4), Unorm),
        Pvrtc2Rgb | Pvrtc2Rgba => compressed(0xa5, &[(0, 0)], 64, 8, (4, 4), Unorm),
        AtcRgb | AtcRgba | Fxt1Rgb => return None,
    })
}
