//! BC1-BC7 (DXT / BPTC / RGTC) block-compressed formats.

pub(crate) mod bc1;
mod bc6h;
pub(crate) mod bc7;
mod tables;

use crate::Format;

/// Decodes one block of `format` into 16 pixels of `format.decoded()`, row major.
/// Invalid BC6H/BC7 blocks decode to zero, as on GPUs.
pub(crate) fn decode_block(format: Format, block: &[u8], out: &mut [u8]) {
    use Format::*;
    let half = |b: &[u8]| -> [u8; 8] {
        let mut h = [0u8; 8];
        h.copy_from_slice(&b[..8]);
        h
    };
    let mut rgba = [[0u8; 4]; 16];
    match format {
        Bc1Unorm | Bc1Srgb => bc1::color(&half(block), false, &mut rgba),
        Bc2Unorm | Bc2Srgb => {
            bc1::color(&half(&block[8..]), true, &mut rgba);
            bc1::alpha4(&half(block), &mut rgba);
        }
        Bc3Unorm | Bc3Srgb => {
            bc1::color(&half(&block[8..]), true, &mut rgba);
            let mut a = [0u8; 16];
            bc1::bc4u(&half(block), &mut a);
            for (p, a) in rgba.iter_mut().zip(a) {
                p[3] = a;
            }
        }
        Bc4Unorm | Bc4Snorm | Bc5Unorm | Bc5Snorm => {
            let signed = matches!(format, Bc4Snorm | Bc5Snorm);
            let channels = if matches!(format, Bc4Unorm | Bc4Snorm) {
                1
            } else {
                2
            };
            for c in 0..channels {
                let mut v = [0u8; 16];
                let b = half(&block[c * 8..]);
                if signed {
                    bc1::bc4s(&b, &mut v);
                } else {
                    bc1::bc4u(&b, &mut v);
                }
                for (i, v) in v.iter().enumerate() {
                    out[i * channels + c] = *v;
                }
            }
            return;
        }
        Bc6hUfloat | Bc6hSfloat => {
            let mut b = [0u8; 16];
            b.copy_from_slice(&block[..16]);
            let mut rgb = [[0u16; 3]; 16];
            if !bc6h::unpack_bc6h(&b, &mut rgb, format == Bc6hSfloat) {
                rgb = [[0; 3]; 16];
            }
            for (i, p) in rgb.iter().enumerate() {
                let o = &mut out[i * 8..i * 8 + 8];
                o[0..2].copy_from_slice(&p[0].to_le_bytes());
                o[2..4].copy_from_slice(&p[1].to_le_bytes());
                o[4..6].copy_from_slice(&p[2].to_le_bytes());
                o[6..8].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0
            }
            return;
        }
        Bc7Unorm | Bc7Srgb => {
            let mut b = [0u8; 16];
            b.copy_from_slice(&block[..16]);
            if !bc7::unpack_bc7(&b, &mut rgba) {
                rgba = [[0; 4]; 16];
            }
        }
        _ => return,
    }
    for (i, p) in rgba.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(p);
    }
}
