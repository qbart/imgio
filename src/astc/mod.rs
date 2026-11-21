//! ASTC LDR and HDR block decoding (via the Basis Universal ASTC helpers).

#[allow(dead_code)]
pub(crate) mod helpers;
#[allow(dead_code)]
pub(crate) mod tables;

use crate::{AstcKind, Format};
use helpers::{DecodePixels, LogAstcBlock, decode_block as decode_log, unpack_block};

/// Decodes one ASTC block into `w * h` pixels of `format.decoded()`, row major.
/// Invalid blocks (and HDR blocks in the LDR profiles) decode to the error color: magenta for
/// LDR, all ones (NaN) for HDR.
pub(crate) fn decode_block(format: Format, block: &[u8], out: &mut [u8]) {
    let Format::Astc(size, kind) = format else {
        return;
    };
    let (w, h) = size.dim();
    let n = (w * h) as usize;
    let mut b = [0u8; 16];
    b.copy_from_slice(&block[..16]);
    let mut log = LogAstcBlock::default();
    // A failed unpack leaves error_flag set, which decode_log turns into the error color.
    let _ = unpack_block(&b, &mut log, w, h);
    match kind {
        AstcKind::Float => {
            let mut px = [[0u16; 4]; helpers::MAX_BLOCK_PIXELS];
            let _ = decode_log(&log, &mut DecodePixels::Hdr16(&mut px[..n]), w, h);
            for (o, p) in out.as_chunks_mut::<8>().0.iter_mut().zip(&px[..n]) {
                for c in 0..4 {
                    o[c * 2..c * 2 + 2].copy_from_slice(&p[c].to_le_bytes());
                }
            }
        }
        AstcKind::Unorm | AstcKind::Srgb => {
            let mut px = [[0u8; 4]; helpers::MAX_BLOCK_PIXELS];
            let pixels = if kind == AstcKind::Srgb {
                DecodePixels::Srgb8(&mut px[..n])
            } else {
                DecodePixels::Ldr8(&mut px[..n])
            };
            let _ = decode_log(&log, &mut { pixels }, w, h);
            for (o, p) in out.as_chunks_mut::<4>().0.iter_mut().zip(&px[..n]) {
                o.copy_from_slice(p);
            }
        }
    }
}
