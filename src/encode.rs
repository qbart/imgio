//! Encoding of uncompressed surfaces to block-compressed formats.
//!
//! `encode(src, format, opts, dst)` converts `src` to RGBA8 when needed, then encodes every
//! block, one row of blocks per work unit (multithreaded for large images). Edge blocks read
//! clamped pixels (the last row / column is repeated).
//!
//! Encoders (all ports of Basis Universal, bit-exact with the C++):
//! - BC1: basisu `encode_bc1` (4-color blocks, opaque); quality picks its pass count.
//! - BC4 / BC5: `encode_bc4` on R (and G). BC3: BC4 on A + BC1. BC2: 4-bit alpha + BC1.

use crate::basis::ll::dxt::Dxt1Block;
use crate::basis::ll::etc::Color32;
use crate::basis::ll::uastc_bc1::{
    ENCODE_BC1_HIGH_QUALITY, ENCODE_BC1_HIGHER_QUALITY, encode_bc1, encode_bc4,
};
use crate::parallel::try_for_each_unit;
use crate::{Buffer, Error, Format, Result, Surface};

/// Speed / quality trade-off; each encoder maps it to its own settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord)]
pub enum Quality {
    Fastest,
    Fast,
    #[default]
    Default,
    High,
    Best,
}

/// Options of [`encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeOptions {
    pub quality: Quality,
}

type Px = [u8; 4];

/// One 4x4 block encoder: 16 RGBA8 pixels (row major) -> `out` (block bytes).
type Encoder4x4 = fn(&[Px; 16], Quality, &mut [u8]);

fn bc1_flags(q: Quality) -> u32 {
    match q {
        Quality::Fastest | Quality::Fast => 0,
        Quality::Default => ENCODE_BC1_HIGH_QUALITY,
        Quality::High | Quality::Best => ENCODE_BC1_HIGHER_QUALITY,
    }
}

fn bc1(px: &[Px; 16], q: Quality, out: &mut [u8]) {
    let colors = px.map(|c| Color32 { c });
    let mut b = Dxt1Block::default();
    encode_bc1(&mut b, &colors, bc1_flags(q));
    out[..8].copy_from_slice(&b.bytes);
}

fn bc4(px: &[Px; 16], c: usize, out: &mut [u8]) {
    let mut b = [0u8; 8];
    encode_bc4(&mut b, &px.map(|p| p[c]));
    out[..8].copy_from_slice(&b);
}

fn encoder(format: Format) -> Option<Encoder4x4> {
    use Format::*;
    Some(match format {
        Bc1Unorm | Bc1Srgb => |px, q, out| bc1(px, q, out),
        Bc2Unorm | Bc2Srgb => |px, q, out| {
            // Explicit alpha: 4 bits per pixel, rounded.
            for (i, pair) in px.as_chunks::<2>().0.iter().enumerate() {
                let a = |p: &Px| (u32::from(p[3]) * 15 + 127) / 255;
                out[i] = (a(&pair[0]) | (a(&pair[1]) << 4)) as u8;
            }
            bc1(px, q, &mut out[8..]);
        },
        Bc3Unorm | Bc3Srgb => |px, q, out| {
            bc4(px, 3, out);
            bc1(px, q, &mut out[8..]);
        },
        Bc4Unorm => |px, _, out| bc4(px, 0, out),
        Bc5Unorm => |px, _, out| {
            bc4(px, 0, out);
            bc4(px, 1, &mut out[8..]);
        },
        _ => return None,
    })
}

/// True when [`encode`] supports `format`.
pub fn can_encode(format: Format) -> bool {
    encoder(format).is_some()
}

/// Encodes `src` (any uncompressed format) to `format` into `dst`
/// (`format.surface_bytes(width, height)` bytes).
///
/// Errors: `Unsupported` (target or source format), `BufferTooSmall`, `OutOfMemory`.
pub fn encode(src: &Surface, format: Format, opts: &EncodeOptions, dst: &mut [u8]) -> Result<()> {
    let enc = encoder(format).ok_or(Error::Unsupported)?;
    let (w, h) = (src.width(), src.height());
    let len = format.surface_bytes(w, h)?;
    let got = dst.len();
    let dst = dst
        .get_mut(..len)
        .ok_or(Error::BufferTooSmall { needed: len, got })?;
    if len == 0 {
        return Ok(());
    }
    // RGBA8 source (converted when needed; labels only, no sRGB transfer).
    let converted;
    let rgba: &[u8] = match src.format() {
        Format::Rgba8Unorm | Format::Rgba8Srgb => src.data(),
        f if crate::convert::can_read(f) => {
            converted = crate::convert::convert_to_buffer(src, Format::Rgba8Unorm)?;
            &converted
        }
        _ => return Err(Error::Unsupported),
    };
    let (wu, hu) = (w as usize, h as usize);
    let bb = format.block_bytes();
    let row = format.row_bytes(w)?;
    let q = opts.quality;
    try_for_each_unit(dst, row, wu * hu, |by, out| {
        for (bx, o) in out.chunks_exact_mut(bb).enumerate() {
            let mut px = [[0u8; 4]; 16];
            for (i, p) in px.iter_mut().enumerate() {
                let x = (bx * 4 + (i & 3)).min(wu - 1);
                let y = (by * 4 + (i >> 2)).min(hu - 1);
                let at = (y * wu + x) * 4;
                p.copy_from_slice(&rgba[at..at + 4]);
            }
            enc(&px, q, o);
        }
        Ok(())
    })
}

/// Like [`encode`] but allocates the output.
pub fn encode_to_buffer(src: &Surface, format: Format, opts: &EncodeOptions) -> Result<Buffer> {
    let mut buf = crate::buffer::zeroed(format.surface_bytes(src.width(), src.height())?)?;
    encode(src, format, opts, &mut buf)?;
    Ok(buf)
}
