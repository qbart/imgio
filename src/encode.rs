//! Encoding of uncompressed surfaces to block-compressed formats.
//!
//! `encode(src, format, opts, dst)` converts `src` to RGBA8 when needed, then encodes every
//! block, one row of blocks per work unit (multithreaded for large images). Edge blocks read
//! clamped pixels (the last row / column is repeated).
//!
//! Encoders (all ports of Basis Universal, bit-exact with the C++):
//! - BC1: basisu `encode_bc1` (4-color blocks, opaque); quality picks its pass count.
//! - BC4 / BC5: `encode_bc4` on R (and G). BC3: BC4 on A + BC1. BC2: 4-bit alpha + BC1.
//! - BC7: `bc7f::fast_pack_bc7_auto_rgba` presets (fastest .. non-analytical).
//! - ETC1 / ETC2 RGB: `etc1f::pack_etc1` (one quality). ETC2 RGBA: EAC alpha + ETC1.
//!   EAC R11 / RG11: `pack_eac` on R (G), `pack_eac_high_quality` from `Quality::High`.
//! - BC6H (unsigned): `fast_encode_bc6h` from RGB halves (source converted to RGBA32F;
//!   negative / NaN -> 0, clamped to 65504); quality picks `FastBc6hParams`.

use crate::basis::ll::astc_6x6_hdr::{FastBc6hParams, fast_encode_bc6h};
use crate::basis::ll::bc7f::{
    PACK_BC7_FLAG_DEFAULT, PACK_BC7_FLAG_DEFAULT_FAST, PACK_BC7_FLAG_DEFAULT_FASTEST,
    PACK_BC7_FLAG_DEFAULT_NON_ANALYTICAL, PACK_BC7_FLAG_DEFAULT_PARTIALLY_ANALYTICAL,
    fast_pack_bc7_auto_rgba,
};
use crate::basis::ll::dxt::Dxt1Block;
use crate::basis::ll::eac::EacBlock;
use crate::basis::ll::etc::Color32;
use crate::basis::ll::etc1f::pack_etc1;
use crate::basis::ll::uastc_bc1::{
    ENCODE_BC1_HIGH_QUALITY, ENCODE_BC1_HIGHER_QUALITY, encode_bc1, encode_bc4,
};
use crate::basis::ll::uastc_etc::{pack_eac, pack_eac_high_quality};
use crate::parallel::try_for_each_unit;
use crate::{Buffer, Error, Format, Result, Surface, half};

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

/// One 4x4 HDR block encoder: 16 RGB half pixels (row major) -> `out`.
type EncoderHdr = fn(&[u16; 48], Quality, &mut [u8]);

#[derive(Clone, Copy)]
enum Encoder {
    Ldr(Encoder4x4),
    Hdr(EncoderHdr),
}

fn bc7_flags(q: Quality) -> u32 {
    match q {
        Quality::Fastest => PACK_BC7_FLAG_DEFAULT_FASTEST,
        Quality::Fast => PACK_BC7_FLAG_DEFAULT_FAST,
        Quality::Default => PACK_BC7_FLAG_DEFAULT,
        Quality::High => PACK_BC7_FLAG_DEFAULT_PARTIALLY_ANALYTICAL,
        Quality::Best => PACK_BC7_FLAG_DEFAULT_NON_ANALYTICAL,
    }
}

fn bc6h_params(q: Quality) -> FastBc6hParams {
    let (modes, pats, hq_ls, brute) = match q {
        Quality::Fastest => (0, 0, false, false),
        Quality::Fast => (2, 0, false, false),
        Quality::Default => (2, 1, true, false),
        Quality::High => (4, 8, true, false),
        Quality::Best => (9, 32, true, true),
    };
    FastBc6hParams {
        num_diff_endpoint_modes_to_try: modes,
        max_2subset_pats_to_try: pats,
        hq_ls,
        brute_force_weight4_assignment: brute,
    }
}

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

fn etc1(px: &[Px; 16], out: &mut [u8]) {
    let mut b = [0u8; 8];
    pack_etc1(&mut b, px);
    out[..8].copy_from_slice(&b);
}

fn eac(px: &[Px; 16], c: usize, q: Quality, out: &mut [u8]) {
    let mut b = EacBlock::default();
    let v = px.map(|p| p[c]);
    if q >= Quality::High {
        pack_eac_high_quality(&mut b, &v);
    } else {
        pack_eac(&mut b, &v);
    }
    out[..8].copy_from_slice(&b.bytes);
}

fn encoder(format: Format) -> Option<Encoder> {
    use Format::*;
    Some(Encoder::Ldr(match format {
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
        Etc1Rgb | Etc2Rgb | Etc2Srgb => |px, _, out| etc1(px, out),
        Etc2Rgba | Etc2RgbaSrgb => |px, q, out| {
            eac(px, 3, q, out);
            etc1(px, &mut out[8..]);
        },
        EacR11Unorm => |px, q, out| eac(px, 0, q, out),
        EacRg11Unorm => |px, q, out| {
            eac(px, 0, q, out);
            eac(px, 1, q, &mut out[8..]);
        },
        Bc7Unorm | Bc7Srgb => |px, q, out| {
            let mut b = [0u8; 16];
            fast_pack_bc7_auto_rgba(&mut b, px, bc7_flags(q));
            out[..16].copy_from_slice(&b);
        },
        Bc6hUfloat => {
            return Some(Encoder::Hdr(|px, q, out| {
                let mut b = [0u8; 16];
                fast_encode_bc6h(px, &mut b, &bc6h_params(q));
                out[..16].copy_from_slice(&b);
            }));
        }
        _ => return None,
    }))
}

/// `v` as a half for the unsigned BC6H encoder: NaN / negative -> 0, clamped to 65504.
fn bc6h_half(v: f32) -> u16 {
    if v > 0.0 {
        half::from_f32(v.min(65504.0))
    } else {
        0
    }
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
    if !crate::convert::can_read(src.format()) {
        return Err(Error::Unsupported);
    }
    let (w, h) = (src.width(), src.height());
    let len = format.surface_bytes(w, h)?;
    let got = dst.len();
    let dst = dst
        .get_mut(..len)
        .ok_or(Error::BufferTooSmall { needed: len, got })?;
    if len == 0 {
        return Ok(());
    }
    let (wu, hu) = (w as usize, h as usize);
    let bb = format.block_bytes();
    let row = format.row_bytes(w)?;
    let q = opts.quality;
    // Source pixel index of pixel `i` of block (`bx`, `by`), edges clamped.
    let at = move |bx: usize, by: usize, i: usize| {
        let x = (bx * 4 + (i & 3)).min(wu - 1);
        let y = (by * 4 + (i >> 2)).min(hu - 1);
        y * wu + x
    };
    match enc {
        Encoder::Ldr(enc) => {
            // RGBA8 source (converted when needed; labels only, no sRGB transfer).
            let converted;
            let rgba: &[u8] = match src.format() {
                Format::Rgba8Unorm | Format::Rgba8Srgb => src.data(),
                _ => {
                    converted = crate::convert::convert_to_buffer(src, Format::Rgba8Unorm)?;
                    &converted
                }
            };
            try_for_each_unit(dst, row, wu * hu, |by, out| {
                for (bx, o) in out.chunks_exact_mut(bb).enumerate() {
                    let mut px = [[0u8; 4]; 16];
                    for (i, p) in px.iter_mut().enumerate() {
                        let a = at(bx, by, i) * 4;
                        p.copy_from_slice(&rgba[a..a + 4]);
                    }
                    enc(&px, q, o);
                }
                Ok(())
            })
        }
        Encoder::Hdr(enc) => {
            let converted;
            let rgba: &[u8] = match src.format() {
                Format::Rgba32Float => src.data(),
                _ => {
                    converted = crate::convert::convert_to_buffer(src, Format::Rgba32Float)?;
                    &converted
                }
            };
            try_for_each_unit(dst, row, wu * hu, |by, out| {
                for (bx, o) in out.chunks_exact_mut(bb).enumerate() {
                    let mut px = [0u16; 48];
                    for i in 0..16 {
                        let a = at(bx, by, i) * 16;
                        for c in 0..3 {
                            let b = &rgba[a + c * 4..a + c * 4 + 4];
                            px[i * 3 + c] = bc6h_half(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                        }
                    }
                    enc(&px, q, o);
                }
                Ok(())
            })
        }
    }
}

/// Like [`encode`] but allocates the output.
pub fn encode_to_buffer(src: &Surface, format: Format, opts: &EncodeOptions) -> Result<Buffer> {
    let mut buf = crate::buffer::zeroed(format.surface_bytes(src.width(), src.height())?)?;
    encode(src, format, opts, &mut buf)?;
    Ok(buf)
}
