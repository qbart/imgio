//! Encoding of uncompressed surfaces to block-compressed formats.
//!
//! `encode(src, format, opts, dst)` converts `src` to RGBA8 when needed, then encodes every
//! block, one row of blocks per work unit (multithreaded for large images). Edge blocks read
//! clamped pixels (the last row / column is repeated).
//!
//! Encoders (all ports of Basis Universal, bit-exact with the C++):
//! - BC1: basisu `encode_bc1` (4-color blocks, opaque); quality picks its pass count.
//! - BC4 / BC5: `encode_bc4` on R (and G). BC3: BC4 on A + BC1. BC2: 4-bit alpha + BC1.
//! - BC7: `bc7f::fast_pack_bc7_auto_rgba` (Fastest: fastest, Fast: fast, Default: partially
//!   analytical) and basisu `bc7e_scalar` (High: slow, Best: slowest; perceptual for sRGB).
//! - ETC1 / ETC2 RGB: `etc1f::pack_etc1` up to Default; High / Best: flip x mode search with
//!   basisu `etc1_optimizer` (slow / uber), see `etc1hq`. ETC2 RGBA: EAC alpha + ETC1.
//!   EAC R11 / RG11: `pack_eac` on R (G), `pack_eac_high_quality` from `Quality::High`.
//! - ASTC (every 2D block size; LDR, sRGB, HDR): astcenc (ARM astc-encoder port, via ktx-rs),
//!   presets fastest (0) / fast (10) / medium (60) / thorough (98) / exhaustive (100); its own
//!   threads share one context.
//! - PVRTC1 4bpp RGB / RGBA: basisu `encode_pvrtc1` (whole image; power-of-two block counts).
//! - ETC2 RGB A1 (punch-through, alpha < 128 transparent): `etc1hq::encode_punchthrough_block`
//!   (differential-only optimizer search for opaque blocks; base / base +- b search with
//!   transparent texels). No T / H / planar modes yet.
//! - BC6H signed: own mode 11 / 12 / 13 / 14 single-region encoder (`bc6h_signed`).
//! - BC6H (unsigned): `fast_encode_bc6h` from RGB halves (source converted to RGBA32F;
//!   negative / NaN -> 0, clamped to 65504); quality picks `FastBc6hParams`.

use crate::basis::ll::astc_6x6_hdr::{FastBc6hParams, fast_encode_bc6h};
use crate::basis::ll::bc7f::{
    PACK_BC7_FLAG_DEFAULT_FAST, PACK_BC7_FLAG_DEFAULT_FASTEST,
    PACK_BC7_FLAG_DEFAULT_PARTIALLY_ANALYTICAL, fast_pack_bc7_auto_rgba,
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
    /// Signed normalized targets: pixels are offset bytes (`round(v * 127) + 128`, 1..255) and
    /// the encoder converts its endpoints / base codewords to two's complement.
    Snorm(Encoder4x4),
}

/// BC7 block: `bc7f` up to `Quality::Default`, `bc7e_scalar` above (`perceptual` for sRGB).
fn bc7(px: &[Px; 16], q: Quality, perceptual: bool, out: &mut [u8]) {
    let flags = match q {
        Quality::Fastest => PACK_BC7_FLAG_DEFAULT_FASTEST,
        Quality::Fast => PACK_BC7_FLAG_DEFAULT_FAST,
        Quality::Default => PACK_BC7_FLAG_DEFAULT_PARTIALLY_ANALYTICAL,
        Quality::High | Quality::Best => {
            let params = bc7e_params(q, perceptual);
            out[..16].copy_from_slice(&crate::bc7e::compress_block(px, params));
            return;
        }
    };
    let mut b = [0u8; 16];
    fast_pack_bc7_auto_rgba(&mut b, px, flags);
    out[..16].copy_from_slice(&b);
}

/// The 4 bc7e presets used (High / Best x linear / perceptual), built once.
fn bc7e_params(q: Quality, perceptual: bool) -> &'static crate::bc7e::Params {
    use crate::bc7e::Params;
    static P: std::sync::OnceLock<[Params; 4]> = std::sync::OnceLock::new();
    let p = P.get_or_init(|| {
        [
            Params::slow(false),
            Params::slow(true),
            Params::slowest(false),
            Params::slowest(true),
        ]
    });
    &p[usize::from(q == Quality::Best) * 2 + usize::from(perceptual)]
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

/// ETC1 color block: `etc1f` up to `Quality::Default`, the `etc1_optimizer` search above
/// (slow for High, uber for Best; `perceptual` for sRGB formats).
fn etc1(px: &[Px; 16], q: Quality, perceptual: bool, out: &mut [u8]) {
    let b = if q >= Quality::High {
        crate::etc1hq::encode_block(px, q == Quality::Best, perceptual)
    } else {
        let mut b = [0u8; 8];
        pack_etc1(&mut b, px);
        b
    };
    out[..8].copy_from_slice(&b);
}

/// ETC2 RGB color block: ETC1 (`etc1`), then from `Quality::High` also T / H / planar
/// candidates (`etc2modes`), lowest error wins.
fn etc2(px: &[Px; 16], q: Quality, perceptual: bool, out: &mut [u8]) {
    etc1(px, q, perceptual, out);
    if q >= Quality::High {
        let mut b = [0u8; 8];
        b.copy_from_slice(&out[..8]);
        let effort = u32::from(q == Quality::Best);
        let b = crate::etc2modes::improve(b, px, Format::Etc2Rgb, effort);
        out[..8].copy_from_slice(&b);
    }
}

/// ETC2 RGB A1: the punch-through ETC1-style encoder, then T / H / planar candidates (wider
/// search from `Quality::High`).
fn etc2_rgba1(px: &[Px; 16], q: Quality, perceptual: bool, out: &mut [u8]) {
    let b = crate::etc1hq::encode_punchthrough_block(px, q == Quality::Best, perceptual);
    let effort = u32::from(q >= Quality::High);
    let b = crate::etc2modes::improve(b, px, Format::Etc2Rgba1, effort);
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
        Etc1Rgb => |px, q, out| etc1(px, q, false, out),
        Etc2Rgb => |px, q, out| etc2(px, q, false, out),
        Etc2Srgb => |px, q, out| etc2(px, q, true, out),
        Etc2Rgba => |px, q, out| {
            eac(px, 3, q, out);
            etc2(px, q, false, &mut out[8..]);
        },
        Etc2RgbaSrgb => |px, q, out| {
            eac(px, 3, q, out);
            etc2(px, q, true, &mut out[8..]);
        },
        Etc2Rgba1 => |px, q, out| etc2_rgba1(px, q, false, out),
        Etc2Rgba1Srgb => |px, q, out| etc2_rgba1(px, q, true, out),
        EacR11Unorm => |px, q, out| eac(px, 0, q, out),
        EacRg11Unorm => |px, q, out| {
            eac(px, 0, q, out);
            eac(px, 1, q, &mut out[8..]);
        },
        Bc7Unorm => |px, q, out| bc7(px, q, false, out),
        Bc7Srgb => |px, q, out| bc7(px, q, true, out),
        Bc4Snorm => return Some(Encoder::Snorm(|px, _, out| bc4_snorm(px, 0, out))),
        Bc5Snorm => {
            return Some(Encoder::Snorm(|px, _, out| {
                bc4_snorm(px, 0, out);
                bc4_snorm(px, 1, &mut out[8..]);
            }));
        }
        EacR11Snorm => return Some(Encoder::Snorm(|px, q, out| eac_snorm(px, 0, q, out))),
        EacRg11Snorm => {
            return Some(Encoder::Snorm(|px, q, out| {
                eac_snorm(px, 0, q, out);
                eac_snorm(px, 1, q, &mut out[8..]);
            }));
        }
        Bc6hSfloat => {
            return Some(Encoder::Hdr(|px, q, out| {
                let effort = match q {
                    Quality::Fastest => 0,
                    Quality::Fast | Quality::Default => 1,
                    Quality::High | Quality::Best => 2,
                };
                out[..16].copy_from_slice(&crate::bc6h_signed::encode_block(px, effort));
            }));
        }
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

/// BC4 snorm: `encode_bc4` on the offset bytes; endpoints to signed (offset binary XOR 0x80
/// keeps their order, so the 8 / 6 interpolant mode is the same).
fn bc4_snorm(px: &[Px; 16], c: usize, out: &mut [u8]) {
    bc4(px, c, out);
    out[0] ^= 0x80;
    out[1] ^= 0x80;
}

/// EAC R11 snorm: `pack_eac` on the offset bytes, base codeword to signed (the signed decode
/// lacks the unsigned `+ 4` bias: under half an 8-bit step).
fn eac_snorm(px: &[Px; 16], c: usize, q: Quality, out: &mut [u8]) {
    eac(px, c, q, out);
    out[0] ^= 0x80;
}

/// `v` as a half for the signed BC6H encoder: NaN -> 0, clamped to +-65504.
fn bc6h_shalf(v: f32) -> u16 {
    if v.is_nan() {
        0
    } else {
        half::from_f32(v.clamp(-65504.0, 65504.0))
    }
}

/// True for the signed (offset byte) source path.
fn enc_kind(format: Format) -> bool {
    matches!(encoder(format), Some(Encoder::Snorm(_)))
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
        || matches!(
            format,
            Format::Astc(..) | Format::Pvrtc1Rgb | Format::Pvrtc1Rgba
        )
}

fn astc_error(e: crate::astcenc::AstcencError) -> Error {
    use crate::astcenc::AstcencError as E;
    match e {
        E::OutOfMem => Error::OutOfMemory,
        E::BadCpuFloat | E::NotImplemented => Error::Unsupported,
        _ => Error::InvalidArgument,
    }
}

/// ASTC through astcenc: one context shared by the worker threads.
fn encode_astc(
    src: &Surface,
    block: crate::AstcBlock,
    kind: crate::AstcKind,
    q: Quality,
    dst: &mut [u8],
) -> Result<()> {
    use crate::astcenc::*;
    let (profile, image_format) = match kind {
        crate::AstcKind::Unorm => (AstcencProfile::Ldr, Format::Rgba8Unorm),
        crate::AstcKind::Srgb => (AstcencProfile::LdrSrgb, Format::Rgba8Unorm),
        crate::AstcKind::Float => (AstcencProfile::Hdr, Format::Rgba32Float),
    };
    let quality = match q {
        Quality::Fastest => ASTCENC_PRE_FASTEST,
        Quality::Fast => ASTCENC_PRE_FAST,
        Quality::Default => ASTCENC_PRE_MEDIUM,
        Quality::High => ASTCENC_PRE_THOROUGH,
        Quality::Best => ASTCENC_PRE_EXHAUSTIVE,
    };
    let (bw, bh) = block.dim();
    let config = astcenc_config_init(profile, bw, bh, 1, quality, 0).map_err(astc_error)?;
    let (w, h) = (src.width(), src.height());
    let pixels = (w as usize).saturating_mul(h as usize);
    let threads = if pixels < crate::parallel::MIN_PARALLEL_PIXELS {
        1
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get()) as u32
    };
    let ctx = astcenc_context_alloc(&config, threads).map_err(astc_error)?;
    // astcenc takes owned per-slice vectors.
    let converted;
    let bytes: &[u8] = if src.format() == image_format {
        src.data()
    } else {
        converted = crate::convert::convert_to_buffer(src, image_format)?;
        &converted
    };
    let data = match image_format {
        Format::Rgba8Unorm => {
            let mut v = Vec::new();
            v.try_reserve_exact(bytes.len())
                .map_err(|_| Error::OutOfMemory)?;
            v.extend_from_slice(bytes);
            AstcencImageData::U8(vec![v])
        }
        _ => {
            let mut v = Vec::new();
            v.try_reserve_exact(bytes.len() / 4)
                .map_err(|_| Error::OutOfMemory)?;
            v.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b)),
            );
            AstcencImageData::F32(vec![v])
        }
    };
    let image = AstcencImage {
        dim_x: w,
        dim_y: h,
        dim_z: 1,
        data,
    };
    let out = std::sync::Mutex::new(dst);
    let run = |t: u32| astcenc_compress_image(&ctx, &image, &AstcencSwizzle::RGBA, &out, t);
    if threads <= 1 {
        return run(0).map_err(astc_error);
    }
    // Every thread index must run (astcenc's task manager waits for all of them), so a failed
    // spawn runs that index on this thread afterwards.
    std::thread::scope(|s| {
        let mut handles = vas::Vector::<_, vas::storage::Dynamic>::default();
        let mut inline = vas::Vector::<u32, vas::storage::Dynamic>::default();
        let mut result = Ok(());
        for t in 1..threads {
            match std::thread::Builder::new().spawn_scoped(s, move || run(t)) {
                Ok(h) => {
                    if handles.push(h).is_err() {
                        result = Err(Error::OutOfMemory);
                    }
                }
                Err(_) => {
                    if inline.push(t).is_err() {
                        result = Err(Error::OutOfMemory);
                    }
                }
            }
        }
        let record = |result: &mut Result<()>, r: core::result::Result<(), AstcencError>| {
            if let Err(e) = r {
                *result = Err(astc_error(e));
            }
        };
        record(&mut result, run(0));
        for &t in inline.iter() {
            record(&mut result, run(t));
        }
        while let Some(h) = handles.pop() {
            match h.join() {
                Ok(r) => record(&mut result, r),
                Err(_) => result = Err(Error::InvalidData),
            }
        }
        result
    })
}

/// Encodes `src` (any uncompressed format) to `format` into `dst`
/// (`format.surface_bytes(width, height)` bytes).
///
/// Errors: `Unsupported` (target or source format), `BufferTooSmall`, `OutOfMemory`.
pub fn encode(src: &Surface, format: Format, opts: &EncodeOptions, dst: &mut [u8]) -> Result<()> {
    if !crate::convert::can_read(src.format()) {
        return Err(Error::Unsupported);
    }
    if matches!(format, Format::Pvrtc1Rgb | Format::Pvrtc1Rgba) {
        let (w, h) = (src.width(), src.height());
        let (bx, by) = format.block_count(w, h);
        if !bx.is_power_of_two() || !by.is_power_of_two() {
            return Err(Error::InvalidArgument);
        }
        let len = format.surface_bytes(w, h)?;
        let got = dst.len();
        let dst = dst
            .get_mut(..len)
            .ok_or(Error::BufferTooSmall { needed: len, got })?;
        let converted;
        let rgba: &[u8] = match src.format() {
            Format::Rgba8Unorm | Format::Rgba8Srgb => src.data(),
            _ => {
                converted = crate::convert::convert_to_buffer(src, Format::Rgba8Unorm)?;
                &converted
            }
        };
        return crate::basis::pvrtc1_enc::encode_pvrtc1(
            rgba,
            w as usize,
            h as usize,
            format == Format::Pvrtc1Rgba,
            false,
            bx,
            by,
            dst,
        );
    }
    if let Format::Astc(block, kind) = format {
        let len = format.surface_bytes(src.width(), src.height())?;
        let got = dst.len();
        let dst = dst
            .get_mut(..len)
            .ok_or(Error::BufferTooSmall { needed: len, got })?;
        if len == 0 {
            return Ok(());
        }
        return encode_astc(src, block, kind, opts.quality, dst);
    }
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
        Encoder::Ldr(enc) | Encoder::Snorm(enc) => {
            // RGBA8 source (converted when needed; labels only, no sRGB transfer), or offset
            // bytes of the [-1, 1] values for signed targets.
            let converted;
            let rgba: &[u8] = match (enc_kind(format), src.format()) {
                (false, Format::Rgba8Unorm | Format::Rgba8Srgb) => src.data(),
                (false, _) => {
                    converted = crate::convert::convert_to_buffer(src, Format::Rgba8Unorm)?;
                    &converted
                }
                (true, _) => {
                    let mut f = crate::convert::convert_to_buffer(src, Format::Rgba32Float)?;
                    let n = f.len() / 4;
                    for i in 0..n {
                        let b = &f[i * 4..i * 4 + 4];
                        let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        let v = if v.is_nan() { 0.0 } else { v.clamp(-1.0, 1.0) };
                        f[i] = ((v * 127.0).round() as i32 + 128) as u8;
                    }
                    f.truncate(n);
                    converted = f;
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
            let signed = format == Format::Bc6hSfloat;
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
                            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                            px[i * 3 + c] = if signed { bc6h_shalf(v) } else { bc6h_half(v) };
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
