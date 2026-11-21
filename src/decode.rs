//! Decoding of block-compressed surfaces.

use crate::block::{BlockDecoder, decode_surface};
use crate::{Buffer, Error, Format, Result, astc, atc, bc, etc, fxt1, pvrtc};

/// How a format is decoded.
enum Decoder {
    /// Block by block.
    Block(BlockDecoder),
    /// PVRTC1: whole surface (blocks interpolate with their neighbors).
    Pvrtc1,
}

fn decoder(format: Format) -> Option<Decoder> {
    use Format::*;
    let block: BlockDecoder = match format {
        Bc1Unorm | Bc1Srgb | Bc2Unorm | Bc2Srgb | Bc3Unorm | Bc3Srgb | Bc4Unorm | Bc4Snorm
        | Bc5Unorm | Bc5Snorm | Bc6hUfloat | Bc6hSfloat | Bc7Unorm | Bc7Srgb => |f, b, o| {
            bc::decode_block(f, b, o);
            Ok(())
        },
        Etc1Rgb | Etc2Rgb | Etc2Srgb | Etc2Rgba1 | Etc2Rgba1Srgb | Etc2Rgba | Etc2RgbaSrgb
        | EacR11Unorm | EacR11Snorm | EacRg11Unorm | EacRg11Snorm => |f, b, o| {
            etc::decode_block(f, b, o);
            Ok(())
        },
        Astc(..) => |f, b, o| {
            astc::decode_block(f, b, o);
            Ok(())
        },
        AtcRgb | AtcRgba => |f, b, o| {
            atc::decode_block(f, b, o);
            Ok(())
        },
        Fxt1Rgb => |_, b, o| fxt1::decode_block(b, o),
        Pvrtc2Rgb | Pvrtc2Rgba => |_, b, o| pvrtc::decode_pvrtc2_block(b, o),
        Pvrtc1Rgb | Pvrtc1Rgba => return Some(Decoder::Pvrtc1),
        _ => return None,
    };
    Some(Decoder::Block(block))
}

/// True when [`decode`] supports `format`.
pub fn can_decode(format: Format) -> bool {
    decoder(format).is_some()
}

/// Decodes a `width` x `height` block-compressed surface from `src` into `dst` as tightly
/// packed rows of `format.decoded()`. Edge blocks are cropped. Multithreaded for large surfaces.
///
/// `src` needs `format.surface_bytes(width, height)` bytes and `dst` needs
/// `format.decoded().surface_bytes(width, height)` bytes; extra bytes are ignored.
///
/// PVRTC1 needs power-of-two block counts (`InvalidArgument` otherwise). FXT1 decodes CC_MIXED
/// opaque blocks and PVRTC2 hard-transition blocks only (the modes Basis Universal emits);
/// other block modes return `Unsupported`.
pub fn decode(format: Format, src: &[u8], width: u32, height: u32, dst: &mut [u8]) -> Result<()> {
    match decoder(format).ok_or(Error::Unsupported)? {
        Decoder::Block(f) => decode_surface(format, src, width, height, dst, f),
        Decoder::Pvrtc1 => {
            let (src_len, dst_len) = (
                format.surface_bytes(width, height)?,
                format.decoded().surface_bytes(width, height)?,
            );
            if src.len() < src_len {
                return Err(Error::BufferTooSmall {
                    needed: src_len,
                    got: src.len(),
                });
            }
            if dst.len() < dst_len {
                return Err(Error::BufferTooSmall {
                    needed: dst_len,
                    got: dst.len(),
                });
            }
            pvrtc::decode_pvrtc1(src, width, height, dst)
        }
    }
}

/// Like [`decode`] but allocates the output buffer.
pub fn decode_to_buffer(format: Format, src: &[u8], width: u32, height: u32) -> Result<Buffer> {
    let len = format.decoded().surface_bytes(width, height)?;
    let mut buf = crate::buffer::zeroed(len)?;
    decode(format, src, width, height, &mut buf)?;
    Ok(buf)
}
