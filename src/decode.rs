//! Decoding of block-compressed surfaces.

use crate::block::{BlockDecoder, decode_surface};
use crate::{Buffer, Error, Format, Result, bc, etc};

fn decoder(format: Format) -> Option<BlockDecoder> {
    use Format::*;
    match format {
        Bc1Unorm | Bc1Srgb | Bc2Unorm | Bc2Srgb | Bc3Unorm | Bc3Srgb | Bc4Unorm | Bc4Snorm
        | Bc5Unorm | Bc5Snorm | Bc6hUfloat | Bc6hSfloat | Bc7Unorm | Bc7Srgb => {
            Some(bc::decode_block)
        }
        Etc1Rgb | Etc2Rgb | Etc2Srgb | Etc2Rgba1 | Etc2Rgba1Srgb | Etc2Rgba | Etc2RgbaSrgb
        | EacR11Unorm | EacR11Snorm | EacRg11Unorm | EacRg11Snorm => Some(etc::decode_block),
        _ => None,
    }
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
pub fn decode(format: Format, src: &[u8], width: u32, height: u32, dst: &mut [u8]) -> Result<()> {
    let f = decoder(format).ok_or(Error::Unsupported)?;
    decode_surface(format, src, width, height, dst, f)
}

/// Like [`decode`] but allocates the output buffer.
pub fn decode_to_buffer(format: Format, src: &[u8], width: u32, height: u32) -> Result<Buffer> {
    let len = format.decoded().surface_bytes(width, height)?;
    let mut buf = crate::buffer::zeroed(len)?;
    decode(format, src, width, height, &mut buf)?;
    Ok(buf)
}
