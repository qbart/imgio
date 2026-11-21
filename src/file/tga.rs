//! Truevision TGA (own implementation).
//!
//! Not implemented yet: every function returns [`Error::Unsupported`].

use super::Image;
use crate::{Buffer, Error, Result, Surface};

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EncodeOptions {}

/// Decodes the main image.
pub fn decode(data: &[u8]) -> Result<Image> {
    let _ = data;
    Err(Error::Unsupported)
}

/// Encodes `src`.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    let _ = (src, opts);
    Err(Error::Unsupported)
}

/// True when `data` has the TGA 2.0 footer or a plausible TGA header (no signature exists).
pub fn looks_like_tga(data: &[u8]) -> bool {
    if data.len() >= 44 && data.ends_with(b"TRUEVISION-XFILE.\0") {
        return true;
    }
    let Some(h) = data.get(..18) else {
        return false;
    };
    let (cmap_type, img_type, bpp) = (h[1], h[2], h[16]);
    let (w, hh) = (
        u16::from_le_bytes([h[12], h[13]]),
        u16::from_le_bytes([h[14], h[15]]),
    );
    cmap_type <= 1
        && matches!(img_type, 1 | 2 | 3 | 9 | 10 | 11)
        && matches!(bpp, 8 | 15 | 16 | 24 | 32)
        && w > 0
        && hh > 0
        && (img_type & 7 != 1 || cmap_type == 1)
}
