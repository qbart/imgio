//! Radiance RGBE (own implementation).
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
