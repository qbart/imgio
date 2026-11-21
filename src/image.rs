//! Image views and mip chain layout.

use crate::{Error, Format, Result};
use core::ops::Range;
use vas::Vector;
use vas::storage::Inline;

/// Maximum mip levels (a 2^31 wide image has 32).
pub const MAX_MIPS: usize = 32;

/// Byte ranges of each mip level inside a tightly packed chain.
pub type MipRanges = Vector<Range<usize>, Inline<MAX_MIPS>>;

/// A read-only 2D surface: `format` pixels or blocks, tightly packed rows.
#[derive(Debug, Clone, Copy)]
pub struct Surface<'a> {
    format: Format,
    width: u32,
    height: u32,
    data: &'a [u8],
}

impl<'a> Surface<'a> {
    /// Wraps `data`; it must hold at least `format.surface_bytes(width, height)` bytes
    /// (extra bytes are cut off).
    pub fn new(format: Format, width: u32, height: u32, data: &'a [u8]) -> Result<Self> {
        let len = format.surface_bytes(width, height)?;
        let data = data.get(..len).ok_or(Error::BufferTooSmall {
            needed: len,
            got: data.len(),
        })?;
        Ok(Self {
            format,
            width,
            height,
            data,
        })
    }

    pub fn format(&self) -> Format {
        self.format
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
}

/// Size of mip `level` of a `width` x `height` image (never below 1x1).
pub fn mip_dim(width: u32, height: u32, level: u32) -> (u32, u32) {
    let shr = |v: u32| v.checked_shr(level).unwrap_or(0).max(1);
    (shr(width), shr(height))
}

/// Number of levels in a full mip chain down to 1x1.
pub fn mip_levels(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// Byte ranges of `levels` mips of `format`, packed one after another from level 0.
pub fn mip_ranges(format: Format, width: u32, height: u32, levels: u32) -> Result<MipRanges> {
    if levels == 0 || levels > mip_levels(width, height) {
        return Err(Error::InvalidArgument);
    }
    let mut out = MipRanges::default();
    let mut offset = 0usize;
    for level in 0..levels {
        let (w, h) = mip_dim(width, height, level);
        let end = offset
            .checked_add(format.surface_bytes(w, h)?)
            .ok_or(Error::InvalidArgument)?;
        out.push(offset..end).map_err(|_| Error::InvalidArgument)?;
        offset = end;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mips() -> Result<()> {
        assert_eq!(mip_levels(1, 1), 1);
        assert_eq!(mip_levels(256, 64), 9);
        assert_eq!(mip_levels(5, 3), 3);
        assert_eq!(mip_dim(256, 64, 7), (2, 1));
        assert_eq!(mip_dim(256, 64, 40), (1, 1));
        let r = mip_ranges(Format::Bc1Unorm, 16, 16, 5)?;
        let sizes: Vec<usize> = r.iter().map(|r| r.len()).collect();
        assert_eq!(sizes, [128, 32, 8, 8, 8]);
        assert_eq!(r[4].end, 184);
        assert!(mip_ranges(Format::Bc1Unorm, 16, 16, 6).is_err());
        Ok(())
    }

    #[test]
    fn surface() {
        let data = [0u8; 20];
        assert!(Surface::new(Format::Rgba8Unorm, 2, 2, &data).is_ok_and(|s| s.data().len() == 16));
        assert!(Surface::new(Format::Rgba8Unorm, 3, 2, &data).is_err());
    }
}
