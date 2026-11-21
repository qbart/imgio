//! Pixel and texture formats.

use crate::{Error, Result};

/// A pixel or block-compressed texture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// 8-bit red, unsigned normalized.
    R8Unorm,
    /// 8-bit red, signed normalized.
    R8Snorm,
    /// 8-bit red/green, unsigned normalized.
    Rg8Unorm,
    /// 8-bit red/green, signed normalized.
    Rg8Snorm,
    /// 8-bit RGBA, unsigned normalized.
    Rgba8Unorm,
    /// 8-bit RGBA, sRGB color with linear alpha.
    Rgba8Srgb,
    /// 16-bit float RGBA (IEEE half, little endian).
    Rgba16Float,
    /// 32-bit float RGBA (little endian).
    Rgba32Float,
    /// BC1 (DXT1) RGB with 1-bit alpha.
    Bc1Unorm,
    /// BC1, sRGB.
    Bc1Srgb,
    /// BC2 (DXT3) RGB with explicit 4-bit alpha.
    Bc2Unorm,
    /// BC2, sRGB.
    Bc2Srgb,
    /// BC3 (DXT5) RGB with interpolated alpha.
    Bc3Unorm,
    /// BC3, sRGB.
    Bc3Srgb,
    /// BC4 single channel, unsigned.
    Bc4Unorm,
    /// BC4 single channel, signed.
    Bc4Snorm,
    /// BC5 two channels, unsigned.
    Bc5Unorm,
    /// BC5 two channels, signed.
    Bc5Snorm,
    /// BC6H HDR RGB, unsigned float.
    Bc6hUfloat,
    /// BC6H HDR RGB, signed float.
    Bc6hSfloat,
    /// BC7 RGBA.
    Bc7Unorm,
    /// BC7, sRGB.
    Bc7Srgb,
}

impl Format {
    /// Block footprint in pixels (width, height). `(1, 1)` for uncompressed formats.
    pub const fn block_dim(self) -> (u32, u32) {
        if self.is_compressed() { (4, 4) } else { (1, 1) }
    }

    /// Bytes per block (per pixel for uncompressed formats).
    pub const fn block_bytes(self) -> usize {
        use Format::*;
        match self {
            R8Unorm | R8Snorm => 1,
            Rg8Unorm | Rg8Snorm => 2,
            Rgba8Unorm | Rgba8Srgb => 4,
            Rgba16Float | Bc1Unorm | Bc1Srgb | Bc4Unorm | Bc4Snorm => 8,
            Rgba32Float | Bc2Unorm | Bc2Srgb | Bc3Unorm | Bc3Srgb | Bc5Unorm | Bc5Snorm
            | Bc6hUfloat | Bc6hSfloat | Bc7Unorm | Bc7Srgb => 16,
        }
    }

    /// True for block-compressed formats.
    pub const fn is_compressed(self) -> bool {
        use Format::*;
        !matches!(
            self,
            R8Unorm
                | R8Snorm
                | Rg8Unorm
                | Rg8Snorm
                | Rgba8Unorm
                | Rgba8Srgb
                | Rgba16Float
                | Rgba32Float
        )
    }

    /// True when color channels are sRGB encoded.
    pub const fn is_srgb(self) -> bool {
        use Format::*;
        matches!(self, Rgba8Srgb | Bc1Srgb | Bc2Srgb | Bc3Srgb | Bc7Srgb)
    }

    /// True for HDR (float) formats.
    pub const fn is_hdr(self) -> bool {
        use Format::*;
        matches!(self, Rgba16Float | Rgba32Float | Bc6hUfloat | Bc6hSfloat)
    }

    /// The uncompressed format a block-compressed format decodes to without loss.
    /// Uncompressed formats return themselves.
    pub const fn decoded(self) -> Format {
        use Format::*;
        match self {
            Bc1Unorm | Bc2Unorm | Bc3Unorm | Bc7Unorm => Rgba8Unorm,
            Bc1Srgb | Bc2Srgb | Bc3Srgb | Bc7Srgb => Rgba8Srgb,
            Bc4Unorm => R8Unorm,
            Bc4Snorm => R8Snorm,
            Bc5Unorm => Rg8Unorm,
            Bc5Snorm => Rg8Snorm,
            Bc6hUfloat | Bc6hSfloat => Rgba16Float,
            other => other,
        }
    }

    /// Number of blocks covering a `width` x `height` image.
    pub fn block_count(self, width: u32, height: u32) -> (u32, u32) {
        let (bw, bh) = self.block_dim();
        (width.div_ceil(bw), height.div_ceil(bh))
    }

    /// Bytes in one row of blocks (one pixel row for uncompressed formats).
    pub fn row_bytes(self, width: u32) -> Result<usize> {
        let (bx, _) = self.block_count(width, 1);
        (bx as usize)
            .checked_mul(self.block_bytes())
            .ok_or(Error::InvalidArgument)
    }

    /// Size in bytes of a `width` x `height` surface.
    pub fn surface_bytes(self, width: u32, height: u32) -> Result<usize> {
        let (_, by) = self.block_count(width, height);
        self.row_bytes(width)?
            .checked_mul(by as usize)
            .ok_or(Error::InvalidArgument)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() -> Result<()> {
        assert_eq!(Format::Rgba8Unorm.surface_bytes(3, 2)?, 24);
        assert_eq!(Format::Bc1Unorm.surface_bytes(5, 5)?, 4 * 8);
        assert_eq!(Format::Bc7Srgb.surface_bytes(4, 4)?, 16);
        assert_eq!(Format::Bc7Srgb.surface_bytes(0, 4)?, 0);
        assert_eq!(Format::Rgba32Float.row_bytes(2)?, 32);
        Ok(())
    }

    #[test]
    fn decoded() {
        assert_eq!(Format::Bc7Srgb.decoded(), Format::Rgba8Srgb);
        assert_eq!(Format::Bc5Snorm.decoded(), Format::Rg8Snorm);
        assert_eq!(Format::Bc6hUfloat.decoded(), Format::Rgba16Float);
        assert!(Format::Bc6hSfloat.is_hdr());
    }
}
