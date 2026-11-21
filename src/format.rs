//! Pixel and texture formats.

use crate::{Error, Result};

/// ASTC block footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AstcBlock {
    B4x4,
    B5x4,
    B5x5,
    B6x5,
    B6x6,
    B8x5,
    B8x6,
    B8x8,
    B10x5,
    B10x6,
    B10x8,
    B10x10,
    B12x10,
    B12x12,
}

impl AstcBlock {
    /// All block sizes.
    pub const ALL: [AstcBlock; 14] = [
        AstcBlock::B4x4,
        AstcBlock::B5x4,
        AstcBlock::B5x5,
        AstcBlock::B6x5,
        AstcBlock::B6x6,
        AstcBlock::B8x5,
        AstcBlock::B8x6,
        AstcBlock::B8x8,
        AstcBlock::B10x5,
        AstcBlock::B10x6,
        AstcBlock::B10x8,
        AstcBlock::B10x10,
        AstcBlock::B12x10,
        AstcBlock::B12x12,
    ];

    /// Footprint in pixels (width, height).
    pub const fn dim(self) -> (u32, u32) {
        use AstcBlock::*;
        match self {
            B4x4 => (4, 4),
            B5x4 => (5, 4),
            B5x5 => (5, 5),
            B6x5 => (6, 5),
            B6x6 => (6, 6),
            B8x5 => (8, 5),
            B8x6 => (8, 6),
            B8x8 => (8, 8),
            B10x5 => (10, 5),
            B10x6 => (10, 6),
            B10x8 => (10, 8),
            B10x10 => (10, 10),
            B12x10 => (12, 10),
            B12x12 => (12, 12),
        }
    }

    /// The block size with footprint `w` x `h`, if any.
    pub fn from_dim(w: u32, h: u32) -> Option<AstcBlock> {
        Self::ALL.into_iter().find(|b| b.dim() == (w, h))
    }
}

/// ASTC profile / color encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AstcKind {
    /// LDR, linear; decodes to RGBA8 unorm.
    Unorm,
    /// LDR, sRGB; decodes to RGBA8 sRGB.
    Srgb,
    /// HDR profile; decodes to RGBA16F.
    Float,
}

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
    /// 16-bit red, unsigned normalized (little endian).
    R16Unorm,
    /// 16-bit red, signed normalized (little endian).
    R16Snorm,
    /// 16-bit red/green, unsigned normalized (little endian).
    Rg16Unorm,
    /// 16-bit red/green, signed normalized (little endian).
    Rg16Snorm,
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
    /// ETC1 RGB.
    Etc1Rgb,
    /// ETC2 RGB.
    Etc2Rgb,
    /// ETC2 RGB, sRGB.
    Etc2Srgb,
    /// ETC2 RGB with 1-bit punch-through alpha.
    Etc2Rgba1,
    /// ETC2 RGB with 1-bit alpha, sRGB.
    Etc2Rgba1Srgb,
    /// ETC2 RGB with EAC alpha.
    Etc2Rgba,
    /// ETC2 RGB with EAC alpha, sRGB.
    Etc2RgbaSrgb,
    /// EAC single channel, unsigned.
    EacR11Unorm,
    /// EAC single channel, signed.
    EacR11Snorm,
    /// EAC two channels, unsigned.
    EacRg11Unorm,
    /// EAC two channels, signed.
    EacRg11Snorm,
    /// ASTC (16-byte blocks of the given footprint).
    Astc(AstcBlock, AstcKind),
}

impl Format {
    /// Block footprint in pixels (width, height). `(1, 1)` for uncompressed formats.
    pub const fn block_dim(self) -> (u32, u32) {
        match self {
            Format::Astc(b, _) => b.dim(),
            _ if self.is_compressed() => (4, 4),
            _ => (1, 1),
        }
    }

    /// Bytes per block (per pixel for uncompressed formats).
    pub const fn block_bytes(self) -> usize {
        use Format::*;
        match self {
            R8Unorm | R8Snorm => 1,
            Rg8Unorm | Rg8Snorm | R16Unorm | R16Snorm => 2,
            Rgba8Unorm | Rgba8Srgb | Rg16Unorm | Rg16Snorm => 4,
            Rgba16Float | Bc1Unorm | Bc1Srgb | Bc4Unorm | Bc4Snorm | Etc1Rgb | Etc2Rgb
            | Etc2Srgb | Etc2Rgba1 | Etc2Rgba1Srgb | EacR11Unorm | EacR11Snorm => 8,
            Rgba32Float | Bc2Unorm | Bc2Srgb | Bc3Unorm | Bc3Srgb | Bc5Unorm | Bc5Snorm
            | Bc6hUfloat | Bc6hSfloat | Bc7Unorm | Bc7Srgb | Etc2Rgba | Etc2RgbaSrgb
            | EacRg11Unorm | EacRg11Snorm | Astc(..) => 16,
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
                | R16Unorm
                | R16Snorm
                | Rg16Unorm
                | Rg16Snorm
                | Rgba8Unorm
                | Rgba8Srgb
                | Rgba16Float
                | Rgba32Float
        )
    }

    /// True when color channels are sRGB encoded.
    pub const fn is_srgb(self) -> bool {
        use Format::*;
        matches!(
            self,
            Rgba8Srgb
                | Bc1Srgb
                | Bc2Srgb
                | Bc3Srgb
                | Bc7Srgb
                | Etc2Srgb
                | Etc2Rgba1Srgb
                | Etc2RgbaSrgb
        )
    }

    /// True for HDR (float) formats.
    pub const fn is_hdr(self) -> bool {
        use Format::*;
        matches!(
            self,
            Rgba16Float | Rgba32Float | Bc6hUfloat | Bc6hSfloat | Astc(_, AstcKind::Float)
        )
    }

    /// The uncompressed format a block-compressed format decodes to without loss.
    /// Uncompressed formats return themselves.
    pub const fn decoded(self) -> Format {
        use Format::*;
        match self {
            Bc1Unorm | Bc2Unorm | Bc3Unorm | Bc7Unorm | Etc1Rgb | Etc2Rgb | Etc2Rgba1
            | Etc2Rgba => Rgba8Unorm,
            Bc1Srgb | Bc2Srgb | Bc3Srgb | Bc7Srgb | Etc2Srgb | Etc2Rgba1Srgb | Etc2RgbaSrgb => {
                Rgba8Srgb
            }
            Astc(_, AstcKind::Unorm) => Rgba8Unorm,
            Astc(_, AstcKind::Srgb) => Rgba8Srgb,
            Astc(_, AstcKind::Float) => Rgba16Float,
            EacR11Unorm => R16Unorm,
            EacR11Snorm => R16Snorm,
            EacRg11Unorm => Rg16Unorm,
            EacRg11Snorm => Rg16Snorm,
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
    fn astc_sizes() -> Result<()> {
        let f = Format::Astc(AstcBlock::B10x6, AstcKind::Float);
        assert_eq!(f.block_dim(), (10, 6));
        assert_eq!(f.surface_bytes(21, 6)?, 3 * 16);
        assert!(f.is_hdr());
        assert_eq!(AstcBlock::from_dim(12, 10), Some(AstcBlock::B12x10));
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
