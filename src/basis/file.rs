//! `.basis` file container: header, slice descriptors and image/level lookup.
//!
//! Layout from Basis Universal `transcoder/basisu_file_headers.h` (packed little endian).

use crate::{AstcBlock, Error, Result};
use std::sync::OnceLock;
use vas::Vector;
use vas::storage::Dynamic;

const SIG: u32 = ((b'B' as u32) << 8) | b's' as u32;
/// `BASISD_SUPPORTED_BASIS_VERSION`.
const VERSION: u32 = 0x13;
const HEADER_SIZE: usize = 77;
const SLICE_DESC_SIZE: usize = 23;

const FLAG_ETC1S: u32 = 1;
const FLAG_Y_FLIPPED: u32 = 2;
const FLAG_HAS_ALPHA_SLICES: u32 = 4;
const FLAG_SRGB: u32 = 16;
const SLICE_FLAG_HAS_ALPHA: u32 = 1;

/// Payload encoding of a Basis Universal texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BasisTexFormat {
    /// ETC1S (BasisLZ), LDR, 4x4 blocks.
    Etc1s,
    /// UASTC LDR, 4x4 blocks.
    UastcLdr4x4,
    /// UASTC HDR, 4x4 blocks.
    UastcHdr4x4,
    /// ASTC HDR 6x6 (plain ASTC blocks).
    AstcHdr6x6,
    /// UASTC HDR 6x6 intermediate (supercompressed).
    UastcHdr6x6Intermediate,
    /// XUASTC LDR: supercompressed ASTC LDR with the given block size.
    XuastcLdr(AstcBlock),
    /// Plain ASTC LDR blocks with the given block size.
    AstcLdr(AstcBlock),
    /// XUBC7: supercompressed BC7.
    Xubc7,
}

/// ASTC block sizes in basisu `basis_tex_format` order (XUASTC 5..18, ASTC LDR 19..32).
pub(crate) const BASIS_ASTC_ORDER: [AstcBlock; 14] = [
    AstcBlock::B4x4,
    AstcBlock::B5x4,
    AstcBlock::B5x5,
    AstcBlock::B6x5,
    AstcBlock::B6x6,
    AstcBlock::B8x5,
    AstcBlock::B8x6,
    AstcBlock::B10x5,
    AstcBlock::B10x6,
    AstcBlock::B8x8,
    AstcBlock::B10x8,
    AstcBlock::B10x10,
    AstcBlock::B12x10,
    AstcBlock::B12x12,
];

impl BasisTexFormat {
    pub(crate) fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Etc1s,
            1 => Self::UastcLdr4x4,
            2 => Self::UastcHdr4x4,
            3 => Self::AstcHdr6x6,
            4 => Self::UastcHdr6x6Intermediate,
            5..=18 => Self::XuastcLdr(BASIS_ASTC_ORDER[v as usize - 5]),
            19..=32 => Self::AstcLdr(BASIS_ASTC_ORDER[v as usize - 19]),
            33 => Self::Xubc7,
            _ => return None,
        })
    }

    /// Block footprint of the payload.
    pub fn block_dim(self) -> (u32, u32) {
        match self {
            Self::AstcHdr6x6 | Self::UastcHdr6x6Intermediate => (6, 6),
            Self::XuastcLdr(b) | Self::AstcLdr(b) => b.dim(),
            _ => (4, 4),
        }
    }

    /// The ASTC block size of XUASTC LDR and ASTC LDR payloads.
    pub fn astc_block(self) -> Option<AstcBlock> {
        match self {
            Self::XuastcLdr(b) | Self::AstcLdr(b) => Some(b),
            _ => None,
        }
    }

    /// True for HDR payloads.
    pub fn is_hdr(self) -> bool {
        matches!(
            self,
            Self::UastcHdr4x4 | Self::AstcHdr6x6 | Self::UastcHdr6x6Intermediate
        )
    }
}

/// `basis_texture_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasisTexType {
    Tex2D,
    Tex2DArray,
    CubemapArray,
    VideoFrames,
    Volume,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SliceDesc {
    pub image_index: u32,
    pub level_index: u32,
    pub flags: u32,
    pub orig_width: u32,
    pub orig_height: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub file_ofs: u32,
    pub file_size: u32,
}

/// Offset and size of a section of the file.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Section {
    pub ofs: u32,
    pub size: u32,
}

/// Size and location of one image level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Blocks horizontally.
    pub blocks_x: u32,
    /// Blocks vertically.
    pub blocks_y: u32,
    /// True when the level carries alpha (ETC1S alpha slice or UASTC alpha).
    pub has_alpha: bool,
}

/// A parsed `.basis` file borrowing its bytes.
#[derive(Debug)]
pub struct BasisFile<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) tex_format: BasisTexFormat,
    pub(crate) tex_type: BasisTexType,
    pub(crate) flags: u32,
    pub(crate) total_images: u32,
    pub(crate) us_per_frame: u32,
    pub(crate) total_endpoints: u32,
    pub(crate) endpoint_cb: Section,
    pub(crate) total_selectors: u32,
    pub(crate) selector_cb: Section,
    pub(crate) tables: Section,
    pub(crate) slices: Vector<SliceDesc, Dynamic>,
    /// ETC1S codebooks and Huffman tables, decoded on first transcode (None if corrupt).
    pub(crate) etc1s: OnceLock<Option<super::ll::etc1s::LowlevelEtc1sTranscoder>>,
}

/// Basis Universal CRC16 (`basist::crc16`).
pub(crate) fn crc16(data: &[u8], crc: u16) -> u16 {
    let mut crc = !crc;
    for &b in data {
        let q = (b as u16) ^ (crc >> 8);
        let k = (q >> 4) ^ q;
        crc = (((crc << 8) ^ k) ^ (k << 5)) ^ (k << 12);
    }
    !crc
}

fn le(b: &[u8], ofs: usize, n: usize) -> u32 {
    b[ofs..ofs + n]
        .iter()
        .rev()
        .fold(0u32, |v, &x| (v << 8) | x as u32)
}

fn section(data: &[u8], ofs: u32, size: u32) -> Result<Section> {
    let end = (ofs as usize).checked_add(size as usize);
    if end.is_none_or(|e| e > data.len()) {
        return Err(Error::InvalidData);
    }
    Ok(Section { ofs, size })
}

impl<'a> BasisFile<'a> {
    /// Parses and validates the header, CRCs and slice descriptors.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() <= HEADER_SIZE {
            return Err(Error::InvalidData);
        }
        let h = |ofs, n| le(data, ofs, n);
        if h(0, 2) != SIG || h(2, 2) != VERSION || h(4, 2) as usize != HEADER_SIZE {
            return Err(Error::InvalidData);
        }
        if crc16(&data[8..HEADER_SIZE], 0) != h(6, 2) as u16 {
            return Err(Error::InvalidData);
        }
        let data_size = h(8, 4) as usize;
        let body = data
            .get(HEADER_SIZE..HEADER_SIZE + data_size)
            .ok_or(Error::InvalidData)?;
        if crc16(body, 0) != h(12, 2) as u16 {
            return Err(Error::InvalidData);
        }
        let total_slices = h(14, 3);
        let total_images = h(17, 3);
        let tex_format = BasisTexFormat::from_u32(h(20, 1)).ok_or(Error::Unsupported)?;
        let flags = h(21, 2);
        let tex_type = match h(23, 1) {
            0 => BasisTexType::Tex2D,
            1 => BasisTexType::Tex2DArray,
            2 => BasisTexType::CubemapArray,
            3 => BasisTexType::VideoFrames,
            4 => BasisTexType::Volume,
            _ => return Err(Error::InvalidData),
        };
        if total_slices == 0 || total_images == 0 || total_images > total_slices {
            return Err(Error::InvalidData);
        }
        let etc1s = tex_format == BasisTexFormat::Etc1s;
        if etc1s != (flags & FLAG_ETC1S != 0)
            || (etc1s && flags & FLAG_HAS_ALPHA_SLICES != 0 && total_slices & 1 != 0)
        {
            return Err(Error::InvalidData);
        }
        let endpoint_cb = section(data, h(41, 4), h(45, 3))?;
        let selector_cb = section(data, h(50, 4), h(54, 3))?;
        let tables = section(data, h(57, 4), h(61, 4))?;
        if etc1s && (endpoint_cb.size == 0 || selector_cb.size == 0 || tables.size == 0) {
            return Err(Error::InvalidData);
        }
        let desc_ofs = h(65, 4) as usize;
        let descs = (total_slices as usize)
            .checked_mul(SLICE_DESC_SIZE)
            .and_then(|n| data.get(desc_ofs..desc_ofs.checked_add(n)?))
            .ok_or(Error::InvalidData)?;
        let mut slices = Vector::<SliceDesc, Dynamic>::default();
        slices
            .reserve(total_slices as usize)
            .map_err(|_| Error::OutOfMemory)?;
        for d in descs.as_chunks::<SLICE_DESC_SIZE>().0 {
            let s = |ofs, n| le(d, ofs, n);
            let desc = SliceDesc {
                image_index: s(0, 3),
                level_index: s(3, 1),
                flags: s(4, 1),
                orig_width: s(5, 2),
                orig_height: s(7, 2),
                blocks_x: s(9, 2),
                blocks_y: s(11, 2),
                file_ofs: s(13, 4),
                file_size: s(17, 4),
            };
            section(data, desc.file_ofs, desc.file_size)?;
            let (bw, bh) = tex_format.block_dim();
            if desc.blocks_x != desc.orig_width.div_ceil(bw)
                || desc.blocks_y != desc.orig_height.div_ceil(bh)
                || desc.image_index >= total_images
            {
                return Err(Error::InvalidData);
            }
            slices.push(desc).map_err(|_| Error::OutOfMemory)?;
        }
        Ok(Self {
            data,
            tex_format,
            tex_type,
            flags,
            total_images,
            us_per_frame: h(24, 3),
            total_endpoints: h(39, 2),
            endpoint_cb,
            total_selectors: h(48, 2),
            selector_cb,
            tables,
            slices,
            etc1s: OnceLock::new(),
        })
    }

    /// Payload format.
    pub fn tex_format(&self) -> BasisTexFormat {
        self.tex_format
    }

    /// Texture type (2D, array, cubemap array, video, volume).
    pub fn tex_type(&self) -> BasisTexType {
        self.tex_type
    }

    /// Number of images (array layers, cube faces, frames or slices).
    pub fn image_count(&self) -> u32 {
        self.total_images
    }

    /// True when the texture has alpha.
    pub fn has_alpha(&self) -> bool {
        self.flags & FLAG_HAS_ALPHA_SLICES != 0
    }

    /// True when the ASTC LDR / XUASTC LDR payload was encoded for the sRGB decode profile.
    pub fn is_srgb(&self) -> bool {
        self.flags & FLAG_SRGB != 0
    }

    /// True when the encoder flipped the image vertically.
    pub fn y_flipped(&self) -> bool {
        self.flags & FLAG_Y_FLIPPED != 0
    }

    /// Microseconds per frame (video textures).
    pub fn us_per_frame(&self) -> u32 {
        self.us_per_frame
    }

    /// Number of mip levels of `image`.
    pub fn level_count(&self, image: u32) -> u32 {
        self.slices
            .iter()
            .filter(|s| s.image_index == image)
            .map(|s| s.level_index + 1)
            .max()
            .unwrap_or(0)
    }

    /// `find_slice`: the slice holding `image`/`level` (the alpha slice for ETC1S when `alpha`).
    pub(crate) fn slice(&self, image: u32, level: u32, alpha: bool) -> Option<&SliceDesc> {
        self.slices.iter().find(|s| {
            s.image_index == image
                && s.level_index == level
                && (self.tex_format != BasisTexFormat::Etc1s
                    || (s.flags & SLICE_FLAG_HAS_ALPHA != 0) == alpha)
        })
    }

    /// Size and block counts of `image`/`level`.
    pub fn level_info(&self, image: u32, level: u32) -> Result<LevelInfo> {
        let s = self
            .slice(image, level, false)
            .ok_or(Error::InvalidArgument)?;
        Ok(LevelInfo {
            width: s.orig_width,
            height: s.orig_height,
            blocks_x: s.blocks_x,
            blocks_y: s.blocks_y,
            has_alpha: self.has_alpha(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc() {
        assert_eq!(crc16(b"", 0), 0);
        assert_eq!(crc16(b"123456789", 0), crc16(b"6789", crc16(b"12345", 0)));
    }

    #[test]
    fn tex_format_ids() {
        assert_eq!(
            BasisTexFormat::from_u32(5),
            Some(BasisTexFormat::XuastcLdr(AstcBlock::B4x4))
        );
        assert_eq!(
            BasisTexFormat::from_u32(14),
            Some(BasisTexFormat::XuastcLdr(AstcBlock::B8x8))
        );
        assert_eq!(
            BasisTexFormat::from_u32(19),
            Some(BasisTexFormat::AstcLdr(AstcBlock::B4x4))
        );
        assert_eq!(
            BasisTexFormat::from_u32(32),
            Some(BasisTexFormat::AstcLdr(AstcBlock::B12x12))
        );
        assert_eq!(BasisTexFormat::from_u32(33), Some(BasisTexFormat::Xubc7));
        assert_eq!(BasisTexFormat::from_u32(34), None);
        assert_eq!(
            BasisTexFormat::AstcLdr(AstcBlock::B10x8).block_dim(),
            (10, 8)
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(BasisFile::parse(&[0u8; 10]).is_err());
        assert!(BasisFile::parse(&[0u8; 200]).is_err());
    }
}
