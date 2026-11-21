//! ARM `.astc` files (astcenc): a 16-byte header (magic `0x5CA1AB13`, block x / y / z, 24-bit
//! little-endian width / height / depth) followed by the blocks, x fastest, then y, then z.
//! The header does not say LDR, sRGB or HDR; [`AstcFile::format`] takes the decode profile.

use crate::{AstcBlock, AstcKind, Buffer, Error, Format, Result};

const MAGIC: u32 = 0x5CA1_AB13;

/// A parsed `.astc` file, borrowing its bytes.
#[derive(Debug, Clone, Copy)]
pub struct AstcFile<'a> {
    block: (u32, u32, u32),
    width: u32,
    height: u32,
    depth: u32,
    data: &'a [u8],
}

fn u24(b: &[u8]) -> u32 {
    u32::from(b[0]) | (u32::from(b[1]) << 8) | (u32::from(b[2]) << 16)
}

impl<'a> AstcFile<'a> {
    /// Parses the header; the data must hold every block.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() < 16 || u32::from_le_bytes([data[0], data[1], data[2], data[3]]) != MAGIC {
            return Err(Error::InvalidData);
        }
        let block = (u32::from(data[4]), u32::from(data[5]), u32::from(data[6]));
        let (width, height, depth) = (u24(&data[7..]), u24(&data[10..]), u24(&data[13..]));
        if block.0 == 0 || block.1 == 0 || block.2 == 0 || width == 0 || height == 0 || depth == 0 {
            return Err(Error::InvalidData);
        }
        let blocks = u64::from(width.div_ceil(block.0))
            * u64::from(height.div_ceil(block.1))
            * u64::from(depth.div_ceil(block.2));
        let len = usize::try_from(blocks * 16).map_err(|_| Error::InvalidData)?;
        let body = data.get(16..16 + len).ok_or(Error::InvalidData)?;
        Ok(AstcFile {
            block,
            width,
            height,
            depth,
            data: body,
        })
    }

    /// Block footprint (x, y, z).
    pub fn block(&self) -> (u32, u32, u32) {
        self.block
    }
    /// Width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Depth in texels (1 for 2D images).
    pub fn depth(&self) -> u32 {
        self.depth
    }
    /// The blocks (16 bytes each, x fastest, then y, then z).
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
    /// The imgio format for 2D blocks with decode profile `kind` (`None` for 3D blocks).
    pub fn format(&self, kind: AstcKind) -> Option<Format> {
        (self.block.2 == 1)
            .then(|| AstcBlock::from_dim(self.block.0, self.block.1))
            .flatten()
            .map(|b| Format::Astc(b, kind))
    }
}

/// Writes a `.astc` file for `block` (x, y, z) blocks covering `width` x `height` x `depth`
/// texels. `data` must hold exactly that many 16-byte blocks.
pub fn write(
    block: (u32, u32, u32),
    width: u32,
    height: u32,
    depth: u32,
    data: &[u8],
) -> Result<Buffer> {
    let max = 1 << 24;
    if block.0 == 0
        || block.1 == 0
        || block.2 == 0
        || block.0 > 255
        || block.1 > 255
        || block.2 > 255
    {
        return Err(Error::InvalidArgument);
    }
    if width == 0 || height == 0 || depth == 0 || width >= max || height >= max || depth >= max {
        return Err(Error::InvalidArgument);
    }
    let blocks = u64::from(width.div_ceil(block.0))
        * u64::from(height.div_ceil(block.1))
        * u64::from(depth.div_ceil(block.2));
    if u64::try_from(data.len()).ok() != Some(blocks * 16) {
        return Err(Error::InvalidArgument);
    }
    let mut out = crate::buffer::zeroed(16 + data.len())?;
    out[..4].copy_from_slice(&MAGIC.to_le_bytes());
    out[4..7].copy_from_slice(&[block.0 as u8, block.1 as u8, block.2 as u8]);
    for (i, v) in [width, height, depth].iter().enumerate() {
        out[7 + i * 3..10 + i * 3].copy_from_slice(&v.to_le_bytes()[..3]);
    }
    out[16..].copy_from_slice(data);
    Ok(out)
}
