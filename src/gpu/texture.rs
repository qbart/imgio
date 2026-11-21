//! Texture sources for GPU upload: [`TextureSource`] over the imgio containers and
//! [`TextureData`] for images held by the caller (encoded or transcoded data).

use crate::dds::DdsFile;
use crate::ktx2::Ktx2File;
use crate::{Buffer, Error, Format, Result};
use vas::Vector;
use vas::storage::Dynamic;

/// A texture whose images can be uploaded to a GPU.
///
/// Images are addressed by mip `level`, array `layer` and cube `face`. For a 3D texture
/// (`depth() > 1`) image `(level, 0, 0)` holds every depth slice of the level, slice after slice.
/// Rows (block rows for compressed formats) are padded to [`TextureSource::row_alignment`]
/// bytes; images and slices follow each other without padding.
pub trait TextureSource {
    /// Pixel / block format of every image (`Unsupported` when there is none, e.g. a KTX2
    /// file with a Basis Universal payload, which must be transcoded first).
    fn format(&self) -> Result<Format>;
    /// Width of level 0 in pixels.
    fn width(&self) -> u32;
    /// Height of level 0 in pixels.
    fn height(&self) -> u32;
    /// Depth of level 0 in pixels: 1, or more for a 3D texture.
    fn depth(&self) -> u32 {
        1
    }
    /// Mip levels (at least 1).
    fn level_count(&self) -> u32;
    /// Array layers (at least 1).
    fn layer_count(&self) -> u32;
    /// 1, or 6 for a cubemap (faces +X, -X, +Y, -Y, +Z, -Z).
    fn face_count(&self) -> u32;
    /// True for an array texture (default: more than one layer).
    fn is_array(&self) -> bool {
        self.layer_count() > 1
    }
    /// Alignment in bytes of each row: 1 (tightly packed, the default), 2, 4 or 8.
    fn row_alignment(&self) -> u32 {
        1
    }
    /// The bytes of one image.
    fn image_data(&self, level: u32, layer: u32, face: u32) -> Result<&[u8]>;
}

impl TextureSource for Ktx2File<'_> {
    fn format(&self) -> Result<Format> {
        Ktx2File::format(self).ok_or(Error::Unsupported)
    }
    fn width(&self) -> u32 {
        Ktx2File::width(self)
    }
    fn height(&self) -> u32 {
        Ktx2File::height(self)
    }
    fn depth(&self) -> u32 {
        Ktx2File::depth(self)
    }
    fn level_count(&self) -> u32 {
        Ktx2File::level_count(self)
    }
    fn layer_count(&self) -> u32 {
        Ktx2File::layer_count(self)
    }
    fn face_count(&self) -> u32 {
        Ktx2File::face_count(self)
    }
    fn image_data(&self, level: u32, layer: u32, face: u32) -> Result<&[u8]> {
        Ktx2File::image_data(self, level, layer, face)
    }
}

impl TextureSource for DdsFile<'_> {
    fn format(&self) -> Result<Format> {
        Ok(DdsFile::format(self))
    }
    fn width(&self) -> u32 {
        DdsFile::width(self)
    }
    fn height(&self) -> u32 {
        DdsFile::height(self)
    }
    fn level_count(&self) -> u32 {
        DdsFile::level_count(self)
    }
    fn layer_count(&self) -> u32 {
        DdsFile::layer_count(self)
    }
    fn face_count(&self) -> u32 {
        DdsFile::face_count(self)
    }
    fn image_data(&self, level: u32, layer: u32, face: u32) -> Result<&[u8]> {
        DdsFile::image_data(self, level, layer, face)
    }
    fn depth(&self) -> u32 {
        DdsFile::depth(self)
    }
}

/// The bytes of one level of a [`TextureData`]: borrowed or owned.
#[derive(Debug)]
pub enum LevelData<'a> {
    Borrowed(&'a [u8]),
    Owned(Buffer),
}

impl LevelData<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            LevelData::Borrowed(b) => b,
            LevelData::Owned(b) => b,
        }
    }
}

/// A texture described and filled by the caller, e.g. with [`crate::encode`] or
/// [`crate::ktx2::Ktx2File::transcode`] output.
///
/// Set the shape first, then push the levels from level 0: each level holds every image of
/// the level, layer after layer and face after face within a layer (the KTX2 level layout).
#[derive(Debug)]
pub struct TextureData<'a> {
    format: Format,
    width: u32,
    height: u32,
    depth: u32,
    layers: u32,
    faces: u32,
    array: bool,
    row_alignment: u32,
    levels: Vector<LevelData<'a>, Dynamic>,
}

impl<'a> TextureData<'a> {
    /// A `width` x `height` 2D texture of `format` without levels.
    pub fn new(format: Format, width: u32, height: u32) -> Self {
        TextureData {
            format,
            width,
            height,
            depth: 1,
            layers: 1,
            faces: 1,
            array: false,
            row_alignment: 1,
            levels: Vector::default(),
        }
    }
    /// Makes this an array texture of `layers` layers.
    pub fn with_layers(mut self, layers: u32) -> Self {
        self.layers = layers;
        self.array = true;
        self
    }
    /// Makes this a cubemap (6 faces).
    pub fn with_cube(mut self) -> Self {
        self.faces = 6;
        self
    }
    /// Makes this a 3D texture of `depth` slices.
    pub fn with_depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }
    /// Rows of the pushed data are padded to `alignment` bytes (1, 2, 4 or 8).
    pub fn with_row_alignment(mut self, alignment: u32) -> Self {
        self.row_alignment = alignment;
        self
    }

    /// Appends the next level, borrowed.
    ///
    /// Errors: `InvalidArgument` (bad shape, too many levels, or `data` is not exactly the
    /// level's size), `OutOfMemory`.
    pub fn push_level(&mut self, data: &'a [u8]) -> Result<()> {
        self.push(LevelData::Borrowed(data))
    }

    /// Appends the next level, owned. Errors as [`TextureData::push_level`].
    pub fn push_owned_level(&mut self, data: Buffer) -> Result<()> {
        self.push(LevelData::Owned(data))
    }

    fn push(&mut self, data: LevelData<'a>) -> Result<()> {
        let level = self.levels.len() as u32;
        let shape = Shape::of_dims(self, level + 1)?;
        let images = (self.layers * self.faces) as usize;
        let size = shape
            .image_bytes(level)?
            .checked_mul(images)
            .ok_or(Error::InvalidArgument)?;
        if data.bytes().len() != size {
            return Err(Error::InvalidArgument);
        }
        self.levels.push(data).map_err(|_| Error::OutOfMemory)
    }
}

impl TextureSource for TextureData<'_> {
    fn format(&self) -> Result<Format> {
        Ok(self.format)
    }
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn depth(&self) -> u32 {
        self.depth
    }
    fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }
    fn layer_count(&self) -> u32 {
        self.layers
    }
    fn face_count(&self) -> u32 {
        self.faces
    }
    fn is_array(&self) -> bool {
        self.array
    }
    fn row_alignment(&self) -> u32 {
        self.row_alignment
    }
    fn image_data(&self, level: u32, layer: u32, face: u32) -> Result<&[u8]> {
        if layer >= self.layers || face >= self.faces {
            return Err(Error::InvalidArgument);
        }
        let data = self
            .levels
            .get(level as usize)
            .ok_or(Error::InvalidArgument)?
            .bytes();
        let size = Shape::of_dims(self, self.level_count())?.image_bytes(level)?;
        let at = (layer * self.faces + face) as usize * size;
        data.get(at..at + size).ok_or(Error::InvalidData)
    }
}

/// The validated shape of a [`TextureSource`] and its image sizes.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(any(feature = "vulkan", feature = "opengl")), allow(dead_code))]
pub(crate) struct Shape {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub levels: u32,
    pub layers: u32,
    pub faces: u32,
    pub array: bool,
    pub row_alignment: u32,
}

#[cfg_attr(not(any(feature = "vulkan", feature = "opengl")), allow(dead_code))]
impl Shape {
    /// Validates `src`: non-zero size, 1 or 6 faces (square, 2D), several layers only for
    /// arrays, 3D textures without layers or faces, levels within the full mip chain, row
    /// alignment 1, 2, 4 or 8.
    pub fn of(src: &(impl TextureSource + ?Sized)) -> Result<Shape> {
        let format = src.format()?;
        let s = Shape {
            format,
            width: src.width(),
            height: src.height(),
            depth: src.depth(),
            levels: src.level_count(),
            layers: src.layer_count(),
            faces: src.face_count(),
            array: src.is_array(),
            row_alignment: src.row_alignment(),
        };
        s.check()
    }

    fn of_dims(t: &TextureData<'_>, levels: u32) -> Result<Shape> {
        Shape {
            format: t.format,
            width: t.width,
            height: t.height,
            depth: t.depth,
            levels,
            layers: t.layers,
            faces: t.faces,
            array: t.array,
            row_alignment: t.row_alignment,
        }
        .check()
    }

    fn check(self) -> Result<Shape> {
        let bad = self.width == 0
            || self.height == 0
            || self.depth == 0
            || self.levels == 0
            || self.layers == 0
            || !(self.faces == 1 || self.faces == 6)
            || (self.faces == 6 && (self.width != self.height || self.depth != 1))
            || (self.layers > 1 && !self.array)
            || (self.depth > 1 && (self.layers > 1 || self.faces > 1 || self.array))
            || !matches!(self.row_alignment, 1 | 2 | 4 | 8)
            || self.levels > self.max_levels();
        if bad {
            return Err(Error::InvalidArgument);
        }
        Ok(self)
    }

    /// Levels of the full mip chain down to 1x1x1.
    pub fn max_levels(&self) -> u32 {
        32 - self
            .width
            .max(self.height)
            .max(self.depth)
            .max(1)
            .leading_zeros()
    }

    /// Size of `level` in pixels (width, height, depth).
    pub fn extent(&self, level: u32) -> (u32, u32, u32) {
        let shr = |v: u32| v.checked_shr(level).unwrap_or(0).max(1);
        (shr(self.width), shr(self.height), shr(self.depth))
    }

    /// Bytes of one tightly packed row of blocks of `level`.
    pub fn row_bytes(&self, level: u32) -> Result<usize> {
        self.format.row_bytes(self.extent(level).0)
    }

    /// Bytes of one row of blocks of `level` in the source (padded to the row alignment).
    pub fn src_row_pitch(&self, level: u32) -> Result<usize> {
        let a = self.row_alignment as usize;
        Ok(self.row_bytes(level)?.div_ceil(a) * a)
    }

    /// Rows of blocks of one slice of `level`.
    pub fn block_rows(&self, level: u32) -> u32 {
        let (w, h, _) = self.extent(level);
        self.format.block_count(w, h).1
    }

    /// Bytes of one source image of `level` (all depth slices, padded rows).
    pub fn image_bytes(&self, level: u32) -> Result<usize> {
        let (_, _, d) = self.extent(level);
        self.src_row_pitch(level)?
            .checked_mul(self.block_rows(level) as usize)
            .and_then(|n| n.checked_mul(d as usize))
            .ok_or(Error::InvalidArgument)
    }

    /// Bytes of one tightly packed image of `level` (all depth slices).
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub fn tight_image_bytes(&self, level: u32) -> Result<usize> {
        let (_, _, d) = self.extent(level);
        self.row_bytes(level)?
            .checked_mul(self.block_rows(level) as usize)
            .and_then(|n| n.checked_mul(d as usize))
            .ok_or(Error::InvalidArgument)
    }

    /// Array layers times faces: the image count of every level.
    pub fn images(&self) -> u32 {
        self.layers * self.faces
    }
}

/// Copies one source image to `dst` with its row padding removed; returns the bytes written.
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub(crate) fn copy_tight(shape: &Shape, level: u32, src: &[u8], dst: &mut [u8]) -> Result<usize> {
    let row = shape.row_bytes(level)?;
    let pitch = shape.src_row_pitch(level)?;
    let rows = shape.block_rows(level) as usize * shape.extent(level).2 as usize;
    let total = shape.tight_image_bytes(level)?;
    let got = dst.len();
    let dst = dst
        .get_mut(..total)
        .ok_or(Error::BufferTooSmall { needed: total, got })?;
    if row == pitch {
        dst.copy_from_slice(src.get(..total).ok_or(Error::InvalidData)?);
        return Ok(total);
    }
    for (r, out) in dst.chunks_exact_mut(row.max(1)).take(rows).enumerate() {
        let at = r * pitch;
        out.copy_from_slice(src.get(at..at + row).ok_or(Error::InvalidData)?);
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_data_layout() -> Result<()> {
        let short = [0u8; 5 * 3 * 2];
        let level0 = [7u8; 8 * 3 * 2];
        let level1 = [1u8; 2 * 4];
        let mut t = TextureData::new(Format::R8Unorm, 5, 3)
            .with_layers(2)
            .with_row_alignment(4);
        // 5 bytes padded to 8: 8 * 3 rows * 2 layers.
        assert_eq!(t.push_level(&short), Err(Error::InvalidArgument));
        t.push_level(&level0)?;
        // 2x1 then 1x1, rows padded to 4; 5x3 has 3 levels.
        t.push_level(&level1)?;
        t.push_level(&level1)?;
        assert_eq!(t.push_level(&level1), Err(Error::InvalidArgument));
        assert_eq!(t.image_data(0, 1, 0)?.len(), 24);
        assert_eq!(t.image_data(1, 1, 0)?, &[1, 1, 1, 1]);
        assert!(t.image_data(3, 0, 0).is_err());
        let shape = Shape::of(&t)?;
        let mut out = [0u8; 15];
        assert_eq!(copy_tight(&shape, 0, t.image_data(0, 0, 0)?, &mut out)?, 15);
        assert_eq!(out, [7u8; 15]);
        Ok(())
    }

    #[test]
    fn shape_rules() {
        let cube = TextureData::new(Format::Rgba8Unorm, 4, 2).with_cube();
        assert!(Shape::of_dims(&cube, 1).is_err());
        let vol = TextureData::new(Format::Rgba8Unorm, 4, 4)
            .with_depth(4)
            .with_layers(2);
        assert!(Shape::of_dims(&vol, 1).is_err());
        let ok = TextureData::new(Format::Bc7Unorm, 8, 8).with_depth(2);
        assert_eq!(Shape::of_dims(&ok, 4).map(|s| s.max_levels()), Ok(4));
        assert!(Shape::of_dims(&ok, 5).is_err());
    }
}
