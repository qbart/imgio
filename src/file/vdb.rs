//! OpenVDB volumes (decode only, partial support).
//!
//! Needs the `vdb` cargo feature; without it every function returns `Unsupported`.
//!
//! imgio's own reader of the OpenVDB file format (file versions 222+, i.e. OpenVDB 3.0 and
//! later) for the standard `5_4_3` trees. The `vdb-rs` crate links the C Blosc library, so it
//! is not used; zlib streams are inflated with `miniz_oxide`.
//!
//! - [`grids`] lists every grid: name, value type, class, compression, index-to-world
//!   transform and active voxel bounding box (from the tree topology; for grids that cannot
//!   be parsed, from the `file_bbox_min` / `file_bbox_max` metadata OpenVDB writes).
//! - [`densify`] / [`densify_region`] turn a scalar float grid (`float`, `double`, or either
//!   saved as half) into a dense [`VdbVolume`] of `R32Float` voxels: active and inactive
//!   values, tiles and the background, exactly as OpenVDB's `getValue`.
//! - [`decode`] returns one 2D `R32Float` slice: the middle z slice
//!   (`min.z + (max.z - min.z) / 2`) of the active bounding box of the first scalar float
//!   grid; `Unsupported` if the file has none (or it has no active voxels).
//!
//! Supported: uncompressed, zip and active-mask compression (all inactive-value modes), half
//! float storage, root / internal tiles, instanced grids, scale / translate / affine /
//! unitary maps (frustum maps are listed without a matrix). Not supported (`Unsupported`):
//! Blosc-compressed data (the OpenVDB default when built with Blosc; such grids are still
//! listed), point data / mask / string grids, other tree configurations, file versions before
//! 222 and files without grid offsets.

use super::Image;
use crate::{Buffer, Error, Result, Surface};
use vas::Vector;
use vas::storage::Dynamic;

/// Grid compression flag: zlib.
pub const COMPRESS_ZIP: u32 = 0x1;
/// Grid compression flag: only active values are stored (inactive ones are reconstructed).
pub const COMPRESS_ACTIVE_MASK: u32 = 0x2;
/// Grid compression flag: Blosc (data not decodable by imgio).
pub const COMPRESS_BLOSC: u32 = 0x4;

/// Encoder options (OpenVDB files cannot be written).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EncodeOptions {}

/// Value type of a grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Float,
    Double,
    Int32,
    Int64,
    Bool,
    Vec3Float,
    Vec3Double,
    Vec3Int,
    /// Any other type (points, masks, strings, ...): listed, never densified.
    Other,
}

/// Index-to-world transform of a grid.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    /// OpenVDB map type (`UniformScaleMap`, `ScaleTranslateMap`, `AffineMap`, ...).
    pub map: String,
    /// Affine matrix in OpenVDB's row-vector convention: `world = [i j k 1] * matrix`
    /// (translation in the last row). `None` for non-linear (frustum) maps.
    pub matrix: Option<[[f64; 4]; 4]>,
}

/// One grid of a file.
#[derive(Debug, Clone, PartialEq)]
pub struct GridInfo {
    /// Grid name (OpenVDB's unique-name suffix for duplicates removed).
    pub name: String,
    /// OpenVDB tree type, e.g. `Tree_float_5_4_3` (without the `_HalfFloat` suffix).
    pub type_name: String,
    pub value_type: ValueType,
    /// Float values are stored as 16-bit halves.
    pub half_float: bool,
    /// Grid class metadata (`level set`, `fog volume`, `staggered`, `unknown`).
    pub class: String,
    /// `COMPRESS_*` flags of the grid.
    pub compression: u32,
    /// Name of the grid whose tree this grid shares (OpenVDB instancing).
    pub instance_of: Option<String>,
    /// `None` if the map type is unknown.
    pub transform: Option<Transform>,
    /// Inclusive `(min, max)` index-space bounding box of the active voxels and tiles;
    /// `None` for an empty grid or when it cannot be determined.
    pub active_bbox: Option<([i32; 3], [i32; 3])>,
}

/// Options of [`densify`] / [`decode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Largest number of voxels a dense volume may have (`InvalidArgument` above).
    pub max_voxels: u64,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            max_voxels: 1 << 26,
        }
    }
}

/// A dense `R32Float` volume: voxel `(x, y, z)` (index space) is the little-endian `f32` at
/// `((z - origin[2]) * height + (y - origin[1])) * width + (x - origin[0])`.
#[derive(Debug)]
pub struct VdbVolume {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Index-space coordinate of the first voxel.
    pub origin: [i32; 3],
    pub data: Buffer,
}

impl VdbVolume {
    /// The value at index-space `(x, y, z)`, `None` outside the volume.
    pub fn get(&self, x: i32, y: i32, z: i32) -> Option<f32> {
        let rel = |v: i32, o: i32, n: u32| {
            let d = i64::from(v) - i64::from(o);
            (d >= 0 && d < i64::from(n)).then_some(d as usize)
        };
        let (x, y, z) = (
            rel(x, self.origin[0], self.width)?,
            rel(y, self.origin[1], self.height)?,
            rel(z, self.origin[2], self.depth)?,
        );
        let i = ((z * self.height as usize + y) * self.width as usize + x) * 4;
        let b = self.data.get(i..i + 4)?;
        Some(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Lists the grids of a file.
pub fn grids(data: &[u8]) -> Result<Vector<GridInfo, Dynamic>> {
    #[cfg(feature = "vdb")]
    {
        imp::grids(data)
    }
    #[cfg(not(feature = "vdb"))]
    {
        let _ = data;
        Err(Error::Unsupported)
    }
}

/// Densifies the active bounding box of grid `grid` (index into [`grids`]); an empty grid
/// gives a 0x0x0 volume.
pub fn densify(data: &[u8], grid: usize, opts: &DecodeOptions) -> Result<VdbVolume> {
    #[cfg(feature = "vdb")]
    {
        imp::densify(data, grid, None, opts)
    }
    #[cfg(not(feature = "vdb"))]
    {
        let _ = (data, grid, opts);
        Err(Error::Unsupported)
    }
}

/// Densifies the inclusive index-space box `min..=max` of grid `grid`.
pub fn densify_region(
    data: &[u8],
    grid: usize,
    min: [i32; 3],
    max: [i32; 3],
    opts: &DecodeOptions,
) -> Result<VdbVolume> {
    #[cfg(feature = "vdb")]
    {
        imp::densify(data, grid, Some((min, max)), opts)
    }
    #[cfg(not(feature = "vdb"))]
    {
        let _ = (data, grid, min, max, opts);
        Err(Error::Unsupported)
    }
}

/// Decodes the middle z slice of the first scalar float grid as an `R32Float` image.
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_with(data, &DecodeOptions::default())
}

/// [`decode`] with options.
pub fn decode_with(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    let list = grids(data)?;
    let (index, (min, max)) = list
        .iter()
        .enumerate()
        .find(|(_, g)| matches!(g.value_type, ValueType::Float | ValueType::Double))
        .and_then(|(i, g)| g.active_bbox.map(|b| (i, b)))
        .ok_or(Error::Unsupported)?;
    let z = min[2] + (max[2] - min[2]) / 2;
    let vol = densify_region(data, index, [min[0], min[1], z], [max[0], max[1], z], opts)?;
    let img = Image::from_surface(&Surface::new(
        crate::Format::R32Float,
        vol.width,
        vol.height,
        &vol.data,
    )?)?;
    Ok(img)
}

/// OpenVDB files cannot be written: always `Unsupported`.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    let _ = (src, opts);
    Err(Error::Unsupported)
}

#[cfg(feature = "vdb")]
mod imp {
    use super::*;

    const MAGIC: [u8; 8] = [0x20, 0x42, 0x44, 0x56, 0, 0, 0, 0];
    /// OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION: per-grid compression, node metadata bytes.
    const MIN_VERSION: u32 = 222;
    const LEAF_WORDS: usize = 8;
    /// Unique-name separator of duplicate grid names (`GridDescriptor::SEP`).
    const NAME_SEP: char = '\x1e';

    // Node metadata of `io::writeCompressedValues`.
    const NO_MASK_OR_INACTIVE_VALS: u8 = 0;
    const NO_MASK_AND_ONE_INACTIVE_VAL: u8 = 2;
    const MASK_AND_NO_INACTIVE_VALS: u8 = 3;
    const MASK_AND_ONE_INACTIVE_VAL: u8 = 4;
    const MASK_AND_TWO_INACTIVE_VALS: u8 = 5;
    const NO_MASK_AND_ALL_VALS: u8 = 6;

    fn push<T>(v: &mut Vector<T, Dynamic>, x: T) -> Result<()> {
        v.push(x).map_err(|_| Error::OutOfMemory)
    }

    struct Reader<'a> {
        data: &'a [u8],
        pos: usize,
    }

    impl<'a> Reader<'a> {
        fn at(data: &'a [u8], pos: usize) -> Result<Self> {
            if pos > data.len() {
                return Err(Error::InvalidData);
            }
            Ok(Self { data, pos })
        }
        fn remaining(&self) -> usize {
            self.data.len() - self.pos
        }
        fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
            let end = self.pos.checked_add(n).ok_or(Error::InvalidData)?;
            let b = self.data.get(self.pos..end).ok_or(Error::InvalidData)?;
            self.pos = end;
            Ok(b)
        }
        fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
            let mut a = [0u8; N];
            a.copy_from_slice(self.bytes(N)?);
            Ok(a)
        }
        fn u8(&mut self) -> Result<u8> {
            Ok(self.arr::<1>()?[0])
        }
        fn u32(&mut self) -> Result<u32> {
            Ok(u32::from_le_bytes(self.arr()?))
        }
        fn i32(&mut self) -> Result<i32> {
            Ok(i32::from_le_bytes(self.arr()?))
        }
        fn i64(&mut self) -> Result<i64> {
            Ok(i64::from_le_bytes(self.arr()?))
        }
        fn f64(&mut self) -> Result<f64> {
            Ok(f64::from_le_bytes(self.arr()?))
        }
        fn vec3d(&mut self) -> Result<[f64; 3]> {
            Ok([self.f64()?, self.f64()?, self.f64()?])
        }
        fn coord(&mut self) -> Result<[i64; 3]> {
            Ok([
                i64::from(self.i32()?),
                i64::from(self.i32()?),
                i64::from(self.i32()?),
            ])
        }
        fn string(&mut self) -> Result<String> {
            let n = self.u32()? as usize;
            Ok(String::from_utf8_lossy(self.bytes(n)?).into_owned())
        }
        /// A node mask of `words` 64-bit words.
        fn mask(&mut self, words: usize) -> Result<Vector<u64, Dynamic>> {
            let b = self.bytes(words.checked_mul(8).ok_or(Error::InvalidData)?)?;
            let mut m = Vector::<u64, Dynamic>::default();
            m.reserve(words).map_err(|_| Error::OutOfMemory)?;
            for w in b.as_chunks::<8>().0 {
                push(&mut m, u64::from_le_bytes(*w))?;
            }
            Ok(m)
        }
    }

    fn bit(mask: &[u64], i: usize) -> bool {
        mask.get(i / 64).is_some_and(|w| w >> (i % 64) & 1 != 0)
    }

    fn count_ones(mask: &[u64]) -> usize {
        mask.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Value type, full value size and stored size of a `Tree_<type>_5_4_3` grid.
    struct TypeInfo {
        value_type: ValueType,
        /// `sizeof(ValueType)`: background, tiles, inactive values.
        size: usize,
        /// Bytes per value in node buffers (halved for real types saved as half).
        stored: usize,
        /// Tree topology can be parsed.
        parsable: bool,
    }

    fn type_info(type_name: &str, half: bool) -> TypeInfo {
        let value = type_name
            .strip_prefix("Tree_")
            .and_then(|t| t.strip_suffix("_5_4_3"));
        let (value_type, size, components, real) = match value {
            Some("float") => (ValueType::Float, 4, 1, true),
            Some("double") => (ValueType::Double, 8, 1, true),
            Some("int32") => (ValueType::Int32, 4, 1, false),
            Some("int64") => (ValueType::Int64, 8, 1, false),
            Some("bool") => (ValueType::Bool, 1, 1, false),
            Some("vec3s") => (ValueType::Vec3Float, 12, 3, true),
            Some("vec3d") => (ValueType::Vec3Double, 24, 3, true),
            Some("vec3i") => (ValueType::Vec3Int, 12, 3, false),
            _ => (ValueType::Other, 0, 0, false),
        };
        TypeInfo {
            value_type,
            size,
            stored: if half && real { 2 * components } else { size },
            parsable: value_type != ValueType::Other,
        }
    }

    struct Descriptor {
        unique_name: String,
        type_name: String,
        half: bool,
        instance_parent: String,
        grid_pos: usize,
        block_pos: usize,
    }

    struct Archive<'a> {
        data: &'a [u8],
        descriptors: Vector<Descriptor, Dynamic>,
    }

    fn position(v: i64, len: usize) -> Result<usize> {
        usize::try_from(v)
            .ok()
            .filter(|&p| p <= len)
            .ok_or(Error::InvalidData)
    }

    fn skip_metadata(r: &mut Reader) -> Result<()> {
        let count = r.u32()?;
        for _ in 0..count {
            r.string()?;
            r.string()?;
            let n = r.u32()? as usize;
            r.bytes(n)?;
        }
        Ok(())
    }

    fn read_archive(data: &[u8]) -> Result<Archive<'_>> {
        let mut r = Reader::at(data, 0)?;
        if r.arr::<8>()? != MAGIC {
            return Err(Error::InvalidData);
        }
        let version = r.u32()?;
        if version < MIN_VERSION {
            return Err(Error::Unsupported);
        }
        r.u32()?; // library major
        r.u32()?; // library minor
        if r.u8()? == 0 {
            // Streams without grid offsets (written to a pipe) are not supported.
            return Err(Error::Unsupported);
        }
        r.bytes(36)?; // UUID
        skip_metadata(&mut r)?;
        let count = r.u32()?;
        let mut descriptors = Vector::<Descriptor, Dynamic>::default();
        for _ in 0..count {
            let unique_name = r.string()?;
            let mut type_name = r.string()?;
            let half = match type_name.strip_suffix("_HalfFloat") {
                Some(t) => {
                    type_name = t.to_owned();
                    true
                }
                None => false,
            };
            let instance_parent = r.string()?;
            let grid_pos = position(r.i64()?, data.len())?;
            let block_pos = position(r.i64()?, data.len())?;
            let end_pos = position(r.i64()?, data.len())?;
            // Instances have no buffers (block position 0).
            let block_pos = if instance_parent.is_empty() {
                block_pos
            } else {
                grid_pos
            };
            if grid_pos < r.pos || block_pos < grid_pos || end_pos < block_pos {
                return Err(Error::InvalidData);
            }
            push(
                &mut descriptors,
                Descriptor {
                    unique_name,
                    type_name,
                    half,
                    instance_parent,
                    grid_pos,
                    block_pos,
                },
            )?;
            r = Reader::at(data, end_pos)?;
        }
        Ok(Archive { data, descriptors })
    }

    fn display_name(unique: &str) -> String {
        unique.split(NAME_SEP).next().unwrap_or(unique).to_owned()
    }

    /// Grid metadata imgio uses.
    #[derive(Default)]
    struct GridMeta {
        class: Option<String>,
        bbox_min: Option<[i32; 3]>,
        bbox_max: Option<[i32; 3]>,
    }

    fn read_grid_meta(r: &mut Reader) -> Result<GridMeta> {
        let mut m = GridMeta::default();
        let count = r.u32()?;
        for _ in 0..count {
            let name = r.string()?;
            let ty = r.string()?;
            let n = r.u32()? as usize;
            let value = r.bytes(n)?;
            let vec3i = || -> Option<[i32; 3]> {
                let (c, _) = value.as_chunks::<4>();
                match c {
                    [x, y, z] => Some([
                        i32::from_le_bytes(*x),
                        i32::from_le_bytes(*y),
                        i32::from_le_bytes(*z),
                    ]),
                    _ => None,
                }
            };
            match (name.as_str(), ty.as_str()) {
                ("class", "string") => {
                    m.class = Some(String::from_utf8_lossy(value).into_owned());
                }
                ("file_bbox_min", "vec3i") => m.bbox_min = vec3i(),
                ("file_bbox_max", "vec3i") => m.bbox_max = vec3i(),
                _ => {}
            }
        }
        Ok(m)
    }

    fn scale_translate(scale: [f64; 3], t: [f64; 3]) -> [[f64; 4]; 4] {
        [
            [scale[0], 0.0, 0.0, 0.0],
            [0.0, scale[1], 0.0, 0.0],
            [0.0, 0.0, scale[2], 0.0],
            [t[0], t[1], t[2], 1.0],
        ]
    }

    fn read_mat4(r: &mut Reader) -> Result<[[f64; 4]; 4]> {
        let mut m = [[0.0; 4]; 4];
        for row in &mut m {
            for v in row.iter_mut() {
                *v = r.f64()?;
            }
        }
        Ok(m)
    }

    /// `math::Transform::read`: `Err(Unsupported)` for unknown maps (their size is unknown).
    fn read_transform(r: &mut Reader) -> Result<Transform> {
        let map = r.string()?;
        let matrix = match map.as_str() {
            "UniformScaleMap" | "ScaleMap" => {
                let scale = r.vec3d()?;
                for _ in 0..4 {
                    r.vec3d()?; // voxel size, inverse scale, inverse scale^2, 1 / (2 scale)
                }
                Some(scale_translate(scale, [0.0; 3]))
            }
            "UniformScaleTranslateMap" | "ScaleTranslateMap" => {
                let t = r.vec3d()?;
                let scale = r.vec3d()?;
                for _ in 0..4 {
                    r.vec3d()?;
                }
                Some(scale_translate(scale, t))
            }
            "TranslationMap" => Some(scale_translate([1.0; 3], r.vec3d()?)),
            "AffineMap" | "UnitaryMap" => Some(read_mat4(r)?),
            "NonlinearFrustumMap" => {
                r.vec3d()?; // bbox min
                r.vec3d()?; // bbox max
                r.f64()?; // taper
                r.f64()?; // depth
                if r.string()? != "AffineMap" {
                    return Err(Error::Unsupported);
                }
                read_mat4(r)?;
                None
            }
            _ => return Err(Error::Unsupported),
        };
        Ok(Transform { map, matrix })
    }

    /// Stream state of one grid's tree.
    struct Ctx {
        compression: u32,
        ty: TypeInfo,
        /// Scalar float grids: values are decoded to `f32`.
        scalar: bool,
        background: f32,
    }

    impl Ctx {
        /// A full-size value (background, tile, inactive value) as `f32` (scalar grids).
        fn value(&self, b: &[u8]) -> f32 {
            match *b {
                [a, b, c, d] if self.ty.value_type == ValueType::Float => {
                    f32::from_le_bytes([a, b, c, d])
                }
                [a, b, c, d, e, f, g, h] if self.ty.value_type == ValueType::Double => {
                    f64::from_le_bytes([a, b, c, d, e, f, g, h]) as f32
                }
                _ => 0.0,
            }
        }

        /// A stored buffer value as `f32` (scalar grids).
        fn stored(&self, b: &[u8]) -> f32 {
            match *b {
                [a, b] => crate::half::to_f32(u16::from_le_bytes([a, b])),
                _ => self.value(b),
            }
        }
    }

    /// `io::readData`: `bytes` bytes of (possibly zip / Blosc compressed) data into `out`.
    fn read_data(r: &mut Reader, ctx: &Ctx, out: &mut [u8]) -> Result<()> {
        if ctx.compression & (COMPRESS_ZIP | COMPRESS_BLOSC) == 0 {
            out.copy_from_slice(r.bytes(out.len())?);
            return Ok(());
        }
        let n = r.i64()?;
        if n <= 0 {
            // Stored uncompressed (compression did not pay off).
            if n.unsigned_abs() != out.len() as u64 {
                return Err(Error::InvalidData);
            }
            out.copy_from_slice(r.bytes(out.len())?);
            return Ok(());
        }
        if ctx.compression & COMPRESS_BLOSC != 0 {
            return Err(Error::Unsupported);
        }
        let src = r.bytes(usize::try_from(n).map_err(|_| Error::InvalidData)?)?;
        let got = miniz_oxide::inflate::decompress_slice_iter_to_slice(
            out,
            core::iter::once(src),
            true,
            false,
        )
        .map_err(|_| Error::InvalidData)?;
        if got != out.len() {
            return Err(Error::InvalidData);
        }
        Ok(())
    }

    /// `io::readCompressedValues` for a node of `count` values. With `out`, the values are
    /// decoded (scalar grids) into it; otherwise they are only skipped.
    fn read_values(
        r: &mut Reader,
        ctx: &Ctx,
        value_mask: &[u64],
        count: usize,
        scratch: &mut Buffer,
        out: Option<&mut [f32]>,
    ) -> Result<()> {
        let meta = r.u8()?;
        if meta > NO_MASK_AND_ALL_VALS {
            return Err(Error::InvalidData);
        }
        let size = ctx.ty.size;
        let mut inactive0 = if meta == NO_MASK_OR_INACTIVE_VALS {
            ctx.background
        } else {
            -ctx.background
        };
        let mut inactive1 = ctx.background;
        if matches!(
            meta,
            NO_MASK_AND_ONE_INACTIVE_VAL | MASK_AND_ONE_INACTIVE_VAL | MASK_AND_TWO_INACTIVE_VALS
        ) {
            inactive0 = ctx.value(r.bytes(size)?);
            if meta == MASK_AND_TWO_INACTIVE_VALS {
                inactive1 = ctx.value(r.bytes(size)?);
            }
        }
        let mut selection = None;
        if matches!(
            meta,
            MASK_AND_NO_INACTIVE_VALS | MASK_AND_ONE_INACTIVE_VAL | MASK_AND_TWO_INACTIVE_VALS
        ) {
            selection = Some(r.mask(count / 64)?);
        }
        let masked = ctx.compression & COMPRESS_ACTIVE_MASK != 0 && meta != NO_MASK_AND_ALL_VALS;
        let stored_count = if masked {
            count_ones(value_mask)
        } else {
            count
        };
        let half = ctx.ty.stored != ctx.ty.size;
        // Half data of no values is not written at all (`HalfReader`).
        if half && stored_count == 0 {
            if let Some(out) = out {
                out.fill(inactive0);
                for (i, v) in out.iter_mut().enumerate() {
                    if selection.as_deref().is_some_and(|s| bit(s, i)) {
                        *v = inactive1;
                    }
                }
            }
            return Ok(());
        }
        let bytes = stored_count
            .checked_mul(ctx.ty.stored)
            .ok_or(Error::InvalidData)?;
        if bytes > r.remaining().saturating_mul(1100).saturating_add(4096) {
            // More than zlib's maximum ratio of the remaining input.
            return Err(Error::InvalidData);
        }
        scratch.clear();
        scratch.resize(bytes, 0).map_err(|_| Error::OutOfMemory)?;
        read_data(r, ctx, scratch)?;
        let Some(out) = out else { return Ok(()) };
        let mut src = scratch.chunks_exact(ctx.ty.stored);
        for (i, v) in out.iter_mut().enumerate().take(count) {
            *v = if !masked || bit(value_mask, i) {
                src.next().map_or(0.0, |b| ctx.stored(b))
            } else if selection.as_deref().is_some_and(|s| bit(s, i)) {
                inactive1
            } else {
                inactive0
            };
        }
        Ok(())
    }

    /// What a topology walk reports.
    trait Visit {
        /// Whether tile values are needed (decoded for scalar grids).
        fn values(&self) -> bool;
        /// A tile (or the whole of an internal node entry) `min..min + size` of `value`.
        fn tile(&mut self, min: [i64; 3], size: i64, value: f32, active: bool);
        /// A leaf with its value mask.
        fn leaf(&mut self, origin: [i64; 3], mask: &[u64]) -> Result<()>;
    }

    /// Leaves of each root child, so their buffers can be read in OpenVDB's order.
    struct RootChild {
        origin: [i64; 3],
        leaves: core::ops::Range<usize>,
    }

    struct Tree {
        children: Vector<RootChild, Dynamic>,
        /// Origin and value mask of every leaf, in topology order.
        leaves: Vector<([i64; 3], [u64; LEAF_WORDS]), Dynamic>,
    }

    /// Local offset `i` of a node with `log2dim` to its child origin (`child_log2` total).
    fn child_origin(origin: [i64; 3], i: usize, log2dim: u32, child_log2: u32) -> [i64; 3] {
        let i = i as i64;
        let dim = 1i64 << log2dim;
        let x = i >> (2 * log2dim);
        let y = (i >> log2dim) & (dim - 1);
        let z = i & (dim - 1);
        [
            origin[0] + (x << child_log2),
            origin[1] + (y << child_log2),
            origin[2] + (z << child_log2),
        ]
    }

    /// `InternalNode::readTopology` for log2dim 5 (children: log2dim 4 nodes) or 4 (leaves).
    #[allow(clippy::too_many_arguments)]
    fn read_internal(
        r: &mut Reader,
        ctx: &Ctx,
        origin: [i64; 3],
        log2dim: u32,
        tree: &mut Tree,
        visit: &mut dyn Visit,
        scratch: &mut Buffer,
        values: &mut Vector<f32, Dynamic>,
    ) -> Result<()> {
        let count = 1usize << (3 * log2dim);
        let child_log2 = if log2dim == 5 { 7 } else { 3 };
        let child_mask = r.mask(count / 64)?;
        let value_mask = r.mask(count / 64)?;
        let want = visit.values() && ctx.scalar;
        if want {
            values.clear();
            values.resize(count, 0.0).map_err(|_| Error::OutOfMemory)?;
        }
        read_values(
            r,
            ctx,
            &value_mask,
            count,
            scratch,
            want.then_some(&mut values[..]),
        )?;
        for i in 0..count {
            if !bit(&child_mask, i) {
                let v = if want {
                    values.get(i).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                let o = child_origin(origin, i, log2dim, child_log2);
                visit.tile(o, 1 << child_log2, v, bit(&value_mask, i));
            }
        }
        for i in 0..count {
            if !bit(&child_mask, i) {
                continue;
            }
            let o = child_origin(origin, i, log2dim, child_log2);
            if log2dim == 5 {
                read_internal(r, ctx, o, 4, tree, visit, scratch, values)?;
            } else {
                let mut mask = [0u64; LEAF_WORDS];
                for (w, b) in mask.iter_mut().zip(r.bytes(64)?.as_chunks::<8>().0) {
                    *w = u64::from_le_bytes(*b);
                }
                visit.leaf(o, &mask)?;
                push(&mut tree.leaves, (o, mask))?;
            }
        }
        Ok(())
    }

    /// `RootNode::readTopology`.
    fn read_topology(r: &mut Reader, ctx: &mut Ctx, visit: &mut dyn Visit) -> Result<Tree> {
        r.i32()?; // buffer count (always 1)
        let size = ctx.ty.size;
        ctx.background = ctx.value(r.bytes(size)?);
        let tiles = r.u32()? as usize;
        let children = r.u32()? as usize;
        if tiles.saturating_mul(13 + size) > r.remaining() {
            return Err(Error::InvalidData);
        }
        for _ in 0..tiles {
            let origin = r.coord()?;
            let value = ctx.value(r.bytes(size)?);
            let active = r.u8()? != 0;
            visit.tile(origin, 1 << 12, value, active);
        }
        let mut tree = Tree {
            children: Vector::default(),
            leaves: Vector::default(),
        };
        let mut scratch = Buffer::default();
        let mut values = Vector::<f32, Dynamic>::default();
        for _ in 0..children {
            let origin = r.coord()?;
            let start = tree.leaves.len();
            read_internal(
                r,
                ctx,
                origin,
                5,
                &mut tree,
                visit,
                &mut scratch,
                &mut values,
            )?;
            let end = tree.leaves.len();
            push(
                &mut tree.children,
                RootChild {
                    origin,
                    leaves: start..end,
                },
            )?;
        }
        // Buffers follow the root's map order (sorted origins).
        tree.children.sort_stable_by_key(|c| c.origin);
        if tree.children.windows(2).any(|w| w[0].origin == w[1].origin) {
            return Err(Error::InvalidData);
        }
        Ok(tree)
    }

    /// Accumulates the active bounding box.
    #[derive(Default)]
    struct Bounds {
        min: Option<[i64; 3]>,
        max: [i64; 3],
    }

    impl Bounds {
        fn add(&mut self, lo: [i64; 3], hi: [i64; 3]) {
            match &mut self.min {
                None => {
                    self.min = Some(lo);
                    self.max = hi;
                }
                Some(min) => {
                    for a in 0..3 {
                        min[a] = min[a].min(lo[a]);
                        self.max[a] = self.max[a].max(hi[a]);
                    }
                }
            }
        }
        fn get(&self) -> Result<Option<([i32; 3], [i32; 3])>> {
            let Some(min) = self.min else { return Ok(None) };
            let c = |v: [i64; 3]| -> Result<[i32; 3]> {
                let mut o = [0i32; 3];
                for (o, v) in o.iter_mut().zip(v) {
                    *o = i32::try_from(v).map_err(|_| Error::InvalidData)?;
                }
                Ok(o)
            };
            Ok(Some((c(min)?, c(self.max)?)))
        }
    }

    impl Visit for Bounds {
        fn values(&self) -> bool {
            false
        }
        fn tile(&mut self, min: [i64; 3], size: i64, _: f32, active: bool) {
            if active {
                self.add(min, min.map(|v| v + size - 1));
            }
        }
        fn leaf(&mut self, origin: [i64; 3], mask: &[u64]) -> Result<()> {
            if mask.iter().all(|&w| w == 0) {
                return Ok(());
            }
            let (mut lo, mut hi) = ([7i64; 3], [0i64; 3]);
            for i in 0..512 {
                if bit(mask, i) {
                    let p = [(i >> 6) as i64, ((i >> 3) & 7) as i64, (i & 7) as i64];
                    for a in 0..3 {
                        lo[a] = lo[a].min(p[a]);
                        hi[a] = hi[a].max(p[a]);
                    }
                }
            }
            self.add(
                [origin[0] + lo[0], origin[1] + lo[1], origin[2] + lo[2]],
                [origin[0] + hi[0], origin[1] + hi[1], origin[2] + hi[2]],
            );
            Ok(())
        }
    }

    /// A grid ready for topology parsing.
    struct Grid {
        info: GridInfo,
        /// Topology position, compression and type of the tree (the parent's for instances);
        /// `Err` if the tree cannot be parsed.
        tree: Result<(usize, usize, u32, TypeInfoKey)>,
        meta_bbox: Option<([i32; 3], [i32; 3])>,
    }

    /// Cheap copy of what `type_info` needs.
    #[derive(Clone)]
    struct TypeInfoKey {
        type_name: String,
        half: bool,
    }

    /// Reads compression, metadata and transform of descriptor `d`; returns the topology
    /// position.
    fn read_grid_header(
        data: &[u8],
        d: &Descriptor,
    ) -> Result<(u32, GridMeta, Result<Transform>, Option<usize>)> {
        let mut r = Reader::at(data, d.grid_pos)?;
        let compression = r.u32()?;
        let meta = read_grid_meta(&mut r)?;
        let transform = read_transform(&mut r);
        let topo = transform.is_ok().then_some(r.pos);
        Ok((compression, meta, transform, topo))
    }

    fn open_grid(archive: &Archive, index: usize) -> Result<Grid> {
        let d = archive
            .descriptors
            .get(index)
            .ok_or(Error::InvalidArgument)?;
        let (compression, meta, transform, topo) = read_grid_header(archive.data, d)?;
        let ty = type_info(&d.type_name, d.half);
        let instance_of = (!d.instance_parent.is_empty()).then(|| display_name(&d.instance_parent));
        let tree = if d.instance_parent.is_empty() {
            topo.map(|p| {
                (
                    p,
                    d.block_pos,
                    compression,
                    TypeInfoKey {
                        type_name: d.type_name.clone(),
                        half: d.half,
                    },
                )
            })
            .ok_or(Error::Unsupported)
        } else {
            archive
                .descriptors
                .iter()
                .find(|p| p.unique_name == d.instance_parent && p.instance_parent.is_empty())
                .ok_or(Error::InvalidData)
                .and_then(|p| {
                    let (c, _, _, topo) = read_grid_header(archive.data, p)?;
                    let topo = topo.ok_or(Error::Unsupported)?;
                    Ok((
                        topo,
                        p.block_pos,
                        c,
                        TypeInfoKey {
                            type_name: p.type_name.clone(),
                            half: p.half,
                        },
                    ))
                })
        };
        let meta_bbox = meta.bbox_min.zip(meta.bbox_max);
        Ok(Grid {
            info: GridInfo {
                name: display_name(&d.unique_name),
                type_name: d.type_name.clone(),
                value_type: ty.value_type,
                half_float: d.half,
                class: meta.class.unwrap_or_else(|| "unknown".to_owned()),
                compression,
                instance_of,
                transform: transform.ok(),
                active_bbox: None,
            },
            tree,
            meta_bbox,
        })
    }

    fn new_ctx(key: &TypeInfoKey, compression: u32) -> Result<Ctx> {
        let ty = type_info(&key.type_name, key.half);
        if !ty.parsable {
            return Err(Error::Unsupported);
        }
        let scalar = matches!(ty.value_type, ValueType::Float | ValueType::Double);
        Ok(Ctx {
            compression,
            ty,
            scalar,
            background: 0.0,
        })
    }

    fn bounds(data: &[u8], grid: &Grid) -> Result<Option<([i32; 3], [i32; 3])>> {
        let (topo, _, compression, key) = grid.tree.as_ref().map_err(Clone::clone)?;
        let mut ctx = new_ctx(key, *compression)?;
        let mut r = Reader::at(data, *topo)?;
        let mut b = Bounds::default();
        read_topology(&mut r, &mut ctx, &mut b)?;
        b.get()
    }

    pub(super) fn grids(data: &[u8]) -> Result<Vector<GridInfo, Dynamic>> {
        let archive = read_archive(data)?;
        let mut out = Vector::<GridInfo, Dynamic>::default();
        for i in 0..archive.descriptors.len() {
            let grid = open_grid(&archive, i)?;
            let bbox = match bounds(data, &grid) {
                Ok(b) => b,
                Err(Error::Unsupported) => grid
                    .meta_bbox
                    .filter(|(lo, hi)| lo.iter().zip(hi).all(|(l, h)| l <= h)),
                Err(e) => return Err(e),
            };
            let mut info = grid.info;
            info.active_bbox = bbox;
            push(&mut out, info)?;
        }
        Ok(out)
    }

    /// Fills a dense region from tiles and leaves.
    struct Dense {
        min: [i64; 3],
        dims: [i64; 3],
        data: Vector<f32, Dynamic>,
    }

    impl Dense {
        fn fill(&mut self, lo: [i64; 3], hi_excl: [i64; 3], value: f32) {
            let mut a = [0i64; 3];
            let mut b = [0i64; 3];
            for k in 0..3 {
                a[k] = (lo[k] - self.min[k]).max(0);
                b[k] = (hi_excl[k] - self.min[k]).min(self.dims[k]);
                if a[k] >= b[k] {
                    return;
                }
            }
            let (w, h) = (self.dims[0], self.dims[1]);
            for z in a[2]..b[2] {
                for y in a[1]..b[1] {
                    let row = ((z * h + y) * w) as usize;
                    if let Some(s) = self.data.get_mut(row + a[0] as usize..row + b[0] as usize) {
                        s.fill(value);
                    }
                }
            }
        }

        fn leaf(&mut self, origin: [i64; 3], values: &[f32]) {
            let outside = (0..3)
                .any(|k| origin[k] + 8 <= self.min[k] || origin[k] >= self.min[k] + self.dims[k]);
            if outside {
                return;
            }
            let (w, h) = (self.dims[0], self.dims[1]);
            for (i, &v) in values.iter().enumerate() {
                let p = [
                    origin[0] + (i >> 6) as i64 - self.min[0],
                    origin[1] + ((i >> 3) & 7) as i64 - self.min[1],
                    origin[2] + (i & 7) as i64 - self.min[2],
                ];
                if (0..3).all(|k| p[k] >= 0 && p[k] < self.dims[k])
                    && let Some(d) = self.data.get_mut(((p[2] * h + p[1]) * w + p[0]) as usize)
                {
                    *d = v;
                }
            }
        }
    }

    impl Visit for Dense {
        fn values(&self) -> bool {
            true
        }
        fn tile(&mut self, min: [i64; 3], size: i64, value: f32, _: bool) {
            self.fill(min, min.map(|v| v + size), value);
        }
        fn leaf(&mut self, _: [i64; 3], _: &[u64]) -> Result<()> {
            Ok(())
        }
    }

    pub(super) fn densify(
        data: &[u8],
        index: usize,
        region: Option<([i32; 3], [i32; 3])>,
        opts: &DecodeOptions,
    ) -> Result<VdbVolume> {
        let archive = read_archive(data)?;
        let grid = open_grid(&archive, index)?;
        if !matches!(grid.info.value_type, ValueType::Float | ValueType::Double) {
            return Err(Error::Unsupported);
        }
        if grid.info.compression & COMPRESS_BLOSC != 0 {
            return Err(Error::Unsupported);
        }
        let (min, max) = match region {
            Some(r) => r,
            None => match bounds(data, &grid)? {
                Some(b) => b,
                None => {
                    return Ok(VdbVolume {
                        width: 0,
                        height: 0,
                        depth: 0,
                        origin: [0; 3],
                        data: Buffer::default(),
                    });
                }
            },
        };
        let mut dims = [0i64; 3];
        for k in 0..3 {
            dims[k] = i64::from(max[k]) - i64::from(min[k]) + 1;
            if dims[k] <= 0 {
                return Err(Error::InvalidArgument);
            }
        }
        let voxels = dims.iter().map(|&d| d as u128).product::<u128>();
        if voxels > u128::from(opts.max_voxels) || voxels > (isize::MAX as u128) / 4 {
            return Err(Error::InvalidArgument);
        }
        let (topo, block, compression, key) = grid.tree?;
        let mut ctx = new_ctx(&key, compression)?;
        let mut dense = Dense {
            min: min.map(i64::from),
            dims,
            data: Vector::default(),
        };
        // The background is known after the topology header: fill once it is read.
        {
            let mut r = Reader::at(data, topo)?;
            r.i32()?;
            ctx.background = ctx.value(r.bytes(ctx.ty.size)?);
        }
        dense
            .data
            .resize(voxels as usize, ctx.background)
            .map_err(|_| Error::OutOfMemory)?;
        let mut r = Reader::at(data, topo)?;
        let tree = read_topology(&mut r, &mut ctx, &mut dense)?;
        // Leaf buffers (`LeafNode::readBuffers`): value mask again, then the values.
        let mut r = Reader::at(data, block)?;
        let mut scratch = Buffer::default();
        let mut values = [0f32; 512];
        for child in tree.children.iter() {
            for (origin, mask) in tree.leaves.get(child.leaves.clone()).unwrap_or(&[]) {
                r.bytes(64)?;
                read_values(&mut r, &ctx, mask, 512, &mut scratch, Some(&mut values))?;
                dense.leaf(*origin, &values);
            }
        }
        let mut out = crate::buffer::zeroed(dense.data.len() * 4)?;
        for (o, v) in out.as_chunks_mut::<4>().0.iter_mut().zip(dense.data.iter()) {
            *o = v.to_le_bytes();
        }
        Ok(VdbVolume {
            width: u32::try_from(dims[0]).map_err(|_| Error::InvalidArgument)?,
            height: u32::try_from(dims[1]).map_err(|_| Error::InvalidArgument)?,
            depth: u32::try_from(dims[2]).map_err(|_| Error::InvalidArgument)?,
            origin: min,
            data: out,
        })
    }
}
