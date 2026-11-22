//! OpenEXR (`exr` crate).
//!
//! Needs the `exr` cargo feature; without it every function returns `Unsupported`.
//!
//! Decoding reads scanline and tiled files, multi-part files, multi-resolution (mip and rip)
//! levels, f16 / f32 / u32 samples and every compression the crate decodes (NONE, RLE, ZIPS,
//! ZIP, PIZ, PXR24, B44, B44A, DWAA, DWAB); HTJ2K, deep data and subsampled channels return
//! [`Error::Unsupported`].
//!
//! [`decode`] returns the main layer of part 0 (the unprefixed channels, else the first layer
//! with color channels) at full resolution, placed in the display window:
//! - R / G / B (/ A): all half -> `Rgb16Float` / `Rgba16Float`; any f32 or u32 channel ->
//!   `Rgba32Float` (missing color channels are 0, missing alpha 1).
//! - Y / RY / BY (full resolution chroma): converted to RGB like OpenEXR's `RgbaInputFile`
//!   (luminance weights from the `chromaticities` attribute, Rec. 709 by default).
//! - Y alone or a single-channel layer: `R16Float` (half) / `R32Float` (f32, u32); Y + A ->
//!   RGBA with R = G = B = Y.
//! - u32 samples are converted to f32 (exact up to 2^24, rounded above); [`decode_planes`]
//!   returns them raw.
//! - Display window pixels outside the data window are 0 (also alpha, when the layer has
//!   alpha; alpha is 1 when it has none); data outside the display window is cropped.
//!   [`Window::Data`] returns the data window instead.
//!
//! [`parts`] lists the parts (channels, layers, windows, compression, tiling, levels);
//! [`decode_with`] decodes any part / layer / level, [`decode_planes`] any channels raw.
//! Encoding writes one part from `Rgba16Float`, `Rgb16Float`, `Rgba32Float`, `R16Float`
//! (channel `Y`), `R32Float` (channel `Y`); other uncompressed formats are converted to
//! `Rgba32Float` (sRGB formats are linearized). Compressions NONE, RLE, ZIPS, ZIP, PIZ,
//! PXR24, B44, B44A; scanlines or tiles, optional mip levels.

use super::Image;
#[cfg(feature = "exr")]
use crate::Format;
use crate::{Buffer, Error, Result, Surface};
use vas::Vector;
use vas::storage::Dynamic;

/// Plane indices per role: single channel, RGB, alpha, Y / RY / BY.
#[cfg(feature = "exr")]
type PlaneRoles = (
    Option<usize>,
    [Option<usize>; 3],
    Option<usize>,
    Option<(usize, usize, usize)>,
);

/// EXR compression methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// Uncompressed.
    None,
    /// Run-length encoding (lossless).
    Rle,
    /// Zlib, one scanline per block (lossless).
    Zips,
    /// Zlib, 16 scanlines per block (lossless).
    #[default]
    Zip,
    /// Wavelet + Huffman (lossless).
    Piz,
    /// f32 rounded to 24 bits + zlib (lossy for f32 only).
    Pxr24,
    /// 4x4 half blocks (lossy for f16 only).
    B44,
    /// B44 with flat-area optimization (lossy for f16 only).
    B44a,
    /// DCT, 32 lines (lossy; decode only).
    Dwaa,
    /// DCT, 256 lines (lossy; decode only).
    Dwab,
    /// High-throughput JPEG 2000 (not supported by the crate).
    Htj2k,
}

/// Channel sample type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleType {
    /// IEEE half.
    F16,
    /// IEEE single.
    F32,
    /// Unsigned 32-bit integer.
    U32,
}

impl SampleType {
    /// Bytes per sample.
    pub fn bytes(self) -> usize {
        match self {
            SampleType::F16 => 2,
            SampleType::F32 | SampleType::U32 => 4,
        }
    }
}

/// A pixel rectangle in file coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Which window [`decode_with`] returns (full resolution level only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Window {
    /// The display window: data placed in it, the rest filled (see the module docs).
    #[default]
    Display,
    /// The data window as stored.
    Data,
}

/// Resolution levels of a part.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Levels {
    /// One level.
    Single,
    /// `count` mip levels (index `(l, l)`).
    Mip { count: u32, round_up: bool },
    /// `x` by `y` rip levels (index `(lx, ly)`).
    Rip { x: u32, y: u32, round_up: bool },
}

/// One channel of a part.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelInfo {
    /// Full name, e.g. `R` or `diffuse.R`.
    pub name: String,
    pub sample_type: SampleType,
    /// Subsampling factors (1 = full resolution).
    pub sampling: (u32, u32),
    /// Quantize linearly (a hint for lossy compressions).
    pub linear: bool,
}

/// One part (header) of a file.
#[derive(Debug)]
pub struct PartInfo {
    /// Part name (multi-part files).
    pub name: Option<String>,
    /// Channels, sorted by name.
    pub channels: Vector<ChannelInfo, Dynamic>,
    /// Layer names: channel name prefixes before the last `.`; `""` = unprefixed channels.
    pub layers: Vector<String, Dynamic>,
    pub data_window: Rect,
    pub display_window: Rect,
    pub compression: Compression,
    /// Tile size, `None` for scanline parts.
    pub tile_size: Option<(u32, u32)>,
    pub levels: Levels,
    /// Deep data (not decodable).
    pub deep: bool,
}

impl PartInfo {
    /// Size of level `(lx, ly)` (mip levels: `lx == ly`).
    pub fn level_size(&self, level: (u32, u32)) -> Result<(u32, u32)> {
        let (w, h) = (self.data_window.width, self.data_window.height);
        let size = |n: u32, l: u32, up: bool| -> u32 {
            let d = 1u64 << l.min(40);
            let v = if up {
                (n as u64).div_ceil(d)
            } else {
                n as u64 / d
            };
            v.max(1) as u32
        };
        match self.levels {
            Levels::Single if level == (0, 0) => Ok((w, h)),
            Levels::Mip { count, round_up } if level.0 == level.1 && level.0 < count => {
                Ok((size(w, level.0, round_up), size(h, level.1, round_up)))
            }
            Levels::Rip { x, y, round_up } if level.0 < x && level.1 < y => {
                Ok((size(w, level.0, round_up), size(h, level.1, round_up)))
            }
            _ => Err(Error::InvalidArgument),
        }
    }
}

/// Raw samples of one channel: `width * height` little-endian samples of `sample_type`.
#[derive(Debug)]
pub struct Plane {
    pub name: String,
    pub sample_type: SampleType,
    pub width: u32,
    pub height: u32,
    pub data: Buffer,
}

/// What [`decode_with`] decodes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DecodeOptions {
    /// Part index.
    pub part: usize,
    /// Layer (channel prefix, `""` = unprefixed channels); `None` = the main layer.
    pub layer: Option<String>,
    /// Resolution level `(lx, ly)`.
    pub level: (u32, u32),
    /// Display or data window (levels other than `(0, 0)` always return their data).
    pub window: Window,
}

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EncodeOptions {
    pub compression: Compression,
    /// Tile size; `None` writes scanlines (mip levels need tiles: 64x64 is used then).
    pub tiles: Option<(u32, u32)>,
    /// Write a full mip chain (round down), generated with `mip_options`.
    pub mips: bool,
    pub mip_options: crate::mip::MipOptions,
}

/// Lists the parts of a file.
pub fn parts(data: &[u8]) -> Result<Vector<PartInfo, Dynamic>> {
    #[cfg(feature = "exr")]
    {
        imp::parts(data)
    }
    #[cfg(not(feature = "exr"))]
    {
        let _ = data;
        Err(Error::Unsupported)
    }
}

/// Decodes the main image (see the module docs).
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_with(data, &DecodeOptions::default())
}

/// Decodes a part / layer / level.
pub fn decode_with(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
    #[cfg(feature = "exr")]
    {
        imp::decode_with(data, opts)
    }
    #[cfg(not(feature = "exr"))]
    {
        let _ = (data, opts);
        Err(Error::Unsupported)
    }
}

/// Decodes the named channels (all channels when `names` is empty) of a part at a level, in
/// `names` order (else file order), over the level's data window.
pub fn decode_planes(
    data: &[u8],
    part: usize,
    level: (u32, u32),
    names: &[&str],
) -> Result<Vector<Plane, Dynamic>> {
    #[cfg(feature = "exr")]
    {
        imp::decode_planes(data, part, level, names)
    }
    #[cfg(not(feature = "exr"))]
    {
        let _ = (data, part, level, names);
        Err(Error::Unsupported)
    }
}

/// Encodes `src`.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    #[cfg(feature = "exr")]
    {
        imp::encode(src, opts)
    }
    #[cfg(not(feature = "exr"))]
    {
        let _ = (src, opts);
        Err(Error::Unsupported)
    }
}

/// Encodes a mip chain (level 0 first; each level half the previous, rounded down, all the
/// same format) as one tiled part. `opts.mips` is ignored.
pub fn encode_levels(levels: &[Surface], opts: &EncodeOptions) -> Result<Buffer> {
    #[cfg(feature = "exr")]
    {
        imp::encode_levels(levels, opts)
    }
    #[cfg(not(feature = "exr"))]
    {
        let _ = (levels, opts);
        Err(Error::Unsupported)
    }
}

/// Splits a channel name into (layer, base name) at the last `.`.
pub fn split_channel_name(name: &str) -> (&str, &str) {
    match name.rfind('.') {
        Some(i) => (name.get(..i).unwrap_or(""), name.get(i + 1..).unwrap_or("")),
        None => ("", name),
    }
}

#[cfg(feature = "exr")]
mod imp {
    use super::*;
    use exr::block::UncompressedBlock;
    use exr::block::reader::ChunksReader;
    use exr::math::{RoundingMode, Vec2};
    use exr::meta::attribute::{LevelMode, SampleType as XSample};
    use exr::meta::header::Header;
    use exr::meta::{BlockDescription, MetaData};
    use std::io::Cursor;

    /// Largest decoded level, in bytes.
    const MAX_BYTES: u64 = 1 << 32;
    /// Smallest output a tiny file may still claim.
    const MIN_LIMIT: u64 = 1 << 26;
    /// Largest plausible decompression ratio (zlib tops out near 1032:1).
    const MAX_RATIO: u64 = 4096;

    fn map_err(e: exr::error::Error) -> Error {
        match e {
            exr::error::Error::NotSupported(_) => Error::Unsupported,
            _ => Error::InvalidData,
        }
    }

    fn oom<T>(_: T) -> Error {
        Error::OutOfMemory
    }

    fn sample_type(t: XSample) -> SampleType {
        match t {
            XSample::F16 => SampleType::F16,
            XSample::F32 => SampleType::F32,
            XSample::U32 => SampleType::U32,
        }
    }

    fn compression(c: exr::compression::Compression) -> Compression {
        use exr::compression::Compression as C;
        match c {
            C::Uncompressed => Compression::None,
            C::RLE => Compression::Rle,
            C::ZIP1 => Compression::Zips,
            C::ZIP16 => Compression::Zip,
            C::PIZ => Compression::Piz,
            C::PXR24 => Compression::Pxr24,
            C::B44 => Compression::B44,
            C::B44A => Compression::B44a,
            C::DWAA(_) => Compression::Dwaa,
            C::DWAB(_) => Compression::Dwab,
            C::HTJ2K32 | C::HTJ2K256 => Compression::Htj2k,
        }
    }

    fn rect(pos: Vec2<i32>, size: Vec2<usize>) -> Result<Rect> {
        Ok(Rect {
            x: pos.0,
            y: pos.1,
            width: u32::try_from(size.0).map_err(|_| Error::InvalidData)?,
            height: u32::try_from(size.1).map_err(|_| Error::InvalidData)?,
        })
    }

    fn level_count(n: u32, up: bool) -> u32 {
        let n = n.max(1);
        let floor = 31 - n.leading_zeros();
        if up && !n.is_power_of_two() {
            floor + 2
        } else {
            floor + 1
        }
    }

    fn part_info(h: &Header) -> Result<PartInfo> {
        let mut channels = Vector::<ChannelInfo, Dynamic>::default();
        let mut layers = Vector::<String, Dynamic>::default();
        for c in &h.channels.list {
            let name = String::from_utf8_lossy(c.name.as_slice()).into_owned();
            let layer = split_channel_name(&name).0;
            if !layers.iter().any(|l| l == layer) {
                layers.push(layer.to_owned()).map_err(oom)?;
            }
            channels
                .push(ChannelInfo {
                    sample_type: sample_type(c.sample_type),
                    sampling: (
                        u32::try_from(c.sampling.0).map_err(|_| Error::InvalidData)?,
                        u32::try_from(c.sampling.1).map_err(|_| Error::InvalidData)?,
                    ),
                    linear: c.quantize_linearly,
                    name,
                })
                .map_err(oom)?;
        }
        let data_window = rect(h.own_attributes.layer_position, h.layer_size)?;
        let dw = h.shared_attributes.display_window;
        let display_window = rect(dw.position, dw.size)?;
        let (tile_size, levels) = match h.blocks {
            BlockDescription::ScanLines => (None, Levels::Single),
            BlockDescription::Tiles(t) => {
                let ts = (
                    u32::try_from(t.tile_size.0).map_err(|_| Error::InvalidData)?,
                    u32::try_from(t.tile_size.1).map_err(|_| Error::InvalidData)?,
                );
                let up = t.rounding_mode == RoundingMode::Up;
                let (w, hh) = (data_window.width, data_window.height);
                let levels = match t.level_mode {
                    LevelMode::Singular => Levels::Single,
                    LevelMode::MipMap => Levels::Mip {
                        count: level_count(w.max(hh), up),
                        round_up: up,
                    },
                    LevelMode::RipMap => Levels::Rip {
                        x: level_count(w, up),
                        y: level_count(hh, up),
                        round_up: up,
                    },
                };
                (Some(ts), levels)
            }
        };
        Ok(PartInfo {
            name: h
                .own_attributes
                .layer_name
                .as_ref()
                .map(|n| String::from_utf8_lossy(n.as_slice()).into_owned()),
            channels,
            layers,
            data_window,
            display_window,
            compression: compression(h.compression),
            tile_size,
            levels,
            deep: h.deep,
        })
    }

    fn read_meta(data: &[u8]) -> Result<MetaData> {
        let meta = MetaData::read_from_buffered(Cursor::new(data), false).map_err(map_err)?;
        // Every chunk needs an 8-byte offset table entry: bounds the tables the reader
        // allocates.
        let chunks = meta
            .headers
            .iter()
            .try_fold(0u64, |acc, h| acc.checked_add(h.chunk_count as u64))
            .ok_or(Error::InvalidData)?;
        if chunks.saturating_mul(8) > data.len() as u64 {
            return Err(Error::InvalidData);
        }
        Ok(meta)
    }

    pub fn parts(data: &[u8]) -> Result<Vector<PartInfo, Dynamic>> {
        let meta = read_meta(data)?;
        let mut out = Vector::<PartInfo, Dynamic>::default();
        for h in &meta.headers {
            out.push(part_info(h)?).map_err(oom)?;
        }
        Ok(out)
    }

    /// Decodes channels `sel` (indices into the part's channel list) of `part` at `level`.
    fn read_planes(
        data: &[u8],
        part: usize,
        level: (u32, u32),
        sel: &[usize],
    ) -> Result<(PartInfo, Vector<Plane, Dynamic>)> {
        let reader = exr::block::read(Cursor::new(data), false).map_err(map_err)?;
        let meta = reader.meta_data();
        let chunks = meta
            .headers
            .iter()
            .try_fold(0u64, |acc, h| acc.checked_add(h.chunk_count as u64))
            .ok_or(Error::InvalidData)?;
        if chunks.saturating_mul(8) > data.len() as u64 {
            return Err(Error::InvalidData);
        }
        let header = meta.headers.get(part).ok_or(Error::InvalidArgument)?;
        let info = part_info(header)?;
        if info.deep {
            return Err(Error::Unsupported);
        }
        if info.channels.iter().any(|c| c.sampling != (1, 1)) {
            return Err(Error::Unsupported);
        }
        let (w, h) = info.level_size(level)?;
        let mut bytes = 0u64;
        for &i in sel {
            let c = info.channels.get(i).ok_or(Error::InvalidArgument)?;
            bytes = bytes.saturating_add(w as u64 * h as u64 * c.sample_type.bytes() as u64);
        }
        let limit = (data.len() as u64)
            .saturating_mul(MAX_RATIO)
            .clamp(MIN_LIMIT, MAX_BYTES);
        if bytes > limit {
            return Err(Error::InvalidData);
        }
        let mut planes = Vector::<Plane, Dynamic>::default();
        for &i in sel {
            let c = info.channels.get(i).ok_or(Error::InvalidArgument)?;
            let len = w as usize * h as usize * c.sample_type.bytes();
            planes
                .push(Plane {
                    name: c.name.clone(),
                    sample_type: c.sample_type,
                    width: w,
                    height: h,
                    data: crate::buffer::zeroed(len)?,
                })
                .map_err(oom)?;
        }
        // Byte size of one line of each channel is width * sample bytes: precompute the
        // per-channel sample sizes of the part.
        let sizes: Vector<usize, Dynamic> = {
            let mut v = Vector::<usize, Dynamic>::default();
            for c in info.channels.iter() {
                v.push(c.sample_type.bytes()).map_err(oom)?;
            }
            v
        };
        let want = Vec2(level.0 as usize, level.1 as usize);
        let filtered = reader
            .filter_chunks(false, |_, _, block| {
                block.layer == part && block.level == want
            })
            .map_err(map_err)?;
        let mut filled = 0u64;
        let mut insert = |block: UncompressedBlock| -> Result<()> {
            place_block(&block, &sizes, sel, &mut planes)?;
            filled += 1;
            Ok(())
        };
        let mut err = None;
        let r = if filtered.len() >= 16 {
            filtered.decompress_parallel(false, |_, block| {
                insert(block).map_err(|e| {
                    err = Some(e);
                    exr::error::Error::Aborted
                })
            })
        } else {
            filtered.decompress_sequential(false, |_, block| {
                insert(block).map_err(|e| {
                    err = Some(e);
                    exr::error::Error::Aborted
                })
            })
        };
        if let Some(e) = err {
            return Err(e);
        }
        r.map_err(map_err)?;
        Ok((info, planes))
    }

    /// Copies the selected channels of a block into the planes.
    fn place_block(
        block: &UncompressedBlock,
        sizes: &[usize],
        sel: &[usize],
        planes: &mut [Plane],
    ) -> Result<()> {
        let bw = block.index.pixel_size.0;
        let bh = block.index.pixel_size.1;
        let (x0, y0) = (block.index.pixel_position.0, block.index.pixel_position.1);
        let mut off = 0usize;
        for row in 0..bh {
            for (ci, &sz) in sizes.iter().enumerate() {
                let n = bw * sz;
                let line = block.data.get(off..off + n).ok_or(Error::InvalidData)?;
                off += n;
                for (pi, _) in sel.iter().enumerate().filter(|(_, s)| **s == ci) {
                    let p = planes.get_mut(pi).ok_or(Error::InvalidData)?;
                    let pw = p.width as usize;
                    let y = y0 + row;
                    if y >= p.height as usize || x0 + bw > pw {
                        return Err(Error::InvalidData);
                    }
                    let start = (y * pw + x0) * sz;
                    let dst = p.data.get_mut(start..start + n).ok_or(Error::InvalidData)?;
                    // Native-endian samples -> little-endian.
                    if cfg!(target_endian = "little") {
                        dst.copy_from_slice(line);
                    } else {
                        for (d, s) in dst.chunks_exact_mut(sz).zip(line.chunks_exact(sz)) {
                            for (a, b) in d.iter_mut().zip(s.iter().rev()) {
                                *a = *b;
                            }
                        }
                    }
                }
            }
        }
        if off != block.data.len() {
            return Err(Error::InvalidData);
        }
        Ok(())
    }

    pub fn decode_planes(
        data: &[u8],
        part: usize,
        level: (u32, u32),
        names: &[&str],
    ) -> Result<Vector<Plane, Dynamic>> {
        let meta = read_meta(data)?;
        let header = meta.headers.get(part).ok_or(Error::InvalidArgument)?;
        let mut sel = Vector::<usize, Dynamic>::default();
        if names.is_empty() {
            for i in 0..header.channels.list.len() {
                sel.push(i).map_err(oom)?;
            }
        } else {
            for n in names {
                let i = header
                    .channels
                    .list
                    .iter()
                    .position(|c| c.name.as_slice() == n.as_bytes())
                    .ok_or(Error::InvalidArgument)?;
                sel.push(i).map_err(oom)?;
            }
        }
        Ok(read_planes(data, part, level, &sel)?.1)
    }

    /// Channel roles of a layer.
    #[derive(Default)]
    struct Roles {
        r: Option<usize>,
        g: Option<usize>,
        b: Option<usize>,
        a: Option<usize>,
        y: Option<usize>,
        ry: Option<usize>,
        by: Option<usize>,
        count: usize,
        first: Option<usize>,
    }

    fn roles(info: &PartInfo, layer: &str) -> Roles {
        let mut r = Roles::default();
        for (i, c) in info.channels.iter().enumerate() {
            let (l, base) = split_channel_name(&c.name);
            if l != layer {
                continue;
            }
            r.count += 1;
            r.first.get_or_insert(i);
            match base {
                "R" => r.r = Some(i),
                "G" => r.g = Some(i),
                "B" => r.b = Some(i),
                "A" => r.a = Some(i),
                "Y" => r.y = Some(i),
                "RY" => r.ry = Some(i),
                "BY" => r.by = Some(i),
                _ => {}
            }
        }
        r
    }

    impl Roles {
        fn has_color(&self) -> bool {
            self.r.is_some() || self.g.is_some() || self.b.is_some() || self.y.is_some()
        }
    }

    fn main_layer(info: &PartInfo) -> Result<String> {
        if roles(info, "").has_color() {
            return Ok(String::new());
        }
        for l in info.layers.iter() {
            if roles(info, l).has_color() {
                return Ok(l.clone());
            }
        }
        info.layers.first().cloned().ok_or(Error::Unsupported)
    }

    /// Output layout of a layer.
    enum Kind {
        /// One channel.
        Single(usize),
        /// Color channels (R, G, B), optional alpha.
        Rgb([Option<usize>; 3], Option<usize>),
        /// Luminance + alpha.
        Ya(usize, usize),
        /// Luminance / chroma, optional alpha.
        Yc(usize, usize, usize, Option<usize>),
    }

    fn sample(p: &Plane, i: usize) -> f32 {
        let sz = p.sample_type.bytes();
        let Some(b) = p.data.get(i * sz..i * sz + sz) else {
            return 0.0;
        };
        match p.sample_type {
            SampleType::F16 => crate::half::to_f32(u16::from_le_bytes([b[0], b[1]])),
            SampleType::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            SampleType::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
        }
    }

    /// Luminance weights of chromaticities (Imath `RGBtoXYZ` row Y), as OpenEXR `computeYw`.
    fn luminance_weights(c: Option<exr::meta::attribute::Chromaticities>) -> [f32; 3] {
        let (red, green, blue, white) = match c {
            Some(c) => (
                (c.red.0, c.red.1),
                (c.green.0, c.green.1),
                (c.blue.0, c.blue.1),
                (c.white.0, c.white.1),
            ),
            None => ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06), (0.3127, 0.3290)),
        };
        let y = 1.0f32;
        let xx = white.0 * y / white.1;
        let zz = (1.0 - white.0 - white.1) * y / white.1;
        let d =
            red.0 * (blue.1 - green.1) + blue.0 * (green.1 - red.1) + green.0 * (red.1 - blue.1);
        let sr = (xx * (blue.1 - green.1) - green.0 * (y * (blue.1 - 1.0) + blue.1 * (xx + zz))
            + blue.0 * (y * (green.1 - 1.0) + green.1 * (xx + zz)))
            / d;
        let sg = (xx * (red.1 - blue.1) + red.0 * (y * (blue.1 - 1.0) + blue.1 * (xx + zz))
            - blue.0 * (y * (red.1 - 1.0) + red.1 * (xx + zz)))
            / d;
        let sb = (xx * (green.1 - red.1) - red.0 * (y * (green.1 - 1.0) + green.1 * (xx + zz))
            + green.0 * (y * (red.1 - 1.0) + red.1 * (xx + zz)))
            / d;
        let w = [sr * red.1, sg * green.1, sb * blue.1];
        if w.iter().all(|v| v.is_finite()) {
            w
        } else {
            [0.2126, 0.7152, 0.0722]
        }
    }

    pub fn decode_with(data: &[u8], opts: &DecodeOptions) -> Result<Image> {
        let meta = read_meta(data)?;
        let header = meta.headers.get(opts.part).ok_or(Error::InvalidArgument)?;
        let info = part_info(header)?;
        let chroma = header.shared_attributes.chromaticities;
        let layer = match &opts.layer {
            Some(l) => {
                if !info.layers.iter().any(|x| x == l) {
                    return Err(Error::InvalidArgument);
                }
                l.clone()
            }
            None => main_layer(&info)?,
        };
        let ro = roles(&info, &layer);
        let kind = if ro.r.is_some() || ro.g.is_some() || ro.b.is_some() {
            Kind::Rgb([ro.r, ro.g, ro.b], ro.a)
        } else if let (Some(y), Some(ry), Some(by)) = (ro.y, ro.ry, ro.by) {
            Kind::Yc(y, ry, by, ro.a)
        } else if let (Some(y), Some(a)) = (ro.y, ro.a) {
            Kind::Ya(y, a)
        } else if let Some(y) = ro.y {
            Kind::Single(y)
        } else if ro.count == 1 {
            Kind::Single(ro.first.ok_or(Error::Unsupported)?)
        } else {
            return Err(Error::Unsupported);
        };
        let mut sel = Vector::<usize, Dynamic>::default();
        let mut add = |i: Option<usize>| -> Result<Option<usize>> {
            match i {
                Some(i) => {
                    sel.push(i).map_err(oom)?;
                    Ok(Some(sel.len() - 1))
                }
                None => Ok(None),
            }
        };
        // Plane indices per role.
        let (single, rgb, alpha, yc): PlaneRoles = match kind {
            Kind::Single(i) => (add(Some(i))?, [None; 3], None, None),
            Kind::Rgb(c, a) => (None, [add(c[0])?, add(c[1])?, add(c[2])?], add(a)?, None),
            Kind::Ya(y, a) => {
                let y = add(Some(y))?;
                (None, [y, y, y], add(Some(a))?, None)
            }
            Kind::Yc(y, ry, by, a) => {
                let y = add(Some(y))?.ok_or(Error::InvalidData)?;
                let ry = add(Some(ry))?.ok_or(Error::InvalidData)?;
                let by = add(Some(by))?.ok_or(Error::InvalidData)?;
                (None, [None; 3], add(a)?, Some((y, ry, by)))
            }
        };
        let (_, planes) = read_planes(data, opts.part, opts.level, &sel)?;
        let half = planes.iter().all(|p| p.sample_type == SampleType::F16);
        let format = match (single.is_some(), half, alpha.is_some()) {
            (true, true, _) => Format::R16Float,
            (true, false, _) => Format::R32Float,
            (false, true, true) => Format::Rgba16Float,
            (false, true, false) => Format::Rgb16Float,
            (false, false, _) => Format::Rgba32Float,
        };
        let first = planes.first().ok_or(Error::InvalidData)?;
        let (lw, lh) = (first.width, first.height);
        // Output window in level data coordinates.
        let (ox, oy, ow, oh) = if opts.level == (0, 0) && opts.window == Window::Display {
            let d = info.display_window;
            (
                d.x as i64 - info.data_window.x as i64,
                d.y as i64 - info.data_window.y as i64,
                d.width,
                d.height,
            )
        } else {
            (0, 0, lw, lh)
        };
        let mut img = Image::new(format, ow, oh)?;
        let n = match format {
            Format::R16Float | Format::R32Float => 1,
            Format::Rgb16Float => 3,
            _ => 4,
        };
        let yw = luminance_weights(chroma);
        let px = format.block_bytes();
        let has_alpha = alpha.is_some();
        for (y, row) in img.data.chunks_exact_mut(ow as usize * px).enumerate() {
            let sy = oy + y as i64;
            for (x, out) in row.chunks_exact_mut(px).enumerate() {
                let sx = ox + x as i64;
                let inside = sx >= 0 && sy >= 0 && sx < lw as i64 && sy < lh as i64;
                let mut v = [0.0f32, 0.0, 0.0, if has_alpha { 0.0 } else { 1.0 }];
                if inside {
                    let i = sy as usize * lw as usize + sx as usize;
                    let get =
                        |p: Option<usize>| p.and_then(|p| planes.get(p)).map(|p| sample(p, i));
                    if let Some(s) = single {
                        v[0] = get(Some(s)).unwrap_or(0.0);
                    } else if let Some((yi, ryi, byi)) = yc {
                        let l = get(Some(yi)).unwrap_or(0.0);
                        let ry = get(Some(ryi)).unwrap_or(0.0);
                        let by = get(Some(byi)).unwrap_or(0.0);
                        if ry == 0.0 && by == 0.0 {
                            v[..3].fill(l);
                        } else {
                            let r = (ry + 1.0) * l;
                            let b = (by + 1.0) * l;
                            let g = (l - r * yw[0] - b * yw[2]) / yw[1];
                            v[..3].copy_from_slice(&[r, g, b]);
                        }
                    } else {
                        for c in 0..3 {
                            v[c] = get(rgb[c]).unwrap_or(0.0);
                        }
                    }
                    if let Some(a) = get(alpha) {
                        v[3] = a;
                    }
                }
                for (c, val) in v.iter().take(n).enumerate() {
                    if half {
                        let hb = crate::half::from_f32(*val).to_le_bytes();
                        if let Some(d) = out.get_mut(c * 2..c * 2 + 2) {
                            d.copy_from_slice(&hb);
                        }
                    } else if let Some(d) = out.get_mut(c * 4..c * 4 + 4) {
                        d.copy_from_slice(&val.to_le_bytes());
                    }
                }
            }
        }
        // Half planes copied through f32 and back are exact (to_f32 is exact, from_f32 of an
        // exact half is the identity, NaN payloads aside).
        Ok(img)
    }

    // ----- encoding -----

    use exr::image::{
        AnyChannel, AnyChannels, Blocks, Encoding, FlatSamples, Layer, Levels as XLevels,
    };
    use exr::prelude::{LayerAttributes, LineOrder, SmallVec, WritableImage, f16};

    fn xcompression(c: Compression) -> Result<exr::compression::Compression> {
        use exr::compression::Compression as C;
        Ok(match c {
            Compression::None => C::Uncompressed,
            Compression::Rle => C::RLE,
            Compression::Zips => C::ZIP1,
            Compression::Zip => C::ZIP16,
            Compression::Piz => C::PIZ,
            Compression::Pxr24 => C::PXR24,
            Compression::B44 => C::B44,
            Compression::B44a => C::B44A,
            Compression::Dwaa | Compression::Dwab | Compression::Htj2k => {
                return Err(Error::Unsupported);
            }
        })
    }

    /// Channel names and sample kind of a storable format.
    fn layout(f: Format) -> Option<(&'static [&'static str], bool)> {
        Some(match f {
            Format::Rgba16Float => (&["R", "G", "B", "A"], true),
            Format::Rgb16Float => (&["R", "G", "B"], true),
            Format::R16Float => (&["Y"], true),
            Format::Rgba32Float => (&["R", "G", "B", "A"], false),
            Format::R32Float => (&["Y"], false),
            _ => return None,
        })
    }

    /// One channel of a surface as flat samples.
    fn channel_samples(src: &Surface, c: usize, half: bool) -> Result<FlatSamples> {
        let px = src.format().block_bytes();
        let count = src.width() as usize * src.height() as usize;
        let data = src.data();
        if half {
            let mut v: Vec<f16> = Vec::new();
            v.try_reserve_exact(count).map_err(oom)?;
            for p in data.chunks_exact(px) {
                let b = p.get(c * 2..c * 2 + 2).ok_or(Error::InvalidArgument)?;
                v.push(f16::from_bits(u16::from_le_bytes([b[0], b[1]])));
            }
            Ok(FlatSamples::F16(v))
        } else {
            let mut v: Vec<f32> = Vec::new();
            v.try_reserve_exact(count).map_err(oom)?;
            for p in data.chunks_exact(px) {
                let b = p.get(c * 4..c * 4 + 4).ok_or(Error::InvalidArgument)?;
                v.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
            Ok(FlatSamples::F32(v))
        }
    }

    /// `src` in a storable format (converted when needed).
    fn storable(src: &Surface) -> Result<Option<Buffer>> {
        if layout(src.format()).is_some() {
            return Ok(None);
        }
        if !crate::convert::can_read(src.format()) {
            return Err(Error::Unsupported);
        }
        let len = Format::Rgba32Float.surface_bytes(src.width(), src.height())?;
        let mut buf = crate::buffer::zeroed(len)?;
        let o = crate::convert::ConvertOptions {
            srgb_transfer: true,
        };
        crate::convert::convert_with(src, Format::Rgba32Float, &o, &mut buf)?;
        Ok(Some(buf))
    }

    pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
        let conv = storable(src)?;
        let src = match &conv {
            Some(b) => Surface::new(Format::Rgba32Float, src.width(), src.height(), b)?,
            None => *src,
        };
        if !opts.mips {
            return write(&[src], opts, false);
        }
        let count = crate::image::mip_levels(src.width(), src.height());
        let mips = crate::mip::generate_mips(&src, count, &opts.mip_options)?;
        let mut levels = Vector::<Surface, Dynamic>::default();
        levels.push(src).map_err(oom)?;
        for (l, b) in mips.iter().enumerate() {
            let (w, h) = crate::image::mip_dim(src.width(), src.height(), l as u32 + 1);
            levels
                .push(Surface::new(src.format(), w, h, b)?)
                .map_err(oom)?;
        }
        write(&levels, opts, true)
    }

    pub fn encode_levels(levels: &[Surface], opts: &EncodeOptions) -> Result<Buffer> {
        let base = levels.first().ok_or(Error::InvalidArgument)?;
        if layout(base.format()).is_none() {
            return Err(Error::Unsupported);
        }
        if levels.len() as u32 > crate::image::mip_levels(base.width(), base.height()) {
            return Err(Error::InvalidArgument);
        }
        for (l, s) in levels.iter().enumerate() {
            let dim = crate::image::mip_dim(base.width(), base.height(), l as u32);
            if s.format() != base.format() || (s.width(), s.height()) != dim {
                return Err(Error::InvalidArgument);
            }
        }
        write(levels, opts, true)
    }

    fn write(levels: &[Surface], opts: &EncodeOptions, mip: bool) -> Result<Buffer> {
        let base = levels.first().ok_or(Error::InvalidArgument)?;
        let (names, half) = layout(base.format()).ok_or(Error::Unsupported)?;
        let (w, h) = (base.width() as usize, base.height() as usize);
        let compression = xcompression(opts.compression)?;
        let tiles = match (opts.tiles, mip) {
            (Some((tw, th)), _) => {
                if tw == 0 || th == 0 {
                    return Err(Error::InvalidArgument);
                }
                Some(Vec2(tw as usize, th as usize))
            }
            (None, true) => Some(Vec2(64, 64)),
            (None, false) => None,
        };
        let blocks = match tiles {
            Some(t) => Blocks::Tiles(t),
            None => Blocks::ScanLines,
        };
        let encoding = Encoding {
            compression,
            blocks,
            line_order: LineOrder::Increasing,
        };
        let mut list: SmallVec<[AnyChannel<XLevels<FlatSamples>>; 4]> = SmallVec::new();
        for (c, name) in names.iter().enumerate() {
            let data = if mip {
                let mut v = Vec::new();
                v.try_reserve_exact(levels.len()).map_err(oom)?;
                for s in levels {
                    v.push(channel_samples(s, c, half)?);
                }
                XLevels::Mip {
                    rounding_mode: RoundingMode::Down,
                    level_data: v,
                }
            } else {
                XLevels::Singular(channel_samples(base, c, half)?)
            };
            let mut ch = AnyChannel::new(*name, data);
            ch.quantize_linearly = *name == "A";
            list.push(ch);
        }
        let layer = Layer::new(
            Vec2(w, h),
            LayerAttributes::default(),
            encoding,
            AnyChannels::sort(list),
        );
        let image = exr::image::Image::from_layer(layer);
        let mut cursor = Cursor::new(Vec::new());
        let writer = image.write();
        // The crate builds a thread pool per call: only worth it for large images.
        let writer = if w * h >= 1 << 18 {
            writer
        } else {
            writer.non_parallel()
        };
        writer.to_buffered(&mut cursor).map_err(map_err)?;
        let bytes = cursor.into_inner();
        let mut out = crate::buffer::zeroed(bytes.len())?;
        out.copy_from_slice(&bytes);
        Ok(out)
    }
}
