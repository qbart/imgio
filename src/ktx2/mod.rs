//! KTX2 container: Basis Universal payloads (ETC1S/BasisLZ, UASTC LDR 4x4 and HDR 4x4,
//! ASTC HDR 6x6, UASTC HDR 6x6 intermediate, ASTC LDR, XUASTC LDR, XUBC7) and plain
//! `VkFormat` payloads of every imgio [`Format`] with a Vulkan equivalent, raw or Zstandard
//! supercompressed; [`write`] creates plain files.
//!
//! Layout from the KTX 2.0 specification; Basis payload detection follows Basis Universal's
//! `ktx2_transcoder` (DFD color model, vkFormat, supercompression scheme).

mod dfd;
mod vk;
mod writer;

pub use dfd::{Dfd, dfd};
pub use vk::{format_from_vk, vk_format};
pub use writer::{Ktx2Desc, Supercompression, WriteOptions, write};

use crate::basis::BasisTexFormat;
use crate::basis::TranscodeOptions;
use crate::basis::ll::etc1s::LowlevelEtc1sTranscoder;
use crate::basis::transcode::{
    SliceRef, TranscodedImage, alloc_items, can_transcode, etc1s_transcoder, transcode_items,
    transcode_slice,
};
use crate::{AstcBlock, Buffer, Error, Format, Result};
use std::sync::OnceLock;
use vas::Vector;
use vas::storage::Dynamic;

pub(crate) const IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];
const HEADER_SIZE: usize = 80;
const LEVEL_INDEX_SIZE: usize = 24;

const SS_NONE: u32 = 0;
const SS_BASISLZ: u32 = 1;
const SS_ZSTD: u32 = 2;
const SS_UASTC_HDR_6X6I: u32 = 4;
const SS_XUASTC_LDR: u32 = 5;
const SS_XUBC7: u32 = 6;

const MODEL_ASTC: u32 = 162;
const MODEL_ETC1S: u32 = 163;
const MODEL_UASTC_LDR_4X4: u32 = 166;
const MODEL_UASTC_HDR_4X4: u32 = 167;
const MODEL_UASTC_HDR_6X6I: u32 = 168;
const MODEL_XUASTC_LDR: u32 = 169;
const MODEL_XUBC7: u32 = 170;

const VK_ASTC_4X4_SFLOAT: u32 = 1_000_066_000;
const VK_ASTC_6X6_SFLOAT: u32 = 1_000_066_004;
/// `VK_FORMAT_ASTC_4x4_UNORM_BLOCK` .. `VK_FORMAT_ASTC_12x12_SRGB_BLOCK` (odd = unorm).
const VK_ASTC_LDR: std::ops::RangeInclusive<u32> = 157..=184;

const TRANSFER_SRGB: u32 = 2;
const CHANNEL_UASTC_RGBA: u32 = 3;
const CHANNEL_UASTC_RRRG: u32 = 5;
const IMAGE_IS_P_FRAME: u32 = 2;

#[derive(Debug, Clone, Copy)]
struct Level {
    ofs: u64,
    len: u64,
    uncompressed_len: u64,
}

/// Where one image (level/layer/face) lives relative to its level's data.
#[derive(Debug, Clone, Copy)]
struct ImageDesc {
    flags: u32,
    rgb: (u32, u32),
    alpha: (u32, u32),
}

/// A parsed KTX2 file with a Basis Universal payload, borrowing its bytes.
#[derive(Debug)]
pub struct Ktx2File<'a> {
    data: &'a [u8],
    /// Basis Universal payload (`None` for plain formats).
    tex_format: Option<BasisTexFormat>,
    /// Plain payload: the `VkFormat`'s imgio format when the DFD model is its standard one.
    format: Option<Format>,
    vk_format: u32,
    width: u32,
    height: u32,
    layers: u32,
    faces: u32,
    scheme: u32,
    srgb: bool,
    /// ASTC LDR: sRGB decode profile (even vkFormat).
    astc_srgb: bool,
    deblock: bool,
    has_alpha: bool,
    video: bool,
    levels: Vector<Level, Dynamic>,
    /// ETC1S, 6x6i, XUASTC or XUBC7 slice locations, indexed `level * layers * faces + layer * faces + face`.
    images: Vector<ImageDesc, Dynamic>,
    etc1s: Option<LowlevelEtc1sTranscoder>,
    /// Zstandard-inflated level data, filled on first use.
    inflated: Vector<OnceLock<Option<Buffer>>, Dynamic>,
}

fn u32le(b: &[u8], ofs: usize) -> Result<u32> {
    b.get(ofs..ofs + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or(Error::InvalidData)
}

fn u64le(b: &[u8], ofs: usize) -> Result<u64> {
    Ok(u64::from(u32le(b, ofs)?) | (u64::from(u32le(b, ofs + 4)?) << 32))
}

fn range(data: &[u8], ofs: u64, len: u64) -> Result<&[u8]> {
    let start = usize::try_from(ofs).map_err(|_| Error::InvalidData)?;
    let len = usize::try_from(len).map_err(|_| Error::InvalidData)?;
    data.get(start..start.checked_add(len).ok_or(Error::InvalidData)?)
        .ok_or(Error::InvalidData)
}

/// True when a DFD sample is alpha (BC1A: "alpha present").
fn dfd_has_alpha(dfd: &[u8], model: u32) -> bool {
    let samples = (dfd.len().saturating_sub(28)) / 16;
    (0..samples).any(|i| {
        let ch = u32le(dfd, 28 + i * 16).map_or(0, |w| (w >> 24) & 15);
        ch == 15 || (model == 0x80 && ch == 1)
    })
}

/// KTX2 `typeSize`: the size of the data type for uncompressed formats, 1 otherwise.
pub(crate) fn type_size_of(format: Format) -> u32 {
    use Format::*;
    match format {
        R16Unorm | R16Snorm | Rg16Unorm | Rg16Snorm | Rgb16Float | Rgba16Float | Rgba16Unorm
        | Rgb565 | Bgr565 | Rgba4444 => 2,
        Rgba32Float | Rgb9e5Float => 4,
        _ => 1,
    }
}

fn oom<T>(_: T) -> Error {
    Error::OutOfMemory
}

/// The value of `key` in the key/value data (bytes after the key's NUL terminator).
fn key_value<'k>(kvd: &'k [u8], key: &[u8]) -> Option<&'k [u8]> {
    let mut at = 0usize;
    while let Ok(n) = u32le(kvd, at) {
        let n = n as usize;
        let kv = kvd.get(at + 4..(at + 4).checked_add(n)?)?;
        if kv.len() > key.len() && kv.starts_with(key) && kv[key.len()] == 0 {
            return kv.get(key.len() + 1..);
        }
        at += 4 + n.div_ceil(4) * 4;
    }
    None
}

impl<'a> Ktx2File<'a> {
    /// Parses and validates the header, level index, DFD, key/values and supercompression data.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() <= HEADER_SIZE || data[..12] != IDENTIFIER {
            return Err(Error::InvalidData);
        }
        let h = |ofs| u32le(data, ofs);
        let (vk_format, type_size) = (h(12)?, h(16)?);
        let (width, height, depth) = (h(20)?, h(24)?, h(28)?);
        let (layer_count, faces, level_count, scheme) = (h(32)?, h(36)?, h(40)?, h(44)?);
        let (dfd_ofs, dfd_len, kvd_ofs, kvd_len) = (h(48)?, h(52)?, h(56)?, h(60)?);
        let (sgd_ofs, sgd_len) = (u64le(data, 64)?, u64le(data, 72)?);
        if width == 0 || height == 0 || depth != 0 {
            return Err(Error::Unsupported);
        }
        if (faces != 1 && faces != 6) || (faces == 6 && width != height) {
            return Err(Error::InvalidData);
        }
        if level_count == 0 || level_count > crate::image::mip_levels(width, height) {
            return Err(Error::InvalidData);
        }
        let layers = layer_count.max(1);

        let mut levels = Vector::<Level, Dynamic>::default();
        for i in 0..level_count as usize {
            let at = HEADER_SIZE + i * LEVEL_INDEX_SIZE;
            let l = Level {
                ofs: u64le(data, at)?,
                len: u64le(data, at + 8)?,
                uncompressed_len: u64le(data, at + 16)?,
            };
            if l.len == 0 || l.ofs < HEADER_SIZE as u64 {
                return Err(Error::InvalidData);
            }
            range(data, l.ofs, l.len)?;
            levels.push(l).map_err(oom)?;
        }

        // Data format descriptor: one basic block (Basis payloads: 1 or 2 samples).
        let dfd = range(data, dfd_ofs.into(), dfd_len.into())?;
        if dfd_len < 28 || u32le(dfd, 0)? != dfd_len {
            return Err(Error::InvalidData);
        }
        let bits = u32le(dfd, 12)?;
        let (model, transfer) = (bits & 255, (bits >> 16) & 255);
        let dims = u32le(dfd, 16)?;
        let dfd_block = AstcBlock::from_dim((dims & 255) + 1, ((dims >> 8) & 255) + 1);
        let chan0 = (u32le(dfd, 28)? >> 24) & 15;
        let uastc_alpha = chan0 == CHANNEL_UASTC_RGBA || chan0 == CHANNEL_UASTC_RRRG;
        let plain_format = format_from_vk(vk_format).filter(|&f| {
            // Standard DFD model of the vkFormat; Basis payloads use their own models.
            dfd::dfd(f).is_some_and(|d| d.words()[3] & 255 == model)
        });
        let basis = (dfd_len == 44 || dfd_len == 60) && type_size == 1;
        let detected = match model {
            _ if !basis => None,
            MODEL_ETC1S if vk_format == 0 && scheme == SS_BASISLZ => {
                Some((BasisTexFormat::Etc1s, dfd_len == 60))
            }
            MODEL_UASTC_LDR_4X4 if vk_format == 0 => {
                Some((BasisTexFormat::UastcLdr4x4, uastc_alpha))
            }
            MODEL_UASTC_HDR_4X4 if vk_format == VK_ASTC_4X4_SFLOAT => {
                Some((BasisTexFormat::UastcHdr4x4, false))
            }
            MODEL_ASTC if vk_format == VK_ASTC_6X6_SFLOAT => {
                Some((BasisTexFormat::AstcHdr6x6, false))
            }
            MODEL_ASTC if VK_ASTC_LDR.contains(&vk_format) => {
                // vkFormat order: 4x4 .. 12x12, each as UNORM then SRGB.
                let b = AstcBlock::ALL[((vk_format - 157) / 2) as usize];
                if dfd_block != Some(b) {
                    return Err(Error::InvalidData);
                }
                Some((BasisTexFormat::AstcLdr(b), uastc_alpha))
            }
            MODEL_XUASTC_LDR if vk_format == 0 && scheme == SS_XUASTC_LDR => {
                let b = dfd_block.ok_or(Error::InvalidData)?;
                Some((BasisTexFormat::XuastcLdr(b), uastc_alpha))
            }
            MODEL_XUBC7 if vk_format == 0 && scheme == SS_XUBC7 => {
                Some((BasisTexFormat::Xubc7, uastc_alpha))
            }
            MODEL_UASTC_HDR_6X6I
                if vk_format == 0 && (scheme == SS_UASTC_HDR_6X6I || scheme == SS_BASISLZ) =>
            {
                Some((BasisTexFormat::UastcHdr6x6Intermediate, false))
            }
            _ => None,
        };
        let (tex_format, has_alpha) = match (detected, plain_format) {
            (Some((t, a)), _) => (Some(t), a),
            (None, Some(_)) => (None, dfd_has_alpha(dfd, model)),
            (None, None) => return Err(Error::Unsupported),
        };
        if let Some(f) = plain_format
            && type_size != type_size_of(f)
        {
            return Err(Error::InvalidData);
        }
        // Payloads stored as whole blocks per image (no supercompression global data).
        let plain = match tex_format {
            None => true,
            Some(t) => matches!(
                t,
                BasisTexFormat::UastcLdr4x4
                    | BasisTexFormat::UastcHdr4x4
                    | BasisTexFormat::AstcHdr6x6
                    | BasisTexFormat::AstcLdr(_)
            ),
        };
        if plain {
            if scheme != SS_NONE && scheme != SS_ZSTD {
                return Err(Error::Unsupported);
            }
            if scheme == SS_ZSTD && levels.iter().any(|l| l.uncompressed_len == 0) {
                return Err(Error::InvalidData);
            }
        }

        let kvd = if kvd_len > 0 {
            range(data, kvd_ofs.into(), kvd_len.into())?
        } else {
            &[]
        };
        let mut video = key_value(kvd, b"KTXanimData").is_some();
        // basisu `DeblockFilterID` = "1": deblock ASTC LDR / XUASTC when transcoding.
        let deblock = matches!(key_value(kvd, b"DeblockFilterID"), Some([b'1', 0, ..]));

        // Supercompression global data.
        let total_images = (level_count as usize)
            .checked_mul(layers as usize)
            .and_then(|n| n.checked_mul(faces as usize))
            .ok_or(Error::InvalidData)?;
        let mut images = Vector::<ImageDesc, Dynamic>::default();
        let mut etc1s = None;
        if !plain {
            let sgd = range(data, sgd_ofs, sgd_len)?;
            if tex_format == Some(BasisTexFormat::Etc1s) {
                let g = |ofs| u32le(sgd, ofs);
                let endpoint_count = g(0)? & 0xFFFF;
                let selector_count = g(0)? >> 16;
                let (ep_len, sel_len, tab_len) = (g(4)? as u64, g(8)? as u64, g(12)? as u64);
                let descs_end = 20 + total_images as u64 * 20;
                for i in 0..total_images {
                    let at = 20 + i * 20;
                    images
                        .push(ImageDesc {
                            flags: g(at)?,
                            rgb: (g(at + 4)?, g(at + 8)?),
                            alpha: (g(at + 12)?, g(at + 16)?),
                        })
                        .map_err(oom)?;
                }
                let ep = range(sgd, descs_end, ep_len)?;
                let sel = range(sgd, descs_end + ep_len, sel_len)?;
                let tab = range(sgd, descs_end + ep_len + sel_len, tab_len)?;
                etc1s = Some(etc1s_transcoder(
                    (endpoint_count, ep),
                    (selector_count, sel),
                    tab,
                )?);
                if faces == 1
                    && layers > 1
                    && images.iter().any(|d| d.flags & IMAGE_IS_P_FRAME != 0)
                {
                    video = true;
                }
            } else {
                // UASTC HDR 6x6i, XUASTC LDR, XUBC7: offset/length (+ profile in the standard
                // layout) per image.
                let std_layout = matches!(scheme, SS_UASTC_HDR_6X6I | SS_XUASTC_LDR | SS_XUBC7);
                let stride = if std_layout { 12 } else { 8 };
                if sgd.len() != total_images * stride {
                    return Err(Error::InvalidData);
                }
                for i in 0..total_images {
                    let at = i * stride;
                    images
                        .push(ImageDesc {
                            flags: 0,
                            rgb: (u32le(sgd, at)?, u32le(sgd, at + 4)?),
                            alpha: (0, 0),
                        })
                        .map_err(oom)?;
                }
            }
        }
        let mut inflated = Vector::<OnceLock<Option<Buffer>>, Dynamic>::default();
        inflated
            .resize_with(level_count as usize, OnceLock::new)
            .map_err(oom)?;
        Ok(Self {
            data,
            tex_format,
            format: plain_format,
            vk_format,
            width,
            height,
            layers,
            faces,
            scheme,
            srgb: transfer == TRANSFER_SRGB,
            astc_srgb: vk_format & 1 == 0,
            deblock,
            has_alpha,
            video,
            levels,
            images,
            etc1s,
            inflated,
        })
    }

    /// The Basis Universal payload format (`None` for plain `VkFormat` payloads).
    pub fn tex_format(&self) -> Option<BasisTexFormat> {
        self.tex_format
    }
    /// The plain payload format: the header's `VkFormat` as an imgio [`Format`], when the DFD
    /// uses that format's standard color model (also set for ASTC files from Basis).
    pub fn format(&self) -> Option<Format> {
        self.format
    }
    /// The header's `VkFormat` (0 for Basis-specific payloads).
    pub fn vk_format(&self) -> u32 {
        self.vk_format
    }
    /// The bytes of one image (`level`, array `layer`, cube `face`) of a plain payload,
    /// inflated if Zstandard supercompressed: `format().surface_bytes(level size)` bytes.
    ///
    /// Errors: `Unsupported` (no plain format), `InvalidArgument` (no such image),
    /// `InvalidData`.
    pub fn image_data(&self, level: u32, layer: u32, face: u32) -> Result<&[u8]> {
        let f = self.format.ok_or(Error::Unsupported)?;
        if level >= self.level_count() || layer >= self.layers || face >= self.faces {
            return Err(Error::InvalidArgument);
        }
        let (w, h) = self.level_dim(level);
        let size = f.surface_bytes(w, h)? as u64;
        let data = self.level_data(level)?;
        range(data, u64::from(layer * self.faces + face) * size, size)
    }
    /// Width of level 0 in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Height of level 0 in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Mip levels.
    pub fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }
    /// Array layers (at least 1).
    pub fn layer_count(&self) -> u32 {
        self.layers
    }
    /// Faces: 1, or 6 for cubemaps.
    pub fn face_count(&self) -> u32 {
        self.faces
    }
    /// True when the payload has alpha.
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }
    /// True when the DFD transfer function is sRGB.
    pub fn is_srgb(&self) -> bool {
        self.srgb
    }
    /// True for animations (`KTXanimData`, or ETC1S P-frames).
    pub fn is_video(&self) -> bool {
        self.video
    }

    /// Size of `level` in pixels.
    pub fn level_dim(&self, level: u32) -> (u32, u32) {
        crate::image::mip_dim(self.width, self.height, level)
    }

    /// True when this file can be transcoded to `format`.
    pub fn can_transcode_to(&self, format: Format) -> bool {
        self.tex_format.is_some_and(|t| can_transcode(t, format))
    }

    /// Bytes needed to transcode one image of `level` to `format` (tightly packed).
    pub fn transcoded_size(&self, level: u32, format: Format) -> Result<usize> {
        let (w, h) = self.level_dim(level);
        format.surface_bytes(w, h)
    }

    /// The level's payload bytes, inflated if Zstandard supercompressed.
    fn level_data(&self, level: u32) -> Result<&[u8]> {
        let l = self
            .levels
            .get(level as usize)
            .ok_or(Error::InvalidArgument)?;
        let raw = range(self.data, l.ofs, l.len)?;
        if self.scheme != SS_ZSTD {
            return Ok(raw);
        }
        let slot = self
            .inflated
            .get(level as usize)
            .ok_or(Error::InvalidArgument)?;
        slot.get_or_init(|| {
            let n = usize::try_from(l.uncompressed_len).ok()?;
            let mut buf = crate::buffer::zeroed(n).ok()?;
            let mut dec = ruzstd::decoding::FrameDecoder::new();
            match dec.decode_all(raw, &mut buf) {
                Ok(written) if written == n => Some(buf),
                _ => None,
            }
        })
        .as_ref()
        .map(|b| &b[..])
        .ok_or(Error::InvalidData)
    }

    /// Transcodes one image (`level`, array `layer`, cube `face`) to `format` into `dst`
    /// (tightly packed blocks or pixels).
    ///
    /// Errors: `Unsupported` (target not available for this payload), `BufferTooSmall`,
    /// `InvalidArgument` (no such level/layer/face), `InvalidData`.
    pub fn transcode(
        &self,
        level: u32,
        layer: u32,
        face: u32,
        format: Format,
        dst: &mut [u8],
    ) -> Result<()> {
        let opts = TranscodeOptions::default();
        self.transcode_with(level, layer, face, format, &opts, dst)
    }

    /// Like [`Ktx2File::transcode`], with transcoder options.
    pub fn transcode_with(
        &self,
        level: u32,
        layer: u32,
        face: u32,
        format: Format,
        opts: &TranscodeOptions,
        dst: &mut [u8],
    ) -> Result<()> {
        if level >= self.level_count() || layer >= self.layers || face >= self.faces {
            return Err(Error::InvalidArgument);
        }
        let (width, height) = self.level_dim(level);
        let tex = self.tex_format.ok_or(Error::Unsupported)?;
        let (bw, bh) = tex.block_dim();
        let (blocks_x, blocks_y) = (width.div_ceil(bw), height.div_ceil(bh));
        let image = (level * self.layers + layer) * self.faces + face;
        let l = self.levels[level as usize];
        let mut s = SliceRef {
            tex,
            data: self.data,
            ofs: 0,
            len: 0,
            alpha: (0, 0),
            width,
            height,
            blocks_x,
            blocks_y,
            level,
            has_alpha: self.has_alpha,
            video: self.video,
            srgb: self.astc_srgb,
            deblock: opts.deblock.resolve(self.deblock),
            opts: *opts,
        };
        match tex {
            BasisTexFormat::Etc1s
            | BasisTexFormat::UastcHdr6x6Intermediate
            | BasisTexFormat::XuastcLdr(_)
            | BasisTexFormat::Xubc7 => {
                let d = self.images.get(image as usize).ok_or(Error::InvalidData)?;
                s.ofs = l.ofs + u64::from(d.rgb.0);
                s.len = d.rgb.1;
                if d.alpha.1 != 0 {
                    s.alpha = (l.ofs + u64::from(d.alpha.0), d.alpha.1);
                }
            }
            _ => {
                let data = self.level_data(level)?;
                let size = u64::from(blocks_x) * u64::from(blocks_y) * 16;
                let ofs = u64::from(layer * self.faces + face) * size;
                range(data, ofs, size)?;
                s.data = data;
                s.ofs = ofs;
                s.len = u32::try_from(size).map_err(|_| Error::InvalidData)?;
            }
        }
        transcode_slice(&s, self.etc1s.as_ref(), format, dst)
    }

    /// Like [`Ktx2File::transcode`] but allocates the output.
    pub fn transcode_to_buffer(
        &self,
        level: u32,
        layer: u32,
        face: u32,
        format: Format,
    ) -> Result<Buffer> {
        let opts = TranscodeOptions::default();
        self.transcode_to_buffer_with(level, layer, face, format, &opts)
    }

    /// Transcodes every level, layer and face to `format`, all in parallel (level-major, then
    /// layer, then face; `TranscodedImage::image` is the layer). Fails if any image fails.
    pub fn transcode_all(
        &self,
        format: Format,
        opts: &TranscodeOptions,
    ) -> Result<vas::Vector<TranscodedImage, vas::storage::Dynamic>> {
        if !self.can_transcode_to(format) {
            return Err(Error::Unsupported);
        }
        let (layers, faces) = (self.layers, self.faces);
        let count = (self.level_count() * layers * faces) as usize;
        let keys = (0..self.level_count()).flat_map(move |level| {
            let (w, h) = self.level_dim(level);
            (0..layers * faces).map(move |i| Ok((i / faces, i % faces, level, w, h)))
        });
        let mut items = alloc_items(format, keys, count)?;
        let tex = self.tex_format.ok_or(Error::Unsupported)?;
        transcode_items(tex, format, &mut items, |t| {
            self.transcode_with(t.level, t.image, t.face, format, opts, &mut t.data)
        })?;
        Ok(items)
    }

    /// Like [`Ktx2File::transcode_with`] but allocates the output.
    pub fn transcode_to_buffer_with(
        &self,
        level: u32,
        layer: u32,
        face: u32,
        format: Format,
        opts: &TranscodeOptions,
    ) -> Result<Buffer> {
        let mut buf = crate::buffer::zeroed(self.transcoded_size(level, format)?)?;
        self.transcode_with(level, layer, face, format, opts, &mut buf)?;
        Ok(buf)
    }
}
