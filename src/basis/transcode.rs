//! Transcoding `.basis` image levels to GPU formats.

use super::file::{BasisFile, BasisTexFormat};
use super::ll::etc1s::{LowlevelEtc1sTranscoder, TranscodeStatus};
use super::ll::uastc_hdr::LowlevelUastcHdr4x4Transcoder;
use super::ll::uastc_hdr_6x6i::LowlevelUastcHdr6x6IntermediateTranscoder;
use super::ll::uastc_ldr::LowlevelUastcLdr4x4Transcoder;
use super::ll::{TranscoderState, TranscoderTextureFormat as T, basis_is_format_supported};
use crate::{AstcBlock, AstcKind, Buffer, Error, Format, Result};

/// The basisu target for `format` (sRGB variants share the unorm target: same bits).
fn target(format: Format) -> Option<T> {
    use Format::*;
    Some(match format {
        Etc1Rgb => T::ETC1_RGB,
        Etc2Rgba | Etc2RgbaSrgb => T::ETC2_RGBA,
        Bc1Unorm | Bc1Srgb => T::BC1_RGB,
        Bc3Unorm | Bc3Srgb => T::BC3_RGBA,
        Bc4Unorm => T::BC4_R,
        Bc5Unorm => T::BC5_RG,
        Bc7Unorm | Bc7Srgb => T::BC7_RGBA,
        Pvrtc1Rgb => T::PVRTC1_4_RGB,
        Pvrtc1Rgba => T::PVRTC1_4_RGBA,
        Astc(AstcBlock::B4x4, AstcKind::Unorm | AstcKind::Srgb) => T::ASTC_LDR_4x4_RGBA,
        AtcRgb => T::ATC_RGB,
        AtcRgba => T::ATC_RGBA,
        Rgba8Unorm | Rgba8Srgb => T::RGBA32,
        Rgb565 => T::RGB565,
        Bgr565 => T::BGR565,
        Rgba4444 => T::RGBA4444,
        Pvrtc2Rgb => T::PVRTC2_4_RGB,
        Pvrtc2Rgba => T::PVRTC2_4_RGBA,
        EacR11Unorm => T::ETC2_EAC_R11,
        EacRg11Unorm => T::ETC2_EAC_RG11,
        Bc6hUfloat => T::BC6H,
        Astc(AstcBlock::B4x4, AstcKind::Float) => T::ASTC_HDR_4x4_RGBA,
        Rgb16Float => T::RGB_HALF,
        Rgba16Float => T::RGBA_HALF,
        Rgb9e5Float => T::RGB_9E5,
        Astc(AstcBlock::B6x6, AstcKind::Float) => T::ASTC_HDR_6x6_RGBA,
        _ => return None,
    })
}

fn ll_format(f: BasisTexFormat) -> super::ll::BasisTexFormat {
    use super::ll::BasisTexFormat as L;
    match f {
        BasisTexFormat::Etc1s => L::Etc1s,
        BasisTexFormat::UastcLdr4x4 => L::UastcLdr4x4,
        BasisTexFormat::UastcHdr4x4 => L::UastcHdr4x4,
        BasisTexFormat::AstcHdr6x6 => L::AstcHdr6x6,
        BasisTexFormat::UastcHdr6x6Intermediate => L::UastcHdr6x6Intermediate,
    }
}

/// Decoded ETC1S codebooks and Huffman tables, shared by all slices of a file.
pub(crate) fn etc1s_transcoder(file: &BasisFile) -> Result<LowlevelEtc1sTranscoder> {
    let sec = |s: super::file::Section| &file.data[s.ofs as usize..(s.ofs + s.size) as usize];
    let mut t = LowlevelEtc1sTranscoder::new();
    if !t.decode_palettes(
        file.total_endpoints,
        sec(file.endpoint_cb),
        file.total_selectors,
        sec(file.selector_cb),
    ) || !t.decode_tables(sec(file.tables))
    {
        return Err(Error::InvalidData);
    }
    Ok(t)
}

fn status(s: TranscodeStatus) -> Result<()> {
    match s {
        TranscodeStatus::Ok => Ok(()),
        TranscodeStatus::Unsupported => Err(Error::Unsupported),
        TranscodeStatus::Failed => Err(Error::InvalidData),
    }
}

impl BasisFile<'_> {
    /// True when this file can be transcoded to `format`.
    pub fn can_transcode_to(&self, format: Format) -> bool {
        target(format).is_some_and(|t| basis_is_format_supported(t, ll_format(self.tex_format)))
            && self.tex_format != BasisTexFormat::AstcHdr6x6
    }

    /// Bytes needed to transcode `image`/`level` to `format` (tightly packed).
    pub fn transcoded_size(&self, image: u32, level: u32, format: Format) -> Result<usize> {
        let info = self.level_info(image, level)?;
        format.surface_bytes(info.width, info.height)
    }

    /// Transcodes `image`/`level` to `format` into `dst` (tightly packed blocks or pixels).
    ///
    /// PVRTC1 needs power-of-two sizes. Errors: `Unsupported` (target not available for this
    /// payload), `BufferTooSmall`, `InvalidArgument` (no such image/level), `InvalidData`.
    pub fn transcode(&self, image: u32, level: u32, format: Format, dst: &mut [u8]) -> Result<()> {
        if !self.can_transcode_to(format) {
            return Err(Error::Unsupported);
        }
        let t = target(format).ok_or(Error::Unsupported)?;
        let needed = self.transcoded_size(image, level, format)?;
        if dst.len() < needed {
            return Err(Error::BufferTooSmall {
                needed,
                got: dst.len(),
            });
        }
        let dst = &mut dst[..needed];
        let color = self
            .slice(image, level, false)
            .ok_or(Error::InvalidArgument)?;
        let units = if format.is_compressed() {
            needed / format.block_bytes()
        } else {
            (color.orig_width * color.orig_height) as usize
        } as u32;
        let mut state = TranscoderState::new();
        let video = self.tex_type == super::file::BasisTexType::VideoFrames;
        match self.tex_format {
            BasisTexFormat::Etc1s => {
                let etc1s = self
                    .etc1s
                    .get_or_init(|| etc1s_transcoder(self).ok())
                    .as_ref()
                    .ok_or(Error::InvalidData)?;
                let alpha = if self.has_alpha() {
                    let a = self.slice(image, level, true).ok_or(Error::InvalidData)?;
                    (a.file_ofs as u64, a.file_size)
                } else {
                    (0, 0)
                };
                status(etc1s.transcode_image(
                    t,
                    dst,
                    units,
                    self.data,
                    color.blocks_x,
                    color.blocks_y,
                    color.orig_width,
                    color.orig_height,
                    level,
                    color.file_ofs as u64,
                    color.file_size,
                    alpha.0,
                    alpha.1,
                    0,
                    self.has_alpha(),
                    video,
                    0,
                    &mut state,
                    0,
                ))
            }
            BasisTexFormat::UastcLdr4x4 => {
                status(LowlevelUastcLdr4x4Transcoder::new().transcode_image(
                    t,
                    dst,
                    units,
                    self.data,
                    color.blocks_x,
                    color.blocks_y,
                    color.orig_width,
                    color.orig_height,
                    level,
                    color.file_ofs as u64,
                    color.file_size,
                    0,
                    self.has_alpha(),
                    video,
                    0,
                    &mut state,
                    0,
                    -1,
                    -1,
                ))
            }
            BasisTexFormat::UastcHdr4x4 => {
                status(LowlevelUastcHdr4x4Transcoder::new().transcode_image(
                    t,
                    dst,
                    units,
                    self.data,
                    color.blocks_x,
                    color.blocks_y,
                    color.orig_width,
                    color.orig_height,
                    level,
                    color.file_ofs as u64,
                    color.file_size,
                    0,
                    self.has_alpha(),
                    video,
                    0,
                    &mut state,
                    0,
                    -1,
                    -1,
                ))
            }
            BasisTexFormat::UastcHdr6x6Intermediate => status(
                LowlevelUastcHdr6x6IntermediateTranscoder::new().transcode_image(
                    t,
                    dst,
                    units,
                    self.data,
                    color.blocks_x,
                    color.blocks_y,
                    color.orig_width,
                    color.orig_height,
                    level,
                    color.file_ofs as u64,
                    color.file_size,
                    0,
                    self.has_alpha(),
                    video,
                    0,
                    &mut state,
                    0,
                    -1,
                    -1,
                ),
            ),
            BasisTexFormat::AstcHdr6x6 => Err(Error::Unsupported),
        }
    }

    /// Like [`BasisFile::transcode`] but allocates the output.
    pub fn transcode_to_buffer(&self, image: u32, level: u32, format: Format) -> Result<Buffer> {
        let mut buf = crate::buffer::zeroed(self.transcoded_size(image, level, format)?)?;
        self.transcode(image, level, format, &mut buf)?;
        Ok(buf)
    }
}
