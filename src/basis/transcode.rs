//! Container-independent transcoding of one Basis slice, plus the `.basis` front end.

use super::file::{BasisFile, BasisTexFormat};
use super::ll::astc_hdr_6x6::LowlevelAstcHdr6x6Transcoder;
use super::ll::etc1s::{LowlevelEtc1sTranscoder, TranscodeStatus};
use super::ll::uastc_hdr::LowlevelUastcHdr4x4Transcoder;
use super::ll::uastc_hdr_6x6i::LowlevelUastcHdr6x6IntermediateTranscoder;
use super::ll::uastc_ldr::LowlevelUastcLdr4x4Transcoder;
use super::ll::{TranscoderState, TranscoderTextureFormat as T, basis_is_format_supported};
use crate::parallel::for_each_unit;
use crate::{AstcBlock, AstcKind, Buffer, Error, Format, Result};
use std::sync::Mutex;

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

fn ll_format(f: BasisTexFormat) -> Option<super::ll::BasisTexFormat> {
    use super::ll::BasisTexFormat as L;
    Some(match f {
        BasisTexFormat::Etc1s => L::Etc1s,
        BasisTexFormat::UastcLdr4x4 => L::UastcLdr4x4,
        BasisTexFormat::UastcHdr4x4 => L::UastcHdr4x4,
        BasisTexFormat::AstcHdr6x6 => L::AstcHdr6x6,
        BasisTexFormat::UastcHdr6x6Intermediate => L::UastcHdr6x6Intermediate,
        // Not transcodable yet.
        BasisTexFormat::XuastcLdr(_) | BasisTexFormat::AstcLdr(_) | BasisTexFormat::Xubc7 => {
            return None;
        }
    })
}

/// True when `tex` payloads can be transcoded to `format`.
pub(crate) fn can_transcode(tex: BasisTexFormat, format: Format) -> bool {
    match (target(format), ll_format(tex)) {
        (Some(t), Some(l)) => basis_is_format_supported(t, l),
        _ => false,
    }
}

/// Decodes ETC1S endpoint/selector codebooks and Huffman tables.
pub(crate) fn etc1s_transcoder(
    endpoints: (u32, &[u8]),
    selectors: (u32, &[u8]),
    tables: &[u8],
) -> Result<LowlevelEtc1sTranscoder> {
    let mut t = LowlevelEtc1sTranscoder::new();
    if !t.decode_palettes(endpoints.0, endpoints.1, selectors.0, selectors.1)
        || !t.decode_tables(tables)
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

/// One compressed image slice: offsets point into `data`.
#[derive(Clone, Copy)]
pub(crate) struct SliceRef<'a> {
    pub tex: BasisTexFormat,
    pub data: &'a [u8],
    pub ofs: u64,
    pub len: u32,
    /// ETC1S alpha slice (offset, length), `(0, 0)` when none.
    pub alpha: (u64, u32),
    pub width: u32,
    pub height: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub level: u32,
    pub has_alpha: bool,
    pub video: bool,
}

/// Transcodes one slice to `format` into `dst` (tightly packed). `etc1s` is required for ETC1S.
pub(crate) fn transcode_slice(
    s: &SliceRef,
    etc1s: Option<&LowlevelEtc1sTranscoder>,
    format: Format,
    dst: &mut [u8],
) -> Result<()> {
    if !can_transcode(s.tex, format) {
        return Err(Error::Unsupported);
    }
    let t = target(format).ok_or(Error::Unsupported)?;
    let needed = format.surface_bytes(s.width, s.height)?;
    if dst.len() < needed {
        return Err(Error::BufferTooSmall {
            needed,
            got: dst.len(),
        });
    }
    let dst = &mut dst[..needed];
    let units = if format.is_compressed() {
        needed / format.block_bytes()
    } else {
        (s.width as usize) * (s.height as usize)
    } as u32;
    let mut state = TranscoderState::new();
    match s.tex {
        BasisTexFormat::Etc1s => status(etc1s.ok_or(Error::InvalidData)?.transcode_image(
            t,
            dst,
            units,
            s.data,
            s.blocks_x,
            s.blocks_y,
            s.width,
            s.height,
            s.level,
            s.ofs,
            s.len,
            s.alpha.0,
            s.alpha.1,
            0,
            s.has_alpha,
            s.video,
            0,
            &mut state,
            0,
        )),
        BasisTexFormat::UastcLdr4x4 | BasisTexFormat::UastcHdr4x4 => {
            transcode_uastc_4x4(s, t, format, dst)
        }
        BasisTexFormat::UastcHdr6x6Intermediate => status(
            LowlevelUastcHdr6x6IntermediateTranscoder::new().transcode_image(
                t,
                dst,
                units,
                s.data,
                s.blocks_x,
                s.blocks_y,
                s.width,
                s.height,
                s.level,
                s.ofs,
                s.len,
                0,
                s.has_alpha,
                s.video,
                0,
                &mut state,
                0,
                -1,
                -1,
            ),
        ),
        BasisTexFormat::AstcHdr6x6 => status(LowlevelAstcHdr6x6Transcoder::new().transcode_image(
            t, dst, units, s.data, s.blocks_x, s.blocks_y, s.width, s.height, s.ofs, s.len, 0,
        )),
        BasisTexFormat::XuastcLdr(_) | BasisTexFormat::AstcLdr(_) | BasisTexFormat::Xubc7 => {
            Err(Error::Unsupported)
        }
    }
}

/// UASTC 4x4 (LDR or HDR): blocks are independent, so rows of blocks are transcoded in
/// parallel (except PVRTC1, which needs the whole image).
fn transcode_uastc_4x4(s: &SliceRef, t: T, format: Format, dst: &mut [u8]) -> Result<()> {
    let hdr = s.tex == BasisTexFormat::UastcHdr4x4;
    let run = |out: &mut [u8], first_row: u32, rows: u32, height: u32| -> TranscodeStatus {
        let bx = s.blocks_x;
        let ofs = s.ofs + u64::from(first_row) * u64::from(bx) * 16;
        let len = bx * rows * 16;
        let units = if format.is_compressed() {
            bx * rows
        } else {
            s.width * height
        };
        let mut state = TranscoderState::new();
        let (w, a, v, l) = (s.width, s.has_alpha, s.video, s.level);
        if hdr {
            LowlevelUastcHdr4x4Transcoder::new().transcode_image(
                t, out, units, s.data, bx, rows, w, height, l, ofs, len, 0, a, v, 0, &mut state, 0,
                -1, -1,
            )
        } else {
            LowlevelUastcLdr4x4Transcoder::new().transcode_image(
                t, out, units, s.data, bx, rows, w, height, l, ofs, len, 0, a, v, 0, &mut state, 0,
                -1, -1,
            )
        }
    };
    if (s.len as u64) < u64::from(s.blocks_x) * u64::from(s.blocks_y) * 16 {
        return Err(Error::InvalidData);
    }
    if matches!(format, Format::Pvrtc1Rgb | Format::Pvrtc1Rgba) {
        return status(run(dst, 0, s.blocks_y, s.height));
    }
    // One unit = one row of 4x4 source blocks.
    let row = format.row_bytes(s.width)?;
    let unit = if format.is_compressed() { row } else { row * 4 };
    let first_error = Mutex::new(None);
    let pixels = (s.width as usize).saturating_mul(s.height as usize);
    for_each_unit(dst, unit, pixels, |by, out| {
        let by = by as u32;
        let height = (s.height - by * 4).min(4);
        if let Err(e) = status(run(out, by, 1, height))
            && let Ok(mut slot) = first_error.lock()
        {
            slot.get_or_insert(e);
        }
    });
    match first_error.into_inner() {
        Ok(Some(e)) => Err(e),
        _ => Ok(()),
    }
}

impl BasisFile<'_> {
    /// True when this file can be transcoded to `format`.
    pub fn can_transcode_to(&self, format: Format) -> bool {
        can_transcode(self.tex_format, format)
    }

    /// Bytes needed to transcode `image`/`level` to `format` (tightly packed).
    pub fn transcoded_size(&self, image: u32, level: u32, format: Format) -> Result<usize> {
        let info = self.level_info(image, level)?;
        format.surface_bytes(info.width, info.height)
    }

    fn etc1s(&self) -> Result<&LowlevelEtc1sTranscoder> {
        self.etc1s
            .get_or_init(|| {
                let sec = |s: super::file::Section| {
                    self.data
                        .get(s.ofs as usize..(s.ofs as usize).saturating_add(s.size as usize))
                        .unwrap_or(&[])
                };
                etc1s_transcoder(
                    (self.total_endpoints, sec(self.endpoint_cb)),
                    (self.total_selectors, sec(self.selector_cb)),
                    sec(self.tables),
                )
                .ok()
            })
            .as_ref()
            .ok_or(Error::InvalidData)
    }

    /// Transcodes `image`/`level` to `format` into `dst` (tightly packed blocks or pixels).
    ///
    /// PVRTC1 needs power-of-two sizes. Errors: `Unsupported` (target not available for this
    /// payload), `BufferTooSmall`, `InvalidArgument` (no such image/level), `InvalidData`.
    pub fn transcode(&self, image: u32, level: u32, format: Format, dst: &mut [u8]) -> Result<()> {
        if !self.can_transcode_to(format) {
            return Err(Error::Unsupported);
        }
        let c = self
            .slice(image, level, false)
            .ok_or(Error::InvalidArgument)?;
        let etc1s = self.tex_format == BasisTexFormat::Etc1s;
        let alpha = if etc1s && self.has_alpha() {
            let a = self.slice(image, level, true).ok_or(Error::InvalidData)?;
            (a.file_ofs as u64, a.file_size)
        } else {
            (0, 0)
        };
        let s = SliceRef {
            tex: self.tex_format,
            data: self.data,
            ofs: c.file_ofs as u64,
            len: c.file_size,
            alpha,
            width: c.orig_width,
            height: c.orig_height,
            blocks_x: c.blocks_x,
            blocks_y: c.blocks_y,
            level,
            has_alpha: self.has_alpha(),
            video: self.tex_type == super::file::BasisTexType::VideoFrames,
        };
        let e = if etc1s { Some(self.etc1s()?) } else { None };
        transcode_slice(&s, e, format, dst)
    }

    /// Like [`BasisFile::transcode`] but allocates the output.
    pub fn transcode_to_buffer(&self, image: u32, level: u32, format: Format) -> Result<Buffer> {
        let mut buf = crate::buffer::zeroed(self.transcoded_size(image, level, format)?)?;
        self.transcode(image, level, format, &mut buf)?;
        Ok(buf)
    }
}
