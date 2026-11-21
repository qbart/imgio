//! Container-independent transcoding of one Basis slice, plus the `.basis` front end.

use super::file::{BasisFile, BasisTexFormat};
use super::ll::astc_hdr_6x6::LowlevelAstcHdr6x6Transcoder;
use super::ll::etc1s::{LowlevelEtc1sTranscoder, TranscodeStatus};
use super::ll::uastc_hdr::LowlevelUastcHdr4x4Transcoder;
use super::ll::uastc_hdr_6x6i::LowlevelUastcHdr6x6IntermediateTranscoder;
use super::ll::uastc_ldr::LowlevelUastcLdr4x4Transcoder;
use super::ll::{TranscoderState, TranscoderTextureFormat as T, basis_is_format_supported};
use crate::parallel::try_for_each_unit;
use crate::{AstcBlock, AstcKind, Buffer, Error, Format, Result};

/// Deblocking of ASTC LDR / XUASTC payloads when transcoding to non-ASTC targets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Deblock {
    /// Container default: `.basis` deblocks block sizes of 80 texels (10x8) or more, KTX2
    /// follows the file's `DeblockFilterID` key.
    #[default]
    Auto,
    /// Never deblock (basisu `cDecodeFlagsNoDeblockFiltering`).
    Off,
    /// Always deblock (basisu `cDecodeFlagsForceDeblockFiltering`).
    Force,
}

impl Deblock {
    /// The decision for a payload whose container default is `auto`.
    pub(crate) fn resolve(self, auto: bool) -> bool {
        match self {
            Deblock::Auto => auto,
            Deblock::Off => false,
            Deblock::Force => true,
        }
    }
}

/// Transcoder options (basisu decode flags). `Default` matches basisu's defaults (flags 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscodeOptions {
    /// Slower, higher quality encoders where available: UASTC -> ETC/EAC/BC1/PVRTC, ASTC LDR /
    /// XUASTC / XUBC7 -> BC1/EAC/BC7 (partially analytical) / ASTC 4x4, HDR -> BC6H
    /// (`cDecodeFlagsHighQuality`).
    pub high_quality: bool,
    /// For files with alpha: transcode the alpha channel instead of color to opaque targets
    /// (ETC1, BC1 from ETC1S, BC4, EAC R11, PVRTC1 RGB, ...)
    /// (`cDecodeFlagsTranscodeAlphaDataToOpaqueFormats`).
    pub alpha_to_opaque: bool,
    /// Never emit BC1 3-color blocks (`cDecodeFlagsBC1ForbidThreeColorBlocks`).
    pub bc1_forbid_three_color: bool,
    /// ETC1S -> BC7 adaptive chroma filtering; off is much faster
    /// (inverse of `cDecodeFlagsNoETC1SChromaFiltering`).
    pub etc1s_chroma_filtering: bool,
    /// ASTC LDR / XUASTC deblocking (`cDecodeFlagsNoDeblockFiltering`,
    /// `cDecodeFlagsForceDeblockFiltering`).
    pub deblock: Deblock,
    /// XUASTC 4x4 / 6x6 / 8x6 -> BC7 direct fast paths (inverse of
    /// `cDecodeFlagXUASTCLDRDisableFastBC7Transcoding`).
    pub xuastc_fast_bc7: bool,
}

impl Default for TranscodeOptions {
    fn default() -> Self {
        TranscodeOptions {
            high_quality: false,
            alpha_to_opaque: false,
            bc1_forbid_three_color: false,
            etc1s_chroma_filtering: true,
            deblock: Deblock::Auto,
            xuastc_fast_bc7: true,
        }
    }
}

impl TranscodeOptions {
    /// The basisu `decode_flags` value.
    pub fn flags(&self) -> u32 {
        use super::ll::decode_flags as f;
        let mut v = 0;
        let mut set = |on: bool, bit: u32| {
            if on {
                v |= bit;
            }
        };
        set(self.high_quality, f::HIGH_QUALITY);
        set(
            self.alpha_to_opaque,
            f::TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS,
        );
        set(
            self.bc1_forbid_three_color,
            f::BC1_FORBID_THREE_COLOR_BLOCKS,
        );
        set(!self.etc1s_chroma_filtering, f::NO_ETC1S_CHROMA_FILTERING);
        set(self.deblock == Deblock::Off, f::NO_DEBLOCK_FILTERING);
        set(self.deblock == Deblock::Force, f::FORCE_DEBLOCK_FILTERING);
        set(
            !self.xuastc_fast_bc7,
            f::XUASTC_LDR_DISABLE_FAST_BC7_TRANSCODING,
        );
        v
    }

    /// Options from basisu `decode_flags` (unknown and internal bits ignored).
    pub fn from_flags(flags: u32) -> Self {
        use super::ll::decode_flags as f;
        let on = |bit: u32| flags & bit != 0;
        TranscodeOptions {
            high_quality: on(f::HIGH_QUALITY),
            alpha_to_opaque: on(f::TRANSCODE_ALPHA_DATA_TO_OPAQUE_FORMATS),
            bc1_forbid_three_color: on(f::BC1_FORBID_THREE_COLOR_BLOCKS),
            etc1s_chroma_filtering: !on(f::NO_ETC1S_CHROMA_FILTERING),
            deblock: if on(f::NO_DEBLOCK_FILTERING) {
                Deblock::Off
            } else if on(f::FORCE_DEBLOCK_FILTERING) {
                Deblock::Force
            } else {
                Deblock::Auto
            },
            xuastc_fast_bc7: !on(f::XUASTC_LDR_DISABLE_FAST_BC7_TRANSCODING),
        }
    }

    /// The 4x4 block encoder options for a payload with (container) alpha `has_alpha`.
    pub(crate) fn block(&self, has_alpha: bool) -> super::astc_ldr::BlockOpts {
        super::astc_ldr::BlockOpts {
            high_quality: self.high_quality,
            from_alpha: has_alpha && self.alpha_to_opaque,
            fast_bc7: self.xuastc_fast_bc7,
        }
    }
}

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
        Fxt1Rgb => T::FXT1_RGB,
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
    if let BasisTexFormat::AstcLdr(b) | BasisTexFormat::XuastcLdr(b) = tex {
        return super::astc_ldr::can_transcode(b, format);
    }
    if tex == BasisTexFormat::Xubc7 {
        return super::xubc7::can_transcode(format);
    }
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
    /// ASTC LDR / XUASTC: sRGB decode profile.
    pub srgb: bool,
    /// ASTC LDR / XUASTC: apply the deblocking filter for non-ASTC targets (options applied).
    pub deblock: bool,
    /// Transcoder options.
    pub opts: TranscodeOptions,
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
    let needed = format.surface_bytes(s.width, s.height)?;
    if dst.len() < needed {
        return Err(Error::BufferTooSmall {
            needed,
            got: dst.len(),
        });
    }
    let dst = &mut dst[..needed];
    match s.tex {
        BasisTexFormat::AstcLdr(block) => return transcode_astc_ldr(s, block, format, dst),
        BasisTexFormat::XuastcLdr(block) => return transcode_xuastc_ldr(s, block, format, dst),
        _ => {}
    }
    if s.tex == BasisTexFormat::Xubc7 {
        return transcode_xubc7(s, format, dst);
    }
    let t = target(format).ok_or(Error::Unsupported)?;
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
            s.opts.flags(),
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
                s.opts.flags(),
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
            t,
            dst,
            units,
            s.data,
            s.blocks_x,
            s.blocks_y,
            s.width,
            s.height,
            s.ofs,
            s.len,
            s.opts.flags(),
        )),
        BasisTexFormat::AstcLdr(_) | BasisTexFormat::XuastcLdr(_) | BasisTexFormat::Xubc7 => {
            Err(Error::Unsupported)
        }
    }
}

/// The slice's compressed bytes.
fn slice_data<'a>(s: &SliceRef<'a>) -> Result<&'a [u8]> {
    let start = usize::try_from(s.ofs).map_err(|_| Error::InvalidData)?;
    s.data
        .get(start..start.saturating_add(s.len as usize))
        .ok_or(Error::InvalidData)
}

/// Plain ASTC LDR blocks.
fn transcode_astc_ldr(
    s: &SliceRef,
    block: AstcBlock,
    format: Format,
    dst: &mut [u8],
) -> Result<()> {
    super::astc_ldr::transcode(
        &super::astc_ldr::AstcLdrImage {
            blocks: super::astc_ldr::Blocks::Physical(slice_data(s)?),
            block,
            blocks_x: s.blocks_x,
            blocks_y: s.blocks_y,
            width: s.width,
            height: s.height,
            srgb: s.srgb,
            deblock: s.deblock,
            has_alpha: true,
            opts: s.opts.block(s.has_alpha),
        },
        format,
        dst,
    )
}

/// XUASTC LDR: decompressed to logical ASTC blocks. The stream header's sRGB decode profile
/// and alpha flag are authoritative; its block size and block counts must match the container.
fn transcode_xuastc_ldr(
    s: &SliceRef,
    block: AstcBlock,
    format: Format,
    dst: &mut [u8],
) -> Result<()> {
    let img = super::xuastc::decompress(slice_data(s)?, Some((block, s.blocks_x, s.blocks_y)))?;
    super::astc_ldr::transcode(
        &super::astc_ldr::AstcLdrImage {
            blocks: super::astc_ldr::Blocks::Logical(&img.blocks),
            block,
            blocks_x: s.blocks_x,
            blocks_y: s.blocks_y,
            width: s.width,
            height: s.height,
            srgb: img.srgb,
            deblock: s.deblock,
            has_alpha: img.has_alpha,
            opts: s.opts.block(s.has_alpha),
        },
        format,
        dst,
    )
}

/// XUBC7 (supercompressed BC7).
fn transcode_xubc7(s: &SliceRef, format: Format, dst: &mut [u8]) -> Result<()> {
    let start = usize::try_from(s.ofs).map_err(|_| Error::InvalidData)?;
    let data = s
        .data
        .get(start..start.saturating_add(s.len as usize))
        .ok_or(Error::InvalidData)?;
    super::xubc7::transcode(
        &super::xubc7::Xubc7Image {
            data,
            blocks_x: s.blocks_x,
            blocks_y: s.blocks_y,
            width: s.width,
            height: s.height,
            has_alpha: s.has_alpha,
            opts: s.opts.block(s.has_alpha),
        },
        format,
        dst,
    )
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
        let (w, a, v, l, f) = (s.width, s.has_alpha, s.video, s.level, s.opts.flags());
        if hdr {
            LowlevelUastcHdr4x4Transcoder::new().transcode_image(
                t, out, units, s.data, bx, rows, w, height, l, ofs, len, f, a, v, 0, &mut state, 0,
                -1, -1,
            )
        } else {
            LowlevelUastcLdr4x4Transcoder::new().transcode_image(
                t, out, units, s.data, bx, rows, w, height, l, ofs, len, f, a, v, 0, &mut state, 0,
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
    let pixels = (s.width as usize).saturating_mul(s.height as usize);
    try_for_each_unit(dst, unit, pixels, |by, out| {
        let by = by as u32;
        let height = (s.height - by * 4).min(4);
        status(run(out, by, 1, height))
    })
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

    /// Transcodes `image`/`level` to `format` into `dst` (tightly packed blocks or pixels)
    /// with the default [`TranscodeOptions`].
    ///
    /// PVRTC1 needs power-of-two sizes. Errors: `Unsupported` (target not available for this
    /// payload), `BufferTooSmall`, `InvalidArgument` (no such image/level), `InvalidData`.
    pub fn transcode(&self, image: u32, level: u32, format: Format, dst: &mut [u8]) -> Result<()> {
        self.transcode_with(image, level, format, &TranscodeOptions::default(), dst)
    }

    /// Like [`BasisFile::transcode`], with transcoder options.
    pub fn transcode_with(
        &self,
        image: u32,
        level: u32,
        format: Format,
        opts: &TranscodeOptions,
        dst: &mut [u8],
    ) -> Result<()> {
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
            srgb: self.is_srgb(),
            // basisu default: deblock block sizes of 80 texels (10x8) or more.
            deblock: {
                let (bw, bh) = self.tex_format.block_dim();
                opts.deblock.resolve(bw * bh >= 80)
            },
            opts: *opts,
        };
        let e = if etc1s { Some(self.etc1s()?) } else { None };
        transcode_slice(&s, e, format, dst)
    }

    /// Like [`BasisFile::transcode`] but allocates the output.
    pub fn transcode_to_buffer(&self, image: u32, level: u32, format: Format) -> Result<Buffer> {
        self.transcode_to_buffer_with(image, level, format, &TranscodeOptions::default())
    }

    /// Like [`BasisFile::transcode_with`] but allocates the output.
    pub fn transcode_to_buffer_with(
        &self,
        image: u32,
        level: u32,
        format: Format,
        opts: &TranscodeOptions,
    ) -> Result<Buffer> {
        let mut buf = crate::buffer::zeroed(self.transcoded_size(image, level, format)?)?;
        self.transcode_with(image, level, format, opts, &mut buf)?;
        Ok(buf)
    }
}
