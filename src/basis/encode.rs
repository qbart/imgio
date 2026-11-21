//! Basis Universal encoding: images to a `.basis` file.
//!
//! Drives the ported `basis_compressor` (`ll::comp`, bit-exact with the C++ encoder) for
//! ETC1S, UASTC LDR 4x4 (optionally RDO), UASTC HDR 4x4, ASTC HDR 6x6, UASTC HDR 6x6
//! intermediate, ASTC LDR and XUASTC LDR (4x4..12x12) and XUBC7. Defaults follow the `basisu`
//! command line tool.

use super::file::{BasisTexFormat, BasisTexType};
use super::ll::comp::{
    BasisCompressor, BasisCompressorErrorCode, BasisCompressorParams, HdrModes, Image, ImageF,
    XuastcLdrDeblockingMode,
};
use super::ll::frontend::{
    C_BASIS_TEX_TYPE_2D, C_BASIS_TEX_TYPE_2D_ARRAY, C_BASIS_TEX_TYPE_CUBEMAP_ARRAY,
    C_BASIS_TEX_TYPE_VIDEO_FRAMES, C_BASIS_TEX_TYPE_VOLUME,
};
use super::xuastc::enc::XuastcSyntax as EncSyntax;
use super::xubc7::encoder::Bc7Encoder;
use crate::convert::{ConvertOptions, convert_to_buffer, convert_with};
use crate::mip::Filter;
use crate::{AstcBlock, Buffer, Error, Format, Result, Surface};

/// Codec of an encoded `.basis` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BasisCodec {
    /// ETC1S (BasisLZ): small, lower quality LDR.
    #[default]
    Etc1s,
    /// UASTC LDR 4x4: high quality LDR.
    UastcLdr4x4,
    /// UASTC HDR 4x4 (ASTC HDR 4x4 subset, transcodes to BC6H quickly).
    UastcHdr4x4,
    /// ASTC HDR 6x6 (optionally RDO).
    AstcHdr6x6,
    /// UASTC HDR 6x6 intermediate (compressed ASTC HDR 6x6).
    UastcHdr6x6Intermediate,
    /// Plain ASTC LDR blocks of the given size (`-ldr_4x4` .. `-ldr_12x12`; basisu v2).
    AstcLdr(AstcBlock),
    /// XUASTC LDR: supercompressed ASTC LDR of the given block size (`-ldr_4x4i` ..
    /// `-ldr_12x12i`; basisu v2). See the `xuastc_*` options.
    XuastcLdr(AstcBlock),
    /// XUBC7: supercompressed BC7 (`-xubc7`; basisu v2.50). See the `xubc7_*` options.
    Xubc7,
}

impl BasisCodec {
    /// True for the HDR codecs (which take float input and ignore alpha).
    pub fn is_hdr(self) -> bool {
        !matches!(
            self,
            BasisCodec::Etc1s
                | BasisCodec::UastcLdr4x4
                | BasisCodec::AstcLdr(_)
                | BasisCodec::XuastcLdr(_)
                | BasisCodec::Xubc7
        )
    }

    /// The `.basis` payload format this codec writes.
    pub fn tex_format(self) -> BasisTexFormat {
        match self {
            BasisCodec::Etc1s => BasisTexFormat::Etc1s,
            BasisCodec::UastcLdr4x4 => BasisTexFormat::UastcLdr4x4,
            BasisCodec::UastcHdr4x4 => BasisTexFormat::UastcHdr4x4,
            BasisCodec::AstcHdr6x6 => BasisTexFormat::AstcHdr6x6,
            BasisCodec::UastcHdr6x6Intermediate => BasisTexFormat::UastcHdr6x6Intermediate,
            BasisCodec::AstcLdr(b) => BasisTexFormat::AstcLdr(b),
            BasisCodec::XuastcLdr(b) => BasisTexFormat::XuastcLdr(b),
            BasisCodec::Xubc7 => BasisTexFormat::Xubc7,
        }
    }
}

/// XUASTC LDR entropy coding syntax (basisu `xuastc_ldr_syntax`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum XuastcSyntax {
    /// Zstandard for everything (`-xuastc_zstd`, default): fastest transcoding, larger files.
    #[default]
    Zstd,
    /// Adaptive arithmetic coding (`-xuastc_arith`): smallest files, slowest transcoding.
    Arith,
    /// Arithmetic coded headers, Zstandard coded weights (`-xuastc_hybrid`).
    Hybrid,
}

/// ASTC LDR / XUASTC LDR deblocking (basisu `xuastc_ldr_deblocking_mode`). "SCD" makes the
/// encoder optimize the blocks for the deblocking filter; "filter" writes the `DeblockFilterID`
/// key that asks decoders to deblock (on transcoding to non-ASTC formats). The "largest" modes
/// apply to block sizes of 80 texels (10x8) or more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum XuastcDeblocking {
    /// No SCD, no filter (`-xuastc_ldr_no_deblocking`).
    Off,
    /// SCD + filter on the largest block sizes (default, `-xuastc_ldr_deblocking_largest`).
    #[default]
    ScdFilterLargest,
    /// SCD + filter on all block sizes (`-xuastc_ldr_deblocking_all`).
    ScdFilterAll,
    /// SCD on all block sizes, no filter (`-xuastc_ldr_deblocking_scd_no_filtering`).
    ScdOnly,
    /// Filter on the largest block sizes, no SCD
    /// (`-xuastc_ldr_deblocking_no_scd_filtering_largest`).
    FilterLargest,
    /// Filter on all block sizes, no SCD (`-xuastc_ldr_deblocking_no_scd_filtering_all`).
    FilterAll,
}

/// XUBC7 BC7 base encoder (basisu `xbc7::bc7_encoder_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Xubc7Encoder {
    /// Real-time analytical bc7f (`-xubc7_bc7f`, default).
    #[default]
    Bc7f,
    /// bc7e_scalar at level 0 (fastest)..=6 (slowest) (`-xubc7_bc7e_scalar_level L`;
    /// `-xubc7_bc7e` = level 2): higher quality, slower.
    Bc7e(u32),
}

/// Options of [`encode_basis`]. [`Default`] matches the `basisu` tool defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasisEncodeOptions {
    pub codec: BasisCodec,
    /// How the images relate (`basis_texture_type`). Every type but `Tex2D` needs the same
    /// size for all images; `CubemapArray` a multiple of 6 images.
    pub tex_type: BasisTexType,
    /// ETC1S quality 1..=255 (`-q`, default 128): sets the codebook sizes.
    pub etc1s_quality: u32,
    /// ETC1S effort 0..=6 (`-comp_level`, default 1).
    pub etc1s_level: u32,
    /// UASTC LDR 4x4 pack level 0 (fastest)..=4 (slowest) (`-uastc_level`, default 2).
    pub uastc_level: u32,
    /// UASTC LDR 4x4 RDO lambda (`-uastc_rdo_l`, e.g. 0.25..10); 0 disables RDO (default).
    pub uastc_rdo_lambda: f32,
    /// UASTC HDR 4x4 level 0..=4 (`-uastc_level` in HDR mode, default 1).
    pub hdr_4x4_level: u32,
    /// ASTC HDR 6x6 / UASTC HDR 6x6 intermediate level 0..=12 (`-hdr_6x6_level`, default 2).
    pub hdr_6x6_level: u32,
    /// ASTC HDR 6x6 / UASTC HDR 6x6 intermediate RDO lambda (`-lambda`); 0 = off (default).
    pub hdr_6x6_lambda: f32,
    /// sRGB content: perceptual error metrics, sRGB-space mipmap filtering and the `.basis`
    /// sRGB flag (default true; `-linear` clears it). HDR codecs only use the error metrics.
    pub srgb: bool,
    /// Generate a full mip chain per image (`-mipmap`).
    pub mipmaps: bool,
    /// Mipmap filter for LDR codecs (`-mip_filter`, default kaiser); HDR mips use box.
    pub mip_filter: Filter,
    /// Wrap at the edges while filtering mipmaps (default true; `-mip_clamp` clears it).
    pub mip_wrap: bool,
    /// Flip the images vertically before encoding (`-y_flip`); sets the file's y-flip flag.
    pub y_flip: bool,
    /// Worker threads; 0 = all available. As in basisu, ETC1S codebook generation and UASTC
    /// RDO (up to 4 jobs) split their work by thread count, so those outputs depend on it.
    pub threads: u32,
    /// ASTC LDR / XUASTC LDR effort 0 (fastest)..=10 (`-effort`, default 3).
    pub astc_ldr_effort: u32,
    /// XUASTC LDR quality 1..=100 (`-quality`, default 100). Below 100 the weight grids are DCT
    /// coded at this quality and lossy supercompression is enabled (smaller, lossier); 100 is
    /// lossless supercompression of the ASTC blocks. ASTC LDR ignores it.
    pub xuastc_quality: u32,
    /// Overrides the lossy supercompression choice of `xuastc_quality` (`-xy` / `-xyd`).
    pub xuastc_lossy_supercompression: Option<bool>,
    /// XUASTC LDR bitstream syntax (default [`XuastcSyntax::Zstd`]).
    pub xuastc_syntax: XuastcSyntax,
    /// ASTC LDR / XUASTC LDR deblocking (default [`XuastcDeblocking::ScdFilterLargest`]).
    pub xuastc_deblocking: XuastcDeblocking,
    /// ASTC LDR / XUASTC LDR / XUBC7 channel error weights, each 1..=256 (`-weights`;
    /// `astc_ldr::compress_image` rejects larger); `None` = by `srgb`: 9, 11, 1, 11 (sRGB) or
    /// 1, 1, 1, 1 (linear), like `-srgb` / `-linear`.
    pub ldr_channel_weights: Option<[u32; 4]>,
    /// XUBC7 quality 1..=100 (`-quality`, default 100 = lossless BC7 weights). Below 100 the
    /// weights are DCT coded at this quality (smaller, lossier).
    pub xubc7_quality: u32,
    /// XUBC7 effort 0 (fastest)..=10 (`-effort`, default 3).
    pub xubc7_effort: u32,
    /// XUBC7 RDO level 0..=100 (`-xubc7_rdo`, default 0 = off): block reuse / endpoint / weight
    /// RDO trading quality for size.
    pub xubc7_rdo_level: u32,
    /// XUBC7 BC7 base encoder (default [`Xubc7Encoder::Bc7f`]).
    pub xubc7_encoder: Xubc7Encoder,
    /// XUBC7 desired stripe count 1..=16 (`-xubc7_stripes`, default 8): transcoding parallelism
    /// vs size. Images under 128 rows use one stripe, others at least 16 block rows per stripe.
    pub xubc7_stripes: u32,
    /// Write the key/value header extension of basisu v2.x (`BasisULibVers` = "2.50", plus
    /// `KTXmapRange` for HDR codecs) as the `basisu` 2.50 tool does (default true). false = the
    /// pre-2.0 layout (KTX-Software / libktx `basis_compressor`) for ETC1S / UASTC / HDR. Keys
    /// that change decoding (XUASTC / ASTC LDR `DeblockFilterID`) are written either way.
    pub basisu_v2_keys: bool,
}

impl Default for BasisEncodeOptions {
    fn default() -> Self {
        BasisEncodeOptions {
            codec: BasisCodec::Etc1s,
            tex_type: BasisTexType::Tex2D,
            etc1s_quality: 128,
            etc1s_level: 1,
            uastc_level: 2,
            uastc_rdo_lambda: 0.0,
            hdr_4x4_level: 1,
            hdr_6x6_level: 2,
            hdr_6x6_lambda: 0.0,
            srgb: true,
            mipmaps: false,
            mip_filter: Filter::Kaiser,
            mip_wrap: true,
            y_flip: false,
            threads: 0,
            astc_ldr_effort: 3,
            xuastc_quality: 100,
            xuastc_lossy_supercompression: None,
            xuastc_syntax: XuastcSyntax::Zstd,
            xuastc_deblocking: XuastcDeblocking::ScdFilterLargest,
            ldr_channel_weights: None,
            xubc7_quality: 100,
            xubc7_effort: 3,
            xubc7_rdo_level: 0,
            xubc7_encoder: Xubc7Encoder::Bc7f,
            xubc7_stripes: 8,
            basisu_v2_keys: true,
        }
    }
}

/// Encodes `images` into a `.basis` file.
///
/// LDR codecs read RGBA8 (other formats are converted, block-compressed ones decoded first;
/// fully opaque inputs produce no alpha). HDR codecs read RGB as linear float (sRGB 8-bit
/// inputs are linearized) and ignore alpha; values above the ASTC HDR range are rescaled,
/// negative / NaN / Inf ones zeroed.
pub fn encode_basis(images: &[Surface], opts: &BasisEncodeOptions) -> Result<Buffer> {
    encode_with(images, opts, |_| {})
}

/// [`encode_basis`] with `prepare` called on the compressor before `process` (test hooks).
fn encode_with(
    images: &[Surface],
    opts: &BasisEncodeOptions,
    prepare: impl FnOnce(&mut BasisCompressor),
) -> Result<Buffer> {
    if images.is_empty() {
        return Err(Error::InvalidArgument);
    }
    let mut p = compressor_params(opts)?;
    for s in images {
        if s.width() == 0 || s.height() == 0 {
            return Err(Error::InvalidArgument);
        }
        if opts.codec.is_hdr() {
            p.source_images_hdr.push(to_image_f(s)?);
        } else {
            p.source_images.push(to_image(s)?);
        }
    }
    let mut comp = BasisCompressor::new();
    if !comp.init(&p) {
        return Err(Error::InvalidArgument);
    }
    prepare(&mut comp);
    use BasisCompressorErrorCode as Ec;
    match comp.process() {
        Ec::Success => {}
        Ec::FailedReadingSourceImages | Ec::FailedValidating | Ec::FailedInvalidParameters => {
            return Err(Error::InvalidArgument);
        }
        _ => return Err(comp.take_encode_error().unwrap_or(Error::InvalidData)),
    }
    let out = comp.get_output_basis_file();
    let mut buf = crate::buffer::zeroed(out.len())?;
    buf.copy_from_slice(out);
    Ok(buf)
}

/// `basis_compressor_params` for `opts` (no source images).
fn compressor_params(opts: &BasisEncodeOptions) -> Result<BasisCompressorParams> {
    let threads = match opts.threads {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get() as u32),
        n => n,
    };
    let mut p = BasisCompressorParams {
        status_output: false,
        job_pool_threads: threads,
        tex_type: match opts.tex_type {
            BasisTexType::Tex2D => C_BASIS_TEX_TYPE_2D,
            BasisTexType::Tex2DArray => C_BASIS_TEX_TYPE_2D_ARRAY,
            BasisTexType::CubemapArray => C_BASIS_TEX_TYPE_CUBEMAP_ARRAY,
            BasisTexType::VideoFrames => C_BASIS_TEX_TYPE_VIDEO_FRAMES,
            BasisTexType::Volume => C_BASIS_TEX_TYPE_VOLUME,
        },
        y_flip: opts.y_flip,
        mip_gen: opts.mipmaps,
        mip_filter: opts.mip_filter.name(),
        mip_wrapping: opts.mip_wrap,
        add_basisu_v2_keys: opts.basisu_v2_keys,
        ..Default::default()
    };
    p.set_srgb_options(opts.srgb);
    match opts.codec {
        BasisCodec::Etc1s => {
            if !(1..=255).contains(&opts.etc1s_quality) || opts.etc1s_level > 6 {
                return Err(Error::InvalidArgument);
            }
            p.quality_level = opts.etc1s_quality as i32;
            p.etc1s_compression_level.set(opts.etc1s_level as i32);
        }
        BasisCodec::UastcLdr4x4 => {
            if opts.uastc_level > 4 || opts.uastc_rdo_lambda.is_nan() || opts.uastc_rdo_lambda < 0.0
            {
                return Err(Error::InvalidArgument);
            }
            p.uastc = true;
            p.pack_uastc_ldr_4x4_flags = opts.uastc_level;
            if opts.uastc_rdo_lambda > 0.0 {
                p.rdo_uastc_ldr_4x4 = true;
                p.rdo_uastc_ldr_4x4_quality_scalar
                    .set(opts.uastc_rdo_lambda);
            }
        }
        BasisCodec::UastcHdr4x4 => {
            if opts.hdr_4x4_level > 4 {
                return Err(Error::InvalidArgument);
            }
            p.uastc = true;
            p.hdr = true;
            p.hdr_mode = HdrModes::UastcHdr4x4;
            p.uastc_hdr_4x4_options
                .set_quality_level(opts.hdr_4x4_level as i32);
        }
        BasisCodec::AstcHdr6x6 | BasisCodec::UastcHdr6x6Intermediate => {
            if opts.hdr_6x6_level > 12 || opts.hdr_6x6_lambda.is_nan() || opts.hdr_6x6_lambda < 0.0
            {
                return Err(Error::InvalidArgument);
            }
            p.uastc = true;
            p.hdr = true;
            p.hdr_mode = if opts.codec == BasisCodec::AstcHdr6x6 {
                HdrModes::AstcHdr6x6
            } else {
                HdrModes::UastcHdr6x6Intermediate
            };
            p.astc_hdr_6x6_options
                .set_user_level(opts.hdr_6x6_level as i32);
            p.astc_hdr_6x6_options.lambda = opts.hdr_6x6_lambda;
        }
        BasisCodec::AstcLdr(_) | BasisCodec::XuastcLdr(_) => set_astc_ldr_params(&mut p, opts)?,
        BasisCodec::Xubc7 => set_xubc7_params(&mut p, opts)?,
    }
    Ok(p)
}

/// The XUBC7 part of [`compressor_params`], as the `basisu` tool sets it for `-xubc7 [-srgb |
/// -linear] [-weights ..] [-xubc7_rdo R] [-xubc7_stripes S] [-xubc7_bc7f | -xubc7_bc7e |
/// -xubc7_bc7e_scalar_level L] -effort E -quality Q`.
fn set_xubc7_params(p: &mut BasisCompressorParams, opts: &BasisEncodeOptions) -> Result<()> {
    let bc7e_level = match opts.xubc7_encoder {
        Xubc7Encoder::Bc7f => None,
        Xubc7Encoder::Bc7e(l) => Some(l),
    };
    if !(1..=100).contains(&opts.xubc7_quality)
        || opts.xubc7_effort > 10
        || opts.xubc7_rdo_level > 100
        || !(1..=16).contains(&opts.xubc7_stripes)
        || bc7e_level.is_some_and(|l| l > 6)
    {
        return Err(Error::InvalidArgument);
    }
    if let Some(w) = opts.ldr_channel_weights
        && w.iter().any(|&c| !(1..=256).contains(&c))
    {
        return Err(Error::InvalidArgument);
    }
    p.set_format_mode(BasisTexFormat::Xubc7);
    p.set_ldr_srgb_channel_weights(opts.srgb);
    if let Some(w) = opts.ldr_channel_weights {
        p.ldr_channel_weights = w;
    }
    p.xubc7_rdo_level.set(opts.xubc7_rdo_level as i32);
    p.xubc7_num_stripes.set(opts.xubc7_stripes as i32);
    match bc7e_level {
        None => p.xubc7_encoder = Bc7Encoder::Bc7f,
        Some(l) => {
            p.xubc7_encoder = Bc7Encoder::Bc7eScalar;
            p.xubc7_bc7e_scalar_level.set(l as i32);
        }
    }
    if !p.set_format_mode_and_quality_effort(
        BasisTexFormat::Xubc7,
        opts.xubc7_quality as i32,
        opts.xubc7_effort as i32,
        false,
    ) {
        return Err(Error::InvalidArgument);
    }
    Ok(())
}

/// The ASTC LDR / XUASTC LDR part of [`compressor_params`], as the `basisu` tool sets it for
/// `-ldr_WxH[i] [-srgb | -linear] [-weights ..] -effort E -quality Q [-xy | -xyd]` plus the
/// syntax and deblocking options.
fn set_astc_ldr_params(p: &mut BasisCompressorParams, opts: &BasisEncodeOptions) -> Result<()> {
    if opts.astc_ldr_effort > 10 || !(1..=100).contains(&opts.xuastc_quality) {
        return Err(Error::InvalidArgument);
    }
    if let Some(w) = opts.ldr_channel_weights
        && w.iter().any(|&c| !(1..=256).contains(&c))
    {
        return Err(Error::InvalidArgument);
    }
    let mode = opts.codec.tex_format();
    p.set_format_mode(mode);
    p.set_ldr_srgb_channel_weights(opts.srgb);
    if let Some(w) = opts.ldr_channel_weights {
        p.ldr_channel_weights = w;
    }
    p.xuastc_ldr_syntax = match opts.xuastc_syntax {
        XuastcSyntax::Zstd => EncSyntax::FullZstd,
        XuastcSyntax::Arith => EncSyntax::FullArith,
        XuastcSyntax::Hybrid => EncSyntax::HybridArithZstd,
    };
    p.xuastc_ldr_deblocking_mode = match opts.xuastc_deblocking {
        XuastcDeblocking::Off => XuastcLdrDeblockingMode::Disabled,
        XuastcDeblocking::ScdFilterLargest => {
            XuastcLdrDeblockingMode::UseScdAndFilteringOnlyLargestBlocks
        }
        XuastcDeblocking::ScdFilterAll => XuastcLdrDeblockingMode::UseScdAndFilteringAllBlockSizes,
        XuastcDeblocking::ScdOnly => XuastcLdrDeblockingMode::UseScdNoFiltering,
        XuastcDeblocking::FilterLargest => {
            XuastcLdrDeblockingMode::NoScdButEnableFilteringOnLargestBlocks
        }
        XuastcDeblocking::FilterAll => XuastcLdrDeblockingMode::NoScdButEnableFilteringOnAllBlocks,
    };
    if !p.set_format_mode_and_quality_effort(
        mode,
        opts.xuastc_quality as i32,
        opts.astc_ldr_effort as i32,
        false,
    ) {
        return Err(Error::InvalidArgument);
    }
    if let Some(lossy) = opts.xuastc_lossy_supercompression {
        p.xuastc_ldr_use_lossy_supercompression = lossy;
    }
    Ok(())
}

/// `s` as uncompressed pixels (block-compressed surfaces are decoded) in its own format.
fn uncompressed<'a>(s: &Surface<'a>, keep: &'a mut Buffer) -> Result<Surface<'a>> {
    let f = s.format();
    if !f.is_compressed() {
        return Ok(*s);
    }
    *keep = crate::decode_to_buffer(f, s.data(), s.width(), s.height())?;
    Surface::new(f.decoded(), s.width(), s.height(), keep)
}

/// RGBA8 `basisu::image` of `s` (channel values as stored; no sRGB conversion).
fn to_image(s: &Surface) -> Result<Image> {
    let mut keep = Buffer::default();
    let s = uncompressed(s, &mut keep)?;
    let conv;
    let data = if matches!(s.format(), Format::Rgba8Unorm | Format::Rgba8Srgb) {
        s.data()
    } else {
        conv = convert_to_buffer(&s, Format::Rgba8Unorm)?;
        &conv[..]
    };
    let mut img = Image::with_size(s.width(), s.height());
    for (d, px) in img.get_pixels_mut().iter_mut().zip(data.as_chunks::<4>().0) {
        *d = *px;
    }
    Ok(img)
}

/// Linear float `basisu::imagef` of `s`.
fn to_image_f(s: &Surface) -> Result<ImageF> {
    let mut keep = Buffer::default();
    let s = uncompressed(s, &mut keep)?;
    let len = Format::Rgba32Float.surface_bytes(s.width(), s.height())?;
    let mut conv = crate::buffer::zeroed(len)?;
    let linearize = ConvertOptions {
        srgb_transfer: true,
    };
    convert_with(&s, Format::Rgba32Float, &linearize, &mut conv)?;
    let mut img = ImageF::with_size(s.width(), s.height());
    for (d, px) in img
        .get_pixels_mut()
        .iter_mut()
        .zip(conv.as_chunks::<16>().0)
    {
        for (c, v) in d.iter_mut().zip(px.as_chunks::<4>().0) {
            *c = f32::from_le_bytes(*v);
        }
    }
    Ok(img)
}

#[cfg(test)]
mod tests {
    //! Everything around the XUASTC / ASTC LDR block encoder, checked against the basisu 2.50
    //! CLI goldens (testdata/basisu_enc/cli_{astc,xuastc}*.basis): the goldens' own encoder
    //! outputs are injected in place of `astc_ldr::compress_image`, the rest (source images,
    //! mipmaps, alpha detection, config / key-value mapping, ASTC packing, `.basis` writer) runs
    //! as is and must reproduce the files byte for byte.
    use super::*;
    use crate::astc::helpers::{LogAstcBlock, unpack_block};
    use crate::basis::BasisFile;
    use crate::basis::ll::comp::BASISU_DEBLOCKING_BLOCK_SIZE_THRESHOLD;
    use crate::basis::xuastc::enc::CompressOutput;
    use crate::{AstcKind, Error};

    type R<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    /// (golden, raw image, options) of `scripts/gen_basis_testdata.sh` (`-basis ...`).
    pub(crate) fn goldens() -> Vec<(&'static str, &'static str, BasisEncodeOptions)> {
        use AstcBlock::*;
        use BasisCodec::{AstcLdr, XuastcLdr};
        let o = |codec| BasisEncodeOptions {
            codec,
            ..Default::default()
        };
        vec![
            ("cli_astc4x4_rgba_64x48", "rgba_64x48", o(AstcLdr(B4x4))),
            (
                "cli_astc4x4_e0_rgb_37x29",
                "rgb_37x29",
                BasisEncodeOptions {
                    astc_ldr_effort: 0,
                    ..o(AstcLdr(B4x4))
                },
            ),
            (
                "cli_astc6x6_linear_mip_rgb_37x29",
                "rgb_37x29",
                BasisEncodeOptions {
                    srgb: false,
                    mipmaps: true,
                    ..o(AstcLdr(B6x6))
                },
            ),
            (
                "cli_astc8x6_e1_rgba_64x48",
                "rgba_64x48",
                BasisEncodeOptions {
                    astc_ldr_effort: 1,
                    ..o(AstcLdr(B8x6))
                },
            ),
            ("cli_astc12x12_rgba_64x48", "rgba_64x48", o(AstcLdr(B12x12))),
            (
                "cli_xuastc4x4_zstd_rgb_37x29",
                "rgb_37x29",
                o(XuastcLdr(B4x4)),
            ),
            (
                "cli_xuastc5x5_e5_mip_rgb_37x29",
                "rgb_37x29",
                BasisEncodeOptions {
                    astc_ldr_effort: 5,
                    mipmaps: true,
                    ..o(XuastcLdr(B5x5))
                },
            ),
            (
                "cli_xuastc6x6_arith_mip_rgba_64x48",
                "rgba_64x48",
                BasisEncodeOptions {
                    xuastc_syntax: XuastcSyntax::Arith,
                    mipmaps: true,
                    ..o(XuastcLdr(B6x6))
                },
            ),
            (
                "cli_xuastc6x5_hybrid_linear_rgba_64x48",
                "rgba_64x48",
                BasisEncodeOptions {
                    xuastc_syntax: XuastcSyntax::Hybrid,
                    srgb: false,
                    ..o(XuastcLdr(B6x5))
                },
            ),
            (
                "cli_xuastc6x6_q60_rgba_64x48",
                "rgba_64x48",
                BasisEncodeOptions {
                    xuastc_quality: 60,
                    ..o(XuastcLdr(B6x6))
                },
            ),
            (
                "cli_xuastc6x6_hybrid_q60_rgba_64x48",
                "rgba_64x48",
                BasisEncodeOptions {
                    xuastc_syntax: XuastcSyntax::Hybrid,
                    xuastc_quality: 60,
                    ..o(XuastcLdr(B6x6))
                },
            ),
            (
                "cli_xuastc8x6_arith_q60_mosaic_rgba_50x38",
                "mosaic_rgba_50x38",
                BasisEncodeOptions {
                    xuastc_syntax: XuastcSyntax::Arith,
                    xuastc_quality: 60,
                    ..o(XuastcLdr(B8x6))
                },
            ),
            (
                "cli_xuastc12x12_rgba_64x48",
                "rgba_64x48",
                o(XuastcLdr(B12x12)),
            ),
            (
                "cli_xuastc12x12_nodeblock_rgba_64x48",
                "rgba_64x48",
                BasisEncodeOptions {
                    xuastc_deblocking: XuastcDeblocking::Off,
                    ..o(XuastcLdr(B12x12))
                },
            ),
        ]
    }

    fn testdata(path: &str) -> R<Vec<u8>> {
        Ok(std::fs::read(format!(
            "{}/testdata/{path}",
            env!("CARGO_MANIFEST_DIR")
        ))?)
    }

    /// Raw RGBA8 test image NAME_WxH -> (width, height, pixels).
    fn raw(name: &str) -> R<(u32, u32, Vec<u8>)> {
        let dims = name.rsplit('_').next().ok_or("name")?;
        let (w, h) = dims.split_once('x').ok_or("dims")?;
        Ok((
            w.parse()?,
            h.parse()?,
            testdata(&format!("images/raw/{name}.rgba"))?,
        ))
    }

    fn unpack_all(blocks: &[u8], block: AstcBlock) -> R<Vec<LogAstcBlock>> {
        let (bw, bh) = block.dim();
        let mut out = Vec::new();
        for b in blocks.as_chunks::<16>().0 {
            let mut log = LogAstcBlock::default();
            if !unpack_block(b, &mut log, bw, bh) {
                return Err("unpack_block".into());
            }
            out.push(log);
        }
        Ok(out)
    }

    /// The `compress_image` outputs of every slice of a C++ `.basis` golden: its payload and the
    /// logical blocks (ASTC LDR: the payload unpacked; XUASTC: the payload decoded to ASTC).
    fn golden_outputs(golden: &[u8]) -> R<Vec<Option<CompressOutput>>> {
        let f = BasisFile::parse(golden)?;
        let block = f.tex_format().astc_block().ok_or("not ASTC")?;
        let mut out = Vec::new();
        for s in f.slices.iter() {
            let payload = golden
                .get(s.file_ofs as usize..(s.file_ofs + s.file_size) as usize)
                .ok_or("slice")?
                .to_vec();
            let astc = if matches!(f.tex_format(), BasisTexFormat::AstcLdr(_)) {
                payload.clone()
            } else {
                let kind = if f.is_srgb() {
                    AstcKind::Srgb
                } else {
                    AstcKind::Unorm
                };
                f.transcode_to_buffer(s.image_index, s.level_index, Format::Astc(block, kind))?
                    .to_vec()
            };
            out.push(Some(CompressOutput {
                comp_data: payload,
                coded_blocks: unpack_all(&astc, block)?,
            }));
        }
        Ok(out)
    }

    #[test]
    fn writer_matches_basisu_goldens() -> R<()> {
        for (name, image, opts) in goldens() {
            let golden = testdata(&format!("basisu_enc/{name}.basis"))?;
            let (w, h, px) = raw(image)?;
            let src = [Surface::new(Format::Rgba8Unorm, w, h, &px)?];
            for threads in [1, 3] {
                let outputs = golden_outputs(&golden)?;
                let ours = encode_with(&src, &BasisEncodeOptions { threads, ..opts }, |c| {
                    c.injected_xuastc_outputs = Some(outputs)
                })?;
                assert!(ours[..] == golden[..], "{name}");
            }
            // 12x12 (>= 80 texels) deblocks by default: SCD + `DeblockFilterID` = "1".
            let kv = crate::basis::ll::basis_file::basis_file_key_values(&golden).ok_or("kv")?;
            let deblock = name.contains("12x12") && !name.contains("nodeblock");
            assert_eq!(kv.len(), 1 + deblock as usize, "{name}");
            assert_eq!(kv.last().ok_or("kv")?.key_str(), b"BasisULibVers");
            assert_eq!(kv.last().ok_or("kv")?.value, b"2.50\0");
            if deblock {
                assert_eq!(kv[0].key_str(), b"DeblockFilterID");
                assert_eq!(kv[0].value, b"1\0");
            }
        }
        Ok(())
    }

    #[test]
    fn writer_without_v2_keys() -> R<()> {
        // Pre-2.0 layout: no key/value data, except keys that change decoding.
        let (w, h, px) = raw("rgba_64x48")?;
        let src = [Surface::new(Format::Rgba8Unorm, w, h, &px)?];
        for (name, deblock_key) in [
            ("cli_xuastc6x6_q60_rgba_64x48", false),
            ("cli_xuastc12x12_rgba_64x48", true),
        ] {
            let golden = testdata(&format!("basisu_enc/{name}.basis"))?;
            let opts = goldens()
                .into_iter()
                .find(|g| g.0 == name)
                .ok_or("golden")?
                .2;
            let outputs = golden_outputs(&golden)?;
            let ours = encode_with(
                &src,
                &BasisEncodeOptions {
                    basisu_v2_keys: false,
                    ..opts
                },
                |c| c.injected_xuastc_outputs = Some(outputs),
            )?;
            let ext = |b: &[u8]| -> (usize, usize) {
                let le = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                (le(69) as usize, le(73) as usize)
            };
            let (gofs, gsize) = ext(&golden);
            let (ofs, size) = ext(&ours);
            // BasisULibVers = "2.50\0": 5-byte entry header + 13 + 5 bytes.
            assert_eq!(gsize - size, if deblock_key { 23 } else { gsize });
            assert_eq!(ours.len(), golden.len() - (gsize - size), "{name}");
            if deblock_key {
                assert_eq!(ofs, gofs);
                let kv = crate::basis::ll::basis_file::parse_key_value_data(&ours[ofs..ofs + size])
                    .ok_or("kv")?;
                assert_eq!(kv.len(), 1);
                assert_eq!(kv[0].key_str(), b"DeblockFilterID");
                assert_eq!(kv[0].value, b"1\0");
            }
            // Same payloads and descriptors otherwise.
            let (a, b) = (BasisFile::parse(&ours)?, BasisFile::parse(&golden)?);
            for (sa, sb) in a.slices.iter().zip(b.slices.iter()) {
                assert_eq!(
                    ours[sa.file_ofs as usize..][..sa.file_size as usize],
                    golden[sb.file_ofs as usize..][..sb.file_size as usize]
                );
            }
        }
        Ok(())
    }

    /// The `astc_ldr_encode_config` of `opts` (`any_alpha`: a source image has alpha).
    fn cfg_of(opts: &BasisEncodeOptions, any_alpha: bool) -> R<AstcLdrEncodeConfigView> {
        let p = compressor_params(opts)?;
        let f = opts.codec.tex_format();
        let (bw, bh) = f.block_dim();
        let c = BasisCompressor::xuastc_or_astc_ldr_config(
            &p,
            bw,
            bh,
            matches!(f, BasisTexFormat::XuastcLdr(_)),
            any_alpha,
        );
        Ok(AstcLdrEncodeConfigView {
            effort: c.effort_level,
            astcf: c.use_astcf,
            merge: c.merge_basisu_into_output,
            scd: c.scd_enabled,
            postfilter: c.scd_will_postfilter,
            passes: c.num_scd_passes,
            dct: c.use_dct,
            dct_quality: c.dct_quality,
            lossy: c.lossy_supercompression,
            weights: c.comp_weights,
            srgb: c.astc_decode_mode_srgb,
            syntax: c.compressed_syntax,
            chroma: c.scd_preserve_chroma,
        })
    }

    #[derive(Debug, PartialEq)]
    struct AstcLdrEncodeConfigView {
        effort: u32,
        astcf: bool,
        merge: bool,
        scd: bool,
        postfilter: bool,
        passes: u32,
        dct: bool,
        dct_quality: f32,
        lossy: bool,
        weights: [u32; 4],
        srgb: bool,
        syntax: EncSyntax,
        chroma: bool,
    }

    #[test]
    fn config_mapping() -> R<()> {
        use AstcBlock::*;
        let x = |b| BasisEncodeOptions {
            codec: BasisCodec::XuastcLdr(b),
            ..Default::default()
        };
        // CLI defaults: effort 3, zstd, sRGB weights, no DCT, no SCD below 10x8.
        let c = cfg_of(&x(B6x6), false)?;
        assert_eq!(
            c,
            AstcLdrEncodeConfigView {
                effort: 3,
                astcf: true,
                merge: false,
                scd: false,
                postfilter: false,
                passes: 8,
                dct: false,
                dct_quality: 85.0,
                lossy: false,
                weights: [9, 11, 1, 11],
                srgb: true,
                syntax: EncSyntax::FullZstd,
                chroma: true,
            }
        );
        // Effort 3 merges basisu with alpha or more than 64 texels per block.
        assert!(cfg_of(&x(B6x6), true)?.merge);
        assert!(!cfg_of(&x(B8x8), false)?.merge);
        assert!(cfg_of(&x(B10x8), false)?.merge);
        // Effort 0: astcf except 4x4 (bc7f path); 1-2 astcf only; >= 4 both.
        for (b, e, astcf, merge, passes) in [
            (B4x4, 0, false, false, 0),
            (B6x6, 0, true, false, 0),
            (B6x6, 1, true, false, 0),
            (B6x6, 2, true, false, 2),
            (B4x4, 4, true, true, 10),
            (B6x6, 5, true, true, 12),
            (B6x6, 7, true, true, 16),
            (B6x6, 9, true, true, 20),
            (B6x6, 10, true, true, 32),
        ] {
            let c = cfg_of(
                &BasisEncodeOptions {
                    astc_ldr_effort: e,
                    ..x(b)
                },
                false,
            )?;
            assert_eq!(
                (c.effort, c.astcf, c.merge, c.passes),
                (e, astcf, merge, passes),
                "{b:?} effort {e}"
            );
        }
        // Deblocking: SCD + filter from 10x8 (80 texels) by default.
        assert_eq!(BASISU_DEBLOCKING_BLOCK_SIZE_THRESHOLD, 80);
        let c = cfg_of(&x(B10x8), false)?;
        assert!(c.scd && c.postfilter && c.passes == 8);
        let c = cfg_of(&x(B10x6), false)?;
        assert!(!c.scd && !c.postfilter);
        let off = |d, b| BasisEncodeOptions {
            xuastc_deblocking: d,
            ..x(b)
        };
        let c = cfg_of(&off(XuastcDeblocking::Off, B12x12), false)?;
        assert!(!c.scd && !c.postfilter);
        let c = cfg_of(&off(XuastcDeblocking::ScdFilterAll, B4x4), false)?;
        assert!(c.scd && c.postfilter);
        let c = cfg_of(&off(XuastcDeblocking::ScdOnly, B4x4), false)?;
        assert!(c.scd && !c.postfilter);
        let c = cfg_of(&off(XuastcDeblocking::FilterLargest, B12x12), false)?;
        assert!(!c.scd && c.postfilter);
        let c = cfg_of(&off(XuastcDeblocking::FilterLargest, B8x8), false)?;
        assert!(!c.scd && !c.postfilter);
        let c = cfg_of(&off(XuastcDeblocking::FilterAll, B4x4), false)?;
        assert!(!c.scd && c.postfilter);
        // Effort 1 has no SCD passes, so SCD is off but the filter key stays.
        let c = cfg_of(
            &BasisEncodeOptions {
                astc_ldr_effort: 1,
                ..x(B12x12)
            },
            false,
        )?;
        assert!(!c.scd && c.postfilter && c.passes == 0);
        // Quality < 100: DCT + lossy supercompression (XUASTC only).
        let q = |codec, q| BasisEncodeOptions {
            codec,
            xuastc_quality: q,
            ..Default::default()
        };
        let c = cfg_of(&q(BasisCodec::XuastcLdr(B6x6), 60), false)?;
        assert!(c.dct && c.lossy && c.dct_quality == 60.0);
        let c = cfg_of(&q(BasisCodec::XuastcLdr(B6x6), 1), false)?;
        assert!(c.dct && c.lossy && c.dct_quality == 1.0);
        let c = cfg_of(&q(BasisCodec::AstcLdr(B6x6), 60), false)?;
        assert!(!c.dct && !c.lossy);
        let c = cfg_of(
            &BasisEncodeOptions {
                xuastc_lossy_supercompression: Some(false),
                ..q(BasisCodec::XuastcLdr(B6x6), 60)
            },
            false,
        )?;
        assert!(c.dct && !c.lossy);
        let c = cfg_of(
            &BasisEncodeOptions {
                xuastc_lossy_supercompression: Some(true),
                ..x(B6x6)
            },
            false,
        )?;
        assert!(!c.dct && c.lossy);
        // -linear: linear decode mode, 1,1,1,1 weights, no chroma preservation; overrides.
        let c = cfg_of(
            &BasisEncodeOptions {
                srgb: false,
                xuastc_syntax: XuastcSyntax::Arith,
                ..x(B6x6)
            },
            false,
        )?;
        assert!(!c.srgb && !c.chroma && c.weights == [1, 1, 1, 1]);
        assert_eq!(c.syntax, EncSyntax::FullArith);
        let c = cfg_of(
            &BasisEncodeOptions {
                ldr_channel_weights: Some([2, 3, 4, 5]),
                xuastc_syntax: XuastcSyntax::Hybrid,
                ..x(B6x6)
            },
            false,
        )?;
        assert_eq!(c.weights, [2, 3, 4, 5]);
        assert_eq!(c.syntax, EncSyntax::HybridArithZstd);
        Ok(())
    }

    #[test]
    fn xubc7_config_mapping() -> R<()> {
        use crate::basis::xubc7::encoder::{BasisuParams, EncodeParams, Stripes};
        let pack = |opts: &BasisEncodeOptions| -> R<EncodeParams> {
            let p = compressor_params(opts)?;
            assert_eq!(p.format_mode, BasisTexFormat::Xubc7);
            Ok(BasisCompressor::xubc7_pack_options(&p))
        };
        let x = BasisEncodeOptions {
            codec: BasisCodec::Xubc7,
            ..Default::default()
        };
        // CLI `-xubc7` defaults: Q 100, effort 3 (weight re-optimization), bc7f, 8 stripes,
        // sRGB weights + perceptual.
        let d = pack(&x)?;
        assert_eq!(d, EncodeParams::default());
        assert!(d.optimize_weights_after_bc7f && d.perceptual && d.dct_q == 100);
        assert_eq!(d.stripes, Stripes::Desired(8));
        // Every knob, as `encode_slices_to_xubc7` replays it.
        let o = BasisEncodeOptions {
            xubc7_quality: 60,
            xubc7_effort: 10,
            xubc7_rdo_level: 40,
            xubc7_encoder: Xubc7Encoder::Bc7e(5),
            xubc7_stripes: 3,
            srgb: false,
            ..x
        };
        let mut b = BasisuParams {
            quality: 60,
            effort_level: 10,
            rdo_level: 40,
            num_stripes: 3,
            encoder: Bc7Encoder::Bc7eScalar,
            bc7e_level: 5,
            ..Default::default()
        };
        b.set_srgb(false);
        assert_eq!(pack(&o)?, EncodeParams::from_basisu(&b));
        let w = pack(&BasisEncodeOptions {
            ldr_channel_weights: Some([2, 3, 4, 5]),
            xubc7_effort: 1,
            ..x
        })?;
        assert_eq!(w.weights, [2, 3, 4, 5]);
        assert!(!w.optimize_weights_after_bc7f);
        assert_eq!(
            w.bc7_pack_flags,
            crate::basis::ll::bc7f::PACK_BC7_FLAG_DEFAULT
        );
        Ok(())
    }

    #[test]
    fn encoder_errors_propagate() -> R<()> {
        // Until the block encoder lands, its error comes through unchanged; afterwards this
        // encodes. Either way no other error (e.g. from the surrounding compressor) occurs.
        let (w, h, px) = raw("rgb_37x29")?;
        let src = [Surface::new(Format::Rgba8Unorm, w, h, &px)?];
        let r = encode_basis(
            &src,
            &BasisEncodeOptions {
                codec: BasisCodec::XuastcLdr(AstcBlock::B6x6),
                ..Default::default()
            },
        );
        assert!(matches!(r, Ok(_) | Err(Error::Unsupported)), "{r:?}");
        Ok(())
    }
}
