//! Basis Universal encoding: images to a `.basis` file.
//!
//! Drives the ported `basis_compressor` (`ll::comp`, bit-exact with the C++ encoder) for
//! ETC1S, UASTC LDR 4x4 (optionally RDO), UASTC HDR 4x4, ASTC HDR 6x6 and UASTC HDR 6x6
//! intermediate. Defaults follow the `basisu` command line tool.

use super::file::BasisTexType;
use super::ll::comp::{
    BasisCompressor, BasisCompressorErrorCode, BasisCompressorParams, HdrModes, Image, ImageF,
};
use super::ll::frontend::{
    C_BASIS_TEX_TYPE_2D, C_BASIS_TEX_TYPE_2D_ARRAY, C_BASIS_TEX_TYPE_CUBEMAP_ARRAY,
    C_BASIS_TEX_TYPE_VIDEO_FRAMES, C_BASIS_TEX_TYPE_VOLUME,
};
use crate::convert::{ConvertOptions, convert_to_buffer, convert_with};
use crate::mip::Filter;
use crate::{Buffer, Error, Format, Result, Surface};

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
}

impl BasisCodec {
    /// True for the HDR codecs (which take float input and ignore alpha).
    pub fn is_hdr(self) -> bool {
        !matches!(self, BasisCodec::Etc1s | BasisCodec::UastcLdr4x4)
    }
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
    use BasisCompressorErrorCode as Ec;
    match comp.process() {
        Ec::Success => {}
        Ec::FailedReadingSourceImages | Ec::FailedValidating | Ec::FailedInvalidParameters => {
            return Err(Error::InvalidArgument);
        }
        _ => return Err(Error::InvalidData),
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
    }
    Ok(p)
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
