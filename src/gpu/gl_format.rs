// SPDX-FileCopyrightText: 2010-2026 Mark Callow, The Khronos Group Inc. and KTX-Software contributors
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from KTX-Software (via ktx-rs) and modified; see NOTICE.md.
//! OpenGL {,ES} internal format, format and type of every imgio [`Format`]: the entries of
//! KTX-Software `vkFormat2glInternalFormat` / `vkFormat2glFormat` / `vkFormat2glType`
//! (`vk2gl.h`) for the formats imgio has, plus ETC1, ATC and FXT1, which have no `VkFormat`.
#![cfg_attr(not(feature = "opengl"), allow(dead_code))]

use crate::{AstcBlock, AstcKind, Format};

/// How a [`Format`] is passed to `glTexImage*` / `glCompressedTexImage*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlFormat {
    /// Sized internal format (`GL_RGBA8`, `GL_COMPRESSED_RGBA_BPTC_UNORM`, ...).
    pub internal_format: u32,
    /// Unsized base internal format (`GL_RGBA`, ...) for contexts without sized formats;
    /// 0 for compressed formats.
    pub base_internal_format: u32,
    /// Pixel data format (`GL_RGBA`, ...); 0 for compressed formats.
    pub format: u32,
    /// Pixel data type (`GL_UNSIGNED_BYTE`, ...); 0 for compressed formats.
    pub ty: u32,
}

impl GlFormat {
    /// True for block-compressed formats (`glCompressedTexImage*`).
    pub fn is_compressed(&self) -> bool {
        self.format == 0
    }
}

const GL_RED: u32 = 0x1903;
const GL_RG: u32 = 0x8227;
const GL_RGB: u32 = 0x1907;
const GL_RGBA: u32 = 0x1908;
const GL_BGRA: u32 = 0x80E1;
const GL_BYTE: u32 = 0x1400;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_SHORT: u32 = 0x1402;
const GL_UNSIGNED_SHORT: u32 = 0x1403;
const GL_FLOAT: u32 = 0x1406;
const GL_HALF_FLOAT: u32 = 0x140B;
const GL_UNSIGNED_SHORT_4_4_4_4: u32 = 0x8033;
const GL_UNSIGNED_SHORT_5_6_5: u32 = 0x8363;
const GL_UNSIGNED_SHORT_5_6_5_REV: u32 = 0x8364;
const GL_UNSIGNED_INT_5_9_9_9_REV: u32 = 0x8C3E;
const GL_RGBA8: u32 = 0x8058;
const GL_ETC1_RGB8_OES: u32 = 0x8D64;
const GL_COMPRESSED_R11_EAC: u32 = 0x9270;
const GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC: u32 = 0x9279;

const ASTC_RGBA_4X4: u32 = 0x93B0;
const ASTC_SRGB8_ALPHA8_4X4: u32 = 0x93D0;

const fn plain(internal_format: u32, format: u32, ty: u32) -> GlFormat {
    GlFormat {
        internal_format,
        base_internal_format: format,
        format,
        ty,
    }
}

const fn compressed(internal_format: u32) -> GlFormat {
    GlFormat {
        internal_format,
        base_internal_format: 0,
        format: 0,
        ty: 0,
    }
}

/// The GL formats of `format` (`None`: no GL equivalent).
///
/// BC1 maps to `GL_COMPRESSED_RGBA_S3TC_DXT1_EXT` (imgio's BC1 has punch-through alpha), ETC1
/// to `GL_ETC1_RGB8_OES`, PVRTC2 RGB to the RGBA enum (there is no RGB one), ASTC HDR to the
/// LDR enum (the HDR profile is a context capability).
pub fn gl_format(format: Format) -> Option<GlFormat> {
    use Format::*;
    Some(match format {
        R8Unorm => plain(0x8229, GL_RED, GL_UNSIGNED_BYTE),
        R8Snorm => plain(0x8F94, GL_RED, GL_BYTE),
        Rg8Unorm => plain(0x822B, GL_RG, GL_UNSIGNED_BYTE),
        Rg8Snorm => plain(0x8F95, GL_RG, GL_BYTE),
        R16Unorm => plain(0x822A, GL_RED, GL_UNSIGNED_SHORT),
        R16Snorm => plain(0x8F98, GL_RED, GL_SHORT),
        Rg16Unorm => plain(0x822C, GL_RG, GL_UNSIGNED_SHORT),
        Rg16Snorm => plain(0x8F99, GL_RG, GL_SHORT),
        Rgba8Unorm => plain(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE),
        Rgba8Srgb => plain(0x8C43, GL_RGBA, GL_UNSIGNED_BYTE),
        Rgb8Unorm => plain(0x8051, GL_RGB, GL_UNSIGNED_BYTE),
        Rgb8Srgb => plain(0x8C41, GL_RGB, GL_UNSIGNED_BYTE),
        Bgra8Unorm => GlFormat {
            internal_format: GL_RGBA8,
            base_internal_format: GL_RGBA,
            format: GL_BGRA,
            ty: GL_UNSIGNED_BYTE,
        },
        Bgra8Srgb => GlFormat {
            internal_format: 0x8C43,
            base_internal_format: GL_RGBA,
            format: GL_BGRA,
            ty: GL_UNSIGNED_BYTE,
        },
        Rgb565 => plain(0x8D62, GL_RGB, GL_UNSIGNED_SHORT_5_6_5),
        Bgr565 => plain(0x8D62, GL_RGB, GL_UNSIGNED_SHORT_5_6_5_REV),
        Rgba4444 => plain(0x8056, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4),
        Rgb16Float => plain(0x881B, GL_RGB, GL_HALF_FLOAT),
        Rgb9e5Float => plain(0x8C3D, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV),
        Rgba16Float => plain(0x881A, GL_RGBA, GL_HALF_FLOAT),
        Rgba16Unorm => plain(0x805B, GL_RGBA, GL_UNSIGNED_SHORT),
        Rgba32Float => plain(0x8814, GL_RGBA, GL_FLOAT),
        R16Float => plain(0x822D, GL_RED, GL_HALF_FLOAT),
        R32Float => plain(0x822E, GL_RED, GL_FLOAT),
        Bc1Unorm => compressed(0x83F1),
        Bc1Srgb => compressed(0x8C4D),
        Bc2Unorm => compressed(0x83F2),
        Bc2Srgb => compressed(0x8C4E),
        Bc3Unorm => compressed(0x83F3),
        Bc3Srgb => compressed(0x8C4F),
        Bc4Unorm => compressed(0x8DBB),
        Bc4Snorm => compressed(0x8DBC),
        Bc5Unorm => compressed(0x8DBD),
        Bc5Snorm => compressed(0x8DBE),
        Bc6hUfloat => compressed(0x8E8F),
        Bc6hSfloat => compressed(0x8E8E),
        Bc7Unorm => compressed(0x8E8C),
        Bc7Srgb => compressed(0x8E8D),
        Etc1Rgb => compressed(GL_ETC1_RGB8_OES),
        Etc2Rgb => compressed(0x9274),
        Etc2Srgb => compressed(0x9275),
        Etc2Rgba1 => compressed(0x9276),
        Etc2Rgba1Srgb => compressed(0x9277),
        Etc2Rgba => compressed(0x9278),
        Etc2RgbaSrgb => compressed(GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC),
        EacR11Unorm => compressed(GL_COMPRESSED_R11_EAC),
        EacR11Snorm => compressed(0x9271),
        EacRg11Unorm => compressed(0x9272),
        EacRg11Snorm => compressed(0x9273),
        Astc(block, kind) => {
            let i = AstcBlock::ALL.iter().position(|&b| b == block)? as u32;
            compressed(match kind {
                AstcKind::Unorm | AstcKind::Float => ASTC_RGBA_4X4 + i,
                AstcKind::Srgb => ASTC_SRGB8_ALPHA8_4X4 + i,
            })
        }
        Pvrtc1Rgb => compressed(0x8C00), // GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG
        Pvrtc1Rgba => compressed(0x8C02), // GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG
        Pvrtc2Rgb | Pvrtc2Rgba => compressed(0x9138), // GL_COMPRESSED_RGBA_PVRTC_4BPPV2_IMG
        AtcRgb => compressed(0x8C92),    // GL_ATC_RGB_AMD
        AtcRgba => compressed(0x87EE),   // GL_ATC_RGBA_INTERPOLATED_ALPHA_AMD
        Fxt1Rgb => compressed(0x86B0),   // GL_COMPRESSED_RGB_FXT1_3DFX
    })
}

/// True for the formats KTX-Software unpacks in software when the context rejects them
/// (`GL_ETC1_RGB8_OES` and `GL_COMPRESSED_R11_EAC` .. `GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC`).
pub fn is_etc(internal_format: u32) -> bool {
    internal_format == GL_ETC1_RGB8_OES
        || (GL_COMPRESSED_R11_EAC..=GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC).contains(&internal_format)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping() {
        assert_eq!(
            gl_format(Format::Rgba8Unorm),
            Some(plain(0x8058, 0x1908, 0x1401))
        );
        let astc = |b, k| gl_format(Format::Astc(b, k)).map(|f| f.internal_format);
        assert_eq!(astc(AstcBlock::B12x12, AstcKind::Unorm), Some(0x93BD));
        assert_eq!(astc(AstcBlock::B6x6, AstcKind::Srgb), Some(0x93D4));
        assert_eq!(astc(AstcBlock::B4x4, AstcKind::Float), Some(0x93B0));
        assert!(gl_format(Format::Bc7Srgb).is_some_and(|f| f.is_compressed()));
        assert!(is_etc(0x9274) && is_etc(GL_ETC1_RGB8_OES) && !is_etc(0x8E8C));
        // Every format has a GL equivalent.
        for f in [Format::Fxt1Rgb, Format::AtcRgba, Format::Pvrtc2Rgb] {
            assert!(gl_format(f).is_some());
        }
    }
}
