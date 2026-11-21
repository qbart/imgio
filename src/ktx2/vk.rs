//! `VkFormat` values of the formats imgio handles (Vulkan 1.3 core, `VK_IMG_format_pvrtc`,
//! `VK_EXT_texture_compression_astc_hdr`).

use crate::{AstcBlock, AstcKind, Format};

const ASTC_LDR_BASE: u32 = 157;
const ASTC_SFLOAT_BASE: u32 = 1_000_066_000;
const PVRTC1_4BPP_UNORM: u32 = 1_000_054_001;
const PVRTC2_4BPP_UNORM: u32 = 1_000_054_003;
const PVRTC1_4BPP_SRGB: u32 = 1_000_054_005;
const PVRTC2_4BPP_SRGB: u32 = 1_000_054_007;

/// The `VkFormat` written for `format` (`None`: no Vulkan equivalent, e.g. ATC, FXT1).
///
/// BC1 is written as `BC1_RGBA` (imgio's BC1 decodes punch-through alpha), ETC1 as
/// `ETC2_R8G8B8` (ETC1 is a subset), PVRTC1 RGB/RGBA share one value.
pub fn vk_format(format: Format) -> Option<u32> {
    use Format::*;
    Some(match format {
        R8Unorm => 9,
        R8Snorm => 10,
        Rg8Unorm => 16,
        Rg8Snorm => 17,
        Rgba8Unorm => 37,
        Rgba8Srgb => 43,
        Rgb8Unorm => 23,
        Rgb8Srgb => 29,
        Bgra8Unorm => 44,
        Bgra8Srgb => 50,
        R16Unorm => 70,
        R16Snorm => 71,
        Rg16Unorm => 77,
        Rg16Snorm => 78,
        Rgb16Float => 90,
        Rgba16Float => 97,
        Rgba16Unorm => 91,
        Rgba32Float => 109,
        R16Float => 76,
        R32Float => 100,
        Rgb9e5Float => 123,
        Rgba4444 => 2,
        Rgb565 => 4,
        Bgr565 => 5,
        Bc1Unorm => 133,
        Bc1Srgb => 134,
        Bc2Unorm => 135,
        Bc2Srgb => 136,
        Bc3Unorm => 137,
        Bc3Srgb => 138,
        Bc4Unorm => 139,
        Bc4Snorm => 140,
        Bc5Unorm => 141,
        Bc5Snorm => 142,
        Bc6hUfloat => 143,
        Bc6hSfloat => 144,
        Bc7Unorm => 145,
        Bc7Srgb => 146,
        Etc1Rgb | Etc2Rgb => 147,
        Etc2Srgb => 148,
        Etc2Rgba1 => 149,
        Etc2Rgba1Srgb => 150,
        Etc2Rgba => 151,
        Etc2RgbaSrgb => 152,
        EacR11Unorm => 153,
        EacR11Snorm => 154,
        EacRg11Unorm => 155,
        EacRg11Snorm => 156,
        Astc(b, kind) => {
            let i = AstcBlock::ALL.iter().position(|&x| x == b)? as u32;
            match kind {
                AstcKind::Unorm => ASTC_LDR_BASE + 2 * i,
                AstcKind::Srgb => ASTC_LDR_BASE + 2 * i + 1,
                AstcKind::Float => ASTC_SFLOAT_BASE + i,
            }
        }
        Pvrtc1Rgb | Pvrtc1Rgba => PVRTC1_4BPP_UNORM,
        Pvrtc2Rgb | Pvrtc2Rgba => PVRTC2_4BPP_UNORM,
        AtcRgb | AtcRgba | Fxt1Rgb => return None,
    })
}

/// The imgio format of `vk` (`None`: not supported). `BC1_RGB` and `BC1_RGBA` both read as
/// BC1; PVRTC sRGB variants read as their unorm formats (same bits).
pub fn format_from_vk(vk: u32) -> Option<Format> {
    use Format::*;
    Some(match vk {
        131 | 133 => Bc1Unorm,
        132 | 134 => Bc1Srgb,
        PVRTC1_4BPP_UNORM | PVRTC1_4BPP_SRGB => Pvrtc1Rgba,
        PVRTC2_4BPP_UNORM | PVRTC2_4BPP_SRGB => Pvrtc2Rgba,
        157..=184 => {
            let i = (vk - ASTC_LDR_BASE) as usize;
            let kind = if i.is_multiple_of(2) {
                AstcKind::Unorm
            } else {
                AstcKind::Srgb
            };
            Astc(AstcBlock::ALL[i / 2], kind)
        }
        v if (ASTC_SFLOAT_BASE..ASTC_SFLOAT_BASE + 14).contains(&v) => Astc(
            AstcBlock::ALL[(v - ASTC_SFLOAT_BASE) as usize],
            AstcKind::Float,
        ),
        _ => {
            return ALL_PLAIN
                .iter()
                .copied()
                .find(|&f| f != Etc1Rgb && vk_format(f) == Some(vk));
        }
    })
}

/// Formats with a unique `VkFormat` (everything except ASTC, BC1 and PVRTC, handled above).
const ALL_PLAIN: [Format; 46] = {
    use Format::*;
    [
        R8Unorm,
        R8Snorm,
        Rg8Unorm,
        Rg8Snorm,
        Rgba8Unorm,
        Rgba8Srgb,
        Rgb8Unorm,
        Rgb8Srgb,
        Bgra8Unorm,
        Bgra8Srgb,
        R16Unorm,
        R16Snorm,
        Rg16Unorm,
        Rg16Snorm,
        Rgb16Float,
        Rgba16Float,
        Rgba16Unorm,
        Rgba32Float,
        R16Float,
        R32Float,
        Rgb9e5Float,
        Rgba4444,
        Rgb565,
        Bgr565,
        Bc2Unorm,
        Bc2Srgb,
        Bc3Unorm,
        Bc3Srgb,
        Bc4Unorm,
        Bc4Snorm,
        Bc5Unorm,
        Bc5Snorm,
        Bc6hUfloat,
        Bc6hSfloat,
        Bc7Unorm,
        Bc7Srgb,
        Etc2Rgb,
        Etc2Srgb,
        Etc2Rgba1,
        Etc2Rgba1Srgb,
        Etc2Rgba,
        Etc2RgbaSrgb,
        EacR11Unorm,
        EacR11Snorm,
        EacRg11Unorm,
        EacRg11Snorm,
    ]
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut n = 0;
        for vk in (0..200)
            .chain(1_000_054_000..1_000_054_010)
            .chain(1_000_066_000..1_000_066_016)
        {
            if let Some(f) = format_from_vk(vk) {
                n += 1;
                let back = vk_format(f);
                // Aliases read to a format whose own value differs.
                let alias = matches!(vk, 131 | 132 | 1_000_054_005 | 1_000_054_007);
                assert!(alias || back == Some(vk), "{vk} -> {f:?} -> {back:?}");
            }
        }
        assert_eq!(n, 46 + 4 + 4 + 28 + 14);
    }
}
