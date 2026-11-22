//! GPU memory of textures: bytes per format with mips, what a texture file occupies once
//! uploaded, and estimates for desktop / mobile formats chosen from the image content.
//!
//! [`texture_bytes`] sums every level, layer and face of a [`Layout`] with block rounding
//! (a 2x2 level of BC7 still takes a 4x4 block). [`of_container`] reads KTX2, DDS, KTX1, Basis
//! and `.astc` headers (no pixel decoding); supercompressed Basis payloads have no fixed GPU
//! format, [`transcode_target`] gives the format a runtime would usually pick. For image files,
//! [`analyze`] classifies alpha, gray and HDR content and [`suggest`] picks a block format per
//! [`Target`].

use crate::basis::BasisTexFormat;
use crate::{AstcBlock, AstcKind, Error, Format, Result, Surface};

/// Texture dimensions and counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub width: u32,
    pub height: u32,
    /// 1 for 2D textures.
    pub depth: u32,
    /// Array layers (1 for plain textures).
    pub layers: u32,
    /// 6 for cubemaps, else 1.
    pub faces: u32,
    /// Mip levels (1 = base only).
    pub levels: u32,
}

impl Layout {
    /// A 2D texture, with the full mip chain when `mips`.
    pub fn new_2d(width: u32, height: u32, mips: bool) -> Layout {
        Layout {
            width,
            height,
            depth: 1,
            layers: 1,
            faces: 1,
            levels: if mips {
                full_levels(width, height, 1)
            } else {
                1
            },
        }
    }

    /// The same texture with the full mip chain (`mips`) or the base level only.
    pub fn with_mips(self, mips: bool) -> Layout {
        Layout {
            levels: if mips {
                full_levels(self.width, self.height, self.depth)
            } else {
                1
            },
            ..self
        }
    }
}

/// Levels of a full mip chain down to 1x1x1.
pub fn full_levels(width: u32, height: u32, depth: u32) -> u32 {
    32 - width.max(height).max(depth).max(1).leading_zeros()
}

/// Bytes of every level, layer and face of `layout` in `format` (levels halve each dimension,
/// at least 1; block formats round up to whole blocks, 3D textures store `depth` slices).
/// `InvalidArgument` for zero sizes, faces other than 1 / 6 or more levels than the full chain.
pub fn texture_bytes(format: Format, layout: &Layout) -> Result<u64> {
    let l = layout;
    let ok = l.width > 0
        && l.height > 0
        && l.depth > 0
        && l.layers > 0
        && matches!(l.faces, 1 | 6)
        && (1..=full_levels(l.width, l.height, l.depth)).contains(&l.levels);
    if !ok {
        return Err(Error::InvalidArgument);
    }
    let mut total = 0u64;
    for i in 0..l.levels {
        let (w, h, d) = (
            (l.width >> i).max(1),
            (l.height >> i).max(1),
            (l.depth >> i).max(1),
        );
        let s = format.surface_bytes(w, h)? as u64;
        total = s
            .checked_mul(u64::from(d) * u64::from(l.layers) * u64::from(l.faces))
            .and_then(|v| total.checked_add(v))
            .ok_or(Error::InvalidArgument)?;
    }
    Ok(total)
}

/// The format an uncompressed image is uploaded as: 24-bit RGB / BGR becomes RGBA8 (GPUs rarely
/// sample 3-byte texels), RGB16F becomes RGBA16F; everything else is itself.
pub fn upload_format(f: Format) -> Format {
    match f {
        Format::Rgb8Unorm => Format::Rgba8Unorm,
        Format::Rgb8Srgb => Format::Rgba8Srgb,
        Format::Rgb16Float => Format::Rgba16Float,
        f => f,
    }
}

/// Alpha usage of an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alpha {
    /// Every alpha is 1 (or the format has none).
    Opaque,
    /// Only 0 and 1 (cutouts).
    Binary,
    /// Partial transparency.
    Full,
}

/// What [`analyze`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Content {
    pub alpha: Alpha,
    /// Red, green and blue equal everywhere (or a one / two channel format).
    pub gray: bool,
    /// A float / shared exponent format.
    pub hdr: bool,
    /// An sRGB labeled format.
    pub srgb: bool,
}

/// Classifies an uncompressed surface (stored values, like [`crate::convert`]).
pub fn analyze(src: &Surface) -> Result<Content> {
    let f = src.format();
    let n = crate::resize::channels(f).ok_or(Error::Unsupported)?;
    let px = crate::convert::convert_to_buffer(src, Format::Rgba32Float)?;
    let (mut gray, mut binary, mut opaque) = (true, true, true);
    for p in px.as_chunks::<16>().0 {
        let v = |c: usize| f32::from_le_bytes([p[c * 4], p[c * 4 + 1], p[c * 4 + 2], p[c * 4 + 3]]);
        let a = match n {
            2 => v(1),
            4 => v(3),
            _ => 1.0,
        };
        if a != 1.0 {
            opaque = false;
            if a != 0.0 {
                binary = false;
            }
        }
        if n >= 3 && (v(0) != v(1) || v(1) != v(2)) {
            gray = false;
        }
    }
    Ok(Content {
        alpha: if opaque {
            Alpha::Opaque
        } else if binary {
            Alpha::Binary
        } else {
            Alpha::Full
        },
        gray,
        hdr: f.is_hdr(),
        srgb: f.is_srgb(),
    })
}

/// A platform family for [`suggest`] / [`transcode_target`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// BC formats (Windows, Linux, consoles).
    Desktop,
    /// ASTC (iOS, Android, Apple silicon).
    Mobile,
}

/// A block format for content on a target: desktop BC6H (HDR), BC4 (opaque gray), BC5 (gray +
/// alpha, or two channel data such as normal map XY: pass `gray` with `Alpha::Full`), BC7
/// (colour); mobile ASTC 6x6 (HDR, or LDR colour / gray) and ASTC 4x4 for gray +
/// alpha and colour with partial alpha (alpha costs ASTC precision). sRGB content keeps sRGB.
pub fn suggest(c: &Content, target: Target) -> Format {
    let kind = if c.hdr {
        AstcKind::Float
    } else if c.srgb {
        AstcKind::Srgb
    } else {
        AstcKind::Unorm
    };
    match target {
        Target::Desktop if c.hdr => Format::Bc6hUfloat,
        Target::Desktop if c.gray && c.alpha == Alpha::Opaque => Format::Bc4Unorm,
        Target::Desktop if c.gray => Format::Bc5Unorm,
        Target::Desktop if c.srgb => Format::Bc7Srgb,
        Target::Desktop => Format::Bc7Unorm,
        Target::Mobile if c.hdr || c.alpha != Alpha::Full => Format::Astc(AstcBlock::B6x6, kind),
        Target::Mobile => Format::Astc(AstcBlock::B4x4, kind),
    }
}

/// The GPU format a Basis payload is usually transcoded to: desktop ETC1S -> BC1 (BC3 with
/// alpha), LDR UASTC / XUASTC / ASTC / XUBC7 -> BC7, HDR -> BC6H; mobile ETC1S -> ETC2 RGB
/// (RGBA with alpha), UASTC LDR / XUBC7 -> ASTC 4x4, XUASTC / ASTC -> ASTC of the same block,
/// HDR -> ASTC HDR of the payload's block size.
pub fn transcode_target(b: BasisTexFormat, alpha: bool, srgb: bool, target: Target) -> Format {
    let k = if srgb {
        AstcKind::Srgb
    } else {
        AstcKind::Unorm
    };
    let hdr = matches!(
        b,
        BasisTexFormat::UastcHdr4x4
            | BasisTexFormat::AstcHdr6x6
            | BasisTexFormat::UastcHdr6x6Intermediate
    );
    match target {
        Target::Desktop => match b {
            _ if hdr => Format::Bc6hUfloat,
            BasisTexFormat::Etc1s => match (alpha, srgb) {
                (false, false) => Format::Bc1Unorm,
                (false, true) => Format::Bc1Srgb,
                (true, false) => Format::Bc3Unorm,
                (true, true) => Format::Bc3Srgb,
            },
            _ if srgb => Format::Bc7Srgb,
            _ => Format::Bc7Unorm,
        },
        Target::Mobile => match b {
            BasisTexFormat::Etc1s => match (alpha, srgb) {
                (false, false) => Format::Etc2Rgb,
                (false, true) => Format::Etc2Srgb,
                (true, false) => Format::Etc2Rgba,
                (true, true) => Format::Etc2RgbaSrgb,
            },
            BasisTexFormat::UastcHdr4x4 => Format::Astc(AstcBlock::B4x4, AstcKind::Float),
            BasisTexFormat::AstcHdr6x6 | BasisTexFormat::UastcHdr6x6Intermediate => {
                Format::Astc(AstcBlock::B6x6, AstcKind::Float)
            }
            BasisTexFormat::XuastcLdr(bl) | BasisTexFormat::AstcLdr(bl) => Format::Astc(bl, k),
            _ => Format::Astc(AstcBlock::B4x4, k),
        },
    }
}

/// A texture container's dimensions and GPU format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContainerMemory {
    pub layout: Layout,
    /// The stored GPU format (`None` for Basis payloads, see `basis`).
    pub format: Option<Format>,
    /// Basis Universal payload kind (`.basis`, or KTX2 with Basis data).
    pub basis: Option<BasisTexFormat>,
    pub has_alpha: bool,
    pub srgb: bool,
    /// Channels of the decoded format (1 R, 2 RG, 3 RGB, 4 RGBA; Basis: 3 or 4).
    pub channels: u8,
    /// [`texture_bytes`] of `format` (`None` for Basis payloads).
    pub gpu_bytes: Option<u64>,
}

/// Reads a KTX2, DDS, KTX1, `.basis` or `.astc` header; `Ok(None)` for other data (image
/// files). Basis files count `image_count` images of the first image's size as layers.
pub fn of_container(data: &[u8]) -> Result<Option<ContainerMemory>> {
    let starts = |m: &[u8]| data.starts_with(m);
    let mk = |layout: Layout,
              format: Option<Format>,
              basis: Option<BasisTexFormat>,
              has_alpha: bool,
              srgb: bool|
     -> Result<Option<ContainerMemory>> {
        let gpu_bytes = format.map(|f| texture_bytes(f, &layout)).transpose()?;
        let channels = match format.and_then(|f| crate::resize::channels(f.decoded())) {
            Some(n) => n as u8,
            None if has_alpha => 4,
            None => 3,
        };
        Ok(Some(ContainerMemory {
            layout,
            format,
            basis,
            has_alpha,
            srgb,
            channels,
            gpu_bytes,
        }))
    };
    let alpha_of = |f: Format| {
        !matches!(
            f.decoded(),
            Format::R8Unorm
                | Format::R8Snorm
                | Format::Rg8Unorm
                | Format::Rg8Snorm
                | Format::Rgb8Unorm
                | Format::Rgb8Srgb
                | Format::R16Unorm
                | Format::R16Float
                | Format::R32Float
                | Format::Rgb16Float
                | Format::Rgb9e5Float
                | Format::Rgb565
                | Format::Bgr565
        )
    };
    if starts(&[0xAB, b'K', b'T', b'X', b' ', b'2', b'0', 0xBB]) {
        let k = crate::ktx2::Ktx2File::parse(data)?;
        let layout = Layout {
            width: k.width(),
            height: k.height().max(1),
            depth: k.depth().max(1),
            layers: k.layer_count().max(1),
            faces: k.face_count().max(1),
            levels: k.level_count().max(1),
        };
        let fmt = k.format();
        // Plain ASTC payloads in Basis KTX2 files have a real GPU format.
        return mk(
            layout,
            fmt,
            if fmt.is_none() { k.tex_format() } else { None },
            fmt.map_or(k.has_alpha(), alpha_of),
            k.is_srgb(),
        );
    }
    if starts(&[0xAB, b'K', b'T', b'X', b' ', b'1', b'1', 0xBB]) {
        let k = crate::ktx1::Ktx1File::parse(data)?;
        let f = k.format();
        let layout = Layout {
            width: k.width(),
            height: k.height().max(1),
            depth: k.depth().max(1),
            layers: k.layer_count().max(1),
            faces: k.face_count().max(1),
            levels: k.level_count().max(1),
        };
        return mk(layout, Some(f), None, alpha_of(f), f.is_srgb());
    }
    if starts(b"DDS ") {
        let d = crate::dds::DdsFile::parse(data)?;
        let f = d.format();
        let layout = Layout {
            width: d.width(),
            height: d.height(),
            depth: d.depth().max(1),
            layers: d.layer_count().max(1),
            faces: d.face_count().max(1),
            levels: d.level_count().max(1),
        };
        return mk(layout, Some(f), None, alpha_of(f), f.is_srgb());
    }
    if starts(&[0x13, 0xAB, 0xA1, 0x5C]) {
        let a = crate::astc_file::AstcFile::parse(data)?;
        let f = a.format(AstcKind::Unorm).ok_or(Error::Unsupported)?;
        let layout = Layout {
            width: a.width(),
            height: a.height(),
            depth: a.depth().max(1),
            layers: 1,
            faces: 1,
            levels: 1,
        };
        return mk(layout, Some(f), None, true, false);
    }
    if starts(b"sB") {
        let b = crate::basis::BasisFile::parse(data)?;
        let li = b.level_info(0, 0)?;
        let layout = Layout {
            width: li.width,
            height: li.height,
            depth: 1,
            layers: b.image_count().max(1),
            faces: 1,
            levels: b.level_count(0).max(1),
        };
        return mk(
            layout,
            None,
            Some(b.tex_format()),
            b.has_alpha(),
            b.is_srgb(),
        );
    }
    Ok(None)
}
