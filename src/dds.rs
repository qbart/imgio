//! DirectDraw Surface (`.dds`) reader and writer (own implementation, Microsoft DDS layout).
//!
//! - Reader: DX10 headers (DXGI formats) and legacy headers (FourCC `DXT1`..`DXT5`, `ATI1/2`,
//!   `BC4U/S`, `BC5U/S`, D3DFMT float codes, RGB / luminance bit masks); 2D textures with mips,
//!   arrays and cubemaps (all six faces). Volume textures and formats without an imgio
//!   [`Format`] (e.g. BGRA8) are `Unsupported`.
//! - Writer: DX10 header (layout as basisu `build_dx10_header`), or a legacy header for the
//!   formats that have one (BC1-BC5, RGBA8, R8, RG8, 565, RGBA16 / RGBA16F / RGBA32F codes).
//! - Data order: for every array element / face, its full mip chain.

use crate::{Buffer, Error, Format, Result};

const MAGIC: u32 = 0x2053_4444; // "DDS "
const HEADER: usize = 128; // magic + DDS_HEADER
const DX10: usize = 20;

const DDSD_CAPS: u32 = 0x1;
const DDSD_HEIGHT: u32 = 0x2;
const DDSD_WIDTH: u32 = 0x4;
const DDSD_PITCH: u32 = 0x8;
const DDSD_PIXELFORMAT: u32 = 0x1000;
const DDSD_MIPMAPCOUNT: u32 = 0x20000;
const DDSD_LINEARSIZE: u32 = 0x80000;
const DDSD_DEPTH: u32 = 0x80_0000;

const DDPF_ALPHAPIXELS: u32 = 0x1;
const DDPF_FOURCC: u32 = 0x4;
const DDPF_RGB: u32 = 0x40;
const DDPF_LUMINANCE: u32 = 0x2_0000;

const DDSCAPS_COMPLEX: u32 = 0x8;
const DDSCAPS_TEXTURE: u32 = 0x1000;
const DDSCAPS_MIPMAP: u32 = 0x40_0000;
const DDSCAPS2_CUBEMAP: u32 = 0x200;
const DDSCAPS2_CUBEMAP_ALLFACES: u32 = 0xFC00;
const DDSCAPS2_VOLUME: u32 = 0x20_0000;

const DIMENSION_TEXTURE2D: u32 = 3;
const MISC_TEXTURECUBE: u32 = 0x4;

const fn fourcc(s: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*s)
}

/// The DXGI format of `format` (`None`: no DXGI equivalent).
pub fn dxgi_format(format: Format) -> Option<u32> {
    use Format::*;
    Some(match format {
        Rgba32Float => 2,
        Rgba16Float => 10,
        Rgba16Unorm => 11,
        Rgba8Unorm => 28,
        Rgba8Srgb => 29,
        Rg16Unorm => 35,
        Rg16Snorm => 37,
        Rg8Unorm => 49,
        Rg8Snorm => 51,
        R16Unorm => 56,
        R16Snorm => 58,
        R8Unorm => 61,
        R8Snorm => 63,
        Rgb9e5Float => 67,
        Bc1Unorm => 71,
        Bc1Srgb => 72,
        Bc2Unorm => 74,
        Bc2Srgb => 75,
        Bc3Unorm => 77,
        Bc3Srgb => 78,
        Bc4Unorm => 80,
        Bc4Snorm => 81,
        Bc5Unorm => 83,
        Bc5Snorm => 84,
        // DXGI B5G6R5: blue in the low bits = imgio's Rgb565.
        Rgb565 => 85,
        Bc6hUfloat => 95,
        Bc6hSfloat => 96,
        Bc7Unorm => 98,
        Bc7Srgb => 99,
        _ => return None,
    })
}

/// The imgio format of DXGI format `dxgi` (typeless BC formats read as unorm).
pub fn format_from_dxgi(dxgi: u32) -> Option<Format> {
    use Format::*;
    Some(match dxgi {
        70 => Bc1Unorm,
        73 => Bc2Unorm,
        76 => Bc3Unorm,
        79 => Bc4Unorm,
        82 => Bc5Unorm,
        94 => Bc6hUfloat,
        97 => Bc7Unorm,
        _ => {
            return ALL.into_iter().find(|&f| dxgi_format(f) == Some(dxgi));
        }
    })
}

const ALL: [Format; 29] = {
    use Format::*;
    [
        Rgba32Float,
        Rgba16Float,
        Rgba16Unorm,
        Rgba8Unorm,
        Rgba8Srgb,
        Rg16Unorm,
        Rg16Snorm,
        Rg8Unorm,
        Rg8Snorm,
        R16Unorm,
        R16Snorm,
        R8Unorm,
        R8Snorm,
        Rgb9e5Float,
        Bc1Unorm,
        Bc1Srgb,
        Bc2Unorm,
        Bc2Srgb,
        Bc3Unorm,
        Bc3Srgb,
        Bc4Unorm,
        Bc4Snorm,
        Bc5Unorm,
        Bc5Snorm,
        Rgb565,
        Bc6hUfloat,
        Bc6hSfloat,
        Bc7Unorm,
        Bc7Srgb,
    ]
};

/// Legacy pixel format: (flags, FourCC, bits, R, G, B, A masks).
type Legacy = (u32, u32, u32, [u32; 4]);

/// The legacy (pre-DX10) pixel format of `format`, if it has one.
fn legacy(format: Format) -> Option<Legacy> {
    use Format::*;
    let fcc = |s: &[u8; 4]| (DDPF_FOURCC, fourcc(s), 0, [0; 4]);
    let code = |c: u32| (DDPF_FOURCC, c, 0, [0; 4]);
    Some(match format {
        Bc1Unorm => fcc(b"DXT1"),
        Bc2Unorm => fcc(b"DXT3"),
        Bc3Unorm => fcc(b"DXT5"),
        Bc4Unorm => fcc(b"ATI1"),
        Bc4Snorm => fcc(b"BC4S"),
        Bc5Unorm => fcc(b"ATI2"),
        Bc5Snorm => fcc(b"BC5S"),
        Rgba16Unorm => code(36),
        Rgba16Float => code(113),
        Rgba32Float => code(116),
        Rgba8Unorm => (
            DDPF_RGB | DDPF_ALPHAPIXELS,
            0,
            32,
            [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000],
        ),
        R8Unorm => (DDPF_LUMINANCE, 0, 8, [0xFF, 0, 0, 0]),
        Rg8Unorm => (DDPF_RGB, 0, 16, [0xFF, 0xFF00, 0, 0]),
        Rg16Unorm => (DDPF_RGB, 0, 32, [0xFFFF, 0xFFFF_0000, 0, 0]),
        Rgb565 => (DDPF_RGB, 0, 16, [0xF800, 0x07E0, 0x1F, 0]),
        Bgr565 => (DDPF_RGB, 0, 16, [0x1F, 0x07E0, 0xF800, 0]),
        Rgba4444 => (
            DDPF_RGB | DDPF_ALPHAPIXELS,
            0,
            16,
            [0xF000, 0x0F00, 0x00F0, 0x000F],
        ),
        _ => return None,
    })
}

/// The imgio format of a legacy pixel format.
fn format_from_legacy(flags: u32, cc: u32, bits: u32, masks: [u32; 4]) -> Option<Format> {
    use Format::*;
    if flags & DDPF_FOURCC != 0 {
        return Some(match cc {
            c if c == fourcc(b"DXT1") => Bc1Unorm,
            c if c == fourcc(b"DXT2") || c == fourcc(b"DXT3") => Bc2Unorm,
            c if c == fourcc(b"DXT4") || c == fourcc(b"DXT5") => Bc3Unorm,
            c if c == fourcc(b"ATI1") || c == fourcc(b"BC4U") => Bc4Unorm,
            c if c == fourcc(b"BC4S") => Bc4Snorm,
            c if c == fourcc(b"ATI2") || c == fourcc(b"BC5U") => Bc5Unorm,
            c if c == fourcc(b"BC5S") => Bc5Snorm,
            36 => Rgba16Unorm,
            113 => Rgba16Float,
            116 => Rgba32Float,
            _ => return None,
        });
    }
    let alpha = flags & DDPF_ALPHAPIXELS != 0;
    let m = [
        masks[0],
        masks[1],
        masks[2],
        if alpha { masks[3] } else { 0 },
    ];
    if flags & DDPF_LUMINANCE != 0 {
        return (bits == 8 && m[0] == 0xFF && m[3] == 0).then_some(R8Unorm);
    }
    if flags & DDPF_RGB == 0 {
        return None;
    }
    [
        Rgba8Unorm, R8Unorm, Rg8Unorm, Rg16Unorm, Rgb565, Bgr565, Rgba4444,
    ]
    .into_iter()
    .find(|&f| {
        legacy(f).is_some_and(|(_, _, b, lm)| {
            b == bits && lm[..3] == m[..3] && (lm[3] == m[3] || (f == R8Unorm && m[3] == 0))
        })
    })
}

fn u32le(b: &[u8], at: usize) -> Result<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or(Error::InvalidData)
}

/// A parsed `.dds` file, borrowing its bytes.
#[derive(Debug, Clone)]
pub struct DdsFile<'a> {
    data: &'a [u8],
    format: Format,
    width: u32,
    height: u32,
    levels: u32,
    layers: u32,
    faces: u32,
    dx10: bool,
    /// Offset of the first image.
    start: usize,
    /// Bytes of one face / array element (its whole mip chain).
    element: usize,
}

impl<'a> DdsFile<'a> {
    /// Parses and validates the headers; the data must hold every image.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() < HEADER || u32le(data, 0)? != MAGIC || u32le(data, 4)? != 124 {
            return Err(Error::InvalidData);
        }
        let h = |i: usize| u32le(data, 4 + i * 4);
        let (flags, height, width) = (h(1)?, h(2)?, h(3)?);
        let depth = h(5)?;
        let mips = if flags & DDSD_MIPMAPCOUNT != 0 {
            h(6)?
        } else {
            1
        }
        .max(1);
        let (pf_flags, cc, bits) = (h(19)?, h(20)?, h(21)?);
        let masks = [h(22)?, h(23)?, h(24)?, h(25)?];
        let caps2 = h(27)?;
        if width == 0 || height == 0 {
            return Err(Error::InvalidData);
        }
        if caps2 & DDSCAPS2_VOLUME != 0 || (flags & DDSD_DEPTH != 0 && depth > 1) {
            return Err(Error::Unsupported);
        }
        let dx10 = pf_flags & DDPF_FOURCC != 0 && cc == fourcc(b"DX10");
        let (format, layers, faces, start) = if dx10 {
            let d = |i: usize| u32le(data, HEADER + i * 4);
            let (dxgi, dim, misc, array) = (d(0)?, d(1)?, d(2)?, d(3)?);
            if dim != DIMENSION_TEXTURE2D {
                return Err(Error::Unsupported);
            }
            let format = format_from_dxgi(dxgi).ok_or(Error::Unsupported)?;
            let faces = if misc & MISC_TEXTURECUBE != 0 { 6 } else { 1 };
            (format, array.max(1), faces, HEADER + DX10)
        } else {
            let format = format_from_legacy(pf_flags, cc, bits, masks).ok_or(Error::Unsupported)?;
            let faces = if caps2 & DDSCAPS2_CUBEMAP != 0 {
                if caps2 & DDSCAPS2_CUBEMAP_ALLFACES != DDSCAPS2_CUBEMAP_ALLFACES {
                    return Err(Error::Unsupported); // partial cubemap
                }
                6
            } else {
                1
            };
            (format, 1, faces, HEADER)
        };
        if mips > crate::image::mip_levels(width, height) || (faces == 6 && width != height) {
            return Err(Error::InvalidData);
        }
        let mut element = 0usize;
        for level in 0..mips {
            let (w, h) = crate::image::mip_dim(width, height, level);
            element = element
                .checked_add(format.surface_bytes(w, h)?)
                .ok_or(Error::InvalidData)?;
        }
        let total = element
            .checked_mul(layers as usize * faces as usize)
            .and_then(|n| n.checked_add(start))
            .ok_or(Error::InvalidData)?;
        if data.len() < total {
            return Err(Error::InvalidData);
        }
        Ok(DdsFile {
            data,
            format,
            width,
            height,
            levels: mips,
            layers,
            faces,
            dx10,
            start,
            element,
        })
    }

    /// Pixel / block format.
    pub fn format(&self) -> Format {
        self.format
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
        self.levels
    }
    /// Array elements (1 for a plain texture; cubes for a cubemap array).
    pub fn layer_count(&self) -> u32 {
        self.layers
    }
    /// 6 for cubemaps, 1 otherwise.
    pub fn face_count(&self) -> u32 {
        self.faces
    }
    /// True when the file has a DX10 header.
    pub fn is_dx10(&self) -> bool {
        self.dx10
    }

    /// The bytes of one image (`level`, array `layer`, cube `face`).
    pub fn image_data(&self, level: u32, layer: u32, face: u32) -> Result<&'a [u8]> {
        if level >= self.levels || layer >= self.layers || face >= self.faces {
            return Err(Error::InvalidArgument);
        }
        let mut at = self.start + (layer * self.faces + face) as usize * self.element;
        for l in 0..level {
            let (w, h) = crate::image::mip_dim(self.width, self.height, l);
            at += self.format.surface_bytes(w, h)?;
        }
        let (w, h) = crate::image::mip_dim(self.width, self.height, level);
        let len = self.format.surface_bytes(w, h)?;
        self.data.get(at..at + len).ok_or(Error::InvalidData)
    }
}

/// Shape of a texture to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdsDesc {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    /// Array layers; 0 or 1 for a plain texture (a DX10 header with array size 1).
    pub layers: u32,
    /// 1, or 6 for a cubemap (square).
    pub faces: u32,
    /// Mip levels (at least 1).
    pub levels: u32,
}

/// Options of [`write()`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DdsWriteOptions {
    /// Write a legacy (pre-DX10) header when the format has one and the texture is not an array
    /// (`Unsupported` otherwise). Default: DX10 header.
    pub legacy: bool,
}

/// Writes a `.dds` file. `levels[i]` holds mip level `i`: `max(layers, 1) * faces` images of
/// `format.surface_bytes(level size)` bytes each, in layer then face order (the same input as
/// [`crate::ktx2::write`]); the file stores each face / layer with its full mip chain.
pub fn write(desc: &DdsDesc, levels: &[&[u8]], opts: &DdsWriteOptions) -> Result<Buffer> {
    let f = desc.format;
    let (w, h) = (desc.width, desc.height);
    let n = levels.len();
    if w == 0
        || h == 0
        || n == 0
        || n != desc.levels as usize
        || desc.levels > crate::image::mip_levels(w, h)
        || !(desc.faces == 1 || (desc.faces == 6 && w == h))
    {
        return Err(Error::InvalidArgument);
    }
    let images = desc.layers.max(1) as usize * desc.faces as usize;
    let mut sizes = [0usize; 32];
    for (i, l) in levels.iter().enumerate() {
        let (lw, lh) = crate::image::mip_dim(w, h, i as u32);
        sizes[i] = f.surface_bytes(lw, lh)?;
        if sizes[i].checked_mul(images) != Some(l.len()) {
            return Err(Error::InvalidArgument);
        }
    }
    let legacy_pf = if opts.legacy {
        if desc.layers > 1 {
            return Err(Error::Unsupported);
        }
        Some(legacy(f).ok_or(Error::Unsupported)?)
    } else {
        None
    };
    let dxgi = match legacy_pf {
        Some(_) => 0,
        None => dxgi_format(f).ok_or(Error::Unsupported)?,
    };
    let start = HEADER + if legacy_pf.is_some() { 0 } else { DX10 };
    let total = levels.iter().map(|l| l.len()).sum::<usize>() + start;
    let mut out = crate::buffer::zeroed(total)?;
    let mut put = |at: usize, v: u32| out[at..at + 4].copy_from_slice(&v.to_le_bytes());

    let compressed = f.is_compressed();
    let mut flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT;
    let pitch = if compressed {
        flags |= DDSD_LINEARSIZE;
        sizes[0]
    } else {
        flags |= DDSD_PITCH;
        f.row_bytes(w)?
    };
    if n > 1 {
        flags |= DDSD_MIPMAPCOUNT;
    }
    let cube = desc.faces == 6;
    let mut caps = DDSCAPS_TEXTURE;
    if n > 1 || cube || desc.layers > 1 {
        caps |= DDSCAPS_COMPLEX;
    }
    if n > 1 {
        caps |= DDSCAPS_MIPMAP;
    }
    let caps2 = if cube {
        DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_ALLFACES
    } else {
        0
    };
    put(0, MAGIC);
    put(4, 124);
    put(8, flags);
    put(12, h);
    put(16, w);
    put(20, pitch as u32);
    put(28, desc.levels);
    // DDS_PIXELFORMAT at 76.
    put(76, 32);
    match legacy_pf {
        Some((pf_flags, cc, bits, masks)) => {
            put(80, pf_flags);
            put(84, cc);
            put(88, bits);
            for (i, m) in masks.iter().enumerate() {
                put(92 + 4 * i, *m);
            }
        }
        None => {
            put(80, DDPF_FOURCC);
            put(84, fourcc(b"DX10"));
        }
    }
    put(108, caps);
    put(112, caps2);
    if legacy_pf.is_none() {
        put(128, dxgi);
        put(132, DIMENSION_TEXTURE2D);
        put(136, if cube { MISC_TEXTURECUBE } else { 0 });
        put(140, desc.layers.max(1));
    }
    // Images: every layer / face with its mip chain.
    let mut at = start;
    for image in 0..images {
        for (level, l) in levels.iter().enumerate() {
            let size = sizes[level];
            out[at..at + size].copy_from_slice(&l[image * size..(image + 1) * size]);
            at += size;
        }
    }
    Ok(out)
}
