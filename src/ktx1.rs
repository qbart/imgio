//! KTX 1.1 (`.ktx`) reader and writer (own implementation of the Khronos KTX 1 layout).
//!
//! - Reader: both endiannesses (big-endian files are byte swapped by `glTypeSize`), 2D
//!   textures with mips, arrays and cubemaps; GL enums -> imgio [`Format`] (compressed:
//!   S3TC / RGTC / BPTC / ETC / EAC / ASTC / PVRTC / ATC / FXT1 incl. sRGB and RGB / RGBA
//!   aliases; uncompressed: sized internal formats, or unsized `format` + `type`). ASTC enums
//!   have no LDR / HDR distinction and read as LDR. `image(level, layer, face)` returns the
//!   tightly packed texels (uncompressed rows are 4-byte aligned in the file; PVRTC1 levels
//!   are stored with at least 2 x 2 blocks, tiled, and cropped back).
//! - Writer: the conventions of basisu `create_ktx_texture_file` for compressed formats
//!   (`glType` / `glFormat` 0, `glTypeSize` 1, BC1 as `GL_COMPRESSED_RGB_S3TC_DXT1`), sized
//!   formats from [`crate::gpu::gl_format`] for uncompressed ones; optional key/values.

use crate::{AstcBlock, AstcKind, Buffer, Error, Format, Result};

const IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];
const ENDIAN: u32 = 0x0403_0201;

const GL_RED: u32 = 0x1903;
const GL_RG: u32 = 0x8227;
const GL_RGB: u32 = 0x1907;
const GL_RGBA: u32 = 0x1908;

/// Compressed internal format -> (format, base internal format). The first entry of a
/// format is the one the writer uses.
const COMPRESSED: &[(u32, Format, u32)] = {
    use Format::*;
    &[
        (0x83F0, Bc1Unorm, GL_RGB),
        (0x83F1, Bc1Unorm, GL_RGBA),
        (0x8C4C, Bc1Srgb, GL_RGB),
        (0x8C4D, Bc1Srgb, GL_RGBA),
        (0x83F2, Bc2Unorm, GL_RGBA),
        (0x8C4E, Bc2Srgb, GL_RGBA),
        (0x83F3, Bc3Unorm, GL_RGBA),
        (0x8C4F, Bc3Srgb, GL_RGBA),
        (0x8DBB, Bc4Unorm, GL_RED),
        (0x8DBC, Bc4Snorm, GL_RED),
        (0x8DBD, Bc5Unorm, GL_RG),
        (0x8DBE, Bc5Snorm, GL_RG),
        (0x8E8F, Bc6hUfloat, GL_RGB),
        (0x8E8E, Bc6hSfloat, GL_RGB),
        (0x8E8C, Bc7Unorm, GL_RGBA),
        (0x8E8D, Bc7Srgb, GL_RGBA),
        (0x8D64, Etc1Rgb, GL_RGB),
        (0x9274, Etc2Rgb, GL_RGB),
        (0x9275, Etc2Srgb, GL_RGB),
        (0x9276, Etc2Rgba1, GL_RGBA),
        (0x9277, Etc2Rgba1Srgb, GL_RGBA),
        (0x9278, Etc2Rgba, GL_RGBA),
        (0x9279, Etc2RgbaSrgb, GL_RGBA),
        (0x9270, EacR11Unorm, GL_RED),
        (0x9271, EacR11Snorm, GL_RED),
        (0x9272, EacRg11Unorm, GL_RG),
        (0x9273, EacRg11Snorm, GL_RG),
        (0x8C00, Pvrtc1Rgb, GL_RGB),
        (0x8A55, Pvrtc1Rgb, GL_RGB),
        (0x8C02, Pvrtc1Rgba, GL_RGBA),
        (0x8A57, Pvrtc1Rgba, GL_RGBA),
        (0x9138, Pvrtc2Rgba, GL_RGBA),
        (0x93F1, Pvrtc2Rgba, GL_RGBA),
        (0x8C92, AtcRgb, GL_RGB),
        (0x87EE, AtcRgba, GL_RGBA),
        (0x86B0, Fxt1Rgb, GL_RGB),
    ]
};

/// sRGB enums of formats whose imgio [`Format`] has no sRGB variant (PVRTC).
const SRGB_ONLY_ENUMS: [(u32, Format); 3] = [
    (0x8A55, Format::Pvrtc1Rgb),
    (0x8A57, Format::Pvrtc1Rgba),
    (0x93F1, Format::Pvrtc2Rgba),
];

/// The writer's (internal, base internal) format of a compressed `format` (`srgb`: the sRGB
/// enum for PVRTC, which has no sRGB `Format`).
fn compressed_gl(format: Format, srgb: bool) -> Option<(u32, u32)> {
    let f = if format == Format::Pvrtc2Rgb {
        Format::Pvrtc2Rgba
    } else {
        format
    };
    if srgb && let Some(e) = SRGB_ONLY_ENUMS.iter().find(|e| e.1 == f) {
        return Some((
            e.0,
            if f == Format::Pvrtc1Rgb {
                GL_RGB
            } else {
                GL_RGBA
            },
        ));
    }
    if let Format::Astc(b, kind) = format {
        let i = AstcBlock::ALL.iter().position(|&x| x == b)? as u32;
        let base = if kind == AstcKind::Srgb {
            0x93D0
        } else {
            0x93B0
        };
        return Some((base + i, GL_RGBA));
    }
    COMPRESSED
        .iter()
        .find(|e| e.1 == f && !SRGB_ONLY_ENUMS.iter().any(|s| s.0 == e.0))
        .map(|e| (e.0, e.2))
}

fn format_from_gl(internal: u32, gl_format: u32, gl_type: u32) -> Option<Format> {
    if let Some(e) = COMPRESSED.iter().find(|e| e.0 == internal) {
        return Some(e.1);
    }
    if (0x93B0..=0x93BD).contains(&internal) {
        return Some(Format::Astc(
            AstcBlock::ALL[(internal - 0x93B0) as usize],
            AstcKind::Unorm,
        ));
    }
    if (0x93D0..=0x93DD).contains(&internal) {
        return Some(Format::Astc(
            AstcBlock::ALL[(internal - 0x93D0) as usize],
            AstcKind::Srgb,
        ));
    }
    // Uncompressed: the sized internal format, or the unsized format + type.
    UNCOMPRESSED.iter().copied().find(|&f| {
        crate::gpu::gl_format(f).is_some_and(|g| {
            g.internal_format == internal
                || (internal == g.format && gl_format == g.format && gl_type == g.ty)
        })
    })
}

const UNCOMPRESSED: [Format; 22] = {
    use Format::*;
    [
        R16Float,
        R32Float,
        Rgb8Unorm,
        Rgb8Srgb,
        R8Unorm,
        R8Snorm,
        Rg8Unorm,
        Rg8Snorm,
        R16Unorm,
        R16Snorm,
        Rg16Unorm,
        Rg16Snorm,
        Rgba8Unorm,
        Rgba8Srgb,
        Rgb565,
        Bgr565,
        Rgba4444,
        Rgb16Float,
        Rgb9e5Float,
        Rgba16Float,
        Rgba16Unorm,
        Rgba32Float,
    ]
};

/// `glTypeSize` of an uncompressed format (the element size for endian conversion).
fn type_size(format: Format) -> u32 {
    use Format::*;
    match format {
        R16Unorm | R16Snorm | Rg16Unorm | Rg16Snorm | Rgb565 | Bgr565 | Rgba4444 | Rgb16Float
        | Rgba16Float | Rgba16Unorm | R16Float => 2,
        Rgb9e5Float | Rgba32Float | R32Float => 4,
        _ => 1,
    }
}

/// True for PVRTC1, whose levels are stored with at least 2 x 2 blocks.
fn is_pvrtc1(format: Format) -> bool {
    matches!(format, Format::Pvrtc1Rgb | Format::Pvrtc1Rgba)
}

/// PVRTC1 block grid of a `w` x `h` level: (tight, stored with the 2 x 2 minimum).
fn pvrtc1_grids(w: u32, h: u32) -> ((usize, usize), (usize, usize)) {
    let (bw, bh) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
    ((bw, bh), (bw.max(2), bh.max(2)))
}

/// Re-twiddles PVRTC1 blocks from grid `from` to grid `to`, block (x, y) of `to` taking block
/// (x % from.w, y % from.h) (PVRTC wraps, so tiling pads losslessly and cropping undoes it).
fn pvrtc1_regrid(src: &[u8], from: (usize, usize), to: (usize, usize), dst: &mut [u8]) {
    use crate::pvrtc::swizzle;
    for y in 0..to.1 {
        for x in 0..to.0 {
            let d = swizzle(to.0, to.1, x, y) * 8;
            let s = swizzle(from.0, from.1, x % from.0, y % from.1) * 8;
            dst[d..d + 8].copy_from_slice(&src[s..s + 8]);
        }
    }
}

/// Bytes of one face / array element of `format` at `w` x `h` as stored (uncompressed rows
/// padded to 4 bytes, PVRTC1 at least 2 x 2 blocks).
fn stored_size(format: Format, w: u32, h: u32) -> Result<usize> {
    if is_pvrtc1(format) {
        let (_, (bw, bh)) = pvrtc1_grids(w, h);
        return Ok(bw * bh * 8);
    }
    if format.is_compressed() {
        return format.surface_bytes(w, h);
    }
    Ok(format.row_bytes(w)?.next_multiple_of(4) * h as usize)
}

/// A parsed `.ktx` file, borrowing its bytes.
#[derive(Debug, Clone)]
pub struct Ktx1File<'a> {
    data: &'a [u8],
    format: Format,
    width: u32,
    height: u32,
    levels: u32,
    layers: u32,
    faces: u32,
    big_endian: bool,
    /// Base depth (1 for 2D textures).
    depth: u32,
    internal: u32,
    kvd: &'a [u8],
    /// Offset of each level's data (after its imageSize).
    level_ofs: [usize; 32],
}

impl<'a> Ktx1File<'a> {
    /// Parses the header and locates every level.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() < 64 || data[..12] != IDENTIFIER {
            return Err(Error::InvalidData);
        }
        let raw = |i: usize| {
            u32::from_le_bytes([
                data[12 + i * 4],
                data[13 + i * 4],
                data[14 + i * 4],
                data[15 + i * 4],
            ])
        };
        let big_endian = match raw(0) {
            ENDIAN => false,
            0x0102_0304 => true,
            _ => return Err(Error::InvalidData),
        };
        let h = |i: usize| {
            if big_endian {
                raw(i).swap_bytes()
            } else {
                raw(i)
            }
        };
        let (gl_type, _type_size, gl_format, internal) = (h(1), h(2), h(3), h(4));
        let (width, height, depth) = (h(6), h(7), h(8));
        let (layers, faces, levels, kvd_len) = (h(9), h(10), h(11), h(12));
        if width == 0 {
            return Err(Error::Unsupported);
        }
        let (height, depth) = (height.max(1), depth.max(1));
        if !(faces == 1 || (faces == 6 && width == height && depth == 1)) {
            return Err(Error::InvalidData);
        }
        let levels = levels.max(1);
        if levels > crate::image::mip_levels(width.max(depth), height) {
            return Err(Error::InvalidData);
        }
        let format = format_from_gl(internal, gl_format, gl_type).ok_or(Error::Unsupported)?;
        let kvd_end = 64usize
            .checked_add(kvd_len as usize)
            .ok_or(Error::InvalidData)?;
        let kvd = data.get(64..kvd_end).ok_or(Error::InvalidData)?;
        let mut level_ofs = [0usize; 32];
        let mut at = kvd_end;
        let elements = layers.max(1) as usize;
        for (l, slot) in level_ofs.iter_mut().enumerate().take(levels as usize) {
            let (w, hh) = crate::image::mip_dim(width, height, l as u32);
            let slices = depth.checked_shr(l as u32).unwrap_or(0).max(1) as usize;
            let face = stored_size(format, w, hh)? * slices;
            let face_padded = if faces == 6 && layers == 0 {
                face.next_multiple_of(4)
            } else {
                face
            };
            *slot = at + 4;
            let size = face_padded * faces as usize * elements;
            at = (at + 4)
                .checked_add(size)
                .ok_or(Error::InvalidData)?
                .next_multiple_of(4);
        }
        if data.len() < at.saturating_sub(3) {
            return Err(Error::InvalidData);
        }
        Ok(Ktx1File {
            data,
            format,
            width,
            height,
            levels,
            layers: layers.max(1),
            faces,
            big_endian,
            depth,
            internal,
            kvd,
            level_ofs,
        })
    }

    /// Pixel / block format (ASTC reads as LDR; the file cannot say HDR).
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
    /// Array elements (1 for a plain texture).
    pub fn layer_count(&self) -> u32 {
        self.layers
    }
    /// 6 for cubemaps, 1 otherwise.
    pub fn face_count(&self) -> u32 {
        self.faces
    }
    /// Depth of level 0 (1 for 2D textures).
    pub fn depth(&self) -> u32 {
        self.depth
    }
    /// Depth slices of `level` (`image` returns them back to back).
    pub fn level_depth(&self, level: u32) -> u32 {
        self.depth.checked_shr(level).unwrap_or(0).max(1)
    }
    /// The raw `glInternalFormat`.
    pub fn gl_internal_format(&self) -> u32 {
        self.internal
    }
    /// True when the GL format is sRGB (also for PVRTC, whose `Format` has no sRGB variant).
    pub fn is_srgb(&self) -> bool {
        self.format.is_srgb() || SRGB_ONLY_ENUMS.iter().any(|e| e.0 == self.internal)
    }
    /// True for big-endian files.
    pub fn is_big_endian(&self) -> bool {
        self.big_endian
    }
    /// The key/value pairs (key without its NUL, raw value).
    pub fn key_values(&self) -> impl Iterator<Item = (&'a [u8], &'a [u8])> + 'a {
        let (kvd, be) = (self.kvd, self.big_endian);
        let mut at = 0usize;
        core::iter::from_fn(move || {
            let n = kvd.get(at..at + 4)?;
            let n = if be {
                u32::from_be_bytes([n[0], n[1], n[2], n[3]])
            } else {
                u32::from_le_bytes([n[0], n[1], n[2], n[3]])
            } as usize;
            let kv = kvd.get(at + 4..(at + 4).checked_add(n)?)?;
            at += 4 + n.next_multiple_of(4);
            let z = kv.iter().position(|&b| b == 0)?;
            Some((&kv[..z], &kv[z + 1..]))
        })
    }

    /// The texels of one image (`level`, array `layer`, cube `face`), tightly packed and in
    /// little-endian element order (3D textures: the level's depth slices back to back).
    pub fn image(&self, level: u32, layer: u32, face: u32) -> Result<Buffer> {
        if level >= self.levels || layer >= self.layers || face >= self.faces {
            return Err(Error::InvalidArgument);
        }
        let f = self.format;
        let (w, h) = crate::image::mip_dim(self.width, self.height, level);
        let slices = self.level_depth(level) as usize;
        let slice_size = stored_size(f, w, h)?;
        let face_size = slice_size * slices;
        let padded = if self.faces == 6 && self.layers == 1 {
            face_size.next_multiple_of(4)
        } else {
            face_size
        };
        let start = self.level_ofs[level as usize] + (layer * self.faces + face) as usize * padded;
        let src = self
            .data
            .get(start..start + face_size)
            .ok_or(Error::InvalidData)?;
        let tight = f.surface_bytes(w, h)?;
        let mut out = crate::buffer::zeroed(tight * slices)?;
        for z in 0..slices {
            let src = &src[z * slice_size..(z + 1) * slice_size];
            let dst = &mut out[z * tight..(z + 1) * tight];
            if is_pvrtc1(f) {
                let (tg, stored) = pvrtc1_grids(w, h);
                pvrtc1_regrid(src, stored, tg, dst);
            } else if f.is_compressed() {
                dst.copy_from_slice(src);
            } else {
                let (row, stored) = (f.row_bytes(w)?, f.row_bytes(w)?.next_multiple_of(4));
                for y in 0..h as usize {
                    dst[y * row..(y + 1) * row].copy_from_slice(&src[y * stored..y * stored + row]);
                }
            }
        }
        if !f.is_compressed() {
            let ts = type_size(f) as usize;
            if self.big_endian && ts > 1 {
                for e in out.chunks_exact_mut(ts) {
                    e.reverse();
                }
            }
        }
        Ok(out)
    }
}

/// Shape of a texture to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ktx1Desc {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    /// Array elements; 0 for a non-array texture.
    pub layers: u32,
    /// 1, or 6 for a cubemap (square).
    pub faces: u32,
    /// Mip levels (at least 1).
    pub levels: u32,
    /// Use the sRGB GL enum for PVRTC formats (other formats carry sRGB in `format`).
    pub srgb: bool,
    /// Depth of a 3D texture; 0 (or 1) for 2D.
    pub depth: u32,
}

/// Writes a little-endian `.ktx` file. `levels[i]` holds mip level `i`: `max(layers, 1) *
/// faces` tightly packed images of `format.surface_bytes(level size)` bytes (layer, then face;
/// the same input as [`crate::ktx2::write()`]). `key_values` are written in the given order
/// (values raw; strings include their NUL).
pub fn write(desc: &Ktx1Desc, levels: &[&[u8]], key_values: &[(&str, &[u8])]) -> Result<Buffer> {
    let f = desc.format;
    let (w, h) = (desc.width, desc.height);
    if w == 0
        || h == 0
        || levels.is_empty()
        || levels.len() != desc.levels as usize
        || desc.levels > crate::image::mip_levels(w.max(desc.depth), h)
        || !(desc.faces == 1 || (desc.faces == 6 && w == h && desc.depth <= 1))
    {
        return Err(Error::InvalidArgument);
    }
    let (gl_type, gl_format, internal, base, ts) = if f.is_compressed() {
        let (i, b) = compressed_gl(f, desc.srgb).ok_or(Error::Unsupported)?;
        (0, 0, i, b, 1)
    } else {
        let g = crate::gpu::gl_format(f).ok_or(Error::Unsupported)?;
        (
            g.ty,
            g.format,
            g.internal_format,
            g.base_internal_format,
            type_size(f),
        )
    };
    let images = desc.layers.max(1) as usize * desc.faces as usize;
    let mut kvd = Vec::new();
    for (k, v) in key_values {
        if k.is_empty() || k.as_bytes().contains(&0) {
            return Err(Error::InvalidArgument);
        }
        let n = k.len() + 1 + v.len();
        kvd.extend_from_slice(&(n as u32).to_le_bytes());
        kvd.extend_from_slice(k.as_bytes());
        kvd.push(0);
        kvd.extend_from_slice(v);
        kvd.resize(kvd.len().next_multiple_of(4), 0);
    }
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&IDENTIFIER);
    for v in [
        ENDIAN,
        gl_type,
        ts,
        gl_format,
        internal,
        base,
        w,
        h,
        if desc.depth > 1 { desc.depth } else { 0 },
        desc.layers,
        desc.faces,
        desc.levels,
        kvd.len() as u32,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&kvd);
    let cube = desc.faces == 6 && desc.layers == 0;
    for (l, data) in levels.iter().enumerate() {
        let (lw, lh) = crate::image::mip_dim(w, h, l as u32);
        let slices = desc.depth.checked_shr(l as u32).unwrap_or(0).max(1) as usize;
        let size = f.surface_bytes(lw, lh)?;
        if size.checked_mul(images * slices) != Some(data.len()) {
            return Err(Error::InvalidArgument);
        }
        let slice_stored = stored_size(f, lw, lh)?;
        let face = slice_stored * slices;
        let image_size = if cube { face } else { face * images };
        out.extend_from_slice(&(image_size as u32).to_le_bytes());
        // Images, depth slices innermost (cubemaps are 2D, so one slice per face).
        for i in 0..images * slices {
            let src = &data[i * size..(i + 1) * size];
            if is_pvrtc1(f) {
                let (tight, stored) = pvrtc1_grids(lw, lh);
                let at = out.len();
                out.resize(at + slice_stored, 0);
                pvrtc1_regrid(src, tight, stored, &mut out[at..]);
            } else if f.is_compressed() {
                out.extend_from_slice(src);
            } else {
                let row = f.row_bytes(lw)?;
                for y in 0..lh as usize {
                    out.extend_from_slice(&src[y * row..(y + 1) * row]);
                    out.resize(out.len().next_multiple_of(4), 0);
                }
            }
            if cube {
                out.resize(out.len().next_multiple_of(4), 0);
            }
        }
        out.resize(out.len().next_multiple_of(4), 0);
    }
    let mut buf = crate::buffer::zeroed(out.len())?;
    buf.copy_from_slice(&out);
    Ok(buf)
}
