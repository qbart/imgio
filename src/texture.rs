//! One-call texture building: source images -> mip chain -> GPU format -> container.
//!
//! [`build`] takes the base images of a 2D texture, array or cubemap (layer-major, faces
//! inside), generates the mips with [`crate::mip`] (sRGB-correct for sRGB targets, linear float
//! for HDR targets, optional normal-map renormalization), encodes every level with
//! [`crate::encode`] (or converts, for uncompressed targets) and returns a [`Texture`] that
//! writes KTX2, DDS or KTX1.

use crate::mip::MipOptions;
use crate::{Buffer, EncodeOptions, Error, Format, Result, Surface};
use vas::Vector;
use vas::storage::Dynamic;

/// How many mip levels to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mips {
    /// Only the base level.
    None,
    /// The full chain down to 1x1.
    #[default]
    Full,
    /// This many levels (clamped to the full chain).
    Count(u32),
}

/// Options of [`build`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildOptions {
    /// Target format (block-compressed or uncompressed).
    pub format: Format,
    pub encode: EncodeOptions,
    pub mips: Mips,
    pub mip: MipOptions,
    /// Treat RGB as a unit normal (x, y, z mapped from [0, 1] to [-1, 1]) and renormalize
    /// every generated level.
    pub normal_map: bool,
}

impl BuildOptions {
    /// Defaults for `format`: default quality, full mip chain, KTX-Software mip filter.
    pub fn new(format: Format) -> Self {
        BuildOptions {
            format,
            encode: EncodeOptions::default(),
            mips: Mips::Full,
            mip: MipOptions::default(),
            normal_map: false,
        }
    }
}

/// A built texture: every level holds `max(layers, 1) * faces` images (layer, then face) of
/// `format.surface_bytes(level size)` bytes.
#[derive(Debug)]
pub struct Texture {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    /// Array layers; 0 for a non-array texture.
    pub layers: u32,
    /// 1, or 6 for a cubemap.
    pub faces: u32,
    pub levels: Vector<Buffer, Dynamic>,
}

impl Texture {
    fn level_refs(&self) -> Result<Vector<&[u8], Dynamic>> {
        let mut v = Vector::<&[u8], Dynamic>::default();
        v.reserve(self.levels.len())
            .map_err(|_| Error::OutOfMemory)?;
        for l in self.levels.iter() {
            v.push(&l[..]).map_err(|_| Error::OutOfMemory)?;
        }
        Ok(v)
    }

    /// Writes a KTX2 file.
    pub fn to_ktx2(&self, opts: &crate::ktx2::WriteOptions) -> Result<Buffer> {
        let desc = crate::ktx2::Ktx2Desc {
            format: self.format,
            width: self.width,
            height: self.height,
            layers: self.layers,
            faces: self.faces,
            levels: self.levels.len() as u32,
            depth: 0,
        };
        crate::ktx2::write(&desc, &self.level_refs()?, opts)
    }

    /// Writes a DDS file.
    pub fn to_dds(&self, opts: &crate::dds::DdsWriteOptions) -> Result<Buffer> {
        let desc = crate::dds::DdsDesc {
            format: self.format,
            width: self.width,
            height: self.height,
            layers: self.layers,
            faces: self.faces,
            levels: self.levels.len() as u32,
            depth: 0,
        };
        crate::dds::write(&desc, &self.level_refs()?, opts)
    }

    /// Writes a KTX1 file.
    pub fn to_ktx1(&self, key_values: &[(&str, &[u8])]) -> Result<Buffer> {
        let desc = crate::ktx1::Ktx1Desc {
            format: self.format,
            width: self.width,
            height: self.height,
            layers: self.layers,
            faces: self.faces,
            levels: self.levels.len() as u32,
            srgb: self.format.is_srgb(),
            depth: 0,
        };
        crate::ktx1::write(&desc, &self.level_refs()?, key_values)
    }
}

/// The working format mips are generated in for `target`.
fn working_format(target: Format, src: Format) -> Format {
    if target.is_hdr() || matches!(target, Format::Bc6hSfloat) {
        Format::Rgba32Float
    } else if target.is_srgb() || src.is_srgb() {
        Format::Rgba8Srgb
    } else if matches!(
        target,
        Format::Rgba16Unorm | Format::R16Unorm | Format::Rg16Unorm
    ) {
        Format::Rgba16Unorm
    } else if matches!(
        target,
        Format::R8Snorm
            | Format::Rg8Snorm
            | Format::R16Snorm
            | Format::Rg16Snorm
            | Format::Bc4Snorm
            | Format::Bc5Snorm
            | Format::EacR11Snorm
            | Format::EacRg11Snorm
            | Format::Rgb16Float
            | Format::Rgba16Float
            | Format::Rgb9e5Float
    ) {
        Format::Rgba32Float
    } else {
        Format::Rgba8Unorm
    }
}

/// Renormalizes RGB as unit vectors in place (`fmt`: RGBA8 / RGBA8 sRGB label / RGBA16 /
/// RGBA32F working formats).
fn renormalize(fmt: Format, data: &mut [u8]) {
    let n = fmt.block_bytes();
    for px in data.chunks_exact_mut(n) {
        let read = |px: &[u8], c: usize| -> f32 {
            match fmt {
                Format::Rgba32Float => {
                    f32::from_le_bytes([px[c * 4], px[c * 4 + 1], px[c * 4 + 2], px[c * 4 + 3]])
                }
                Format::Rgba16Unorm => {
                    f32::from(u16::from_le_bytes([px[c * 2], px[c * 2 + 1]])) / 65535.0
                }
                _ => f32::from(px[c]) / 255.0,
            }
        };
        let v = [0, 1, 2].map(|c| read(px, c) * 2.0 - 1.0);
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len <= 1e-6 {
            continue;
        }
        for c in 0..3 {
            let u = (v[c] / len + 1.0) * 0.5;
            match fmt {
                Format::Rgba32Float => px[c * 4..c * 4 + 4].copy_from_slice(&u.to_le_bytes()),
                Format::Rgba16Unorm => px[c * 2..c * 2 + 2]
                    .copy_from_slice(&((u.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_le_bytes()),
                _ => px[c] = (u.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            }
        }
    }
}

/// Builds a texture from base `images` (`max(layers, 1) * faces` surfaces of equal size and
/// format, layer then face order; `layers` 0 for a non-array texture, `faces` 1 or 6).
///
/// Errors: `InvalidArgument` (shape, sizes), `Unsupported` (source or target format), plus
/// those of [`crate::mip`] / [`crate::encode`].
pub fn build(images: &[Surface], layers: u32, faces: u32, opts: &BuildOptions) -> Result<Texture> {
    let count = layers.max(1) as usize * faces as usize;
    let first = images.first().ok_or(Error::InvalidArgument)?;
    let (w, h) = (first.width(), first.height());
    if images.len() != count
        || !(faces == 1 || (faces == 6 && w == h))
        || images
            .iter()
            .any(|s| s.width() != w || s.height() != h || s.format() != first.format())
    {
        return Err(Error::InvalidArgument);
    }
    let target = opts.format;
    let full = crate::image::mip_levels(w, h);
    let levels = match opts.mips {
        Mips::None => 1,
        Mips::Full => full,
        Mips::Count(n) => n.clamp(1, full),
    };
    let work = working_format(target, first.format());
    // Per image: the working-format chain.
    let mut out = Vector::<Buffer, Dynamic>::default();
    out.reserve(levels as usize)
        .map_err(|_| Error::OutOfMemory)?;
    for level in 0..levels {
        let (lw, lh) = crate::image::mip_dim(w, h, level);
        let size = target.surface_bytes(lw, lh)?;
        out.push(crate::buffer::zeroed(size * count)?)
            .map_err(|_| Error::OutOfMemory)?;
    }
    for (i, img) in images.iter().enumerate() {
        let base = if img.format() == work {
            let mut b = crate::buffer::zeroed(img.data().len())?;
            b.copy_from_slice(img.data());
            b
        } else {
            crate::convert::convert_to_buffer(img, work)?
        };
        let mut chain =
            crate::mip::generate_mips(&Surface::new(work, w, h, &base)?, levels, &opts.mip)?;
        if opts.normal_map {
            for l in chain.iter_mut() {
                renormalize(work, l);
            }
        }
        for level in 0..levels {
            let (lw, lh) = crate::image::mip_dim(w, h, level);
            let data: &[u8] = if level == 0 {
                &base
            } else {
                &chain[level as usize - 1]
            };
            let s = Surface::new(work, lw, lh, data)?;
            let size = target.surface_bytes(lw, lh)?;
            let dst = &mut out[level as usize][i * size..(i + 1) * size];
            if target.is_compressed() {
                crate::encode(&s, target, &opts.encode, dst)?;
            } else {
                crate::convert::convert(&s, target, dst)?;
            }
        }
    }
    Ok(Texture {
        format: target,
        width: w,
        height: h,
        layers,
        faces,
        levels: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_and_writes_every_container() -> Result<()> {
        let raw = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/images/raw/rgba_64x48.rgba"
        ))
        .unwrap_or_default();
        let s = Surface::new(Format::Rgba8Unorm, 64, 48, &raw)?;
        let t = build(&[s], 0, 1, &BuildOptions::new(Format::Bc7Srgb))?;
        assert_eq!(t.levels.len(), 7);
        let k2 = t.to_ktx2(&crate::ktx2::WriteOptions::default())?;
        let f = crate::ktx2::Ktx2File::parse(&k2)?;
        assert_eq!((f.format(), f.level_count()), (Some(Format::Bc7Srgb), 7));
        let dds = t.to_dds(&crate::dds::DdsWriteOptions::default())?;
        assert_eq!(crate::dds::DdsFile::parse(&dds)?.level_count(), 7);
        let k1 = t.to_ktx1(&[])?;
        assert_eq!(crate::ktx1::Ktx1File::parse(&k1)?.level_count(), 7);
        // Level 1 = the sRGB-correct mip encoded.
        let mip = crate::mip::generate_mips(
            &Surface::new(Format::Rgba8Srgb, 64, 48, &raw)?,
            2,
            &MipOptions::default(),
        )?;
        let want = crate::encode_to_buffer(
            &Surface::new(Format::Rgba8Srgb, 32, 24, &mip[0])?,
            Format::Bc7Srgb,
            &EncodeOptions::default(),
        )?;
        assert_eq!(&t.levels[1][..], &want[..]);
        Ok(())
    }

    #[test]
    fn cubemap_and_normal_maps() -> Result<()> {
        // A tilted normal field: renormalized mips keep unit length.
        let px: Vec<u8> = (0..16 * 16)
            .flat_map(|i| {
                let (x, y) = ((i % 16) as f32 / 15.0, (i / 16) as f32 / 15.0);
                let v = [x - 0.5, y - 0.5, 0.7];
                let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                [v[0] / l, v[1] / l, v[2] / l]
                    .map(|c| ((c + 1.0) * 127.5) as u8)
                    .into_iter()
                    .chain([255])
            })
            .collect();
        let s = Surface::new(Format::Rgba8Unorm, 16, 16, &px)?;
        let opts = BuildOptions {
            normal_map: true,
            ..BuildOptions::new(Format::Rgba8Unorm)
        };
        let t = build(&[s; 6], 0, 6, &opts)?;
        assert_eq!((t.faces, t.levels.len()), (6, 5));
        for l in t.levels.iter().skip(1) {
            for p in l.as_chunks::<4>().0 {
                let v = [0, 1, 2].map(|c| f32::from(p[c]) / 255.0 * 2.0 - 1.0);
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                assert!((len - 1.0).abs() < 0.02, "{len}");
            }
        }
        assert!(build(&[s], 0, 6, &opts).is_err());
        Ok(())
    }
}
