//! RenderMan zfile depth maps (own implementation).
//!
//! Layout (PRMan `zfile` display driver; OpenImageIO `zfile` plugin as the format reference):
//!
//! | Offset | Size | Field |
//! |---|---|---|
//! | 0 | 4 | magic `0x2f0867ab` (native byte order of the writer) |
//! | 4 | 2 | width (u16) |
//! | 6 | 2 | height (u16) |
//! | 8 | 64 | world-to-screen matrix, 16 f32 |
//! | 72 | 64 | world-to-camera matrix, 16 f32 |
//! | 136 | 4 * width * height | depths, f32, rows top to bottom |
//!
//! Every field uses the byte order the magic shows (`ab 67 08 2f` little endian,
//! `2f 08 67 ab` big endian); both decode. Matrices are kept as the 16 stored floats (row
//! major, Imath `M44f` / RenderMan convention). Gzip-compressed zfiles (an OpenImageIO option)
//! are not recognized.
//!
//! Decodes to `R32Float`; [`decode_zfile`] also returns the matrices. Encoding takes
//! `R32Float` (or `R16Float`, `R8Unorm`, `R16Unorm`, converted losslessly to f32) and writes
//! the matrices of [`EncodeOptions`] (identity by default, as OpenImageIO) in little endian
//! unless [`EncodeOptions::big_endian`].

use super::Image;
use crate::convert::convert_to_buffer;
use crate::{Buffer, Error, Format, Result, Surface};

/// The zfile magic number.
pub const MAGIC: u32 = 0x2f08_67ab;
/// Header size in bytes.
pub const HEADER_BYTES: usize = 136;

const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// Encoder options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    /// World-to-screen matrix (default identity).
    pub world_to_screen: [f32; 16],
    /// World-to-camera matrix (default identity).
    pub world_to_camera: [f32; 16],
    /// Write big endian (default little endian).
    pub big_endian: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        EncodeOptions {
            world_to_screen: IDENTITY,
            world_to_camera: IDENTITY,
            big_endian: false,
        }
    }
}

/// A decoded zfile: depths and the camera matrices.
#[derive(Debug)]
pub struct ZfileImage {
    /// `R32Float` depths.
    pub image: Image,
    pub world_to_screen: [f32; 16],
    pub world_to_camera: [f32; 16],
    /// The file was big endian.
    pub big_endian: bool,
}

/// Decodes the main image.
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_zfile(data).map(|z| z.image)
}

/// Decodes a zfile with its matrices.
pub fn decode_zfile(d: &[u8]) -> Result<ZfileImage> {
    let h: &[u8; HEADER_BYTES] = d.first_chunk().ok_or(Error::InvalidData)?;
    let magic = [h[0], h[1], h[2], h[3]];
    let big_endian = if magic == MAGIC.to_le_bytes() {
        false
    } else if magic == MAGIC.to_be_bytes() {
        true
    } else {
        return Err(Error::InvalidData);
    };
    let u16_at = |o: usize| {
        let b = [h[o], h[o + 1]];
        if big_endian {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        }
    };
    let f32_of = |b: &[u8]| {
        let b = [b[0], b[1], b[2], b[3]];
        if big_endian {
            f32::from_be_bytes(b)
        } else {
            f32::from_le_bytes(b)
        }
    };
    let (w, ht) = (u16_at(4) as usize, u16_at(6) as usize);
    if w == 0 || ht == 0 {
        return Err(Error::InvalidData);
    }
    let matrix = |o: usize| {
        let mut m = [0f32; 16];
        for (v, b) in m.iter_mut().zip(h[o..o + 64].as_chunks::<4>().0) {
            *v = f32_of(b);
        }
        m
    };
    let (world_to_screen, world_to_camera) = (matrix(8), matrix(72));
    let src = d
        .get(HEADER_BYTES..HEADER_BYTES + w * ht * 4)
        .ok_or(Error::InvalidData)?;
    let mut image = Image::new(Format::R32Float, w as u32, ht as u32)?;
    for (o, s) in image
        .data
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<4>().0)
    {
        o.copy_from_slice(&f32_of(s).to_le_bytes());
    }
    Ok(ZfileImage {
        image,
        world_to_screen,
        world_to_camera,
        big_endian,
    })
}

/// Encodes `src` (single channel: `R32Float`, `R16Float`, `R8Unorm`, `R16Unorm`).
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    let (w, h) = (src.width(), src.height());
    if w > 0xFFFF || h > 0xFFFF {
        return Err(Error::Unsupported);
    }
    let converted;
    let depths: &[u8] = match src.format() {
        Format::R32Float => src.data(),
        Format::R16Float | Format::R8Unorm | Format::R16Unorm => {
            converted = convert_to_buffer(src, Format::R32Float)?;
            &converted
        }
        _ => return Err(Error::Unsupported),
    };
    let be = opts.big_endian;
    let mut out = crate::buffer::zeroed(HEADER_BYTES + depths.len())?;
    let mut hd = [0u8; HEADER_BYTES];
    let mut put = |o: usize, b: &[u8]| {
        if let Some(d) = hd.get_mut(o..o + b.len()) {
            d.copy_from_slice(b);
        }
    };
    let f = |v: f32| if be { v.to_be_bytes() } else { v.to_le_bytes() };
    put(
        0,
        &if be {
            MAGIC.to_be_bytes()
        } else {
            MAGIC.to_le_bytes()
        },
    );
    for (o, v) in [(4, w as u16), (6, h as u16)] {
        put(o, &if be { v.to_be_bytes() } else { v.to_le_bytes() });
    }
    for (i, v) in opts.world_to_screen.iter().enumerate() {
        put(8 + 4 * i, &f(*v));
    }
    for (i, v) in opts.world_to_camera.iter().enumerate() {
        put(72 + 4 * i, &f(*v));
    }
    let (head, body) = out.split_at_mut(HEADER_BYTES);
    head.copy_from_slice(&hd);
    for (o, s) in body
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(depths.as_chunks::<4>().0)
    {
        o.copy_from_slice(&f(f32::from_le_bytes([s[0], s[1], s[2], s[3]])));
    }
    Ok(out)
}
