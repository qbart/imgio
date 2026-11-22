//! JPEG (decode only).
//!
//! Needs the `jpeg` cargo feature (crate `jpeg-decoder`); without it every function returns
//! `Unsupported`.
//!
//! - Baseline / extended (SOF0 / SOF1) and progressive (SOF2) 8-bit DCT files, restart
//!   markers, any chroma subsampling (4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1), and lossless
//!   (SOF3) files of 2..16 bits.
//! - Gray files decode to `R8Unorm` (lossless > 8 bits: `R16Unorm`); YCbCr, RGB, CMYK and
//!   YCCK (Adobe APP14 transform 0 / 2) files to `Rgb8Srgb` (lossless RGB > 8 bits:
//!   `Rgba16Unorm`). CMYK -> RGB is Pillow's `R = (255 - K) - C * (255 - K) / 255`.
//! - 12-bit DCT, hierarchical and arithmetic-coded files are `Unsupported` (no pure Rust
//!   decoder handles them).
//! - The EXIF orientation is metadata ([`decode_with_metadata`], [`exif_orientation`]): pixels
//!   are returned as stored, not rotated.
//!
//! Output is close to libjpeg-turbo but not bit-exact: `jpeg-decoder` uses its own integer
//! IDCT and YCbCr -> RGB fixed point (measured bounds in `docs/spec/file_jpeg.md`); lossless
//! files are exact.

use super::Image;
use crate::{Buffer, Result, Surface};

/// Encoder options (there is no JPEG encoder: [`encode`] returns `Unsupported`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EncodeOptions {}

/// Largest image decoded, in pixels.
pub const MAX_PIXELS: u64 = 1 << 28;
/// Largest pixel count decoded per byte of input. Every 8x8 block of every component costs at
/// least one bit (its DC code), so a valid file has at most 512 pixels per byte; larger
/// ratios are truncated or hostile headers.
pub const MAX_PIXELS_PER_BYTE: u64 = 4096;

/// File metadata returned with the pixels.
#[derive(Debug, Default)]
pub struct Metadata {
    /// EXIF orientation (1..=8, 1 = as stored; see [`exif_orientation`]). The pixels are
    /// not rotated: apply it yourself when displaying.
    pub orientation: u8,
    /// Raw EXIF payload (TIFF header onwards), if present.
    pub exif: Option<Buffer>,
    /// ICC profile (APP2 chunks joined), if present.
    pub icc_profile: Option<Buffer>,
}

/// Decodes the image (`R8Unorm` for gray files, `Rgb8Srgb` otherwise).
pub fn decode(data: &[u8]) -> Result<Image> {
    decode_with_metadata(data).map(|(img, _)| img)
}

/// Decodes the image and returns its metadata (EXIF orientation, EXIF, ICC profile).
pub fn decode_with_metadata(data: &[u8]) -> Result<(Image, Metadata)> {
    #[cfg(feature = "jpeg")]
    {
        imp::decode(data)
    }
    #[cfg(not(feature = "jpeg"))]
    {
        let _ = data;
        Err(crate::Error::Unsupported)
    }
}

/// There is no JPEG encoder: always `Unsupported`.
pub fn encode(src: &Surface, opts: &EncodeOptions) -> Result<Buffer> {
    let _ = (src, opts);
    Err(crate::Error::Unsupported)
}

/// The orientation tag (0x0112) of an EXIF payload: its TIFF header (`II*\0` / `MM\0*`)
/// onwards, optionally preceded by `Exif\0\0`. `None` without a valid tag (1..=8).
///
/// 1 = as stored, 2 = mirrored horizontally, 3 = rotated 180, 4 = mirrored vertically,
/// 5 = transposed, 6 = rotated 90 clockwise to display, 7 = transverse, 8 = rotated 90
/// counter-clockwise.
pub fn exif_orientation(exif: &[u8]) -> Option<u8> {
    let t = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    let be = match t.get(..4)? {
        b"II*\0" => false,
        b"MM\0*" => true,
        _ => return None,
    };
    let u16_at = |o: usize| {
        let b = t.get(o..o + 2)?;
        Some(if be {
            u16::from_be_bytes([b[0], b[1]])
        } else {
            u16::from_le_bytes([b[0], b[1]])
        })
    };
    let u32_at = |o: usize| {
        let b = t.get(o..o + 4)?;
        let a = [b[0], b[1], b[2], b[3]];
        Some(if be {
            u32::from_be_bytes(a)
        } else {
            u32::from_le_bytes(a)
        })
    };
    let ifd = u32_at(4)? as usize;
    let count = u16_at(ifd)? as usize;
    for i in 0..count.min(512) {
        let e = ifd.checked_add(2 + 12 * i)?;
        if u16_at(e)? == 0x0112 {
            // SHORT, count 1: the value sits in the first two bytes of the value field.
            if u16_at(e + 2)? != 3 || u32_at(e + 4)? != 1 {
                return None;
            }
            let v = u16_at(e + 8)?;
            return (1..=8).contains(&v).then_some(v as u8);
        }
    }
    None
}

#[cfg(feature = "jpeg")]
mod imp {
    use super::{MAX_PIXELS, MAX_PIXELS_PER_BYTE, Metadata, exif_orientation};
    use crate::file::Image;
    use crate::{Buffer, Error, Format, Result};
    use jpeg_decoder::{Decoder, PixelFormat};
    use std::io::Cursor;

    fn map_err(e: jpeg_decoder::Error) -> Error {
        match e {
            jpeg_decoder::Error::Unsupported(_) => Error::Unsupported,
            _ => Error::InvalidData,
        }
    }

    /// The frame header (SOFn) fields imgio needs.
    struct FrameHeader {
        lossless: bool,
        precision: u8,
        components: u8,
    }

    /// Walks the markers up to the frame header. `Unsupported` for coding processes
    /// jpeg-decoder does not decode (hierarchical, arithmetic) and DCT precisions != 8.
    fn frame_header(data: &[u8]) -> Result<FrameHeader> {
        if data.get(..2) != Some(&[0xFF, 0xD8]) {
            return Err(Error::InvalidData);
        }
        let mut pos = 2;
        loop {
            // Fill bytes before a marker.
            while data.get(pos) == Some(&0xFF) && data.get(pos + 1) == Some(&0xFF) {
                pos += 1;
            }
            let (Some(&0xFF), Some(&m)) = (data.get(pos), data.get(pos + 1)) else {
                return Err(Error::InvalidData);
            };
            pos += 2;
            if (0xD0..=0xD7).contains(&m) || m == 0x01 {
                continue;
            }
            let seg = data.get(pos..pos + 2).ok_or(Error::InvalidData)?;
            let len = usize::from(u16::from_be_bytes([seg[0], seg[1]]));
            match m {
                0xC0..=0xC3 => {
                    let h = data.get(pos + 2..pos + 8).ok_or(Error::InvalidData)?;
                    let (lossless, precision) = (m == 0xC3, h[0]);
                    if !lossless && precision != 8 {
                        return Err(Error::Unsupported);
                    }
                    return Ok(FrameHeader {
                        lossless,
                        precision,
                        components: h[5],
                    });
                }
                0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => return Err(Error::Unsupported),
                0xDA | 0xD9 => return Err(Error::InvalidData),
                _ => {}
            }
            if len < 2 {
                return Err(Error::InvalidData);
            }
            pos += len;
        }
    }

    fn to_buffer(b: &[u8]) -> Result<Buffer> {
        let mut out = Buffer::default();
        out.extend_from_slice(b).map_err(|_| Error::OutOfMemory)?;
        Ok(out)
    }

    /// Pillow's `MULDIV255`: `a * b / 255`, rounded.
    fn muldiv255(a: u32, b: u32) -> u32 {
        let t = a * b + 128;
        ((t >> 8) + t) >> 8
    }

    pub fn decode(data: &[u8]) -> Result<(Image, Metadata)> {
        let fh = frame_header(data)?;
        let mut dec = Decoder::new(Cursor::new(data));
        dec.read_info().map_err(map_err)?;
        let info = dec.info().ok_or(Error::InvalidData)?;
        let (w, h) = (u32::from(info.width), u32::from(info.height));
        let px = u64::from(w) * u64::from(h);
        if w == 0 || h == 0 {
            return Err(Error::InvalidData);
        }
        if px > MAX_PIXELS {
            return Err(Error::Unsupported);
        }
        if px > (data.len() as u64).saturating_mul(MAX_PIXELS_PER_BYTE) {
            return Err(Error::InvalidData);
        }
        let wide = fh.lossless && fh.precision > 8;
        let samples = usize::from(fh.components) * if wide { 2 } else { 1 };
        dec.set_max_decoding_buffer_size(px as usize * samples.max(1));
        let pixels = dec.decode().map_err(map_err)?;
        let n = px as usize;
        let out_fmt = match (info.pixel_format, wide) {
            (PixelFormat::L8, _) => Format::R8Unorm,
            (PixelFormat::L16, _) => Format::R16Unorm,
            (PixelFormat::RGB24, false) | (PixelFormat::CMYK32, false) => Format::Rgb8Srgb,
            (PixelFormat::RGB24, true) => Format::Rgba16Unorm,
            _ => return Err(Error::Unsupported),
        };
        let mut img = Image::new(out_fmt, w, h)?;
        let in_samples = match info.pixel_format {
            PixelFormat::L8 => 1,
            PixelFormat::L16 => 2,
            PixelFormat::RGB24 if wide => 6,
            PixelFormat::RGB24 => 3,
            PixelFormat::CMYK32 => 4,
        };
        if pixels.len() != n * in_samples {
            return Err(Error::InvalidData);
        }
        // Lossless > 8 bits: native-endian samples of `precision` bits, scaled to 16 bits.
        let max = (1u32 << fh.precision.clamp(2, 16)) - 1;
        let scale16 = |b: &[u8]| -> [u8; 2] {
            let v = u32::from(u16::from_ne_bytes([b[0], b[1]])).min(max);
            (((v * 65535 + max / 2) / max) as u16).to_le_bytes()
        };
        match (info.pixel_format, wide) {
            (PixelFormat::CMYK32, _) => {
                for (o, p) in img
                    .data
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(pixels.as_chunks::<4>().0)
                {
                    let nk = 255 - u32::from(p[3]);
                    for c in 0..3 {
                        o[c] = (nk - muldiv255(u32::from(p[c]), nk)) as u8;
                    }
                }
            }
            (PixelFormat::L16, _) => {
                for (o, p) in img
                    .data
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(pixels.as_chunks::<2>().0)
                {
                    *o = scale16(p);
                }
            }
            (PixelFormat::RGB24, true) => {
                for (o, p) in img
                    .data
                    .as_chunks_mut::<8>()
                    .0
                    .iter_mut()
                    .zip(pixels.as_chunks::<6>().0)
                {
                    o[0..2].copy_from_slice(&scale16(&p[0..2]));
                    o[2..4].copy_from_slice(&scale16(&p[2..4]));
                    o[4..6].copy_from_slice(&scale16(&p[4..6]));
                    o[6..8].copy_from_slice(&[255, 255]);
                }
            }
            _ => img.data.copy_from_slice(&pixels),
        }

        let mut meta = Metadata {
            orientation: 1,
            ..Metadata::default()
        };
        if let Some(exif) = dec.exif_data() {
            meta.orientation = exif_orientation(exif).unwrap_or(1);
            meta.exif = Some(to_buffer(exif)?);
        }
        if let Some(icc) = dec.icc_profile() {
            meta.icc_profile = Some(to_buffer(&icc)?);
        }
        Ok((img, meta))
    }
}
