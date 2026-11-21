//! Image file formats: decode files from byte slices, encode [`Surface`]s to bytes.
//!
//! [`FileFormat::detect`] identifies a file from its first bytes; [`decode`] returns its main
//! image, [`decode_animation`] every frame (animated GIF / APNG / WebP; one frame otherwise),
//! [`encode`] writes a surface with the format's default options. Each format module
//! (`imgio::file::png`, ...) also has its own `decode` / `encode` with format-specific options.
//!
//! Formats backed by a pure Rust crate are behind a cargo feature of the same name (`png`,
//! `gif`, `webp`, `jpeg`, `exr`, `tiff`, `jxl`, `jpeg2000`, `vdb`, `avif`); without it their
//! functions return [`Error::Unsupported`]. BMP, TGA, Radiance HDR and RenderMan zfile are
//! imgio's own code and always available.
//!
//! Decoded pixel formats (native depth, no silent precision loss):
//! - 8-bit gray `R8Unorm`, gray + alpha `Rg8Unorm` (R = gray, G = alpha), RGB `Rgb8Srgb`,
//!   RGBA `Rgba8Srgb` (8-bit color in these files is sRGB encoded; relabel with
//!   [`Image::with_format`] if a file is known to be linear).
//! - 16-bit gray `R16Unorm`, gray + alpha `Rg16Unorm`, RGB / RGBA `Rgba16Unorm` (RGB: alpha
//!   65535).
//! - Half float RGB `Rgb16Float`, RGBA `Rgba16Float`; 32-bit float `Rgba32Float`.
//! - Palette, 1/2/4-bit and packed (5:6:5, 5:5:5) samples are expanded to the 8-bit formats.
//! - Animation frames are fully composited, canvas-sized images.

use crate::{Buffer, Error, Format, Result, Surface};
use vas::Vector;
use vas::storage::Dynamic;

pub mod avif;
pub mod bmp;
pub mod exr;
pub mod gif;
pub mod hdr;
pub mod jpeg;
pub mod jpeg2000;
pub mod jxl;
pub mod png;
pub mod tga;
pub mod tiff;
pub mod vdb;
pub mod webp;
pub mod zfile;

/// An image file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileFormat {
    /// Windows / OS/2 bitmap (`.bmp`).
    Bmp,
    /// Truevision TGA (`.tga`).
    Tga,
    /// GIF, including animation (`.gif`).
    Gif,
    /// WebP, including animation (`.webp`).
    WebP,
    /// PNG, including APNG (`.png`).
    Png,
    /// JPEG (`.jpg`).
    Jpeg,
    /// OpenEXR (`.exr`).
    Exr,
    /// TIFF and BigTIFF (`.tif`).
    Tiff,
    /// Radiance RGBE (`.hdr`).
    Hdr,
    /// JPEG XL, codestream or container (`.jxl`).
    JpegXl,
    /// JPEG 2000, JP2 container or J2K codestream (`.jp2`, `.j2k`).
    Jpeg2000,
    /// OpenVDB volume (`.vdb`).
    Vdb,
    /// RenderMan depth file (`.zfile`).
    Zfile,
    /// AVIF (`.avif`).
    Avif,
}

/// Every [`FileFormat`].
pub const ALL: [FileFormat; 14] = [
    FileFormat::Bmp,
    FileFormat::Tga,
    FileFormat::Gif,
    FileFormat::WebP,
    FileFormat::Png,
    FileFormat::Jpeg,
    FileFormat::Exr,
    FileFormat::Tiff,
    FileFormat::Hdr,
    FileFormat::JpegXl,
    FileFormat::Jpeg2000,
    FileFormat::Vdb,
    FileFormat::Zfile,
    FileFormat::Avif,
];

impl FileFormat {
    /// Identifies a file from its first bytes. TGA has no signature: it is recognized by its
    /// 2.0 footer, or (last) by a plausible header.
    pub fn detect(data: &[u8]) -> Option<FileFormat> {
        let starts = |m: &[u8]| data.starts_with(m);
        let fmt = if starts(b"\x89PNG\r\n\x1a\n") {
            FileFormat::Png
        } else if starts(b"GIF87a") || starts(b"GIF89a") {
            FileFormat::Gif
        } else if starts(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
            FileFormat::WebP
        } else if starts(&[0xFF, 0xD8, 0xFF]) {
            FileFormat::Jpeg
        } else if starts(&[0x76, 0x2F, 0x31, 0x01]) {
            FileFormat::Exr
        } else if starts(b"II*\0") || starts(b"MM\0*") || starts(b"II+\0") || starts(b"MM\0+") {
            FileFormat::Tiff
        } else if starts(b"#?RADIANCE") || starts(b"#?RGBE") {
            FileFormat::Hdr
        } else if starts(&[0xFF, 0x0A]) || starts(b"\0\0\0\x0cJXL \r\n\x87\n") {
            FileFormat::JpegXl
        } else if starts(b"\0\0\0\x0cjP  \r\n\x87\n") || starts(&[0xFF, 0x4F, 0xFF, 0x51]) {
            FileFormat::Jpeg2000
        } else if starts(&[0x20, 0x42, 0x44, 0x56]) {
            FileFormat::Vdb
        } else if starts(&[0xAB, 0x67, 0x08, 0x2F]) || starts(&[0x2F, 0x08, 0x67, 0xAB]) {
            FileFormat::Zfile
        } else if is_avif(data) {
            FileFormat::Avif
        } else if starts(b"BM") && data.len() >= 26 {
            FileFormat::Bmp
        } else if tga::looks_like_tga(data) {
            FileFormat::Tga
        } else {
            return None;
        };
        Some(fmt)
    }

    /// Usual file extension (lowercase, without the dot).
    pub fn extension(self) -> &'static str {
        match self {
            FileFormat::Bmp => "bmp",
            FileFormat::Tga => "tga",
            FileFormat::Gif => "gif",
            FileFormat::WebP => "webp",
            FileFormat::Png => "png",
            FileFormat::Jpeg => "jpg",
            FileFormat::Exr => "exr",
            FileFormat::Tiff => "tif",
            FileFormat::Hdr => "hdr",
            FileFormat::JpegXl => "jxl",
            FileFormat::Jpeg2000 => "jp2",
            FileFormat::Vdb => "vdb",
            FileFormat::Zfile => "zfile",
            FileFormat::Avif => "avif",
        }
    }

    /// The format of a file extension (case-insensitive, without the dot).
    pub fn from_extension(ext: &str) -> Option<FileFormat> {
        let e = ext.to_ascii_lowercase();
        Some(match e.as_str() {
            "bmp" | "dib" => FileFormat::Bmp,
            "tga" | "icb" | "vda" | "vst" => FileFormat::Tga,
            "gif" => FileFormat::Gif,
            "webp" => FileFormat::WebP,
            "png" | "apng" => FileFormat::Png,
            "jpg" | "jpeg" | "jpe" | "jfif" => FileFormat::Jpeg,
            "exr" => FileFormat::Exr,
            "tif" | "tiff" => FileFormat::Tiff,
            "hdr" | "rgbe" | "pic" => FileFormat::Hdr,
            "jxl" => FileFormat::JpegXl,
            "jp2" | "j2k" | "j2c" | "jpc" | "jpx" => FileFormat::Jpeg2000,
            "vdb" => FileFormat::Vdb,
            "zfile" | "z" => FileFormat::Zfile,
            "avif" => FileFormat::Avif,
            _ => return None,
        })
    }
}

/// ISO BMFF `ftyp` box with an AVIF major or compatible brand.
fn is_avif(data: &[u8]) -> bool {
    if data.get(4..8) != Some(b"ftyp") {
        return false;
    }
    let size = data
        .get(..4)
        .map_or(0, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .min(data.len());
    let brands = |b: &[u8]| b == b"avif" || b == b"avis";
    data.get(8..12).is_some_and(brands)
        || data
            .get(16..size.max(16))
            .is_some_and(|c| c.as_chunks::<4>().0.iter().any(|b| brands(b)))
}

/// An owned image: `format` pixels, tightly packed rows.
#[derive(Debug)]
pub struct Image {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub data: Buffer,
}

impl Image {
    /// A zeroed image.
    pub fn new(format: Format, width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidArgument);
        }
        let data = crate::buffer::zeroed(format.surface_bytes(width, height)?)?;
        Ok(Image {
            format,
            width,
            height,
            data,
        })
    }

    /// Copies a surface.
    pub fn from_surface(src: &Surface) -> Result<Self> {
        let mut img = Image::new(src.format(), src.width(), src.height())?;
        img.data.copy_from_slice(src.data());
        Ok(img)
    }

    /// A view of the pixels.
    pub fn surface(&self) -> Result<Surface<'_>> {
        Surface::new(self.format, self.width, self.height, &self.data)
    }

    /// The same pixels with another label of the same layout (e.g. `Rgba8Srgb` ->
    /// `Rgba8Unorm` for a linear file); `InvalidArgument` if the byte sizes differ.
    pub fn with_format(mut self, format: Format) -> Result<Self> {
        if format.surface_bytes(self.width, self.height)? != self.data.len() {
            return Err(Error::InvalidArgument);
        }
        self.format = format;
        Ok(self)
    }
}

/// One frame of an animation.
#[derive(Debug)]
pub struct Frame {
    /// The composited canvas after this frame.
    pub image: Image,
    /// Display time as a fraction of a second (`numerator`, `denominator`).
    pub delay: (u32, u32),
}

/// Frames of an animated file (a still image is one frame with zero delay).
#[derive(Debug)]
pub struct Animation {
    pub frames: Vector<Frame, Dynamic>,
    /// Number of plays; 0 = loop forever.
    pub loops: u32,
}

impl Animation {
    /// A one-frame animation.
    pub fn still(image: Image) -> Result<Self> {
        let mut frames = Vector::<Frame, Dynamic>::default();
        frames
            .push(Frame {
                image,
                delay: (0, 1),
            })
            .map_err(|_| Error::OutOfMemory)?;
        Ok(Animation { frames, loops: 0 })
    }
}

/// Decodes the main image of a file (the first frame of an animation, the first page /
/// part / layer of multi-image formats), detecting its format.
pub fn decode(data: &[u8]) -> Result<Image> {
    let fmt = FileFormat::detect(data).ok_or(Error::Unsupported)?;
    decode_as(fmt, data)
}

/// Decodes the main image of a file of a known format.
pub fn decode_as(fmt: FileFormat, data: &[u8]) -> Result<Image> {
    match fmt {
        FileFormat::Bmp => bmp::decode(data),
        FileFormat::Tga => tga::decode(data),
        FileFormat::Gif => gif::decode(data),
        FileFormat::WebP => webp::decode(data),
        FileFormat::Png => png::decode(data),
        FileFormat::Jpeg => jpeg::decode(data),
        FileFormat::Exr => exr::decode(data),
        FileFormat::Tiff => tiff::decode(data),
        FileFormat::Hdr => hdr::decode(data),
        FileFormat::JpegXl => jxl::decode(data),
        FileFormat::Jpeg2000 => jpeg2000::decode(data),
        FileFormat::Vdb => vdb::decode(data),
        FileFormat::Zfile => zfile::decode(data),
        FileFormat::Avif => avif::decode(data),
    }
}

/// Decodes every frame of a file (GIF, APNG and WebP animations; a still image for the other
/// formats).
pub fn decode_animation(data: &[u8]) -> Result<Animation> {
    match FileFormat::detect(data).ok_or(Error::Unsupported)? {
        FileFormat::Gif => gif::decode_animation(data),
        FileFormat::Png => png::decode_animation(data),
        FileFormat::WebP => webp::decode_animation(data),
        fmt => Animation::still(decode_as(fmt, data)?),
    }
}

/// Encodes `src` with the format's default options. `Unsupported` for formats without an
/// encoder (JPEG, JPEG 2000, OpenVDB) or a pixel format the encoder cannot store.
pub fn encode(fmt: FileFormat, src: &Surface) -> Result<Buffer> {
    match fmt {
        FileFormat::Bmp => bmp::encode(src, &bmp::EncodeOptions::default()),
        FileFormat::Tga => tga::encode(src, &tga::EncodeOptions::default()),
        FileFormat::Gif => gif::encode(src, &gif::EncodeOptions::default()),
        FileFormat::WebP => webp::encode(src, &webp::EncodeOptions::default()),
        FileFormat::Png => png::encode(src, &png::EncodeOptions::default()),
        FileFormat::Exr => exr::encode(src, &exr::EncodeOptions::default()),
        FileFormat::Tiff => tiff::encode(src, &tiff::EncodeOptions::default()),
        FileFormat::Hdr => hdr::encode(src, &hdr::EncodeOptions::default()),
        FileFormat::JpegXl => jxl::encode(src, &jxl::EncodeOptions::default()),
        FileFormat::Zfile => zfile::encode(src, &zfile::EncodeOptions::default()),
        FileFormat::Avif => avif::encode(src, &avif::EncodeOptions::default()),
        FileFormat::Jpeg | FileFormat::Jpeg2000 | FileFormat::Vdb => Err(Error::Unsupported),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_signatures() {
        let cases: [(&[u8], FileFormat); 13] = [
            (b"\x89PNG\r\n\x1a\n....", FileFormat::Png),
            (b"GIF89a......", FileFormat::Gif),
            (b"RIFF\0\0\0\0WEBPVP8 ", FileFormat::WebP),
            (&[0xFF, 0xD8, 0xFF, 0xE0], FileFormat::Jpeg),
            (&[0x76, 0x2F, 0x31, 0x01, 2, 0, 0, 0], FileFormat::Exr),
            (b"MM\0*\0\0\0\x08", FileFormat::Tiff),
            (b"#?RADIANCE\n", FileFormat::Hdr),
            (&[0xFF, 0x0A, 0, 0], FileFormat::JpegXl),
            (b"\0\0\0\x0cjP  \r\n\x87\n", FileFormat::Jpeg2000),
            (&[0x20, 0x42, 0x44, 0x56, 0, 0, 0, 0], FileFormat::Vdb),
            (&[0xAB, 0x67, 0x08, 0x2F], FileFormat::Zfile),
            (b"\0\0\0\x1cftypavif\0\0\0\0mif1miaf", FileFormat::Avif),
            (
                b"BM\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
                FileFormat::Bmp,
            ),
        ];
        for (data, fmt) in cases {
            assert_eq!(FileFormat::detect(data), Some(fmt), "{fmt:?}");
            assert_eq!(FileFormat::from_extension(fmt.extension()), Some(fmt));
        }
        assert_eq!(FileFormat::detect(b"hello"), None);
    }
}
