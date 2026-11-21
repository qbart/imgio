//! `.astc` files written by ARM astcenc 6.0 (testdata/astc/files, `astcenc -cl ... -fast`):
//! parsed, rewritten byte for byte, and decoded by imgio exactly like `astcenc -dl` (its
//! legacy-header RGBA8 DDS output, read with `imgio::dds`).

use imgio::astc_file::{self, AstcFile};
use imgio::dds::DdsFile;
use imgio::{AstcKind, Format};
use std::error::Error;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/astc/files/");

#[test]
fn astcenc_files() -> Result<(), Box<dyn Error>> {
    for (b, dim) in [("4x4", (4, 4)), ("6x5", (6, 5)), ("12x12", (12, 12))] {
        let data = std::fs::read(format!("{DIR}rgba_64x48_{b}.astc"))?;
        let f = AstcFile::parse(&data)?;
        assert_eq!(
            (f.block(), f.width(), f.height(), f.depth()),
            ((dim.0, dim.1, 1), 64, 48, 1)
        );
        let out = astc_file::write(f.block(), 64, 48, 1, f.data())?;
        assert!(out[..] == data[..], "{b}: rewrite differs");
        let format = f.format(AstcKind::Unorm).ok_or("2D block")?;
        let ours = imgio::decode_to_buffer(format, f.data(), 64, 48)?;
        let dds = std::fs::read(format!("{DIR}rgba_64x48_{b}.decoded.dds"))?;
        let d = DdsFile::parse(&dds)?;
        assert_eq!(d.format(), Format::Rgba8Unorm);
        assert!(
            &ours[..] == d.image_data(0, 0, 0)?,
            "{b}: decode differs from astcenc"
        );
    }
    assert!(AstcFile::parse(&[0u8; 16]).is_err());
    assert!(astc_file::write((4, 4, 1), 4, 4, 1, &[0; 15]).is_err());
    Ok(())
}
