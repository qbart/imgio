//! ASTC decode vs. the ktx-rs port of Arm astcenc (testdata/astc, tools/astc-golden):
//! encoded synthetic LDR/sRGB/HDR images and random blocks at several block sizes.

use imgio::{AstcBlock, AstcKind, Format};
use std::error::Error;

#[test]
fn matches_astcenc() -> Result<(), Box<dyn Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/astc/");
    let cases = std::fs::read_to_string(format!("{dir}cases.txt"))?;
    let mut failures = Vec::new();
    for line in cases.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let [name, kind, bx, by, w, h] = f[..] else {
            return Err(format!("bad line {line}").into());
        };
        let kind = match kind {
            "srgb" => AstcKind::Srgb,
            "float" => AstcKind::Float,
            _ => AstcKind::Unorm,
        };
        let block = AstcBlock::from_dim(bx.parse()?, by.parse()?).ok_or("block size")?;
        let (w, h): (u32, u32) = (w.parse()?, h.parse()?);
        let src = std::fs::read(format!("{dir}{name}.astc"))?;
        let want = std::fs::read(format!("{dir}{name}.expected"))?;
        let got = imgio::decode_to_buffer(Format::Astc(block, kind), &src, w, h)?;
        let px = if kind == AstcKind::Float { 8 } else { 4 };
        let bad = got
            .chunks_exact(px)
            .zip(want.chunks_exact(px))
            .enumerate()
            .filter(|(_, (g, w))| g != w)
            .map(|(i, (g, w))| format!("px {i}: {g:?} vs {w:?}"))
            .collect::<Vec<_>>();
        if !bad.is_empty() {
            failures.push(format!("{name}: {} px differ, first {}", bad.len(), bad[0]));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
