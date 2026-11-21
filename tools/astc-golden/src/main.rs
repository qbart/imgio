//! Writes testdata/astc: synthetic images compressed and decompressed by the ktx-rs port of
//! Arm astcenc, plus random blocks (error paths, all modes) decompressed by it.
//! Run: cargo run --release --manifest-path tools/astc-golden/Cargo.toml

use ktx::astcenc::*;
use std::fmt::Write as _;
use std::sync::Mutex;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFF) as f32 / 65535.0
    }
}

/// Gradients, a hard edge, noise and an alpha ramp; values scaled by `scale` (HDR > 1).
fn image(w: u32, h: u32, scale: f32, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            let edge = if (x / 7 + y / 5) % 2 == 0 { 0.2 } else { 0.0 };
            let n = rng.unit() * 0.15;
            out.extend_from_slice(&[
                (u + edge + n).min(1.0) * scale,
                (v * 0.8 + n).min(1.0) * scale,
                ((1.0 - u) * v + edge).min(1.0) * scale,
                if scale > 1.0 { 1.0 } else { (0.3 + u * 0.7).min(1.0) },
            ]);
        }
    }
    out
}

fn profile(kind: &str) -> AstcencProfile {
    match kind {
        "srgb" => AstcencProfile::LdrSrgb,
        "float" => AstcencProfile::Hdr,
        _ => AstcencProfile::Ldr,
    }
}

fn ctx(kind: &str, bx: u32, by: u32) -> Result<Box<AstcencContext>, AstcencError> {
    let flags = if kind == "float" { 0 } else { ASTCENC_FLG_USE_DECODE_UNORM8 };
    let cfg = astcenc_config_init(profile(kind), bx, by, 1, 60.0, flags)?;
    astcenc_context_alloc(&cfg, 1)
}

fn decompress(kind: &str, bx: u32, by: u32, w: u32, h: u32, blocks: &[u8]) -> Result<Vec<u8>, AstcencError> {
    let mut c = ctx(kind, bx, by)?;
    let n = (w * h * 4) as usize;
    let data = if kind == "float" {
        AstcencImageData::F16(vec![vec![0; n]])
    } else {
        AstcencImageData::U8(vec![vec![0; n]])
    };
    let mut img = AstcencImage { dim_x: w, dim_y: h, dim_z: 1, data };
    astcenc_decompress_image(&mut c, blocks, &mut img, &AstcencSwizzle::RGBA, 0)?;
    Ok(match img.data {
        AstcencImageData::F16(v) => v[0].iter().flat_map(|h| h.to_le_bytes()).collect(),
        AstcencImageData::U8(v) => v[0].clone(),
        AstcencImageData::F32(_) => Vec::new(),
    })
}

fn compress(kind: &str, bx: u32, by: u32, w: u32, h: u32, px: &[f32]) -> Result<Vec<u8>, AstcencError> {
    let c = ctx(kind, bx, by)?;
    let data = if kind == "float" {
        AstcencImageData::F16(vec![px.iter().map(|v| float_to_float16(*v)).collect()])
    } else {
        AstcencImageData::U8(vec![px.iter().map(|v| (v * 255.0 + 0.5) as u8).collect()])
    };
    let img = AstcencImage { dim_x: w, dim_y: h, dim_z: 1, data };
    let blocks = (w.div_ceil(bx) * h.div_ceil(by) * 16) as usize;
    let mut out = vec![0u8; blocks];
    {
        let o: AstcencOutput = Mutex::new(&mut out[..]);
        astcenc_compress_image(&c, &img, &AstcencSwizzle::RGBA, &o, 0)?;
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/astc");
    std::fs::create_dir_all(dir)?;
    let mut cases = String::new();
    let mut write = |name: String, kind: &str, bx: u32, by: u32, w: u32, h: u32, blocks: &[u8]| -> Result<(), Box<dyn std::error::Error>> {
        let expected = decompress(kind, bx, by, w, h, blocks).map_err(|e| format!("{e:?}"))?;
        std::fs::write(format!("{dir}/{name}.astc"), blocks)?;
        std::fs::write(format!("{dir}/{name}.expected"), &expected)?;
        writeln!(cases, "{name} {kind} {bx} {by} {w} {h}")?;
        println!("{name}");
        Ok(())
    };
    let (w, h) = (37u32, 29u32);
    for (bx, by) in [(4, 4), (5, 5), (6, 6), (8, 5), (8, 8), (10, 6), (12, 12)] {
        for kind in ["unorm", "srgb", "float"] {
            let scale = if kind == "float" { 6.0 } else { 1.0 };
            let px = image(w, h, scale, (bx * 100 + by) as u64);
            let blocks = compress(kind, bx, by, w, h, &px).map_err(|e| format!("{e:?}"))?;
            write(format!("img_{kind}_{bx}x{by}"), kind, bx, by, w, h, &blocks)?;
        }
    }
    // Random blocks: mostly invalid encodings, some void extent and every mode by chance.
    for (bx, by) in [(4, 4), (6, 6), (12, 12)] {
        let mut rng = Lcg((bx * 31 + by) as u64);
        let (w, h) = (bx * 16, by * 8);
        let blocks: Vec<u8> = (0..128 * 16).map(|_| rng.next() as u8).collect();
        for kind in ["unorm", "float"] {
            write(format!("rand_{kind}_{bx}x{by}"), kind, bx, by, w, h, &blocks)?;
        }
    }
    std::fs::write(format!("{dir}/cases.txt"), cases)?;
    Ok(())
}
