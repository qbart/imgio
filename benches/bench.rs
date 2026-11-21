//! Throughput benchmarks (no external harness): `make bench` / `cargo bench`.
//! Prints the best of N runs in megapixels per second.

use imgio::ktx2::Ktx2File;
use imgio::{AstcBlock, AstcKind, Format, Surface};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn bench(name: &str, pixels: u64, mut f: impl FnMut() -> imgio::Result<()>) {
    let mut best = Duration::MAX;
    let start = Instant::now();
    let mut runs = 0;
    while runs < 3 || (start.elapsed() < Duration::from_millis(500) && runs < 200) {
        let t = Instant::now();
        if let Err(e) = f() {
            println!("{name:<40} error: {e}");
            return;
        }
        best = best.min(t.elapsed());
        runs += 1;
    }
    let mpix = pixels as f64 / best.as_secs_f64() / 1e6;
    println!(
        "{name:<40} {:>9.3} ms {mpix:>10.1} Mpix/s",
        best.as_secs_f64() * 1e3
    );
}

fn random(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 56) as u8
        })
        .collect()
}

/// `src` blocks repeated to cover `w` x `h` of `format`.
fn tiled(format: Format, src: &[u8], w: u32, h: u32) -> imgio::Result<Vec<u8>> {
    let n = format.surface_bytes(w, h)?;
    Ok(src.iter().copied().cycle().take(n).collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = env!("CARGO_MANIFEST_DIR");
    let (w, h) = (1024u32, 1024u32);
    let px = u64::from(w * h);

    println!("-- decode {w}x{h}");
    let mut out = vec![0u8; (w * h * 8) as usize];
    for (name, f) in [
        ("bc1", Format::Bc1Unorm),
        ("bc3", Format::Bc3Unorm),
        ("bc5", Format::Bc5Unorm),
        ("bc6h", Format::Bc6hUfloat),
        ("bc7", Format::Bc7Unorm),
        ("etc1", Format::Etc1Rgb),
        ("etc2 rgba", Format::Etc2Rgba),
        ("eac rg11", Format::EacRg11Unorm),
    ] {
        let src = random(f.surface_bytes(w, h)?, 7);
        bench(&format!("decode {name}"), px, || {
            imgio::decode(f, black_box(&src), w, h, &mut out)
        });
    }
    for (kind, file) in [(AstcKind::Unorm, "unorm"), (AstcKind::Float, "float")] {
        for (block, dim) in [(AstcBlock::B4x4, "4x4"), (AstcBlock::B8x8, "8x8")] {
            let blocks = std::fs::read(format!("{root}/testdata/astc/img_{file}_{dim}.astc"))?;
            let f = Format::Astc(block, kind);
            let src = tiled(f, &blocks, w, h)?;
            bench(&format!("decode astc {dim} {file}"), px, || {
                imgio::decode(f, black_box(&src), w, h, &mut out)
            });
        }
    }

    println!("-- convert {w}x{h}");
    let rgba = random((w * h * 4) as usize, 3);
    let s = Surface::new(Format::Rgba8Unorm, w, h, &rgba)?;
    let mut fout = vec![0u8; (w * h * 16) as usize];
    bench("convert rgba8 -> rgba32f", px, || {
        imgio::convert::convert(&s, Format::Rgba32Float, &mut fout)
    });

    for name in ["etc1s_512", "uastc_512"] {
        let data = std::fs::read(format!("{root}/testdata/bench/{name}.ktx2"))?;
        let k = Ktx2File::parse(&data)?;
        let px = u64::from(k.width() * k.height());
        println!("-- transcode {name}.ktx2");
        let mut out = vec![0u8; k.transcoded_size(0, Format::Rgba8Unorm)?];
        for (tname, f) in [
            ("etc1", Format::Etc1Rgb),
            ("etc2 rgba", Format::Etc2Rgba),
            ("bc1", Format::Bc1Unorm),
            ("bc3", Format::Bc3Unorm),
            ("bc7", Format::Bc7Unorm),
            ("astc 4x4", Format::Astc(AstcBlock::B4x4, AstcKind::Unorm)),
            ("rgba32", Format::Rgba8Unorm),
        ] {
            bench(&format!("{name} -> {tname}"), px, || {
                k.transcode(0, 0, 0, f, &mut out)
            });
        }
    }
    Ok(())
}
