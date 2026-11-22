//! The `imgio` binary end to end: every subcommand on generated inputs, outputs read back.
#![cfg(feature = "cli")]

use std::path::{Path, PathBuf};
use std::process::Command;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn dir(name: &str) -> R<PathBuf> {
    let d = std::env::temp_dir().join(format!("imgio-cli-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

fn run(args: &[&str]) -> R<String> {
    let out = Command::new(env!("CARGO_BIN_EXE_imgio"))
        .args(args)
        .output()?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    if !out.status.success() {
        return Err(format!("{args:?}: {}", String::from_utf8_lossy(&out.stderr)).into());
    }
    Ok(text)
}

fn fails(args: &[&str]) -> R<String> {
    let out = Command::new(env!("CARGO_BIN_EXE_imgio"))
        .args(args)
        .output()?;
    assert!(!out.status.success(), "{args:?} should fail");
    Ok(String::from_utf8_lossy(&out.stderr).to_string())
}

fn png(path: &Path, w: u32, h: u32) -> R {
    let px: Vec<u8> = (0..w * h * 4).map(|i| (i * 7 % 251) as u8).collect();
    let s = imgio::Surface::new(imgio::Format::Rgba8Srgb, w, h, &px)?;
    std::fs::write(
        path,
        &imgio::file::encode(imgio::file::FileFormat::Png, &s)?[..],
    )?;
    Ok(())
}

fn p(d: &Path, n: &str) -> String {
    d.join(n).to_string_lossy().into_owned()
}

#[test]
fn every_command() -> R {
    let d = dir("all")?;
    png(&d.join("a.png"), 32, 32)?;
    png(&d.join("b.png"), 20, 12)?;
    let (a, b) = (p(&d, "a.png"), p(&d, "b.png"));
    assert!(run(&["--version"])?.starts_with("imgio "));
    assert!(run(&["formats"])?.contains("astc6x6-srgb"));
    assert!(run(&["info", &a])?.contains("Png 32x32 rgba8-srgb"));

    // Image processing chain; same input, same output.
    let o1 = p(&d, "o1.png");
    let args = [
        "convert",
        &a,
        &o1,
        "--resize",
        "16x0",
        "--flip",
        "h",
        "--blur",
        "0.8",
        "--saturation",
        "0.5",
        "--color",
        "p3",
    ];
    run(&args)?;
    let first = std::fs::read(&o1)?;
    run(&args)?;
    assert_eq!(std::fs::read(&o1)?, first);
    assert!(run(&["info", &o1])?.contains("16x16"));

    // Textures: BC7 KTX2, decoded back; Basis; DDS cube.
    let k = p(&d, "t.ktx2");
    run(&[
        "convert",
        &a,
        &k,
        "--format",
        "bc7-srgb",
        "--quality",
        "fastest",
    ])?;
    let i = run(&["info", &k])?;
    assert!(i.contains("levels 6") && i.contains("bc7-srgb"), "{i}");
    let back = p(&d, "back.png");
    run(&["convert", &k, &back])?;
    let cmp = run(&["compare", &a, &back])?;
    assert!(cmp.contains("PSNR"), "{cmp}");
    let bk = p(&d, "u.ktx2");
    run(&["convert", &a, &bk, "--basis", "uastc", "--no-mips"])?;
    assert!(run(&["info", &bk])?.contains("UastcLdr4x4"));
    let cube = p(&d, "cube.dds");
    run(&[
        "texture",
        "-o",
        &cube,
        "--cube",
        &a,
        &a,
        &a,
        &a,
        &a,
        &a,
        "--format",
        "bc1",
        "--no-mips",
    ])?;
    assert!(run(&["info", &cube])?.contains("format bc1"));

    // Pack, atlas, IBL, ICO.
    let orm = p(&d, "orm.png");
    run(&["pack", "-o", &orm, "--orm", &a, &a, &a])?;
    run(&[
        "pack",
        "-o",
        &orm,
        "--r",
        &format!("{a}:g:invert"),
        "--a",
        "0.25",
    ])?;
    let at = p(&d, "atlas.png");
    let js = p(&d, "atlas.json");
    let msg = run(&[
        "atlas", "-o", &at, "--json", &js, &a, &b, &o1, "--rotate", "--pot",
    ])?;
    assert!(msg.contains("3 sprites on 1 page"), "{msg}");
    assert!(std::fs::read_to_string(&js)?.contains("\"frames\""));
    let env = p(&d, "env.ktx2");
    run(&[
        "ibl",
        &a,
        "-o",
        &env,
        "--size",
        "8",
        "--specular",
        &p(&d, "s.ktx2"),
        "--samples",
        "8",
        "--brdf",
        &p(&d, "brdf.png"),
        "--brdf-size",
        "8",
    ])?;
    assert!(run(&["info", &env])?.contains("faces 6"));
    let ico = p(&d, "i.ico");
    run(&["convert", &a, &ico])?;
    assert!(run(&["info", &ico])?.contains("entry 2: 32x32"));

    // Errors are messages and a failing exit code.
    assert!(fails(&["nonsense"])?.contains("unknown command"));
    assert!(fails(&["compare", &a, &b])?.contains("sizes differ"));
    assert!(fails(&["texture", "-o", &cube, "--cube", &b, &b, &b, &b, &b, &b])?.contains("square"));
    assert!(fails(&["convert", &a, &k, "--format", "bc99"])?.contains("unknown format"));
    assert!(fails(&["convert", &p(&d, "missing.png"), &o1])?.contains("missing.png"));
    std::fs::remove_dir_all(&d)?;
    Ok(())
}

#[test]
fn lut_command() -> R {
    use imgio::lut::Cube;
    let d = dir("lut")?;
    png(&d.join("a.png"), 24, 16)?;
    let a = p(&d, "a.png");
    // Identity and a desaturating grade.
    let id = p(&d, "id.cube");
    run(&["lut", "-o", &id, "--size", "9", "--title", "Identity"])?;
    let c = Cube::parse(&std::fs::read(&id)?)?;
    assert_eq!(c.title.as_deref(), Some("Identity"));
    let l = c.lut.as_ref().ok_or("no 3D")?;
    assert_eq!(l.size, 9);
    assert_eq!(l.table[1 + 9 * (2 + 9 * 8)], [0.125, 0.25, 1.0]);
    let g = p(&d, "gray.cube");
    run(&["lut", "-o", &g, "--size", "17", "--saturation", "0"])?;
    let c = Cube::parse(&std::fs::read(&g)?)?;
    let l = c.lut.as_ref().ok_or("no 3D")?;
    assert!(l.table.iter().all(|v| v[0] == v[1] && v[1] == v[2]));
    // Applied to an image: gray pixels, alpha kept.
    let out = p(&d, "out.png");
    run(&["convert", &a, &out, "--lut", &g])?;
    let img = imgio::file::decode(&std::fs::read(&out)?)?;
    let src = imgio::file::decode(&std::fs::read(&a)?)?;
    for (q, s) in img.data.chunks(4).zip(src.data.chunks(4)) {
        assert!(q[0] == q[1] && q[1] == q[2] && q[3] == s[3], "{q:?}");
    }
    // As a strip image (17 is not a square) and back: the same result within 16-bit rounding.
    let strip = p(&d, "gray.png");
    run(&["lut", &g, "-o", &strip])?;
    let si = imgio::file::decode(&std::fs::read(&strip)?)?;
    assert_eq!((si.width, si.height), (289, 17));
    let out2 = p(&d, "out2.png");
    run(&[
        "convert",
        &a,
        &out2,
        "--lut",
        &strip,
        "--lut-interp",
        "trilinear",
    ])?;
    let img2 = imgio::file::decode(&std::fs::read(&out2)?)?;
    let tri = p(&d, "out3.png");
    run(&[
        "convert",
        &a,
        &tri,
        "--lut",
        &g,
        "--lut-interp",
        "trilinear",
    ])?;
    let img3 = imgio::file::decode(&std::fs::read(&tri)?)?;
    assert!(
        img2.data
            .iter()
            .zip(img3.data.iter())
            .all(|(x, y)| x.abs_diff(*y) <= 1)
    );
    // HALD for square sizes.
    let hald = p(&d, "hald.png");
    run(&["lut", "-o", &hald, "--size", "16", "--exposure", "1"])?;
    let hi = imgio::file::decode(&std::fs::read(&hald)?)?;
    assert_eq!((hi.width, hi.height), (64, 64));
    // A colour transform as a device link, applied with --lut.
    let link = p(&d, "p3.icc");
    run(&[
        "lut", "-o", &link, "--size", "17", "--from", "p3", "--to", "srgb",
    ])?;
    let prof = imgio::color::Profile::from_icc(&std::fs::read(&link)?)?;
    assert!(prof.is_device_link());
    run(&["convert", &a, &p(&d, "out4.png"), "--lut", &link])?;
    // A .cube input with nothing else is copied as is (shaper kept).
    let shaper = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/lut/shaper.cube");
    let copy = p(&d, "copy.cube");
    run(&["lut", &shaper.to_string_lossy(), "-o", &copy])?;
    let c = Cube::parse(&std::fs::read(&copy)?)?;
    assert_eq!(c.shaper.as_ref().ok_or("no shaper")?.size, 256);
    // Errors.
    assert!(fails(&["lut", "-o", &id, "--blur", "2"])?.contains("per-pixel"));
    assert!(fails(&["lut", "-o", &id, "--from", "p3"])?.contains("--to"));
    assert!(fails(&["convert", &a, &out, "--lut", &a])?.contains("HALD"));
    std::fs::remove_dir_all(&d)?;
    Ok(())
}

#[test]
fn memory_command() -> R {
    use imgio::memory::{self, Layout};
    let d = dir("memory")?;
    std::fs::create_dir_all(d.join("sub"))?;
    png(&d.join("a.png"), 64, 32)?;
    png(&d.join("sub/b.png"), 20, 12)?;
    std::fs::write(d.join("notes.txt"), b"ignored")?;
    let ktx = p(&d, "sub/t.ktx2");
    run(&["convert", &p(&d, "a.png"), &ktx, "--format", "bc7-srgb", "--mips"])?;
    let dir_s = d.to_string_lossy().into_owned();
    let text = run(&["memory", &dir_s])?;
    assert!(text.contains("total 3 files"), "{text}");
    let json = run(&["memory", &dir_s, "--json"])?;
    // a.png: RGBA8 64x32 with 7 mips; desktop BC7 (colour, alpha varies), mobile ASTC 4x4.
    let l = Layout::new_2d(64, 32, true);
    let rgba = memory::texture_bytes(imgio::Format::Rgba8Srgb, &l)?;
    let bc7 = memory::texture_bytes(imgio::Format::Bc7Srgb, &l)?;
    assert!(json.contains(&format!(
        "\"now\":{{\"format\":\"rgba8-srgb\",\"bytes\":{rgba}}}"
    )));
    assert!(json.contains(&format!("\"desktop\":{{\"format\":\"bc7-srgb\",\"bytes\":{bc7}}}")));
    // The KTX2 keeps its BC7 for desktop.
    assert!(json.contains(&format!(
        "\"now\":{{\"format\":\"bc7-srgb\",\"bytes\":{bc7}}},\"note\":\"\",\"desktop\":{{\"format\":\"bc7-srgb\",\"bytes\":{bc7}}}"
    )));
    let nomips = run(&["memory", &p(&d, "a.png"), "--json", "--no-mips"])?;
    assert!(nomips.contains(&format!("\"bytes\":{}", 64 * 32 * 4)));
    // Budget: under passes, over fails with exit code 1.
    run(&["memory", &dir_s, "--budget", "1M"])?;
    assert!(fails(&["memory", &dir_s, "--budget", "2K"])?.contains("over budget"));
    assert!(fails(&["memory", &dir_s, "--budget", "lots"])?.contains("bad size"));
    std::fs::remove_dir_all(&d)?;
    Ok(())
}
