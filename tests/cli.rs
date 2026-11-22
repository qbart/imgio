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
