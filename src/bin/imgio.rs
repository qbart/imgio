//! `imgio` command line tool: image and texture conversion, processing, packing, atlases,
//! image based lighting and comparison. Build with `cargo install imgio --features cli`.

use imgio::atlas::{self, Algorithm, AtlasOptions, MaxRectsHeuristic, PageSize, SortOrder, Trim};
use imgio::color::{ColorSpace, Intent, Profile};
use imgio::file::{self, FileFormat, Image};
use imgio::filter;
use imgio::ibl;
use imgio::lut::{self, Cube, Interpolation, Lut3d};
use imgio::memory::{self, Layout, Target};
use imgio::metrics::{self, CompareOptions, Scope, Thresholds};
use imgio::mip::Filter;
use imgio::pack::{self, Channel, ChannelSpec, PackOptions, PackSize};
use imgio::resize::{self, Fit, ResizeOptions};
use imgio::texture::{self, BuildOptions, Mips, Texture};
use imgio::{AstcBlock, AstcKind, EncodeOptions, Format, Quality, Surface};
use std::process::ExitCode;

type Res<T = ()> = Result<T, String>;

const USAGE: &str = "imgio - image and GPU texture tool

USAGE:
  imgio info <file>...
  imgio convert <input> <output> [image options] [texture options]
  imgio texture -o <out.ktx2|.dds|.ktx> <input>... [--cube | --array] [texture options]
  imgio pack -o <output> [--r SRC] [--g SRC] [--b SRC] [--a SRC] [--orm AO ROUGH METAL]
             [--size same|largest|first|WxH] [--pixel FORMAT]
             SRC = file[:r|g|b|a|luma|avg|max][:invert] or a number (constant)
  imgio atlas -o <atlas.png> [--json FILE] <input>... [atlas options]
  imgio ibl <panorama> -o <env.ktx2> [--size N] [--irradiance FILE] [--irradiance-size N]
            [--specular FILE] [--samples N] [--levels N] [--brdf FILE] [--brdf-size N]
  imgio compare <reference> <test> [--diff FILE|DIR] [--amplify N] [--peak P] [--no-ssim]
                [--min-psnr DB] [--min-ssim S] [--max-error E] [--max-mean-error E]
                [--max-mse M] [--scope channels|rgb|all] [--json] [--strict]
                (two files, or two directories matched by relative path; exit 1 on a
                failed limit, size mismatch or missing file; --strict also fails extras)
  imgio memory <file|dir>... [--no-mips] [--desktop FORMAT] [--mobile FORMAT] [--json]
               [--budget SIZE]   (SIZE like 512M, 2G; exit 1 when the current total exceeds it)
  imgio lut [input LUT] -o <out.cube|.icc|image> [--size N] [--from SPACE --to SPACE]
            [--intent I] [--title T] [--layout hald|strip] [per-pixel image options]
  imgio formats

Inputs: any image file, or a texture container (.ktx2 .dds .ktx .basis .astc: level 0, decoded
or transcoded to RGBA). Outputs are picked by extension; .ktx2 .dds .ktx make textures.

IMAGE OPTIONS (applied in this order):
  --color SPACE         colour manage to srgb|linear|p3|adobe|prophoto|rec2020|pq|hlg|acescg|gray
  --intent I            perceptual|relative|saturation|absolute (default perceptual)
  --crop X,Y,W,H        --resize WxH (0 keeps aspect)  --fit exact|contain|cover|width|height
  --filter NAME         lanczos4 (default), lanczos3, mitchell, catmullrom, kaiser, box, tent, ...
  --flip h|v            --rotate 90|180|270            --transpose
  --blur SIGMA          --box-blur RADIUS              --sharpen SIGMA[,AMOUNT[,THRESHOLD]]
  --exposure EV --brightness B --contrast C --gamma G --saturation S
  --grayscale           --premultiply                  --unpremultiply
  --swizzle rgba        (letters r g b a 0 1)
  --normal-map STRENGTH [--directx]                    --tonemap clamp|reinhard|aces [--white W]
  --lut FILE            apply a LUT: .cube, ICC device link (.icc .icm), HALD or strip image
  --lut-interp I        tetrahedral (default) | trilinear
  --pixel FORMAT        convert the pixels to this uncompressed format (see `imgio formats`)

TEXTURE OPTIONS:
  --format FORMAT       bc1..bc7, etc1, etc2, eac-r11, astc4x4[-srgb|-hdr], rgba8, rgba16f, ...
  --basis CODEC         etc1s|uastc|uastc-hdr|astc-hdr6x6|astc6x6|xuastc6x6|xubc7 (KTX2 only)
  --quality Q           fastest|fast|default|high|best   --mips|--no-mips  --levels N
  --normal              renormalize mips as a normal map --zstd LEVEL      --legacy-dds
  --srgb | --linear     (Basis) colour space of the input (default srgb)

LUT: a new table (identity of --size, default 33, or the input LUT) is baked through the
colour conversion and the per-pixel image options (exposure .. saturation, grayscale, swizzle,
tonemap, --lut), then written as .cube, an ICC device link (.icc / .icm) or an image (HALD
for square sizes 4, 9, .. 256, else a size*size x size strip; 16-bit where the file allows).
A .cube input with no other options is copied as is (shaper and domain kept).

ATLAS OPTIONS:
  --algorithm maxrects|maxrects-bl|maxrects-baf|maxrects-blsf|maxrects-cp|skyline|shelf
  --sort maxside|area|height|width|perimeter|none  --max WxH (4096x4096)  --fixed WxH
  --pot --square --multiple N --padding N (2) --border N --extrude N --rotate
  --trim [ALPHA] --dedupe --pages --pixel FORMAT
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("imgio: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Res {
    let Some(cmd) = args.first() else {
        print!("{USAGE}");
        return Ok(());
    };
    let rest = &args[1..];
    match cmd.as_str() {
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            Ok(())
        }
        "-V" | "--version" => {
            println!("imgio {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "info" => info(rest),
        "convert" => convert(rest),
        "texture" => texture_cmd(rest),
        "pack" => pack_cmd(rest),
        "atlas" => atlas_cmd(rest),
        "ibl" => ibl_cmd(rest),
        "compare" => compare(rest),
        "lut" => lut_cmd(rest),
        "memory" => memory_cmd(rest),
        "formats" => {
            for (name, _) in formats() {
                println!("{name}");
            }
            Ok(())
        }
        c => Err(format!("unknown command `{c}` (see `imgio --help`)")),
    }
}

// ---------------------------------------------------------------------------------------------
// Arguments

/// Parsed `--key value` options, `--flag`s and positional arguments.
struct Args {
    pos: Vec<String>,
    opts: Vec<(String, Option<String>)>,
}

/// Options that take no value.
const FLAGS: &[&str] = &[
    "--cube",
    "--array",
    "--mips",
    "--no-mips",
    "--normal",
    "--legacy-dds",
    "--srgb",
    "--linear",
    "--pot",
    "--square",
    "--rotate",
    "--dedupe",
    "--pages",
    "--transpose",
    "--grayscale",
    "--premultiply",
    "--unpremultiply",
    "--directx",
    "--no-ssim",
];

/// Options with an optional value (taken when the next argument is a number).
const OPTIONAL: &[&str] = &["--trim"];

/// Options with several values.
const MULTI: &[(&str, usize)] = &[("--orm", 3)];

impl Args {
    fn parse(a: &[String]) -> Res<Args> {
        Args::parse_with(a, &[])
    }

    /// Like [`Args::parse`] with command specific value-less options (`--json` takes a file in
    /// `atlas` but is a flag in `memory` / `compare`).
    fn parse_with(a: &[String], flags: &[&str]) -> Res<Args> {
        let mut pos = Vec::new();
        let mut opts = Vec::new();
        let mut i = 0;
        while i < a.len() {
            let s = &a[i];
            if s.starts_with("--") || (s.starts_with('-') && s.len() == 2 && s != "-") {
                let key = if s == "-o" {
                    "--output".to_string()
                } else {
                    s.clone()
                };
                if FLAGS.contains(&key.as_str()) || flags.contains(&key.as_str()) {
                    opts.push((key, None));
                } else if OPTIONAL.contains(&key.as_str()) {
                    let v = a.get(i + 1).filter(|v| v.parse::<f32>().is_ok()).cloned();
                    if v.is_some() {
                        i += 1;
                    }
                    opts.push((key, v.or(Some(String::new()))));
                } else if let Some((_, n)) = MULTI.iter().find(|(k, _)| *k == key) {
                    let vals: Vec<String> = a
                        .get(i + 1..i + 1 + n)
                        .ok_or(format!("{key} needs {n} values"))?
                        .to_vec();
                    i += n;
                    opts.push((key, Some(vals.join("\u{1f}"))));
                } else {
                    let v = a.get(i + 1).ok_or(format!("{key} needs a value"))?;
                    i += 1;
                    opts.push((key, Some(v.clone())));
                }
            } else {
                pos.push(s.clone());
            }
            i += 1;
        }
        Ok(Args { pos, opts })
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.opts
            .iter()
            .rev()
            .find(|(n, _)| n == k)
            .and_then(|(_, v)| v.as_deref())
    }

    fn has(&self, k: &str) -> bool {
        self.opts.iter().any(|(n, _)| n == k)
    }

    fn num<T: std::str::FromStr>(&self, k: &str) -> Res<Option<T>> {
        self.get(k)
            .map(|v| {
                v.parse::<T>()
                    .map_err(|_| format!("bad number for {k}: `{v}`"))
            })
            .transpose()
    }

    fn output(&self) -> Res<&str> {
        self.get("--output")
            .ok_or_else(|| "missing -o <output>".to_string())
    }
}

fn size2(s: &str) -> Res<(u32, u32)> {
    let (a, b) = s
        .split_once(['x', 'X'])
        .ok_or(format!("expected WxH, got `{s}`"))?;
    let p = |v: &str| v.parse::<u32>().map_err(|_| format!("bad size `{s}`"));
    Ok((p(a)?, p(b)?))
}

fn e<E: std::fmt::Display>(ctx: &str) -> impl Fn(E) -> String + '_ {
    move |err| format!("{ctx}: {err}")
}

fn read(path: &str) -> Res<Vec<u8>> {
    std::fs::read(path).map_err(e(path))
}

fn write(path: &str, data: &[u8]) -> Res {
    std::fs::write(path, data).map_err(e(path))
}

fn ext(path: &str) -> String {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

// ---------------------------------------------------------------------------------------------
// Formats

fn formats() -> Vec<(String, Format)> {
    use Format::*;
    let plain = [
        ("r8", R8Unorm),
        ("r8-snorm", R8Snorm),
        ("rg8", Rg8Unorm),
        ("rg8-snorm", Rg8Snorm),
        ("rgb8", Rgb8Unorm),
        ("rgb8-srgb", Rgb8Srgb),
        ("rgba8", Rgba8Unorm),
        ("rgba8-srgb", Rgba8Srgb),
        ("bgra8", Bgra8Unorm),
        ("bgra8-srgb", Bgra8Srgb),
        ("r16", R16Unorm),
        ("r16-snorm", R16Snorm),
        ("rg16", Rg16Unorm),
        ("rg16-snorm", Rg16Snorm),
        ("rgba16", Rgba16Unorm),
        ("r16f", R16Float),
        ("r32f", R32Float),
        ("rgb16f", Rgb16Float),
        ("rgba16f", Rgba16Float),
        ("rgba32f", Rgba32Float),
        ("rgb9e5", Rgb9e5Float),
        ("rgb565", Rgb565),
        ("bgr565", Bgr565),
        ("rgba4444", Rgba4444),
        ("bc1", Bc1Unorm),
        ("bc1-srgb", Bc1Srgb),
        ("bc2", Bc2Unorm),
        ("bc2-srgb", Bc2Srgb),
        ("bc3", Bc3Unorm),
        ("bc3-srgb", Bc3Srgb),
        ("bc4", Bc4Unorm),
        ("bc4-snorm", Bc4Snorm),
        ("bc5", Bc5Unorm),
        ("bc5-snorm", Bc5Snorm),
        ("bc6h", Bc6hUfloat),
        ("bc6h-signed", Bc6hSfloat),
        ("bc7", Bc7Unorm),
        ("bc7-srgb", Bc7Srgb),
        ("etc1", Etc1Rgb),
        ("etc2", Etc2Rgb),
        ("etc2-srgb", Etc2Srgb),
        ("etc2-a1", Etc2Rgba1),
        ("etc2-a1-srgb", Etc2Rgba1Srgb),
        ("etc2-rgba", Etc2Rgba),
        ("etc2-rgba-srgb", Etc2RgbaSrgb),
        ("eac-r11", EacR11Unorm),
        ("eac-r11-snorm", EacR11Snorm),
        ("eac-rg11", EacRg11Unorm),
        ("eac-rg11-snorm", EacRg11Snorm),
        ("pvrtc1-rgb", Pvrtc1Rgb),
        ("pvrtc1-rgba", Pvrtc1Rgba),
        ("pvrtc2-rgb", Pvrtc2Rgb),
        ("pvrtc2-rgba", Pvrtc2Rgba),
        ("atc-rgb", AtcRgb),
        ("atc-rgba", AtcRgba),
        ("fxt1", Fxt1Rgb),
    ];
    let mut v: Vec<(String, Format)> = plain.iter().map(|(n, f)| (n.to_string(), *f)).collect();
    for b in AstcBlock::ALL {
        let n = format!("astc{}", &format!("{b:?}")[1..]);
        v.push((n.clone(), Astc(b, AstcKind::Unorm)));
        v.push((format!("{n}-srgb"), Astc(b, AstcKind::Srgb)));
        v.push((format!("{n}-hdr"), Astc(b, AstcKind::Float)));
    }
    v
}

fn parse_format(s: &str) -> Res<Format> {
    let s = s.to_ascii_lowercase();
    formats()
        .into_iter()
        .find(|(n, _)| *n == s)
        .map(|(_, f)| f)
        .ok_or(format!("unknown format `{s}` (see `imgio formats`)"))
}

fn format_name(f: Format) -> String {
    formats()
        .into_iter()
        .find(|(_, g)| *g == f)
        .map_or(format!("{f:?}"), |(n, _)| n)
}

// ---------------------------------------------------------------------------------------------
// Loading

/// Loads any image file, or level 0 of a texture container (decoded to RGBA).
fn load(path: &str) -> Res<Image> {
    let data = read(path)?;
    load_bytes(path, &data)
}

fn decoded(f: Format, data: &[u8], w: u32, h: u32) -> Res<Image> {
    if !f.is_compressed() {
        let mut img = Image::new(f, w, h).map_err(e("image"))?;
        let n = img.data.len();
        img.data
            .copy_from_slice(data.get(..n).ok_or("short texture data")?);
        return Ok(img);
    }
    let buf = imgio::decode_to_buffer(f, data, w, h).map_err(e("decode"))?;
    let mut img = Image::new(f.decoded(), w, h).map_err(e("image"))?;
    img.data.copy_from_slice(&buf);
    Ok(img)
}

fn load_bytes(path: &str, data: &[u8]) -> Res<Image> {
    let x = ext(path);
    match x.as_str() {
        "ktx2" => {
            let k = imgio::ktx2::Ktx2File::parse(data).map_err(e(path))?;
            let (w, h) = (k.width(), k.height());
            if k.tex_format().is_some() {
                let hdr = k.tex_format().is_some_and(|t| t.is_hdr());
                let target = if hdr {
                    Format::Rgba16Float
                } else {
                    Format::Rgba8Unorm
                };
                let b = k.transcode_to_buffer(0, 0, 0, target).map_err(e(path))?;
                let f = if !hdr && k.is_srgb() {
                    Format::Rgba8Srgb
                } else {
                    target
                };
                decoded(f, &b, w, h)
            } else {
                let f = k.format().ok_or("unsupported KTX2 format")?;
                decoded(f, k.image_data(0, 0, 0).map_err(e(path))?, w, h)
            }
        }
        "dds" => {
            let d = imgio::dds::DdsFile::parse(data).map_err(e(path))?;
            decoded(
                d.format(),
                d.image_data(0, 0, 0).map_err(e(path))?,
                d.width(),
                d.height(),
            )
        }
        "ktx" => {
            let k = imgio::ktx1::Ktx1File::parse(data).map_err(e(path))?;
            let img = k.image(0, 0, 0).map_err(e(path))?;
            decoded(k.format(), &img, k.width(), k.height())
        }
        "basis" => {
            let b = imgio::basis::BasisFile::parse(data).map_err(e(path))?;
            let li = b.level_info(0, 0).map_err(e(path))?;
            let hdr = b.tex_format().is_hdr();
            let target = if hdr {
                Format::Rgba16Float
            } else {
                Format::Rgba8Unorm
            };
            let px = b.transcode_to_buffer(0, 0, target).map_err(e(path))?;
            let f = if !hdr && b.is_srgb() {
                Format::Rgba8Srgb
            } else {
                target
            };
            decoded(f, &px, li.width, li.height)
        }
        "astc" => {
            let a = imgio::astc_file::AstcFile::parse(data).map_err(e(path))?;
            let f = a.format(AstcKind::Unorm).ok_or("unsupported ASTC block")?;
            decoded(f, a.data(), a.width(), a.height())
        }
        _ => file::decode(data).map_err(e(path)),
    }
}

// ---------------------------------------------------------------------------------------------
// info

fn info(a: &[String]) -> Res {
    if a.is_empty() {
        return Err("info needs a file".into());
    }
    for path in a {
        let data = read(path)?;
        println!("{path}:");
        match ext(path).as_str() {
            "ktx2" => {
                let k = imgio::ktx2::Ktx2File::parse(&data).map_err(e(path))?;
                println!(
                    "  KTX2 {}x{} depth {} levels {} layers {} faces {} srgb {}",
                    k.width(),
                    k.height(),
                    k.depth(),
                    k.level_count(),
                    k.layer_count(),
                    k.face_count(),
                    k.is_srgb()
                );
                match (k.tex_format(), k.format()) {
                    (Some(t), _) => println!("  Basis Universal {t:?}"),
                    (None, Some(f)) => println!("  format {}", format_name(f)),
                    _ => println!("  vkFormat {}", k.vk_format()),
                }
                continue;
            }
            "dds" => {
                let d = imgio::dds::DdsFile::parse(&data).map_err(e(path))?;
                println!(
                    "  DDS {}x{} levels {} format {}",
                    d.width(),
                    d.height(),
                    d.level_count(),
                    format_name(d.format())
                );
                continue;
            }
            "ktx" => {
                let k = imgio::ktx1::Ktx1File::parse(&data).map_err(e(path))?;
                println!(
                    "  KTX1 {}x{} levels {} layers {} faces {} format {}",
                    k.width(),
                    k.height(),
                    k.level_count(),
                    k.layer_count(),
                    k.face_count(),
                    format_name(k.format())
                );
                continue;
            }
            "basis" => {
                let b = imgio::basis::BasisFile::parse(&data).map_err(e(path))?;
                let li = b.level_info(0, 0).map_err(e(path))?;
                println!(
                    "  Basis {:?} {:?} images {} {}x{} levels {} alpha {} srgb {}",
                    b.tex_format(),
                    b.tex_type(),
                    b.image_count(),
                    li.width,
                    li.height,
                    b.level_count(0),
                    b.has_alpha(),
                    b.is_srgb()
                );
                continue;
            }
            _ => {}
        }
        let fmt = FileFormat::detect(&data).ok_or("unknown file format")?;
        let img = file::decode(&data).map_err(e(path))?;
        println!(
            "  {fmt:?} {}x{} {}",
            img.width,
            img.height,
            format_name(img.format)
        );
        if let Ok(anim) = file::decode_animation(&data)
            && anim.frames.len() > 1
        {
            println!("  {} frames, loops {}", anim.frames.len(), anim.loops);
        }
        match file::colorimetry(&data).map_err(e(path))? {
            file::Colorimetry::Icc(b) => {
                let ch = Profile::from_icc(&b)
                    .map(|p| format!("{:?}", p.data_channels()))
                    .unwrap_or_else(|_| "unparsed".into());
                println!("  colour: ICC profile, {} bytes ({ch})", b.len());
            }
            c => println!("  colour: {c:?}"),
        }
        if fmt == FileFormat::Ico
            && let Ok(inf) = file::ico::info(&data)
        {
            for (i, en) in inf.entries.iter().enumerate() {
                println!(
                    "  entry {i}: {}x{} bits {} png {} hotspot {:?}",
                    en.width, en.height, en.bits, en.png, en.hotspot
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// convert: image operations

fn color_space(s: &str) -> Res<ColorSpace> {
    Ok(match s {
        "srgb" => ColorSpace::Srgb,
        "linear" | "linear-srgb" => ColorSpace::LinearSrgb,
        "p3" | "display-p3" => ColorSpace::DisplayP3,
        "dci-p3" => ColorSpace::DciP3,
        "adobe" | "adobergb" => ColorSpace::AdobeRgb,
        "prophoto" => ColorSpace::ProPhoto,
        "rec2020" | "bt2020" => ColorSpace::Rec2020,
        "pq" | "rec2100-pq" => ColorSpace::Rec2100Pq,
        "hlg" | "rec2100-hlg" => ColorSpace::Rec2100Hlg,
        "acescg" => ColorSpace::AcesCg,
        "aces" | "aces2065" => ColorSpace::Aces2065,
        "gray" => ColorSpace::GraySrgb,
        "gray-linear" => ColorSpace::Gray(1.0),
        _ => return Err(format!("unknown colour space `{s}`")),
    })
}

fn intent(a: &Args) -> Res<Intent> {
    Ok(match a.get("--intent").unwrap_or("perceptual") {
        "perceptual" => Intent::Perceptual,
        "relative" => Intent::RelativeColorimetric,
        "saturation" => Intent::Saturation,
        "absolute" => Intent::AbsoluteColorimetric,
        s => return Err(format!("unknown intent `{s}`")),
    })
}

fn resample_filter(s: &str) -> Res<Filter> {
    Ok(match s {
        "box" => Filter::Box,
        "tent" | "bilinear" => Filter::Tent,
        "bell" => Filter::Bell,
        "bspline" => Filter::BSpline,
        "mitchell" => Filter::Mitchell,
        "blackman" => Filter::Blackman,
        "lanczos3" => Filter::Lanczos3,
        "lanczos4" | "lanczos" => Filter::Lanczos4,
        "lanczos6" => Filter::Lanczos6,
        "lanczos12" => Filter::Lanczos12,
        "kaiser" => Filter::Kaiser,
        "gaussian" => Filter::Gaussian,
        "catmullrom" => Filter::CatmullRom,
        _ => return Err(format!("unknown filter `{s}`")),
    })
}

fn swizzle_map(s: &str) -> Res<[filter::Channel; 4]> {
    let c: Vec<char> = s.chars().collect();
    if c.len() != 4 {
        return Err("--swizzle needs 4 letters".into());
    }
    let m = |ch: char| -> Res<filter::Channel> {
        Ok(match ch {
            'r' => filter::Channel::R,
            'g' => filter::Channel::G,
            'b' => filter::Channel::B,
            'a' => filter::Channel::A,
            '0' => filter::Channel::Zero,
            '1' => filter::Channel::One,
            _ => return Err(format!("bad swizzle letter `{ch}`")),
        })
    };
    Ok([m(c[0])?, m(c[1])?, m(c[2])?, m(c[3])?])
}

/// Applies the image options of `a` to `img` in the documented order.
fn surf(i: &Image) -> Res<Surface<'_>> {
    i.surface().map_err(e("image"))
}

fn process(mut img: Image, a: &Args, data: Option<&[u8]>) -> Res<Image> {
    let s = surf;
    if let Some(space) = a.get("--color") {
        let target = Profile::new(color_space(space)?).map_err(e("colour"))?;
        img = match data {
            Some(d) if FileFormat::detect(d).is_some() => {
                file::decode_to(d, &target, intent(a)?).map_err(e("colour"))?
            }
            _ => {
                let gray = matches!(
                    img.format,
                    Format::R8Unorm | Format::R16Unorm | Format::R16Float | Format::R32Float
                );
                let from = Profile::new(if gray {
                    ColorSpace::GraySrgb
                } else {
                    ColorSpace::Srgb
                })
                .map_err(e("colour"))?;
                let fmt = if img.format.is_hdr() {
                    Format::Rgba32Float
                } else {
                    img.format
                };
                imgio::color::convert(&s(&img)?, &from, &target, intent(a)?, fmt)
                    .map_err(e("colour"))?
            }
        };
    }
    if let Some(c) = a.get("--crop") {
        let v: Vec<u32> = c
            .split(',')
            .map(|x| x.parse().map_err(|_| format!("bad --crop `{c}`")))
            .collect::<Res<_>>()?;
        if v.len() != 4 {
            return Err("--crop needs X,Y,W,H".into());
        }
        img = resize::crop(&s(&img)?, v[0], v[1], v[2], v[3]).map_err(e("crop"))?;
    }
    if let Some(r) = a.get("--resize") {
        let (w, h) = size2(r)?;
        let fit = match a.get("--fit").unwrap_or(if w == 0 {
            "height"
        } else if h == 0 {
            "width"
        } else {
            "exact"
        }) {
            "exact" => Fit::Exact,
            "contain" => Fit::Contain,
            "cover" => Fit::Cover,
            "width" => Fit::Width,
            "height" => Fit::Height,
            f => return Err(format!("unknown fit `{f}`")),
        };
        let o = ResizeOptions {
            fit,
            filter: resample_filter(a.get("--filter").unwrap_or("lanczos4"))?,
            ..Default::default()
        };
        img = resize::resize(&s(&img)?, w, h, &o).map_err(e("resize"))?;
    }
    match a.get("--flip") {
        Some("h") => img = filter::flip_horizontal(&s(&img)?).map_err(e("flip"))?,
        Some("v") => img = filter::flip_vertical(&s(&img)?).map_err(e("flip"))?,
        Some(f) => return Err(format!("--flip h|v, got `{f}`")),
        None => {}
    }
    match a.get("--rotate") {
        Some("90") => img = filter::rotate90(&s(&img)?).map_err(e("rotate"))?,
        Some("180") => img = filter::rotate180(&s(&img)?).map_err(e("rotate"))?,
        Some("270") => img = filter::rotate270(&s(&img)?).map_err(e("rotate"))?,
        Some(r) => return Err(format!("--rotate 90|180|270, got `{r}`")),
        None => {}
    }
    if a.has("--transpose") {
        img = filter::transpose(&s(&img)?).map_err(e("transpose"))?;
    }
    if let Some(sg) = a.num::<f32>("--blur")? {
        img = filter::blur_gaussian(&s(&img)?, sg).map_err(e("blur"))?;
    }
    if let Some(r) = a.num::<u32>("--box-blur")? {
        img = filter::blur_box(&s(&img)?, r).map_err(e("blur"))?;
    }
    if let Some(sh) = a.get("--sharpen") {
        let v: Vec<f32> = sh
            .split(',')
            .map(|x| x.parse().map_err(|_| format!("bad --sharpen `{sh}`")))
            .collect::<Res<_>>()?;
        let (sigma, amount, threshold) = (
            v.first().copied().unwrap_or(1.0),
            v.get(1).copied().unwrap_or(0.5),
            v.get(2).copied().unwrap_or(0.0),
        );
        img = filter::unsharp_mask(&s(&img)?, sigma, amount, threshold).map_err(e("sharpen"))?;
    }
    let adj = filter::Adjust {
        exposure: a.num("--exposure")?.unwrap_or(0.0),
        brightness: a.num("--brightness")?.unwrap_or(0.0),
        contrast: a.num("--contrast")?.unwrap_or(1.0),
        gamma: a.num("--gamma")?.unwrap_or(1.0),
        saturation: a.num("--saturation")?.unwrap_or(1.0),
    };
    if adj != filter::Adjust::default() {
        img = filter::adjust(&s(&img)?, &adj).map_err(e("adjust"))?;
    }
    if a.has("--grayscale") {
        img = filter::grayscale(&s(&img)?, filter::Luma::Rec709).map_err(e("grayscale"))?;
    }
    if a.has("--premultiply") {
        img = filter::premultiply_alpha(&s(&img)?).map_err(e("premultiply"))?;
    }
    if a.has("--unpremultiply") {
        img = filter::unpremultiply_alpha(&s(&img)?).map_err(e("unpremultiply"))?;
    }
    if let Some(m) = a.get("--swizzle") {
        img = filter::swizzle(&s(&img)?, swizzle_map(m)?).map_err(e("swizzle"))?;
    }
    if let Some(st) = a.num::<f32>("--normal-map")? {
        let o = filter::NormalMapOptions {
            strength: st,
            directx: a.has("--directx"),
            ..Default::default()
        };
        img = filter::normal_map(&s(&img)?, &o).map_err(e("normal map"))?;
    }
    if let Some(t) = a.get("--tonemap") {
        let curve = match t {
            "clamp" => filter::ToneMap::Clamp,
            "reinhard" => match a.num::<f32>("--white")? {
                Some(w) => filter::ToneMap::ReinhardExtended { white: w },
                None => filter::ToneMap::Reinhard,
            },
            "aces" => filter::ToneMap::Aces,
            _ => return Err(format!("unknown tone map `{t}`")),
        };
        let o = filter::ToneMapOptions {
            curve,
            ..Default::default()
        };
        img = filter::tone_map(&s(&img)?, &o).map_err(e("tone map"))?;
    }
    if let Some(path) = a.get("--lut") {
        let l = load_lut(path)?;
        img = apply_lut(&img, &l, lut_interp(a)?)?;
    }
    if let Some(p) = a.get("--pixel") {
        let f = parse_format(p)?;
        if f.is_compressed() {
            return Err("--pixel needs an uncompressed format (use --format for textures)".into());
        }
        let mut out = Image::new(f, img.width, img.height).map_err(e("image"))?;
        imgio::convert::convert(&s(&img)?, f, &mut out.data).map_err(e("convert"))?;
        img = out;
    }
    Ok(img)
}

// ---------------------------------------------------------------------------------------------
// Textures

fn quality(a: &Args) -> Res<Quality> {
    Ok(match a.get("--quality").unwrap_or("default") {
        "fastest" => Quality::Fastest,
        "fast" => Quality::Fast,
        "default" => Quality::Default,
        "high" => Quality::High,
        "best" => Quality::Best,
        q => return Err(format!("unknown quality `{q}`")),
    })
}

fn mips(a: &Args) -> Res<Mips> {
    Ok(if a.has("--no-mips") {
        Mips::None
    } else if let Some(n) = a.num::<u32>("--levels")? {
        Mips::Count(n)
    } else {
        Mips::Full
    })
}

/// Writes `tex` by the output extension.
fn write_texture(out: &str, tex: &Texture, a: &Args) -> Res {
    let bytes = match ext(out).as_str() {
        "ktx2" => {
            let sc = match a.num::<i32>("--zstd")? {
                Some(l) => imgio::ktx2::Supercompression::Zstd(l),
                None => imgio::ktx2::Supercompression::None,
            };
            tex.to_ktx2(&imgio::ktx2::WriteOptions {
                supercompression: sc,
                key_values: &[],
            })
            .map_err(e(out))?
        }
        "dds" => tex
            .to_dds(&imgio::dds::DdsWriteOptions {
                legacy: a.has("--legacy-dds"),
            })
            .map_err(e(out))?,
        "ktx" => tex.to_ktx1(&[]).map_err(e(out))?,
        x => {
            return Err(format!(
                "`.{x}` is not a texture container (ktx2, dds, ktx)"
            ));
        }
    };
    write(out, &bytes)
}

fn basis_codec(s: &str) -> Res<imgio::basis::BasisCodec> {
    use imgio::basis::BasisCodec as C;
    let astc = |rest: &str| -> Res<AstcBlock> {
        AstcBlock::ALL
            .into_iter()
            .find(|b| format!("{b:?}")[1..].eq_ignore_ascii_case(rest))
            .ok_or(format!("unknown ASTC block `{rest}`"))
    };
    Ok(match s {
        "etc1s" => C::Etc1s,
        "uastc" => C::UastcLdr4x4,
        "uastc-hdr" => C::UastcHdr4x4,
        "astc-hdr6x6" => C::AstcHdr6x6,
        "uastc-hdr6x6" => C::UastcHdr6x6Intermediate,
        "xubc7" => C::Xubc7,
        x if x.starts_with("xuastc") => C::XuastcLdr(astc(&x[6..])?),
        x if x.starts_with("astc") => C::AstcLdr(astc(&x[4..])?),
        _ => return Err(format!("unknown Basis codec `{s}`")),
    })
}

/// Builds and writes a texture from `images` (`faces` 6 for a cube, `layers` for arrays).
fn make_texture(out: &str, images: &[Image], layers: u32, faces: u32, a: &Args) -> Res {
    let surfaces: Vec<Surface> = images
        .iter()
        .map(|i| i.surface())
        .collect::<Result<_, _>>()
        .map_err(e("image"))?;
    if let Some(codec) = a.get("--basis") {
        if ext(out) != "ktx2" {
            return Err("--basis writes KTX2 only".into());
        }
        let mut o = imgio::basis::BasisEncodeOptions {
            codec: basis_codec(codec)?,
            srgb: !a.has("--linear"),
            mipmaps: !a.has("--no-mips"),
            ..Default::default()
        };
        if faces == 6 {
            o.tex_type = imgio::basis::BasisTexType::CubemapArray;
        } else if layers > 0 {
            o.tex_type = imgio::basis::BasisTexType::Tex2DArray;
        }
        let k = imgio::ktx2::BasisKtx2Options {
            zstd: a.num("--zstd")?,
            key_values: &[],
        };
        let bytes = imgio::ktx2::encode_basis(&surfaces, &o, &k).map_err(e("basis"))?;
        return write(out, &bytes);
    }
    let default_fmt = match images.first().map(|i| i.format) {
        Some(f) if f.is_hdr() => "rgba16f",
        _ => "rgba8-srgb",
    };
    let fmt = parse_format(a.get("--format").unwrap_or(default_fmt))?;
    let mut o = BuildOptions::new(fmt);
    o.mips = mips(a)?;
    o.normal_map = a.has("--normal");
    o.encode = EncodeOptions {
        quality: quality(a)?,
        ..Default::default()
    };
    let tex = texture::build(&surfaces, layers, faces, &o).map_err(e("texture"))?;
    write_texture(out, &tex, a)
}

fn is_texture_ext(x: &str) -> bool {
    matches!(x, "ktx2" | "dds" | "ktx")
}

fn save_image(out: &str, img: &Image) -> Res {
    let fmt =
        FileFormat::from_extension(&ext(out)).ok_or(format!("unknown output type `{out}`"))?;
    let s = img.surface().map_err(e("image"))?;
    let bytes = match file::encode(fmt, &s) {
        Ok(b) => b,
        // Formats the file cannot store (e.g. float into PNG): try 8-bit sRGB / RGBA.
        Err(imgio::Error::Unsupported) if !matches!(img.format, Format::Rgba8Srgb) => {
            let mut c = Image::new(Format::Rgba8Srgb, img.width, img.height).map_err(e("image"))?;
            imgio::convert::convert(&s, Format::Rgba8Srgb, &mut c.data).map_err(e("convert"))?;
            file::encode(fmt, &c.surface().map_err(e("image"))?).map_err(e(out))?
        }
        Err(err) => return Err(format!("{out}: {err}")),
    };
    write(out, &bytes)
}

fn convert(rest: &[String]) -> Res {
    let a = Args::parse(rest)?;
    let [input, output] = a.pos.as_slice() else {
        return Err("convert <input> <output>".into());
    };
    let data = read(input)?;
    let img = load_bytes(input, &data)?;
    let img = process(img, &a, Some(&data))?;
    if is_texture_ext(&ext(output)) {
        make_texture(output, &[img], 0, 1, &a)
    } else {
        save_image(output, &img)
    }
}

fn texture_cmd(rest: &[String]) -> Res {
    let a = Args::parse(rest)?;
    let out = a.output()?;
    if a.pos.is_empty() {
        return Err("texture needs input images".into());
    }
    let mut images = Vec::new();
    for p in &a.pos {
        let data = read(p)?;
        images.push(process(load_bytes(p, &data)?, &a, Some(&data))?);
    }
    let n = images.len() as u32;
    if images
        .iter()
        .any(|i| (i.width, i.height) != (images[0].width, images[0].height))
    {
        return Err("all texture images need the same size".into());
    }
    if a.has("--cube") && images[0].width != images[0].height {
        return Err("cube faces must be square".into());
    }
    let (layers, faces) = if a.has("--cube") {
        if !n.is_multiple_of(6) {
            return Err("--cube needs 6 images per cube (+X -X +Y -Y +Z -Z)".into());
        }
        (if n > 6 { n / 6 } else { 0 }, 6)
    } else if a.has("--array") || n > 1 {
        (n, 1)
    } else {
        (0, 1)
    };
    make_texture(out, &images, layers, faces, &a)
}

// ---------------------------------------------------------------------------------------------
// pack

fn pack_cmd(rest: &[String]) -> Res {
    let a = Args::parse(rest)?;
    let out = a.output()?;
    let mut files: Vec<String> = Vec::new();
    let idx = |f: &str, files: &mut Vec<String>| -> usize {
        match files.iter().position(|x| x == f) {
            Some(i) => i,
            None => {
                files.push(f.to_string());
                files.len() - 1
            }
        }
    };
    let mut o = PackOptions::new(parse_format(a.get("--pixel").unwrap_or("rgba8"))?);
    if let Some(v) = a.get("--orm") {
        let p: Vec<&str> = v.split('\u{1f}').collect();
        for (c, f) in p.iter().enumerate() {
            o.channels[c] = Some(ChannelSpec::input(idx(f, &mut files), Channel::R));
        }
    }
    for (c, key) in ["--r", "--g", "--b", "--a"].iter().enumerate() {
        let Some(src) = a.get(key) else { continue };
        if let Ok(v) = src.parse::<f32>() {
            o.channels[c] = Some(ChannelSpec::constant(v));
            continue;
        }
        let mut parts = src.split(':');
        let f = parts.next().unwrap_or("");
        let mut spec = ChannelSpec::input(idx(f, &mut files), Channel::R);
        for p in parts {
            let ch = match p {
                "r" => Some(Channel::R),
                "g" => Some(Channel::G),
                "b" => Some(Channel::B),
                "a" => Some(Channel::A),
                "luma" => Some(Channel::Luma),
                "avg" => Some(Channel::Average),
                "max" => Some(Channel::Max),
                "invert" => {
                    spec = spec.inverted();
                    None
                }
                _ => return Err(format!("bad channel source `{src}`")),
            };
            if let (Some(ch), pack::Source::Input { input, .. }) = (ch, spec.source) {
                spec.source = pack::Source::Input { input, channel: ch };
            }
        }
        o.channels[c] = Some(spec);
    }
    o.size = match a.get("--size").unwrap_or("same") {
        "same" => PackSize::Same,
        "largest" => PackSize::Largest,
        "first" => PackSize::First,
        s => {
            let (w, h) = size2(s)?;
            PackSize::Fixed(w, h)
        }
    };
    let images: Vec<Image> = files.iter().map(|f| load(f)).collect::<Res<_>>()?;
    let surfaces: Vec<Surface> = images
        .iter()
        .map(|i| i.surface())
        .collect::<Result<_, _>>()
        .map_err(e("image"))?;
    let img = pack::pack(&surfaces, &o).map_err(e("pack"))?;
    if is_texture_ext(&ext(out)) {
        make_texture(out, &[img], 0, 1, &a)
    } else {
        save_image(out, &img)
    }
}

// ---------------------------------------------------------------------------------------------
// atlas

fn page_path(out: &str, i: usize, pages: usize) -> String {
    if pages == 1 {
        return out.to_string();
    }
    let p = std::path::Path::new(out);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("atlas");
    let x = p.extension().and_then(|s| s.to_str()).unwrap_or("png");
    p.with_file_name(format!("{stem}{i}.{x}"))
        .to_string_lossy()
        .into_owned()
}

fn atlas_cmd(rest: &[String]) -> Res {
    let a = Args::parse(rest)?;
    let out = a.output()?;
    if a.pos.is_empty() {
        return Err("atlas needs input images".into());
    }
    let mut o = AtlasOptions {
        multiple_of: 1,
        ..Default::default()
    };
    o.algorithm = match a.get("--algorithm").unwrap_or("maxrects") {
        "maxrects" | "maxrects-bssf" => Algorithm::MaxRects(MaxRectsHeuristic::BestShortSideFit),
        "maxrects-blsf" => Algorithm::MaxRects(MaxRectsHeuristic::BestLongSideFit),
        "maxrects-baf" => Algorithm::MaxRects(MaxRectsHeuristic::BestAreaFit),
        "maxrects-bl" => Algorithm::MaxRects(MaxRectsHeuristic::BottomLeft),
        "maxrects-cp" => Algorithm::MaxRects(MaxRectsHeuristic::ContactPoint),
        "skyline" => Algorithm::Skyline,
        "shelf" => Algorithm::Shelf,
        x => return Err(format!("unknown algorithm `{x}`")),
    };
    o.sort = match a.get("--sort").unwrap_or("maxside") {
        "maxside" => SortOrder::MaxSide,
        "area" => SortOrder::Area,
        "height" => SortOrder::Height,
        "width" => SortOrder::Width,
        "perimeter" => SortOrder::Perimeter,
        "none" => SortOrder::None,
        x => return Err(format!("unknown sort `{x}`")),
    };
    if let Some(m) = a.get("--max") {
        (o.max_width, o.max_height) = size2(m)?;
    }
    if let Some(f) = a.get("--fixed") {
        let (w, h) = size2(f)?;
        o.size = PageSize::Fixed(w, h);
    }
    o.power_of_two = a.has("--pot");
    o.square = a.has("--square");
    o.multiple_of = a.num("--multiple")?.unwrap_or(1);
    o.padding = a.num("--padding")?.unwrap_or(2);
    o.border = a.num("--border")?.unwrap_or(0);
    o.extrude = a.num("--extrude")?.unwrap_or(0);
    o.rotation = a.has("--rotate");
    o.dedupe = a.has("--dedupe");
    o.multipage = a.has("--pages");
    if a.has("--trim") {
        o.trim = Trim::Alpha(a.get("--trim").and_then(|v| v.parse().ok()).unwrap_or(0.0));
    }
    if let Some(p) = a.get("--pixel") {
        o.format = parse_format(p)?;
    }
    let images: Vec<Image> = a.pos.iter().map(|f| load(f)).collect::<Res<_>>()?;
    let surfaces: Vec<Surface> = images
        .iter()
        .map(|i| i.surface())
        .collect::<Result<_, _>>()
        .map_err(e("image"))?;
    let at = atlas::build(&surfaces, &o).map_err(e("atlas"))?;
    let n = at.pages.len();
    for (i, page) in at.pages.iter().enumerate() {
        let p = page_path(out, i, n);
        if is_texture_ext(&ext(&p)) {
            let mut c = Image::new(page.format, page.width, page.height).map_err(e("image"))?;
            c.data.copy_from_slice(&page.data);
            make_texture(&p, &[c], 0, 1, &a)?;
        } else {
            save_image(&p, page)?;
        }
    }
    if let Some(j) = a.get("--json") {
        let names: Vec<String> = a
            .pos
            .iter()
            .map(|p| {
                std::path::Path::new(p)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(p)
                    .to_string()
            })
            .collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let base = std::path::Path::new(out)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(out);
        let pattern = if n == 1 {
            base.to_string()
        } else {
            page_path(base, 0, 2).replacen('0', "{}", 1)
        };
        let json = at.to_json(&refs, &pattern).map_err(e("json"))?;
        write(j, &json)?;
    }
    println!(
        "{} sprites on {n} page(s): {}",
        a.pos.len(),
        at.pages
            .iter()
            .map(|p| format!("{}x{}", p.width, p.height))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// ibl

fn cube_texture(faces: &[Image; 6]) -> Res<Texture> {
    let f = faces[0].format;
    let mut level = Vec::new();
    for img in faces {
        level.extend_from_slice(&img.data);
    }
    let mut levels = vas::Vector::default();
    let mut b = imgio::Buffer::default();
    b.extend_from_slice(&level).map_err(|_| "out of memory")?;
    levels.push(b).map_err(|_| "out of memory")?;
    Ok(Texture {
        format: f,
        width: faces[0].width,
        height: faces[0].height,
        layers: 0,
        faces: 6,
        levels,
    })
}

fn ibl_cmd(rest: &[String]) -> Res {
    let a = Args::parse(rest)?;
    let [input] = a.pos.as_slice() else {
        return Err("ibl <panorama> -o <env.ktx2>".into());
    };
    let out = a.output()?;
    let pano = load(input)?;
    let size: u32 = a.num("--size")?.unwrap_or(512);
    let fmt = parse_format(a.get("--pixel").unwrap_or("rgba16f"))?;
    let faces = ibl::equirect_to_cube(&pano.surface().map_err(e("image"))?, size, fmt)
        .map_err(e("cube"))?;
    let surf = ibl::surfaces(&faces).map_err(e("cube"))?;
    if a.has("--no-mips") {
        write_texture(out, &cube_texture(&faces)?, &a)?;
    } else {
        let tex = texture::build(&surf, 0, 6, &{
            let mut o = BuildOptions::new(fmt);
            o.mips = mips(&a)?;
            o
        })
        .map_err(e("texture"))?;
        write_texture(out, &tex, &a)?;
    }
    if let Some(p) = a.get("--irradiance") {
        let n = a.num("--irradiance-size")?.unwrap_or(32);
        let irr = ibl::irradiance(&surf, n, ibl::Irradiance::Sh, fmt).map_err(e("irradiance"))?;
        write_texture(p, &cube_texture(&irr)?, &a)?;
    }
    if let Some(p) = a.get("--specular") {
        let o = ibl::SpecularOptions {
            size: a.num("--specular-size")?.unwrap_or(0),
            levels: a.num("--spec-levels")?.unwrap_or(0),
            samples: a.num("--samples")?.unwrap_or(512),
            format: fmt,
        };
        let lv = ibl::prefilter_specular(&surf, &o).map_err(e("specular"))?;
        let mut levels = vas::Vector::default();
        for faces in lv.iter() {
            let mut b = imgio::Buffer::default();
            for img in faces {
                b.extend_from_slice(&img.data)
                    .map_err(|_| "out of memory")?;
            }
            levels.push(b).map_err(|_| "out of memory")?;
        }
        let w = lv.first().map_or(1, |f| f[0].width);
        write_texture(
            p,
            &Texture {
                format: fmt,
                width: w,
                height: w,
                layers: 0,
                faces: 6,
                levels,
            },
            &a,
        )?;
    }
    if let Some(p) = a.get("--brdf") {
        let n = a.num("--brdf-size")?.unwrap_or(256);
        let f = if is_texture_ext(&ext(p)) {
            Format::Rg16Unorm
        } else {
            Format::Rgba16Unorm
        };
        let lut = ibl::brdf_lut(n, a.num("--samples")?.unwrap_or(512), f).map_err(e("brdf"))?;
        if is_texture_ext(&ext(p)) {
            let mut o = BuildOptions::new(f);
            o.mips = Mips::None;
            let tex = texture::build(&[lut.surface().map_err(e("image"))?], 0, 1, &o)
                .map_err(e("brdf"))?;
            write_texture(p, &tex, &a)?;
        } else {
            save_image(p, &lut)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// compare

fn thresholds(a: &Args) -> Res<Thresholds> {
    Ok(Thresholds {
        min_psnr: a.num("--min-psnr")?,
        min_ssim: a.num("--min-ssim")?,
        max_error: a.num("--max-error")?,
        max_mean_error: a.num("--max-mean-error")?,
        max_mse: a.num("--max-mse")?,
        scope: match a.get("--scope").unwrap_or("channels") {
            "channels" => Scope::Channels,
            "rgb" => Scope::Rgb,
            "all" => Scope::All,
            s => return Err(format!("unknown scope `{s}` (channels|rgb|all)")),
        },
    })
}

/// Outcome of comparing one pair.
struct Cmp {
    path: String,
    /// `pass`, `fail`, `missing` (no test file), `extra` (no reference) or `error`.
    status: &'static str,
    metrics: Option<metrics::Metrics>,
    failures: Vec<metrics::Failure>,
    message: String,
}

fn compare_pair(ra: &str, rb: &str, a: &Args, th: &Thresholds, diff: Option<&str>) -> Cmp {
    let mut c = Cmp {
        path: rb.to_string(),
        status: "pass",
        metrics: None,
        failures: Vec::new(),
        message: String::new(),
    };
    let run = |c: &mut Cmp| -> Res {
        let (ia, ib) = (load(ra)?, load(rb)?);
        if (ia.width, ia.height) != (ib.width, ib.height) {
            c.status = "fail";
            c.message = format!(
                "sizes differ: {}x{} vs {}x{}",
                ia.width, ia.height, ib.width, ib.height
            );
            return Ok(());
        }
        let (sa, sb) = (ia.surface().map_err(e(ra))?, ib.surface().map_err(e(rb))?);
        let o = CompareOptions {
            peak: a.num("--peak")?.unwrap_or(1.0),
            ssim: !a.has("--no-ssim"),
        };
        let m = metrics::compare(&sa, &sb, &o).map_err(e("compare"))?;
        c.failures = th
            .check(&m)
            .map_err(e("compare"))?
            .iter()
            .copied()
            .collect();
        if !c.failures.is_empty() {
            c.status = "fail";
        }
        c.metrics = Some(m);
        if let Some(d) = diff {
            let img = metrics::difference(&sa, &sb, a.num("--amplify")?.unwrap_or(1.0))
                .map_err(e("diff"))?;
            if let Some(parent) = std::path::Path::new(d).parent() {
                std::fs::create_dir_all(parent).map_err(e(d))?;
            }
            save_image(d, &img)?;
        }
        Ok(())
    };
    if let Err(err) = run(&mut c) {
        c.status = "error";
        c.message = err;
    }
    c
}

fn json_num(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".into()
    }
}

fn cmp_json(c: &Cmp) -> String {
    let ch = |m: &metrics::ChannelMetrics| {
        format!(
            "{{\"mse\":{},\"psnr\":{},\"max_error\":{},\"mean_error\":{},\"ssim\":{}}}",
            json_num(m.mse),
            json_num(m.psnr),
            json_num(m.max_error),
            json_num(m.mean_error),
            json_num(m.ssim)
        )
    };
    let metrics = c.metrics.as_ref().map_or("null".to_string(), |m| {
        let names = ["R", "G", "B", "A"];
        let per: Vec<String> = names
            .iter()
            .zip(m.per_channel.iter())
            .take(m.channels)
            .map(|(n, v)| format!("\"{n}\":{}", ch(v)))
            .collect();
        format!(
            "{{{},\"rgb\":{},\"all\":{}}}",
            per.join(","),
            ch(&m.rgb),
            ch(&m.all)
        )
    });
    let fails: Vec<String> = c
        .failures
        .iter()
        .map(|f| {
            format!(
                "{{\"metric\":\"{}\",\"channel\":\"{}\",\"value\":{},\"limit\":{}}}",
                f.metric,
                f.channel,
                json_num(f.value),
                json_num(f.limit)
            )
        })
        .collect();
    format!(
        "{{\"path\":{},\"status\":\"{}\",\"message\":{},\"metrics\":{metrics},\"failures\":[{}]}}",
        json_str(&c.path),
        c.status,
        json_str(&c.message),
        fails.join(",")
    )
}

fn cmp_reasons(c: &Cmp) -> String {
    let mut r: Vec<String> = c
        .failures
        .iter()
        .map(|f| {
            let op = if matches!(f.metric, "psnr" | "ssim") {
                "<"
            } else {
                ">"
            };
            format!("{} {} {:.6} {op} {}", f.metric, f.channel, f.value, f.limit)
        })
        .collect();
    if !c.message.is_empty() {
        r.insert(0, c.message.clone());
    }
    r.join("; ")
}

fn compare(rest: &[String]) -> Res {
    let a = Args::parse_with(rest, &["--json", "--strict"])?;
    let [ra, rb] = a.pos.as_slice() else {
        return Err("compare <reference> <test> (files or directories)".into());
    };
    let th = thresholds(&a)?;
    let json = a.has("--json");
    let is_dir = |p: &str| std::path::Path::new(p).is_dir();
    if !(is_dir(ra) && is_dir(rb)) {
        if is_dir(ra) || is_dir(rb) {
            return Err("compare two files or two directories".into());
        }
        let c = compare_pair(ra, rb, &a, &th, a.get("--diff"));
        if json {
            println!("{}", cmp_json(&c));
        } else if let Some(m) = &c.metrics {
            let names = ["R", "G", "B", "A"];
            println!(
                "{:>4} {:>12} {:>9} {:>10} {:>10} {:>8}",
                "", "MSE", "PSNR dB", "max err", "mean err", "SSIM"
            );
            let row = |n: &str, c: &metrics::ChannelMetrics| {
                println!(
                    "{n:>4} {:>12.6e} {:>9.3} {:>10.6} {:>10.6} {:>8.5}",
                    c.mse, c.psnr, c.max_error, c.mean_error, c.ssim
                );
            };
            for (n, ch) in names.iter().zip(m.per_channel.iter()).take(m.channels) {
                row(n, ch);
            }
            row("RGB", &m.rgb);
            row("all", &m.all);
            if th.any() {
                println!(
                    "{}",
                    if c.status == "pass" {
                        "PASS".to_string()
                    } else {
                        format!("FAIL: {}", cmp_reasons(&c))
                    }
                );
            }
        }
        return match c.status {
            "pass" => Ok(()),
            "error" => Err(c.message),
            _ => Err(cmp_reasons(&c)),
        };
    }
    // Directories: every image / texture under the reference, matched by relative path.
    let rel = |root: &str, p: &str| -> String {
        std::path::Path::new(p)
            .strip_prefix(root)
            .map(|r| r.to_string_lossy().into_owned())
            .unwrap_or_else(|_| p.to_string())
    };
    let (mut fa, mut fb) = (Vec::new(), Vec::new());
    collect_files(ra, &mut fa)?;
    collect_files(rb, &mut fb)?;
    let ra_rel: Vec<String> = fa.iter().map(|p| rel(ra, p)).collect();
    let mut results = Vec::new();
    for (pa, r) in fa.iter().zip(&ra_rel) {
        let pb = std::path::Path::new(rb).join(r);
        let pbs = pb.to_string_lossy().into_owned();
        if !pb.is_file() {
            results.push(Cmp {
                path: r.clone(),
                status: "missing",
                metrics: None,
                failures: Vec::new(),
                message: format!("no {r} in {rb}"),
            });
            continue;
        }
        let mut c = compare_pair(pa, &pbs, &a, &th, None);
        // Diff images for the failures only (CI artifacts).
        if let Some(d) = a.get("--diff")
            && c.status == "fail"
            && c.metrics.is_some()
        {
            let out = std::path::Path::new(d).join(format!("{r}.diff.png"));
            let o = out.to_string_lossy().into_owned();
            let again = compare_pair(pa, &pbs, &a, &Thresholds::default(), Some(&o));
            if again.status == "error" {
                c.message = again.message;
            }
        }
        c.path = r.clone();
        results.push(c);
    }
    for pb in &fb {
        let r = rel(rb, pb);
        if !ra_rel.contains(&r) {
            results.push(Cmp {
                path: r.clone(),
                status: "extra",
                metrics: None,
                failures: Vec::new(),
                message: format!("no reference {r} in {ra}"),
            });
        }
    }
    let count = |s: &str| results.iter().filter(|c| c.status == s).count();
    let strict = a.has("--strict");
    let bad =
        count("fail") + count("missing") + count("error") + if strict { count("extra") } else { 0 };
    if json {
        let items: Vec<String> = results.iter().map(cmp_json).collect();
        println!(
            "{{\"files\":[{}],\"passed\":{},\"failed\":{},\"missing\":{},\"extra\":{},\"errors\":{}}}",
            items.join(","),
            count("pass"),
            count("fail"),
            count("missing"),
            count("extra"),
            count("error")
        );
    } else {
        for c in &results {
            let summary = c.metrics.as_ref().map_or(String::new(), |m| {
                format!(
                    "  psnr {:.2} dB  ssim {:.5}  max {:.6}",
                    m.all.psnr, m.all.ssim, m.all.max_error
                )
            });
            let why = cmp_reasons(c);
            println!(
                "{:<7} {}{summary}{}",
                c.status.to_uppercase(),
                c.path,
                if why.is_empty() {
                    String::new()
                } else {
                    format!("  [{why}]")
                }
            );
        }
        println!(
            "{} passed, {} failed, {} missing, {} extra, {} errors",
            count("pass"),
            count("fail"),
            count("missing"),
            count("extra"),
            count("error")
        );
    }
    if bad > 0 {
        return Err(format!("{bad} of {} comparisons failed", results.len()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// LUTs

/// A LUT file: a `.cube` or an ICC device link.
enum LutFile {
    Cube(Cube),
    Link(Box<Profile>),
}

fn lut_interp(a: &Args) -> Res<Interpolation> {
    Ok(match a.get("--lut-interp").unwrap_or("tetrahedral") {
        "tetrahedral" => Interpolation::Tetrahedral,
        "trilinear" => Interpolation::Trilinear,
        s => return Err(format!("unknown LUT interpolation `{s}`")),
    })
}

/// `.cube`, `.icc` / `.icm` device link, or a HALD (square) / strip (`n*n x n`) image.
fn load_lut(path: &str) -> Res<LutFile> {
    let data = read(path)?;
    match ext(path).as_str() {
        "cube" => Ok(LutFile::Cube(Cube::parse(&data).map_err(e(path))?)),
        "icc" | "icm" => {
            let p = Profile::from_icc(&data).map_err(e(path))?;
            if !p.is_device_link() {
                return Err(format!("{path}: not a device link profile"));
            }
            Ok(LutFile::Link(Box::new(p)))
        }
        _ => {
            let img = load_bytes(path, &data)?;
            let s = img.surface().map_err(e(path))?;
            let (w, h) = (img.width as u64, img.height as u64);
            let lut = if w == h {
                Lut3d::from_hald(&s)
            } else if w == h * h {
                Lut3d::from_strip(&s)
            } else {
                return Err(format!(
                    "{path}: {w}x{h} is neither a HALD (square, side level^3) nor a strip (n*n x n)"
                ));
            }
            .map_err(e(path))?;
            Ok(LutFile::Cube(Cube::new(lut)))
        }
    }
}

fn apply_lut(img: &Image, l: &LutFile, interp: Interpolation) -> Res<Image> {
    let s = img.surface().map_err(e("lut"))?;
    match l {
        LutFile::Cube(c) => lut::apply(&s, c, interp, img.format),
        LutFile::Link(p) => imgio::color::apply_link(&s, p, img.format),
    }
    .map_err(e("lut"))
}

fn lut_cmd(rest: &[String]) -> Res {
    let a = Args::parse(rest)?;
    let out = a.output()?;
    let src = match a.pos.as_slice() {
        [] => None,
        [p] => Some(load_lut(p)?),
        _ => return Err("lut [input] -o <output>".into()),
    };
    const SPATIAL: &[&str] = &[
        "--crop",
        "--resize",
        "--flip",
        "--rotate",
        "--transpose",
        "--blur",
        "--box-blur",
        "--sharpen",
        "--normal-map",
        "--premultiply",
        "--unpremultiply",
        "--color",
    ];
    if let Some((k, _)) = a.opts.iter().find(|(k, _)| SPATIAL.contains(&k.as_str())) {
        return Err(format!("{k} is not a per-pixel colour operation (lut)"));
    }
    const KEEP: &[&str] = &["--output", "--title", "--lut-interp"];
    let copy = ext(out) == "cube" && a.opts.iter().all(|(k, _)| KEEP.contains(&k.as_str()));
    let mut cube = match src {
        Some(LutFile::Cube(c)) if copy => c,
        src => {
            let size = match (&src, a.num::<usize>("--size")?) {
                (_, Some(n)) => n,
                (Some(LutFile::Cube(Cube { lut: Some(l), .. })), None) => l.size,
                _ => 33,
            };
            let id = Lut3d::identity(size).map_err(e("--size"))?;
            let mut img = id.to_strip(Format::Rgba32Float).map_err(e("lut"))?;
            if let Some(l) = &src {
                img = apply_lut(&img, l, lut_interp(&a)?)?;
            }
            match (a.get("--from"), a.get("--to")) {
                (None, None) => {}
                (Some(f), Some(t)) => {
                    let from = Profile::new(color_space(f)?).map_err(e("--from"))?;
                    let to = Profile::new(color_space(t)?).map_err(e("--to"))?;
                    img = imgio::color::convert(
                        &surf(&img)?,
                        &from,
                        &to,
                        intent(&a)?,
                        Format::Rgba32Float,
                    )
                    .map_err(e("colour"))?;
                }
                _ => return Err("--from and --to go together".into()),
            }
            let img = process(img, &a, None)?;
            Cube::new(Lut3d::from_strip(&surf(&img)?).map_err(e("lut"))?)
        }
    };
    let title = a.get("--title");
    match ext(out).as_str() {
        "cube" => {
            if let Some(t) = title {
                let mut s = vas::String::default();
                s.push_str(t).map_err(|_| "title too long".to_string())?;
                cube.title = Some(s);
            }
            write(out, &cube.to_bytes().map_err(e(out))?)
        }
        x => {
            let l = cube.lut.as_ref().ok_or("no 3D table")?;
            if x == "icc" || x == "icm" {
                let b = l
                    .to_device_link(title.unwrap_or("imgio LUT"))
                    .map_err(e(out))?;
                return write(out, &b);
            }
            let square = (2..=16).any(|k| k * k == l.size);
            let img = match a
                .get("--layout")
                .unwrap_or(if square { "hald" } else { "strip" })
            {
                "hald" => l.to_hald(Format::Rgba16Unorm),
                "strip" => l.to_strip(Format::Rgba16Unorm),
                s => return Err(format!("unknown layout `{s}`")),
            }
            .map_err(e(out))?;
            save_image(out, &img)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// memory

/// One file of the memory report.
struct MemRow {
    path: String,
    layout: Layout,
    /// Current GPU format and bytes (`None` for Basis payloads).
    now: Option<(Format, u64)>,
    note: String,
    desktop: (Format, u64),
    mobile: (Format, u64),
}

fn human(b: u64) -> String {
    let v = b as f64;
    if b >= 1 << 30 {
        format!("{:.2} GiB", v / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.2} MiB", v / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KiB", v / 1024.0)
    } else {
        format!("{b} B")
    }
}

fn parse_size(s: &str) -> Res<u64> {
    let t = s.trim().trim_end_matches(['B', 'b']).trim_end_matches('i');
    let (num, mul) = match t.chars().last() {
        Some('K' | 'k') => (&t[..t.len() - 1], 1u64 << 10),
        Some('M' | 'm') => (&t[..t.len() - 1], 1 << 20),
        Some('G' | 'g') => (&t[..t.len() - 1], 1 << 30),
        _ => (t, 1),
    };
    let v: f64 = num.trim().parse().map_err(|_| format!("bad size `{s}`"))?;
    if !(v >= 0.0 && v.is_finite()) {
        return Err(format!("bad size `{s}`"));
    }
    Ok((v * mul as f64).round() as u64)
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn collect_files(path: &str, out: &mut Vec<String>) -> Res {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(p)
            .map_err(e(path))?
            .filter_map(|x| x.ok().map(|x| x.path()))
            .collect();
        entries.sort();
        for en in entries {
            let s = en.to_string_lossy().into_owned();
            if en.is_dir() {
                collect_files(&s, out)?;
            } else if is_texture_ext(&ext(&s))
                || matches!(ext(&s).as_str(), "basis" | "astc")
                || FileFormat::from_extension(&ext(&s)).is_some()
            {
                out.push(s);
            }
        }
    } else {
        out.push(path.to_string());
    }
    Ok(())
}

fn mem_row(path: &str, a: &Args) -> Res<MemRow> {
    let data = read(path)?;
    let mips = !a.has("--no-mips");
    let fixed = |k: &str| a.get(k).map(parse_format).transpose();
    let (fd, fm) = (fixed("--desktop")?, fixed("--mobile")?);
    let est =
        |c: &memory::Content, t: Target, over: Option<Format>, l: &Layout| -> Res<(Format, u64)> {
            let f = over.unwrap_or_else(|| memory::suggest(c, t));
            Ok((f, memory::texture_bytes(f, l).map_err(e(path))?))
        };
    if let Some(m) = memory::of_container(&data).map_err(e(path))? {
        let l = m.layout;
        let hdr = m.format.is_some_and(|f| f.is_hdr())
            || matches!(
                m.basis,
                Some(
                    imgio::basis::BasisTexFormat::UastcHdr4x4
                        | imgio::basis::BasisTexFormat::AstcHdr6x6
                        | imgio::basis::BasisTexFormat::UastcHdr6x6Intermediate
                )
            );
        // Plain containers: level 0 decoded and analyzed; else the header (R data is gray, RG
        // data two channels: BC5 / ASTC 4x4).
        let decoded = if m.basis.is_none() {
            load_bytes(path, &data)
                .ok()
                .and_then(|img| memory::analyze(&img.surface().ok()?).ok())
        } else {
            None
        };
        let header = memory::Content {
            alpha: if m.has_alpha || m.channels == 2 {
                memory::Alpha::Full
            } else {
                memory::Alpha::Opaque
            },
            gray: m.channels <= 2,
            hdr,
            srgb: m.srgb,
        };
        let c = match decoded {
            // Two channel data stays two channel.
            Some(d) if m.channels == 2 => memory::Content {
                alpha: memory::Alpha::Full,
                gray: true,
                ..d
            },
            Some(d) => memory::Content {
                hdr,
                srgb: m.srgb,
                ..d
            },
            None => header,
        };
        // Already in the target's block family: keep it.
        let family = |f: Format| {
            let n = format_name(f);
            if n.starts_with("bc") {
                Some(Target::Desktop)
            } else if ["astc", "etc", "eac", "pvrtc"]
                .iter()
                .any(|p| n.starts_with(p))
            {
                Some(Target::Mobile)
            } else {
                None
            }
        };
        let keep = |t: Target| m.format.filter(|f| family(*f) == Some(t));
        let (fd, fm) = (fd.or(keep(Target::Desktop)), fm.or(keep(Target::Mobile)));
        let (desktop, mobile, note) = match m.basis {
            Some(b) => {
                let t = |t: Target, over: Option<Format>| -> Res<(Format, u64)> {
                    let f = over.unwrap_or(memory::transcode_target(b, m.has_alpha, m.srgb, t));
                    Ok((f, memory::texture_bytes(f, &l).map_err(e(path))?))
                };
                (
                    t(Target::Desktop, fd)?,
                    t(Target::Mobile, fm)?,
                    format!("basis {b:?}"),
                )
            }
            None => (
                est(&c, Target::Desktop, fd, &l)?,
                est(&c, Target::Mobile, fm, &l)?,
                String::new(),
            ),
        };
        return Ok(MemRow {
            path: path.to_string(),
            layout: l,
            now: m.format.zip(m.gpu_bytes),
            note,
            desktop,
            mobile,
        });
    }
    let img = file::decode(&data).map_err(e(path))?;
    let c = memory::analyze(&img.surface().map_err(e(path))?).map_err(e(path))?;
    let l = Layout::new_2d(img.width, img.height, mips);
    let f = memory::upload_format(img.format);
    let now = (f, memory::texture_bytes(f, &l).map_err(e(path))?);
    let note = match (c.alpha, c.gray) {
        (memory::Alpha::Opaque, false) => String::new(),
        (al, g) => format!(
            "{}{}",
            match al {
                memory::Alpha::Opaque => "",
                memory::Alpha::Binary => "cutout alpha",
                memory::Alpha::Full => "alpha",
            },
            if g {
                if al == memory::Alpha::Opaque {
                    "gray"
                } else {
                    ", gray"
                }
            } else {
                ""
            }
        ),
    };
    Ok(MemRow {
        path: path.to_string(),
        layout: l,
        now: Some(now),
        note,
        desktop: est(&c, Target::Desktop, fd, &l)?,
        mobile: est(&c, Target::Mobile, fm, &l)?,
    })
}

fn memory_cmd(rest: &[String]) -> Res {
    let a = Args::parse_with(rest, &["--json"])?;
    if a.pos.is_empty() {
        return Err("memory <file|dir>...".into());
    }
    let mut paths = Vec::new();
    for p in &a.pos {
        collect_files(p, &mut paths)?;
    }
    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    for p in &paths {
        match mem_row(p, &a) {
            Ok(r) => rows.push(r),
            Err(err) => skipped.push(err),
        }
    }
    let now: u64 = rows.iter().filter_map(|r| r.now.map(|n| n.1)).sum();
    let desk: u64 = rows.iter().map(|r| r.desktop.1).sum();
    let mob: u64 = rows.iter().map(|r| r.mobile.1).sum();
    let unknown = rows.iter().filter(|r| r.now.is_none()).count();
    if a.has("--json") {
        let fmt_obj =
            |f: Format, b: u64| format!("{{\"format\":\"{}\",\"bytes\":{b}}}", format_name(f));
        let mut o = String::from("{\"files\":[");
        for (i, r) in rows.iter().enumerate() {
            let l = r.layout;
            o.push_str(&format!(
                "{}{{\"path\":{},\"width\":{},\"height\":{},\"depth\":{},\"layers\":{},\"faces\":{},\"levels\":{},\"now\":{},\"note\":{},\"desktop\":{},\"mobile\":{}}}",
                if i > 0 { "," } else { "" },
                json_str(&r.path),
                l.width,
                l.height,
                l.depth,
                l.layers,
                l.faces,
                l.levels,
                r.now.map_or("null".to_string(), |(f, b)| fmt_obj(f, b)),
                json_str(&r.note),
                fmt_obj(r.desktop.0, r.desktop.1),
                fmt_obj(r.mobile.0, r.mobile.1),
            ));
        }
        o.push_str(&format!(
            "],\"total\":{{\"now\":{now},\"desktop\":{desk},\"mobile\":{mob},\"unknown\":{unknown}}},\"skipped\":["
        ));
        for (i, s) in skipped.iter().enumerate() {
            o.push_str(&format!("{}{}", if i > 0 { "," } else { "" }, json_str(s)));
        }
        o.push_str("]}");
        println!("{o}");
    } else {
        let w = rows
            .iter()
            .map(|r| r.path.len())
            .max()
            .unwrap_or(4)
            .clamp(4, 60);
        println!(
            "{:<w$} {:>11} {:>5} {:>22} {:>22} {:>24}",
            "file", "size", "mips", "now", "desktop", "mobile"
        );
        let cell = |f: Format, b: u64| format!("{} {}", human(b), format_name(f));
        for r in &rows {
            let l = r.layout;
            let mut dims = format!("{}x{}", l.width, l.height);
            if l.depth > 1 {
                dims.push_str(&format!("x{}", l.depth));
            }
            let mut extra = Vec::new();
            if l.faces == 6 {
                extra.push("cube".to_string());
            }
            if l.layers > 1 {
                extra.push(format!("{} layers", l.layers));
            }
            if !r.note.is_empty() {
                extra.push(r.note.clone());
            }
            println!(
                "{:<w$} {:>11} {:>5} {:>22} {:>22} {:>24}{}",
                r.path,
                dims,
                l.levels,
                r.now.map_or("transcoded".to_string(), |(f, b)| cell(f, b)),
                cell(r.desktop.0, r.desktop.1),
                cell(r.mobile.0, r.mobile.1),
                if extra.is_empty() {
                    String::new()
                } else {
                    format!("  ({})", extra.join(", "))
                }
            );
        }
        let pct = |x: u64| {
            if now > 0 {
                format!(" ({:+.0}%)", (x as f64 / now as f64 - 1.0) * 100.0)
            } else {
                String::new()
            }
        };
        println!(
            "total {} files: now {}{}, desktop {}{}, mobile {}{}",
            rows.len(),
            human(now),
            if unknown > 0 {
                format!(" + {unknown} transcoded")
            } else {
                String::new()
            },
            human(desk),
            if unknown == 0 {
                pct(desk)
            } else {
                String::new()
            },
            human(mob),
            if unknown == 0 {
                pct(mob)
            } else {
                String::new()
            },
        );
        for s in &skipped {
            eprintln!("skipped: {s}");
        }
    }
    if let Some(b) = a.get("--budget") {
        let limit = parse_size(b)?;
        if now > limit {
            return Err(format!(
                "over budget: {} > {} ({} over)",
                human(now),
                human(limit),
                human(now - limit)
            ));
        }
    }
    Ok(())
}
