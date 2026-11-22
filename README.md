# imgio

Pure Rust image and GPU texture codecs: BC1-BC7, ETC / EAC, ASTC, PVRTC, Basis Universal
(ETC1S, UASTC, XUASTC, XUBC7) encoding and transcoding, RGBA and HDR variants, KTX2 / KTX1 /
DDS / `.astc` / `.basis` containers, mipmaps, image files (PNG, JPEG, WebP, AVIF, EXR, TIFF,
JPEG XL, GIF, ...), resize and filters. No C/C++ bindings.

Status: 0.1, every [ROADMAP.md](ROADMAP.md) item done; the API may still change before 1.0.
Minimum Rust version: 1.88 (the `avif` feature needs 1.90, `jpeg2000` 1.92).

## Usage

```toml
[dependencies]
imgio = "0.1"
```

Encode RGBA pixels to BC7 and write a KTX2 file with a full mip chain:

```rust
use imgio::texture::{build, BuildOptions};
use imgio::{EncodeOptions, Format, Quality, Surface};

fn main() -> imgio::Result<()> {
    let (w, h) = (64, 64);
    let rgba = vec![200u8; (w * h * 4) as usize];
    let src = Surface::new(Format::Rgba8Srgb, w, h, &rgba)?;

    // One surface, high quality BC7.
    let opts = EncodeOptions { quality: Quality::High, ..Default::default() };
    let bc7 = imgio::encode_to_buffer(&src, Format::Bc7Srgb, &opts)?;
    assert_eq!(bc7.len(), (w * h) as usize);

    // Mips + encode + container in one call.
    let tex = build(&[src], 0, 1, &BuildOptions::new(Format::Bc7Srgb))?;
    let ktx2 = tex.to_ktx2(&imgio::ktx2::WriteOptions::default())?;

    // Read it back and decode level 1 to RGBA8.
    let file = imgio::ktx2::Ktx2File::parse(&ktx2)?;
    let level1 = file.image_data(1, 0, 0)?;
    let rgba1 = imgio::decode_to_buffer(Format::Bc7Srgb, level1, 32, 32)?;
    assert_eq!(rgba1.len(), 32 * 32 * 4);
    Ok(())
}
```

Transcode a Basis Universal KTX2 file to whatever the GPU supports:

```rust,no_run
use imgio::Format;

fn main() -> imgio::Result<()> {
    let bytes = std::fs::read("texture.ktx2").map_err(|_| imgio::Error::InvalidData)?;
    let file = imgio::ktx2::Ktx2File::parse(&bytes)?;
    let target = if file.can_transcode_to(Format::Bc7Unorm) { Format::Bc7Unorm } else { Format::Rgba8Unorm };
    let level0 = file.transcode_to_buffer(0, 0, 0, target)?;
    println!("{} bytes", level0.len());
    Ok(())
}
```

Load an image file, resize it, sharpen it and save it as PNG:

```rust
use imgio::file::{self, FileFormat};
use imgio::filter;
use imgio::resize::{resize, Fit, ResizeOptions};
use imgio::{Format, Surface};

fn main() -> imgio::Result<()> {
    // Any supported file: the format is detected from the bytes.
    let rgba = vec![128u8; 64 * 48 * 4];
    let png = file::encode(FileFormat::Png, &Surface::new(Format::Rgba8Srgb, 64, 48, &rgba)?)?;
    let img = file::decode(&png)?;

    // Fit inside 32x32 (keeps the aspect ratio: 32x24), sRGB-correct Lanczos.
    let opts = ResizeOptions { fit: Fit::Contain, ..Default::default() };
    let small = resize(&img.surface()?, 32, 32, &opts)?;
    assert_eq!((small.width, small.height), (32, 24));

    let sharp = filter::unsharp_mask(&small.surface()?, 1.0, 0.5, 0.0)?;
    let out = file::encode(FileFormat::Png, &sharp.surface()?)?;
    assert!(!out.is_empty());
    Ok(())
}
```

### Cargo features

| Feature | Default | Enables |
|---|---|---|
| `png`, `jpeg` | yes | PNG / APNG and JPEG files (`imgio::file`) |
| `gif`, `webp`, `exr`, `tiff`, `jxl`, `jpeg2000`, `vdb`, `avif` | no | the other image file formats (pure Rust crates) |
| `vulkan`, `opengl` | no | GPU texture upload (the only bindings; loaded at run time) |

## Formats

Encoders run through `imgio::encode` (quality `Fastest`..`Best`), decoders through
`imgio::decode`, uncompressed formats through `imgio::convert`. "Port" means a Rust
translation that is byte identical with the named C++ reference; "own" means imgio's own code.

### Uncompressed

| Type | Encoder | Decoder | Why / typical use |
|---|---|---|---|
| R8 / RG8 (unorm, snorm) | `convert` | `convert` | Masks, roughness / AO, 2-channel normal maps |
| R16 / RG16 (unorm, snorm) | `convert` | `convert` | Height maps, precise single / dual channel data |
| RGBA8 / RGBA8 sRGB | `convert` | `convert` | Universal 8-bit color, the default decode target |
| RGB8 / RGB8 sRGB | `convert` | `convert` | Opaque color without alpha padding |
| BGRA8 / BGRA8 sRGB | `convert` | `convert` | Swapchain / D3D / DDS native byte order |
| RGB565 / BGR565 / RGBA4444 | `convert` | `convert` | Low-memory legacy and mobile textures, UI |
| RGBA16 unorm | `convert` | `convert` | 16-bit masters, high precision LDR |
| RGB16F / RGBA16F | `convert` | `convert` | HDR render targets, environment maps |
| RGB9E5 | `convert` | `convert` | Compact HDR (shared exponent), lighting data |
| RGBA32F | `convert` | `convert` | Full precision HDR, interchange / processing |
| R16F / R32F | `convert` | `convert` | Depth, height and single-channel float data (EXR, zfile) |

### Block compressed (desktop: BC)

| Type | Encoder | Decoder | Why / typical use |
|---|---|---|---|
| BC1 (+ sRGB) | port, basisu `encode_bc1` | own | Opaque color at 4 bpp, every desktop GPU |
| BC2 (+ sRGB) | port, BC1 + 4-bit alpha | own | Sharp explicit alpha (legacy) |
| BC3 (+ sRGB) | port, BC1 + BC4 alpha | own | Color with smooth alpha at 8 bpp |
| BC4 (unorm, snorm) | port, basisu `encode_bc4` (snorm: own offset) | own | Single channel: masks, height, roughness |
| BC5 (unorm, snorm) | port, BC4 x 2 (snorm: own offset) | own | Tangent-space normal maps (XY) |
| BC6H unsigned | port, basisu `fast_encode_bc6h` | port, basisu | HDR color: skyboxes, lightmaps |
| BC6H signed | own (single-region modes) | port, basisu | Signed HDR data |
| BC7 (+ sRGB) | port, basisu `bc7f` (Fastest..Default), `bc7e` (High, Best) | port, basisu | High quality color + alpha at 8 bpp |

### Block compressed (mobile: ETC / EAC)

| Type | Encoder | Decoder | Why / typical use |
|---|---|---|---|
| ETC1 | port, basisu `etc1f`; High / Best: basisu `etc1_optimizer` | port, KTX-Software | Opaque color on every OpenGL ES 2 device |
| ETC2 RGB (+ sRGB) | ETC1 + own T / H / planar search at High+ | port, KTX-Software | Opaque color, OpenGL ES 3 / Vulkan mobile baseline |
| ETC2 RGB A1 (+ sRGB) | own punch-through optimizer + T / H / planar | port, KTX-Software | Cut-out alpha (foliage, fences) |
| ETC2 RGBA (+ sRGB) | ETC2 RGB + port, basisu EAC | port, KTX-Software | Color with smooth alpha on mobile |
| EAC R11 / RG11 (unorm, snorm) | port, basisu `pack_eac` (snorm: own offset) | port, KTX-Software | Single / dual channel data, normal maps on mobile |

### Block compressed (ASTC and legacy)

| Type | Encoder | Decoder | Why / typical use |
|---|---|---|---|
| ASTC 4x4..12x12 LDR / sRGB / HDR | port, ARM astcenc 6.0 (presets fastest..exhaustive, LZ RDO, alpha scale; 3D blocks via `astcenc` API) | port, basisu ASTC helpers | Flexible bitrate (8 to 0.89 bpp), modern mobile and Apple GPUs |
| PVRTC1 4bpp RGB / RGBA | port, basisu `encode_pvrtc1` | port, basisu | Older iOS (PowerVR) devices |
| PVRTC2 4bpp RGB / RGBA | own (hard-transition blocks) | port, basisu (hard-transition blocks) | PowerVR GPUs with PVRTC2 support |
| ATC RGB / RGBA | own | port, basisu | Older Adreno (Qualcomm) devices |
| FXT1 RGB | own (CC_MIXED blocks) | port, basisu (CC_MIXED blocks) | 3dfx / Intel legacy GPUs |

### Basis Universal (supercompressed, transcoded at load time)

`imgio::basis::encode_basis` / `imgio::ktx2::encode_basis` encode; `transcode` decodes to the
GPU format the device supports (BC1-7, ETC, ASTC, PVRTC, ATC, FXT1, RGBA ...).

| Type | Encoder | Decoder | Why / typical use |
|---|---|---|---|
| ETC1S (BasisLZ) | port, basisu v2.50 | port, basisu transcoder | Smallest files, one texture for every GPU (web, mobile) |
| UASTC LDR 4x4 (+ RDO) | port, basisu v2.50 | port, basisu transcoder | High quality universal LDR, near-BC7 quality |
| UASTC HDR 4x4 | port, basisu v2.50 | port, basisu transcoder | Universal HDR, fast transcode to BC6H / ASTC HDR |
| ASTC HDR 6x6 / UASTC HDR 6x6 intermediate | port, basisu v2.50 | port, basisu transcoder | Smaller universal HDR |
| ASTC LDR 4x4..12x12 | port, basisu v2.50 | port, basisu transcoder | Plain ASTC payload in a Basis / KTX2 file |
| XUASTC LDR 4x4..12x12 | port, basisu v2.50 | port, basisu transcoder | Supercompressed ASTC, small downloads |
| XUBC7 | port, basisu v2.50 | port, basisu transcoder | Supercompressed BC7, small desktop downloads |

### Image files

`imgio::file`: `FileFormat::detect`, `decode` / `decode_animation` / `encode` on byte slices.

| Type | Encoder | Decoder | Why / typical use |
|---|---|---|---|
| BMP | own (uncompressed: 8-bit gray palette, 24-bit, 32-bit V4 with alpha) | own (OS/2 1.x / 2.x, INFO / V2-V5 headers, 1-32 bpp, bit fields, RLE4 / RLE8) | Windows bitmaps, tool interchange |
| TGA | own (gray / 24 / 32-bit, raw or RLE) | own (color-mapped, truecolor, gray; raw / RLE; all origins, 2.0 extension) | Game art pipelines, legacy texture sources |
| Radiance HDR (RGBE) | own (new-style RLE, bit-exact with ffmpeg / imagecodecs) | own (flat / old / new RLE, all orientations) to RGBA32F | HDR environment maps, light probes |
| RenderMan zfile | own (matrices as options, either byte order) | own (both byte orders, matrices kept) to R32F | Depth maps / shadow maps from RenderMan |
| JPEG XL (`.jxl`) | `zune-jpegxl` (feature `jxl`; lossless, 8/16-bit gray, gray+alpha, RGB, RGBA) | `jxl-oxide` (conformant: VarDCT, modular, 8/16-bit, float, alpha, animation, container; sRGB / linear sRGB out) | Modern lossy / lossless photos and HDR, lossless archival of 16-bit sources |
| JPEG 2000 (`.jp2`, `.j2k`) | - (no pure Rust encoder) | `hayro-jpeg2000` (feature `jpeg2000`): JP2 / J2K, 5/3 + 9/7, tiles, layers, reduce, sub-sampling, up to 16-bit out | Digital cinema, medical / geospatial imagery, PDF embedded images |
| OpenVDB (`.vdb`) | - (only C++ exists) | own (feature `vdb`, partial): grid list, float / double grids densified to `R32Float` volumes, none / zip / active-mask compression (no Blosc) | Sparse volumes from DCC tools (smoke, clouds, level sets) as 3D textures |
| PNG / APNG | `png` crate (8/16-bit, palette, APNG) | `png` crate + own APNG compositing | Lossless images, UI, screenshots, animated stickers |
| GIF | `gif` + `color_quant` (exact palette or NeuQuant, dithering, animation) | `gif` crate + own compositing | Legacy animations, simple graphics |
| WebP (`.webp`) | `image-webp` (feature `webp`): lossless only | `image-webp` + own animation compositing: lossy (bit-exact with libwebp), lossless, alpha, animation | Web images, smaller than PNG / JPEG |
| JPEG (`.jpg`) | - (decode only) | `jpeg-decoder` (feature `jpeg`): baseline, progressive, lossless, all subsamplings, CMYK / YCCK, EXIF orientation as metadata | Photos, the most common image format |
| AVIF (`.avif`) | `ravif` (rav1e, no asm; feature `avif`): 8 / 10-bit 4:4:4, alpha | `avif-parse` + `re_rav1d` (dav1d port, no asm) + own YUV -> RGB (bit-exact with libavif): 8-12 bit, all subsamplings, alpha | Modern web photos, HDR stills at small sizes |
| OpenEXR (`.exr`) | `exr` crate (feature `exr`): half / f32, NONE / RLE / ZIP(S) / PIZ / PXR24 / B44(A), scanline / tiled, mips | `exr` crate: multi-part, layers, mip / rip, f16 / f32 / u32, all compressions incl. DWAA / DWAB | VFX / film HDR interchange, renders, light probes |
| TIFF / BigTIFF (`.tif`) | `tiff` crate (feature `tiff`): 8 / 16-bit, half / f32, LZW / Deflate / PackBits, predictors, multi-page | own on `weezl` / `miniz_oxide` / `zune-jpeg`: strips / tiles, planar, 1-64 bit, palette / CMYK / YCbCr, JPEG-in-TIFF | Print, scanning, GIS, 16-bit / float masters |

### Image processing

| Module | What | Typical use |
|---|---|---|
| `imgio::mip` | Mip chains, every resampler filter (basisu resampler port), sRGB-correct, wrap modes | Texture mipmaps (`texture::build` renormalizes normal maps) |
| `imgio::resize` | Resize with exact / contain / cover / width / height fits, crop; premultiplied, sRGB-correct | Thumbnails, power-of-two textures |
| `imgio::filter` | Flip / rotate / transpose, box / Gaussian blur, unsharp mask, exposure / brightness / contrast / gamma / saturation, grayscale, swizzle, premultiply, normal map from height, HDR tone mapping | Asset pipelines, texture prep |
| `imgio::pack` | Any output channel from any channel of any input (luma, average, max, constants, invert, scale / bias), resizing mismatched inputs; ORM and mask map presets; `split` | ORM / mask maps, alpha from a separate mask, channel extraction |
| `imgio::metrics` | MSE, PSNR, max / mean error, SSIM (matches scikit-image) per channel and aggregated; difference image | Encoder quality checks, regression tests |

### Containers

`imgio::texture::build` makes a whole texture in one call (mips, encode, container).

| Type | Encoder | Decoder | Why / typical use |
|---|---|---|---|
| KTX2 | own `ktx2::write` (raw / Zstd), `write_basis` (Basis payloads) | own `Ktx2File` (raw / Zstd / Basis payloads) | Khronos standard for GPU textures (Vulkan, glTF) |
| KTX1 | own `ktx1::write` | own `Ktx1File` | Legacy OpenGL / OpenGL ES texture files |
| DDS | own `dds::write` (legacy and DX10 headers) | own `DdsFile` | DirectX / game engine textures |
| `.astc` | own `astc_file::write` | own `AstcFile` | ARM astcenc output files |
| `.basis` | port, basisu `encode_basis` | port, basisu `BasisFile` | Basis Universal native files |

## License

Apache-2.0. See [NOTICE.md](NOTICE.md) for third-party code and test data.
