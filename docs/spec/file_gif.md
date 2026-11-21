# GIF (`imgio::file::gif`)

Backed by the `gif` crate 0.14 (feature `gif`; `std` + `raii_no_panic`, no default features)
and `color_quant` 2.0 (NeuQuant). Pure Rust (`weezl` LZW). Without the feature every function
returns `Unsupported`.

- Decode (`decode` = first composited frame, `decode_animation[_with]`): global / local color
  tables, interlaced images (crate deinterlaces), transparency, `Rgba8Srgb` output. Browser
  compositing: canvas (logical screen) starts transparent black; per frame save the rect for
  disposal 3, draw non-transparent indices (out-of-table index -> opaque black), emit the
  canvas, then dispose: 2 clears the rect to transparent (background color index ignored), 3
  restores, 0 / 1 keep. Frames are clipped to the screen. No color table at all -> `InvalidData`.
- Delays `(centiseconds, 100)`. Loops: no NETSCAPE2.0 block -> 1 play; count 0 -> 0 (forever);
  count N -> N + 1 plays (Chrome / Firefox play once and repeat N times; Pillow reports raw N as
  `info["loop"]` and no key without the block).
- Limits: screen pixels <= max(input_len * 4096 * 8 / 3, 4 Mi) and frame pixels <= input_len *
  4096 * 8 / 3 (an LZW code of >= 3 bits yields <= 4096 pixels); `DecodeOptions::max_bytes`
  (default 1 GiB) bounds all composited frames and the crate's frame buffer (`OutOfMemory`).
- Encode (`encode`, `encode_animation`): `Rgba8*`, `Bgra8*`, `Rgb8*`, `R8Unorm` (gray),
  `Rg8Unorm` (gray + alpha); others `Unsupported`, > 65535 px sides `InvalidArgument`. Per frame
  (parallel, `parallel::try_for_each_item`): alpha < 128 -> transparent index (color 0,0,0, after
  the colors), other alpha dropped; <= 256 opaque colors (255 with transparency) -> exact sorted
  table (lossless); else NeuQuant (`quantizer_speed` 1..=30, default 10, capped at pixels / 64Ki so
  small frames use every pixel) + optional Floyd-Steinberg `dither`; local color table per frame,
  LZW pre-encoded in parallel. Any transparency in an animation -> every frame gets a
  transparent index and disposal 2 (transparent pixels show the cleared canvas, also in Pillow);
  else disposal 1. Delay rounded to centiseconds. `loops` option (else `Animation::loops`, still
  image 1): 0 -> NETSCAPE 0, 1 -> no block, N -> NETSCAPE N - 1.
- Tests (`tests/file_gif.rs`, `testdata/file/gif`, `scripts/gen_file_testdata_pnggif.py`): 7
  files (script writer: global / local tables, interlace, transparency, every disposal, offsets,
  loop none / 0 / 3; Pillow and ffmpeg encoders); every composited frame, delay and loop count
  equals Pillow `ImageSequence` exactly (files avoid Pillow's non-browser paths: first frame
  covers the screen or has transparency, background-disposed frames have a transparent index,
  disposal never falls back to 0 after a non-zero method). Clipping, lossless round trips
  (<= 256 / 255 colors, gray, gray + alpha, RGB, BGRA, animations with transparency, loop
  mapping), NeuQuant PSNR bounds (dither off / on, speeds), 6 encoder outputs decoded exactly by
  Pillow (`encoded_verified.txt`), corrupt-input loops, size limits.

## Audit

- 2026-11-22: decode with browser compositing, loop count mapping, encode (exact palette /
  NeuQuant + dithering, transparency, animation, parallel frames); exact vs Pillow.
