# Channel packing (`imgio::pack`)

Spec: build each output channel (R, G, B, A) from any channel of any input image, a constant
or nothing (0 RGB / 1 alpha); `split` returns one single-channel image per channel.

- Sources: `Input { input, channel }` with `R`, `G`, `B`, `A`, `Luma` (Rec. 709 of stored RGB),
  `Average`, `Max`; `Constant(v)`. Up to `MAX_INPUTS` (64) inputs.
- Remap per channel: `bias + scale * (invert ? 1 - v : v)`, optional clamp to 0..1 (integer
  output formats always clamp and round through `convert`).
- Values as stored: sRGB inputs are not linearized (packed maps are data).
- Sizes: `Same` (mismatch -> `InvalidArgument`), `Largest`, `First`, `Fixed(w, h)`; resized
  inputs go through `resize` (Lanczos4, linear, no premultiply). Unused inputs are ignored.
- Presets: `PackOptions::orm()` (glTF occlusion / roughness / metallic), `mask_map()` (Unity
  HDRP: metallic, occlusion, detail, smoothness = 1 - roughness).
- Rows in parallel.

## Audit

- tests/pack.rs: ORM and mask map exact on 8-bit inputs, any channel / constant / luma / invert /
  scale / bias, unclamped float output, size modes, index and compressed-format errors,
  split -> pack round trip exact, 16-bit split precision.
