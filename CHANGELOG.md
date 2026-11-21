# Changelog

## 0.1.0 (unreleased)

First functional release.

- GPU texture encoders for BC1-BC7, ETC1 / ETC2 / EAC, ASTC (astcenc 6.0 port, LZ RDO), PVRTC1/2,
  ATC and FXT1; decoders for all of them (`imgio::encode`, `imgio::decode`).
- Basis Universal v2.50 encoder and transcoder: ETC1S, UASTC LDR / HDR, ASTC HDR 6x6, ASTC LDR,
  XUASTC LDR, XUBC7; byte identical with the C++ reference.
- Containers: KTX2 (raw / Zstd / Basis payloads), KTX1, DDS (legacy and DX10), `.astc`, `.basis`;
  2D, arrays, cubemaps and 3D textures.
- Uncompressed format conversion, sRGB-correct mip generation (`imgio::mip`), one-call texture
  builder (`imgio::texture::build`).
- Optional Vulkan / OpenGL upload (`vulkan`, `opengl` features).
- Image file API (`imgio::file`): format detection and the shared decode / encode interface.
