#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline golden generator for imgio's KTX2 reader/writer (dev tool, not part of the crate).

Creates testdata/ktx2/plain/*.ktx2 with the reference `ktx create --raw` (KTX-Software CLI, here
the ktx-rs port: KTX=path/to/ktx) from deterministic raw images, and cases.txt:
"name vkformat width height layers faces levels zstd". Image k of a file (level-major, then
layer, then face) has bytes (i * 7 + k * 13 + 3) & 255.
Usage: KTX=.../ktx python3 tools/ktx2-golden/gen.py
"""
import os
import subprocess
import tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "testdata", "ktx2", "plain")
KTX = os.environ.get("KTX", "ktx")

# (vkformat, name, block w, block h, block/pixel bytes)
FORMATS = [
    (9, "R8_UNORM", 1, 1, 1), (10, "R8_SNORM", 1, 1, 1), (16, "R8G8_UNORM", 1, 1, 2),
    (17, "R8G8_SNORM", 1, 1, 2), (37, "R8G8B8A8_UNORM", 1, 1, 4), (43, "R8G8B8A8_SRGB", 1, 1, 4),
    (70, "R16_UNORM", 1, 1, 2), (71, "R16_SNORM", 1, 1, 2), (77, "R16G16_UNORM", 1, 1, 4),
    (78, "R16G16_SNORM", 1, 1, 4), (90, "R16G16B16_SFLOAT", 1, 1, 6),
    (97, "R16G16B16A16_SFLOAT", 1, 1, 8), (109, "R32G32B32A32_SFLOAT", 1, 1, 16),
    (123, "E5B9G9R9_UFLOAT_PACK32", 1, 1, 4), (2, "R4G4B4A4_UNORM_PACK16", 1, 1, 2),
    (4, "R5G6B5_UNORM_PACK16", 1, 1, 2), (5, "B5G6R5_UNORM_PACK16", 1, 1, 2),
    (133, "BC1_RGBA_UNORM_BLOCK", 4, 4, 8), (134, "BC1_RGBA_SRGB_BLOCK", 4, 4, 8),
    (135, "BC2_UNORM_BLOCK", 4, 4, 16), (136, "BC2_SRGB_BLOCK", 4, 4, 16),
    (137, "BC3_UNORM_BLOCK", 4, 4, 16), (138, "BC3_SRGB_BLOCK", 4, 4, 16),
    (139, "BC4_UNORM_BLOCK", 4, 4, 8), (140, "BC4_SNORM_BLOCK", 4, 4, 8),
    (141, "BC5_UNORM_BLOCK", 4, 4, 16), (142, "BC5_SNORM_BLOCK", 4, 4, 16),
    (143, "BC6H_UFLOAT_BLOCK", 4, 4, 16), (144, "BC6H_SFLOAT_BLOCK", 4, 4, 16),
    (145, "BC7_UNORM_BLOCK", 4, 4, 16), (146, "BC7_SRGB_BLOCK", 4, 4, 16),
    (147, "ETC2_R8G8B8_UNORM_BLOCK", 4, 4, 8), (148, "ETC2_R8G8B8_SRGB_BLOCK", 4, 4, 8),
    (149, "ETC2_R8G8B8A1_UNORM_BLOCK", 4, 4, 8), (150, "ETC2_R8G8B8A1_SRGB_BLOCK", 4, 4, 8),
    (151, "ETC2_R8G8B8A8_UNORM_BLOCK", 4, 4, 16), (152, "ETC2_R8G8B8A8_SRGB_BLOCK", 4, 4, 16),
    (153, "EAC_R11_UNORM_BLOCK", 4, 4, 8), (154, "EAC_R11_SNORM_BLOCK", 4, 4, 8),
    (155, "EAC_R11G11_UNORM_BLOCK", 4, 4, 16), (156, "EAC_R11G11_SNORM_BLOCK", 4, 4, 16),
    (1000054001, "PVRTC1_4BPP_UNORM_BLOCK_IMG", 4, 4, 8),
    (1000054003, "PVRTC2_4BPP_UNORM_BLOCK_IMG", 4, 4, 8),
]
ASTC = ["4x4", "5x4", "5x5", "6x5", "6x6", "8x5", "8x6", "8x8", "10x5", "10x6", "10x8", "10x10",
        "12x10", "12x12"]
for i, b in enumerate(ASTC):
    w, h = map(int, b.split("x"))
    FORMATS.append((157 + 2 * i, f"ASTC_{b}_UNORM_BLOCK", w, h, 16))
    FORMATS.append((158 + 2 * i, f"ASTC_{b}_SRGB_BLOCK", w, h, 16))
    FORMATS.append((1000066000 + i, f"ASTC_{b}_SFLOAT_BLOCK", w, h, 16))
BY_NAME = {f[1]: f for f in FORMATS}


def image_size(fmt, w, h):
    _, _, bw, bh, bb = fmt
    return -(-w // bw) * -(-h // bh) * bb


def create(name, fmt_name, w, h, layers=0, faces=1, levels=1, zstd=0):
    fmt = BY_NAME[fmt_name]
    with tempfile.TemporaryDirectory() as tmp:
        inputs, k = [], 0
        for level in range(levels):
            lw, lh = max(w >> level, 1), max(h >> level, 1)
            for _ in range(max(layers, 1) * faces):
                p = os.path.join(tmp, f"{k}.raw")
                with open(p, "wb") as f:
                    f.write(bytes((i * 7 + k * 13 + 3) & 255 for i in range(image_size(fmt, lw, lh))))
                inputs.append(p)
                k += 1
        args = [KTX, "create", "--raw", "--format", "VK_FORMAT_" + fmt_name, "--width", str(w),
                "--height", str(h), "--levels", str(levels)]
        if layers:
            args += ["--layers", str(layers)]
        if faces == 6:
            args += ["--cubemap"]
        if zstd:
            args += ["--zstd", str(zstd)]
        subprocess.run(args + inputs + [os.path.join(OUT, name + ".ktx2")], check=True)
    return f"{name} {fmt[0]} {w} {h} {layers} {faces} {levels} {zstd}"


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = []
    for f in FORMATS:
        cases.append(create(f"fmt_{f[0]}", f[1], 16, 8))
    cases.append(create("rgba8_mips", "R8G8B8A8_UNORM", 20, 12, levels=5))
    cases.append(create("rgb16f_mips", "R16G16B16_SFLOAT", 9, 5, levels=4))
    cases.append(create("bc7_mips", "BC7_UNORM_BLOCK", 24, 16, levels=5))
    cases.append(create("bc1_array", "BC1_RGBA_UNORM_BLOCK", 16, 8, layers=3, levels=3))
    cases.append(create("rgba16f_cube", "R16G16B16A16_SFLOAT", 8, 8, faces=6, levels=4))
    cases.append(create("astc6x5_cube_array", "ASTC_6x5_SRGB_BLOCK", 12, 12, layers=2, faces=6,
                        levels=2))
    cases.append(create("bc7_mips_zstd3", "BC7_UNORM_BLOCK", 24, 16, levels=5, zstd=3))
    cases.append(create("rgba8_array_zstd19", "R8G8B8A8_UNORM", 20, 12, layers=2, levels=3,
                        zstd=19))
    cases.append(create("eac_cube_zstd1", "EAC_R11G11_UNORM_BLOCK", 8, 8, faces=6, levels=2,
                        zstd=1))
    with open(os.path.join(OUT, "cases.txt"), "w") as f:
        f.write("\n".join(cases) + "\n")


if __name__ == "__main__":
    main()
