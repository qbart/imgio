#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Encodes testdata/astcenc/cli/*.astc with the C++ astc-encoder 6.0 CLI (offline dev tool):
the byte-identity references of tests/astcenc_cli.rs.

The sources are the raw copies in testdata/images/raw, handed to the CLI losslessly: RGBA8 as an
uncompressed top-down TGA, RGBA16F as a DX10 DDS (the CLI reads it as FP16 data, the same values
imgio converts to RGBA32F). Every case uses the CLI defaults (`-j 1`, RGBA swizzle, no alpha
weighting, no -decode_unorm8), matching `imgio::encode`'s astcenc setup.

Usage: python3 scripts/gen_astcenc_testdata.py [path/to/astcenc-avx2]
(build the CLI from ref/astcenc: cmake -DASTCENC_ISA_AVX2=ON ..; make)
"""
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "testdata", "images", "raw")
OUT = os.path.join(ROOT, "testdata", "astcenc", "cli")

# (output name, raw source, width, height, block, CLI mode, CLI preset); keep in sync with
# tests/astcenc_cli.rs
CASES = [
    ("rgba_64x48_4x4_l_medium", "rgba_64x48.rgba", 64, 48, "4x4", "-cl", "-medium"),
    ("rgba_64x48_6x6_s_fast", "rgba_64x48.rgba", 64, 48, "6x6", "-cs", "-fast"),
    ("rgba_64x48_10x5_l_fastest", "rgba_64x48.rgba", 64, 48, "10x5", "-cl", "-fastest"),
    ("rgb_37x29_5x4_l_thorough", "rgb_37x29.rgba", 37, 29, "5x4", "-cl", "-thorough"),
    ("rgb_37x29_8x5_s_exhaustive", "rgb_37x29.rgba", 37, 29, "8x5", "-cs", "-exhaustive"),
    ("mosaic_rgba_50x38_8x8_l_exhaustive", "mosaic_rgba_50x38.rgba", 50, 38, "8x8", "-cl",
     "-exhaustive"),
    ("mosaic_rgb_50x38_12x12_s_medium", "mosaic_rgb_50x38.rgba", 50, 38, "12x12", "-cs",
     "-medium"),
    ("mosaic_rgba_50x38_4x4_s_thorough", "mosaic_rgba_50x38.rgba", 50, 38, "4x4", "-cs",
     "-thorough"),
    ("hdr_40x24_6x6_H_medium", "hdr_40x24.rgba16f", 40, 24, "6x6", "-cH", "-medium"),
    ("hdr_40x24_4x4_H_thorough", "hdr_40x24.rgba16f", 40, 24, "4x4", "-cH", "-thorough"),
    ("hdr_40x24_12x12_H_fastest", "hdr_40x24.rgba16f", 40, 24, "12x12", "-cH", "-fastest"),
]


def tga(path, w, h, rgba):
    """Uncompressed 32-bit BGRA TGA, top-left origin."""
    bgra = bytearray(len(rgba))
    for i in range(0, len(rgba), 4):
        bgra[i : i + 4] = bytes((rgba[i + 2], rgba[i + 1], rgba[i], rgba[i + 3]))
    hdr = struct.pack("<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, w, h, 32, 0x28)
    with open(path, "wb") as f:
        f.write(hdr + bytes(bgra))


def dds_rgba16f(path, w, h, data):
    """DDS with a DX10 header, DXGI_FORMAT_R16G16B16A16_FLOAT."""
    pf = struct.pack("<II4sIIIII", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
    hdr = struct.pack("<IIIIIII", 124, 0x1 | 0x2 | 0x4 | 0x1000 | 0x8, h, w, w * 8, 0, 1)
    hdr += b"\0" * 44 + pf + struct.pack("<IIIII", 0x1000, 0, 0, 0, 0)
    dx10 = struct.pack("<IIIII", 10, 3, 0, 1, 0)
    with open(path, "wb") as f:
        f.write(b"DDS " + hdr + dx10 + data)


def main():
    cli = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "ref", "astcenc", "build", "Source", "astcenc-avx2")
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, src, w, h, block, mode, preset in CASES:
            with open(os.path.join(RAW, src), "rb") as f:
                data = f.read()
            if src.endswith(".rgba16f"):
                inp = os.path.join(tmp, name + ".dds")
                dds_rgba16f(inp, w, h, data)
            else:
                inp = os.path.join(tmp, name + ".tga")
                tga(inp, w, h, data)
            out = os.path.join(OUT, name + ".astc")
            subprocess.run([cli, mode, inp, out, block, preset, "-j", "1", "-silent"],
                           check=True)
            print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
