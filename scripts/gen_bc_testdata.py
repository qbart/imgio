#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates testdata/bc: random BC blocks (fixed seed) and their decodes by Pillow.

Pillow has its own BCn decoders, so the expected outputs are an independent reference.
Usage: python3 scripts/gen_bc_testdata.py   (needs pillow and numpy)

Each case writes <name>.bc (raw blocks) and <name>.expected (Pillow's decode as raw bytes:
uint8 channels; Pillow clamps BC6H to 8-bit RGB). BC4 snorm is not supported by Pillow. See testdata/bc/cases.txt.
"""
import io
import os
import random
import struct

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "bc")

# name, DXGI format, bytes per block, width, height
CASES = [
    ("bc1_unorm", 71, 8, 64, 64),
    ("bc1_unorm_odd", 71, 8, 30, 18),
    ("bc2_unorm", 74, 16, 64, 64),
    ("bc3_unorm", 77, 16, 64, 64),
    ("bc4_unorm", 80, 8, 64, 64),
    ("bc5_unorm", 83, 16, 64, 64),
    ("bc5_snorm", 84, 16, 64, 64),
    ("bc6h_ufloat", 95, 16, 64, 64),
    ("bc6h_sfloat", 96, 16, 64, 64),
    ("bc7_unorm", 98, 16, 128, 128),
]


def dds(dxgi, width, height, data):
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000
    pf = struct.pack("<II4sIIIII", 32, 0x4, b"DX10", 0, 0, 0, 0, 0)
    hdr = struct.pack("<IIIIIII", 124, flags, height, width, len(data), 0, 0)
    hdr += b"\0" * 44 + pf + struct.pack("<IIIII", 0x1000, 0, 0, 0, 0)
    dx10 = struct.pack("<IIIII", dxgi, 3, 0, 1, 0)
    return b"DDS " + hdr + dx10 + data


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    for i, (name, dxgi, bpb, w, h) in enumerate(CASES):
        rng = random.Random(0x1A6E10 + i)
        blocks = ((w + 3) // 4) * ((h + 3) // 4)
        data = bytes(rng.getrandbits(8) for _ in range(blocks * bpb))
        im = Image.open(io.BytesIO(dds(dxgi, w, h, data)))
        im.load()
        arr = np.asarray(im)
        if arr.dtype != np.uint8:
            arr = arr.astype("<f4")
        with open(os.path.join(OUT, name + ".bc"), "wb") as f:
            f.write(data)
        with open(os.path.join(OUT, name + ".expected"), "wb") as f:
            f.write(arr.tobytes())
        lines.append(f"{name} {w} {h} {im.mode} {arr.dtype} {arr.shape}")
        print(lines[-1])
    with open(os.path.join(OUT, "cases.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
