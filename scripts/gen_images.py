#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates the synthetic source images in testdata/images (deterministic).

Usage: python3 scripts/gen_images.py   (needs pillow and numpy)
"""
import os
import struct

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(__file__), "..", "testdata", "images")


def ldr(w, h, seed, alpha):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / max(w - 1, 1), y / max(h - 1, 1)
    checker = (((x // 6) + (y // 6)) % 2) * 0.25
    circle = ((x - w * 0.6) ** 2 + (y - h * 0.4) ** 2 < (min(w, h) * 0.25) ** 2) * 0.5
    noise = rng.random((h, w)) * 0.12
    r = np.clip(u + checker * 0.5 + noise, 0, 1)
    g = np.clip(v * 0.7 + circle + noise, 0, 1)
    b = np.clip((1 - u) * 0.8 + checker, 0, 1)
    chans = [r, g, b]
    if alpha:
        chans.append(np.clip(0.2 + 0.8 * u * (1 - circle), 0, 1))
    return (np.stack(chans, -1) * 255 + 0.5).astype(np.uint8)


def write_hdr(path, rgb):
    """Radiance RGBE (.hdr), flat (uncompressed) scanlines."""
    h, w, _ = rgb.shape
    m = rgb.max(-1)
    e = np.where(m > 1e-32, np.floor(np.log2(np.maximum(m, 1e-32))) + 1, 0)
    scale = np.where(m > 1e-32, 256.0 / np.exp2(e), 0)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    out[..., 3] = np.where(m > 1e-32, e + 128, 0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {h} +X {w}\n".encode())
        f.write(out.tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(ldr(64, 48, 1, True), "RGBA").save(os.path.join(OUT, "rgba_64x48.png"))
    Image.fromarray(ldr(37, 29, 2, False), "RGB").save(os.path.join(OUT, "rgb_37x29.png"))
    # Power-of-two source (PVRTC1 targets).
    Image.fromarray(ldr(32, 32, 5, True), "RGBA").save(os.path.join(OUT, "rgba_32x32.png"))
    # Benchmark source.
    Image.fromarray(ldr(512, 512, 4, True), "RGBA").save(os.path.join(OUT, "bench_512.png"))
    # HDR: LDR pattern scaled to a 0..16 range with a hot spot.
    base = ldr(40, 24, 3, False).astype(np.float32) / 255.0
    y, x = np.mgrid[0:24, 0:40].astype(np.float32)
    hot = 15.0 * np.exp(-((x - 28) ** 2 + (y - 10) ** 2) / 30.0)
    write_hdr(os.path.join(OUT, "hdr_40x24.hdr"), base * 2.0 + hot[..., None])


if __name__ == "__main__":
    main()
