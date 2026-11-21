#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates the synthetic source images in testdata/images (deterministic).

Usage: python3 scripts/gen_images.py   (needs numpy; pillow for the PNGs, otherwise only the
raw/*.rgba copies are written)
"""
import os
import struct

import numpy as np

try:
    from PIL import Image
except ImportError:  # raw files only
    Image = None

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


def mosaic(w, h, seed, alpha):
    """12x12 tiles of solid colours, 2-colour stripes, gradients, hard edges, uncorrelated
    channels and circles (exercises the XUASTC -> BC7 fast paths: solid, shared endpoints,
    2 subsets, dual plane), some tiles with an alpha ramp."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 4), np.float32)
    img[..., 3] = 1.0
    pal = np.array([[0.9, 0.1, 0.1], [0.1, 0.2, 0.9], [0.95, 0.9, 0.2], [0.2, 0.7, 0.3]], np.float32)
    for ry in range(0, h, 12):
        for rx in range(0, w, 12):
            k = int(rng.integers(0, 8))
            sl = (slice(ry, ry + 12), slice(rx, rx + 12))
            yy, xx = y[sl] - ry, x[sl] - rx
            a, b = pal[rng.integers(0, 4)], pal[rng.integers(0, 4)]
            if k <= 1:  # solid
                img[sl][..., :3] = a
            elif k == 2:  # 2-colour stripes (same endpoints in every block)
                m = ((xx // 2) % 2)[..., None]
                img[sl][..., :3] = a * (1 - m) + b * m
            elif k == 3:  # low contrast gradient
                t = (xx + yy)[..., None] / 22.0
                img[sl][..., :3] = a * (1 - 0.08 * t) + 0.08 * t * b
            elif k == 4:  # hard diagonal edge
                m = (xx + 0.6 * yy > 7)[..., None]
                img[sl][..., :3] = np.where(m, a, b)
            elif k == 5:  # uncorrelated channels
                img[sl][..., 0] = xx / 11.0
                img[sl][..., 1] = rng.random(xx.shape) * 0.6
                img[sl][..., 2] = yy / 11.0
            elif k == 6:  # strong gradient
                t = (xx / 11.0)[..., None]
                img[sl][..., :3] = a * (1 - t) + b * t
            else:  # circle on a solid background
                m = (((xx - 5) ** 2 + (yy - 6) ** 2) < 16)[..., None]
                img[sl][..., :3] = np.where(m, a, b)
            if alpha and rng.random() < 0.3:
                img[sl][..., 3] = 0.3 + 0.7 * yy / 11.0
    out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return out if alpha else out[..., :3]


def cutout(w, h, seed):
    """RGBA with mostly fully transparent alpha over noisy RGB (exercises astcenc alpha scaling,
    `-a`): a soft-edged disc, a hard-edged opaque rectangle, a horizontal alpha ramp and a few
    isolated near-zero alpha specks (1..3 / 255)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / max(w - 1, 1), y / max(h - 1, 1)
    noise = rng.random((h, w, 3)).astype(np.float32)
    rgb = np.stack([u, v, 1 - u], -1) * 0.5 + noise * 0.5
    d = np.sqrt((x - w * 0.3) ** 2 + (y - h * 0.45) ** 2)
    alpha = np.clip((h * 0.3 - d) / 6.0, 0, 1)
    rect = (x >= w * 0.67) & (x < w * 0.875) & (y >= h * 0.125) & (y < h * 0.375)
    alpha = np.where(rect, 1.0, alpha)
    ramp = (y >= h * 0.69) & (y < h * 0.81) & (x >= w * 0.5)
    alpha = np.where(ramp, (x - w * 0.5) / (w * 0.5 - 1), alpha)
    out = (np.clip(np.concatenate([rgb, alpha[..., None]], -1), 0, 1) * 255 + 0.5).astype(np.uint8)
    for (sx, sy, a) in ((w * 10 // 96, h * 58 // 64, 1), (w * 88 // 96, h * 40 // 64, 2),
                        (w * 50 // 96, h * 4 // 64, 3)):
        out[sy, sx, 3] = a
    return out


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


def hdr_image():
    """40x24 RGB float: LDR pattern scaled to 0..2 plus a 15.0 hot spot."""
    base = ldr(40, 24, 3, False).astype(np.float32) / 255.0
    y, x = np.mgrid[0:24, 0:40].astype(np.float32)
    hot = 15.0 * np.exp(-((x - 28) ** 2 + (y - 10) ** 2) / 30.0)
    return base * 2.0 + hot[..., None]


def raw(name, px):
    """Raw RGBA8 copy (alpha 255 for RGB) for tests without a PNG decoder: raw/NAME_WxH.rgba."""
    h, w = px.shape[:2]
    if px.shape[2] == 3:
        px = np.concatenate([px, np.full((h, w, 1), 255, np.uint8)], -1)
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    with open(os.path.join(OUT, "raw", f"{name}_{w}x{h}.rgba"), "wb") as f:
        f.write(px.astype(np.uint8).tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    raw("rgba", ldr(64, 48, 1, True))
    raw("rgb", ldr(37, 29, 2, False))
    raw("mosaic_rgba", mosaic(50, 38, 7, True))
    raw("mosaic_rgb", mosaic(50, 38, 8, False))
    raw("cutout_rgba", cutout(96, 64, 9))
    # HDR source as raw RGBA16F (alpha 1) for the BC6H encoder tests.
    hdr = hdr_image()
    with open(os.path.join(OUT, "raw", "hdr_40x24.rgba16f"), "wb") as f:
        rgba = np.concatenate([hdr, np.ones((24, 40, 1), np.float32)], -1)
        f.write(rgba.astype(np.float16).tobytes())
    if Image is None:
        print("pillow not installed: PNG / HDR files skipped")
        return
    Image.fromarray(ldr(64, 48, 1, True), "RGBA").save(os.path.join(OUT, "rgba_64x48.png"))
    Image.fromarray(ldr(37, 29, 2, False), "RGB").save(os.path.join(OUT, "rgb_37x29.png"))
    # Power-of-two source (PVRTC1 targets).
    Image.fromarray(ldr(32, 32, 5, True), "RGBA").save(os.path.join(OUT, "rgba_32x32.png"))
    # Block-structured mosaics (XUASTC -> BC7 fast paths).
    Image.fromarray(mosaic(50, 38, 7, True), "RGBA").save(os.path.join(OUT, "mosaic_rgba_50x38.png"))
    Image.fromarray(mosaic(50, 38, 8, False), "RGB").save(os.path.join(OUT, "mosaic_rgb_50x38.png"))
    # Transparent regions (astcenc alpha scaling).
    Image.fromarray(cutout(96, 64, 9), "RGBA").save(os.path.join(OUT, "cutout_rgba_96x64.png"))
    # Benchmark source.
    Image.fromarray(ldr(512, 512, 4, True), "RGBA").save(os.path.join(OUT, "bench_512.png"))
    # HDR: LDR pattern scaled to a 0..16 range with a hot spot.
    write_hdr(os.path.join(OUT, "hdr_40x24.hdr"), hdr_image())


if __name__ == "__main__":
    main()
