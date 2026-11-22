#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates the quality metrics test data (deterministic).

Usage: VENV/bin/python -I scripts/gen_metrics_testdata.py

Needs numpy, Pillow and scikit-image (offline reference tools, never linked). Sources are
imgio's own images (testdata/images/raw). Writes testdata/metrics/:

- `<case>.a` / `<case>.b`: reference and distorted pixels (imgio layout: R8 / RGB8 / RGBA8,
  R16 little endian or RGBA f16), distortions: seeded noise, box blur, JPEG (Pillow q 40),
  quantization, a shift.
- `expected.txt`: `case format width height peak` then per channel `mse max_error mean_error
  ssim`, computed in float64 by numpy and scikit-image `structural_similarity(data_range=peak,
  gaussian_weights=True, sigma=1.5, use_sample_covariance=False)` on the values imgio's
  `convert` produces (unorm / 255 or / 65535 rounded to f32, f16 widened).
"""
import io
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "testdata/images/raw")
OUT = os.path.join(ROOT, "testdata/metrics")


def load8(name, w, h, ch):
    a = np.fromfile(os.path.join(RAW, name), np.uint8).reshape(h, w, 4)
    return a[:, :, :ch].copy()


def box(a, r=1):
    f = a.astype(np.float64)
    p = np.pad(f, ((r, r), (r, r), (0, 0)), mode="edge")
    s = sum(p[dy:dy + a.shape[0], dx:dx + a.shape[1]] for dy in range(2 * r + 1) for dx in range(2 * r + 1))
    return s / (2 * r + 1) ** 2


def noise(a, sigma, seed, maxv):
    rng = np.random.default_rng(seed)
    return a.astype(np.float64) + rng.normal(0, sigma, a.shape)


def to8(f):
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


def jpeg(a, q):
    mode = {1: "L", 3: "RGB"}[a.shape[2]]
    im = Image.fromarray(a[:, :, 0] if a.shape[2] == 1 else a, mode)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    d = np.asarray(Image.open(io.BytesIO(b.getvalue())))
    return d.reshape(a.shape)


def values(a, fmt):
    if fmt in ("R8Unorm", "Rgb8Unorm", "Rgba8Unorm"):
        return (a.astype(np.float32) / np.float32(255)).astype(np.float64)
    if fmt == "R16Unorm":
        return (a.astype(np.float32) / np.float32(65535)).astype(np.float64)
    if fmt == "Rgba16Float":
        return a.astype(np.float64)
    raise ValueError(fmt)


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = []
    rgb = load8("rgb_37x29.rgba", 37, 29, 3)
    rgba = load8("rgba_64x48.rgba", 64, 48, 4)
    mosaic = load8("mosaic_rgb_50x38.rgba", 50, 38, 3)
    gray = mosaic[:, :, :1].copy()
    cases.append(("rgb_noise", "Rgb8Unorm", rgb, to8(noise(rgb, 6, 1, 255)), 1.0))
    cases.append(("rgb_jpeg", "Rgb8Unorm", mosaic, jpeg(mosaic, 40), 1.0))
    cases.append(("rgba_blur", "Rgba8Unorm", rgba, to8(box(rgba)), 1.0))
    cases.append(("rgba_same", "Rgba8Unorm", rgba, rgba.copy(), 1.0))
    cases.append(("gray_quant", "R8Unorm", gray, (gray // 16) * 16, 1.0))
    cases.append(("gray_shift", "R8Unorm", gray, np.roll(gray, 1, axis=1), 1.0))
    g16 = (gray.astype(np.uint16) * 257) ^ np.uint16(0x0013)
    g16b = np.clip(noise(g16, 300, 2, 65535), 0, 65535).astype(np.uint16)
    cases.append(("gray16_noise", "R16Unorm", g16, g16b, 1.0))
    hdr = np.fromfile(os.path.join(RAW, "hdr_40x24.rgba16f"), np.float16).reshape(24, 40, 4)
    hdrb = (hdr.astype(np.float32) * np.float32(1.05) + np.float32(0.01)).astype(np.float16)
    peak = float(np.max(hdr[:, :, :3].astype(np.float64)))
    cases.append(("hdr_scale", "Rgba16Float", hdr, hdrb, peak))

    lines = ["# case format width height peak | per channel: mse max_error mean_error ssim"]
    for name, fmt, a, b, peak in cases:
        h, w, ch = a.shape
        a.tofile(os.path.join(OUT, name + ".a"))
        b.astype(a.dtype).tofile(os.path.join(OUT, name + ".b"))
        x, y = values(a, fmt), values(b.astype(a.dtype), fmt)
        f = [name, fmt, str(w), str(h), repr(peak)]
        for c in range(ch):
            d = x[:, :, c] - y[:, :, c]
            mse = float(np.mean(d * d))
            s = structural_similarity(
                x[:, :, c], y[:, :, c], data_range=peak, gaussian_weights=True, sigma=1.5,
                use_sample_covariance=False)
            f += [repr(mse), repr(float(np.max(np.abs(d)))), repr(float(np.mean(np.abs(d)))), repr(float(s))]
        lines.append(" ".join(f))
    with open(os.path.join(OUT, "expected.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
