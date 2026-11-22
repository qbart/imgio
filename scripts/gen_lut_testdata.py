#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates the LUT test data (deterministic).

Usage: VENV/bin/python -I scripts/gen_lut_testdata.py [--verify]

Needs numpy and colour-science 0.4.7 (offline reference, never linked). Writes testdata/lut/:

- `grade.cube` (3D 17, Iridas writer, TITLE), `domain.cube` (3D 9, DOMAIN_MIN / MAX),
  `curve.cube` (1D 1024 over 0..2), `shaper.cube` (Resolve writer: 1D 256 shaper over
  -0.25..4 + 3D 21): written by colour-science with 7 decimals.
- `input.f32`: 64 x 32 RGBA f32 (rows 0-15 in -0.1..1.1, rows 16-31 in -0.5..4.5, the first
  pixels on tetrahedron boundaries and lattice points).
- `<lut>_<tri|tet>.f32` (3D) / `<lut>.f32` (1D): RGB f32 per pixel, from colour-science
  `read_LUT` then `LUT.apply` with `table_interpolation_trilinear` / `_tetrahedral` (inputs
  clamped to the domain first for 1D tables, as the .cube spec says; colour clamps 3D inputs
  itself), in float64.
- `p3_srgb.link.icc`, `gray.link.icc`: device links made by LittleCMS 2.14 `linkicc` (perceptual,
  default precision, creation date cleared) from testdata/color `p3.icc` -> `lcms_srgb.icc` and `gray22.icc` ->
  `gray10.icc`; `<name>_link.f32`: LittleCMS `transicc -c0 -l` (no optimization, float) of the
  first 1024 input pixels clamped to 0..1 (RGB, or red for gray), divided by 255.

`--verify` applies the device link imgio's tests wrote to `enc/` (IMGIO_WRITE_ENCODED=1,
`Lut3d::to_device_link` of `grade.cube`) with `transicc -c0 -l`, checks it against
colour-science tetrahedral interpolation of the 16-bit table (within 2 / 65535: LittleCMS
interpolates in 16-bit fixed point), stores the LittleCMS output as
`grade_link_lcms.f32` and writes `encoded_verified.txt` (name, fnv1a64 of the bytes).
"""
import os
import subprocess
import sys

import numpy as np
import colour
from colour.algebra import table_interpolation_tetrahedral, table_interpolation_trilinear

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "testdata/lut")
W, H = 64, 32


def grade(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.stack(
        [
            np.clip(r, 0, None) ** 0.8 + 0.1 * g * b,
            0.9 * g + 0.15 * np.sin(3 * r) * b,
            (b + 0.05) / (1.05 + 0.3 * r * g),
        ],
        -1,
    )


def lattice(n, lo, hi):
    t = colour.LUT3D.linear_table(n)  # [r, g, b, 3] over 0..1
    return lo + t * (np.array(hi) - np.array(lo))


def inputs():
    rng = np.random.default_rng(20261122)
    a = np.empty((H, W, 4))
    a[:16, :, :3] = rng.uniform(-0.1, 1.1, (16, W, 3))
    a[16:, :, :3] = rng.uniform(-0.5, 4.5, (16, W, 3))
    a[..., 3] = rng.uniform(0, 1, (H, W))
    special = [
        [0.5, 0.5, 0.5], [0.7, 0.7, 0.2], [0.2, 0.7, 0.7], [0.7, 0.2, 0.7], [0.3, 0.3, 0.9],
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.25, 0.5, 0.75], [0.0625, 0.125, 0.9375], [1.0, 0.0, 0.5],
    ]
    a[0, : len(special), :3] = special
    return a.astype(np.float32)


COLOR = os.path.join(ROOT, "testdata/color")
N_LINK = 1024


def link_inputs():
    a = np.fromfile(os.path.join(OUT, "input.f32"), np.float32).reshape(-1, 4)[:N_LINK, :3]
    return np.clip(a.astype(np.float64), 0, 1)


def transicc(link, vals):
    """Applies a device link to rows of 0..1 values (LittleCMS works in 0..255 here)."""
    text = "".join(" ".join(f"{v * 255:.9g}" for v in row) + "\n" for row in vals)
    out = subprocess.run(["transicc", "-n", "-c0", "-l" + link], input=text, capture_output=True,
                         text=True, check=True).stdout
    rows = [l.split() for l in out.splitlines() if l.strip() and l[0] in "0123456789-"]
    return np.array(rows, np.float64) / 255


def fnv(b):
    h = 0xCBF29CE484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def links():
    rgb = link_inputs()
    for name, a, b, gray in [("p3_srgb", "p3.icc", "lcms_srgb.icc", False),
                             ("gray", "gray22.icc", "gray10.icc", True)]:
        path = os.path.join(OUT, name + ".link.icc")
        subprocess.run(["linkicc", "-t0", "-o" + path, os.path.join(COLOR, a), os.path.join(COLOR, b)],
                       check=True, capture_output=True)
        # Deterministic: clear the creation date (no profile ID is stored).
        b = bytearray(open(path, "rb").read())
        assert b[84:100] == bytes(16)
        b[24:36] = bytes(12)
        open(path, "wb").write(b)
        out = transicc(path, rgb[:, :1] if gray else rgb)
        assert out.shape == (N_LINK, 1 if gray else 3), out.shape
        out.astype(np.float32).tofile(os.path.join(OUT, f"{name}_link.f32"))


def verify():
    name = "grade_link.icc"
    data = open(os.path.join(OUT, "enc", name), "rb").read()
    rgb = link_inputs()
    got = transicc(os.path.join(OUT, "enc", name), rgb)
    cube = colour.read_LUT(os.path.join(OUT, "grade.cube"))
    q = colour.LUT3D(np.round(np.clip(cube.table, 0, 1) * 65535) / 65535)
    want = q.apply(rgb, interpolator=table_interpolation_tetrahedral)
    err = np.abs(got - want).max()
    # LittleCMS interpolates 16-bit CLUTs in 16-bit fixed point.
    if err > 2 / 65535:
        sys.exit(f"{name}: LittleCMS differs from the table by {err}")
    got.astype(np.float32).tofile(os.path.join(OUT, "grade_link_lcms.f32"))
    with open(os.path.join(OUT, "encoded_verified.txt"), "w") as fh:
        fh.write("# imgio device links applied by LittleCMS transicc as the table says "
                 "(scripts/gen_lut_testdata.py --verify)\n")
        fh.write(f"{name} {fnv(data):016x}\n")
    print(name, "verified, max error", err)


def main():
    if "--verify" in sys.argv:
        verify()
        return
    os.makedirs(OUT, exist_ok=True)
    img = inputs()
    img.tofile(os.path.join(OUT, "input.f32"))
    rgb = img[..., :3].astype(np.float64)

    luts = {}
    luts["grade"] = colour.LUT3D(grade(lattice(17, [0, 0, 0], [1, 1, 1])), "Grade test")
    dlo, dhi = [-0.1, -0.2, 0.0], [1.5, 2.0, 4.0]
    luts["domain"] = colour.LUT3D(
        grade(lattice(9, dlo, dhi) / np.array(dhi)), "Domain", np.array([dlo, dhi])
    )
    x = np.linspace(0, 2, 1024)
    luts["curve"] = colour.LUT3x1D(
        np.stack([x ** 0.45, np.sqrt(x) * 0.8, x / (1 + x)], -1), "Curve", np.array([[0] * 3, [2] * 3])
    )
    s = np.linspace(-0.25, 4, 256)
    shaper = colour.LUT3x1D(np.tile(np.log2(s + 0.5)[:, None] / 3.2 + 0.32, (1, 3)), "Shaper",
                            np.array([[-0.25] * 3, [4] * 3]))
    cube = colour.LUT3D(grade(lattice(21, [0, 0, 0], [1, 1, 1])), "Cube")
    luts["shaper"] = colour.LUTSequence(shaper, cube)

    for name, lut in luts.items():
        path = os.path.join(OUT, name + ".cube")
        if name == "shaper":
            colour.io.write_LUT_ResolveCube(lut, path, decimals=7)
        else:
            colour.io.write_LUT_IridasCube(lut, path, decimals=7)
        back = colour.read_LUT(path)
        if isinstance(back, colour.LUT3x1D):
            lo, hi = back.domain
            out = back.apply(np.clip(rgb, lo, hi))
            out.astype(np.float32).tofile(os.path.join(OUT, name + ".f32"))
            continue
        for tag, interp in [("tri", table_interpolation_trilinear), ("tet", table_interpolation_tetrahedral)]:
            if isinstance(back, colour.LUTSequence):
                lo, hi = back[0].domain
                v = back[0].apply(np.clip(rgb, lo, hi))
                out = back[1].apply(v, interpolator=interp)
            else:
                out = back.apply(rgb, interpolator=interp)
            out.astype(np.float32).tofile(os.path.join(OUT, f"{name}_{tag}.f32"))
    links()


if __name__ == "__main__":
    main()
