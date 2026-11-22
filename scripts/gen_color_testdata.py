#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates the colour management test data (deterministic).

Usage: VENV/bin/python -I scripts/gen_color_testdata.py

Needs numpy and Pillow with LittleCMS (offline reference, never linked). Writes
testdata/color/:

- `*.icc`: ICC profiles built here from their published definitions: v4 matrix / TRC
  profiles (Display P3, Adobe RGB, ProPhoto, Rec. 2020, gray gamma 2.2, gray linear), a v2
  LUT profile (`lut16` A2B0 / B2A0 with XYZ connection space, 17^3 / 33^3 grids of a
  non-linear device model), and LittleCMS's built-in sRGB (`lcms_srgb.icc`).
- `src_rgb.raw` (RGB8, 64x48 from testdata/images/raw) and `src_gray.raw` (gray8, its G).
- `<from>__<to>__<intent>.raw`: LittleCMS 8-bit transforms of the source (`NOOPTIMIZE`: the
  full pipeline, not LittleCMS's precalculated 8-bit device link).
- `cases.txt`: `from to intent width height channels_in channels_out`.
- `files/`: images carrying colour data: Pillow PNG / JPEG / WebP / TIFF / AVIF with an
  embedded P3 profile (`*_p3.*`; the JPEG also with the 245 KB LUT profile, split over APP2
  segments), PNG with hand-inserted cICP (BT.2020 PQ) and gAMA + cHRM chunks, OpenEXR with a
  chromaticities attribute (OpenEXR Python; attribute bytes patched), Radiance HDR with a `PRIMARIES=` line;
  `png_p3_to_srgb.raw` / `png_gama_chrm_to_srgb.raw`: LittleCMS transforms of the PNG pixels
  to sRGB (perceptual, `NOOPTIMIZE`; the gAMA / cHRM profile is built here from the chunk
  values).
"""
import io
import os
import struct

import numpy as np
from PIL import Image, ImageCms

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "testdata/images/raw")
OUT = os.path.join(ROOT, "testdata/color")

D50 = np.array([0.9642, 1.0, 0.8249])
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def s15(v):
    return struct.pack(">i", int(round(v * 65536)))


def xy_to_xyz(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz(prim, white):
    m = np.stack([xy_to_xyz(*p) for p in prim], 1)
    w = xy_to_xyz(*white)
    s = np.linalg.solve(m, w)
    return m * s


def adapt(white):
    """Bradford matrix from `white` (XYZ) to D50."""
    src = BRADFORD @ white
    dst = BRADFORD @ D50
    return np.linalg.inv(BRADFORD) @ np.diag(dst / src) @ BRADFORD


def tag_xyz(v):
    return b"XYZ \0\0\0\0" + b"".join(s15(c) for c in v)


def tag_para(params):
    kind = {1: 0, 3: 1, 4: 2, 5: 3, 7: 4}[len(params)]
    return b"para\0\0\0\0" + struct.pack(">HH", kind, 0) + b"".join(s15(p) for p in params)


def tag_mluc(text):
    t = text.encode("utf-16-be")
    return b"mluc\0\0\0\0" + struct.pack(">II", 1, 12) + b"enUS" + struct.pack(">II", len(t), 28) + t


def tag_sf32(m):
    return b"sf32\0\0\0\0" + b"".join(s15(v) for v in np.asarray(m).flatten())


def profile(tags, cls=b"mntr", space=b"RGB ", version=0x04400000):
    """ICC bytes from (signature, data) tags (4-byte aligned)."""
    n = len(tags)
    off = 128 + 4 + 12 * n
    table = struct.pack(">I", n)
    body = b""
    for sig, data in tags:
        table += sig + struct.pack(">II", off + len(body), len(data))
        body += data + b"\0" * ((-len(data)) % 4)
    size = off + len(body)
    hdr = struct.pack(">I", size) + b"\0\0\0\0" + struct.pack(">I", version) + cls + space + b"XYZ "
    hdr += b"\0" * 12 + b"acsp" + b"\0" * 24 + struct.pack(">I", 0)
    hdr += b"".join(s15(c) for c in D50) + b"\0" * 4 + b"\0" * 16 + b"\0" * 28
    assert len(hdr) == 128, len(hdr)
    return hdr + table + body


def matrix_profile(name, prim, white, trc):
    w = xy_to_xyz(*white)
    a = adapt(w)
    m = a @ rgb_to_xyz(prim, white)
    tags = [
        (b"desc", tag_mluc(name)),
        (b"cprt", tag_mluc("Public domain test profile (imgio)")),
        (b"wtpt", tag_xyz(D50)),
        (b"chad", tag_sf32(a)),
        (b"rXYZ", tag_xyz(m[:, 0])),
        (b"gXYZ", tag_xyz(m[:, 1])),
        (b"bXYZ", tag_xyz(m[:, 2])),
    ]
    t = tag_para(trc)
    tags += [(b"rTRC", t), (b"gTRC", t), (b"bTRC", t)]
    return profile(tags)


def gray_profile(name, gamma):
    tags = [
        (b"desc", tag_mluc(name)),
        (b"cprt", tag_mluc("Public domain test profile (imgio)")),
        (b"wtpt", tag_xyz(D50)),
        (b"kTRC", tag_para([gamma])),
    ]
    return profile(tags, space=b"GRAY")


# Device model of the LUT profile: Adobe RGB primaries, per-channel curves, a mild cross talk.
LUT_M = adapt(xy_to_xyz(0.3127, 0.3290)) @ rgb_to_xyz(
    [(0.64, 0.33), (0.21, 0.71), (0.15, 0.06)], (0.3127, 0.3290))


def lut_fwd(rgb):
    lin = np.stack([rgb[..., 0] ** 2.2, rgb[..., 1] ** 2.0, rgb[..., 2] ** 2.4], -1)
    mix = lin + 0.04 * (lin[..., [1, 2, 0]] - lin)
    return mix @ LUT_M.T


def lut_inv(xyz):
    mix = xyz @ np.linalg.inv(LUT_M).T
    # Invert the cross talk (linear): mix = 0.96 lin + 0.04 roll(lin).
    c = 0.96 * np.eye(3) + 0.04 * np.eye(3)[[1, 2, 0]]
    lin = np.clip(mix @ np.linalg.inv(c).T, 0, 1)
    return np.stack([lin[..., 0] ** (1 / 2.2), lin[..., 1] ** (1 / 2.0), lin[..., 2] ** (1 / 2.4)], -1)


def lut16(fn, grid, in_max, out_max, shaper=None):
    """lut16Type with an identity matrix, input tables (identity, or `shaper` sampled at 4096
    points: table input x in 0..1 -> grid coordinate), a `grid`^3 CLUT and identity output
    tables. The CLUT is sampled where the shaped coordinate hits the grid."""
    g = np.linspace(0, 1, grid)
    if shaper is None:
        n_in = 2
        tin = struct.pack(">HH", 0, 65535) * 3
        x = g
    else:
        fwd, inv = shaper
        n_in = 4096
        t = np.clip(np.rint(fwd(np.linspace(0, 1, n_in)) * 65535), 0, 65535).astype(">u2").tobytes()
        tin = t * 3
        x = inv(g)
    r, gg, b = np.meshgrid(x, x, x, indexing="ij")
    vals = fn(np.stack([r, gg, b], -1) * in_max) / out_max
    clut = np.clip(np.rint(vals * 65535), 0, 65535).astype(">u2").tobytes()
    eye = b"".join(s15(v) for v in [1, 0, 0, 0, 1, 0, 0, 0, 1])
    t = b"mft2\0\0\0\0" + struct.pack(">BBBB", 3, 3, grid, 0) + eye + struct.pack(">HH", n_in, 2)
    return t + tin + clut + struct.pack(">HH", 0, 65535) * 3


def lut_profile():
    xyz_max = 65535 / 32768  # u1Fixed15 XYZ encoding
    tags = [
        (b"desc", b"desc\0\0\0\0" + struct.pack(">I", 12) + b"LUT device\0\0" + b"\0" * 8 + b"\0" * 3 + b"\0" * 67),
        (b"cprt", b"text\0\0\0\0Public domain test profile (imgio)\0"),
        (b"wtpt", tag_xyz(D50)),
        (b"A2B0", lut16(lut_fwd, 17, 1.0, xyz_max)),
        (b"B2A0", lut16(lut_inv, 33, xyz_max, 1.0, (np.cbrt, lambda t: t ** 3))),
    ]
    return profile(tags, cls=b"scnr", version=0x02100000)


SRGB_TRC = [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045]
REC709_TRC = [1 / 0.45, 1 / 1.099, 0.099 / 1.099, 1 / 4.5, 0.081]


def main():
    os.makedirs(OUT, exist_ok=True)
    profiles = {
        "p3": matrix_profile("Display P3", [(0.68, 0.32), (0.265, 0.69), (0.15, 0.06)], (0.3127, 0.3290), SRGB_TRC),
        "adobe": matrix_profile("Adobe RGB", [(0.64, 0.33), (0.21, 0.71), (0.15, 0.06)], (0.3127, 0.3290), [563 / 256]),
        "prophoto": matrix_profile("ProPhoto", [(0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)], (0.3457, 0.3585), [1.8]),
        "rec2020": matrix_profile("Rec. 2020", [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)], (0.3127, 0.3290), REC709_TRC),
        "gray22": gray_profile("Gray 2.2", 2.2),
        "gray10": gray_profile("Gray linear", 1.0),
        "lut": lut_profile(),
        "lcms_srgb": ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes(),
    }
    for k, v in profiles.items():
        with open(os.path.join(OUT, k + ".icc"), "wb") as f:
            f.write(v)
    src = np.fromfile(os.path.join(RAW, "rgba_64x48.rgba"), np.uint8).reshape(48, 64, 4)[:, :, :3].copy()
    # Add saturated ramps (gamut edges) in the first rows.
    for x in range(64):
        v = x * 255 // 63
        src[0, x] = (v, 0, 0)
        src[1, x] = (0, v, 0)
        src[2, x] = (0, 0, v)
        src[3, x] = (v, v, v)
        src[4, x] = (255, v, 255 - v)
    gray = src[:, :, 1].copy()
    src.tofile(os.path.join(OUT, "src_rgb.raw"))
    gray.tofile(os.path.join(OUT, "src_gray.raw"))
    rgb_img = Image.fromarray(src, "RGB")
    gray_img = Image.fromarray(gray, "L")
    prof = {k: ImageCms.ImageCmsProfile(io.BytesIO(v)) for k, v in profiles.items()}
    cases = [
        ("lcms_srgb", "p3", 0), ("p3", "lcms_srgb", 0), ("lcms_srgb", "adobe", 1),
        ("adobe", "lcms_srgb", 1), ("prophoto", "lcms_srgb", 0), ("lcms_srgb", "prophoto", 1),
        ("rec2020", "p3", 1), ("lcms_srgb", "rec2020", 0), ("adobe", "prophoto", 3),
        ("lut", "lcms_srgb", 0), ("lcms_srgb", "lut", 0), ("lut", "adobe", 1),
        ("gray22", "gray10", 0), ("gray10", "gray22", 1),
    ]
    lines = ["# from to intent width height channels_in channels_out"]
    for a, b, intent in cases:
        gin = a.startswith("gray")
        gout = b.startswith("gray")
        img = gray_img if gin else rgb_img
        t = ImageCms.buildTransform(prof[a], prof[b], "L" if gin else "RGB", "L" if gout else "RGB",
                                    renderingIntent=intent, flags=ImageCms.Flags.NOOPTIMIZE)
        out = np.asarray(ImageCms.applyTransform(img, t))
        out.tofile(os.path.join(OUT, f"{a}__{b}__{intent}.raw"))
        lines.append(f"{a} {b} {intent} 64 48 {1 if gin else 3} {1 if gout else 3}")
    with open(os.path.join(OUT, "cases.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def png_chunk(t, body):
    import zlib
    return struct.pack(">I", len(body)) + t + body + struct.pack(">I", zlib.crc32(t + body) & 0xFFFFFFFF)


def insert_after_ihdr(png, chunks):
    return png[:33] + b"".join(chunks) + png[33:]


def files(profiles):
    import OpenEXR
    import Imath
    out = os.path.join(OUT, "files")
    os.makedirs(out, exist_ok=True)
    src = np.fromfile(os.path.join(RAW, "rgb_37x29.rgba"), np.uint8).reshape(29, 37, 4)[:, :, :3].copy()
    img = Image.fromarray(src, "RGB")
    p3 = profiles["p3"]
    for ext, kw in [("png", {}), ("jpg", {"quality": 95}), ("webp", {"lossless": True}),
                    ("tif", {}), ("avif", {"quality": 90})]:
        img.save(os.path.join(out, f"p3.{ext}"), icc_profile=p3, **kw)
    img.save(os.path.join(out, "lut.jpg"), icc_profile=profiles["lut"], quality=95)
    b = io.BytesIO()
    img.save(b, "PNG")
    plain = b.getvalue()
    with open(os.path.join(out, "cicp.png"), "wb") as f:
        f.write(insert_after_ihdr(plain, [png_chunk(b"cICP", bytes([9, 16, 0, 1]))]))
    # gAMA 1/1.8 with Adobe-like primaries and a D50 white.
    chrm = [34570, 35850, 64000, 33000, 21000, 71000, 15000, 6000]
    with open(os.path.join(out, "gama_chrm.png"), "wb") as f:
        f.write(insert_after_ihdr(plain, [png_chunk(b"gAMA", struct.pack(">I", 55556)),
                                          png_chunk(b"cHRM", struct.pack(">8I", *chrm))]))
    # LittleCMS references.
    srgb = ImageCms.ImageCmsProfile(io.BytesIO(profiles["lcms_srgb"]))
    def lcms(pbytes, name):
        t = ImageCms.buildTransform(ImageCms.ImageCmsProfile(io.BytesIO(pbytes)), srgb, "RGB", "RGB",
                                    renderingIntent=0, flags=ImageCms.Flags.NOOPTIMIZE)
        np.asarray(ImageCms.applyTransform(img, t)).tofile(os.path.join(out, name))
    lcms(p3, "png_p3_to_srgb.raw")
    v = [c / 100000 for c in chrm]
    gprof = matrix_profile("gAMA cHRM", [(v[2], v[3]), (v[4], v[5]), (v[6], v[7])], (v[0], v[1]),
                           [100000 / 55556])
    lcms(gprof, "png_gama_chrm_to_srgb.raw")
    # OpenEXR with chromaticities (P3-D65), 8x4 half RGB.
    h, w = 4, 8
    hdr = OpenEXR.Header(w, h)
    hdr["chromaticities"] = Imath.Chromaticities(Imath.V2f(0.68, 0.32), Imath.V2f(0.265, 0.69),
                                                 Imath.V2f(0.15, 0.06), Imath.V2f(0.3127, 0.329))
    hdr["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)) for c in "RGB"}
    px = (np.arange(w * h, dtype=np.float16) / 16).astype(np.float16).tobytes()
    f = OpenEXR.OutputFile(os.path.join(out, "p3.exr"), hdr)
    f.writePixels({"R": px, "G": px, "B": px})
    f.close()
    # This OpenEXR Python version writes the chromaticities attribute as zeros: patch it.
    path = os.path.join(out, "p3.exr")
    d = bytearray(open(path, "rb").read())
    key = b"chromaticities\0chromaticities\0" + struct.pack("<i", 32)
    i = d.find(key) + len(key)
    d[i:i + 32] = struct.pack("<8f", 0.68, 0.32, 0.265, 0.69, 0.15, 0.06, 0.3127, 0.329)
    open(path, "wb").write(bytes(d))
    rgbe = bytes([128, 64, 32, 129]) * 6
    with open(os.path.join(out, "primaries.hdr"), "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nPRIMARIES= 0.708 0.292 0.170 0.797 0.131 0.046 0.3127 0.3290\n\n-Y 2 +X 3\n" + rgbe)


if __name__ == "__main__":
    main()
    files({k: open(os.path.join(OUT, k + ".icc"), "rb").read() for k in ["p3", "lut", "lcms_srgb"]})
