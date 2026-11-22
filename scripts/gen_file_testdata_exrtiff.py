#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates testdata/file/exr and testdata/file/tiff (deterministic synthetic images).

Offline reference tools only (never linked): OpenEXR 3.x Python bindings, OpenImageIO
(mip levels), imagecodecs / tifffile (libtiff) and Pillow. Usage:

    python scripts/gen_file_testdata_exrtiff.py              # write files + expected.txt
    python scripts/gen_file_testdata_exrtiff.py verify DIR   # decode imgio-encoded files in
                                                             # DIR with the reference decoders

`expected.txt` lines: `<file> <case args...> <format> <width> <height> <fnv1a64 of the
expected decoded bytes>` (format = imgio `Format` name). Expected bytes are computed from the
reference decoder's output with imgio's documented output policy.

`verify DIR` checks the files written by `IMGIO_FILE_DUMP=DIR cargo test --features exr,tiff
--test file_exr --test file_tiff` (encoder output) against the reference decoders.
"""
import ctypes
import os
import struct
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXR_OUT = os.path.join(ROOT, "testdata", "file", "exr")
TIFF_OUT = os.path.join(ROOT, "testdata", "file", "tiff")


def fnv(b):
    h = 0xCBF29CE484222325
    for x in bytes(b):
        h ^= x
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def fnv_np(a):
    """FNV-1a over a numpy array's bytes (vectorized enough for small test images)."""
    return fnv(np.ascontiguousarray(a).tobytes())


# ---------------------------------------------------------------- source images


def hdr_rgba(w, h, seed):
    """Float RGBA: wide dynamic range (2^-12 .. 2^10), a few negatives, zeros, alpha 0..1."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / max(w - 1, 1), y / max(h - 1, 1)
    e = (u * 22 - 12)[..., None] + rng.random((h, w, 3)) * 2
    rgb = np.exp2(e) * (0.5 + 0.5 * np.stack([u, v, 1 - u], -1))
    rgb[(x.astype(int) + y.astype(int)) % 11 == 0] *= -0.25
    rgb[(x.astype(int) * 7 + y.astype(int)) % 13 == 0] = 0.0
    a = np.clip(0.1 + 0.9 * v + rng.random((h, w)) * 0.05, 0, 1)
    return np.concatenate([rgb, a[..., None]], -1).astype(np.float32)


# ---------------------------------------------------------------- EXR

EXR_LINES = []


def exr_expect(name, args, fmt, arr):
    """arr: (h, w, c) array already in the output dtype."""
    h, w = arr.shape[:2]
    EXR_LINES.append(f"{name} {args} {fmt} {w} {h} {fnv_np(arr)}")


def exr_read(path):
    """Reference decode: {part index: {channel: array}} via the OpenEXR bindings."""
    import OpenEXR

    f = OpenEXR.File(path, separate_channels=True)
    return [{k: v.pixels for k, v in p.channels.items()} for p in f.parts], f


def as_out(chans, alpha_default=True):
    """RGB(A) policy: all half -> Rgb16Float / Rgba16Float; otherwise Rgba32Float."""
    half = all(c.dtype == np.float16 for c in chans.values())
    names = [n for n in "RGB" if n in chans]
    shape = next(iter(chans.values())).shape
    if half:
        cs = [chans.get(n, np.zeros(shape, np.float16)) for n in "RGB"]
        if "A" in chans:
            return "Rgba16Float", np.stack(cs + [chans["A"]], -1).astype("<f2")
        return "Rgb16Float", np.stack(cs, -1).astype("<f2")
    cs = [chans[n].astype(np.float32) if n in chans else np.zeros(shape, np.float32) for n in "RGB"]
    a = chans["A"].astype(np.float32) if "A" in chans else np.ones(shape, np.float32)
    assert names
    return "Rgba32Float", np.stack(cs + [a], -1).astype("<f4")


def place(arr, data_win, disp_win, fill):
    """Puts a data-window image into the display window (pixels outside = fill per channel)."""
    (dx0, dy0), (dx1, dy1) = data_win
    (px0, py0), (px1, py1) = disp_win
    ow, oh = px1 - px0 + 1, py1 - py0 + 1
    out = np.empty((oh, ow, arr.shape[2]), arr.dtype)
    out[...] = np.array(fill, dtype=arr.dtype)
    for y in range(oh):
        sy = py0 + y - dy0
        if not 0 <= sy < arr.shape[0]:
            continue
        for x in range(ow):
            sx = px0 + x - dx0
            if 0 <= sx < arr.shape[1]:
                out[y, x] = arr[sy, sx]
    return out


def gen_exr():
    import OpenEXR

    os.makedirs(EXR_OUT, exist_ok=True)
    w, h = 37, 29
    src = hdr_rgba(w, h, 1)
    half = src.astype(np.float16)

    comps = [
        ("none", OpenEXR.NO_COMPRESSION),
        ("rle", OpenEXR.RLE_COMPRESSION),
        ("zips", OpenEXR.ZIPS_COMPRESSION),
        ("zip", OpenEXR.ZIP_COMPRESSION),
        ("piz", OpenEXR.PIZ_COMPRESSION),
        ("pxr24", OpenEXR.PXR24_COMPRESSION),
        ("b44", OpenEXR.B44_COMPRESSION),
        ("b44a", OpenEXR.B44A_COMPRESSION),
        ("dwaa", OpenEXR.DWAA_COMPRESSION),
        ("dwab", OpenEXR.DWAB_COMPRESSION),
    ]
    # RGBA half, scanlines, every compression (B44 / DWA are lossy: expected = OpenEXR decode).
    for cname, c in comps:
        name = f"rgba16_{cname}.exr"
        path = os.path.join(EXR_OUT, name)
        chans = {n: half[..., i] for i, n in enumerate("RGBA")}
        OpenEXR.File({"compression": c, "type": OpenEXR.scanlineimage}, chans).write(path)
        parts, _ = exr_read(path)
        fmt, arr = as_out(parts[0])
        exr_expect(name, "decode", fmt, arr)

    # RGB f32 (PXR24 rounds f32 to 24 bits).
    for cname, c in [("zip", OpenEXR.ZIP_COMPRESSION), ("piz", OpenEXR.PIZ_COMPRESSION),
                     ("pxr24", OpenEXR.PXR24_COMPRESSION), ("rle", OpenEXR.RLE_COMPRESSION)]:
        name = f"rgb32_{cname}.exr"
        path = os.path.join(EXR_OUT, name)
        chans = {n: src[..., i].copy() for i, n in enumerate("RGB")}
        OpenEXR.File({"compression": c}, chans).write(path)
        parts, _ = exr_read(path)
        fmt, arr = as_out(parts[0])
        exr_expect(name, "decode", fmt, arr)

    # u32 single channel (values above 2^24 round in R32Float); raw plane check too.
    u = (np.arange(w * h, dtype=np.uint64).reshape(h, w) * 2654435761 % (1 << 32)).astype(np.uint32)
    u[0, :4] = [0, 1, (1 << 24) + 1, 0xFFFFFFFF]
    name = "id_u32_zip.exr"
    path = os.path.join(EXR_OUT, name)
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION}, {"id": u}).write(path)
    parts, _ = exr_read(path)
    exr_expect(name, "decode", "R32Float", parts[0]["id"].astype("<f4")[..., None])
    exr_expect(name, "plane 0 0,0 id", "U32", parts[0]["id"].astype("<u4")[..., None])

    # Mixed sample types -> Rgba32Float; u32 alpha converted to float.
    name = "mixed_piz.exr"
    path = os.path.join(EXR_OUT, name)
    chans = {"R": half[..., 0], "G": src[..., 1].copy(), "B": half[..., 2],
             "A": (src[..., 3] * 1000).astype(np.uint32)}
    OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION}, chans).write(path)
    parts, _ = exr_read(path)
    fmt, arr = as_out(parts[0])
    exr_expect(name, "decode", fmt, arr)

    # Single channels: Y half -> R16Float, Z f32 -> R32Float; Y + A -> Rgba16Float.
    name = "y16_zip.exr"
    path = os.path.join(EXR_OUT, name)
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION}, {"Y": half[..., 1]}).write(path)
    parts, _ = exr_read(path)
    exr_expect(name, "decode", "R16Float", parts[0]["Y"].astype("<f2")[..., None])
    name = "z32_none.exr"
    path = os.path.join(EXR_OUT, name)
    OpenEXR.File({"compression": OpenEXR.NO_COMPRESSION}, {"Z": src[..., 0].copy()}).write(path)
    parts, _ = exr_read(path)
    exr_expect(name, "decode", "R32Float", parts[0]["Z"].astype("<f4")[..., None])
    name = "ya16_rle.exr"
    path = os.path.join(EXR_OUT, name)
    OpenEXR.File({"compression": OpenEXR.RLE_COMPRESSION},
                 {"Y": half[..., 1], "A": half[..., 3]}).write(path)
    parts, _ = exr_read(path)
    yy, aa = parts[0]["Y"], parts[0]["A"]
    exr_expect(name, "decode", "Rgba16Float", np.stack([yy, yy, yy, aa], -1).astype("<f2"))

    # Luminance / chroma (full resolution): RGB like OpenEXR RgbaInputFile, Rec. 709 weights.
    name = "yc16_zip.exr"
    path = os.path.join(EXR_OUT, name)
    lum = np.abs(half[..., 1]).astype(np.float16)
    ry = (src[..., 0] * 0.0 + np.linspace(-0.5, 0.5, w)[None, :]).astype(np.float16)
    by = (np.linspace(0.4, -0.4, h)[:, None] + src[..., 2] * 0.0).astype(np.float16)
    ry[0, :3] = 0
    by[0, :3] = 0
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION}, {"Y": lum, "RY": ry, "BY": by}).write(path)
    parts, _ = exr_read(path)
    exr_expect(name, "decode", "Rgb16Float", yc_to_rgb(parts[0]["Y"], parts[0]["RY"], parts[0]["BY"]))

    # Tiled RGBA half, partial edge tiles.
    for cname, c in [("none", OpenEXR.NO_COMPRESSION), ("zip", OpenEXR.ZIP_COMPRESSION),
                     ("piz", OpenEXR.PIZ_COMPRESSION), ("b44a", OpenEXR.B44A_COMPRESSION)]:
        name = f"tiled16_{cname}.exr"
        path = os.path.join(EXR_OUT, name)
        td = OpenEXR.TileDescription()
        td.xSize, td.ySize = 16, 8
        chans = {n: half[..., i] for i, n in enumerate("RGBA")}
        OpenEXR.File({"compression": c, "type": OpenEXR.tiledimage, "tiles": td}, chans).write(path)
        parts, _ = exr_read(path)
        fmt, arr = as_out(parts[0])
        exr_expect(name, "decode", fmt, arr)

    # Data window inside / overlapping the display window.
    name = "window_zip.exr"
    path = os.path.join(EXR_OUT, name)
    data_win = ((5, -3), (5 + w - 1, -3 + h - 1))
    disp_win = ((0, 0), (39, 31))
    hdr = {"compression": OpenEXR.ZIP_COMPRESSION,
           "dataWindow": (np.array(data_win[0], "i4"), np.array(data_win[1], "i4")),
           "displayWindow": (np.array(disp_win[0], "i4"), np.array(disp_win[1], "i4"))}
    OpenEXR.File(hdr, {n: half[..., i] for i, n in enumerate("RGBA")}).write(path)
    parts, _ = exr_read(path)
    fmt, arr = as_out(parts[0])
    exr_expect(name, "decode", fmt, place(arr, data_win, disp_win, [0, 0, 0, 0]))
    exr_expect(name, "part=0 layer=- level=0,0 window=data", fmt, arr)
    name = "window_rgb_zip.exr"  # no alpha: outside pixels are (0, 0, 0) / alpha 1 in f32
    path = os.path.join(EXR_OUT, name)
    OpenEXR.File(hdr, {n: src[..., i].copy() for i, n in enumerate("RGB")}).write(path)
    parts, _ = exr_read(path)
    fmt, arr = as_out(parts[0])
    exr_expect(name, "decode", fmt, place(arr, data_win, disp_win, [0, 0, 0, 1]))

    # Multi-part, layers: part 0 "beauty" (RGBA + diffuse.* + specular.*), part 1 "depth"
    # (tiled Z f32 with its own data window).
    name = "multipart.exr"
    path = os.path.join(EXR_OUT, name)
    b = {n: half[..., i] for i, n in enumerate("RGBA")}
    b.update({f"diffuse.{n}": (half[..., i] * 0.5).astype(np.float16) for i, n in enumerate("RGB")})
    b.update({f"specular.{n}": src[::-1, :, i].copy() for i, n in enumerate("RGB")})
    td = OpenEXR.TileDescription()
    td.xSize, td.ySize = 8, 8
    zh = src[:20, :30, 0].copy()
    dhdr = {"compression": OpenEXR.PIZ_COMPRESSION, "type": OpenEXR.tiledimage, "tiles": td,
            "dataWindow": (np.array([2, 3], "i4"), np.array([31, 22], "i4")),
            "displayWindow": (np.array([0, 0], "i4"), np.array([w - 1, h - 1], "i4"))}
    p0 = OpenEXR.Part({"compression": OpenEXR.ZIP_COMPRESSION}, b, "beauty")
    p1 = OpenEXR.Part(dhdr, {"Z": zh}, "depth")
    OpenEXR.File([p0, p1]).write(path)
    parts, _ = exr_read(path)
    fmt, arr = as_out({k: v for k, v in parts[0].items() if "." not in k})
    exr_expect(name, "decode", fmt, arr)
    fmt, arr = as_out({k.split(".")[1]: v for k, v in parts[0].items() if k.startswith("diffuse.")})
    exr_expect(name, "part=0 layer=diffuse level=0,0 window=display", fmt, arr)
    fmt, arr = as_out({k.split(".")[1]: v for k, v in parts[0].items() if k.startswith("specular.")})
    exr_expect(name, "part=0 layer=specular level=0,0 window=display", fmt, arr)
    z = parts[1]["Z"].astype("<f4")[..., None]
    exr_expect(name, "part=1 layer=- level=0,0 window=data", "R32Float", z)
    exr_expect(name, "part=1 layer=- level=0,0 window=display", "R32Float",
               place(z, ((2, 3), (31, 22)), ((0, 0), (w - 1, h - 1)), [0]))

    gen_exr_mip(half)
    gen_exr_rip(half)
    gen_exr_subsampled(half)

    with open(os.path.join(EXR_OUT, "expected.txt"), "w") as f:
        f.write("# generated by scripts/gen_file_testdata_exrtiff.py\n")
        f.write("\n".join(EXR_LINES) + "\n")


def luminance_weights():
    """OpenEXR computeYw (Imath RGBtoXYZ row Y) for Rec. 709, in float32 like imgio."""
    f = np.float32
    red, green, blue, white = (f(0.64), f(0.33)), (f(0.30), f(0.60)), (f(0.15), f(0.06)), (f(0.3127), f(0.3290))
    y = f(1.0)
    xx = white[0] * y / white[1]
    zz = (f(1.0) - white[0] - white[1]) * y / white[1]
    d = red[0] * (blue[1] - green[1]) + blue[0] * (green[1] - red[1]) + green[0] * (red[1] - blue[1])
    sr = (xx * (blue[1] - green[1]) - green[0] * (y * (blue[1] - f(1.0)) + blue[1] * (xx + zz))
          + blue[0] * (y * (green[1] - f(1.0)) + green[1] * (xx + zz))) / d
    sg = (xx * (red[1] - blue[1]) + red[0] * (y * (blue[1] - f(1.0)) + blue[1] * (xx + zz))
          - blue[0] * (y * (red[1] - f(1.0)) + red[1] * (xx + zz))) / d
    sb = (xx * (green[1] - red[1]) - red[0] * (y * (green[1] - f(1.0)) + green[1] * (xx + zz))
          + green[0] * (y * (red[1] - f(1.0)) + red[1] * (xx + zz))) / d
    return f(sr * red[1]), f(sg * green[1]), f(sb * blue[1])


def yc_to_rgb(lum, ry, by):
    yw = luminance_weights()
    l, ry, by = lum.astype(np.float32), ry.astype(np.float32), by.astype(np.float32)
    r = (ry + np.float32(1)) * l
    b = (by + np.float32(1)) * l
    g = (l - r * yw[0] - b * yw[2]) / yw[1]
    gray = (ry == 0) & (by == 0)
    out = np.stack([np.where(gray, l, r), np.where(gray, l, g), np.where(gray, l, b)], -1)
    return out.astype("<f2")


def box_down(a):
    """2x box downsample (round down sizes), float32 math."""
    h, w = a.shape[:2]
    nh, nw = max(h // 2, 1), max(w // 2, 1)
    a = a.astype(np.float32)
    if h > 1:
        a = (a[0:nh * 2:2] + a[1:nh * 2:2]) * 0.5
    if w > 1:
        a = (a[:, 0:nw * 2:2] + a[:, 1:nw * 2:2]) * 0.5
    return a


def gen_exr_mip(half):
    """Tiled mip chain written by OpenImageIO (OpenEXR C++), our own box-filtered levels."""
    import OpenImageIO as oiio

    name = "mip16_zip.exr"
    path = os.path.join(EXR_OUT, name)
    levels = [half.astype(np.float32)]
    while max(levels[-1].shape[:2]) > 1:
        levels.append(box_down(levels[-1]))
    levels = [l.astype(np.float16) for l in levels]
    out = oiio.ImageOutput.create(path)
    for i, lv in enumerate(levels):
        spec = oiio.ImageSpec(lv.shape[1], lv.shape[0], 4, "half")
        spec.tile_width, spec.tile_height = 16, 16
        spec.attribute("compression", "zip")
        spec.attribute("textureformat", "Plain Texture")
        assert out.open(path, spec, "Create" if i == 0 else "AppendMIPLevel"), out.geterror()
        assert out.write_image(lv), out.geterror()
    assert out.close(), out.geterror()
    del out
    inp = oiio.ImageInput.open(path)
    m = 0
    while inp.seek_subimage(0, m):
        px = inp.read_image(0, m, 0, 4, "half")
        assert px is not None, inp.geterror()
        exr_expect(name, f"part=0 layer=- level={m},{m} window=data", "Rgba16Float", px.astype("<f2"))
        m += 1
    assert m == len(levels), (m, len(levels))
    inp.close()


def exr_attr(name, typ, data):
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data


def exr_header(w, h, chans, tiles=None, compression=0, line_order=0):
    """Single-part header; chans = [(name, pixel type, xs, ys)]; tiles = (tw, th, mode)."""
    cl = b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", t, 0, xs, ys) for n, t, xs, ys in chans) + b"\0"
    box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    hdr = exr_attr("channels", "chlist", cl)
    hdr += exr_attr("compression", "compression", bytes([compression]))
    hdr += exr_attr("dataWindow", "box2i", box)
    hdr += exr_attr("displayWindow", "box2i", box)
    hdr += exr_attr("lineOrder", "lineOrder", bytes([line_order]))
    hdr += exr_attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    hdr += exr_attr("screenWindowCenter", "v2f", struct.pack("<ff", 0, 0))
    hdr += exr_attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    flags = 0
    if tiles:
        hdr += exr_attr("tiles", "tiledesc", struct.pack("<IIB", tiles[0], tiles[1], tiles[2]))
        flags = 0x200
    return struct.pack("<ii", 20000630, 2 | flags) + hdr + b"\0"


def gen_exr_rip(half):
    """Uncompressed tiled rip map (no Python writer exists: packed by hand)."""
    name = "rip16_none.exr"
    path = os.path.join(EXR_OUT, name)
    base = half[:21, :, :2].astype(np.float32)  # G, R channels of a 37x21 crop
    h, w = base.shape[:2]
    nx, ny = w.bit_length(), h.bit_length()
    tw = th = 8
    levels = {}
    for ly in range(ny):
        for lx in range(nx):
            lw, lh = max(w >> lx, 1), max(h >> ly, 1)
            # deterministic content per level: base sampled at the level grid + level marker
            ys = (np.arange(lh) * h // lh)[:, None]
            xs = (np.arange(lw) * w // lw)[None, :]
            levels[(lx, ly)] = (base[ys, xs] + np.float32(lx * 0.25 + ly * 0.5)).astype(np.float16)
    chunks = []
    for ly in range(ny):
        for lx in range(nx):
            lv = levels[(lx, ly)]
            lh, lw = lv.shape[:2]
            for ty in range((lh + th - 1) // th):
                for tx in range((lw + tw - 1) // tw):
                    t = lv[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                    # per line: channel G then R (sorted), little-endian halves
                    data = b"".join(t[r, :, 0].astype("<f2").tobytes() + t[r, :, 1].astype("<f2").tobytes()
                                    for r in range(t.shape[0]))
                    chunks.append(struct.pack("<iiiii", tx, ty, lx, ly, len(data)) + data)
    hdr = exr_header(w, h, [("G", 1, 1, 1), ("R", 1, 1, 1)], tiles=(tw, th, 2))
    off = len(hdr) + 8 * len(chunks)
    table = b""
    for c in chunks:
        table += struct.pack("<Q", off)
        off += len(c)
    with open(path, "wb") as f:
        f.write(hdr + table + b"".join(chunks))
    # Reference check: OpenEXR 3.5 (Python bindings) accepts the chunk table and reads level
    # (0, 0); the other levels follow the spec's offset table order (ly, lx, ty, tx), which the
    # bindings cannot read (OpenImageIO and OpenEXRCore 3.1 mis-index rip levels).
    import OpenEXR

    ch = OpenEXR.File(path, separate_channels=True).parts[0].channels
    assert np.array_equal(ch["G"].pixels, levels[(0, 0)][..., 0])
    assert np.array_equal(ch["R"].pixels, levels[(0, 0)][..., 1])
    for (lx, ly), lv in levels.items():
        out = np.stack([lv[..., 1], lv[..., 0], np.zeros_like(lv[..., 0])], -1).astype("<f2")
        exr_expect(name, f"part=0 layer=- level={lx},{ly} window=data", "Rgb16Float", out)


def gen_exr_subsampled(half):
    """Y + RY/BY subsampled 2x2 (as written by RgbaOutputFile WRITE_YC), uncompressed scanlines,
    packed by hand: the exr crate cannot decode subsampled channels (Unsupported)."""
    name = "yc_sub_none.exr"
    path = os.path.join(EXR_OUT, name)
    w, h = 16, 8
    y = half[:h, :w, 1]
    c = half[:h:2, :w:2, 0]
    hdr = exr_header(w, h, [("BY", 1, 2, 2), ("RY", 1, 2, 2), ("Y", 1, 1, 1)])
    chunks = []
    for row in range(h):
        data = b""
        if row % 2 == 0:
            data += c[row // 2].astype("<f2").tobytes() * 2
        data += y[row].astype("<f2").tobytes()
        chunks.append(struct.pack("<ii", row, len(data)) + data)
    off = len(hdr) + 8 * len(chunks)
    table = b""
    for ch in chunks:
        table += struct.pack("<Q", off)
        off += len(ch)
    with open(path, "wb") as f:
        f.write(hdr + table + b"".join(chunks))
    import OpenEXR

    f = OpenEXR.File(path, separate_channels=True)
    ch = f.parts[0].channels
    assert np.array_equal(ch["Y"].pixels, y) and ch["RY"].xSampling == 2
    EXR_LINES.append(f"{name} unsupported")


def verify_exr(d):
    """imgio-encoded EXR files: OpenEXR's decode equals the source (lossless compressions) and
    imgio's own decode (all compressions); mip levels read with OpenImageIO."""
    import OpenEXR
    import OpenImageIO as oiio

    ok = True
    names = sorted(n for n in os.listdir(d) if n.startswith("exr_") and n.endswith(".exr"))
    chans = {"Rgba16Float": ("RGBA", "<f2"), "Rgb16Float": ("RGB", "<f2"), "R16Float": ("Y", "<f2"),
             "Rgba32Float": ("RGBA", "<f4"), "R32Float": ("Y", "<f4")}
    for n in names:
        parts = n[4:-4].split("_")
        fmt, comp = parts[0], parts[1]
        names_, dt = chans[fmt]
        if parts[2] == "mips":
            inp = oiio.ImageInput.open(os.path.join(d, n))
            m = 0
            while inp.seek_subimage(0, m):
                spec = inp.spec()
                px = inp.read_image(0, m, 0, len(names_), "half" if dt == "<f2" else "float")
                want = np.frombuffer(open(os.path.join(d, f"exr_{fmt}_mip{m}.raw"), "rb").read(), dt)
                good = np.array_equal(px.astype(dt).reshape(-1).view(np.uint8), want.view(np.uint8))
                ok &= good
                print(f"{n} level {m} {spec.width}x{spec.height}: {'ok' if good else 'MISMATCH'}")
                m += 1
            inp.close()
            continue
        f = OpenEXR.File(os.path.join(d, n), separate_channels=True)
        ch = f.parts[0].channels
        assert sorted(ch.keys()) == sorted(names_), (n, ch.keys())
        h, w = ch[names_[0]].pixels.shape
        got = np.stack([ch[c].pixels for c in names_], -1).astype(dt)
        src = np.frombuffer(open(os.path.join(d, f"exr_src_{fmt}_{w}x{h}.raw"), "rb").read(), dt)
        dec = np.frombuffer(open(os.path.join(d, n + ".dec"), "rb").read(), dt)
        lossless = not ((comp == "pxr24" and dt == "<f4") or (comp in ("b44", "b44a") and dt == "<f2"))
        g = got.reshape(-1).view(np.uint8)
        good = np.array_equal(g, dec.view(np.uint8))
        if lossless:
            good &= np.array_equal(g, src.view(np.uint8))
        ok &= good
        print(f"{n}: {'ok' if good else 'MISMATCH'} ({'lossless' if lossless else 'lossy = imgio decode'})")
    if ok and os.path.exists(os.path.join(d, "exr_encode.txt")):
        src = open(os.path.join(d, "exr_encode.txt")).read()
        open(os.path.join(EXR_OUT, "encode.txt"), "w").write(src)
        print("testdata/file/exr/encode.txt updated")
    return ok


# ---------------------------------------------------------------- TIFF

TIFF_LINES = []


def raw_rgba(name):
    """testdata/images/raw/<name>.rgba as (h, w, 4) uint8 (size from the name)."""
    w, h = (int(v) for v in name.rsplit("_", 1)[1].split("x"))
    d = np.fromfile(os.path.join(ROOT, "testdata", "images", "raw", name + ".rgba"), np.uint8)
    return d.reshape(h, w, 4)


def tiff_expect(name, page, fmt, arr):
    h, w = arr.shape[:2]
    TIFF_LINES.append(f"{name} page={page} {fmt} {w} {h} {fnv_np(arr)}")


def libtiff_rgba(path, page=0):
    """libtiff TIFFReadRGBAImageOriented (top-left): (h, w, 4) uint8."""
    lib = ctypes.CDLL("libtiff.so.6")
    lib.TIFFOpen.restype = ctypes.c_void_p
    lib.TIFFOpen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.TIFFSetWarningHandler(None)
    tif = lib.TIFFOpen(path.encode(), b"r")
    assert tif, path
    tif = ctypes.c_void_p(tif)
    assert lib.TIFFSetDirectory(tif, page) == 1
    w, h = ctypes.c_uint32(), ctypes.c_uint32()
    lib.TIFFGetField(tif, 256, ctypes.byref(w))
    lib.TIFFGetField(tif, 257, ctypes.byref(h))
    raster = (ctypes.c_uint32 * (w.value * h.value))()
    assert lib.TIFFReadRGBAImageOriented(tif, w, h, raster, 1, 0) == 1, path
    lib.TIFFClose(tif)
    a = np.frombuffer(raster, np.uint32).reshape(h.value, w.value)
    return np.stack([(a >> s) & 0xFF for s in (0, 8, 16, 24)], -1).astype(np.uint8)


def policy(a, photometric="rgb", alpha=False):
    """Expected imgio output of tifffile-decoded samples (h, w[, c])."""
    if a.ndim == 2:
        a = a[..., None]
    dt = a.dtype
    gray = photometric in ("minisblack", "miniswhite")
    if dt == np.uint8:
        fmt = {(True, False): "R8Unorm", (True, True): "Rg8Unorm", (False, False): "Rgb8Srgb",
               (False, True): "Rgba8Srgb"}[(gray, alpha)]
        return fmt, a
    if dt == np.uint16:
        if gray:
            return ("Rg16Unorm" if alpha else "R16Unorm"), a.astype("<u2")
        if not alpha:
            a = np.concatenate([a, np.full(a.shape[:2] + (1,), 65535, np.uint16)], -1)
        return "Rgba16Unorm", a.astype("<u2")
    if dt == np.float16:
        if gray and not alpha:
            return "R16Float", a.astype("<f2")
        if gray:
            a = np.concatenate([a[..., :1]] * 3 + [a[..., 1:2]], -1)
            return "Rgba16Float", a.astype("<f2")
        return ("Rgba16Float" if alpha else "Rgb16Float"), a.astype("<f2")
    if dt == np.uint32:
        f = (a.astype(np.float64) / 4294967295.0).astype(np.float32)
    elif dt == np.int8:
        f = a.astype(np.float32) / np.float32(127.0)
    elif dt == np.int16:
        f = a.astype(np.float32) / np.float32(32767.0)
    elif dt == np.int32:
        f = (a.astype(np.float64) / 2147483647.0).astype(np.float32)
    else:
        f = a.astype(np.float32)
    if gray and not alpha:
        return "R32Float", f.astype("<f4")
    if gray:
        f = np.concatenate([f[..., :1]] * 3 + [f[..., 1:2]], -1)
    elif not alpha:
        f = np.concatenate([f, np.ones(f.shape[:2] + (1,), np.float32)], -1)
    return "Rgba32Float", f.astype("<f4")


def write_tiff(path, w, h, tags, strips, le=True):
    """Minimal classic TIFF writer for cases no tool writes: tags = {tag: (type, values)};
    strip offsets / byte counts are filled in."""
    e = "<" if le else ">"
    fmt = {1: "B", 3: "H", 4: "I", 5: "II", 7: "B"}
    size = {1: 1, 3: 2, 4: 4, 5: 8, 7: 1}
    tags = dict(tags)
    tags[256] = (4, [w])
    tags[257] = (4, [h])
    tags[273] = (4, [0] * len(strips))
    tags[279] = (4, [len(s) for s in strips])
    n = len(tags)
    ifd_off = 8
    data_off = ifd_off + 2 + n * 12 + 4
    extra = b""
    entries = []
    strip_start = None
    for tag in sorted(tags):
        typ, vals = tags[tag]
        count = len(vals) // (2 if typ == 5 else 1)
        payload = b"".join(struct.pack(e + fmt[typ], *((v,) if typ != 5 else v)) for v in
                           ([vals[i:i + 2] for i in range(0, len(vals), 2)] if typ == 5 else vals))
        if tag == 273:
            strip_start = len(entries)
        if len(payload) <= 4:
            entries.append([tag, typ, count, payload.ljust(4, b"\0"), None])
        else:
            entries.append([tag, typ, count, None, len(extra)])
            extra += payload
            if len(extra) % 2:
                extra += b"\0"
    pix_off = data_off + len(extra)
    offs, o = [], pix_off
    for s_ in strips:
        offs.append(o)
        o += len(s_)
    # rewrite strip offsets
    ent = entries[strip_start]
    payload = b"".join(struct.pack(e + "I", v) for v in offs)
    if ent[4] is None:
        ent[3] = payload.ljust(4, b"\0")
    else:
        extra = extra[:ent[4]] + payload + extra[ent[4] + len(payload):]
    out = (b"II*\0" if le else b"MM\0*") + struct.pack(e + "I", ifd_off)
    out += struct.pack(e + "H", n)
    for tag, typ, count, inline, off in entries:
        val = inline if inline is not None else struct.pack(e + "I", data_off + off)
        out += struct.pack(e + "HHI", tag, typ, count) + val
    out += struct.pack(e + "I", 0) + extra + b"".join(strips)
    with open(path, "wb") as f:
        f.write(out)


def ycbcr_units(ycc, sh, sv):
    """Packs full-resolution YCbCr (h, w, 3) into subsampled data units (chroma = top-left)."""
    h, w = ycc.shape[:2]
    out = bytearray()
    for uy in range(0, h + (-h % sv), sv):
        for ux in range(0, w + (-w % sh), sh):
            for dy in range(sv):
                for dx in range(sh):
                    y, x = min(uy + dy, h - 1), min(ux + dx, w - 1)
                    out.append(ycc[y, x, 0])
            out += bytes([ycc[uy, ux, 1], ycc[uy, ux, 2]])
    return bytes(out)


def gen_tiff():
    import tifffile
    from PIL import Image

    os.makedirs(TIFF_OUT, exist_ok=True)
    rgba = raw_rgba("rgba_64x48")
    rgb = raw_rgba("rgb_37x29")[..., :3].copy()
    cut = raw_rgba("cutout_rgba_96x64")[:40, :52].copy()
    gray = rgb[..., 1].copy()
    rng = np.random.default_rng(5)
    g16 = (gray.astype(np.uint16) * 257 + rng.integers(0, 256, gray.shape)).astype(np.uint16)
    rgb16 = (rgb.astype(np.uint16) * 257 + rng.integers(0, 256, rgb.shape)).astype(np.uint16)
    rgba16 = (cut.astype(np.uint16) * 257 + rng.integers(0, 256, cut.shape)).astype(np.uint16)
    hdr = hdr_rgba(37, 29, 7)

    def tf(name, data, page_args=None, **kw):
        path = os.path.join(TIFF_OUT, name)
        tifffile.imwrite(path, data, **kw)
        return path

    def check(name, photometric="rgb", alpha=False, page=0):
        path = os.path.join(TIFF_OUT, name)
        with tifffile.TiffFile(path) as t:
            pg = t.pages[page]
            a = pg.asarray()
            if pg.planarconfig == tifffile.PLANARCONFIG.SEPARATE and pg.samplesperpixel > 1:
                a = np.moveaxis(a, 0, -1)
        fmt, arr = policy(a, photometric, alpha)
        tiff_expect(name, page, fmt, arr)
        return a

    # RGB 8-bit: compressions, predictor, byte order, tiles, planar, BigTIFF.
    for comp, pred in [(None, None), ("lzw", None), ("lzw", 2), ("zlib", None), ("zlib", 2),
                       ("deflate", 2), ("packbits", None)]:
        name = f"rgb8_{comp or 'none'}{'_pred' if pred else ''}.tif"
        tf(name, rgb, photometric="rgb", compression=comp, predictor=pred, rowsperstrip=7)
        check(name)
    tf("rgb8_lzw_be_tiled.tif", rgb, photometric="rgb", compression="lzw", predictor=2,
       byteorder=">", tile=(16, 16))
    check("rgb8_lzw_be_tiled.tif")
    tf("rgb8_planar_zlib.tif", np.moveaxis(rgb, -1, 0), photometric="rgb", compression="zlib", planarconfig="separate",
       rowsperstrip=10)
    check("rgb8_planar_zlib.tif")
    tf("rgb8_planar_tiled_big.tif", np.moveaxis(rgb, -1, 0), photometric="rgb", compression="lzw", planarconfig="separate",
       tile=(16, 32), bigtiff=True, byteorder=">")
    check("rgb8_planar_tiled_big.tif")
    tf("rgba8_bigtiff_packbits.tif", rgba, photometric="rgb", compression="packbits", bigtiff=True,
       extrasamples=["unassalpha"])
    check("rgba8_bigtiff_packbits.tif", alpha=True)
    tf("rgba8_assoc_lzw.tif", cut, photometric="rgb", compression="lzw", extrasamples=["assocalpha"])
    check("rgba8_assoc_lzw.tif", alpha=True)
    # RGB + unspecified extra sample: dropped.
    tf("rgbx8_none.tif", rgba, photometric="rgb", extrasamples=["unspecified"])
    tiff_expect("rgbx8_none.tif", 0, "Rgb8Srgb", rgba[..., :3])

    # Gray and gray + alpha, 8 / 16 bits; min-is-white (integers inverted).
    tf("gray8_deflate.tif", gray, photometric="minisblack", compression="deflate", predictor=2)
    check("gray8_deflate.tif", "minisblack")
    tf("graya8_lzw.tif", np.stack([gray, gray[::-1]], -1), photometric="minisblack",
       compression="lzw", extrasamples=["unassalpha"])
    check("graya8_lzw.tif", "minisblack", True)
    tf("gray16_be_lzw_pred.tif", g16, photometric="minisblack", compression="lzw", predictor=2,
       byteorder=">", tile=(16, 16))
    check("gray16_be_lzw_pred.tif", "minisblack")
    tf("graya16_zlib.tif", np.stack([g16, g16[::-1]], -1), photometric="minisblack",
       compression="zlib", predictor=2, extrasamples=["unassalpha"])
    check("graya16_zlib.tif", "minisblack", True)
    tf("gray8_miniswhite.tif", gray, photometric="miniswhite", compression="packbits")
    tiff_expect("gray8_miniswhite.tif", 0, "R8Unorm", 255 - gray)
    assert np.array_equal(libtiff_rgba(os.path.join(TIFF_OUT, "gray8_miniswhite.tif"))[..., 0], 255 - gray)
    tf("gray16_miniswhite.tif", g16, photometric="miniswhite")
    tiff_expect("gray16_miniswhite.tif", 0, "R16Unorm", (65535 - g16).astype("<u2"))

    # Sub-byte gray (1/2/4 bits), checked against libtiff's RGBA reader too.
    for bits in (1, 2, 4):
        v = (gray >> (8 - bits)).astype(np.uint8)
        name = f"gray{bits}.tif"
        tf(name, v, photometric="minisblack", bitspersample=bits)
        a = tifffile.imread(os.path.join(TIFF_OUT, name))
        assert np.array_equal(a, v)
        exp = (a.astype(np.uint32) * 255 // ((1 << bits) - 1)).astype(np.uint8)
        assert np.array_equal(libtiff_rgba(os.path.join(TIFF_OUT, name))[..., 0], exp)
        tiff_expect(name, 0, "R8Unorm", exp)
    # 1-bit min-is-white with FillOrder 2 (LSB first), hand-packed.
    w1, h1 = 37, 29
    bits1 = (gray > 128).astype(np.uint8)
    rows = []
    for y in range(h1):
        bytes_ = np.packbits(bits1[y], bitorder="little")
        rows.append(bytes(bytes_))
    write_tiff(os.path.join(TIFF_OUT, "bilevel_fill2_miniswhite.tif"), w1, h1,
               {258: (3, [1]), 259: (3, [1]), 262: (3, [0]), 266: (3, [2]), 277: (3, [1]),
                278: (4, [h1])}, [b"".join(rows)])
    ref = libtiff_rgba(os.path.join(TIFF_OUT, "bilevel_fill2_miniswhite.tif"))[..., 0]
    assert np.array_equal(ref, (1 - bits1) * 255)
    tiff_expect("bilevel_fill2_miniswhite.tif", 0, "R8Unorm", ref)

    # Palette 8 / 4 bits (16-bit colormap -> >> 8, like libtiff).
    cmap = np.zeros((3, 256), np.uint16)
    cmap[0] = np.arange(256) * 257
    cmap[1] = (255 - np.arange(256)) * 256 + 7
    cmap[2] = (np.arange(256) * 7 % 256) * 257
    idx = gray
    tf("palette8_lzw.tif", idx, photometric="palette", colormap=cmap, compression="lzw")
    pal_exp = np.stack([(cmap[c][idx] >> 8) for c in range(3)], -1).astype(np.uint8)
    assert np.array_equal(libtiff_rgba(os.path.join(TIFF_OUT, "palette8_lzw.tif"))[..., :3], pal_exp)
    tiff_expect("palette8_lzw.tif", 0, "Rgb8Srgb", pal_exp)
    idx4 = (gray >> 4).astype(np.uint8)
    cmap4 = cmap[:, ::16].copy()
    rows4 = b"".join(bytes(np.packbits(np.unpackbits(r[:, None], axis=1)[:, 4:].reshape(-1)))
                     for r in idx4)
    write_tiff(os.path.join(TIFF_OUT, "palette4.tif"), idx4.shape[1], idx4.shape[0],
               {258: (3, [4]), 259: (3, [1]), 262: (3, [3]), 277: (3, [1]), 278: (4, [idx4.shape[0]]),
                320: (3, [int(v) for v in cmap4.reshape(-1)])}, [rows4], le=False)
    assert np.array_equal(tifffile.imread(os.path.join(TIFF_OUT, "palette4.tif")), idx4)
    pal4 = np.stack([(cmap4[c][idx4] >> 8) for c in range(3)], -1).astype(np.uint8)
    assert np.array_equal(libtiff_rgba(os.path.join(TIFF_OUT, "palette4.tif"))[..., :3], pal4)
    tiff_expect("palette4.tif", 0, "Rgb8Srgb", pal4)

    # CMYK 8-bit (libtiff RGBA formula) and 16-bit.
    cmyk = np.concatenate([255 - rgb, (rgb.min(-1, keepdims=True) // 3)], -1).astype(np.uint8)
    tf("cmyk8_lzw.tif", cmyk, photometric="separated", compression="lzw")
    k = 255 - cmyk[..., 3:4].astype(np.uint32)
    exp = (k * (255 - cmyk[..., :3].astype(np.uint32)) // 255).astype(np.uint8)
    assert np.array_equal(libtiff_rgba(os.path.join(TIFF_OUT, "cmyk8_lzw.tif"))[..., :3], exp)
    tiff_expect("cmyk8_lzw.tif", 0, "Rgb8Srgb", exp)
    cmyk16 = np.concatenate([65535 - rgb16, rgb16.min(-1, keepdims=True) // 3], -1).astype(np.uint16)
    tf("cmyk16.tif", cmyk16, photometric="separated")
    k = 65535 - cmyk16[..., 3:4].astype(np.uint64)
    exp = (k * (65535 - cmyk16[..., :3].astype(np.uint64)) // 65535).astype(np.uint16)
    exp = np.concatenate([exp, np.full(exp.shape[:2] + (1,), 65535, np.uint16)], -1)
    tiff_expect("cmyk16.tif", 0, "Rgba16Unorm", exp.astype("<u2"))

    # YCbCr: libtiff tables / subsampling (expected = libtiff RGBA reader).
    ycc = rng.integers(16, 240, (h1, w1, 3)).astype(np.uint8)
    ycc[..., 0] = gray
    for sh, sv, comp in [(1, 1, 1), (2, 2, 1), (2, 1, 5), (4, 2, 1), (1, 2, 32773)]:
        name = f"ycbcr{sh}{sv}.tif"
        data = ycbcr_units(ycc, sh, sv) if (sh, sv) != (1, 1) else ycc.tobytes()
        if comp == 5:
            import imagecodecs
            data = imagecodecs.lzw_encode(data)
        elif comp == 32773:
            import imagecodecs
            data = imagecodecs.packbits_encode(data)
        tags = {258: (3, [8, 8, 8]), 259: (3, [comp]), 262: (3, [6]), 277: (3, [3]),
                278: (4, [h1]), 530: (3, [sh, sv])}
        if (sh, sv) == (2, 2):  # explicit coefficients / reference range
            tags[529] = (5, [2990, 10000, 5870, 10000, 1140, 10000])
            tags[532] = (5, [16, 1, 235, 1, 128, 1, 240, 1, 128, 1, 240, 1])
        write_tiff(os.path.join(TIFF_OUT, name), w1, h1, tags, [data])
        ref = libtiff_rgba(os.path.join(TIFF_OUT, name))[..., :3]
        tiff_expect(name, 0, "Rgb8Srgb", ref)

    # Float / signed / 32-bit integer samples.
    f32 = hdr[..., :3].copy()
    tf("rgb32f_pred3_lzw.tif", f32, photometric="rgb", compression="lzw", predictor=3)
    check("rgb32f_pred3_lzw.tif")
    tf("rgb32f_be_pred3_tiled.tif", f32, photometric="rgb", compression="zlib", predictor=3,
       byteorder=">", tile=(16, 16))
    check("rgb32f_be_pred3_tiled.tif")
    tf("rgba32f_planar.tif", np.moveaxis(hdr, -1, 0), photometric="rgb", planarconfig="separate",
       extrasamples=["unassalpha"], compression="zlib")
    check("rgba32f_planar.tif", alpha=True)
    tf("gray32f_deflate.tif", hdr[..., 0].copy(), photometric="minisblack", compression="deflate")
    check("gray32f_deflate.tif", "minisblack")
    tf("rgb16f_pred3.tif", f32.astype(np.float16), photometric="rgb", compression="zlib", predictor=3)
    check("rgb16f_pred3.tif")
    tf("rgba16f_be.tif", hdr.astype(np.float16), photometric="rgb", byteorder=">",
       extrasamples=["unassalpha"])
    check("rgba16f_be.tif", alpha=True)
    tf("gray16f_pred3.tif", hdr[..., 1].astype(np.float16), photometric="minisblack",
       compression="lzw", predictor=3, tile=(16, 16))
    check("gray16f_pred3.tif", "minisblack")
    tf("gray64f_be.tif", hdr[..., 2].astype(np.float64) * 1.0000001, photometric="minisblack",
       byteorder=">")
    check("gray64f_be.tif", "minisblack")
    u32 = (g16.astype(np.uint32) * 65537 + 12345).astype(np.uint32)
    tf("gray32u_lzw.tif", u32, photometric="minisblack", compression="lzw", predictor=2)
    check("gray32u_lzw.tif", "minisblack")
    tf("gray16s.tif", (g16.astype(np.int32) - 32768).astype(np.int16), photometric="minisblack")
    check("gray16s.tif", "minisblack")
    tf("gray8s_be.tif", (gray.astype(np.int16) - 128).astype(np.int8), photometric="minisblack",
       byteorder=">")
    check("gray8s_be.tif", "minisblack")
    tf("rgb16_zlib_pred.tif", rgb16, photometric="rgb", compression="zlib", predictor=2)
    check("rgb16_zlib_pred.tif")
    tf("rgba16_be_packbits.tif", rgba16, photometric="rgb", compression="packbits", byteorder=">",
       extrasamples=["unassalpha"])
    check("rgba16_be_packbits.tif", alpha=True)

    # Multi-page (mixed page types).
    path = os.path.join(TIFF_OUT, "multipage.tif")
    with tifffile.TiffWriter(path) as t:
        t.write(rgb, photometric="rgb", compression="lzw")
        t.write(g16, photometric="minisblack", compression="zlib")
        t.write(f32, photometric="rgb", compression="lzw", predictor=3, tile=(16, 16))
    check("multipage.tif", page=0)
    check("multipage.tif", "minisblack", page=1)
    check("multipage.tif", page=2)

    # JPEG: tifffile (YCbCr 2x2, full streams), Pillow / libtiff (RGB with JPEGTables, gray).
    # Expected = libtiff's decode (libjpeg); imgio (zune-jpeg) is compared with bounds.
    jpeg_cases = []
    tf("jpeg_ycbcr_tifffile.tif", rgba[..., :3].copy(), photometric="ycbcr", compression="jpeg",
       rowsperstrip=16)
    jpeg_cases.append(("jpeg_ycbcr_tifffile.tif", 3))
    tf("jpeg_ycbcr_tiled.tif", rgba[..., :3].copy(), photometric="ycbcr", compression="jpeg",
       tile=(32, 32))
    jpeg_cases.append(("jpeg_ycbcr_tiled.tif", 3))
    Image.fromarray(rgb).save(os.path.join(TIFF_OUT, "jpeg_rgb_pillow.tif"), compression="jpeg")
    jpeg_cases.append(("jpeg_rgb_pillow.tif", 3))
    Image.fromarray(gray).save(os.path.join(TIFF_OUT, "jpeg_gray_pillow.tif"), compression="jpeg")
    jpeg_cases.append(("jpeg_gray_pillow.tif", 1))
    for name, n in jpeg_cases:
        ref = libtiff_rgba(os.path.join(TIFF_OUT, name))[..., :n]
        fmt = "Rgb8Srgb" if n == 3 else "R8Unorm"
        ref.tofile(os.path.join(TIFF_OUT, name[:-4] + ".ref"))
        h_, w_ = ref.shape[:2]
        TIFF_LINES.append(f"{name} jpeg {fmt} {w_} {h_} {name[:-4]}.ref")

    # Unsupported: CCITT G4 (Pillow / libtiff).
    Image.fromarray((gray > 128).astype(np.uint8) * 255).convert("1").save(
        os.path.join(TIFF_OUT, "bilevel_g4.tif"), compression="group4")
    TIFF_LINES.append("bilevel_g4.tif unsupported")

    with open(os.path.join(TIFF_OUT, "expected.txt"), "w") as f:
        f.write("# generated by scripts/gen_file_testdata_exrtiff.py\n")
        f.write("\n".join(TIFF_LINES) + "\n")


def verify_tiff(d):
    """imgio-encoded TIFF files decode (tifffile) to the source pixels exactly."""
    import tifffile

    ok = True
    for n in sorted(x for x in os.listdir(d) if x.startswith("tiff_") and x.endswith(".tif")):
        src = open(os.path.join(d, n[:-4] + ".src"), "rb").read()
        with tifffile.TiffFile(os.path.join(d, n)) as t:
            pages = [p.asarray() for p in t.pages]
            got = b"".join(np.ascontiguousarray(p).astype(p.dtype.newbyteorder("<")).tobytes() for p in pages)
            info = (t.is_bigtiff, [(p.compression.name, p.predictor) for p in t.pages][:1])
        good = got == src
        ok &= good
        print(f"{n}: {'ok' if good else 'MISMATCH'} {info}")
    if ok and os.path.exists(os.path.join(d, "tiff_encode.txt")):
        src = open(os.path.join(d, "tiff_encode.txt")).read()
        open(os.path.join(TIFF_OUT, "encode.txt"), "w").write(src)
        print("testdata/file/tiff/encode.txt updated")
    return ok


# ---------------------------------------------------------------- main


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        verify(sys.argv[2])
        return
    gen_exr()
    if "gen_tiff" in globals():
        gen_tiff()


def verify(d):
    ok = True
    if "verify_exr" in globals():
        ok &= verify_exr(d)
    if "verify_tiff" in globals():
        ok &= verify_tiff(d)
    print("all verified" if ok else "MISMATCHES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
