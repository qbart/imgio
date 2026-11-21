#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates the BMP / TGA / Radiance HDR test files of imgio's own decoders (deterministic).

Usage: VENV/bin/python -I scripts/gen_file_testdata_own.py [--skip-encoded]

Needs numpy, Pillow, imagecodecs and the `ffmpeg` CLI (offline reference tools, never linked).
Sources are imgio's own images (testdata/images/raw). Writes testdata/file/{bmp,tga,hdr}/:

- `*.bmp` / `*.tga` / `*.hdr`: `pil_*` written by Pillow, `ff_*` by ffmpeg, `ic_*` by
  imagecodecs, everything else hand-built here from the format specs (variants the tools cannot
  write: OS/2 headers, RLE4 / RLE8 with deltas, bit fields, colormapped TGA, odd origins,
  old-style HDR RLE, flipped / X-major resolution strings).
- `expected/*.raw`: the expected decoded pixels (imgio layout: R8 / RG8 / RGB8 / RGBA8 bytes,
  or RGBA f32 little endian for HDR), deduplicated by content.
- `cases.txt`: `file format width height expected reference`, where reference names the
  decoder the expected pixels were checked against when this script ran (`pillow`, `ffmpeg`,
  `imagecodecs`, several joined by `+`), or `source` when no tool reads the variant (then the
  expected pixels are the hand-built source).

Then (unless --skip-encoded) it runs `cargo test --release --test file_<fmt>` with
IMGIO_WRITE_ENCODED=1, which makes the tests write imgio's encoder output to `enc/`, and checks
those files with the reference decoders (exact). The tests later compare the encoders against
the committed `enc/` files byte for byte.
"""
import hashlib
import os
import struct
import subprocess
import sys
import tempfile

import imagecodecs
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "testdata", "images", "raw")
OUT = os.path.join(ROOT, "testdata", "file")


def load(name, w, h):
    return np.fromfile(os.path.join(RAW, name), np.uint8).reshape(h, w, 4)


RGB = load("rgb_37x29.rgba", 37, 29)[..., :3].copy()
MOS = load("mosaic_rgb_50x38.rgba", 50, 38)[..., :3].copy()
MOSA = load("mosaic_rgba_50x38.rgba", 50, 38).copy()
CUT = load("cutout_rgba_96x64.rgba", 96, 64)[10:39, 20:57].copy()  # 37x29, alpha 0..255
GRAY = RGB[..., 0].copy()  # the tests use the red channel of rgb_37x29 as gray source
HDR = np.fromfile(os.path.join(RAW, "hdr_40x24.rgba16f"), np.float16).reshape(24, 40, 4)
HDR = HDR[..., :3].astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Case bookkeeping

class Cases:
    def __init__(self, fmt):
        self.dir = os.path.join(OUT, fmt)
        os.makedirs(os.path.join(self.dir, "expected"), exist_ok=True)
        self.rows = []
        self.used = set()

    def add(self, name, data, fmt, expected, refs):
        """Writes `name` and its expected pixels (numpy array in imgio layout)."""
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(data)
        h, w = expected.shape[:2]
        raw = np.ascontiguousarray(expected).tobytes()
        key = hashlib.sha256(raw).hexdigest()[:16] + ".raw"
        self.used.add(key)
        with open(os.path.join(self.dir, "expected", key), "wb") as f:
            f.write(raw)
        self.rows.append(f"{name} {fmt} {w} {h} {key} {'+'.join(refs) or 'source'}")
        assert len(data) < 50_000, name

    def finish(self):
        for f in os.listdir(os.path.join(self.dir, "expected")):
            if f not in self.used:
                os.remove(os.path.join(self.dir, "expected", f))
        with open(os.path.join(self.dir, "cases.txt"), "w") as f:
            f.write("# file format width height expected reference "
                    "(scripts/gen_file_testdata_own.py)\n")
            f.write("\n".join(self.rows) + "\n")


def fmt_of(a, gray_alpha=False):
    if a.dtype == np.float32:
        return "Rgba32Float"
    if a.ndim == 2:
        return "R8Unorm"
    return {2: "Rg8Unorm", 3: "Rgb8Srgb", 4: "Rgba8Srgb"}[a.shape[2]]


def pillow_pixels(data, like):
    """Pillow's decode of `data` shaped like the expected array `like`."""
    with tempfile.NamedTemporaryFile() as t:
        t.write(data)
        t.flush()
        im = Image.open(t.name)
        im.load()
        if like.ndim == 2:
            return np.asarray(im.convert("L"))
        mode = {2: "LA", 3: "RGB", 4: "RGBA"}[like.shape[2]]
        return np.asarray(im.convert(mode))


def ffmpeg_pixels(data, like, ext):
    """ffmpeg's decode of `data` shaped like `like` (8-bit) or RGBA f32 (HDR)."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "in." + ext)
        with open(p, "wb") as f:
            f.write(data)
        if like.dtype == np.float32:
            pix, ch, dt = "gbrpf32le", 3, np.float32
        else:
            pix, ch, dt = {2: ("ya8", 2, np.uint8), 3: ("rgb24", 3, np.uint8),
                           4: ("rgba", 4, np.uint8)}.get(
                like.shape[2] if like.ndim == 3 else 1, ("gray", 1, np.uint8))
        out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", p, "-f", "rawvideo",
                              "-pix_fmt", pix, "-"], capture_output=True, check=True).stdout
        h, w = like.shape[:2]
        if ch == 1:  # palette files: pal8 -> gray conversion is not exact, rgb24 is
            out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", p, "-f", "rawvideo",
                                  "-pix_fmt", "rgb24", "-"], capture_output=True,
                                 check=True).stdout
            a = np.frombuffer(out, np.uint8).reshape(h, w, 3)
            return a[..., 0] if (a == a[..., :1]).all() else a.mean(-1).astype(np.uint8) + 1
        a = np.frombuffer(out, dt)
        if pix == "gbrpf32le":
            g, b, r = a.reshape(3, h, w)
            return np.stack([r, g, b, np.ones_like(r)], -1)
        return a.reshape((h, w) if ch == 1 else (h, w, ch))


def check(name, got, expected, tol=0):
    if got.shape != expected.shape:
        raise SystemExit(f"{name}: reference shape {got.shape} != {expected.shape}")
    diff = np.abs(got.astype(np.int64) - expected.astype(np.int64)).max() if \
        expected.dtype != np.float32 else (0 if np.array_equal(got, expected) else 1)
    if diff > tol:
        raise SystemExit(f"{name}: reference decode differs by {diff}")


def pil_ref(name, data, expected, tol=0, rgb_only=False):
    """["pillow"] when Pillow decodes `data` to `expected`, [] when Pillow cannot read the
    variant; a mismatch is fatal. rgb_only: Pillow drops the alpha of this variant, compare
    the color channels ("pillow-rgb")."""
    if rgb_only:
        expected = expected[..., :3]
    try:
        got = pillow_pixels(data, expected)
    except (OSError, ValueError) as e:
        print(f"note: Pillow cannot read {name}: {e}")
        return []
    check(name, got, expected, tol)
    return ["pillow-rgb" if rgb_only else "pillow"]


def run(cmd, data=None):
    return subprocess.run(cmd, input=data, capture_output=True, check=True).stdout


def ffmpeg_encode(raw, pix_fmt, w, h, ext, codec_args):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "out." + ext)
        run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt,
             "-s", f"{w}x{h}", "-i", "-", "-frames:v", "1", *codec_args, p], raw)
        with open(p, "rb") as f:
            return f.read()


def pillow_encode(img, ext, **kw):
    with tempfile.NamedTemporaryFile(suffix="." + ext) as t:
        img.save(t.name, **kw)
        with open(t.name, "rb") as f:
            return f.read()


def quantize(rgb, colors):
    """(indices, palette Nx3) with at most `colors` entries (Pillow median cut, no dither)."""
    q = Image.fromarray(rgb, "RGB").quantize(colors, method=Image.Quantize.MEDIANCUT,
                                              dither=Image.Dither.NONE)
    pal = np.array(q.getpalette()[:3 * colors], np.uint8).reshape(-1, 3)
    return np.asarray(q).copy(), pal


def expand(v, bits):
    """imgio's n-bit to 8-bit expansion: round(v * 255 / (2^n - 1))."""
    m = (1 << bits) - 1
    return ((v.astype(np.int64) * 255 + m // 2) // m).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# BMP

def bmp_rows(packed_rows, top_down=False):
    """Pads rows (list of bytes) to 4 bytes and orders them bottom-up unless top_down."""
    rows = [r + b"\0" * (-len(r) % 4) for r in packed_rows]
    return b"".join(rows if top_down else rows[::-1])


def pack_bits(idx, bits):
    out = []
    for row in idx:
        acc, n, b = 0, 0, bytearray()
        for v in row:
            acc = (acc << bits) | int(v)
            n += bits
            if n == 8:
                b.append(acc)
                acc, n = 0, 0
        if n:
            b.append(acc << (8 - n))
        out.append(bytes(b))
    return out


def bmp_file(w, h, bpp, body, hsize=40, comp=0, palette=b"", masks=None, colors_used=0,
             top_down=False, cstype=b"BGRs"):
    """BMP with a `hsize`-byte header. masks: (r, g, b[, a]); in the header for hsize >= 52,
    after a 40-byte header otherwise."""
    after = b""
    if hsize == 12:
        hdr = struct.pack("<IHHHH", 12, w, h, 1, bpp)
    else:
        info = struct.pack("<IiiHHIIiiII", hsize, w, -h if top_down else h, 1, bpp, comp,
                           len(body), 2835, 2835, colors_used, 0)
        hdr = info[:hsize]
        if hsize > 40:
            ext = b""
            if masks is not None and hsize >= 52:
                m = list(masks) + [0] * (4 - len(masks))
                ext = struct.pack("<III", *m[:3]) + (struct.pack("<I", m[3]) if hsize >= 56
                                                      else b"")
            ext = ext.ljust(hsize - 40, b"\0")
            if hsize >= 108:
                ext = ext[:16] + cstype + ext[20:]
            hdr = info + ext
        elif masks is not None:
            after = struct.pack(f"<{len(masks)}I", *masks)
    off = 14 + len(hdr) + len(after) + len(palette)
    return b"BM" + struct.pack("<IHHI", off + len(body), 0, 0, off) + hdr + after + palette + body


def pal_bytes(pal, entry=4):
    return b"".join(bytes([b, g, r]) + (b"\0" if entry == 4 else b"") for r, g, b in pal)


def rle_encode(idx, rle4):
    """BMP RLE8 / RLE4 stream of index rows (bottom-up), fully drawn."""
    out = bytearray()
    for row in idx[::-1]:
        row = [int(v) for v in row]
        i, n = 0, len(row)
        while i < n:
            if rle4:
                # Encoded run of an alternating pair (a, b, a, b, ...).
                j = i + 1
                while j < n and j - i < 255 and row[j] == row[i + (j - i) % 2]:
                    j += 1
                if j - i >= 4 or j == n:
                    pair = (row[i] << 4) | (row[i + 1] if i + 1 < n else 0)
                    out += bytes([j - i, pair])
                    i = j
                    continue
            else:
                j = i + 1
                while j < n and j - i < 255 and row[j] == row[i]:
                    j += 1
                if j - i >= 3:
                    out += bytes([j - i, row[i]])
                    i = j
                    continue
            # Absolute run until the next run of 3 (at least 3 values), or a single run.
            j = i
            while j < n and j - i < 255 and not (j + 2 < n and row[j] == row[j + 1] == row[j + 2]):
                j += 1
            if j - i < 3:
                v = row[i]
                out += bytes([1, (v << 4) if rle4 else v])
                i += 1
                continue
            vals = row[i:j]
            if rle4:
                data = bytes(((vals[k] << 4) | (vals[k + 1] if k + 1 < len(vals) else 0))
                             for k in range(0, len(vals), 2))
            else:
                data = bytes(vals)
            out += bytes([0, len(vals)]) + data + (b"\0" if len(data) % 2 else b"")
            i = j
        out += b"\0\0"  # end of line
    out[-2:] = b"\0\1"  # end of bitmap replaces the last end of line
    return bytes(out)


def rle_with_holes(idx, rle4):
    """RLE stream with a delta, early end-of-lines and an early end of bitmap. Returns
    (stream, drawn mask in top-down order)."""
    h, w = idx.shape
    drawn = np.zeros((h, w), bool)
    out = bytearray()
    y = 0  # bottom-up row index

    def put_run(x, y, n):
        nonlocal out
        row = idx[h - 1 - y]
        v = int(row[x])
        out += bytes([n, (v << 4 | v) if rle4 else v])
        drawn[h - 1 - y, x:x + n] = True
        # the run repeats row[x] (rle4: the same nibble twice)
        idx_exp[h - 1 - y, x:x + n] = v

    def put_abs(x, y, n):
        nonlocal out
        vals = [int(v) for v in idx[h - 1 - y, x:x + n]]
        if rle4:
            data = bytes(((vals[k] << 4) | (vals[k + 1] if k + 1 < n else 0))
                         for k in range(0, n, 2))
        else:
            data = bytes(vals)
        out += bytes([0, n]) + data + (b"\0" if len(data) % 2 else b"")
        drawn[h - 1 - y, x:x + n] = True
        idx_exp[h - 1 - y, x:x + n] = vals

    idx_exp = idx.copy()
    # Row 0: 5 pixels, then a delta (+7, +2) -> row 2 x 12, 6 pixels absolute, end of line.
    put_run(0, 0, 5)
    out += bytes([0, 2, 7, 2])
    put_abs(12, 2, 6)
    out += b"\0\0"
    # Rows 3 .. h-4: full rows (runs of 4 then absolute rest).
    for y in range(3, h - 3):
        put_run(0, y, 4)
        put_abs(4, y, w - 4)
        out += b"\0\0"
    # Row h-3: half the row, end of bitmap (rows h-2, h-1 undrawn).
    put_abs(0, h - 3, w // 2)
    out += b"\0\1"
    return bytes(out), drawn, idx_exp


def gen_bmp():
    c = Cases("bmp")
    w, h = 37, 29
    rgba_cut = CUT

    # Pillow-written: 24-bit, 32-bit BI_RGB with alpha (BITMAPINFOHEADER), 8-bit P / L, 1-bit.
    d = pillow_encode(Image.fromarray(RGB, "RGB"), "bmp")
    check("pil_rgb24", pillow_pixels(d, RGB), RGB)
    c.add("pil_rgb24.bmp", d, "Rgb8Srgb", RGB, ["pillow"])
    d = pillow_encode(Image.fromarray(rgba_cut, "RGBA"), "bmp")
    c.add("pil_rgba32.bmp", d, "Rgba8Srgb", rgba_cut, pil_ref("pil_rgba32", d, rgba_cut, rgb_only=True))
    idx, pal = quantize(RGB, 256)
    p = Image.fromarray(idx, "P")
    p.putpalette(pal.flatten().tolist())
    d = pillow_encode(p, "bmp")
    exp = pal[idx]
    check("pil_p8", pillow_pixels(d, exp), exp)
    c.add("pil_p8.bmp", d, "Rgb8Srgb", exp, ["pillow"])
    d = pillow_encode(Image.fromarray(GRAY, "L"), "bmp")
    check("pil_l8", pillow_pixels(d, GRAY), GRAY)
    c.add("pil_l8.bmp", d, "R8Unorm", GRAY, ["pillow"])
    bw = np.where(GRAY > 127, 255, 0).astype(np.uint8)
    d = pillow_encode(Image.fromarray(bw, "L").convert("1", dither=Image.Dither.NONE), "bmp")
    check("pil_1", pillow_pixels(d, bw), bw)
    c.add("pil_1bit.bmp", d, "R8Unorm", bw, ["pillow"])

    # ffmpeg-written: bgr24, bgra (BI_RGB 32 with alpha), rgb8 (3:3:2 palette), gray, monob,
    # rgb565 / rgb444 (bit fields), rgb555.
    bgr = RGB[..., ::-1].tobytes()
    d = ffmpeg_encode(bgr, "bgr24", w, h, "bmp", [])
    check("ff_bgr24", pillow_pixels(d, RGB), RGB)
    c.add("ff_bgr24.bmp", d, "Rgb8Srgb", RGB, ["pillow", "ffmpeg"])
    bgra = rgba_cut[..., [2, 1, 0, 3]].tobytes()
    d = ffmpeg_encode(bgra, "bgra", w, h, "bmp", [])
    c.add("ff_bgra.bmp", d, "Rgba8Srgb", rgba_cut, pil_ref("ff_bgra", d, rgba_cut, rgb_only=True))
    rgb8 = (RGB[..., 0] & 0xE0) | ((RGB[..., 1] >> 3) & 0x1C) | (RGB[..., 2] >> 6)
    d = ffmpeg_encode(rgb8.astype(np.uint8).tobytes(), "rgb8", w, h, "bmp", [])
    exp = pillow_pixels(d, RGB)
    check("ff_rgb8", ffmpeg_pixels(d, RGB, "bmp"), exp)
    c.add("ff_rgb8.bmp", d, "Rgb8Srgb", exp, ["pillow", "ffmpeg"])
    d = ffmpeg_encode(GRAY.tobytes(), "gray", w, h, "bmp", [])
    check("ff_gray", pillow_pixels(d, GRAY), GRAY)
    c.add("ff_gray.bmp", d, "R8Unorm", GRAY, ["pillow", "ffmpeg"])
    mono = np.packbits(GRAY > 127, axis=1)
    d = ffmpeg_encode(mono.tobytes(), "monob", w, h, "bmp", [])
    exp = pillow_pixels(d, bw)
    check("ff_monob", exp, bw)
    c.add("ff_monob.bmp", d, "R8Unorm", bw, ["pillow"])
    r5, g6, b5 = RGB[..., 0] >> 3, RGB[..., 1] >> 2, RGB[..., 2] >> 3
    v565 = (r5.astype(np.uint16) << 11) | (g6.astype(np.uint16) << 5) | b5
    d = ffmpeg_encode(v565.astype("<u2").tobytes(), "rgb565le", w, h, "bmp", [])
    exp = np.stack([expand(r5, 5), expand(g6, 6), expand(b5, 5)], -1)
    check("ff_rgb565", pillow_pixels(d, exp), exp, tol=1)
    c.add("ff_rgb565.bmp", d, "Rgb8Srgb", exp, ["pillow"])
    g5 = RGB[..., 1] >> 3
    v555 = (r5.astype(np.uint16) << 10) | (g5.astype(np.uint16) << 5) | b5
    d = ffmpeg_encode(v555.astype("<u2").tobytes(), "rgb555le", w, h, "bmp", [])
    exp555 = np.stack([expand(r5, 5), expand(g5, 5), expand(b5, 5)], -1)
    check("ff_rgb555", pillow_pixels(d, exp555), exp555, tol=1)
    c.add("ff_rgb555.bmp", d, "Rgb8Srgb", exp555, ["pillow"])
    r4, g4, b4 = RGB[..., 0] >> 4, RGB[..., 1] >> 4, RGB[..., 2] >> 4
    v444 = (r4.astype(np.uint16) << 8) | (g4.astype(np.uint16) << 4) | b4
    d = ffmpeg_encode(v444.astype("<u2").tobytes(), "rgb444le", w, h, "bmp", [])
    exp = np.stack([expand(r4, 4), expand(g4, 4), expand(b4, 4)], -1)
    c.add("ff_rgb444.bmp", d, "Rgb8Srgb", exp, pil_ref("ff_rgb444", d, exp, 1))

    # Hand-built palettes: OS/2 1.x (3-byte entries), OS/2 2.x (64 and 16-byte headers),
    # 2-bit and 4-bit BITMAPINFOHEADER.
    idx16, pal16 = quantize(MOS, 16)
    mw, mh = 50, 38
    exp16 = pal16[idx16]
    for name, hsize, bpp, idx_, pal_, entry in (
        ("os2v1_p4", 12, 4, idx16, pal16, 3),
        ("os2v2_p8", 64, 8, idx16, pal16, 4),
        ("info_p4", 40, 4, idx16, pal16, 4),
    ):
        d = bmp_file(mw, mh, bpp, bmp_rows(pack_bits(idx_, bpp)), hsize,
                     palette=pal_bytes(pal_.tolist() + [[0, 0, 0]] * (
                         (1 << bpp if hsize == 12 else len(pal_)) - len(pal_)), entry),
                     colors_used=0 if hsize == 12 else len(pal_))
        check(name, pillow_pixels(d, exp16), exp16)
        c.add(name + ".bmp", d, "Rgb8Srgb", exp16, ["pillow"])
    idx4, pal4 = quantize(MOS, 4)
    exp4 = pal4[idx4]
    d = bmp_file(mw, mh, 2, bmp_rows(pack_bits(idx4, 2)), palette=pal_bytes(pal4),
                 colors_used=4)
    c.add("info_p2.bmp", d, "Rgb8Srgb", exp4, pil_ref("info_p2", d, exp4))
    d = bmp_file(w, h, 24, bmp_rows([r.tobytes() for r in RGB[..., ::-1]]), 12)
    check("os2v1_rgb24", pillow_pixels(d, RGB), RGB)
    c.add("os2v1_rgb24.bmp", d, "Rgb8Srgb", RGB, ["pillow"])
    d = bmp_file(w, h, 24, bmp_rows([r.tobytes() for r in RGB[..., ::-1]]), 16)
    c.add("os2v2_16_rgb24.bmp", d, "Rgb8Srgb", RGB, pil_ref("os2v2_16", d, RGB))
    d = bmp_file(w, h, 24, bmp_rows([r.tobytes() for r in RGB[..., ::-1]], True), top_down=True)
    check("topdown_rgb24", pillow_pixels(d, RGB), RGB)
    c.add("topdown_rgb24.bmp", d, "Rgb8Srgb", RGB, ["pillow"])
    gray_pal = [[i, i, i] for i in range(256)]
    d = bmp_file(w, h, 8, bmp_rows([r.tobytes() for r in GRAY], True), palette=pal_bytes(gray_pal),
                 colors_used=256, top_down=True)
    c.add("topdown_gray8.bmp", d, "R8Unorm", GRAY, pil_ref("topdown_gray8", d, GRAY))

    # RLE8 / RLE4: fully drawn and with delta / early end of line / early end of bitmap.
    idx256, pal256 = quantize(MOS, 256)
    for name, rle4, idx_, pal_ in (("rle8", False, idx256, pal256), ("rle4", True, idx16, pal16)):
        comp, bpp = (2, 4) if rle4 else (1, 8)
        s = rle_encode(idx_, rle4)
        d = bmp_file(mw, mh, bpp, s, comp=comp, palette=pal_bytes(pal_), colors_used=len(pal_))
        exp = pal_[idx_]
        check(name, pillow_pixels(d, exp), exp)
        c.add(name + ".bmp", d, "Rgb8Srgb", exp, ["pillow"])
        s, drawn, idx_exp = rle_with_holes(idx_, rle4)
        d = bmp_file(mw, mh, bpp, s, comp=comp, palette=pal_bytes(pal_), colors_used=len(pal_))
        exp = np.concatenate([pal_[idx_exp], np.where(drawn, 255, 0)[..., None]], -1)
        exp = np.where(drawn[..., None], exp, 0).astype(np.uint8)
        # Pillow rejects an early end of bitmap: check it on the stream with the missing rows
        # painted (index 0, masked out below).
        s2 = s[:-2] + b"\0\0" + (bytes([mw, 0]) + b"\0\0") * 2 + b"\0\1"
        d2 = bmp_file(mw, mh, bpp, s2, comp=comp, palette=pal_bytes(pal_), colors_used=len(pal_))
        got = pillow_pixels(d2, exp[..., :3])
        check(name + "_holes", np.where(drawn[..., None], got, 0), exp[..., :3])
        c.add(name + "_holes.bmp", d, "Rgba8Srgb", exp, ["pillow"])
    # Gray palette with holes -> Rg8Unorm.
    gidx = (MOS[..., 1] // 16).astype(np.uint8)
    gpal = [[i * 17] * 3 for i in range(16)]
    s, drawn, idx_exp = rle_with_holes(gidx, True)
    d = bmp_file(mw, mh, 4, s, comp=2, palette=pal_bytes(gpal), colors_used=16)
    exp = np.stack([np.where(drawn, idx_exp * 17, 0), np.where(drawn, 255, 0)], -1).astype(np.uint8)
    c.add("rle4_gray_holes.bmp", d, "Rg8Unorm", exp, [])

    # 16-bit: BI_RGB 5:5:5, BI_BITFIELDS 5:6:5 (masks after the header), V3 4:4:4:4 with alpha.
    rows = [r.astype("<u2").tobytes() for r in v555]
    d = bmp_file(w, h, 16, bmp_rows(rows))
    check("info_555", pillow_pixels(d, exp555), exp555, tol=1)
    c.add("info_rgb555.bmp", d, "Rgb8Srgb", exp555, ["pillow"])
    rows = [r.astype("<u2").tobytes() for r in v565]
    d = bmp_file(w, h, 16, bmp_rows(rows), comp=3, masks=(0xF800, 0x07E0, 0x001F))
    exp = np.stack([expand(r5, 5), expand(g6, 6), expand(b5, 5)], -1)
    check("bf_565", pillow_pixels(d, exp), exp, tol=1)
    c.add("bf_rgb565.bmp", d, "Rgb8Srgb", exp, ["pillow"])
    a4 = CUT[..., 3] >> 4
    c4 = [CUT[..., k] >> 4 for k in range(3)]
    v4444 = (a4.astype(np.uint16) << 12) | (c4[0].astype(np.uint16) << 8) | \
        (c4[1].astype(np.uint16) << 4) | c4[2]
    rows = [r.astype("<u2").tobytes() for r in v4444]
    d = bmp_file(w, h, 16, bmp_rows(rows), hsize=56, comp=3,
                 masks=(0x0F00, 0x00F0, 0x000F, 0xF000))
    exp = np.stack([expand(c4[0], 4), expand(c4[1], 4), expand(c4[2], 4), expand(a4, 4)], -1)
    c.add("v3_argb4444.bmp", d, "Rgba8Srgb", exp, pil_ref("v3_4444", d, exp, 1))

    # 32-bit: BI_RGB with zero fourth byte (-> RGB), bit fields in R, G, B byte order after a
    # 40-byte header, BI_ALPHABITFIELDS, V2 (no alpha), V4 / V5 with alpha, 10:10:10:2.
    rows = [np.concatenate([r[..., ::-1], np.zeros((w, 1), np.uint8)], -1).tobytes() for r in RGB]
    d = bmp_file(w, h, 32, bmp_rows(rows))
    check("info_bgrx", pillow_pixels(d, RGB), RGB)
    c.add("info_bgrx32.bmp", d, "Rgb8Srgb", RGB, ["pillow"])
    rgbx = [np.concatenate([r, np.full((w, 1), 7, np.uint8)], -1).tobytes() for r in RGB]
    d = bmp_file(w, h, 32, bmp_rows(rgbx), comp=3, masks=(0xFF, 0xFF00, 0xFF0000))
    c.add("bf_rgbx32.bmp", d, "Rgb8Srgb", RGB, pil_ref("bf_rgbx", d, RGB))
    bgra_rows = [r[..., [2, 1, 0, 3]].tobytes() for r in CUT]
    d = bmp_file(w, h, 32, bmp_rows(bgra_rows), comp=6,
                 masks=(0xFF0000, 0xFF00, 0xFF, 0xFF000000))
    c.add("alphabf_bgra32.bmp", d, "Rgba8Srgb", CUT, pil_ref("alphabf", d, CUT))
    d = bmp_file(w, h, 32, bmp_rows(bgra_rows), hsize=52, comp=3,
                 masks=(0xFF0000, 0xFF00, 0xFF))
    c.add("v2_bgrx32.bmp", d, "Rgb8Srgb", CUT[..., :3], pil_ref("v2", d, CUT[..., :3]))
    for name, hsize in (("v4_bgra32", 108), ("v5_bgra32", 124)):
        d = bmp_file(w, h, 32, bmp_rows(bgra_rows), hsize=hsize, comp=3,
                     masks=(0xFF0000, 0xFF00, 0xFF, 0xFF000000))
        check(name, pillow_pixels(d, CUT), CUT)
        c.add(name + ".bmp", d, "Rgba8Srgb", CUT, ["pillow"])
    c10 = [(RGB[..., k].astype(np.uint32) * 4 + 1) for k in range(3)]  # 10-bit values
    a2 = (GRAY >> 6).astype(np.uint32)
    v = (a2 << 30) | (c10[0] << 20) | (c10[1] << 10) | c10[2]
    rows = [r.astype("<u4").tobytes() for r in v]
    d = bmp_file(w, h, 32, bmp_rows(rows), hsize=124, comp=3,
                 masks=(0x3FF00000, 0x000FFC00, 0x000003FF, 0xC0000000))
    exp = np.stack([expand(c10[0], 10), expand(c10[1], 10), expand(c10[2], 10),
                    expand(a2, 2)], -1)
    c.add("v5_a2rgb10.bmp", d, "Rgba8Srgb", exp, pil_ref("v5_a2rgb10", d, exp))
    c.finish()


# ---------------------------------------------------------------------------------------------
# TGA

def tga_rle(pixels, pb):
    """RLE packet stream of a flat pixel byte string (packets may cross scanlines)."""
    px = [pixels[i:i + pb] for i in range(0, len(pixels), pb)]
    out, i, n = bytearray(), 0, len(px)
    while i < n:
        j = i + 1
        while j < n and j - i < 128 and px[j] == px[i]:
            j += 1
        if j - i >= 2:
            out += bytes([0x80 | (j - i - 1)]) + px[i]
            i = j
            continue
        j = i + 1
        while j < n and j - i < 128 and not (j + 1 < n and px[j] == px[j + 1]):
            j += 1
        out += bytes([j - i - 1]) + b"".join(px[i:j])
        i = j
    return bytes(out)


def tga_file(img_type, w, h, bpp, desc, pixels, cmap=None, image_id=b"", ext_attr=None,
             footer=False):
    """cmap: (first, bits, entry bytes) or None. pixels: stored pixel bytes (already in
    storage order); RLE-compressed here for types 9..11."""
    first, bits, entries = cmap if cmap else (0, 0, b"")
    eb = (bits + 7) // 8
    hdr = struct.pack("<BBBHHBHHHHBB", len(image_id), 1 if cmap else 0, img_type, first,
                      len(entries) // eb if cmap else 0, bits, 0, 0, w, h, bpp, desc)
    body = tga_rle(pixels, (bpp + 7) // 8) if img_type >= 9 else pixels
    d = hdr + image_id + entries + body
    if ext_attr is not None:
        ext_off = len(d)
        ext = bytearray(495)
        ext[0:2] = struct.pack("<H", 495)
        ext[494] = ext_attr
        d += bytes(ext)
        d += struct.pack("<II", ext_off, 0) + b"TRUEVISION-XFILE.\0"
    elif footer:
        d += b"\0" * 8 + b"TRUEVISION-XFILE.\0"
    return d


def orient(a, desc):
    """Stored pixel order of top-left image `a` for descriptor origin bits."""
    if not desc & 0x20:
        a = a[::-1]
    if desc & 0x10:
        a = a[:, ::-1]
    return np.ascontiguousarray(a)


def gen_tga():
    c = Cases("tga")
    w, h = 37, 29

    # Pillow: gray, gray + alpha, palette, RGB, RGBA; raw and RLE; both origins.
    idx, pal = quantize(RGB, 256)
    p = Image.fromarray(idx, "P")
    p.putpalette(pal.flatten().tolist())
    la = np.stack([GRAY, CUT[..., 3]], -1)
    for name, img, exp in (
        ("l8", Image.fromarray(GRAY, "L"), GRAY),
        ("la16", Image.fromarray(la, "LA"), la),
        ("p8", p, pal[idx]),
        ("rgb24", Image.fromarray(RGB, "RGB"), RGB),
        ("rgba32", Image.fromarray(CUT, "RGBA"), CUT),
    ):
        for comp in (None, "tga_rle"):
            for orientation in (-1, 1):
                kw = {"orientation": orientation}
                if comp:
                    kw["compression"] = comp
                d = pillow_encode(img, "tga", **kw)
                top = "_top" if d[17] & 0x20 else "_bottom"
                fname = f"pil_{name}{'_rle' if comp else ''}{top}"
                check(fname, pillow_pixels(d, exp), exp)
                c.add(fname + ".tga", d, fmt_of(exp), exp, ["pillow"])

    # ffmpeg: bgr24, bgra, rgb555, gray; raw and RLE.
    r5, g5, b5 = RGB[..., 0] >> 3, RGB[..., 1] >> 3, RGB[..., 2] >> 3
    v555 = (r5.astype(np.uint16) << 10) | (g5.astype(np.uint16) << 5) | b5
    exp555 = np.stack([expand(r5, 5), expand(g5, 5), expand(b5, 5)], -1)
    for rle in (0, 1):
        sfx = "_rle" if rle else ""
        args = ["-rle", str(rle)]
        d = ffmpeg_encode(RGB[..., ::-1].tobytes(), "bgr24", w, h, "tga", args)
        check("ff_bgr24", pillow_pixels(d, RGB), RGB)
        check("ff_bgr24", ffmpeg_pixels(d, RGB, "tga"), RGB)
        c.add(f"ff_bgr24{sfx}.tga", d, "Rgb8Srgb", RGB, ["pillow", "ffmpeg"])
        d = ffmpeg_encode(CUT[..., [2, 1, 0, 3]].tobytes(), "bgra", w, h, "tga", args)
        check("ff_bgra", pillow_pixels(d, CUT), CUT)
        check("ff_bgra", ffmpeg_pixels(d, CUT, "tga"), CUT)
        c.add(f"ff_bgra{sfx}.tga", d, "Rgba8Srgb", CUT, ["pillow", "ffmpeg"])
        d = ffmpeg_encode(v555.astype("<u2").tobytes(), "rgb555le", w, h, "tga", args)
        check("ff_rgb555", pillow_pixels(d, exp555), exp555, tol=1)
        c.add(f"ff_rgb555{sfx}.tga", d, "Rgb8Srgb", exp555, ["pillow"])
        d = ffmpeg_encode(GRAY.tobytes(), "gray", w, h, "tga", args)
        check("ff_gray", pillow_pixels(d, GRAY), GRAY)
        c.add(f"ff_gray{sfx}.tga", d, "R8Unorm", GRAY, ["pillow", "ffmpeg"])

    # Hand-built color-mapped: 15 / 16 / 24 / 32-bit maps, map offset, 16-bit indices,
    # raw and RLE, image ID.
    idx16, pal16 = quantize(MOS, 16)
    mw, mh = 50, 38
    pal_a = np.concatenate([pal16, (np.arange(16) * 16 + 15)[:, None]], -1).astype(np.uint8)
    p5 = [(pal16[:, k] >> 3).astype(np.uint16) for k in range(3)]
    abit = (np.arange(16) % 2).astype(np.uint16)
    e15 = (p5[0] << 10) | (p5[1] << 5) | p5[2]
    exp_p5 = np.stack([expand(p5[k], 5) for k in range(3)], -1)
    maps = {
        15: (e15.astype("<u2").tobytes(), exp_p5, 0),
        16: ((e15 | (abit << 15)).astype("<u2").tobytes(),
             np.concatenate([exp_p5, (abit * 255)[:, None]], -1).astype(np.uint8), 1),
        24: (pal16[:, ::-1].tobytes(), pal16, 0),
        32: (pal_a[:, [2, 1, 0, 3]].tobytes(), pal_a, 8),
    }
    for bits, (entries, pexp, abits) in maps.items():
        for rle in (False, True):
            for ibits, first in ((8, 0), (16, 3)):
                stored = orient(idx16.astype(np.uint16) + first, 0x20)
                px = stored.astype(np.uint8 if ibits == 8 else "<u2").tobytes()
                name = f"cmap{bits}_i{ibits}{'_rle' if rle else ''}"
                d = tga_file(9 if rle else 1, mw, mh, ibits, 0x20 | abits, px,
                             cmap=(first, bits, entries),
                             image_id=b"imgio colormapped test" if rle else b"")
                exp = pexp[idx16]
                c.add(name + ".tga", d, fmt_of(exp), exp, pil_ref(name, d, exp, 1, rgb_only=bits == 16))

    # Truecolor 15 / 16-bit (alpha bit), 24-bit in all four origins, RLE across scanlines.
    v16 = v555 | ((CUT[..., 3] > 127).astype(np.uint16) << 15)
    exp16 = np.concatenate([exp555, np.where(CUT[..., 3:] > 127, 255, 0)], -1).astype(np.uint8)
    d = tga_file(2, w, h, 15, 0x20, v555.astype("<u2").tobytes())
    c.add("tc15.tga", d, "Rgb8Srgb", exp555, pil_ref("tc15", d, exp555, 1))
    d = tga_file(10, w, h, 16, 0x01, orient(v16, 0x01).astype("<u2").tobytes())
    c.add("tc16_a1_rle_bottom.tga", d, "Rgba8Srgb", exp16, pil_ref("tc16", d, exp16, 1, rgb_only=True))
    for desc, oname in ((0x00, "bl"), (0x10, "br"), (0x20, "tl"), (0x30, "tr")):
        for rle in (False, True):
            px = orient(RGB[..., ::-1], desc).tobytes()
            name = f"rgb24_{oname}{'_rle' if rle else ''}"
            d = tga_file(10 if rle else 2, w, h, 24, desc, px, image_id=b"origin " + oname.encode())
            refs = []
            if not desc & 0x10:  # Pillow ignores right-to-left
                check(name, pillow_pixels(d, RGB), RGB)
                refs = ["pillow"]
            c.add(name + ".tga", d, "Rgb8Srgb", RGB, refs)
    # Gray RLE, bottom-right origin.
    d = tga_file(11, w, h, 8, 0x10, orient(GRAY, 0x10).tobytes())
    c.add("gray8_rle_br.tga", d, "R8Unorm", GRAY, [])  # Pillow ignores right-to-left

    # Alpha rules: 32-bit with alpha bits 0 (non-zero alpha -> RGBA; all zero -> RGB),
    # extension area attribute type 0 (ignore alpha) and 3 (alpha), 2.0 footer.
    bgra = orient(CUT[..., [2, 1, 0, 3]], 0x20).tobytes()
    d = tga_file(2, w, h, 32, 0x20, bgra)
    c.add("rgba32_abits0.tga", d, "Rgba8Srgb", CUT, pil_ref("abits0", d, CUT))
    bgr0 = orient(np.concatenate([RGB[..., ::-1], np.zeros((h, w, 1), np.uint8)], -1), 0x20)
    d = tga_file(2, w, h, 32, 0x20, bgr0.tobytes(), footer=True)
    c.add("rgbx32_footer.tga", d, "Rgb8Srgb", RGB, pil_ref("rgbx32", d, RGB))
    d = tga_file(10, w, h, 32, 0x28, bgra, ext_attr=0)
    c.add("rgba32_attr0_rle.tga", d, "Rgb8Srgb", CUT[..., :3], pil_ref("attr0", d, CUT[..., :3]))
    d = tga_file(2, w, h, 32, 0x20, bgra, ext_attr=3, image_id=b"ext attr 3")
    c.add("rgba32_attr3.tga", d, "Rgba8Srgb", CUT, pil_ref("attr3", d, CUT))
    c.finish()


# ---------------------------------------------------------------------------------------------
# HDR

def rgbe_of(rgb):
    """imagecodecs (Walter rgbe.c) float -> RGBE bytes, (h, w, 4) uint8."""
    h, w = rgb.shape[:2]
    raw = imagecodecs.rgbe_encode(np.ascontiguousarray(rgb, np.float32), header=False, rle=False)
    return np.frombuffer(raw, np.uint8).reshape(h, w, 4)


def float_of(rgbe):
    """RGBE -> RGBA f32: m * 2^(e - 136), e == 0 -> 0 (ffmpeg / imagecodecs)."""
    m = rgbe[..., :3].astype(np.float32)
    e = rgbe[..., 3].astype(np.int32)
    f = np.where(e[..., None] == 0, np.float32(0),
                 np.ldexp(m, (e - 136)[..., None]).astype(np.float32)).astype(np.float32)
    return np.concatenate([f, np.ones(f.shape[:2] + (1,), np.float32)], -1)


def new_rle(row):
    """Walter RGBE_WriteBytes_RLE of one scanline (w, 4)."""
    out = bytearray([2, 2, len(row) >> 8, len(row) & 255])
    for c in range(4):
        data = bytes(row[:, c])
        n, cur = len(data), 0
        while cur < n:
            beg, run, old = cur, 0, 0
            while run < 4 and beg < n:
                beg += run
                old = run
                run = 1
                while beg + run < n and run < 127 and data[beg] == data[beg + run]:
                    run += 1
            if old > 1 and old == beg - cur:
                out += bytes([128 + old, data[cur]])
                cur = beg
            while cur < beg:
                k = min(beg - cur, 128)
                out += bytes([k]) + data[cur:cur + k]
                cur += k
            if run >= 4:
                out += bytes([128 + run, data[beg]])
                cur += run
    return bytes(out)


def old_rle(row):
    """Old-style RLE of one scanline: a pixel, then `1 1 1 n` repeats (n shifted by 8 per
    consecutive record)."""
    out, i, n = bytearray(), 0, len(row)
    while i < n:
        out += bytes(row[i])
        j = i + 1
        while j < n and bytes(row[j]) == bytes(row[i]):
            j += 1
        k = j - i - 1
        shift = 0
        while k > 0:
            out += bytes([1, 1, 1, k & 255])
            k >>= 8
            shift += 8
        i = j
    return bytes(out)


def hdr_file(rgbe, res="-Y", minor="+X", mode="rle", header=b"#?RADIANCE\n",
             extra=b"FORMAT=32-bit_rle_rgbe\n"):
    """Radiance file of top-left image `rgbe` with resolution axes (major, minor)."""
    h, w = rgbe.shape[:2]
    a = rgbe
    # Scanline order: rows (Y major) or columns (X major).
    if res[1] == "X":
        a = a.transpose(1, 0, 2)  # a[x][y]
        if res[0] == "-":
            a = a[::-1]
        if minor[0] == "+":  # +Y: bottom to top
            a = a[:, ::-1]
        dims = f"{res} {w} {minor} {h}"
    else:
        if res[0] == "+":
            a = a[::-1]
        if minor[0] == "-":
            a = a[:, ::-1]
        dims = f"{res} {h} {minor} {w}"
    lines = [new_rle(r) if mode == "rle" else old_rle(r) if mode == "old" else r.tobytes()
             for r in a]
    return header + extra + b"\n" + dims.encode() + b"\n" + b"".join(lines)


def ic_decode_top_left(d):
    """imagecodecs decode (stored order) reoriented to top-left, RGBA f32."""
    info = imagecodecs.rgbe_info(d)
    a = imagecodecs.rgbe_decode(d)
    return info, np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], -1)


def gen_hdr():
    c = Cases("hdr")
    h, w = HDR.shape[:2]
    rgbe = rgbe_of(HDR)
    exp = float_of(rgbe)

    # ffmpeg (from float input) and imagecodecs (RLE / flat).
    g = np.stack([HDR[..., 1], HDR[..., 2], HDR[..., 0]]).astype("<f4").tobytes()
    d = ffmpeg_encode(g, "gbrpf32le", w, h, "hdr", [])
    check("ff_rle", ffmpeg_pixels(d, exp, "hdr"), exp)
    check("ff_rle ic", ic_decode_top_left(d)[1], exp)
    c.add("ff_rle.hdr", d, "Rgba32Float", exp, ["ffmpeg", "imagecodecs"])
    for rle in (True, False):
        d = imagecodecs.rgbe_encode(HDR, header=True, rle=rle)
        check("ic", ic_decode_top_left(d)[1], exp)
        check("ic ff", ffmpeg_pixels(d, exp, "hdr"), exp)
        c.add(f"ic_{'rle' if rle else 'flat'}.hdr", d, "Rgba32Float", exp,
              ["imagecodecs", "ffmpeg"])

    # Hand-built: flat with EXPOSURE / #?RGBE / extra variables, old-style RLE (a blocky image
    # with long runs, incl. > 255 for the shifted count), every resolution string, narrow
    # (width < 8: always flat) and XYZE.
    d = hdr_file(rgbe, mode="flat", header=b"#?RGBE\n",
                 extra=b"# made by imgio\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=2.0\n"
                       b"GAMMA=1.0\nEXPOSURE=0.5\nPRIMARIES=0.640 0.330 0.290 0.600 0.150 "
                       b"0.060 0.333 0.333\n")
    check("flat_exposure", ic_decode_top_left(d)[1], exp)
    c.add("flat_exposure.hdr", d, "Rgba32Float", exp, ["imagecodecs"])
    blocky = np.repeat(np.repeat(rgbe[::4, ::4], 4, 0), 4, 1)[:6]
    wide = np.repeat(blocky, 8, 1)[:, :300]  # 300 wide: runs of 32, and > 255 below
    wide[3, 10:290] = wide[3, 10]
    d = hdr_file(wide, mode="old")
    # No reference: imagecodecs (Walter rgbe.c) has no old-style RLE, and ffmpeg 6.1 counts
    # each repeated pixel as 4 (`w -= 4` in hdrdec.c decompress) and ends runs early. Expected
    # pixels follow Radiance's oldreadcolrs (one pixel per repeat).
    c.add("old_rle.hdr", d, "Rgba32Float", float_of(wide), [])
    small = rgbe[3:12, 5:18]  # 13x9
    for major, minor in (("-Y", "+X"), ("-Y", "-X"), ("+Y", "+X"), ("+Y", "-X"),
                         ("+X", "-Y"), ("+X", "+Y"), ("-X", "-Y"), ("-X", "+Y")):
        name = f"orient_{major[0]}{major[1]}{minor[0]}{minor[1]}".replace("-", "m").replace(
            "+", "p")
        mode = "rle" if major[1] == "Y" else "flat"  # X-major scanlines are 9 long
        d = hdr_file(small, major, minor, mode=mode)
        refs = []
        if (major, minor) == ("-Y", "+X"):
            check(name, ic_decode_top_left(d)[1], float_of(small))
            refs = ["imagecodecs"]
        c.add(name + ".hdr", d, "Rgba32Float", float_of(small), refs)
    tiny = rgbe[:4, :5]
    d = hdr_file(tiny, mode="flat")
    check("narrow", ic_decode_top_left(d)[1], float_of(tiny))
    c.add("narrow_5x4.hdr", d, "Rgba32Float", float_of(tiny), ["imagecodecs"])
    d = hdr_file(small, extra=b"FORMAT=32-bit_rle_xyze\n")
    c.add("xyze.hdr", d, "Rgba32Float", float_of(small), [])
    c.finish()


# ---------------------------------------------------------------------------------------------
# Encoder output check (imgio -> reference decoders)

def check_encoded():
    env = dict(os.environ, IMGIO_WRITE_ENCODED="1")
    for t in ("file_bmp", "file_tga", "file_hdr"):
        subprocess.run(["cargo", "test", "--release", "--test", t, "encoded"], cwd=ROOT,
                       env=env, check=True)
    mosa = MOSA
    srcs = {"rgb": RGB, "rgba": mosa, "gray": GRAY}
    for fmt in ("bmp", "tga"):
        d = os.path.join(OUT, fmt, "enc")
        for f in sorted(os.listdir(d)):
            data = open(os.path.join(d, f), "rb").read()
            exp = srcs[f.split(".")[0].split("_")[0]]
            check(f"{fmt}/enc/{f} pillow", pillow_pixels(data, exp), exp)
            check(f"{fmt}/enc/{f} ffmpeg", ffmpeg_pixels(data, exp, fmt), exp)
            print(f"ok {fmt}/enc/{f}: Pillow and ffmpeg decode == source")
    d = os.path.join(OUT, "hdr", "enc")
    ref = imagecodecs.rgbe_encode(HDR, header=True, rle=True)
    exp = float_of(rgbe_of(HDR))
    for f in sorted(os.listdir(d)):
        data = open(os.path.join(d, f), "rb").read()
        check(f"hdr/enc/{f} ffmpeg", ffmpeg_pixels(data, exp, "hdr"), exp)
        check(f"hdr/enc/{f} imagecodecs", ic_decode_top_left(data)[1], exp)
        if f == "hdr_rle.hdr":
            body = data[data.index(b"+X 40\n") + 6:]
            ref_body = ref[ref.index(b"+X 40\n") + 6:]
            assert body == ref_body, "RLE scanlines differ from imagecodecs rgbe_encode"
            print("ok hdr/enc/hdr_rle.hdr: scanlines byte-identical to imagecodecs rgbe_encode")
        print(f"ok hdr/enc/{f}: ffmpeg and imagecodecs decode == imagecodecs rgbe of source")


def main():
    gen_bmp()
    gen_tga()
    gen_hdr()
    if "--skip-encoded" not in sys.argv:
        check_encoded()
    print("done")


if __name__ == "__main__":
    main()
