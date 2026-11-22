#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates the ICO / CUR test files (deterministic).

Usage: VENV/bin/python -I scripts/gen_file_testdata_ico.py [--verify]

Needs numpy and Pillow (offline reference tools, never linked). Sources are imgio's own images
(testdata/images/raw). Writes testdata/file/ico/:

- `pil_*.ico`: written by Pillow (PNG entries, and `bitmap_format="bmp"` DIB entries).
- `dib_*.ico` / `*.cur`: hand-built from the format description: 1 / 4 / 8-bit palette
  (color and gray), 24-bit and 32-bit DIBs with AND masks, PNG entries, two entries of one
  size with different depths, a cursor with hotspots (an ICO with type 2).
- `expected/<file>.<index>.rgba`: Pillow's decode of each directory entry (`IcoFile.frame`,
  matched by offset; cursors decoded as the same bytes with type 1 and the bit count of each
  DIB, since Pillow takes the hotspot field for the bit count) as RGBA8.
- `cases.txt`: `file entries largest` then per entry `width height bits hotspot_x hotspot_y
  png`.

`--verify` decodes the files imgio's tests wrote to `enc/` (IMGIO_WRITE_ENCODED=1) with Pillow,
checks them against imgio's own decode of the same file (`enc/<name>.<i>.rgba`) and writes
`encoded_verified.txt` (name, fnv1a64 of the encoded bytes).
"""
import io
import os
import struct
import sys

import numpy as np
from PIL import IcoImagePlugin, Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "testdata/images/raw")
OUT = os.path.join(ROOT, "testdata/file/ico")


def load(name, w, h):
    return np.fromfile(os.path.join(RAW, name), np.uint8).reshape(h, w, 4)


def square(a, s):
    """Nearest-neighbour s x s crop / scale of an RGBA array (deterministic)."""
    h, w = a.shape[:2]
    ys = (np.arange(s) * h) // s
    xs = (np.arange(s) * w) // s
    return a[ys][:, xs].copy()


def and_mask(alpha):
    h, w = alpha.shape
    stride = (w + 31) // 32 * 4
    rows = []
    for y in range(h - 1, -1, -1):
        row = bytearray(stride)
        for x in range(w):
            if alpha[y, x] == 0:
                row[x // 8] |= 0x80 >> (x % 8)
        rows.append(bytes(row))
    return b"".join(rows)


def dib(a, bits, gray=False):
    """BITMAPINFOHEADER DIB (height doubled) + AND mask (alpha 0 = transparent)."""
    h, w = a.shape[:2]
    alpha = a[:, :, 3]
    pal = b""
    if bits <= 8:
        n = 1 << bits
        if gray:
            levels = np.linspace(0, 255, n).round().astype(np.uint8)
            luma = a[:, :, 0]
            idx = np.abs(luma[:, :, None].astype(int) - levels[None, None, :].astype(int)).argmin(2)
            colors = [(int(v), int(v), int(v)) for v in levels]
        else:
            # Palette = the n most frequent colors (others map to the nearest one).
            rgb = a[:, :, :3].reshape(-1, 3)
            uniq, counts = np.unique(rgb, axis=0, return_counts=True)
            order = np.argsort(-counts, kind="stable")
            colors = [tuple(int(c) for c in uniq[i]) for i in order[:n]]
            while len(colors) < n:
                colors.append((0, 0, 0))
            cp = np.array(colors, dtype=int)
            d = ((a[:, :, None, :3].astype(int) - cp[None, None]) ** 2).sum(3)
            idx = d.argmin(2)
        pal = b"".join(bytes([c[2], c[1], c[0], 0]) for c in colors)
        stride = (w * bits + 31) // 32 * 4
        rows = []
        for y in range(h - 1, -1, -1):
            row = bytearray(stride)
            for x in range(w):
                v = int(idx[y, x])
                bitpos = x * bits
                row[bitpos // 8] |= v << (8 - bits - bitpos % 8)
            rows.append(bytes(row))
        px = b"".join(rows)
    elif bits == 24:
        stride = (w * 3 + 3) // 4 * 4
        px = b"".join(
            a[y, :, 2::-1].tobytes() + bytes(stride - w * 3) for y in range(h - 1, -1, -1))
    else:
        bgra = a[:, :, [2, 1, 0, 3]]
        px = b"".join(bgra[y].tobytes() for y in range(h - 1, -1, -1))
    hdr = struct.pack("<IiiHHIIiiII", 40, w, 2 * h, 1, bits, 0, len(px), 0, 0, 0, 0)
    return hdr + pal + px + and_mask(alpha)


def png(a):
    b = io.BytesIO()
    Image.fromarray(a, "RGBA").save(b, "PNG")
    return b.getvalue()


def ico(entries, cursor=False):
    """entries: (body, width, height, colors, a, b) with a/b = planes/bits or hotspot."""
    out = struct.pack("<HHH", 0, 2 if cursor else 1, len(entries))
    off = 6 + 16 * len(entries)
    for body, w, h, colors, a, b in entries:
        out += struct.pack("<BBBBHHII", w % 256, h % 256, colors, 0, a, b, len(body), off)
        off += len(body)
    return out + b"".join(e[0] for e in entries)


def pillow_frames(data):
    """Pillow's decode of each directory entry (in directory order) as RGBA8."""
    # Cursors: type 1 and the planes / bit count fields restored from each DIB header (Pillow
    # reads the hotspot y as the bit count and would ignore 32-bit alpha).
    as_ico = bytearray(data[:2] + b"\x01\x00" + data[4:])
    n = struct.unpack_from("<H", data, 4)[0]
    if data[2] == 2:
        for i in range(n):
            off = struct.unpack_from("<I", data, 6 + 16 * i + 12)[0]
            bits = 32 if data[off:off + 4] == b"\x89PNG" else struct.unpack_from("<H", data, off + 14)[0]
            struct.pack_into("<HH", as_ico, 6 + 16 * i + 4, 1, bits)
    f = IcoImagePlugin.IcoFile(io.BytesIO(bytes(as_ico)))
    offs = [struct.unpack_from("<I", data, 6 + 16 * i + 12)[0] for i in range(n)]
    out = []
    for o in offs:
        k = next(i for i, e in enumerate(f.entry) if e.offset == o)
        out.append(np.asarray(f.frame(k).convert("RGBA")))
    return out


def fnv(b):
    h = 0xCBF29CE484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def verify():
    enc = os.path.join(OUT, "enc")
    lines = []
    for name in sorted(os.listdir(enc)):
        if not (name.endswith(".ico") or name.endswith(".cur")):
            continue
        data = open(os.path.join(enc, name), "rb").read()
        for i, fr in enumerate(pillow_frames(data)):
            mine = np.fromfile(os.path.join(enc, f"{name}.{i}.rgba"), np.uint8).reshape(fr.shape)
            if not np.array_equal(fr, mine):
                sys.exit(f"{name} entry {i}: Pillow decode differs")
        lines.append(f"{name} {fnv(data):016x}")
    with open(os.path.join(OUT, "encoded_verified.txt"), "w") as fh:
        fh.write("# imgio encoder output decoded exactly by Pillow (scripts/gen_file_testdata_ico.py --verify)\n")
        fh.write("\n".join(lines) + "\n")
    print(len(lines), "encoded files verified")


def main():
    if "--verify" in sys.argv:
        verify()
        return
    os.makedirs(os.path.join(OUT, "expected"), exist_ok=True)
    cut = load("cutout_rgba_96x64.rgba", 96, 64)
    rgba = load("rgba_64x48.rgba", 64, 48)
    mosaic = load("mosaic_rgba_50x38.rgba", 50, 38)
    # Binary alpha for the masked depths.
    cut_bin = cut.copy()
    cut_bin[:, :, 3] = np.where(cut[:, :, 3] >= 128, 255, 0)

    files = {}
    s16, s32, s48 = square(cut_bin, 16), square(cut_bin, 32), square(cut_bin, 48)
    files["dib_depths.ico"] = ico([
        (dib(s16, 1), 16, 16, 2, 1, 1),
        (dib(s16, 4), 16, 16, 16, 1, 4),
        (dib(s32, 8), 32, 32, 0, 1, 8),
        (dib(s32, 24), 32, 32, 0, 1, 24),
        (dib(square(cut, 32), 32), 32, 32, 0, 1, 32),
        (dib(s48, 8, gray=True), 48, 48, 0, 1, 8),
    ])
    files["dib_odd.ico"] = ico([
        (dib(cut_bin[:13, :37].copy(), 1), 37, 13, 2, 1, 1),
        (dib(cut_bin[:9, :33].copy(), 24), 33, 9, 0, 1, 24),
        (dib(cut[:7, :35].copy(), 32), 35, 7, 0, 1, 32),
    ])
    big = square(cut, 256)
    files["mixed.ico"] = ico([
        (dib(square(cut, 16), 32), 16, 16, 0, 1, 32),
        (png(square(rgba, 64)), 64, 64, 0, 1, 32),
        (png(big), 256, 256, 0, 1, 32),
    ])
    files["hotspot.cur"] = ico([
        (dib(square(cut, 32), 32), 32, 32, 0, 5, 7),
        (dib(s16, 4), 16, 16, 16, 2, 3),
        (png(square(mosaic, 48)), 48, 48, 0, 23, 40),
    ], cursor=True)
    b = io.BytesIO()
    Image.fromarray(square(mosaic, 128), "RGBA").save(b, "ICO", sizes=[(16, 16), (32, 32), (128, 128)])
    files["pil_png.ico"] = b.getvalue()
    b = io.BytesIO()
    Image.fromarray(square(cut, 64), "RGBA").save(b, "ICO", sizes=[(24, 24), (64, 64)], bitmap_format="bmp")
    files["pil_bmp.ico"] = b.getvalue()

    lines = ["# file entries largest | per entry: width height bits hotspot_x hotspot_y png"]
    for name, data in files.items():
        with open(os.path.join(OUT, name), "wb") as fh:
            fh.write(data)
        frames = pillow_frames(data)
        cursor = struct.unpack_from("<H", data, 2)[0] == 2
        n = len(frames)
        ents = []
        best = None
        for i, fr in enumerate(frames):
            fr.tofile(os.path.join(OUT, "expected", f"{name}.{i}.rgba"))
            w, h, colors, _, a, bb, size, off = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
            w, h = w or 256, h or 256
            ispng = data[off:off + 8] == b"\x89PNG\r\n\x1a\n"
            bits = 0 if cursor else bb
            hx, hy = (a, bb) if cursor else (0, 0)
            assert fr.shape[:2] == (h, w), (name, i)
            ents.append(f"{w} {h} {bits} {hx} {hy} {int(ispng)}")
            depth = bits or ((colors - 1).bit_length() if colors else 256)
            key = (w * h, depth)
            if best is None or key > best[1]:
                best = (i, key)
        lines.append(f"{name} {n} {best[0]} " + " ".join(ents))
    with open(os.path.join(OUT, "cases.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
