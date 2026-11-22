#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates testdata/file/{webp,jpeg,avif}: test files made by the offline reference encoders
and the reference decoders' output (never linked into imgio).

Tools: Pillow 13 (libwebp 1.6, libjpeg-turbo 3.2, libavif), imagecodecs (libwebp, libjpeg-turbo,
libavif 1.4 + aom / dav1d), ffmpeg (libaom-av1 encoder, libdav1d decoder) and numpy.

Usage:
  python scripts/gen_file_testdata_webjpgavif.py            # reference files
  python scripts/gen_file_testdata_webjpgavif.py --encoded  # verify + store imgio encoder output

`--encoded` runs the ignored `dump_encoded` tests of tests/file_webp.rs and tests/file_avif.rs
(they write imgio's encoder output to $IMGIO_DUMP_DIR), decodes the files with Pillow /
imagecodecs, checks them (WebP: exact, AVIF: PSNR) and stores them as `enc_*` with the reference
decode, so the Rust tests can check that imgio still writes the same verified bytes.

Each directory has a `manifest.txt`: `<file> <width> <height> <channels> <bits> [key=value...]`
and the reference decodes as raw files: `<stem>.ref` (8-bit: bytes, >8-bit: u16 little endian,
interleaved channels; animations: frames concatenated), `<stem>.yuv` (AVIF: dav1d Y, U, V planes).
"""
import io
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import imagecodecs
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "testdata", "images", "raw")
OUT = os.path.join(ROOT, "testdata", "file")

SOURCES = {
    "rgb_37x29": (37, 29),
    "rgba_64x48": (64, 48),
    "cutout_rgba_96x64": (96, 64),
    "mosaic_rgb_50x38": (50, 38),
    "mosaic_rgba_50x38": (50, 38),
}


def src(name):
    w, h = SOURCES[name]
    return np.fromfile(os.path.join(RAW, name + ".rgba"), np.uint8).reshape(h, w, 4)


class Manifest:
    def __init__(self, fmt):
        self.dir = os.path.join(OUT, fmt)
        os.makedirs(self.dir, exist_ok=True)
        self.lines = []

    def file(self, name, data):
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(data)

    def ref(self, name, arr):
        stem = name.rsplit(".", 1)[0]
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.uint16:
            arr = arr.astype("<u2")
        self.file(stem + ".ref", arr.tobytes())

    def add(self, name, w, h, ch, bits, **kv):
        extra = "".join(f" {k}={v}" for k, v in kv.items())
        self.lines.append(f"{name} {w} {h} {ch} {bits}{extra}")

    def write(self, header):
        with open(os.path.join(self.dir, "manifest.txt"), "w") as f:
            f.write("# " + header + "\n")
            f.write("# file width height channels bits [key=value...]\n")
            f.write("\n".join(self.lines) + "\n")


# ---------------------------------------------------------------------------------------- WebP


def riff_chunks(data):
    """(fourcc, payload) of a RIFF WEBP file."""
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    pos, out = 12, []
    while pos + 8 <= len(data):
        cc, n = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        out.append((cc, data[pos + 8 : pos + 8 + n]))
        pos += 8 + n + (n & 1)
    return out


def chunk(cc, payload):
    return cc + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")


def riff(chunks):
    body = b"WEBP" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def u24(v):
    return struct.pack("<I", v)[:3]


def vp8x(flags, w, h):
    return chunk(b"VP8X", bytes([flags, 0, 0, 0]) + u24(w - 1) + u24(h - 1))


def pil_webp(arr, **kw):
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    b = io.BytesIO()
    Image.fromarray(arr, mode).save(b, "WEBP", **kw)
    return b.getvalue()


def lossless_chunk(arr):
    """VP8L chunk payload of `arr` (RGBA) encoded by libwebp."""
    chunks = dict(riff_chunks(pil_webp(arr, lossless=True, exact=True)))
    return chunks[b"VP8L"]


def alpha_filter(a, method):
    """WebP ALPH prediction filter (residuals), per the container spec."""
    a = a.astype(np.int32)
    h, w = a.shape
    pred = np.zeros_like(a)
    for y in range(h):
        for x in range(w):
            if x == 0 and y == 0:
                p = 0
            elif y == 0:
                p = a[0, x - 1]
            elif x == 0:
                p = a[y - 1, 0]
            elif method == 1:
                p = a[y, x - 1]
            elif method == 2:
                p = a[y - 1, x]
            else:
                p = min(max(a[y, x - 1] + a[y - 1, x] - a[y - 1, x - 1], 0), 255)
            pred[y, x] = p
    if method == 0:
        pred[:] = 0
    return ((a - pred) & 255).astype(np.uint8)


def alph_payload(alpha, method, compressed):
    res = alpha_filter(alpha, method)
    if compressed:
        h, w = res.shape
        img = np.zeros((h, w, 4), np.uint8)
        img[..., 1] = res
        img[..., 3] = 255
        data = lossless_chunk(img)[5:]  # ALPH holds the image stream without the VP8L header
    else:
        data = res.tobytes()
    return bytes([(method << 2) | int(compressed)]) + data


def webp_ref_still(data):
    """libwebp decode (simple API, fancy upsampling)."""
    return imagecodecs.webp_decode(data)


def pil_frames(data):
    im = Image.open(io.BytesIO(data))
    frames, delays = [], []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append(np.asarray(im.convert("RGBA")))
        delays.append(im.info.get("duration", 0))
    return np.stack(frames), delays, im.info.get("loop", 0)


def anim_frames(w, h, n):
    """Synthetic RGBA animation: a gradient with a moving square and a fading disc on a
    partly transparent background."""
    y, x = np.mgrid[0:h, 0:w]
    frames = []
    for i in range(n):
        f = np.zeros((h, w, 4), np.uint8)
        f[..., 0] = (x * 255 // max(w - 1, 1)).astype(np.uint8)
        f[..., 1] = (y * 255 // max(h - 1, 1)).astype(np.uint8)
        f[..., 2] = 90
        f[..., 3] = np.where(x < w // 3, 0, np.where(x < 2 * w // 3, 128, 255))
        sx, sy = 2 + 5 * i, 3 + 3 * i
        f[sy : sy + 10, sx : sx + 10] = [250, 40, 30, 255]
        disc = (x - w // 2) ** 2 + (y - h // 2) ** 2 < 64
        f[disc] = [20, 200, 240, 60 + 30 * i]
        frames.append(f)
    return frames


def gen_webp():
    m = Manifest("webp")
    # Lossless: decodes must equal the source exactly (no .ref; the test reads testdata/images).
    m.file("lossless_rgb.webp", pil_webp(src("mosaic_rgb_50x38")[..., :3], lossless=True))
    m.add("lossless_rgb.webp", 50, 38, 3, 8, source="mosaic_rgb_50x38")
    m.file("lossless_rgba.webp", pil_webp(src("cutout_rgba_96x64"), lossless=True, exact=True))
    m.add("lossless_rgba.webp", 96, 64, 4, 8, source="cutout_rgba_96x64")

    # Lossy stills: reference = libwebp's decode.
    stills = [
        ("lossy_rgb.webp", src("rgb_37x29")[..., :3], dict(quality=75)),
        ("lossy_rgb_q30.webp", src("mosaic_rgb_50x38")[..., :3], dict(quality=30, method=6)),
        ("lossy_rgba.webp", src("rgba_64x48"), dict(quality=70)),
        ("lossy_rgba_aq50.webp", src("cutout_rgba_96x64"), dict(quality=60, alpha_quality=50, method=6)),
    ]
    for name, arr, kw in stills:
        data = pil_webp(arr, **kw)
        ref = webp_ref_still(data)
        pil = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA" if arr.shape[-1] == 4 else "RGB"))
        assert (pil == ref).all(), name
        m.file(name, data)
        m.ref(name, ref)
        m.add(name, arr.shape[1], arr.shape[0], arr.shape[2], 8)

    # ALPH chunk variants (filters 0..3, raw / lossless-compressed) around one libwebp VP8 frame.
    a = src("rgba_64x48")
    vp8 = dict(riff_chunks(pil_webp(np.ascontiguousarray(a[..., :3]), quality=80)))[b"VP8 "]
    for method in range(4):
        for comp in (0, 1):
            name = f"alpha_f{method}_c{comp}.webp"
            data = riff([vp8x(0x10, 64, 48), chunk(b"ALPH", alph_payload(a[..., 3], method, comp)), chunk(b"VP8 ", vp8)])
            ref = webp_ref_still(data)
            assert (ref[..., 3] == a[..., 3]).all(), name
            m.file(name, data)
            m.ref(name, ref)
            m.add(name, 64, 48, 4, 8)

    # Animations: reference = Pillow's composited frames (libwebp WebPAnimDecoder).
    frames = anim_frames(48, 40, 6)
    durs = [100, 40, 250, 70, 70, 500]
    for name, kw in (
        ("anim_lossless.webp", dict(lossless=True)),
        ("anim_lossy.webp", dict(quality=60, allow_mixed=True)),
    ):
        b = io.BytesIO()
        ims = [Image.fromarray(f, "RGBA") for f in frames]
        ims[0].save(b, "WEBP", save_all=True, append_images=ims[1:], duration=durs, loop=2, **kw)
        data = b.getvalue()
        ref, delays, loops = pil_frames(data)
        assert delays == durs and loops == 2, (delays, loops)
        m.file(name, data)
        m.ref(name, ref)
        m.add(name, 48, 40, 4, 8, frames=len(ref), loops=loops, delays=",".join(map(str, delays)))

    # Hand-assembled animation: offsets, blend / no-blend, dispose none / background, a VP8+ALPH
    # and an opaque VP8 frame, a non-zero ANIM background colour (WebPAnimDecoder ignores it).
    cw, chh = 40, 32
    rng = np.random.default_rng(7)

    def rgba(w, h, alpha):
        f = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
        f[..., 3] = alpha(w, h)
        return f

    full_a = lambda w, h: np.where(np.arange(w)[None, :] < w // 4, 255, 160).repeat(h, 0).astype(np.uint8)
    mixed_a = lambda w, h: rng.choice(np.array([0, 40, 128, 200, 255], np.uint8), (h, w))
    opaque = lambda w, h: np.full((h, w), 255, np.uint8)

    def vp8l_frame(f):
        return chunk(b"VP8L", lossless_chunk(f))

    def lossy_alpha_frame(f):
        c = dict(riff_chunks(pil_webp(f, quality=85)))
        return chunk(b"ALPH", c[b"ALPH"]) + chunk(b"VP8 ", c[b"VP8 "])

    def lossy_frame(f):
        return chunk(b"VP8 ", dict(riff_chunks(pil_webp(np.ascontiguousarray(f[..., :3]), quality=85)))[b"VP8 "])

    spec = [  # (x, y, w, h, duration, no_blend, dispose_bg, frame bytes)
        (0, 0, cw, chh, 80, False, False, vp8l_frame(rgba(cw, chh, full_a))),
        (8, 6, 16, 12, 60, False, True, vp8l_frame(rgba(16, 12, mixed_a))),
        (14, 10, 20, 16, 60, False, False, vp8l_frame(rgba(20, 16, mixed_a))),
        (2, 2, 12, 12, 30, True, False, vp8l_frame(rgba(12, 12, mixed_a))),
        (20, 14, 16, 16, 30, False, True, lossy_alpha_frame(rgba(16, 16, mixed_a))),
        (4, 4, 18, 10, 120, False, False, lossy_frame(rgba(18, 10, opaque))),
        (0, 0, cw, chh, 90, True, True, vp8l_frame(rgba(cw, chh, mixed_a))),
        (10, 8, 14, 14, 45, False, False, vp8l_frame(rgba(14, 14, mixed_a))),
    ]
    anmf = []
    for x, y, w, h, d, nb, db, payload in spec:
        head = u24(x // 2) + u24(y // 2) + u24(w - 1) + u24(h - 1) + u24(d) + bytes([(nb << 1) | db])
        anmf.append(chunk(b"ANMF", head + payload))
    anim = chunk(b"ANIM", bytes([0x30, 0x60, 0x90, 0xC0]) + struct.pack("<H", 3))
    data = riff([vp8x(0x12, cw, chh), anim] + anmf)
    ref, delays, loops = pil_frames(data)
    assert delays == [s[4] for s in spec] and loops == 3
    m.file("anim_crafted.webp", data)
    m.ref("anim_crafted.webp", ref)
    m.add("anim_crafted.webp", cw, chh, 4, 8, frames=len(ref), loops=loops, delays=",".join(map(str, delays)))

    m.write("WebP: Pillow 13 / libwebp 1.6 files; .ref = libwebp decode (imagecodecs webp_decode, animations: Pillow frames)")


# ---------------------------------------------------------------------------------------- JPEG


def pil_jpeg(arr, mode, **kw):
    b = io.BytesIO()
    Image.fromarray(arr, mode).save(b, "JPEG", **kw)
    return b.getvalue()


def pil_decode(data, mode):
    im = Image.open(io.BytesIO(data))
    im.draft(None, None)
    return np.asarray(im.convert(mode))


def exif_orientation(value, big_endian):
    """Minimal EXIF (TIFF IFD0 with the Orientation tag), without the `Exif\\0\\0` prefix."""
    e = ">" if big_endian else "<"
    head = (b"MM\0*" if big_endian else b"II*\0") + struct.pack(e + "I", 8)
    ifd = struct.pack(e + "H", 2)
    ifd += struct.pack(e + "HHI", 0x010F, 2, 4) + b"imo\0"  # Make, ASCII, inline
    ifd += struct.pack(e + "HHIH", 0x0112, 3, 1, value) + b"\0\0"
    ifd += struct.pack(e + "I", 0)
    return head + ifd


def gen_jpeg():
    m = Manifest("jpeg")
    odd = np.ascontiguousarray(src("rgb_37x29")[..., :3])
    even = np.ascontiguousarray(src("rgba_64x48")[..., :3])
    mosaic = np.ascontiguousarray(src("mosaic_rgb_50x38")[..., :3])

    def add_rgb(name, data, arr):
        m.file(name, data)
        m.ref(name, pil_decode(data, "RGB"))
        m.add(name, arr.shape[1], arr.shape[0], 3, 8)

    add_rgb("baseline_444.jpg", pil_jpeg(odd, "RGB", quality=90, subsampling=0), odd)
    add_rgb("baseline_422.jpg", pil_jpeg(odd, "RGB", quality=90, subsampling=1), odd)
    add_rgb("baseline_420.jpg", pil_jpeg(odd, "RGB", quality=90, subsampling=2), odd)
    add_rgb("baseline_420_q40.jpg", pil_jpeg(mosaic, "RGB", quality=40, subsampling=2), mosaic)
    add_rgb("baseline_440.jpg", imagecodecs.jpeg8_encode(odd, level=90, subsampling="440"), odd)
    add_rgb("baseline_411.jpg", imagecodecs.jpeg8_encode(even, level=90, subsampling="411"), even)
    add_rgb("progressive_420.jpg", pil_jpeg(even, "RGB", quality=85, progressive=True), even)
    add_rgb("progressive_444.jpg", pil_jpeg(odd, "RGB", quality=85, subsampling=0, progressive=True), odd)
    add_rgb("restart_420.jpg", pil_jpeg(even, "RGB", quality=85, restart_marker_blocks=3), even)
    add_rgb("restart_rows_prog.jpg", pil_jpeg(mosaic, "RGB", quality=85, progressive=True, restart_marker_rows=1), mosaic)
    add_rgb("optimized_422.jpg", pil_jpeg(mosaic, "RGB", quality=75, subsampling=1, optimize=True), mosaic)

    gray = np.ascontiguousarray(even[..., 1])
    for name, data in (
        ("gray.jpg", pil_jpeg(gray, "L", quality=90)),
        ("gray_progressive.jpg", pil_jpeg(np.ascontiguousarray(odd[..., 0]), "L", quality=80, progressive=True)),
    ):
        m.file(name, data)
        m.ref(name, pil_decode(data, "L"))
        h, w = pil_decode(data, "L").shape
        m.add(name, w, h, 1, 8)

    # CMYK (Adobe APP14 transform 0, inverted) and YCCK (transform 2); the reference is
    # libjpeg-turbo's CMYK output converted by Pillow (R = 255 - C - K + C*K/255, ...).
    a = src("rgba_64x48")
    cmyk = np.stack([255 - a[..., 0], 255 - a[..., 1], 255 - a[..., 2], a[..., 3] // 3], -1).astype(np.uint8)
    for name, data in (
        ("cmyk_adobe.jpg", pil_jpeg(cmyk, "CMYK", quality=90)),
        ("ycck_adobe.jpg", imagecodecs.jpeg8_encode(cmyk, level=90, colorspace="CMYK", outcolorspace="YCCK")),
    ):
        im = Image.open(io.BytesIO(data))
        assert im.mode == "CMYK", (name, im.mode)
        m.file(name, data)
        m.ref(name, np.asarray(im.convert("RGB")))
        m.add(name, 64, 48, 3, 8, cmyk=1)

    # EXIF orientation (metadata only, pixels are not rotated).
    for name, val, be in (("exif_orient6.jpg", 6, False), ("exif_orient3_be.jpg", 3, True)):
        data = pil_jpeg(odd, "RGB", quality=90, exif=b"Exif\0\0" + exif_orientation(val, be))
        m.file(name, data)
        m.ref(name, pil_decode(data, "RGB"))
        m.add(name, 37, 29, 3, 8, orientation=val)

    # Lossless (SOF3): exact; reference = libjpeg-turbo (imagecodecs jpeg8_decode).
    g16 = (src("rgba_64x48")[..., 0].astype(np.uint32) * 257 ^ src("rgba_64x48")[..., 2]).astype(np.uint16)
    for name, arr, kw in (
        ("lossless_gray8.jpg", gray, dict(predictor=1)),
        ("lossless_rgb8.jpg", odd, dict(predictor=7)),
        ("lossless_gray12.jpg", (g16 >> 4).astype(np.uint16), dict(bitspersample=12, predictor=4)),
        ("lossless_gray16.jpg", g16, dict(bitspersample=16, predictor=6)),
    ):
        data = imagecodecs.jpeg8_encode(arr, lossless=True, **kw)
        ref = imagecodecs.jpeg8_decode(data)
        assert (ref == arr).all(), name
        bits = kw.get("bitspersample", 8)
        m.file(name, data)
        m.ref(name, ref)
        h, w = arr.shape[:2]
        m.add(name, w, h, 1 if arr.ndim == 2 else 3, bits)

    # Unsupported (no pure Rust decoder): 12-bit DCT.
    d12 = (even[..., 0].astype(np.uint16) * 16).astype(np.uint16)
    m.file("twelve_bit.jpg", imagecodecs.jpeg8_encode(d12, level=90, bitspersample=12))
    m.add("twelve_bit.jpg", 64, 48, 1, 12, unsupported=1)

    m.write("JPEG: Pillow 13 / imagecodecs (libjpeg-turbo 3.2) files; .ref = libjpeg-turbo decode via Pillow (islow IDCT, fancy upsampling)")


# ---------------------------------------------------------------------------------------- AVIF

AV = imagecodecs.AVIF
PIXFMT = {"420": "yuv420p", "422": "yuv422p", "444": "yuv444p", "400": "gray"}


def dav1d_planes(data, sub, bits):
    """Y, U, V planes decoded by dav1d (ffmpeg libdav1d), raw (u16 le above 8 bits)."""
    pix = PIXFMT[sub] + (f"{bits}le" if bits > 8 else "")
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "in.avif")
        with open(p, "wb") as f:
            f.write(data)
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-c:v", "libdav1d", "-i", p, "-f", "rawvideo", "-pix_fmt", pix, "-"],
            check=True, capture_output=True,
        ).stdout
    return out


def ffmpeg_avif(arr, args):
    h, w = arr.shape[:2]
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "out.avif")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
             "-c:v", "libaom-av1", "-still-picture", "1", "-cpu-used", "6"] + args + ["-y", p],
            input=np.ascontiguousarray(arr[..., :3]).tobytes(), check=True,
        )
        with open(p, "rb") as f:
            return f.read()


def scale_bits(arr, bits):
    if bits == 8:
        return np.ascontiguousarray(arr)
    return (arr.astype(np.uint32) * ((1 << bits) - 1) // 255).astype(np.uint16)


def gen_avif():
    m = Manifest("avif")
    PF = {"420": AV.PIXEL_FORMAT.YUV420, "422": AV.PIXEL_FORMAT.YUV422, "444": AV.PIXEL_FORMAT.YUV444, "400": AV.PIXEL_FORMAT.YUV400}
    MC = {
        "601": AV.MATRIX_COEFFICIENTS.BT601,
        "709": AV.MATRIX_COEFFICIENTS.BT709,
        "2020": AV.MATRIX_COEFFICIENTS.BT2020_NCL,
        "identity": AV.MATRIX_COEFFICIENTS.IDENTITY,
        "ycgco": AV.MATRIX_COEFFICIENTS.YCGCO,
        "fcc": AV.MATRIX_COEFFICIENTS.FCC,
        "240": AV.MATRIX_COEFFICIENTS.SMPTE240,
    }

    def store(name, data, sub, bits, alpha, range_):
        info = imagecodecs.avif_info(data)
        assert info["bitspersample"] == bits, (name, info)
        ref = imagecodecs.avif_decode(data)
        h, w = ref.shape[:2]
        if ref.ndim == 2:
            ref = ref[..., None]
        if ref.shape[-1] in (1, 2):  # gray (+ alpha) -> RGB(A), as libavif converts it
            g = ref[..., :1].repeat(3, -1)
            ref = np.concatenate([g, ref[..., 1:]], -1)
        ch = ref.shape[-1]
        assert (ch == 4) == alpha, name
        m.file(name, data)
        m.ref(name, ref)
        stem = name.rsplit(".", 1)[0]
        m.file(stem + ".yuv", dav1d_planes(data, sub, bits))
        m.add(name, w, h, ch, bits, sub=sub, range=range_, matrix=str(info["matrix"].value))

    cases = [  # name, source, subsampling, bits, matrix, quality
        ("yuv420_8.avif", "rgb_37x29", "420", 8, "601", 70),
        ("yuv422_8.avif", "rgba_64x48", "422", 8, "601", 80),
        ("yuv444_8.avif", "mosaic_rgb_50x38", "444", 8, "709", 80),
        ("mono_8.avif", "rgba_64x48", "400", 8, "601", 80),
        ("yuv420_10.avif", "rgb_37x29", "420", 10, "601", 80),
        ("yuv422_10.avif", "mosaic_rgb_50x38", "422", 10, "709", 75),
        ("yuv444_12.avif", "rgba_64x48", "444", 12, "2020", 85),
        ("yuv420_12_2020.avif", "rgb_37x29", "420", 12, "2020", 80),
        ("mono_12.avif", "rgb_37x29", "400", 12, "601", 80),
        ("identity_444_8.avif", "mosaic_rgb_50x38", "444", 8, "identity", 90),
        ("ycgco_420_10.avif", "rgb_37x29", "420", 10, "ycgco", 80),
        ("fcc_444_10.avif", "rgba_64x48", "444", 10, "fcc", 80),
        ("smpte240_422_12.avif", "rgb_37x29", "422", 12, "240", 80),
    ]
    for name, s, sub, bits, mc, q in cases:
        arr = scale_bits(src(s)[..., :3], bits)
        data = imagecodecs.avif_encode(arr, q, bitspersample=bits, pixelformat=PF[sub], matrix=MC[mc])
        store(name, data, sub, bits, False, "full")

    # Alpha (auxiliary image).
    for name, s, sub, bits in (
        ("alpha_420_8.avif", "cutout_rgba_96x64", "420", 8),
        ("alpha_444_10.avif", "rgba_64x48", "444", 10),
        ("alpha_mono_8.avif", "mosaic_rgba_50x38", "400", 8),
    ):
        arr = scale_bits(src(s), bits)
        data = imagecodecs.avif_encode(arr, 80, bitspersample=bits, pixelformat=PF[sub])
        store(name, data, sub, bits, True, "full")

    # Limited range (ffmpeg libaom-av1; nclx written by ffmpeg's AVIF muxer).
    for name, s, pix, sub, bits, cs in (
        ("limited_420_8.avif", "rgba_64x48", "yuv420p", "420", 8, "bt709"),
        ("limited_444_10.avif", "rgb_37x29", "yuv444p10le", "444", 10, "bt470bg"),
        ("limited_422_10.avif", "mosaic_rgb_50x38", "yuv422p10le", "422", 10, "bt2020nc"),
    ):
        data = ffmpeg_avif(src(s), ["-crf", "28", "-pix_fmt", pix, "-color_range", "tv", "-colorspace", cs,
                                    "-color_primaries", "bt709", "-color_trc", "iec61966-2-1"])
        store(name, data, sub, bits, False, "limited")

    # Unspecified matrix (2) in the nclx box: libavif uses BT.601.
    data = ffmpeg_avif(src("rgba_64x48"), ["-crf", "20", "-pix_fmt", "yuv444p10le", "-color_range", "pc"])
    store("unspecified_444_10.avif", data, "444", 10, False, "full")

    m.write("AVIF: imagecodecs (libavif 1.4 + aom) and ffmpeg libaom-av1 files; .ref = libavif decode (imagecodecs avif_decode, native depth), .yuv = dav1d planes (ffmpeg libdav1d)")


# ------------------------------------------------------------------------- encoder verification


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def gen_encoded():
    """Runs imgio's encoders (ignored Rust tests), verifies with the reference decoders."""
    with tempfile.TemporaryDirectory() as t:
        env = dict(os.environ, IMGIO_DUMP_DIR=t)
        subprocess.run(
            ["cargo", "test", "--release", "--features", "webp,avif", "--test", "file_webp", "--test", "file_avif",
             "--", "--ignored", "dump_encoded"],
            cwd=ROOT, env=env, check=True,
        )
        for fmt in ("webp", "avif"):
            lines = []
            for name in sorted(os.listdir(t)):
                if not name.endswith("." + fmt):
                    continue
                with open(os.path.join(t, name), "rb") as f:
                    data = f.read()
                source = name[len("enc_") :].rsplit(".", 1)[0]
                s = src(source)
                ch = 4 if "rgba" in source else 3
                s = s[..., :ch]
                if fmt == "webp":
                    ref = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA" if ch == 4 else "RGB"))
                    assert (ref == s).all(), name  # lossless: exact in libwebp
                    lines.append(f"{name} {s.shape[1]} {s.shape[0]} {ch} 8 source={source}")
                else:
                    ref = imagecodecs.avif_decode(data)
                    bits = imagecodecs.avif_info(data)["bitspersample"]
                    r8 = np.round(ref.astype(np.float64) * 255 / ((1 << bits) - 1))
                    mask = s[..., 3] > 0 if ch == 4 else np.ones(s.shape[:2], bool)
                    p = psnr(r8[..., :3][mask], s[..., :3][mask])
                    assert p > 28, (name, p)
                    if ch == 4:
                        assert psnr(r8[..., 3], s[..., 3]) > 35, name
                    if ref.dtype == np.uint16:
                        ref = ref.astype("<u2")
                    lines.append(f"{name} {s.shape[1]} {s.shape[0]} {ch} {bits} source={source} psnr={p:.2f}")
                with open(os.path.join(OUT, fmt, name), "wb") as f:
                    f.write(data)
                stem = name.rsplit(".", 1)[0]
                with open(os.path.join(OUT, fmt, stem + ".ref"), "wb") as f:
                    f.write(np.ascontiguousarray(ref).tobytes())
            with open(os.path.join(OUT, fmt, "encoded.txt"), "w") as f:
                f.write("# imgio encoder output verified by " + ("Pillow (libwebp), exact" if fmt == "webp" else "imagecodecs (libavif + dav1d), PSNR") + "\n")
                f.write("\n".join(lines) + "\n")
                print(fmt, *lines, sep="\n  ")


if __name__ == "__main__":
    if "--encoded" in sys.argv:
        gen_encoded()
    else:
        gen_webp()
        gen_jpeg()
        gen_avif()
