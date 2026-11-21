#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates testdata/file/{jxl,jpeg2000,vdb} with offline reference tools (never linked).

Usage: <venv>/bin/python scripts/gen_file_testdata_jxlj2kvdb.py [jxl] [jpeg2000] [vdb]
       (no argument: all three; the venv needs numpy, Pillow 13 and imagecodecs)
       <venv>/bin/python scripts/gen_file_testdata_jxlj2kvdb.py --verify-jxl-encoder DIR

== JPEG XL ==
JPEG XL: libjxl through imagecodecs (`jpegxl_encode` / `jpegxl_decode`). Each `<name>.jxl`
has a `<name>.raw` twin: libjxl's decode of the file in imgio's decoded layout (8-bit
interleaved; 16-bit little endian with RGB padded to RGBA (alpha 65535); float as f32 little
endian, RGB / gray+alpha expanded to RGBA). Lossless files are asserted equal to their
source here; `meta.txt` lists per file what the Rust test checks (exact or a max error).

The `--verify-jxl-encoder` form decodes every `*.jxl` that `tests/file_jxl.rs` wrote to DIR
(run it with `IMGIO_JXL_ENC_DUMP=DIR`) with libjxl and checks it equals the `*.src` pixels next
to it.

== JPEG 2000 ==
Offline reference tools only (never linked into imgio):
- imagecodecs `jpeg2k` (OpenJPEG) for the basic files and every reference decode,
- Pillow 13 (OpenJPEG 2.5.4) for tiles, quality layers, progression orders, image offsets and
  sYCC,
- the OpenJPEG CLI `opj_compress` (env `OPJ_BIN` = directory holding it, default: PATH) for
  sub-sampled components (raw input with per-component sub-sampling); skipped if missing.

Writes `<name>` (the file), `<name>.raw` (expected decoded pixels in imgio's output layout:
8-bit samples, or 16-bit little endian) and `manifest.txt` (one line per decode case:
`file format width height reduce max_diff`, max_diff 0 = exact).

== OpenVDB ==
See `gen_vdb` (C++ OpenVDB 10 via tools/vdb-golden/gen.cpp, env OVDB_PREFIX).
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import imagecodecs
import imagecodecs as ic
import numpy as np
from PIL import Image


# ---------------------------------------------------------------- JPEG XL

def jxl_raw(root, name, w, h):
    a = np.fromfile(root / "testdata/images/raw" / name, dtype=np.uint8)
    return a.reshape(h, w, 4)


def jxl_hdr(root):
    a = np.fromfile(root / "testdata/images/raw/hdr_40x24.rgba16f", dtype="<f2")
    return a.reshape(24, 40, 4)


def jxl_gray(rgb):
    y = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    return np.clip(y + 0.5, 0, 255).astype(np.uint8)


def jxl_to16(a8, salt):
    """8-bit -> 16-bit with real low-byte content."""
    h, w = a8.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    low = ((x * 37 + y * 101 + salt) & 0xFF).astype(np.uint16)
    if a8.ndim == 3:
        low = low[..., None]
    return (a8.astype(np.uint16) << 8) | low


def jxl_layout(a):
    """imgio decoded layout of a libjxl decode (h, w[, c])."""
    if a.ndim == 2:
        a = a[..., None]
    c = a.shape[2]
    if a.dtype == np.uint8:
        return a.tobytes()
    if a.dtype == np.uint16:
        if c == 3:
            a = np.concatenate([a, np.full(a.shape[:2] + (1,), 65535, np.uint16)], -1)
        return a.astype("<u2").tobytes()
    a = a.astype("<f4")
    if c == 2:
        a = np.concatenate([a[..., :1]] * 3 + [a[..., 1:]], -1)
    elif c == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), "<f4")], -1)
    return a.tobytes()


def jxl_srgb16(a):
    v = a.astype(np.float64) / 65535
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    return np.clip(np.round(s * 65535), 0, 65535).astype(np.uint16)


def gen_jxl(root):
    out = root / "testdata/file/jxl"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    rgba = jxl_raw(root, "rgba_64x48.rgba", 64, 48)
    rgb = jxl_raw(root, "rgb_37x29.rgba", 37, 29)[..., :3].copy()
    mos = jxl_raw(root, "mosaic_rgba_50x38.rgba", 50, 38)
    mos_rgb = jxl_raw(root, "mosaic_rgb_50x38.rgba", 50, 38)[..., :3].copy()
    cut = jxl_raw(root, "cutout_rgba_96x64.rgba", 96, 64)[8:56, 16:80].copy()
    hdr = jxl_hdr(root)
    crop = rgba[8:32, 16:48].copy()  # 32x24
    meta = []

    def put(name, data, ref, check, *dec_args, **dec_kw):
        (out / f"{name}.jxl").write_bytes(data)
        dec = ic.jpegxl_decode(data, *dec_args, **dec_kw)
        if check == "exact" and ref is not None:
            assert dec.dtype == ref.dtype and np.array_equal(dec, ref), name
        (out / f"{name}.raw").write_bytes(jxl_layout(dec))
        meta.append(f"{name} {check}")
        return dec

    L = dict(lossless=True, effort=7)
    # Lossless modular, 8-bit.
    put("rgba8", ic.jpegxl_encode(rgba, **L), rgba, "exact")
    put("rgb8_container", ic.jpegxl_encode(rgb, usecontainer=True, **L), rgb, "exact")
    g = jxl_gray(mos_rgb)
    put("gray8", ic.jpegxl_encode(g, **L), g, "exact")
    ga = np.stack([jxl_gray(mos), mos[..., 3]], -1)
    put("graya8", ic.jpegxl_encode(ga, **L), ga, "exact")
    # Lossless modular, 16-bit, sRGB transfer.
    S = dict(transfer="SRGB", **L)
    a = jxl_to16(rgb, 3)
    put("rgb16", ic.jpegxl_encode(a, **S), a, "exact")
    a = jxl_to16(crop, 5)
    put("rgba16_container", ic.jpegxl_encode(a, usecontainer=True, **S), a, "exact")
    a = jxl_to16(jxl_gray(crop), 7)
    put("gray16", ic.jpegxl_encode(a, **S), a, "exact")
    a = jxl_to16(np.stack([jxl_gray(crop), crop[..., 3]], -1), 9)
    put("graya16", ic.jpegxl_encode(a, **S), a, "exact")
    # 16-bit linear transfer, color managed to sRGB by the decoder. libjxl cannot change the
    # color encoding of non-XYB images, so the reference is its (exact) linear decode put
    # through the sRGB curve in float64.
    a = jxl_to16(crop[..., :3].copy(), 11)
    data = ic.jpegxl_encode(a, transfer="LINEAR", **L)
    (out / "rgb16_linear.jxl").write_bytes(data)
    dec = ic.jpegxl_decode(data)
    assert np.array_equal(dec, a)
    (out / "rgb16_linear.raw").write_bytes(jxl_layout(jxl_srgb16(dec)))
    meta.append("rgb16_linear max 8")
    # Lossless float (linear sRGB): HDR values above 1 and float16 / gray.
    a = hdr.astype(np.float32)
    put("rgba_f32", ic.jpegxl_encode(a, **L), a, "exact")
    a = hdr[:16, :24, :3].copy()
    put("rgb_f16", ic.jpegxl_encode(a, **L), a, "exact")
    a = (hdr[:16, :24, 0] * 0.25).astype(np.float32)
    put("gray_f32", ic.jpegxl_encode(a, **L), a, "exact")
    # Lossy VarDCT (XYB): libjxl's decode is the reference, with a bound.
    put("lossy_rgb8", ic.jpegxl_encode(mos_rgb, distance=1.0), None, "max 1")
    put("lossy_rgba8", ic.jpegxl_encode(cut, distance=1.5, usecontainer=True), None, "max 1")
    put("lossy_gray8", ic.jpegxl_encode(g, distance=1.0), None, "max 1")
    a = jxl_to16(rgb, 13)
    put("lossy_rgb16", ic.jpegxl_encode(a, distance=0.5, transfer="SRGB"), None, "max 8")
    # Animation: 3 lossless RGBA frames.
    frames = np.stack([rgba[0:16, 0:24], rgba[16:32, 20:44], rgba[32:48, 40:64]])
    d = put("anim_rgba8", ic.jpegxl_encode(frames, **L), frames, "exact")
    assert d.shape == frames.shape
    (out / "meta.txt").write_text("\n".join(meta) + "\n")


def jxl_verify_encoder(d):
    d = Path(d)
    n = 0
    for f in sorted(d.glob("*.jxl")):
        src = (d / (f.stem + ".src")).read_bytes()
        dec = ic.jpegxl_decode(f.read_bytes())
        got = dec.astype("<u2").tobytes() if dec.dtype == np.uint16 else dec.tobytes()
        assert got == src, f.name
        n += 1
    print(f"libjxl decode of {n} imgio-encoded files: exact")


# ---------------------------------------------------------------- JPEG 2000

def j2k_src(root, name, w, h):
    data = (root / "testdata/images/raw" / f"{name}.rgba").read_bytes()
    return np.frombuffer(data, np.uint8).reshape(h, w, 4)


def j2k_gray(rgb):
    r, g, b = (rgb[..., i].astype(np.uint32) for i in range(3))
    return ((77 * r + 150 * g + 29 * b + 128) >> 8).astype(np.uint8)


def j2k_widen(px8, bits):
    """Deterministic `bits`-bit samples from 8-bit ones (all bits in use)."""
    h, w = px8.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    low = (x * 7 + y * 13) % (1 << (bits - 8))
    if px8.ndim == 3:
        low = low[..., None] + np.arange(px8.shape[2])
        low %= 1 << (bits - 8)
    return ((px8.astype(np.uint16) << (bits - 8)) | low).astype(np.uint16)


def j2k_to_output(px, bits):
    """Reference integer samples (h, w[, c]) -> imgio output bytes and format name."""
    px = np.asarray(px).astype(np.int64)
    if px.ndim == 2:
        px = px[..., None]
    c = px.shape[2]
    m = (1 << bits) - 1
    if bits <= 8:
        out = ((px * 255 + m // 2) // m).astype(np.uint8)
        fmt = {1: "R8Unorm", 2: "Rg8Unorm", 3: "Rgb8Srgb", 4: "Rgba8Srgb"}[c]
    else:
        out = ((px * 65535 + m // 2) // m).astype("<u2")
        if c == 3:
            a = np.full(out.shape[:2] + (1,), 65535, "<u2")
            out = np.concatenate([out, a], 2)
        fmt = {1: "R16Unorm", 2: "Rg16Unorm", 3: "Rgba16Unorm", 4: "Rgba16Unorm"}[c]
    return out.tobytes(), fmt


class J2kWriter:
    def __init__(self, out):
        self.out = out
        self.lines = []

    def add(self, name, data, ref, bits, max_diff=0, reduce=0, write=True):
        if write:
            (self.out / name).write_bytes(data)
        raw, fmt = j2k_to_output(ref, bits)
        h, w = np.asarray(ref).shape[:2]
        suffix = "" if reduce == 0 else f".r{reduce}"
        (self.out / f"{name}{suffix}.raw").write_bytes(raw)
        self.lines.append(f"{name} {fmt} {w} {h} {reduce} {max_diff}")
        print(f"{name}{suffix}: {len(data)} bytes, {fmt} {w}x{h}, max_diff {max_diff}")


def j2k_check_lossless(data, src):
    dec = imagecodecs.jpeg2k_decode(data)
    assert np.array_equal(dec, src), "reference decode is not lossless"


def j2k_pillow(img, **kw):
    buf = io.BytesIO()
    img.save(buf, "JPEG2000", **kw)
    return buf.getvalue()


def j2k_opj(opj, args):
    subprocess.run([str(opj)] + args, check=True, stdout=subprocess.DEVNULL)


def gen_jpeg2000(root):
    out = root / "testdata/file/jpeg2000"
    out.mkdir(parents=True, exist_ok=True)
    wr = J2kWriter(out)
    enc = imagecodecs.jpeg2k_encode
    dec = imagecodecs.jpeg2k_decode

    rgb37 = j2k_src(root, "rgb_37x29", 37, 29)[..., :3]
    mos_rgba = j2k_src(root, "mosaic_rgba_50x38", 50, 38)
    mos_rgb = j2k_src(root, "mosaic_rgb_50x38", 50, 38)[..., :3]
    rgba64 = j2k_src(root, "rgba_64x48", 64, 48)
    cutout = j2k_src(root, "cutout_rgba_96x64", 96, 64)

    # Lossless 5/3: every layout and bit depth (imagecodecs / OpenJPEG).
    lossless = [
        ("gray8_53.j2k", j2k_gray(rgb37), 8, dict(codecformat="J2K")),
        ("gray4_53.jp2", j2k_gray(rgb37) >> 4, 4, dict(bitspersample=4)),
        ("graya8_53.jp2", np.stack([j2k_gray(mos_rgba), mos_rgba[..., 3]], -1), 8,
         dict(colorspace="GRAY")),
        ("rgb8_53_rct.jp2", mos_rgb, 8, dict(mct=True)),
        ("rgb8_53_nomct.j2k", rgb37, 8, dict(codecformat="J2K", mct=False)),
        ("rgba8_53.jp2", rgba64, 8, dict(resolutions=4)),
        ("gray12_53.jp2", j2k_widen(j2k_gray(rgb37), 12), 12, dict(bitspersample=12)),
        ("graya16_53.j2k", j2k_widen(np.stack([j2k_gray(mos_rgba), mos_rgba[..., 3]], -1), 16),
         16, dict(codecformat="J2K", colorspace="GRAY")),
        ("rgb12_53.j2k", j2k_widen(rgb37, 12), 12, dict(codecformat="J2K", bitspersample=12)),
        ("rgba16_53.jp2", j2k_widen(mos_rgba, 16), 16, dict()),
    ]
    for name, px, bits, kw in lossless:
        data = enc(px, level=0, reversible=True, **kw)
        j2k_check_lossless(data, px)
        wr.add(name, data, px, bits)

    # Lower resolution levels of a lossless file: exact against OpenJPEG `skipres`.
    data = (out / "rgb8_53_rct.jp2").read_bytes()
    for r in (1, 2):
        wr.add("rgb8_53_rct.jp2", data, dec(data, skipres=r), 8, reduce=r, write=False)
    data = (out / "gray12_53.jp2").read_bytes()
    wr.add("gray12_53.jp2", data, dec(data, skipres=1), 12, reduce=1, write=False)

    # Lossy 9/7 (irreversible, ICT for RGB): bounded against OpenJPEG's decode. Both decoders
    # run the 9/7 inverse in f32 with different lifting order / rounding: 1 LSB at most
    # (16 in 16-bit output units for 12-bit samples).
    lossy = [
        ("rgb8_97.jp2", mos_rgb, 8, dict(level=38), 1),
        ("rgba8_97.j2k", rgba64, 8, dict(level=40, codecformat="J2K"), 1),
        ("gray12_97.j2k", j2k_widen(j2k_gray(rgb37), 12), 12,
         dict(level=60, codecformat="J2K", bitspersample=12), 16),
    ]
    for name, px, bits, kw, md in lossy:
        data = enc(px, reversible=False, **kw)
        wr.add(name, data, dec(data), bits, max_diff=md)
    data = (out / "rgb8_97.jp2").read_bytes()
    wr.add("rgb8_97.jp2", data, dec(data, skipres=1), 8, max_diff=1, reduce=1, write=False)

    # Tiles, quality layers, progression orders, image offset (Pillow / OpenJPEG 2.5.4).
    img = Image.fromarray(cutout, "RGBA")
    data = j2k_pillow(img, tile_size=(32, 32), quality_mode="rates",
                      quality_layers=[40, 10, 1], num_resolutions=3, progression="RPCL")
    j2k_check_lossless(data, cutout)
    wr.add("rgba8_tiles_layers.jp2", data, cutout, 8)

    img = Image.fromarray(mos_rgb, "RGB")
    data = j2k_pillow(img, no_jp2=True, irreversible=True, offset=(3, 5), tile_offset=(0, 0),
                      tile_size=(24, 20), quality_mode="dB", quality_layers=[28, 36, 44],
                      progression="PCRL", num_resolutions=4)
    wr.add("rgb8_tiles_97_offset.j2k", data, dec(data), 8, max_diff=1)

    img = Image.fromarray(rgb37, "RGB")
    data = j2k_pillow(img, tile_size=(16, 16), progression="CPRL", num_resolutions=2,
                      quality_layers=[20, 5, 1], mct=1)
    j2k_check_lossless(data, rgb37)
    wr.add("rgb8_tiles_cprl.jp2", data, rgb37, 8)

    # sYCC colour space (JP2 enumerated colour space 18), lossless YCbCr samples: the YCC ->
    # RGB conversion is done in floating point by both decoders.
    ycc = np.asarray(img.convert("YCbCr"))
    data = j2k_pillow(Image.fromarray(ycc, "YCbCr"))
    wr.add("ycc8_sycc.jp2", data, dec(data), 8, max_diff=1)

    # Sub-sampled components (OpenJPEG CLI raw input).
    opj_dir = os.environ.get("OPJ_BIN")
    opj = Path(opj_dir) / "opj_compress" if opj_dir else shutil.which("opj_compress")
    if opj is None or not Path(opj).exists():
        print("opj_compress not found (set OPJ_BIN): sub-sampled files kept as they are",
              file=sys.stderr)
        for name in ("graya8_sub.j2k", "ycc420.j2k"):
            prev = [ln for ln in (out / "manifest.txt").read_text().splitlines()
                    if ln.startswith(name + " ")]
            wr.lines.extend(prev)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Gray + alpha with the alpha plane sub-sampled 2x2: replication of the source.
            g = j2k_gray(mos_rgba)
            a = mos_rgba[0::2, 0::2, 3].copy()
            (tmp / "ga.raw").write_bytes(g.tobytes() + a.tobytes())
            j2k_opj(opj, ["-i", str(tmp / "ga.raw"), "-o", str(tmp / "ga.j2k"),
                          "-F", "50,38,2,8,u@1x1:2x2"])
            data = (tmp / "ga.j2k").read_bytes()
            ref = np.stack([g, np.repeat(np.repeat(a, 2, 0), 2, 1)[:38, :50]], -1)
            wr.add("graya8_sub.j2k", data, ref, 8)

            # YCbCr 4:2:0 codestream (three components, chroma sub-sampled): decoders treat
            # it as sYCC; compared with OpenJPEG's integer sycc420_to_rgb.
            ycc = np.asarray(Image.fromarray(rgb37, "RGB").convert("YCbCr"))
            planes = [ycc[..., 0], ycc[0::2, 0::2, 1].copy(), ycc[0::2, 0::2, 2].copy()]
            (tmp / "ycc.raw").write_bytes(b"".join(p.tobytes() for p in planes))
            j2k_opj(opj, ["-i", str(tmp / "ycc.raw"), "-o", str(tmp / "ycc.j2k"),
                          "-F", "37,29,3,8,u@1x1:2x2:2x2", "-mct", "0", "-n", "4"])
            data = (tmp / "ycc.j2k").read_bytes()
            dopj = Path(opj).with_name("opj_decompress")
            subprocess.run([str(dopj), "-i", str(tmp / "ycc.j2k"), "-o", str(tmp / "ycc.png")],
                           check=True, stdout=subprocess.DEVNULL)
            ref = np.asarray(Image.open(tmp / "ycc.png").convert("RGB"))
            wr.add("ycc420.j2k", data, ref, 8, max_diff=1)

    (out / "manifest.txt").write_text("\n".join(wr.lines) + "\n")


# ---------------------------------------------------------------- OpenVDB

def gen_vdb(root):
    """testdata/file/vdb: OpenVDB 10 files written by tools/vdb-golden/gen.cpp.

    Needs g++ and the OpenVDB 10 C++ library with headers in OVDB_PREFIX (a directory with
    include/ and lib/). On Ubuntu 24.04 without root: `apt-get download libopenvdb-dev
    libopenvdb10.0t64 libtbb-dev libtbb12 libtbbmalloc2 libtbbbind-2-5 libimath-dev libblosc1
    libblosc-dev liblog4cplus-2.0.5t64 libboost1.83-dev libboost-iostreams1.83-dev` and
    `dpkg -x <deb> <dir>` each, then OVDB_PREFIX=<dir>/usr. Offline only, never linked.
    """
    prefix = os.environ.get("OVDB_PREFIX")
    if not prefix:
        print("vdb: OVDB_PREFIX not set, skipping", file=sys.stderr)
        return
    prefix = Path(prefix)
    lib = prefix / "lib" / "x86_64-linux-gnu"
    out = root / "testdata" / "file" / "vdb"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exe = tmp / "vdb_gen"
        subprocess.run(
            ["g++", "-O1", "-std=c++17", "-w", f"-I{prefix / 'include'}",
             str(root / "tools" / "vdb-golden" / "gen.cpp"), "-o", str(exe), f"-L{lib}",
             "-l:libopenvdb.so.10.0", "-l:libtbb.so.12", "-l:libImath-3_1.so.29", "-lpthread", f"-Wl,-rpath-link,{lib}"],
            check=True)
        work = tmp / "out"
        work.mkdir()
        env = dict(os.environ, LD_LIBRARY_PATH=str(lib))
        subprocess.run([str(exe), str(work)], check=True, env=env)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        # Dense expectations: one file per distinct content.
        dense = {}
        lines = []
        for line in (work / "expected.txt").read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "grid" and parts[-1] != "-":
                data = (work / parts[-1]).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest not in dense:
                    dense[digest] = parts[-1]
                    (out / parts[-1]).write_bytes(data)
                parts[-1] = dense[digest]
                line = " ".join(parts)
            lines.append(line)
        (out / "expected.txt").write_text("\n".join(lines) + "\n")
        for f in sorted(work.glob("*.vdb")):
            data = f.read_bytes()
            if f.name == "float_none.vdb":
                # Uncompressed internal nodes are large: stored zlib-wrapped, the test inflates.
                (out / (f.name + ".zz")).write_bytes(zlib.compress(data, 9))
            else:
                (out / f.name).write_bytes(data)
    print(f"vdb: wrote {len(list(out.iterdir()))} files to {out}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--verify-jxl-encoder":
        jxl_verify_encoder(args[1])
    else:
        gens = {"jxl": gen_jxl, "jpeg2000": gen_jpeg2000, "vdb": gen_vdb}
        for name in args or list(gens):
            gens[name](root)
