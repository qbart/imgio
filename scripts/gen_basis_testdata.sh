#!/bin/sh
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
# Encodes the .basis / .ktx2 test files in testdata/{basis,ktx2} from testdata/images with the
# C++ Basis Universal CLI (offline dev tool; build it in ref/basisu, see `make ref`).
# Usage: sh scripts/gen_basis_testdata.sh [file-name-filter]
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=${BASISU:-$ROOT/ref/basisu/bin/basisu}
I=$ROOT/testdata/images
ONLY=$1

enc() {
    out=$1
    shift
    case "$out" in *"$ONLY"*) ;; *) return 0 ;; esac
    case "$out" in *.ktx2) dir=ktx2 ;; *) dir=basis ;; esac
    "$B" -quiet "$@" -output_file "$ROOT/testdata/$dir/$out" >/dev/null
}

# ETC1S / UASTC / HDR
enc etc1s_rgba.basis -basis -file "$I/rgba_64x48.png" -mipmap
enc etc1s_rgb.basis -basis -file "$I/rgb_37x29.png"
enc uastc_rgba.basis -basis -uastc -file "$I/rgba_64x48.png" -mipmap
enc uastc_rgb.basis -basis -uastc -file "$I/rgb_37x29.png"
enc uastc_hdr4x4.basis -basis -hdr -file "$I/hdr_40x24.hdr"
enc astc_hdr6x6.basis -basis -hdr_6x6 -file "$I/hdr_40x24.hdr"
enc uastc_hdr6x6i.basis -basis -hdr_6x6i -file "$I/hdr_40x24.hdr"
enc etc1s_rgba.ktx2 -ktx2 -file "$I/rgba_64x48.png" -mipmap
enc etc1s_array.ktx2 -ktx2 -tex_array -file "$I/rgb_37x29.png" -file "$I/rgb_37x29.png"
enc uastc_rgba_zstd.ktx2 -ktx2 -uastc -file "$I/rgba_64x48.png" -mipmap
enc uastc_rgb_raw.ktx2 -ktx2 -uastc -ktx2_no_zstandard -file "$I/rgb_37x29.png"
enc uastc_hdr4x4.ktx2 -ktx2 -hdr -file "$I/hdr_40x24.hdr"
enc astc_hdr6x6.ktx2 -ktx2 -hdr_6x6 -file "$I/hdr_40x24.hdr"
enc uastc_hdr6x6i.ktx2 -ktx2 -hdr_6x6i -file "$I/hdr_40x24.hdr"

# Plain ASTC LDR (basisu v2 tex formats 19..32)
enc astc_ldr4x4_rgba.basis -basis -ldr_4x4 -file "$I/rgba_64x48.png" -mipmap
enc astc_ldr6x5_rgb.basis -basis -ldr_6x5 -linear -file "$I/rgb_37x29.png"
enc astc_ldr8x8_pow2.basis -basis -ldr_8x8 -file "$I/rgba_32x32.png" -mipmap
enc astc_ldr12x12_rgba.basis -basis -ldr_12x12 -file "$I/rgba_64x48.png"
enc astc_ldr8x5_rgba.ktx2 -ktx2 -ldr_8x5 -file "$I/rgba_64x48.png" -mipmap
enc astc_ldr10x10_rgb.ktx2 -ktx2 -ldr_10x10 -linear -file "$I/rgb_37x29.png"

# XUASTC LDR (tex formats 5..18): zstd, arithmetic and hybrid syntaxes, DCT
enc xuastc4x4_rgba.basis -basis -ldr_4x4i -xuastc_zstd -file "$I/rgba_64x48.png" -mipmap
enc xuastc6x6_rgba_arith.basis -basis -ldr_6x6i -xuastc_arith -file "$I/rgba_64x48.png" -mipmap
enc xuastc8x6_rgb_hybrid.basis -basis -ldr_8x6i -xuastc_hybrid -linear -file "$I/rgb_37x29.png"
enc xuastc5x4_dct.basis -basis -ldr_5x4i -quality 60 -file "$I/rgba_64x48.png"
enc xuastc12x12_pow2.basis -basis -ldr_12x12i -file "$I/rgba_32x32.png" -mipmap
# Images of <= 64 blocks always use the full arithmetic syntax; these exceed it.
enc xuastc4x4_hybrid_dct.basis -basis -ldr_4x4i -xuastc_hybrid -quality 60 -file "$I/rgba_64x48.png" -mipmap
enc xuastc6x5_dct_arith.basis -basis -ldr_6x5i -xuastc_arith -quality 50 -file "$I/rgba_64x48.png"
enc xuastc6x6_rgba.ktx2 -ktx2 -ldr_6x6i -file "$I/rgba_64x48.png" -mipmap
enc xuastc10x8_dct_arith.ktx2 -ktx2 -ldr_10x8i -xuastc_arith -quality 40 -file "$I/rgb_37x29.png"
# Block-structured mosaics: solid / shared-endpoint / 2-subset / dual-plane / alpha blocks for
# the XUASTC -> BC7 fast paths (mips cover the small edge cases).
enc xuastc6x6_mosaic_rgba.basis -basis -ldr_6x6i -file "$I/mosaic_rgba_50x38.png" -mipmap
enc xuastc6x6_mosaic_rgb.basis -basis -ldr_6x6i -linear -file "$I/mosaic_rgb_50x38.png"
enc xuastc8x6_mosaic_rgba.basis -basis -ldr_8x6i -file "$I/mosaic_rgba_50x38.png" -mipmap
enc xuastc8x6_mosaic_rgb.basis -basis -ldr_8x6i -linear -file "$I/mosaic_rgb_50x38.png"
enc xuastc4x4_array.ktx2 -ktx2 -ldr_4x4i -tex_array -file "$I/rgb_37x29.png" -file "$I/rgb_37x29.png"

# XUBC7 (tex format 33)
enc xubc7_rgba.basis -basis -xubc7 -file "$I/rgba_64x48.png" -mipmap
enc xubc7_rgb.basis -basis -xubc7 -file "$I/rgb_37x29.png"
enc xubc7_rgba.ktx2 -ktx2 -xubc7 -file "$I/rgba_32x32.png" -mipmap
