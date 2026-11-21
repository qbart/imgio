#!/bin/sh
# Builds ./dump (offline golden generator) against ref/basisu; writes testdata/{basis,ktx2}/golden.txt
# and golden_flags.txt (selected files transcoded with non-default basisu decode flags).
set -e
cd "$(dirname "$0")"
OUT=${OUT:-./dump}
gcc -O2 -c ../../ref/basisu/zstd/zstddeclib.c -o "$OUT.zstd.o"
g++ -O2 -std=c++17 -I../../ref/basisu/transcoder -I../../ref/basisu/zstd -DBASISD_SUPPORT_KTX2_ZSTD=1 \
    dump.cpp ../../ref/basisu/transcoder/basisu_transcoder.cpp "$OUT.zstd.o" -o "$OUT"
"$OUT" ../../testdata/basis/*.basis > ../../testdata/basis/golden.txt
"$OUT" ../../testdata/ktx2/*.ktx2 > ../../testdata/ktx2/golden.txt
B=../../testdata/basis
K=../../testdata/ktx2
{
    "$OUT" -f 4 $B/etc1s_rgba.basis $B/uastc_rgba.basis $B/astc_ldr4x4_rgba.basis \
        $B/xuastc6x6_mosaic_rgba.basis $B/xubc7_rgba.basis
    "$OUT" -f 8 $B/etc1s_rgba.basis $B/uastc_rgba.basis $B/astc_ldr4x4_rgba.basis $B/xubc7_rgba.basis
    "$OUT" -f 32 $B/uastc_rgba.basis $B/uastc_hdr4x4.basis $B/uastc_hdr6x6i.basis $B/astc_hdr6x6.basis \
        $B/astc_ldr4x4_rgba.basis $B/xuastc4x4_rgba.basis $B/xuastc6x6_mosaic_rgba.basis \
        $B/xuastc8x6_mosaic_rgb.basis $B/xubc7_rgba.basis
    "$OUT" -f 64 $B/etc1s_rgba.basis $B/etc1s_rgb.basis
    "$OUT" -f 128 $B/astc_ldr12x12_rgba.basis $B/xuastc12x12_pow2.basis
    "$OUT" -f 512 $B/astc_ldr4x4_rgba.basis $B/xuastc6x6_rgba_arith.basis
    "$OUT" -f 1024 $B/xuastc4x4_rgba.basis $B/xuastc6x6_mosaic_rgba.basis $B/xuastc8x6_mosaic_rgb.basis
    "$OUT" -f 1056 $B/xuastc6x6_mosaic_rgba.basis $B/xuastc8x6_mosaic_rgb.basis
} > $B/golden_flags.txt
{
    "$OUT" -f 4 $K/etc1s_rgba.ktx2 $K/uastc_rgba_zstd.ktx2 $K/xubc7_rgba.ktx2
    "$OUT" -f 128 $K/xuastc10x8_dct_arith.ktx2 $K/astc_ldr10x10_rgb.ktx2
    "$OUT" -f 512 $K/astc_ldr8x5_rgba.ktx2 $K/xuastc6x6_rgba.ktx2
} > $K/golden_flags.txt
# PVRTC1/2, ATC, FXT1 decode golden (needs the basisu_encoder static library).
BU=${BU:-./bu}
cmake -S ../../ref/basisu -B "$BU" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BU" --target basisu_encoder -j8 >/dev/null
g++ -O2 -std=c++17 -I../../ref/basisu/transcoder -I../../ref/basisu/encoder -DBASISU_SUPPORT_SSE=0 \
    unpack.cpp "$BU/libbasisu_encoder.a" -lpthread -o "$OUT.unpack"
"$OUT.unpack" $B/*.basis > $B/golden_decode.txt
# ASTC LDR / XUASTC LDR common encoder API golden (testdata/basisu_enc/basisu_astc_ldr_common.txt).
# BASISU_ENC_LIB overrides the basisu_encoder static library (default: the one built above).
BASISU_ENC_LIB=${BASISU_ENC_LIB:-$BU/libbasisu_encoder.a}
g++ -O2 -std=c++17 -I../../ref/basisu/transcoder -I../../ref/basisu/encoder -DBASISU_SUPPORT_SSE=0 \
    astc_ldr_common.cpp "$BASISU_ENC_LIB" -lpthread -o "$OUT.astc_ldr_common"
"$OUT.astc_ldr_common" > ../../testdata/basisu_enc/basisu_astc_ldr_common.txt
# Block encoder golden (testdata/encode/golden.txt) from testdata/images/raw.
g++ -O2 -std=c++17 -I../../ref/basisu/transcoder -I../../ref/basisu/zstd -DBASISD_SUPPORT_KTX2_ZSTD=0 \
    encode.cpp -o "$OUT.encode"
"$OUT.encode" ../../testdata/images/raw/*.rgba ../../testdata/images/raw/*.rgba16f > ../../testdata/encode/golden.txt
# BC7 encoder golden (testdata/encode/bc7e_golden.txt): basisu bc7e_scalar, all presets.
g++ -O2 -std=c++17 -ffp-contract=off -I../../ref/basisu/encoder \
    bc7e.cpp ../../ref/basisu/encoder/basisu_bc7e_scalar.cpp -o "$OUT.bc7e"
"$OUT.bc7e" ../../testdata/images/raw/*.rgba > ../../testdata/encode/bc7e_golden.txt
# XUBC7 encoder golden (testdata/encode/xubc7_golden.txt): xbc7::pack_image configured like the
# basisu CLI -xubc7 path; needs the basisu_encoder library built without FP contraction.
BUX=${BUX:-./bu-nofma}
cmake -S ../../ref/basisu -B "$BUX" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS=-ffp-contract=off -DCMAKE_C_FLAGS=-ffp-contract=off >/dev/null
cmake --build "$BUX" --target basisu_encoder -j8 >/dev/null
g++ -O2 -std=c++17 -ffp-contract=off -I../../ref/basisu/transcoder -I../../ref/basisu/encoder \
    -DBASISU_SUPPORT_SSE=0 xubc7_encode.cpp "$BUX/libbasisu_encoder.a" -lpthread -o "$OUT.xubc7"
"$OUT.xubc7" ../../testdata/images/raw/*.rgba > ../../testdata/encode/xubc7_golden.txt
# XUASTC LDR full encoder, block level (testdata/basisu_enc/basisu_astc_ldr_block.txt). Includes
# basisu_astc_ldr_encode.cpp directly (static functions); links the basisu_encoder library above.
g++ -O2 -std=c++17 -I../../ref/basisu/transcoder -I../../ref/basisu/encoder -DBASISU_SUPPORT_SSE=0 \
    astc_ldr_block.cpp "$BU/libbasisu_encoder.a" -lpthread -o "$OUT.astc_ldr_block"
"$OUT.astc_ldr_block" 200 > ../../testdata/basisu_enc/basisu_astc_ldr_block.txt
