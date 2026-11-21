#!/bin/sh
# Builds ./dump (offline golden generator) against ref/basisu; writes testdata/{basis,ktx2}/golden.txt.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-./dump}
gcc -O2 -c ../../ref/basisu/zstd/zstddeclib.c -o "$OUT.zstd.o"
g++ -O2 -std=c++17 -I../../ref/basisu/transcoder -I../../ref/basisu/zstd -DBASISD_SUPPORT_KTX2_ZSTD=1 \
    dump.cpp ../../ref/basisu/transcoder/basisu_transcoder.cpp "$OUT.zstd.o" -o "$OUT"
"$OUT" ../../testdata/basis/*.basis > ../../testdata/basis/golden.txt
"$OUT" ../../testdata/ktx2/*.ktx2 > ../../testdata/ktx2/golden.txt
