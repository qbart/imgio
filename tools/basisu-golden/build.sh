#!/bin/sh
# Builds ./dump (offline golden generator) against ref/basisu; writes testdata/basis/golden.txt.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-./dump}
g++ -O2 -std=c++17 -I../../ref/basisu/transcoder -I../../ref/basisu/zstd -DBASISD_SUPPORT_KTX2_ZSTD=0 \
    dump.cpp ../../ref/basisu/transcoder/basisu_transcoder.cpp -o "$OUT"
"$OUT" ../../testdata/basis/*.basis > ../../testdata/basis/golden.txt
