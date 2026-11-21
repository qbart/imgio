#!/bin/sh
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
# Regenerates testdata/astcenc/compress_golden.txt (src/astcenc/tests.rs
# `compress_matches_astcenc`) with the C++ astc-encoder 6.0 in ref/astcenc, compiled like its
# CMake AVX2 build (C++20, AVX2 + BMI / BMI2 + POPCNT + F16C, invariance on: no FP contraction).
# Offline dev tool: the C++ code is only built here, never linked into imgio. Requires g++.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${ASTCENC_SRC:-$ROOT/ref/astcenc/Source}
WORK=${WORK:-$(mktemp -d)}
mkdir -p "$WORK"
# shellcheck disable=SC2046
g++ -O3 -w -std=c++20 -mavx2 -mbmi -mbmi2 -mpopcnt -mf16c -ffp-contract=off -fno-math-errno \
    -DNDEBUG -DASTCENC_NEON=0 -DASTCENC_SVE=0 -DASTCENC_SSE=41 -DASTCENC_AVX=2 \
    -DASTCENC_X86_GATHERS=1 -DASTCENC_POPCNT=1 -DASTCENC_F16C=1 -I"$SRC" \
    "$ROOT/tools/astcenc-golden/compress.cpp" $(ls "$SRC"/astcenc_*.cpp) \
    -o "$WORK/astcenc_compress_golden" -lpthread
"$WORK/astcenc_compress_golden" > "$ROOT/testdata/astcenc/compress_golden.txt"
echo "wrote testdata/astcenc/compress_golden.txt ($(wc -l < "$ROOT/testdata/astcenc/compress_golden.txt") cases)"
