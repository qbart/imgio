#!/bin/sh
# SPDX-FileCopyrightText: 2026 imgio contributors
# SPDX-License-Identifier: Apache-2.0
# Dev tool for the `#[ignore]`d XUASTC encoder comparison tests (docs/spec/xuastc_encode.md).
# Needs a basisu CLI built from ref/basisu with xuastc_dump.patch applied, e.g.
#   cp -r ref/basisu /tmp/bu && patch -d /tmp/bu -p1 < tools/basisu-golden/xuastc_dump.patch
#   cmake -S /tmp/bu -B /tmp/bu/build -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/bu/build -j
# Usage: BASISU=/tmp/bu/bin/basisu xuastc_dumps.sh OUTDIR NAME IMAGE.png [basisu options...]
# Writes OUTDIR/NAME.{first,basisu,pre,final}.xud (the encoder candidate lists at each stage
# that runs: first encoder, basisu encoder before merging, before SCD, supercompressor input),
# NAME.cfg (effective encoder config) and NAME.basis. Use single-level images (no -mipmap).
set -e
OUT=$1; NAME=$2; IMG=$3; shift 3
mkdir -p "$OUT"
rm -f "$OUT/$NAME".*
XU_DUMP_FIRST="$OUT/$NAME.first.xud" XU_DUMP_BASISU="$OUT/$NAME.basisu.xud" \
XU_DUMP_PRE="$OUT/$NAME.pre.xud" XU_DUMP="$OUT/$NAME.final.xud" XU_DUMP_CFG="$OUT/$NAME.cfg" \
    "${BASISU:?set BASISU}" -quiet -basis "$@" -file "$IMG" -output_file "$OUT/$NAME.basis" >/dev/null
