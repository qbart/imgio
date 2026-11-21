#!/bin/sh
# Generates the astcf / fast 4x4 ASTC LDR encoder verification dumps of the C++ encoder (basisu
# v2.50 with the `.xud` dump hooks, see docs/spec/xuastc_encode.md) and their manifest
# (name bw bh effort srgb quality scd merge disable_subsets disable_rgb_dp).
#
#   BASISU=/path/to/patched/basisu OUT=/tmp/astcf_dumps tools/basisu-golden/astcf_dumps.sh
#   XU_DUMP_DIR=/tmp/astcf_dumps cargo test --release --lib astcf_matches_cpp_dumps -- --ignored
#
# With GOLDEN=1 it also rewrites testdata/basisu_enc/basisu_astcf.txt (hashes of the C++ output
# for the cases listed in `astcf.rs` GOLDEN_CASES, checked by `astcf_matches_cpp_golden`).
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
I=$R/testdata/images
OUT=${OUT:-/tmp/astcf_dumps}
: "${BASISU:?set BASISU to the patched basisu binary}"
mkdir -p "$OUT"
M=$OUT/astcf_manifest.txt
: > "$M"
gen() { # NAME IMAGE.png [basisu options...] -> $OUT/NAME.first.xud
  gn=$1; gi=$2; shift 2
  if [ "$STAGE" = pre ]; then gv=XU_DUMP_PRE; else gv=XU_DUMP_FIRST; fi
  rm -f "$OUT/$gn".*
  env "$gv=$OUT/$gn.first.xud" "$BASISU" -quiet -basis "$@" -file "$gi" -output_file "$OUT/$gn.basis" >/dev/null
  rm -f "$OUT/$gn.basis"
}
run() { # name img bw bh effort linear(0/1) quality(0=none) deblock(0/1) alpha(0/1) [xs(0/1) xp(0/1)]
  n=$1; img=$2; bw=$3; bh=$4; e=$5; lin=$6; q=$7; db=$8; al=$9; xs=${10:-0}; xp=${11:-0}
  xa=""; [ "$xs" = 1 ] && xa="$xa -xs"; [ "$xp" = 1 ] && xa="$xa -xp"
  st=""; [ "$e" = 0 ] && [ "$bw$bh" = 44 ] && st=pre
  if [ "$q" = 0 ]; then fmt="-ldr_${bw}x${bh}"; qa=""; else fmt="-ldr_${bw}x${bh}i"; qa="-quality $q"; fi
  la=""; srgb=1; [ "$lin" = 1 ] && { la="-linear"; srgb=0; }
  dba="-xuastc_ldr_no_deblocking"; scd=0
  if [ "$db" = 1 ]; then dba=""; if [ $((bw*bh)) -ge 80 ] && [ "$e" -ge 2 ]; then scd=1; fi; fi
  merge=0
  if [ "$e" -ge 4 ]; then merge=1; elif [ "$e" = 3 ] && { [ $((bw*bh)) -gt 64 ] || [ "$al" = 1 ]; }; then merge=1; fi
  STAGE=$st gen "$n" "$I/$img.png" $fmt -effort $e $la $qa $dba $xa
  echo "$n $bw $bh $e $srgb $q $scd $merge $xs $xp" >> "$M"
}
alpha() { case $1 in *rgba*) echo 1;; *) echo 0;; esac; }
for img in rgb_37x29 mosaic_rgb_50x38 rgba_64x48 mosaic_rgba_50x38 rgba_32x32; do
  for bs in 4x4 5x5 6x6 8x6 8x8 10x8 12x12; do
    bw=${bs%x*}; bh=${bs#*x}
    for e in 1 2 3; do run "a_${img}_${bs}_e$e" $img $bw $bh $e 0 0 0 $(alpha $img); done
  done
done
for img in rgb_37x29 rgba_64x48; do
  for bs in 4x4 5x5 6x6 8x6 8x8 10x8 12x12; do
    bw=${bs%x*}; bh=${bs#*x}
    for e in 2 3; do run "l_${img}_${bs}_e$e" $img $bw $bh $e 1 0 0 $(alpha $img); done
  done
done
for img in mosaic_rgb_50x38 rgba_64x48; do
  for bs in 4x4 5x5 6x6 8x6 8x8 10x8 12x12; do
    bw=${bs%x*}; bh=${bs#*x}
    for e in 1 3; do run "q_${img}_${bs}_e$e" $img $bw $bh $e 0 60 0 $(alpha $img); done
  done
done
for bs in 5x4 6x5 8x5 10x5 10x6 10x10 12x10; do
  bw=${bs%x*}; bh=${bs#*x}
  run "o_mosaic_rgba_50x38_${bs}_e3" mosaic_rgba_50x38 $bw $bh 3 0 0 0 1
  run "o_rgb_37x29_${bs}_e2" rgb_37x29 $bw $bh 2 0 0 0 0
done
for e in 0 5 10; do
  run "h_rgb_37x29_6x6_e$e" rgb_37x29 6 6 $e 0 0 0 0
  run "h_rgba_32x32_8x8_e$e" rgba_32x32 8 8 $e 0 0 0 1
  run "hq_rgba_64x48_6x6_e$e" rgba_64x48 6 6 $e 1 40 0 1
done
for bs in 10x8 12x12; do
  bw=${bs%x*}; bh=${bs#*x}
  run "d_rgb_37x29_${bs}_e3" rgb_37x29 $bw $bh 3 0 0 1 0
  run "d_rgba_64x48_${bs}_e3" rgba_64x48 $bw $bh 3 0 0 1 1
  run "dq_rgba_64x48_${bs}_e2" rgba_64x48 $bw $bh 2 0 75 1 1
done
for img in rgb_37x29 mosaic_rgb_50x38 rgba_64x48 mosaic_rgba_50x38 rgba_32x32; do
  run "f_${img}_4x4_e0" $img 4 4 0 0 0 0 $(alpha $img)
done
run "fl_rgba_64x48_4x4_e0" rgba_64x48 4 4 0 1 0 0 1
run "fq_mosaic_rgb_50x38_4x4_e0" mosaic_rgb_50x38 4 4 0 0 60 0 0
run "fq_rgba_64x48_4x4_e0" rgba_64x48 4 4 0 1 30 0 1
run "fx_rgba_64x48_4x4_e0" rgba_64x48 4 4 0 0 0 0 1 1 1
run "fxp_mosaic_rgb_50x38_4x4_e0" mosaic_rgb_50x38 4 4 0 0 0 0 0 0 1
run "x_rgba_64x48_6x6_e3" rgba_64x48 6 6 3 0 0 0 1 1 0
run "xp_mosaic_rgb_50x38_6x6_e3" mosaic_rgb_50x38 6 6 3 0 0 0 0 0 1
run "xpq_rgba_64x48_8x8_e5" rgba_64x48 8 8 5 0 50 0 1 1 1
wc -l "$M"
if [ "$GOLDEN" = 1 ]; then
  cd "$R"
  XU_DUMP_DIR=$OUT XU_GOLDEN_OUT=$R/testdata/basisu_enc/basisu_astcf.txt \
    cargo test --release --lib astcf_golden_from_dumps -- --ignored
fi
