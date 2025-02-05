#!/bin/bash

filename=sgemm
impls=("cublas" "device")
data=data

if [ $# -eq 0 ]; then
    echo "No arguments provided."
    exit 1
fi

mkdir -p $data

case "$1" in
  size_matrix)
  make $filename.bin
  for impl in "${impls[@]}"; do
  > $data/${impl}_perf
    for ((i=8; i<=14; i++)); do
      echo $((2 ** i)) >> $data/${impl}_perf
      build/$filename $impl $((2 ** i)) >> $data/${impl}_perf
    done
  done
  ;;
  hgemm)
  make hgemm.bin
  for impl in "${impls[@]}"; do
  > $data/${impl}_perf_hgemm
    for ((i=8; i<=14; i++)); do
      echo $((2 ** i)) >> $data/${impl}_perf_hgemm
      build/hgemm $impl $((2 ** i)) >> $data/${impl}_perf_hgemm
    done
  done
  ;;
esac