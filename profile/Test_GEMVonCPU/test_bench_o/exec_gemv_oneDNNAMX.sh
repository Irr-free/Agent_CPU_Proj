#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH="/home/bing/tool/onednn-tbb/lib:${LD_LIBRARY_PATH}"

log=./temp.log
: > "$log"
for i in {1..79}; do
  echo "i=${i}" >> "$log"
  taskset -c 0-$i ./test_bench_o/gemv_oneDNN_bf16_batched_pinCore 4096 128 32 10000 >> "$log"
  echo "" >> "$log"
done
