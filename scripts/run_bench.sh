#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/ntt_bench \
  --mode all \
  --min-exp 10 \
  --max-exp 18 \
  --direct-limit-exp 13 \
  --threads 1,2,4,8 \
  --repeats 5 \
  --csv results/bench.csv
