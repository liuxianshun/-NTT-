$ErrorActionPreference = "Stop"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

if (Test-Path ".\build\Release\ntt_bench.exe") {
    $exe = ".\build\Release\ntt_bench.exe"
} elseif (Test-Path ".\build\ntt_bench.exe") {
    $exe = ".\build\ntt_bench.exe"
} else {
    $exe = ".\build\ntt_bench"
}

& $exe `
  --mode all `
  --min-exp 10 `
  --max-exp 18 `
  --direct-limit-exp 13 `
  --threads 1,2,4,8 `
  --repeats 5 `
  --csv results/bench.csv
