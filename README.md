# 多项式乘法及其 NTT 优化实验代码

本项目用于《并行计算》课程期末实验，围绕多项式乘法和 NTT 优化设计了三个可测试问题：

1. 直接卷积：比较串行、OpenMP 输出划分、pthread 输出划分和手工 AVX2 内层向量化。
2. 单模数 NTT：比较 Cooley-Tukey NTT 在不同线程数、静态/动态调度下的性能。
3. 三模数 CRT-NTT：使用 998244353、1004535809、469762049 三个 NTT 友好素数，比较模数级任务并行和线程内 stage 并行。

## 构建

Linux/WSL/macOS with GCC or Clang:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Windows 可使用 MinGW、MSYS2 或 WSL。MSVC 也可构建 OpenMP 版本，但 pthread 分支仅在类 Unix 平台启用。

## 运行

```bash
./build/ntt_bench --mode all --min-exp 10 --max-exp 18 --threads 1,2,4,8 --repeats 5 --csv results/bench.csv
```

常用参数：

- `--mode all|direct|ntt|crt`：选择实验组。
- `--threads 1,2,4,8`：线程数列表。
- `--min-exp 10 --max-exp 18`：多项式长度为 `2^exp`。
- `--direct-limit-exp 13`：直接卷积复杂度较高，默认只测到 `2^13`。
- `--max-coeff 1024`：随机系数上界。直接卷积 SIMD 路径使用 64 位累加，建议保持较小系数以避免长向量累加溢出。

脚本 `scripts/run_bench.sh` 和 `scripts/run_bench.ps1` 给出了默认运行方式。
