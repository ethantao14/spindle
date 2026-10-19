# Performance notes

Each entry records what a change bought, with the command, machine, compiler and thread count.

## Baseline: naive matmul (M1)

Triple loop in `src/ops.cpp`, float32, one thread, no blocking or SIMD.

```sh
cmake --preset release && cmake --build --preset release
./build/release/bench/spindle_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

Machine: Apple M3 (4 performance + 4 efficiency cores, 16 GB), macOS 26, Apple clang 17.0.0, `-O3` (CMake Release), 1 thread. Measured 2026-10-18 with a load average around 4.8, so other processes were running; treat these as approximate.

| M x K x N | Median time | GFLOPS |
| --- | --- | --- |
| 256 x 256 x 256 | 11.8 ms | 2.84 |
| 512 x 512 x 512 | 113 ms | 2.40 |
| 1 x 768 x 3072 (GPT-2 MLP, one decode token) | 2.13 ms | 2.22 |
| 64 x 768 x 768 (GPT-2 attention projection, 64 tokens) | 32.8 ms | 2.30 |

The inner loop walks `b` down a column, one cache line per multiply-add, which is the main thing M4's blocking will fix.
