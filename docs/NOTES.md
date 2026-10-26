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

## KV cache: per-token decode latency (M3)

Greedy decoding of 1000 tokens after a short prompt, timing each step (one token through the model, logits, sampling). With the KV cache, each step does the same matmul work and only the attention over stored keys grows.

```sh
./build/release/bench/spindle_latency --model build/reference/gpt2/model --tokens 1000 --out docs/m3-latency.csv
python3 tools/plot_latency.py docs/m3-latency.csv docs/m3-latency.svg
```

Machine: Apple M3 (16 GB), macOS 26, Apple clang 17.0.0, `-O3` (CMake Release), 1 thread, naive matmul. Measured 2026-10-26 on battery power with sleep blocked by `caffeinate -i`; total 107 s.

| Steps | Median per token |
| --- | --- |
| first 100 | 99.2 ms |
| 400 to 500 | 104.7 ms |
| last 100 | 112.1 ms |

Last 100 over first 100: 1.13. The slow rise is attention reading more stored positions; without the cache each step would rerun every earlier token (not measured here). Isolated spikes up to 157 ms are other processes. Plot: [m3-latency.svg](m3-latency.svg).

## Blocked NEON matmul (M4)

`src/ops.cpp` now has two NEON paths. Row-major `b`: blocks of 64 rows x 128 columns x 128 inner, with a register tile of 4 rows x 16 columns (16 vector accumulators) fed by `vfmaq`. Transposed `b` (attention scores, tied-embedding logits): dot products along K with four independent accumulators, four output columns at a time. Other strides use the old loop. Packing `b` panels was tried and made decode-size shapes slower, so it was dropped.

```sh
./build/release/bench/spindle_peak_flops
./build/release/bench/spindle_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
./build/release/bench/spindle_latency --model build/reference/gpt2/model --tokens 200 --out latency.csv
```

Machine: Apple M3, macOS 26, Apple clang 17.0.0, `-O3`, 1 thread, on battery with `caffeinate -dimsu`. Measured 2026-10-26. Load average 1.6 for the before run and 4.5 for the after run, so the after numbers are if anything low.

Measured single-thread peak (`spindle_peak_flops`, independent FMA chains): 105.6 GFLOPS.

| M x K x N | Before (naive) | After | Share of peak |
| --- | --- | --- | --- |
| 256 x 256 x 256 | 2.86 GFLOPS | 85.6 GFLOPS | 81% |
| 512 x 512 x 512 | 2.44 GFLOPS | 85.3 GFLOPS | 81% |
| 64 x 768 x 768 | 2.26 GFLOPS | 81.8 GFLOPS | 77% |
| 1 x 768 x 3072 (decode MLP) | 2.16 GFLOPS | 24.8 GFLOPS | 23% |
| 1 x 768 x 50257, `b` transposed (logits) | 3.32 GFLOPS | 22.9 GFLOPS | 22% |

Single-row shapes read every weight once per token, so they are limited by memory bandwidth rather than FMA throughput: 1 x 768 x 3072 moves 9.4 MB in 0.19 ms, about 50 GB/s.

GPT-2 greedy decode, 200 tokens: median 105.4 ms per token before (9.5 tokens/s), 19.7 ms after (50.7 tokens/s), first 100 steps. The exact-match tests (activations within 1e-4, 100-token greedy output) still pass.
