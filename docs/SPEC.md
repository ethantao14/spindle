# LLM Inference Engine Spec

As of 2026-10-17

## Overview

A from-scratch LLM inference engine in C++20 that loads real open model weights and generates text on a laptop CPU, with no ML libraries. Every output is checked against Hugging Face Transformers, and every speed claim is benchmarked against llama.cpp on the same machine.

**Goals**

- Exact correctness: logits match the PyTorch reference at every layer, and greedy output is token-for-token identical.
- Real performance work: SIMD, multithreading, cache-aware matmul, quantization, with numbers reported as a percent of the machine's peak.
- Two architectures from one config-driven codebase: GPT-2 first, then Llama.
- A results table and live demo, like quarry's.

**Non-goals**

- Training or backprop (inference only).
- A GPU backend in the core milestones (stretch goal).
- Python at runtime. Python is used only in tests, to produce reference outputs.

**Target models**

| Model | Params | Why |
| --- | --- | --- |
| GPT-2 small | 124M | Simplest transformer, fast iteration, byte-level BPE |
| TinyLlama-1.1B | 1.1B | Llama architecture (RMSNorm, RoPE, SwiGLU, grouped-query attention), ungated, Apache 2.0 |

**Allowed dependencies**

- C++ standard library, CMake.
- nlohmann/json, for safetensors headers and model configs.
- GoogleTest, Google Benchmark, libFuzzer.
- Everything else is written by hand: tensors, kernels, tokenizer, weight loader, sampler, server.

**Don't copy from:** llama.cpp, llama2.c, or ggml. You can read them for ideas after writing your own version, but reviewers know these codebases well.

## Milestones

Eight milestones, each a set of PRs that ends when every pass criterion is met and reproducible from a fresh clone. M2 is the hardest checkpoint; once GPT-2 matches the reference exactly, the rest is extension and speed.

### M0: Repo, CI, reference harness

- CMake build, clang-format, clang-tidy, GoogleTest.
- CI builds a Release target plus an AddressSanitizer/UBSan target and runs all tests on both.
- `tools/reference.py`: runs a model in Hugging Face Transformers and dumps token IDs, per-layer activations and logits to binary files the C++ tests load.

**Pass:** CI green on both builds; one test proven able to fail by injecting a bug; reference dumps generated for GPT-2 on 5 fixed prompts.

### M1: Tensors and core ops

- Tensor type: float32, shape and strides, views without copies, 64-byte aligned allocation.
- Ops: naive matmul, add, softmax, LayerNorm, GELU, embedding lookup.
- Numerically stable softmax (subtract the max).

**Pass:** each op matches a NumPy reference within 1e-5 relative error on random and edge-case inputs (very large, very small, all-equal values); naive matmul GFLOPS recorded as the baseline.

### M2: GPT-2 forward pass

- safetensors loader using mmap, with shape and dtype validation.
- Byte-level BPE tokenizer written by hand (encode and decode).
- Causal multi-head attention, the full transformer block, greedy generation, a CLI.

**Pass:** tokenizer output identical to Hugging Face on 1,000+ strings including Unicode, emoji, whitespace runs and code; activations match at every layer within 1e-4; a 100-token greedy generation is identical to Hugging Face on all 5 prompts.

### M3: KV cache and sampling

- KV cache, so each new token attends to stored keys and values instead of recomputing the whole sequence.
- Temperature, top-k and top-p sampling with a seeded RNG; streaming output in the CLI.

**Pass:** cached and uncached runs give identical logits; per-token latency stays flat as the sequence grows (plot it); same seed gives the same output across runs.

### M4: Performance

- Cache-blocked matmul, then SIMD (AVX2 on x86 or NEON on Apple Silicon).
- A thread pool that splits matmul and attention across cores.
- Fused ops where profiling shows memory traffic dominates.
- Profile with perf or Instruments and write down what each change bought in NOTES.md.

**Pass:** matmul reaches at least 50% of the machine's measured peak FLOPS; GPT-2 FP32 tokens/sec within 2x of llama.cpp FP32 on the same machine and thread count; every optimization keeps M2's exact-match tests green.

### M5: Llama architecture

- Config-driven model so GPT-2 and Llama share one engine.
- RMSNorm, rotary position embeddings (RoPE), SwiGLU, grouped-query attention.
- Load TinyLlama's tokenizer format.

**Pass:** TinyLlama activations match Hugging Face at every layer within 1e-4; 100-token greedy output identical on 5 prompts; GPT-2 tests still pass.

### M6: Quantization

- int8 block quantization (one scale per 32 weights), then 4-bit with grouped scales.
- Quantized matmul kernels that dequantize in registers, not into a full float copy.

**Pass:** perplexity on a fixed WikiText-2 slice reported for FP32, int8 and int4, with int8 within 1% of FP32; model memory and tokens/sec reported for each; results compared against llama.cpp's matching formats.

### M7: Serving and batching

- HTTP server that streams tokens over SSE.
- Batching of concurrent requests, so several users share each forward pass.
- A load-testing benchmark tool.

**Pass:** throughput and p50/p99 latency reported at 1, 4 and 8 concurrent clients; batched throughput beats running requests one at a time; outputs under batching identical to unbatched.

## Testing standards

These apply to every PR, plus checks specific to numerical code.

- **Tests proven able to fail:** for each new test, inject a bug (flip a sign, drop a scale, off-by-one in the causal mask) and confirm the test catches it.
- **Fresh-clone verification:** every milestone's pass criteria reproduce from a clean clone with one documented command.
- **Reproducible claims:** every number in the README has a command next to it, plus the machine, compiler and thread count it was measured with.
- **Layer-by-layer reference checks:** compare every intermediate tensor, not only final logits, so a bug is located to one layer.
- **Tolerances written down:** each comparison states its tolerance and why (1e-5 for single ops, 1e-4 across a full forward pass, since float32 errors accumulate).
- **Sanitizers:** the AddressSanitizer/UBSan build runs the full suite in CI.
- **Fuzzing:** libFuzzer targets for the safetensors loader and the tokenizer. Malformed files and invalid UTF-8 produce errors, never crashes.
- **Determinism:** multithreaded results must match single-threaded results within tolerance, so thread scheduling can't change output.

## Results table (README template)

Fill this in as milestones land; it becomes the README's headline and the source of resume numbers. All rows on one machine, same thread count, llama.cpp as the baseline.

| Model | Format | Memory (MB) | Tokens/sec (ours) | Tokens/sec (llama.cpp) | Perplexity | Matches Hugging Face |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 124M | FP32 | | | | | |
| GPT-2 124M | int8 | | | | | |
| TinyLlama 1.1B | FP32 | | | | | |
| TinyLlama 1.1B | int8 | | | | | |
| TinyLlama 1.1B | int4 | | | | | |

A second table for M7 records throughput and p50/p99 latency at 1, 4 and 8 concurrent clients, batched versus unbatched.

## Stretch goals, prep and open decisions

**Stretch goals** (only after M7)

- GPU backend in Metal or CUDA, checked against the CPU path.
- Tiled attention in the style of FlashAttention, to cut memory traffic on long contexts.
- Speculative decoding, using GPT-2 as a draft model for a larger one.
- Compile GPT-2 to WebAssembly so the demo runs in the browser on Vercel.

**Prep before M2**

- "Attention Is All You Need" (Vaswani et al., 2017), for the core architecture.
- Andrej Karpathy's "Let's build GPT" video and nanoGPT, for a readable PyTorch version.
- The GPT-2 paper and Hugging Face's GPT-2 modeling code, which is the reference.
- Before M5: the RoFormer paper (RoPE) and the Llama paper.

**Timeline:** about 5 to 7 weeks at part-time pace. M0 to M2 is roughly half of it, because exact matching takes the most debugging.

**Open decisions**

- [x] Project name: spindle
- [x] Main dev machine: Apple Silicon (M3, NEON, Instruments)
- [ ] Demo: browser WebAssembly build or a recorded CLI GIF
