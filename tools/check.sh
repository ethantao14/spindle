#!/usr/bin/env bash
# Runs every check CI runs: format, reference dumps, both test builds, a benchmark
# smoke run, and clang-tidy.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/clang-tidy ]]; then
  python3 -m venv .venv
  .venv/bin/pip install --quiet -r requirements-dev.txt
fi

# Tracked plus new (not ignored) files, so checks cover work not yet committed.
list_files() { git ls-files --cached --others --exclude-standard "$@"; }
cpp_files=$(list_files '*.cpp')
sources=$(list_files '*.cpp' '*.h')
if [[ -z "$cpp_files" ]]; then
  echo "error: no C++ sources found" >&2
  exit 1
fi

echo "== clang-format"
.venv/bin/clang-format --dry-run --Werror $sources

echo "== reference dumps (first run downloads PyTorch and the 548 MB GPT-2 weights)"
if [[ ! -x .venv-reference/bin/python ]]; then
  python3 -m venv .venv-reference
fi
.venv-reference/bin/pip install --quiet -r requirements-reference.txt
.venv-reference/bin/python tools/reference.py \
  --prompts tests/reference/gpt2_prompts.json --out build/reference/gpt2
.venv-reference/bin/python tools/op_reference.py --out build/reference/ops
.venv-reference/bin/python tools/tokenizer_reference.py --out build/reference/tokenizer

for preset in release asan; do
  echo "== build and test: $preset"
  cmake --preset "$preset"
  cmake --build --preset "$preset" -j
  # A skipped test is not a pass: fail loudly if ctest reports any.
  ctest --preset "$preset" --no-tests=error | tee "build/$preset/ctest.log"
  if grep -q "Skipped" "build/$preset/ctest.log"; then
    echo "error: tests were skipped in $preset build" >&2
    exit 1
  fi
done

echo "== benchmark smoke run (one iteration each, timings not meaningful)"
build/release/bench/spindle_bench --benchmark_min_time=1x

echo "== clang-tidy (uses the sanitizer build, which compiles every source)"
# The PyPI clang-tidy does not know the macOS SDK location, so pass it explicitly.
tidy_args=()
if [[ "$(uname)" == "Darwin" ]]; then
  tidy_args+=("--extra-arg=-isysroot$(xcrun --show-sdk-path)")
fi
.venv/bin/clang-tidy --quiet -p build/asan "${tidy_args[@]}" $cpp_files

echo "All checks passed."
