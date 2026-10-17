#!/usr/bin/env bash
# Runs every check CI runs: format, clang-tidy, and the test suite in both builds.
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

echo "== clang-tidy (uses the sanitizer build, which compiles every source)"
# The PyPI clang-tidy does not know the macOS SDK location, so pass it explicitly.
tidy_args=()
if [[ "$(uname)" == "Darwin" ]]; then
  tidy_args+=("--extra-arg=-isysroot$(xcrun --show-sdk-path)")
fi
.venv/bin/clang-tidy --quiet -p build/asan "${tidy_args[@]}" $cpp_files

echo "All checks passed."
