# spindle

A from-scratch LLM inference engine in C++20 that runs GPT-2 and TinyLlama on a laptop CPU, with no ML libraries. See [docs/SPEC.md](docs/SPEC.md) for the goals and milestones.

## Build and check

Requires CMake 3.25+, a C++20 compiler and Python 3 (for the pinned clang-format and clang-tidy).

```sh
./tools/check.sh
```

This is the same command CI runs. It checks formatting, builds and tests both the Release and the AddressSanitizer/UBSan builds, fails if any test is skipped, and runs clang-tidy.

To build and run the CLI on its own:

```sh
cmake --preset release && cmake --build --preset release
./build/release/spindle --version
```
