include(FetchContent)
FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz
  URL_HASH SHA256=9631341c82bac4a288bef951f8b26b41f69021794184ece969f8473977eaa340
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(spindle_bench matmul_bench.cpp)
target_link_libraries(spindle_bench PRIVATE spindle_core benchmark::benchmark_main)
target_compile_options(spindle_bench PRIVATE ${SPINDLE_WARNINGS})

add_executable(spindle_latency latency.cpp)
target_link_libraries(spindle_latency PRIVATE spindle_core)
target_compile_options(spindle_latency PRIVATE ${SPINDLE_WARNINGS})

add_executable(spindle_peak_flops peak_flops.cpp)
target_compile_options(spindle_peak_flops PRIVATE ${SPINDLE_WARNINGS})
