#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "spindle/ops.h"
#include "spindle/tensor.h"

namespace {

spindle::Tensor random_tensor(spindle::Shape shape, unsigned seed) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> distribution;
  std::vector<float> values(spindle::element_count(shape));
  for (float& value : values) {
    value = distribution(generator);
  }
  return {std::move(shape), values};
}

// Args: rows (M), inner (K), cols (N). Reports GFLOPS as 2*M*K*N per run.
void BM_MatmulNaive(benchmark::State& state) {
  const auto rows = static_cast<std::size_t>(state.range(0));
  const auto inner = static_cast<std::size_t>(state.range(1));
  const auto cols = static_cast<std::size_t>(state.range(2));
  const spindle::Tensor a = random_tensor({rows, inner}, 1);
  const spindle::Tensor b = random_tensor({inner, cols}, 2);
  spindle::Tensor out({rows, cols});

  for (auto _ : state) {
    spindle::matmul(a.view(), b.view(), out.view());
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  const double flops_per_run = 2.0 * static_cast<double>(rows * inner * cols);
  state.counters["GFLOPS"] =
      benchmark::Counter(flops_per_run / 1e9, benchmark::Counter::kIsIterationInvariantRate);
}

}  // namespace

BENCHMARK(BM_MatmulNaive)
    ->ArgNames({"M", "K", "N"})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    // GPT-2 small shapes: one decode token through the MLP, and a 64-token prompt.
    ->Args({1, 768, 3072})
    ->Args({64, 768, 768})
    ->Unit(benchmark::kMillisecond);
