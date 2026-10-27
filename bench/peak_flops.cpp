#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <string_view>

#if defined(__ARM_NEON)  // NOLINT(readability-use-concise-preprocessor-directives)
#include <arm_neon.h>
#endif

namespace {

// Register-only FP32 vector FMAs measure one core's arithmetic ceiling, excluding memory traffic.
float fma_loop(std::size_t iterations) {
#if defined(__ARM_NEON)  // NOLINT(readability-use-concise-preprocessor-directives)
  std::array<float32x4_t, 16> sums;
  for (std::size_t i = 0; i < sums.size(); ++i) {
    sums[i] = vdupq_n_f32(static_cast<float>(i) * 0.01F);
  }
  const auto multiplier = vdupq_n_f32(0.9999F);
  const auto addend = vdupq_n_f32(0.0001F);
  for (std::size_t i = 0; i < iterations; ++i) {
    sums[0] = vfmaq_f32(sums[0], multiplier, addend);
    sums[1] = vfmaq_f32(sums[1], multiplier, addend);
    sums[2] = vfmaq_f32(sums[2], multiplier, addend);
    sums[3] = vfmaq_f32(sums[3], multiplier, addend);
    sums[4] = vfmaq_f32(sums[4], multiplier, addend);
    sums[5] = vfmaq_f32(sums[5], multiplier, addend);
    sums[6] = vfmaq_f32(sums[6], multiplier, addend);
    sums[7] = vfmaq_f32(sums[7], multiplier, addend);
    sums[8] = vfmaq_f32(sums[8], multiplier, addend);
    sums[9] = vfmaq_f32(sums[9], multiplier, addend);
    sums[10] = vfmaq_f32(sums[10], multiplier, addend);
    sums[11] = vfmaq_f32(sums[11], multiplier, addend);
    sums[12] = vfmaq_f32(sums[12], multiplier, addend);
    sums[13] = vfmaq_f32(sums[13], multiplier, addend);
    sums[14] = vfmaq_f32(sums[14], multiplier, addend);
    sums[15] = vfmaq_f32(sums[15], multiplier, addend);
    // Keep the work in registers and prevent algebraic elimination of repeated iterations.
    asm volatile(""
                 : "+w"(sums[0]), "+w"(sums[1]), "+w"(sums[2]), "+w"(sums[3]), "+w"(sums[4]),
                   "+w"(sums[5]), "+w"(sums[6]), "+w"(sums[7]), "+w"(sums[8]), "+w"(sums[9]),
                   "+w"(sums[10]), "+w"(sums[11]), "+w"(sums[12]), "+w"(sums[13]), "+w"(sums[14]),
                   "+w"(sums[15]));
  }
  float result = 0;
  for (const auto& sum : sums) {
    result += vaddvq_f32(sum);
  }
  return result;
#else
  (void)iterations;
  return 0;
#endif
}

}  // namespace

int main(int argc, char** argv) {
#if defined(__ARM_NEON)  // NOLINT(readability-use-concise-preprocessor-directives)
  const bool smoke = argc == 2 && std::string_view(argv[1]) == "--smoke";
  const std::size_t iterations = smoke ? 100000 : 20000000;
  std::array<double, 7> rates{};
  float checksum = fma_loop(10000);
  for (auto& rate : rates) {
    const auto start = std::chrono::steady_clock::now();
    checksum += fma_loop(iterations);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    rate = static_cast<double>(iterations) * 16 * 4 * 2 / seconds / 1e9;
  }
  std::ranges::sort(rates);
  std::printf("Single-thread NEON FP32 peak: %.2f GFLOPS (median of %zu runs; checksum %.3f)\n",
              rates[rates.size() / 2], rates.size(), static_cast<double>(checksum));
#else
  (void)argc;
  (void)argv;
  std::printf("NEON FP32 peak unavailable on this architecture (checksum %.1f)\n",
              static_cast<double>(fma_loop(0)));
#endif
}
