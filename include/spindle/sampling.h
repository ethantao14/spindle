#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace spindle {

// PCG32 (pcg_basic's pcg32_srandom_r and pcg32_random_r), so a seed gives the same
// numbers on every compiler and platform.
class Pcg32 {
 public:
  explicit Pcg32(std::uint64_t seed, std::uint64_t stream = 54);

  std::uint32_t next();
  // Uniform in [0, 1): the top 24 bits of next() times 2^-24.
  float next_float();

 private:
  std::uint64_t state_ = 0;
  std::uint64_t increment_ = 0;
};

// Each filter is off at its default. temperature 0 means greedy decoding.
struct SamplingConfig {
  float temperature = 1.0F;
  std::size_t top_k = 0;  // keep the k most likely tokens; 0 keeps all
  float top_p = 1.0F;     // keep the smallest most-likely set with this much probability
};

// Greedy returns the first maximum. Otherwise ranks softmax(logits / temperature)
// (ties by lower id), keeps top_k, then the shortest prefix reaching top_p, and
// draws from what is left by inverse CDF with one next_float(). Throws
// std::invalid_argument for empty logits, temperature < 0, or top_p outside (0, 1].
std::int32_t sample(std::span<const float> logits, const SamplingConfig& config, Pcg32& rng);

}  // namespace spindle
