#include "spindle/sampling.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace spindle {

// Seed and stream follow the agreed PCG interface.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Pcg32::Pcg32(std::uint64_t seed, std::uint64_t stream) : increment_((stream << 1U) | 1U) {
  next();
  state_ += seed;
  next();
}

std::uint32_t Pcg32::next() {
  const std::uint64_t old = state_;
  state_ = (old * 6364136223846793005ULL) + increment_;
  const auto shifted = static_cast<std::uint32_t>(((old >> 18U) ^ old) >> 27U);
  const auto rotation = static_cast<std::uint32_t>(old >> 59U);
  return (shifted >> rotation) | (shifted << ((0U - rotation) & 31U));
}

float Pcg32::next_float() { return static_cast<float>(next() >> 8U) * 0x1p-24F; }

std::int32_t sample(std::span<const float> logits, const SamplingConfig& config, Pcg32& rng) {
  if (logits.empty() || !std::isfinite(config.temperature) || config.temperature < 0 ||
      !std::isfinite(config.top_p) || config.top_p <= 0 || config.top_p > 1) {
    throw std::invalid_argument("invalid sampling logits or configuration");
  }
  const auto best = std::ranges::max_element(logits);
  if (config.temperature == 0) {
    return static_cast<std::int32_t>(best - logits.begin());
  }
  std::vector<std::size_t> ranked(logits.size());
  std::iota(ranked.begin(), ranked.end(), 0);
  std::ranges::sort(ranked, [&](std::size_t left, std::size_t right) {
    return logits[left] == logits[right] ? left < right : logits[left] > logits[right];
  });
  if (config.top_k != 0 && config.top_k < ranked.size()) {
    ranked.resize(config.top_k);
  }
  std::vector<double> weights;
  weights.reserve(ranked.size());
  double total = 0;
  for (const auto id : ranked) {
    const double weight = std::exp((static_cast<double>(logits[id]) - *best) / config.temperature);
    weights.push_back(weight);
    total += weight;
  }
  const double threshold = total * config.top_p;
  double retained = 0;
  std::size_t keep = 0;
  do {
    retained += weights[keep++];
  } while (keep < weights.size() && retained < threshold);
  const double draw = rng.next_float() * retained;
  double cumulative = 0;
  for (std::size_t i = 0; i < keep; ++i) {
    cumulative += weights[i];
    if (draw < cumulative) {
      return static_cast<std::int32_t>(ranked[i]);
    }
  }
  return static_cast<std::int32_t>(ranked[keep - 1]);
}

}  // namespace spindle
