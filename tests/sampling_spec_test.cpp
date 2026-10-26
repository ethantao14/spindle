#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "spindle/sampling.h"

namespace {

TEST(Pcg32Spec, MatchesPcgBasicVector) {
  spindle::Pcg32 rng(42, 54);
  for (const auto expected :
       {0xa15c02b7U, 0x7b47f409U, 0xba1d3330U, 0x83d2f293U, 0xbfa4784bU, 0xcbed606eU}) {
    EXPECT_EQ(rng.next(), expected);
  }
}

TEST(Pcg32Spec, SeedsAndStreamsControlSequence) {
  spindle::Pcg32 first(7);
  spindle::Pcg32 same(7);
  spindle::Pcg32 other_seed(8);
  spindle::Pcg32 other_stream(7, 55);
  bool seed_differs = false;
  bool stream_differs = false;
  for (int i = 0; i < 1000; ++i) {
    const auto value = first.next();
    EXPECT_EQ(value, same.next());
    seed_differs |= value != other_seed.next();
    stream_differs |= value != other_stream.next();
  }
  EXPECT_TRUE(seed_differs);
  EXPECT_TRUE(stream_differs);
}

TEST(Pcg32Spec, FloatUsesTop24BitsAndIsHalfOpen) {
  spindle::Pcg32 rng(42);
  spindle::Pcg32 parallel(42);
  for (int i = 0; i < 20000; ++i) {
    const float value = rng.next_float();
    EXPECT_GE(value, 0.0F);
    EXPECT_LT(value, 1.0F);
    EXPECT_EQ(value, static_cast<float>(parallel.next() >> 8) * 0x1p-24F);
  }
}

TEST(SamplingSpec, GreedyUsesFirstMaximumAndConsumesNoRandomness) {
  const std::array<float, 4> logits = {-4, 3, 3, 1};
  spindle::Pcg32 rng(7);
  spindle::Pcg32 parallel(7);
  const spindle::SamplingConfig config{.temperature = 0};
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(spindle::sample(logits, config, rng), 1);
  }
  EXPECT_EQ(rng.next(), parallel.next());
}

TEST(SamplingSpec, SingleTokenFiltersAndSmallTemperatureMatchGreedy) {
  spindle::Pcg32 inputs(19);
  spindle::Pcg32 rng(7);
  for (int trial = 0; trial < 100; ++trial) {
    std::array<float, 16> logits{};
    for (auto& value : logits) {
      value = inputs.next_float() * 20.0F;
    }
    const auto best = std::ranges::max_element(logits) - logits.begin();
    EXPECT_EQ(spindle::sample(logits, {.top_k = 1}, rng), best);
    EXPECT_EQ(spindle::sample(logits, {.top_p = 1e-6F}, rng), best);
  }
  const std::array<float, 3> separated = {1, 3, 2};
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(spindle::sample(separated, {.temperature = 1e-6F}, rng), 1);
  }
}

TEST(SamplingSpec, FiltersExcludeTokensOutsideRankedPrefix) {
  // Probabilities by id: .1, .6, .05, .25. Both filters retain ids 1 and 3.
  const std::array<float, 4> logits = {std::log(.1F), std::log(.6F), std::log(.05F),
                                       std::log(.25F)};
  for (const auto config :
       {spindle::SamplingConfig{.top_k = 2}, spindle::SamplingConfig{.top_p = .8F}}) {
    spindle::Pcg32 rng(42);
    std::array<int, 4> counts{};
    for (int i = 0; i < 10000; ++i) {
      const auto token = spindle::sample(logits, config, rng);
      ASSERT_GE(token, 0);
      ASSERT_LT(token, 4);
      ++counts[static_cast<std::size_t>(token)];
    }
    EXPECT_EQ(counts[0], 0);
    EXPECT_EQ(counts[2], 0);
    EXPECT_GT(counts[1], 0);
    EXPECT_GT(counts[3], 0);
  }
}

TEST(SamplingSpec, FrequenciesMatchTemperatureScaledSoftmax) {
  const std::array<float, 3> logits = {-1, 0, 1};
  constexpr int draws = 20000;
  // Absolute .02 is over five worst-case binomial standard deviations.
  constexpr double tolerance = .02;
  for (const float temperature : {1.0F, .5F}) {
    SCOPED_TRACE(temperature);
    spindle::Pcg32 rng(42);
    std::array<int, 3> counts{};
    double total = 0;
    for (const float value : logits) {
      total += std::exp(static_cast<double>(value) / temperature);
    }
    for (int i = 0; i < draws; ++i) {
      const auto token = spindle::sample(logits, {.temperature = temperature}, rng);
      ASSERT_GE(token, 0);
      ASSERT_LT(token, 3);
      ++counts[static_cast<std::size_t>(token)];
    }
    for (std::size_t i = 0; i < logits.size(); ++i) {
      EXPECT_NEAR(static_cast<double>(counts[i]) / draws,
                  std::exp(static_cast<double>(logits[i]) / temperature) / total, tolerance);
    }
  }
}

TEST(SamplingSpec, LargeLogitsRemainStable) {
  spindle::Pcg32 rng(42);
  for (const auto logits :
       {std::array<float, 3>{10000, 9999, -10000}, std::array<float, 3>{-10000, -10001, -20000}}) {
    for (int i = 0; i < 1000; ++i) {
      const auto token = spindle::sample(logits, {}, rng);
      EXPECT_GE(token, 0);
      EXPECT_LT(token, 2);
    }
  }
}

TEST(SamplingSpec, RejectsInvalidInputs) {
  spindle::Pcg32 rng(42);
  const std::array<float, 2> logits = {0, 1};
  EXPECT_THROW((void)spindle::sample({}, {}, rng), std::invalid_argument);
  for (const auto config :
       {spindle::SamplingConfig{.temperature = -1}, spindle::SamplingConfig{.top_p = 0},
        spindle::SamplingConfig{.top_p = 1.01F}}) {
    EXPECT_THROW((void)spindle::sample(logits, config, rng), std::invalid_argument);
  }
}

TEST(SamplingSpec, SameSeedReproducesDrawSequence) {
  const std::array<float, 4> logits = {1, -1, 2, 0};
  spindle::Pcg32 first(7);
  spindle::Pcg32 second(7);
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(spindle::sample(logits, {.temperature = .8F, .top_k = 3, .top_p = .9F}, first),
              spindle::sample(logits, {.temperature = .8F, .top_k = 3, .top_p = .9F}, second));
  }
}

TEST(SamplingSpec, InverseCdfUsesRankOrderAndOneFloat) {
  // Ranked order is id 1 (.6), then id 0 (.4).
  const std::array<float, 2> logits = {std::log(.4F), std::log(.6F)};
  spindle::Pcg32 rng(42);
  spindle::Pcg32 parallel(42);
  const auto expected = parallel.next_float() < .6F ? 1 : 0;
  EXPECT_EQ(spindle::sample(logits, {}, rng), expected);
  EXPECT_EQ(rng.next(), parallel.next());
}

}  // namespace
