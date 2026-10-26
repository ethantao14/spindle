#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

#include "spindle/gpt2.h"
#include "support/reference.h"

namespace {

const spindle::Gpt2Model& sampling_model() {
  static const auto model =
      spindle::Gpt2Model::load(std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" / "model");
  return model;
}

std::vector<std::int32_t> reference_tokens(const char* filename) {
  return spindle::test::load_reference_tensor(std::filesystem::path(SPINDLE_REFERENCE_DIR) /
                                              "gpt2" / "prompt_1" / filename)
      .ints;
}

TEST(GenerateSamplingSpec, SeedControlsContinuationAndCallbacksMatchReturnedTokens) {
  const auto prompt = reference_tokens("tokens.bin");
  const spindle::SamplingConfig config{.temperature = 1};
  std::vector<std::int32_t> observed;
  const auto first = spindle::generate(sampling_model(), prompt, 30, config, 7,
                                       [&](std::int32_t token) { observed.push_back(token); });
  EXPECT_FALSE(first.empty());
  EXPECT_EQ(observed, first);
  EXPECT_EQ(first, spindle::generate(sampling_model(), prompt, 30, config, 7));
  EXPECT_NE(first, spindle::generate(sampling_model(), prompt, 30, config, 8));
}

TEST(GenerateSamplingSpec, GreedyAndTopOneMatchReference) {
  const auto prompt = reference_tokens("tokens.bin");
  auto expected = reference_tokens("greedy.bin");
  ASSERT_GE(expected.size(), 5U);
  expected.resize(5);
  const auto greedy = spindle::generate_greedy(sampling_model(), prompt, 5);
  EXPECT_EQ(greedy, expected);
  EXPECT_EQ(spindle::generate(sampling_model(), prompt, 5, {.temperature = 0}, 7), greedy);
  EXPECT_EQ(spindle::generate(sampling_model(), prompt, 5, {.top_k = 1}, 8), greedy);
}

TEST(GenerateSamplingSpec, UsesLastPositionAndDefaultStreamOncePerToken) {
  const auto prompt = reference_tokens("tokens.bin");
  const auto& model = sampling_model();
  const spindle::SamplingConfig config{.temperature = .8F, .top_k = 40};
  spindle::Pcg32 rng(7);
  spindle::KvCache cache(model.config(), prompt.size() + 3);
  auto hidden = model.forward(prompt, cache);
  std::vector<std::int32_t> expected;
  for (int i = 0; i < 3; ++i) {
    const auto logits = model.logits(hidden.view());
    const auto width = model.config().vocab_size;
    const auto row = logits.shape()[0] - 1;
    const auto token =
        spindle::sample(std::span<const float>(logits.data() + (row * width), width), config, rng);
    expected.push_back(token);
    if (token == 50256 || i == 2) {
      break;
    }
    hidden = model.forward(std::span<const std::int32_t>(&token, 1), cache);
  }
  EXPECT_EQ(spindle::generate(model, prompt, 3, config, 7), expected);
}

TEST(GenerateSamplingSpec, RejectsEmptyPromptAndBadConfig) {
  EXPECT_THROW((void)spindle::generate(sampling_model(), {}, 1, {}, 0), std::invalid_argument);
  const auto prompt = reference_tokens("tokens.bin");
  for (const auto config :
       {spindle::SamplingConfig{.temperature = -1}, spindle::SamplingConfig{.top_p = 0},
        spindle::SamplingConfig{.top_p = 1.01F}}) {
    EXPECT_THROW((void)spindle::generate(sampling_model(), prompt, 1, config, 0),
                 std::invalid_argument);
  }
}

}  // namespace
