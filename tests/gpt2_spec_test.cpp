#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "spindle/gpt2.h"
#include "support/reference.h"

namespace {

std::filesystem::path prompt_dir(int prompt) {
  return std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" /
         ("prompt_" + std::to_string(prompt));
}

const spindle::Gpt2Model& model() {
  static const auto gpt2 =
      spindle::Gpt2Model::load(std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" / "model");
  return gpt2;
}

// Max absolute error divided by max absolute reference value. Float32 errors
// accumulate across layers, so reference forward comparisons allow 1e-4.
void expect_close(const spindle::ConstTensorView& actual,
                  const spindle::test::ReferenceTensor& expected, float tolerance) {
  ASSERT_EQ(expected.dtype, spindle::test::DType::kFloat32);
  ASSERT_EQ(actual.shape(), expected.shape);
  ASSERT_EQ(actual.rank(), 2U);
  float max_error = 0.0F;
  float max_value = 0.0F;
  for (std::size_t row = 0; row < actual.dim(0); ++row) {
    for (std::size_t col = 0; col < actual.dim(1); ++col) {
      const auto index = (row * actual.dim(1)) + col;
      const float value = actual.at({row, col});
      ASSERT_TRUE(std::isfinite(value));
      ASSERT_TRUE(std::isfinite(expected.floats[index]));
      max_error = std::max(max_error, std::abs(value - expected.floats[index]));
      max_value = std::max(max_value, std::abs(expected.floats[index]));
    }
  }
  EXPECT_LE(max_error, tolerance * max_value);
}

class Gpt2SpecTest : public testing::TestWithParam<int> {};

TEST(Gpt2ConfigSpecTest, MatchesGpt2Small) {
  const auto& config = model().config();
  EXPECT_EQ(config.layers, 12U);
  EXPECT_EQ(config.heads, 12U);
  EXPECT_EQ(config.hidden, 768U);
  EXPECT_EQ(config.max_positions, 1024U);
  EXPECT_EQ(config.vocab_size, 50257U);
  EXPECT_FLOAT_EQ(config.layer_norm_epsilon, 1e-5F);
}

TEST(Gpt2CacheSpecTest, TracksLengthAndRejectsCapacityOverflow) {
  const auto& gpt2 = model();
  const std::vector<std::int32_t> tokens = {15496, 11, 616};
  spindle::KvCache cache(gpt2.config(), 3);
  EXPECT_EQ(cache.capacity(), 3U);
  EXPECT_EQ(cache.length(), 0U);
  (void)gpt2.forward(std::span(tokens).first(2), cache);
  EXPECT_EQ(cache.length(), 2U);
  (void)gpt2.forward(std::span(tokens).last(1), cache);
  EXPECT_EQ(cache.length(), 3U);
  EXPECT_THROW((void)gpt2.forward(std::span(tokens).first(1), cache), std::out_of_range);
  spindle::KvCache short_cache(gpt2.config(), 2);
  EXPECT_THROW((void)gpt2.forward(tokens, short_cache), std::out_of_range);
  EXPECT_THROW((void)spindle::KvCache(gpt2.config(), gpt2.config().max_positions + 1),
               std::invalid_argument);
}

TEST(Gpt2CacheSpecTest, RejectsUnknownTokenIds) {
  for (const std::int32_t token : {-1, 50257}) {
    spindle::KvCache cache(model().config(), 1);
    EXPECT_THROW((void)model().forward(std::span(&token, 1), cache), std::out_of_range);
  }
}

TEST_P(Gpt2SpecTest, ObservesStagesInOrderAndMatchesReferenceLogits) {
  const auto dir = prompt_dir(GetParam());
  const auto tokens = spindle::test::load_reference_tensor(dir / "tokens.bin").ints;
  std::vector<std::string> expected_names = {"embed"};
  for (int layer = 0; layer < 12; ++layer) {
    expected_names.push_back((layer < 10 ? "block_0" : "block_") + std::to_string(layer));
  }
  expected_names.emplace_back("ln_f");
  std::vector<std::string> seen;
  spindle::KvCache cache(model().config(), tokens.size());
  const auto hidden = model().forward(
      tokens, cache, [&](std::string_view name, const spindle::ConstTensorView& value) {
        seen.emplace_back(name);
        ASSERT_LE(seen.size(), expected_names.size());
        EXPECT_EQ(name, expected_names[seen.size() - 1]);
        // Use the expected name so an unexpected callback cannot select a file.
        expect_close(
            value,
            spindle::test::load_reference_tensor(dir / (expected_names[seen.size() - 1] + ".bin")),
            1e-4F);
      });
  EXPECT_EQ(seen, expected_names);
  EXPECT_EQ(hidden.shape(), (spindle::Shape{tokens.size(), 768}));
  expect_close(hidden.view(), spindle::test::load_reference_tensor(dir / "ln_f.bin"), 1e-4F);
  const auto logits = model().logits(hidden.view());
  expect_close(logits.view(), spindle::test::load_reference_tensor(dir / "logits.bin"), 1e-4F);
}

TEST_P(Gpt2SpecTest, CachedTokensMatchFullPromptFinalLogits) {
  const auto tokens =
      spindle::test::load_reference_tensor(prompt_dir(GetParam()) / "tokens.bin").ints;
  ASSERT_FALSE(tokens.empty());
  spindle::KvCache full_cache(model().config(), tokens.size());
  const auto full_hidden = model().forward(tokens, full_cache);
  const auto full_logits = model().logits(full_hidden.view());
  ASSERT_EQ(full_logits.shape(), (spindle::Shape{tokens.size(), model().config().vocab_size}));
  spindle::KvCache incremental(model().config(), tokens.size());
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    const auto hidden = model().forward(std::span(tokens).subspan(index, 1), incremental);
    EXPECT_EQ(incremental.length(), index + 1);
    if (index + 1 == tokens.size()) {
      const auto logits = model().logits(hidden.view());
      ASSERT_EQ(logits.shape(), (spindle::Shape{1, model().config().vocab_size}));
      spindle::test::ReferenceTensor expected;
      expected.shape = logits.shape();
      const float* last = full_logits.data() + (index * model().config().vocab_size);
      expected.floats.assign(last, last + model().config().vocab_size);
      // Same model and weights: tighter than the cross-engine reference check.
      expect_close(logits.view(), expected, 1e-5F);
    }
  }
}

TEST_P(Gpt2SpecTest, GreedyContinuationMatchesHuggingFace) {
  const auto dir = prompt_dir(GetParam());
  const auto prompt = spindle::test::load_reference_tensor(dir / "tokens.bin").ints;
  const auto expected = spindle::test::load_reference_tensor(dir / "greedy.bin").ints;
  const auto actual = spindle::generate_greedy(model(), prompt, 100);
  EXPECT_EQ(actual, expected);
  EXPECT_LE(actual.size(), 100U);
  const auto end = std::ranges::find(actual, 50256);
  if (end != actual.end()) {
    EXPECT_EQ(end + 1, actual.end());
  } else {
    EXPECT_EQ(actual.size(), 100U);
  }
}

TEST(Gpt2GenerateSpecTest, ZeroCountReturnsNoTokens) {
  const auto prompt = spindle::test::load_reference_tensor(prompt_dir(0) / "tokens.bin").ints;
  EXPECT_TRUE(spindle::generate_greedy(model(), prompt, 0).empty());
}

INSTANTIATE_TEST_SUITE_P(AllPrompts, Gpt2SpecTest, testing::Range(0, 5));

}  // namespace
