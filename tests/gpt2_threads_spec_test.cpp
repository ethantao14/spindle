#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "spindle/gpt2.h"
#include "support/reference.h"

namespace {

std::filesystem::path model_dir() {
  return std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" / "model";
}

void expect_bits_equal(const spindle::ConstTensorView& actual,
                       const spindle::ConstTensorView& expected) {
  ASSERT_EQ(actual.shape(), expected.shape());
  ASSERT_EQ(actual.rank(), 2U);
  for (std::size_t row = 0; row < actual.dim(0); ++row) {
    for (std::size_t col = 0; col < actual.dim(1); ++col) {
      ASSERT_EQ(std::bit_cast<std::uint32_t>(actual.at({row, col})),
                std::bit_cast<std::uint32_t>(expected.at({row, col})))
          << "row=" << row << " col=" << col;
    }
  }
}

TEST(Gpt2ThreadsSpec, ForwardLogitsAndSamplingAreBitIdentical) {
  const auto baseline = spindle::Gpt2Model::load(model_dir(), 1);
  EXPECT_EQ(baseline.threads(), 1U);
  const spindle::SamplingConfig config{.temperature = 0.8F};
  for (const std::size_t threads : std::array<std::size_t, 2>{4, 3}) {
    SCOPED_TRACE(threads);
    const auto parallel = spindle::Gpt2Model::load(model_dir(), threads);
    EXPECT_EQ(parallel.threads(), threads);
    for (const int prompt : {0, 1}) {
      SCOPED_TRACE(prompt);
      const auto tokens = spindle::test::load_reference_tensor(
                              std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" /
                              ("prompt_" + std::to_string(prompt)) / "tokens.bin")
                              .ints;
      spindle::KvCache baseline_cache(baseline.config(), tokens.size());
      spindle::KvCache parallel_cache(parallel.config(), tokens.size());
      const auto expected_hidden = baseline.forward(tokens, baseline_cache);
      const auto actual_hidden = parallel.forward(tokens, parallel_cache);
      expect_bits_equal(actual_hidden.view(), expected_hidden.view());
      const auto expected_logits = baseline.logits(expected_hidden.view());
      const auto actual_logits = parallel.logits(actual_hidden.view());
      expect_bits_equal(actual_logits.view(), expected_logits.view());
      const auto expected_tokens = spindle::generate(baseline, tokens, 5, config, 7);
      ASSERT_FALSE(expected_tokens.empty());
      EXPECT_EQ(spindle::generate(parallel, tokens, 5, config, 7), expected_tokens);
    }
  }
}

TEST(Gpt2ThreadsSpec, ZeroThreadsThrows) {
  EXPECT_THROW((void)spindle::Gpt2Model::load(model_dir(), 0), std::invalid_argument);
}

}  // namespace
