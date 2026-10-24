// GPT-2 forward pass and greedy generation against Hugging Face Transformers
// (dumps from tools/reference.py). Every stage is compared, so a bug is located
// to one layer.
#include "spindle/gpt2.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "support/reference.h"

namespace {

using spindle::ConstTensorView;
using spindle::Gpt2Model;
using spindle::KvCache;
using spindle::Tensor;
using spindle::test::load_float_tensor;
using spindle::test::load_reference_tensor;

constexpr int kPromptCount = 5;
constexpr std::size_t kGreedyTokens = 100;
// Float32 rounding accumulates over 12 layers, so the spec allows 1e-4 for a full
// forward pass rather than the 1e-5 for single ops.
constexpr double kTolerance = 1e-4;

std::filesystem::path model_dir() {
  return std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" / "model";
}

std::filesystem::path prompt_dir(int prompt) {
  return std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" /
         ("prompt_" + std::to_string(prompt));
}

// Loading copies ~500 MB of weights, so the tests in one process share a model.
const Gpt2Model& model() {
  static const Gpt2Model gpt2 = Gpt2Model::load(model_dir());
  return gpt2;
}

std::vector<std::int32_t> prompt_tokens(int prompt) {
  return load_reference_tensor(prompt_dir(prompt) / "tokens.bin").ints;
}

// Largest error relative to the scale of the row it is in (the row's largest
// magnitude). GPT-2's residual stream has a few huge channels, and errors scale
// with them through LayerNorm and the matmuls, not with each element's own size.
double max_row_relative_error(const ConstTensorView& actual, const Tensor& expected) {
  const std::size_t width = expected.shape().back();
  double worst = 0.0;
  for (std::size_t row = 0; row * width < expected.element_count(); ++row) {
    const float* expected_row = expected.data() + (row * width);
    const float* actual_row = actual.data() + (row * width);
    double scale = 0.0;
    for (std::size_t i = 0; i < width; ++i) {
      scale = std::max(scale, std::abs(static_cast<double>(expected_row[i])));
    }
    for (std::size_t i = 0; i < width; ++i) {
      const double error = std::abs(static_cast<double>(actual_row[i]) - expected_row[i]);
      worst = std::max(worst, error / scale);
    }
  }
  return worst;
}

class Gpt2Forward : public testing::TestWithParam<int> {};
class Gpt2Greedy : public testing::TestWithParam<int> {};

}  // namespace

TEST_P(Gpt2Forward, MatchesReferenceAtEveryStage) {
  const std::vector<std::int32_t> tokens = prompt_tokens(GetParam());
  std::map<std::string, Tensor> stages;
  KvCache cache(model().config(), tokens.size());
  const Tensor hidden = model().forward(
      tokens, cache, [&stages](std::string_view name, const ConstTensorView& value) {
        Tensor copy(value.shape());
        std::copy_n(value.data(), value.element_count(), copy.data());
        stages.emplace(name, std::move(copy));
      });
  ASSERT_EQ(stages.size(), model().config().layers + 2);
  stages.emplace("logits", model().logits(hidden.view()));

  for (const auto& [name, actual] : stages) {
    const Tensor expected = load_float_tensor(prompt_dir(GetParam()) / (name + ".bin"));
    ASSERT_EQ(actual.shape(), expected.shape()) << name;
    const double error = max_row_relative_error(actual.view(), expected);
    std::printf("%-9s max row-relative error %.2e\n", name.c_str(), error);
    EXPECT_LE(error, kTolerance) << name;
  }
}

INSTANTIATE_TEST_SUITE_P(AllPrompts, Gpt2Forward, testing::Range(0, kPromptCount));

TEST_P(Gpt2Greedy, MatchesReferenceFor100Tokens) {
  const std::vector<std::int32_t> expected =
      load_reference_tensor(prompt_dir(GetParam()) / "greedy.bin").ints;
  ASSERT_EQ(expected.size(), kGreedyTokens);
  EXPECT_EQ(spindle::generate_greedy(model(), prompt_tokens(GetParam()), kGreedyTokens), expected);
}

INSTANTIATE_TEST_SUITE_P(AllPrompts, Gpt2Greedy, testing::Range(0, kPromptCount));

// Feeding tokens one at a time through the cache must give exactly the logits of
// one pass over the whole sequence: the same sums in the same order.
TEST(Gpt2Cache, IncrementalMatchesFullPass) {
  const std::vector<std::int32_t> tokens = prompt_tokens(2);
  constexpr std::size_t kPrefill = 10;
  ASSERT_GT(tokens.size(), kPrefill);

  KvCache full_cache(model().config(), tokens.size());
  const Tensor full = model().logits(model().forward(tokens, full_cache).view());

  KvCache cache(model().config(), tokens.size());
  const Tensor prefill =
      model().logits(model().forward(std::span(tokens).first(kPrefill), cache).view());
  const std::size_t vocab = model().config().vocab_size;
  ASSERT_TRUE(std::equal(prefill.data(), prefill.data() + prefill.element_count(), full.data()));
  for (std::size_t position = kPrefill; position < tokens.size(); ++position) {
    const Tensor step =
        model().logits(model().forward(std::span(tokens).subspan(position, 1), cache).view());
    ASSERT_TRUE(std::equal(step.data(), step.data() + vocab, full.data() + (position * vocab)))
        << "position " << position;
  }
  EXPECT_EQ(cache.length(), tokens.size());
}

TEST(Gpt2Cache, RejectsOverflow) {
  KvCache cache(model().config(), 2);
  const std::vector<std::int32_t> tokens = {1, 2, 3};
  EXPECT_THROW((void)model().forward(tokens, cache), std::out_of_range);
  EXPECT_EQ(cache.length(), 0U);
  EXPECT_THROW(KvCache(model().config(), model().config().max_positions + 1),
               std::invalid_argument);
}

TEST(Gpt2Model, RejectsUnknownToken) {
  KvCache cache(model().config(), 1);
  const std::vector<std::int32_t> tokens = {50257};
  EXPECT_THROW((void)model().forward(tokens, cache), std::out_of_range);
}

TEST(Gpt2Model, ReportsMissingModel) {
  EXPECT_THROW((void)Gpt2Model::load(std::filesystem::path(testing::TempDir()) / "no_model"),
               std::runtime_error);
}
