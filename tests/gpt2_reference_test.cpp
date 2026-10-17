// Checks the GPT-2 reference dumps from tools/reference.py load with the
// expected shapes and contents. Later milestones compare the engine to them.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "support/reference.h"

namespace {

using spindle::test::DType;
using spindle::test::load_reference_tensor;
using spindle::test::ReferenceTensor;

constexpr int kPromptCount = 5;
constexpr int kLayerCount = 12;
constexpr std::size_t kHiddenSize = 768;
constexpr std::size_t kVocabSize = 50257;

std::filesystem::path prompt_dir(int prompt) {
  return std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" /
         ("prompt_" + std::to_string(prompt));
}

bool all_finite(const std::vector<float>& values) {
  return std::ranges::all_of(values, [](float value) { return std::isfinite(value); });
}

std::vector<std::string> activation_names() {
  std::vector<std::string> names = {"embed"};
  for (int layer = 0; layer < kLayerCount; ++layer) {
    names.push_back((layer < 10 ? "block_0" : "block_") + std::to_string(layer));
  }
  names.emplace_back("ln_f");
  return names;
}

class Gpt2Reference : public testing::TestWithParam<int> {};

}  // namespace

TEST_P(Gpt2Reference, HasConsistentShapes) {
  const auto dir = prompt_dir(GetParam());
  const ReferenceTensor tokens = load_reference_tensor(dir / "tokens.bin");
  ASSERT_EQ(tokens.dtype, DType::kInt32);
  ASSERT_EQ(tokens.shape.size(), 1U);
  const std::size_t token_count = tokens.shape[0];
  ASSERT_GT(token_count, 0U);
  for (const std::int32_t token : tokens.ints) {
    EXPECT_GE(token, 0);
    EXPECT_LT(static_cast<std::size_t>(token), kVocabSize);
  }

  for (const std::string& name : activation_names()) {
    SCOPED_TRACE(name);
    const ReferenceTensor activation = load_reference_tensor(dir / (name + ".bin"));
    EXPECT_EQ(activation.dtype, DType::kFloat32);
    EXPECT_EQ(activation.shape, (std::vector<std::size_t>{token_count, kHiddenSize}));
    EXPECT_TRUE(all_finite(activation.floats));
  }

  const ReferenceTensor logits = load_reference_tensor(dir / "logits.bin");
  EXPECT_EQ(logits.dtype, DType::kFloat32);
  EXPECT_EQ(logits.shape, (std::vector<std::size_t>{token_count, kVocabSize}));
  EXPECT_TRUE(all_finite(logits.floats));
}

INSTANTIATE_TEST_SUITE_P(AllPrompts, Gpt2Reference, testing::Range(0, kPromptCount));

// Known GPT-2 token IDs, so a wrong tokenizer or a scrambled dump is caught.
TEST(Gpt2ReferenceTokens, HelloPromptMatchesKnownIds) {
  const ReferenceTensor tokens = load_reference_tensor(prompt_dir(1) / "tokens.bin");
  EXPECT_EQ(tokens.ints, (std::vector<std::int32_t>{15496, 11, 616, 1438, 318}));
}
