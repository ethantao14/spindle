// Core ops against NumPy references from tools/op_reference.py. Expected values
// are float64 results rounded to float32. Each op's tolerance is 1e-5 relative
// (docs/SPEC.md); the comment on each says what it is relative to and why.
#include "spindle/ops.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "spindle/tensor.h"
#include "support/reference.h"

namespace {

using spindle::Shape;
using spindle::Tensor;
using spindle::test::load_float_tensor;
using spindle::test::load_reference_tensor;

constexpr double kRelativeTolerance = 1e-5;
constexpr float kLayerNormEpsilon = 1e-5F;
// Below float32's normal range, results cannot keep 1e-5 relative precision.
constexpr double kSmallestNormal = std::numeric_limits<float>::min();

std::filesystem::path case_dir(const std::string& op, const std::string& name) {
  return std::filesystem::path(SPINDLE_REFERENCE_DIR) / "ops" / op / name;
}

// Checks every element; tolerance(i) gives the allowed absolute error at flat index i.
void expect_close(const Tensor& actual, const Tensor& expected,
                  const std::function<double(std::size_t)>& tolerance) {
  ASSERT_EQ(actual.shape(), expected.shape());
  for (std::size_t i = 0; i < actual.element_count(); ++i) {
    ASSERT_NEAR(actual.data()[i], expected.data()[i], tolerance(i)) << "at flat index " << i;
  }
}

class AddReference : public testing::TestWithParam<std::string> {};
class GeluReference : public testing::TestWithParam<std::string> {};
class SoftmaxReference : public testing::TestWithParam<std::string> {};
class LayerNormReference : public testing::TestWithParam<std::string> {};

}  // namespace

// add: one rounding per element, so relative to the result itself.
TEST_P(AddReference, MatchesNumpy) {
  const auto dir = case_dir("add", GetParam());
  const Tensor a = load_float_tensor(dir / "a.bin");
  const Tensor b = load_float_tensor(dir / "b.bin");
  const Tensor expected = load_float_tensor(dir / "expected.bin");
  Tensor out(a.shape());
  spindle::add(a.view(), b.view(), out.view());
  expect_close(out, expected,
               [&](std::size_t i) { return kRelativeTolerance * std::abs(expected.data()[i]); });
}

INSTANTIATE_TEST_SUITE_P(Cases, AddReference,
                         testing::Values("random", "large_values", "small_values", "all_equal",
                                         "bias"));

// gelu: 1 + tanh(...) cancels for negative x, so the error scales with |x|, not
// the tiny result. PyTorch's float32 GELU, the M2 reference, cancels the same way.
TEST_P(GeluReference, MatchesNumpy) {
  const auto dir = case_dir("gelu", GetParam());
  const Tensor in = load_float_tensor(dir / "in.bin");
  const Tensor expected = load_float_tensor(dir / "expected.bin");
  Tensor out(in.shape());
  spindle::gelu(in.view(), out.view());
  expect_close(out, expected, [&](std::size_t i) {
    return kRelativeTolerance * std::max(std::abs(expected.data()[i]), std::abs(in.data()[i]));
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, GeluReference,
                         testing::Values("random", "large_values", "small_values", "all_equal"));

// softmax: relative to each probability, floored at the smallest normal float.
TEST_P(SoftmaxReference, MatchesNumpy) {
  const auto dir = case_dir("softmax", GetParam());
  const Tensor in = load_float_tensor(dir / "in.bin");
  const Tensor expected = load_float_tensor(dir / "expected.bin");
  Tensor out(in.shape());
  spindle::softmax(in.view(), out.view());
  expect_close(out, expected, [&](std::size_t i) {
    return (kRelativeTolerance * std::abs(expected.data()[i])) + kSmallestNormal;
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, SoftmaxReference,
                         testing::Values("random", "gpt2_width", "large_values", "small_values",
                                         "all_equal", "masked", "long_tail"));

// layer_norm: normalized values are order 1 before the gain, so the error is
// relative to the result plus that unit scale times |gamma|.
TEST_P(LayerNormReference, MatchesNumpy) {
  const auto dir = case_dir("layer_norm", GetParam());
  const Tensor in = load_float_tensor(dir / "in.bin");
  const Tensor gamma = load_float_tensor(dir / "gamma.bin");
  const Tensor beta = load_float_tensor(dir / "beta.bin");
  const Tensor expected = load_float_tensor(dir / "expected.bin");
  Tensor out(in.shape());
  spindle::layer_norm(in.view(), gamma.view(), beta.view(), kLayerNormEpsilon, out.view());
  const std::size_t width = gamma.element_count();
  expect_close(out, expected, [&](std::size_t i) {
    return kRelativeTolerance * (std::abs(expected.data()[i]) + std::abs(gamma.data()[i % width]));
  });
}

INSTANTIATE_TEST_SUITE_P(Cases, LayerNormReference,
                         testing::Values("random", "gpt2_width", "large_values", "small_values",
                                         "all_equal"));

// embedding is a copy, so it must be exact.
TEST(Embedding, MatchesNumpyExactly) {
  const auto dir = case_dir("embedding", "random");
  const Tensor table = load_float_tensor(dir / "table.bin");
  const std::vector<std::int32_t> tokens = load_reference_tensor(dir / "tokens.bin").ints;
  const Tensor expected = load_float_tensor(dir / "expected.bin");
  Tensor out(expected.shape());
  spindle::embedding(table.view(), tokens, out.view());
  expect_close(out, expected, [](std::size_t) { return 0.0; });
}

TEST(Embedding, RejectsTokensOutsideTable) {
  const Tensor table({4, 2});
  Tensor out({1, 2});
  const std::vector<std::int32_t> too_large = {4};
  const std::vector<std::int32_t> negative = {-1};
  EXPECT_THROW(spindle::embedding(table.view(), too_large, out.view()), std::out_of_range);
  EXPECT_THROW(spindle::embedding(table.view(), negative, out.view()), std::out_of_range);
}

TEST(Softmax, RowsSumToOne) {
  const Tensor in({2, 3}, {1, 2, 3, -5, 0, 5});
  Tensor out(in.shape());
  spindle::softmax(in.view(), out.view());
  EXPECT_NEAR(out.data()[0] + out.data()[1] + out.data()[2], 1.0F, 1e-6F);
  EXPECT_NEAR(out.data()[3] + out.data()[4] + out.data()[5], 1.0F, 1e-6F);
}

TEST(Ops, RunInPlace) {
  Tensor tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  const Tensor expected({2, 3}, {2, 4, 6, 8, 10, 12});
  spindle::add(tensor.view(), tensor.view(), tensor.view());
  expect_close(tensor, expected, [](std::size_t) { return 0.0; });
}

TEST(Ops, RejectShapeMismatch) {
  const Tensor in({2, 3});
  Tensor wrong_out({3, 2});
  const Tensor wrong_bias({2});
  const Tensor gamma({3});
  Tensor out({2, 3});
  EXPECT_THROW(spindle::add(in.view(), wrong_bias.view(), out.view()), std::invalid_argument);
  EXPECT_THROW(spindle::add(in.view(), in.view(), wrong_out.view()), std::invalid_argument);
  EXPECT_THROW(spindle::softmax(in.view(), wrong_out.view()), std::invalid_argument);
  EXPECT_THROW(spindle::gelu(in.view(), wrong_out.view()), std::invalid_argument);
  EXPECT_THROW(spindle::layer_norm(in.view(), wrong_bias.view(), gamma.view(), 1e-5F, out.view()),
               std::invalid_argument);
  const std::vector<std::int32_t> tokens = {0, 1, 2};
  EXPECT_THROW(spindle::embedding(in.view(), tokens, out.view()), std::invalid_argument);
}

TEST(Ops, RejectNonContiguousViews) {
  const Tensor in({2, 3});
  Tensor out({3, 2});
  EXPECT_THROW(spindle::softmax(in.view().transpose(0, 1), out.view()), std::invalid_argument);
}
