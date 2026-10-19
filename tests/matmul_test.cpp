#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "spindle/ops.h"
#include "spindle/tensor.h"
#include "support/reference.h"

namespace {

using spindle::ConstTensorView;
using spindle::matmul;
using spindle::Tensor;
using spindle::test::load_float_tensor;

// Relative tolerance for single ops (docs/SPEC.md). For a dot product the error
// scales with the sum of |a_ik * b_kj|, so that sum is what it is relative to.
constexpr double kRelativeTolerance = 1e-5;

std::filesystem::path case_dir(const std::string& name) {
  return std::filesystem::path(SPINDLE_REFERENCE_DIR) / "ops" / "matmul" / name;
}

double abs_product_sum(const ConstTensorView& a, const ConstTensorView& b, std::size_t row,
                       std::size_t col) {
  double sum = 0.0;
  for (std::size_t k = 0; k < a.dim(1); ++k) {
    sum += std::abs(static_cast<double>(a.at({row, k})) * static_cast<double>(b.at({k, col})));
  }
  return sum;
}

void expect_matches_reference(const std::string& name, bool b_is_transposed) {
  const Tensor a = load_float_tensor(case_dir(name) / "a.bin");
  const Tensor b_stored = load_float_tensor(case_dir(name) / "b.bin");
  const Tensor expected = load_float_tensor(case_dir(name) / "expected.bin");
  const ConstTensorView b = b_is_transposed ? b_stored.view().transpose(0, 1) : b_stored.view();

  Tensor out(expected.shape());
  matmul(a.view(), b, out.view());

  for (std::size_t row = 0; row < out.shape()[0]; ++row) {
    for (std::size_t col = 0; col < out.shape()[1]; ++col) {
      const double tolerance = kRelativeTolerance * abs_product_sum(a.view(), b, row, col);
      ASSERT_NEAR(out.view().at({row, col}), expected.view().at({row, col}), tolerance)
          << name << " at (" << row << ", " << col << ")";
    }
  }
}

}  // namespace

TEST(Matmul, MatchesNumpyRandomSmall) { expect_matches_reference("random_small", false); }
TEST(Matmul, MatchesNumpyAtGpt2Width) { expect_matches_reference("random_gpt2_width", false); }
TEST(Matmul, MatchesNumpyWithTransposedView) { expect_matches_reference("transposed_b", true); }
TEST(Matmul, MatchesNumpyLargeValues) { expect_matches_reference("large_values", false); }
TEST(Matmul, MatchesNumpySmallValues) { expect_matches_reference("small_values", false); }

TEST(Matmul, AllEqualValuesAreExact) {
  const Tensor a = load_float_tensor(case_dir("all_equal") / "a.bin");
  const Tensor b = load_float_tensor(case_dir("all_equal") / "b.bin");
  const Tensor expected = load_float_tensor(case_dir("all_equal") / "expected.bin");
  Tensor out(expected.shape());
  matmul(a.view(), b.view(), out.view());
  for (std::size_t i = 0; i < out.element_count(); ++i) {
    EXPECT_EQ(out.data()[i], expected.data()[i]);
  }
}

TEST(Matmul, EmptyInnerDimensionGivesZeros) {
  const Tensor a({2, 0});
  const Tensor b({0, 3});
  Tensor out({2, 3}, {1, 1, 1, 1, 1, 1});
  matmul(a.view(), b.view(), out.view());
  for (std::size_t i = 0; i < out.element_count(); ++i) {
    EXPECT_EQ(out.data()[i], 0.0F);
  }
}

TEST(Matmul, WritesIntoStridedOutputView) {
  const Tensor a({2, 2}, {1, 2, 3, 4});
  const Tensor identity({2, 2}, {1, 0, 0, 1});
  Tensor out({2, 2});
  matmul(a.view(), identity.view(), out.view().transpose(0, 1));
  EXPECT_EQ(out.view().at({0, 1}), 3.0F);
  EXPECT_EQ(out.view().at({1, 0}), 2.0F);
}

TEST(Matmul, RejectsShapeMismatch) {
  const Tensor a({2, 3});
  const Tensor b({4, 5});
  Tensor out({2, 5});
  EXPECT_THROW(matmul(a.view(), b.view(), out.view()), std::invalid_argument);
  const Tensor b_ok({3, 5});
  Tensor out_wrong({2, 4});
  EXPECT_THROW(matmul(a.view(), b_ok.view(), out_wrong.view()), std::invalid_argument);
  const Tensor vector({3});
  EXPECT_THROW(matmul(vector.view(), b_ok.view(), out.view()), std::invalid_argument);
}
