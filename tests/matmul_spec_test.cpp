#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "spindle/ops.h"
#include "spindle/tensor.h"

namespace {

using spindle::ConstTensorView;
using spindle::matmul;
using spindle::Tensor;
using spindle::TensorView;

constexpr std::array<std::size_t, 14> kEdges{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 64, 65};
constexpr std::array<std::size_t, 6> kInner{1, 3, 4, 17, 64, 129};

void fill_random(Tensor& tensor, std::mt19937& generator) {
  std::normal_distribution<float> normal;
  for (std::size_t i = 0; i < tensor.element_count(); ++i) {
    tensor.data()[i] = normal(generator);
  }
}

void expect_reference(const ConstTensorView& a, const ConstTensorView& b, const TensorView& out) {
  std::vector<double> expected(out.element_count());
  double max_reference = 0;
  for (std::size_t i = 0; i < a.dim(0); ++i) {
    for (std::size_t j = 0; j < b.dim(1); ++j) {
      double sum = 0;
      for (std::size_t k = 0; k < a.dim(1); ++k) {
        sum += static_cast<double>(a.data()[(i * a.strides()[0]) + (k * a.strides()[1])]) *
               static_cast<double>(b.data()[(k * b.strides()[0]) + (j * b.strides()[1])]);
      }
      expected[(i * b.dim(1)) + j] = sum;
      max_reference = std::max(max_reference, std::abs(sum));
    }
  }
  matmul(a, b, out);
  // A global absolute bound tolerates float accumulation and cancellation near zero.
  const double tolerance = 1e-5 * (1 + max_reference);
  for (std::size_t i = 0; i < out.dim(0); ++i) {
    for (std::size_t j = 0; j < out.dim(1); ++j) {
      ASSERT_NEAR(out.at({i, j}), expected[(i * out.dim(1)) + j], tolerance)
          << "at (" << i << ", " << j << ")";
    }
  }
}

void random_case(std::size_t m, std::size_t k, std::size_t n, bool transpose_b,
                 std::mt19937& generator) {
  SCOPED_TRACE(::testing::Message() << m << "x" << k << "x" << n << " transposed=" << transpose_b);
  Tensor a({m, k});
  Tensor b(transpose_b ? spindle::Shape{n, k} : spindle::Shape{k, n});
  Tensor out({m, n});
  fill_random(a, generator);
  fill_random(b, generator);
  std::fill_n(out.data(), out.element_count(), std::numeric_limits<float>::quiet_NaN());
  expect_reference(a.view(), transpose_b ? b.view().transpose(0, 1) : b.view(), out.view());
}

TEST(MatmulSpec, AllRemainders) {
  // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility.
  std::mt19937 generator(104729);
  for (const auto m : kEdges) {
    for (const auto n : kEdges) {
      for (const auto k : kInner) {
        random_case(m, k, n, false, generator);
      }
    }
  }
}

TEST(MatmulSpec, LargerBlocks) {
  // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility.
  std::mt19937 generator(104743);
  for (const auto& shape : std::array<std::array<std::size_t, 3>, 4>{
           {{1, 768, 3072}, {7, 768, 2304}, {130, 257, 129}, {256, 512, 384}}}) {
    random_case(shape[0], shape[1], shape[2], false, generator);
  }
}

TEST(MatmulSpec, TransposedBAndLogits) {
  // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility.
  std::mt19937 generator(104759);
  for (const auto n : kEdges) {
    for (const auto k : kInner) {
      random_case(7, k, n, true, generator);
      random_case(n, k, 5, true, generator);
    }
  }
  random_case(130, 257, 129, true, generator);
  random_case(1, 768, 50257, true, generator);
}

TEST(MatmulSpec, SlicedInputsAndGuardedOutput) {
  // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility.
  std::mt19937 generator(104761);
  constexpr float kSentinel = -98765.0F;
  for (const bool transpose_b : {false, true}) {
    Tensor a({17, 36});
    Tensor b(transpose_b ? spindle::Shape{19, 38} : spindle::Shape{33, 24});
    Tensor storage({19, 26});
    fill_random(a, generator);
    fill_random(b, generator);
    std::fill_n(storage.data(), storage.element_count(), kSentinel);
    const auto av = a.view().slice(1, 1, 34);
    const auto bv =
        transpose_b ? b.view().slice(1, 3, 36).transpose(0, 1) : b.view().slice(1, 2, 21);
    const auto out = storage.view().slice(0, 1, 18).slice(1, 3, 22);
    expect_reference(av, bv, out);
    for (std::size_t i = 0; i < 19; ++i) {
      for (std::size_t j = 0; j < 26; ++j) {
        if (i == 0 || i == 18 || j < 3 || j >= 22) {
          EXPECT_EQ(storage.view().at({i, j}), kSentinel);
        }
      }
    }
  }
}

TEST(MatmulSpec, ArbitraryStridesAndTransposedOutput) {
  Tensor a({33, 7});
  Tensor b({19, 33});
  Tensor out({19, 7});
  // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility.
  std::mt19937 generator(104773);
  fill_random(a, generator);
  fill_random(b, generator);
  expect_reference(a.view().transpose(0, 1), b.view().transpose(0, 1), out.view().transpose(0, 1));
}

TEST(MatmulSpec, OverwritesAndIsBitIdentical) {
  // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility.
  std::mt19937 generator(104779);
  for (const bool transpose_b : {false, true}) {
    Tensor a({17, 129});
    Tensor b(transpose_b ? spindle::Shape{33, 129} : spindle::Shape{129, 33});
    Tensor first({17, 33});
    Tensor second({17, 33});
    fill_random(a, generator);
    fill_random(b, generator);
    std::fill_n(first.data(), first.element_count(), std::numeric_limits<float>::quiet_NaN());
    std::fill_n(second.data(), second.element_count(), 12345.0F);
    const ConstTensorView bv = transpose_b ? b.view().transpose(0, 1) : b.view();
    expect_reference(a.view(), bv, first.view());
    matmul(a.view(), bv, second.view());
    for (std::size_t i = 0; i < first.element_count(); ++i) {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(first.data()[i]),
                std::bit_cast<std::uint32_t>(second.data()[i]));
    }
  }
}

TEST(MatmulSpec, SpecialValuesStayInTheirRows) {
  for (const bool transpose_b : {false, true}) {
    for (const float special :
         {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
      Tensor a({7, 17});
      Tensor b(transpose_b ? spindle::Shape{19, 17} : spindle::Shape{17, 19});
      Tensor out({7, 19});
      std::fill_n(a.data(), a.element_count(), 1.0F);
      std::fill_n(b.data(), b.element_count(), 2.0F);
      a.view().at({3, 16}) = special;
      matmul(a.view(), transpose_b ? b.view().transpose(0, 1) : b.view(), out.view());
      for (std::size_t i = 0; i < 7; ++i) {
        for (std::size_t j = 0; j < 19; ++j) {
          const float value = out.view().at({i, j});
          if (i != 3) {
            EXPECT_EQ(value, 34.0F);
          } else if (std::isnan(special)) {
            EXPECT_TRUE(std::isnan(value));
          } else {
            EXPECT_EQ(value, special);
          }
        }
      }
    }
  }
}

TEST(MatmulSpec, ZerosAndExactCancellation) {
  for (const auto k : std::array<std::size_t, 6>{0, 3, 4, 17, 64, 129}) {
    for (const bool transpose_b : {false, true}) {
      Tensor a({5, k});
      Tensor b(transpose_b ? spindle::Shape{7, k} : spindle::Shape{k, 7});
      Tensor out({5, 7});
      std::fill_n(b.data(), b.element_count(), 1.0F);
      const ConstTensorView bv = transpose_b ? b.view().transpose(0, 1) : b.view();
      for (const bool cancellation : {false, true}) {
        for (std::size_t i = 0; i < 5; ++i) {
          for (std::size_t p = 0; p < k; ++p) {
            float value = 0.0F;
            if (cancellation && p < k - (k % 2)) {
              value = p % 2 == 0 ? 1.0F : -1.0F;
            }
            a.view().at({i, p}) = value;
          }
        }
        std::fill_n(out.data(), out.element_count(), 987.0F);
        matmul(a.view(), bv, out.view());
        for (std::size_t i = 0; i < out.element_count(); ++i) {
          EXPECT_EQ(out.data()[i], 0.0F);
        }
      }
    }
  }
}

TEST(MatmulSpec, RejectsShapeMismatch) {
  const Tensor a({3, 7});
  const Tensor b({7, 5});
  const Tensor wrong_b({8, 5});
  Tensor out({3, 5});
  Tensor wrong_rows({4, 5});
  Tensor wrong_cols({3, 6});
  EXPECT_THROW(matmul(a.view(), wrong_b.view(), out.view()), std::invalid_argument);
  EXPECT_THROW(matmul(a.view(), b.view(), wrong_rows.view()), std::invalid_argument);
  EXPECT_THROW(matmul(a.view(), b.view(), wrong_cols.view()), std::invalid_argument);
}

}  // namespace
