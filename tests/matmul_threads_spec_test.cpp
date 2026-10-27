#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

#include "spindle/ops.h"

namespace {

void fill_random(spindle::Tensor& tensor, std::mt19937& generator) {
  std::normal_distribution<float> normal;
  for (std::size_t i = 0; i < tensor.element_count(); ++i) {
    tensor.data()[i] = normal(generator);
  }
}

void compare_pools(const spindle::ConstTensorView& a, const spindle::ConstTensorView& b) {
  spindle::Tensor expected({a.dim(0), b.dim(1)});
  spindle::matmul(a, b, expected.view(), nullptr);
  for (const std::size_t threads : std::array<std::size_t, 5>{1, 2, 3, 4, 8}) {
    SCOPED_TRACE(threads);
    spindle::ThreadPool pool(threads);
    spindle::Tensor actual(expected.shape());
    std::fill_n(actual.data(), actual.element_count(), std::numeric_limits<float>::quiet_NaN());
    spindle::matmul(a, b, actual.view(), &pool);
    for (std::size_t i = 0; i < actual.element_count(); ++i) {
      ASSERT_EQ(std::bit_cast<std::uint32_t>(actual.data()[i]),
                std::bit_cast<std::uint32_t>(expected.data()[i]))
          << "element=" << i;
    }
  }
}

TEST(MatmulThreadsSpec, ShapesAndBLayoutsAreBitIdentical) {
  // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility.
  std::mt19937 generator(104789);
  for (const auto& shape : std::array<std::array<std::size_t, 3>, 7>{{{1, 768, 3072},
                                                                      {1, 768, 50257},
                                                                      {5, 17, 33},
                                                                      {64, 768, 768},
                                                                      {130, 257, 129},
                                                                      {7, 3, 2},
                                                                      {1, 1, 1}}}) {
    const auto m = shape[0];
    const auto k = shape[1];
    const auto n = shape[2];
    for (int layout = 0; layout < 3; ++layout) {
      SCOPED_TRACE(::testing::Message() << m << "x" << k << "x" << n << " layout=" << layout);
      spindle::Tensor a({m, k});
      spindle::Tensor b(layout == 1 ? spindle::Shape{n, k}
                                    : spindle::Shape{k, n + (layout == 2 ? 3 : 0)});
      fill_random(a, generator);
      fill_random(b, generator);
      const spindle::ConstTensorView bv = [&]() -> spindle::ConstTensorView {
        if (layout == 1) {
          return b.view().transpose(0, 1);
        }
        if (layout == 2) {
          return b.view().slice(1, 1, n + 1);
        }
        return b.view();
      }();
      compare_pools(a.view(), bv);
    }
  }
}

TEST(MatmulThreadsSpec, StridedAAndOutputAreBitIdenticalAndGuarded) {
  // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility.
  std::mt19937 generator(104791);
  spindle::Tensor a({33, 7});
  spindle::Tensor b({19, 33});
  fill_random(a, generator);
  fill_random(b, generator);
  spindle::Tensor expected({19, 9});
  std::fill_n(expected.data(), expected.element_count(), -98765.0F);
  const auto expected_view = expected.view().slice(1, 1, 8).transpose(0, 1);
  spindle::matmul(a.view().transpose(0, 1), b.view().transpose(0, 1), expected_view);
  for (const std::size_t threads : std::array<std::size_t, 5>{1, 2, 3, 4, 8}) {
    spindle::ThreadPool pool(threads);
    spindle::Tensor actual(expected.shape());
    std::fill_n(actual.data(), actual.element_count(), -98765.0F);
    spindle::matmul(a.view().transpose(0, 1), b.view().transpose(0, 1),
                    actual.view().slice(1, 1, 8).transpose(0, 1), &pool);
    for (std::size_t i = 0; i < actual.element_count(); ++i) {
      ASSERT_EQ(std::bit_cast<std::uint32_t>(actual.data()[i]),
                std::bit_cast<std::uint32_t>(expected.data()[i]));
    }
  }
}

TEST(MatmulThreadsSpec, ShapeMismatchStillThrows) {
  const spindle::Tensor a({3, 7});
  const spindle::Tensor b({7, 5});
  const spindle::Tensor wrong_b({8, 5});
  spindle::Tensor out({3, 5});
  spindle::Tensor wrong_rows({4, 5});
  spindle::Tensor wrong_cols({3, 6});
  for (const std::size_t threads : std::array<std::size_t, 5>{1, 2, 3, 4, 8}) {
    spindle::ThreadPool pool(threads);
    EXPECT_THROW(spindle::matmul(a.view(), wrong_b.view(), out.view(), &pool),
                 std::invalid_argument);
    EXPECT_THROW(spindle::matmul(a.view(), b.view(), wrong_rows.view(), &pool),
                 std::invalid_argument);
    EXPECT_THROW(spindle::matmul(a.view(), b.view(), wrong_cols.view(), &pool),
                 std::invalid_argument);
  }
}

}  // namespace
