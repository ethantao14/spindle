#include "spindle/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using spindle::ConstTensorView;
using spindle::Shape;
using spindle::Strides;
using spindle::Tensor;
using spindle::TensorView;

// 2x3 tensor holding 0..5 in row-major order.
Tensor counting_tensor() { return {{2, 3}, {0, 1, 2, 3, 4, 5}}; }

}  // namespace

TEST(Tensor, AllocationIs64ByteAligned) {
  // Many live allocations of varied sizes, so a weaker alignment cannot pass by luck.
  std::vector<Tensor> tensors;
  for (std::size_t size = 1; size <= 200; ++size) {
    tensors.emplace_back(Shape{size});
  }
  for (const Tensor& tensor : tensors) {
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(tensor.data()) % spindle::kTensorAlignment, 0U);
  }
}

TEST(Tensor, StartsZeroed) {
  // macOS often hands out zeroed memory anyway; the sanitizer build fills new
  // allocations with 0xbe, so that build is where a missing zero fill fails.
  const Tensor tensor({4096});
  for (std::size_t i = 0; i < tensor.element_count(); ++i) {
    ASSERT_EQ(tensor.data()[i], 0.0F) << "index " << i;
  }
}

TEST(Tensor, RejectsWrongValueCount) {
  EXPECT_THROW(Tensor({2, 2}, {1, 2, 3}), std::invalid_argument);
}

TEST(Tensor, RejectsShapeThatOverflows) {
  EXPECT_THROW(Tensor({std::size_t{1} << 40U, std::size_t{1} << 40U}), std::overflow_error);
}

TEST(TensorView, ContiguousViewHasRowMajorStrides) {
  Tensor tensor({2, 3, 4});
  const TensorView view = tensor.view();
  EXPECT_EQ(view.shape(), (Shape{2, 3, 4}));
  EXPECT_EQ(view.strides(), (Strides{12, 4, 1}));
  EXPECT_TRUE(view.is_contiguous());
  EXPECT_EQ(view.element_count(), 24U);
}

TEST(TensorView, AtReadsRowMajorElements) {
  const Tensor tensor = counting_tensor();
  EXPECT_EQ(tensor.view().at({0, 0}), 0.0F);
  EXPECT_EQ(tensor.view().at({0, 2}), 2.0F);
  EXPECT_EQ(tensor.view().at({1, 0}), 3.0F);
  EXPECT_EQ(tensor.view().at({1, 2}), 5.0F);
}

TEST(TensorView, AtRejectsOutOfRangeIndex) {
  const Tensor tensor = counting_tensor();
  EXPECT_THROW((void)tensor.view().at({2, 0}), std::out_of_range);
  EXPECT_THROW((void)tensor.view().at({0, 3}), std::out_of_range);
  EXPECT_THROW((void)tensor.view().at({0}), std::out_of_range);
}

TEST(TensorView, ViewsShareStorageWithoutCopying) {
  Tensor tensor = counting_tensor();
  const TensorView transposed = tensor.view().transpose(0, 1);
  EXPECT_EQ(transposed.data(), tensor.data());
  transposed.at({2, 1}) = 42.0F;
  EXPECT_EQ(tensor.view().at({1, 2}), 42.0F);
}

TEST(TensorView, TransposeSwapsShapeAndStrides) {
  const Tensor tensor = counting_tensor();
  const ConstTensorView transposed = tensor.view().transpose(0, 1);
  EXPECT_EQ(transposed.shape(), (Shape{3, 2}));
  EXPECT_EQ(transposed.strides(), (Strides{1, 3}));
  EXPECT_FALSE(transposed.is_contiguous());
  EXPECT_EQ(transposed.at({2, 0}), 2.0F);
  EXPECT_EQ(transposed.at({0, 1}), 3.0F);
}

TEST(TensorView, SliceSelectsRangeOnOneAxis) {
  const Tensor tensor = counting_tensor();
  const ConstTensorView columns = tensor.view().slice(1, 1, 3);
  EXPECT_EQ(columns.shape(), (Shape{2, 2}));
  EXPECT_EQ(columns.at({0, 0}), 1.0F);
  EXPECT_EQ(columns.at({1, 1}), 5.0F);
  const ConstTensorView second_row = tensor.view().slice(0, 1, 2);
  EXPECT_EQ(second_row.at({0, 0}), 3.0F);
  EXPECT_TRUE(second_row.is_contiguous());
}

TEST(TensorView, SliceRejectsBadRange) {
  const Tensor tensor = counting_tensor();
  EXPECT_THROW((void)tensor.view().slice(1, 2, 4), std::out_of_range);
  EXPECT_THROW((void)tensor.view().slice(1, 2, 1), std::out_of_range);
  EXPECT_THROW((void)tensor.view().slice(2, 0, 1), std::out_of_range);
  EXPECT_EQ(tensor.view().slice(1, 3, 3).element_count(), 0U);
}

TEST(TensorView, ReshapeKeepsElementOrder) {
  const Tensor tensor = counting_tensor();
  const ConstTensorView reshaped = tensor.view().reshape({3, 2});
  EXPECT_EQ(reshaped.data(), tensor.data());
  EXPECT_EQ(reshaped.at({1, 0}), 2.0F);
  EXPECT_EQ(reshaped.at({2, 1}), 5.0F);
}

TEST(TensorView, SingletonAxisStridesDoNotBreakContiguity) {
  const Tensor tensor({1, 3}, {1, 2, 3});
  const ConstTensorView column = tensor.view().transpose(0, 1);
  EXPECT_TRUE(column.is_contiguous());
  const ConstTensorView flat = column.reshape({3});
  EXPECT_EQ(flat.at({2}), 3.0F);
}

TEST(TensorView, EmptySliceOfEmptyTensorKeepsBasePointer) {
  Tensor tensor({0, 3});
  const TensorView sliced = tensor.view().slice(1, 2, 3);
  EXPECT_EQ(sliced.data(), tensor.data());
  EXPECT_EQ(sliced.element_count(), 0U);
}

TEST(TensorView, ReshapeRejectsNonContiguousOrWrongCount) {
  const Tensor tensor = counting_tensor();
  EXPECT_THROW((void)tensor.view().transpose(0, 1).reshape({6}), std::invalid_argument);
  EXPECT_THROW((void)tensor.view().reshape({4}), std::invalid_argument);
}

TEST(TensorView, RejectsMismatchedShapeAndStrides) {
  float value = 0.0F;
  EXPECT_THROW(TensorView(&value, {1, 1}, {1}), std::invalid_argument);
}
