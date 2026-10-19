#include "spindle/ops.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace spindle {

namespace {

std::string shape_to_string(const Shape& shape) {
  std::string text = "[";
  for (std::size_t axis = 0; axis < shape.size(); ++axis) {
    text += (axis == 0 ? "" : ", ") + std::to_string(shape[axis]);
  }
  return text + "]";
}

}  // namespace

void matmul(const ConstTensorView& a, const ConstTensorView& b, const TensorView& out) {
  if (a.rank() != 2 || b.rank() != 2 || out.rank() != 2 || a.dim(1) != b.dim(0) ||
      out.dim(0) != a.dim(0) || out.dim(1) != b.dim(1)) {
    throw std::invalid_argument("matmul shape mismatch: " + shape_to_string(a.shape()) + " @ " +
                                shape_to_string(b.shape()) + " -> " + shape_to_string(out.shape()));
  }
  const std::size_t rows = a.dim(0);
  const std::size_t inner = a.dim(1);
  const std::size_t cols = b.dim(1);
  const float* a_data = a.data();
  const float* b_data = b.data();
  float* out_data = out.data();

  // Naive triple loop: the correctness baseline that later kernels are measured against.
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      float sum = 0.0F;
      for (std::size_t k = 0; k < inner; ++k) {
        sum += a_data[(row * a.strides()[0]) + (k * a.strides()[1])] *
               b_data[(k * b.strides()[0]) + (col * b.strides()[1])];
      }
      out_data[(row * out.strides()[0]) + (col * out.strides()[1])] = sum;
    }
  }
}

}  // namespace spindle
