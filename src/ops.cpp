#include "spindle/ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
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

void require_contiguous(const char* op, const ConstTensorView& view) {
  if (!view.is_contiguous()) {
    throw std::invalid_argument(std::string(op) + " needs contiguous views");
  }
}

// Checks in and out are contiguous with equal shapes of rank >= 1; returns the row width.
std::size_t check_row_op(const char* op, const ConstTensorView& in, const TensorView& out) {
  if (in.rank() == 0 || in.shape() != out.shape()) {
    throw std::invalid_argument(std::string(op) +
                                " shape mismatch: " + shape_to_string(in.shape()) + " -> " +
                                shape_to_string(out.shape()));
  }
  require_contiguous(op, in);
  require_contiguous(op, out);
  return in.dim(in.rank() - 1);
}

std::size_t row_count(const ConstTensorView& view, std::size_t width) {
  return width == 0 ? 0 : view.element_count() / width;
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

void add(const ConstTensorView& a, const ConstTensorView& b, const TensorView& out) {
  const std::size_t width = check_row_op("add", a, out);
  require_contiguous("add", b);
  const float* a_data = a.data();
  const float* b_data = b.data();
  float* out_data = out.data();

  if (b.shape() == a.shape()) {
    for (std::size_t i = 0; i < a.element_count(); ++i) {
      out_data[i] = a_data[i] + b_data[i];
    }
    return;
  }
  if (b.shape() != Shape{width}) {
    throw std::invalid_argument("add shape mismatch: " + shape_to_string(a.shape()) + " + " +
                                shape_to_string(b.shape()));
  }
  // b is a bias: add it to every row.
  for (std::size_t row = 0; row < row_count(a, width); ++row) {
    for (std::size_t i = 0; i < width; ++i) {
      out_data[(row * width) + i] = a_data[(row * width) + i] + b_data[i];
    }
  }
}

void softmax(const ConstTensorView& in, const TensorView& out) {
  const std::size_t width = check_row_op("softmax", in, out);
  for (std::size_t row = 0; row < row_count(in, width); ++row) {
    const float* in_row = in.data() + (row * width);
    float* out_row = out.data() + (row * width);

    const float max_value = *std::max_element(in_row, in_row + width);
    // Sum in double: many tiny terms would each round away in a float sum.
    double sum = 0.0;
    for (std::size_t i = 0; i < width; ++i) {
      out_row[i] = std::exp(in_row[i] - max_value);
      sum += out_row[i];
    }
    for (std::size_t i = 0; i < width; ++i) {
      out_row[i] = static_cast<float>(out_row[i] / sum);
    }
  }
}

void layer_norm(const ConstTensorView& in, const ConstTensorView& gamma,
                const ConstTensorView& beta, float epsilon, const TensorView& out) {
  const std::size_t width = check_row_op("layer_norm", in, out);
  if (gamma.shape() != Shape{width} || beta.shape() != Shape{width}) {
    throw std::invalid_argument("layer_norm gamma and beta must have shape [" +
                                std::to_string(width) + "]");
  }
  require_contiguous("layer_norm", gamma);
  require_contiguous("layer_norm", beta);

  for (std::size_t row = 0; row < row_count(in, width); ++row) {
    const float* in_row = in.data() + (row * width);
    float* out_row = out.data() + (row * width);

    // Accumulate in double so long rows do not lose precision in the sums.
    double sum = 0.0;
    for (std::size_t i = 0; i < width; ++i) {
      sum += in_row[i];
    }
    const double mean = sum / static_cast<double>(width);
    double squared_deviation_sum = 0.0;
    for (std::size_t i = 0; i < width; ++i) {
      const double deviation = in_row[i] - mean;
      squared_deviation_sum += deviation * deviation;
    }
    const double variance = squared_deviation_sum / static_cast<double>(width);
    const double inverse_std = 1.0 / std::sqrt(variance + epsilon);
    for (std::size_t i = 0; i < width; ++i) {
      const double normalized = (in_row[i] - mean) * inverse_std;
      out_row[i] = static_cast<float>((normalized * gamma.data()[i]) + beta.data()[i]);
    }
  }
}

void gelu(const ConstTensorView& in, const TensorView& out) {
  check_row_op("gelu", in, out);
  const auto sqrt_2_over_pi = static_cast<float>(std::sqrt(2.0 / std::numbers::pi));
  for (std::size_t i = 0; i < in.element_count(); ++i) {
    const float x = in.data()[i];
    const float inner = sqrt_2_over_pi * (x + (0.044715F * x * x * x));
    out.data()[i] = 0.5F * x * (1.0F + std::tanh(inner));
  }
}

void embedding(const ConstTensorView& table, std::span<const std::int32_t> tokens,
               const TensorView& out) {
  if (table.rank() != 2 || out.rank() != 2 || out.dim(0) != tokens.size() ||
      out.dim(1) != table.dim(1)) {
    throw std::invalid_argument(
        "embedding shape mismatch: table " + shape_to_string(table.shape()) + ", " +
        std::to_string(tokens.size()) + " tokens -> " + shape_to_string(out.shape()));
  }
  require_contiguous("embedding", table);
  require_contiguous("embedding", out);
  const std::size_t width = table.dim(1);

  for (std::size_t t = 0; t < tokens.size(); ++t) {
    const std::int32_t token = tokens[t];
    if (token < 0 || static_cast<std::size_t>(token) >= table.dim(0)) {
      throw std::out_of_range("token " + std::to_string(token) + " outside embedding table of " +
                              std::to_string(table.dim(0)) + " rows");
    }
    const float* source = table.data() + (static_cast<std::size_t>(token) * width);
    std::copy_n(source, width, out.data() + (t * width));
  }
}

}  // namespace spindle
