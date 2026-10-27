#include "spindle/ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <string>

#if defined(__ARM_NEON)  // NOLINT(readability-use-concise-preprocessor-directives)
#include <arm_neon.h>
#endif

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

#if defined(__ARM_NEON)  // NOLINT(readability-use-concise-preprocessor-directives)
// Each tile keeps four vectors per output row live across the K panel.
template <std::size_t Rows>
void row_tile(const ConstTensorView& a, const ConstTensorView& b, const TensorView& out,
              std::size_t row, std::size_t col, std::size_t begin, std::size_t end) {
  std::array<std::array<float32x4_t, 4>, Rows> sums;
  for (std::size_t r = 0; r < Rows; ++r) {
    for (std::size_t v = 0; v < 4; ++v) {
      sums[r][v] = begin == 0
                       ? vdupq_n_f32(0)
                       : vld1q_f32(out.data() + ((row + r) * out.strides()[0]) + col + (4 * v));
    }
  }
  for (std::size_t k = begin; k < end; ++k) {
    const float* bp = b.data() + (k * b.strides()[0]) + col;
    const float32x4_t b0 = vld1q_f32(bp + 0);
    const float32x4_t b1 = vld1q_f32(bp + 4);
    const float32x4_t b2 = vld1q_f32(bp + 8);
    const float32x4_t b3 = vld1q_f32(bp + 12);
    for (std::size_t r = 0; r < Rows; ++r) {
      const float av = a.data()[((row + r) * a.strides()[0]) + k];
      sums[r][0] = vfmaq_n_f32(sums[r][0], b0, av);
      sums[r][1] = vfmaq_n_f32(sums[r][1], b1, av);
      sums[r][2] = vfmaq_n_f32(sums[r][2], b2, av);
      sums[r][3] = vfmaq_n_f32(sums[r][3], b3, av);
    }
  }
  for (std::size_t r = 0; r < Rows; ++r) {
    for (std::size_t v = 0; v < 4; ++v) {
      vst1q_f32(out.data() + ((row + r) * out.strides()[0]) + col + (4 * v), sums[r][v]);
    }
  }
}

void matmul_rows(const ConstTensorView& a, const ConstTensorView& b, const TensorView& out) {
  constexpr std::size_t kRows = 64;
  constexpr std::size_t kCols = 128;
  constexpr std::size_t kInner = 128;
  for (std::size_t i = 0; i < a.dim(0); i += kRows) {
    const auto row_end = std::min(i + kRows, a.dim(0));
    for (std::size_t j = 0; j < b.dim(1); j += kCols) {
      const auto col_end = std::min(j + kCols, b.dim(1));
      for (std::size_t p = 0; p < a.dim(1); p += kInner) {
        const auto end = std::min(p + kInner, a.dim(1));
        std::size_t row = i;
        for (; row + 4 <= row_end; row += 4) {
          for (std::size_t col = j; col + 16 <= col_end; col += 16) {
            row_tile<4>(a, b, out, row, col, p, end);
          }
        }
        for (; row < row_end; ++row) {
          for (std::size_t col = j; col + 16 <= col_end; col += 16) {
            row_tile<1>(a, b, out, row, col, p, end);
          }
        }
        for (row = i; row < row_end; ++row) {
          for (std::size_t col = j + (((col_end - j) / 16) * 16); col < col_end; ++col) {
            float sum = p == 0 ? 0 : out.data()[(row * out.strides()[0]) + col];
            for (std::size_t k = p; k < end; ++k) {
              sum += a.data()[(row * a.strides()[0]) + k] * b.data()[(k * b.strides()[0]) + col];
            }
            out.data()[(row * out.strides()[0]) + col] = sum;
          }
        }
      }
    }
  }
}

// Four independent K chains per column hide FMA latency; A is shared across columns.
template <std::size_t Cols>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): explicit dimensions and strides.
void dot_tile(const float* a, const float* b, std::size_t stride, std::size_t inner, float* out,
              std::size_t out_stride) {
  std::array<std::array<float32x4_t, 4>, Cols> sums;
  for (auto& column : sums) {
    for (auto& sum : column) {
      sum = vdupq_n_f32(0);
    }
  }
  std::size_t k = 0;
  for (; k + 16 <= inner; k += 16) {
    const float32x4_t a0 = vld1q_f32(a + k + 0);
    const float32x4_t a1 = vld1q_f32(a + k + 4);
    const float32x4_t a2 = vld1q_f32(a + k + 8);
    const float32x4_t a3 = vld1q_f32(a + k + 12);
    for (std::size_t j = 0; j < Cols; ++j) {
      sums[j][0] = vfmaq_f32(sums[j][0], a0, vld1q_f32(b + (j * stride) + k + 0));
      sums[j][1] = vfmaq_f32(sums[j][1], a1, vld1q_f32(b + (j * stride) + k + 4));
      sums[j][2] = vfmaq_f32(sums[j][2], a2, vld1q_f32(b + (j * stride) + k + 8));
      sums[j][3] = vfmaq_f32(sums[j][3], a3, vld1q_f32(b + (j * stride) + k + 12));
    }
  }
  for (std::size_t j = 0; j < Cols; ++j) {
    const auto pair0 = vaddq_f32(sums[j][0], sums[j][1]);
    const auto pair1 = vaddq_f32(sums[j][2], sums[j][3]);
    float sum = vaddvq_f32(vaddq_f32(pair0, pair1));
    for (std::size_t p = k; p < inner; ++p) {
      sum += a[p] * b[(j * stride) + p];
    }
    out[j * out_stride] = sum;
  }
}

void matmul_dots(const ConstTensorView& a, const ConstTensorView& b, const TensorView& out) {
  for (std::size_t row = 0; row < a.dim(0); ++row) {
    const float* ap = a.data() + (row * a.strides()[0]);
    float* op = out.data() + (row * out.strides()[0]);
    std::size_t col = 0;
    for (; col + 4 <= b.dim(1); col += 4) {
      dot_tile<4>(ap, b.data() + (col * b.strides()[1]), b.strides()[1], a.dim(1),
                  op + (col * out.strides()[1]), out.strides()[1]);
    }
    for (; col < b.dim(1); ++col) {
      dot_tile<1>(ap, b.data() + (col * b.strides()[1]), b.strides()[1], a.dim(1),
                  op + (col * out.strides()[1]), out.strides()[1]);
    }
  }
}
#endif

}  // namespace

void matmul(const ConstTensorView& a, const ConstTensorView& b, const TensorView& out,
            ThreadPool* pool) {
  if (a.rank() != 2 || b.rank() != 2 || out.rank() != 2 || a.dim(1) != b.dim(0) ||
      out.dim(0) != a.dim(0) || out.dim(1) != b.dim(1)) {
    throw std::invalid_argument("matmul shape mismatch: " + shape_to_string(a.shape()) + " @ " +
                                shape_to_string(b.shape()) + " -> " + shape_to_string(out.shape()));
  }
  const std::size_t rows = a.dim(0);
  const std::size_t inner = a.dim(1);
  const std::size_t cols = b.dim(1);
  // Below 512K multiply-adds, measured dispatch overhead exceeds the savings.
  // Larger splits preserve the kernel's column tiles.
  if (pool != nullptr && pool->size() > 1 && rows * inner * cols >= 524288) {
    std::size_t block = 1;
#if defined(__ARM_NEON)  // NOLINT(readability-use-concise-preprocessor-directives)
    if (a.strides()[1] == 1 && b.strides()[1] == 1 && out.strides()[1] == 1 &&
        (rows == 1 || out.strides()[0] >= cols)) {
      block = 128;
    } else if (a.strides()[1] == 1 && b.strides()[0] == 1) {
      block = 4;
    }
#endif
    const auto blocks = (cols + block - 1) / block;
    const auto chunks = std::min(blocks, pool->size());
    if (chunks > 1) {
      pool->run(chunks, [&](std::size_t chunk, std::size_t count) {
        const auto first = (blocks * chunk / count) * block;
        const auto last = std::min(cols, (blocks * (chunk + 1) / count) * block);
        matmul(a, b.slice(1, first, last), out.slice(1, first, last));
      });
      return;
    }
  }
  const float* a_data = a.data();
  const float* b_data = b.data();
  float* out_data = out.data();

#if defined(__ARM_NEON)  // NOLINT(readability-use-concise-preprocessor-directives)
  if (rows != 0 && cols != 0 && inner != 0 && a.strides()[1] == 1) {
    if (b.strides()[1] == 1 && out.strides()[1] == 1 && (rows == 1 || out.strides()[0] >= cols)) {
      matmul_rows(a, b, out);
      return;
    }
    if (b.strides()[0] == 1) {
      matmul_dots(a, b, out);
      return;
    }
  }
#endif

  // General strided fallback also handles empty inner dimensions.
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
