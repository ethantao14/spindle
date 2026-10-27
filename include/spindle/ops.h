#pragma once

#include <cstdint>
#include <span>

#include "spindle/tensor.h"
#include "spindle/thread_pool.h"

namespace spindle {

// Ops throw std::invalid_argument on a shape mismatch. Except matmul, they need
// contiguous row-major views. Except matmul and embedding, out may be the same
// view as an input (in place).

// out[M, N] = a[M, K] @ b[K, N]. Accepts strided views (e.g. a transposed b).
// out must not overlap a or b. With a pool, output columns are split across its
// threads, and every element is computed exactly as on one thread (bit-identical).
void matmul(const ConstTensorView& a, const ConstTensorView& b, const TensorView& out,
            ThreadPool* pool = nullptr);

// out = a + b. b has a's shape, or is 1-D with a's last dimension (a bias added to every row).
void add(const ConstTensorView& a, const ConstTensorView& b, const TensorView& out);

// Softmax over the last axis, subtracting each row's max first so exp cannot overflow.
void softmax(const ConstTensorView& in, const TensorView& out);

// LayerNorm over the last axis: (x - mean) / sqrt(variance + epsilon) * gamma + beta.
void layer_norm(const ConstTensorView& in, const ConstTensorView& gamma,
                const ConstTensorView& beta, float epsilon, const TensorView& out);

// GELU with the tanh approximation, the variant GPT-2 uses ("gelu_new" in Hugging Face).
void gelu(const ConstTensorView& in, const TensorView& out);

// out[t, :] = table[tokens[t], :]. out must not overlap table.
// Throws std::out_of_range for a token outside the table.
void embedding(const ConstTensorView& table, std::span<const std::int32_t> tokens,
               const TensorView& out);

}  // namespace spindle
