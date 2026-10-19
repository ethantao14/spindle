#pragma once

#include "spindle/tensor.h"

namespace spindle {

// out[M, N] = a[M, K] @ b[K, N]. Accepts strided views (e.g. a transposed b).
// out must not overlap a or b. Throws std::invalid_argument on a shape mismatch.
void matmul(const ConstTensorView& a, const ConstTensorView& b, const TensorView& out);

}  // namespace spindle
