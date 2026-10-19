#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>

namespace spindle {

using Shape = std::vector<std::size_t>;
using Strides = std::vector<std::size_t>;  // in elements, not bytes

constexpr std::size_t kTensorAlignment = 64;

// A non-owning view of float32 data with a shape and strides. Views never copy;
// the caller keeps the underlying storage alive. T is float or const float.
template <typename T>
class BasicTensorView {
  static_assert(std::is_same_v<std::remove_const_t<T>, float>, "tensors hold float32");

 public:
  BasicTensorView(T* data, Shape shape, Strides strides);

  // A view of densely packed row-major data.
  static BasicTensorView contiguous(T* data, Shape shape);

  // Lets a writable view be passed where a read-only view is expected.
  operator BasicTensorView<const float>() const  // NOLINT(google-explicit-constructor)
    requires(!std::is_const_v<T>)
  {
    return {data_, shape_, strides_};
  }

  [[nodiscard]] T* data() const { return data_; }
  [[nodiscard]] const Shape& shape() const { return shape_; }
  [[nodiscard]] const Strides& strides() const { return strides_; }
  [[nodiscard]] std::size_t rank() const { return shape_.size(); }
  [[nodiscard]] std::size_t dim(std::size_t axis) const;
  [[nodiscard]] std::size_t element_count() const;
  [[nodiscard]] bool is_contiguous() const;

  // Bounds-checked element access; throws std::out_of_range.
  [[nodiscard]] T& at(std::initializer_list<std::size_t> index) const;

  [[nodiscard]] BasicTensorView transpose(std::size_t axis_a, std::size_t axis_b) const;
  // Keeps indices [begin, end) along one axis.
  [[nodiscard]] BasicTensorView slice(std::size_t axis, std::size_t begin, std::size_t end) const;
  // Same elements under a new shape. Requires a contiguous view.
  [[nodiscard]] BasicTensorView reshape(Shape new_shape) const;

 private:
  T* data_;
  Shape shape_;
  Strides strides_;
};

using TensorView = BasicTensorView<float>;
using ConstTensorView = BasicTensorView<const float>;

extern template class BasicTensorView<float>;
extern template class BasicTensorView<const float>;

// Owns zero-initialized, 64-byte aligned, row-major float32 storage.
class Tensor {
 public:
  explicit Tensor(Shape shape);
  Tensor(Shape shape, const std::vector<float>& values);

  [[nodiscard]] TensorView view() { return TensorView::contiguous(data_.get(), shape_); }
  [[nodiscard]] ConstTensorView view() const {
    return ConstTensorView::contiguous(data_.get(), shape_);
  }
  [[nodiscard]] float* data() { return data_.get(); }
  [[nodiscard]] const float* data() const { return data_.get(); }
  [[nodiscard]] const Shape& shape() const { return shape_; }
  [[nodiscard]] std::size_t element_count() const;

 private:
  struct AlignedDelete {
    void operator()(float* pointer) const;
  };

  Shape shape_;
  std::unique_ptr<float, AlignedDelete> data_;
};

// Product of the dimensions; throws std::overflow_error if it does not fit.
std::size_t element_count(const Shape& shape);

}  // namespace spindle
