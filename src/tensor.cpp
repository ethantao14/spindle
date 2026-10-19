#include "spindle/tensor.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace spindle {

namespace {

Strides row_major_strides(const Shape& shape) {
  Strides strides(shape.size());
  std::size_t stride = 1;
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    strides[axis] = stride;
    stride *= shape[axis];
  }
  return strides;
}

void check_axis(std::size_t axis, std::size_t rank) {
  if (axis >= rank) {
    throw std::out_of_range("axis " + std::to_string(axis) + " out of range for rank " +
                            std::to_string(rank));
  }
}

}  // namespace

std::size_t element_count(const Shape& shape) {
  std::size_t count = 1;
  for (const std::size_t dim : shape) {
    if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
      throw std::overflow_error("tensor shape overflows size_t");
    }
    count *= dim;
  }
  return count;
}

template <typename T>
BasicTensorView<T>::BasicTensorView(T* data, Shape shape, Strides strides)
    : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {
  if (shape_.size() != strides_.size()) {
    throw std::invalid_argument("shape and strides must have the same rank");
  }
}

template <typename T>
BasicTensorView<T> BasicTensorView<T>::contiguous(T* data, Shape shape) {
  Strides strides = row_major_strides(shape);
  return {data, std::move(shape), std::move(strides)};
}

template <typename T>
std::size_t BasicTensorView<T>::dim(std::size_t axis) const {
  check_axis(axis, rank());
  return shape_[axis];
}

template <typename T>
std::size_t BasicTensorView<T>::element_count() const {
  return spindle::element_count(shape_);
}

template <typename T>
bool BasicTensorView<T>::is_contiguous() const {
  if (element_count() == 0) {
    return true;
  }
  // Size-one axes never move to another element, so their stride does not matter.
  std::size_t expected_stride = 1;
  for (std::size_t axis = rank(); axis-- > 0;) {
    if (shape_[axis] != 1 && strides_[axis] != expected_stride) {
      return false;
    }
    expected_stride *= shape_[axis];
  }
  return true;
}

template <typename T>
T& BasicTensorView<T>::at(std::initializer_list<std::size_t> index) const {
  if (index.size() != rank()) {
    throw std::out_of_range("index rank does not match tensor rank");
  }
  std::size_t offset = 0;
  std::size_t axis = 0;
  for (const std::size_t position : index) {
    if (position >= shape_[axis]) {
      throw std::out_of_range("index " + std::to_string(position) + " out of range on axis " +
                              std::to_string(axis));
    }
    offset += position * strides_[axis];
    ++axis;
  }
  return data_[offset];
}

template <typename T>
BasicTensorView<T> BasicTensorView<T>::transpose(std::size_t axis_a, std::size_t axis_b) const {
  check_axis(axis_a, rank());
  check_axis(axis_b, rank());
  Shape shape = shape_;
  Strides strides = strides_;
  std::swap(shape[axis_a], shape[axis_b]);
  std::swap(strides[axis_a], strides[axis_b]);
  return {data_, std::move(shape), std::move(strides)};
}

template <typename T>
BasicTensorView<T> BasicTensorView<T>::slice(std::size_t axis, std::size_t begin,
                                             std::size_t end) const {
  check_axis(axis, rank());
  if (begin > end || end > shape_[axis]) {
    throw std::out_of_range("slice [" + std::to_string(begin) + ", " + std::to_string(end) +
                            ") out of range for dimension " + std::to_string(shape_[axis]));
  }
  Shape shape = shape_;
  shape[axis] = end - begin;
  // An empty result keeps the base pointer so it never points past the data.
  T* start = (spindle::element_count(shape) == 0) ? data_ : data_ + (begin * strides_[axis]);
  return {start, std::move(shape), strides_};
}

template <typename T>
BasicTensorView<T> BasicTensorView<T>::reshape(Shape new_shape) const {
  if (!is_contiguous()) {
    throw std::invalid_argument("reshape requires a contiguous view");
  }
  if (spindle::element_count(new_shape) != element_count()) {
    throw std::invalid_argument("reshape must keep the element count");
  }
  return contiguous(data_, std::move(new_shape));
}

template class BasicTensorView<float>;
template class BasicTensorView<const float>;

void Tensor::AlignedDelete::operator()(float* pointer) const {
  ::operator delete[](pointer, std::align_val_t{kTensorAlignment});
}

Tensor::Tensor(Shape shape) : shape_(std::move(shape)) {
  const std::size_t count = spindle::element_count(shape_);
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("tensor too large to allocate");
  }
  void* memory = ::operator new[](count * sizeof(float), std::align_val_t{kTensorAlignment});
  data_.reset(static_cast<float*>(memory));
  std::fill_n(data_.get(), count, 0.0F);
}

Tensor::Tensor(Shape shape, const std::vector<float>& values) : Tensor(std::move(shape)) {
  if (values.size() != element_count()) {
    throw std::invalid_argument("value count does not match tensor shape");
  }
  std::ranges::copy(values, data_.get());
}

std::size_t Tensor::element_count() const { return spindle::element_count(shape_); }

}  // namespace spindle
