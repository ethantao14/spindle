#include "reference.h"

#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace spindle::test {

namespace {

static_assert(std::endian::native == std::endian::little,
              "reference files are little-endian and read without byte swapping");

constexpr std::array<char, 4> kMagic = {'S', 'P', 'T', 'N'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kMaxDims = 8;

// Reads raw bytes sequentially, failing with the file name on any short read.
class ByteReader {
 public:
  ByteReader(std::vector<char> bytes, std::string name)
      : bytes_(std::move(bytes)), name_(std::move(name)) {}

  template <typename T>
  T read() {
    T value{};
    std::memcpy(&value, take(sizeof(T)), sizeof(T));
    return value;
  }

  const char* take(std::size_t count) {
    if (count > bytes_.size() - offset_) {
      fail("file is truncated");
    }
    const char* start = bytes_.data() + offset_;
    offset_ += count;
    return start;
  }

  [[nodiscard]] std::size_t remaining() const { return bytes_.size() - offset_; }

  [[noreturn]] void fail(const std::string& reason) const {
    throw std::runtime_error(name_ + ": " + reason);
  }

 private:
  std::vector<char> bytes_;
  std::string name_;
  std::size_t offset_ = 0;
};

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error(path.string() + ": cannot open (run tools/check.sh to generate)");
  }
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

template <typename T>
std::vector<T> read_elements(ByteReader& reader, std::size_t count) {
  if (count > reader.remaining() / sizeof(T)) {
    reader.fail("file is truncated");
  }
  std::vector<T> values(count);
  std::memcpy(values.data(), reader.take(count * sizeof(T)), count * sizeof(T));
  return values;
}

}  // namespace

std::size_t ReferenceTensor::element_count() const {
  std::size_t count = 1;
  for (const std::size_t dim : shape) {
    count *= dim;
  }
  return count;
}

ReferenceTensor load_reference_tensor(const std::filesystem::path& path) {
  ByteReader reader(read_file(path), path.string());

  if (std::memcmp(reader.take(kMagic.size()), kMagic.data(), kMagic.size()) != 0) {
    reader.fail("bad magic, not a reference tensor file");
  }
  if (reader.read<std::uint32_t>() != kFormatVersion) {
    reader.fail("unsupported format version");
  }
  const auto dtype_code = reader.read<std::uint32_t>();
  if (dtype_code > static_cast<std::uint32_t>(DType::kInt32)) {
    reader.fail("unknown dtype " + std::to_string(dtype_code));
  }
  const auto ndim = reader.read<std::uint32_t>();
  if (ndim > kMaxDims) {
    reader.fail("too many dimensions");
  }

  ReferenceTensor tensor;
  tensor.dtype = static_cast<DType>(dtype_code);
  std::size_t count = 1;
  for (std::uint32_t i = 0; i < ndim; ++i) {
    const auto dim = reader.read<std::uint64_t>();
    if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
      reader.fail("shape overflows");
    }
    count *= dim;
    tensor.shape.push_back(dim);
  }

  if (tensor.dtype == DType::kFloat32) {
    tensor.floats = read_elements<float>(reader, count);
  } else {
    tensor.ints = read_elements<std::int32_t>(reader, count);
  }
  if (reader.remaining() != 0) {
    reader.fail("unexpected trailing bytes");
  }
  return tensor;
}

Tensor load_float_tensor(const std::filesystem::path& path) {
  ReferenceTensor reference = load_reference_tensor(path);
  if (reference.dtype != DType::kFloat32) {
    throw std::runtime_error(path.string() + ": expected a float32 tensor");
  }
  return {reference.shape, reference.floats};
}

}  // namespace spindle::test
