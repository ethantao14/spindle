#include "spindle/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spindle {

namespace {

static_assert(std::endian::native == std::endian::little,
              "safetensors data is little-endian and read without byte swapping");

constexpr std::size_t kHeaderLengthBytes = 8;
// The format caps the header at 100 MB, which also bounds a hostile length field.
constexpr std::uint64_t kMaxHeaderBytes = 100'000'000;

std::size_t dtype_size(const std::string& dtype) {
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") {
    return 8;
  }
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") {
    return 4;
  }
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") {
    return 2;
  }
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") {
    return 1;
  }
  return 0;
}

// Closes a file descriptor when it goes out of scope.
class FileDescriptor {
 public:
  explicit FileDescriptor(int descriptor) : descriptor_(descriptor) {}
  ~FileDescriptor() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&&) = delete;
  FileDescriptor& operator=(FileDescriptor&&) = delete;
  [[nodiscard]] int get() const { return descriptor_; }

 private:
  int descriptor_;
};

}  // namespace

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : path_(path) {
  const auto fail = [this](const std::string& reason) {
    throw std::runtime_error(path_.string() + ": " + reason);
  };
  const FileDescriptor file(::open(path.c_str(), O_RDONLY));
  if (file.get() < 0) {
    fail("cannot open");
  }
  struct stat info{};
  if (::fstat(file.get(), &info) != 0 || info.st_size < 0) {
    fail("cannot stat");
  }
  file_size_ = static_cast<std::size_t>(info.st_size);
  if (file_size_ < kHeaderLengthBytes) {
    fail("too small to hold a header length");
  }
  void* mapping = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, file.get(), 0);
  if (mapping == MAP_FAILED) {
    fail("mmap failed");
  }
  mapping_ = static_cast<const char*>(mapping);

  // From here on the destructor will not run if we throw, so unmap on failure.
  try {
    std::uint64_t header_bytes = 0;
    std::memcpy(&header_bytes, mapping_, sizeof(header_bytes));
    if (header_bytes > kMaxHeaderBytes || header_bytes > file_size_ - kHeaderLengthBytes) {
      fail("header length " + std::to_string(header_bytes) + " does not fit in the file");
    }
    const auto header_size = static_cast<std::size_t>(header_bytes);
    data_ = mapping_ + kHeaderLengthBytes + header_size;
    const std::size_t data_size = file_size_ - kHeaderLengthBytes - header_size;

    const nlohmann::json header =
        nlohmann::json::parse(mapping_ + kHeaderLengthBytes, data_, nullptr, false);
    if (!header.is_object()) {
      fail("header is not a JSON object");
    }
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    for (const auto& [name, value] : header.items()) {
      if (name == "__metadata__") {
        continue;
      }
      const auto bad = [&fail, &name](const std::string& reason) {
        std::string message = name;
        message += ": ";
        message += reason;
        fail(message);
      };
      if (!value.is_object() || !value.contains("dtype") || !value.contains("shape") ||
          !value.contains("data_offsets") || !value["dtype"].is_string() ||
          !value["shape"].is_array() || !value["data_offsets"].is_array() ||
          value["data_offsets"].size() != 2) {
        bad("expected dtype, shape and two data_offsets");
      }
      SafetensorsEntry entry;
      entry.dtype = value["dtype"].get<std::string>();
      const std::size_t element_bytes = dtype_size(entry.dtype);
      if (element_bytes == 0) {
        bad("unknown dtype " + entry.dtype);
      }
      for (const nlohmann::json& dim : value["shape"]) {
        if (!dim.is_number_unsigned()) {
          bad("shape dimensions must be non-negative integers");
        }
        entry.shape.push_back(dim.get<std::size_t>());
      }
      for (const nlohmann::json& offset : value["data_offsets"]) {
        if (!offset.is_number_unsigned()) {
          bad("data_offsets must be non-negative integers");
        }
      }
      entry.begin = value["data_offsets"][0].get<std::size_t>();
      entry.end = value["data_offsets"][1].get<std::size_t>();
      if (entry.begin > entry.end || entry.end > data_size) {
        bad("data_offsets fall outside the data section");
      }
      std::size_t expected_bytes = 0;
      try {
        expected_bytes = element_count(entry.shape);
      } catch (const std::overflow_error&) {
        bad("shape is too large");
      }
      if (expected_bytes > std::numeric_limits<std::size_t>::max() / element_bytes ||
          expected_bytes * element_bytes != entry.end - entry.begin) {
        bad("byte range does not match shape and dtype");
      }
      ranges.emplace_back(entry.begin, entry.end);
      entries_.emplace(name, std::move(entry));
    }
    std::ranges::sort(ranges);
    for (std::size_t i = 1; i < ranges.size(); ++i) {
      if (ranges[i].first < ranges[i - 1].second) {
        fail("tensor byte ranges overlap");
      }
    }
  } catch (...) {
    ::munmap(const_cast<char*>(mapping_),
             file_size_);  // NOLINT(cppcoreguidelines-pro-type-const-cast)
    throw;
  }
}

SafetensorsFile::~SafetensorsFile() {
  ::munmap(const_cast<char*>(mapping_),
           file_size_);  // NOLINT(cppcoreguidelines-pro-type-const-cast)
}

Tensor SafetensorsFile::load_f32(const std::string& name, const Shape& expected_shape) const {
  const auto found = entries_.find(name);
  if (found == entries_.end()) {
    throw std::runtime_error(path_.string() + ": missing tensor " + name);
  }
  const SafetensorsEntry& entry = found->second;
  if (entry.dtype != "F32") {
    throw std::runtime_error(path_.string() + ": " + name + " is " + entry.dtype +
                             ", expected F32");
  }
  if (entry.shape != expected_shape) {
    throw std::runtime_error(path_.string() + ": " + name + " has an unexpected shape");
  }
  Tensor tensor(entry.shape);
  std::memcpy(tensor.data(), data_ + entry.begin, entry.end - entry.begin);
  return tensor;
}

}  // namespace spindle
