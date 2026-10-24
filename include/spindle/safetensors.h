#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "spindle/tensor.h"

namespace spindle {

// One tensor's entry in a safetensors header. Offsets are relative to the data section.
struct SafetensorsEntry {
  std::string dtype;
  Shape shape;
  std::size_t begin = 0;
  std::size_t end = 0;
};

// A memory-mapped safetensors file. Opening validates the whole header: every
// entry's dtype is known, its byte range matches its shape, lies inside the file,
// and does not overlap another entry. Throws std::runtime_error on any problem.
class SafetensorsFile {
 public:
  explicit SafetensorsFile(const std::filesystem::path& path);
  ~SafetensorsFile();
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  SafetensorsFile(SafetensorsFile&&) = delete;
  SafetensorsFile& operator=(SafetensorsFile&&) = delete;

  [[nodiscard]] const std::map<std::string, SafetensorsEntry>& entries() const { return entries_; }

  // Copies a float32 tensor into aligned storage. The file's data is not 4-byte
  // aligned in general, so it cannot be used in place. Throws std::runtime_error if
  // the tensor is missing, is not F32, or does not have the expected shape.
  [[nodiscard]] Tensor load_f32(const std::string& name, const Shape& expected_shape) const;

 private:
  std::filesystem::path path_;
  const char* mapping_ = nullptr;
  std::size_t file_size_ = 0;
  const char* data_ = nullptr;  // start of the data section, after the header
  std::map<std::string, SafetensorsEntry> entries_;
};

}  // namespace spindle
