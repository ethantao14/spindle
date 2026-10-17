#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace spindle::test {

// Element types in a reference tensor file; values match tools/reference.py.
enum class DType : std::uint8_t { kFloat32 = 0, kInt32 = 1 };

// A tensor dumped by tools/reference.py. Only the vector matching dtype is filled.
struct ReferenceTensor {
  DType dtype = DType::kFloat32;
  std::vector<std::size_t> shape;
  std::vector<float> floats;
  std::vector<std::int32_t> ints;

  [[nodiscard]] std::size_t element_count() const;
};

// Reads a tensor file. Throws std::runtime_error if the file is missing or malformed.
ReferenceTensor load_reference_tensor(const std::filesystem::path& path);

}  // namespace spindle::test
