#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/reference.h"

namespace {

using spindle::test::DType;
using spindle::test::load_reference_tensor;

// Builds a tensor file byte by byte, mirroring write_tensor in tools/reference.py.
class FileBuilder {
 public:
  FileBuilder& magic(const std::string& text) {
    bytes_.insert(bytes_.end(), text.begin(), text.end());
    return *this;
  }

  template <typename T>
  FileBuilder& value(T number) {
    const auto* raw = reinterpret_cast<const char*>(&number);
    bytes_.insert(bytes_.end(), raw, raw + sizeof(T));
    return *this;
  }

  FileBuilder& header(std::uint32_t dtype, std::initializer_list<std::uint64_t> shape) {
    magic("SPTN").value<std::uint32_t>(1).value(dtype);
    value(static_cast<std::uint32_t>(shape.size()));
    for (const std::uint64_t dim : shape) {
      value(dim);
    }
    return *this;
  }

  [[nodiscard]] std::filesystem::path write(const std::string& name) const {
    const std::filesystem::path path = std::filesystem::path(testing::TempDir()) / name;
    std::ofstream file(path, std::ios::binary);
    file.write(bytes_.data(), static_cast<std::streamsize>(bytes_.size()));
    return path;
  }

 private:
  std::vector<char> bytes_;
};

std::string load_error(const std::filesystem::path& path) {
  try {
    load_reference_tensor(path);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return "";
}

}  // namespace

TEST(ReferenceLoader, ReadsFloatTensor) {
  const auto path = FileBuilder()
                        .header(0, {2, 3})
                        .value(1.0F)
                        .value(2.0F)
                        .value(3.0F)
                        .value(4.0F)
                        .value(5.0F)
                        .value(-6.5F)
                        .write("float.bin");
  const auto tensor = load_reference_tensor(path);
  EXPECT_EQ(tensor.dtype, DType::kFloat32);
  EXPECT_EQ(tensor.shape, (std::vector<std::size_t>{2, 3}));
  EXPECT_EQ(tensor.floats, (std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, -6.5F}));
  EXPECT_TRUE(tensor.ints.empty());
}

TEST(ReferenceLoader, ReadsIntTensor) {
  const auto path = FileBuilder()
                        .header(1, {3})
                        .value<std::int32_t>(15496)
                        .value<std::int32_t>(11)
                        .value<std::int32_t>(-1)
                        .write("int.bin");
  const auto tensor = load_reference_tensor(path);
  EXPECT_EQ(tensor.dtype, DType::kInt32);
  EXPECT_EQ(tensor.shape, (std::vector<std::size_t>{3}));
  EXPECT_EQ(tensor.ints, (std::vector<std::int32_t>{15496, 11, -1}));
}

TEST(ReferenceLoader, RejectsMissingFile) {
  EXPECT_NE(load_error("/nonexistent/tensor.bin").find("cannot open"), std::string::npos);
}

TEST(ReferenceLoader, RejectsBadMagic) {
  const auto path = FileBuilder().magic("NOPE").value<std::uint32_t>(1).write("magic.bin");
  EXPECT_NE(load_error(path).find("bad magic"), std::string::npos);
}

TEST(ReferenceLoader, RejectsUnknownDtype) {
  const auto path = FileBuilder().header(7, {1}).value(1.0F).write("dtype.bin");
  EXPECT_NE(load_error(path).find("unknown dtype"), std::string::npos);
}

TEST(ReferenceLoader, RejectsTruncatedData) {
  const auto path = FileBuilder().header(0, {4}).value(1.0F).write("short.bin");
  EXPECT_NE(load_error(path).find("truncated"), std::string::npos);
}

TEST(ReferenceLoader, RejectsTruncatedHeader) {
  const auto path = FileBuilder().magic("SPTN").value<std::uint32_t>(1).write("header.bin");
  EXPECT_NE(load_error(path).find("truncated"), std::string::npos);
}

TEST(ReferenceLoader, RejectsTrailingBytes) {
  const auto path = FileBuilder().header(0, {1}).value(1.0F).value(2.0F).write("long.bin");
  EXPECT_NE(load_error(path).find("trailing"), std::string::npos);
}

TEST(ReferenceLoader, RejectsShapeThatOverflows) {
  const auto path =
      FileBuilder().header(0, {std::uint64_t{1} << 40U, std::uint64_t{1} << 40U}).write("big.bin");
  EXPECT_NE(load_error(path).find("overflows"), std::string::npos);
}
