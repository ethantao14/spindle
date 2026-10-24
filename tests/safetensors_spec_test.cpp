#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "spindle/safetensors.h"

namespace {

class SafetensorsSpecTest : public testing::Test {
 protected:
  std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / "safetensors_spec";

  void SetUp() override { std::filesystem::create_directories(dir); }
  void TearDown() override { std::filesystem::remove_all(dir); }

  static void append_u64(std::string& bytes, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
      bytes.push_back(static_cast<char>((value >> shift) & 255U));
    }
  }

  static std::string file_bytes(const std::string& header) {
    std::string bytes;
    append_u64(bytes, header.size());
    bytes += header;
    for (float value : {1.0F, -2.5F}) {
      const auto bits = std::bit_cast<std::uint32_t>(value);
      for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<char>((bits >> shift) & 255U));
      }
    }
    return bytes;
  }

  [[nodiscard]] std::filesystem::path write(const std::string& bytes) const {
    const auto path = dir / "test.safetensors";
    std::ofstream file(path, std::ios::binary);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
  }

  static std::string valid_header() {
    return R"({"a":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}})";
  }
};

TEST_F(SafetensorsSpecTest, LoadsEntriesAndCopiesAlignedValues) {
  const spindle::SafetensorsFile file(write(file_bytes(valid_header())));
  ASSERT_EQ(file.entries().size(), 1U);
  const auto& entry = file.entries().at("a");
  EXPECT_EQ(entry.dtype, "F32");
  EXPECT_EQ(entry.shape, (spindle::Shape{1, 2}));
  EXPECT_EQ(entry.begin, 0U);
  EXPECT_EQ(entry.end, 8U);
  auto tensor = file.load_f32("a", {1, 2});
  EXPECT_EQ(tensor.shape(), (spindle::Shape{1, 2}));
  EXPECT_EQ(tensor.data()[0], 1.0F);
  EXPECT_EQ(tensor.data()[1], -2.5F);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(tensor.data()) % spindle::kTensorAlignment, 0U);
  tensor.data()[0] = 99.0F;
  EXPECT_EQ(file.load_f32("a", {1, 2}).data()[0], 1.0F);
}

TEST_F(SafetensorsSpecTest, AcceptsMetadataAndMultipleEntries) {
  const spindle::SafetensorsFile file(write(file_bytes(
      R"({"__metadata__":{"source":"test"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})")));
  ASSERT_EQ(file.entries().size(), 2U);
  EXPECT_EQ(file.entries().count("__metadata__"), 0U);
  EXPECT_EQ(file.load_f32("a", {1}).data()[0], 1.0F);
  EXPECT_EQ(file.load_f32("b", {1}).data()[0], -2.5F);
}

TEST_F(SafetensorsSpecTest, RejectsTruncatedLengthAndOversizedHeader) {
  for (std::size_t size = 0; size < 8; ++size) {
    EXPECT_THROW((void)spindle::SafetensorsFile(write(std::string(size, '\0'))),
                 std::runtime_error);
  }
  std::string bytes;
  append_u64(bytes, 1024);
  bytes += "{}";
  EXPECT_THROW((void)spindle::SafetensorsFile(write(bytes)), std::runtime_error);
}

TEST_F(SafetensorsSpecTest, RejectsMalformedHeadersAndRanges) {
  const std::vector<std::string> headers = {
      "{oops", R"({"a":{"dtype":"UNKNOWN","shape":[2],"data_offsets":[0,8]}})",
      R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})",
      R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}})",
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})"};
  for (const auto& header : headers) {
    SCOPED_TRACE(header);
    EXPECT_THROW((void)spindle::SafetensorsFile(write(file_bytes(header))), std::runtime_error);
  }
}

TEST_F(SafetensorsSpecTest, RejectsInvalidLoadRequests) {
  {
    const spindle::SafetensorsFile file(write(file_bytes(valid_header())));
    EXPECT_THROW((void)file.load_f32("missing", {1, 2}), std::runtime_error);
    EXPECT_THROW((void)file.load_f32("a", {2, 1}), std::runtime_error);
  }
  const auto other_path = dir / "integer.safetensors";
  std::filesystem::copy_file(
      write(file_bytes(R"({"a":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}})")), other_path);
  const spindle::SafetensorsFile integer_file(other_path);
  EXPECT_THROW((void)integer_file.load_f32("a", {2}), std::runtime_error);
}

TEST_F(SafetensorsSpecTest, RejectsMissingFile) {
  EXPECT_THROW((void)spindle::SafetensorsFile(dir / "missing"), std::runtime_error);
}

TEST_F(SafetensorsSpecTest, RandomCorruptionSucceedsOrThrowsRuntimeError) {
  // NOLINTNEXTLINE(bugprone-random-generator-seed): fixed seed for reproducibility.
  std::mt19937 rng(7193);
  const auto valid = file_bytes(valid_header());
  std::uniform_int_distribution<std::size_t> position(0, valid.size() - 1);
  std::uniform_int_distribution<int> byte(0, 255);
  for (int round = 0; round < 400; ++round) {
    SCOPED_TRACE(round);
    auto bytes = valid;
    for (int mutation = 0; mutation < 1 + (round % 4); ++mutation) {
      bytes[position(rng)] = static_cast<char>(byte(rng));
    }
    if (round % 3 == 0) {
      bytes.resize(position(rng));
    }
    const auto path = write(bytes);
    try {
      const spindle::SafetensorsFile file(path);
    } catch (const std::runtime_error&) {
      // Malformed files are the expected alternative to successful validation.
      continue;
    } catch (...) {
      ADD_FAILURE() << "exception other than std::runtime_error";
    }
  }
}

}  // namespace
