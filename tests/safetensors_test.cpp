#include "spindle/safetensors.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using spindle::SafetensorsFile;
using spindle::Shape;
using spindle::Tensor;
using testing::HasSubstr;

// Two F32 tensors: "a" [2, 3] holding 0..5 and "b" [2] holding 10, 11.
constexpr const char* kValidHeader =
    R"({"__metadata__": {"format": "pt"},)"
    R"( "a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},)"
    R"( "b": {"dtype": "F32", "shape": [2], "data_offsets": [24, 32]}})";

std::string valid_data() {
  const std::vector<float> values = {0, 1, 2, 3, 4, 5, 10, 11};
  std::string bytes(values.size() * sizeof(float), '\0');
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

// The 8-byte little-endian header length, the header, then the data.
std::string file_bytes(const std::string& header, const std::string& data) {
  const std::uint64_t length = header.size();
  std::string bytes(sizeof(length), '\0');
  std::memcpy(bytes.data(), &length, sizeof(length));
  return bytes + header + data;
}

std::filesystem::path write_file(const std::filesystem::path& name, const std::string& bytes) {
  const std::filesystem::path path = std::filesystem::path(testing::TempDir()) / name;
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return path;
}

std::string open_error(const std::string& name, const std::string& bytes) {
  try {
    const SafetensorsFile file(write_file(name, bytes));
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return "";
}

// A file whose header has one tensor entry with the given JSON value.
std::string single_entry(const std::string& entry) {
  return file_bytes(R"({"t": )" + entry + "}", valid_data());
}

}  // namespace

TEST(Safetensors, LoadsTensorsIntoAlignedStorage) {
  // The header length (and so the data start) is not a multiple of 4 here.
  const std::string header = std::string(kValidHeader) + " ";
  ASSERT_NE((8 + header.size()) % 4, 0U);
  const SafetensorsFile file(write_file("valid.safetensors", file_bytes(header, valid_data())));
  EXPECT_EQ(file.entries().size(), 2U);

  const Tensor a = file.load_f32("a", {2, 3});
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % spindle::kTensorAlignment, 0U);
  EXPECT_EQ(std::vector<float>(a.data(), a.data() + 6), (std::vector<float>{0, 1, 2, 3, 4, 5}));
  const Tensor b = file.load_f32("b", {2});
  EXPECT_EQ(std::vector<float>(b.data(), b.data() + 2), (std::vector<float>{10, 11}));
}

TEST(Safetensors, LoadF32ChecksNameShapeAndDtype) {
  const std::string header = R"({"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},)"
                             R"( "h": {"dtype": "F16", "shape": [4], "data_offsets": [24, 32]}})";
  const SafetensorsFile file(write_file("checks.safetensors", file_bytes(header, valid_data())));
  EXPECT_THROW((void)file.load_f32("missing", {2, 3}), std::runtime_error);
  EXPECT_THROW((void)file.load_f32("a", {3, 2}), std::runtime_error);
  EXPECT_THROW((void)file.load_f32("h", {4}), std::runtime_error);
}

TEST(Safetensors, RejectsBadFileStructure) {
  EXPECT_THAT(open_error("missing_dir/none", ""), HasSubstr("cannot open"));
  EXPECT_THAT(open_error("tiny", "1234"), HasSubstr("too small"));
  std::string huge_length(8, '\xff');
  EXPECT_THAT(open_error("huge_length", huge_length + "{}"), HasSubstr("does not fit"));
  EXPECT_THAT(open_error("past_end", file_bytes("{}", "").substr(0, 8) + "{"),
              HasSubstr("does not fit"));
  EXPECT_THAT(open_error("not_json", file_bytes("{oops", "")), HasSubstr("not a JSON object"));
  EXPECT_THAT(open_error("array", file_bytes("[]", "")), HasSubstr("not a JSON object"));
}

TEST(Safetensors, RejectsBadEntries) {
  const auto entry_error = [](const std::string& name, const std::string& entry) {
    return open_error(name, single_entry(entry));
  };
  EXPECT_THAT(entry_error("no_dtype", R"({"shape": [2], "data_offsets": [0, 8]})"),
              HasSubstr("expected dtype"));
  EXPECT_THAT(entry_error("one_offset", R"({"dtype": "F32", "shape": [2], "data_offsets": [0]})"),
              HasSubstr("expected dtype"));
  EXPECT_THAT(entry_error("bad_dtype", R"({"dtype": "F12", "shape": [2], "data_offsets": [0, 8]})"),
              HasSubstr("unknown dtype"));
  EXPECT_THAT(entry_error("neg_dim", R"({"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]})"),
              HasSubstr("non-negative"));
  EXPECT_THAT(entry_error("neg_off", R"({"dtype": "F32", "shape": [2], "data_offsets": [-8, 0]})"),
              HasSubstr("non-negative"));
  EXPECT_THAT(entry_error("reversed", R"({"dtype": "F32", "shape": [0], "data_offsets": [8, 0]})"),
              HasSubstr("outside the data"));
  EXPECT_THAT(
      entry_error("past_data", R"({"dtype": "F32", "shape": [2], "data_offsets": [32, 40]})"),
      HasSubstr("outside the data"));
  EXPECT_THAT(entry_error("size", R"({"dtype": "F32", "shape": [3], "data_offsets": [0, 8]})"),
              HasSubstr("does not match"));
  EXPECT_THAT(entry_error("overflow", R"({"dtype": "F32", "shape": [4294967296, 4294967296],)"
                                      R"( "data_offsets": [0, 8]})"),
              HasSubstr("too large"));
}

TEST(Safetensors, RejectsOverlappingTensors) {
  const std::string header = R"({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},)"
                             R"( "b": {"dtype": "F32", "shape": [4], "data_offsets": [12, 28]}})";
  EXPECT_THAT(open_error("overlap", file_bytes(header, valid_data())), HasSubstr("overlap"));
}

// Stands in for fuzzing until a libFuzzer toolchain is chosen: corrupted and
// truncated files must open or throw runtime_error, never crash (checked under ASan).
TEST(Safetensors, CorruptedFilesOpenOrThrow) {
  const std::string valid = file_bytes(kValidHeader, valid_data());
  std::mt19937 rng(1234);  // NOLINT(bugprone-random-generator-seed): reproducible on purpose
  std::uniform_int_distribution<std::size_t> position(0, valid.size() - 1);
  std::uniform_int_distribution<int> byte(0, 255);
  std::size_t opened = 0;
  std::size_t rejected = 0;
  for (int round = 0; round < 2000; ++round) {
    std::string bytes = valid;
    if (round % 4 == 0) {
      bytes.resize(position(rng));
    } else {
      for (int flips = (round % 3) + 1; flips > 0; --flips) {
        bytes[position(rng)] = static_cast<char>(byte(rng));
      }
    }
    try {
      const SafetensorsFile file(write_file("corrupt.safetensors", bytes));
      for (const auto& [name, entry] : file.entries()) {
        if (entry.dtype == "F32") {
          (void)file.load_f32(name, entry.shape);
        }
      }
      ++opened;
    } catch (const std::runtime_error&) {
      ++rejected;
    }
  }
  // Both outcomes must occur, or the inputs are not exercising the checks.
  EXPECT_GT(opened, 0U);
  EXPECT_GT(rejected, 0U);
}
