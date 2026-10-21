#include "spindle/unicode.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "support/reference.h"

namespace {

using spindle::char_class;
using spindle::CharClass;
using spindle::decode_utf8;

const char* class_name(CharClass value) {
  switch (value) {
    case CharClass::kLetter:
      return "letter";
    case CharClass::kNumber:
      return "number";
    case CharClass::kWhitespace:
      return "whitespace";
    case CharClass::kOther:
      return "other";
  }
  return "?";
}

}  // namespace

// Every code point except surrogates, classified by the Hugging Face pre-tokenizer.
TEST(CharClass, MatchesHuggingFaceOnEveryCodePoint) {
  const spindle::test::ReferenceTensor ranges = spindle::test::load_reference_tensor(
      std::filesystem::path(SPINDLE_REFERENCE_DIR) / "tokenizer" / "char_classes.bin");
  ASSERT_EQ(ranges.shape.size(), 2U);
  ASSERT_EQ(ranges.shape[1], 3U);
  std::size_t checked = 0;
  std::size_t mismatches = 0;
  for (std::size_t row = 0; row < ranges.shape[0]; ++row) {
    const auto first = static_cast<char32_t>(ranges.ints[row * 3]);
    const auto last = static_cast<char32_t>(ranges.ints[(row * 3) + 1]);
    const auto expected = static_cast<CharClass>(ranges.ints[(row * 3) + 2]);
    for (char32_t code_point = first; code_point <= last; ++code_point) {
      ++checked;
      if (char_class(code_point) != expected && ++mismatches <= 10) {
        ADD_FAILURE() << "U+" << std::hex << static_cast<std::uint32_t>(code_point) << ": got "
                      << class_name(char_class(code_point)) << ", expected "
                      << class_name(expected);
      }
    }
  }
  EXPECT_EQ(mismatches, 0U);
  EXPECT_EQ(checked, 0x110000U - 0x800U);  // all code points minus the surrogates
}

TEST(Utf8, DecodesEachSequenceLength) {
  EXPECT_EQ(decode_utf8("a\xc3\xa9\xe6\x9d\xb1\xf0\x9f\x9a\x80"),
            (std::vector<char32_t>{U'a', U'é', U'東', U'\U0001f680'}));
  EXPECT_EQ(decode_utf8(""), std::vector<char32_t>{});
  EXPECT_EQ(decode_utf8(std::string(1, '\0')), std::vector<char32_t>{U'\0'});
}

TEST(Utf8, RejectsInvalidSequences) {
  for (const std::string bad : {
           "\x80",              // continuation byte with no lead
           "\xff",              // never valid in UTF-8
           "\xc3",              // truncated 2-byte sequence
           "\xe6\x9d",          // truncated 3-byte sequence
           "\xc3\x41",          // lead byte followed by a non-continuation
           "\xc0\x80",          // overlong NUL
           "\xe0\x80\xaf",      // overlong '/'
           "\xf0\x82\x82\xac",  // overlong euro sign
           "\xed\xa0\x80",      // surrogate U+D800
           "\xf4\x90\x80\x80",  // U+110000, above the Unicode range
       }) {
    EXPECT_THROW(decode_utf8("ok" + bad), std::invalid_argument) << testing::PrintToString(bad);
  }
}

// The bytes after the view are a valid continuation, so only the length check catches this.
TEST(Utf8, RejectsSequenceCutOffByTheViewEnd) {
  const std::string text = "\xe6\x9d\xb1";
  EXPECT_THROW(decode_utf8(std::string_view(text).substr(0, 2)), std::invalid_argument);
}

TEST(Utf8, EncodeRoundTripsBoundaryCodePoints) {
  for (const char32_t code_point :
       {0x0U, 0x7fU, 0x80U, 0x7ffU, 0x800U, 0xd7ffU, 0xe000U, 0xffffU, 0x10000U, 0x10ffffU}) {
    std::string bytes;
    spindle::append_utf8(code_point, bytes);
    EXPECT_EQ(decode_utf8(bytes), std::vector<char32_t>{code_point})
        << static_cast<std::uint32_t>(code_point);
  }
}

// Stands in for fuzzing until a libFuzzer toolchain is chosen: random bytes must
// decode or throw invalid_argument, never crash (the ASan build checks this).
TEST(Utf8, RandomBytesDecodeOrThrow) {
  std::mt19937 rng(1234);  // NOLINT(bugprone-random-generator-seed): reproducible on purpose
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> length(0, 16);
  std::size_t valid = 0;
  std::size_t invalid = 0;
  for (int round = 0; round < 20000; ++round) {
    std::string bytes;
    for (int i = length(rng); i > 0; --i) {
      bytes.push_back(static_cast<char>(byte(rng)));
    }
    try {
      std::string encoded;
      for (const char32_t code_point : decode_utf8(bytes)) {
        spindle::append_utf8(code_point, encoded);
      }
      EXPECT_EQ(encoded, bytes);
      ++valid;
    } catch (const std::invalid_argument&) {
      ++invalid;
    }
  }
  // Both outcomes must occur, or the inputs are not exercising the check.
  EXPECT_GT(valid, 0U);
  EXPECT_GT(invalid, 0U);
}
