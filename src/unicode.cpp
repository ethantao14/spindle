#include "spindle/unicode.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace spindle {

namespace {

struct CodePointRange {
  char32_t first;
  char32_t last;
};

#include "unicode_data.inc"

// The White_Space property. Python's unicodedata does not expose it, and it rarely changes.
constexpr std::array<CodePointRange, 10> kWhitespaceRanges = {{
    {.first = 0x9, .last = 0xd},
    {.first = 0x20, .last = 0x20},
    {.first = 0x85, .last = 0x85},
    {.first = 0xa0, .last = 0xa0},
    {.first = 0x1680, .last = 0x1680},
    {.first = 0x2000, .last = 0x200a},
    {.first = 0x2028, .last = 0x2029},
    {.first = 0x202f, .last = 0x202f},
    {.first = 0x205f, .last = 0x205f},
    {.first = 0x3000, .last = 0x3000},
}};

bool in_ranges(std::span<const CodePointRange> ranges, char32_t code_point) {
  // First range whose last code point is at or after this one.
  const auto range = std::ranges::lower_bound(ranges, code_point, {}, &CodePointRange::last);
  return range != ranges.end() && range->first <= code_point;
}

[[noreturn]] void invalid_utf8(std::size_t offset) {
  throw std::invalid_argument("invalid UTF-8 at byte " + std::to_string(offset));
}

}  // namespace

CharClass char_class(char32_t code_point) {
  if (in_ranges(kLetterRanges, code_point)) {
    return CharClass::kLetter;
  }
  if (in_ranges(kNumberRanges, code_point)) {
    return CharClass::kNumber;
  }
  if (in_ranges(kWhitespaceRanges, code_point)) {
    return CharClass::kWhitespace;
  }
  return CharClass::kOther;
}

std::vector<char32_t> decode_utf8(std::string_view text) {
  std::vector<char32_t> code_points;
  std::size_t offset = 0;
  while (offset < text.size()) {
    const auto lead = static_cast<unsigned char>(text[offset]);
    std::size_t length = 0;
    char32_t code_point = 0;
    char32_t smallest = 0;  // below this the sequence is overlong
    if (lead < 0x80) {
      length = 1;
      code_point = lead;
    } else if ((lead & 0xe0U) == 0xc0) {
      length = 2;
      code_point = lead & 0x1fU;
      smallest = 0x80;
    } else if ((lead & 0xf0U) == 0xe0) {
      length = 3;
      code_point = lead & 0x0fU;
      smallest = 0x800;
    } else if ((lead & 0xf8U) == 0xf0) {
      length = 4;
      code_point = lead & 0x07U;
      smallest = 0x10000;
    } else {
      invalid_utf8(offset);
    }
    if (length > text.size() - offset) {
      invalid_utf8(offset);
    }
    for (std::size_t i = 1; i < length; ++i) {
      const auto continuation = static_cast<unsigned char>(text[offset + i]);
      if ((continuation & 0xc0U) != 0x80) {
        invalid_utf8(offset);
      }
      code_point = (code_point << 6U) | (continuation & 0x3fU);
    }
    const bool surrogate = code_point >= 0xd800 && code_point <= 0xdfff;
    if (code_point < smallest || surrogate || code_point > 0x10ffff) {
      invalid_utf8(offset);
    }
    code_points.push_back(code_point);
    offset += length;
  }
  return code_points;
}

void append_utf8(char32_t code_point, std::string& out) {
  const auto byte = [&out](char32_t value) { out.push_back(static_cast<char>(value)); };
  if (code_point < 0x80) {
    byte(code_point);
  } else if (code_point < 0x800) {
    byte(0xc0U | (code_point >> 6U));
    byte(0x80U | (code_point & 0x3fU));
  } else if (code_point < 0x10000) {
    byte(0xe0U | (code_point >> 12U));
    byte(0x80U | ((code_point >> 6U) & 0x3fU));
    byte(0x80U | (code_point & 0x3fU));
  } else {
    byte(0xf0U | (code_point >> 18U));
    byte(0x80U | ((code_point >> 12U) & 0x3fU));
    byte(0x80U | ((code_point >> 6U) & 0x3fU));
    byte(0x80U | (code_point & 0x3fU));
  }
}

}  // namespace spindle
