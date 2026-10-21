#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace spindle {

// The character classes GPT-2's pre-tokenizer splits text on. Values match
// tools/tokenizer_reference.py.
enum class CharClass : std::uint8_t { kLetter = 0, kNumber = 1, kWhitespace = 2, kOther = 3 };

// Unicode general category L* or N*, the White_Space property, or anything else.
CharClass char_class(char32_t code_point);

// Throws std::invalid_argument on invalid UTF-8: truncated or overlong sequences,
// surrogates, and values above U+10FFFF.
std::vector<char32_t> decode_utf8(std::string_view text);

// Appends the UTF-8 encoding of a valid code point.
void append_utf8(char32_t code_point, std::string& out);

}  // namespace spindle
