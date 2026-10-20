#include "spindle/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "spindle/unicode.h"

namespace spindle {

namespace {

constexpr std::string_view kEndOfText = "<|endoftext|>";

// GPT-2 maps each byte to a printable character so tokens never hold raw
// whitespace or control bytes. Printable Latin-1 bytes map to themselves.
std::array<char32_t, 256> byte_to_code_point() {
  std::array<char32_t, 256> mapping{};
  char32_t next_unprintable = 256;
  for (std::size_t byte = 0; byte < mapping.size(); ++byte) {
    const bool printable = (byte >= '!' && byte <= '~') || (byte >= 0xa1 && byte <= 0xac) ||
                           (byte >= 0xae && byte <= 0xff);
    mapping[byte] = printable ? static_cast<char32_t>(byte) : next_unprintable++;
  }
  return mapping;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot open " + path.string());
  }
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

[[noreturn]] void malformed(const std::filesystem::path& path, const std::string& reason) {
  throw std::runtime_error(path.string() + ": " + reason);
}

bool is_contraction_suffix(std::u32string_view rest) {
  constexpr std::array<std::u32string_view, 7> kSuffixes = {U"s", U"t",  U"re", U"ve",
                                                            U"m", U"ll", U"d"};
  return std::ranges::any_of(
      kSuffixes, [rest](std::u32string_view suffix) { return rest.starts_with(suffix); });
}

std::size_t contraction_length(std::u32string_view rest) {
  return (rest.starts_with(U"re") || rest.starts_with(U"ve") || rest.starts_with(U"ll")) ? 3 : 2;
}

// Length in code points of the piece GPT-2's pre-tokenizer regex matches at the
// start of text: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::size_t next_piece_length(std::u32string_view text) {
  if (text[0] == U'\'' && is_contraction_suffix(text.substr(1))) {
    return contraction_length(text.substr(1));
  }
  const std::size_t start = (text[0] == U' ' && text.size() > 1) ? 1 : 0;
  const CharClass run_class = char_class(text[start]);
  if (run_class != CharClass::kWhitespace) {
    std::size_t end = start + 1;
    while (end < text.size() && char_class(text[end]) == run_class) {
      ++end;
    }
    return end;
  }
  std::size_t end = 1;
  while (end < text.size() && char_class(text[end]) == CharClass::kWhitespace) {
    ++end;
  }
  // \s+(?!\S) leaves the last space of a run to attach to the word after it.
  const bool followed_by_text = end < text.size();
  return (followed_by_text && end > 1) ? end - 1 : end;
}

}  // namespace

Tokenizer Tokenizer::load(const std::filesystem::path& model_dir) {
  Tokenizer tokenizer;
  const std::array<char32_t, 256> mapping = byte_to_code_point();
  std::unordered_map<char32_t, char> code_point_to_byte;
  for (std::size_t byte = 0; byte < mapping.size(); ++byte) {
    append_utf8(mapping[byte], tokenizer.byte_symbols_[byte]);
    code_point_to_byte[mapping[byte]] = static_cast<char>(byte);
  }

  const std::filesystem::path vocab_path = model_dir / "vocab.json";
  const nlohmann::json vocab = nlohmann::json::parse(read_file(vocab_path), nullptr, false);
  if (!vocab.is_object() || vocab.empty() ||
      vocab.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    malformed(vocab_path, "expected a non-empty object of token -> id");
  }
  tokenizer.token_bytes_.resize(vocab.size());
  std::vector<bool> seen(vocab.size(), false);
  for (const auto& [token, id_json] : vocab.items()) {
    if (!id_json.is_number_integer() || id_json.get<std::int64_t>() < 0 ||
        std::cmp_greater_equal(id_json.get<std::int64_t>(), vocab.size())) {
      malformed(vocab_path, "token id out of range for '" + token + "'");
    }
    const auto id = id_json.get<std::int32_t>();
    if (seen[static_cast<std::size_t>(id)]) {
      malformed(vocab_path, "duplicate id " + std::to_string(id));
    }
    seen[static_cast<std::size_t>(id)] = true;
    std::string bytes;
    for (const char32_t code_point : decode_utf8(token)) {
      const auto byte = code_point_to_byte.find(code_point);
      if (byte == code_point_to_byte.end()) {
        malformed(vocab_path, "token '" + token + "' has a character outside the byte mapping");
      }
      bytes.push_back(byte->second);
    }
    tokenizer.token_bytes_[static_cast<std::size_t>(id)] = bytes;
    tokenizer.ids_.emplace(token, id);
  }
  const auto end_of_text = tokenizer.ids_.find(std::string(kEndOfText));
  if (end_of_text == tokenizer.ids_.end()) {
    malformed(vocab_path, "missing " + std::string(kEndOfText));
  }
  tokenizer.end_of_text_id_ = end_of_text->second;
  for (const std::string& symbol : tokenizer.byte_symbols_) {
    if (!tokenizer.ids_.contains(symbol)) {
      malformed(vocab_path, "missing the single-byte token '" + symbol + "'");
    }
  }

  const std::filesystem::path merges_path = model_dir / "merges.txt";
  std::istringstream merges(read_file(merges_path));
  std::string line;
  while (std::getline(merges, line)) {
    if (line.empty() || line.starts_with("#version")) {
      continue;
    }
    const std::size_t space = line.find(' ');
    if (space == std::string::npos || line.find(' ', space + 1) != std::string::npos) {
      malformed(merges_path, "expected 'left right', got '" + line + "'");
    }
    const std::string merged = line.substr(0, space) + line.substr(space + 1);
    if (!tokenizer.ids_.contains(merged)) {
      malformed(merges_path, "merge result '" + merged + "' is not in the vocabulary");
    }
    tokenizer.merge_ranks_.emplace(line, tokenizer.merge_ranks_.size());
  }
  return tokenizer;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const {
  std::vector<std::int32_t> tokens;
  // The end-of-text marker is matched in the raw text, before pre-tokenizing.
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t marker = text.find(kEndOfText, start);
    const std::size_t end = marker == std::string_view::npos ? text.size() : marker;
    encode_plain(text.substr(start, end - start), tokens);
    if (marker == std::string_view::npos) {
      break;
    }
    tokens.push_back(end_of_text_id_);
    start = marker + kEndOfText.size();
  }
  return tokens;
}

void Tokenizer::encode_plain(std::string_view text, std::vector<std::int32_t>& out) const {
  const std::vector<char32_t> code_points = decode_utf8(text);
  const std::u32string_view remaining_all(code_points.data(), code_points.size());
  std::size_t position = 0;
  std::string piece;
  while (position < code_points.size()) {
    const std::size_t length = next_piece_length(remaining_all.substr(position));
    piece.clear();
    for (std::size_t i = position; i < position + length; ++i) {
      append_utf8(code_points[i], piece);
    }
    encode_piece(piece, out);
    position += length;
  }
}

void Tokenizer::encode_piece(std::string_view piece, std::vector<std::int32_t>& out) const {
  std::vector<std::string> symbols;
  symbols.reserve(piece.size());
  for (const char byte : piece) {
    symbols.push_back(byte_symbols_[static_cast<unsigned char>(byte)]);
  }
  // Repeatedly merge the adjacent pair learned earliest, until no pair has a merge.
  while (symbols.size() > 1) {
    std::size_t best_rank = std::numeric_limits<std::size_t>::max();
    std::size_t best_index = 0;
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto rank = merge_ranks_.find(symbols[i] + ' ' + symbols[i + 1]);
      if (rank != merge_ranks_.end() && rank->second < best_rank) {
        best_rank = rank->second;
        best_index = i;
      }
    }
    if (best_rank == std::numeric_limits<std::size_t>::max()) {
      break;
    }
    symbols[best_index] += symbols[best_index + 1];
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index) + 1);
  }
  for (const std::string& symbol : symbols) {
    // Single bytes and every merge result are in the vocabulary (checked at load).
    out.push_back(ids_.at(symbol));
  }
}

std::string Tokenizer::decode(std::span<const std::int32_t> tokens) const {
  std::string text;
  for (const std::int32_t token : tokens) {
    if (token < 0 || std::cmp_greater_equal(token, token_bytes_.size())) {
      throw std::out_of_range("token id " + std::to_string(token) + " is outside the vocabulary");
    }
    text += token_bytes_[static_cast<std::size_t>(token)];
  }
  return text;
}

}  // namespace spindle
