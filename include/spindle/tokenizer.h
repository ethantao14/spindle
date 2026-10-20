#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace spindle {

// GPT-2's byte-level BPE tokenizer, loaded from vocab.json and merges.txt.
class Tokenizer {
 public:
  // Reads model_dir/vocab.json and model_dir/merges.txt. Throws std::runtime_error
  // if either is missing or malformed.
  static Tokenizer load(const std::filesystem::path& model_dir);

  // Throws std::invalid_argument if text is not valid UTF-8.
  [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;

  // Returns raw bytes, which are not valid UTF-8 if a token splits a character.
  // Throws std::out_of_range for an id outside the vocabulary.
  [[nodiscard]] std::string decode(std::span<const std::int32_t> tokens) const;

  [[nodiscard]] std::size_t vocab_size() const { return token_bytes_.size(); }

 private:
  Tokenizer() = default;

  // Appends the BPE tokens of one pre-tokenized piece of plain text.
  void encode_piece(std::string_view piece, std::vector<std::int32_t>& out) const;
  void encode_plain(std::string_view text, std::vector<std::int32_t>& out) const;

  std::vector<std::string> token_bytes_;  // id -> the raw bytes it stands for
  // Merges and the vocabulary are keyed by GPT-2's printable stand-ins for bytes.
  std::unordered_map<std::string, std::int32_t> ids_;
  std::unordered_map<std::string, std::size_t> merge_ranks_;  // "left right" -> rank
  std::array<std::string, 256> byte_symbols_;
  std::int32_t end_of_text_id_ = -1;
};

}  // namespace spindle
