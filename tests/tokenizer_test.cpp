#include "spindle/tokenizer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using spindle::Tokenizer;

std::filesystem::path model_dir() {
  return std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" / "model";
}

struct TokenizerCase {
  std::string text;
  std::vector<std::int32_t> ids;
};

// Strings and their Hugging Face token ids from tools/tokenizer_reference.py.
std::vector<TokenizerCase> load_cases() {
  const std::filesystem::path path =
      std::filesystem::path(SPINDLE_REFERENCE_DIR) / "tokenizer" / "cases.json";
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error(path.string() + ": cannot open (run tools/check.sh to generate)");
  }
  std::vector<TokenizerCase> cases;
  for (const nlohmann::json& entry : nlohmann::json::parse(file)) {
    cases.push_back({.text = entry.at("text").get<std::string>(),
                     .ids = entry.at("ids").get<std::vector<std::int32_t>>()});
  }
  return cases;
}

// Loading takes about 0.1 s, so the tests share one instance.
const Tokenizer& tokenizer() {
  static const Tokenizer gpt2 = Tokenizer::load(model_dir());
  return gpt2;
}

void write_file(const std::filesystem::path& path, const std::string& contents) {
  std::ofstream file(path, std::ios::binary);
  file << contents;
}

struct ModelFiles {
  std::string vocab;
  std::string merges;
};

// A model directory holding the given vocab.json and merges.txt.
std::filesystem::path fake_model(const std::string& name, const ModelFiles& files) {
  const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / name;
  std::filesystem::create_directories(dir);
  write_file(dir / "vocab.json", files.vocab);
  write_file(dir / "merges.txt", files.merges);
  return dir;
}

std::string load_error(const std::filesystem::path& dir) {
  try {
    (void)Tokenizer::load(dir);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return "";
}

}  // namespace

TEST(Tokenizer, MatchesHuggingFaceOnReferenceStrings) {
  const std::vector<TokenizerCase> cases = load_cases();
  ASSERT_GE(cases.size(), 1000U);
  std::size_t mismatches = 0;
  for (const TokenizerCase& reference : cases) {
    if (tokenizer().encode(reference.text) != reference.ids && ++mismatches <= 10) {
      ADD_FAILURE() << "text: " << testing::PrintToString(reference.text) << "\n  got:      "
                    << testing::PrintToString(tokenizer().encode(reference.text))
                    << "\n  expected: " << testing::PrintToString(reference.ids);
    }
  }
  EXPECT_EQ(mismatches, 0U);
}

TEST(Tokenizer, DecodeRoundTripsReferenceStrings) {
  for (const TokenizerCase& reference : load_cases()) {
    EXPECT_EQ(tokenizer().decode(reference.ids), reference.text);
  }
}

TEST(Tokenizer, EncodesKnownIds) {
  EXPECT_EQ(tokenizer().encode("Hello, my name is"),
            (std::vector<std::int32_t>{15496, 11, 616, 1438, 318}));
  EXPECT_EQ(tokenizer().encode("<|endoftext|>"), std::vector<std::int32_t>{50256});
  EXPECT_EQ(tokenizer().vocab_size(), 50257U);
}

TEST(Tokenizer, DecodeCanSplitACharacter) {
  // The rocket emoji's 4 bytes span several tokens, so the first decodes to a partial character.
  const std::string rocket = "\xf0\x9f\x9a\x80";
  const std::vector<std::int32_t> tokens = tokenizer().encode(rocket);
  ASSERT_GT(tokens.size(), 1U);
  const std::string first = tokenizer().decode({tokens.data(), 1});
  EXPECT_LT(first.size(), rocket.size());
  EXPECT_EQ(first, rocket.substr(0, first.size()));
}

TEST(Tokenizer, RejectsInvalidInput) {
  EXPECT_THROW((void)tokenizer().encode("bad \xff byte"), std::invalid_argument);
  const std::vector<std::int32_t> out_of_range = {50257};
  EXPECT_THROW((void)tokenizer().decode(out_of_range), std::out_of_range);
  const std::vector<std::int32_t> negative = {-1};
  EXPECT_THROW((void)tokenizer().decode(negative), std::out_of_range);
}

// Stands in for fuzzing until a libFuzzer toolchain is chosen: random bytes must
// encode or throw invalid_argument, and valid text must round-trip.
TEST(Tokenizer, RandomBytesEncodeOrThrow) {
  std::mt19937 rng(1234);  // NOLINT(bugprone-random-generator-seed): reproducible on purpose
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> length(0, 24);
  std::size_t valid = 0;
  std::size_t invalid = 0;
  for (int round = 0; round < 5000; ++round) {
    std::string bytes;
    for (int i = length(rng); i > 0; --i) {
      bytes.push_back(static_cast<char>(byte(rng)));
    }
    try {
      EXPECT_EQ(tokenizer().decode(tokenizer().encode(bytes)), bytes);
      ++valid;
    } catch (const std::invalid_argument&) {
      ++invalid;
    }
  }
  // Both outcomes must occur, or the inputs are not exercising the check.
  EXPECT_GT(valid, 0U);
  EXPECT_GT(invalid, 0U);
}

TEST(TokenizerLoad, RejectsMissingFiles) {
  EXPECT_NE(load_error(std::filesystem::path(testing::TempDir()) / "no_such_model"), "");
}

TEST(TokenizerLoad, RejectsMalformedVocab) {
  using testing::HasSubstr;
  const auto vocab_error = [](const std::string& name, const std::string& vocab) {
    return load_error(fake_model(name, {.vocab = vocab, .merges = ""}));
  };
  EXPECT_THAT(vocab_error("vocab_not_json", "{oops"), HasSubstr("vocab.json"));
  EXPECT_THAT(vocab_error("vocab_array", "[1, 2]"), HasSubstr("vocab.json"));
  EXPECT_THAT(vocab_error("vocab_bad_id", R"({"a": 5})"), HasSubstr("out of range"));
  EXPECT_THAT(vocab_error("vocab_dup", R"({"a": 0, "b": 0})"), HasSubstr("duplicate"));
  // Valid ids, but no end-of-text token and no single-byte tokens.
  EXPECT_THAT(vocab_error("vocab_small", R"({"a": 0})"), HasSubstr("<|endoftext|>"));
}

TEST(TokenizerLoad, RejectsMalformedMerges) {
  using testing::HasSubstr;
  // A complete byte vocabulary plus the end-of-text token, copied from GPT-2's.
  std::ifstream real_vocab(model_dir() / "vocab.json");
  const nlohmann::json full = nlohmann::json::parse(real_vocab);
  nlohmann::json bytes_only = nlohmann::json::object();
  for (const auto& [token, id] : full.items()) {
    if (id.get<int>() < 256) {
      bytes_only[token] = id;
    }
  }
  bytes_only["<|endoftext|>"] = 256;
  const auto merges_error = [vocab = bytes_only.dump()](const std::string& name,
                                                        const std::string& merges) {
    return load_error(fake_model(name, {.vocab = vocab, .merges = merges}));
  };
  EXPECT_EQ(merges_error("merges_ok", "#version: 0.2\n"), "");
  EXPECT_THAT(merges_error("merges_one_part", "a\n"), HasSubstr("expected"));
  EXPECT_THAT(merges_error("merges_three_parts", "a b c\n"), HasSubstr("expected"));
  EXPECT_THAT(merges_error("merges_unknown", "a b\n"), HasSubstr("not in the vocabulary"));
}
