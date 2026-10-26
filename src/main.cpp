#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "spindle/gpt2.h"
#include "spindle/tokenizer.h"
#include "spindle/version.h"

namespace {

constexpr int kUsageError = 2;
constexpr int kRuntimeError = 1;

constexpr std::string_view kUsage =
    "usage: spindle --version\n"
    "       spindle tokenize --model DIR TEXT\n"
    "       spindle detokenize --model DIR TOKEN_ID...\n"
    "       spindle generate --model DIR --tokens N [--temperature T] [--top-k K] [--top-p P] "
    "[--seed S] PROMPT\n";

int usage() {
  std::fwrite(kUsage.data(), 1, kUsage.size(), stderr);
  return kUsageError;
}

int tokenize(const spindle::Tokenizer& tokenizer, std::string_view text) {
  const std::vector<std::int32_t> tokens = tokenizer.encode(text);
  std::string line;
  for (const std::int32_t token : tokens) {
    line += (line.empty() ? "" : " ") + std::to_string(token);
  }
  std::printf("%s\n", line.c_str());
  return 0;
}

int detokenize(const spindle::Tokenizer& tokenizer, std::span<char* const> ids) {
  std::vector<std::int32_t> tokens;
  for (const std::string_view id : ids) {
    std::int32_t token = 0;
    const auto [end, error] = std::from_chars(id.data(), id.data() + id.size(), token);
    if (error != std::errc() || end != id.data() + id.size()) {
      std::fprintf(stderr, "error: '%.*s' is not a token id\n", static_cast<int>(id.size()),
                   id.data());
      return kUsageError;
    }
    tokens.push_back(token);
  }
  const std::string text = tokenizer.decode(tokens);
  std::fwrite(text.data(), 1, text.size(), stdout);
  std::fputc('\n', stdout);
  return 0;
}

template <typename T>
T parse_number(std::string_view text) {
  T value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    throw std::invalid_argument("invalid numeric argument: " + std::string(text));
  }
  return value;
}

int generate(std::span<char* const> args) {
  std::size_t count = 0;
  std::uint64_t seed = 0;
  spindle::SamplingConfig config{.temperature = 0};
  try {
    if (args.size() < 7 || args.size() % 2 == 0 || std::string_view(args[4]) != "--tokens") {
      return usage();
    }
    try {
      count = parse_number<std::size_t>(args[5]);
    } catch (const std::invalid_argument&) {
      std::fprintf(stderr, "error: '%s' is not a token count\n", args[5]);
      return kUsageError;
    }
    for (std::size_t i = 6; i + 1 < args.size(); i += 2) {
      const std::string_view flag = args[i];
      if (i + 2 >= args.size()) {
        throw std::invalid_argument("flag missing its value or final prompt");
      }
      const std::string_view value = args[i + 1];
      if (flag == "--temperature") {
        config.temperature = parse_number<float>(value);
      } else if (flag == "--top-k") {
        config.top_k = parse_number<std::size_t>(value);
      } else if (flag == "--top-p") {
        config.top_p = parse_number<float>(value);
      } else if (flag == "--seed") {
        seed = parse_number<std::uint64_t>(value);
      } else {
        throw std::invalid_argument("unknown flag: " + std::string(flag));
      }
    }
    if (std::string_view(args.back()).starts_with("--")) {
      throw std::invalid_argument("unknown flag or flag missing its value");
    }
    if (!std::isfinite(config.temperature) || config.temperature < 0 ||
        !std::isfinite(config.top_p) || config.top_p <= 0 || config.top_p > 1) {
      throw std::invalid_argument("invalid sampling configuration");
    }
  } catch (const std::invalid_argument& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return kUsageError;
  }
  const spindle::Tokenizer tokenizer = spindle::Tokenizer::load(args[3]);
  const spindle::Gpt2Model model = spindle::Gpt2Model::load(args[3]);
  const std::string_view prompt = args.back();
  const auto prompt_tokens = tokenizer.encode(prompt);
  std::fwrite(prompt.data(), 1, prompt.size(), stdout);
  std::fflush(stdout);
  spindle::generate(model, prompt_tokens, count, config, seed, [&](std::int32_t token) {
    const std::string bytes = tokenizer.decode(std::span(&token, 1));
    std::fwrite(bytes.data(), 1, bytes.size(), stdout);
    std::fflush(stdout);
  });
  std::fputc('\n', stdout);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  if (args.size() == 2 && std::string_view(args[1]) == "--version") {
    const std::string_view version = spindle::version();
    std::printf("spindle %.*s\n", static_cast<int>(version.size()), version.data());
    return 0;
  }
  if (args.size() < 5 || std::string_view(args[2]) != "--model") {
    return usage();
  }
  const std::string_view command = args[1];
  try {
    if (command == "generate") {
      return generate(args);
    }
    if (command == "tokenize" && args.size() == 5) {
      return tokenize(spindle::Tokenizer::load(args[3]), args[4]);
    }
    if (command == "detokenize") {
      return detokenize(spindle::Tokenizer::load(args[3]), args.subspan(4));
    }
    return usage();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return kRuntimeError;
  }
}
