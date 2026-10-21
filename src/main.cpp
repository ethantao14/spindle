#include <charconv>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "spindle/tokenizer.h"
#include "spindle/version.h"

namespace {

constexpr int kUsageError = 2;
constexpr int kRuntimeError = 1;

constexpr std::string_view kUsage =
    "usage: spindle --version\n"
    "       spindle tokenize --model DIR TEXT\n"
    "       spindle detokenize --model DIR TOKEN_ID...\n";

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
  if (command != "tokenize" && command != "detokenize") {
    return usage();
  }
  try {
    const spindle::Tokenizer tokenizer = spindle::Tokenizer::load(args[3]);
    if (command == "tokenize") {
      return args.size() == 5 ? tokenize(tokenizer, args[4]) : usage();
    }
    return detokenize(tokenizer, args.subspan(4));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return kRuntimeError;
  }
}
