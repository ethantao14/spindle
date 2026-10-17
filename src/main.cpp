#include <cstdio>
#include <span>
#include <string_view>

#include "spindle/version.h"

namespace {

constexpr int kUsageError = 2;

void print_usage(std::FILE* out) { std::fputs("usage: spindle --version\n", out); }

}  // namespace

int main(int argc, char** argv) {
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  if (args.size() == 2 && std::string_view(args[1]) == "--version") {
    const std::string_view version = spindle::version();
    std::printf("spindle %.*s\n", static_cast<int>(version.size()), version.data());
    return 0;
  }
  print_usage(stderr);
  return kUsageError;
}
