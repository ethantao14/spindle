#include <gtest/gtest.h>
#include <sys/wait.h>

#include <array>
#include <cstdio>
#include <string>

namespace {

struct CommandResult {
  int exit_code;
  std::string output;
};

// Runs the spindle CLI with the given arguments, capturing stdout and stderr.
CommandResult run_cli(const std::string& args) {
  const std::string command = std::string("'") + SPINDLE_CLI_PATH + "' " + args + " 2>&1";
  // Running the real binary through a shell is the point of this test.
  FILE* pipe = popen(command.c_str(), "r");  // NOLINT(bugprone-command-processor)
  if (pipe == nullptr) {
    return {.exit_code = -1, .output = ""};
  }
  std::string output;
  std::array<char, 256> buffer{};
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  const int status = pclose(pipe);
  return {.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1, .output = output};
}

}  // namespace

TEST(Cli, PrintsVersion) {
  const CommandResult result = run_cli("--version");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.output, "spindle 0.1.0\n");
}

TEST(Cli, RejectsUnknownArguments) {
  const CommandResult result = run_cli("--bogus");
  EXPECT_EQ(result.exit_code, 2);
  EXPECT_EQ(result.output, "usage: spindle --version\n");
}
