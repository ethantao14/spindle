#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

struct CommandResult {
  int exit_code;
  std::string output;
  std::string error;
};

std::string shell_quote(const std::string& text) {
  std::string quoted = "'";
  for (const char ch : text) {
    quoted += ch == '\'' ? "'\\''" : std::string(1, ch);
  }
  return quoted + "'";
}

std::filesystem::path model_dir() {
  return std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" / "model";
}

CommandResult run_cli(const std::string& args) {
  // Capture stderr separately to verify the CLI's stream contract.
  std::string pattern =
      (std::filesystem::path(testing::TempDir()) / "threads_spec_XXXXXX").string();
  const int fd = mkstemp(pattern.data());
  if (fd == -1) {
    throw std::runtime_error("cannot create stderr capture");
  }
  close(fd);
  const std::string command =
      shell_quote(SPINDLE_CLI_PATH) + " " + args + " 2>" + shell_quote(pattern);
  FILE* pipe = popen(command.c_str(), "r");  // NOLINT(bugprone-command-processor)
  if (pipe == nullptr) {
    std::filesystem::remove(pattern);
    throw std::runtime_error("cannot run spindle CLI");
  }
  std::string output;
  std::array<char, 256> buffer{};
  while (const auto count = fread(buffer.data(), 1, buffer.size(), pipe)) {
    output.append(buffer.data(), count);
  }
  const int status = pclose(pipe);
  std::ifstream file(pattern, std::ios::binary);
  const std::string error{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  std::filesystem::remove(pattern);
  return {
      .exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1, .output = output, .error = error};
}

std::string arguments(const std::string& flags, int count = 0) {
  return "generate --model " + shell_quote(model_dir().string()) + " --tokens " +
         std::to_string(count) + " " + flags + " 'Hello, my name is'";
}

TEST(ThreadsCliSpec, GreedyOutputIsByteIdenticalAndDefaultIsOne) {
  const auto single = run_cli(arguments("--threads 1", 20));
  const auto parallel = run_cli(arguments("--threads 4", 20));
  const auto implicit = run_cli(arguments("", 20));
  for (const auto* result : {&single, &parallel, &implicit}) {
    ASSERT_EQ(result->exit_code, 0) << result->error;
    EXPECT_TRUE(result->error.empty());
    ASSERT_FALSE(result->output.empty());
  }
  EXPECT_EQ(single.output, parallel.output);
  EXPECT_EQ(single.output, implicit.output);
}

TEST(ThreadsCliSpec, ThreadsBeforeAndAfterSamplingFlagsPreservesBytes) {
  const auto single =
      run_cli(arguments("--threads 1 --temperature 0.8 --top-k 40 --top-p 0.9 --seed 7", 5));
  const auto before =
      run_cli(arguments("--threads 4 --temperature 0.8 --top-k 40 --top-p 0.9 --seed 7", 5));
  const auto after =
      run_cli(arguments("--temperature 0.8 --top-k 40 --top-p 0.9 --seed 7 --threads 4", 5));
  for (const auto* result : {&single, &before, &after}) {
    ASSERT_EQ(result->exit_code, 0) << result->error;
    EXPECT_TRUE(result->error.empty());
    ASSERT_FALSE(result->output.empty());
  }
  EXPECT_EQ(single.output, before.output);
  EXPECT_EQ(single.output, after.output);
}

TEST(ThreadsCliSpec, RejectsInvalidThreadCountsOnStderr) {
  for (const std::string value : {"0", "-1", "x"}) {
    SCOPED_TRACE(value);
    const auto result = run_cli(arguments("--threads " + value));
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_TRUE(result.output.empty());
    EXPECT_FALSE(result.error.empty());
  }
}

}  // namespace
