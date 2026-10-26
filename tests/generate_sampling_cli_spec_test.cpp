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
      (std::filesystem::path(testing::TempDir()) / "generate_spec_XXXXXX").string();
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

TEST(GenerateSamplingCliSpec, SeedIsRepeatableAndFlagsCanBeReordered) {
  const auto first = run_cli(arguments("--temperature 0.8 --seed 7 --top-k 40 --top-p 0.9", 20));
  const auto same = run_cli(arguments("--temperature 0.8 --seed 7 --top-k 40 --top-p 0.9", 20));
  const auto reordered =
      run_cli(arguments("--top-p 0.9 --top-k 40 --seed 7 --temperature 0.8", 20));
  const auto other = run_cli(arguments("--temperature 0.8 --seed 8 --top-k 40 --top-p 0.9", 20));
  for (const auto* result : {&first, &same, &reordered, &other}) {
    ASSERT_EQ(result->exit_code, 0) << result->error;
    EXPECT_TRUE(result->error.empty());
    EXPECT_EQ(result->output.find("Hello, my name is"), 0U);
    ASSERT_FALSE(result->output.empty());
    EXPECT_EQ(result->output.back(), '\n');
  }
  EXPECT_EQ(first.output, same.output);
  EXPECT_EQ(first.output, reordered.output);
  EXPECT_NE(first.output, other.output);
}

TEST(GenerateSamplingCliSpec, TopKWithoutTemperaturePreservesGreedyBytes) {
  const auto result = run_cli(arguments("--top-k 40 --top-p 0.9", 5));
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(result.error.empty());
  EXPECT_EQ(result.output, "Hello, my name is John. I'm a\n");
}

TEST(GenerateSamplingCliSpec, DefaultSeedIsZero) {
  const auto implicit = run_cli(arguments("--temperature 0.8", 1));
  const auto explicit_zero = run_cli(arguments("--seed 0 --temperature 0.8", 1));
  ASSERT_EQ(implicit.exit_code, 0) << implicit.error;
  ASSERT_EQ(explicit_zero.exit_code, 0) << explicit_zero.error;
  EXPECT_TRUE(implicit.error.empty());
  EXPECT_TRUE(explicit_zero.error.empty());
  EXPECT_EQ(implicit.output, explicit_zero.output);
}

TEST(GenerateSamplingCliSpec, RejectsInvalidFlagsOnStderr) {
  const std::array<std::string, 12> cases = {
      "--unknown 1", "--temperature nope", "--top-k nope",        "--top-p nope",
      "--seed nope", "--temperature -0.1", "--top-p 0",           "--top-p -0.1",
      "--top-p 1.1", "--top-k -1",         "--temperature 1oops", "--seed 7oops"};
  for (const auto& flags : cases) {
    SCOPED_TRACE(flags);
    const auto result = run_cli(arguments(flags));
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_TRUE(result.output.empty());
    EXPECT_FALSE(result.error.empty());
  }
}

TEST(GenerateSamplingCliSpec, RejectsEveryFlagWithoutValue) {
  for (const std::string flag : {"--temperature", "--top-k", "--top-p", "--seed"}) {
    SCOPED_TRACE(flag);
    const auto result =
        run_cli("generate --model " + shell_quote(model_dir().string()) + " --tokens 0 " + flag);
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_TRUE(result.output.empty());
    EXPECT_FALSE(result.error.empty());
  }
}

TEST(GenerateSamplingCliSpec, BadModelStillExitsOne) {
  const auto result =
      run_cli("generate --model /nonexistent --tokens 1 --temperature 0.8 --seed 7 Hello");
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_TRUE(result.output.empty());
  EXPECT_FALSE(result.error.empty());
}

}  // namespace
