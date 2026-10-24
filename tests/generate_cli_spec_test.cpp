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

#include "spindle/tokenizer.h"
#include "support/reference.h"

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

std::string arguments(const std::string& count, const std::string& prompt) {
  return "generate --model " + shell_quote(model_dir().string()) + " --tokens " +
         shell_quote(count) + " " + shell_quote(prompt);
}

TEST(GenerateCliSpecTest, PrintsPromptAndReferenceContinuation) {
  // Prompt 1 from tests/reference/gpt2_prompts.json.
  const std::string prompt = "Hello, my name is";
  const auto tokenizer = spindle::Tokenizer::load(model_dir());
  const auto dir = std::filesystem::path(SPINDLE_REFERENCE_DIR) / "gpt2" / "prompt_1";
  EXPECT_EQ(tokenizer.encode(prompt),
            spindle::test::load_reference_tensor(dir / "tokens.bin").ints);
  const auto continuation = spindle::test::load_reference_tensor(dir / "greedy.bin").ints;
  const auto result = run_cli(arguments("100", prompt));
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.output, prompt + tokenizer.decode(continuation) + "\n");
  EXPECT_TRUE(result.error.empty());
}

TEST(GenerateCliSpecTest, ZeroTokensPrintsOnlyPrompt) {
  const std::string prompt = "A prompt with 'quotes' and\na second line";
  const auto result = run_cli(arguments("0", prompt));
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.output, prompt + "\n");
  EXPECT_TRUE(result.error.empty());
}

TEST(GenerateCliSpecTest, RejectsNonNumericCount) {
  const auto result = run_cli(arguments("not-a-number", "Hello"));
  EXPECT_EQ(result.exit_code, 2);
  EXPECT_TRUE(result.output.empty());
  EXPECT_FALSE(result.error.empty());
}

TEST(GenerateCliSpecTest, RejectsWrongArgumentShapeWithUsage) {
  const std::array<std::string, 3> cases = {
      "generate", "generate --model " + shell_quote(model_dir().string()) + " --tokens 1",
      arguments("1", "Hello") + " extra"};
  for (const auto& args : cases) {
    SCOPED_TRACE(args);
    const auto result = run_cli(args);
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_TRUE(result.output.empty());
    EXPECT_NE(result.error.find("usage:"), std::string::npos);
    EXPECT_NE(result.error.find("spindle"), std::string::npos);
  }
}

TEST(GenerateCliSpecTest, RejectsMissingAndBadModelDirectories) {
  // The fixture directory exists but contains no model; its child is missing.
  std::string pattern =
      (std::filesystem::path(testing::TempDir()) / "bad_model_spec_XXXXXX").string();
  ASSERT_NE(mkdtemp(pattern.data()), nullptr);
  for (const auto& dir :
       {std::filesystem::path(pattern) / "missing", std::filesystem::path(pattern)}) {
    SCOPED_TRACE(dir.string());
    const auto result =
        run_cli("generate --model " + shell_quote(dir.string()) + " --tokens 1 Hello");
    EXPECT_EQ(result.exit_code, 1);
    EXPECT_TRUE(result.output.empty());
    EXPECT_EQ(result.error.find("error: "), 0U);
  }
  std::filesystem::remove_all(pattern);
}

}  // namespace
