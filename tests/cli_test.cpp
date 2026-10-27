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

std::string model_dir() { return std::string(SPINDLE_REFERENCE_DIR) + "/gpt2/model"; }

}  // namespace

TEST(Cli, PrintsVersion) {
  const CommandResult result = run_cli("--version");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.output, "spindle 0.1.0\n");
}

TEST(Cli, RejectsUnknownArguments) {
  const CommandResult result = run_cli("--bogus");
  EXPECT_EQ(result.exit_code, 2);
  EXPECT_EQ(result.output,
            "usage: spindle --version\n"
            "       spindle tokenize --model DIR TEXT\n"
            "       spindle detokenize --model DIR TOKEN_ID...\n"
            "       spindle generate --model DIR --tokens N [--temperature T] [--top-k K] [--top-p "
            "P] [--seed S] [--threads N] PROMPT\n");
}

TEST(Cli, TokenizesText) {
  const CommandResult result =
      run_cli("tokenize --model '" + model_dir() + "' 'Hello, my name is'");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.output, "15496 11 616 1438 318\n");
}

TEST(Cli, DetokenizesIds) {
  const CommandResult result = run_cli("detokenize --model '" + model_dir() + "' 15496 11 616");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.output, "Hello, my\n");
}

TEST(Cli, ReportsBadTokenId) {
  const CommandResult result = run_cli("detokenize --model '" + model_dir() + "' 12x");
  EXPECT_EQ(result.exit_code, 2);
  EXPECT_EQ(result.output, "error: '12x' is not a token id\n");
}

TEST(Cli, ReportsMissingModel) {
  const CommandResult result = run_cli("tokenize --model /nonexistent hi");
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_EQ(result.output, "error: cannot open /nonexistent/vocab.json\n");
}

// Expected text is Hugging Face's greedy continuation (the first tokens of prompt_1's greedy.bin).
TEST(Cli, GeneratesGreedily) {
  const CommandResult result =
      run_cli("generate --model '" + model_dir() + "' --tokens 5 'Hello, my name is'");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.output, "Hello, my name is John. I'm a\n");
}

TEST(Cli, ReportsBadTokenCount) {
  const CommandResult result = run_cli("generate --model '" + model_dir() + "' --tokens -1 hi");
  EXPECT_EQ(result.exit_code, 2);
  EXPECT_EQ(result.output, "error: '-1' is not a token count\n");
}
