#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "spindle/gpt2.h"
#include "spindle/tokenizer.h"

namespace {

double median(std::vector<double> values) {
  std::ranges::sort(values);
  const auto middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2 : values[middle];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string_view model_dir;
    std::string_view output_path;
    std::size_t count = 0;
    std::size_t threads = 1;
    if (argc % 2 != 1) {
      throw std::invalid_argument("flags need values");
    }
    for (int i = 1; i + 1 < argc; i += 2) {
      const std::string_view flag = argv[i];
      const std::string_view text = argv[i + 1];
      if (flag == "--model") {
        model_dir = text;
      } else if (flag == "--out") {
        output_path = text;
      } else if (flag == "--tokens" || flag == "--threads") {
        std::size_t value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc() || end != text.data() + text.size() || value == 0) {
          throw std::invalid_argument("tokens and threads must be positive integers");
        }
        (flag == "--tokens" ? count : threads) = value;
      } else {
        throw std::invalid_argument("unknown flag");
      }
    }
    if (model_dir.empty() || output_path.empty() || count == 0) {
      throw std::invalid_argument(
          "usage: spindle_latency --model DIR --tokens N --out FILE.csv [--threads N]");
    }
    const auto tokenizer = spindle::Tokenizer::load(model_dir);
    const auto model = spindle::Gpt2Model::load(model_dir, threads);
    const auto prompt = tokenizer.encode("The meaning of life is");
    const auto limit = std::min<std::size_t>(1024, model.config().max_positions);
    if (prompt.size() > limit || count > limit - prompt.size()) {
      throw std::invalid_argument("prompt plus tokens exceeds context length");
    }
    std::ofstream output{std::string(output_path)};
    if (!output) {
      throw std::runtime_error("cannot open output CSV");
    }
    output << "position,milliseconds\n";
    spindle::KvCache cache(model.config(), prompt.size() + count);
    // Prefill all but the last prompt token so every timed step forwards one token.
    model.forward(std::span(prompt).first(prompt.size() - 1), cache);
    auto token = prompt.back();
    spindle::Pcg32 rng(0);
    std::vector<double> times;
    times.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const auto start = std::chrono::steady_clock::now();
      const auto hidden = model.forward(std::span(&token, 1), cache);
      const auto logits = model.logits(hidden.view());
      token = spindle::sample(std::span<const float>(logits.data(), model.config().vocab_size),
                              {.temperature = 0}, rng);
      const double elapsed =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count();
      times.push_back(elapsed);
      output << prompt.size() + i << ',' << elapsed << '\n';
    }
    output.close();
    if (!output) {
      throw std::runtime_error("cannot write output CSV");
    }
    const auto window = std::min<std::size_t>(100, times.size());
    const double first =
        median({times.begin(), times.begin() + static_cast<std::ptrdiff_t>(window)});
    const double last = median({times.end() - static_cast<std::ptrdiff_t>(window), times.end()});
    std::printf("First %zu median: %.3f ms; last %zu median: %.3f ms; ratio: %.3f\n", window, first,
                window, last, last / first);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
