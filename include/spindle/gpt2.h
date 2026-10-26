#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "spindle/sampling.h"
#include "spindle/tensor.h"

namespace spindle {

struct Gpt2Config {
  std::size_t layers = 0;
  std::size_t heads = 0;
  std::size_t hidden = 0;
  std::size_t max_positions = 0;
  std::size_t vocab_size = 0;
  float layer_norm_epsilon = 0.0F;
};

// Keys and values of every position run so far, so a new token attends to them
// without recomputing the whole sequence.
class KvCache {
 public:
  // Holds up to capacity positions; throws std::invalid_argument if that exceeds
  // the model's context length.
  KvCache(const Gpt2Config& config, std::size_t capacity);

  [[nodiscard]] std::size_t length() const { return length_; }
  [[nodiscard]] std::size_t capacity() const { return capacity_; }

 private:
  friend class Gpt2Model;
  std::size_t length_ = 0;
  std::size_t capacity_;
  std::vector<Tensor> keys_;    // per layer, [capacity, hidden]
  std::vector<Tensor> values_;  // per layer, [capacity, hidden]
};

// Called with the name and value of each stage: "embed", "block_00".."block_NN", "ln_f".
using ActivationObserver = std::function<void(std::string_view name, const ConstTensorView&)>;

// GPT-2 (Hugging Face layout), float32 weights loaded from model.safetensors and config.json.
class Gpt2Model {
 public:
  // Throws std::runtime_error if a file is missing, malformed, or has the wrong shapes.
  static Gpt2Model load(const std::filesystem::path& model_dir);

  [[nodiscard]] const Gpt2Config& config() const { return config_; }

  // Runs tokens at the positions after those already in the cache, appends their
  // keys and values, and returns the final hidden states [tokens, hidden] (after ln_f).
  // Throws std::out_of_range for an unknown token or a full cache.
  Tensor forward(std::span<const std::int32_t> tokens, KvCache& cache,
                 const ActivationObserver& observer = nullptr) const;

  // Logits [rows, vocab] for hidden states [rows, hidden], using the tied embedding.
  [[nodiscard]] Tensor logits(const ConstTensorView& hidden) const;

 private:
  struct Layer {
    Tensor ln_1_weight, ln_1_bias;
    Tensor attn_weight, attn_bias;  // [hidden, 3 * hidden]: query, key, value
    Tensor attn_proj_weight, attn_proj_bias;
    Tensor ln_2_weight, ln_2_bias;
    Tensor mlp_fc_weight, mlp_fc_bias;  // [hidden, 4 * hidden]
    Tensor mlp_proj_weight, mlp_proj_bias;
  };

  Gpt2Model(Gpt2Config config, Tensor token_embedding, Tensor position_embedding,
            std::vector<Layer> layers, Tensor ln_f_weight, Tensor ln_f_bias);

  void attention(const Layer& layer, std::size_t layer_index, const ConstTensorView& in,
                 KvCache& cache, const TensorView& out) const;

  Gpt2Config config_;
  Tensor token_embedding_;     // [vocab, hidden]
  Tensor position_embedding_;  // [max_positions, hidden]
  std::vector<Layer> layers_;
  Tensor ln_f_weight_, ln_f_bias_;
};

// Called with each new token as soon as it is chosen, for streaming output.
using TokenCallback = std::function<void(std::int32_t token)>;

// Samples until count tokens are generated or the model emits end-of-text (which
// is included). Returns only the new tokens. Throws std::invalid_argument for an
// empty prompt or a bad config, std::out_of_range past the context length.
std::vector<std::int32_t> generate(const Gpt2Model& model, std::span<const std::int32_t> prompt,
                                   std::size_t count, const SamplingConfig& config,
                                   std::uint64_t seed, const TokenCallback& on_token = nullptr);

// generate with temperature 0.
std::vector<std::int32_t> generate_greedy(const Gpt2Model& model,
                                          std::span<const std::int32_t> prompt, std::size_t count);

}  // namespace spindle
