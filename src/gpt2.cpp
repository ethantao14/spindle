#include "spindle/gpt2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

#include "spindle/ops.h"
#include "spindle/safetensors.h"

namespace spindle {

namespace {

constexpr std::int32_t kEndOfTextToken = 50256;

Gpt2Config read_config(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot open " + path.string());
  }
  const nlohmann::json json = nlohmann::json::parse(file, nullptr, false);
  const auto positive = [&json, &path](const char* key) {
    if (!json.is_object() || !json.contains(key) || !json[key].is_number_unsigned() ||
        json[key].get<std::size_t>() == 0) {
      throw std::runtime_error(path.string() + ": " + key + " must be a positive integer");
    }
    return json[key].get<std::size_t>();
  };
  Gpt2Config config{
      .layers = positive("n_layer"),
      .heads = positive("n_head"),
      .hidden = positive("n_embd"),
      .max_positions = positive("n_positions"),
      .vocab_size = positive("vocab_size"),
      .layer_norm_epsilon = 0.0F,
  };
  if (!json.contains("layer_norm_epsilon") || !json["layer_norm_epsilon"].is_number_float()) {
    throw std::runtime_error(path.string() + ": layer_norm_epsilon must be a number");
  }
  config.layer_norm_epsilon = json["layer_norm_epsilon"].get<float>();
  if (config.hidden % config.heads != 0) {
    throw std::runtime_error(path.string() + ": n_embd must be divisible by n_head");
  }
  return config;
}

}  // namespace

KvCache::KvCache(const Gpt2Config& config, std::size_t capacity) : capacity_(capacity) {
  if (capacity > config.max_positions) {
    throw std::invalid_argument("KV cache capacity " + std::to_string(capacity) +
                                " exceeds the model's " + std::to_string(config.max_positions) +
                                " positions");
  }
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    keys_.emplace_back(Shape{capacity, config.hidden});
    values_.emplace_back(Shape{capacity, config.hidden});
  }
}

Gpt2Model::Gpt2Model(Gpt2Config config, Tensor token_embedding, Tensor position_embedding,
                     std::vector<Layer> layers, Tensor ln_f_weight, Tensor ln_f_bias)
    : config_(config),
      token_embedding_(std::move(token_embedding)),
      position_embedding_(std::move(position_embedding)),
      layers_(std::move(layers)),
      ln_f_weight_(std::move(ln_f_weight)),
      ln_f_bias_(std::move(ln_f_bias)) {}

Gpt2Model Gpt2Model::load(const std::filesystem::path& model_dir) {
  const Gpt2Config config = read_config(model_dir / "config.json");
  const SafetensorsFile weights(model_dir / "model.safetensors");
  const std::size_t hidden = config.hidden;

  std::vector<Layer> layers;
  for (std::size_t index = 0; index < config.layers; ++index) {
    const std::string prefix = "h." + std::to_string(index) + ".";
    const auto load = [&](const std::string& name, const Shape& shape) {
      return weights.load_f32(prefix + name, shape);
    };
    layers.push_back(Layer{
        .ln_1_weight = load("ln_1.weight", {hidden}),
        .ln_1_bias = load("ln_1.bias", {hidden}),
        .attn_weight = load("attn.c_attn.weight", {hidden, 3 * hidden}),
        .attn_bias = load("attn.c_attn.bias", {3 * hidden}),
        .attn_proj_weight = load("attn.c_proj.weight", {hidden, hidden}),
        .attn_proj_bias = load("attn.c_proj.bias", {hidden}),
        .ln_2_weight = load("ln_2.weight", {hidden}),
        .ln_2_bias = load("ln_2.bias", {hidden}),
        .mlp_fc_weight = load("mlp.c_fc.weight", {hidden, 4 * hidden}),
        .mlp_fc_bias = load("mlp.c_fc.bias", {4 * hidden}),
        .mlp_proj_weight = load("mlp.c_proj.weight", {4 * hidden, hidden}),
        .mlp_proj_bias = load("mlp.c_proj.bias", {hidden}),
    });
  }
  return {config,
          weights.load_f32("wte.weight", {config.vocab_size, hidden}),
          weights.load_f32("wpe.weight", {config.max_positions, hidden}),
          std::move(layers),
          weights.load_f32("ln_f.weight", {hidden}),
          weights.load_f32("ln_f.bias", {hidden})};
}

Tensor Gpt2Model::forward(std::span<const std::int32_t> tokens, KvCache& cache,
                          const ActivationObserver& observer) const {
  const std::size_t count = tokens.size();
  const std::size_t start = cache.length_;
  if (count > cache.capacity_ - start) {
    throw std::out_of_range("KV cache is full: " + std::to_string(start) + " + " +
                            std::to_string(count) + " > " + std::to_string(cache.capacity_));
  }
  const std::size_t hidden = config_.hidden;
  const auto notify = [&observer](std::string_view name, const ConstTensorView& value) {
    if (observer) {
      observer(name, value);
    }
  };

  Tensor x({count, hidden});
  embedding(token_embedding_.view(), tokens, x.view());
  add(x.view(), position_embedding_.view().slice(0, start, start + count), x.view());
  notify("embed", x.view());

  Tensor normed({count, hidden});
  Tensor attention_out({count, hidden});
  Tensor mlp_hidden({count, 4 * hidden});
  Tensor mlp_out({count, hidden});
  for (std::size_t index = 0; index < layers_.size(); ++index) {
    const Layer& layer = layers_[index];
    layer_norm(x.view(), layer.ln_1_weight.view(), layer.ln_1_bias.view(),
               config_.layer_norm_epsilon, normed.view());
    attention(layer, index, normed.view(), cache, attention_out.view());
    add(x.view(), attention_out.view(), x.view());

    layer_norm(x.view(), layer.ln_2_weight.view(), layer.ln_2_bias.view(),
               config_.layer_norm_epsilon, normed.view());
    matmul(normed.view(), layer.mlp_fc_weight.view(), mlp_hidden.view());
    add(mlp_hidden.view(), layer.mlp_fc_bias.view(), mlp_hidden.view());
    gelu(mlp_hidden.view(), mlp_hidden.view());
    matmul(mlp_hidden.view(), layer.mlp_proj_weight.view(), mlp_out.view());
    add(mlp_out.view(), layer.mlp_proj_bias.view(), mlp_out.view());
    add(x.view(), mlp_out.view(), x.view());

    const std::string name = std::string(index < 10 ? "block_0" : "block_") + std::to_string(index);
    notify(name, x.view());
  }
  cache.length_ = start + count;

  layer_norm(x.view(), ln_f_weight_.view(), ln_f_bias_.view(), config_.layer_norm_epsilon,
             normed.view());
  notify("ln_f", normed.view());
  return normed;
}

// Causal multi-head self-attention over the cached positions plus the new ones.
void Gpt2Model::attention(const Layer& layer, std::size_t layer_index, const ConstTensorView& in,
                          KvCache& cache, const TensorView& out) const {
  const std::size_t count = in.dim(0);
  const std::size_t hidden = config_.hidden;
  const std::size_t head_size = hidden / config_.heads;
  const std::size_t start = cache.length_;
  const std::size_t total = start + count;

  Tensor qkv({count, 3 * hidden});
  matmul(in, layer.attn_weight.view(), qkv.view());
  add(qkv.view(), layer.attn_bias.view(), qkv.view());

  // Append this step's keys and values after the cached ones.
  const TensorView keys = cache.keys_[layer_index].view();
  const TensorView values = cache.values_[layer_index].view();
  for (std::size_t row = 0; row < count; ++row) {
    const float* source = qkv.data() + (row * 3 * hidden);
    std::copy_n(source + hidden, hidden, keys.data() + ((start + row) * hidden));
    std::copy_n(source + (2 * hidden), hidden, values.data() + ((start + row) * hidden));
  }

  const ConstTensorView queries = qkv.view().slice(1, 0, hidden);
  const ConstTensorView all_keys = keys.slice(0, 0, total);
  const ConstTensorView all_values = values.slice(0, 0, total);
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));
  Tensor scores({count, total});
  Tensor context({count, hidden});
  for (std::size_t head = 0; head < config_.heads; ++head) {
    const std::size_t first = head * head_size;
    const std::size_t last = first + head_size;
    matmul(queries.slice(1, first, last), all_keys.slice(1, first, last).transpose(0, 1),
           scores.view());
    for (std::size_t row = 0; row < count; ++row) {
      float* score_row = scores.data() + (row * total);
      for (std::size_t column = 0; column < total; ++column) {
        // Query at position start + row may only see keys at or before it.
        score_row[column] = column <= start + row ? score_row[column] * scale
                                                  : -std::numeric_limits<float>::infinity();
      }
    }
    softmax(scores.view(), scores.view());
    matmul(scores.view(), all_values.slice(1, first, last), context.view().slice(1, first, last));
  }

  matmul(context.view(), layer.attn_proj_weight.view(), out);
  add(out, layer.attn_proj_bias.view(), out);
}

Tensor Gpt2Model::logits(const ConstTensorView& hidden) const {
  Tensor result({hidden.dim(0), config_.vocab_size});
  matmul(hidden, token_embedding_.view().transpose(0, 1), result.view());
  return result;
}

std::vector<std::int32_t> generate(const Gpt2Model& model, std::span<const std::int32_t> prompt,
                                   std::size_t count, const SamplingConfig& config,
                                   std::uint64_t seed, const TokenCallback& on_token) {
  if (prompt.empty()) {
    throw std::invalid_argument("generation needs at least one prompt token");
  }
  if (!std::isfinite(config.temperature) || config.temperature < 0 ||
      !std::isfinite(config.top_p) || config.top_p <= 0 || config.top_p > 1) {
    throw std::invalid_argument("invalid sampling configuration");
  }
  if (prompt.size() > model.config().max_positions ||
      count > model.config().max_positions - prompt.size()) {
    throw std::out_of_range("generation exceeds context length");
  }
  Pcg32 rng(seed);
  KvCache cache(model.config(), prompt.size() + count);
  std::vector<std::int32_t> generated;
  Tensor hidden = model.forward(prompt, cache);
  while (generated.size() < count) {
    const std::size_t last = hidden.shape()[0] - 1;
    const Tensor logits = model.logits(hidden.view().slice(0, last, last + 1));
    const auto next =
        sample(std::span<const float>(logits.data(), model.config().vocab_size), config, rng);
    generated.push_back(next);
    if (on_token) {
      on_token(next);
    }
    if (next == kEndOfTextToken || generated.size() == count) {
      break;
    }
    hidden = model.forward(std::span(&generated.back(), 1), cache);
  }
  return generated;
}

std::vector<std::int32_t> generate_greedy(const Gpt2Model& model,
                                          std::span<const std::int32_t> prompt, std::size_t count) {
  return generate(model, prompt, count, {.temperature = 0}, 0);
}

}  // namespace spindle
