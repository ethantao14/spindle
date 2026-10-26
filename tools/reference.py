"""Dump GPT-2 reference outputs from Hugging Face Transformers for the C++ tests.

Links out/model to the downloaded model files. For each prompt, writes token IDs,
the output of every stage of the forward pass, the logits, and a 100-token greedy
continuation, one tensor per file (see tools/tensorfile.py).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from tensorfile import write_tensor
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_ID = "openai-community/gpt2"
# Pinned so the reference cannot change underneath the tests (cache-key check).
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GREEDY_TOKENS = 100
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "vocab.json", "merges.txt"]


def capture_stages(model: GPT2LMHeadModel) -> tuple[dict[str, torch.Tensor], list]:
    """Register hooks recording the embedding output, each block's output and ln_f's output."""
    captured: dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output[0] if isinstance(output, tuple) else output

        return hook

    transformer = model.transformer
    # In eval mode dropout is the identity, so its output is token plus position embeddings.
    handles = [transformer.drop.register_forward_hook(save("embed"))]
    for index, block in enumerate(transformer.h):
        handles.append(block.register_forward_hook(save(f"block_{index:02d}")))
    handles.append(transformer.ln_f.register_forward_hook(save("ln_f")))
    return captured, handles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="JSON list of prompt strings")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    model_dir = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir, dtype=torch.float32).eval()
    captured, _handles = capture_stages(model)

    # The C++ tests and CLI read the model files through this link.
    args.out.mkdir(parents=True, exist_ok=True)
    model_link = args.out / "model"
    model_link.unlink(missing_ok=True)
    model_link.symlink_to(model_dir, target_is_directory=True)

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    for index, prompt in enumerate(prompts):
        prompt_dir = args.out / f"prompt_{index}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        token_ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(token_ids).logits

        write_tensor(prompt_dir / "tokens.bin", token_ids[0].numpy().astype(np.int32))
        for name, activation in captured.items():
            write_tensor(prompt_dir / f"{name}.bin", activation[0].numpy())
        write_tensor(prompt_dir / "logits.bin", logits[0].numpy())

        # Greedy decoding stops early if the model emits end-of-text.
        with torch.no_grad():
            output = model.generate(
                token_ids,
                attention_mask=torch.ones_like(token_ids),
                do_sample=False,
                max_new_tokens=GREEDY_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
            )
        write_tensor(prompt_dir / "greedy.bin", output[0, token_ids.shape[1] :].numpy().astype(np.int32))
        print(f"prompt_{index}: {token_ids.shape[1]} tokens -> {prompt_dir}")


if __name__ == "__main__":
    main()
