"""Encode test strings with the Hugging Face GPT-2 tokenizer for the C++ tokenizer tests.

Writes cases.json (a list of {"text", "ids"}) and char_classes.bin: inclusive code point
ranges [first, last, class] showing how the pre-tokenizer classifies every code point.
The strings are written by hand or generated here with a fixed seed.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from reference import MODEL_FILES, MODEL_ID, MODEL_REVISION
from tensorfile import write_tensor
from tokenizers.pre_tokenizers import ByteLevel
from transformers import GPT2TokenizerFast

SEED = 1234
RANDOM_CASES = 1200

# Hand-written cases for each rule of the pre-tokenizer and each kind of text.
FIXED_CASES = [
    "",
    " ",
    "  ",
    "a",
    "Hello world",
    " Hello world ",
    "Hello  world",
    "Hello\tworld",
    "Hello\n\nworld\n",
    "trailing spaces   ",
    "\n\n\n",
    " \t \n ",
    " non-breaking　ideographic space",
    "don't I'll we've they're he'd she's I'm",
    "DON'T I'LL 'S",
    "'s'sup''t'''",
    "rock'n'roll",
    "12345 67890 3.14159 1,000,000 -42",
    "١٢٣ ½ Ⅷ ①",
    "!!! ??? ... --- ***",
    "a.b,c;d:e!f?g",
    "def f(x):\n    return x ** 2  # square\n",
    "for (int i = 0; i < n; ++i) { sum += a[i]; }",
    '{"key": [1, 2, 3], "nested": {"a": null}}',
    "<html><body class=\"x\">Hi</body></html>",
    "SELECT * FROM users WHERE id = 7;",
    "café naïve résumé façade",
    "é combining accent",
    "東京は日本の首都です。",
    "한국어 텍스트",
    "Привет, мир!",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "🚀🔥 emoji 😀😃",
    "👨‍👩‍👧‍👦 family, 🏳️‍🌈 flag, 🇯🇵 flag",
    "mixed: abc123def 456ghi",
    "<|endoftext|>",
    "Hello<|endoftext|>world",
    " <|endoftext|> spaced <|endoftext|><|endoftext|>",
    "<|endoftext",
    "\x00\x01\x1f\x7f control bytes",
    "\U0010fffd private use ",
]

# Pools the random strings draw characters from, weighted toward the tricky cases.
POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    " ",
    "    \t\n\r  　",
    "'s't're've'm'll'd",
    "éèüñçøåßæœ",
    "東京日本語中文字",
    "абвгдежзий",
    "😀🚀🔥👍🏽❤️‍🇺🇸",
    "́̈⃝",
    "١½Ⅷ",
]


def random_code_point(rng: random.Random) -> str:
    """Any code point except surrogates, which a Python str cannot encode to UTF-8."""
    while True:
        code_point = rng.randrange(0x110000)
        if not 0xD800 <= code_point <= 0xDFFF:
            return chr(code_point)


def random_cases(rng: random.Random) -> list[str]:
    cases = []
    for _ in range(RANDOM_CASES):
        length = rng.randint(1, 40)
        chars = []
        for _ in range(length):
            if rng.random() < 0.05:
                chars.append(random_code_point(rng))
            else:
                chars.append(rng.choice(rng.choice(POOLS)))
        cases.append("".join(chars))
    return cases


def char_classes() -> np.ndarray:
    """Classify each code point by how the Hugging Face pre-tokenizer splits around it.

    0 letter, 1 number, 2 whitespace, 3 other, matching spindle::CharClass.
    """
    pre_tokenizer = ByteLevel(add_prefix_space=False)

    def one_piece(text: str) -> bool:
        return len(pre_tokenizer.pre_tokenize_str(text)) == 1

    def classify(char: str) -> int:
        if one_piece("a" + char + "a"):
            return 0
        if one_piece("1" + char + "1"):
            return 1
        if one_piece("!" + char + "!"):
            return 3
        return 2

    ranges: list[list[int]] = []
    for code_point in range(0x110000):
        if 0xD800 <= code_point <= 0xDFFF:
            continue
        char_class = classify(chr(code_point))
        if ranges and ranges[-1][2] == char_class and ranges[-1][1] == code_point - 1:
            ranges[-1][1] = code_point
        else:
            ranges.append([code_point, code_point, char_class])
    return np.array(ranges, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()

    model_dir = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    texts = FIXED_CASES + random_cases(random.Random(SEED))
    cases = [{"text": text, "ids": tokenizer(text).input_ids} for text in texts]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    write_tensor(args.out / "char_classes.bin", char_classes())
    print(f"{len(cases)} tokenizer cases -> {args.out}")


if __name__ == "__main__":
    main()
