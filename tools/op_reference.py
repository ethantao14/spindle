"""Generate NumPy reference cases for the core ops tests.

Each case directory holds the op's inputs and the expected output. Expected
values are computed in float64 and rounded to float32, so they are the exact
answer to float32 precision rather than another float32 implementation.
"""

import argparse
from pathlib import Path

import numpy as np
from tensorfile import write_tensor

SEED = 1234


def matmul_cases(rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray, bool]]:
    """Name -> (a, b, b_is_transposed). A transposed b is stored as [N, K]."""

    def normal(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape)

    return {
        "random_small": (normal(7, 13), normal(13, 5), False),
        "random_gpt2_width": (normal(16, 768), normal(768, 96), False),
        "transposed_b": (normal(7, 13), normal(5, 13), True),
        # Products near 1e32 and 1e-36: far from 1, but inside float32's normal range.
        "large_values": (normal(7, 13) * 1e16, normal(13, 5) * 1e16, False),
        "small_values": (normal(7, 13) * 1e-18, normal(13, 5) * 1e-18, False),
        "all_equal": (np.full((7, 13), 0.5), np.full((13, 5), 0.25), False),
    }


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = np.exp(x - x.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + epsilon) * gamma + beta


LAYER_NORM_EPSILON = 1e-5  # GPT-2's value


def f32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).astype(np.float32)


def write_case(out: Path, op: str, name: str, inputs: dict[str, np.ndarray], expected) -> None:
    """Inputs are rounded to float32 first; expected is computed from those in float64."""
    case_dir = out / op / name
    for input_name, array in inputs.items():
        write_tensor(case_dir / f"{input_name}.bin", array)
    write_tensor(case_dir / "expected.bin", f32(expected))
    print(f"{op}/{name} -> {case_dir}")


def write_matmul_cases(out: Path, rng: np.random.Generator) -> None:
    for name, (a, b, b_is_transposed) in matmul_cases(rng).items():
        a32, b32 = f32(a), f32(b)
        b_logical = b32.T if b_is_transposed else b32
        write_case(out, "matmul", name, {"a": a32, "b": b32},
                   a32.astype(np.float64) @ b_logical.astype(np.float64))


def write_elementwise_cases(out: Path, rng: np.random.Generator) -> None:
    shape = (4, 37)
    unit = lambda: rng.standard_normal(shape)  # noqa: E731
    value_sets = {
        "random": unit() * 3,
        "large_values": unit() * 1e30,
        "small_values": unit() * 1e-30,
        "all_equal": np.full(shape, 0.5),
    }
    for name, values in value_sets.items():
        x = f32(values)
        b = f32(unit() * np.abs(values).max())
        write_case(out, "add", name, {"a": x, "b": b}, x.astype(np.float64) + b)
        write_case(out, "gelu", name, {"in": x}, gelu_tanh(x.astype(np.float64)))
    bias = f32(unit()[0])
    x = f32(unit())
    write_case(out, "add", "bias", {"a": x, "b": bias}, x.astype(np.float64) + bias)


def write_row_cases(out: Path, rng: np.random.Generator) -> None:
    masked = rng.standard_normal((4, 37))
    masked[:, 20:] = -np.inf  # like a causal attention mask
    value_sets = {
        "random": rng.standard_normal((4, 37)),
        "gpt2_width": rng.standard_normal((3, 768)),
        # Softmax overflows exp() here unless it subtracts the row max.
        "large_values": rng.standard_normal((4, 37)) * 1e4,
        "small_values": rng.standard_normal((4, 37)) * 1e-30,
        "all_equal": np.full((4, 37), 0.5),
    }
    for name, values in value_sets.items():
        x = f32(values)
        width = x.shape[-1]
        gamma = f32(1.0 + 0.1 * rng.standard_normal(width))
        beta = f32(0.1 * rng.standard_normal(width))
        write_case(out, "softmax", name, {"in": x}, softmax(x.astype(np.float64)))
        write_case(out, "layer_norm", name, {"in": x, "gamma": gamma, "beta": beta},
                   layer_norm(x.astype(np.float64), gamma, beta, LAYER_NORM_EPSILON))
    x = f32(masked)
    write_case(out, "softmax", "masked", {"in": x}, softmax(x.astype(np.float64)))
    # Many terms too small to change a float32 running sum on their own.
    long_tail = np.full((1, 768), -17.0)
    long_tail[0, 0] = 0.0
    x = f32(long_tail)
    write_case(out, "softmax", "long_tail", {"in": x}, softmax(x.astype(np.float64)))


def write_embedding_cases(out: Path, rng: np.random.Generator) -> None:
    table = f32(rng.standard_normal((50, 8)))
    tokens = np.array([0, 49, 7, 7, 23], dtype=np.int32)
    write_case(out, "embedding", "random", {"table": table, "tokens": tokens}, table[tokens])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    write_matmul_cases(args.out, rng)
    write_elementwise_cases(args.out, rng)
    write_row_cases(args.out, rng)
    write_embedding_cases(args.out, rng)


if __name__ == "__main__":
    main()
