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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    for name, (a, b, b_is_transposed) in matmul_cases(rng).items():
        a32 = a.astype(np.float32)
        b32 = b.astype(np.float32)
        b_logical = b32.T if b_is_transposed else b32
        expected = (a32.astype(np.float64) @ b_logical.astype(np.float64)).astype(np.float32)
        case_dir = args.out / "matmul" / name
        write_tensor(case_dir / "a.bin", a32)
        write_tensor(case_dir / "b.bin", b32)
        write_tensor(case_dir / "expected.bin", expected)
        print(f"matmul/{name}: {a32.shape} @ {b_logical.shape} -> {case_dir}")


if __name__ == "__main__":
    main()
