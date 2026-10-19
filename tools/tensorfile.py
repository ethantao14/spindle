"""Writes tensors in the binary format read by tests/support/reference.h."""

import struct
from pathlib import Path

import numpy as np

MAGIC = b"SPTN"
FORMAT_VERSION = 1
DTYPE_CODES = {np.dtype(np.float32): 0, np.dtype(np.int32): 1}


def write_tensor(path: Path, array: np.ndarray) -> None:
    """Header: magic, version, dtype, ndim (u32 each), dims (u64 each). Then raw little-endian data."""
    array = np.ascontiguousarray(array)
    header = MAGIC + struct.pack("<III", FORMAT_VERSION, DTYPE_CODES[array.dtype], array.ndim)
    header += struct.pack(f"<{array.ndim}Q", *array.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + array.astype(array.dtype.newbyteorder("<")).tobytes())
