# src/encoding/full_encoding.py
import numpy as np
from typing import Tuple, Optional


def encode_full(
    counts: np.ndarray,
    revealed: np.ndarray,
    *,
    channels_first: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    """
    Paper's "full encoding".

    Channels (channels-last):
      0..8: one-hot for numbers 0..8 (1 only on revealed cells that equal that number)
      9:    unknown/unrevealed channel (1 where unrevealed)

    Args:
      counts: int array shape (H, W). For unrevealed cells values may be arbitrary.
      revealed: bool/int array shape (H, W). True where the cell is revealed (number visible).
      channels_first: if True, return shape (C, H, W) instead of (H, W, C).
      dtype: output dtype (default np.float32).

    Returns:
      obs: (H, W, 10) or (10, H, W) array with 0/1 values in `dtype`.
    """
    counts = np.asarray(counts)
    revealed = np.asarray(revealed).astype(bool)

    if counts.shape != revealed.shape:
        raise ValueError("counts and revealed must have the same shape")

    H, W = counts.shape
    out = np.zeros((H, W, 10), dtype=dtype)

    # channels 0..8: number one-hots for revealed cells
    for v in range(9):
        out[..., v] = (revealed & (counts == v)).astype(dtype)

    # channel 9: unknown (unrevealed)
    out[..., 9] = (~revealed).astype(dtype)

    if channels_first:
        out = np.transpose(out, (2, 0, 1))

    return out


def encode_full_batch(
    counts_batch: np.ndarray,
    revealed_batch: np.ndarray,
    *,
    channels_first: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    """
    Batch version. Inputs have shape (N, H, W). Returns (N, H, W, 10) or (N, 10, H, W).
    """
    counts_batch = np.asarray(counts_batch)
    revealed_batch = np.asarray(revealed_batch).astype(bool)

    if counts_batch.shape != revealed_batch.shape:
        raise ValueError("counts_batch and revealed_batch must have same shape (N,H,W)")

    N, H, W = counts_batch.shape
    out = np.zeros((N, H, W, 10), dtype=dtype)

    for v in range(9):
        out[..., v] = (revealed_batch & (counts_batch == v)).astype(dtype)

    out[..., 9] = (~revealed_batch).astype(dtype)

    if channels_first:
        out = np.transpose(out, (0, 3, 1, 2))

    return out


def make_action_mask(revealed: np.ndarray, *, flatten: bool = True) -> np.ndarray:
    """
    Return a boolean mask where True = valid clickable cell (not revealed).
    Accepts revealed of shape (H,W) or (N,H,W). If flatten is True, returns flattened mask(s).
    """
    revealed = np.asarray(revealed).astype(bool)
    if revealed.ndim == 2:
        mask = ~revealed
        return mask.reshape(-1) if flatten else mask
    elif revealed.ndim == 3:
        N, H, W = revealed.shape
        mask = (~revealed).reshape(N, -1)
        return mask if flatten else (~revealed)
    else:
        raise ValueError("revealed must be 2D (H,W) or 3D (N,H,W)")


def flatten_xy(r: int, c: int, H: int, W: int) -> int:
    return r * W + c


def unflatten_index(idx: int, H: int, W: int) -> Tuple[int, int]:
    return divmod(int(idx), W)