"""Dataset generator utilities compatible with the original paper repo.

Produces channel-first encodings and one-hot action labels (length H*W).
Default labeling strategy is `random`. Output is a compressed `.npz` file
with fields: `inputs` (N,C,H,W), `masks` (N,H*W), `labels` (N,H*W), `values` (N,) .
"""
from __future__ import annotations

import argparse
import copy
import os
from typing import Tuple, List

import numpy as np

from encoding.full_encoding import encode_full, make_action_mask
from game.game import MinesweeperEnv
from constants import BrickType


def _to_one_hot(index: int, size: int) -> np.ndarray:
    v = np.zeros((size,), dtype=np.float32)
    v[index] = 1.0
    return v


def generate_examples(n_examples: int,
                      rows: int = 6,
                      cols: int = 6,
                      mines: int = 4,
                      strategy: str = 'random',
                      rollout_samples: int = 20,
                      max_init_moves: int = 5,
                      seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate dataset examples.

    Returns:
      inputs: (N, C, H, W) float32
      masks:  (N, H*W) uint8
      labels: (N, H*W) float32 (one-hot)
      values: (N,) float32 (zeros if not used)
    """
    rng = np.random.RandomState(seed)

    env = MinesweeperEnv(rows=rows, cols=cols, mines=mines, mode='rgb_array')

    inputs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    values: List[float] = []

    flat_size = rows * cols

    for i in range(n_examples):
        env.reset()

        # optional warm-up moves
        k_init = rng.randint(0, max_init_moves + 1)
        for _ in range(k_init):
            revealed = (env.player_board != BrickType.UNKNOWN)
            mask = make_action_mask(revealed).astype(np.uint8)
            legal = np.where(mask.flatten() == 1)[0]
            if len(legal) == 0:
                break
            a = int(rng.choice(legal))
            env.button = 'left'
            env.step(a)

        counts = env.board.copy()
        revealed = (env.player_board != BrickType.UNKNOWN)

        # encode channel-first to match original repo
        enc = encode_full(counts, revealed, channels_first=True).astype(np.float32)

        mask = make_action_mask(revealed).astype(np.uint8).reshape(-1)
        legal_inds = np.where(mask == 1)[0]

        if len(legal_inds) == 0:
            # no legal moves (shouldn't happen) — pick first
            # produce a safe default label (one-hot first index)
            label_idx = 0
            value = 0.0
            onehot = _to_one_hot(label_idx, flat_size)
        elif strategy == 'random':
            label_idx = int(rng.choice(legal_inds))
            value = 0.0
            onehot = _to_one_hot(label_idx, flat_size)
        elif strategy == 'mine_map':
            # label is the true mine map (multi-hot): 1 if cell contains a mine
            mine_map = (counts == BrickType.MINE).astype(np.float32).reshape(-1)
            onehot = mine_map
            value = 0.0
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        inputs.append(enc)
        masks.append(mask)
        labels.append(onehot)
        values.append(float(value))

    X = np.stack(inputs, axis=0)
    M = np.stack(masks, axis=0)
    Y = np.stack(labels, axis=0)
    V = np.array(values, dtype=np.float32)

    return X, M, Y, V


def save_npz(path: str, X: np.ndarray, M: np.ndarray, Y: np.ndarray, V: np.ndarray):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, inputs=X, masks=M, labels=Y, values=V)


def _cli():
    p = argparse.ArgumentParser(description='Generate Minesweeper dataset (.npz)')
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--rows', type=int, default=6)
    p.add_argument('--cols', type=int, default=6)
    p.add_argument('--mines', type=int, default=4)
    p.add_argument('--strategy', choices=['random', 'mine_map'], default='random',
                   help=("Labeling strategy: 'random' = random legal click (one-hot),"
                         " 'mine_map' = ground-truth mine map (multi-hot)"))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    X, M, Y, V = generate_examples(args.n, args.rows, args.cols, args.mines, args.strategy, seed=args.seed)
    save_npz(args.out, X, M, Y, V)
    print(f"Saved {args.out}: inputs={X.shape}, masks={M.shape}, labels={Y.shape}")


if __name__ == '__main__':
    _cli()
