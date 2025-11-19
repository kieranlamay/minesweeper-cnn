#!/usr/bin/env python3
"""Show a sample Minesweeper board and its 'full encoding' channels.

Run from repo root:
    python scripts/show_encoding.py
"""
import sys
import pathlib

# Ensure repo root is on sys.path so `src` can be imported when running scripts
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
from src.encoding.full_encoding import encode_full, make_action_mask


def pretty_print(mat):
    for row in mat:
        print(' '.join(str(int(x)) for x in row))


def pretty_board_view(counts, revealed):
    H, W = counts.shape
    print("\nPretty board view (numbers where revealed, # for unrevealed):")
    for r in range(H):
        row = []
        for c in range(W):
            if revealed[r, c]:
                row.append(str(int(counts[r, c])))
            else:
                row.append('#')
        print(' '.join(row))


def main():
    # sample 6x6 board (matches paper)
    H, W = 6, 6
    counts = np.zeros((H, W), dtype=int)
    revealed = np.zeros((H, W), dtype=bool)

    # set some revealed cells with numbers
    revealed[0, 0] = True; counts[0, 0] = 1
    revealed[0, 1] = True; counts[0, 1] = 3
    revealed[1, 2] = True; counts[1, 2] = 0
    revealed[4, 4] = True; counts[4, 4] = 8
    revealed[5, 3] = True; counts[5, 3] = 2

    print("Counts (only meaningful on revealed cells):")
    pretty_print(counts)
    print("\nRevealed mask (1=revealed):")
    pretty_print(revealed.astype(int))

    # print combined human-friendly view
    pretty_board_view(counts, revealed)

    enc = encode_full(counts, revealed, channels_first=False)
    print(f"\nEncoded shape: {enc.shape} (H, W, C=10)")

    # Print each channel
    for ch in range(enc.shape[-1]):
        print(f"\nChannel {ch}:")
        pretty_print(enc[..., ch])

    # Print action mask
    mask = make_action_mask(revealed)
    print("\nAction mask (flattened, 1=clickable):")
    # show as HxW
    print('\n'.join(' '.join(str(int(x)) for x in mask.reshape(H, W)[r]) for r in range(H)))


if __name__ == '__main__':
    main()
