#!/usr/bin/env python3
"""CLI wrapper to generate Minesweeper datasets using src.min esweeper.dataset

Run from repo root:
    python scripts/generate_dataset.py --out data/train.npz --n 1000
"""
import sys
import pathlib

# ensure repo root on path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.minesweeper.dataset import _cli


if __name__ == '__main__':
    _cli()
