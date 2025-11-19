#!/usr/bin/env python3
"""Simple W&B demo with a dummy training loop (no encoding).

This script logs scalar metrics and a generated loss curve to Weights & Biases.
It runs in offline mode if `WANDB_API_KEY` is not set. No TensorFlow or project code
is required — just `wandb` and `matplotlib`/`numpy` for plotting.

Usage:
  pip install wandb matplotlib numpy
  # Optional: login to W&B to view runs online
  wandb login
  python scripts/wandb_demo.py
"""
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
except Exception:
    print("wandb is not installed. Install with: pip install wandb")
    raise


def run_dummy_training(steps=20, seed=0):
    rng = np.random.default_rng(seed)
    losses = []
    accs = []
    for step in range(1, steps + 1):
        # dummy loss decaying and accuracy rising with noise
        loss = np.exp(-step / (steps / 3.0)) + rng.normal(scale=0.02)
        acc = min(1.0, (step / steps) + rng.normal(scale=0.02))
        losses.append(float(loss))
        accs.append(float(max(0.0, min(1.0, acc))))
    return losses, accs


def plot_metrics(losses, accs):
    fig, ax1 = plt.subplots(figsize=(6, 3))
    steps = list(range(1, len(losses) + 1))
    ax1.plot(steps, losses, label='loss', color='C0')
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss', color='C0')
    ax2 = ax1.twinx()
    ax2.plot(steps, accs, label='acc', color='C1')
    ax2.set_ylabel('acc', color='C1')
    ax1.grid(True)
    fig.tight_layout()
    return fig


def main():
    project = os.getenv('WANDB_PROJECT', 'minesweeper-demo')
    api_key = os.getenv('WANDB_API_KEY')
    mode = 'online' if api_key else 'offline'

    print(f'Initializing W&B in {mode} mode (set WANDB_API_KEY to enable online logging)')
    run = wandb.init(project=project, name=f"dummy-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}", mode=mode)

    # Dummy training
    losses, accs = run_dummy_training(steps=20)

    # Log scalars per step
    for step, (l, a) in enumerate(zip(losses, accs), start=1):
        wandb.log({'train/loss': l, 'train/acc': a}, step=step)

    # Log the summary plot
    fig = plot_metrics(losses, accs)
    wandb.log({'metrics/plot': wandb.Image(fig)})

    print('W&B demo finished. If online mode, view the run in your W&B project.')
    run.finish()


if __name__ == '__main__':
    main()
