#!/usr/bin/env python3
"""
Learning Rate Sweep Experiment

This script performs a hyperparameter sweep over learning rates to find optimal values.
It trains multiple models with different learning rates and logs results to W&B.

Usage:
    python experiments/learning_rate_sweep.py --sweep_type grid --device cuda
    python experiments/learning_rate_sweep.py --sweep_type stability --device cuda
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from cs336_basics.utils import setup_device
from experiment_utils import run_single_experiment


def grid_sweep(device: str, learning_rates: list[float], dataset: str = "tinystories"):
    """
    Perform a grid sweep over learning rates.

    Args:
        device: Device to use for training
        learning_rates: List of learning rates to try
        dataset: Dataset to use ('tinystories' or 'owt')
    """
    results = {}

    for lr in learning_rates:
        run_name = f"lr_{lr:.0e}".replace(".", "_").replace("-", "_")
        success = run_single_experiment(
            dataset=dataset,
            learning_rate=lr,
            run_name=run_name,
            device=device,
            checkpoint_dir="checkpoints/lr_sweep",
            wandb_project="cs336-lr-sweep",
            experiment_params={"Learning Rate": lr}
        )
        results[lr] = success

    print("\n" + "="*80)
    print("Grid Sweep Results:")
    print("="*80)
    for lr, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  LR {lr:.0e}: {status}")
    print("="*80 + "\n")


def stability_sweep(device: str, start_lr: float, max_lr: float, num_steps: int = 10, dataset: str = "tinystories"):
    """
    Perform a stability sweep to find the edge of stability.

    Gradually increase learning rate until divergence is observed.

    Args:
        device: Device to use for training
        start_lr: Starting learning rate
        max_lr: Maximum learning rate to try
        num_steps: Number of learning rates to try
        dataset: Dataset to use ('tinystories' or 'owt')
    """
    # Generate learning rates on log scale
    import numpy as np
    learning_rates = np.logspace(np.log10(start_lr), np.log10(max_lr), num_steps)

    results = {}
    diverged = False

    for lr in learning_rates:
        if diverged:
            print(f"\nSkipping LR {lr:.0e} (already found divergence)\n")
            results[lr] = False
            continue

        run_name = f"stability_lr_{lr:.0e}".replace(".", "_").replace("-", "_")
        success = run_single_experiment(
            dataset=dataset,
            learning_rate=lr,
            run_name=run_name,
            device=device,
            checkpoint_dir="checkpoints/lr_sweep",
            wandb_project="cs336-lr-sweep",
            experiment_params={"Learning Rate": lr}
        )
        results[lr] = success

        if not success:
            diverged = True
            print(f"\n⚠️  Divergence detected at LR {lr:.0e}")

    print("\n" + "="*80)
    print("Stability Sweep Results:")
    print("="*80)
    for lr, success in results.items():
        status = "✓ Converged" if success else "✗ Diverged"
        print(f"  LR {lr:.0e}: {status}")

    # Find the edge of stability
    converged_lrs = [lr for lr, success in results.items() if success]
    if converged_lrs:
        best_lr = max(converged_lrs)
        print(f"\n🎯 Best stable learning rate: {best_lr:.0e}")
    else:
        print(f"\n⚠ All learning rates diverged!")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Learning Rate Sweep Experiments")
    parser.add_argument(
        "--sweep_type",
        type=str,
        choices=["grid", "stability", "both"],
        default="grid",
        help="Type of sweep to perform"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["tinystories", "owt"],
        default="tinystories",
        help="Dataset to use for sweep"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )

    args = parser.parse_args()

    # Check device availability
    args.device = setup_device(args.device, verbose=True)

    # Define learning rates for grid sweep
    grid_learning_rates = [
        1e-5,   # Very small
        # 5e-5,   # Small
        1e-4,   # Small-medium
        # 3e-4,   # Common default
        # 5e-4,   # Medium
        1e-3,   # Large
        # 3e-3,   # Very large
        5e-3,   # Likely too large
    ]

    # Run experiments
    if args.sweep_type in ["grid", "both"]:
        print("\n" + "="*80)
        print("GRID SWEEP: Testing multiple learning rates")
        print("="*80 + "\n")
        grid_sweep(args.device, grid_learning_rates, args.dataset)

    if args.sweep_type in ["stability", "both"]:
        print("\n" + "="*80)
        print("STABILITY SWEEP: Finding edge of stability")
        print("="*80 + "\n")
        stability_sweep(args.device, start_lr=1e-4, max_lr=1e-2, num_steps=10, dataset=args.dataset)


if __name__ == "__main__":
    main()
