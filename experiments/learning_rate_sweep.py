"""Learning Rate Sweep - find optimal LR and edge of stability."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.config import TrainingConfig
from cs336_basics.utils import run_experiment


def grid_sweep(device: str, learning_rates: list[float], dataset: str = "tinystories"):
    """Test multiple learning rates."""
    results = {}
    for lr in learning_rates:
        print(f"\n{'='*80}\nTesting LR: {lr:.0e}\n{'='*80}\n")
        config = TrainingConfig.from_dataset(
            dataset,
            learning_rate=lr,
            device=device,
            checkpoint_dir=f"checkpoints/lr_sweep/lr_{lr:.0e}",
            wandb_project="cs336-lr-sweep",
            wandb_run_name=f"lr_{lr:.0e}",
            use_wandb=True,
        )
        results[lr] = run_experiment(config)

    print(f"\n{'='*80}\nResults:\n{'='*80}")
    for lr, success in results.items():
        print(f"  LR {lr:.0e}: {'✓' if success else '✗'}")
    print("="*80 + "\n")


def stability_sweep(device: str, start_lr: float, max_lr: float, num_steps: int = 10, dataset: str = "tinystories"):
    """Find edge of stability by gradually increasing LR until divergence."""
    import numpy as np
    learning_rates = np.logspace(np.log10(start_lr), np.log10(max_lr), num_steps)

    results = {}
    for lr in learning_rates:
        print(f"\n{'='*80}\nTesting LR: {lr:.0e}\n{'='*80}\n")
        config = TrainingConfig.from_dataset(
            dataset,
            learning_rate=lr,
            device=device,
            checkpoint_dir=f"checkpoints/lr_sweep/stability_{lr:.0e}",
            wandb_project="cs336-lr-sweep",
            wandb_run_name=f"stability_{lr:.0e}",
            use_wandb=True,
        )
        success = run_experiment(config)
        results[lr] = success
        if not success:
            print(f"\n⚠️ Divergence at LR {lr:.0e}")
            break

    print(f"\n{'='*80}\nResults:\n{'='*80}")
    for lr, success in results.items():
        print(f"  LR {lr:.0e}: {'✓ Converged' if success else '✗ Diverged'}")

    converged = [lr for lr, s in results.items() if s]
    if converged:
        print(f"\n🎯 Edge of stability: {max(converged):.0e}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Rate Sweep")
    parser.add_argument("--sweep_type", choices=["grid", "stability", "both"], default="grid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", choices=["tinystories", "owt"], default="tinystories")
    args = parser.parse_args()

    lrs = [1e-5, 1e-4, 1e-3, 5e-3]  # Grid sweep LRs

    if args.sweep_type in ["grid", "both"]:
        print("\n" + "="*80 + "\nGRID SWEEP\n" + "="*80)
        grid_sweep(args.device, lrs, args.dataset)

    if args.sweep_type in ["stability", "both"]:
        print("\n" + "="*80 + "\nSTABILITY SWEEP\n" + "="*80)
        stability_sweep(args.device, 1e-4, 1e-2, 10, args.dataset)
