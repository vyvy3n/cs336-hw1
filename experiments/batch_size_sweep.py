"""Batch Size Sweep - test different batch sizes up to memory limit."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.config import TrainingConfig
from cs336_basics.utils import run_experiment


def batch_size_sweep(device: str, batch_sizes: list[int], dataset: str = "tinystories", base_lr: float = 3e-4):
    """Test multiple batch sizes."""
    results = {}
    for bs in batch_sizes:
        print(f"\n{'='*80}\nTesting batch_size: {bs}\n{'='*80}\n")
        config = TrainingConfig.from_dataset(
            dataset,
            batch_size=bs,
            learning_rate=base_lr,
            device=device,
            checkpoint_dir=f"checkpoints/batch_sweep/bs_{bs}",
            wandb_project="cs336-batch-sweep",
            wandb_run_name=f"bs_{bs}",
            use_wandb=True,
        )
        success = run_experiment(config, handle_oom=True)
        results[bs] = success
        if not success:
            print(f"\n⚠️ OOM at batch_size={bs}, stopping\n")
            break

    print(f"\n{'='*80}\nResults:\n{'='*80}")
    for bs, success in results.items():
        print(f"  Batch {bs}: {'✓' if success else '✗ OOM'}")
    max_bs = max([bs for bs, s in results.items() if s], default=0)
    if max_bs:
        print(f"\n🎯 Max batch size: {max_bs}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Size Sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", choices=["tinystories", "owt"], default="tinystories")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--base_lr", type=float, default=3e-4)
    args = parser.parse_args()

    batch_size_sweep(args.device, args.batch_sizes, args.dataset, args.base_lr)
