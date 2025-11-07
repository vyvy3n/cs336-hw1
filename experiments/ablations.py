"""Ablation studies: no_rmsnorm, post_norm, no_rope, silu_only."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.config import TrainingConfig
from cs336_basics.utils import run_experiment


ABLATIONS = {
    "no_rmsnorm": {"ablation_type": "no_rmsnorm"},
    "post_norm": {"ablation_type": "post_norm"},
    "no_rope": {"use_rope": False},
    "silu_only": {"ablation_type": "silu_only"},
}


def run_ablation(ablation: str, lr: float, device: str):
    """Run an ablation experiment."""
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation}")

    print(f"\n{'='*80}\nAblation: {ablation} (LR={lr:.0e})\n{'='*80}\n")

    config = TrainingConfig.from_dataset(
        "tinystories",
        learning_rate=lr,
        device=device,
        checkpoint_dir=f"checkpoints/ablations/{ablation}",
        wandb_project="cs336-ablations",
        wandb_run_name=ablation,
        use_wandb=True,
        **ABLATIONS[ablation]
    )

    return run_experiment(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Experiments")
    parser.add_argument("--ablation", required=True, choices=list(ABLATIONS.keys()) + ["all"])
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.ablation == "all":
        for abl in ABLATIONS:
            run_ablation(abl, args.lr, args.device)
    else:
        run_ablation(args.ablation, args.lr, args.device)
