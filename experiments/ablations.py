#!/usr/bin/env python3
"""
Ablation Experiments for Transformer Architecture

This script implements four ablation studies:
1. layer_norm: Remove all RMSNorm layers
2. pre_norm: Switch from pre-norm to post-norm architecture
3. no_pos_emb: Remove positional embeddings (RoPE)
4. swiglu: Replace SwiGLU with SiLU-only FFN

Usage:
    python experiments/ablations.py --ablation layer_norm --device cuda
    python experiments/ablations.py --ablation pre_norm --device cuda
    python experiments/ablations.py --ablation no_pos_emb --device cuda
    python experiments/ablations.py --ablation swiglu --device cuda
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.config import TrainingConfig
from cs336_basics.training import Trainer
from cs336_basics.utils import setup_device, print_experiment_header


def run_experiment(ablation: str, learning_rate: float, device: str, use_wandb: bool = True):
    """
    Run a single ablation experiment.

    Args:
        ablation: Type of ablation ('layer_norm', 'pre_norm', 'no_pos_emb', 'swiglu')
        learning_rate: Learning rate to use
        device: Device to train on
        use_wandb: Whether to use W&B logging
    """
    # Set ablation-specific configuration
    ablation_configs = {
        "layer_norm": {
            "run_name": f"no_rmsnorm_lr_{learning_rate:.0e}",
            "checkpoint_dir": "checkpoints/ablations/no_rmsnorm",
            "ablation_type": "no_rmsnorm",
        },
        "pre_norm": {
            "run_name": f"post_norm_lr_{learning_rate:.0e}",
            "checkpoint_dir": "checkpoints/ablations/post_norm",
            "ablation_type": "post_norm",
        },
        "no_pos_emb": {
            "run_name": f"no_rope_lr_{learning_rate:.0e}",
            "checkpoint_dir": "checkpoints/ablations/no_rope",
            "use_rope": False,
        },
        "swiglu": {
            "run_name": f"silu_only_lr_{learning_rate:.0e}",
            "checkpoint_dir": "checkpoints/ablations/silu_only",
            "ablation_type": "silu_only",
        },
    }

    if ablation not in ablation_configs:
        raise ValueError(f"Unknown ablation type: {ablation}")

    ablation_cfg = ablation_configs[ablation]

    # Create config using dataset factory, then override for ablation
    config = TrainingConfig.from_dataset(
        dataset="tinystories",
        learning_rate=learning_rate,
        device=device,
        use_wandb=use_wandb,
        wandb_project="cs336-ablations",
        wandb_run_name=ablation_cfg["run_name"],
        checkpoint_dir=ablation_cfg["checkpoint_dir"],
    )

    # Apply ablation-specific overrides
    if "ablation_type" in ablation_cfg:
        config.ablation_type = ablation_cfg["ablation_type"]
    if "use_rope" in ablation_cfg:
        config.use_rope = ablation_cfg["use_rope"]

    print_experiment_header(
        f"Ablation: {ablation}",
        {"Learning Rate": learning_rate, "Device": device, "W&B": use_wandb}
    )

    # Train the model
    try:
        trainer = Trainer(config)
        trainer.train()
        print(f"\n✅ {ablation} ablation completed successfully!\n")
        return True
    except Exception as e:
        print(f"\n❌ {ablation} ablation failed: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Ablation Experiments")
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=["layer_norm", "pre_norm", "no_pos_emb", "swiglu"],
        help="Type of ablation to run"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )

    args = parser.parse_args()

    # Check device availability
    args.device = setup_device(args.device, verbose=True)

    # Run the experiment
    success = run_experiment(
        ablation=args.ablation,
        learning_rate=args.learning_rate,
        device=args.device,
        use_wandb=not args.no_wandb
    )

    if success:
        print(f"\n🎉 Experiment completed! Check results in checkpoints/ablations/")
    else:
        print(f"\n⚠️  Experiment failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

