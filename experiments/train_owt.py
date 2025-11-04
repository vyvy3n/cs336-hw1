#!/usr/bin/env python3
"""
Train on OpenWebText (OWT) Dataset

This script trains a language model on OpenWebText with the same model architecture
and total training iterations as TinyStories, allowing for direct comparison.

The assignment asks:
- Train your language model on OpenWebText with the same model architecture and 
  total training iterations as TinyStories
- Deliverable: A learning curve of your language model on OpenWebText
- Describe the difference in losses from TinyStories - how should we interpret these losses?

Usage:
    # Quick test (100 iterations)
    python experiments/train_owt.py --device cuda --max_iters 100 --eval_interval 50

    # Full training (same as TinyStories default: 5000 iterations)
    python experiments/train_owt.py --device cuda --use_wandb

    # Custom iterations
    python experiments/train_owt.py --device cuda --max_iters 10000 --use_wandb
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.config import TrainingConfig, ModelConfig, DataConfig, OptimizerConfig, SchedulerConfig
from cs336_basics.training import Trainer
from cs336_basics.utils import setup_device, print_experiment_header


def train_owt(
    device: str = "cuda",
    max_iters: int = 5000,
    eval_interval: int = 100,
    use_wandb: bool = False,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
):
    """
    Train on OpenWebText dataset.

    Args:
        device: Device to use ('cuda' or 'cpu')
        max_iters: Total training iterations (default: 5000, same as TinyStories)
        eval_interval: Evaluation frequency
        use_wandb: Whether to use Weights & Biases logging
        learning_rate: Learning rate (default: 1e-3)
        batch_size: Batch size (default: 32)
    """
    # Create configuration with same architecture as TinyStories
    # NOTE: OWT uses vocab_size=32000, TinyStories uses vocab_size=10000
    config = TrainingConfig(
        model=ModelConfig(
            vocab_size=32000,  # OWT vocab size
            context_length=256,
            num_layers=4,
            d_model=512,
            num_heads=16,
            d_ff=1344,  # 512 * 2.625 for SwiGLU
            use_rope=True,
            theta=10000,
        ),
        data=DataConfig(
            train_data_path="data/owt_train_tokens.npy",
            val_data_path="data/owt_valid_tokens.npy",
            batch_size=batch_size,
            context_length=256,
        ),
        optimizer=OptimizerConfig(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            grad_clip_norm=1.0,
        ),
        scheduler=SchedulerConfig(
            warmup_iters=100,
            max_iters=max_iters,
            min_lr_ratio=0.1,
        ),
        device=device,
        seed=42,
        checkpoint_dir="checkpoints/owt",
        checkpoint_interval=1000,
        log_interval=10,
        eval_interval=eval_interval,
        eval_iters=100,
        use_wandb=use_wandb,
        wandb_project="cs336-owt",
        wandb_run_name=f"owt_lr{learning_rate:.0e}_bs{batch_size}_iters{max_iters}",
    )

    print_experiment_header(
        "OpenWebText Training",
        {
            "Dataset": "OpenWebText",
            "Model": "TransformerLM",
            "Vocab Size": config.model.vocab_size,
            "Context Length": config.model.context_length,
            "Layers": config.model.num_layers,
            "d_model": config.model.d_model,
            "Heads": config.model.num_heads,
            "d_ff": config.model.d_ff,
            "Learning Rate": learning_rate,
            "Batch Size": batch_size,
            "Max Iterations": max_iters,
            "Device": device,
        }
    )

    # Train
    trainer = Trainer(config)
    trainer.train()

    print("\n" + "="*80)
    print("✓ Training Complete!")
    print("="*80)
    print(f"\nCheckpoints saved to: {config.checkpoint_dir}")
    if use_wandb:
        print("View results on Weights & Biases dashboard")


def main():
    parser = argparse.ArgumentParser(description="Train on OpenWebText")
    
    # Training parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=5000,
        help="Total training iterations (default: 5000, same as TinyStories)"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=100,
        help="Evaluation frequency (default: 100)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    
    # Logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Train
    train_owt(
        device=device,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        use_wandb=args.use_wandb,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

