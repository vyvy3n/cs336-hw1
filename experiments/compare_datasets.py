#!/usr/bin/env python3
"""
Compare TinyStories vs OpenWebText Training

This script trains models on both TinyStories and OpenWebText with identical
configurations, allowing for direct comparison of learning curves and losses.

The assignment asks:
- Train your language model on OpenWebText with the same model architecture and
  total training iterations as TinyStories
- Deliverable: A learning curve of your language model on OpenWebText
- Describe the difference in losses from TinyStories - how should we interpret these losses?

Usage:
    # Quick test (100 iterations each)
    python experiments/compare_datasets.py --device cuda --max_iters 100

    # Full comparison (5000 iterations each)
    python experiments/compare_datasets.py --device cuda --use_wandb

    # Train only one dataset
    python experiments/compare_datasets.py --device cuda --dataset tinystories
    python experiments/compare_datasets.py --device cuda --dataset owt
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.config import TrainingConfig
from cs336_basics.training import Trainer
from cs336_basics.utils import setup_device, print_experiment_header


def train_on_dataset(
    dataset_name: str,
    device: str = "cuda",
    max_iters: int = 40000,
    eval_interval: int = 500,
    use_wandb: bool = False,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
):
    """
    Train on a specific dataset.

    Args:
        dataset_name: 'tinystories' or 'owt'
        device: Device to use ('cuda' or 'cpu')
        max_iters: Total training iterations
        eval_interval: Evaluation frequency
        use_wandb: Whether to use Weights & Biases logging
        learning_rate: Learning rate
        batch_size: Batch size
    """
    # Create configuration using the dataset factory method
    config = TrainingConfig.from_dataset(
        dataset=dataset_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_iters=max_iters,
        eval_interval=eval_interval,
        device=device,
        use_wandb=use_wandb,
        wandb_project="cs336-dataset-comparison",
    )

    print_experiment_header(
        f"Training on {dataset_name.upper()}",
        {
            "Dataset": dataset_name.upper(),
            "Model": "TransformerLM",
            "Vocab Size": config.vocab_size,
            "Context Length": config.context_length,
            "Layers": config.num_layers,
            "d_model": config.d_model,
            "Heads": config.num_heads,
            "d_ff": config.d_ff,
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
    print(f"✓ {dataset_name.upper()} Training Complete!")
    print("="*80)
    print(f"Checkpoints saved to: {config.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare TinyStories vs OpenWebText training"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["tinystories", "owt", "both"],
        default="both",
        help="Which dataset(s) to train on (default: both)"
    )
    
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
        default=40000,
        help="Total training iterations per dataset (default: 40000)"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Evaluation frequency (default: 500)"
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
        help="Enable Weights & Biases logging (recommended for comparison)"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Determine which datasets to train
    datasets = []
    if args.dataset == "both":
        datasets = ["tinystories", "owt"]
    else:
        datasets = [args.dataset]
    
    # Train on each dataset
    for dataset in datasets:
        print("\n" + "="*80)
        print(f"Starting training on {dataset.upper()}")
        print("="*80 + "\n")
        
        train_on_dataset(
            dataset_name=dataset,
            device=device,
            max_iters=args.max_iters,
            eval_interval=args.eval_interval,
            use_wandb=args.use_wandb,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    
    # Print summary
    print("\n" + "="*80)
    print("✓ All Training Complete!")
    print("="*80)
    
    if args.use_wandb:
        print("\n📊 View comparison on Weights & Biases:")
        print("   - Project: cs336-dataset-comparison")
        print("   - Compare the learning curves side-by-side")
        print("\n💡 Expected observations:")
        print("   - OWT should have HIGHER loss than TinyStories")
        print("   - Why? OWT is more diverse, complex, and realistic text")
        print("   - TinyStories is simpler, more repetitive, easier to model")
        print("   - Lower loss ≠ better model (depends on task/domain)")
    
    print("\n📁 Checkpoints saved to:")
    if "tinystories" in datasets:
        print("   - checkpoints/tinystories/")
    if "owt" in datasets:
        print("   - checkpoints/owt/")


if __name__ == "__main__":
    main()

