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

from cs336_basics.config import TrainingConfig, ModelConfig, DataConfig, OptimizerConfig, SchedulerConfig
from cs336_basics.training import Trainer
from cs336_basics.utils import setup_device, print_experiment_header


def train_on_dataset(
    dataset_name: str,
    device: str = "cuda",
    max_iters: int = 5000,
    eval_interval: int = 100,
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
    # Set dataset-specific paths
    if dataset_name == "tinystories":
        train_path = "data/tinystories_train_tokens.npy"
        val_path = "data/tinystories_valid_tokens.npy"
        checkpoint_dir = "checkpoints/tinystories"
        wandb_project = "cs336-dataset-comparison"
        run_name = f"tinystories_lr{learning_rate:.0e}_bs{batch_size}"
    elif dataset_name == "owt":
        train_path = "data/owt_train_tokens.npy"
        val_path = "data/owt_valid_tokens.npy"
        checkpoint_dir = "checkpoints/owt"
        wandb_project = "cs336-dataset-comparison"
        run_name = f"owt_lr{learning_rate:.0e}_bs{batch_size}"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create configuration (identical for both datasets)
    config = TrainingConfig(
        model=ModelConfig(
            vocab_size=10000,
            context_length=256,
            num_layers=4,
            d_model=512,
            num_heads=16,
            d_ff=1344,  # 512 * 2.625 for SwiGLU
            use_rope=True,
            theta=10000,
        ),
        data=DataConfig(
            train_data_path=train_path,
            val_data_path=val_path,
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
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=1000,
        log_interval=10,
        eval_interval=eval_interval,
        eval_iters=100,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=run_name,
    )

    print_experiment_header(
        f"Training on {dataset_name.upper()}",
        {
            "Dataset": dataset_name.upper(),
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
        default=5000,
        help="Total training iterations per dataset (default: 5000)"
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

