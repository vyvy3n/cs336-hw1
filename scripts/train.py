#!/usr/bin/env python3
"""
Main training script for GPT-2 style transformer language models.

This script provides a simplified command-line interface for training transformer models.
Most users should use --dataset presets for optimal defaults.

Usage examples:
    # Use dataset preset (recommended)
    python train.py --dataset tinystories --device cuda
    python train.py --dataset owt --learning_rate 1e-3 --batch_size 64

    # Custom dataset
    python train.py --train_data data/my_train.npy --val_data data/my_val.npy --vocab_size 50000

    # Resume training
    python train.py --dataset tinystories --resume_from checkpoints/checkpoint_5000.pt
"""
import argparse
import sys

from cs336_basics.config import TrainingConfig
from cs336_basics.training import Trainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 style transformer language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Use dataset preset (recommended)
  %(prog)s --dataset tinystories --device cuda
  %(prog)s --dataset owt --learning_rate 1e-3 --batch_size 64

  # Custom dataset
  %(prog)s --train_data data/my_train.npy --val_data data/my_val.npy --vocab_size 50000

  # Resume training
  %(prog)s --dataset tinystories --resume_from checkpoints/checkpoint_5000.pt
        """
    )

    # Dataset preset (most common usage)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["tinystories", "owt"],
        help="Use dataset preset with optimized defaults. Recommended for most users."
    )

    # Most commonly overridden hyperparameters
    parser.add_argument("--learning_rate", type=float, help="Learning rate (default: 6e-4 or from preset)")
    parser.add_argument("--batch_size", type=int, help="Batch size (default: 32 or from preset)")
    parser.add_argument("--max_iters", type=int, help="Total training iterations (default: 5000 or from preset)")

    # Custom data paths (for advanced users not using presets)
    parser.add_argument("--train_data", type=str, help="Training data path (required if not using --dataset)")
    parser.add_argument("--val_data", type=str, help="Validation data path (required if not using --dataset)")
    parser.add_argument("--vocab_size", type=int, help="Vocabulary size (required if not using --dataset)")

    # Environment
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to train on")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, help="Checkpoint directory (default: checkpoints/)")
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint path")

    # Logging
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, help="W&B run name")

    return parser.parse_args()


def create_config_from_args(args) -> TrainingConfig:
    """Create a TrainingConfig from command-line arguments."""

    # Collect overrides from command-line arguments
    overrides = {}
    for key, value in vars(args).items():
        if value is not None and key != 'dataset':
            # Map argument names to config attribute names
            if key == 'train_data':
                overrides['train_data_path'] = value
            elif key == 'val_data':
                overrides['val_data_path'] = value
            elif key == 'no_wandb':
                # If --no_wandb is provided, explicitly disable wandb
                if value:
                    overrides['use_wandb'] = False
            else:
                overrides[key] = value

    # If dataset preset is specified, use it with overrides
    if args.dataset:
        config = TrainingConfig.from_dataset(args.dataset, **overrides)
    else:
        # Validate required arguments for custom dataset
        if not args.train_data or not args.val_data:
            raise ValueError(
                "When not using --dataset preset, you must specify both "
                "--train_data and --val_data paths"
            )
        if not args.vocab_size:
            raise ValueError(
                "When not using --dataset preset, you must specify --vocab_size"
            )

        # Create config with overrides
        config = TrainingConfig(**overrides)

    return config


def print_config(config: TrainingConfig):
    """Print the training configuration (key settings only)."""
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)

    # Calculate total tokens for reference
    total_tokens = config.batch_size * config.max_iters * config.context_length

    print(f"\n📊 Model: {config.num_layers}L-{config.d_model}D-{config.num_heads}H "
          f"(vocab={config.vocab_size:,}, ctx={config.context_length})")

    print(f"\n🎯 Training:")
    print(f"   • Learning rate:  {config.learning_rate:.2e}")
    print(f"   • Batch size:     {config.batch_size}")
    print(f"   • Max iterations: {config.max_iters:,} ({total_tokens:,} tokens)")
    print(f"   • Device:         {config.device}")

    print(f"\n📁 Data:")
    print(f"   • Train: {config.train_data_path}")
    print(f"   • Val:   {config.val_data_path}")

    print(f"\n💾 Checkpoints: {config.checkpoint_dir}")

    if config.resume_from:
        print(f"   • Resuming from: {config.resume_from}")

    if config.use_wandb:
        print(f"\n📈 W&B: {config.wandb_project}", end="")
        if config.wandb_run_name:
            print(f" / {config.wandb_run_name}")
        else:
            print()

    print("=" * 80 + "\n")


def main():
    args = parse_args()
    
    # Create configuration
    try:
        config = create_config_from_args(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print configuration
    print_config(config)
    
    # Start training
    try:
        trainer = Trainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
