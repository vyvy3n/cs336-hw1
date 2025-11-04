#!/usr/bin/env python3
"""
Quick Learning Rate Test

A simplified script for quick testing of learning rates with reduced training steps.
Useful for debugging and rapid iteration.

Usage:
    python experiments/quick_lr_test.py --learning_rate 3e-4 --max_iters 1000
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from cs336_basics.config import TrainingConfig, ModelConfig, OptimizerConfig, SchedulerConfig, DataConfig
from cs336_basics.training import train


def main():
    parser = argparse.ArgumentParser(description="Quick Learning Rate Test")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate to test"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=1000,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="Context length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable W&B logging"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="W&B run name"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    model_config = ModelConfig(
        vocab_size=10000,
        context_length=args.context_length,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        use_rope=True,
        theta=10000.0,
    )
    
    optimizer_config = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip_norm=1.0,
    )
    
    scheduler_config = SchedulerConfig(
        warmup_iters=int(args.max_iters * 0.05),  # 5% warmup
        max_iters=args.max_iters,
        min_lr_ratio=0.1,
    )
    
    data_config = DataConfig(
        train_data_path="data/tinystories_train_tokens.npy",
        val_data_path="data/tinystories_valid_tokens.npy",
        batch_size=args.batch_size,
        context_length=args.context_length,
    )
    
    run_name = args.run_name or f"quick_test_lr_{args.learning_rate:.0e}"
    
    config = TrainingConfig(
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        data=data_config,
        eval_interval=200,
        eval_iters=50,
        log_interval=50,
        checkpoint_interval=500,
        checkpoint_dir=f"checkpoints/quick_test/{run_name}",
        use_wandb=args.use_wandb,
        wandb_project="cs336-quick-test",
        wandb_run_name=run_name,
        device=args.device,
    )
    
    print("\n" + "="*80)
    print("Quick Learning Rate Test")
    print("="*80)
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    print(f"Device: {args.device}")
    print(f"W&B logging: {args.use_wandb}")
    print("="*80 + "\n")
    
    # Run training
    train(config)
    
    print("\n" + "="*80)
    print("✓ Quick test complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

