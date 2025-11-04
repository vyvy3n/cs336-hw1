#!/usr/bin/env python3
"""
Main training script for GPT-2 style transformer language models.

This script provides a command-line interface for training transformer models
with configurable hyperparameters, checkpointing, and logging.

Usage examples:
    # Train with default configuration
    python train.py
    
    # Train with custom hyperparameters
    python train.py --batch_size 64 --learning_rate 1e-3 --max_iters 10000
    
    # Resume from checkpoint
    python train.py --resume_from checkpoints/checkpoint_latest.pt
    
    # Train with Weights & Biases logging
    python train.py --use_wandb --wandb_project my-gpt2 --wandb_run_name experiment-1
    
    # Use a preset configuration
    python train.py --config small  # or medium, large
"""
import argparse
import sys
from pathlib import Path

from cs336_basics.config import (
    TrainingConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    DataConfig,
)
from cs336_basics.training import Trainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 style transformer language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Note: Preset configurations removed for simplicity
    # All configuration is done via command-line arguments
    
    # Model architecture
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    model_group.add_argument("--context_length", type=int, default=256, help="Maximum sequence length")
    model_group.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    model_group.add_argument("--d_model", type=int, default=384, help="Model dimension")
    model_group.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    model_group.add_argument("--d_ff", type=int, default=None, help="Feed-forward dimension (None = auto)")
    model_group.add_argument("--use_rope", action="store_true", default=True, help="Use RoPE")
    model_group.add_argument("--no_rope", action="store_false", dest="use_rope", help="Don't use RoPE")
    model_group.add_argument("--theta", type=float, default=10000.0, help="RoPE theta parameter")
    
    # Optimizer
    opt_group = parser.add_argument_group("Optimizer")
    opt_group.add_argument("--learning_rate", type=float, default=6e-4, help="Maximum learning rate")
    opt_group.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay coefficient")
    opt_group.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    opt_group.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    opt_group.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    opt_group.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm")
    
    # Learning rate schedule
    sched_group = parser.add_argument_group("Learning Rate Schedule")
    sched_group.add_argument("--warmup_iters", type=int, default=100, help="Warmup iterations")
    sched_group.add_argument("--max_iters", type=int, default=5000, help="Total training iterations")
    sched_group.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as ratio of max LR")
    
    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train_data", type=str, default="data/owt_train_tokens.npy", help="Training data path")
    data_group.add_argument("--val_data", type=str, default="data/owt_valid_tokens.npy", help="Validation data path")
    data_group.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    train_group.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpointing
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    ckpt_group.add_argument("--checkpoint_interval", type=int, default=500, help="Checkpoint interval")
    ckpt_group.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log_interval", type=int, default=10, help="Log interval")
    log_group.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    log_group.add_argument("--eval_iters", type=int, default=20, help="Number of eval batches")
    
    # Weights & Biases
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    wandb_group.add_argument("--wandb_project", type=str, default="gpt2-training", help="W&B project name")
    wandb_group.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    
    return parser.parse_args()


def create_config_from_args(args) -> TrainingConfig:
    """Create a TrainingConfig from command-line arguments."""

    # Create custom configuration from arguments
    config = TrainingConfig()
    
    # Override with command-line arguments (only if not using preset or explicitly set)
    # Model
    config.model.vocab_size = args.vocab_size
    config.model.context_length = args.context_length
    config.model.num_layers = args.num_layers
    config.model.d_model = args.d_model
    config.model.num_heads = args.num_heads
    config.model.d_ff = args.d_ff
    config.model.use_rope = args.use_rope
    config.model.theta = args.theta
    
    # Optimizer
    config.optimizer.learning_rate = args.learning_rate
    config.optimizer.weight_decay = args.weight_decay
    config.optimizer.beta1 = args.beta1
    config.optimizer.beta2 = args.beta2
    config.optimizer.eps = args.eps
    config.optimizer.grad_clip_norm = args.grad_clip_norm
    
    # Scheduler
    config.scheduler.warmup_iters = args.warmup_iters
    config.scheduler.max_iters = args.max_iters
    config.scheduler.min_lr_ratio = args.min_lr_ratio
    
    # Data
    config.data.train_data_path = args.train_data
    config.data.val_data_path = args.val_data
    config.data.batch_size = args.batch_size
    config.data.context_length = args.context_length
    
    # Training
    config.device = args.device
    config.seed = args.seed
    
    # Checkpointing
    config.checkpoint_dir = args.checkpoint_dir
    config.checkpoint_interval = args.checkpoint_interval
    config.resume_from = args.resume_from
    
    # Logging
    config.log_interval = args.log_interval
    config.eval_interval = args.eval_interval
    config.eval_iters = args.eval_iters
    
    # Weights & Biases
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    config.wandb_run_name = args.wandb_run_name
    
    return config


def print_config(config: TrainingConfig):
    """Print the training configuration."""
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    
    print("\nModel Architecture:")
    print(f"  Vocabulary size:    {config.model.vocab_size:,}")
    print(f"  Context length:     {config.model.context_length}")
    print(f"  Number of layers:   {config.model.num_layers}")
    print(f"  Model dimension:    {config.model.d_model}")
    print(f"  Number of heads:    {config.model.num_heads}")
    print(f"  Feed-forward dim:   {config.model.d_ff}")
    print(f"  Use RoPE:           {config.model.use_rope}")
    if config.model.use_rope:
        print(f"  RoPE theta:         {config.model.theta}")
    
    print("\nOptimizer:")
    print(f"  Learning rate:      {config.optimizer.learning_rate}")
    print(f"  Weight decay:       {config.optimizer.weight_decay}")
    print(f"  Beta1:              {config.optimizer.beta1}")
    print(f"  Beta2:              {config.optimizer.beta2}")
    print(f"  Epsilon:            {config.optimizer.eps}")
    print(f"  Grad clip norm:     {config.optimizer.grad_clip_norm}")
    
    print("\nLearning Rate Schedule:")
    print(f"  Warmup iterations:  {config.scheduler.warmup_iters}")
    print(f"  Max iterations:     {config.scheduler.max_iters}")
    print(f"  Min LR ratio:       {config.scheduler.min_lr_ratio}")
    
    print("\nData:")
    print(f"  Train data:         {config.data.train_data_path}")
    print(f"  Val data:           {config.data.val_data_path}")
    print(f"  Batch size:         {config.data.batch_size}")
    print(f"  Context length:     {config.data.context_length}")
    
    print("\nTraining:")
    print(f"  Device:             {config.device}")
    print(f"  Random seed:        {config.seed}")
    print(f"  Checkpoint dir:     {config.checkpoint_dir}")
    print(f"  Checkpoint interval:{config.checkpoint_interval}")
    print(f"  Log interval:       {config.log_interval}")
    print(f"  Eval interval:      {config.eval_interval}")
    print(f"  Eval iterations:    {config.eval_iters}")
    
    if config.resume_from:
        print(f"  Resume from:        {config.resume_from}")
    
    if config.use_wandb:
        print("\nWeights & Biases:")
        print(f"  Project:            {config.wandb_project}")
        if config.wandb_run_name:
            print(f"  Run name:           {config.wandb_run_name}")
    
    print("=" * 80)
    print()


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
