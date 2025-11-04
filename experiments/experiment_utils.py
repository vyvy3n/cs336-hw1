"""
Shared utilities for experiment scripts.

This module provides common functions used across different experiment scripts
to reduce code duplication and improve maintainability.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from cs336_basics.config import TrainingConfig, ModelConfig, OptimizerConfig, SchedulerConfig, DataConfig


def create_base_config(
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_iters: int = 40_000,
    vocab_size: int = 10_000,
    context_length: int = 256,
    num_layers: int = 4,
    d_model: int = 512,
    num_heads: int = 16,
    d_ff: int = 1344,
    use_rope: bool = True,
    theta: float = 10000.0,
    warmup_ratio: float = 0.05,
    eval_interval: int = 500,
    eval_iters: int = 100,
    log_interval: int = 100,
    checkpoint_interval: int = 10_000,
    checkpoint_dir: str = "checkpoints",
    wandb_project: str = "cs336-experiments",
    device: str = "cuda",
) -> TrainingConfig:
    """
    Create a base training configuration with sensible defaults.
    
    This function provides a centralized way to create training configurations
    with consistent defaults across all experiments.
    
    Args:
        batch_size: Batch size for training
        learning_rate: Peak learning rate
        max_iters: Total number of training iterations
        vocab_size: Vocabulary size (10000 for TinyStories)
        context_length: Maximum sequence length
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension (None = auto-compute as 8/3 * d_model)
        use_rope: Whether to use Rotary Position Embeddings
        theta: RoPE theta parameter
        warmup_ratio: Fraction of max_iters to use for warmup
        eval_interval: Evaluate every N iterations
        eval_iters: Number of batches to use for evaluation
        log_interval: Log every N iterations
        checkpoint_interval: Save checkpoint every N iterations
        checkpoint_dir: Directory to save checkpoints
        wandb_project: Weights & Biases project name
        device: Device to use for training
    
    Returns:
        TrainingConfig object with specified parameters
    """
    model_config = ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        use_rope=use_rope,
        theta=theta,
    )
    
    optimizer_config = OptimizerConfig(
        learning_rate=learning_rate,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip_norm=1.0,
    )
    
    warmup_iters = int(warmup_ratio * max_iters)
    scheduler_config = SchedulerConfig(
        warmup_iters=warmup_iters,
        max_iters=max_iters,
        min_lr_ratio=0.1,
    )
    
    data_config = DataConfig(
        train_data_path="data/tinystories_train_tokens.npy",
        val_data_path="data/tinystories_valid_tokens.npy",
        batch_size=batch_size,
        context_length=context_length,
    )
    
    config = TrainingConfig(
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        data=data_config,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        use_wandb=True,
        wandb_project=wandb_project,
        device=device,
    )
    
    return config


def print_experiment_header(title: str, params: dict):
    """
    Print a formatted experiment header.
    
    Args:
        title: Experiment title
        params: Dictionary of parameters to display
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    for key, value in params.items():
        print(f"{key}: {value}")
    print(f"{'='*80}\n")


def handle_oom_error(batch_size: int) -> bool:
    """
    Handle out-of-memory errors during training.

    Args:
        batch_size: The batch size that caused OOM

    Returns:
        False (indicating failure)
    """
    print(f"\n✗ Out of Memory")
    print(f"Batch size {batch_size} is too large for available GPU memory\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return False
