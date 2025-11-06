#!/usr/bin/env python3
"""
Shared utilities for experiment scripts.

This module provides common functions used across different experiment scripts
to reduce code duplication and ensure consistency.
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.config import TrainingConfig
from cs336_basics.training import Trainer
from cs336_basics.utils import print_experiment_header, handle_oom_error


def run_single_experiment(
    dataset: str = "tinystories",
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    run_name: str = None,
    device: str = "cuda",
    use_wandb: bool = True,
    checkpoint_dir: str = None,
    wandb_project: str = None,
    max_iters: int = 5000,
    eval_interval: int = 500,
    experiment_params: dict = None,
    handle_oom: bool = False,
) -> bool:
    """
    Run a single training experiment with specified parameters.
    
    This is a generalized experiment runner used by sweep scripts to run
    multiple experiments with different hyperparameters.
    
    Args:
        dataset: Dataset name ('tinystories' or 'owt')
        learning_rate: Learning rate to use
        batch_size: Batch size to use
        run_name: Name for this run (for W&B and checkpoints)
        device: Device to use for training
        use_wandb: Whether to use W&B logging
        checkpoint_dir: Base checkpoint directory (run_name will be appended)
        wandb_project: W&B project name
        max_iters: Total training iterations
        eval_interval: Evaluation frequency
        experiment_params: Dict of parameters to display in experiment header
        handle_oom: Whether to handle OOM errors gracefully (for batch size sweeps)
    
    Returns:
        bool: True if experiment succeeded, False if it failed
    
    Example:
        >>> success = run_single_experiment(
        ...     dataset='tinystories',
        ...     learning_rate=1e-4,
        ...     run_name='lr_1e-4',
        ...     device='cuda'
        ... )
    """
    # Auto-generate run name if not provided
    if run_name is None:
        run_name = f"{dataset}_lr{learning_rate:.0e}_bs{batch_size}"

    # Set checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/{dataset}_experiments/{run_name}"
    else:
        checkpoint_dir = f"{checkpoint_dir}/{run_name}"

    # Create config using the dataset factory method
    config = TrainingConfig.from_dataset(
        dataset=dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_iters=max_iters,
        eval_interval=eval_interval,
        checkpoint_dir=checkpoint_dir,
        wandb_project=wandb_project,
        wandb_run_name=run_name,
        device=device,
        use_wandb=use_wandb,
    )

    # Print experiment header
    if experiment_params is None:
        experiment_params = {
            "Learning Rate": learning_rate,
            "Batch Size": batch_size,
        }

    total_tokens = batch_size * config.max_iters * config.context_length
    experiment_params["Total steps"] = config.max_iters
    experiment_params["Total tokens"] = f"{total_tokens:,}"
    
    print_experiment_header(run_name, experiment_params)
    
    # Run training
    try:
        trainer = Trainer(config)
        trainer.train()
        print(f"\n✓ Completed: {run_name}\n")
        return True
    except RuntimeError as e:
        if handle_oom and "out of memory" in str(e).lower():
            return handle_oom_error(batch_size)
        else:
            print(f"\n✗ Failed: {run_name}")
            print(f"Error: {e}\n")
            return False
    except Exception as e:
        print(f"\n✗ Failed: {run_name}")
        print(f"Error: {e}\n")
        return False
