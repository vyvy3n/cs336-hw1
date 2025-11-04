#!/usr/bin/env python3
"""
Batch Size Sweep Experiment

This script performs a hyperparameter sweep over batch sizes to understand their impact on training.
It trains multiple models with different batch sizes and logs results to W&B.

Usage:
    python experiments/batch_size_sweep.py --device cuda
    python experiments/batch_size_sweep.py --device cuda --optimize_lr
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from cs336_basics.config import TrainingConfig, ModelConfig, DataConfig, OptimizerConfig
from cs336_basics.training import Trainer
from cs336_basics.utils import setup_device, print_experiment_header, handle_oom_error


def run_single_experiment(batch_size: int, learning_rate: float, run_name: str, device: str, use_wandb: bool):
    """
    Run a single training experiment with a specific batch size.

    Args:
        batch_size: Batch size to use
        learning_rate: Learning rate to use
        run_name: Name for this run (for W&B and checkpoints)
        device: Device to use for training
        use_wandb: Whether to use W&B logging
    """
    config = TrainingConfig(
        model=ModelConfig(vocab_size=10000),
        data=DataConfig(
            train_data_path="data/tinystories_train_tokens.npy",
            val_data_path="data/tinystories_valid_tokens.npy",
            batch_size=batch_size,
        ),
        optimizer=OptimizerConfig(learning_rate=learning_rate),
        checkpoint_dir=f"checkpoints/batch_size_sweep/{run_name}",
        wandb_project="cs336-batch-size-sweep",
        wandb_run_name=run_name,
        use_wandb=use_wandb,
        device=device,
    )

    total_tokens = batch_size * config.scheduler.max_iters * config.data.context_length
    print_experiment_header(
        run_name,
        {
            "Batch size": batch_size,
            "Learning rate": learning_rate,
            "Total steps": config.scheduler.max_iters,
            "Total tokens": f"{total_tokens:,}",
        }
    )

    try:
        trainer = Trainer(config)
        trainer.train()
        print(f"\n✓ Completed: {run_name}\n")
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return handle_oom_error(batch_size)
        else:
            print(f"\n✗ Failed: {run_name}")
            print(f"Error: {e}\n")
            return False
    except Exception as e:
        print(f"\n✗ Failed: {run_name}")
        print(f"Error: {e}\n")
        return False


def batch_size_sweep(device: str, use_wandb: bool, optimize_lr: bool = False, base_lr: float = 3e-4,
                     batch_sizes: list[int] = None):
    """
    Perform a sweep over batch sizes.

    Args:
        device: Device to use for training
        use_wandb: Whether to use W&B logging
        optimize_lr: Whether to scale learning rate with batch size
        base_lr: Base learning rate (for batch_size=32)
        batch_sizes: Specific batch sizes to test (if None, use default range)
    """
    # Test batch sizes from 1 to GPU memory limit
    # Include typical sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    results = {}
    max_successful_batch_size = 0
    
    print("\n" + "="*80)
    print("BATCH SIZE SWEEP")
    print("="*80)
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Base learning rate: {base_lr}")
    print(f"LR optimization: {'Enabled (sqrt scaling)' if optimize_lr else 'Disabled (fixed LR)'}")
    print("="*80 + "\n")
    
    for batch_size in batch_sizes:
        # Optionally scale learning rate with batch size
        # Common heuristic: LR ∝ sqrt(batch_size)
        if optimize_lr:
            # Scale LR proportionally to sqrt(batch_size / 32)
            lr_scale = (batch_size / 32.0) ** 0.5
            learning_rate = base_lr * lr_scale
        else:
            learning_rate = base_lr
        
        run_name = f"bs_{batch_size}_lr_{learning_rate:.0e}".replace(".", "_").replace("-", "_")
        
        success = run_single_experiment(
            batch_size=batch_size,
            learning_rate=learning_rate,
            run_name=run_name,
            device=device,
            use_wandb=use_wandb
        )
        
        results[batch_size] = {
            'success': success,
            'learning_rate': learning_rate
        }
        
        if success:
            max_successful_batch_size = batch_size
        else:
            # If we hit OOM, stop trying larger batch sizes
            if "out of memory" in str(success).lower():
                print(f"\n⚠ Reached memory limit at batch size {batch_size}")
                print(f"Skipping larger batch sizes...\n")
                break
    
    # Print summary
    print("\n" + "="*80)
    print("Batch Size Sweep Results:")
    print("="*80)
    print(f"{'Batch Size':<15} {'Learning Rate':<15} {'Status':<20}")
    print("-" * 80)
    for batch_size, result in results.items():
        lr = result['learning_rate']
        status = "✓ Success" if result['success'] else "✗ Failed/OOM"
        print(f"{batch_size:<15} {lr:<15.2e} {status:<20}")
    
    print("-" * 80)
    print(f"\n🎯 Maximum successful batch size: {max_successful_batch_size}")
    print("="*80 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch Size Sweep Experiments")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--optimize_lr",
        action="store_true",
        help="Scale learning rate with batch size (sqrt scaling)"
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=3e-4,
        help="Base learning rate (for batch_size=32)"
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Specific batch sizes to test (e.g., --batch_sizes 32 64 128)"
    )

    args = parser.parse_args()

    # Check device availability
    args.device = setup_device(args.device, verbose=True)

    # Run sweep
    batch_size_sweep(
        device=args.device,
        use_wandb=not args.no_wandb,
        optimize_lr=args.optimize_lr,
        base_lr=args.base_lr,
        batch_sizes=args.batch_sizes
    )


if __name__ == "__main__":
    main()

