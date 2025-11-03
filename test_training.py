#!/usr/bin/env python3
"""
Quick test script to verify the training pipeline works.

This runs a very short training run with a tiny model to ensure
all components are working correctly.
"""
import sys
import numpy as np
import torch

from cs336_basics.config import TrainingConfig, ModelConfig, OptimizerConfig, SchedulerConfig, DataConfig
from cs336_basics.training import train


def test_training():
    """Run a quick training test."""
    print("=" * 80)
    print("Testing Training Pipeline")
    print("=" * 80)
    
    # Create a minimal configuration for testing
    config = TrainingConfig(
        model=ModelConfig(
            vocab_size=50257,
            context_length=64,  # Very short context
            num_layers=2,  # Very shallow
            d_model=128,  # Very small
            num_heads=2,
            d_ff=None,  # Auto-compute
            use_rope=True,
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-3,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
        ),
        scheduler=SchedulerConfig(
            warmup_iters=5,
            max_iters=20,  # Very short training
        ),
        data=DataConfig(
            train_data_path="data/owt_train_tokens.npy",
            val_data_path="data/owt_valid_tokens.npy",
            batch_size=4,  # Small batch
            context_length=64,
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        checkpoint_dir="test_checkpoints",
        checkpoint_interval=10,
        log_interval=5,
        eval_interval=10,
        eval_iters=2,
        use_wandb=False,
    )
    
    print("\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Model: {config.model.num_layers} layers, {config.model.d_model} dim")
    print(f"  Training: {config.scheduler.max_iters} iterations")
    print(f"  Data: {config.data.batch_size} batch size, {config.data.context_length} context")
    print()
    
    try:
        # Run training
        train(config)
        print("\n" + "=" * 80)
        print("✓ Training test PASSED!")
        print("=" * 80)
        return True
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ Training test FAILED!")
        print(f"Error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1)

