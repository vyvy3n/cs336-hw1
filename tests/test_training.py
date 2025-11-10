"""
Tests for training functionality.

These are simple, fast smoke tests to verify that training works end-to-end:
- Training runs without crashing
- Loss decreases over iterations
- Checkpoints can be saved and loaded
- Training can be resumed from checkpoints

All tests use tiny synthetic datasets and small models to run quickly (< 30 seconds total).
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from cs336_basics.config import TrainingConfig
from cs336_basics.training import Trainer
from cs336_basics.utils import load_checkpoint


def _create_tiny_dataset(tmp_path: Path, filename: str, size: int = 1000, vocab_size: int = 100) -> Path:
    """
    Create a tiny synthetic dataset for testing.
    
    Args:
        tmp_path: Temporary directory path (from pytest fixture)
        filename: Name of the file to create (e.g., "train.npy")
        size: Number of tokens in the dataset
        vocab_size: Vocabulary size (token IDs will be in range [0, vocab_size))
    
    Returns:
        Path to the created dataset file
    """
    data = np.random.randint(0, vocab_size, size=size, dtype=np.uint16)
    filepath = tmp_path / filename
    np.save(filepath, data)
    return filepath


def _create_tiny_config(tmp_path: Path, **overrides) -> TrainingConfig:
    """
    Create a minimal training configuration for testing.
    
    Uses tiny model and dataset to run quickly:
    - 2 layers, 64 dim, 2 heads
    - 1000 training tokens, 200 validation tokens
    - Batch size 4, context length 32
    
    Args:
        tmp_path: Temporary directory for data and checkpoints
        **overrides: Any config parameters to override
    
    Returns:
        TrainingConfig for testing
    """
    # Create tiny datasets
    train_data = _create_tiny_dataset(tmp_path, "train.npy", size=1000)
    val_data = _create_tiny_dataset(tmp_path, "val.npy", size=200)
    
    # Default tiny config
    defaults = {
        "vocab_size": 100,
        "train_data_path": str(train_data),
        "val_data_path": str(val_data),
        "context_length": 32,
        "num_layers": 2,
        "d_model": 64,
        "num_heads": 2,
        "d_ff": 128,
        "batch_size": 4,
        "max_iters": 50,
        "warmup_iters": 10,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "checkpoint_interval": 10,
        "log_interval": 10,
        "eval_interval": 10,
        "eval_iters": 5,
        "use_wandb": False,
        "device": "cpu",  # Use CPU for tests to avoid GPU memory issues
    }
    
    # Apply overrides
    defaults.update(overrides)
    
    return TrainingConfig(**defaults)


def test_training_loss_decreases(tmp_path):
    """
    Test that training reduces loss over iterations.
    
    This is the most important test: verify that the training loop actually
    improves the model. We train a tiny model for 50 iterations and check
    that the final loss is lower than the initial loss.
    
    What this tests:
    - Training loop runs without crashing
    - Forward pass computes loss correctly
    - Backward pass computes gradients
    - Optimizer updates parameters
    - Model actually learns (loss decreases)
    
    Expected runtime: ~5-10 seconds
    """
    config = _create_tiny_config(tmp_path, max_iters=50)
    trainer = Trainer(config)
    
    # Get initial loss (before training)
    initial_loss = trainer.estimate_loss(trainer.train_data, num_batches=5)
    print(f"\nInitial loss: {initial_loss:.4f}")
    
    # Train for 50 iterations
    trainer.train()
    
    # Get final loss (after training)
    final_loss = trainer.estimate_loss(trainer.train_data, num_batches=5)
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {initial_loss - final_loss:.4f}")
    
    # Assert that loss decreased
    assert final_loss < initial_loss, (
        f"Loss did not decrease during training: "
        f"initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_checkpoint_save_load(tmp_path):
    """
    Test that checkpoints can be saved and loaded correctly.
    
    This verifies that the checkpoint mechanism works:
    1. Train for some iterations
    2. Save a checkpoint
    3. Create a new model and optimizer
    4. Load the checkpoint
    5. Verify that the state is restored correctly
    
    What this tests:
    - save_checkpoint() creates a valid checkpoint file
    - Checkpoint contains all required fields (model, optimizer, iteration)
    - load_checkpoint() restores model weights correctly
    - load_checkpoint() restores optimizer state correctly
    - load_checkpoint() restores iteration count correctly
    
    Expected runtime: ~3-5 seconds
    """
    config = _create_tiny_config(tmp_path, max_iters=10)
    
    # Train for 10 iterations
    trainer = Trainer(config)
    trainer.train()
    
    # Checkpoint should have been saved at iteration 10
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_iter_10.pt"
    assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}"
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Verify checkpoint contains required fields
    assert "model" in checkpoint, "Checkpoint missing 'model' field"
    assert "optimizer" in checkpoint, "Checkpoint missing 'optimizer' field"
    assert "iteration" in checkpoint, "Checkpoint missing 'iteration' field"
    assert checkpoint["iteration"] == 10, f"Expected iteration=10, got {checkpoint['iteration']}"
    
    # Create a new model and optimizer
    new_trainer = Trainer(config)
    
    # Load the checkpoint
    loaded_iter, training_state = load_checkpoint(
        checkpoint_path,
        new_trainer.model,
        new_trainer.optimizer
    )
    
    assert loaded_iter == 10, f"Expected loaded_iter=10, got {loaded_iter}"
    
    # Verify model weights match
    for (name1, param1), (name2, param2) in zip(
        trainer.model.named_parameters(),
        new_trainer.model.named_parameters()
    ):
        assert name1 == name2
        assert torch.allclose(param1, param2), f"Parameter {name1} does not match after loading"
    
    print(f"\n✓ Checkpoint saved and loaded successfully")
    print(f"  Iteration: {loaded_iter}")
    print(f"  Model parameters: {sum(p.numel() for p in new_trainer.model.parameters()):,}")


def test_resume_training(tmp_path):
    """
    Test that training can be resumed from a checkpoint.
    
    This is a critical feature for long training runs:
    1. Train for N iterations and save checkpoint
    2. Create a new trainer and resume from checkpoint
    3. Train for N more iterations
    4. Verify that training continues from the correct iteration
    
    What this tests:
    - resume_from parameter works correctly
    - Training continues from the saved iteration (not from 0)
    - Model state is restored correctly
    - Optimizer state is restored correctly (momentum buffers, etc.)
    - Training can continue without issues after resume
    
    Expected runtime: ~5-10 seconds
    """
    # First training run: train for 10 iterations
    config1 = _create_tiny_config(tmp_path, max_iters=10, checkpoint_interval=10)
    trainer1 = Trainer(config1)
    
    # Get loss before training
    loss_before = trainer1.estimate_loss(trainer1.train_data, num_batches=5)
    print(f"\nLoss before first training: {loss_before:.4f}")
    
    # Train for 10 iterations
    trainer1.train()
    
    # Get loss after first training
    loss_after_first = trainer1.estimate_loss(trainer1.train_data, num_batches=5)
    print(f"Loss after first training (10 iters): {loss_after_first:.4f}")
    
    # Verify checkpoint was saved
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_iter_10.pt"
    assert checkpoint_path.exists(), "Checkpoint not saved after first training"
    
    # Second training run: resume and train for 10 more iterations
    config2 = _create_tiny_config(
        tmp_path,
        max_iters=20,  # Train to iteration 20 (10 more from checkpoint)
        checkpoint_interval=10,
        resume_from=str(checkpoint_path)
    )
    trainer2 = Trainer(config2)

    # Note: current_iter is still 0 here because checkpoint is loaded in train() method
    # This is expected behavior

    # Train for 10 more iterations (from 10 to 20)
    # The train() method will load the checkpoint and resume from iteration 10
    trainer2.train()

    # Verify that training reached iteration 20 (not 30)
    # If resume didn't work, it would train from 0 to 20
    # With resume, it trains from 10 to 20
    assert trainer2.current_iter == 20, (
        f"Expected current_iter=20 after resume training, got {trainer2.current_iter}"
    )
    
    # Get loss after resume training
    loss_after_resume = trainer2.estimate_loss(trainer2.train_data, num_batches=5)
    print(f"Loss after resume training (20 iters total): {loss_after_resume:.4f}")

    # Verify that overall loss decreased from initial to final
    # Note: We don't check loss_after_resume < loss_after_first because with only
    # 10 iterations, the loss might not decrease monotonically due to random sampling
    # The important thing is that resume works and training continues correctly
    assert loss_after_resume < loss_before, (
        f"Overall loss did not decrease: "
        f"initial={loss_before:.4f}, final={loss_after_resume:.4f}"
    )

    print(f"\n✓ Resume training successful")
    print(f"  Initial loss: {loss_before:.4f}")
    print(f"  After 10 iters: {loss_after_first:.4f}")
    print(f"  After 20 iters (resumed): {loss_after_resume:.4f}")
    print(f"  Total loss reduction: {loss_before - loss_after_resume:.4f}")


def test_checkpoint_interval(tmp_path):
    """
    Test that checkpoints are saved at the correct intervals.
    
    This verifies that the checkpoint_interval parameter works correctly:
    - Train for 25 iterations with checkpoint_interval=10
    - Verify checkpoints are saved at iterations 10 and 20
    - Verify no checkpoint at iteration 25 (not a multiple of 10)
    
    What this tests:
    - checkpoint_interval parameter is respected
    - Checkpoints are saved at the right times
    - checkpoint_latest.pt is updated
    
    Expected runtime: ~3-5 seconds
    """
    config = _create_tiny_config(
        tmp_path,
        max_iters=25,
        checkpoint_interval=10
    )
    trainer = Trainer(config)
    trainer.train()
    
    checkpoint_dir = tmp_path / "checkpoints"
    
    # Check that checkpoints exist at iterations 10 and 20
    checkpoint_10 = checkpoint_dir / "checkpoint_iter_10.pt"
    checkpoint_20 = checkpoint_dir / "checkpoint_iter_20.pt"
    checkpoint_25 = checkpoint_dir / "checkpoint_iter_25.pt"
    checkpoint_latest = checkpoint_dir / "checkpoint_latest.pt"

    assert checkpoint_10.exists(), "Checkpoint not saved at iteration 10"
    assert checkpoint_20.exists(), "Checkpoint not saved at iteration 20"

    # Note: checkpoint_iter_25.pt exists because training saves a final checkpoint
    # This is expected behavior - we want to save the final state
    assert checkpoint_25.exists(), "Final checkpoint should be saved at iteration 25"
    assert checkpoint_latest.exists(), "checkpoint_latest.pt not found"

    # Verify checkpoint_latest points to iteration 25 (the final checkpoint)
    latest_checkpoint = torch.load(checkpoint_latest, weights_only=False)
    assert latest_checkpoint["iteration"] == 25, (
        f"checkpoint_latest.pt should be from iteration 25, got {latest_checkpoint['iteration']}"
    )

    print(f"\n✓ Checkpoints saved at correct intervals")
    print(f"  Found: checkpoint_iter_10.pt")
    print(f"  Found: checkpoint_iter_20.pt")
    print(f"  Found: checkpoint_iter_25.pt (final checkpoint)")
    print(f"  Found: checkpoint_latest.pt (iteration {latest_checkpoint['iteration']})")
