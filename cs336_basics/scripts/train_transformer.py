"""
Training script for Transformer Language Model

This script implements the complete training loop that combines:
- Data loading with memory-efficient np.memmap
- Transformer LM architecture
- AdamW optimizer with cosine learning rate scheduling
- Cross-entropy loss with gradient clipping
- Checkpointing and resuming
- Validation and logging

Usage (from project root):
    uv run python cs336_basics/scripts/train_transformer.py --config path/to/config.json
    uv run python cs336_basics/scripts/train_transformer.py --data path/to/data.npy --vocab_size 10000 --max_steps 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from torch.utils.tensorboard.writer import SummaryWriter

from cs336_basics.data import get_batch
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.models import TransformerLM
from cs336_basics.training.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.lr_schedules import cosine_learning_rate_schedule
from cs336_basics.training.optimizers import AdamW


@dataclass
class TrainingConfig:
    """Configuration for training the Transformer language model."""

    # Data parameters
    train_data_path: str
    val_data_path: str | None = None
    vocab_size: int = 10000
    context_length: int = 256

    # Model parameters
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    eps: float = 1e-5

    # Training parameters
    max_steps: int = 10000
    batch_size: int = 64
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0

    # Logging and checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    resume_from: str | None = None

    # Hardware
    device: str = "cuda"
    compile_model: bool = True

    # Wandb logging
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.context_length > 0, "context_length must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.d_ff > 0, "d_ff must be positive"
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.min_learning_rate >= 0, "min_learning_rate must be non-negative"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert 0 <= self.beta1 < 1, "beta1 must be in [0, 1)"
        assert 0 <= self.beta2 < 1, "beta2 must be in [0, 1)"
        assert self.grad_clip_norm > 0, "grad_clip_norm must be positive"

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


class TrainingLogger:
    """Handles logging to console, tensorboard, and wandb."""

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize training logger with console, file, tensorboard and wandb logging.

        Args:
            config: Training configuration containing logging settings
                    like log directory, wandb project name etc.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(Path(config.log_dir) / "training.log")],
        )

        self.tb_writer = SummaryWriter(log_dir=config.log_dir)

        self.use_wandb = config.wandb_project is not None
        if self.use_wandb:
            wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=asdict(config))

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        """
        Log metrics for a training step.

        Logs training metrics to console, tensorboard, and wandb (if enabled).
        Console logging includes loss and learning rate. All metrics are logged
        to tensorboard under the "train/" prefix. If wandb is enabled, all
        metrics are logged directly to wandb.

        Args:
            step: Current training step number
            metrics: Dictionary of metric names to values to log. Expected to
                     contain at least 'loss' and 'lr' keys.
        """
        loss = metrics.get("loss", 0)
        lr = metrics.get("lr", 0)
        self.logger.info(f"Step: {step}: loss={loss:.4f}, lr={lr:.2e}")

        for key, value in metrics.items():
            self.tb_writer.add_scalar(f"train/{key}", value, step)

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_eval(self, step: int, metrics: dict[str, Any]) -> None:
        """
        Log evaluation metrics.

        Logs validation metrics to console, tensorboard, and wandb (if enabled).
        Console logging includes validation loss and perplexity. All metrics are
        to tensorboard under the "val/" prefix. If wandb is enabled, all
        metrics are logged to wandb with "val_" prefix.

        Args:
            step: Current training step number
            metrics: Dictionary of metric names to values to log. Expected to
                     contain at least 'val_loss' and 'val_perplexity' keys.
        """
        val_loss = metrics.get("val_loss", 0)
        val_ppl = metrics.get("val_perplexity", 0)
        self.logger.info(f"Step: {step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

        for key, value in metrics.items():
            self.tb_writer.add_scalar(f"val/{key}", value, step)

        if self.use_wandb:
            wandb.log({f"val_{k}": v for k, v in metrics.items()}, step=step)

    def close(self) -> None:
        """Close all logging handles."""
        self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()


class DataLoader:
    """Memory-efficient data loader using np.memmap."""

    def __init__(self, data_path: str, batch_size: int, context_length: int, device: str) -> None:
        """
        Initialize memory-mapped data loader.

        Args:
            data_path: Path to numpy array file containing tokenized data
            batch_size: Number of sequences per batch
            context_length: Length of each sequence
            device: Device to load batches to ('cpu', 'cuda', etc.)

        Raises:
            ValueError: If dataset is smaller than context_length + 1
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.data_size = len(self.data)

        if self.data_size < context_length + 1:
            raise ValueError(f"Dataset too small: {self.data_size} < {context_length + 1}")

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of training data.

        Returns:
            The input and target tensors
        """
        return get_batch(self.data.astype(np.int_), self.batch_size, self.context_length, self.device)


class Trainer:
    """Main trainer class that orchestrates the training process."""

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize the Trainer with the given configuration.

        This sets up all components needed for training:
        - Logging and tracking
        - Device (CPU/GPU) configuration
        - Model initialization and compilation
        - Optimizer with configured hyperparameters
        - Data loaders for training and validation
        - Checkpoint loading if resuming training

        Args:
            config: Configuration object containing all training parameters

        Raises:
            RuntimeWarning: If CUDA is specified but not available
        """
        self.config = config
        self.logger = TrainingLogger(config)
        self.step = 0
        self.start_time = time.time()

        if config.device == "cuda" and not torch.cuda.is_available():
            self.logger.logger.warning("CUDA not available, using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.model = TransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            eps=config.eps,
            device=self.device,
        )

        self.original_model = self.model

        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        self.train_loader = DataLoader(
            config.train_data_path, config.batch_size, config.context_length, str(self.device)
        )

        self.val_loader = None
        if config.val_data_path:
            self.val_loader = DataLoader(
                config.val_data_path, config.batch_size, config.context_length, str(self.device)
            )

        if config.resume_from:
            self.load_checkpoint(config.resume_from)

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.logger.info(f"Model initialized with {total_params:,} parameters")
        self.logger.logger.info(f"Training on device: {self.device}")

    def get_lr(self, step: int) -> float:
        """
        Get learning rate for the current step.

        Calculates the learning rate using a cosine schedule with linear warmup.
        The learning rate starts at 0, linearly increases to max_learning_rate during
        warmup_steps, then follows a cosine decay to min_learning_rate over max_steps.

        Args:
            step: Current training step

        Returns:
            Learning rate for the current step
        """
        return cosine_learning_rate_schedule(
            iteration=step,
            max_learning_rate=self.config.learning_rate,
            min_learning_rate=self.config.min_learning_rate,
            warmup_iters=self.config.warmup_steps,
            cosine_cycle_iters=self.config.max_steps,
        )

    def train_step(self) -> dict[str, Any]:
        """
        Perform a single training step.

        This method:
        1. Sets model to training mode
        2. Updates learning rate based on current step
        3. Gets a batch of data from the training loader
        4. Performs forward pass through model
        5. Calculates cross entropy loss
        6. Performs backward pass and gradient updates
        7. Applies gradient clipping
        8. Takes optimizer step

        Returns:
            Dictionary containing:
                - loss: Training loss for this step
                - lr: Current learning rate
                - step: Current training step
        """
        self.model.train()

        lr = self.get_lr(self.step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        inputs, targets = self.train_loader.get_batch()

        logits = self.model(inputs)
        loss = cross_entropy(logits, targets)

        self.optimizer.zero_grad()
        loss.backward()

        gradient_clipping(self.model.parameters(), self.config.grad_clip_norm)

        self.optimizer.step()

        return {"loss": loss.item(), "lr": lr, "step": self.step}

    def evaluate(self) -> dict[str, Any]:
        """
        Evaluate the model on validation set.

        This method:
        1. Checks if validation data is available
        2. Sets model to evaluation mode
        3. Runs forward passes on validation batches without gradients
        4. Calculates average loss and perplexity across batches

        Returns:
            Dictionary containing:
                - loss: Average validation loss
                - perplexity: Perplexity calculated as exp(loss)
                Returns empty dict if no validation data available
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for _ in range(100):
                inputs, targets = self.val_loader.get_batch()
                logits = self.model(inputs)
                loss = cross_entropy(logits, targets)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)

        return {"loss": avg_loss, "perplexity": perplexity}

    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.

        This method saves the current state of training, including:
        1. Model state dict (from original unwrapped model)
        2. Optimizer state dict
        3. Current training step

        Args:
            path: File path where checkpoint should be saved

        Note:
            Uses the original unwrapped model rather than compiled/DDP wrapped version
            to ensure checkpoint compatibility.
        """
        save_checkpoint(self.original_model, self.optimizer, self.step, path)
        self.logger.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        This method loads a previously saved training checkpoint, restoring:
        1. Model state dict (into original unwrapped model)
        2. Optimizer state dict
        3. Current training step

        Args:
            path: File path to checkpoint to load

        Note:
            Loads into the original unwrapped model rather than compiled/DDP wrapped version
            to ensure checkpoint compatibility.
        """
        self.step = load_checkpoint(path, self.original_model, self.optimizer)
        self.logger.logger.info(f"Checkpoint loaded from {path}, resuming from step {self.step}")

    def train(self) -> None:
        """Main training loop."""
        self.logger.logger.info("Starting training...")
        self.logger.logger.info(f"Training for {self.config.max_steps} steps")

        while self.step < self.config.max_steps:
            metrics = self.train_step()

            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - self.start_time
                steps_per_sec = self.step / elapsed if elapsed > 0 else 0
                metrics.update({"steps_per_sec": steps_per_sec, "elapsed_time": elapsed})
                self.logger.log_step(self.step, metrics)

            if self.step % self.config.eval_interval == 0 and self.step > 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    self.logger.log_eval(self.step, eval_metrics)

            if self.step % self.config.save_interval == 0 and self.step > 0:
                checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_step_{self.step}.pt")
                self.save_checkpoint(checkpoint_path)

            self.step += 1

        final_checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_final.pt")
        self.save_checkpoint(final_checkpoint_path)

        final_eval_metrics = self.evaluate()
        if final_eval_metrics:
            self.logger.log_eval(self.step, final_eval_metrics)

        self.logger.logger.info("Training completed!")
        self.logger.close()


def load_config(config_path: str) -> TrainingConfig:
    """
    Load training configuration from JSON file.

    Args:
        Path to JSON configuration file containing model and training parameters.
        See cs336_basics/scripts/example_config.json for expected format.

    Returns:
        Configuration object with all training parameters loaded from the JSON file.
        The JSON fields are mapped to TrainingConfig attributes.
    """
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return TrainingConfig(**config_dict)


def save_config(config: TrainingConfig, path: str) -> None:
    """
    Save training configuration to JSON file.

    Args:
        config: Configuration object containing model and training parameters
                to be saved.
        path: Path where the JSON configuration file should be saved.
    """
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)


def main() -> None:
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--train_data_path", type=str, help="Path to training data (.npy file)")
    parser.add_argument("--val_data_path", type=str, help="Path to validation data (.npy file)")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="Feed-forward dimension")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, help="Wandb run name")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = TrainingConfig(
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            resume_from=args.resume_from,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )

    config_save_path = os.path.join(config.checkpoint_dir, "config.json")
    save_config(config, config_save_path)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
