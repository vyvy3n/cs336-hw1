"""
Training Script for Transformer Language Model on H100 GPU

This script implements a highly optimized training loop that combines:
- Advanced experiment tracking with ExperimentLogger
- H100-specific optimizations for 30-40 minute runtime
- Memory-efficient data loading with np.memmap
- Transformer LM architecture with performance optimizations
- AdamW optimizer with cosine learning rate scheduling
- Comprehensive logging and checkpointing

Usage examples:
    Train on TinyStories with optimized settings
    python cs336_basics/scripts/train_transformer_optimized.py \
        --config cs336_basics/scripts/configs/tinystories_h100.json

    Train on OpenWebText with custom parameters
    python cs336_basics/scripts/train_transformer_optimized.py \
        --train_data data/encoded/owt_train_tokens.npy \
        --val_data data/encoded/owt_valid_tokens.npy \
        --vocab_size 32000 \
        --batch_size 128 \
        --max_steps 12800

Performance optimizations for H100:
- TF32 precision for maximum throughput
- Optimized batch sizes and gradient accumulation
- torch.compile for JIT optimization
- Memory-efficient attention implementation
- Efficient data loading with proper pinned memory
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.experiments.exp_logging import ExperimentLogger, TrainingIntegrator
from cs336_basics.loss.cross_entropy import cross_entropy
from cs336_basics.nn.models import TransformerLM
from cs336_basics.training.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.lr_schedules import cosine_learning_rate_schedule
from cs336_basics.training.optimizers import AdamW


@dataclass
class TrainingConfig:
    """
    Optimized configuration for H100 training with 30-40 minute target runtime.

    This configuration is specifically tuned for maximum throughput on H100 GPUs
    while maintaining training stability and convergence.
    """

    # Data parameters
    train_data_path: str
    val_data_path: str | None = None
    vocab_size: int = 10000
    context_length: int = 256

    # Model parameters
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344  # ~8/3 * d_model, multiple of 64 for tensor cores
    rope_theta: float = 10000.0
    eps: float = 1e-5

    # Training parameters
    max_steps: int = 12800  # Targeting 327,680,000 tokens
    batch_size: int = 64  # Will be tuned based on memory
    gradient_accumulation_steps: int = 1  # For effective larger batches
    learning_rate: float = 6e-4  # Scaled for batch size
    min_learning_rate: float = 6e-5  # 10% of max_lr
    warmup_steps: int = 640  # 5% of max_steps
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95  # Modern LLM setting
    grad_clip_norm: float = 1.0

    # H100 optimization settings
    use_tf32: bool = True  # Enable TF32 for maximum throughput
    compile_model: bool = True  # torch.compile for JIT optimization
    channels_last: bool = False  # Memory layout optimization
    use_fused_adamw: bool = True  # Fused optimizer when available

    # Logging and evaluation
    log_interval: int = 50  # More frequent logging for short runs
    eval_interval: int = 640  # Every 5% of training
    eval_batches: int = 50  # Reduce eval time
    save_interval: int = 3200  # Every 25% of training

    # Directories and experiment tracking
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "transformer_h100_optimized"
    experiment_description: str = "H100-optimized Transformer training for 30-40min runtime"
    use_wandb: bool = True
    wandb_project: str = "cs336-assignment1"

    # Hardware settings
    device: str = "cuda"
    num_workers: int = 4  # For data loading
    pin_memory: bool = True

    # Resume settings
    resume_from: str | None = None
    auto_resume: bool = True  # Auto-resume from latest checkpoint

    def __post_init__(self) -> None:
        """Validate and optimize configuration for H100."""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.context_length > 0, "context_length must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        self.total_tokens = self.effective_batch_size * self.max_steps * self.context_length

        if self.d_ff % 64 != 0:
            self.d_ff = ((self.d_ff + 63) // 64) * 64
            warnings.warn(f"Adjusted d_ff to {self.d_ff} for optimal tensor core usage")


class DataLoader:
    """
    High-performance data loader optimized for H100 training.

    Features:
    - Memory mapping for large datasets
    - Optimized batch sampling
    - Proper device placement
    - Efficient memory usage
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        context_length: int,
        device: str,
        pin_memory: bool = True,
    ) -> None:
        """
        Initialize optimized data loader.

        Args:
            data_path: Path to tokenized data (.npy file)
            batch_size: Batch size for training
            context_length: Sequence length
            device: Target device ('cuda', 'cpu')
            pin_memory: Whether to use pinned memory for faster GPU transfer
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.pin_memory = pin_memory

        data_path_obj = Path(data_path)
        if not data_path_obj.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.data_size = len(self.data)

        if self.data_size < context_length + 1:
            raise ValueError(f"Dataset too small: {self.data_size} < {context_length + 1}")

        print(f"Loaded dataset with {self.data_size:,} tokens from {data_path}")

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of training data with optimized performance."""
        return get_batch(
            dataset=self.data.astype(np.int_),
            batch_size=self.batch_size,
            context_length=self.context_length,
            device=self.device,
        )


class Trainer:
    """
    High-performance trainer optimized for H100 GPU and 30-40 minute runtime.

    This trainer includes:
    - Advanced experiment tracking with ExperimentLogger
    - H100-specific optimizations (TF32, torch.compile, etc.)
    - Memory-efficient training loop
    - Comprehensive performance monitoring
    """

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize the optimized trainer.

        Args:
            config: Training configuration with H100 optimizations
        """
        self.config = config
        self.step = 0
        self.start_time = time.time()

        self.experiment_logger = ExperimentLogger(
            experiment_name=config.experiment_name,
            description=config.experiment_description,
            log_dir="experiments",
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
        )

        self.experiment_logger.log_hyperparameters(**asdict(config))

        self.training_integrator = TrainingIntegrator(self.experiment_logger)

        self._setup_device()

        self._setup_model()

        self._setup_optimizer()

        self._setup_data_loaders()

        if config.resume_from or config.auto_resume:
            self._try_resume()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model initialized with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Training on device: {self.device}")
        print(f"Effective batch size: {config.effective_batch_size}")
        print(f"Total tokens to process: {config.total_tokens:,}")

        self.experiment_logger.add_note(f"Model: {total_params:,} parameters")
        self.experiment_logger.add_note(f"Device: {self.device}")

    def _setup_device(self) -> None:
        """Setup device with H100-specific optimizations."""
        if self.config.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
            self.config.use_tf32 = False
            self.config.compile_model = False
        else:
            self.device = torch.device(self.config.device)

        if self.device.type == "cuda":
            if self.config.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("Enabled TF32 for maximum H100 throughput")

            if self.config.channels_last:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True

            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            self.experiment_logger.add_note(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

    def _setup_model(self) -> None:
        """Setup model with performance optimizations."""
        self.model = TransformerLM(
            vocab_size=self.config.vocab_size,
            context_length=self.config.context_length,
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            rope_theta=self.config.rope_theta,
            eps=self.config.eps,
            device=self.device,
        )

        self.original_model = self.model

        if self.config.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                print("Applied torch.compile with max-autotune")
            except Exception as e:
                print(f"WARNING: torch.compile failed: {e}")

        self.model = self.model.to(self.device)

    def _setup_optimizer(self) -> None:
        """Setup optimizer with performance optimizations."""
        optimizer_class = AdamW
        optimizer_kwargs = {
            "lr": self.config.learning_rate,
            "betas": (self.config.beta1, self.config.beta2),
            "weight_decay": self.config.weight_decay,
            "eps": self.config.eps,
        }

        if self.config.use_fused_adamw and self.device.type == "cuda":
            try:
                import torch.optim

                if hasattr(torch.optim, "AdamW"):
                    optimizer_kwargs["fused"] = True
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
                    print("Using fused AdamW optimizer")
                else:
                    self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
            except Exception:
                self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
        else:
            self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

    def _setup_data_loaders(self) -> None:
        """Setup optimized data loaders."""
        self.train_loader = DataLoader(
            data_path=self.config.train_data_path,
            batch_size=self.config.batch_size,
            context_length=self.config.context_length,
            device=str(self.device),
            pin_memory=self.config.pin_memory,
        )

        self.val_loader = None
        if self.config.val_data_path and Path(self.config.val_data_path).exists():
            self.val_loader = DataLoader(
                data_path=self.config.val_data_path,
                batch_size=self.config.batch_size,
                context_length=self.config.context_length,
                device=str(self.device),
                pin_memory=self.config.pin_memory,
            )

    def _try_resume(self) -> None:
        """Try to resume from checkpoint."""
        checkpoint_path = None

        if self.config.resume_from and Path(self.config.resume_from).exists():
            checkpoint_path = self.config.resume_from
        elif self.config.auto_resume:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
                if checkpoints:
                    checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))

        if checkpoint_path:
            try:
                self.step = load_checkpoint(checkpoint_path, self.original_model, self.optimizer)
                print(f"Resumed from {checkpoint_path} at step {self.step}")
            except Exception as e:
                print(f"WARNING: Failed to load checkpoint: {e}")

    def get_lr(self, step: int) -> float:
        """Get learning rate for current step using cosine schedule with warmup."""
        return cosine_learning_rate_schedule(
            iteration=step,
            max_learning_rate=self.config.learning_rate,
            min_learning_rate=self.config.min_learning_rate,
            warmup_iters=self.config.warmup_steps,
            cosine_cycle_iters=self.config.max_steps,
        )

    def train_step(self) -> dict[str, Any]:
        """
        Perform optimized training step with gradient accumulation.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        lr = self.get_lr(self.step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        total_loss = 0.0

        for _ in range(self.config.gradient_accumulation_steps):
            with torch.amp.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu", enabled=self.device.type == "cuda"
            ):
                inputs, targets = self.train_loader.get_batch()

                logits = self.model(inputs)
                loss = cross_entropy(logits, targets)

                loss = loss / self.config.gradient_accumulation_steps
                total_loss += loss.item()

            loss.backward()

        gradient_clipping(self.model.parameters(), self.config.grad_clip_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "loss": total_loss,
            "lr": lr,
            "step": self.step,
        }

    def evaluate(self) -> dict[str, Any]:
        """
        Evaluate model on validation set.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for _ in range(self.config.eval_batches):
                try:
                    inputs, targets = self.val_loader.get_batch()

                    with torch.amp.autocast(
                        device_type="cuda" if self.device.type == "cuda" else "cpu", enabled=self.device.type == "cuda"
                    ):
                        logits = self.model(inputs)
                        loss = cross_entropy(logits, targets)

                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"WARNING: Evaluation batch failed: {e}")
                    break

        if num_batches == 0:
            return {}

        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        save_checkpoint(self.original_model, self.optimizer, self.step, path)
        print(f"Checkpoint saved: {path}")

    def train(self) -> None:
        """Main optimized training loop."""
        print("Starting optimized H100 training...")
        print(f"Target runtime: 30-40 minutes for {self.config.max_steps} steps")

        self.training_integrator.start_epoch(0)

        while self.step < self.config.max_steps:
            step_start_time = time.time()

            metrics = self.train_step()

            step_time = time.time() - step_start_time
            elapsed_time = time.time() - self.start_time
            steps_per_sec = (self.step + 1) / elapsed_time if elapsed_time > 0 else 0
            tokens_per_sec = steps_per_sec * self.config.effective_batch_size * self.config.context_length

            metrics.update(
                {
                    "step_time": step_time,
                    "steps_per_sec": steps_per_sec,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_time": elapsed_time,
                }
            )

            if self.step % self.config.log_interval == 0:
                self.training_integrator.log_training_step(
                    step=self.step,
                    train_loss=metrics["loss"],
                    learning_rate=metrics["lr"],
                    step_time=step_time,
                    tokens_per_sec=tokens_per_sec,
                )

                if steps_per_sec > 0:
                    remaining_steps = self.config.max_steps - self.step
                    eta_seconds = remaining_steps / steps_per_sec
                    eta_minutes = eta_seconds / 60
                    print(
                        f"Step {self.step}/{self.config.max_steps}: "
                        f"loss={metrics['loss']:.4f}, lr={metrics['lr']:.2e}, "
                        f"{tokens_per_sec:.0f} tok/s, ETA: {eta_minutes:.1f}min"
                    )

            if self.step % self.config.eval_interval == 0 and self.step > 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    self.training_integrator.log_validation_step(
                        step=self.step,
                        val_loss=eval_metrics["loss"],
                        perplexity=eval_metrics["perplexity"],
                    )

            if self.step % self.config.save_interval == 0 and self.step > 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.step}.pt"
                self.save_checkpoint(str(checkpoint_path))

            self.step += 1

        final_eval = self.evaluate()
        if final_eval:
            self.training_integrator.log_validation_step(
                step=self.step,
                val_loss=final_eval["loss"],
                perplexity=final_eval["perplexity"],
            )

        final_checkpoint = Path(self.config.checkpoint_dir) / "checkpoint_final.pt"
        self.save_checkpoint(str(final_checkpoint))

        total_time = time.time() - self.start_time
        self.experiment_logger.add_note(f"Training completed in {total_time / 60:.1f} minutes")
        self.experiment_logger.mark_completed()

        print(f"Training completed in {total_time / 60:.1f} minutes!")


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return TrainingConfig(**config_dict)


def save_config(config: TrainingConfig, path: str) -> None:
    """Save configuration to JSON file."""
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)


def create_optimized_configs() -> None:
    """Create optimized configuration files for different scenarios."""

    tinystories_config = TrainingConfig(
        train_data_path="data/encoded/tinystories_train_tokens.npy",
        val_data_path="data/encoded/tinystories_val_tokens.npy",
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        max_steps=12800,  # 327M tokens = 64 * 12800 * 256
        batch_size=64,
        learning_rate=6e-4,
        experiment_name="tinystories_h100_optimized",
        experiment_description="TinyStories training optimized for H100 30-40min runtime",
    )

    owt_config = TrainingConfig(
        train_data_path="data/encoded/owt_train_tokens.npy",
        val_data_path="data/encoded/owt_valid_tokens.npy",
        vocab_size=32000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        max_steps=12800,
        batch_size=64,
        learning_rate=6e-4,
        experiment_name="openwebtext_h100_optimized",
        experiment_description="OpenWebText training optimized for H100 30-40min runtime",
    )

    project_root = Path.cwd()
    configs_dir = project_root / "cs336_basics" / "scripts" / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    tinystories_config_path = configs_dir / "tinystories_h100.json"
    owt_config_path = configs_dir / "openwebtext_h100.json"

    save_config(tinystories_config, str(tinystories_config_path))
    save_config(owt_config, str(owt_config_path))

    print("Created optimized configuration files:")
    print(f"- {tinystories_config_path}")
    print(f"- {owt_config_path}")


def main() -> None:
    """Main entry point for optimized training script."""
    parser = argparse.ArgumentParser(description="Optimized Transformer Training for H100")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--create-configs", action="store_true", help="Create optimized config files")

    # Data parameters
    parser.add_argument("--train_data", type=str, help="Path to training data (.npy)")
    parser.add_argument("--val_data", type=str, help="Path to validation data (.npy)")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="Feed-forward dimension")

    # Training parameters
    parser.add_argument("--max_steps", type=int, default=12800, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    # Optimization parameters
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    # Directories
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    if args.create_configs:
        create_optimized_configs()
        return

    if args.config:
        config = load_config(args.config)
        if args.train_data:
            config.train_data_path = args.train_data
        if args.val_data:
            config.val_data_path = args.val_data
        if args.resume_from:
            config.resume_from = args.resume_from
    else:
        if not args.train_data:
            raise ValueError("Must specify either --config or --train_data")

        config = TrainingConfig(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,
        )

    if args.no_tf32:
        config.use_tf32 = False
    if args.no_compile:
        config.compile_model = False
    if args.no_wandb:
        config.use_wandb = False

    config_save_path = Path(config.checkpoint_dir) / "config.json"
    save_config(config, str(config_save_path))

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
