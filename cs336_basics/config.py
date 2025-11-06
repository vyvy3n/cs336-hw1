"""
Configuration dataclass for training hyperparameters.

This module provides a single, flat configuration for all training settings.
"""
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class TrainingConfig:
    """Complete training configuration - all settings in one place."""

    # ========== Model Architecture ==========
    vocab_size: int = 50257  # Vocabulary size
    context_length: int = 256  # Maximum sequence length
    num_layers: int = 6  # Number of transformer blocks
    d_model: int = 384  # Model dimension
    num_heads: int = 6  # Number of attention heads
    d_ff: Optional[int] = None  # Feed-forward dimension (None = auto-compute as 8/3 * d_model)
    use_rope: bool = True  # Whether to use Rotary Position Embeddings
    theta: float = 10000.0  # RoPE theta parameter
    ablation_type: str = "none"  # Ablation type: "none", "no_rmsnorm", "post_norm", "silu_only"

    # ========== Data ==========
    train_data_path: str = "data/owt_train_tokens.npy"  # Path to training data
    val_data_path: str = "data/owt_valid_tokens.npy"  # Path to validation data
    batch_size: int = 32  # Batch size for training

    # ========== Optimizer ==========
    learning_rate: float = 6e-4  # Initial/max learning rate
    weight_decay: float = 0.1  # Weight decay coefficient
    beta1: float = 0.9  # Adam beta1
    beta2: float = 0.95  # Adam beta2
    eps: float = 1e-8  # Adam epsilon for numerical stability
    grad_clip_norm: float = 1.0  # Maximum gradient norm for clipping

    # ========== Learning Rate Schedule ==========
    warmup_iters: int = 2000  # Number of warmup iterations
    max_iters: int = 40000  # Total number of training iterations
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of max LR

    # ========== Training Settings ==========
    device: str = "cuda"  # Device to train on ('cuda' or 'cpu')
    seed: int = 42  # Random seed for reproducibility

    # ========== Checkpointing ==========
    checkpoint_dir: str = "checkpoints"  # Directory to save checkpoints
    checkpoint_interval: int = 10000  # Save checkpoint every N iterations
    resume_from: Optional[str] = None  # Path to checkpoint to resume from

    # ========== Logging ==========
    log_interval: int = 100  # Log training metrics every N iterations
    eval_interval: int = 500  # Evaluate on validation set every N iterations
    eval_iters: int = 100  # Number of batches to use for validation

    # ========== Early Stopping (Optional) ==========
    early_stopping_patience: Optional[int] = None  # Stop if val loss doesn't improve for N evals
    early_stopping_min_delta: float = 0.001  # Minimum change to qualify as improvement

    # ========== Weights & Biases (Optional) ==========
    use_wandb: bool = True  # Whether to use Weights & Biases for logging
    wandb_project: str = "cs336-hw1"  # W&B project name
    wandb_run_name: Optional[str] = None  # W&B run name (None = auto-generate)

    def __post_init__(self):
        """Validate and setup configuration."""
        # Validate model architecture
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")

        # Auto-compute d_ff if not specified
        if self.d_ff is None:
            approx = 8.0 * self.d_model / 3.0
            self.d_ff = max(64, int(round(approx / 64.0) * 64))

        # Validate data paths exist
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(f"Training data not found: {self.train_data_path}")
        if not os.path.exists(self.val_data_path):
            raise FileNotFoundError(f"Validation data not found: {self.val_data_path}")

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Validate device
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = "cpu"

    @property
    def cosine_cycle_iters(self) -> int:
        """Total iterations for cosine annealing (same as max_iters)."""
        return self.max_iters

    def get_min_lr(self) -> float:
        """Compute minimum learning rate from max learning rate."""
        return self.learning_rate * self.min_lr_ratio

    @classmethod
    def from_dataset(cls, dataset: str, **overrides) -> 'TrainingConfig':
        """
        Create a TrainingConfig for a specific dataset with sensible defaults.

        Args:
            dataset: Dataset name ('tinystories' or 'owt')
            **overrides: Any config parameters to override

        Returns:
            TrainingConfig with dataset-specific defaults

        Example:
            >>> config = TrainingConfig.from_dataset('tinystories', learning_rate=1e-3)
            >>> config = TrainingConfig.from_dataset('owt', batch_size=64, max_iters=50000)
        """
        if dataset == "tinystories":
            defaults = {
                "vocab_size": 10000,
                "train_data_path": "data/tinystories_train_tokens.npy",
                "val_data_path": "data/tinystories_valid_tokens.npy",
                "checkpoint_dir": "checkpoints/tinystories",
                "wandb_project": "cs336-tinystories",
                # Standard architecture for experiments
                "context_length": 256,
                "num_layers": 4,
                "d_model": 512,
                "num_heads": 16,
                "d_ff": 1344,
                "use_rope": True,
                "theta": 10000,
                # Standard training settings
                "warmup_iters": 100,
                "max_iters": 40000,
                "log_interval": 50,
                "eval_interval": 500,
                "checkpoint_interval": 1000,
            }
        elif dataset == "owt":
            defaults = {
                "vocab_size": 32000,
                "train_data_path": "data/owt_train_tokens.npy",
                "val_data_path": "data/owt_valid_tokens.npy",
                "checkpoint_dir": "checkpoints/owt",
                "wandb_project": "cs336-owt",
                # Standard architecture for experiments
                "context_length": 256,
                "num_layers": 4,
                "d_model": 512,
                "num_heads": 16,
                "d_ff": 1344,
                "use_rope": True,
                "theta": 10000,
                # Standard training settings
                "warmup_iters": 100,
                "max_iters": 40000,
                "log_interval": 50,
                "eval_interval": 500,
                "checkpoint_interval": 1000,
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Must be 'tinystories' or 'owt'")

        # Apply overrides
        defaults.update(overrides)
        return cls(**defaults)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging."""
        return {
            "model": {
                "vocab_size": self.vocab_size,
                "context_length": self.context_length,
                "num_layers": self.num_layers,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "d_ff": self.d_ff,
                "use_rope": self.use_rope,
                "theta": self.theta,
                "ablation_type": self.ablation_type,
            },
            "optimizer": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "eps": self.eps,
                "grad_clip_norm": self.grad_clip_norm,
            },
            "scheduler": {
                "warmup_iters": self.warmup_iters,
                "max_iters": self.max_iters,
                "min_lr_ratio": self.min_lr_ratio,
            },
            "data": {
                "train_data_path": self.train_data_path,
                "val_data_path": self.val_data_path,
                "batch_size": self.batch_size,
                "context_length": self.context_length,
            },
            "training": {
                "device": self.device,
                "seed": self.seed,
                "checkpoint_interval": self.checkpoint_interval,
                "log_interval": self.log_interval,
                "eval_interval": self.eval_interval,
                "eval_iters": self.eval_iters,
            },
        }
