"""
Configuration dataclasses for training hyperparameters.

This module provides structured configuration for model architecture,
training hyperparameters, and data loading settings.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the Transformer language model architecture."""

    vocab_size: int = 50257  # GPT-2 vocabulary size
    context_length: int = 256  # Maximum sequence length
    num_layers: int = 6  # Number of transformer blocks
    d_model: int = 384  # Model dimension
    num_heads: int = 6  # Number of attention heads
    d_ff: Optional[int] = None  # Feed-forward dimension (None = auto-compute as 8/3 * d_model)
    use_rope: bool = True  # Whether to use Rotary Position Embeddings
    theta: float = 10000.0  # RoPE theta parameter
    ablation_type: str = "none"  # Ablation type: "none", "no_rmsnorm", "post_norm", "silu_only"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        if self.d_ff is None:
            # Auto-compute d_ff as 8/3 * d_model, rounded to multiple of 64
            approx = 8.0 * self.d_model / 3.0
            self.d_ff = max(64, int(round(approx / 64.0) * 64))


@dataclass
class OptimizerConfig:
    """Configuration for the AdamW optimizer."""
    
    learning_rate: float = 6e-4  # Initial/max learning rate
    weight_decay: float = 0.1  # Weight decay coefficient
    beta1: float = 0.9  # Adam beta1
    beta2: float = 0.95  # Adam beta2 (LLMs often use 0.95 instead of 0.999)
    eps: float = 1e-8  # Adam epsilon for numerical stability
    grad_clip_norm: float = 1.0  # Maximum gradient norm for clipping


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduling."""
    
    warmup_iters: int = 100  # Number of warmup iterations
    max_iters: int = 5000  # Total number of training iterations
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of max LR (min_lr = min_lr_ratio * max_lr)
    
    @property
    def cosine_cycle_iters(self) -> int:
        """Total iterations for cosine annealing (same as max_iters)."""
        return self.max_iters
    
    def get_min_lr(self, max_lr: float) -> float:
        """Compute minimum learning rate from max learning rate."""
        return max_lr * self.min_lr_ratio


@dataclass
class DataConfig:
    """Configuration for data loading."""

    train_data_path: str = "data/owt_train_tokens.npy"  # Path to training data
    val_data_path: str = "data/owt_valid_tokens.npy"  # Path to validation data
    batch_size: int = 32  # Batch size for training
    context_length: int = 256  # Context length (should match ModelConfig.context_length)

    def __post_init__(self):
        """Validate data paths exist."""
        import os
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(f"Training data not found: {self.train_data_path}")
        if not os.path.exists(self.val_data_path):
            raise FileNotFoundError(f"Validation data not found: {self.val_data_path}")


@dataclass
class TrainingConfig:
    """Main configuration for the entire training run."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Training settings
    device: str = "cuda"  # Device to train on ('cuda' or 'cpu')
    seed: int = 42  # Random seed for reproducibility
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"  # Directory to save checkpoints
    checkpoint_interval: int = 500  # Save checkpoint every N iterations
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    
    # Logging
    log_interval: int = 10  # Log training metrics every N iterations
    eval_interval: int = 100  # Evaluate on validation set every N iterations
    eval_iters: int = 20  # Number of batches to use for validation
    
    # Weights & Biases (optional)
    use_wandb: bool = False  # Whether to use Weights & Biases for logging
    wandb_project: str = "gpt2-training"  # W&B project name
    wandb_run_name: Optional[str] = None  # W&B run name (None = auto-generate)
    
    def __post_init__(self):
        """Validate and setup configuration."""
        import os
        
        # Ensure context lengths match
        if self.model.context_length != self.data.context_length:
            raise ValueError(
                f"Model context_length ({self.model.context_length}) must match "
                f"data context_length ({self.data.context_length})"
            )
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Validate device
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = "cpu"
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging."""
        return {
            "model": {
                "vocab_size": self.model.vocab_size,
                "context_length": self.model.context_length,
                "num_layers": self.model.num_layers,
                "d_model": self.model.d_model,
                "num_heads": self.model.num_heads,
                "d_ff": self.model.d_ff,
                "use_rope": self.model.use_rope,
                "theta": self.model.theta,
            },
            "optimizer": {
                "learning_rate": self.optimizer.learning_rate,
                "weight_decay": self.optimizer.weight_decay,
                "beta1": self.optimizer.beta1,
                "beta2": self.optimizer.beta2,
                "eps": self.optimizer.eps,
                "grad_clip_norm": self.optimizer.grad_clip_norm,
            },
            "scheduler": {
                "warmup_iters": self.scheduler.warmup_iters,
                "max_iters": self.scheduler.max_iters,
                "min_lr_ratio": self.scheduler.min_lr_ratio,
            },
            "data": {
                "train_data_path": self.data.train_data_path,
                "val_data_path": self.data.val_data_path,
                "batch_size": self.data.batch_size,
                "context_length": self.data.context_length,
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
