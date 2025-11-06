"""
Training utilities for transformer language models.

This module provides a Trainer class that encapsulates the training loop,
evaluation, checkpointing, and logging functionality.
"""
import os
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .models import TransformerLM
from .optimizers import AdamW, CrossEntropyLoss, get_lr_cosine_schedule, gradient_clipping
from .utils import (
    get_batch,
    setup_device,
    get_checkpoint_paths,
    safe_wandb_call,
    count_parameters,
    set_seed,
    save_checkpoint as save_checkpoint_impl,
    load_checkpoint as load_checkpoint_impl,
)
from .config import TrainingConfig


class Trainer:
    """
    Trainer class that encapsulates training loop, evaluation, and checkpointing.

    This class manages the entire training process including:
    - Model and optimizer initialization
    - Training loop with learning rate scheduling
    - Evaluation on validation set
    - Checkpointing and resuming
    - Logging to console and Weights & Biases

    Example:
        >>> config = TrainingConfig(...)
        >>> trainer = Trainer(config)
        >>> trainer.train()
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Set random seed
        set_seed(config.seed)

        # Setup device
        self.device = setup_device(config.device, verbose=True)
        self.config.device = self.device

        # Load datasets
        self._load_datasets()

        # Initialize model
        self.model = self._create_model()

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        # Initialize loss function
        self.loss_fn = CrossEntropyLoss(device=self.device)

        # Initialize W&B if requested
        self.use_wandb = config.use_wandb and self._initialize_wandb()

        # Training state
        self.current_iter = 0
        self.train_metrics = {}

        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False

    def _load_datasets(self):
        """Load training and validation datasets with memory mapping."""
        print(f"Loading training data from {self.config.train_data_path}")
        self.train_data = np.load(self.config.train_data_path, mmap_mode='r')
        print(f"Training data shape: {self.train_data.shape}")

        print(f"Loading validation data from {self.config.val_data_path}")
        self.val_data = np.load(self.config.val_data_path, mmap_mode='r')
        print(f"Validation data shape: {self.val_data.shape}")

    def _create_model(self) -> nn.Module:
        """Create and initialize the model."""
        print("Initializing model...")

        # Use ablation model if specified
        if self.config.ablation_type != "none":
            from .ablation_models import TransformerLMAblation
            print(f"Using ablation model: {self.config.ablation_type}")
            model = TransformerLMAblation(
                vocab_size=self.config.vocab_size,
                context_length=self.config.context_length,
                num_layers=self.config.num_layers,
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                use_rope=self.config.use_rope,
                ablation_type=self.config.ablation_type,
                theta=self.config.theta,
                device=self.device,
            ).to(self.device)
        else:
            model = TransformerLM(
                vocab_size=self.config.vocab_size,
                context_length=self.config.context_length,
                num_layers=self.config.num_layers,
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                use_rope=self.config.use_rope,
                theta=self.config.theta,
                device=self.device,
            ).to(self.device)

        num_params = count_parameters(model)
        print(f"Model has {num_params:,} trainable parameters")

        return model

    def _initialize_wandb(self) -> bool:
        """Initialize Weights & Biases logging."""
        result = safe_wandb_call(
            'init',
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config.to_dict(),
        )

        if result is not None:
            import wandb
            print(f"Initialized Weights & Biases: {wandb.run.url}")
            return True
        else:
            print("Warning: wandb not installed or failed to initialize")
            return False

    def estimate_loss(self, dataset: np.ndarray, num_batches: int = 20) -> float:
        """
        Estimate loss on a dataset by averaging over multiple batches.

        Args:
            dataset: Dataset to evaluate on
            num_batches: Number of batches to average over

        Returns:
            Average loss over the batches
        """
        self.model.eval()
        losses = []

        with torch.no_grad():
            for _ in range(num_batches):
                x, y = get_batch(
                    dataset,
                    batch_size=self.config.data.batch_size,
                    context_length=self.config.data.context_length,
                    device=self.device,
                )

                logits = self.model(x)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                losses.append(loss.item())

        self.model.train()
        return np.mean(losses)


    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step.

        Returns:
            Dictionary with training metrics (loss, lr, etc.)
        """
        self.model.train()

        # Get learning rate for this iteration
        lr = get_lr_cosine_schedule(
            it=self.current_iter,
            max_learning_rate=self.config.learning_rate,
            min_learning_rate=self.config.get_min_lr(),
            warmup_iters=self.config.warmup_iters,
            cosine_cycle_iters=self.config.cosine_cycle_iters,
        )

        # Update learning rate in optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Sample a batch
        x, y = get_batch(
            self.train_data,
            batch_size=self.config.batch_size,
            context_length=self.config.context_length,
            device=self.device,
        )

        # Forward pass
        logits = self.model(x)

        # Compute loss
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(self.model.parameters(), max_l2_norm=self.config.grad_clip_norm)

        # Optimizer step
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'lr': lr,
        }


    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Log training metrics to console and optionally to Weights & Biases.

        Args:
            metrics: Dictionary of metrics to log
        """
        # Console logging
        log_str = f"Iter {self.current_iter:5d}"
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f" | {key}: {value:.4f}"
            else:
                log_str += f" | {key}: {value}"
        print(log_str)

        # Weights & Biases logging
        if self.use_wandb:
            safe_wandb_call('log', metrics, step=self.current_iter)


    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None):
        """
        Save a training checkpoint.

        Args:
            metrics: Optional metrics to save with checkpoint
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Get checkpoint paths
        numbered_path, latest_path = get_checkpoint_paths(
            self.config.checkpoint_dir,
            self.current_iter
        )

        # Prepare training state (for early stopping, etc.)
        training_state = {
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
        }

        # Save numbered checkpoint
        print(f"Saving checkpoint to {numbered_path}")
        save_checkpoint_impl(self.model, self.optimizer, self.current_iter, numbered_path, training_state)

        # Save latest checkpoint
        save_checkpoint_impl(self.model, self.optimizer, self.current_iter, latest_path, training_state)

        # Save metrics if provided
        if metrics is not None:
            metrics_path = os.path.join(
                self.config.checkpoint_dir,
                f"metrics_iter_{self.current_iter}.txt"
            )
            with open(metrics_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")


    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a training checkpoint and resume training.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        self.current_iter, training_state = load_checkpoint_impl(checkpoint_path, self.model, self.optimizer)

        # Restore early stopping state if available
        if training_state:
            self.best_val_loss = training_state.get('best_val_loss', float('inf'))
            self.patience_counter = training_state.get('patience_counter', 0)
            print(f"Resumed from iteration {self.current_iter}")
            print(f"  Best val loss: {self.best_val_loss:.4f}")
            print(f"  Patience counter: {self.patience_counter}")
        else:
            print(f"Resumed from iteration {self.current_iter} (no training state found)")


    def train(self):
        """
        Main training loop.

        Runs the complete training process including:
        - Training steps with learning rate scheduling
        - Periodic evaluation on validation set
        - Checkpointing at specified intervals
        - Logging to console and W&B
        """
        # Resume from checkpoint if specified
        if self.config.resume_from is not None:
            self.load_checkpoint(self.config.resume_from)

        start_iter = self.current_iter

        # Print training info
        print(f"\nStarting training from iteration {start_iter} to {self.config.max_iters}")
        print(f"Logging every {self.config.log_interval} iterations")
        print(f"Evaluating every {self.config.eval_interval} iterations")
        print(f"Checkpointing every {self.config.checkpoint_interval} iterations")
        print("-" * 80)

        # Training loop
        for iteration in tqdm(
            range(start_iter, self.config.max_iters),
            initial=start_iter,
            total=self.config.max_iters
        ):
            self.current_iter = iteration

            # Training step
            self.train_metrics = self.train_step()

            # Logging
            if iteration % self.config.log_interval == 0:
                self.log_metrics(self.train_metrics)

            # Evaluation
            if iteration % self.config.eval_interval == 0:
                val_loss = self.estimate_loss(self.val_data, num_batches=self.config.eval_iters)
                eval_metrics = {
                    'val_loss': val_loss,
                    'train_loss': self.train_metrics['loss'],
                    'lr': self.train_metrics['lr'],
                }
                self.log_metrics(eval_metrics)

                # Early stopping check
                if self.config.early_stopping_patience is not None:
                    if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                        # Improvement detected
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                    else:
                        # No improvement
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.early_stopping_patience:
                            print(f"\n🛑 Early stopping triggered at iteration {iteration}")
                            print(f"   Best val loss: {self.best_val_loss:.4f}")
                            print(f"   No improvement for {self.config.early_stopping_patience} evaluations")
                            self.early_stopped = True
                            break

            # Checkpointing
            if iteration % self.config.checkpoint_interval == 0 and iteration > 0:
                self.save_checkpoint(metrics=self.train_metrics)

        # Final evaluation and checkpoint
        if not self.early_stopped:
            self.current_iter = self.config.max_iters

        print(f"\nPerforming final evaluation at iteration {self.current_iter}...")
        val_loss = self.estimate_loss(self.val_data, num_batches=self.config.eval_iters)
        final_metrics = {
            'val_loss': val_loss,
            'train_loss': self.train_metrics['loss'],
            'lr': self.train_metrics['lr'],
        }
        self.log_metrics(final_metrics)

        # Save final checkpoint
        if self.early_stopped:
            print(f"\n✅ Training stopped early at iteration {self.current_iter}")
            print(f"   Best validation loss: {self.best_val_loss:.4f}")
        else:
            print("\n✅ Training complete!")
        print("Saving final checkpoint...")
        self.save_checkpoint(metrics=final_metrics)

        # Finish W&B
        if self.use_wandb:
            safe_wandb_call('finish')
