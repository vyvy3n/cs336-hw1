"""
Training utilities for transformer language models.

This module provides helper functions for training, evaluation, and logging.
"""
import os
import time
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .models import TransformerLM
from .optimizers import AdamW, CrossEntropyLoss, get_lr_cosine_schedule, gradient_clipping
from .utils import get_batch
from .config import TrainingConfig


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_loss(
    model: nn.Module,
    dataset: np.ndarray,
    config: TrainingConfig,
    num_batches: int = 20,
) -> float:
    """
    Estimate loss on a dataset by averaging over multiple batches.
    
    Args:
        model: The model to evaluate
        dataset: Dataset to evaluate on (memory-mapped numpy array)
        config: Training configuration
        num_batches: Number of batches to average over
    
    Returns:
        Average loss over the batches
    """
    model.eval()
    losses = []
    loss_fn = CrossEntropyLoss(device=config.device)
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Sample a batch
            x, y = get_batch(
                dataset,
                batch_size=config.data.batch_size,
                context_length=config.data.context_length,
                device=config.device,
            )
            
            # Forward pass
            logits = model(x)  # (batch_size, seq_len, vocab_size)
            
            # Reshape for loss computation
            # CrossEntropyLoss expects (batch_size, vocab_size) and (batch_size,)
            # So we flatten the sequence dimension
            logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
            targets_flat = y.view(-1)  # (batch_size * seq_len,)
            
            # Compute loss
            loss = loss_fn(logits_flat, targets_flat)
            losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    config: TrainingConfig,
    iteration: int,
) -> Dict[str, float]:
    """
    Perform a single training step.
    
    Args:
        model: The model to train
        optimizer: The optimizer
        train_data: Training dataset
        config: Training configuration
        iteration: Current iteration number
    
    Returns:
        Dictionary with training metrics (loss, lr, etc.)
    """
    model.train()
    
    # Get learning rate for this iteration
    lr = get_lr_cosine_schedule(
        it=iteration,
        max_learning_rate=config.optimizer.learning_rate,
        min_learning_rate=config.scheduler.get_min_lr(config.optimizer.learning_rate),
        warmup_iters=config.scheduler.warmup_iters,
        cosine_cycle_iters=config.scheduler.cosine_cycle_iters,
    )
    
    # Update learning rate in optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Sample a batch
    x, y = get_batch(
        train_data,
        batch_size=config.data.batch_size,
        context_length=config.data.context_length,
        device=config.device,
    )
    
    # Forward pass
    logits = model(x)  # (batch_size, seq_len, vocab_size)
    
    # Reshape for loss computation
    logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
    targets_flat = y.view(-1)  # (batch_size * seq_len,)
    
    # Compute loss
    loss_fn = CrossEntropyLoss(device=config.device)
    loss = loss_fn(logits_flat, targets_flat)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    gradient_clipping(model.parameters(), max_l2_norm=config.optimizer.grad_clip_norm)
    
    # Optimizer step
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'lr': lr,
    }


def log_metrics(
    iteration: int,
    metrics: Dict[str, Any],
    use_wandb: bool = False,
):
    """
    Log training metrics to console and optionally to Weights & Biases.
    
    Args:
        iteration: Current iteration number
        metrics: Dictionary of metrics to log
        use_wandb: Whether to log to Weights & Biases
    """
    # Console logging
    log_str = f"Iter {iteration:5d}"
    for key, value in metrics.items():
        if isinstance(value, float):
            log_str += f" | {key}: {value:.4f}"
        else:
            log_str += f" | {key}: {value}"
    print(log_str)
    
    # Weights & Biases logging
    if use_wandb:
        try:
            import wandb
            wandb.log(metrics, step=iteration)
        except ImportError:
            pass


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: TrainingConfig,
    metrics: Optional[Dict[str, float]] = None,
):
    """
    Save a training checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        iteration: Current iteration number
        config: Training configuration
        metrics: Optional metrics to save with checkpoint
    """
    from .utils import save_checkpoint as save_checkpoint_impl

    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"checkpoint_iter_{iteration}.pt"
    )

    print(f"Saving checkpoint to {checkpoint_path}")
    save_checkpoint_impl(model, optimizer, iteration, checkpoint_path)
    
    # Also save a "latest" checkpoint for easy resuming
    latest_path = os.path.join(config.checkpoint_dir, "checkpoint_latest.pt")
    save_checkpoint_impl(model, optimizer, iteration, latest_path)
    
    # Save metrics if provided
    if metrics is not None:
        metrics_path = os.path.join(
            config.checkpoint_dir,
            f"metrics_iter_{iteration}.txt"
        )
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> int:
    """
    Load a training checkpoint.
    
    Args:
        model: The model to load into
        optimizer: The optimizer to load into
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        The iteration number from the checkpoint
    """
    from .utils import load_checkpoint as load_checkpoint_impl
    
    print(f"Loading checkpoint from {checkpoint_path}")
    iteration = load_checkpoint_impl(checkpoint_path, model, optimizer)
    print(f"Resumed from iteration {iteration}")
    return iteration


def initialize_wandb(config: TrainingConfig):
    """
    Initialize Weights & Biases logging.
    
    Args:
        config: Training configuration
    """
    try:
        import wandb
        
        # Initialize wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.to_dict(),
        )
        
        print(f"Initialized Weights & Biases: {wandb.run.url}")
        return True
    except ImportError:
        print("Warning: wandb not installed, skipping W&B logging")
        return False
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return False


def train(config: TrainingConfig):
    """
    Main training loop.
    
    Args:
        config: Training configuration
    """
    # Set random seed
    set_seed(config.seed)
    
    # Initialize Weights & Biases if requested
    use_wandb = config.use_wandb and initialize_wandb(config)
    
    # Load datasets with memory mapping for efficiency
    print(f"Loading training data from {config.data.train_data_path}")
    train_data = np.load(config.data.train_data_path, mmap_mode='r')
    print(f"Training data shape: {train_data.shape}")
    
    print(f"Loading validation data from {config.data.val_data_path}")
    val_data = np.load(config.data.val_data_path, mmap_mode='r')
    print(f"Validation data shape: {val_data.shape}")
    
    # Initialize model
    print("Initializing model...")

    # Use ablation model if ablation_type is specified
    if hasattr(config.model, 'ablation_type') and config.model.ablation_type != "none":
        from cs336_basics.ablation_models import TransformerLMAblation
        print(f"Using ablation model: {config.model.ablation_type}")
        model = TransformerLMAblation(
            vocab_size=config.model.vocab_size,
            context_length=config.model.context_length,
            num_layers=config.model.num_layers,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            d_ff=config.model.d_ff,
            use_rope=config.model.use_rope,
            ablation_type=config.model.ablation_type,
            theta=config.model.theta,
            device=config.device,
        ).to(config.device)
    else:
        model = TransformerLM(
            vocab_size=config.model.vocab_size,
            context_length=config.model.context_length,
            num_layers=config.model.num_layers,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            d_ff=config.model.d_ff,
            use_rope=config.model.use_rope,
            theta=config.model.theta,
            device=config.device,
        ).to(config.device)
    
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if config.resume_from is not None:
        start_iter = load_checkpoint(model, optimizer, config.resume_from)
    
    # Training loop
    print(f"\nStarting training from iteration {start_iter} to {config.scheduler.max_iters}")
    print(f"Logging every {config.log_interval} iterations")
    print(f"Evaluating every {config.eval_interval} iterations")
    print(f"Checkpointing every {config.checkpoint_interval} iterations")
    print("-" * 80)
    
    for iteration in tqdm(range(start_iter, config.scheduler.max_iters), initial=start_iter, total=config.scheduler.max_iters):
        # Training step
        train_metrics = train_step(model, optimizer, train_data, config, iteration)

        # Logging
        if iteration % config.log_interval == 0:
            log_metrics(iteration, train_metrics, use_wandb=use_wandb)

        # Evaluation
        if iteration % config.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, config, num_batches=config.eval_iters)
            eval_metrics = {
                'val_loss': val_loss,
                'train_loss': train_metrics['loss'],
                'lr': train_metrics['lr'],
            }
            log_metrics(iteration, eval_metrics, use_wandb=use_wandb)

        # Checkpointing
        if iteration % config.checkpoint_interval == 0 and iteration > 0:
            save_checkpoint(model, optimizer, iteration, config, metrics=train_metrics)

    # Final evaluation and logging at max_iters
    final_iteration = config.scheduler.max_iters
    print(f"\nPerforming final evaluation at iteration {final_iteration}...")
    train_metrics = train_step(model, optimizer, train_data, config, final_iteration)
    val_loss = estimate_loss(model, val_data, config, num_batches=config.eval_iters)
    final_metrics = {
        'val_loss': val_loss,
        'train_loss': train_metrics['loss'],
        'lr': train_metrics['lr'],
    }
    log_metrics(final_iteration, final_metrics, use_wandb=use_wandb)

    # Final checkpoint
    print("\nTraining complete! Saving final checkpoint...")
    save_checkpoint(model, optimizer, final_iteration, config, metrics=train_metrics)
    
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
