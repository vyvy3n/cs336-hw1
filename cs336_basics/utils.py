import numpy as np
import torch
import os
from typing import BinaryIO, IO


### Data utils for language modeling


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input sequences and their corresponding next-token targets from a dataset.

    Args:
        dataset (np.ndarray): 1D numpy array of integer token IDs in the dataset.
            For large datasets, use memory-mapped mode: np.load(path, mmap_mode='r')
    
    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels (next tokens).

    Example:
        >>> dataset = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> x, y = get_batch(dataset, batch_size=2, context_length=3, device='cpu')
        >>> # If we sample starting at index 2: x[0] = [2, 3, 4], y[0] = [3, 4, 5]
    """
    # Calculate the maximum valid starting index
    # If x = dataset[i:i+context_length], then y = dataset[i+1:i+context_length+1]
    # We need i+context_length+1 <= len(dataset), so i < len(dataset) - context_length
    max_start_idx = len(dataset) - context_length

    # Randomly sample batch_size starting indices from [0, max_start_idx)
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    # Optimization: Load a contiguous chunk (context_length + 1) and slice it
    # This reduces the number of separate array accesses from 2*batch_size to batch_size
    # and allows us to reuse data between x and y
    x_batch = np.empty((batch_size, context_length), dtype=np.int64)
    y_batch = np.empty((batch_size, context_length), dtype=np.int64)

    for i, start_idx in enumerate(start_indices):
        # Load a single contiguous chunk of size (context_length + 1)
        chunk = dataset[start_idx : start_idx + context_length + 1]
        # Split into input (first context_length tokens) and target (last context_length tokens)
        x_batch[i] = chunk[:context_length]
        y_batch[i] = chunk[1:]

    # Convert to PyTorch tensors and move to the specified device
    # Using torch.from_numpy creates a tensor that shares memory with the numpy array (zero-copy)
    # Then .to(device) copies to the target device if needed
    x_tensor = torch.from_numpy(x_batch).to(device)
    y_tensor = torch.from_numpy(y_batch).to(device)

    return x_tensor, y_tensor


### Model checkpointing utilities for saving and loading training state, including 
### model weights, optimizer state, and training iteration count.


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    training_state: dict = None,
) -> None:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    # Save additional training state (e.g., early stopping state)
    if training_state is not None:
        checkpoint['training_state'] = training_state
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict]:
    # Load checkpoint from disk
    # torch.load() automatically handles both file paths and file-like objects
    checkpoint = torch.load(src, weights_only=False)

    # Restore model state
    # load_state_dict() updates the model parameters in-place
    model.load_state_dict(checkpoint['model'])

    # Restore optimizer state
    # This includes momentum buffers, learning rate, etc.
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Get training state if available (for early stopping, etc.)
    training_state = checkpoint.get('training_state', {})

    # Return the iteration number and training state
    return checkpoint['iteration'], training_state


def get_checkpoint_paths(checkpoint_dir: str, iteration: int) -> tuple[str, str]:
    """
    Generate checkpoint file paths.

    Args:
        checkpoint_dir: Directory to save checkpoints
        iteration: Current iteration number

    Returns:
        Tuple of (numbered_checkpoint_path, latest_checkpoint_path)

    Example:
        >>> numbered, latest = get_checkpoint_paths("checkpoints", 1000)
        >>> print(numbered)  # "checkpoints/checkpoint_iter_1000.pt"
        >>> print(latest)    # "checkpoints/checkpoint_latest.pt"
    """
    numbered = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pt")
    latest = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    return numbered, latest


### Device and GPU utilities


def setup_device(requested_device: str = "cuda", verbose: bool = True) -> str:
    """
    Check device availability and optionally print GPU info.

    Args:
        requested_device: Requested device ('cuda' or 'cpu')
        verbose: Whether to print device information

    Returns:
        Valid device string ('cuda' or 'cpu')

    Example:
        >>> device = setup_device("cuda", verbose=True)
        📊 GPU Information:
          Device: NVIDIA A100
          Memory: 40.00 GB
    """
    if requested_device == "cuda" and torch.cuda.is_available():
        if verbose:
            print(f"\n📊 GPU Information:")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        return "cuda"
    else:
        if verbose:
            if requested_device == "cuda":
                print("\n⚠️  CUDA not available, falling back to CPU\n")
            else:
                print("\n⚠️  Using CPU\n")
        return "cpu"


### Weights & Biases utilities


def safe_wandb_call(func_name: str, *args, **kwargs):
    """
    Safely call a wandb function, handling import errors gracefully.

    Args:
        func_name: Function name as string (e.g., 'log', 'init', 'finish')
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Result of the function call, or None if wandb not available

    Example:
        >>> safe_wandb_call('log', {'loss': 0.5}, step=100)
        >>> safe_wandb_call('finish')
    """
    try:
        import wandb
        wandb_func = getattr(wandb, func_name)
        return wandb_func(*args, **kwargs)
    except (ImportError, AttributeError):
        return None


### Model utilities


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters

    Example:
        >>> model = TransformerLM(...)
        >>> num_params = count_parameters(model)
        >>> print(f"Model has {num_params:,} parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


### Experiment utilities


def print_experiment_header(title: str, params: dict):
    """
    Print a formatted experiment header.

    Args:
        title: Experiment title
        params: Dictionary of parameters to display

    Example:
        >>> print_experiment_header("Batch Size Sweep", {"batch_size": 32, "lr": 1e-3})
        ================================================================================
        Batch Size Sweep
        ================================================================================
        batch_size: 32
        lr: 0.001
        ================================================================================
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    for key, value in params.items():
        print(f"{key}: {value}")
    print(f"{'='*80}\n")


def handle_oom_error(batch_size: int) -> bool:
    """
    Handle out-of-memory errors during training.

    Args:
        batch_size: The batch size that caused OOM

    Returns:
        False (indicating failure)

    Example:
        >>> if not handle_oom_error(512):
        ...     print("Need to reduce batch size")
        ✗ Out of Memory
        Batch size 512 is too large for available GPU memory
    """
    print(f"\n✗ Out of Memory")
    print(f"Batch size {batch_size} is too large for available GPU memory\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return False
