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
) -> None:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    # Load checkpoint from disk
    # torch.load() automatically handles both file paths and file-like objects
    checkpoint = torch.load(src, weights_only=False)
    
    # Restore model state
    # load_state_dict() updates the model parameters in-place
    model.load_state_dict(checkpoint['model'])
    
    # Restore optimizer state
    # This includes momentum buffers, learning rate, etc.
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Return the iteration number so training can resume from the correct point
    return checkpoint['iteration']
