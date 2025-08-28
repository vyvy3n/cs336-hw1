from torch import Tensor
import numpy as np
import torch

def data_loader(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:

    if len(x) < context_length + 1:
        raise ValueError(f"Dataset too short: {len(x)} < {context_length + 1}")

    max_start = len(x) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    seq_indices = torch.arange(context_length).unsqueeze(0) + torch.tensor(starts).unsqueeze(1)
    
    # Extract input and target sequences
    inputs = x[seq_indices]  # Shape: (batch_size, context_length)
    targets = x[seq_indices + 1]  # Shape: (batch_size, context_length)

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets