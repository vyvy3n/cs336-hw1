"""
Data loading utilities for training language models.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray[np.int_], batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and their corresponding labels from a dataset.

    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample random contiguous sequences from the dataset to create
    training batches for language modeling.

    Args:
        dataset: 1D numpy array of integer token IDs in the dataset
        batch_size: Desired batch size to sample
        context_length: Desired context length of each sampled example
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the decvice
                to place the sampled input sequences and labels on

    Returns:
        A tuple of torch.LongTensors, each of shape (batch_size, context_length).
        The first tensor contains the sampled input sequences, and the second
        tensor contains the corresponding language modeling labels (next tokens)

    Raises:
        ValueError: If the dataset is too short to sample sequences of the requested length.
        RunTimeError: If the specified device is invalid or unavailable.
    """
    if len(dataset) < context_length + 1:
        raise ValueError(
            f"Dataset length {len(dataset)} is too short to sample sequences of "
            f"context_length {context_length}. Need at least {context_length + 1} tokens."
        )

    num_valid_positions = len(dataset) - context_length
    starting_indices = np.random.randint(0, num_valid_positions, size=batch_size)

    input_sequences = np.stack([dataset[start_idx : start_idx + context_length] for start_idx in starting_indices])
    target_sequences = np.stack(
        [dataset[start_idx + 1 : start_idx + 1 + context_length] for start_idx in starting_indices]
    )

    input_tensor = torch.from_numpy(input_sequences).long().to(device)
    target_tensor = torch.from_numpy(target_sequences).long().to(device)

    return input_tensor, target_tensor
