"""
Model and optimizer checkpointing utilities.
"""

from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save a checkpoint containing model state, optimizer state, and iteration number.

    Args:
        model: the model to save
        optimizer: the optimizer to save
        iteration: current iteration number
        out: path or file-like object to save the checkpoint to
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    """
    Load a checkpoint and restore model state, optimizer state, and iteration number.

    Args:
        src: path or file-like object to load the checkpoint from
        model: the model to restore state to
        optimizer: the optimizer to restore state to

    Returns:
        iteration number from the checkpoint
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
