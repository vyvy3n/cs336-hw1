import torch
import os
from typing import Union, BinaryIO, IO
from pathlib import Path

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
):
    """
    Save model checkpoint with all necessary state.
    
    Args:
        model: PyTorch model to save
        optimizer: PyTorch optimizer to save
        iteration: Current training iteration
        out: File path or file-like object to save to
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    Load model checkpoint and restore state.
    
    Args:
        src: File path or file-like object to load from
        model: PyTorch model to restore state to
        optimizer: PyTorch optimizer to restore state to
    
    Returns:
        int: The iteration number from the checkpoint
    """
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']
