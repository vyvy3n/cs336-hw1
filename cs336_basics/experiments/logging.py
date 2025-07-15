"""
Experiment logging infrastructure.

This module provides comprehensive experiment tracking capabilities including:
- Experiment tracking and hyperparameter logging
- Metrics tracking with timestamps and step numbers
- Learning curve visualization
- Integration with training loops
- Optional Weights & Biases integration
- Experiment persistence and loading
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import wandb


@dataclass
class MetricPoint:
    """A single metric measurement point."""

    step: int
    wall_time: float
    value: float
    epoch: int | None = None


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    # Model hyperparameters
    vocab_size: int | None = None
    context_length: int | None = None
    d_model: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None
    d_ff: int | None = None

    # Training hyperparameters
    learning_rate: float | None = None
    batch_size: int | None = None
    weight_decay: float | None = None
    beta1: float | None = None
    beta2: float | None = None
    eps: float | None = None
    warmup_steps: int | None = None
    max_steps: int | None = None

    # Other settings
    seed: int | None = None
    dataset: str | None = None
    tokenizer: str | None = None

    # Custom hyperparameters
    custom: dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom[key] = value
