#!/usr/bin/env python3
"""Shared utilities for experiment scripts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.config import TrainingConfig
from cs336_basics.training import Trainer


def run_experiment(config: TrainingConfig, handle_oom: bool = False) -> bool:
    """Run a training experiment. Returns True if successful."""
    try:
        trainer = Trainer(config)
        trainer.train()
        print(f"\n✓ Completed\n")
        return True
    except RuntimeError as e:
        if handle_oom and "out of memory" in str(e).lower():
            print(f"\n⚠ OOM at batch_size={config.batch_size}\n")
            return False
        print(f"\n✗ Failed: {e}\n")
        return False
    except Exception as e:
        print(f"\n✗ Failed: {e}\n")
        return False
