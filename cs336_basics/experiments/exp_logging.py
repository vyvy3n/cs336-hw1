"""
Advanced experiment logging with comprehensive performance monitoring.
"""

from __future__ import annotations

import json
import os
import time

import pynvml
import torch
import wandb


class MemoryMonitor:
    """Monitor GPU memory usage and efficiency."""

    def __init__(self):
        self.peak_memory = 0
        self.baseline_memory = 0
        self.reset()

    def reset(self):
        """Reset memory monitoring."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.baseline_memory = torch.cuda.memory_allocated()
            self.peak_memory = self.baseline_memory

    def get_stats(self) -> dict[str, float]:
        """Get comprehensive memory statistics."""
        if not torch.cuda.is_available():
            return {}

        stats = {}

        current_allocated = torch.cuda.memory_allocated()
        current_reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()

        stats["memory_allocated_gb"] = current_allocated / 1e9
        stats["memory_reserved_gb"] = current_reserved / 1e9
        stats["max_memory_allocated_gb"] = max_allocated / 1e9
        stats["max_memory_reserved_gb"] = max_reserved / 1e9

        total_memory = torch.cuda.get_device_properties(0).total_memory
        stats["total_memory_gb"] = total_memory / 1e9
        stats["memory_utilization"] = current_allocated / total_memory
        stats["peak_memory_utilization"] = max_allocated / total_memory

        if current_reserved > 0:
            stats["memory_efficiency"] = current_allocated / current_reserved
        else:
            stats["memory_efficiency"] = 0.0

        return stats


class PerformanceMonitor:
    """Monitor training performance metrics."""

    def __init__(self):
        self.step_times = []
        self.throughput_history = []
        self.loss_history = []
        self.start_time = time.time()
        self.memory_monitor = MemoryMonitor()

    def log_step(self, step_time: float, tokens_per_sec: float, loss: float, batch_size: int, sequence_length: int):
        """Log performance metrics for a training step."""
        self.step_times.append(step_time)
        self.throughput_history.append(tokens_per_sec)
        self.loss_history.append(loss)

        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]
            self.throughput_history = self.throughput_history[-100:]
            self.loss_history = self.loss_history[-100:]

    def get_stats(self) -> dict[str, float]:
        """Get comprehensive performance statistics."""
        stats = {}

        if self.step_times:
            stats["avg_step_time"] = sum(self.step_times) / len(self.step_times)
            stats["min_step_time"] = min(self.step_times)
            stats["max_step_time"] = max(self.step_times)

        if self.throughput_history:
            stats["avg_tokens_per_sec"] = sum(self.throughput_history) / len(self.throughput_history)
            stats["max_tokens_per_sec"] = max(self.throughput_history)

        if self.loss_history:
            recent_losses = self.loss_history[-10:]
            if len(recent_losses) > 1:
                loss_trend = recent_losses[-1] - recent_losses[0]
                stats["loss_trend"] = loss_trend

        memory_stats = self.memory_monitor.get_stats()
        stats.update(memory_stats)

        stats["total_runtime_hours"] = (time.time() - self.start_time) / 3600

        return stats


class ExperimentLogger:
    """
    Enhanced experiment logger with comprehensive tracking.
    """

    def __init__(
        self,
        experiment_name: str,
        description: str = "",
        log_dir: str = "experiments",
        use_wandb: bool = True,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
    ):
        self.experiment_name = experiment_name
        self.description = description
        self.log_dir = log_dir
        self.use_wandb = use_wandb

        os.makedirs(log_dir, exist_ok=True)
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        if use_wandb:
            wandb.init(
                project=wandb_project or "ml-experiments",
                entity=wandb_entity,
                name=experiment_name,
                notes=description,
            )
            self.wandb = wandb
            print(f"Initialized W&B logging for project: {wandb_project}")
        else:
            self.wandb = None

        self.performance_monitor = PerformanceMonitor()

        self.metadata = {
            "name": experiment_name,
            "description": description,
            "start_time": time.time(),
            "status": "running",
            "notes": [],
        }

        self._log_system_info()

    def _log_system_info(self):
        """Log system and hardware information."""
        system_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            system_info.update(
                {
                    "cuda_version": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                    "gpu_count": torch.cuda.device_count(),
                }
            )

        self.metadata["system_info"] = system_info
        self._save_metadata()

    def log_hyperparameters(self, **hyperparams):
        """Log experiment hyperparameters."""
        self.metadata["hyperparameters"] = hyperparams
        self._save_metadata()

        if self.wandb:
            self.wandb.config.update(hyperparams)

    def log_metrics(self, step: int, **metrics):
        """Log training metrics."""
        perf_stats = self.performance_monitor.get_stats()
        metrics.update(perf_stats)

        log_entry = {"step": step, "timestamp": time.time(), **metrics}
        log_file = os.path.join(self.experiment_dir, "metrics.jsonl")

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if self.wandb:
            self.wandb.log(metrics, step=step)

    def add_note(self, note: str):
        """Add a note to the experiment."""
        timestamped_note = f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {note}"
        self.metadata["notes"].append(timestamped_note)
        self._save_metadata()

        if self.wandb:
            self.wandb.notes = "\n".join(self.metadata["notes"])

    def mark_completed(self, success: bool = True):
        """Mark experiment as completed."""
        self.metadata["status"] = "completed" if success else "failed"
        self.metadata["end_time"] = time.time()
        self.metadata["duration_hours"] = (self.metadata["end_time"] - self.metadata["start_time"]) / 3600

        final_stats = self.performance_monitor.get_stats()
        self.metadata["final_performance"] = final_stats

        self._save_metadata()

        if self.wandb:
            self.wandb.finish()

    def _save_metadata(self):
        """Save experiment metadata to file."""
        metadata_file = os.path.join(self.experiment_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)


class TrainingIntegrator:
    """
    Integration layer for training loop with comprehensive monitoring.
    """

    def __init__(
        self,
        experiment_logger: ExperimentLogger,
        hardware_log_interval: int = 100,
    ):
        self.logger = experiment_logger
        self.hardware_log_interval = hardware_log_interval
        self.step_count = 0

        self.current_epoch = 0
        self.total_tokens_processed = 0
        self.total_samples_processed = 0

        self.best_val_loss = float("inf")
        self.steps_since_improvement = 0

    def start_epoch(self, epoch: int):
        """Start a new training epoch."""
        self.current_epoch = epoch
        self.logger.add_note(f"Started epoch {epoch}")

    def log_training_step(
        self,
        step: int,
        train_loss: float,
        learning_rate: float,
        tokens_processed: int,
        samples_processed: int,
        step_time: float,
        tokens_per_sec: float,
        wallclock_time: float | None = None,
        **additional_metrics,
    ):
        """Log comprehensive training step metrics with optional wallclock time."""
        self.step_count = step
        self.total_tokens_processed += tokens_processed
        self.total_samples_processed += samples_processed

        batch_size = samples_processed
        sequence_length = tokens_processed // batch_size if batch_size > 0 else 0

        self.logger.performance_monitor.log_step(step_time, tokens_per_sec, train_loss, batch_size, sequence_length)

        metrics = {
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "step_time": step_time,
            "tokens_per_second": tokens_per_sec,
            "total_tokens_processed": self.total_tokens_processed,
            "total_samples_processed": self.total_samples_processed,
            "epoch": self.current_epoch,
            **additional_metrics,
        }

        # Use wallclock time as step for W&B if provided, otherwise use step number
        if wallclock_time is not None:
            metrics["wallclock_hours"] = wallclock_time
            metrics["step_number"] = step  # Keep original step as metric
            # Convert wallclock hours to integer deciseconds for W&B step (good resolution)
            wallclock_step = int(wallclock_time * 36000)  # hours -> deciseconds
            self.logger.log_metrics(wallclock_step, **metrics)
        else:
            self.logger.log_metrics(step, **metrics)

        if step % self.hardware_log_interval == 0:
            self._log_hardware_stats(step if wallclock_time is None else int(wallclock_time * 36000))

    def log_validation_step(self, step: int, val_loss: float, perplexity: float, wallclock_time: float | None = None, **additional_metrics):
        """Log validation metrics with optional wallclock time."""
        metrics = {"val_loss": val_loss, "val_perplexity": perplexity, **additional_metrics}

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.steps_since_improvement = 0
            metrics["best_val_loss"] = self.best_val_loss
            self.logger.add_note(f"New best validation loss: {val_loss:.4f}")
        else:
            self.steps_since_improvement += 1
            metrics["steps_since_improvement"] = self.steps_since_improvement

        # Use wallclock time as step for W&B if provided, otherwise use step number  
        if wallclock_time is not None:
            metrics["wallclock_hours"] = wallclock_time
            metrics["step_number"] = step  # Keep original step as metric
            # Convert wallclock hours to integer deciseconds for W&B step (good resolution)
            wallclock_step = int(wallclock_time * 36000)  # hours -> deciseconds
            self.logger.log_metrics(wallclock_step, **metrics)
        else:
            self.logger.log_metrics(step, **metrics)

    def _log_hardware_stats(self, step: int):
        """Log detailed hardware utilization stats."""
        if not torch.cuda.is_available():
            return

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000

        hw_metrics = {
            "gpu_utilization_percent": utilization.gpu,
            "gpu_memory_utilization_percent": utilization.memory,
            "gpu_temperature_c": temp,
            "gpu_power_usage_w": power,
        }

        self.logger.log_metrics(step, **hw_metrics)
