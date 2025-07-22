"""
Experiment logging infrastructure.

This module provides comprehensive experiment tracking capabilities including:
- Experiment metadata and hyperparameter logging
- Metrics tracking with timestamps and step numbers
- Learning curve visualization
- Integration with training loops
- Optional Weights & Biases integration with hardware monitoring
- Experiment persistence and loading
"""

import json
import time
import warnings
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
class HardwareSnapshot:
    """Snapshot of hardware utilization at a specific point in time."""

    step: int
    wall_time: float

    tokens_per_second: float | None = None
    samples_per_second: float | None = None
    memory_efficiency: float | None = None

    # W&B automatically tracks these system metrics:
    # - gpu.{gpu_index}.gpu (GPU utilization %)
    # - gpu.{gpu_index}.memory (GPU memory utilization %)
    # - gpu.{gpu_index}.memoryAllocated (GPU memory allocated %)
    # - gpu.{gpu_index}.temp (GPU temperature)
    # - gpu.{gpu_index}.powerWatts (GPU power usage)
    # - cpu (CPU utilization %)
    # - proc.memory.percent (Process memory usage %)
    # - proc.memory.rssMB (Process memory RSS in MB)
    # - memory_percent (System memory usage %)
    # - disk.in/out (Disk I/O in MB)
    # - network.sent/recv (Network I/O)


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


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment."""

    experiment_id: str
    name: str
    description: str
    start_time: datetime
    end_time: datetime | None = None
    status: str = "running"
    tags: list[str] = field(default_factory=list)
    notes: str = ""


class HardwareMonitor:
    """
    Lightweight hardware monitoring that complements W&B's built-in system monitoring.

    This class focuses on ML-specific performance metrics while W&B automatically
    handles comprehensive system monitoring (GPU, CPU, memory, disk, network).
    """

    def __init__(
        self,
        enable_gpu_monitoring: bool = True,
        monitor_interval: float = 15.0,
    ) -> None:
        """
        Initialize hardware monitor.

        Args:
            enable_gpu_monitoring: Whether to enable GPU-specific monitoring
            monitor_interval: Interval between hardware snapshots (seconds)
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.monitor_interval = monitor_interval
        self.hardware_snapshots: list[HardwareSnapshot] = []
        self.start_time = time.time()

        self._last_tokens_processed = 0
        self._last_samples_processed = 0
        self._last_snapshot_time = time.time()

    def capture_snapshot(
        self,
        step: int,
        tokens_processed: int | None = None,
        samples_processed: int | None = None,
    ) -> HardwareSnapshot:
        """
        Capture a hardware utilization snapshot.

        Args:
            step: Training step number
            tokens_processed: Total tokens processed so far
            samples_processed: Total samples processed so far

        Returns:
            HardwareSnapshot with performance metrics
        """
        current_time = time.time()
        wall_time = current_time - self.start_time

        tokens_per_second = None
        samples_per_second = None

        if tokens_processed is not None and self._last_tokens_processed > 0:
            time_delta = current_time - self._last_snapshot_time
            if time_delta > 0:
                tokens_delta = tokens_processed - self._last_tokens_processed
                tokens_per_second = tokens_delta / time_delta

        if samples_processed is not None and self._last_samples_processed > 0:
            time_delta = current_time - self._last_snapshot_time
            if time_delta > 0:
                samples_delta = samples_processed - self._last_samples_processed
                samples_per_second = samples_delta / time_delta

        memory_efficiency = None
        if self.enable_gpu_monitoring:
            try:
                import torch

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    if reserved > 0:
                        memory_efficiency = allocated / reserved
            except ImportError:
                pass

        snapshot = HardwareSnapshot(
            step=step,
            wall_time=wall_time,
            tokens_per_second=tokens_per_second,
            samples_per_second=samples_per_second,
            memory_efficiency=memory_efficiency,
        )

        self.hardware_snapshots.append(snapshot)

        if tokens_processed is not None:
            self._last_tokens_processed = tokens_processed
        if samples_processed is not None:
            self._last_samples_processed = samples_processed
        self._last_snapshot_time = current_time

        return snapshot

    def get_latest_snapshot(self) -> HardwareSnapshot | None:
        """Get the most recent hardware snapshot."""
        return self.hardware_snapshots[-1] if self.hardware_snapshots else None

    def get_average_metrics(self) -> dict[str, float]:
        """Get average performance metrics across all snapshots."""
        if not self.hardware_snapshots:
            return {}

        metrics = {}
        valid_snapshots = [s for s in self.hardware_snapshots if s.tokens_per_second is not None]

        if valid_snapshots:
            metrics["avg_tokens_per_second"] = sum(s.tokens_per_second for s in valid_snapshots) / len(valid_snapshots)

        valid_snapshots = [s for s in self.hardware_snapshots if s.samples_per_second is not None]
        if valid_snapshots:
            metrics["avg_samples_per_second"] = sum(s.samples_per_second for s in valid_snapshots) / len(
                valid_snapshots
            )

        valid_snapshots = [s for s in self.hardware_snapshots if s.memory_efficiency is not None]
        if valid_snapshots:
            metrics["avg_memory_efficiency"] = sum(s.memory_efficiency for s in valid_snapshots) / len(valid_snapshots)

        return metrics

    def __enter__(self) -> "HardwareMonitor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass


class ExperimentLogger:
    """
    Comprehensive experiment logger for tracking ML experiments.

    Features:
    - Track metrics over training steps and wall time
    - Log hyperparameters and experiment metadata
    - Generate learning curve visualizations
    - Save/load experiment logs
    - Optional Weights & Biases integration with automatic system monitoring
    - Hardware performance tracking optimized for ML training
    - Easy integration with training loops
    """

    def __init__(
        self,
        experiment_name: str,
        experiment_id: str | None = None,
        description: str = "",
        log_dir: str | Path = "experiments",
        use_wandb: bool = True,
        wandb_project: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        auto_save_interval: int = 100,
        enable_hardware_monitoring: bool = True,
        hardware_monitor_interval: float = 15.0,
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Human-readable name for the experiment
            experiment_id: Unique identifier (auto-generated if None)
            description: Description of the experiment
            log_dir: Directory to save experiment logs
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            wandb_config: Additional W&B configuration
            auto_save_interval: Automatically save logs every N steps
            enable_hardware_monitoring: Whether to enable hardware monitoring
            hardware_monitor_interval: Interval between hardware snapshots (seconds)
        """
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.description = description
        self.start_time = datetime.now()

        self.log_dir = Path(log_dir)
        self.experiment_dir = self.log_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = ExperimentMetadata(
            experiment_id=self.experiment_id, name=experiment_name, description=description, start_time=self.start_time
        )

        self.config = ExperimentConfig()

        self.metrics: dict[str, list[MetricPoint]] = {}
        self.current_step = 0
        self.start_wall_time = time.time()

        self.auto_save_interval = auto_save_interval
        self.last_save_step = 0

        self.use_wandb = use_wandb
        self.wandb_run = None

        self.enable_hardware_monitoring = enable_hardware_monitoring
        self.hardware_monitor: HardwareMonitor | None = None

        if self.enable_hardware_monitoring:
            self.hardware_monitor = HardwareMonitor(
                enable_gpu_monitoring=True,
                monitor_interval=hardware_monitor_interval,
            )

        if self.use_wandb:
            self._init_wandb(wandb_project, wandb_config)

        self._save_metadata()

    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID.

        Returns:
            A string containing a unique experiment ID with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"

    def _init_wandb(self, project: str | None, config: dict[str, Any] | None) -> None:
        """Initialize Weights & Biases logging with system monitoring.

        Args:
            project: Name of the W&B project to log to. Defaults to "cs336-assignment1" if None
            config: Optional dictionary of configuration parameters to log to W&B
        """
        try:
            self.wandb_run = wandb.init(
                project=project or "cs336-assignment1",
                name=self.experiment_name,
                id=self.experiment_id,
                config=config or {},
                resume="allow",
            )

            if self.enable_hardware_monitoring:
                self.add_note(
                    "Hardware monitoring enabled: leveraging W&B system metrics + custom ML performance tracking"
                )

        except Exception as e:
            warnings.warn(f"Failed to initialize W&B: {e}")
            self.use_wandb = False

    def update_config(self, **kwargs) -> None:
        """Update experiment configuration."""
        self.config.update(**kwargs)

        if self.use_wandb and self.wandb_run:
            self.wandb_run.config.update(kwargs)

    def log_metric(self, name: str, value: float, step: int | None = None, epoch: int | None = None) -> None:
        """
        Log a metric value.

        Args:
            name: Metric name (e.g., 'train_loss', 'val_loss')
            value: Metric value
            step: Training step (uses internal counter if None)
            epoch: Training epoch (optional)
        """
        if step is None:
            step = self.current_step
        else:
            self.current_step = max(self.current_step, step)

        wall_time = time.time() - self.start_wall_time

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(MetricPoint(step=step, wall_time=wall_time, value=value, epoch=epoch))

        if self.use_wandb and self.wandb_run:
            log_dict = {name: value}
            if epoch is not None:
                log_dict["epoch"] = epoch
            self.wandb_run.log(log_dict, step=step)

        if step - self.last_save_step >= self.auto_save_interval:
            self.save()
            self.last_save_step = step

    def log_hardware_metrics(
        self,
        step: int,
        tokens_processed: int | None = None,
        samples_processed: int | None = None,
    ) -> None:
        """
        Log hardware performance metrics.

        Args:
            step: Training step number
            tokens_processed: Total tokens processed so far
            samples_processed: Total samples processed so far
        """
        if not self.enable_hardware_monitoring or not self.hardware_monitor:
            return

        snapshot = self.hardware_monitor.capture_snapshot(step, tokens_processed, samples_processed)

        if self.use_wandb and self.wandb_run:
            log_dict = {}

            if snapshot.tokens_per_second is not None:
                log_dict["tokens_per_second"] = snapshot.tokens_per_second

            if snapshot.samples_per_second is not None:
                log_dict["samples_per_second"] = snapshot.samples_per_second

            if snapshot.memory_efficiency is not None:
                log_dict["memory_efficiency"] = snapshot.memory_efficiency

            if log_dict:
                self.wandb_run.log(log_dict, step=step)

    def get_hardware_utilization_summary(self) -> dict[str, Any]:
        """
        Get a summary of hardware utilization metrics.

        Returns:
            Dictionary containing hardware utilization statistics
        """
        if not self.enable_hardware_monitoring or not self.hardware_monitor:
            return {"hardware_monitoring_enabled": False}

        average_metrics = self.hardware_monitor.get_average_metrics()
        latest_snapshot = self.hardware_monitor.get_latest_snapshot()

        summary = {
            "hardware_monitoring_enabled": True,
            "total_snapshots": len(self.hardware_monitor.hardware_snapshots),
            "average_metrics": average_metrics,
        }

        if latest_snapshot:
            summary["latest_metrics"] = {
                "step": latest_snapshot.step,
                "tokens_per_second": latest_snapshot.tokens_per_second,
                "samples_per_second": latest_snapshot.samples_per_second,
                "memory_efficiency": latest_snapshot.memory_efficiency,
            }

        summary["wandb_system_metrics"] = {
            "note": "GPU, CPU, memory, disk, and network metrics are automatically logged by W&B",
            "view_in_dashboard": "Check the 'System' tab in your W&B run dashboard",
        }

        return summary

    def log_metrics(self, metrics: dict[str, float], **kwargs) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary mapping metric names to values
            **kwargs: Additional arguments passed to log_metric()
        """
        for name, value in metrics.items():
            self.log_metric(name, value, **kwargs)

    def log_hyperparameters(self, **hyperparams) -> None:
        """Log hyperparameters."""
        self.update_config(**hyperparams)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment.

        Args:
            tag: Tag string to add to experiment metadata
        """
        if tag not in self.metadata.tags:
            self.metadata.tags.append(tag)

    def add_note(self, note: str) -> None:
        """Add a note to the experiment.

        Args:
            note: Text note to add to experiment metadata
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata.notes += f"\n[{timestamp}] {note}"

    def mark_completed(self) -> None:
        """Mark experiment as completed."""
        self.metadata.status = "completed"
        self.metadata.end_time = datetime.now()
        self.save()

        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

    def mark_failed(self, error_message: str = "") -> None:
        """Mark experiment as failed.

        Args:
            error_message: Optional error message to add to experiment notes
        """
        self.metadata.status = "failed"
        self.metadata.end_time = datetime.now()
        if error_message:
            self.add_note(f"FAILED: {error_message}")
        self.save()

        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish(exit_code=1)

    def get_metric_history(self, metric_name: str) -> list[MetricPoint]:
        """Get history of a specific metric.

        Args:
            metric_name: Name of the metric to retrieve history for

        Returns:
            list of MetricPoint objects containing the metric values and timestamps,
            or empty list if metric not found
        """
        return self.metrics.get(metric_name, [])

    def get_latest_metric(self, metric_name: str) -> float | None:
        """Get the latest value of a metric.

        Args:
            metric_name: Name of the metric to retrieve the latest value for

        Returns:
            The value of the latest recorded metric point, or None if no values exist
        """
        history = self.get_metric_history(metric_name)
        return history[-1].value if history else None

    def plot_metrics(
        self,
        metric_names: list[str],
        x_axis: str = "step",
        title: str | None = None,
        save_path: str | Path | None = None,
        figsize: tuple = (12, 8),
    ):
        """
        Plot learning curves for specified metrics.

        Args:
            metric_names: list of metric names to plot
            x_axis: What to use for x-axis ("step" or "time")
            title: Plot title
            save_path: Path to save plot (optional)
            figsize: Figure size

        Returns:
            Matplotlib figure object if matplotlib is available, None otherwise
        """
        assert mplstyle is not None
        assert plt is not None

        if x_axis == "step":
            x_label = "Training Step"
        elif x_axis == "time":
            x_label = "Wall Time (hours)"
        else:
            raise ValueError(f"Invalid x_axis: {x_axis}")

        mplstyle.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=figsize)

        for metric_name in metric_names:
            history = self.get_metric_history(metric_name)
            if not history:
                continue

            if x_axis == "step":
                x_data = [point.step for point in history]
            else:
                x_data = [point.wall_time / 3600 for point in history]

            y_data = [point.value for point in history]
            ax.plot(x_data, y_data, label=metric_name, linewidth=2)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Value")
        ax.set_title(title or f"Learning Curves - {self.experiment_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plot_path = self.experiment_dir / f"learning_curves_{x_axis}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

        return fig

    def generate_summary_report(self) -> str:
        """Generate a summary report of the experiment.

        Returns:
            A formatted string containing a summary report of the experiment, including
            experiment metadata, configuration, metrics, and hardware utilization.
        """
        report = f"""
Experiment Summary Report
========================

Experiment ID: {self.experiment_id}
Name: {self.experiment_name}
Description: {self.description}
Status: {self.metadata.status}
Start Time: {self.metadata.start_time.strftime("%Y-%m-%d %H:%M:%S")}
"""

        if self.metadata.end_time:
            duration = self.metadata.end_time - self.metadata.start_time
            report += f"End Time: {self.metadata.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"Duration: {duration}\n"

        report += f"Tags: {', '.join(self.metadata.tags) if self.metadata.tags else 'None'}\n\n"

        config_dict = asdict(self.config)
        non_none_config = {k: v for k, v in config_dict.items() if v is not None and k != "custom"}
        if non_none_config:
            report += "Configuration:\n"
            for key, value in non_none_config.items():
                report += f"  {key}: {value}\n"

        if self.config.custom:
            report += "Custom Parameters:\n"
            for key, value in self.config.custom.items():
                report += f"  {key}: {value}\n"

        if self.metrics:
            report += "\nMetrics Summary:\n"
            for metric_name, history in self.metrics.items():
                if history:
                    latest = history[-1]
                    report += f"  {metric_name}: {latest.value:.6f} (step {latest.step})\n"

        hardware_summary = self.get_hardware_utilization_summary()
        if hardware_summary.get("hardware_monitoring_enabled", False):
            report += "\nHardware Utilization Summary:\n"

            avg_metrics = hardware_summary.get("average_metrics", {})
            if avg_metrics:
                report += "  Average Performance Metrics:\n"
                for metric, value in avg_metrics.items():
                    report += f"    {metric}: {value:.2f}\n"

            latest_metrics = hardware_summary.get("latest_metrics", {})
            if latest_metrics and latest_metrics.get("step") is not None:
                report += f"  Latest Metrics (step {latest_metrics['step']}):\n"
                for metric, value in latest_metrics.items():
                    if metric != "step" and value is not None:
                        report += f"    {metric}: {value:.2f}\n"

            report += "\n  Note: Comprehensive system metrics (GPU, CPU, memory, disk, network)\n"
            report += "        are automatically logged by W&B. View them in the dashboard.\n"

        if self.metadata.notes.strip():
            report += f"\nNotes:{self.metadata.notes}\n"

        return report

    def save(self) -> None:
        """Save experiment data to disk."""
        self._save_metadata()

        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        metrics_path = self.experiment_dir / "metrics.json"
        metrics_data = {}
        for name, history in self.metrics.items():
            metrics_data[name] = [asdict(point) for point in history]

        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)

        if self.enable_hardware_monitoring and self.hardware_monitor:
            hardware_path = self.experiment_dir / "hardware_metrics.json"
            hardware_data = {
                "snapshots": [asdict(snapshot) for snapshot in self.hardware_monitor.hardware_snapshots],
                "summary": self.get_hardware_utilization_summary(),
            }
            with open(hardware_path, "w") as f:
                json.dump(hardware_data, f, indent=2, default=str)

        report_path = self.experiment_dir / "summary.txt"
        with open(report_path, "w") as f:
            f.write(self.generate_summary_report())

    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(self.metadata), f, indent=2, default=str)

    @classmethod
    def load(cls, experiment_id: str, log_dir: str | Path = "experiments") -> "ExperimentLogger":
        """
        Load an existing experiment.

        Args:
            experiment_id: ID of experiment to load
            log_dir: Directory containing experiment logs

        Returns:
            Loaded ExperimentLogger instance
        """
        log_dir = Path(log_dir)
        experiment_dir = log_dir / experiment_id

        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found in {log_dir}")

        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)

        logger = cls(
            experiment_name=metadata_dict["name"],
            experiment_id=experiment_id,
            description=metadata_dict["description"],
            log_dir=log_dir,
            use_wandb=False,
            enable_hardware_monitoring=False,
        )

        logger.metadata = ExperimentMetadata(**metadata_dict)
        if isinstance(logger.metadata.start_time, str):
            logger.metadata.start_time = datetime.fromisoformat(logger.metadata.start_time)
        if logger.metadata.end_time and isinstance(logger.metadata.end_time, str):
            logger.metadata.end_time = datetime.fromisoformat(logger.metadata.end_time)

        config_path = experiment_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            logger.config = ExperimentConfig(**config_dict)

        metrics_path = experiment_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)

            for name, history_data in metrics_data.items():
                logger.metrics[name] = [MetricPoint(**point_data) for point_data in history_data]

        hardware_path = experiment_dir / "hardware_metrics.json"
        if hardware_path.exists():
            with open(hardware_path, "r") as f:
                hardware_data = json.load(f)

            logger.enable_hardware_monitoring = True
            logger.hardware_monitor = HardwareMonitor()
            logger.hardware_monitor.hardware_snapshots = [
                HardwareSnapshot(**snapshot_data) for snapshot_data in hardware_data.get("snapshots", [])
            ]

        return logger


class TrainingIntegrator:
    """Helper class for easy integration with training loops and hardware monitoring."""

    def __init__(
        self,
        logger: ExperimentLogger,
        hardware_log_interval: int = 50,
    ):
        """
        Initialize training integrator.

        Args:
            logger: ExperimentLogger instance
            hardware_log_interval: Log hardware metrics every N steps
        """
        self.logger = logger
        self.step_start_time = time.time()
        self.hardware_log_interval = hardware_log_interval

        self.total_tokens_processed = 0
        self.total_samples_processed = 0

    def log_training_step(
        self,
        step: int,
        train_loss: float,
        learning_rate: float,
        tokens_processed: int | None = None,
        samples_processed: int | None = None,
        **additional_metrics,
    ) -> None:
        """
        Log metrics for a training step with optional hardware monitoring.

        Args:
            step: Training step number
            train_loss: Training loss value
            learning_rate: Current learning rate
            tokens_processed: Number of tokens processed in this step
            samples_processed: Number of samples processed in this step
            **additional_metrics: Any additional metrics to log
        """
        metrics = {"train_loss": train_loss, "learning_rate": learning_rate, **additional_metrics}

        for name, value in metrics.items():
            self.logger.log_metric(name, value, step=step)

        if tokens_processed is not None:
            self.total_tokens_processed += tokens_processed
        if samples_processed is not None:
            self.total_samples_processed += samples_processed

        if step % self.hardware_log_interval == 0:
            self.logger.log_hardware_metrics(
                step=step,
                tokens_processed=self.total_tokens_processed if tokens_processed is not None else None,
                samples_processed=self.total_samples_processed if samples_processed is not None else None,
            )

    def log_validation_step(
        self, step: int, val_loss: float, perplexity: float | None = None, **additional_metrics
    ) -> None:
        """
        Log validation metrics.

        Args:
            step: Training step number
            val_loss: Validation loss
            perplexity: Perplexity score (optional)
            **additional_metrics: Any additional metrics to log
        """
        metrics = {"val_loss": val_loss, **additional_metrics}

        if perplexity is not None:
            metrics["val_perplexity"] = perplexity

        for name, value in metrics.items():
            self.logger.log_metric(name, value, step=step)

    def start_epoch(self, epoch: int) -> None:
        """Signal start of a new epoch.

        Args:
            epoch: The epoch number to log
        """
        self.logger.log_metric("epoch", epoch, step=self.logger.current_step)

    def log_step_time(self, step: int) -> None:
        """Log the time taken for the current step.

        Args:
            step: Training step number to log the time for
        """
        current_time = time.time()
        step_time = current_time - self.step_start_time
        self.logger.log_metric("step_time", step_time, step=step)
        self.step_start_time = current_time


def create_experiment_logger(name: str, description: str = "", **config_kwargs) -> ExperimentLogger:
    """
    Convenience function to create and configure an experiment logger.

    Args:
        name: Experiment name
        description: Experiment description
        **config_kwargs: Configuration parameters to log

    Returns:
        Configured ExperimentLogger instance
    """
    logger = ExperimentLogger(experiment_name=name, description=description, use_wandb=True)

    if config_kwargs:
        logger.update_config(**config_kwargs)

    return logger


def compare_experiments(
    experiment_ids: list[str], metric_name: str, log_dir: str | Path = "experiments", x_axis: str = "step"
):
    """
    Compare multiple experiments by plotting a specific metric.

    Args:
        experiment_ids: list of experiment IDs to compare
        metric_name: Name of metric to compare
        log_dir: Directory containing experiment logs
        x_axis: What to use for x-axis ("step" or "time")

    Returns:
        Matplotlib figure with comparison plot if matplotlib is available, None otherwise
    """
    assert mplstyle is not None
    assert plt is not None

    if x_axis == "step":
        x_label = "Training Step"
    elif x_axis == "time":
        x_label = "Wall Time (hours)"
    else:
        raise ValueError(f"Invalid x_axis: {x_axis}")

    mplstyle.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 8))

    for exp_id in experiment_ids:
        try:
            logger = ExperimentLogger.load(exp_id, log_dir)
            history = logger.get_metric_history(metric_name)

            if not history:
                continue

            if x_axis == "step":
                x_data = [point.step for point in history]
            else:
                x_data = [point.wall_time / 3600 for point in history]

            y_data = [point.value for point in history]
            ax.plot(x_data, y_data, label=f"{logger.experiment_name} ({exp_id})", linewidth=2)

        except Exception as e:
            warnings.warn(f"Failed to load experiment {exp_id}: {e}")

    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Experiment Comparison - {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
