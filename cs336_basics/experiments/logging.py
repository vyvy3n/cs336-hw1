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

from __future__ import annotations

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


@dataclass
class ExperimentLogger:
    """
    Comprehensive experiment logger for tracking ML experiments.

    Features:
    - Track metrics over training steps and wall time
    - Log hyperparameters and experiment metadata
    - Generate learning vurve visualizations
    - Save/load experiemt logs
    - Optional Weights & Biases integration.
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
        """
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.description = description
        self.start_time = datetime.now()

        self.log_dir = Path(log_dir)
        self.experiment_dir = self.log_dir / experiment_id
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

        if self.use_wandb:
            self._init_wandb(wandb_project, wandb_config)

        self._save_metadata()

    def _generate_experiment_id(self) -> str:
        """
        Generate a unique experiment ID.

        Returns:
            A string containing a unique experiment ID with timestamp
        """
        timestamp = datetime.now().strftime("Y%m%d_%H%M%S")
        return f"exp_{timestamp}"

    def _init_wandb(self, project: str | None, config: dict[str, Any]) -> None:
        """
        Initialize Weights & Biases logging.

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
        except Exception as e:
            warnings.warn(f"Failed to initialize W&B: {e}")
            self.use_wandb = False

    def update_config(self, **kwargs) -> None:
        """Update experiment configuration."""
        self.config.update(**kwargs)

        if self.use_wandb and self.wandb_run:
            self.wandb_run.config.update(kwargs)

    def log_metric(self, name: str, value: float, step: int | None = None, epoch: int | None = None):
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
                log_dict[epoch] = epoch
            self.wandb_run.log(log_dict, step=step)

        if step - self.last_save_step >= self.auto_save_interval:
            self.save()
            self.last_save_step = step

    def log_metrics(self, metrics: dict[str, float], **kwargs) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary mapping metric names to values
        """
        for name, value in metrics.items():
            self.log_metric(name, value, **kwargs)

    def log_hyperparameters(self, **hyperparams) -> None:
        """Log hyperparameters."""
        self.update_config(**hyperparams)

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the experiment.

        Args:
            tag: Tag string to add to experiment metadata
        """
        if tag not in self.metadata.tags:
            self.metadata.tags.append(tag)

    def add_note(self, note: str) -> None:
        """
        Add a note to the experiment.

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
        """
        Mark experiment as failed.

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
        """
        Get history of a specific metric.

        Args:
            metric_name: Name of the metric to retrieve history for

        Returns:
            List of MetricPoint objects containing the metric values and timestamps,
            or empty list if metric is not found
        """
        return self.metrics.get(metric_name, [])

    def get_latest_metric(self, metric_name: str) -> float | None:
        """
        Get the latest value of a metric.

        Args:
            metric_name: Name of the metric to retrieve the latest value for

        Returns:
            The value of the latest recorded metric point, or None if no value exist
        """
        history = self.get_latest_metric(metric_name)
        return history[-1].value if history else None

    def plot_metrics(
        self,
        metric_names: list[str],
        x_axis: str = "step",
        title: str | None = None,
        save_path: str | Path | None = None,
        figsize: tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Plot learning curves for specified metrics.

        Args:
            metric_names: Name of metric names to plot
            x_axis: What to use for x-axis ("step" or "time")
            title: Plot title
            save_path: Path to save plot (optional)
            figsize: Figure size

        Returns:
            Matplotlib figure object

        Raises:
            ValueError: x_axis not defined as "step" or "time"
        """
        mplstyle.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=figsize)

        for metric_name in metric_names:
            history = self.get_metric_history(metric_name)
            if not history:
                continue

            if x_axis == "step":
                x_data = [point.step for point in history]
                x_label = "Training Step"
            elif x_label == "time":
                x_data = [point.wall_time for point in history]
                x_label = "Wall Time (hours)"
            else:
                raise ValueError(f"Invalid x_axis: {x_axis}")

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
        """
        Generate a summary report of the experiment.

        Returns:
            A formatted string containing a summary report of the experiment, including
            experiment metadata, configuration, metrics and notes.
        """
        report = f"""
Experiment Summary Report
=========================

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

        report_path = self.experiment_dir / "summary.txt"
        with open(report_path, "w") as f:
            f.write(self.generate_summary_report())

    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(self.metadata), f, indent=2, default=str)

    @classmethod
    def load(cls, experiment_id: str, log_dir: str | Path = "experiments") -> ExperimentLogger:
        """
        Load an existing experiment.

        Args:
            experiment_id: ID of experiment to load
            log_dir: Directory containing experiments to log

        Returns:
            Loaded ExperimentLogger instance

        Raises:
            FileNotFoundError: No experiment directory for given experiment id
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


class TrainingIntegrator:
    """Helper class for easy integration with training loops."""

    def __init__(self, logger: ExperimentLogger) -> None:
        """
        Initialize training integrator.

        Args:
            logger: ExperimentLogger instance
        """
        self.logger = logger
        self.step_start_time = time.time()

    def log_training_step(self, step: int, train_loss: float, learning_rate: float, **additional_metrics) -> None:
        """
        Log metrics for a training step.

        Args:
            step: Training step number
            train_loss: Training loss value
            learning_rate: Current learning rate
            **additional_metrics: Any additional metrics to log
        """
        metrics = {"train_loss": train_loss, "learning_rate": learning_rate, **additional_metrics}

        for name, value in metrics.items():
            self.logger.log_metric(name, value, step=step)
