"""
Comprehensive tests for experiment logging infrastructure.

This module tests the complete experiment tracking capabilities including:
- Experiment metadata and hyperparameter logging
- Metrics tracking with timestamps and step numbers
- Learning curve visualization
- Integration with training loops
- Weights & Biases integration (mocked)
- Hardware monitoring and performance tracking
- Experiment persistence and loading
- Error handling and edge cases
"""

import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cs336_basics.experiments.exp_logging import (
    ExperimentConfig,
    ExperimentLogger,
    ExperimentMetadata,
    HardwareMonitor,
    HardwareSnapshot,
    MetricPoint,
    TrainingIntegrator,
    compare_experiments,
    create_experiment_logger,
)


class TestMetricPoint:
    """Test the MetricPoint dataclass."""

    def test_metric_point_creation(self) -> None:
        """Test basic MetricPoint creation."""
        point = MetricPoint(step=100, wall_time=1.5, value=0.75)
        assert point.step == 100
        assert point.wall_time == 1.5
        assert point.value == 0.75
        assert point.epoch is None

    def test_metric_point_with_epoch(self) -> None:
        """Test MetricPoint creation with epoch."""
        point = MetricPoint(step=100, wall_time=1.5, value=0.75, epoch=5)
        assert point.epoch == 5

    def test_metric_point_equality(self) -> None:
        """Test MetricPoint equality comparison."""
        point1 = MetricPoint(step=100, wall_time=1.5, value=0.75, epoch=5)
        point2 = MetricPoint(step=100, wall_time=1.5, value=0.75, epoch=5)
        point3 = MetricPoint(step=101, wall_time=1.5, value=0.75, epoch=5)

        assert point1 == point2
        assert point1 != point3


class TestHardwareSnapshot:
    """Test the HardwareSnapshot dataclass."""

    def test_hardware_snapshot_creation(self) -> None:
        """Test basic HardwareSnapshot creation."""
        snapshot = HardwareSnapshot(step=100, wall_time=1.5)
        assert snapshot.step == 100
        assert snapshot.wall_time == 1.5
        assert snapshot.tokens_per_second is None
        assert snapshot.samples_per_second is None
        assert snapshot.memory_efficiency is None

    def test_hardware_snapshot_with_metrics(self) -> None:
        """Test HardwareSnapshot creation with performance metrics."""
        snapshot = HardwareSnapshot(
            step=100,
            wall_time=1.5,
            tokens_per_second=1000.0,
            samples_per_second=50.0,
            memory_efficiency=0.85,
        )
        assert snapshot.tokens_per_second == 1000.0
        assert snapshot.samples_per_second == 50.0
        assert snapshot.memory_efficiency == 0.85

    def test_hardware_snapshot_equality(self) -> None:
        """Test HardwareSnapshot equality comparison."""
        snapshot1 = HardwareSnapshot(step=100, wall_time=1.5, tokens_per_second=1000.0)
        snapshot2 = HardwareSnapshot(step=100, wall_time=1.5, tokens_per_second=1000.0)
        snapshot3 = HardwareSnapshot(step=101, wall_time=1.5, tokens_per_second=1000.0)

        assert snapshot1 == snapshot2
        assert snapshot1 != snapshot3


class TestHardwareMonitor:
    """Test the HardwareMonitor class."""

    def test_hardware_monitor_creation(self) -> None:
        """Test HardwareMonitor initialization."""
        monitor = HardwareMonitor(enable_gpu_monitoring=True, monitor_interval=10.0)
        assert monitor.enable_gpu_monitoring is True
        assert monitor.monitor_interval == 10.0
        assert len(monitor.hardware_snapshots) == 0
        assert monitor._last_tokens_processed == 0
        assert monitor._last_samples_processed == 0

    def test_hardware_monitor_context_manager(self) -> None:
        """Test HardwareMonitor as context manager."""
        with HardwareMonitor() as monitor:
            assert monitor is not None
            assert isinstance(monitor, HardwareMonitor)

    def test_capture_snapshot_basic(self) -> None:
        """Test basic snapshot capture."""
        monitor = HardwareMonitor()

        snapshot = monitor.capture_snapshot(step=0, tokens_processed=1000, samples_processed=32)

        assert snapshot.step == 0
        assert snapshot.wall_time >= 0
        assert len(monitor.hardware_snapshots) == 1

        # First snapshot should have None for throughput metrics
        assert snapshot.tokens_per_second is None
        assert snapshot.samples_per_second is None

    def test_capture_snapshot_throughput_calculation(self) -> None:
        """Test throughput calculation in snapshots."""
        monitor = HardwareMonitor()

        # First snapshot
        snapshot1 = monitor.capture_snapshot(step=0, tokens_processed=1000, samples_processed=32)
        time.sleep(0.1)  # Small delay to ensure time difference

        # Second snapshot - should calculate throughput
        snapshot2 = monitor.capture_snapshot(step=1, tokens_processed=2000, samples_processed=64)

        assert snapshot1.tokens_per_second is None
        assert snapshot1.samples_per_second is None

        # Second snapshot should have throughput metrics
        assert snapshot2.tokens_per_second is not None
        assert snapshot2.samples_per_second is not None
        assert snapshot2.tokens_per_second > 0
        assert snapshot2.samples_per_second > 0

    @patch("builtins.__import__")
    def test_capture_snapshot_memory_efficiency(self, mock_import) -> None:
        """Test memory efficiency calculation."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 850 * 1024 * 1024  # 850MB
        mock_torch.cuda.memory_reserved.return_value = 1000 * 1024 * 1024  # 1000MB

        def import_side_effect(name, *args, **kwargs):
            if name == "torch":
                return mock_torch
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_side_effect

        monitor = HardwareMonitor(enable_gpu_monitoring=True)
        snapshot = monitor.capture_snapshot(step=0)

        assert snapshot.memory_efficiency is not None
        assert abs(snapshot.memory_efficiency - 0.85) < 0.001  # 850/1000 = 0.85

    @patch("builtins.__import__")
    def test_capture_snapshot_no_gpu(self, mock_import) -> None:
        """Test snapshot capture when GPU is not available."""
        # Create a mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        def import_side_effect(name, *args, **kwargs):
            if name == "torch":
                return mock_torch
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_side_effect

        monitor = HardwareMonitor(enable_gpu_monitoring=True)
        snapshot = monitor.capture_snapshot(step=0)

        assert snapshot.memory_efficiency is None

    def test_capture_snapshot_gpu_disabled(self) -> None:
        """Test snapshot capture when GPU monitoring is disabled."""
        monitor = HardwareMonitor(enable_gpu_monitoring=False)
        snapshot = monitor.capture_snapshot(step=0)

        assert snapshot.memory_efficiency is None

    def test_get_latest_snapshot(self) -> None:
        """Test getting the latest snapshot."""
        monitor = HardwareMonitor()

        # No snapshots initially
        assert monitor.get_latest_snapshot() is None

        # Add snapshots
        snapshot1 = monitor.capture_snapshot(step=0)
        assert monitor.get_latest_snapshot() == snapshot1

        snapshot2 = monitor.capture_snapshot(step=1)
        assert monitor.get_latest_snapshot() == snapshot2

    def test_get_average_metrics_empty(self) -> None:
        """Test getting average metrics when no snapshots exist."""
        monitor = HardwareMonitor()
        averages = monitor.get_average_metrics()
        assert averages == {}

    def test_get_average_metrics_with_data(self) -> None:
        """Test getting average metrics with snapshot data."""
        monitor = HardwareMonitor()

        # Create snapshots with known values
        snapshot1 = HardwareSnapshot(
            step=0, wall_time=1.0, tokens_per_second=1000.0, samples_per_second=50.0, memory_efficiency=0.8
        )
        snapshot2 = HardwareSnapshot(
            step=1, wall_time=2.0, tokens_per_second=1200.0, samples_per_second=60.0, memory_efficiency=0.9
        )

        monitor.hardware_snapshots = [snapshot1, snapshot2]

        averages = monitor.get_average_metrics()

        assert abs(averages["avg_tokens_per_second"] - 1100.0) < 0.001  # (1000+1200)/2
        assert abs(averages["avg_samples_per_second"] - 55.0) < 0.001  # (50+60)/2
        assert abs(averages["avg_memory_efficiency"] - 0.85) < 0.001  # (0.8+0.9)/2

    def test_get_average_metrics_partial_data(self) -> None:
        """Test getting average metrics with partial data."""
        monitor = HardwareMonitor()

        # Create snapshots with some None values
        snapshot1 = HardwareSnapshot(
            step=0, wall_time=1.0, tokens_per_second=1000.0, samples_per_second=None, memory_efficiency=0.8
        )
        snapshot2 = HardwareSnapshot(
            step=1, wall_time=2.0, tokens_per_second=None, samples_per_second=60.0, memory_efficiency=None
        )

        monitor.hardware_snapshots = [snapshot1, snapshot2]

        averages = monitor.get_average_metrics()

        assert abs(averages["avg_tokens_per_second"] - 1000.0) < 0.001  # Only one valid value
        assert abs(averages["avg_samples_per_second"] - 60.0) < 0.001  # Only one valid value
        assert abs(averages["avg_memory_efficiency"] - 0.8) < 0.001  # Only one valid value


class TestExperimentConfig:
    """Test the ExperimentConfig dataclass."""

    def test_config_creation_defaults(self) -> None:
        """Test ExperimentConfig creation with defaults."""
        config = ExperimentConfig()
        assert config.vocab_size is None
        assert config.learning_rate is None
        assert config.custom == {}

    def test_config_creation_with_values(self) -> None:
        """Test ExperimentConfig creation with specific values."""
        config = ExperimentConfig(
            vocab_size=50000,
            context_length=1024,
            d_model=768,
            learning_rate=1e-4,
            batch_size=32,
            custom={"special_param": "test_value"},
        )
        assert config.vocab_size == 50000
        assert config.context_length == 1024
        assert config.d_model == 768
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.custom["special_param"] == "test_value"

    def test_config_update_existing_field(self) -> None:
        """Test updating existing configuration fields."""
        config = ExperimentConfig(learning_rate=1e-3)
        config.update(learning_rate=5e-4, batch_size=64)

        assert config.learning_rate == 5e-4
        assert config.batch_size == 64

    def test_config_update_custom_field(self) -> None:
        """Test updating custom configuration fields."""
        config = ExperimentConfig()
        config.update(custom_param="value", another_param=42)

        assert config.custom["custom_param"] == "value"
        assert config.custom["another_param"] == 42

    def test_config_update_mixed_fields(self) -> None:
        """Test updating both standard and custom fields."""
        config = ExperimentConfig()
        config.update(learning_rate=1e-4, vocab_size=30000, custom_optimizer="adamw_custom")

        assert config.learning_rate == 1e-4
        assert config.vocab_size == 30000
        assert config.custom["custom_optimizer"] == "adamw_custom"


class TestExperimentMetadata:
    """Test the ExperimentMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Test basic metadata creation."""
        start_time = datetime.now()
        metadata = ExperimentMetadata(
            experiment_id="test_exp_001", name="Test Experiment", description="A test experiment", start_time=start_time
        )

        assert metadata.experiment_id == "test_exp_001"
        assert metadata.name == "Test Experiment"
        assert metadata.description == "A test experiment"
        assert metadata.start_time == start_time
        assert metadata.end_time is None
        assert metadata.status == "running"
        assert metadata.tags == []
        assert metadata.notes == ""

    def test_metadata_with_optional_fields(self) -> None:
        """Test metadata creation with optional fields."""
        start_time = datetime.now()
        end_time = datetime.now()
        metadata = ExperimentMetadata(
            experiment_id="test_exp_002",
            name="Test Experiment 2",
            description="Another test experiment",
            start_time=start_time,
            end_time=end_time,
            status="completed",
            tags=["test", "experiment"],
            notes="Initial notes",
        )

        assert metadata.end_time == end_time
        assert metadata.status == "completed"
        assert metadata.tags == ["test", "experiment"]
        assert metadata.notes == "Initial notes"


class TestExperimentLogger:
    """Test the ExperimentLogger class."""

    def test_logger_creation_basic(self) -> None:
        """Test basic logger creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            assert logger.experiment_name == "test_experiment"
            assert logger.description == ""
            assert logger.use_wandb is False
            assert logger.experiment_id.startswith("exp_")
            assert len(logger.metrics) == 0
            assert logger.current_step == 0

    def test_logger_creation_with_params(self) -> None:
        """Test logger creation with custom parameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="custom_experiment",
                experiment_id="custom_exp_001",
                description="Custom test experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                auto_save_interval=50,
            )

            assert logger.experiment_name == "custom_experiment"
            assert logger.experiment_id == "custom_exp_001"
            assert logger.description == "Custom test experiment"
            assert logger.auto_save_interval == 50

    def test_logger_directory_creation(self) -> None:
        """Test that logger creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "experiments"
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=log_dir, use_wandb=False)

            assert logger.experiment_dir.exists()
            assert logger.experiment_dir.is_dir()
            assert logger.experiment_dir.name == logger.experiment_id

    def test_generate_experiment_id(self) -> None:
        """Test experiment ID generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            exp_id = logger.experiment_id
            assert exp_id.startswith("exp_")
            assert len(exp_id) > 4  # Should have timestamp

    def test_update_config(self) -> None:
        """Test configuration updates."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            logger.update_config(learning_rate=1e-4, batch_size=32, vocab_size=50000, custom_param="test_value")

            assert logger.config.learning_rate == 1e-4
            assert logger.config.batch_size == 32
            assert logger.config.vocab_size == 50000
            assert logger.config.custom["custom_param"] == "test_value"

    def test_log_single_metric(self) -> None:
        """Test logging a single metric."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            logger.log_metric("train_loss", 0.5, step=100, epoch=1)

            assert "train_loss" in logger.metrics
            assert len(logger.metrics["train_loss"]) == 1

            point = logger.metrics["train_loss"][0]
            assert point.step == 100
            assert point.value == 0.5
            assert point.epoch == 1
            assert point.wall_time > 0

    def test_log_metric_auto_step(self) -> None:
        """Test logging metric with automatic step increment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            logger.log_metric("train_loss", 0.5)
            logger.log_metric("train_loss", 0.4)

            assert len(logger.metrics["train_loss"]) == 2
            assert logger.metrics["train_loss"][0].step == 0
            assert logger.metrics["train_loss"][1].step == 0  # Should use current_step

    def test_log_multiple_metrics(self) -> None:
        """Test logging multiple metrics at once."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            metrics = {"train_loss": 0.5, "val_loss": 0.6, "learning_rate": 1e-4}
            logger.log_metrics(metrics, step=100)

            for metric_name, expected_value in metrics.items():
                assert metric_name in logger.metrics
                assert len(logger.metrics[metric_name]) == 1
                assert logger.metrics[metric_name][0].value == expected_value
                assert logger.metrics[metric_name][0].step == 100

    def test_log_hyperparameters(self) -> None:
        """Test logging hyperparameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            logger.log_hyperparameters(learning_rate=1e-4, batch_size=32, model_type="transformer")

            assert logger.config.learning_rate == 1e-4
            assert logger.config.batch_size == 32
            assert logger.config.custom["model_type"] == "transformer"

    def test_add_tag(self) -> None:
        """Test adding tags to experiment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            logger.add_tag("baseline")
            logger.add_tag("experiment_1")
            logger.add_tag("baseline")  # Should not duplicate

            assert "baseline" in logger.metadata.tags
            assert "experiment_1" in logger.metadata.tags
            assert len(logger.metadata.tags) == 2

    def test_add_note(self) -> None:
        """Test adding notes to experiment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            logger.add_note("First observation")
            logger.add_note("Second observation")

            notes = logger.metadata.notes
            assert "First observation" in notes
            assert "Second observation" in notes
            assert notes.count("\n") >= 2  # Should have timestamps

    def test_get_metric_history(self) -> None:
        """Test retrieving metric history."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            logger.log_metric("train_loss", 0.5, step=1)
            logger.log_metric("train_loss", 0.4, step=2)
            logger.log_metric("train_loss", 0.3, step=3)

            history = logger.get_metric_history("train_loss")
            assert len(history) == 3
            assert [point.value for point in history] == [0.5, 0.4, 0.3]

            # Test non-existent metric
            empty_history = logger.get_metric_history("nonexistent_metric")
            assert len(empty_history) == 0

    def test_get_latest_metric(self) -> None:
        """Test retrieving latest metric value."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            logger.log_metric("train_loss", 0.5, step=1)
            logger.log_metric("train_loss", 0.4, step=2)
            logger.log_metric("train_loss", 0.3, step=3)

            latest = logger.get_latest_metric("train_loss")
            assert latest == 0.3

            # Test non-existent metric
            no_metric = logger.get_latest_metric("nonexistent_metric")
            assert no_metric is None

    def test_mark_completed(self) -> None:
        """Test marking experiment as completed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            logger.mark_completed()

            assert logger.metadata.status == "completed"
            assert logger.metadata.end_time is not None

    def test_mark_failed(self) -> None:
        """Test marking experiment as failed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            error_msg = "Out of memory error"
            logger.mark_failed(error_msg)

            assert logger.metadata.status == "failed"
            assert logger.metadata.end_time is not None
            assert error_msg in logger.metadata.notes

    def test_auto_save_interval(self) -> None:
        """Test automatic saving based on step interval."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False, auto_save_interval=2
            )

            # Log metrics that should trigger auto-save
            logger.log_metric("train_loss", 0.5, step=1)
            logger.log_metric("train_loss", 0.4, step=3)  # Should trigger save

            # Check that files were created
            config_file = logger.experiment_dir / "config.json"
            metrics_file = logger.experiment_dir / "metrics.json"
            metadata_file = logger.experiment_dir / "metadata.json"

            assert config_file.exists()
            assert metrics_file.exists()
            assert metadata_file.exists()

    def test_save_and_load_experiment(self) -> None:
        """Test saving and loading experiment data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and configure logger
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                experiment_id="test_exp_001",
                description="Test experiment for save/load",
                log_dir=tmp_dir,
                use_wandb=False,
            )

            # Add some data
            logger.update_config(learning_rate=1e-4, batch_size=32)
            logger.log_metric("train_loss", 0.5, step=1)
            logger.log_metric("val_loss", 0.6, step=1)
            logger.add_tag("test")
            logger.add_note("Test note")

            # Save explicitly
            logger.save()

            # Load the experiment
            loaded_logger = ExperimentLogger.load("test_exp_001", tmp_dir)

            # Verify loaded data
            assert loaded_logger.experiment_name == "test_experiment"
            assert loaded_logger.experiment_id == "test_exp_001"
            assert loaded_logger.description == "Test experiment for save/load"
            assert loaded_logger.config.learning_rate == 1e-4
            assert loaded_logger.config.batch_size == 32

            # Check metrics
            assert "train_loss" in loaded_logger.metrics
            assert "val_loss" in loaded_logger.metrics
            assert loaded_logger.get_latest_metric("train_loss") == 0.5
            assert loaded_logger.get_latest_metric("val_loss") == 0.6

            # Check metadata
            assert "test" in loaded_logger.metadata.tags
            assert "Test note" in loaded_logger.metadata.notes

    def test_load_nonexistent_experiment(self) -> None:
        """Test loading a non-existent experiment raises error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(FileNotFoundError):
                ExperimentLogger.load("nonexistent_exp", tmp_dir)

    def test_hardware_monitoring_enabled(self) -> None:
        """Test that hardware monitoring is enabled by default."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=True,
            )

            assert logger.enable_hardware_monitoring is True
            assert logger.hardware_monitor is not None
            assert isinstance(logger.hardware_monitor, HardwareMonitor)

    def test_hardware_monitoring_disabled(self) -> None:
        """Test that hardware monitoring can be disabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=False,
            )

            assert logger.enable_hardware_monitoring is False
            assert logger.hardware_monitor is None

    def test_log_hardware_metrics_disabled(self) -> None:
        """Test logging hardware metrics when monitoring is disabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=False,
            )

            # Should not raise an error
            logger.log_hardware_metrics(step=1, tokens_processed=1000, samples_processed=32)

            # Should not have created any snapshots
            summary = logger.get_hardware_utilization_summary()
            assert summary["hardware_monitoring_enabled"] is False

    def test_log_hardware_metrics_enabled(self) -> None:
        """Test logging hardware metrics when monitoring is enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=True,
            )

            logger.log_hardware_metrics(step=1, tokens_processed=1000, samples_processed=32)

            assert logger.hardware_monitor is not None
            assert len(logger.hardware_monitor.hardware_snapshots) == 1

            snapshot = logger.hardware_monitor.hardware_snapshots[0]
            assert snapshot.step == 1

    @patch("cs336_basics.experiments.exp_logging.wandb")
    def test_log_hardware_metrics_with_wandb(self, mock_wandb) -> None:
        """Test logging hardware metrics to W&B."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=True,
                enable_hardware_monitoring=True,
            )

            # Mock throughput calculation by adding a previous snapshot
            logger.hardware_monitor._last_tokens_processed = 500
            logger.hardware_monitor._last_samples_processed = 16
            logger.hardware_monitor._last_snapshot_time = time.time() - 1.0

            logger.log_hardware_metrics(step=1, tokens_processed=1000, samples_processed=32)

            # Check that wandb.log was called
            mock_run.log.assert_called()
            call_args = mock_run.log.call_args[0][0]  # Get the logged dictionary

            # Should include performance metrics
            assert "tokens_per_second" in call_args or "samples_per_second" in call_args

    def test_get_hardware_utilization_summary_disabled(self) -> None:
        """Test hardware utilization summary when monitoring is disabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=False,
            )

            summary = logger.get_hardware_utilization_summary()
            assert summary["hardware_monitoring_enabled"] is False
            assert "average_metrics" not in summary

    def test_get_hardware_utilization_summary_enabled(self) -> None:
        """Test hardware utilization summary when monitoring is enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=True,
            )

            # Add some hardware data
            logger.log_hardware_metrics(step=1, tokens_processed=1000, samples_processed=32)
            time.sleep(0.1)
            logger.log_hardware_metrics(step=2, tokens_processed=2000, samples_processed=64)

            summary = logger.get_hardware_utilization_summary()
            assert summary["hardware_monitoring_enabled"] is True
            assert summary["total_snapshots"] == 2
            assert "average_metrics" in summary
            assert "latest_metrics" in summary
            assert "wandb_system_metrics" in summary

    def test_save_and_load_with_hardware_monitoring(self) -> None:
        """Test saving and loading experiment with hardware monitoring data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create logger with hardware monitoring
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                experiment_id="test_exp_hw_001",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=True,
            )

            # Add hardware metrics
            logger.log_hardware_metrics(step=1, tokens_processed=1000, samples_processed=32)
            logger.log_hardware_metrics(step=2, tokens_processed=2000, samples_processed=64)

            # Save
            logger.save()

            # Load
            loaded_logger = ExperimentLogger.load("test_exp_hw_001", tmp_dir)

            # Verify hardware monitoring data was loaded
            assert loaded_logger.enable_hardware_monitoring is True
            assert loaded_logger.hardware_monitor is not None
            assert len(loaded_logger.hardware_monitor.hardware_snapshots) == 2

            summary = loaded_logger.get_hardware_utilization_summary()
            assert summary["hardware_monitoring_enabled"] is True
            assert summary["total_snapshots"] == 2

    def test_generate_summary_report_with_hardware(self) -> None:
        """Test generation of summary report with hardware metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                description="Test experiment with hardware monitoring",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=True,
            )

            # Add some metrics and hardware data - need multiple snapshots to calculate averages
            logger.log_metric("train_loss", 0.5, step=1)
            logger.log_hardware_metrics(step=1, tokens_processed=1000, samples_processed=32)
            time.sleep(0.1)  # Small delay to ensure time difference
            logger.log_hardware_metrics(step=2, tokens_processed=2000, samples_processed=64)

            report = logger.generate_summary_report()

            assert "Hardware Utilization Summary:" in report
            # Check for either the average metrics section or just the latest metrics
            has_metrics = (
                "Average Performance Metrics:" in report
                or "Latest Metrics" in report
                or "automatically logged by W&B" in report
            )
            assert has_metrics, f"Expected hardware metrics in report, got: {report}"

    def test_generate_summary_report(self) -> None:
        """Test generation of summary report."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                description="Test experiment for summary",
                log_dir=tmp_dir,
                use_wandb=False,
            )

            # Add some data
            logger.update_config(learning_rate=1e-4, batch_size=32)
            logger.log_metric("train_loss", 0.5, step=100)
            logger.log_metric("val_loss", 0.6, step=100)
            logger.add_tag("baseline")
            logger.add_note("Experiment completed successfully")

            report = logger.generate_summary_report()

            # Check that key information is in the report
            assert "test_experiment" in report
            assert "Test experiment for summary" in report
            assert "learning_rate: 1e-4" in report or "learning_rate: 0.0001" in report
            assert "batch_size: 32" in report
            assert "train_loss: 0.5" in report
            assert "val_loss: 0.6" in report
            assert "baseline" in report
            assert "Experiment completed successfully" in report

    def test_plot_metrics(self) -> None:
        """Test plotting metrics functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            # Add time-series data
            for step in range(10):
                logger.log_metric("train_loss", 1.0 - step * 0.1, step=step)
                logger.log_metric("val_loss", 1.1 - step * 0.08, step=step)
                time.sleep(0.01)  # Small delay to create wall time differences

            # Test plotting by step
            fig = logger.plot_metrics(["train_loss", "val_loss"], x_axis="step")
            assert fig is not None

            # Test plotting by time
            fig_time = logger.plot_metrics(["train_loss", "val_loss"], x_axis="time")
            assert fig_time is not None

            # Test with save path
            save_path = Path(tmp_dir) / "test_plot.png"
            fig_saved = logger.plot_metrics(["train_loss", "val_loss"], x_axis="step", save_path=save_path)
            assert fig_saved is not None
            assert save_path.exists()

    def test_plot_metrics_invalid_axis(self) -> None:
        """Test plotting with invalid x-axis raises error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            logger.log_metric("train_loss", 0.5, step=1)

            with pytest.raises(ValueError, match="Invalid x_axis"):
                logger.plot_metrics(["train_loss"], x_axis="invalid")

    @patch("wandb.init")
    def test_wandb_integration_initialization(self, mock_wandb_init: MagicMock) -> None:
        """Test Weights & Biases integration initialization."""
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=True,
                wandb_project="test_project",
                wandb_config={"test": "config"},
            )

            mock_wandb_init.assert_called_once_with(
                project="test_project",
                name="test_experiment",
                id=logger.experiment_id,
                config={"test": "config"},
                resume="allow",
            )
            assert logger.wandb_run == mock_run

    @patch("wandb.init")
    def test_wandb_integration_default_project(self, mock_wandb_init: MagicMock) -> None:
        """Test Weights & Biases integration with default project."""
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=True)

            mock_wandb_init.assert_called_once_with(
                project="cs336-assignment1", name="test_experiment", id=logger.experiment_id, config={}, resume="allow"
            )

    @patch("wandb.init")
    def test_wandb_integration_failed_init(self, mock_wandb_init: MagicMock) -> None:
        """Test handling of W&B initialization failure."""
        mock_wandb_init.side_effect = Exception("W&B connection failed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=True)

                assert len(w) == 1
                assert "Failed to initialize W&B" in str(w[0].message)
                assert logger.use_wandb is False
                assert logger.wandb_run is None

    @patch("wandb.init")
    def test_wandb_metric_logging(self, mock_wandb_init: MagicMock) -> None:
        """Test metric logging to W&B."""
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=True)

            logger.log_metric("train_loss", 0.5, step=100, epoch=5)

            mock_run.log.assert_called_once_with({"train_loss": 0.5, "epoch": 5}, step=100)

    @patch("wandb.init")
    def test_wandb_config_update(self, mock_wandb_init: MagicMock) -> None:
        """Test configuration updates to W&B."""
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=True)

            logger.update_config(learning_rate=1e-4, batch_size=32)

            mock_run.config.update.assert_called_once_with({"learning_rate": 1e-4, "batch_size": 32})

    @patch("wandb.init")
    def test_wandb_experiment_completion(self, mock_wandb_init: MagicMock) -> None:
        """Test W&B run finishing on experiment completion."""
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=True)

            logger.mark_completed()
            mock_run.finish.assert_called_once()

    @patch("wandb.init")
    def test_wandb_experiment_failure(self, mock_wandb_init: MagicMock) -> None:
        """Test W&B run finishing with error code on experiment failure."""
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=True)

            logger.mark_failed("Test error")
            mock_run.finish.assert_called_once_with(exit_code=1)


class TestTrainingIntegrator:
    """Test the TrainingIntegrator helper class."""

    def test_training_integrator_creation(self) -> None:
        """Test TrainingIntegrator creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)
            integrator = TrainingIntegrator(logger)

            assert integrator.logger == logger
            assert hasattr(integrator, "step_start_time")

    def test_log_training_step(self) -> None:
        """Test logging training step metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)
            integrator = TrainingIntegrator(logger)

            integrator.log_training_step(step=100, train_loss=0.5, learning_rate=1e-4, gradient_norm=0.1)

            assert logger.get_latest_metric("train_loss") == 0.5
            assert logger.get_latest_metric("learning_rate") == 1e-4
            assert logger.get_latest_metric("gradient_norm") == 0.1

            # Check that all metrics have the same step
            for metric in ["train_loss", "learning_rate", "gradient_norm"]:
                history = logger.get_metric_history(metric)
                assert len(history) == 1
                assert history[0].step == 100

    def test_log_validation_step(self) -> None:
        """Test logging validation step metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)
            integrator = TrainingIntegrator(logger)

            integrator.log_validation_step(step=100, val_loss=0.6, perplexity=1.8, accuracy=0.85)

            assert logger.get_latest_metric("val_loss") == 0.6
            assert logger.get_latest_metric("val_perplexity") == 1.8
            assert logger.get_latest_metric("accuracy") == 0.85

    def test_log_validation_step_without_perplexity(self) -> None:
        """Test logging validation step without perplexity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)
            integrator = TrainingIntegrator(logger)

            integrator.log_validation_step(step=100, val_loss=0.6, accuracy=0.85)

            assert logger.get_latest_metric("val_loss") == 0.6
            assert logger.get_latest_metric("accuracy") == 0.85
            assert logger.get_latest_metric("val_perplexity") is None

    def test_training_integrator_creation_with_hardware_monitoring(self) -> None:
        """Test TrainingIntegrator creation with hardware monitoring."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=True,
            )
            integrator = TrainingIntegrator(logger, hardware_log_interval=10)

            assert integrator.logger == logger
            assert integrator.hardware_log_interval == 10
            assert integrator.total_tokens_processed == 0
            assert integrator.total_samples_processed == 0

    def test_log_training_step_with_throughput_tracking(self) -> None:
        """Test logging training step with throughput tracking."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=True,
            )
            integrator = TrainingIntegrator(logger, hardware_log_interval=2)

            # First step
            integrator.log_training_step(
                step=1,
                train_loss=0.5,
                learning_rate=1e-4,
                tokens_processed=1000,
                samples_processed=32,
            )

            assert integrator.total_tokens_processed == 1000
            assert integrator.total_samples_processed == 32

            # Second step - should trigger hardware logging (interval=2)
            integrator.log_training_step(
                step=2,
                train_loss=0.4,
                learning_rate=1e-4,
                tokens_processed=1000,
                samples_processed=32,
            )

            assert integrator.total_tokens_processed == 2000
            assert integrator.total_samples_processed == 64

            # Check that hardware metrics were logged
            assert logger.hardware_monitor is not None
            assert len(logger.hardware_monitor.hardware_snapshots) >= 1

    def test_log_training_step_hardware_interval(self) -> None:
        """Test that hardware metrics are logged at correct intervals."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=True,
            )
            integrator = TrainingIntegrator(logger, hardware_log_interval=3)

            # Steps 1 and 2 should not trigger hardware logging
            integrator.log_training_step(
                step=1, train_loss=0.5, learning_rate=1e-4, tokens_processed=1000, samples_processed=32
            )
            integrator.log_training_step(
                step=2, train_loss=0.4, learning_rate=1e-4, tokens_processed=1000, samples_processed=32
            )

            assert len(logger.hardware_monitor.hardware_snapshots) == 0

            # Step 3 should trigger hardware logging (interval=3)
            integrator.log_training_step(
                step=3, train_loss=0.3, learning_rate=1e-4, tokens_processed=1000, samples_processed=32
            )

            assert len(logger.hardware_monitor.hardware_snapshots) == 1

    def test_log_training_step_without_throughput(self) -> None:
        """Test logging training step without throughput tracking."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                enable_hardware_monitoring=True,
            )
            integrator = TrainingIntegrator(logger, hardware_log_interval=1)

            # Should work without tokens_processed/samples_processed
            integrator.log_training_step(step=1, train_loss=0.5, learning_rate=1e-4)

            assert integrator.total_tokens_processed == 0
            assert integrator.total_samples_processed == 0

            # Hardware metrics should still be logged but without throughput data
            assert len(logger.hardware_monitor.hardware_snapshots) == 1
            snapshot = logger.hardware_monitor.hardware_snapshots[0]
            assert snapshot.step == 1

    def test_start_epoch(self) -> None:
        """Test logging epoch start."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)
            integrator = TrainingIntegrator(logger)

            integrator.start_epoch(5)

            assert logger.get_latest_metric("epoch") == 5

    def test_log_step_time(self) -> None:
        """Test logging step timing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)
            integrator = TrainingIntegrator(logger)

            # Simulate some work
            time.sleep(0.01)
            integrator.log_step_time(100)

            step_time = logger.get_latest_metric("step_time")
            assert step_time is not None
            assert step_time > 0

    def test_integrated_training_simulation(self) -> None:
        """Test integrated training loop simulation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)
            integrator = TrainingIntegrator(logger)

            # Simulate training steps
            for step in range(5):
                integrator.log_training_step(step=step, train_loss=1.0 - step * 0.1, learning_rate=1e-4)

                if step % 2 == 0:  # Log validation every other step
                    integrator.log_validation_step(step=step, val_loss=1.1 - step * 0.08, perplexity=3.0 - step * 0.2)

                integrator.log_step_time(step)

            # Check that metrics were logged correctly
            train_history = logger.get_metric_history("train_loss")
            val_history = logger.get_metric_history("val_loss")
            time_history = logger.get_metric_history("step_time")

            assert len(train_history) == 5
            assert len(val_history) == 3  # Every other step
            assert len(time_history) == 5

            # Check decreasing loss trend
            train_losses = [point.value for point in train_history]
            assert train_losses == [1.0, 0.9, 0.8, 0.7, 0.6]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_experiment_logger(self) -> None:
        """Test convenience function for creating experiment logger."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("cs336_basics.experiments.exp_logging.ExperimentLogger") as mock_logger_cls:
                mock_logger = MagicMock()
                mock_logger_cls.return_value = mock_logger

                result = create_experiment_logger(
                    name="test_experiment", description="Test description", learning_rate=1e-4, batch_size=32
                )

                mock_logger_cls.assert_called_once_with(
                    experiment_name="test_experiment", description="Test description", use_wandb=True
                )
                mock_logger.update_config.assert_called_once_with(learning_rate=1e-4, batch_size=32)
                assert result == mock_logger

    def test_compare_experiments(self) -> None:
        """Test experiment comparison functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create two experiments
            logger1 = ExperimentLogger(
                experiment_name="experiment_1", experiment_id="exp_001", log_dir=tmp_dir, use_wandb=False
            )
            logger2 = ExperimentLogger(
                experiment_name="experiment_2", experiment_id="exp_002", log_dir=tmp_dir, use_wandb=False
            )

            # Add different loss curves
            for step in range(10):
                logger1.log_metric("train_loss", 1.0 - step * 0.1, step=step)
                logger2.log_metric("train_loss", 1.0 - step * 0.05, step=step)

            # Save experiments
            logger1.save()
            logger2.save()

            # Compare experiments
            fig = compare_experiments(
                experiment_ids=["exp_001", "exp_002"], metric_name="train_loss", log_dir=tmp_dir, x_axis="step"
            )

            assert fig is not None

    def test_compare_experiments_with_missing_experiment(self) -> None:
        """Test experiment comparison with missing experiment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="experiment_1", experiment_id="exp_001", log_dir=tmp_dir, use_wandb=False
            )
            logger.log_metric("train_loss", 0.5, step=1)
            logger.save()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fig = compare_experiments(
                    experiment_ids=["exp_001", "nonexistent"], metric_name="train_loss", log_dir=tmp_dir
                )

                assert len(w) == 1
                assert "Failed to load experiment nonexistent" in str(w[0].message)
                assert fig is not None

    def test_compare_experiments_invalid_axis(self) -> None:
        """Test experiment comparison with invalid x-axis."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="experiment_1", experiment_id="exp_001", log_dir=tmp_dir, use_wandb=False
            )
            logger.log_metric("train_loss", 0.5, step=1)
            logger.save()

            with pytest.raises(ValueError, match="Invalid x_axis"):
                compare_experiments(
                    experiment_ids=["exp_001"], metric_name="train_loss", log_dir=tmp_dir, x_axis="invalid"
                )


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_metric_operations(self) -> None:
        """Test operations on empty metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            # Test getting history for non-existent metric
            history = logger.get_metric_history("nonexistent")
            assert history == []

            # Test getting latest value for non-existent metric
            latest = logger.get_latest_metric("nonexistent")
            assert latest is None

            # Test plotting non-existent metrics (should handle gracefully)
            fig = logger.plot_metrics(["nonexistent_metric"])
            assert fig is not None

    def test_concurrent_metric_logging(self) -> None:
        """Test concurrent metric logging doesn't cause issues."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            # Log metrics with the same step number
            logger.log_metric("metric1", 0.5, step=100)
            logger.log_metric("metric2", 0.6, step=100)
            logger.log_metric("metric1", 0.4, step=100)  # Same metric, same step

            history1 = logger.get_metric_history("metric1")
            history2 = logger.get_metric_history("metric2")

            assert len(history1) == 2  # Two entries for metric1
            assert len(history2) == 1  # One entry for metric2

    def test_large_metric_values(self) -> None:
        """Test handling of large metric values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            # Test very large and very small values
            large_value = 1e10
            small_value = 1e-10

            logger.log_metric("large_metric", large_value, step=1)
            logger.log_metric("small_metric", small_value, step=1)

            assert logger.get_latest_metric("large_metric") == large_value
            assert logger.get_latest_metric("small_metric") == small_value

    def test_special_characters_in_names(self) -> None:
        """Test handling of special characters in experiment names."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test with special characters that are valid in filenames
            logger = ExperimentLogger(experiment_name="test_experiment_v1.0-beta", log_dir=tmp_dir, use_wandb=False)

            logger.log_metric("train_loss", 0.5, step=1)
            logger.save()

            # Should be able to save and load successfully
            loaded = ExperimentLogger.load(logger.experiment_id, tmp_dir)
            assert loaded.experiment_name == "test_experiment_v1.0-beta"

    def test_datetime_serialization(self) -> None:
        """Test proper datetime serialization and deserialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            original_start_time = logger.metadata.start_time
            logger.mark_completed()
            original_end_time = logger.metadata.end_time

            logger.save()
            loaded = ExperimentLogger.load(logger.experiment_id, tmp_dir)

            # Check that datetime objects are properly restored
            assert isinstance(loaded.metadata.start_time, datetime)
            assert isinstance(loaded.metadata.end_time, datetime)
            assert loaded.metadata.start_time == original_start_time
            assert loaded.metadata.end_time == original_end_time

    def test_config_with_complex_types(self) -> None:
        """Test configuration with complex data types."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            # Test with various data types
            logger.update_config(
                string_param="test",
                int_param=42,
                float_param=3.14,
                bool_param=True,
                list_param=[1, 2, 3],
                dict_param={"nested": "value"},
            )

            logger.save()
            loaded = ExperimentLogger.load(logger.experiment_id, tmp_dir)

            # Verify all types are preserved
            assert loaded.config.custom["string_param"] == "test"
            assert loaded.config.custom["int_param"] == 42
            assert loaded.config.custom["float_param"] == 3.14
            assert loaded.config.custom["bool_param"] is True
            assert loaded.config.custom["list_param"] == [1, 2, 3]
            assert loaded.config.custom["dict_param"] == {"nested": "value"}


class TestPerformanceAndScalability:
    """Test performance and scalability aspects."""

    def test_large_number_of_metrics(self) -> None:
        """Test handling large numbers of metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(
                experiment_name="test_experiment",
                log_dir=tmp_dir,
                use_wandb=False,
                auto_save_interval=10000,  # Disable auto-save for performance
            )

            # Log many metrics
            num_metrics = 1000
            num_steps = 100

            for step in range(num_steps):
                for metric_idx in range(num_metrics):
                    metric_name = f"metric_{metric_idx}"
                    value = np.random.random()
                    logger.log_metric(metric_name, value, step=step)

            # Verify all metrics were logged
            assert len(logger.metrics) == num_metrics
            for metric_idx in range(num_metrics):
                metric_name = f"metric_{metric_idx}"
                history = logger.get_metric_history(metric_name)
                assert len(history) == num_steps

    def test_metric_retrieval_performance(self) -> None:
        """Test performance of metric retrieval operations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(experiment_name="test_experiment", log_dir=tmp_dir, use_wandb=False)

            # Log a long time series
            num_steps = 10000
            for step in range(num_steps):
                logger.log_metric("train_loss", 1.0 - step * 0.0001, step=step)

            # Test retrieval performance (should be fast)
            import time

            start_time = time.time()

            history = logger.get_metric_history("train_loss")
            latest = logger.get_latest_metric("train_loss")

            end_time = time.time()
            retrieval_time = end_time - start_time

            # Should be very fast (less than 1 second for 10k points)
            assert retrieval_time < 1.0
            assert len(history) == num_steps
            assert latest == 1.0 - (num_steps - 1) * 0.0001


if __name__ == "__main__":
    pytest.main([__file__])
