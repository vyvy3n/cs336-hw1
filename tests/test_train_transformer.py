"""
Tests for the train_transformer.py training script.

This module contains comprehensive tests for all components of the transformer training pipeline,
including configuration validation, logging, data loading, and training orchestration.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from cs336_basics.scripts.train_transformer import (
    DataLoader,
    Trainer,
    TrainingConfig,
    TrainingLogger,
    load_config,
    save_config,
)


class TestTrainingConfig:
    """Test the TrainingConfig dataclass."""

    def test_training_config_initialization(self) -> None:
        """Test that TrainingConfig initializes with valid parameters."""
        config = TrainingConfig(
            train_data_path="test_data.npy",
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            max_steps=1000,
            batch_size=32,
            learning_rate=3e-4,
        )

        assert config.train_data_path == "test_data.npy"
        assert config.vocab_size == 10000
        assert config.context_length == 256
        assert config.d_model == 512
        assert config.num_layers == 4
        assert config.num_heads == 16
        assert config.max_steps == 1000
        assert config.batch_size == 32
        assert config.learning_rate == 3e-4

    def test_training_config_defaults(self) -> None:
        """Test that TrainingConfig uses appropriate defaults."""
        config = TrainingConfig(
            train_data_path="test_data.npy",
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            max_steps=1000,
            batch_size=32,
            learning_rate=3e-4,
        )

        assert config.val_data_path is None
        assert config.d_ff == 1344
        assert config.rope_theta == 10000.0
        assert config.eps == 1e-5
        assert config.min_learning_rate == 3e-5
        assert config.warmup_steps == 1000
        assert config.weight_decay == 0.01
        assert config.beta1 == 0.9
        assert config.beta2 == 0.95
        assert config.grad_clip_norm == 1.0
        assert config.device == "cuda"
        assert config.compile_model is True

    def test_training_config_validation_positive_values(self) -> None:
        """Test that TrainingConfig validates positive values."""
        with pytest.raises(AssertionError, match="vocab_size must be positive"):
            TrainingConfig(
                train_data_path="test_data.npy",
                vocab_size=0,
                context_length=256,
                d_model=512,
                num_layers=4,
                num_heads=16,
                max_steps=1000,
                batch_size=32,
                learning_rate=3e-4,
            )

        with pytest.raises(AssertionError, match="context_length must be positive"):
            TrainingConfig(
                train_data_path="test_data.npy",
                vocab_size=10000,
                context_length=0,
                d_model=512,
                num_layers=4,
                num_heads=16,
                max_steps=1000,
                batch_size=32,
                learning_rate=3e-4,
            )

    def test_training_config_validation_d_model_divisible_by_num_heads(self) -> None:
        """Test that d_model must be divisible by num_heads."""
        with pytest.raises(AssertionError, match="d_model must be divisible by num_heads"):
            TrainingConfig(
                train_data_path="test_data.npy",
                vocab_size=10000,
                context_length=256,
                d_model=513,
                num_layers=4,
                num_heads=16,
                max_steps=1000,
                batch_size=32,
                learning_rate=3e-4,
            )

    def test_training_config_validation_beta_values(self) -> None:
        """Test that beta values are in valid range [0, 1)."""
        with pytest.raises(AssertionError, match="beta1 must be in"):
            TrainingConfig(
                train_data_path="test_data.npy",
                vocab_size=10000,
                context_length=256,
                d_model=512,
                num_layers=4,
                num_heads=16,
                max_steps=1000,
                batch_size=32,
                learning_rate=3e-4,
                beta1=1.0,
            )

        with pytest.raises(AssertionError, match="beta2 must be in"):
            TrainingConfig(
                train_data_path="test_data.npy",
                vocab_size=10000,
                context_length=256,
                d_model=512,
                num_layers=4,
                num_heads=16,
                max_steps=1000,
                batch_size=32,
                learning_rate=3e-4,
                beta2=-0.1,
            )

    def test_training_config_creates_directories(self) -> None:
        """Test that TrainingConfig creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = os.path.join(tmp_dir, "checkpoints")
            log_dir = os.path.join(tmp_dir, "logs")

            config = TrainingConfig(
                train_data_path="test_data.npy",
                vocab_size=10000,
                context_length=256,
                d_model=512,
                num_layers=4,
                num_heads=16,
                max_steps=1000,
                batch_size=32,
                learning_rate=3e-4,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
            )

            assert os.path.exists(checkpoint_dir)
            assert os.path.exists(log_dir)


class TestTrainingLogger:
    """Test the TrainingLogger class."""

    def test_training_logger_initialization(self) -> None:
        """Test that TrainingLogger initializes correctly."""
        config = TrainingConfig(
            train_data_path="test_data.npy",
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            max_steps=1000,
            batch_size=32,
            learning_rate=3e-4,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.log_dir = tmp_dir
            logger = TrainingLogger(config)

            assert logger.config == config
            assert logger.logger is not None
            assert logger.tb_writer is not None
            assert logger.use_wandb is False

    def test_training_logger_wandb_initialization(self) -> None:
        """Test that TrainingLogger initializes wandb when project is specified."""
        config = TrainingConfig(
            train_data_path="test_data.npy",
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            max_steps=1000,
            batch_size=32,
            learning_rate=3e-4,
            wandb_project="test_project",
            wandb_run_name="test_run",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.log_dir = tmp_dir
            with patch("wandb.init") as mock_wandb_init:
                logger = TrainingLogger(config)

                assert logger.use_wandb is True
                mock_wandb_init.assert_called_once()

    def test_training_logger_log_step(self) -> None:
        """Test that log_step works correctly."""
        config = TrainingConfig(
            train_data_path="test_data.npy",
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            max_steps=1000,
            batch_size=32,
            learning_rate=3e-4,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.log_dir = tmp_dir
            logger = TrainingLogger(config)

            logger.tb_writer = MagicMock()

            metrics = {"loss": 1.5, "lr": 3e-4}
            logger.log_step(100, metrics)

            assert logger.tb_writer.add_scalar.call_count == 2
            logger.tb_writer.add_scalar.assert_any_call("train/loss", 1.5, 100)
            logger.tb_writer.add_scalar.assert_any_call("train/lr", 3e-4, 100)

    def test_training_logger_log_eval(self) -> None:
        """Test that log_eval works correctly."""
        config = TrainingConfig(
            train_data_path="test_data.npy",
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            max_steps=1000,
            batch_size=32,
            learning_rate=3e-4,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.log_dir = tmp_dir
            logger = TrainingLogger(config)

            logger.tb_writer = MagicMock()

            metrics = {"val_loss": 1.2, "val_perplexity": 3.32}
            logger.log_eval(500, metrics)

            assert logger.tb_writer.add_scalar.call_count == 2
            logger.tb_writer.add_scalar.assert_any_call("val/val_loss", 1.2, 500)
            logger.tb_writer.add_scalar.assert_any_call("val/val_perplexity", 3.32, 500)

    def test_training_logger_close(self) -> None:
        """Test that close method works correctly."""
        config = TrainingConfig(
            train_data_path="test_data.npy",
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            max_steps=1000,
            batch_size=32,
            learning_rate=3e-4,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.log_dir = tmp_dir
            logger = TrainingLogger(config)

            logger.tb_writer = MagicMock()

            logger.close()

            logger.tb_writer.close.assert_called_once()


class TestDataLoader:
    """Test the DataLoader class."""

    def test_data_loader_initialization(self) -> None:
        """Test that DataLoader initializes correctly."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_file:
            data = np.arange(100, dtype=np.uint16)
            data.tofile(tmp_file.name)
            tmp_file.flush()

            try:
                loader = DataLoader(data_path=tmp_file.name, batch_size=32, context_length=50, device="cpu")

                assert loader.data_path == tmp_file.name
                assert loader.batch_size == 32
                assert loader.context_length == 50
                assert loader.device == "cpu"
                assert loader.data_size == 100
                assert len(loader.data) == 100
            finally:
                os.unlink(tmp_file.name)

    def test_data_loader_too_small_dataset(self) -> None:
        """Test that DataLoader raises error for too small dataset."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_file:
            data = np.arange(10, dtype=np.uint16)
            data.tofile(tmp_file.name)
            tmp_file.flush()

            try:
                with pytest.raises(ValueError, match="Dataset too small"):
                    DataLoader(data_path=tmp_file.name, batch_size=32, context_length=128, device="cpu")
            finally:
                os.unlink(tmp_file.name)

    def test_data_loader_get_batch(self) -> None:
        """Test that get_batch returns correct shapes and values."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_file:
            data = np.arange(100, dtype=np.uint16)
            data.tofile(tmp_file.name)
            tmp_file.flush()

            try:
                loader = DataLoader(data_path=tmp_file.name, batch_size=4, context_length=8, device="cpu")

                inputs, targets = loader.get_batch()

                assert inputs.shape == (4, 8)
                assert targets.shape == (4, 8)

                assert inputs.dtype == torch.long
                assert targets.dtype == torch.long

                assert inputs.device.type == "cpu"
                assert targets.device.type == "cpu"

                torch.testing.assert_close(targets, inputs + 1)

            finally:
                os.unlink(tmp_file.name)


class TestTrainer:
    """Test the Trainer class."""

    def create_test_data(self, size: int = 1000, vocab_size: int = 100) -> str:
        """Create a test data file and return its path."""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        data = np.random.randint(0, vocab_size, size=size, dtype=np.uint16)
        data.tofile(tmp_file.name)
        tmp_file.close()
        return tmp_file.name

    def test_trainer_initialization(self) -> None:
        """Test that Trainer initializes correctly."""
        train_data_path = self.create_test_data(vocab_size=100)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = TrainingConfig(
                    train_data_path=train_data_path,
                    vocab_size=100,
                    context_length=32,
                    d_model=64,
                    num_layers=2,
                    num_heads=4,
                    max_steps=10,
                    batch_size=2,
                    learning_rate=1e-3,
                    warmup_steps=2,
                    checkpoint_dir=tmp_dir,
                    log_dir=tmp_dir,
                    device="cpu",
                    compile_model=False,
                )

                trainer = Trainer(config)

                assert trainer.config == config
                assert trainer.step == 0
                assert trainer.device.type == "cpu"
                assert trainer.model is not None
                assert trainer.optimizer is not None
                assert trainer.train_loader is not None
                assert trainer.val_loader is None

        finally:
            os.unlink(train_data_path)

    def test_trainer_with_validation_data(self) -> None:
        """Test that Trainer initializes with validation data."""
        train_data_path = self.create_test_data(vocab_size=100)
        val_data_path = self.create_test_data(500, vocab_size=100)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = TrainingConfig(
                    train_data_path=train_data_path,
                    val_data_path=val_data_path,
                    vocab_size=100,
                    context_length=32,
                    d_model=64,
                    num_layers=2,
                    num_heads=4,
                    max_steps=10,
                    batch_size=2,
                    learning_rate=1e-3,
                    warmup_steps=2,
                    checkpoint_dir=tmp_dir,
                    log_dir=tmp_dir,
                    device="cpu",
                    compile_model=False,
                )

                trainer = Trainer(config)

                assert trainer.val_loader is not None

        finally:
            os.unlink(train_data_path)
            os.unlink(val_data_path)

    def test_trainer_get_lr(self) -> None:
        """Test that get_lr returns correct learning rates."""
        train_data_path = self.create_test_data(vocab_size=100)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = TrainingConfig(
                    train_data_path=train_data_path,
                    vocab_size=100,
                    context_length=32,
                    d_model=64,
                    num_layers=2,
                    num_heads=4,
                    max_steps=100,
                    batch_size=2,
                    learning_rate=1e-3,
                    min_learning_rate=1e-4,
                    warmup_steps=10,
                    checkpoint_dir=tmp_dir,
                    log_dir=tmp_dir,
                    device="cpu",
                    compile_model=False,
                )

                trainer = Trainer(config)

                lr_warmup = trainer.get_lr(5)
                assert 0 < lr_warmup < config.learning_rate

                lr_peak = trainer.get_lr(10)
                assert abs(lr_peak - config.learning_rate) < 1e-6

                lr_mid = trainer.get_lr(50)
                assert config.min_learning_rate < lr_mid < config.learning_rate

                lr_end = trainer.get_lr(150)
                assert abs(lr_end - config.min_learning_rate) < 1e-6

        finally:
            os.unlink(train_data_path)

    def test_trainer_train_step(self) -> None:
        """Test that train_step works correctly."""
        train_data_path = self.create_test_data(vocab_size=100)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = TrainingConfig(
                    train_data_path=train_data_path,
                    vocab_size=100,
                    context_length=32,
                    d_model=64,
                    num_layers=2,
                    num_heads=4,
                    max_steps=10,
                    batch_size=2,
                    learning_rate=1e-3,
                    warmup_steps=2,
                    checkpoint_dir=tmp_dir,
                    log_dir=tmp_dir,
                    device="cpu",
                    compile_model=False,
                )

                trainer = Trainer(config)

                metrics = trainer.train_step()

                assert "loss" in metrics
                assert "lr" in metrics
                assert "step" in metrics

                assert trainer.step == 0

                assert isinstance(metrics["loss"], float)
                assert metrics["loss"] > 0

        finally:
            os.unlink(train_data_path)

    def test_trainer_evaluate(self) -> None:
        """Test that evaluate works correctly."""
        train_data_path = self.create_test_data(vocab_size=100)
        val_data_path = self.create_test_data(500, vocab_size=100)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = TrainingConfig(
                    train_data_path=train_data_path,
                    val_data_path=val_data_path,
                    vocab_size=100,
                    context_length=32,
                    d_model=64,
                    num_layers=2,
                    num_heads=4,
                    max_steps=10,
                    batch_size=2,
                    learning_rate=1e-3,
                    warmup_steps=2,
                    checkpoint_dir=tmp_dir,
                    log_dir=tmp_dir,
                    device="cpu",
                    compile_model=False,
                )

                trainer = Trainer(config)

                metrics = trainer.evaluate()

                assert "loss" in metrics
                assert "perplexity" in metrics

                assert isinstance(metrics["loss"], float)
                assert isinstance(metrics["perplexity"], float)
                assert metrics["loss"] > 0
                assert metrics["perplexity"] > 1

        finally:
            os.unlink(train_data_path)
            os.unlink(val_data_path)

    def test_trainer_evaluate_no_validation_data(self) -> None:
        """Test that evaluate returns empty dict when no validation data."""
        train_data_path = self.create_test_data(vocab_size=100)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = TrainingConfig(
                    train_data_path=train_data_path,
                    vocab_size=100,
                    context_length=32,
                    d_model=64,
                    num_layers=2,
                    num_heads=4,
                    max_steps=10,
                    batch_size=2,
                    learning_rate=1e-3,
                    warmup_steps=2,
                    checkpoint_dir=tmp_dir,
                    log_dir=tmp_dir,
                    device="cpu",
                    compile_model=False,
                )

                trainer = Trainer(config)

                metrics = trainer.evaluate()

                assert metrics == {}

        finally:
            os.unlink(train_data_path)

    def test_trainer_save_and_load_checkpoint(self) -> None:
        """Test that save_checkpoint and load_checkpoint work correctly."""
        train_data_path = self.create_test_data(vocab_size=100)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = TrainingConfig(
                    train_data_path=train_data_path,
                    vocab_size=100,
                    context_length=32,
                    d_model=64,
                    num_layers=2,
                    num_heads=4,
                    max_steps=10,
                    batch_size=2,
                    learning_rate=1e-3,
                    warmup_steps=2,
                    checkpoint_dir=tmp_dir,
                    log_dir=tmp_dir,
                    device="cpu",
                    compile_model=False,
                )

                trainer = Trainer(config)

                for _ in range(3):
                    trainer.train_step()
                    trainer.step += 1

                checkpoint_path = os.path.join(tmp_dir, "test_checkpoint.pt")
                trainer.save_checkpoint(checkpoint_path)

                assert os.path.exists(checkpoint_path)

                trainer2 = Trainer(config)
                trainer2.load_checkpoint(checkpoint_path)

                assert trainer2.step == 3

        finally:
            os.unlink(train_data_path)

    def test_trainer_short_training_loop(self) -> None:
        """Test a short training loop."""
        train_data_path = self.create_test_data(vocab_size=100)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = TrainingConfig(
                    train_data_path=train_data_path,
                    vocab_size=100,
                    context_length=32,
                    d_model=64,
                    num_layers=2,
                    num_heads=4,
                    max_steps=3,
                    batch_size=2,
                    learning_rate=1e-3,
                    warmup_steps=1,
                    log_interval=1,
                    eval_interval=2,
                    save_interval=5,
                    checkpoint_dir=tmp_dir,
                    log_dir=tmp_dir,
                    device="cpu",
                    compile_model=False,
                )

                trainer = Trainer(config)

                trainer.logger = MagicMock()

                trainer.train()

                assert trainer.step == config.max_steps

                final_checkpoint = os.path.join(tmp_dir, "checkpoint_final.pt")
                assert os.path.exists(final_checkpoint)

        finally:
            os.unlink(train_data_path)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_load_config(self) -> None:
        """Test that load_config works correctly."""
        config_dict = {
            "train_data_path": "test_data.npy",
            "vocab_size": 10000,
            "context_length": 256,
            "d_model": 512,
            "num_layers": 4,
            "num_heads": 16,
            "max_steps": 1000,
            "batch_size": 32,
            "learning_rate": 3e-4,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(config_dict, tmp_file)
            tmp_file.flush()

            try:
                config = load_config(tmp_file.name)

                assert config.train_data_path == "test_data.npy"
                assert config.vocab_size == 10000
                assert config.context_length == 256
                assert config.d_model == 512
                assert config.num_layers == 4
                assert config.num_heads == 16
                assert config.max_steps == 1000
                assert config.batch_size == 32
                assert config.learning_rate == 3e-4

            finally:
                os.unlink(tmp_file.name)

    def test_save_config(self) -> None:
        """Test that save_config works correctly."""
        config = TrainingConfig(
            train_data_path="test_data.npy",
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            max_steps=1000,
            batch_size=32,
            learning_rate=3e-4,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            try:
                save_config(config, tmp_file.name)

                with open(tmp_file.name, "r") as f:
                    loaded_dict = json.load(f)

                assert loaded_dict["train_data_path"] == "test_data.npy"
                assert loaded_dict["vocab_size"] == 10000
                assert loaded_dict["context_length"] == 256
                assert loaded_dict["d_model"] == 512
                assert loaded_dict["num_layers"] == 4
                assert loaded_dict["num_heads"] == 16
                assert loaded_dict["max_steps"] == 1000
                assert loaded_dict["batch_size"] == 32
                assert loaded_dict["learning_rate"] == 3e-4

            finally:
                os.unlink(tmp_file.name)

    def test_load_config_file_not_found(self) -> None:
        """Test that load_config raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_file.json")

    def test_load_config_invalid_json(self) -> None:
        """Test that load_config raises error for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            tmp_file.write("invalid json content")
            tmp_file.flush()

            try:
                with pytest.raises(json.JSONDecodeError):
                    load_config(tmp_file.name)
            finally:
                os.unlink(tmp_file.name)


class TestIntegration:
    """Integration tests that test multiple components together."""

    def test_config_serialization_round_trip(self) -> None:
        """Test that configuration can be saved and loaded correctly."""
        original_config = TrainingConfig(
            train_data_path="test_data.npy",
            val_data_path="test_val_data.npy",
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            max_steps=1000,
            batch_size=32,
            learning_rate=3e-4,
            min_learning_rate=1e-5,
            warmup_steps=100,
            weight_decay=0.01,
            wandb_project="test_project",
            wandb_run_name="test_run",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            try:
                save_config(original_config, tmp_file.name)
                loaded_config = load_config(tmp_file.name)

                assert loaded_config.train_data_path == original_config.train_data_path
                assert loaded_config.val_data_path == original_config.val_data_path
                assert loaded_config.vocab_size == original_config.vocab_size
                assert loaded_config.context_length == original_config.context_length
                assert loaded_config.d_model == original_config.d_model
                assert loaded_config.num_layers == original_config.num_layers
                assert loaded_config.num_heads == original_config.num_heads
                assert loaded_config.max_steps == original_config.max_steps
                assert loaded_config.batch_size == original_config.batch_size
                assert loaded_config.learning_rate == original_config.learning_rate
                assert loaded_config.min_learning_rate == original_config.min_learning_rate
                assert loaded_config.warmup_steps == original_config.warmup_steps
                assert loaded_config.weight_decay == original_config.weight_decay
                assert loaded_config.wandb_project == original_config.wandb_project
                assert loaded_config.wandb_run_name == original_config.wandb_run_name

            finally:
                os.unlink(tmp_file.name)

    def test_trainer_with_all_components(self) -> None:
        """Test that trainer works with all components integrated."""
        train_data = np.random.randint(0, 200, size=2000, dtype=np.uint16)
        val_data = np.random.randint(0, 200, size=1000, dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as train_file:
            train_data.tofile(train_file.name)
            train_file.flush()

            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as val_file:
                val_data.tofile(val_file.name)
                val_file.flush()

                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        config = TrainingConfig(
                            train_data_path=train_file.name,
                            val_data_path=val_file.name,
                            vocab_size=200,
                            context_length=16,
                            d_model=32,
                            num_layers=2,
                            num_heads=4,
                            max_steps=5,
                            batch_size=2,
                            learning_rate=1e-3,
                            warmup_steps=1,
                            log_interval=2,
                            eval_interval=3,
                            save_interval=4,
                            checkpoint_dir=tmp_dir,
                            log_dir=tmp_dir,
                            device="cpu",
                            compile_model=False,
                        )

                        trainer = Trainer(config)

                        trainer.logger = MagicMock()

                        initial_loss = None
                        for i in range(3):
                            metrics = trainer.train_step()
                            if initial_loss is None:
                                initial_loss = metrics["loss"]
                            trainer.step += 1

                        assert trainer.step == 3
                        assert isinstance(initial_loss, float)
                        assert initial_loss > 0

                        eval_metrics = trainer.evaluate()
                        assert "loss" in eval_metrics
                        assert "perplexity" in eval_metrics

                        checkpoint_path = os.path.join(tmp_dir, "test_checkpoint.pt")
                        trainer.save_checkpoint(checkpoint_path)
                        assert os.path.exists(checkpoint_path)

                        trainer2 = Trainer(config)
                        trainer2.load_checkpoint(checkpoint_path)
                        assert trainer2.step == 3

                finally:
                    os.unlink(train_file.name)
                    os.unlink(val_file.name)
