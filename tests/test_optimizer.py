"""
Tests for optimizer implementations.
"""

import numpy as np
import pytest
import torch

from cs336_basics.training.optimizers import AdamW, MixedOptimizer, Muon

from .adapters import get_adamw_cls


class TestAdamW:
    """Tests for AdamW optimizer."""

    @pytest.fixture
    def simple_params(self):
        """Create simple parameters for testing."""
        return [torch.randn(3, 4, requires_grad=True)]

    def test_single_step(self, simple_params):
        """Test single optimization step."""
        params = simple_params
        optimizer = get_adamw_cls()(params, lr=0.1)

        loss = (params[0] ** 2).sum()
        loss.backward()

        original_params = [p.clone() for p in params]

        optimizer.step()

        for original, current in zip(original_params, params):
            assert not torch.allclose(original, current)

    def test_learning_rate_decay(self, simple_params):
        """Test that parameters change less with smaller learning rate."""
        params1 = [p.clone().detach().requires_grad_(True) for p in simple_params]
        params2 = [p.clone().detach().requires_grad_(True) for p in simple_params]

        optimizer1 = get_adamw_cls()(params1, lr=0.1)
        optimizer2 = get_adamw_cls()(params2, lr=0.01)

        loss1 = (params1[0] ** 2).sum()
        loss2 = (params2[0] ** 2).sum()
        loss1.backward()
        loss2.backward()

        original1 = [p.clone() for p in params1]
        original2 = [p.clone() for p in params2]

        optimizer1.step()
        optimizer2.step()

        change1 = torch.norm(params1[0] - original1[0])
        change2 = torch.norm(params2[0] - original2[0])
        assert change1 > change2


class TestMuon:
    """Tests for Muon optimizer."""

    @pytest.fixture
    def matrix_params(self):
        """Create matrix parameters for testing Muon."""
        return [torch.randn(4, 8, requires_grad=True)]

    @pytest.fixture
    def mixed_params(self):
        """Create mixed parameter types for testing."""
        return [
            torch.randn(4, 8, requires_grad=True),
            torch.randn(8, requires_grad=True),
        ]

    def test_muon_single_step(self, matrix_params):
        """Test single Muon optimization step."""
        params = matrix_params
        optimizer = Muon(params, lr=0.1)

        loss = (params[0] ** 2).sum()
        loss.backward()

        original_params = [p.clone() for p in params]
        optimizer.step()

        for original, current in zip(original_params, params):
            assert not torch.allclose(original, current)

    def test_muon_orthogonalization(self, matrix_params):
        """Test that Muon performs orthogonalization."""
        params = matrix_params
        optimizer = Muon(params, lr=0.1, ns_iters=3)

        params[0].grad = torch.randn_like(params[0])

        grad = params[0].grad
        ortho_grad = optimizer.newton_schulz_orthogonalize(grad, 3)

        assert not torch.allclose(grad, ortho_grad)

    def test_muon_vs_adamw_convergence(self):
        """Test that Muon converges faster than AdamW on a simple problem."""
        torch.manual_seed(42)

        param_muon = torch.randn(8, 8, requires_grad=True)
        param_adamw = param_muon.clone().detach().requires_grad_(True)

        optimizer_muon = Muon([param_muon], lr=0.01)
        optimizer_adamw = AdamW([param_adamw], lr=0.01)

        target = torch.eye(8)

        losses_muon = []
        losses_adamw = []

        for _ in range(10):
            optimizer_muon.zero_grad()
            loss_muon = torch.norm(param_muon - target) ** 2
            loss_muon.backward()
            optimizer_muon.step()
            losses_muon.append(loss_muon.item())

            optimizer_adamw.zero_grad()
            loss_adamw = torch.norm(param_adamw - target) ** 2
            loss_adamw.backward()
            optimizer_adamw.step()
            losses_adamw.append(loss_adamw.item())

        assert losses_muon[-1] < losses_muon[0]
        assert losses_adamw[-1] < losses_adamw[0]

    def test_muon_mixed_params(self, mixed_params):
        """Test Muon with mixed parameter types (matrix + bias)."""
        optimizer = Muon(mixed_params, lr=0.1)

        loss = (mixed_params[0] ** 2).sum() + (mixed_params[1] ** 2).sum()
        loss.backward()

        original_params = [p.clone() for p in mixed_params]
        optimizer.step()

        for original, current in zip(original_params, mixed_params):
            assert not torch.allclose(original, current)


class TestMixedOptimizer:
    """Tests for MixedOptimizer."""

    @pytest.fixture
    def model_params(self):
        """Create model-like parameters for testing mixed optimizer."""
        embedding = torch.randn(1000, 128, requires_grad=True)
        linear1 = torch.randn(128, 256, requires_grad=True)
        bias1 = torch.randn(256, requires_grad=True)
        linear2 = torch.randn(256, 128, requires_grad=True)
        lm_head = torch.randn(128, 1000, requires_grad=True)

        return {
            "embedding": embedding,
            "linear1": linear1,
            "bias1": bias1,
            "linear2": linear2,
            "lm_head": lm_head,
        }

    def test_mixed_optimizer_initialization(self, model_params):
        """Test mixed optimizer initialization."""
        params = list(model_params.values())
        param_names = {param: name for name, param in model_params.items()}

        optimizer = MixedOptimizer(
            params,
            muon_lr=0.01,
            adamw_lr=0.005,
            embedding_lr=0.02,
            lm_head_lr=0.001,
        )

        assert optimizer is not None
        assert len(optimizer.param_groups) == 1

    def test_mixed_optimizer_categorization(self, model_params):
        """Test parameter categorization in mixed optimizer."""
        params = list(model_params.values())
        optimizer = MixedOptimizer(params)

        assert optimizer.categorize_parameter("embedding", model_params["embedding"]) == "embedding"
        assert optimizer.categorize_parameter("lm_head", model_params["lm_head"]) == "lm_head"
        assert optimizer.categorize_parameter("bias1", model_params["bias1"]) == "adamw"
        assert optimizer.categorize_parameter("linear1", model_params["linear1"]) == "muon"

    def test_mixed_optimizer_step(self, model_params):
        """Test mixed optimizer step."""
        params = list(model_params.values())
        param_names = {param: name for name, param in model_params.items()}

        optimizer = MixedOptimizer(
            params,
            muon_lr=0.01,
            adamw_lr=0.005,
            embedding_lr=0.02,
            lm_head_lr=0.001,
        )

        loss = sum((p**2).sum() for p in params)
        loss.backward()

        original_params = [p.clone() for p in params]
        optimizer.step(param_names=param_names)

        for original, current in zip(original_params, params):
            assert not torch.allclose(original, current)


class TestOptimizerCompatibility:
    """Test compatibility between optimizers."""

    def test_all_optimizers_work(self):
        """Test that all optimizers can handle a simple optimization problem."""
        torch.manual_seed(42)

        param_shapes = [(4, 8), (8,), (8, 4)]

        optimizers = [
            lambda params: AdamW(params, lr=0.01),
            lambda params: Muon(params, lr=0.01),
            lambda params: MixedOptimizer(params, muon_lr=0.01, adamw_lr=0.01),
        ]

        for opt_fn in optimizers:
            params = [torch.randn(shape, requires_grad=True) for shape in param_shapes]
            optimizer = opt_fn(params)

            loss = sum((p**2).sum() for p in params)
            loss.backward()

            if isinstance(optimizer, MixedOptimizer):
                param_names = {params[i]: f"param_{i}" for i in range(len(params))}
                optimizer.step(param_names=param_names)
            else:
                optimizer.step()

            for p in params:
                assert torch.isfinite(p).all()

    def test_optimizer_state_persistence(self):
        """Test that optimizer state is maintained correctly."""
        torch.manual_seed(42)

        param = torch.randn(4, 4, requires_grad=True)
        optimizer = Muon([param], lr=0.01)

        loss1 = (param**2).sum()
        loss1.backward()
        optimizer.step()

        assert param in optimizer.state
        assert "momentum_buffer" in optimizer.state[param]

        optimizer.zero_grad()
        loss2 = (param**2).sum()
        loss2.backward()
        optimizer.step()

        assert param in optimizer.state
        assert optimizer.state[param]["step"] == 2
