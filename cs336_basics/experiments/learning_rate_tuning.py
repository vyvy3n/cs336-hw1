"""
Learning rate tuning experiment.

This module implements the SGD optimizer with learning rate decay and runs
experiments to demonstrate the impact of different learning rates on training dynamics.
"""

import math
from collections.abc import Callable
from typing import Iterable

import torch


class SGD(torch.optim.Optimizer):
    """
    Stochastic Gradient Descent optimizer with learning rate decay.

    This implementation uses a learning rate schedule of the form:
    lr_t = lr / sqrt(t + 1)
    where t is the iteration number.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3) -> None:
        """
        Initialize the SGD optimizer.

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1e-3)

        Raises:
            ValueError: If learning rate is negative
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure to re-evaluate the model and return the loss

        Returns:
            Loss if closure is provided, None otherwise
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data

                effective_lr = lr / math.sqrt(t + 1)
                p.data -= effective_lr * grad

                state["t"] = t + 1

        return loss


def run_learning_rate_experiment(lr: float, iterations: int = 10) -> list[float]:
    """
    Run a simple optimization experiment with a given learning rate.

    This function creates a simple quadratic loss function (mean of squared weights)
    and optimizes it using SGD with the specified learning rate.

    Args:
        lr: Learning rate to use
        iterations: Number of optimization steps to perform

    Returns:
        List of loss values at each iteration
    """
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    optimizer = SGD([weights], lr=lr)

    losses = []

    for _ in range(iterations):
        optimizer.zero_grad()

        loss = (weights**2).mean()
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return losses


def analyze_learning_rates() -> None:
    """
    Run the learning rate tuning experiment as specified in the assignment.

    Tests learning rates of 1e1, 1e2, and 1e3 for 10 iterations each
    and analyzes the behavior.
    """
    learning_rates = [1e1, 1e2, 1e3]
    iterations = 10

    print("Learning Rate Tuning Experiment")
    print("=" * 40)

    results = {}

    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        print("-" * 30)

        try:
            losses = run_learning_rate_experiment(lr, iterations)
            results[lr] = losses

            print(f"Initial loss: {losses[0]:.6f}")
            print(f"Final loss: {losses[-1]:.6f}")

            if losses[-1] < losses[0]:
                trend = "DECREASING"
            elif losses[-1] > losses[0]:
                if any(math.isnan(loss) or math.isinf(loss) for loss in losses):
                    trend = "DIVERGED (NaN/Inf)"
                else:
                    trend = "INCREASING"
            else:
                trend = "STABLE"

            print(f"Trend: {trend}")

            print("Loss progression:")
            for i, loss in enumerate(losses):
                if math.isnan(loss) or math.isinf(loss):
                    print(f"  Step {i}: {loss}")
                    break
                else:
                    print(f"  Step {i}: {loss:.6f}")

        except Exception as e:
            print(f"Error with lr={lr}: {e}")
            results[lr] = None

    print("\n" + "=" * 40)
    print("SUMMARY ANALYSIS")
    print("=" * 40)

    for lr in learning_rates:
        if results[lr] is not None:
            losses = results[lr]
            initial = losses[0]
            final = losses[-1]

            if math.isnan(final) or math.isinf(final):
                behavior = "diverged to NaN/Inf"
            elif final < initial:
                rate = (initial - final) / initial * 100
                behavior = f"decreased by {rate:.1f}%"
            elif final > initial:
                rate = (final - initial) / initial * 100
                behavior = f"increased by {rate:.1f}%"
            else:
                behavior = "remained stable"

            print(f"LR {lr}: {behavior}")


if __name__ == "__main__":
    analyze_learning_rates()
