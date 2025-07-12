"""
Text generation and decoding functionality for Transformer language models.

This module implements various decoding strategies for generating text from a trained
Transformer language model, including temperature scaling and top-p (nucleus) sampling.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from jaxtyping import Float

from cs336_basics.nn.activations import softmax
from cs336_basics.nn.models import TransformerLM
from cs336_basics.tokenization.tokenizer import Tokenizer


class DecodingStrategy(Protocol):
    """Protocol for decoding strategies."""

    def sample(self, logits: Float[torch.Tensor, "vocab_size"], temperature: float = 1.0) -> int:
        """
        Sample a token from the logits.

        Args:
            logits: Raw logits from the model with shape [vocab size]
            temperature: Temperature parameter for scaling logits. Higher value make the
                         distribution more uniform, lower values make it more peaky. Default: 1.0

        Returns:
            The sampled token ID
        """
        ...


class GreedyDecoding:
    """Greedy decoding strategy that always selects the most likely token."""

    def sample(self, logits: Float[torch.Tensor, "vocab_size"], temperature: float = 1.0) -> int:
        "Sample the most likely token (greedy)."
        return int(logits.argmax().item())


class MultinomialDecoding:
    """Multinomial sampling with optional temperature scaling."""

    def sample(self, logits: Float[torch.Tensor, "vocab_size"], temperature: float = 1.0) -> int:
        """Sample from the probability distribution with temperature scaling."""
        if temperature != 1.0:
            logits = logits / temperature

        probs = softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


class TopPDecoding:
    """Top-p (nucleus) sampling with temperature scaling."""

    def __init__(self, p: float = 0.9) -> None:
        """
        Initialize top-p sampling.

        Args:
            p: Cumulative probability threshold for nucleus sampling.
        """

    def sample(self, logits: Float[torch.Tensor, "vocab_size"], temperature: float = 1.0) -> int:
        """Sample using top-p (nucleus) sampling."""
        if temperature != 1.0:
            logits = logits / temperature

        probs = softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.p

        sorted_indices_to_remove[0] = False
        sorted_probs[sorted_indices_to_remove] = 0.00

        sorted_probs = sorted_probs / sorted_probs.sum()
        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)

        return int(sorted_indices[sampled_idx].item())


def decode_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float | None = None,
    end_token: str = "<|endoftext|>",
    device: torch.device | str = "cpu",
) -> str:
    """
    Generate text completion for a given prompt using a trained Transformer language model.
    """
    pass
