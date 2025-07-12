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
        """Sample the most likely token (greedy)."""
        return int(logits.argmax().item())


class MultinomialDecoding:
    """Multinomial sampling with optional temperature scaling."""

    def sample(self, logits: Float[torch.Tensor, "vocab_size"], temperature: float = 1.0) -> int:
        """Sample from the probability distribution with temperature scaling."""
        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


class TopPDecoding:
    """Top-p (nucleus) sampling with temperature scaling."""

    def __init__(self, p: float = 0.9) -> None:
        """
        Initialize top-p sampling.

        Args:
            p: Cumulative probability threshold for nucleus sampling.
        """
        self.p = p

    def sample(self, logits: Float[torch.Tensor, "vocab_size"], temperature: float = 1.0) -> int:
        """Sample using top-p (nucleus) sampling."""
        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.p

        sorted_indices_to_remove[0] = False
        sorted_probs[sorted_indices_to_remove] = 0.0

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

    This function implements autoregressive text generation using various sampling strategies
    including temperature and top-p (nucleus) sampling.

    Args:
        model: Trained TransformerLM model
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Input text prompt to complete
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more random, lower = more deterministic)
        top_p: If provided, use top-p sampling with this threshold
        end_token: Token that signals end of generation
        device: Device to run inference on

    Returns:
        Generated text completion (prompt + generated tokens)

    Raises:
        ValueError: If temperature is not positive
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    model.eval()
    model = model.to(device)

    if top_p is not None:
        decoder = TopPDecoding(p=top_p)
    elif temperature <= 0.1:
        decoder = GreedyDecoding()
    else:
        decoder = MultinomialDecoding()

    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    end_token_id = tokenizer.encode(end_token)[0] if end_token in tokenizer.encode(end_token) else None

    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.size(1) >= model.context_length:
                input_tensor = input_tensor[:, -model.context_length :]

            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            next_token_id = decoder.sample(next_token_logits, temperature)

            if end_token_id is not None and next_token_id == end_token_id:
                break

            generated_tokens.append(next_token_id)

            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)

    if generated_tokens:
        generated_text = tokenizer.decode(generated_tokens)
        return prompt + generated_text
    else:
        return prompt


def generate_completions(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float | None = None,
    end_token: str = "<|endoftext|>",
    device: torch.device | str = "cpu",
) -> list[str]:
    """
    Generate completions for multiple prompts.

    Args:
        model: Trained TransformerLM model
        tokenizer: Tokenizer for encoding/decoding text
        prompts: List of input text prompts to complete
        max_new_tokens: Maximum number of new tokens to generate per prompt
        temperature: Temperature for sampling
        top_p: If provided, use top-p sampling with this threshold
        end_token: Token that signals the end of generation
        device: Device to run inference on

    Returns:
        List of generated completions
    """
    completions = []

    for prompt in prompts:
        completion = decode_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            end_token=end_token,
            device=device,
        )
        completions.append(completion)

    return completions


def compute_perplexity(
    model: TransformerLM, tokenizer: Tokenizer, text: str, device: torch.device | str = "cpu"
) -> float:
    """
    Compute perplexity of a text sequence under the model.

    Args:
        model: Trained TransformerLM model
        tokenizer: Tokenizer for encoding text
        text: Input text to compute perplexity for
        device: Device to run inference on

    Returns:
        Perplexity value
    """
    model.eval()
    model = model.to(device)

    input_ids = tokenizer.encode(text)

    if len(input_ids) < 2:
        return float("inf")

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    total_loss = 0.0
    num_tokens = 0

    with torch.no_grad():
        for start_idx in range(0, len(input_ids), model.context_length):
            end_idx = min(start_idx + model.context_length, len(input_ids))

            input_chunk = input_tensor[:, start_idx:end_idx]

            if input_chunk.size(1) < 2:
                break

            logits = model(input_chunk)

            input_logits = logits[:, :-1, :]
            target_tokens = input_chunk[:, 1:]

            loss = F.cross_entropy(input_logits.view(-1, model.vocab_size), target_tokens.view(-1), reduction="sum")

            total_loss += loss.item()
            num_tokens += target_tokens.numel()

    avg_loss = total_loss / num_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def interactive_generation(
    model: TransformerLM,
    tokenizer: Tokenizer,
    device: torch.device | str = "cpu",
) -> None:
    """
    Interactive text generation session.

    Args:
        model: Trained TransformerLM model
        tokenizer: Tokenizer for encoding/decoding text
        device: Device to run inference on
    """
    print("Interactive Text Generation")
    print("Type 'quit' to exit")
    print("Available commands:")
    print("  /temp <value>  - Set temperature (default: 1.0)")
    print("  /top_p <value> - Set top-p threshold (default: None)")
    print("  /max <value>   - Set max new tokens (default: 100)")
    print("  /help          - Show this help message")
    print()

    temperature = 1.0
    top_p = None
    max_new_tokens = 100

    while True:
        try:
            user_input = input("Prompt: ").strip()

            if user_input.lower() == "quit":
                break
            elif user_input.startswith("/temp "):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                except (ValueError, IndexError):
                    print("Invalid temperature value")
                continue
            elif user_input.startswith("/top_p "):
                try:
                    top_p = float(user_input.split()[1])
                    print(f"Top-p set to {top_p}")
                except (ValueError, IndexError):
                    print("Invalid top-p value")
                continue
            elif user_input.startswith("/max "):
                try:
                    max_new_tokens = int(user_input.split()[1])
                    print(f"Max new tokens set to {max_new_tokens}")
                except (ValueError, IndexError):
                    print("Invalid max tokens value")
                continue
            elif user_input == "/help":
                print("Available commands:")
                print("  /temp <value>  - Set temperature")
                print("  /top_p <value> - Set top-p threshold")
                print("  /max <value>   - Set max new tokens")
                print("  /help          - Show this help message")
                continue

            if not user_input:
                continue

            completion = decode_text(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )

            print(f"Generated: {completion}")
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
