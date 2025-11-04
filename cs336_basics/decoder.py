"""
Text generation and decoding utilities for transformer language models.

This module provides functions for generating text from trained language models,
including support for:
- Temperature scaling
- Top-p (nucleus) sampling
- Configurable maximum generation length
- End-of-text token detection
"""

from typing import Optional, List
import torch
import torch.nn as nn


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically-stable softmax along the specified dimension.

    Args:
        x: Input tensor
        dim: Dimension to apply softmax to

    Returns:
        Tensor with softmax applied along the specified dimension
    """
    # Subtract max for numerical stability, then exponentiate
    y = x - torch.amax(x, dim=dim, keepdim=True)
    y = torch.exp(y)
    # Normalize
    y = y / torch.sum(y, dim=dim, keepdim=True)
    return y


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Sample a token from logits with optional temperature scaling and top-p sampling.

    Args:
        logits: Logits tensor of shape (vocab_size,)
        temperature: Temperature for scaling logits. Must be > 0.
                    higher values make distribution more uniform,
                    lower values make distribution more peaked,
                    -> 0 makes the largest logits dominate so that the output of softmax becomes one-hot.
        top_p: If provided, only sample from the smallest set of tokens whose
              cumulative probability exceeds top_p (nucleus sampling).

    Returns:
        Sampled token ID as a scalar tensor
    """
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Convert logits to probabilities
    probs = softmax(logits, dim=-1)
    
    # Apply top-p (nucleus) sampling if specified
    if top_p is not None and top_p < 1.0:
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find the smallest set of tokens whose cumulative probability exceeds top_p
        # We want to include the first token that pushes us over top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the mask to the right to keep at least one token
        # and to include the first token that exceeds top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        # Zero out probabilities for tokens not in the nucleus
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        
        # Renormalize probabilities
        probs = probs / probs.sum()
    
    # Sample from the distribution
    token = torch.multinomial(probs, num_samples=1)
    
    return token


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt: torch.Tensor,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate text from a language model given a prompt.
    
    This function implements autoregressive text generation by repeatedly:
    1. Running the model on the current sequence
    2. Sampling the next token from the output distribution
    3. Appending the sampled token to the sequence
    4. Repeating until max_tokens is reached or eos_token_id is generated
    
    Args:
        model: The language model (should be a TransformerLM or similar)
        prompt: Input token IDs of shape (batch_size, seq_len) or (seq_len,)
               If 1D, will be unsqueezed to (1, seq_len)
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling. Higher = more random, lower = more deterministic.
                    Must be > 0. Default is 1.0 (no scaling).
        top_p: If provided, use nucleus sampling with this threshold (0 < top_p <= 1.0).
              Only sample from the smallest set of tokens whose cumulative probability
              exceeds top_p.
        eos_token_id: If provided, stop generation when this token is sampled.
                     Typically set to the <|endoftext|> token ID.
    
    Returns:
        Generated token IDs of shape (batch_size, seq_len + num_generated_tokens)
        where num_generated_tokens <= max_tokens
    
    Example:
        >>> model = TransformerLM(...)
        >>> tokenizer = Tokenizer.from_files(...)
        >>> prompt_text = "Once upon a time"
        >>> prompt_ids = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0)
        >>> generated_ids = generate(model, prompt_ids, max_tokens=50, temperature=0.8)
        >>> generated_text = tokenizer.decode(generated_ids[0].tolist())
    """
    model.eval()
    
    # Handle 1D input
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    
    # Start with the prompt
    generated = prompt.clone()
    
    # Generate tokens one at a time
    for _ in range(max_tokens):
        # Get model predictions for the current sequence
        # Output shape: (batch_size, seq_len, vocab_size)
        logits = model(generated)
        
        # Get logits for the last position (next token prediction)
        # Shape: (batch_size, vocab_size)
        next_token_logits = logits[:, -1, :]
        
        # Sample next token for each sequence in the batch
        next_tokens = []
        for i in range(generated.size(0)):
            next_token = sample_from_logits(
                next_token_logits[i],
                temperature=temperature,
                top_p=top_p,
            )
            next_tokens.append(next_token)
        
        # Stack next tokens: (batch_size, 1)
        next_tokens = torch.stack(next_tokens, dim=0)
        
        # Append to generated sequence
        generated = torch.cat([generated, next_tokens], dim=1)
        
        # Check for end-of-text token
        if eos_token_id is not None:
            # If all sequences have generated eos_token, stop
            if (next_tokens == eos_token_id).all():
                break
    
    return generated


def generate_batch(
    model: nn.Module,
    prompts: List[str],
    tokenizer,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
) -> List[str]:
    """
    Generate text for a batch of prompts.
    
    Args:
        model: The language model
        prompts: List of prompt strings
        tokenizer: Tokenizer with encode() and decode() methods
        max_tokens: Maximum number of tokens to generate per prompt
        temperature: Temperature for sampling
        top_p: Top-p threshold for nucleus sampling
        eos_token_id: End-of-text token ID
        device: Device to run generation on
    
    Returns:
        List of generated text strings
    """
    model.eval()
    
    # Encode all prompts
    prompt_ids = [torch.tensor(tokenizer.encode(prompt), device=device) for prompt in prompts]
    
    # Pad to same length for batching
    max_prompt_len = max(len(p) for p in prompt_ids)
    padded_prompts = []
    for p in prompt_ids:
        if len(p) < max_prompt_len:
            # Pad with zeros (or use a pad token if available)
            padding = torch.zeros(max_prompt_len - len(p), dtype=torch.long, device=device)
            p = torch.cat([p, padding])
        padded_prompts.append(p)
    
    # Stack into batch: (batch_size, max_prompt_len)
    batch_prompts = torch.stack(padded_prompts, dim=0)
    
    # Generate
    generated_ids = generate(
        model=model,
        prompt=batch_prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )
    
    # Decode each sequence
    generated_texts = []
    for ids in generated_ids:
        text = tokenizer.decode(ids.tolist())
        generated_texts.append(text)
    
    return generated_texts
