# Text Generation Module

This module implements text generation (decoding) for transformer language models with support for various sampling strategies.

## Features

The implementation supports all the features described in the assignment:

1. **Basic Autoregressive Generation**: Generate text token-by-token until hitting the `<|endoftext|>` token or reaching the maximum token limit.

2. **Temperature Scaling**: Control the randomness of generation by scaling the logits before applying softmax:
   ```
   softmax(v, τ)_i = exp(v_i/τ) / Σ_j exp(v_j/τ)
   ```
   - `τ → 0`: More deterministic (greedy-like)
   - `τ = 1`: Standard sampling
   - `τ > 1`: More random/diverse

3. **Top-p (Nucleus) Sampling**: Sample from the smallest set of tokens whose cumulative probability exceeds a threshold `p`:
   ```
   P(x_{t+1} = i | q) = {
       q_i / Σ_{j∈V(p)} q_j   if i ∈ V(p)
       0                       otherwise
   }
   ```
   where `V(p)` is the smallest set of vocabulary indices such that `Σ_{j∈V(p)} q_j ≥ p`.

4. **Configurable Generation Length**: Control the maximum number of tokens to generate.

## Module Structure

```
cs336_basics/
├── generation.py          # Core generation functions
│   ├── sample_from_logits()   # Sample with temperature and top-p
│   ├── generate()             # Main generation function
│   └── generate_batch()       # Batch generation utility
├── models.py              # TransformerLM model
└── tokenizer.py           # BPE tokenizer

Scripts:
├── generate_text.py       # Command-line generation script
└── demo_generation.py     # Interactive demo with examples
```

## Usage

### Basic Usage

```python
import torch
from cs336_basics import TransformerLM, Tokenizer, generate

# Load model and tokenizer
model = TransformerLM(...)
tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
eos_token_id = tokenizer.bytes_to_id[b"<|endoftext|>"]

# Encode prompt
prompt = "Once upon a time"
prompt_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

# Generate with default settings (temperature=1.0, no top-p)
generated_ids = generate(
    model=model,
    prompt=prompt_ids,
    max_tokens=100,
    eos_token_id=eos_token_id,
)

# Decode
generated_text = tokenizer.decode(generated_ids[0].tolist())
print(generated_text)
```

### Temperature Sampling

```python
# Low temperature (more deterministic)
generated_ids = generate(
    model=model,
    prompt=prompt_ids,
    max_tokens=100,
    temperature=0.5,
    eos_token_id=eos_token_id,
)

# High temperature (more random)
generated_ids = generate(
    model=model,
    prompt=prompt_ids,
    max_tokens=100,
    temperature=1.5,
    eos_token_id=eos_token_id,
)
```

### Top-p (Nucleus) Sampling

```python
# Sample from top 90% probability mass
generated_ids = generate(
    model=model,
    prompt=prompt_ids,
    max_tokens=100,
    temperature=1.0,
    top_p=0.9,
    eos_token_id=eos_token_id,
)

# More conservative (top 50%)
generated_ids = generate(
    model=model,
    prompt=prompt_ids,
    max_tokens=100,
    temperature=1.0,
    top_p=0.5,
    eos_token_id=eos_token_id,
)
```

### Combining Temperature and Top-p

```python
# Use both temperature scaling and nucleus sampling
generated_ids = generate(
    model=model,
    prompt=prompt_ids,
    max_tokens=100,
    temperature=0.8,
    top_p=0.9,
    eos_token_id=eos_token_id,
)
```

## Command-Line Interface

### Basic Generation

```bash
python generate_text.py \
    --checkpoint path/to/checkpoint.pt \
    --vocab artifacts/tinystories_bpe.yaml \
    --merges artifacts/tinystories_bpe_merges.txt \
    --prompt "Once upon a time" \
    --max-tokens 100
```

### With Temperature

```bash
python generate_text.py \
    --checkpoint path/to/checkpoint.pt \
    --vocab artifacts/tinystories_bpe.yaml \
    --merges artifacts/tinystories_bpe_merges.txt \
    --prompt "The little girl" \
    --max-tokens 100 \
    --temperature 0.8
```

### With Top-p Sampling

```bash
python generate_text.py \
    --checkpoint path/to/checkpoint.pt \
    --vocab artifacts/tinystories_bpe.yaml \
    --merges artifacts/tinystories_bpe_merges.txt \
    --prompt "In a magical forest" \
    --max-tokens 100 \
    --temperature 1.0 \
    --top-p 0.9
```

### With Random Seed (for reproducibility)

```bash
python generate_text.py \
    --checkpoint path/to/checkpoint.pt \
    --vocab artifacts/tinystories_bpe.yaml \
    --merges artifacts/tinystories_bpe_merges.txt \
    --prompt "Once upon a time" \
    --seed 42
```

## Interactive Demo

Run the demo script to see examples of different sampling strategies:

```bash
# Demo mode (shows examples)
python demo_generation.py

# Interactive mode (enter your own prompts)
python demo_generation.py --interactive
```

**Note**: You need to update the checkpoint and tokenizer paths in `demo_generation.py` before running.

## API Reference

### `sample_from_logits(logits, temperature=1.0, top_p=None)`

Sample a token from logits with optional temperature scaling and top-p sampling.

**Parameters:**
- `logits` (torch.Tensor): Logits tensor of shape `(vocab_size,)`
- `temperature` (float): Temperature for scaling. Must be > 0. Default: 1.0
- `top_p` (float, optional): Nucleus sampling threshold (0 < top_p <= 1.0)

**Returns:**
- `torch.Tensor`: Sampled token ID as a scalar tensor

### `generate(model, prompt, max_tokens=100, temperature=1.0, top_p=None, eos_token_id=None)`

Generate text from a language model given a prompt.

**Parameters:**
- `model` (nn.Module): The language model (TransformerLM)
- `prompt` (torch.Tensor): Input token IDs of shape `(batch_size, seq_len)` or `(seq_len,)`
- `max_tokens` (int): Maximum number of tokens to generate. Default: 100
- `temperature` (float): Sampling temperature. Default: 1.0
- `top_p` (float, optional): Nucleus sampling threshold
- `eos_token_id` (int, optional): End-of-text token ID to stop generation

**Returns:**
- `torch.Tensor`: Generated token IDs of shape `(batch_size, seq_len + num_generated)`

### `generate_batch(model, prompts, tokenizer, max_tokens=100, temperature=1.0, top_p=None, eos_token_id=None, device="cpu")`

Generate text for a batch of prompts.

**Parameters:**
- `model` (nn.Module): The language model
- `prompts` (List[str]): List of prompt strings
- `tokenizer`: Tokenizer with `encode()` and `decode()` methods
- `max_tokens` (int): Maximum tokens per prompt. Default: 100
- `temperature` (float): Sampling temperature. Default: 1.0
- `top_p` (float, optional): Nucleus sampling threshold
- `eos_token_id` (int, optional): End-of-text token ID
- `device` (str): Device to run on. Default: "cpu"

**Returns:**
- `List[str]`: List of generated text strings

## Implementation Details

### Temperature Scaling

Temperature scaling modifies the softmax distribution:
- **Low temperature** (τ < 1): Sharpens the distribution, making high-probability tokens even more likely
- **High temperature** (τ > 1): Flattens the distribution, making all tokens more equally likely
- **Temperature → 0**: Approaches greedy decoding (always pick argmax)

### Top-p (Nucleus) Sampling

The implementation:
1. Converts logits to probabilities via softmax
2. Sorts probabilities in descending order
3. Computes cumulative probabilities
4. Finds the smallest set of tokens whose cumulative probability exceeds `p`
5. Zeros out probabilities for tokens outside this set
6. Renormalizes and samples

This approach helps avoid sampling from the "long tail" of unlikely tokens while maintaining diversity.

## Tips for Good Generation

1. **Start with standard settings**: `temperature=1.0`, no top-p
2. **For more coherent text**: Lower temperature (0.5-0.8) or use top-p (0.9)
3. **For more creative text**: Higher temperature (1.2-1.5)
4. **Avoid very low temperatures**: Can lead to repetitive text
5. **Combine strategies**: Temperature + top-p often works well (e.g., `temperature=0.8, top_p=0.9`)

## References

- Holtzman et al. (2020): "The Curious Case of Neural Text Degeneration" - Introduces nucleus (top-p) sampling
- Original paper: https://arxiv.org/abs/1904.09751

