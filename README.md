# CS336 Spring 2025 Assignment 1: Basics

This repository contains a complete implementation of a Transformer language model with training infrastructure, tokenization, and experiment utilities for CS336 Assignment 1.

For the full assignment description, see [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

## Table of Contents

- [Setup](#setup)
- [Quick Start](#quick-start)
- [Codebase Structure](#codebase-structure)
- [Data Preparation](#data-preparation)
- [Experiments & Deliverables](#experiments--deliverables)
  - [1. Text Generation & Decoding](#1-text-generation--decoding)
  - [2. Experiment Logging](#2-experiment-logging)
  - [3. TinyStories Pretraining (Base)](#3-tinystories-pretraining-base)
  - [4. Learning Rate Sweep](#4-learning-rate-sweep)
  - [5. Batch Size Sweep](#5-batch-size-sweep)
  - [6. Generate from TinyStories Model](#6-generate-from-tinystories-model)
  - [7. Ablation Studies](#7-ablation-studies)
  - [8. OpenWebText Training](#8-openwebtext-training)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)

---

## Setup

### Environment

We use `uv` for environment management to ensure reproducibility and ease of use.

**Install `uv`:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv

# Or via Homebrew
brew install uv
```

**Run any Python script:**
```bash
uv run <python_file_path>
```
The environment will be automatically solved and activated when necessary.

### Run Unit Tests

```bash
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s. To connect your implementation to the tests, complete the functions in [./tests/adapters.py](./tests/adapters.py).

### Download Data

Download the TinyStories and OpenWebText datasets:

```bash
mkdir -p data
cd data

# TinyStories dataset
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText sample
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

---

## Quick Start

### 1. Prepare Datasets (Tokenization)

```bash
# Train BPE tokenizer and encode TinyStories dataset
uv run scripts/prepare_tinystories.py --data_dir data --vocab_size 10000

# Or do it step-by-step:
# Step 1: Train BPE tokenizer
uv run scripts/train_bpe.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output artifacts/tinystories_bpe.yaml \
    --vocab_size 10000

# Step 2: Encode datasets
uv run scripts/encode_dataset.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output data/tinystories_train_tokens.npy \
    --tokenizer artifacts/tinystories_bpe.yaml

uv run scripts/encode_dataset.py \
    --input data/TinyStoriesV2-GPT4-valid.txt \
    --output data/tinystories_valid_tokens.npy \
    --tokenizer artifacts/tinystories_bpe.yaml
```

**Note:** The assignment provides pre-trained tokenizers in `artifacts/` for both TinyStories and OpenWebText. You can use those directly if you prefer.

### 2. Train a Model

```bash
# Train on TinyStories (recommended for quick experiments)
uv run scripts/train.py --dataset tinystories --device cuda

# Train on OpenWebText
uv run scripts/train.py --dataset owt --device cuda

# Custom training with specific hyperparameters
uv run scripts/train.py --dataset tinystories \
    --learning_rate 3e-4 \
    --batch_size 32 \
    --max_iters 40000 \
    --device cuda
```

### 3. Generate Text

```bash
# Generate from TinyStories model
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --vocab artifacts/tinystories_vocab.json \
    --merges artifacts/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --temperature 0.8 \
    --max-tokens 256
```

---

## Codebase Structure

```
cs336-hw1/
├── cs336_basics/              # Core implementation
│   ├── models.py              # TransformerLM, TransformerBlock, MultiheadSelfAttention
│   ├── layers.py              # RMSNorm, SwiGLU, RoPE, Softmax, Linear, Embedding
│   ├── tokenizer.py           # BPE tokenizer (encode/decode)
│   ├── bpe.py                 # BPE training algorithm
│   ├── pretokenization.py     # GPT-2 style pre-tokenization
│   ├── generation.py          # Text generation (temperature, top-p sampling)
│   ├── optimizers.py          # AdamW, CrossEntropyLoss, LR scheduling
│   ├── training.py            # Trainer class
│   ├── config.py              # TrainingConfig dataclass
│   ├── utils.py               # Checkpointing, logging, device setup
│   └── ablation_models.py     # Ablation variants (NoRMSNorm, PostNorm, etc.)
│
├── scripts/                   # Standalone scripts
│   ├── train.py               # Main training script
│   ├── generate_text.py       # Text generation script
│   ├── train_bpe.py           # Train BPE tokenizer
│   ├── encode_dataset.py      # Encode dataset with tokenizer
│   ├── prepare_tinystories.py # All-in-one dataset preparation
│   └── find_max_batch_size.py # Find max batch size before OOM
│
├── experiments/               # Experiment scripts for deliverables
│   ├── learning_rate_sweep.py # LR sweep (grid + stability)
│   ├── batch_size_sweep.py    # Batch size sweep
│   └── ablations.py           # Architecture ablations
│
├── data/                      # Datasets (download here)
│   ├── TinyStoriesV2-GPT4-train.txt
│   ├── TinyStoriesV2-GPT4-valid.txt
│   ├── owt_train.txt
│   ├── owt_valid.txt
│   ├── tinystories_train_tokens.npy  # Encoded tokens
│   ├── tinystories_valid_tokens.npy
│   ├── owt_train_tokens.npy
│   └── owt_valid_tokens.npy
│
├── artifacts/                 # Tokenizers (provided by assignment)
│   ├── tinystories_vocab.json
│   ├── tinystories_merges.txt
│   ├── tinystories_bpe.yaml
│   ├── owt_vocab.json
│   ├── owt_merges.txt
│   └── owt_bpe.yaml
│
├── checkpoints/               # Model checkpoints
│   ├── tinystories/
│   ├── owt/
│   ├── lr_sweep/
│   ├── batch_size_sweep/
│   ├── ablations/
│   └── edge_of_stability/
│
├── tests/                     # Unit tests
│   ├── test_model.py
│   ├── test_tokenizer.py
│   ├── test_optimizer.py
│   └── ...
│
├── README.md                  # This file
├── TOKENIZER_GUIDE.md         # BPE tokenizer guide
└── pyproject.toml             # Dependencies
```

---

## Data Preparation

### Option 1: Use Provided Tokenizers (Recommended)

The assignment provides pre-trained tokenizers in `artifacts/`:
- **TinyStories**: `tinystories_vocab.json`, `tinystories_merges.txt`, `tinystories_bpe.yaml`
- **OpenWebText**: `owt_vocab.json`, `owt_merges.txt`, `owt_bpe.yaml`

You only need to encode the datasets:

```bash
# Encode TinyStories
uv run scripts/encode_dataset.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output data/tinystories_train_tokens.npy \
    --tokenizer artifacts/tinystories_bpe.yaml

uv run scripts/encode_dataset.py \
    --input data/TinyStoriesV2-GPT4-valid.txt \
    --output data/tinystories_valid_tokens.npy \
    --tokenizer artifacts/tinystories_bpe.yaml

# Encode OpenWebText
uv run scripts/encode_dataset.py \
    --input data/owt_train.txt \
    --output data/owt_train_tokens.npy \
    --tokenizer artifacts/owt_bpe.yaml

uv run scripts/encode_dataset.py \
    --input data/owt_valid.txt \
    --output data/owt_valid_tokens.npy \
    --tokenizer artifacts/owt_bpe.yaml
```

### Option 2: Train Your Own Tokenizer

```bash
# Train BPE tokenizer on TinyStories
uv run scripts/train_bpe.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output artifacts/my_tinystories_bpe.yaml \
    --vocab_size 10000

# Then encode datasets with your tokenizer
uv run scripts/encode_dataset.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output data/tinystories_train_tokens.npy \
    --tokenizer artifacts/my_tinystories_bpe.yaml
```

For more details on BPE training, see [TOKENIZER_GUIDE.md](./TOKENIZER_GUIDE.md).

---

## Experiments & Deliverables

This section maps each assignment deliverable to the corresponding code and commands.

### 1. Text Generation & Decoding

**Deliverable:** Implement greedy/temperature/top-p decoding. Generate 256+ tokens and provide fluency notes.

**Implementation:** `cs336_basics/generation.py`
- `generate()` - Autoregressive generation with temperature and top-p sampling
- `sample_from_logits()` - Sampling with temperature scaling and nucleus sampling
- `softmax()` - Numerically stable softmax

**Usage:**

```bash
# Greedy decoding (temperature=1.0, no top-p)
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --vocab artifacts/tinystories_vocab.json \
    --merges artifacts/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --max-tokens 256

# Temperature sampling (more random)
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --vocab artifacts/tinystories_vocab.json \
    --merges artifacts/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --temperature 0.8 \
    --max-tokens 256

# Top-p (nucleus) sampling
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --vocab artifacts/tinystories_vocab.json \
    --merges artifacts/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --temperature 0.8 \
    --top-p 0.9 \
    --max-tokens 256

# Low temperature (more deterministic)
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --vocab artifacts/tinystories_vocab.json \
    --merges artifacts/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --temperature 0.3 \
    --max-tokens 256
```

Or, inference using yaml

```bash
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --tokenizer artifacts/tinystories_bpe.yaml \
    --prompt "Once upon a time" \
    --max-tokens 250
```

**Features:**
- ✅ Stops on `<|endoftext|>` token
- ✅ Supports `max_tokens` parameter
- ✅ Temperature scaling (0 < temperature < ∞)
- ✅ Top-p (nucleus) sampling (0 < top_p ≤ 1.0)
- ✅ Batch generation support

**Deliverable Notes:**
- Generate samples with different temperature/top-p settings
- Compare fluency: low temp (deterministic) vs high temp (creative)
- Note: Lower temperature → more coherent but repetitive; Higher temperature → more diverse but less coherent

---

### 2. Experiment Logging

**Deliverable:** Log step, wall-clock time, train loss, val loss, and config. Provide logging code + experiment log with curves.

**Implementation:** `cs336_basics/training.py` + `cs336_basics/utils.py`
- Logs to console and Weights & Biases (W&B)
- Tracks: step, time, train loss, val loss, learning rate, gradient norms
- Saves checkpoints periodically

**Setup W&B (Optional but Recommended):**

```bash
# Login to W&B
wandb login

# Your API key will be saved for future runs
```

**Usage:**

> **⚠️ IMPORTANT:** W&B logging is **disabled by default**. You must explicitly add `--use_wandb` flag to enable it.

```bash
# Train WITH W&B logging (must specify --use_wandb)
uv run scripts/train.py --dataset tinystories --device cuda --use_wandb

# Train WITHOUT W&B (default - no flag needed)
uv run scripts/train.py --dataset tinystories --device cuda
```

**Logged Metrics:**
- `train/loss` - Training loss (per batch)
- `val/loss` - Validation loss (averaged over eval_iters batches)
- `train/lr` - Current learning rate
- `train/grad_norm` - Gradient norm (before clipping)
- `train/tokens_per_sec` - Training throughput
- `train/time_elapsed` - Wall-clock time since start

**Console Output:**
```
Iter  1000 | loss: 8.2341 | lr: 0.0003 | time: 45.2s
Iter  2000 | loss: 6.5432 | lr: 0.0003 | time: 90.5s
Iter  2000 | val_loss: 6.4123 | train_loss: 6.5432 | lr: 0.0003
```

**W&B Dashboard:**
- View real-time training curves
- Compare multiple runs
- Export data as CSV for plotting

**Deliverable Notes:**
- Include W&B dashboard screenshot or exported CSV
- Show train/val loss curves over time
- Document your config (learning rate, batch size, etc.)

---

### 3. TinyStories Pretraining (Base)

**Deliverable:** Train ~17M non-embedding param model on TinyStories. Achieve ≤1.45 validation loss. Provide curves and tuned config.

**Target Architecture:**
- Vocabulary: ~10,000 tokens
- Context length: 256
- d_model: 512
- d_ff: 1344 (≈ 8/3 × d_model, multiple of 64)
- Layers: 4
- Heads: 16
- RoPE: θ = 10000
- **Non-embedding params: ~17M**

**Training Setup:**
- Total tokens: 327,680,000
- Batch size: 32
- Context length: 256
- Total steps: 40,000 (= 327,680,000 / (32 × 256))
- Learning rate: Tune (suggested: 3e-4 to 6e-4)
- Warmup: 2000 steps
- Weight decay: 0.1
- AdamW: β1=0.9, β2=0.95, ε=1e-8
- Gradient clipping: 1.0

**Usage:**

```bash
# Train base model with default config
uv run scripts/train.py \
    --dataset tinystories \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --d_ff 1344 \
    --num_layers 4 \
    --num_heads 16 \
    --learning_rate 3e-4 \
    --batch_size 32 \
    --max_iters 40000 \
    --warmup_iters 2000 \
    --device cuda

# Or use the config defaults (already set for this experiment)
uv run scripts/train.py --dataset tinystories --device cuda
```

**Hyperparameter Tuning:**

Try different learning rates to find the best one:

```bash
# Try LR = 3e-4
uv run scripts/train.py --dataset tinystories --learning_rate 3e-4 --device cuda

# Try LR = 5e-4
uv run scripts/train.py --dataset tinystories --learning_rate 5e-4 --device cuda

# Try LR = 6e-4
uv run scripts/train.py --dataset tinystories --learning_rate 6e-4 --device cuda
```

**Expected Results:**
- Training loss: ~1.3-1.4
- Validation loss: ≤1.45 (target)
- Training time: ~30-40 minutes on H100 GPU

**Deliverable Notes:**
- Report final train/val loss
- Show learning curves (train/val loss vs. steps)
- Document your tuned hyperparameters (LR, warmup, etc.)
- Verify model has ~17M non-embedding params (logged at start of training)

---

### 4. Learning Rate Sweep

**Deliverable:** Run multiple LRs to convergence/divergence. Show curves with at least one divergent run + analysis of "edge of stability".

**Implementation:** `experiments/learning_rate_sweep.py`

**Usage:**

```bash
# Grid sweep: Test multiple learning rates
uv run experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cuda

# Stability sweep: Find edge of stability
uv run experiments/learning_rate_sweep.py \
    --sweep_type stability \
    --device cuda

# Both sweeps
uv run experiments/learning_rate_sweep.py \
    --sweep_type both \
    --device cuda
```

**Grid Sweep LRs:** `[1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]`

**Stability Sweep LRs:** Gradually increase LR until divergence

**Expected Behavior:**
- **Too low LR** (1e-5, 5e-5): Slow convergence, high final loss
- **Optimal LR** (3e-4, 5e-4): Fast convergence, low final loss
- **High LR** (1e-3): Slower convergence or slight instability
- **Too high LR** (3e-3, 5e-3): Divergence (loss → NaN or explodes)

**Deliverable Notes:**
- Show learning curves for all LRs on same plot
- Identify which LRs diverge (loss explodes or becomes NaN)
- Discuss "edge of stability": highest LR that still converges
- Analyze trade-off: higher LR → faster convergence but less stable

**Runtime:** ~4-6 hours for full sweep on H100 GPU

---

### 5. Batch Size Sweep

**Deliverable:** Test batch sizes from 1 up to memory limit. Retune LR if needed. Provide curves + discussion.

**Implementation:** `experiments/batch_size_sweep.py`

**Step 1: Find Max Batch Size**

```bash
# Binary search to find max batch size before OOM
uv run scripts/find_max_batch_size.py --device cuda
```

This will output something like:
```
Max batch size: 128 (fits in GPU memory)
```

**Step 2: Run Batch Size Sweep**

```bash
# Test batch sizes: 1, 2, 4, 8, 16, 32, 64, 128
uv run experiments/batch_size_sweep.py --device cuda

# Or specify custom batch sizes
uv run experiments/batch_size_sweep.py \
    --batch_sizes 1,4,16,32,64 \
    --device cuda
```

**Important:** You may need to retune learning rate for different batch sizes:
- Larger batch size → can use higher LR (more stable gradients)
- Smaller batch size → need lower LR (noisier gradients)

**Expected Behavior:**
- **Small batch (1-4)**: Noisy gradients, slower convergence, may need lower LR
- **Medium batch (16-32)**: Good balance, stable training
- **Large batch (64-128)**: Smooth gradients, faster per-step, but fewer steps per epoch

**Deliverable Notes:**
- Show learning curves for different batch sizes
- Discuss trade-offs: training speed, convergence, memory usage
- Note if you retuned LR for different batch sizes
- Compare final validation loss across batch sizes

**Runtime:** ~3-5 hours for full sweep on H100 GPU

---

### 6. Generate from TinyStories Model

**Deliverable:** Use your decoder to generate 256+ tokens. Adjust temperature/top-p for fluent samples. Discuss 2+ factors affecting quality.

**Usage:**

```bash
# After training, generate from your model
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --vocab artifacts/tinystories_vocab.json \
    --merges artifacts/tinystories_merges.txt \
    --prompt "Once upon a time, there was a little girl named Lily" \
    --temperature 0.8 \
    --top-p 0.9 \
    --max-tokens 256
```

**Experiment with Different Settings:**

```bash
# Low temperature (more deterministic, coherent)
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --vocab artifacts/tinystories_vocab.json \
    --merges artifacts/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --temperature 0.3 \
    --max-tokens 256

# High temperature (more creative, diverse)
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --vocab artifacts/tinystories_vocab.json \
    --merges artifacts/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --temperature 1.2 \
    --max-tokens 256

# Top-p sampling (nucleus sampling)
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --vocab artifacts/tinystories_vocab.json \
    --merges artifacts/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --temperature 0.8 \
    --top-p 0.95 \
    --max-tokens 256
```

**Factors Affecting Quality:**

1. **Temperature:**
   - Low (0.1-0.5): More deterministic, coherent, but repetitive
   - Medium (0.6-0.9): Balanced creativity and coherence
   - High (1.0+): More diverse, creative, but less coherent

2. **Top-p (Nucleus Sampling):**
   - Low (0.5-0.7): Only sample from most likely tokens → more focused
   - Medium (0.8-0.9): Good balance
   - High (0.95-1.0): Sample from broader distribution → more diverse

3. **Model Quality (Validation Loss):**
   - Lower val loss → better generation quality
   - Model trained longer → more coherent stories

4. **Prompt Quality:**
   - Clear, specific prompts → better continuations
   - Prompts similar to training data → better results

**Deliverable Notes:**
- Generate 3-5 samples with different temperature/top-p settings
- Include the generated text (256+ tokens each)
- Discuss at least 2 factors affecting quality (e.g., temperature, top-p, model quality, prompt)
- Compare samples qualitatively (coherence, creativity, repetition)

---

### 7. Ablation Studies

**Deliverable:** Test architectural variants and compare with baseline. Provide curves + commentary for each ablation.

**Implementation:** `experiments/ablations.py` + `cs336_basics/ablation_models.py`

#### Ablation 1: Remove RMSNorm

Test training stability without layer normalization. Try to restabilize with lower LR.

```bash
# Run no_rmsnorm ablation
uv run experiments/ablations.py --ablation no_rmsnorm --device cuda

# Try with lower LR to stabilize
uv run experiments/ablations.py \
    --ablation no_rmsnorm \
    --learning_rate 1e-4 \
    --device cuda
```

**Expected Behavior:**
- Training likely unstable or diverges
- Lower LR may help but still worse than baseline
- Demonstrates importance of layer normalization

#### Ablation 2: Pre-norm vs Post-norm

Compare pre-norm (baseline) with post-norm architecture.

```bash
# Run post_norm ablation
uv run experiments/ablations.py --ablation post_norm --device cuda
```

**Expected Behavior:**
- Post-norm: Normalization after residual connection
- Pre-norm (baseline): Normalization before attention/FFN
- Pre-norm typically more stable and converges better

#### Ablation 3: No Positional Encoding (NoPE) vs RoPE

Remove positional encodings entirely and compare with RoPE baseline.

```bash
# Run no_rope ablation (NoPE)
uv run experiments/ablations.py --ablation no_rope --device cuda
```

**Expected Behavior:**
- Without positional info, model can't distinguish token positions
- Worse performance on tasks requiring positional understanding
- RoPE baseline should significantly outperform

#### Ablation 4: SwiGLU vs SiLU FFN

Compare SwiGLU (baseline) with simple SiLU activation, matching parameter count.

```bash
# Run silu_only ablation
uv run experiments/ablations.py --ablation silu_only --device cuda
```

**Expected Behavior:**
- SwiGLU: Gated linear unit with SiLU activation
- SiLU: Simple SiLU(Wx + b) activation
- SwiGLU typically performs better (gating mechanism helps)

#### Run All Ablations

```bash
# Run all ablations sequentially
uv run experiments/ablations.py --ablation all --device cuda
```

**Deliverable Notes for Each Ablation:**
- Show learning curves comparing ablation vs baseline
- Report final train/val loss
- Discuss why the ablation helps or hurts performance
- For no_rmsnorm: Show attempts to restabilize with lower LR

**Runtime:** ~2-3 hours per ablation on H100 GPU

---

### 8. OpenWebText Training

**Deliverable:** Train on OWT with same architecture and total tokens as TinyStories. Provide learning curve, generated sample, and comparison/interpretation.

**Setup:**

Same architecture as TinyStories base:
- Vocabulary: 32,000 tokens (OWT BPE tokenizer)
- Context length: 256
- d_model: 512
- d_ff: 1344
- Layers: 4
- Heads: 16
- RoPE: θ = 10000
- Total tokens: 327,680,000
- Batch size: 32
- Total steps: 40,000

**Usage:**

```bash
# Train on OpenWebText
uv run scripts/train.py \
    --dataset owt \
    --vocab_size 32000 \
    --context_length 256 \
    --d_model 512 \
    --d_ff 1344 \
    --num_layers 4 \
    --num_heads 16 \
    --learning_rate 3e-4 \
    --batch_size 32 \
    --max_iters 40000 \
    --device cuda

# Or use defaults (already configured for OWT)
uv run scripts/train.py --dataset owt --device cuda
```

**Generate from OWT Model:**

```bash
uv run scripts/generate_text.py \
    --checkpoint checkpoints/owt/checkpoint_latest.pt \
    --tokenizer artifacts/owt_bpe.yaml \
    --prompt "The future of artificial intelligence" \
    --temperature 0.8 \
    --top-p 0.9 \
    --max-tokens 256
```

**Expected Results:**
- **Higher validation loss than TinyStories** (OWT is more diverse and complex)
- TinyStories val loss: ~1.3-1.45
- OWT val loss: ~3.5-4.0 (typical for this setup)

**Comparison: TinyStories vs OpenWebText**

| Aspect | TinyStories | OpenWebText |
|--------|-------------|-------------|
| **Domain** | Children's stories | General web text |
| **Complexity** | Simple vocabulary, repetitive | Diverse topics, complex |
| **Validation Loss** | Lower (~1.3-1.45) | Higher (~3.5-4.0) #

---

## Monitoring & Logging

### Weights & Biases (Recommended)

> **⚠️ IMPORTANT:** W&B logging is **disabled by default**. You must add `--use_wandb` flag when running `scripts/train.py` to enable W&B logging.

W&B provides real-time monitoring and experiment tracking.

**Setup:**
```bash
wandb login
```

**Features:**
- Real-time training curves
- Compare multiple runs
- Hyperparameter tracking
- System metrics (GPU usage, memory)
- Export data as CSV

**View Dashboard:**
After starting training, you'll see:
```
wandb: 🚀 View run at: https://wandb.ai/your-username/cs336-hw1/runs/abc123
```

### Console Logging

Training progress is printed to console:

```
================================================================================
Training Configuration
================================================================================
Model: TransformerLM
  - vocab_size: 10000
  - context_length: 256
  - d_model: 512
  - d_ff: 1344
  - num_layers: 4
  - num_heads: 16
  - Non-embedding params: 17,235,968

Training:
  - Total steps: 40000
  - Batch size: 32
  - Learning rate: 3e-4
  - Warmup: 2000 steps
================================================================================

Iter  100 | loss: 9.2341 | lr: 0.000015 | time: 5.2s
Iter  200 | loss: 8.5432 | lr: 0.000030 | time: 10.5s
Iter  500 | val_loss: 7.4123 | train_loss: 7.5432 | lr: 0.000075
...
```

### Checkpoints

Checkpoints are saved periodically:

```
checkpoints/
├── tinystories/
│   ├── checkpoint_iter_10000.pt
│   ├── checkpoint_iter_20000.pt
│   ├── checkpoint_iter_30000.pt
│   ├── checkpoint_iter_40000.pt
│   └── checkpoint_latest.pt  # Symlink to most recent
```

**Resume from Checkpoint:**
```bash
uv run scripts/train.py \
    --dataset tinystories \
    --resume_from checkpoints/tinystories/checkpoint_iter_20000.pt \
    --device cuda
```

---

## Troubleshooting

### Out of Memory (OOM)

**Find max batch size:**
```bash
uv run scripts/find_max_batch_size.py --device cuda
```

**Reduce batch size:**
```bash
uv run scripts/train.py --dataset tinystories --batch_size 16 --device cuda
```

**Reduce model size:**
```bash
uv run scripts/train.py \
    --dataset tinystories \
    --d_model 384 \
    --num_layers 3 \
    --device cuda
```

### Training Diverges (Loss → NaN)

**Symptoms:** Loss becomes NaN or explodes

**Solutions:**
1. **Lower learning rate:**
   ```bash
   uv run scripts/train.py --dataset tinystories --learning_rate 1e-4 --device cuda
   ```

2. **Increase warmup:**
   ```bash
   uv run scripts/train.py --dataset tinystories --warmup_iters 5000 --device cuda
   ```

3. **Check gradient clipping:**
   ```bash
   uv run scripts/train.py --dataset tinystories --grad_clip_norm 0.5 --device cuda
   ```

### Slow Training

**Use GPU:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Train on GPU
uv run scripts/train.py --dataset tinystories --device cuda
```

**Increase batch size (if memory allows):**
```bash
uv run scripts/train.py --dataset tinystories --batch_size 64 --device cuda
```

### Dataset Not Found

**Error:** `FileNotFoundError: Training data not found`

**Solution:** Encode datasets first:
```bash
# For TinyStories
uv run scripts/encode_dataset.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output data/tinystories_train_tokens.npy \
    --tokenizer artifacts/tinystories_bpe.yaml

# For OpenWebText
uv run scripts/encode_dataset.py \
    --input data/owt_train.txt \
    --output data/owt_train_tokens.npy \
    --tokenizer artifacts/owt_bpe.yaml
```

### CUDA Out of Memory During Generation

**Reduce batch size or use CPU:**
```bash
uv run scripts/generate_text.py \
    --checkpoint checkpoints/tinystories/checkpoint_latest.pt \
    --vocab artifacts/tinystories_vocab.json \
    --merges artifacts/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --device cpu \
    --max-tokens 256
```

