# Experiments

This directory contains scripts for running various experiments including dataset comparison, learning rate sweeps, batch size sweeps, and architecture ablations.

## Overview

The experiments train a small Transformer language model on the TinyStories dataset with the following specifications:

- **Model Architecture:**
  - `d_model`: 512
  - `d_ff`: 1344 (≈ 8/3 × d_model, multiple of 64)
  - `num_layers`: 4
  - `num_heads`: 16
  - RoPE positional embeddings with θ = 10000
  - ~17M non-embedding parameters

- **Training:**
  - Total tokens: 327,680,000
  - Context length: 256
  - Batch size: 32
  - Total steps: 40,000
  - Target validation loss: ≤ 1.45 per-token

## Setup

### 1. Install Dependencies

```bash
# Install required packages
uv pip install tokenizers tqdm
```

### 2. Prepare TinyStories Dataset

The TinyStories dataset needs to be downloaded and tokenized before training.

#### Option A: Automatic (Recommended)

```bash
# Download, extract, tokenize, and save
uv run python scripts/prepare_tinystories.py --vocab_size 10000 --output_dir data
```

This will create:
- `data/TinyStories_train.npy` - Training tokens
- `data/TinyStories_valid.npy` - Validation tokens
- `data/tokenizer_v10000.json` - BPE tokenizer

#### Option B: Manual Download

If automatic download fails:

1. Download TinyStories from: https://huggingface.co/datasets/roneneldan/TinyStories
2. Place the files in `data/TinyStories_raw/`
3. Run: `uv run python scripts/prepare_tinystories.py --skip_download --vocab_size 10000`

### 3. Login to Weights & Biases

```bash
wandb login
```

Enter your API key when prompted. This enables experiment tracking and visualization.

## Running Experiments

### 1. OpenWebText Training & Comparison

Train on OpenWebText and compare with TinyStories to understand dataset effects.

#### Quick Test (100 iterations)

```bash
# Test OWT training only (using main train.py script)
uv run python train.py --device cuda --max_iters 100

# Test both datasets for comparison
uv run python experiments/compare_datasets.py --device cuda --max_iters 100
```

#### Full Training (5000 iterations, same as TinyStories)

```bash
# Train on OWT only (using main train.py script)
uv run python train.py --device cuda --use_wandb

# Train on TinyStories (using main train.py script)
uv run python train.py --train_data data/tinystories_train_tokens.npy \
                       --val_data data/tinystories_valid_tokens.npy \
                       --vocab_size 10000 --device cuda --use_wandb

# Compare both datasets (recommended)
uv run python experiments/compare_datasets.py --device cuda --use_wandb
```

**Expected Results:**
- OpenWebText will have **higher loss** than TinyStories
- OWT is more diverse, complex, and realistic → harder to model
- TinyStories is simpler and more repetitive → easier to model
- Lower loss ≠ better model (depends on target domain)

---

### 2. Learning Rate Sweep

#### Grid Sweep

Perform a hyperparameter sweep over multiple learning rates:

```bash
# Full grid sweep (will take several hours)
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cuda
```

This tests learning rates: `[1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]`

#### Stability Sweep

Find the "edge of stability" by gradually increasing learning rate until divergence:

```bash
# Stability sweep
uv run python experiments/learning_rate_sweep.py \
    --sweep_type stability \
    --device cuda
```

#### Both Sweeps

Run both grid and stability sweeps sequentially:

```bash
uv run python experiments/learning_rate_sweep.py \
    --sweep_type both \
    --device cuda
```

### Running Without W&B

If you don't want to use Weights & Biases:

```bash
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cuda \
    --no_wandb
```

### CPU Training

For CPU:

```bash
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cpu
```

**Note:** CPU training will be significantly slower than GPU training.

## Expected Runtime

On an H100 GPU with the full configuration:
- **Single run (40,000 steps)**: ~30-40 minutes
- **Grid sweep (8 learning rates)**: ~4-5 hours
- **Stability sweep (10 learning rates)**: ~5-6 hours

On CPU/MPS, expect significantly longer runtimes.

## Monitoring Training

### Weights & Biases Dashboard

If using W&B, you can monitor training in real-time:

1. After starting training, you'll see a URL like:
   ```
   wandb: 🚀 View run at: https://wandb.ai/your-username/cs336-lr-sweep/runs/abc123
   ```

2. Click the link to view:
   - Training and validation loss curves
   - Learning rate schedule
   - Gradient norms
   - Model parameter statistics

### Local Logs

Training progress is also printed to the console:

```
Iter  1000 | loss: 8.2341 | lr: 0.0003
Iter  2000 | loss: 6.5432 | lr: 0.0003
Iter  2000 | val_loss: 6.4123 | train_loss: 6.5432 | lr: 0.0003
```

### Checkpoints

Model checkpoints are saved periodically to `checkpoints/lr_sweep/<run_name>/`:

```
checkpoints/lr_sweep/
├── lr_1e_04/
│   ├── checkpoint_iter_10000.pt
│   ├── checkpoint_iter_20000.pt
│   └── ...
├── lr_3e_04/
│   └── ...
└── ...
```

## Analyzing Results

### Deliverable (a): Learning Curves

After running the grid sweep, you can:

1. **View in W&B**: Go to your project page and compare runs
   - Select multiple runs
   - Compare validation loss curves
   - Identify which learning rates converge best

2. **Export data**: Download CSV from W&B for plotting

3. **Key metrics to report**:
   - Final validation loss for each learning rate
   - Which learning rates diverged (if any)
   - Best learning rate based on final validation loss

### Deliverable (b): Edge of Stability

After running the stability sweep:

1. **Identify divergence point**: Find the learning rate where training becomes unstable
2. **Best stable LR**: The highest learning rate that still converges
3. **Analysis**: Discuss how this relates to convergence rates

## Troubleshooting

### Dataset Not Found

Make sure you've run the data preparation script:

```bash
uv run python scripts/prepare_tinystories.py --vocab_size 10000
```

### Out of Memory (OOM)

If you get OOM errors, you can find the maximum batch size that fits in your GPU:

```bash
# Find max batch size using binary search
uv run python scripts/find_max_batch_size.py --device cuda

# Then use that batch size in your experiments
uv run python experiments/batch_size_sweep.py --device cuda --batch_sizes 16,32,64
```

### Training Diverges Immediately

This usually means the learning rate is too high. Try a smaller learning rate in the sweep.

### Slow Training

- Make sure you're using GPU: `--device cuda`
- Check that CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## File Structure

```
experiments/
├── README.md                    # This file
├── experiment_utils.py          # Shared utilities for experiments
├── compare_datasets.py          # Compare TinyStories vs OpenWebText
├── learning_rate_sweep.py       # Learning rate sweep experiments
├── batch_size_sweep.py          # Batch size sweep experiments
└── ablations.py                 # Architecture ablation experiments

scripts/
├── prepare_tinystories.py       # Dataset preparation
└── find_max_batch_size.py       # Find max batch size before OOM

checkpoints/
├── tinystories/                 # TinyStories checkpoints
├── owt/                         # OpenWebText checkpoints
├── lr_sweep/                    # LR sweep checkpoints
└── ablations/                   # Ablation checkpoints

data/
├── tinystories_train_tokens.npy  # TinyStories training tokens
├── tinystories_valid_tokens.npy  # TinyStories validation tokens
├── owt_train_tokens.npy          # OpenWebText training tokens
├── owt_valid_tokens.npy          # OpenWebText validation tokens
└── tokenizer_v10000.json         # BPE tokenizer

Note: For single-dataset training, use the main train.py script in the root directory.
```

## Tips

1. **Use W&B**: It makes comparing runs much easier
2. **Monitor early**: Check the first few hundred iterations - if loss isn't decreasing, stop and adjust
3. **Save results**: W&B automatically saves everything, but you can also export CSVs
4. **Document**: Keep notes on what you observe for your report

## Assignment Deliverables

For the assignment, you need to submit:

1. **Learning curves** showing validation loss vs. training steps for multiple learning rates
2. **Analysis** of your hyperparameter search strategy
3. **Best model** with validation loss ≤ 1.45 per-token
4. **Stability analysis** showing learning curves with at least one divergent run
5. **Discussion** of how divergence point relates to best learning rate

All of this can be generated from the W&B dashboard after running the experiments!

