# Learning Rate Experiments

This directory contains scripts for running learning rate hyperparameter sweep experiments as specified in the assignment.

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

### Quick Test (Recommended First)

Test your setup with a short training run:

```bash
# Quick test with 1000 iterations (~5 minutes on GPU)
uv run python experiments/quick_lr_test.py \
    --learning_rate 3e-4 \
    --max_iters 1000 \
    --device cuda \
    --use_wandb

# Test without W&B
uv run python experiments/quick_lr_test.py \
    --learning_rate 3e-4 \
    --max_iters 1000 \
    --device cuda
```

### Full Learning Rate Sweep

#### Part (a): Grid Sweep

Perform a hyperparameter sweep over multiple learning rates:

```bash
# Full grid sweep (will take several hours)
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cuda

# Quick grid sweep for testing (1000 iters per LR)
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cuda \
    --max_iters 1000
```

This tests learning rates: `[1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]`

#### Part (b): Stability Sweep

Find the "edge of stability" by gradually increasing learning rate until divergence:

```bash
# Stability sweep
uv run python experiments/learning_rate_sweep.py \
    --sweep_type stability \
    --device cuda

# Quick stability sweep for testing
uv run python experiments/learning_rate_sweep.py \
    --sweep_type stability \
    --device cuda \
    --max_iters 1000
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

### CPU/MPS Training

For CPU or Apple Silicon (MPS):

```bash
# CPU
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cpu \
    --max_iters 5000

# Apple Silicon (MPS)
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device mps \
    --max_iters 5000
```

**Note:** As per assignment instructions, for CPU/MPS you may want to reduce total tokens to 40,000,000 and increase target validation loss to 2.00.

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

### Out of Memory (OOM)

If you get OOM errors:

```bash
# Reduce batch size
uv run python experiments/quick_lr_test.py \
    --batch_size 16 \
    --learning_rate 3e-4
```

### Dataset Not Found

Make sure you've run the data preparation script:

```bash
uv run python scripts/prepare_tinystories.py --vocab_size 10000
```

### Training Diverges Immediately

This usually means the learning rate is too high. Try:

```bash
# Test with a smaller learning rate
uv run python experiments/quick_lr_test.py \
    --learning_rate 1e-4 \
    --max_iters 1000
```

### Slow Training

- Make sure you're using GPU: `--device cuda`
- Check that CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce `eval_iters` and `eval_interval` for faster iteration

## File Structure

```
experiments/
├── README.md                    # This file
├── learning_rate_sweep.py       # Main experiment script
└── quick_lr_test.py            # Quick testing script

scripts/
└── prepare_tinystories.py      # Dataset preparation

checkpoints/
└── lr_sweep/                   # Saved model checkpoints

data/
├── TinyStories_train.npy       # Training tokens
├── TinyStories_valid.npy       # Validation tokens
└── tokenizer_v10000.json       # BPE tokenizer
```

## Tips

1. **Start small**: Run `quick_lr_test.py` first to verify everything works
2. **Use W&B**: It makes comparing runs much easier
3. **Monitor early**: Check the first few hundred iterations - if loss isn't decreasing, stop and adjust
4. **Save results**: W&B automatically saves everything, but you can also export CSVs
5. **Document**: Keep notes on what you observe for your report

## Assignment Deliverables

For the assignment, you need to submit:

1. **Learning curves** showing validation loss vs. training steps for multiple learning rates
2. **Analysis** of your hyperparameter search strategy
3. **Best model** with validation loss ≤ 1.45 per-token
4. **Stability analysis** showing learning curves with at least one divergent run
5. **Discussion** of how divergence point relates to best learning rate

All of this can be generated from the W&B dashboard after running the experiments!

