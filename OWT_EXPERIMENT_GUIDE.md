# OpenWebText Experiment Guide

## 📋 Assignment Question

**Problem (main_experiment): Experiment on OWT (2 points) (3 H100 hrs)**

> Train your language model on OpenWebText with the same model architecture and total training iterations as TinyStories. How well does this model do?
> 
> **Deliverable:** A learning curve of your language model on OpenWebText. Describe the difference in losses from TinyStories – how should we interpret these losses?

---

## 🎯 Quick Answer

**Use the existing experiment infrastructure with different data paths!**

We have two minimal scripts that reuse all existing code:
1. `experiments/train_owt.py` - Train on OpenWebText only
2. `experiments/compare_datasets.py` - Train on both datasets for comparison

---

## 🚀 Running the Experiment

### Option 1: Train on OpenWebText Only

```bash
# Quick test (100 iterations)
uv run python experiments/train_owt.py \
    --device cuda \
    --max_iters 100 \
    --eval_interval 50

# Full training (5000 iterations, same as TinyStories)
uv run python experiments/train_owt.py \
    --device cuda \
    --use_wandb
```

### Option 2: Compare Both Datasets (Recommended)

```bash
# Quick test (100 iterations each)
uv run python experiments/compare_datasets.py \
    --device cuda \
    --max_iters 100

# Full comparison (5000 iterations each)
uv run python experiments/compare_datasets.py \
    --device cuda \
    --use_wandb
```

### Option 3: Train Separately

```bash
# Train on TinyStories
uv run python experiments/compare_datasets.py \
    --dataset tinystories \
    --device cuda \
    --use_wandb

# Train on OpenWebText
uv run python experiments/compare_datasets.py \
    --dataset owt \
    --device cuda \
    --use_wandb
```

---

## 📊 What to Expect

### Dataset Statistics

```
OpenWebText:
  Train: 2,850,391,059 tokens (~2.85B)
  Valid: 69,465,323 tokens (~69M)
  Total: 2,919,856,382 tokens (~2.92B)

TinyStories:
  Train: 542,447,487 tokens (~542M)
  Valid: 5,478,232 tokens (~5.5M)
  Total: 547,925,719 tokens (~548M)
```

**Key difference:** OWT is ~5.3x larger and much more diverse.

---

### Expected Loss Comparison

| Dataset | Expected Final Loss | Interpretation |
|---------|-------------------|----------------|
| **TinyStories** | ~1.4 - 1.5 | Lower loss (simpler, repetitive text) |
| **OpenWebText** | ~3.0 - 3.5 | Higher loss (complex, diverse text) |

**Why is OWT loss higher?**

1. **More diverse vocabulary usage** - OWT uses the full vocabulary more uniformly
2. **More complex syntax** - Real web text has varied sentence structures
3. **More varied topics** - OWT covers many domains (sports, science, news, etc.)
4. **Less repetition** - TinyStories has repetitive patterns that are easier to learn
5. **More realistic language** - OWT reflects actual human writing

---

## 🧠 How to Interpret the Losses

### ❌ Common Misconception

> "Lower loss = better model"

### ✅ Correct Interpretation

**Loss measures how well the model fits the training distribution.**

- **TinyStories (lower loss):**
  - Model fits the simple, repetitive distribution well
  - Good for generating children's stories
  - Poor for general language understanding

- **OpenWebText (higher loss):**
  - Model faces a more challenging, diverse distribution
  - Better for general language tasks
  - More realistic language modeling

### 📈 What to Report

1. **Learning curves:** Plot train/val loss over iterations for both datasets
2. **Final losses:** Report final validation loss for each dataset
3. **Comparison:** Explain why OWT has higher loss
4. **Interpretation:** Discuss what this means for model quality

**Example interpretation:**

> "The OpenWebText model achieves a validation loss of 3.2, compared to 1.4 for TinyStories. This higher loss is expected because OpenWebText contains more diverse, complex, and realistic text. The TinyStories dataset is simpler and more repetitive, making it easier to model. However, the OWT model is likely more useful for general language tasks despite the higher loss, as it has been exposed to a more representative sample of natural language."

---

## ⚙️ Configuration

Both scripts use **identical model architecture** to TinyStories:

```python
Model Architecture:
  - vocab_size: 10,000
  - context_length: 256
  - num_layers: 4
  - d_model: 512
  - num_heads: 16
  - d_ff: 1,344 (512 * 2.625 for SwiGLU)
  - use_rope: True
  - theta: 10,000

Training:
  - learning_rate: 1e-3
  - batch_size: 32
  - max_iters: 5,000 (default)
  - warmup_iters: 100
  - optimizer: AdamW (β1=0.9, β2=0.95)
  - grad_clip_norm: 1.0
```

---

## 🎛️ Command-Line Options

### `train_owt.py`

```bash
python experiments/train_owt.py [OPTIONS]

Options:
  --device DEVICE              Device to use (cuda or cpu) [default: cuda]
  --max_iters MAX_ITERS        Total training iterations [default: 5000]
  --eval_interval EVAL_INTERVAL Evaluation frequency [default: 100]
  --learning_rate LR           Learning rate [default: 1e-3]
  --batch_size BS              Batch size [default: 32]
  --use_wandb                  Enable W&B logging
```

### `compare_datasets.py`

```bash
python experiments/compare_datasets.py [OPTIONS]

Options:
  --dataset {tinystories,owt,both}  Which dataset(s) to train [default: both]
  --device DEVICE                   Device to use [default: cuda]
  --max_iters MAX_ITERS            Total iterations per dataset [default: 5000]
  --eval_interval EVAL_INTERVAL    Evaluation frequency [default: 100]
  --learning_rate LR               Learning rate [default: 1e-3]
  --batch_size BS                  Batch size [default: 32]
  --use_wandb                      Enable W&B logging (recommended)
```

---

## 📁 Output Files

### Checkpoints

```
checkpoints/
├── tinystories/
│   ├── checkpoint_iter_1000.pt
│   ├── checkpoint_iter_2000.pt
│   └── ...
└── owt/
    ├── checkpoint_iter_1000.pt
    ├── checkpoint_iter_2000.pt
    └── ...
```

### Weights & Biases

If `--use_wandb` is enabled:
- **Project:** `cs336-dataset-comparison`
- **Runs:** 
  - `tinystories_lr1e-03_bs32`
  - `owt_lr1e-03_bs32`

You can compare learning curves side-by-side in the W&B dashboard.

---

## 💡 Tips

1. **Use W&B logging** (`--use_wandb`) for easy comparison
2. **Start with quick test** (100 iterations) to verify setup
3. **Monitor GPU memory** - OWT uses same memory as TinyStories
4. **Compare on same plot** - Use W&B to overlay learning curves
5. **Check initial loss** - Should be ~log(vocab_size) ≈ 9.2 for both

---

## 🐛 Troubleshooting

### Data files not found

```bash
# Check if OWT data exists
ls -lh data/owt_*.npy

# If missing, you need to prepare OWT data
# (Check with instructor for OWT preparation script)
```

### Out of memory

```bash
# Find max batch size
uv run python scripts/find_max_batch_size.py --device cuda

# Use smaller batch size
uv run python experiments/train_owt.py --device cuda --batch_size 16
```

### Training too slow

```bash
# Reduce iterations for testing
uv run python experiments/train_owt.py --device cuda --max_iters 1000

# Or use CPU (much slower)
uv run python experiments/train_owt.py --device cpu --max_iters 100
```

---

## 📚 Summary

**Minimal viable solution:**

1. ✅ **Reuse existing code** - No new model/training code needed
2. ✅ **Two simple scripts** - `train_owt.py` and `compare_datasets.py`
3. ✅ **Same architecture** - Identical model config as TinyStories
4. ✅ **Easy comparison** - W&B logging for side-by-side curves

**To answer the assignment:**

```bash
# Run this command
uv run python experiments/compare_datasets.py --device cuda --use_wandb

# Then report:
# 1. Learning curves from W&B
# 2. Final validation losses
# 3. Interpretation of why OWT has higher loss
```

**Expected answer:**
> "OWT achieves higher loss (~3.2) than TinyStories (~1.4) because it contains more diverse, complex, and realistic text. This doesn't mean the OWT model is worse - it's actually more useful for general language tasks. Lower loss simply means the model fits the training distribution better, and TinyStories has a simpler, more repetitive distribution."

