# 🚀 OpenWebText Experiments Guide

Complete guide for running OpenWebText experiments with 40K iterations and eval every 500 steps.

---

## 📋 Quick Reference

All experiments use:
- **Max iterations:** 40,000
- **Eval interval:** 500 steps
- **Device:** CUDA (configurable)
- **W&B logging:** Enabled by default

---

## 🎯 Main Experiments

### 1. Train on OpenWebText Only

```bash
# Full run with W&B
./run_owt_experiments.sh

# Without W&B
./run_owt_experiments.sh --no-wandb

# On CPU (slow!)
./run_owt_experiments.sh --device cpu

# Custom iterations
./run_owt_experiments.sh --max-iters 10000 --eval-interval 250
```

**What it does:**
1. Trains a model on OpenWebText (40K iterations)
2. Compares TinyStories vs OpenWebText side-by-side

**Expected runtime:** ~6-8 hours on H100 (for both experiments)

**Output:**
- Checkpoints: `checkpoints/owt/` and `checkpoints/tinystories/`
- W&B dashboard with learning curves

---

## 🔬 Learning Rate Sweep

### Run LR Sweep on OpenWebText

```bash
# Full sweep with default LRs
./run_owt_lr_sweep.sh

# Without W&B
./run_owt_lr_sweep.sh --no-wandb

# Custom iterations
./run_owt_lr_sweep.sh --max-iters 10000 --eval-interval 250
```

**Learning rates tested:**
- 1e-4 (conservative)
- 3e-4 (common default)
- 5e-4 (moderate)
- 1e-3 (aggressive)
- 3e-3 (very aggressive)

**Expected runtime:** ~30-40 hours on H100 (5 runs × 6-8 hours each)

**Output:**
- Checkpoints: `checkpoints/owt_lr_sweep/lr_*/`
- W&B project: `cs336-owt-lr-sweep`

**To customize learning rates:**
Edit `run_owt_lr_sweep.sh` line 14:
```bash
LR_VALUES=(1e-4 3e-4 5e-4 1e-3 3e-3)  # Modify this array
```

---

## 📦 Batch Size Sweep

### Run Batch Size Sweep on OpenWebText

```bash
# Full sweep with default batch sizes
./run_owt_batch_sweep.sh

# Without W&B
./run_owt_batch_sweep.sh --no-wandb

# With custom learning rate
./run_owt_batch_sweep.sh --learning-rate 0.0003

# Custom iterations
./run_owt_batch_sweep.sh --max-iters 10000 --eval-interval 250
```

**Batch sizes tested:**
- 8 (small)
- 16 (small-medium)
- 32 (default)
- 64 (large)
- 128 (very large, may OOM)

**Expected runtime:** ~30-40 hours on H100 (5 runs × 6-8 hours each)

**Output:**
- Checkpoints: `checkpoints/owt_batch_sweep/bs_*/`
- W&B project: `cs336-owt-batch-sweep`

**To customize batch sizes:**
Edit `run_owt_batch_sweep.sh` line 14:
```bash
BATCH_SIZES=(8 16 32 64 128)  # Modify this array
```

**💡 Tip:** Find your GPU's max batch size first:
```bash
uv run python scripts/find_max_batch_size.py --device cuda
```

---

## 🧪 Individual Experiment Commands

If you prefer to run experiments individually:

### Single OWT Training

```bash
uv run python experiments/train_owt.py \
    --device cuda \
    --max_iters 40000 \
    --eval_interval 500 \
    --use_wandb
```

### Compare TinyStories vs OWT

```bash
uv run python experiments/compare_datasets.py \
    --device cuda \
    --max_iters 40000 \
    --eval_interval 500 \
    --use_wandb
```

### Single LR Experiment

```bash
uv run python experiments/train_owt.py \
    --device cuda \
    --max_iters 40000 \
    --eval_interval 500 \
    --learning_rate 0.0003 \
    --checkpoint_dir checkpoints/owt_lr_3e-4 \
    --wandb_project cs336-owt-lr-sweep \
    --wandb_run_name owt_lr_3e-4 \
    --use_wandb
```

### Single Batch Size Experiment

```bash
uv run python experiments/train_owt.py \
    --device cuda \
    --max_iters 40000 \
    --eval_interval 500 \
    --batch_size 64 \
    --checkpoint_dir checkpoints/owt_bs_64 \
    --wandb_project cs336-owt-batch-sweep \
    --wandb_run_name owt_bs_64 \
    --use_wandb
```

---

## 📊 Expected Results

### Validation Loss Ranges

| Dataset | Initial Loss | Final Loss (40K iters) |
|---------|--------------|------------------------|
| TinyStories | ~9.2 | ~1.2-1.4 |
| OpenWebText | ~10.4 | ~3.0-3.5 |

**Why is OWT loss higher?**
- More diverse vocabulary (32K vs 10K tokens)
- More complex language patterns
- More varied topics and domains
- Less repetitive structure

### Learning Rate Impact

| Learning Rate | Expected Behavior |
|---------------|-------------------|
| 1e-4 | Slow but stable convergence |
| 3e-4 | Good balance (common default) |
| 5e-4 | Faster convergence, still stable |
| 1e-3 | Fast but may be unstable |
| 3e-3 | Likely too aggressive, may diverge |

### Batch Size Impact

| Batch Size | Training Speed | Memory Usage | Convergence |
|------------|----------------|--------------|-------------|
| 8 | Slow | Low | Noisy gradients |
| 16 | Moderate | Low-Medium | Moderate noise |
| 32 | Good | Medium | Balanced |
| 64 | Fast | High | Smoother |
| 128 | Very fast | Very high | Very smooth |

**Trade-off:** Larger batch sizes = faster training but require more memory and may need LR adjustment.

---

## 🎓 Assignment Deliverables

For the OpenWebText assignment, you need:

1. **Learning curve of OWT training** ✅
   - Run: `./run_owt_experiments.sh`
   - Export learning curve from W&B

2. **Comparison with TinyStories** ✅
   - Automatically included in `run_owt_experiments.sh`
   - Shows both datasets on same plot

3. **Interpretation of loss differences** ✅
   - See "Expected Results" section above
   - Key point: Higher loss ≠ worse model, just harder dataset

**Minimal command to complete assignment:**
```bash
./run_owt_experiments.sh
```

This runs both required experiments in one go!

---

## 💾 Disk Space Requirements

| Experiment | Checkpoints | Approx. Size |
|------------|-------------|--------------|
| Single OWT run | ~80 checkpoints | ~3-4 GB |
| LR sweep (5 runs) | ~400 checkpoints | ~15-20 GB |
| Batch sweep (5 runs) | ~400 checkpoints | ~15-20 GB |

**💡 Tip:** Checkpoints are saved every 1000 iterations by default. Adjust with `--checkpoint_interval` if needed.

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Find max batch size for your GPU
uv run python scripts/find_max_batch_size.py --device cuda

# Use smaller batch size
./run_owt_batch_sweep.sh --learning-rate 0.001
# Then edit the script to use smaller batch sizes: (4 8 16 32)
```

### Training Too Slow

```bash
# Reduce iterations for testing
./run_owt_experiments.sh --max-iters 5000 --eval-interval 250

# Or use smaller model (edit train_owt.py to reduce num_layers, d_model)
```

### W&B Login Issues

```bash
# Login to W&B first
wandb login

# Or disable W&B
./run_owt_experiments.sh --no-wandb
```

### CUDA Not Available

```bash
# Use CPU (much slower!)
./run_owt_experiments.sh --device cpu --max-iters 1000
```

---

## 📁 File Structure

```
cs336-hw1/
├── run_owt_experiments.sh      # Main experiments
├── run_owt_lr_sweep.sh         # Learning rate sweep
├── run_owt_batch_sweep.sh      # Batch size sweep
├── experiments/
│   ├── train_owt.py            # Single OWT training
│   ├── compare_datasets.py     # Compare TS vs OWT
│   ├── learning_rate_sweep.py  # LR sweep (for TinyStories)
│   └── batch_size_sweep.py     # Batch sweep (for TinyStories)
├── checkpoints/
│   ├── owt/                    # OWT checkpoints
│   ├── tinystories/            # TinyStories checkpoints
│   ├── owt_lr_sweep/           # LR sweep checkpoints
│   └── owt_batch_sweep/        # Batch sweep checkpoints
└── data/
    ├── owt_train_tokens.npy    # OWT training data
    ├── owt_valid_tokens.npy    # OWT validation data
    ├── tinystories_train_tokens.npy
    └── tinystories_valid_tokens.npy
```

---

## 🎯 Summary

**For the assignment (minimal):**
```bash
./run_owt_experiments.sh
```

**For comprehensive analysis:**
```bash
./run_owt_experiments.sh      # Main experiments
./run_owt_lr_sweep.sh         # LR sweep
./run_owt_batch_sweep.sh      # Batch sweep
```

**For quick testing:**
```bash
./run_owt_experiments.sh --max-iters 1000 --eval-interval 100
```

🎉 **You're all set to run OpenWebText experiments!**

