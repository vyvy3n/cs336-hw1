# Quick Start Guide

## 🚀 Running Experiments

### 1. Prepare Data (First Time Only)

```bash
uv run python scripts/prepare_tinystories.py --vocab_size 10000
```

### 2. Find Maximum Batch Size (Optional but Recommended)

```bash
# Find the max batch size your GPU can handle
uv run python scripts/find_max_batch_size.py --device cuda
```

This will output something like:
```
Maximum batch size: 128
Peak memory usage: 15.2 GB
```

### 3. Run Quick Tests

Test each experiment with reduced iterations to verify everything works:

```bash
# Test learning rate sweep (grid)
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cuda \
    --max_iters 100

# Test learning rate sweep (stability)
uv run python experiments/learning_rate_sweep.py \
    --sweep_type stability \
    --device cuda \
    --max_iters 100

# Test batch size sweep
uv run python experiments/batch_size_sweep.py \
    --device cuda \
    --max_iters 100

# Test one ablation
uv run python experiments/ablations.py \
    --ablation layer_norm \
    --device cuda \
    --max_iters 100
```

### 4. Run Full Experiments

Once tests pass, run the full experiments:

```bash
# Learning rate grid sweep
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cuda \
    --use_wandb

# Learning rate stability sweep
uv run python experiments/learning_rate_sweep.py \
    --sweep_type stability \
    --device cuda \
    --use_wandb

# Batch size sweep
uv run python experiments/batch_size_sweep.py \
    --device cuda \
    --use_wandb

# All ablations
for ablation in layer_norm pre_norm no_pos_emb swiglu; do
    uv run python experiments/ablations.py \
        --ablation $ablation \
        --device cuda \
        --use_wandb
done
```

---

## 📁 Project Structure

```
cs336-hw1/
├── cs336_basics/              # Core library (models, layers, training)
├── experiments/               # Experiment scripts (LR sweep, ablations, etc.)
├── scripts/                   # Utility scripts (data prep, find max batch)
├── data/                      # Training/validation data
├── checkpoints/               # Model checkpoints
└── wandb/                     # W&B logs
```

**See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed explanation.**

---

## 🛠️ Utility Scripts

### Find Maximum Batch Size

```bash
# Automatic search
uv run python scripts/find_max_batch_size.py --device cuda

# With custom upper bound
uv run python scripts/find_max_batch_size.py \
    --device cuda \
    --upper_bound 512
```

### Prepare Dataset

```bash
# Default vocab size (10000)
uv run python scripts/prepare_tinystories.py --vocab_size 10000

# Custom vocab size
uv run python scripts/prepare_tinystories.py --vocab_size 5000
```

---

## 🧪 Experiment Scripts

### Learning Rate Sweep

```bash
# Grid search over predefined LRs
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cuda

# Stability search (find edge of stability)
uv run python experiments/learning_rate_sweep.py \
    --sweep_type stability \
    --device cuda
```

**Options:**
- `--max_iters`: Number of training iterations (default: 5000)
- `--eval_interval`: Evaluation frequency (default: 100)
- `--use_wandb`: Enable W&B logging
- `--device`: Device to use (cuda/cpu)

### Batch Size Sweep

```bash
# Default batch sizes
uv run python experiments/batch_size_sweep.py --device cuda

# With LR optimization per batch size
uv run python experiments/batch_size_sweep.py \
    --device cuda \
    --optimize_lr

# Custom batch sizes
uv run python experiments/batch_size_sweep.py \
    --device cuda \
    --batch_sizes 16,32,64,128
```

### Ablation Studies

```bash
# Single ablation
uv run python experiments/ablations.py \
    --ablation layer_norm \
    --device cuda

# All ablations
for ablation in layer_norm pre_norm no_pos_emb swiglu; do
    uv run python experiments/ablations.py \
        --ablation $ablation \
        --device cuda \
        --use_wandb
done
```

**Available ablations:**
- `layer_norm`: Remove all RMSNorm layers
- `pre_norm`: Switch from pre-norm to post-norm
- `no_pos_emb`: Remove positional embeddings (RoPE)
- `swiglu`: Replace SwiGLU with SiLU-only FFN

---

## 💡 Tips

1. **Always run quick tests first** with `--max_iters 100` to verify setup
2. **Use W&B** (`--use_wandb`) for experiment tracking and comparison
3. **Find max batch size** before running batch size sweeps
4. **Monitor GPU memory** with `nvidia-smi` during experiments
5. **Check data paths** - make sure data files exist before running

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Find your max batch size
uv run python scripts/find_max_batch_size.py --device cuda

# Use smaller batch size in experiments
uv run python experiments/batch_size_sweep.py \
    --device cuda \
    --batch_sizes 16,32,64
```

### Dataset Not Found

```bash
# Prepare the dataset
uv run python scripts/prepare_tinystories.py --vocab_size 10000

# Verify files exist
ls -lh data/tinystories_*.npy
```

### CUDA Not Available

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU instead
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cpu
```

### Training Diverges

- Learning rate is too high
- Try smaller learning rates in the sweep
- Check initial loss values (should be ~log(vocab_size) ≈ 9.2 for vocab_size=10000)

---

## 📊 Expected Results

### Learning Rate Sweep
- **Time**: ~2-4 hours for full grid sweep (8 LRs × 5000 iters)
- **Output**: W&B dashboard with loss curves for each LR
- **Goal**: Find optimal learning rate

### Batch Size Sweep
- **Time**: ~1-2 hours for 4-5 batch sizes
- **Output**: W&B dashboard comparing batch sizes
- **Goal**: Understand batch size vs. training dynamics

### Ablations
- **Time**: ~30-60 minutes per ablation
- **Output**: W&B dashboard comparing architectures
- **Goal**: Understand importance of each component

---

## 📚 More Information

- **Experiments**: See [experiments/README.md](experiments/README.md)
- **Project Structure**: See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Code Requirements**: Only uses allowed PyTorch components (no torch.nn layers except Module/Parameter/containers)

