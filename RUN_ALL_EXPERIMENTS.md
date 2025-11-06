# 🚀 Run All OpenWebText Experiments

Complete guide for running all OWT experiments with a single command.

---

## 🎯 Quick Start

### Run ALL experiments (full version):
```bash
./run_all_owt_experiments.sh
```

**What it runs:**
1. ✅ Single OWT training (40K iters)
2. ✅ Compare TinyStories vs OWT (40K iters each)
3. ✅ Learning rate sweep (5 LRs × 40K iters)
4. ✅ Batch size sweep (5 batch sizes × 40K iters)

**Total runtime:** ~80-100 hours on H100

---

### Test first (recommended):
```bash
./test_all_owt_experiments.sh
```

**What it runs:**
1. ✅ Single OWT training (500 iters)
2. ✅ Compare TinyStories vs OWT (500 iters each)
3. ✅ Learning rate sweep (3 LRs × 500 iters)
4. ✅ Batch size sweep (3 batch sizes × 500 iters)

**Total runtime:** ~30-45 minutes on H100

---

## 📋 All Available Scripts

### 1. **`run_all_owt_experiments.sh`** - Complete Suite
Runs all 4 experiment groups in sequence.

```bash
# Full run with W&B
./run_all_owt_experiments.sh

# Without W&B
./run_all_owt_experiments.sh --no-wandb

# On CPU
./run_all_owt_experiments.sh --device cpu

# Custom iterations
./run_all_owt_experiments.sh --max-iters 10000 --eval-interval 250

# Skip confirmation prompt
./run_all_owt_experiments.sh --skip-confirmation
```

**Experiments included:**
- Single OWT training (40K iters, LR=0.001, BS=32)
- TinyStories vs OWT comparison (40K iters each)
- LR sweep: [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
- Batch size sweep: [8, 16, 32, 64, 128]

---

### 2. **`test_all_owt_experiments.sh`** - Quick Test
Tests all experiments with reduced iterations.

```bash
# Quick test (no W&B by default)
./test_all_owt_experiments.sh

# With W&B
./test_all_owt_experiments.sh --use-wandb

# Custom iterations
./test_all_owt_experiments.sh --max-iters 1000
```

**Experiments included:**
- Single OWT training (500 iters)
- TinyStories vs OWT comparison (500 iters each)
- LR sweep: [1e-4, 1e-3, 3e-3] (3 LRs only)
- Batch size sweep: [16, 32, 64] (3 batch sizes only)

---

### 3. **`run_owt_experiments.sh`** - Main Experiments Only
Runs only the core experiments (no sweeps).

```bash
./run_owt_experiments.sh
```

**Experiments included:**
- Single OWT training (40K iters)
- TinyStories vs OWT comparison (40K iters each)

**Runtime:** ~12-16 hours on H100

---

### 4. **`run_owt_lr_sweep.sh`** - Learning Rate Sweep Only
Runs only the learning rate sweep.

```bash
./run_owt_lr_sweep.sh
```

**Experiments included:**
- LR sweep: [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]

**Runtime:** ~30-40 hours on H100

---

### 5. **`run_owt_batch_sweep.sh`** - Batch Size Sweep Only
Runs only the batch size sweep.

```bash
./run_owt_batch_sweep.sh
```

**Experiments included:**
- Batch size sweep: [8, 16, 32, 64, 128]

**Runtime:** ~30-40 hours on H100

---

## 🎓 For the Assignment

**Minimal requirement:**
```bash
./run_owt_experiments.sh
```

This gives you:
- ✅ Learning curve on OpenWebText
- ✅ Comparison with TinyStories
- ✅ Interpretation of loss differences

**For comprehensive analysis:**
```bash
./run_all_owt_experiments.sh
```

This gives you everything above plus:
- ✅ Learning rate sensitivity analysis
- ✅ Batch size impact analysis
- ✅ Optimal hyperparameter recommendations

---

## 📊 What Gets Run

### Experiment 1: Single OWT Training
```bash
uv run python experiments/train_owt.py \
    --device cuda \
    --max_iters 40000 \
    --eval_interval 500 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --use_wandb
```

### Experiment 2: Compare Datasets
```bash
uv run python experiments/compare_datasets.py \
    --device cuda \
    --max_iters 40000 \
    --eval_interval 500 \
    --use_wandb
```

### Experiment 3: Learning Rate Sweep
For each LR in [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]:
```bash
uv run python experiments/train_owt.py \
    --device cuda \
    --max_iters 40000 \
    --eval_interval 500 \
    --learning_rate $LR \
    --batch_size 32 \
    --checkpoint_dir checkpoints/owt_lr_sweep/lr_$LR \
    --wandb_project cs336-owt-lr-sweep \
    --wandb_run_name owt_lr_$LR \
    --use_wandb
```

### Experiment 4: Batch Size Sweep
For each BS in [8, 16, 32, 64, 128]:
```bash
uv run python experiments/train_owt.py \
    --device cuda \
    --max_iters 40000 \
    --eval_interval 500 \
    --learning_rate 0.001 \
    --batch_size $BS \
    --checkpoint_dir checkpoints/owt_batch_sweep/bs_$BS \
    --wandb_project cs336-owt-batch-sweep \
    --wandb_run_name owt_bs_$BS \
    --use_wandb
```

---

## 📁 Output Structure

After running all experiments:

```
cs336-hw1/
├── checkpoints/
│   ├── owt/                          # Experiment 1
│   │   └── checkpoint_iter_*.pt
│   ├── tinystories/                  # Experiment 2 (TS part)
│   │   └── checkpoint_iter_*.pt
│   ├── owt_lr_sweep/                 # Experiment 3
│   │   ├── lr_0.0001/
│   │   ├── lr_0.0003/
│   │   ├── lr_0.0005/
│   │   ├── lr_0.001/
│   │   └── lr_0.003/
│   └── owt_batch_sweep/              # Experiment 4
│       ├── bs_8/
│       ├── bs_16/
│       ├── bs_32/
│       ├── bs_64/
│       └── bs_128/
```

---

## 📊 W&B Projects

If W&B is enabled, results will be logged to:

1. **Default project** - Single OWT training
2. **cs336-dataset-comparison** - TinyStories vs OWT
3. **cs336-owt-lr-sweep** - Learning rate sweep
4. **cs336-owt-batch-sweep** - Batch size sweep

---

## ⏱️ Runtime Estimates (H100)

| Script | Experiments | Iterations | Runtime |
|--------|-------------|------------|---------|
| `test_all_owt_experiments.sh` | 4 groups | 500 each | ~30-45 min |
| `run_owt_experiments.sh` | 2 experiments | 40K each | ~12-16 hours |
| `run_owt_lr_sweep.sh` | 5 LRs | 40K each | ~30-40 hours |
| `run_owt_batch_sweep.sh` | 5 batch sizes | 40K each | ~30-40 hours |
| **`run_all_owt_experiments.sh`** | **All above** | **40K each** | **~80-100 hours** |

---

## 💡 Tips

### 1. Test First
Always run the test script before committing to the full run:
```bash
./test_all_owt_experiments.sh
```

### 2. Monitor Progress
```bash
# Check W&B dashboard
# Or check checkpoint directories
ls -lh checkpoints/owt/
```

### 3. Resume from Checkpoint
If a run fails, you can resume by adding `--resume_from_checkpoint`:
```bash
uv run python experiments/train_owt.py \
    --resume_from_checkpoint checkpoints/owt/checkpoint_iter_10000.pt \
    --max_iters 40000 \
    ...
```

### 4. Customize Hyperparameters
Edit the bash scripts to change:
- Learning rates: Edit `LR_VALUES` array
- Batch sizes: Edit `BATCH_SIZES` array
- Iterations: Use `--max-iters` flag
- Eval frequency: Use `--eval-interval` flag

### 5. Run in Background
```bash
# Run in background with nohup
nohup ./run_all_owt_experiments.sh > experiments.log 2>&1 &

# Check progress
tail -f experiments.log
```

---

## 🐛 Troubleshooting

### Out of Memory
```bash
# Find max batch size first
uv run python scripts/find_max_batch_size.py --device cuda

# Then edit run_owt_batch_sweep.sh to use smaller batch sizes
```

### Slow Training
```bash
# Reduce iterations for testing
./run_all_owt_experiments.sh --max-iters 5000 --eval-interval 250
```

### W&B Issues
```bash
# Login first
wandb login

# Or disable W&B
./run_all_owt_experiments.sh --no-wandb
```

---

## 🎯 Summary

**For the assignment (minimal):**
```bash
./run_owt_experiments.sh
```

**For comprehensive analysis:**
```bash
./run_all_owt_experiments.sh
```

**To test everything first:**
```bash
./test_all_owt_experiments.sh
```

🎉 **You're all set!**

