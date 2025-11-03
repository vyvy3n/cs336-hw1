# Learning Rate Experiments - Quick Start Guide

This guide will help you run the learning rate hyperparameter sweep experiments for the assignment.

## 📋 Assignment Requirements

You need to train a Transformer model with these specifications:
- **d_model**: 512
- **d_ff**: 1344
- **num_layers**: 4
- **num_heads**: 16
- **RoPE theta**: 10000
- **Total tokens**: 327,680,000
- **Target validation loss**: ≤ 1.45 per-token

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Data (~10-30 minutes)

```bash
# Download and tokenize TinyStories dataset
uv run python scripts/prepare_tinystories.py --vocab_size 10000 --output_dir data
```

**Note**: If automatic download fails, manually download from:
https://huggingface.co/datasets/roneneldan/TinyStories

### Step 2: Login to W&B

```bash
wandb login
```

Get your API key from: https://wandb.ai/authorize

### Step 3: Run Experiments

#### Option A: Quick Test First (Recommended)

```bash
# Test with 1000 iterations (~5 minutes on GPU)
uv run python experiments/quick_lr_test.py \
    --learning_rate 3e-4 \
    --max_iters 1000 \
    --device cuda \
    --use_wandb
```

#### Option B: Full Grid Sweep

```bash
# Full sweep over 8 learning rates (~4-5 hours on H100)
uv run python experiments/learning_rate_sweep.py \
    --sweep_type grid \
    --device cuda
```

#### Option C: Stability Sweep

```bash
# Find edge of stability (~5-6 hours on H100)
uv run python experiments/learning_rate_sweep.py \
    --sweep_type stability \
    --device cuda
```

#### Option D: Run Everything

```bash
# Run both sweeps sequentially
bash experiments/run_all_experiments.sh cuda
```

## 📊 Monitoring Results

### Real-time Monitoring

After starting training, you'll see a W&B URL:
```
wandb: 🚀 View run at: https://wandb.ai/your-username/cs336-lr-sweep/runs/abc123
```

Click it to see:
- Training and validation loss curves
- Learning rate schedule
- Gradient norms
- Model statistics

### Analyze Results

```bash
# Analyze W&B results
uv run python experiments/analyze_results.py --wandb_project cs336-lr-sweep

# Analyze local checkpoints
uv run python experiments/analyze_results.py --checkpoint_dir checkpoints/lr_sweep
```

## 📁 Files Created

```
cs336-hw1/
├── experiments/
│   ├── README.md                    # Detailed documentation
│   ├── learning_rate_sweep.py       # Main experiment script
│   ├── quick_lr_test.py            # Quick testing
│   ├── run_all_experiments.sh      # Batch runner
│   └── analyze_results.py          # Results analysis
│
├── scripts/
│   └── prepare_tinystories.py      # Dataset preparation
│
├── data/
│   ├── TinyStories_train.npy       # Training tokens (created by prepare script)
│   ├── TinyStories_valid.npy       # Validation tokens (created by prepare script)
│   └── tokenizer_v10000.json       # BPE tokenizer (created by prepare script)
│
└── checkpoints/
    └── lr_sweep/                   # Model checkpoints (created during training)
        ├── lr_1e_04/
        ├── lr_3e_04/
        └── ...
```

## 🎯 Assignment Deliverables

### Part (a): Grid Sweep

**What to submit:**
1. Learning curves for multiple learning rates
2. Explanation of your search strategy
3. Model with validation loss ≤ 1.45

**How to get it:**
```bash
# Run grid sweep
uv run python experiments/learning_rate_sweep.py --sweep_type grid --device cuda

# Analyze results
uv run python experiments/analyze_results.py --wandb_project cs336-lr-sweep

# Export plots from W&B dashboard
```

### Part (b): Stability Analysis

**What to submit:**
1. Learning curves showing at least one divergent run
2. Analysis of how divergence point relates to best LR

**How to get it:**
```bash
# Run stability sweep
uv run python experiments/learning_rate_sweep.py --sweep_type stability --device cuda

# Analyze results
uv run python experiments/analyze_results.py --wandb_project cs336-lr-sweep
```

## ⚙️ Configuration Options

### Learning Rates Tested (Grid Sweep)

- `1e-5` - Very small
- `5e-5` - Small
- `1e-4` - Small-medium
- `3e-4` - Common default
- `5e-4` - Medium
- `1e-3` - Large
- `3e-3` - Very large
- `5e-3` - Likely too large

### Stability Sweep

- Starts at `1e-4`
- Ends at `1e-2`
- Tests 10 learning rates on log scale
- Stops after finding divergence

## 🐛 Troubleshooting

### "Dataset not found"

```bash
# Make sure you ran data preparation
uv run python scripts/prepare_tinystories.py --vocab_size 10000
```

### Out of Memory

```bash
# Reduce batch size
uv run python experiments/quick_lr_test.py --batch_size 16 --learning_rate 3e-4
```

### Training Diverges Immediately

```bash
# Try smaller learning rate
uv run python experiments/quick_lr_test.py --learning_rate 1e-4 --max_iters 1000
```

### Slow Training

- Make sure you're using GPU: `--device cuda`
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- For CPU/MPS, reduce `--max_iters` significantly

### W&B Login Issues

```bash
# Re-login
wandb login --relogin

# Or disable W&B
uv run python experiments/learning_rate_sweep.py --sweep_type grid --no_wandb
```

## 💡 Tips

1. **Start with quick test**: Always run `quick_lr_test.py` first to verify setup
2. **Monitor early**: Check first 100-200 iterations - if loss isn't decreasing, stop
3. **Use W&B**: Makes comparing runs much easier
4. **Save notes**: Document what you observe for your report
5. **Check target**: Make sure at least one run achieves ≤ 1.45 validation loss

## ⏱️ Expected Runtimes

### H100 GPU
- Quick test (1000 iters): ~5 minutes
- Single full run (40,000 iters): ~30-40 minutes
- Grid sweep (8 runs): ~4-5 hours
- Stability sweep (10 runs): ~5-6 hours

### CPU/MPS
- Significantly longer (10-20x slower)
- Recommended: Reduce to 5,000 iterations per run
- Adjust target validation loss to 2.00

## 📚 Additional Resources

- **Detailed docs**: See `experiments/README.md`
- **W&B docs**: https://docs.wandb.ai
- **TinyStories paper**: https://arxiv.org/abs/2305.07759
- **Kingma & Ba (2015)**: Adam optimizer paper

## ✅ Checklist

Before running full experiments:

- [ ] Data prepared (`TinyStories_train.npy` and `TinyStories_valid.npy` exist)
- [ ] W&B logged in (`wandb login`)
- [ ] Quick test passed (`quick_lr_test.py` runs successfully)
- [ ] GPU available (if using CUDA)
- [ ] Enough disk space (~10GB for checkpoints)

Ready to run:

- [ ] Grid sweep completed
- [ ] Stability sweep completed
- [ ] Results analyzed
- [ ] Best model achieves ≤ 1.45 validation loss
- [ ] Learning curves exported from W&B
- [ ] Report written

## 🆘 Getting Help

If you encounter issues:

1. Check the detailed README: `experiments/README.md`
2. Run quick test to isolate the problem
3. Check W&B dashboard for error messages
4. Review console output for error traces
5. Verify data files exist and are not corrupted

## 🎓 Learning Objectives

By completing these experiments, you will:

1. Understand the importance of learning rate tuning
2. Learn to identify training instability
3. Practice systematic hyperparameter search
4. Gain experience with experiment tracking (W&B)
5. Develop intuition for "edge of stability"

Good luck with your experiments! 🚀

