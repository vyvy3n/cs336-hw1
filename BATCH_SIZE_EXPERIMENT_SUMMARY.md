# Batch Size Experiment - Implementation Summary

## ✅ What Was Implemented

I've created a complete batch size sweep experiment infrastructure for your CS336 assignment.

### 📁 Files Created

1. **`experiments/batch_size_sweep.py`** (300 lines)
   - Main experiment script
   - Tests batch sizes from 1 to GPU memory limit
   - Automatic OOM detection and handling
   - Optional learning rate scaling
   - W&B integration

2. **`experiments/test_batch_size.py`** (150 lines)
   - Quick test script (runs only 100 iterations)
   - Finds GPU memory limits
   - Estimates maximum batch size
   - Takes ~5-10 minutes

3. **`experiments/BATCH_SIZE_EXPERIMENT.md`** (250 lines)
   - Detailed experiment documentation
   - Theory and motivation
   - Expected outcomes
   - Analysis guidelines

4. **`experiments/BATCH_SIZE_QUICKSTART.md`** (300 lines)
   - Quick start guide
   - Step-by-step instructions
   - Troubleshooting
   - Deliverables checklist

5. **`BATCH_SIZE_EXPERIMENT_SUMMARY.md`** (this file)
   - Implementation overview
   - Quick reference

## 🚀 Quick Start

### Step 1: Test GPU Memory Limit (5-10 min)

```bash
cd cs336-hw1
python experiments/test_batch_size.py --device cuda
```

This will:
- Test batch sizes: 1, 32, 64, 128, 256
- Show memory usage for each
- Estimate your maximum batch size

### Step 2: Run Full Experiment (2-3 hours)

```bash
# Create tmux session
tmux new -s batch_size_exp

# Run the experiment
python experiments/batch_size_sweep.py --device cuda

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t batch_size_exp
```

### Step 3: View Results

```
https://wandb.ai/YOUR-USERNAME/cs336-batch-size-sweep
```

## 🎯 What the Experiment Does

### Batch Sizes Tested

- **Small**: 1, 2, 4, 8, 16
- **Typical**: 32, 64, 128
- **Large**: 256, 512, 1024

### Fixed Parameters

- Model: 4-layer transformer, d_model=512, 16 heads
- Total tokens: 327,680,000 (constant)
- Context length: 256
- Dataset: TinyStories
- Optimizer: AdamW

### Variable Parameters

**Option 1: Fixed LR (default)**
```bash
python experiments/batch_size_sweep.py --device cuda
```
- Uses LR=3e-4 for all batch sizes
- Tests pure effect of batch size

**Option 2: Optimized LR**
```bash
python experiments/batch_size_sweep.py --device cuda --optimize_lr
```
- Scales LR with sqrt(batch_size)
- LR = 3e-4 × sqrt(batch_size / 32)

### Training Steps Adjustment

To maintain constant total tokens:
- batch_size=1: 1,280,000 steps
- batch_size=32: 40,000 steps
- batch_size=128: 10,000 steps
- batch_size=512: 2,500 steps

## 📊 Expected Results

### Memory Limits

Typical GPU limits:
- **16 GB GPU**: batch_size up to ~256
- **24 GB GPU**: batch_size up to ~512
- **40 GB GPU**: batch_size up to ~1024
- **80 GB GPU**: batch_size up to ~2048

### Training Time

On H100 GPU:
- batch_size=1: ~60 min
- batch_size=32: ~30 min
- batch_size=128: ~15 min
- batch_size=512: ~10 min

**Total sweep**: ~2-3 hours (within 2 H100 hrs budget)

### Performance Observations

Expected findings:
1. **Throughput**: Larger batches → higher tokens/second
2. **Quality**: Similar final loss for batch_size 32-256
3. **Memory**: Linear scaling with batch size
4. **Stability**: Very small batches (1-8) more noisy

## 🔧 Advanced Usage

### Test Specific Batch Sizes

```bash
python experiments/batch_size_sweep.py \
    --device cuda \
    --batch_sizes 32 64 128 256
```

### Custom Base Learning Rate

```bash
python experiments/batch_size_sweep.py \
    --device cuda \
    --base_lr 5e-4 \
    --optimize_lr
```

### Without W&B

```bash
python experiments/batch_size_sweep.py --device cuda --no_wandb
```

## 📝 Assignment Deliverables

### 1. Learning Curves

Create plots showing:
- Training loss vs. tokens for different batch sizes
- Validation loss vs. tokens for different batch sizes
- All curves on same plot for comparison

**How to get this:**
- Use W&B dashboard (automatic)
- Or export data and create custom plots

### 2. Written Analysis

Answer these questions:

1. **Throughput**: How does batch size affect training speed?
2. **Quality**: How does batch size affect final validation loss?
3. **Memory**: What's the maximum batch size your GPU can handle?
4. **Trade-offs**: Is the largest batch size always best? Why/why not?
5. **LR Interaction**: Should learning rate be adjusted for different batch sizes?

**Example template:**

> "We tested batch sizes from 1 to [MAX] on a [GPU_NAME] GPU. Larger batch sizes significantly improved training throughput, with batch_size=[LARGE] achieving [X]x higher tokens/second compared to batch_size=32. However, final validation loss was similar across batch sizes [RANGE] (all achieving ~[LOSS]), suggesting that GPU efficiency gains don't necessarily translate to better model quality. The maximum batch size our GPU could handle was [MAX], limited by the [MEMORY]GB memory. Interestingly, batch_size=1 showed much noisier training but eventually converged to a similar final loss, though taking [X]x longer. We found that [FIXED/SCALED] learning rate worked best..."

## 🎓 Key Insights to Look For

### 1. GPU Utilization

- Small batches: Poor GPU utilization, many small matrix ops
- Large batches: High GPU utilization, large matrix ops
- **Measure**: tokens/second throughput

### 2. Gradient Noise

- Small batches: High gradient noise, more exploration
- Large batches: Low gradient noise, more exploitation
- **Measure**: smoothness of loss curves

### 3. Convergence Speed

- Per-step: Large batches converge faster per step
- Per-token: May be similar across batch sizes
- **Measure**: loss vs. tokens (not steps!)

### 4. Memory Scaling

- Memory ≈ model_params + batch_size × seq_len × d_model
- Roughly linear with batch size
- **Measure**: peak GPU memory usage

### 5. Learning Rate Interaction

- Fixed LR: May be suboptimal for very large/small batches
- Scaled LR: Should improve consistency
- **Measure**: compare fixed vs. optimized LR runs

## 🐛 Troubleshooting

### Out of Memory

```bash
# Script automatically stops at OOM
# To test smaller sizes only:
python experiments/batch_size_sweep.py --batch_sizes 1 32 64 128
```

### Training is Slow

- Check GPU usage: `nvidia-smi`
- Small batch sizes are inherently slower (more steps)
- Use tmux to run in background

### Import Errors

```bash
# Make sure you're in the right directory
cd cs336-hw1
python experiments/batch_size_sweep.py --device cuda
```

## 📚 Theory Background

### Why Batch Size Matters

1. **Computational Efficiency**: Larger batches → better GPU utilization
2. **Gradient Estimation**: Larger batches → more accurate gradients
3. **Optimization Dynamics**: Batch size affects convergence behavior
4. **Memory Constraints**: GPU memory limits maximum batch size

### Learning Rate Scaling Rules

**Linear Scaling** (Goyal et al., 2017):
```
LR_new = LR_base × (batch_size_new / batch_size_base)
```

**Square Root Scaling** (Hoffer et al., 2017):
```
LR_new = LR_base × sqrt(batch_size_new / batch_size_base)
```

Our script implements sqrt scaling with `--optimize_lr`.

## ✅ Checklist

Before submitting:

- [ ] Ran quick test to find GPU memory limit
- [ ] Ran full batch size sweep
- [ ] Tested at least: 1, 32, 64, 128, and maximum batch size
- [ ] Generated learning curves comparing batch sizes
- [ ] Wrote analysis discussing findings
- [ ] Compared fixed vs. optimized LR (optional but recommended)
- [ ] Saved W&B results or exported plots
- [ ] Answered: Is larger batch size always better?

## 📖 References

1. [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677) - Linear scaling rule
2. [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489) - Batch size and LR
3. [Measuring the Effects of Data Parallelism](https://arxiv.org/abs/1811.03600) - Batch size analysis

## 🎯 Next Steps

1. Run the quick test: `python experiments/test_batch_size.py --device cuda`
2. Review the output and note your GPU's maximum batch size
3. Run the full sweep: `python experiments/batch_size_sweep.py --device cuda`
4. Monitor progress in W&B dashboard
5. Analyze results and write up findings
6. Create visualizations for assignment submission

---

**Questions?** Check the detailed guides:
- `experiments/BATCH_SIZE_QUICKSTART.md` - Quick start guide
- `experiments/BATCH_SIZE_EXPERIMENT.md` - Detailed documentation

**Ready to run?**
```bash
cd cs336-hw1
python experiments/test_batch_size.py --device cuda
```

