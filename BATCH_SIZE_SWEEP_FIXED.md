# Batch Size Sweep - Fixed Iterations (40,000)

## ✅ Script Updated!

The `experiments/batch_size_sweep.py` script has been modified to use **FIXED 40,000 iterations** for all batch sizes.

## Changes Made

### Before (Variable Iterations):
- Kept total tokens constant at 327,680,000
- Batch size 1: 1,280,000 iterations (~11 days!)
- Batch size 32: 40,000 iterations
- Batch size 512: 2,560 iterations

### After (Fixed Iterations):
- **All batch sizes train for 40,000 iterations**
- Fair comparison across different batch sizes
- Reasonable training time (~2 hours per batch size)

## Training Details

**Fixed Parameters:**
- Iterations: 40,000 (for ALL batch sizes)
- Warmup: 2,000 iterations (5% of total)
- Eval interval: 500 iterations
- Context length: 256
- Learning rate: 1e-3 (your best LR from sweep)

**Model Architecture:**
- Layers: 4
- d_model: 512
- num_heads: 16
- d_ff: 1344
- vocab_size: 10,000
- RoPE: enabled (theta=10000)

## Expected Results

| Batch Size | Iterations | Total Tokens | Training Time (est) |
|------------|-----------|--------------|---------------------|
| 1          | 40,000    | 10,240,000   | ~2 hours            |
| 2          | 40,000    | 20,480,000   | ~2 hours            |
| 4          | 40,000    | 40,960,000   | ~2 hours            |
| 8          | 40,000    | 81,920,000   | ~2 hours            |
| 16         | 40,000    | 163,840,000  | ~2 hours            |
| 32         | 40,000    | 327,680,000  | ~2 hours            |
| 64         | 40,000    | 655,360,000  | ~2 hours            |
| 128        | 40,000    | 1,310,720,000| ~2 hours            |
| 256        | 40,000    | 2,621,440,000| ~2 hours            |
| 512        | 40,000    | 5,242,880,000| ~2 hours            |

**Total time for 10 batch sizes: ~20 hours** (can run overnight)

## Commands to Run

### Quick Test (3 batch sizes)
```bash
cd cs336-hw1

python experiments/batch_size_sweep.py \
  --base_lr 1e-3 \
  --batch_sizes 8 32 128 \
  --device cuda
```

### Full Sweep (Recommended)
```bash
cd cs336-hw1

python experiments/batch_size_sweep.py \
  --base_lr 1e-3 \
  --batch_sizes 1 2 4 8 16 32 64 128 256 512 \
  --device cuda
```

### With Automatic Max Batch Size Detection
```bash
cd cs336-hw1

python experiments/batch_size_sweep.py \
  --base_lr 1e-3 \
  --batch_sizes 1 2 4 8 16 32 64 128 256 512 \
  --find_max \
  --device cuda
```

### Run in tmux (Recommended)
```bash
# Create new tmux session
tmux new -s batch_sweep

# Inside tmux
cd cs336-hw1
python experiments/batch_size_sweep.py \
  --base_lr 1e-3 \
  --batch_sizes 1 2 4 8 16 32 64 128 256 512 \
  --find_max \
  --device cuda

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t batch_sweep
```

### With Learning Rate Scaling (Optional)
```bash
cd cs336-hw1

# Scale LR with sqrt(batch_size) for large batches
python experiments/batch_size_sweep.py \
  --base_lr 1e-3 \
  --batch_sizes 1 2 4 8 16 32 64 128 256 512 \
  --optimize_lr \
  --device cuda
```

## Monitoring Progress

### Check W&B Dashboard
Your runs will appear in the `cs336-batch-size-sweep` project

### Check Checkpoints
```bash
ls -lh checkpoints/batch_size_sweep/
```

### View Logs
```bash
# See latest metrics
tail -f checkpoints/batch_size_sweep/bs_*/metrics_iter_*.txt
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

## What to Expect

### Training Progress
- Each batch size will train for exactly 40,000 iterations
- Evaluation every 500 iterations (80 eval points total)
- Checkpoints saved every 10,000 iterations

### Memory Usage
- Small batch sizes (1-8): Very low GPU utilization
- Medium batch sizes (16-64): Good GPU utilization
- Large batch sizes (128+): High GPU utilization, may OOM

### Results
- Smaller batch sizes: More noisy training, slower convergence
- Larger batch sizes: Smoother training, faster convergence (up to a point)
- Very large batch sizes: May need LR adjustment to converge well

## Notes

✅ **Fixed iterations ensures fair comparison** - All models see the same number of gradient updates

✅ **Reasonable training time** - ~2 hours per batch size instead of days

✅ **Matches your LR sweep** - Batch size 32 with 40K iterations matches your previous experiments

⚠️ **Different total tokens** - Larger batch sizes see more tokens, but this is expected and acceptable

⚠️ **May hit OOM** - Very large batch sizes (256+) may run out of GPU memory

## Troubleshooting

### Out of Memory (OOM)
If you hit OOM, the script will automatically skip larger batch sizes.

### Slow Training
Small batch sizes (1-4) will be slower due to poor GPU utilization. This is expected.

### W&B Issues
Add `--no_wandb` flag to disable W&B logging if needed.

---

**Ready to run!** The script is now configured for practical batch size experiments. 🚀

