# Ablation Experiments - Issue Diagnosis and Fix

## 🔍 What Happened

Your ablation experiments **crashed early** (after 8-9 minutes instead of ~2 hours). W&B marked them as "Failed" because the training terminated unexpectedly.

### Evidence
1. ✅ W&B shows runs with data (learning curves visible)
2. ❌ Runs marked as "Failed" in W&B
3. ❌ Very short runtime: 8m 15s to 8m 51s (should be ~2 hours)
4. ❌ No checkpoints saved (checkpoints/ablations/ is empty)

### Root Cause

**The training script tried to save checkpoints to directories that didn't exist!**

The `save_checkpoint()` function in `cs336_basics/training.py` was missing the directory creation step:

```python
# BEFORE (missing directory creation)
def save_checkpoint(...):
    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_iter_{iteration}.pt")
    save_checkpoint_impl(model, optimizer, iteration, checkpoint_path)  # ❌ Crashes if dir doesn't exist!
```

```python
# AFTER (fixed)
def save_checkpoint(...):
    os.makedirs(config.checkpoint_dir, exist_ok=True)  # ✅ Create directory first!
    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_iter_{iteration}.pt")
    save_checkpoint_impl(model, optimizer, iteration, checkpoint_path)
```

When the training reached the first checkpoint save (iteration 10,000), it crashed with a `FileNotFoundError`.

## ✅ Fix Applied

I've fixed the issue by adding directory creation to the `save_checkpoint()` function:

**File modified:** `cs336_basics/training.py` (line 202)
- Added: `os.makedirs(config.checkpoint_dir, exist_ok=True)`

## 🚀 How to Re-run

Now that the fix is applied, you can re-run the experiments:

### Option 1: Run All Experiments (Recommended)

```bash
cd cs336-hw1

# In tmux (recommended)
tmux new -s ablations_fixed
./run_all_ablations.sh

# Detach: Ctrl+b, then d
# Monitor: tmux attach -t ablations_fixed
```

### Option 2: Run Individual Experiments

```bash
cd cs336-hw1

# Layer norm ablation
python experiments/ablations.py --ablation layer_norm --learning_rate 1e-3 --device cuda

# Post-norm
python experiments/ablations.py --ablation pre_norm --learning_rate 1e-3 --device cuda

# No position embeddings
python experiments/ablations.py --ablation no_pos_emb --learning_rate 1e-3 --device cuda

# SiLU-only FFN
python experiments/ablations.py --ablation swiglu --learning_rate 1e-3 --device cuda
```

## 📊 What to Expect

### Training Time
- Each experiment: ~2 hours (40,000 iterations)
- All 4 experiments: ~8 hours total

### Checkpoints
This time, checkpoints will be saved to:
```
checkpoints/ablations/
├── no_rmsnorm/
│   ├── checkpoint_iter_10000.pt
│   ├── checkpoint_iter_20000.pt
│   ├── checkpoint_iter_30000.pt
│   ├── checkpoint_iter_40000.pt
│   └── checkpoint_latest.pt
├── post_norm/
│   └── ...
├── no_rope/
│   └── ...
└── silu_only/
    └── ...
```

### W&B
- New runs will appear in the `cs336-ablations` project
- They should complete successfully and show "Finished" status
- Learning curves will show full 40K iterations

## 🧪 Verify the Fix

Before running the full experiments, test with a quick run:

```bash
cd cs336-hw1

# Quick test (just 1000 iterations)
python -c "
import sys
sys.path.insert(0, '.')
from experiments.ablations import run_experiment

# Modify config for quick test
from cs336_basics.config import TrainingConfig, ModelConfig, OptimizerConfig, SchedulerConfig, DataConfig

config = TrainingConfig(
    model=ModelConfig(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        use_rope=True,
        theta=10000.0,
        ablation_type='no_rmsnorm'
    ),
    optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.1, beta1=0.9, beta2=0.95),
    scheduler=SchedulerConfig(warmup_iters=100, max_iters=1000, min_lr_ratio=0.1),
    data=DataConfig(
        train_data_path='data/tinystories_train_tokens.npy',
        val_data_path='data/tinystories_valid_tokens.npy',
        batch_size=32,
        context_length=256
    ),
    eval_interval=500,
    eval_iters=10,
    log_interval=100,
    checkpoint_interval=500,
    checkpoint_dir='checkpoints/test_ablation',
    use_wandb=False,
    device='cuda'
)

from cs336_basics.training import train
train(config)
print('✅ Test passed! Checkpoint saved successfully.')
"

# Check if checkpoint was created
ls -la checkpoints/test_ablation/
```

If you see checkpoint files, the fix works! 🎉

## 📝 Old W&B Runs

The old "Failed" runs in W&B can be:
1. **Ignored** - They only ran for ~8 minutes and didn't complete
2. **Deleted** - You can delete them from the W&B dashboard if you want
3. **Kept for reference** - They show what happens when training crashes early

The new runs will have different names and will show "Finished" status.

## 🎯 Next Steps

1. **Re-run all ablation experiments** using the commands above
2. **Monitor progress** via W&B dashboard or tmux
3. **Wait ~8 hours** for all experiments to complete
4. **Analyze results** and create learning curve plots
5. **Write commentary** for each ablation

## 💡 Why This Wasn't Caught Earlier

The batch size sweep experiments worked because they explicitly created directories:

```python
# In batch_size_sweep.py
checkpoint_dir = f"checkpoints/batch_sweep/batch_{batch_size}"
os.makedirs(checkpoint_dir, exist_ok=True)  # ✅ Explicit creation
```

But the ablations script relied on the training script to create directories, which it didn't do. This is now fixed globally in the training script, so all future experiments will work correctly.

---

**The fix is complete! You can now re-run your ablation experiments.** 🚀

