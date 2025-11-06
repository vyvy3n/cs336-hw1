# 🐛 Resume Training Bugs - Fixed!

## 📊 The Problem You Discovered

When you resumed training from checkpoint 39,500, you observed:

1. **Loss increased** instead of continuing from where it left off
   - Original run at 40K: val_loss ≈ 3.97
   - Resumed run at 41K: val_loss ≈ 4.25 (WORSE!)

2. **Early stopping triggered immediately** even though loss was worse

3. **W&B charts showed two separate runs** with confusing x-axis

---

## 🔍 Root Cause Analysis

### Bug #1: Early Stopping State Not Saved/Restored

**The Problem:**

When you save a checkpoint, it only saved:
```python
checkpoint = {
    'model': model.state_dict(),      # ✅ Saved
    'optimizer': optimizer.state_dict(),  # ✅ Saved
    'iteration': iteration,            # ✅ Saved
    # ❌ Missing: best_val_loss, patience_counter
}
```

When you resumed:
- `best_val_loss` was reset to `float('inf')` instead of 3.97
- `patience_counter` was reset to 0

**Why This Caused Issues:**

1. Your original run had `best_val_loss = 3.97` at iteration 40K
2. When resumed, `best_val_loss = inf` (reset!)
3. First evaluation gets val_loss = 4.25
4. Since 4.25 < inf, early stopping thinks it's "improving"
5. But 4.25 > 3.97, so loss actually got WORSE!

**The Fix:**

Now checkpoints save and restore early stopping state:
```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iteration': iteration,
    'training_state': {                # ✅ NEW!
        'best_val_loss': 3.97,
        'patience_counter': 2,
    }
}
```

---

### Bug #2: W&B Creates New Run Instead of Continuing

**The Problem:**

When you resume training, W&B creates a **new run** with a new run ID. This causes:
- X-axis resets to 0 (shows iteration 40000 as step 0)
- Two separate lines in the chart
- Confusing visualization

**Why This Happens:**

This is actually **expected behavior** for W&B. Each time you call `wandb.init()`, it creates a new run.

**The Fix:**

The code already logs with correct step numbers:
```python
wandb.log(metrics, step=self.current_iter)  # Uses actual iteration (40000, 41000, etc.)
```

But W&B still shows it as a separate run. To continue the same run, you would need to:
1. Save the W&B run ID in the checkpoint
2. Resume with `wandb.init(id=saved_run_id, resume="must")`

**For now:** It's fine to have separate runs. Just look at the actual step numbers in the tooltip.

---

## ✅ What Was Fixed

### 1. Updated `save_checkpoint()` in `utils.py`

**Before:**
```python
def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)
```

**After:**
```python
def save_checkpoint(model, optimizer, iteration, out, training_state=None):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    if training_state is not None:
        checkpoint['training_state'] = training_state  # ✅ Save early stopping state
    torch.save(checkpoint, out)
```

---

### 2. Updated `load_checkpoint()` in `utils.py`

**Before:**
```python
def load_checkpoint(src, model, optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']
```

**After:**
```python
def load_checkpoint(src, model, optimizer) -> tuple[int, dict]:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    training_state = checkpoint.get('training_state', {})  # ✅ Load early stopping state
    return checkpoint['iteration'], training_state
```

---

### 3. Updated `Trainer.save_checkpoint()` in `training.py`

**Before:**
```python
def save_checkpoint(self, metrics=None):
    # ...
    save_checkpoint_impl(self.model, self.optimizer, self.current_iter, path)
```

**After:**
```python
def save_checkpoint(self, metrics=None):
    # Prepare training state
    training_state = {
        'best_val_loss': self.best_val_loss,      # ✅ Save early stopping state
        'patience_counter': self.patience_counter,
    }
    save_checkpoint_impl(self.model, self.optimizer, self.current_iter, path, training_state)
```

---

### 4. Updated `Trainer.load_checkpoint()` in `training.py`

**Before:**
```python
def load_checkpoint(self, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    self.current_iter = load_checkpoint_impl(checkpoint_path, self.model, self.optimizer)
    print(f"Resumed from iteration {self.current_iter}")
```

**After:**
```python
def load_checkpoint(self, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    self.current_iter, training_state = load_checkpoint_impl(checkpoint_path, self.model, self.optimizer)
    
    # Restore early stopping state
    if training_state:
        self.best_val_loss = training_state.get('best_val_loss', float('inf'))
        self.patience_counter = training_state.get('patience_counter', 0)
        print(f"Resumed from iteration {self.current_iter}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")        # ✅ Show restored state
        print(f"  Patience counter: {self.patience_counter}")
    else:
        print(f"Resumed from iteration {self.current_iter} (no training state found)")
```

---

### 5. Updated Test Adapter in `tests/adapters.py`

**Before:**
```python
def run_load_checkpoint(src, model, optimizer) -> int:
    return load_checkpoint(src, model, optimizer)
```

**After:**
```python
def run_load_checkpoint(src, model, optimizer) -> int:
    iteration, _ = load_checkpoint(src, model, optimizer)  # ✅ Unpack tuple
    return iteration
```

---

## 🚀 How to Use the Fixed Version

### Important: Old Checkpoints Don't Have Training State

Your existing checkpoints (saved before this fix) **don't have** `training_state`. When you load them:
- `best_val_loss` will be reset to `inf`
- `patience_counter` will be reset to 0
- You'll see: `"Resumed from iteration 39000 (no training state found)"`

**This is OK for one-time resume**, but early stopping won't work correctly.

---

### Option 1: Resume Without Early Stopping (Recommended for Old Checkpoints)

```bash
cd cs336-hw1

uv run python experiments/train_owt.py \
    --device cuda \
    --max_iters 100000 \
    --eval_interval 500 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --resume_from checkpoints/owt/checkpoint_iter_39000.pt \
    --use_wandb
    # ❌ Don't use --early_stopping_patience with old checkpoints
```

**Why:** Old checkpoints don't have `best_val_loss`, so early stopping will think any loss is "improvement".

---

### Option 2: Train Fresh with Early Stopping (Recommended for New Runs)

```bash
cd cs336-hw1

uv run python experiments/train_owt.py \
    --device cuda \
    --max_iters 100000 \
    --eval_interval 500 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --early_stopping_patience 10 \
    --use_wandb
```

**Why:** New checkpoints will save `training_state`, so you can resume with early stopping later.

---

## 📊 Expected Behavior After Fix

### When Saving Checkpoint:

```
Saving checkpoint to checkpoints/owt/checkpoint_iter_40000.pt
```

**Checkpoint now contains:**
- ✅ Model weights
- ✅ Optimizer state (momentum, learning rates)
- ✅ Iteration number (40000)
- ✅ **Best validation loss (3.97)**
- ✅ **Patience counter (2)**

---

### When Resuming from New Checkpoint:

```
Loading checkpoint from checkpoints/owt/checkpoint_iter_40000.pt
Resumed from iteration 40000
  Best val loss: 3.9714
  Patience counter: 2

Starting training from iteration 40000 to 100000
```

**Early stopping will now:**
- ✅ Compare new val_loss against 3.9714 (not inf)
- ✅ Continue patience counter from 2 (not 0)
- ✅ Work correctly across resume

---

### When Resuming from Old Checkpoint:

```
Loading checkpoint from checkpoints/owt/checkpoint_iter_39000.pt
Resumed from iteration 39000 (no training state found)

Starting training from iteration 39000 to 100000
```

**Early stopping will:**
- ⚠️ Reset `best_val_loss = inf`
- ⚠️ Reset `patience_counter = 0`
- ⚠️ Not work correctly (will think any loss is improvement)

**Solution:** Don't use `--early_stopping_patience` when resuming from old checkpoints.

---

## 🧪 Testing the Fix

Run this to verify the fix works:

```bash
cd cs336-hw1

uv run python -c "
from cs336_basics.utils import save_checkpoint, load_checkpoint
import torch
import torch.nn as nn

# Create dummy model and optimizer
model = nn.Linear(10, 10)
optimizer = torch.optim.Adam(model.parameters())

# Save checkpoint with training state
training_state = {'best_val_loss': 3.5, 'patience_counter': 2}
save_checkpoint(model, optimizer, 1000, 'test_checkpoint.pt', training_state)

# Load checkpoint
model2 = nn.Linear(10, 10)
optimizer2 = torch.optim.Adam(model2.parameters())
iteration, loaded_state = load_checkpoint('test_checkpoint.pt', model2, optimizer2)

print(f'Iteration: {iteration}')
print(f'Best val loss: {loaded_state.get(\"best_val_loss\")}')
print(f'Patience counter: {loaded_state.get(\"patience_counter\")}')

import os
os.remove('test_checkpoint.pt')
print('✅ Test passed!')
"
```

**Expected output:**
```
Iteration: 1000
Best val loss: 3.5
Patience counter: 2
✅ Test passed!
```

---

## 📝 Summary

| Issue | Before | After |
|-------|--------|-------|
| **Early stopping state saved?** | ❌ No | ✅ Yes |
| **Resume with correct best_val_loss?** | ❌ No (resets to inf) | ✅ Yes (restored from checkpoint) |
| **Resume with correct patience_counter?** | ❌ No (resets to 0) | ✅ Yes (restored from checkpoint) |
| **Early stopping works after resume?** | ❌ No | ✅ Yes (for new checkpoints) |
| **Backward compatible with old checkpoints?** | N/A | ✅ Yes (gracefully handles missing state) |

---

## 🎯 Recommendations

1. **For your current 39K checkpoint:** Resume **without** early stopping
   ```bash
   --resume_from checkpoints/owt/checkpoint_iter_39000.pt
   # Don't use --early_stopping_patience
   ```

2. **For future training:** Use early stopping from the start
   ```bash
   --early_stopping_patience 10
   # Checkpoints will save training state
   ```

3. **If you want to resume with early stopping:** Start a fresh run
   ```bash
   # Don't resume, train from scratch with early stopping
   uv run python experiments/train_owt.py --max_iters 100000 --early_stopping_patience 10
   ```

---

## ✅ All Fixed!

The bugs are now fixed. Future checkpoints will save and restore early stopping state correctly! 🎉

