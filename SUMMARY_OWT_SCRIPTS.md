# Summary: OpenWebText Experiment Scripts

## ✅ What Was Created

I created **2 minimal scripts** that **reuse all existing code** to answer the OpenWebText assignment question.

---

## 📝 Assignment Question

> **Problem (main_experiment): Experiment on OWT (2 points) (3 H100 hrs)**
> 
> Train your language model on OpenWebText with the same model architecture and total training iterations as TinyStories. How well does this model do?
> 
> **Deliverable:** A learning curve of your language model on OpenWebText. Describe the difference in losses from TinyStories – how should we interpret these losses?

---

## 🎯 Solution: Reuse Existing Code!

### Script 1: `experiments/train_owt.py`

**Purpose:** Train on OpenWebText only

**What it does:**
- Uses existing `Trainer` class from `cs336_basics/training.py`
- Uses existing `TrainingConfig` from `cs336_basics/config.py`
- Simply changes data paths to OWT instead of TinyStories
- Same model architecture, same hyperparameters

**Usage:**
```bash
# Quick test
uv run python experiments/train_owt.py --device cuda --max_iters 100

# Full training
uv run python experiments/train_owt.py --device cuda --use_wandb
```

---

### Script 2: `experiments/compare_datasets.py`

**Purpose:** Train on both datasets for direct comparison

**What it does:**
- Trains on TinyStories with config A
- Trains on OpenWebText with config A (same config)
- Logs both to W&B for side-by-side comparison
- Provides interpretation guidance

**Usage:**
```bash
# Quick test both
uv run python experiments/compare_datasets.py --device cuda --max_iters 100

# Full comparison (recommended)
uv run python experiments/compare_datasets.py --device cuda --use_wandb

# Train only one
uv run python experiments/compare_datasets.py --dataset owt --device cuda
```

---

## 🔧 Why This is Minimal

### ✅ No New Core Code

- **No new model code** - Uses existing `TransformerLM`
- **No new training code** - Uses existing `Trainer` class
- **No new config code** - Uses existing `TrainingConfig`
- **No new utilities** - Uses existing `setup_device`, `print_experiment_header`

### ✅ Only Changes Data Paths

The **only difference** between TinyStories and OWT experiments:

```python
# TinyStories
train_data_path="data/tinystories_train_tokens.npy"
val_data_path="data/tinystories_valid_tokens.npy"

# OpenWebText
train_data_path="data/owt_train_tokens.npy"
val_data_path="data/owt_valid_tokens.npy"
```

Everything else is **identical**!

---

## 📊 Expected Results

### Dataset Statistics

| Dataset | Train Tokens | Valid Tokens | Total |
|---------|-------------|--------------|-------|
| TinyStories | 542M | 5.5M | 548M |
| OpenWebText | 2.85B | 69M | 2.92B |

### Expected Losses

| Dataset | Final Val Loss | Why? |
|---------|---------------|------|
| TinyStories | ~1.4 - 1.5 | Simple, repetitive text |
| OpenWebText | ~3.0 - 3.5 | Complex, diverse text |

### Interpretation

**OWT has higher loss because:**
1. More diverse vocabulary usage
2. More complex syntax and grammar
3. More varied topics and domains
4. Less repetition and patterns
5. More realistic, natural language

**This doesn't mean OWT model is worse!**
- Lower loss = better fit to training distribution
- TinyStories has simpler distribution → easier to fit
- OWT model is more useful for general language tasks

---

## 🚀 Quick Start

### Step 1: Verify Data Exists

```bash
ls -lh data/owt_*.npy data/tinystories_*.npy
```

You should see:
- `owt_train_tokens.npy`
- `owt_valid_tokens.npy`
- `tinystories_train_tokens.npy`
- `tinystories_valid_tokens.npy`

### Step 2: Quick Test (2 minutes)

```bash
# Test that everything works
uv run python experiments/compare_datasets.py \
    --device cuda \
    --max_iters 100 \
    --eval_interval 50
```

### Step 3: Full Experiment (~3 H100 hours)

```bash
# Run full comparison with W&B logging
uv run python experiments/compare_datasets.py \
    --device cuda \
    --use_wandb
```

### Step 4: View Results

1. Go to W&B dashboard
2. Project: `cs336-dataset-comparison`
3. Compare learning curves side-by-side
4. Note the difference in final losses

### Step 5: Write Report

Report:
1. **Learning curves** - Screenshot from W&B
2. **Final losses** - TinyStories: ~1.4, OWT: ~3.2
3. **Interpretation** - Explain why OWT has higher loss and what it means

---

## 📁 Files Created

### New Experiment Scripts

1. **`experiments/train_owt.py`** (195 lines)
   - Train on OpenWebText only
   - Minimal script that reuses all existing code

2. **`experiments/compare_datasets.py`** (245 lines)
   - Train on both datasets for comparison
   - Provides interpretation guidance

### Documentation

3. **`OWT_EXPERIMENT_GUIDE.md`** (Comprehensive guide)
   - Detailed explanation of the experiment
   - Expected results and interpretation
   - Troubleshooting tips

4. **`SUMMARY_OWT_SCRIPTS.md`** (This file)
   - Quick summary of what was created
   - Why it's minimal
   - How to use it

### Updated Documentation

5. **`experiments/README.md`** (Updated)
   - Added OWT experiment section
   - Updated file structure

---

## 🎓 Key Takeaways

### 1. Reuse > Rewrite

Instead of creating new training infrastructure, we **reused** existing code:
- ✅ Same `Trainer` class
- ✅ Same `TrainingConfig`
- ✅ Same model architecture
- ✅ Only changed data paths

### 2. Minimal = Better

The scripts are minimal because:
- Less code to maintain
- Less chance of bugs
- Easier to understand
- Consistent with existing experiments

### 3. Comparison is Key

The `compare_datasets.py` script is valuable because:
- Trains both datasets with **identical** configs
- Logs to same W&B project for easy comparison
- Provides interpretation guidance
- Answers the assignment question directly

---

## 🔍 Code Structure

Both scripts follow the same pattern:

```python
# 1. Import existing infrastructure
from cs336_basics.config import TrainingConfig, ModelConfig, DataConfig, ...
from cs336_basics.training import Trainer
from cs336_basics.utils import setup_device, print_experiment_header

# 2. Create config (only data paths differ)
config = TrainingConfig(
    model=ModelConfig(...),  # Same as TinyStories
    data=DataConfig(
        train_data_path="data/owt_train_tokens.npy",  # Only difference!
        val_data_path="data/owt_valid_tokens.npy",    # Only difference!
    ),
    optimizer=OptimizerConfig(...),  # Same as TinyStories
    ...
)

# 3. Train using existing Trainer
trainer = Trainer(config)
trainer.train()
```

**That's it!** No new core code needed.

---

## 📊 Comparison Table

| Aspect | TinyStories | OpenWebText |
|--------|-------------|-------------|
| **Data Path** | `data/tinystories_*.npy` | `data/owt_*.npy` |
| **Model** | TransformerLM | TransformerLM (same) |
| **Architecture** | 4L, 512d, 16h | 4L, 512d, 16h (same) |
| **Iterations** | 5000 | 5000 (same) |
| **Batch Size** | 32 | 32 (same) |
| **Learning Rate** | 1e-3 | 1e-3 (same) |
| **Expected Loss** | ~1.4 | ~3.2 |
| **Interpretation** | Simple text | Complex text |

---

## ✅ Checklist

Before running the experiment:

- [ ] OWT data files exist (`data/owt_*.npy`)
- [ ] TinyStories data files exist (`data/tinystories_*.npy`)
- [ ] W&B is configured (`wandb login`)
- [ ] GPU is available (check with `nvidia-smi`)
- [ ] Quick test passes (100 iterations)

To complete the assignment:

- [ ] Run full experiment (5000 iterations)
- [ ] Collect learning curves from W&B
- [ ] Note final validation losses
- [ ] Write interpretation of loss differences
- [ ] Submit deliverable

---

## 🎯 Bottom Line

**You asked for:** Minimal viable script to answer the OWT question

**I delivered:**
- ✅ 2 minimal scripts (195 and 245 lines)
- ✅ Reuse all existing code
- ✅ Only change data paths
- ✅ Easy to run and compare
- ✅ Comprehensive documentation

**To answer the assignment:**
```bash
uv run python experiments/compare_datasets.py --device cuda --use_wandb
```

That's it! 🎉

