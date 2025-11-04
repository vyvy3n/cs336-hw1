# Project Structure Guide

This document explains the organization of the codebase and where different types of files should go.

## 📁 Directory Structure

```
cs336-hw1/
├── cs336_basics/              # Core library code (reusable components)
│   ├── models.py              # Main model implementations
│   ├── ablation_models.py     # Model variants for ablation studies
│   ├── layers.py              # Custom layer implementations
│   ├── optimizers.py          # Custom optimizer implementations
│   ├── training.py            # Training infrastructure (Trainer class)
│   ├── config.py              # Configuration dataclasses
│   ├── utils.py               # Utility functions
│   ├── decoder.py             # Text generation/decoding
│   ├── tokenizer.py           # Tokenizer implementation
│   ├── bpe.py                 # BPE training
│   └── pretokenization.py     # Pre-tokenization utilities
│
├── experiments/               # Experiment scripts (produce scientific results)
│   ├── learning_rate_sweep.py # LR hyperparameter sweep
│   ├── batch_size_sweep.py    # Batch size hyperparameter sweep
│   ├── ablations.py           # Architecture ablation experiments
│   └── README.md              # Experiment documentation
│
├── scripts/                   # Utility scripts (tools/helpers)
│   ├── prepare_tinystories.py # Data preparation
│   └── find_max_batch_size.py # Find max batch size before OOM
│
├── data/                      # Training/validation data
├── checkpoints/               # Model checkpoints
└── wandb/                     # Weights & Biases logs
```

---

## 🎯 Where to Put Different Types of Files

### `cs336_basics/` - Core Library Code

**Put here:**
- ✅ Reusable model classes and components
- ✅ Training infrastructure (Trainer, loss functions, etc.)
- ✅ Custom layers, optimizers, schedulers
- ✅ Configuration dataclasses
- ✅ Utility functions used across the codebase
- ✅ Model variants (like `ablation_models.py`)

**Examples:**
- `models.py` - Main TransformerLM implementation
- `ablation_models.py` - TransformerLMAblation variants
- `layers.py` - Linear, Embedding, RMSNorm, SwiGLU, etc.
- `training.py` - Trainer class
- `optimizers.py` - AdamW, SGD implementations

**Why here?**
- These are **library components** that can be imported and reused
- They provide the **building blocks** for experiments
- They should be **well-tested and stable**

---

### `experiments/` - Experiment Scripts

**Put here:**
- ✅ Scripts that run scientific experiments
- ✅ Hyperparameter sweeps
- ✅ Ablation studies
- ✅ Comparative analyses
- ✅ Scripts that produce results for papers/reports

**Examples:**
- `learning_rate_sweep.py` - Test multiple learning rates
- `batch_size_sweep.py` - Test multiple batch sizes
- `ablations.py` - Test architectural variants

**Why here?**
- These scripts **produce experimental results**
- They **compare different configurations**
- Results are typically logged to W&B or saved for analysis
- They answer **research questions**

**Characteristics:**
- Usually run multiple training runs
- Log metrics for comparison
- May take hours to complete
- Produce plots, tables, or reports

---

### `scripts/` - Utility Scripts

**Put here:**
- ✅ Data preparation/preprocessing scripts
- ✅ Diagnostic/debugging tools
- ✅ One-off utility scripts
- ✅ Helper scripts that don't produce experimental results

**Examples:**
- `prepare_tinystories.py` - Tokenize and prepare dataset
- `find_max_batch_size.py` - Find max batch size before OOM

**Why here?**
- These are **tools** that support the workflow
- They **don't produce scientific results**
- They're typically run **once or occasionally**
- They help with **setup, debugging, or optimization**

**Characteristics:**
- Usually run once or infrequently
- Produce artifacts (data files, configs) rather than experimental results
- Help with workflow but aren't part of the research

---

## 🚫 Common Mistakes

### ❌ Don't put model code in `experiments/`

**Bad:**
```
experiments/
└── ablation_models.py  # ❌ This is model code, not an experiment!
```

**Why?** Model classes are reusable components, not experiments. They belong in `cs336_basics/`.

---

### ❌ Don't put utility scripts in `experiments/`

**Bad:**
```
experiments/
└── find_max_batch_size.py  # ❌ This is a utility, not an experiment!
```

**Why?** This script is a tool/helper, not a scientific experiment. It belongs in `scripts/`.

---

### ❌ Don't put experiment scripts in `cs336_basics/`

**Bad:**
```
cs336_basics/
└── learning_rate_sweep.py  # ❌ This is an experiment, not library code!
```

**Why?** Experiment scripts use the library but aren't part of it. They belong in `experiments/`.

---

## 📋 Decision Tree

**When adding a new file, ask:**

1. **Is it a reusable component/class?**
   - YES → `cs336_basics/`
   - NO → Continue to #2

2. **Does it run experiments and produce scientific results?**
   - YES → `experiments/`
   - NO → Continue to #3

3. **Is it a utility/tool that supports the workflow?**
   - YES → `scripts/`
   - NO → Consider if it belongs in the project at all

---

## ✅ Current Structure is Correct

### `cs336_basics/ablation_models.py` ✅ Correctly placed

- Contains reusable model classes (`TransformerLMAblation`, `SiLUFFN`, etc.)
- Imported by `training.py` (core infrastructure)
- Provides building blocks for ablation experiments
- **Should stay in `cs336_basics/`**

### `scripts/find_max_batch_size.py` ✅ Correctly placed

- Utility tool to find max batch size before OOM
- Doesn't produce experimental results
- Run occasionally for optimization/debugging
- **Should stay in `scripts/`**

### `experiments/ablations.py` ✅ Correctly placed

- Runs ablation experiments using `ablation_models.py`
- Produces scientific results comparing architectures
- Logs metrics to W&B for analysis
- **Should stay in `experiments/`**

---

## 🔄 Summary

| Type | Location | Examples |
|------|----------|----------|
| **Library code** | `cs336_basics/` | models, layers, training, config |
| **Experiments** | `experiments/` | LR sweep, batch size sweep, ablations |
| **Utilities** | `scripts/` | data prep, find max batch size |

**Key principle:** Separate **what** (library code) from **how you use it** (experiments) from **supporting tools** (scripts).

