# Code Optimization Analysis
## Files: utils.py, training.py, experiment_utils.py

---

## Current Structure Assessment

### ✅ **What's Good:**

1. **Clear Separation of Concerns**
   - `utils.py`: Low-level primitives (get_batch, save/load checkpoint)
   - `training.py`: Training loop and related functions
   - `experiment_utils.py`: Experiment-specific configuration helpers

2. **Minimal and Focused**
   - `utils.py`: 100 lines, 3 functions
   - `training.py`: 410 lines, 11 functions
   - `experiment_utils.py`: 177 lines, 5 functions

3. **Good Documentation**
   - All functions have clear docstrings
   - Type hints throughout

4. **No Major Redundancy**
   - Most code serves a unique purpose

---

## ⚠️ **Identified Issues:**

### **1. Loss Computation Repeated 3 Times**

**Location:** `training.py` lines 68-71, 127-129, and similar pattern

```python
# Repeated in estimate_loss(), train_step(), and implicitly elsewhere
logits_flat = logits.view(-1, logits.size(-1))
targets_flat = y.view(-1)
loss = loss_fn(logits_flat, targets_flat)
```

**Impact:** 9 lines of duplication  
**Fix:** Extract to helper function

---

### **2. Checkpoint Path Generation Logic**

**Location:** `training.py` lines 200-210

```python
checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_iter_{iteration}.pt")
# ... save ...
latest_path = os.path.join(config.checkpoint_dir, "checkpoint_latest.pt")
# ... save again ...
```

**Impact:** Could be simplified  
**Fix:** Helper function for path generation

---

### **3. Device Checking Duplication**

**Location:** 
- `experiment_utils.py` lines 17-24 (`print_gpu_info`)
- `experiment_utils.py` lines 163-176 (`check_device`)

```python
# Two separate functions that both check CUDA availability
def print_gpu_info():
    if torch.cuda.is_available():
        # print info
        
def check_device(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        # fallback to CPU
```

**Impact:** Minor duplication  
**Fix:** Combine into single utility

---

### **4. Checkpoint Wrappers Add Minimal Value**

**Location:** `training.py` lines 178-244

The `save_checkpoint` and `load_checkpoint` in `training.py` are thin wrappers around `utils.py` functions that mainly add:
- Directory creation (1 line)
- Print statements (2-3 lines)
- Metrics saving (optional, 7 lines)

**Impact:** 66 lines for wrappers  
**Fix:** Could inline or simplify

---

### **5. W&B Logging Try-Except Pattern Repeated**

**Location:** `training.py` lines 170-175, 254-271, 404-409

```python
try:
    import wandb
    wandb.something()
except ImportError:
    pass
```

**Impact:** Minor duplication  
**Fix:** Single helper function

---

## 🎯 Optimization Options

### **Option 1: Conservative (Recommended)**

**Changes:**
1. Extract `compute_loss()` helper function
2. Combine `print_gpu_info()` and `check_device()` into `setup_device()`
3. Extract `safe_wandb_call()` helper for W&B operations
4. Add `get_checkpoint_paths()` helper

**Lines Saved:** ~15-20  
**Risk:** Very low  
**Benefit:** Clearer code, less repetition

---

### **Option 2: Moderate**

**All of Option 1, plus:**
5. Simplify checkpoint wrappers (inline or reduce)
6. Move `print_experiment_header` to `training.py` (consolidate logging)
7. Extract model initialization logic from `create_model()`
8. Add `load_datasets()` helper function

**Lines Saved:** ~40-50  
**Risk:** Low  
**Benefit:** Better organization, clearer responsibilities

---

### **Option 3: Aggressive**

**All of Option 2, plus:**
9. Create `Trainer` class to encapsulate training state
10. Create `ExperimentConfig` class to extend `TrainingConfig`
11. Merge related functions into class methods
12. Refactor training loop into smaller methods

**Lines Saved:** ~80-100  
**Risk:** Medium (changes API)  
**Benefit:** More maintainable, but requires updating all experiment scripts

---

## 📝 Recommended Changes (Option 1)

### **Change 1: Extract Loss Computation**

**File:** `training.py`

**Add after line 30:**
```python
def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: CrossEntropyLoss,
) -> torch.Tensor:
    """
    Compute cross-entropy loss from logits and targets.
    
    Flattens logits and targets before computing loss.
    
    Args:
        logits: Model output of shape (batch_size, seq_len, vocab_size)
        targets: Target token IDs of shape (batch_size, seq_len)
        loss_fn: CrossEntropyLoss instance
    
    Returns:
        Scalar loss tensor
    """
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    return loss_fn(logits_flat, targets_flat)
```

**Update 3 locations:**
- Line 68-71 in `estimate_loss()`
- Line 127-129 in `train_step()`

**Lines saved:** 6 lines (3 occurrences × 3 lines - 1 function definition)

---

### **Change 2: Consolidate Device Utilities**

**File:** `utils.py`

**Add after line 60:**
```python
def setup_device(requested_device: str = "cuda", verbose: bool = True) -> str:
    """
    Check device availability and optionally print GPU info.
    
    Args:
        requested_device: Requested device ('cuda' or 'cpu')
        verbose: Whether to print device information
    
    Returns:
        Valid device string ('cuda' or 'cpu')
    """
    if requested_device == "cuda" and torch.cuda.is_available():
        if verbose:
            print(f"\n📊 GPU Information:")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        return "cuda"
    else:
        if verbose:
            if requested_device == "cuda":
                print("\n⚠️  CUDA not available, falling back to CPU\n")
            else:
                print("\n⚠️  Using CPU\n")
        return "cpu"
```

**Remove from `experiment_utils.py`:**
- Lines 17-24 (`print_gpu_info`)
- Lines 163-176 (`check_device`)

**Update imports in experiment scripts**

**Lines saved:** ~15 lines

---

### **Change 3: Extract W&B Helper**

**File:** `training.py`

**Add after line 176:**
```python
def safe_wandb_call(func, *args, **kwargs):
    """
    Safely call a wandb function, handling import errors gracefully.
    
    Args:
        func: Function name as string (e.g., 'log', 'init', 'finish')
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Result of the function call, or None if wandb not available
    """
    try:
        import wandb
        wandb_func = getattr(wandb, func)
        return wandb_func(*args, **kwargs)
    except (ImportError, AttributeError):
        return None
```

**Update 3 locations:**
- Line 173 in `log_metrics()`
- Line 258 in `initialize_wandb()`
- Line 407 in `train()`

**Lines saved:** ~8 lines

---

### **Change 4: Add Checkpoint Path Helper**

**File:** `training.py`

**Add after line 176:**
```python
def get_checkpoint_paths(checkpoint_dir: str, iteration: int) -> tuple[str, str]:
    """
    Generate checkpoint file paths.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        iteration: Current iteration number
    
    Returns:
        Tuple of (numbered_checkpoint_path, latest_checkpoint_path)
    """
    numbered = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pt")
    latest = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    return numbered, latest
```

**Update in `save_checkpoint()`:**
- Lines 200-210

**Lines saved:** ~3 lines

---

## 📊 Summary of Option 1

| Change | Lines Saved | Risk | Files Modified |
|--------|-------------|------|----------------|
| Extract `compute_loss()` | 6 | Very Low | training.py |
| Consolidate device utils | 15 | Low | utils.py, experiment_utils.py, 3 experiment scripts |
| Extract W&B helper | 8 | Very Low | training.py |
| Add checkpoint path helper | 3 | Very Low | training.py |
| **Total** | **~32 lines** | **Low** | **6 files** |

---

## 🎯 Recommendation

**Implement Option 1** for the following reasons:

1. **Minimal Risk:** Changes are localized and don't affect the API
2. **Clear Benefit:** Reduces duplication and improves readability
3. **Easy to Verify:** Each change can be tested independently
4. **Assignment-Friendly:** Doesn't require major restructuring

**Next Steps:**
1. Implement the 4 changes above
2. Run tests to verify nothing breaks
3. Update experiment scripts to use new device utility
4. Verify all experiments still run correctly

---

## 🔍 Detailed Line-by-Line Analysis

### **utils.py (100 lines)**

| Lines | Function | Status | Notes |
|-------|----------|--------|-------|
| 1-60 | `get_batch()` | ✅ Optimal | Well-documented, efficient implementation |
| 67-78 | `save_checkpoint()` | ✅ Good | Simple, focused function |
| 81-99 | `load_checkpoint()` | ✅ Good | Simple, focused function |

**Verdict:** Minimal, well-structured. Only add `setup_device()` helper.

---

### **training.py (410 lines)**

| Lines | Function | Status | Notes |
|-------|----------|--------|-------|
| 19-24 | `set_seed()` | ✅ Optimal | Simple utility |
| 27-29 | `count_parameters()` | ✅ Optimal | One-liner helper |
| 32-75 | `estimate_loss()` | ⚠️ Has duplication | Lines 68-71 repeated |
| 78-144 | `train_step()` | ⚠️ Has duplication | Lines 127-129 repeated |
| 147-175 | `log_metrics()` | ⚠️ W&B pattern | Lines 170-175 repeated |
| 178-220 | `save_checkpoint()` | ⚠️ Wrapper | Could be simplified |
| 223-244 | `load_checkpoint()` | ⚠️ Wrapper | Could be simplified |
| 247-271 | `initialize_wandb()` | ⚠️ W&B pattern | Try-except repeated |
| 274-313 | `create_model()` | ✅ Good | Clear logic |
| 316-409 | `train()` | ⚠️ Long | Could extract helpers |

**Verdict:** Good structure, but has repetition. Apply Option 1 changes.

---

### **experiment_utils.py (177 lines)**

| Lines | Function | Status | Notes |
|-------|----------|--------|-------|
| 17-24 | `print_gpu_info()` | ⚠️ Duplicate | Overlaps with `check_device()` |
| 27-127 | `create_base_config()` | ✅ Good | Central config creation |
| 130-143 | `print_experiment_header()` | ✅ Good | Could move to training.py |
| 146-160 | `handle_oom_error()` | ✅ Good | Experiment-specific |
| 163-176 | `check_device()` | ⚠️ Duplicate | Overlaps with `print_gpu_info()` |

**Verdict:** Good utilities, but device functions should be consolidated.

---

## ✅ Conclusion

Your code is **already well-structured and minimal**. The proposed Option 1 changes will:
- Reduce duplication by ~32 lines
- Improve code clarity
- Make future changes easier
- Maintain backward compatibility

**Would you like me to implement Option 1 changes?**

