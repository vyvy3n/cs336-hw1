# 🏗️ Trainer Architecture Explained

## Your Two Questions:

1. **How does the Trainer class support resume/checkpointing?**
2. **Why do we have a separate `train_owt()` function when we already have a Trainer class?**

---

## 📚 Part 1: How Resume/Checkpointing Works (Line by Line)

### Step 1: Checkpoint Saving (`utils.py`)

<augment_code_snippet path="cs336-hw1/cs336_basics/utils.py" mode="EXCERPT">
````python
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        'model': model.state_dict(),      # Line 74: Save all model weights
        'optimizer': optimizer.state_dict(),  # Line 75: Save optimizer state (momentum, etc.)
        'iteration': iteration,            # Line 76: Save current iteration number
    }
    torch.save(checkpoint, out)           # Line 78: Serialize to disk
````
</augment_code_snippet>

**What each line does:**
- **Line 74:** `model.state_dict()` - Extracts all model parameters (weights, biases) as a dictionary
- **Line 75:** `optimizer.state_dict()` - Extracts optimizer state (momentum buffers, learning rate, etc.)
- **Line 76:** `iteration` - Saves the current training iteration number
- **Line 78:** `torch.save()` - Serializes the entire dictionary to a `.pt` file

**Why we need all three:**
- **Model weights** → Resume with same learned parameters
- **Optimizer state** → Resume with same momentum/adaptive learning rates
- **Iteration number** → Resume from correct training step

---

### Step 2: Checkpoint Loading (`utils.py`)

<augment_code_snippet path="cs336-hw1/cs336_basics/utils.py" mode="EXCERPT">
````python
def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, weights_only=False)  # Line 88: Load from disk
    
    model.load_state_dict(checkpoint['model'])        # Line 92: Restore model weights
    
    optimizer.load_state_dict(checkpoint['optimizer']) # Line 96: Restore optimizer state
    
    return checkpoint['iteration']                     # Line 99: Return iteration number
````
</augment_code_snippet>

**What each line does:**
- **Line 88:** `torch.load()` - Deserializes the `.pt` file into a Python dictionary
- **Line 92:** `model.load_state_dict()` - Restores all model parameters **in-place**
- **Line 96:** `optimizer.load_state_dict()` - Restores optimizer state **in-place**
- **Line 99:** Returns the iteration number so training loop knows where to start

**Key insight:** `load_state_dict()` modifies the model/optimizer **in-place**, so the same objects are updated.

---

### Step 3: Trainer's `load_checkpoint()` Method (`training.py`)

<augment_code_snippet path="cs336-hw1/cs336_basics/training.py" mode="EXCERPT">
````python
def load_checkpoint(self, checkpoint_path: str):
    print(f"Loading checkpoint from {checkpoint_path}")           # Line 302: User feedback
    self.current_iter = load_checkpoint_impl(                     # Line 303: Call utils function
        checkpoint_path, 
        self.model,      # Pass model to be updated in-place
        self.optimizer   # Pass optimizer to be updated in-place
    )
    print(f"Resumed from iteration {self.current_iter}")          # Line 304: Confirm
````
</augment_code_snippet>

**What happens:**
1. **Line 302:** Prints which checkpoint is being loaded
2. **Line 303:** Calls `load_checkpoint_impl()` which:
   - Updates `self.model` in-place with saved weights
   - Updates `self.optimizer` in-place with saved state
   - Returns the iteration number
3. **Line 303 (assignment):** Stores returned iteration in `self.current_iter`
4. **Line 304:** Confirms successful resume

---

### Step 4: Resume Logic in `train()` Method (`training.py`)

<augment_code_snippet path="cs336-hw1/cs336_basics/training.py" mode="EXCERPT">
````python
def train(self):
    # Resume from checkpoint if specified
    if self.config.resume_from is not None:              # Line 318: Check if resume requested
        self.load_checkpoint(self.config.resume_from)    # Line 319: Load checkpoint
    
    start_iter = self.current_iter                       # Line 321: Get starting iteration
    
    print(f"\nStarting training from iteration {start_iter} to {self.config.scheduler.max_iters}")
    
    # Training loop
    for iteration in tqdm(
        range(start_iter, self.config.scheduler.max_iters),  # Line 332: Start from start_iter
        initial=start_iter,                                   # Line 333: Set progress bar
        total=self.config.scheduler.max_iters
    ):
        self.current_iter = iteration
        # ... training continues ...
````
</augment_code_snippet>

**What each line does:**
- **Line 318:** Checks if `config.resume_from` is set (not None)
- **Line 319:** If yes, loads checkpoint (updates model, optimizer, current_iter)
- **Line 321:** Stores starting iteration (0 if fresh start, or loaded value if resuming)
- **Line 332:** Training loop starts from `start_iter` instead of 0
- **Line 333:** Progress bar shows correct position

**Key insight:** If resuming from iteration 39,000:
- `start_iter = 39000`
- Loop runs: `range(39000, 100000)` → trains for 61,000 more iterations
- Progress bar shows 39% complete at start

---

## 🎯 Part 2: Why Have Both Trainer Class AND `train_owt()` Function?

### The Architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    train_owt() Function                      │
│                  (experiments/train_owt.py)                  │
│                                                              │
│  Role: Configuration & Setup                                │
│  - Parse command-line arguments                             │
│  - Create TrainingConfig with specific hyperparameters      │
│  - Set up experiment-specific settings (W&B, paths, etc.)   │
│  - Print experiment header                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Creates config and passes to:
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Trainer Class                           │
│                  (cs336_basics/training.py)                  │
│                                                              │
│  Role: Core Training Logic                                  │
│  - Initialize model, optimizer, datasets                    │
│  - Run training loop                                        │
│  - Handle checkpointing, evaluation, logging                │
│  - Manage early stopping                                    │
│  - Resume from checkpoints                                  │
└─────────────────────────────────────────────────────────────┘
```

---

### Why This Separation?

#### **Trainer Class** = Reusable Training Engine

**Purpose:** Generic, reusable training logic that works for ANY experiment

**Responsibilities:**
1. ✅ Training loop (forward pass, backward pass, optimizer step)
2. ✅ Evaluation on validation set
3. ✅ Checkpointing (save/load)
4. ✅ Learning rate scheduling
5. ✅ Gradient clipping
6. ✅ Logging (console + W&B)
7. ✅ Early stopping

**Key point:** The Trainer doesn't know or care about:
- Which dataset you're using (TinyStories vs OWT)
- What hyperparameters you chose
- What experiment you're running
- How you want to name your W&B runs

---

#### **`train_owt()` Function** = Experiment-Specific Configuration

**Purpose:** Set up a specific experiment with specific hyperparameters

**Responsibilities:**
1. ✅ Parse command-line arguments
2. ✅ Create `TrainingConfig` with OWT-specific settings:
   - `vocab_size=32000` (OWT-specific)
   - `train_data_path="data/owt_train_tokens.npy"` (OWT-specific)
   - `val_data_path="data/owt_valid_tokens.npy"` (OWT-specific)
3. ✅ Set experiment-specific W&B project name
4. ✅ Set checkpoint directory
5. ✅ Print experiment header
6. ✅ Create Trainer and call `trainer.train()`

---

### Analogy: Car vs Driver

| Component | Analogy | Code |
|-----------|---------|------|
| **Trainer Class** | The car engine | Generic training logic |
| **`train_owt()` Function** | The driver's settings | Specific experiment configuration |
| **TrainingConfig** | Dashboard controls | Hyperparameters, paths, settings |

**The car engine (Trainer):**
- Doesn't care where you're driving
- Just knows how to drive (train)
- Can be reused for any trip

**The driver (train_owt):**
- Decides destination (OWT dataset)
- Sets GPS (checkpoint paths)
- Chooses route (hyperparameters)
- Starts the engine (creates Trainer)

---

### Code Example: How They Work Together

<augment_code_snippet path="cs336-hw1/experiments/train_owt.py" mode="EXCERPT">
````python
def train_owt(...):
    # 1. Create experiment-specific configuration
    config = TrainingConfig(
        model=ModelConfig(vocab_size=32000, ...),  # OWT-specific
        data=DataConfig(
            train_data_path="data/owt_train_tokens.npy",  # OWT-specific
            val_data_path="data/owt_valid_tokens.npy",    # OWT-specific
        ),
        checkpoint_dir=checkpoint_dir,  # From command-line
        wandb_project=wandb_project,    # From command-line
        resume_from=resume_from,        # From command-line
        # ... more settings ...
    )
    
    # 2. Create Trainer with this config
    trainer = Trainer(config)
    
    # 3. Start training (Trainer handles everything)
    trainer.train()
````
</augment_code_snippet>

---

### Why Not Just Use Trainer Directly?

**You could**, but you'd have to:

```python
# Without train_owt() - you'd have to do this every time:
from cs336_basics.training import Trainer
from cs336_basics.config import TrainingConfig, ModelConfig, DataConfig, ...

config = TrainingConfig(
    model=ModelConfig(
        vocab_size=32000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        use_rope=True,
        theta=10000,
    ),
    data=DataConfig(
        train_data_path="data/owt_train_tokens.npy",
        val_data_path="data/owt_valid_tokens.npy",
        batch_size=32,
        context_length=256,
    ),
    optimizer=OptimizerConfig(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip_norm=1.0,
    ),
    scheduler=SchedulerConfig(
        warmup_iters=100,
        max_iters=40000,
        min_lr_ratio=0.1,
    ),
    device="cuda",
    seed=42,
    checkpoint_dir="checkpoints/owt",
    checkpoint_interval=1000,
    log_interval=50,
    eval_interval=500,
    eval_iters=100,
    use_wandb=True,
    wandb_project="cs336-owt",
    wandb_run_name="my_experiment",
)

trainer = Trainer(config)
trainer.train()
```

**That's 40+ lines of boilerplate!**

---

### With `train_owt()` - Much Cleaner:

```bash
# Just one command:
uv run python experiments/train_owt.py \
    --device cuda \
    --max_iters 100000 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --resume_from checkpoints/owt/checkpoint_iter_39000.pt \
    --use_wandb
```

**Benefits:**
1. ✅ Command-line interface (easy to use)
2. ✅ Sensible defaults (don't need to specify everything)
3. ✅ Experiment-specific logic (OWT paths, vocab size)
4. ✅ Reusable across different experiments

---

## 🎓 Summary

### How Resume Works:

1. **Save checkpoint:** Model weights + optimizer state + iteration number → `.pt` file
2. **Load checkpoint:** Read `.pt` file → restore model + optimizer + iteration
3. **Resume training:** Start loop from loaded iteration instead of 0

### Why Two Layers:

| Layer | Purpose | Reusability |
|-------|---------|-------------|
| **Trainer Class** | Generic training engine | Used by ALL experiments |
| **`train_owt()` Function** | OWT-specific configuration | Used only for OWT experiments |

### Design Pattern:

This is the **Strategy Pattern**:
- **Trainer** = The algorithm (how to train)
- **train_owt()** = The strategy (what to train)

---

## 📁 Other Experiment Functions:

You also have:
- `experiments/compare_datasets.py` - Trains on both TinyStories AND OWT
- `experiments/learning_rate_sweep.py` - Runs multiple experiments with different LRs
- `experiments/batch_size_sweep.py` - Runs multiple experiments with different batch sizes

**All of them:**
1. Create different `TrainingConfig` objects
2. Pass to `Trainer` class
3. Call `trainer.train()`

**Same engine, different configurations!** 🚗💨

