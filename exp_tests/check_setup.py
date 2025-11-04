#!/usr/bin/env python3
"""
Check Experiment Setup

Verifies that everything is ready to run learning rate experiments.

Usage:
    python experiments/check_setup.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (need >= 3.8)")
        return False


def check_pytorch():
    """Check PyTorch installation."""
    print("\nChecking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  ✓ MPS (Apple Silicon) available")
        
        return True
    except ImportError:
        print(f"  ✗ PyTorch not installed")
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\nChecking dependencies...")
    
    required = {
        'numpy': 'numpy',
        'tqdm': 'tqdm',
        'einops': 'einops',
    }
    
    optional = {
        'wandb': 'wandb',
        'tokenizers': 'tokenizers',
    }
    
    all_good = True
    
    for name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (required)")
            all_good = False
    
    for name, import_name in optional.items():
        try:
            __import__(import_name)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} (optional, but recommended)")
    
    return all_good


def check_data_files():
    """Check if data files exist."""
    print("\nChecking data files...")
    
    data_dir = Path("data")
    
    required_files = [
        "tinystories_train_tokens.npy",
        "tinystories_valid_tokens.npy",
    ]
    
    optional_files = [
        "tokenizer_v10000.json",
    ]
    
    all_good = True
    
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename} (required)")
            all_good = False
    
    for filename in optional_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ⚠ {filename} (optional)")
    
    if not all_good:
        print("\n  To prepare data, run:")
        print("    uv run python scripts/prepare_tinystories.py --vocab_size 10000")
    
    return all_good


def check_wandb():
    """Check W&B setup."""
    print("\nChecking Weights & Biases...")
    
    try:
        import wandb
        print(f"  ✓ wandb installed")
        
        # Check if logged in
        try:
            api = wandb.Api()
            user = api.viewer
            print(f"  ✓ Logged in as: {user.username}")
            return True
        except Exception as e:
            print(f"  ⚠ Not logged in")
            print(f"    Run: wandb login")
            return False
    except ImportError:
        print(f"  ⚠ wandb not installed (optional)")
        print(f"    Install with: pip install wandb")
        return False


def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        free_gb = free / (1024 ** 3)
        print(f"  Available: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print(f"  ⚠ Low disk space (need ~10GB for checkpoints)")
            return False
        else:
            print(f"  ✓ Sufficient disk space")
            return True
    except Exception as e:
        print(f"  ⚠ Could not check disk space: {e}")
        return True


def check_model_import():
    """Check if model can be imported."""
    print("\nChecking model imports...")

    try:
        from cs336_basics.models import TransformerLM
        from cs336_basics.training import train
        from cs336_basics.config import TrainingConfig
        print(f"  ✓ All model imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def run_quick_test():
    """Run a very quick forward pass test."""
    print("\nRunning quick model test...")

    try:
        import torch
        from cs336_basics.models import TransformerLM
        from cs336_basics.config import ModelConfig

        # Create tiny model
        config = ModelConfig(
            vocab_size=1000,
            context_length=64,
            num_layers=2,
            d_model=128,
            num_heads=4,
            d_ff=256,
            use_rope=True,
            theta=10000.0,
        )

        model = TransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            use_rope=config.use_rope,
            theta=config.theta,
        )
        
        # Forward pass
        x = torch.randint(0, 1000, (2, 32))  # batch_size=2, seq_len=32
        logits = model(x)
        
        expected_shape = (2, 32, 1000)
        if logits.shape == expected_shape:
            print(f"  ✓ Model forward pass successful")
            print(f"    Output shape: {logits.shape}")
            return True
        else:
            print(f"  ✗ Unexpected output shape: {logits.shape} (expected {expected_shape})")
            return False
            
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("Experiment Setup Check")
    print("="*80)
    
    checks = [
        ("Python version", check_python_version),
        ("PyTorch", check_pytorch),
        ("Dependencies", check_dependencies),
        ("Data files", check_data_files),
        ("W&B setup", check_wandb),
        ("Disk space", check_disk_space),
        ("Model imports", check_model_import),
        ("Model test", run_quick_test),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n  ✗ Error during {name} check: {e}")
            results[name] = False
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    critical_checks = ["Python version", "PyTorch", "Dependencies", "Data files", "Model imports", "Model test"]
    optional_checks = ["W&B setup", "Disk space"]
    
    critical_passed = all(results.get(check, False) for check in critical_checks)
    optional_passed = all(results.get(check, False) for check in optional_checks)
    
    for name, passed in results.items():
        status = "✓" if passed else ("⚠" if name in optional_checks else "✗")
        print(f"  {status} {name}")
    
    print("="*80)
    
    if critical_passed and optional_passed:
        print("\n🎉 All checks passed! You're ready to run experiments.")
        print("\nNext steps:")
        print("  1. Quick test: uv run python experiments/quick_lr_test.py --max_iters 100 --device cuda")
        print("  2. Full sweep: uv run python experiments/learning_rate_sweep.py --sweep_type grid --device cuda")
    elif critical_passed:
        print("\n✓ Critical checks passed. You can run experiments.")
        print("⚠ Some optional features may not work (e.g., W&B logging).")
        print("\nNext steps:")
        print("  1. Quick test: uv run python experiments/quick_lr_test.py --max_iters 100 --device cuda")
        print("  2. Full sweep: uv run python experiments/learning_rate_sweep.py --sweep_type grid --device cuda --no_wandb")
    else:
        print("\n✗ Some critical checks failed. Please fix the issues above before running experiments.")
        
        if not results.get("Data files", False):
            print("\nTo prepare data:")
            print("  uv run python scripts/prepare_tinystories.py --vocab_size 10000")
        
        if not results.get("W&B setup", False):
            print("\nTo setup W&B:")
            print("  wandb login")
    
    print()


if __name__ == "__main__":
    main()

