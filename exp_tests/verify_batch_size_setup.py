#!/usr/bin/env python3
"""
Verify that the batch size experiment setup is correct.

This script checks:
1. All required files exist
2. Scripts can be imported
3. Configuration is valid
4. Data files are accessible

Usage:
    python experiments/verify_batch_size_setup.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_files():
    """Check that all required files exist."""
    print("\n" + "="*80)
    print("CHECKING FILES")
    print("="*80)
    
    required_files = [
        "experiments/batch_size_sweep.py",
        "experiments/test_batch_size.py",
        "experiments/BATCH_SIZE_EXPERIMENT.md",
        "experiments/BATCH_SIZE_QUICKSTART.md",
        "BATCH_SIZE_EXPERIMENT_SUMMARY.md",
        "data/tinystories_train_tokens.npy",
        "data/tinystories_valid_tokens.npy",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(file_path)
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False
    
    print("="*80)
    return all_exist


def check_imports():
    """Check that scripts can be imported."""
    print("\n" + "="*80)
    print("CHECKING IMPORTS")
    print("="*80)
    
    try:
        from batch_size_sweep import get_base_config, run_single_experiment, batch_size_sweep
        print("  ✓ batch_size_sweep.py imports successfully")
        print("    - get_base_config")
        print("    - run_single_experiment")
        print("    - batch_size_sweep")
    except Exception as e:
        print(f"  ✗ Failed to import batch_size_sweep.py: {e}")
        return False
    
    try:
        from test_batch_size import quick_test
        print("  ✓ test_batch_size.py imports successfully")
        print("    - quick_test")
    except Exception as e:
        print(f"  ✗ Failed to import test_batch_size.py: {e}")
        return False
    
    print("="*80)
    return True


def check_config():
    """Check that configuration is valid."""
    print("\n" + "="*80)
    print("CHECKING CONFIGURATION")
    print("="*80)
    
    try:
        from batch_size_sweep import get_base_config
        
        # Test different batch sizes
        batch_sizes = [1, 32, 128]
        for bs in batch_sizes:
            config = get_base_config(batch_size=bs, learning_rate=3e-4)
            total_steps = config.scheduler.max_iters
            total_tokens = bs * total_steps * config.data.context_length
            print(f"  ✓ batch_size={bs}")
            print(f"    - total_steps: {total_steps:,}")
            print(f"    - total_tokens: {total_tokens:,}")
            print(f"    - context_length: {config.data.context_length}")
            
            # Verify total tokens is constant
            expected_tokens = 327_680_000
            if abs(total_tokens - expected_tokens) > 1000:
                print(f"    ⚠ Warning: total_tokens ({total_tokens:,}) != expected ({expected_tokens:,})")
        
        print("="*80)
        return True
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        print("="*80)
        return False


def check_gpu():
    """Check GPU availability."""
    print("\n" + "="*80)
    print("CHECKING GPU")
    print("="*80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    - Device: {torch.cuda.get_device_name(0)}")
            print(f"    - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"    - CUDA version: {torch.version.cuda}")
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
        
        print("="*80)
        return True
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        print("="*80)
        return False


def check_data():
    """Check data files."""
    print("\n" + "="*80)
    print("CHECKING DATA")
    print("="*80)
    
    try:
        import numpy as np
        
        train_data = np.load("data/tinystories_train_tokens.npy")
        val_data = np.load("data/tinystories_valid_tokens.npy")
        
        print(f"  ✓ Training data loaded")
        print(f"    - Shape: {train_data.shape}")
        print(f"    - Size: {train_data.size:,} tokens")
        print(f"    - Dtype: {train_data.dtype}")
        
        print(f"  ✓ Validation data loaded")
        print(f"    - Shape: {val_data.shape}")
        print(f"    - Size: {val_data.size:,} tokens")
        print(f"    - Dtype: {val_data.dtype}")
        
        print("="*80)
        return True
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        print("="*80)
        return False


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    print("\n1. Quick Test (5-10 minutes):")
    print("   python experiments/test_batch_size.py --device cuda")
    print("\n2. Full Experiment (2-3 hours):")
    print("   tmux new -s batch_size_exp")
    print("   python experiments/batch_size_sweep.py --device cuda")
    print("   # Detach: Ctrl+b, then d")
    print("\n3. With LR Optimization:")
    print("   python experiments/batch_size_sweep.py --device cuda --optimize_lr")
    print("\n4. Test Specific Batch Sizes:")
    print("   python experiments/batch_size_sweep.py --batch_sizes 32 64 128")
    print("\n5. View Results:")
    print("   https://wandb.ai/YOUR-USERNAME/cs336-batch-size-sweep")
    print("="*80 + "\n")


def main():
    print("\n" + "="*80)
    print("BATCH SIZE EXPERIMENT SETUP VERIFICATION")
    print("="*80)
    
    checks = [
        ("Files", check_files),
        ("Imports", check_imports),
        ("Configuration", check_config),
        ("GPU", check_gpu),
        ("Data", check_data),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ Error in {name} check: {e}")
            results[name] = False
    
    # Print summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✅ All checks passed! Ready to run experiments.")
        print_usage()
        return 0
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

