#!/usr/bin/env python3
"""
Diagnostic script to identify why ablation experiments failed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from cs336_basics.config import TrainingConfig, ModelConfig, OptimizerConfig, SchedulerConfig, DataConfig
from cs336_basics.ablation_models import TransformerLMAblation

def test_ablation_config(ablation_type: str):
    """Test if ablation configuration works."""
    print(f"\n{'='*60}")
    print(f"Testing: {ablation_type}")
    print(f"{'='*60}")
    
    try:
        # Create config
        model_config = ModelConfig(
            vocab_size=10000,
            context_length=256,
            num_layers=4,
            d_model=512,
            num_heads=16,
            d_ff=1344,
            use_rope=True if ablation_type != "no_rope" else False,
            theta=10000.0,
            ablation_type=ablation_type if ablation_type != "no_rope" else "none",
        )
        print(f"✓ Config created: ablation_type={model_config.ablation_type}, use_rope={model_config.use_rope}")
        
        # Create model
        if model_config.ablation_type != "none":
            model = TransformerLMAblation(
                vocab_size=model_config.vocab_size,
                context_length=model_config.context_length,
                num_layers=model_config.num_layers,
                d_model=model_config.d_model,
                num_heads=model_config.num_heads,
                d_ff=model_config.d_ff,
                use_rope=model_config.use_rope,
                ablation_type=model_config.ablation_type,
                theta=model_config.theta,
                device='cpu'
            )
        else:
            from cs336_basics.models import TransformerLM
            model = TransformerLM(
                vocab_size=model_config.vocab_size,
                context_length=model_config.context_length,
                num_layers=model_config.num_layers,
                d_model=model_config.d_model,
                num_heads=model_config.num_heads,
                d_ff=model_config.d_ff,
                use_rope=model_config.use_rope,
                theta=model_config.theta,
                device='cpu'
            )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {num_params:,} parameters")
        
        # Test forward pass
        x = torch.randint(0, 10000, (2, 256))
        output = model(x)
        print(f"✓ Forward pass: {output.shape}")
        
        # Test loss computation
        targets = torch.randint(0, 10000, (2, 256))
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output.view(-1, 10000), targets.view(-1))
        print(f"✓ Loss computation: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass")
        
        # Check for NaN
        has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        if has_nan:
            print(f"⚠️  NaN in gradients!")
        else:
            print(f"✓ No NaN in gradients")
        
        print(f"✅ {ablation_type} test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ {ablation_type} test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_files():
    """Check if data files exist."""
    print(f"\n{'='*60}")
    print("Checking data files...")
    print(f"{'='*60}")
    
    train_path = Path("data/tinystories_train_tokens.npy")
    val_path = Path("data/tinystories_valid_tokens.npy")
    
    if train_path.exists():
        print(f"✓ Training data exists: {train_path}")
    else:
        print(f"❌ Training data missing: {train_path}")
        
    if val_path.exists():
        print(f"✓ Validation data exists: {val_path}")
    else:
        print(f"❌ Validation data missing: {val_path}")


def check_checkpoint_dirs():
    """Check checkpoint directories."""
    print(f"\n{'='*60}")
    print("Checking checkpoint directories...")
    print(f"{'='*60}")
    
    ablation_dir = Path("checkpoints/ablations")
    if ablation_dir.exists():
        print(f"✓ Ablations directory exists: {ablation_dir}")
        subdirs = list(ablation_dir.iterdir())
        if subdirs:
            print(f"  Found {len(subdirs)} subdirectories:")
            for subdir in subdirs:
                files = list(subdir.glob("*"))
                print(f"    - {subdir.name}: {len(files)} files")
        else:
            print(f"  ⚠️  Directory is empty")
    else:
        print(f"❌ Ablations directory missing: {ablation_dir}")


def main():
    print("\n" + "="*60)
    print("ABLATION EXPERIMENTS DIAGNOSTIC")
    print("="*60)
    
    # Check data files
    check_data_files()
    
    # Check checkpoint directories
    check_checkpoint_dirs()
    
    # Test each ablation type
    ablations = [
        "no_rmsnorm",
        "post_norm",
        "no_rope",
        "silu_only",
    ]
    
    results = {}
    for ablation in ablations:
        results[ablation] = test_ablation_config(ablation)
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    for ablation, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{ablation:<20} {status}")
    
    all_passed = all(results.values())
    print("="*60)
    if all_passed:
        print("🎉 All diagnostics passed!")
        print("\nThe models work correctly. The issue is likely:")
        print("  1. Training script error (check logs)")
        print("  2. W&B logging issue")
        print("  3. Early termination (OOM, keyboard interrupt, etc.)")
        return 0
    else:
        print("⚠️  Some diagnostics failed - see errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

