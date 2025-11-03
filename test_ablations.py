#!/usr/bin/env python3
"""
Quick test to verify ablation models can be instantiated and run forward pass.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from cs336_basics.ablation_models import TransformerLMAblation
from cs336_basics.models import TransformerLM

def test_model(ablation_type: str, use_rope: bool = True):
    """Test a single ablation model."""
    print(f"\n{'='*60}")
    print(f"Testing: {ablation_type}")
    print(f"{'='*60}")
    
    # Model config
    vocab_size = 10000
    context_length = 256
    num_layers = 2  # Small for quick test
    d_model = 128
    num_heads = 4
    d_ff = 256
    batch_size = 2
    seq_len = 32
    
    try:
        # Create model
        if ablation_type == "baseline":
            model = TransformerLM(
                vocab_size=vocab_size,
                context_length=context_length,
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_rope=use_rope,
                theta=10000.0,
                device="cpu"
            )
        else:
            model = TransformerLMAblation(
                vocab_size=vocab_size,
                context_length=context_length,
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_rope=use_rope,
                ablation_type=ablation_type,
                theta=10000.0,
                device="cpu"
            )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created: {num_params:,} parameters")
        
        # Test forward pass
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (batch_size, seq_len, vocab_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Forward pass successful: {output.shape}")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        print(f"✓ Backward pass successful")
        
        # Check for NaN
        has_nan = torch.isnan(output).any()
        assert not has_nan, "Output contains NaN!"
        print(f"✓ No NaN values")
        
        print(f"✅ {ablation_type} test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ {ablation_type} test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("ABLATION MODELS TEST SUITE")
    print("="*60)
    
    tests = [
        ("baseline", True),
        ("no_rmsnorm", True),
        ("post_norm", True),
        ("no_rope", False),  # NoPE
        ("silu_only", True),
    ]
    
    results = {}
    for ablation_type, use_rope in tests:
        results[ablation_type] = test_model(ablation_type, use_rope)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for ablation_type, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{ablation_type:<20} {status}")
    
    all_passed = all(results.values())
    print("="*60)
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

