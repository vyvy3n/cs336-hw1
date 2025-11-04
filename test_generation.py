#!/usr/bin/env python3
"""
Simple tests for the decoder module.

This script tests the decoding functions with a small dummy model.
"""

import torch
import torch.nn as nn
from cs336_basics.decoder import sample_from_logits, generate


class DummyModel(nn.Module):
    """
    A simple dummy model for testing generation.
    Always predicts the same distribution regardless of input.
    """
    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size
        # Create a simple linear layer
        self.linear = nn.Linear(vocab_size, vocab_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Create dummy embeddings
        embeddings = torch.zeros(batch_size, seq_len, self.vocab_size)
        
        # Simple transformation
        logits = self.linear(embeddings)
        
        return logits


def test_sample_from_logits():
    """Test the sample_from_logits function."""
    print("Testing sample_from_logits...")
    
    # Create simple logits
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test basic sampling
    token = sample_from_logits(logits, temperature=1.0)
    assert token.shape == torch.Size([1])
    assert 0 <= token.item() < 5
    print("  ✓ Basic sampling works")
    
    # Test temperature scaling
    # Low temperature should favor higher logits
    samples = []
    for _ in range(100):
        token = sample_from_logits(logits, temperature=0.1)
        samples.append(token.item())
    # Should mostly sample token 4 (highest logit)
    assert samples.count(4) > 50
    print("  ✓ Low temperature sampling works")
    
    # Test top-p sampling
    token = sample_from_logits(logits, temperature=1.0, top_p=0.5)
    assert token.shape == torch.Size([1])
    print("  ✓ Top-p sampling works")
    
    print("✓ All sample_from_logits tests passed!\n")


def test_generate():
    """Test the generate function."""
    print("Testing generate...")
    
    # Create dummy model
    vocab_size = 100
    model = DummyModel(vocab_size=vocab_size)
    model.eval()
    
    # Create dummy prompt
    prompt = torch.tensor([[1, 2, 3]])  # shape: (1, 3)
    
    # Test basic generation
    generated = generate(
        model=model,
        prompt=prompt,
        max_tokens=10,
        temperature=1.0,
    )
    assert generated.shape[0] == 1  # batch size
    assert generated.shape[1] == 3 + 10  # prompt + generated
    print("  ✓ Basic generation works")
    
    # Test with 1D prompt
    prompt_1d = torch.tensor([1, 2, 3])
    generated = generate(
        model=model,
        prompt=prompt_1d,
        max_tokens=5,
        temperature=1.0,
    )
    assert generated.shape[0] == 1
    assert generated.shape[1] == 3 + 5
    print("  ✓ 1D prompt handling works")
    
    # Test with EOS token
    eos_token_id = 50
    generated = generate(
        model=model,
        prompt=prompt,
        max_tokens=100,
        temperature=1.0,
        eos_token_id=eos_token_id,
    )
    # Should stop early if EOS is generated
    assert generated.shape[1] <= 3 + 100
    print("  ✓ EOS token handling works")
    
    # Test with different temperatures
    generated_low = generate(
        model=model,
        prompt=prompt,
        max_tokens=10,
        temperature=0.1,
    )
    generated_high = generate(
        model=model,
        prompt=prompt,
        max_tokens=10,
        temperature=2.0,
    )
    assert generated_low.shape == generated_high.shape
    print("  ✓ Temperature parameter works")
    
    # Test with top-p
    generated_topp = generate(
        model=model,
        prompt=prompt,
        max_tokens=10,
        temperature=1.0,
        top_p=0.9,
    )
    assert generated_topp.shape[1] == 3 + 10
    print("  ✓ Top-p parameter works")
    
    print("✓ All generate tests passed!\n")


def test_generation_determinism():
    """Test that generation is deterministic with fixed seed."""
    print("Testing generation determinism...")
    
    vocab_size = 100
    model = DummyModel(vocab_size=vocab_size)
    model.eval()
    
    prompt = torch.tensor([[1, 2, 3]])
    
    # Generate with seed
    torch.manual_seed(42)
    generated1 = generate(
        model=model,
        prompt=prompt,
        max_tokens=10,
        temperature=1.0,
    )
    
    # Generate again with same seed
    torch.manual_seed(42)
    generated2 = generate(
        model=model,
        prompt=prompt,
        max_tokens=10,
        temperature=1.0,
    )
    
    # Should be identical
    assert torch.equal(generated1, generated2)
    print("  ✓ Generation is deterministic with fixed seed")
    
    print("✓ Determinism test passed!\n")


def test_batch_generation():
    """Test generation with batch size > 1."""
    print("Testing batch generation...")
    
    vocab_size = 100
    model = DummyModel(vocab_size=vocab_size)
    model.eval()
    
    # Create batch of prompts
    batch_size = 3
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    generated = generate(
        model=model,
        prompt=prompt,
        max_tokens=10,
        temperature=1.0,
    )
    
    assert generated.shape[0] == batch_size
    assert generated.shape[1] == 3 + 10
    print("  ✓ Batch generation works")
    
    print("✓ Batch generation test passed!\n")


def main():
    """Run all tests."""
    print("="*80)
    print("Running Decoder Module Tests")
    print("="*80 + "\n")
    
    try:
        test_sample_from_logits()
        test_generate()
        test_generation_determinism()
        test_batch_generation()
        
        print("="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()

