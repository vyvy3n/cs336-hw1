#!/usr/bin/env python3
"""
Find Maximum Batch Size Using Binary Search

This script uses binary search to find the exact maximum batch size that fits in GPU memory.

Algorithm:
1. Start with a wide range [1, upper_bound]
2. Test the midpoint batch size
3. If it fits, search higher [mid+1, upper_bound]
4. If OOM, search lower [1, mid-1]
5. Repeat until we find the exact maximum

Usage:
    python scripts/find_max_batch_size.py --device cuda
    python scripts/find_max_batch_size.py --device cuda --upper_bound 4096
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from cs336_basics.config import TrainingConfig, ModelConfig, OptimizerConfig, SchedulerConfig, DataConfig
from cs336_basics.models import TransformerLM


def test_batch_size(batch_size: int, device: str, num_steps: int = 10) -> tuple[bool, float]:
    """
    Test if a specific batch size fits in GPU memory.

    Args:
        batch_size: Batch size to test
        device: Device to use
        num_steps: Number of training steps to run (to ensure memory is stable)

    Returns:
        (success, peak_memory_gb): Whether it succeeded and peak memory used
    """
    print(f"\n{'='*80}")
    print(f"Testing batch_size={batch_size}")
    print(f"{'='*80}")

    try:
        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Create minimal config
        context_length = 256
        model_config = ModelConfig(
            vocab_size=10000,
            context_length=context_length,
            num_layers=4,
            d_model=512,
            num_heads=16,
            d_ff=1344,
            use_rope=True,
            theta=10000.0,
        )

        optimizer_config = OptimizerConfig(
            learning_rate=3e-4,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            grad_clip_norm=1.0,
        )

        # Create model and optimizer
        model = TransformerLM(
            vocab_size=model_config.vocab_size,
            context_length=model_config.context_length,
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            use_rope=model_config.use_rope,
            theta=model_config.theta,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )

        # Load data
        import numpy as np
        train_data = np.load("data/tinystories_train_tokens.npy")

        # Run a few training steps to ensure memory is stable
        model.train()
        for step in range(num_steps):
            # Get random batch
            indices = torch.randint(0, len(train_data) - context_length - 1, (batch_size,))
            batch = torch.stack([
                torch.from_numpy(train_data[i:i+context_length+1].astype(np.int64))
                for i in indices
            ]).to(device)

            x = batch[:, :-1]
            y = batch[:, 1:]

            # Forward pass
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")

        # Get peak memory
        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"  ✓ Success! Peak memory: {peak_memory:.2f} GB")
        else:
            peak_memory = 0.0
            print(f"  ✓ Success!")

        # Cleanup
        del model, optimizer, batch, x, y, logits, loss
        if device == "cuda":
            torch.cuda.empty_cache()

        return True, peak_memory

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ✗ Out of Memory")
            # Cleanup
            if device == "cuda":
                torch.cuda.empty_cache()
            return False, 0.0
        else:
            print(f"  ✗ Error: {e}")
            raise
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        raise


def binary_search_max_batch_size(device: str, lower: int = 1, upper: int = 8192) -> int:
    """
    Use binary search to find the maximum batch size that fits in memory.

    Args:
        device: Device to use
        lower: Lower bound for search
        upper: Upper bound for search

    Returns:
        Maximum batch size that fits in memory
    """
    print("\n" + "="*80)
    print("BINARY SEARCH FOR MAXIMUM BATCH SIZE")
    print("="*80)
    print(f"Search range: [{lower}, {upper}]")
    print(f"Device: {device}")

    if device == "cuda":
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total memory: {total_memory:.2f} GB")

    print("="*80)

    max_working = lower
    tested_sizes = {}

    while lower <= upper:
        mid = (lower + upper) // 2

        # Skip if already tested
        if mid in tested_sizes:
            if tested_sizes[mid]:
                lower = mid + 1
            else:
                upper = mid - 1
            continue

        print(f"\nSearch range: [{lower}, {upper}], testing mid={mid}")

        success, memory = test_batch_size(mid, device)
        tested_sizes[mid] = success

        if success:
            max_working = mid
            print(f"  → Success! New max: {max_working}")
            # Search higher
            lower = mid + 1
        else:
            print(f"  → Failed! Searching lower...")
            # Search lower
            upper = mid - 1

    return max_working


def exponential_search_upper_bound(device: str, start: int = 1) -> int:
    """
    Find an upper bound for binary search using exponential search.

    Start with a small batch size and keep doubling until we hit OOM.

    Args:
        device: Device to use
        start: Starting batch size

    Returns:
        Upper bound for binary search
    """
    print("\n" + "="*80)
    print("EXPONENTIAL SEARCH FOR UPPER BOUND")
    print("="*80)
    print(f"Starting from batch_size={start}, doubling until OOM...")
    print("="*80)

    batch_size = start
    last_working = start

    while True:
        success, memory = test_batch_size(batch_size, device, num_steps=5)

        if success:
            last_working = batch_size
            print(f"  → batch_size={batch_size} works, trying {batch_size * 2}...")
            batch_size *= 2
        else:
            print(f"  → batch_size={batch_size} failed!")
            print(f"  → Upper bound found: {batch_size}")
            return batch_size


def main():
    parser = argparse.ArgumentParser(description="Find Maximum Batch Size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--lower_bound",
        type=int,
        default=1,
        help="Lower bound for binary search"
    )
    parser.add_argument(
        "--upper_bound",
        type=int,
        default=None,
        help="Upper bound for binary search (if None, will find automatically)"
    )
    parser.add_argument(
        "--skip_exponential",
        action="store_true",
        help="Skip exponential search and use provided upper_bound"
    )

    args = parser.parse_args()

    # Check GPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Find upper bound if not provided
    if args.upper_bound is None:
        if args.skip_exponential:
            args.upper_bound = 8192
            print(f"Using default upper bound: {args.upper_bound}")
        else:
            args.upper_bound = exponential_search_upper_bound(args.device, start=args.lower_bound)

    # Binary search for exact maximum
    max_batch_size = binary_search_max_batch_size(
        device=args.device,
        lower=args.lower_bound,
        upper=args.upper_bound
    )

    # Print final result
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    print(f"🎯 Maximum batch size: {max_batch_size}")

    if args.device == "cuda":
        # Test the max one more time to get accurate memory
        print(f"\nVerifying maximum batch size...")
        success, memory = test_batch_size(max_batch_size, args.device, num_steps=10)
        if success:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n✅ Verified!")
            print(f"  Maximum batch size: {max_batch_size}")
            print(f"  Peak memory usage: {memory:.2f} GB")
            print(f"  Total GPU memory: {total_memory:.2f} GB")
            print(f"  Memory utilization: {memory/total_memory*100:.1f}%")

    print("="*80)
    print(f"\n💡 Use this for your batch size sweep:")
    print(f"   python experiments/batch_size_sweep.py --batch_sizes 1 32 64 128 256 512 {max_batch_size}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
