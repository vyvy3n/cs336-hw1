#!/usr/bin/env python3
"""Find maximum batch size using binary search.

Usage:
    python scripts/find_max_batch_size.py --device cuda
    python scripts/find_max_batch_size.py --device cuda --upper_bound 4096
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from cs336_basics.config import TrainingConfig
from cs336_basics.models import TransformerLM


def test_batch_size(batch_size: int, device: str, num_steps: int = 10) -> bool:
    """Test if a batch size fits in GPU memory."""
    try:
        # Create minimal config for testing
        config = TrainingConfig.from_dataset(
            "tinystories",
            batch_size=batch_size,
            device=device,
            max_iters=num_steps,
            use_wandb=False,
        )
        
        # Create model and optimizer
        model = TransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            attn_pdrop=0.0,
            residual_pdrop=0.0,
            use_rope=config.use_rope,
            theta=config.theta,
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        
        # Run a few training steps
        for _ in range(num_steps):
            # Generate random batch
            x = torch.randint(0, config.vocab_size, (batch_size, config.context_length), device=device)
            y = torch.randint(0, config.vocab_size, (batch_size, config.context_length), device=device)
            
            # Forward pass
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, config.vocab_size),
                y.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Clean up
        del model, optimizer, x, y, logits, loss
        torch.cuda.empty_cache()
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        raise


def binary_search_max_batch_size(device: str, lower: int = 1, upper: int = 2048, num_steps: int = 10) -> int:
    """Use binary search to find maximum batch size."""
    print(f"\n{'='*80}")
    print(f"Finding Maximum Batch Size")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Search range: [{lower}, {upper}]")
    print(f"Test steps: {num_steps}")
    print(f"{'='*80}\n")
    
    max_working = 0
    
    while lower <= upper:
        mid = (lower + upper) // 2
        print(f"Testing batch_size={mid}...", end=" ", flush=True)
        
        if test_batch_size(mid, device, num_steps):
            print("✓ Success")
            max_working = mid
            lower = mid + 1
        else:
            print("✗ OOM")
            upper = mid - 1
    
    print(f"\n{'='*80}")
    print(f"🎯 Maximum batch size: {max_working}")
    print(f"{'='*80}\n")
    
    return max_working


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find maximum batch size")
    parser.add_argument("--device", default="cuda", help="Device to test on")
    parser.add_argument("--lower", type=int, default=1, help="Lower bound for search")
    parser.add_argument("--upper", type=int, default=2048, help="Upper bound for search")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of test steps")
    args = parser.parse_args()
    
    binary_search_max_batch_size(args.device, args.lower, args.upper, args.num_steps)

