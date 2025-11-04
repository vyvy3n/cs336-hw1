#!/usr/bin/env python3
"""
Quick test script for batch size experiments.

This runs a very short training run with different batch sizes to verify:
1. The configuration is correct
2. GPU memory limits
3. Training runs without errors

Usage:
    python experiments/test_batch_size.py --device cuda
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from batch_size_sweep import get_base_config, run_single_experiment


def quick_test(device: str, batch_sizes: list[int] = None):
    """
    Run quick tests with different batch sizes.
    
    Args:
        device: Device to use
        batch_sizes: List of batch sizes to test
    """
    if batch_sizes is None:
        batch_sizes = [1, 32, 64, 128, 256]
    
    print("\n" + "="*80)
    print("QUICK BATCH SIZE TEST")
    print("="*80)
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Device: {device}")
    print("Note: Running only 100 iterations per test")
    print("="*80 + "\n")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*80}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'='*80}\n")
        
        # Get config with very short training
        config = get_base_config(batch_size=batch_size, learning_rate=3e-4)
        config.device = device
        config.use_wandb = False  # Disable W&B for quick tests
        config.scheduler.max_iters = 100  # Only 100 iterations
        config.eval_interval = 50
        config.log_interval = 10
        config.checkpoint_interval = 100
        config.checkpoint_dir = f"test_checkpoints/batch_size_test/bs_{batch_size}"

        # Create checkpoint directory
        import os
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Calculate memory requirements
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1e9
        
        try:
            from cs336_basics.training import train
            train(config)
            
            if device == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                print(f"\n✓ Success!")
                print(f"  Initial memory: {initial_memory:.2f} GB")
                print(f"  Peak memory: {peak_memory:.2f} GB")
                print(f"  Memory used: {peak_memory - initial_memory:.2f} GB")
                results[batch_size] = {
                    'success': True,
                    'memory_gb': peak_memory - initial_memory
                }
            else:
                print(f"\n✓ Success!")
                results[batch_size] = {'success': True}
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n✗ Out of Memory!")
                results[batch_size] = {'success': False, 'error': 'OOM'}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Stop testing larger batch sizes
                print(f"\nStopping tests - reached memory limit at batch size {batch_size}")
                break
            else:
                print(f"\n✗ Error: {e}")
                results[batch_size] = {'success': False, 'error': str(e)}
        except Exception as e:
            print(f"\n✗ Error: {e}")
            results[batch_size] = {'success': False, 'error': str(e)}
    
    # Print summary
    print("\n" + "="*80)
    print("QUICK TEST SUMMARY")
    print("="*80)
    print(f"{'Batch Size':<15} {'Status':<15} {'Memory (GB)':<15}")
    print("-" * 80)
    
    for batch_size, result in results.items():
        status = "✓ Success" if result['success'] else "✗ Failed"
        memory = f"{result.get('memory_gb', 0):.2f}" if result['success'] and device == "cuda" else "N/A"
        print(f"{batch_size:<15} {status:<15} {memory:<15}")
    
    print("="*80)
    
    # Estimate maximum batch size
    if device == "cuda" and any(r['success'] for r in results.values()):
        successful_results = [(bs, r['memory_gb']) for bs, r in results.items() 
                             if r['success'] and 'memory_gb' in r]
        if successful_results:
            # Get the largest successful batch size
            max_bs, max_memory = max(successful_results, key=lambda x: x[0])
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            available_memory = total_memory * 0.9  # Use 90% to be safe
            
            print(f"\n📊 Memory Analysis:")
            print(f"  Total GPU memory: {total_memory:.2f} GB")
            print(f"  Safe limit (90%): {available_memory:.2f} GB")
            print(f"  Largest tested: batch_size={max_bs}, memory={max_memory:.2f} GB")
            
            # Estimate maximum batch size (memory scales roughly linearly with batch size)
            if max_memory > 0:
                estimated_max_bs = int(max_bs * (available_memory / max_memory))
                print(f"  Estimated max batch size: ~{estimated_max_bs}")
                print(f"\n💡 Recommendation: Test batch sizes up to {estimated_max_bs}")
    
    print("\n" + "="*80 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Quick Batch Size Test")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for testing"
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Specific batch sizes to test (default: 1 32 64 128 256)"
    )
    
    args = parser.parse_args()
    
    # Check GPU availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"\n📊 GPU Information:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Run tests
    quick_test(device=args.device, batch_sizes=args.batch_sizes)


if __name__ == "__main__":
    main()

