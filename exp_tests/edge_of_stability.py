#!/usr/bin/env python3
"""
Edge of Stability Experiment

This script helps you find the "edge of stability" - the learning rate at which
training begins to diverge. It tests increasingly high learning rates until
divergence is detected.

Usage:
    # Test high learning rates to find divergence
    python experiments/edge_of_stability.py --device cuda
    
    # Test specific range
    python experiments/edge_of_stability.py --device cuda --start_lr 0.003 --max_lr 0.03
    
    # Quick test (fewer iterations)
    python experiments/edge_of_stability.py --device cuda --quick
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from cs336_basics.config import TrainingConfig, ModelConfig, OptimizerConfig, SchedulerConfig, DataConfig
from cs336_basics.training import train


def get_base_config(learning_rate: float, max_iters: int = 40000):
    """Get base configuration for edge of stability experiments."""
    batch_size = 32
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
        learning_rate=learning_rate,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        grad_clip_norm=1.0,
    )
    
    scheduler_config = SchedulerConfig(
        warmup_iters=int(max_iters * 0.05),  # 5% warmup
        max_iters=max_iters,
        min_lr_ratio=0.1,
    )
    
    data_config = DataConfig(
        train_data_path="data/tinystories_train_tokens.npy",
        val_data_path="data/tinystories_valid_tokens.npy",
        batch_size=batch_size,
        context_length=context_length,
    )
    
    config = TrainingConfig(
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        data=data_config,
        eval_interval=500,
        eval_iters=100,
        log_interval=100,
        checkpoint_interval=10000,
        checkpoint_dir="checkpoints/edge_of_stability",
        use_wandb=True,
        wandb_project="cs336-lr-sweep",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    return config


def run_experiment(learning_rate: float, device: str, use_wandb: bool, max_iters: int = 40000):
    """
    Run a single experiment with the given learning rate.
    
    Returns:
        (success, diverged, final_loss): Whether training completed, whether it diverged, and final loss
    """
    run_name = f"edge_lr_{learning_rate:.0e}".replace(".", "_").replace("-", "_")
    
    config = get_base_config(learning_rate, max_iters)
    config.device = device
    config.use_wandb = use_wandb
    config.wandb_run_name = run_name
    config.checkpoint_dir = f"{config.checkpoint_dir}/{run_name}"
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Testing LR = {learning_rate:.6f}")
    print(f"{'='*80}\n")
    
    try:
        train(config)
        print(f"\n✓ Completed: LR={learning_rate:.6f}\n")
        return True, False, None
    except Exception as e:
        error_msg = str(e).lower()
        if "nan" in error_msg or "inf" in error_msg or "diverge" in error_msg:
            print(f"\n✗ Diverged: LR={learning_rate:.6f}")
            print(f"   Error: {e}\n")
            return False, True, None
        else:
            print(f"\n✗ Failed: LR={learning_rate:.6f}")
            print(f"   Error: {e}\n")
            return False, False, None


def find_edge_of_stability(start_lr: float, max_lr: float, device: str, use_wandb: bool, 
                          num_steps: int = 10, quick: bool = False):
    """
    Find the edge of stability by testing increasingly high learning rates.
    
    Args:
        start_lr: Starting learning rate
        max_lr: Maximum learning rate to test
        device: Device to use
        use_wandb: Whether to use W&B
        num_steps: Number of learning rates to test
        quick: If True, use fewer iterations for faster testing
    """
    max_iters = 5000 if quick else 40000
    
    print("\n" + "="*80)
    print("EDGE OF STABILITY SEARCH")
    print("="*80)
    print(f"Testing learning rates from {start_lr:.6f} to {max_lr:.6f}")
    print(f"Number of steps: {num_steps}")
    print(f"Max iterations per run: {max_iters}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # Generate learning rates on log scale
    learning_rates = np.logspace(np.log10(start_lr), np.log10(max_lr), num_steps)
    
    results = []
    edge_found = False
    edge_lr = None
    
    for i, lr in enumerate(learning_rates):
        print(f"\n[{i+1}/{num_steps}] Testing LR = {lr:.6f}")
        
        success, diverged, final_loss = run_experiment(lr, device, use_wandb, max_iters)
        
        results.append({
            'lr': lr,
            'success': success,
            'diverged': diverged,
            'final_loss': final_loss
        })
        
        if diverged and not edge_found:
            edge_found = True
            edge_lr = lr
            print(f"\n🎯 EDGE OF STABILITY FOUND!")
            print(f"   Divergence detected at LR = {lr:.6f}")
            
            # Find the highest stable LR
            stable_lrs = [r['lr'] for r in results if r['success']]
            if stable_lrs:
                highest_stable = max(stable_lrs)
                print(f"   Highest stable LR = {highest_stable:.6f}")
                print(f"   Edge is between {highest_stable:.6f} and {lr:.6f}")
            
            # Ask if user wants to continue
            if i < len(learning_rates) - 1:
                print(f"\n   {len(learning_rates) - i - 1} more LRs to test.")
                print(f"   You can stop now or continue to test higher LRs.")
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Learning Rate':<15} {'Status':<15} {'Notes':<30}")
    print("-"*80)
    
    for result in results:
        lr = result['lr']
        if result['diverged']:
            status = "✗ Diverged"
            notes = "Training became unstable"
        elif result['success']:
            status = "✓ Converged"
            notes = "Training completed successfully"
        else:
            status = "⚠ Failed"
            notes = "Error occurred"
        
        print(f"{lr:<15.6f} {status:<15} {notes:<30}")
    
    print("="*80)
    
    # Analysis
    stable_lrs = [r['lr'] for r in results if r['success']]
    diverged_lrs = [r['lr'] for r in results if r['diverged']]
    
    if stable_lrs and diverged_lrs:
        highest_stable = max(stable_lrs)
        lowest_diverged = min(diverged_lrs)
        
        print(f"\n📊 EDGE OF STABILITY ANALYSIS")
        print(f"   Highest stable LR:   {highest_stable:.6f}")
        print(f"   Lowest divergent LR: {lowest_diverged:.6f}")
        print(f"   Edge range:          [{highest_stable:.6f}, {lowest_diverged:.6f}]")
        print(f"   Ratio:               {lowest_diverged/highest_stable:.2f}x")
        
        print(f"\n💡 RECOMMENDATIONS")
        print(f"   1. Your best LR is likely in the range [{highest_stable*0.3:.6f}, {highest_stable:.6f}]")
        print(f"   2. For fine-grained search, test LRs between {highest_stable:.6f} and {lowest_diverged:.6f}")
        print(f"   3. Compare your best LR from grid search to {highest_stable:.6f}")
        
    elif stable_lrs and not diverged_lrs:
        highest_stable = max(stable_lrs)
        print(f"\n⚠ NO DIVERGENCE FOUND")
        print(f"   All tested learning rates converged!")
        print(f"   Highest tested: {highest_stable:.6f}")
        print(f"   Recommendation: Test higher learning rates (e.g., up to {highest_stable*10:.6f})")
        
    elif diverged_lrs and not stable_lrs:
        lowest_diverged = min(diverged_lrs)
        print(f"\n⚠ ALL RUNS DIVERGED")
        print(f"   Lowest tested: {lowest_diverged:.6f}")
        print(f"   Recommendation: Test lower learning rates (e.g., down to {lowest_diverged*0.1:.6f})")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Edge of Stability Experiment")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--start_lr",
        type=float,
        default=0.003,
        help="Starting learning rate (default: 0.003)"
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.03,
        help="Maximum learning rate (default: 0.03)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of learning rates to test (default: 10)"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with only 5000 iterations per run"
    )
    
    args = parser.parse_args()
    
    find_edge_of_stability(
        start_lr=args.start_lr,
        max_lr=args.max_lr,
        device=args.device,
        use_wandb=not args.no_wandb,
        num_steps=args.num_steps,
        quick=args.quick
    )


if __name__ == "__main__":
    main()

