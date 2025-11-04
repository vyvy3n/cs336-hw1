#!/usr/bin/env python3
"""
Analyze Learning Rate Experiment Results

This script analyzes the results from learning rate sweep experiments.
It can work with W&B data or local checkpoint files.

Usage:
    python experiments/analyze_results.py --wandb_project cs336-lr-sweep
    python experiments/analyze_results.py --checkpoint_dir checkpoints/lr_sweep
"""

import argparse
import os
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_wandb_results(project_name: str, entity: str = None):
    """
    Analyze results from Weights & Biases.
    
    Args:
        project_name: W&B project name
        entity: W&B entity (username or team)
    """
    try:
        import wandb
    except ImportError:
        print("Error: wandb not installed. Install with: pip install wandb")
        return
    
    api = wandb.Api()
    
    # Get all runs from the project
    if entity:
        runs = api.runs(f"{entity}/{project_name}")
    else:
        runs = api.runs(project_name)
    
    print(f"\nFound {len(runs)} runs in project '{project_name}'")
    print("="*80)
    
    results = []
    
    for run in runs:
        # Extract learning rate from run name or config
        lr = None
        if 'learning_rate' in run.config:
            lr = run.config['learning_rate']
        elif 'lr_' in run.name:
            # Parse from name like "lr_3e_04"
            lr_str = run.name.split('lr_')[1].split('_')[0:2]
            try:
                lr = float(f"{lr_str[0]}e-{lr_str[1]}")
            except:
                pass
        
        # Get final validation loss
        history = run.history(keys=['val_loss', 'train_loss', 'iteration'])
        
        if len(history) == 0:
            print(f"⚠ Run '{run.name}' has no data")
            continue
        
        final_val_loss = history['val_loss'].dropna().iloc[-1] if 'val_loss' in history else None
        final_train_loss = history['train_loss'].dropna().iloc[-1] if 'train_loss' in history else None
        final_iter = history['iteration'].iloc[-1] if 'iteration' in history else None
        
        # Check if run diverged (loss > 100 or NaN)
        diverged = False
        if final_val_loss is None or final_val_loss > 100:
            diverged = True
        
        results.append({
            'name': run.name,
            'learning_rate': lr,
            'final_val_loss': final_val_loss,
            'final_train_loss': final_train_loss,
            'final_iteration': final_iter,
            'diverged': diverged,
            'state': run.state,
            'url': run.url,
        })
    
    # Sort by learning rate
    results.sort(key=lambda x: x['learning_rate'] if x['learning_rate'] else 0)
    
    # Print summary
    print("\nLearning Rate Sweep Results:")
    print("="*80)
    print(f"{'Learning Rate':<15} {'Val Loss':<12} {'Train Loss':<12} {'Iters':<8} {'Status':<12}")
    print("-"*80)
    
    best_result = None
    best_val_loss = float('inf')
    
    for result in results:
        lr = result['learning_rate']
        val_loss = result['final_val_loss']
        train_loss = result['final_train_loss']
        iters = result['final_iteration']
        
        if result['diverged']:
            status = "❌ Diverged"
        elif result['state'] == 'finished':
            status = "✓ Finished"
        elif result['state'] == 'running':
            status = "⏳ Running"
        else:
            status = f"⚠ {result['state']}"
        
        lr_str = f"{lr:.0e}" if lr else "Unknown"
        val_str = f"{val_loss:.4f}" if val_loss and not result['diverged'] else "N/A"
        train_str = f"{train_loss:.4f}" if train_loss and not result['diverged'] else "N/A"
        iter_str = f"{int(iters)}" if iters else "N/A"
        
        print(f"{lr_str:<15} {val_str:<12} {train_str:<12} {iter_str:<8} {status:<12}")
        
        # Track best result
        if val_loss and not result['diverged'] and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_result = result
    
    print("="*80)
    
    # Print best result
    if best_result:
        print(f"\n🎯 Best Result:")
        print(f"   Learning Rate: {best_result['learning_rate']:.0e}")
        print(f"   Validation Loss: {best_result['final_val_loss']:.4f}")
        print(f"   Training Loss: {best_result['final_train_loss']:.4f}")
        print(f"   Run: {best_result['name']}")
        print(f"   URL: {best_result['url']}")
        
        if best_result['final_val_loss'] <= 1.45:
            print(f"\n   ✓ Meets target validation loss (≤ 1.45)!")
        else:
            print(f"\n   ⚠ Does not meet target validation loss (≤ 1.45)")
            print(f"   Gap: {best_result['final_val_loss'] - 1.45:.4f}")
    
    # Find edge of stability
    converged_lrs = [r['learning_rate'] for r in results if not r['diverged'] and r['learning_rate']]
    diverged_lrs = [r['learning_rate'] for r in results if r['diverged'] and r['learning_rate']]
    
    if converged_lrs and diverged_lrs:
        max_stable_lr = max(converged_lrs)
        min_diverged_lr = min(diverged_lrs)
        
        print(f"\n📊 Stability Analysis:")
        print(f"   Highest stable LR: {max_stable_lr:.0e}")
        print(f"   Lowest diverged LR: {min_diverged_lr:.0e}")
        print(f"   Edge of stability: between {max_stable_lr:.0e} and {min_diverged_lr:.0e}")
    
    print("\n" + "="*80)


def analyze_checkpoint_results(checkpoint_dir: str):
    """
    Analyze results from local checkpoint files.
    
    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories
    """
    import torch
    
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    print(f"\nAnalyzing checkpoints in: {checkpoint_dir}")
    print("="*80)
    
    results = []
    
    # Find all checkpoint directories
    for run_dir in sorted(checkpoint_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Find the latest checkpoint
        checkpoints = list(run_dir.glob("checkpoint_iter_*.pt"))
        if not checkpoints:
            print(f"⚠ No checkpoints found in {run_dir.name}")
            continue
        
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        
        # Load checkpoint
        try:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            
            # Extract learning rate from run name
            lr = None
            if 'lr_' in run_dir.name:
                lr_str = run_dir.name.split('lr_')[1].split('_')[0:2]
                try:
                    lr = float(f"{lr_str[0]}e-{lr_str[1]}")
                except:
                    pass
            
            results.append({
                'name': run_dir.name,
                'learning_rate': lr,
                'iteration': checkpoint.get('iteration', None),
                'train_loss': checkpoint.get('train_loss', None),
                'val_loss': checkpoint.get('val_loss', None),
                'checkpoint_path': str(latest_checkpoint),
            })
            
        except Exception as e:
            print(f"⚠ Error loading {latest_checkpoint}: {e}")
    
    # Sort by learning rate
    results.sort(key=lambda x: x['learning_rate'] if x['learning_rate'] else 0)
    
    # Print summary
    print("\nCheckpoint Analysis:")
    print("="*80)
    print(f"{'Learning Rate':<15} {'Val Loss':<12} {'Train Loss':<12} {'Iteration':<10}")
    print("-"*80)
    
    for result in results:
        lr_str = f"{result['learning_rate']:.0e}" if result['learning_rate'] else "Unknown"
        val_str = f"{result['val_loss']:.4f}" if result['val_loss'] else "N/A"
        train_str = f"{result['train_loss']:.4f}" if result['train_loss'] else "N/A"
        iter_str = f"{result['iteration']}" if result['iteration'] else "N/A"
        
        print(f"{lr_str:<15} {val_str:<12} {train_str:<12} {iter_str:<10}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze Learning Rate Experiment Results")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name to analyze"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (username or team)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing checkpoint subdirectories"
    )
    
    args = parser.parse_args()
    
    if args.wandb_project:
        analyze_wandb_results(args.wandb_project, args.wandb_entity)
    elif args.checkpoint_dir:
        analyze_checkpoint_results(args.checkpoint_dir)
    else:
        print("Error: Must specify either --wandb_project or --checkpoint_dir")
        print("\nExamples:")
        print("  python experiments/analyze_results.py --wandb_project cs336-lr-sweep")
        print("  python experiments/analyze_results.py --checkpoint_dir checkpoints/lr_sweep")


if __name__ == "__main__":
    main()

