#!/usr/bin/env python3
"""
Check Divergence in Existing Runs

This script analyzes your existing W&B runs to check if any have diverged,
and helps you determine what additional experiments you need to run.

Usage:
    python experiments/check_divergence.py
    python experiments/check_divergence.py --project cs336-lr-sweep
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ W&B not available. Install with: pip install wandb")


def check_divergence(project_name: str = "cs336-lr-sweep", entity: str = "tianweiyue-org"):
    """Check existing runs for divergence."""
    
    if not WANDB_AVAILABLE:
        print("Cannot check W&B runs without wandb package.")
        return
    
    print("\n" + "="*80)
    print("CHECKING EXISTING RUNS FOR DIVERGENCE")
    print("="*80)
    print(f"Project: {entity}/{project_name}")
    print("="*80 + "\n")
    
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project_name}")
        
        if not runs:
            print("No runs found in this project.")
            return
        
        results = []
        
        for run in runs:
            # Get learning rate from config
            lr = run.config.get('optimizer', {}).get('learning_rate')
            if lr is None:
                continue
            
            # Get final metrics
            summary = run.summary
            final_val_loss = summary.get('val_loss')
            final_train_loss = summary.get('train_loss')
            
            # Get history to check for divergence
            try:
                history = run.history(keys=['iteration', 'val_loss', 'train_loss'], samples=10000)
                
                if len(history) == 0:
                    status = "⚠ No data"
                    diverged = False
                else:
                    # Check for divergence indicators
                    max_val_loss = history['val_loss'].max()
                    min_val_loss = history['val_loss'].min()
                    
                    # Divergence criteria
                    diverged = False
                    reason = ""
                    
                    if final_val_loss is None or (isinstance(final_val_loss, float) and 
                                                  (final_val_loss != final_val_loss)):  # NaN check
                        diverged = True
                        reason = "NaN loss"
                        status = "✗ Diverged (NaN)"
                    elif max_val_loss > 15:
                        diverged = True
                        reason = f"Loss exploded to {max_val_loss:.2f}"
                        status = f"✗ Diverged (loss>{max_val_loss:.1f})"
                    elif run.state == "crashed" or run.state == "failed":
                        diverged = True
                        reason = f"Run {run.state}"
                        status = f"✗ Diverged ({run.state})"
                    elif final_val_loss and final_val_loss > 10:
                        diverged = True
                        reason = f"High final loss: {final_val_loss:.2f}"
                        status = f"✗ Diverged (loss={final_val_loss:.2f})"
                    elif run.state == "finished":
                        status = f"✓ Converged (loss={final_val_loss:.2f})"
                    elif run.state == "running":
                        status = "⏳ Running"
                    else:
                        status = f"⚠ {run.state}"
                    
                    results.append({
                        'lr': lr,
                        'status': status,
                        'diverged': diverged,
                        'reason': reason,
                        'final_val_loss': final_val_loss,
                        'max_val_loss': max_val_loss,
                        'run_name': run.name,
                        'url': run.url
                    })
            except Exception as e:
                print(f"Error processing run {run.name}: {e}")
                continue
        
        # Sort by learning rate
        results.sort(key=lambda x: x['lr'])
        
        # Print results
        print(f"{'Learning Rate':<15} {'Status':<30} {'Notes':<40}")
        print("-"*85)
        
        converged_runs = []
        diverged_runs = []
        
        for result in results:
            lr = result['lr']
            status = result['status']
            notes = result['reason'] if result['reason'] else f"Final loss: {result['final_val_loss']:.3f}" if result['final_val_loss'] else ""
            
            print(f"{lr:<15.6f} {status:<30} {notes:<40}")
            
            if result['diverged']:
                diverged_runs.append(result)
            elif '✓' in status:
                converged_runs.append(result)
        
        print("="*85)
        
        # Analysis
        print(f"\n📊 SUMMARY")
        print(f"   Total runs: {len(results)}")
        print(f"   Converged: {len(converged_runs)}")
        print(f"   Diverged: {len(diverged_runs)}")
        
        if converged_runs:
            best_run = min(converged_runs, key=lambda x: x['final_val_loss'] if x['final_val_loss'] else float('inf'))
            highest_stable_lr = max(converged_runs, key=lambda x: x['lr'])
            
            print(f"\n✅ CONVERGED RUNS")
            print(f"   Best LR: {best_run['lr']:.6f} (loss={best_run['final_val_loss']:.3f})")
            print(f"   Highest stable LR: {highest_stable_lr['lr']:.6f} (loss={highest_stable_lr['final_val_loss']:.3f})")
        
        if diverged_runs:
            lowest_diverged_lr = min(diverged_runs, key=lambda x: x['lr'])
            
            print(f"\n❌ DIVERGED RUNS")
            print(f"   Lowest divergent LR: {lowest_diverged_lr['lr']:.6f}")
            print(f"   Reason: {lowest_diverged_lr['reason']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        
        if diverged_runs and converged_runs:
            highest_stable = max(converged_runs, key=lambda x: x['lr'])['lr']
            lowest_diverged = min(diverged_runs, key=lambda x: x['lr'])['lr']
            
            print(f"   ✅ You have both converged and diverged runs!")
            print(f"   📍 Edge of stability is between {highest_stable:.6f} and {lowest_diverged:.6f}")
            print(f"\n   Next steps:")
            print(f"   1. Create learning curves plot showing all runs")
            print(f"   2. Write analysis comparing best LR ({best_run['lr']:.6f}) to edge ({highest_stable:.6f})")
            print(f"   3. Optional: Fine-grained search between {highest_stable:.6f} and {lowest_diverged:.6f}")
            
        elif converged_runs and not diverged_runs:
            highest_stable = max(converged_runs, key=lambda x: x['lr'])['lr']
            
            print(f"   ⚠ All runs converged - no divergence found yet!")
            print(f"   📍 Highest tested LR: {highest_stable:.6f}")
            print(f"\n   Next steps:")
            print(f"   1. Test higher learning rates to find divergence")
            print(f"   2. Suggested range: {highest_stable*2:.6f} to {highest_stable*10:.6f}")
            print(f"\n   Run this command:")
            print(f"   python experiments/edge_of_stability.py --device cuda --start_lr {highest_stable*1.5:.6f} --max_lr {highest_stable*10:.6f}")
            
        elif diverged_runs and not converged_runs:
            lowest_diverged = min(diverged_runs, key=lambda x: x['lr'])['lr']
            
            print(f"   ⚠ All runs diverged!")
            print(f"   📍 Lowest tested LR: {lowest_diverged:.6f}")
            print(f"\n   Next steps:")
            print(f"   1. Test lower learning rates")
            print(f"   2. Suggested range: {lowest_diverged*0.1:.6f} to {lowest_diverged:.6f}")
            
        else:
            print(f"   ⚠ No completed runs found")
            print(f"   Run the grid sweep first:")
            print(f"   python experiments/learning_rate_sweep.py --sweep_type grid --device cuda")
        
        print("="*85 + "\n")
        
    except Exception as e:
        print(f"Error accessing W&B: {e}")
        print("\nMake sure you're logged in: wandb login")


def main():
    parser = argparse.ArgumentParser(description="Check for divergence in existing runs")
    parser.add_argument(
        "--project",
        type=str,
        default="cs336-lr-sweep",
        help="W&B project name"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="tianweiyue-org",
        help="W&B entity (username or team)"
    )
    
    args = parser.parse_args()
    
    check_divergence(args.project, args.entity)


if __name__ == "__main__":
    main()

