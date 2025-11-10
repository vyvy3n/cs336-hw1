#!/usr/bin/env python3
"""
Minimal script to inspect checkpoint format and model statistics.
Usage: python inspect_checkpoint.py
"""

import torch

# Change this path to your checkpoint file
CHECKPOINT_PATH = "checkpoints/owt/checkpoint_latest.pt"

def count_parameters(state_dict):
    """Count total and trainable parameters from state dict."""
    total_params = 0
    trainable_params = 0
    for key, tensor in state_dict.items():
        num_params = tensor.numel()
        total_params += num_params
        trainable_params += num_params  # All parameters in state_dict are trainable
    return total_params, trainable_params

def format_size(size_bytes):
    """Format size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    print("=" * 80)
    print("CHECKPOINT INSPECTION")
    print("=" * 80)
    print(f"\nLoading checkpoint from: {CHECKPOINT_PATH}\n")
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    # Print all top-level keys
    print("Top-level checkpoint keys:")
    print("-" * 80)
    for key in checkpoint.keys():
        print(f"  • {key}")
    print()
    
    # Check for model state dict
    state_dict = None
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("✓ Found 'model' key (state dict)")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✓ Found 'model_state_dict' key (state dict)")
    else:
        print("⚠ Warning: No 'model' or 'model_state_dict' found")
        # Try to use the checkpoint itself as state dict
        if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
            print("  Using checkpoint as state dict")
    
    # Print model statistics
    if state_dict:
        print("\n" + "=" * 80)
        print("MODEL STATISTICS")
        print("=" * 80)
        
        total_params, trainable_params = count_parameters(state_dict)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Estimate size
        total_size = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
        print(f"Estimated size: {format_size(total_size)}")
        
        # Print layer information
        print("\n" + "-" * 80)
        print("Layer breakdown:")
        print("-" * 80)
        
        # Group by layer type
        layer_groups = {}
        for key in sorted(state_dict.keys()):
            parts = key.split('.')
            if len(parts) > 0:
                layer_type = parts[0]
                if layer_type not in layer_groups:
                    layer_groups[layer_type] = []
                layer_groups[layer_type].append(key)
        
        for layer_type, keys in layer_groups.items():
            print(f"\n{layer_type}:")
            for key in keys:  # Show first 5 keys
                tensor = state_dict[key]
                print(f"  {key:50s} shape={str(tensor.shape):20s} dtype={str(tensor.dtype):10s}")
            # for key in keys[:5]:  # Show first 5 keys
            #     tensor = state_dict[key]
            #     print(f"  {key:50s} shape={str(tensor.shape):20s} dtype={str(tensor.dtype):10s}")
            # if len(keys) > 5:
            #     print(f"  ... and {len(keys) - 5} more")
    
    # Print checkpoint metadata
    print("\n" + "=" * 80)
    print("CHECKPOINT METADATA")
    print("=" * 80)
    
    if 'iteration' in checkpoint:
        print(f"\nIteration: {checkpoint['iteration']}")
    
    if 'config' in checkpoint:
        print("\nModel config:")
        config = checkpoint['config']
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")
    else:
        print("\n⚠ No 'config' found in checkpoint")
    
    if 'training_state' in checkpoint:
        print("\nTraining state:")
        training_state = checkpoint['training_state']
        for key, value in sorted(training_state.items()):
            print(f"  {key}: {value}")
    
    if 'optimizer' in checkpoint:
        print("\nOptimizer state:")
        optimizer_state = checkpoint['optimizer']
        if isinstance(optimizer_state, dict):
            print(f"  Keys: {list(optimizer_state.keys())}")
            if 'state' in optimizer_state:
                print(f"  Number of parameter groups: {len(optimizer_state.get('param_groups', []))}")
                print(f"  Number of parameters tracked: {len(optimizer_state['state'])}")
    
    print("\n" + "=" * 80)
    print("Sample weight shapes (first 10 layers):")
    print("=" * 80)
    if state_dict:
        for i, (key, tensor) in enumerate(list(state_dict.items())[:10]):
            print(f"{key:50s} {str(tensor.shape):20s} {str(tensor.dtype):10s}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
