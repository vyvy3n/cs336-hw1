import torch
import numpy as np


def data_loading(x, batch_size, context_length, device):
    if len(x) < context_length + 1:
        raise ValueError(f"Data length {len(x)} is too short for context_length {context_length}")

    max_start_idx = len(x) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    input_sequences = []
    targets = []
    
    for start_idx in start_indices:
        input_seq = x[start_idx:start_idx + context_length]
        target_seq = x[start_idx + 1:start_idx + context_length + 1]
        
        input_sequences.append(input_seq)
        targets.append(target_seq)
    
    input_sequences = np.array(input_sequences)
    targets = np.array(targets)
    
    input_tensor = torch.tensor(input_sequences, dtype=torch.long, device=device)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    
    return input_tensor, target_tensor


if __name__ == "__main__":
    print("Testing data_loading function...")
    test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    batch_size = 3
    context_length = 5
    device = 'cpu'
    
    print(f"Test data: {test_data}")
    print(f"Batch size: {batch_size}")
    print(f"Context length: {context_length}")
    
    # Test the function
    inputs, targets = data_loading(test_data, batch_size, context_length, device)
    
    print(f"\nInput tensor shape: {inputs.shape}")
    print(f"Target tensor shape: {targets.shape}")
    print(f"\nInput sequences:")
    print(inputs)
    print(f"\nTarget sequences:")
    print(targets)
    
    # Verify that targets are shifted by 1
    print(f"\nVerifying target shift:")
    for i in range(batch_size):
        print(f"Batch {i}: input[0] = {inputs[i][0].item()}, target[0] = {targets[i][0].item()}")
        print(f"Batch {i}: input[-1] = {inputs[i][-1].item()}, target[-1] = {targets[i][-1].item()}")
    
    # Test error case
    print(f"\nTesting error case with short data...")
    try:
        short_data = np.array([1, 2, 3])
        data_loading(short_data, batch_size, context_length, device)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\ndata_loading test completed successfully!")