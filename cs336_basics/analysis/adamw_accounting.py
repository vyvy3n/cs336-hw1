"""
Resource accounting for training with AdamW optimizer.

This module provides functions to calculate memory usage and FLOPs
for training Transformer models with the AdamW optimizer.
"""


def calculate_parameters_memory(
    vocab_size: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
) -> int:
    """
    Calculate memory usage for model parameters in bytes (float32).

    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum context length (not used in parameter calculation)
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension

    Returns:
        Memory usage in bytes
    """
    # Token embedding: vocab_size × d_model
    token_embedding = vocab_size * d_model

    per_block = (
        d_model  # RMSNorm pre-attention
        + 3 * d_model * d_model  # Q, K, V projections
        + d_model * d_model  # output projection
        + d_model  # RMSNorm pre-ffn
        + d_model * d_ff  # FFN first linear
        + d_ff * d_model  # FFN second linear
    )

    # Final RMSNorm: d_model
    final_norm = d_model

    # Output embedding (LM head): d_model × vocab_size
    output_embedding = d_model * vocab_size

    total_params = token_embedding + num_layers * per_block + final_norm + output_embedding

    # Convert to bytes (float32 = 4 bytes)
    return total_params * 4


def calculate_activations_memory(
    batch_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int,
) -> int:
    """
    Calculate memory usage for activations in bytes (float32).

    Args:
        batch_size: Batch size
        context_length: Maximum context length
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        vocab_size: Size of the vocabulary

    Returns:
        Memory usage in bytes
    """
    per_block = (
        2 * batch_size * context_length * d_model  # RMSNorm (2 instances)
        + 3 * batch_size * context_length * d_model  # Q, K, V projections
        + batch_size * num_heads * context_length * context_length  # Q^T K
        + batch_size * num_heads * context_length * context_length  # Softmax
        + batch_size * context_length * d_model  # Weighted sum
        + batch_size * context_length * d_model  # Output projection
        + batch_size * context_length * d_ff  # FFN W1 multiply
        + batch_size * context_length * d_ff  # SiLU activation
        + batch_size * context_length * d_model  # FFN W2 multiply
    )

    # Final RMSNorm: batch_size × context_length × d_model
    final_norm = batch_size * context_length * d_model

    # Output embedding: batch_size × context_length × vocab_size
    output_embedding = batch_size * context_length * vocab_size

    # Cross-entropy logits: batch_size × context_length × vocab_size
    cross_entropy = batch_size * context_length * vocab_size

    total_activations = num_layers * per_block + final_norm + output_embedding + cross_entropy

    return total_activations * 4


def calculate_gradients_memory(
    vocab_size: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
) -> int:
    """
    Calculate memory usage for gradients in bytes (float32).

    This is the same as parameters memory since we need gradients for each parameter.

    Args:
        vocab_size: Size of the vocabulary
        num_layers: Number of transformer layers
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension

    Returns:
        Memory usage in bytes
    """
    return calculate_parameters_memory(vocab_size, num_layers, d_model, d_ff)


def calculate_optimizer_state_memory(
    vocab_size: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
) -> int:
    """
    Calculate memory usage for AdamW optimizer state in bytes (float32).

    AdamW keeps first and second moment estimates (m and v) for each parameter.

    Args:
        vocab_size: Size of the vocabulary
        num_layers: Number of transformer layers
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension

    Returns:
        Memory usage in bytes
    """
    # AdamW needs 2 states per parameter (first and second moments)
    params_memory = calculate_parameters_memory(vocab_size, num_layers, d_model, d_ff)
    return 2 * params_memory


def calculate_total_memory(
    batch_size: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
) -> dict[str, int]:
    """
    Calculate total memory usage for training with AdamW.

    Args:
        batch_size: Batch size
        vocab_size: Size of the vocabulary
        context_length: Maximum context length
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension

    Returns:
        Dictionary with memory breakdown in bytes
    """
    parameters = calculate_parameters_memory(vocab_size, num_layers, d_model, d_ff)
    activations = calculate_activations_memory(
        batch_size, context_length, num_layers, d_model, num_heads, d_ff, vocab_size
    )
    gradients = calculate_gradients_memory(vocab_size, num_layers, d_model, d_ff)
    optimizer_state = calculate_optimizer_state_memory(vocab_size, num_layers, d_model, d_ff)

    total = parameters + activations + gradients + optimizer_state

    return {
        "parameters": parameters,
        "activations": activations,
        "gradients": gradients,
        "optimizer_state": optimizer_state,
        "total": total,
    }


def calculate_gpt2_xl_memory(batch_size: int) -> tuple[str, int]:
    """
    Calculate memory usage for GPT-2 XL model.

    Args:
        batch_size: Batch size

    Returns:
        Tuple of (expression string, max batch size for 80GB)
    """
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 6400

    memory_breakdown = calculate_total_memory(
        batch_size=1,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    activations_per_batch = calculate_activations_memory(
        batch_size=1,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
    )

    fixed_memory = memory_breakdown["parameters"] + memory_breakdown["gradients"] + memory_breakdown["optimizer_state"]

    variable_memory = activations_per_batch

    a = variable_memory
    b = fixed_memory

    max_memory = 80 * 1024**3
    max_batch_size = int((max_memory - b) / a)

    return f"{a} * batch_size + {b}", max_batch_size


def calculate_adamw_flops(
    vocab_size: int,
    num_layers: int,
    d_model: int,
    d_ff: int,
) -> int:
    """
    Calculate FLOPs for one step of AdamW optimizer.

    AdamW operations per parameter:
    - First moment update: m = β1 * m + (1 - β1) * g (3 ops)
    - Second moment update: v = β2 * v + (1 - β2) * g^2 (4 ops)
    - Bias correction: α_t calculation (8 ops total, amortized)
    - Parameter update: θ = θ - α_t * m / (sqrt(v) + ε) (3 ops)
    - Weight decay: θ = θ - α * λ * θ (2 ops)
    Total: ~12 ops per parameter

    Args:
        vocab_size: Size of the vocabulary
        num_layers: Number of transformer layers
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension

    Returns:
        Number of FLOPs
    """
    total_params = calculate_parameters_memory(vocab_size, num_layers, d_model, d_ff) // 4

    ops_per_param = 12

    return total_params * ops_per_param


def calculate_training_time(
    batch_size: int,
    num_steps: int,
    context_length: int,
    vocab_size: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    mfu: float = 0.5,
    peak_flops: float = 19.5e12,
) -> float:
    """
    Calculate training time in days.

    Args:
        batch_size: Batch size
        num_steps: Number of training steps
        context_length: Maximum context length
        vocab_size: Size of the vocabulary
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        mfu: Model FLOPs utilization (fraction)
        peak_flops: Peak FLOPs/s of the hardware

    Returns:
        Training time in days
    """
    # Total number of parameters
    total_params = calculate_parameters_memory(vocab_size, num_layers, d_model, d_ff) // 4

    # Tokens processed per step
    tokens_per_step = batch_size * context_length

    # Forward pass FLOPs per step using the 6N rule
    forward_flops = 6 * total_params * tokens_per_step

    # Total FLOPs per step (forward + backward)
    # Backward pass: 2x forward pass FLOPs (standard assumption)
    total_flops_per_step = 3 * forward_flops  # forward + 2*forward for backward

    # AdamW optimizer FLOPs are negligible compared to forward/backward
    # (only ~12 ops per parameter vs 6N ops per token)

    # Total FLOPs for training
    total_flops = total_flops_per_step * num_steps

    # Effective FLOPs/s with MFU
    effective_flops_per_second = peak_flops * mfu

    # Time in seconds
    time_seconds = total_flops / effective_flops_per_second

    # Convert to days
    time_days = time_seconds / (24 * 3600)

    return time_days


def solve_adamw_accounting() -> None:
    """
    Solve the AdamW accounting problem.
    """
    print("=== AdamW Resource Accounting ===\n")

    print("Part (a): Peak memory usage expressions")
    print(
        "Parameters: vocab_size × d_model + num_layers × (2 × d_model + 4 × d_model² + 2 × d_model × d_ff) + d_model + d_model × vocab_size"
    )
    print(
        "Activations: batch_size × context_length × (num_layers × (2 × d_model + 3 × d_model + 2 × num_heads × context_length + 2 × d_model + 2 × d_ff + d_model) + d_model + 2 × vocab_size)"
    )
    print("Gradients: Same as parameters")
    print("Optimizer state: 2 × parameters")
    print("Total: parameters + activations + gradients + optimizer_state\n")

    print("Part (b): GPT-2 XL memory calculation")
    expression, max_batch_size = calculate_gpt2_xl_memory(1)
    print(f"Expression: {expression}")
    print(f"Maximum batch size for 80GB: {max_batch_size}\n")

    print("Part (c): AdamW FLOPs calculation")
    gpt2_xl_flops = calculate_adamw_flops(vocab_size=50257, num_layers=48, d_model=1600, d_ff=6400)
    print(f"AdamW FLOPs per step: {gpt2_xl_flops:,}")
    print(
        "AdamW performs ~12 operations per parameter (moment updates, bias correction, parameter update, weight decay)\n"
    )

    print("Part (d): Training time calculation")

    total_params = calculate_parameters_memory(50257, 48, 1600, 6400) // 4
    print(f"Total parameters: {total_params:,}")

    batch_size = 1024
    context_length = 1024
    tokens_per_step = batch_size * context_length

    forward_flops = 6 * total_params * tokens_per_step
    total_flops_per_step = 3 * forward_flops

    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Forward pass FLOPs per step (6N rule): {forward_flops:,}")
    print(f"Total FLOPs per step (forward + 2×backward): {total_flops_per_step:,}")

    training_days = calculate_training_time(
        batch_size=1024,
        num_steps=400000,
        context_length=1024,
        vocab_size=50257,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=6400,
        mfu=0.5,
        peak_flops=19.5e12,
    )
    print(f"\nTraining time for GPT-2 XL (400K steps, batch size 1024, 50% MFU): {training_days:.1f} days")
    print("This is ~40 years, which shows why distributed training is necessary!")


if __name__ == "__main__":
    solve_adamw_accounting()
