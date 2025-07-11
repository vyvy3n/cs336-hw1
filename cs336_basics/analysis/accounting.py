"""
Transformer LM Resource Accounting
"""

from dataclasses import dataclass
from typing import Any, NamedTuple


@dataclass
class ModelConfig:
    """Configuration for a transformer model."""

    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int

    def __post_init__(self):
        """Validate configuration."""
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_mdoel ({self.d_model}) must be divisible by num_heads ({self.num_heads})")

        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_k


class ResourceBreakdown(NamedTuple):
    """Breakdown of resources by component."""

    embedding: float
    transformer_blocks: float
    output_layer: float
    total: float


class FLOPBreakdown(NamedTuple):
    """Breakdown of FLOPs by operation."""

    embedding_lookup: float
    attention_qkv: float
    attention_scores: float
    attention_output: float
    feedforward: float
    output_projection: float
    total: float


def calculate_parameters(config: ModelConfig) -> dict[str, int]:
    """
    Calculate the number of trainable parameters in a transformer model.

    Args:
        config: Model configuration

    Returns:
        Dictionary with parameter counts by component
    """
    params = {}

    # Token embedding: vocab_size × d_model
    params["token_embedding"] = config.vocab_size * config.d_model

    # Per-layer parameters
    layer_params = {}

    # Multi-head attention
    # Q, K, V projections: 3 × (d_model × d_model)
    layer_params["attention_qkv"] = 3 * config.d_model * config.d_model

    # Output projection: d_model × d_model
    layer_params["attention_output"] = config.d_model * config.d_model

    # Feed-forward network (SwiGLU)
    # W1: d_model × d_ff, W2: d_ff × d_model, W3: d_model × d_ff
    layer_params["feedforward"] = (2 * config.d_model * config.d_ff) + (config.d_ff * config.d_model)

    # RMSNorm parameters: 2 × d_model (one for attention, one for FFN)
    layer_params["rmsnorm"] = 2 * config.d_model

    # Total per layer
    params_per_layer = sum(layer_params.values())
    params["transformer_blocks"] = config.num_layers * params_per_layer

    # Final RMSNorm
    params["final_rmsnorm"] = config.d_model

    # Output embedding (LM head): d_model × vocab_size
    params["output_embedding"] = config.d_model * config.vocab_size

    # Total parameters
    params["total"] = sum(params.values())

    # Store per-layer breakdown for reference
    params["per_layer"] = layer_params
    params["per_layer_total"] = params_per_layer

    return params


def calculate_memory_requirements(config: ModelConfig, batch_size: int = 1) -> dict[str, float]:
    """
    Calculate memory requirements in bytes for float32 tensors.

    Args:
        config: Model configuration
        batch_size: Batch size for activation memory calculation

    Returns:
        Dictionary with memory requirements by category (in bytes)
    """
    BYTES_PER_FLOAT32 = 4

    memory = {}

    # Parameters
    param_counts = calculate_parameters(config)
    memory["parameters"] = param_counts["total"] * BYTES_PER_FLOAT32

    # Gradients (same size as parameters)
    memory["gradients"] = memory["parameters"]

    # Optimizer state (AdamW: 2x parameters for momentum estimates)
    memory["optimizer_state"] = 2 * memory["parameters"]

    # Activations (most memory-intensive during training)
    seq_len = config.context_length

    # Token embeddings: batch_size × seq_len × d_model
    embedding_activations = batch_size * seq_len * config.d_model

    # Per-layer activations
    layer_activations = 0

    # RMSNorm activations: batch_size × seq_len × d_model (×2 for both norms)
    layer_activations += 2 * batch_size * seq_len * config.d_model

    # Attention activations
    # QKV projections: batch_size × seq_len × (3 × d_model)
    layer_activations += batch_size * seq_len * 3 * config.d_model

    # Attention scores: batch_size × num_heads × seq_len × seq_len
    layer_activations += batch_size * config.num_heads * seq_len * seq_len

    # Attention output: batch_size × seq_len × d_model
    layer_activations += batch_size * seq_len * config.d_model

    # Feed-forward activations
    # W1, W3 outputs: batch_size × seq_len × d_ff (×2)
    layer_activations += 2 * batch_size * seq_len * config.d_ff

    # W2 output: batch_size × seq_len × d_model
    layer_activations += batch_size * seq_len * config.d_model

    # Total activation memory
    total_activations = embedding_activations + (config.num_layers * layer_activations)

    # Final RMSNorm and output projection
    total_activations += batch_size * seq_len * config.d_model  # Final RMSNorm
    total_activations += batch_size * seq_len * config.vocab_size  # Output logits

    memory["activations"] = total_activations * BYTES_PER_FLOAT32

    # Cross-entropy loss computation (minimal additional memory)
    memory["loss_computation"] = batch_size * seq_len * BYTES_PER_FLOAT32

    # Total memory
    memory["total"] = sum(memory.values())

    return memory


def calculate_forward_pass_flops(config: ModelConfig) -> dict[str, Any]:
    """
    Calculate FLOPs for a single forward pass.

    Rule: Matrix multiply A(m×n) @ B(n×p) requires 2*m*n*p FLOPs

    Args:
        config: Model configuration

    Returns:
        Dictionary with FLOP counts by operation
    """
    flops: dict[str, Any] = {}
    seq_len = config.context_length

    # Embedding lookup is not a matrix multiply, minimal FLOPs
    flops["embedding_lookup"] = 0

    # Per-layer FLOPs
    layer_flops: dict[str, int] = {}

    # Multi-head attention
    # QKV projections: (batch×seq×d_model) @ (d_model×3*d_model) = 2*batch*seq*d_model*3*d_model
    # But we consider batch=1 and seq=seq_len for base calculation
    layer_flops["attention_qkv"] = 2 * seq_len * config.d_model * (3 * config.d_model)

    # Attention scores: Q^T @ K for each head
    # Q: (seq×d_k), K: (seq×d_k) -> Q^T@K: (seq×seq)
    # Per head: 2*seq*d_k*seq, Total: num_heads * 2*seq*d_k*seq
    layer_flops["attention_scores"] = config.num_heads * 2 * seq_len * config.d_k * seq_len

    # Attention output: (attention_weights @ V) for each head
    # attention_weights: (seq×seq), V: (seq×d_k) -> output: (seq×d_k)
    # Per head: 2*seq*seq*d_k, Total: num_heads * 2*seq*seq*d_k
    layer_flops["attention_weighted_sum"] = config.num_heads * 2 * seq_len * seq_len * config.d_k

    # Output projection: (seq×d_model) @ (d_model×d_model)
    layer_flops["attention_output"] = 2 * seq_len * config.d_model * config.d_model

    # Feed-forward network (SwiGLU)
    # W1: (seq×d_model) @ (d_model×d_ff)
    layer_flops["feedforward_w1"] = 2 * seq_len * config.d_model * config.d_ff

    # W2: (seq×d_ff) @ (d_ff×d_model)
    layer_flops["feedforward_w2"] = 2 * seq_len * config.d_ff * config.d_model

    # W3: (seq×d_model) @ (d_model×d_ff)
    layer_flops["feedforward_w3"] = 2 * seq_len * config.d_model * config.d_ff

    # Total per layer
    layer_flops_total = sum(layer_flops.values())
    flops["transformer_blocks"] = config.num_layers * layer_flops_total

    # Output projection: (seq×d_model) @ (d_model×vocab_size)
    flops["output_projection"] = 2 * seq_len * config.d_model * config.vocab_size

    # Total FLOPs
    flops["total"] = sum(flops.values())

    # Store detailed breakdown
    flops["per_layer"] = layer_flops
    flops["per_layer_total"] = layer_flops_total

    return flops


def analyze_model_scaling(configs: list[tuple[str, ModelConfig]]) -> dict[str, dict[str, float]]:
    """
    Analyze how different model sizes affect FLOP distribution.

    Args:
        configs: List of (name, config) tuples

    Returns:
        Dictionary with analysis for each model
    """
    results = {}

    for name, config in configs:
        flops = calculate_forward_pass_flops(config)
        total_flops = flops["total"]

        proportions = {
            "embedding_lookup": flops["embedding_lookup"] / total_flops,
            "attention": (
                flops["per_layer"]["attention_qkv"]
                + flops["per_layer"]["attention_scores"]
                + flops["per_layer"]["attention_weighted_sum"]
                + flops["per_layer"]["attention_output"]
            )
            * config.num_layers
            / total_flops,
            "feedforward": (
                flops["per_layer"]["feedforward_w1"]
                + flops["per_layer"]["feedforward_w2"]
                + flops["per_layer"]["feedforward_w3"]
            )
            * config.num_layers
            / total_flops,
            "output_projection": flops["output_projection"] / total_flops,
            "total_flops": total_flops,
        }

        results[name] = proportions

    return results


def solve_transformer_accounting():
    """
    Solve the transformer accounting problem from the assignment.
    """
    print("=" * 60)
    print("TRANSFORMER LM RESOURCE ACCOUNTING")
    print("=" * 60)

    gpt2_xl = ModelConfig(vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=6400)

    print("\n(a) GPT-2 XL Parameter Count and Memory Requirements")
    print("-" * 50)

    params = calculate_parameters(gpt2_xl)
    print(f"Total parameters: {params['total']:,}")

    # Memory for model weights (float32)
    memory_gb = params["total"] * 4 / (1024**3)  # 4 bytes per float32
    print(f"Memory for model weights: {memory_gb:.2f} GB")

    print("\n(b) Matrix Multiplies and FLOPs for Forward Pass")
    print("-" * 50)

    flops = calculate_forward_pass_flops(gpt2_xl)

    print("Matrix multiplies per forward pass:")
    print(f"  - Token embedding: 0 FLOPs (lookup operation)")
    print(f"  - Per layer ({gpt2_xl.num_layers} layers):")

    per_layer = flops["per_layer"]
    print(f"    * QKV projections: {per_layer['attention_qkv']:,} FLOPs")
    print(f"    * Attention scores (Q^T @ K): {per_layer['attention_scores']:,} FLOPs")
    print(f"    * Attention output (weights @ V): {per_layer['attention_weighted_sum']:,} FLOPs")
    print(f"    * Attention output projection: {per_layer['attention_output']:,} FLOPs")
    print(f"    * Feed-forward W1: {per_layer['feedforward_w1']:,} FLOPs")
    print(f"    * Feed-forward W2: {per_layer['feedforward_w2']:,} FLOPs")
    print(f"    * Feed-forward W3: {per_layer['feedforward_w3']:,} FLOPs")
    print(f"  - Output projection: {flops['output_projection']:,} FLOPs")
    print(f"\nTotal FLOPs: {flops['total']:,}")

    print("\n(c) FLOP Distribution Analysis")
    print("-" * 50)

    total_flops = flops["total"]
    attention_flops = (
        flops["per_layer"]["attention_qkv"]
        + flops["per_layer"]["attention_scores"]
        + flops["per_layer"]["attention_weighted_sum"]
        + flops["per_layer"]["attention_output"]
    ) * gpt2_xl.num_layers

    feedforward_flops = (
        flops["per_layer"]["feedforward_w1"]
        + flops["per_layer"]["feedforward_w2"]
        + flops["per_layer"]["feedforward_w3"]
    ) * gpt2_xl.num_layers

    output_flops = flops["output_projection"]

    print(f"Attention layers: {attention_flops / total_flops * 100:.1f}% ({attention_flops:,} FLOPs)")
    print(f"Feed-forward layers: {feedforward_flops / total_flops * 100:.1f}% ({feedforward_flops:,} FLOPs)")
    print(f"Output projection: {output_flops / total_flops * 100:.1f}% ({output_flops:,} FLOPs)")

    print("\n(d) Comparison with Other GPT-2 Sizes")
    print("-" * 50)

    # GPT-2 configurations
    gpt2_configs = [
        ("GPT-2 Small", ModelConfig(50257, 1024, 12, 768, 12, 3072)),
        ("GPT-2 Medium", ModelConfig(50257, 1024, 24, 1024, 16, 4096)),
        ("GPT-2 Large", ModelConfig(50257, 1024, 36, 1280, 20, 5120)),
        ("GPT-2 XL", gpt2_xl),
    ]

    analysis = analyze_model_scaling(gpt2_configs)

    print(f"{'Model':<15} {'Total FLOPs':<15} {'Attention %':<12} {'FFN %':<8} {'Output %':<8}")
    print("-" * 60)
    for name, results in analysis.items():
        print(
            f"{name:<15} {results['total_flops']:<15,.0f} {results['attention'] * 100:<12.1f} "
            f"{results['feedforward'] * 100:<8.1f} {results['output_projection'] * 100:<8.1f}"
        )

    print("\n(e) Impact of Increasing Context Length")
    print("-" * 50)

    gpt2_xl_long = ModelConfig(
        vocab_size=50257, context_length=16384, num_layers=48, d_model=1600, num_heads=25, d_ff=6400
    )

    flops_long = calculate_forward_pass_flops(gpt2_xl_long)

    flop_ratio = flops_long["total"] / flops["total"]
    print(
        f"Context length increase: {gpt2_xl.context_length} -> {gpt2_xl_long.context_length} "
        f"({gpt2_xl_long.context_length // gpt2_xl.context_length}x)"
    )
    print(f"Total FLOP increase: {flop_ratio:.1f}x")

    attention_ratio = (
        flops_long["per_layer"]["attention_scores"] + flops_long["per_layer"]["attention_weighted_sum"]
    ) / (flops["per_layer"]["attention_scores"] + flops["per_layer"]["attention_weighted_sum"])

    linear_ratio = flops_long["per_layer"]["attention_qkv"] / flops["per_layer"]["attention_qkv"]

    print(f"Attention computation scaling: {attention_ratio:.1f}x (quadratic in sequence length)")
    print(f"Linear layer scaling: {linear_ratio:.1f}x (linear in sequence length)")

    total_flops_long = flops_long["total"]
    attention_flops_long = (
        flops_long["per_layer"]["attention_qkv"]
        + flops_long["per_layer"]["attention_scores"]
        + flops_long["per_layer"]["attention_weighted_sum"]
        + flops_long["per_layer"]["attention_output"]
    ) * gpt2_xl_long.num_layers

    feedforward_flops_long = (
        flops_long["per_layer"]["feedforward_w1"]
        + flops_long["per_layer"]["feedforward_w2"]
        + flops_long["per_layer"]["feedforward_w3"]
    ) * gpt2_xl_long.num_layers

    print(f"\nProportion changes with longer context:")
    print(
        f"Attention: {attention_flops / total_flops * 100:.1f}% -> {attention_flops_long / total_flops_long * 100:.1f}%"
    )
    print(
        f"Feed-forward: {feedforward_flops / total_flops * 100:.1f}% -> {feedforward_flops_long / total_flops_long * 100:.1f}%"
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    solve_transformer_accounting()
