#!/usr/bin/env python3
"""
Generate text from a trained TinyStories model for the assignment.

This script loads a trained checkpoint and generates at least 256 tokens of text
as required by the assignment deliverable.

Usage:
    # Generate with default settings
    python ts_generate_example.py

    # Generate with custom checkpoint
    python ts_generate_example.py --checkpoint checkpoints/lr_sweep/lr_1e_03/checkpoint_iter_40000.pt
    
    # Generate with different temperature
    python ts_generate_example.py --temperature 0.8
    
    # Generate with top-p sampling
    python ts_generate_example.py --top_p 0.9
"""

import argparse
import torch
import yaml
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.models import TransformerLM
from cs336_basics.generation import generate


def load_tokenizer_from_yaml(yaml_path: str) -> Tokenizer:
    """Load tokenizer from a YAML file containing vocab and merges."""
    print(f"  Loading YAML file (this may take a moment)...")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    print(f"  Processing vocab...")
    vocab = data['vocab']
    merges_list = data.get('merges', [])

    print(f"  Processing {len(merges_list)} merges...")
    # Convert merges from list of strings to list of tuples of bytes
    merges = []
    for merge_str in merges_list:
        parts = merge_str.split()
        if len(parts) == 2:
            merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))

    print(f"  Creating tokenizer...")
    return Tokenizer(
        vocab={int(k): v.encode('utf-8') if isinstance(v, str) else v for k, v in vocab.items()},
        merges=merges,
        special_tokens=["<|endoftext|>"]
    )


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained TinyStories model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_latest.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="artifacts/tinystories_bpe.yaml",
        help="Path to tokenizer YAML file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt to start generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=300,
        help="Maximum number of tokens to generate (assignment requires at least 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random, lower = more deterministic)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold (e.g., 0.9)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu or cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
    
    print("="*80)
    print("TINYSTORIES TEXT GENERATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Device: {args.device}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Max tokens: {args.max_tokens}")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer_from_yaml(args.tokenizer)
    
    # Get end-of-text token ID
    eos_token_id = tokenizer.bytes_to_id.get(b"<|endoftext|>")
    print(f"End-of-text token ID: {eos_token_id}")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Extract model configuration
    if "config" in checkpoint:
        config = checkpoint["config"]
        model_config = config.model if hasattr(config, 'model') else config['model']
    elif "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
    else:
        # Try to infer from model state dict
        print("Warning: No config found in checkpoint, using default TinyStories config")
        model_config = {
            'vocab_size': 10000,
            'context_length': 256,
            'num_layers': 4,
            'd_model': 512,
            'num_heads': 16,
            'd_ff': 1344,
            'use_rope': True,
            'theta': 10000.0,
        }
    
    # Create model
    print("\nCreating model...")
    if hasattr(model_config, '__dict__'):
        # It's a config object
        model = TransformerLM(
            vocab_size=model_config.vocab_size,
            context_length=model_config.context_length,
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            use_rope=model_config.use_rope,
            theta=getattr(model_config, 'theta', 10000.0),
        )
    else:
        # It's a dict
        model = TransformerLM(
            vocab_size=model_config['vocab_size'],
            context_length=model_config['context_length'],
            num_layers=model_config['num_layers'],
            d_model=model_config['d_model'],
            num_heads=model_config['num_heads'],
            d_ff=model_config['d_ff'],
            use_rope=model_config.get('use_rope', True),
            theta=model_config.get('theta', 10000.0),
        )
    
    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  Context length: {model.context_length}")
    print(f"  Number of layers: {model.num_layers}")
    print(f"  d_model: {model.token_embeddings.embedding_dim}")
    
    # Encode prompt
    print(f"\n{'='*80}")
    print(f"PROMPT: {args.prompt}")
    print(f"{'='*80}")
    
    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), device=args.device).unsqueeze(0)
    print(f"Prompt tokens: {prompt_ids.shape[1]}")
    
    # Generate
    print(f"\nGenerating {args.max_tokens} tokens...")
    print(f"{'='*80}\n")
    
    generated_ids = generate(
        model=model,
        prompt=prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    # Print results
    print("GENERATED TEXT:")
    print("="*80)
    print(generated_text)
    print("="*80)
    
    # Count tokens
    total_tokens = generated_ids.shape[1]
    generated_tokens = total_tokens - prompt_ids.shape[1]
    
    print(f"\nSTATISTICS:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Prompt tokens: {prompt_ids.shape[1]}")
    print(f"  Generated tokens: {generated_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    
    # Check if we met the assignment requirement
    if generated_tokens >= 256:
        print(f"\n✅ SUCCESS: Generated {generated_tokens} tokens (requirement: ≥256)")
    else:
        print(f"\n⚠ WARNING: Only generated {generated_tokens} tokens (requirement: ≥256)")
        print(f"   The model may have hit the <|endoftext|> token early.")
        print(f"   Try running again or increasing --max_tokens")
    
    print("="*80)
    
    # Save to file
    output_file = "generated_output.txt"
    with open(output_file, 'w') as f:
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Top-p: {args.top_p}\n")
        f.write(f"Generated tokens: {generated_tokens}\n")
        f.write(f"\n{'='*80}\n\n")
        f.write(generated_text)
    
    print(f"\n✅ Output saved to: {output_file}")


if __name__ == "__main__":
    main()

