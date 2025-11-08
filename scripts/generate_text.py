#!/usr/bin/env python3
"""
Simple script to generate text from a trained language model.

Usage:
    python generate_text.py --checkpoint path/to/checkpoint.pt --prompt "Once upon a time"
"""

import argparse
import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.models import TransformerLM
from cs336_basics.generation import generate
from scripts.encode_dataset import load_tokenizer_yaml


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained language model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer YAML file (e.g., artifacts/owt_bpe.yaml)",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Path to vocabulary JSON file (e.g., artifacts/tinystories_vocab.json)",
    )
    parser.add_argument(
        "--merges",
        type=str,
        default=None,
        help="Path to merges TXT file (e.g., artifacts/tinystories_merges.txt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt to start generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cpu or cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-eos-stop",
        action="store_true",
        help="Don't stop generation at EOS token",
    )

    # Manual architecture specification (for checkpoints without config)
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Number of attention heads (if checkpoint doesn't contain config)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context length (if checkpoint doesn't contain config)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=10000.0,
        help="RoPE theta parameter (default: 10000.0)",
    )

    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load tokenizer
    if args.tokenizer:
        # Load from YAML file
        print(f"Loading tokenizer from {args.tokenizer}...")
        vocab, merges = load_tokenizer_yaml(args.tokenizer)
        tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    elif args.vocab and args.merges:
        # Load from JSON vocab + TXT merges
        print(f"Loading tokenizer from {args.vocab} and {args.merges}...")
        tokenizer = Tokenizer.from_files(
            vocab_filepath=args.vocab,
            merges_filepath=args.merges,
            special_tokens=["<|endoftext|>"],
        )
    else:
        raise ValueError("Must provide either --tokenizer (YAML) or both --vocab and --merges (JSON/TXT)")
    
    # Get the end-of-text token ID
    eos_token_id = tokenizer.bytes_to_id[b"<|endoftext|>"]
    print(f"End-of-text token ID: {eos_token_id}")
    
    # Load model checkpoint
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # Helper function to infer config from state_dict
    def infer_config_from_state_dict(state_dict):
        """Infer model configuration from state dict weights."""
        vocab_size = state_dict["token_embeddings.weight"].shape[0]
        d_model = state_dict["token_embeddings.weight"].shape[1]
        num_layers = sum(1 for key in state_dict.keys()
                       if key.startswith("transformer_blocks.")
                       and key.endswith(".ln1.weight"))
        d_ff = state_dict["transformer_blocks.0.ffn.w1.weight"].shape[0]
        use_rope = any("rope" in key for key in state_dict.keys())

        # Guess num_heads based on common configurations
        if d_model == 512:
            num_heads = 16
        elif d_model == 768:
            num_heads = 12
        elif d_model == 1024:
            num_heads = 16
        else:
            num_heads = max(1, d_model // 64)

        print(f"WARNING: Cannot infer num_heads from checkpoint. Guessing num_heads={num_heads}")
        print(f"         If generation quality is poor, specify --num-heads explicitly")

        return {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'num_layers': num_layers,
            'd_ff': d_ff,
            'use_rope': use_rope,
            'num_heads': num_heads,
            'context_length': 512,  # Default
            'theta': 10000.0,  # Default
        }

    # Helper function to apply manual overrides
    def apply_overrides(config, args):
        """Apply manual parameter overrides from command line args."""
        if args.num_heads is not None:
            if 'num_heads' in config and config['num_heads'] != args.num_heads:
                print(f"  Overriding num_heads: {config['num_heads']} -> {args.num_heads}")
            config['num_heads'] = args.num_heads

        if args.context_length is not None:
            if 'context_length' in config and config['context_length'] != args.context_length:
                print(f"  Overriding context_length: {config['context_length']} -> {args.context_length}")
            config['context_length'] = args.context_length

        if args.theta != 10000.0:
            if 'theta' in config and config.get('theta', 10000.0) != args.theta:
                print(f"  Overriding theta: {config.get('theta', 10000.0)} -> {args.theta}")
            config['theta'] = args.theta

        return config

    # Extract model configuration from checkpoint
    if "config" in checkpoint:
        # Checkpoint with config (preferred)
        print("✓ Found model config in checkpoint")
        config = checkpoint["config"].copy()
        config = apply_overrides(config, args)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
    else:
        # Checkpoint without config - need to infer
        print("Warning: Checkpoint does not contain config. Attempting to infer...")
        state_dict = checkpoint.get("model", checkpoint)
        config = infer_config_from_state_dict(state_dict)
        config = apply_overrides(config, args)

    # Print final config
    print(f"Model config: vocab_size={config['vocab_size']}, d_model={config['d_model']}, "
          f"num_layers={config['num_layers']}, num_heads={config['num_heads']}, "
          f"d_ff={config['d_ff']}, use_rope={config['use_rope']}, context_length={config['context_length']}")

    # Create and load model
    model = TransformerLM(**config)
    model.load_state_dict(state_dict)
    
    model = model.to(args.device)
    model.eval()
    
    print(f"\nModel loaded successfully!")
    print(f"Vocab size: {model.vocab_size}")
    print(f"Context length: {model.context_length}")
    print(f"Number of layers: {model.num_layers}")
    
    # Encode prompt
    print(f"\nPrompt: {args.prompt}")
    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), device=args.device).unsqueeze(0)
    print(f"Prompt tokens: {prompt_ids.shape[1]}")
    
    # Generate
    print(f"\nGenerating with temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}...")
    generated_ids = generate(
        model=model,
        prompt=prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=None if args.no_eos_stop else eos_token_id,
    )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    print("\n" + "="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(generated_text)
    print("="*80)
    
    # Print some statistics
    num_generated = generated_ids.shape[1] - prompt_ids.shape[1]
    print(f"\nGenerated {num_generated} tokens")
    
    # Check if generation stopped due to EOS token
    if eos_token_id in generated_ids[0]:
        eos_position = (generated_ids[0] == eos_token_id).nonzero(as_tuple=True)[0][0].item()
        if eos_position >= prompt_ids.shape[1]:
            print(f"Generation stopped at EOS token (position {eos_position})")


if __name__ == "__main__":
    main()
