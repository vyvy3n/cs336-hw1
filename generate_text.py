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


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained language model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to vocabulary file (e.g., artifacts/tinystories_bpe.yaml)",
    )
    parser.add_argument(
        "--merges",
        type=str,
        required=True,
        help="Path to merges file (e.g., artifacts/tinystories_bpe_merges.txt)",
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

    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab} and {args.merges}...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=["<|endoftext|>"],
    )
    
    # Get the end-of-text token ID
    eos_token_id = tokenizer.bytes_to_id[b"<|endoftext|>"]
    print(f"End-of-text token ID: {eos_token_id}")
    
    # Load model checkpoint
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Extract model configuration from checkpoint
    # Handle different checkpoint formats
    if "config" in checkpoint:
        # Format 1: checkpoint with config
        config = checkpoint["config"]
        model = TransformerLM(**config)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
        model.load_state_dict(state_dict)
    elif "model" in checkpoint:
        # Format 2: checkpoint with 'model' key (common format)
        print("Warning: Checkpoint does not contain config. Attempting to infer...")
        state_dict = checkpoint["model"]

        # Infer vocab size from embedding layer
        vocab_size = state_dict["token_embeddings.weight"].shape[0]
        d_model = state_dict["token_embeddings.weight"].shape[1]

        # Count transformer blocks
        num_layers = sum(1 for key in state_dict.keys() if key.startswith("transformer_blocks.") and key.endswith(".ln1.weight"))

        # Get other dimensions from first transformer block
        num_heads = state_dict["transformer_blocks.0.attn.q_proj.weight"].shape[0] // d_model
        d_ff = state_dict["transformer_blocks.0.ffn.w1.weight"].shape[0]

        # Assume some defaults
        context_length = 512  # Default, may not be accurate
        use_rope = "transformer_blocks.0.attn.rope.inv_freq" in state_dict

        print(f"Inferred config: vocab_size={vocab_size}, d_model={d_model}, num_layers={num_layers}, "
              f"num_heads={num_heads}, d_ff={d_ff}, use_rope={use_rope}")

        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            use_rope=use_rope,
        )
        model.load_state_dict(state_dict)
    else:
        # Format 3: checkpoint is the state dict itself
        print("Warning: Checkpoint does not contain config. Attempting to infer...")
        state_dict = checkpoint

        # Infer vocab size from embedding layer
        vocab_size = state_dict["token_embeddings.weight"].shape[0]
        d_model = state_dict["token_embeddings.weight"].shape[1]

        # Count transformer blocks
        num_layers = sum(1 for key in state_dict.keys() if key.startswith("transformer_blocks.") and key.endswith(".ln1.weight"))

        # Get other dimensions from first transformer block
        num_heads = state_dict["transformer_blocks.0.attn.q_proj.weight"].shape[0] // d_model
        d_ff = state_dict["transformer_blocks.0.ffn.w1.weight"].shape[0]

        # Assume some defaults
        context_length = 512  # Default, may not be accurate
        use_rope = "transformer_blocks.0.attn.rope.inv_freq" in state_dict

        print(f"Inferred config: vocab_size={vocab_size}, d_model={d_model}, num_layers={num_layers}, "
              f"num_heads={num_heads}, d_ff={d_ff}, use_rope={use_rope}")

        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            use_rope=use_rope,
        )
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

