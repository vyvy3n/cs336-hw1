#!/usr/bin/env python3
"""
Interactive demo for text generation with various sampling strategies.

This script demonstrates different decoding strategies:
- Greedy decoding (temperature=0.01)
- Temperature sampling
- Top-p (nucleus) sampling
"""

import torch
from cs336_basics import TransformerLM, Tokenizer, generate


def demo_generation():
    """
    Demo function showing different generation strategies.
    
    This is a template - you'll need to provide your own checkpoint and tokenizer files.
    """
    
    # Configuration - UPDATE THESE PATHS
    checkpoint_path = "path/to/your/checkpoint.pt"
    vocab_path = "artifacts/tinystories_bpe.yaml"
    merges_path = "artifacts/tinystories_bpe_merges.txt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("Text Generation Demo")
    print("="*80)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"],
    )
    eos_token_id = tokenizer.bytes_to_id[b"<|endoftext|>"]
    print(f"   Vocabulary size: {len(tokenizer.id_to_bytes)}")
    print(f"   EOS token ID: {eos_token_id}")
    
    # Load model
    print("\n2. Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" in checkpoint:
        config = checkpoint["config"]
        model = TransformerLM(**config)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
        model.load_state_dict(state_dict)
    elif "model" in checkpoint:
        # Infer from state dict
        state_dict = checkpoint["model"]
        vocab_size = state_dict["token_embeddings.weight"].shape[0]
        d_model = state_dict["token_embeddings.weight"].shape[1]
        num_layers = sum(1 for key in state_dict.keys() if key.startswith("transformer_blocks.") and key.endswith(".ln1.weight"))
        num_heads = state_dict["transformer_blocks.0.attn.q_proj.weight"].shape[0] // d_model
        d_ff = state_dict["transformer_blocks.0.ffn.w1.weight"].shape[0]
        use_rope = "transformer_blocks.0.attn.rope.inv_freq" in state_dict

        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=512,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            use_rope=use_rope,
        )
        model.load_state_dict(state_dict)
    else:
        # Fallback: load state dict directly
        state_dict = checkpoint
        vocab_size = state_dict["token_embeddings.weight"].shape[0]
        d_model = state_dict["token_embeddings.weight"].shape[1]
        num_layers = sum(1 for key in state_dict.keys() if key.startswith("transformer_blocks.") and key.endswith(".ln1.weight"))
        num_heads = state_dict["transformer_blocks.0.attn.q_proj.weight"].shape[0] // d_model
        d_ff = state_dict["transformer_blocks.0.ffn.w1.weight"].shape[0]
        use_rope = "transformer_blocks.0.attn.rope.inv_freq" in state_dict

        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=512,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            use_rope=use_rope,
        )
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    print(f"   Model loaded on {device}")
    
    # Demo prompts
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a magical forest",
    ]
    
    # Demo different sampling strategies
    strategies = [
        {"name": "Greedy (temperature≈0)", "temperature": 0.01, "top_p": None},
        {"name": "Low temperature", "temperature": 0.5, "top_p": None},
        {"name": "Standard sampling", "temperature": 1.0, "top_p": None},
        {"name": "High temperature", "temperature": 1.5, "top_p": None},
        {"name": "Nucleus sampling (p=0.9)", "temperature": 1.0, "top_p": 0.9},
        {"name": "Nucleus sampling (p=0.5)", "temperature": 1.0, "top_p": 0.5},
    ]
    
    print("\n" + "="*80)
    print("GENERATION EXAMPLES")
    print("="*80)
    
    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*80}")
        
        for strategy in strategies:
            print(f"\n--- {strategy['name']} ---")
            
            # Encode prompt
            prompt_ids = torch.tensor(
                tokenizer.encode(prompt), 
                device=device
            ).unsqueeze(0)
            
            # Generate
            with torch.no_grad():
                generated_ids = generate(
                    model=model,
                    prompt=prompt_ids,
                    max_tokens=50,
                    temperature=strategy["temperature"],
                    top_p=strategy["top_p"],
                    eos_token_id=eos_token_id,
                )
            
            # Decode
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            print(generated_text)
    
    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)


def interactive_generation():
    """
    Interactive mode where user can input prompts.
    """
    
    # Configuration - UPDATE THESE PATHS
    checkpoint_path = "path/to/your/checkpoint.pt"
    vocab_path = "artifacts/tinystories_bpe.yaml"
    merges_path = "artifacts/tinystories_bpe_merges.txt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("Interactive Text Generation")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"],
    )
    eos_token_id = tokenizer.bytes_to_id[b"<|endoftext|>"]
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" in checkpoint:
        config = checkpoint["config"]
        model = TransformerLM(**config)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
        model.load_state_dict(state_dict)
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        vocab_size = state_dict["token_embeddings.weight"].shape[0]
        d_model = state_dict["token_embeddings.weight"].shape[1]
        num_layers = sum(1 for key in state_dict.keys() if key.startswith("transformer_blocks.") and key.endswith(".ln1.weight"))
        num_heads = state_dict["transformer_blocks.0.attn.q_proj.weight"].shape[0] // d_model
        d_ff = state_dict["transformer_blocks.0.ffn.w1.weight"].shape[0]
        use_rope = "transformer_blocks.0.attn.rope.inv_freq" in state_dict

        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=512,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            use_rope=use_rope,
        )
        model.load_state_dict(state_dict)
    else:
        state_dict = checkpoint
        vocab_size = state_dict["token_embeddings.weight"].shape[0]
        d_model = state_dict["token_embeddings.weight"].shape[1]
        num_layers = sum(1 for key in state_dict.keys() if key.startswith("transformer_blocks.") and key.endswith(".ln1.weight"))
        num_heads = state_dict["transformer_blocks.0.attn.q_proj.weight"].shape[0] // d_model
        d_ff = state_dict["transformer_blocks.0.ffn.w1.weight"].shape[0]
        use_rope = "transformer_blocks.0.attn.rope.inv_freq" in state_dict

        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=512,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            use_rope=use_rope,
        )
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}\n")
    
    # Interactive loop
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print("You can also adjust parameters by typing 'config'.\n")
    
    # Default parameters
    max_tokens = 100
    temperature = 1.0
    top_p = None
    
    while True:
        prompt = input("Prompt: ").strip()
        
        if prompt.lower() == "quit":
            print("Goodbye!")
            break
        
        if prompt.lower() == "config":
            print("\nCurrent configuration:")
            print(f"  max_tokens: {max_tokens}")
            print(f"  temperature: {temperature}")
            print(f"  top_p: {top_p}")
            
            try:
                max_tokens = int(input("New max_tokens (or press Enter to keep): ") or max_tokens)
                temperature = float(input("New temperature (or press Enter to keep): ") or temperature)
                top_p_input = input("New top_p (or press Enter to keep): ")
                if top_p_input:
                    top_p = float(top_p_input) if top_p_input.lower() != "none" else None
            except ValueError:
                print("Invalid input, keeping previous values.")
            
            print()
            continue
        
        if not prompt:
            continue
        
        # Encode and generate
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt),
            device=device
        ).unsqueeze(0)
        
        with torch.no_grad():
            generated_ids = generate(
                model=model,
                prompt=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )
        
        # Decode and display
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        print(f"\n{generated_text}\n")
        print("-"*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_generation()
    else:
        print("Running demo mode. Use --interactive for interactive mode.")
        print("Note: You need to update the checkpoint and tokenizer paths in the script.\n")
        demo_generation()

