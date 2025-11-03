#!/usr/bin/env python3
"""
Complete example of text generation workflow.

This script demonstrates the full pipeline:
1. Load a trained model checkpoint
2. Load the tokenizer
3. Generate text with different sampling strategies
4. Compare outputs

This is a working example that you can adapt for your own use.
"""

import torch
from cs336_basics import TransformerLM, Tokenizer, generate


def example_generation_workflow():
    """
    Complete example showing how to use the generation module.
    """
    
    print("="*80)
    print("Text Generation Example Workflow")
    print("="*80)
    
    # ========================================================================
    # Step 1: Configuration
    # ========================================================================
    print("\n[Step 1] Configuration")
    print("-" * 80)
    
    # You'll need to update these paths to point to your actual files
    CHECKPOINT_PATH = "test_checkpoints/checkpoint_latest.pt"  # Example path
    VOCAB_PATH = "artifacts/tinystories_bpe.yaml"
    MERGES_PATH = "artifacts/tinystories_bpe_merges.txt"
    
    # Generation parameters
    PROMPT = "Once upon a time"
    MAX_TOKENS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Vocabulary: {VOCAB_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Max tokens: {MAX_TOKENS}")
    
    # ========================================================================
    # Step 2: Load Tokenizer
    # ========================================================================
    print("\n[Step 2] Loading Tokenizer")
    print("-" * 80)
    
    try:
        tokenizer = Tokenizer.from_files(
            vocab_filepath=VOCAB_PATH,
            merges_filepath=MERGES_PATH,
            special_tokens=["<|endoftext|>"],
        )
        
        # Get the end-of-text token ID
        eos_token_id = tokenizer.bytes_to_id[b"<|endoftext|>"]
        
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocabulary size: {len(tokenizer.id_to_bytes)}")
        print(f"  EOS token ID: {eos_token_id}")
        
    except FileNotFoundError as e:
        print(f"✗ Error loading tokenizer: {e}")
        print("\nPlease make sure you have:")
        print("  1. Downloaded the data (see README.md)")
        print("  2. Trained a BPE tokenizer")
        return
    
    # ========================================================================
    # Step 3: Load Model
    # ========================================================================
    print("\n[Step 3] Loading Model")
    print("-" * 80)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        # Check if checkpoint contains config
        if "config" in checkpoint:
            # Best case: checkpoint has config
            config = checkpoint["config"]
            model = TransformerLM(**config)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
            model.load_state_dict(state_dict)
            print("✓ Model loaded from checkpoint with config")
        elif "model" in checkpoint:
            # Fallback: infer config from state dict
            print("⚠ Checkpoint doesn't contain config, inferring from state dict...")
            state_dict = checkpoint["model"]

            # Infer model architecture from state dict
            vocab_size = state_dict["token_embeddings.weight"].shape[0]
            d_model = state_dict["token_embeddings.weight"].shape[1]
            num_layers = sum(1 for key in state_dict.keys() if key.startswith("transformer_blocks.") and key.endswith(".ln1.weight"))
            num_heads = state_dict["transformer_blocks.0.attn.q_proj.weight"].shape[0] // d_model
            d_ff = state_dict["transformer_blocks.0.ffn.w1.weight"].shape[0]
            context_length = 512  # Default
            use_rope = "transformer_blocks.0.attn.rope.inv_freq" in state_dict

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
            print("✓ Model loaded (inferred config)")
        else:
            # Last resort: state dict directly
            print("⚠ Checkpoint doesn't contain config, inferring from state dict...")
            state_dict = checkpoint

            # Infer model architecture from state dict
            vocab_size = state_dict["token_embeddings.weight"].shape[0]
            d_model = state_dict["token_embeddings.weight"].shape[1]
            num_layers = sum(1 for key in state_dict.keys() if key.startswith("transformer_blocks.") and key.endswith(".ln1.weight"))
            num_heads = state_dict["transformer_blocks.0.attn.q_proj.weight"].shape[0] // d_model
            d_ff = state_dict["transformer_blocks.0.ffn.w1.weight"].shape[0]
            context_length = 512  # Default
            use_rope = "transformer_blocks.0.attn.rope.inv_freq" in state_dict

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
            print("✓ Model loaded (inferred config)")
        
        model = model.to(DEVICE)
        model.eval()
        
        print(f"  Vocabulary size: {model.vocab_size}")
        print(f"  Context length: {model.context_length}")
        print(f"  Number of layers: {model.num_layers}")
        
    except FileNotFoundError:
        print(f"✗ Checkpoint not found: {CHECKPOINT_PATH}")
        print("\nPlease train a model first using train.py")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # ========================================================================
    # Step 4: Encode Prompt
    # ========================================================================
    print("\n[Step 4] Encoding Prompt")
    print("-" * 80)
    
    prompt_ids = torch.tensor(
        tokenizer.encode(PROMPT),
        device=DEVICE
    ).unsqueeze(0)
    
    print(f"Prompt: '{PROMPT}'")
    print(f"Token IDs: {prompt_ids[0].tolist()}")
    print(f"Number of tokens: {prompt_ids.shape[1]}")
    
    # ========================================================================
    # Step 5: Generate with Different Strategies
    # ========================================================================
    print("\n[Step 5] Generating Text with Different Strategies")
    print("="*80)
    
    strategies = [
        {
            "name": "Greedy (temperature ≈ 0)",
            "temperature": 0.01,
            "top_p": None,
        },
        {
            "name": "Standard Sampling (temperature = 1.0)",
            "temperature": 1.0,
            "top_p": None,
        },
        {
            "name": "High Temperature (temperature = 1.5)",
            "temperature": 1.5,
            "top_p": None,
        },
        {
            "name": "Nucleus Sampling (top-p = 0.9)",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        {
            "name": "Conservative Nucleus (top-p = 0.5)",
            "temperature": 1.0,
            "top_p": 0.5,
        },
        {
            "name": "Combined (temp = 0.8, top-p = 0.9)",
            "temperature": 0.8,
            "top_p": 0.9,
        },
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n[{i}/{len(strategies)}] {strategy['name']}")
        print("-" * 80)
        print(f"Parameters: temperature={strategy['temperature']}, top_p={strategy['top_p']}")
        print()
        
        # Generate
        with torch.no_grad():
            generated_ids = generate(
                model=model,
                prompt=prompt_ids,
                max_tokens=MAX_TOKENS,
                temperature=strategy["temperature"],
                top_p=strategy["top_p"],
                eos_token_id=eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        # Print result
        print(generated_text)
        
        # Statistics
        num_generated = generated_ids.shape[1] - prompt_ids.shape[1]
        print()
        print(f"Generated {num_generated} tokens")
        
        # Check if stopped at EOS
        if eos_token_id in generated_ids[0]:
            eos_positions = (generated_ids[0] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0 and eos_positions[0] >= prompt_ids.shape[1]:
                print(f"Stopped at EOS token")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("Example Complete!")
    print("="*80)
    print("\nKey Observations:")
    print("  • Greedy decoding (low temperature) produces deterministic, focused text")
    print("  • High temperature produces more diverse but potentially less coherent text")
    print("  • Nucleus sampling (top-p) balances diversity and quality")
    print("  • Combining temperature and top-p often gives best results")
    print("\nNext Steps:")
    print("  • Try different prompts")
    print("  • Experiment with different parameter combinations")
    print("  • Use generate_text.py for command-line generation")
    print("  • Use demo_generation.py --interactive for interactive mode")


if __name__ == "__main__":
    example_generation_workflow()

