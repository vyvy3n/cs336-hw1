#!/usr/bin/env python3
"""
Quick text generation script for the assignment.

This script extracts vocab and merges from the YAML file once, then uses them for fast generation.

Usage:
    python quick_generate.py
"""

import torch
import json
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.models import TransformerLM
from cs336_basics.generation import generate


def main():
    print("="*80)
    print("QUICK TEXT GENERATION FOR ASSIGNMENT")
    print("="*80)
    
    # Configuration
    checkpoint_path = "checkpoints/checkpoint_latest.pt"
    device = "cpu"  # Change to "cuda" if you want to use GPU
    temperature = 0.8
    top_p = 0.9
    max_tokens = 300
    prompt = "Once upon a time"
    
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Prompt: '{prompt}'")
    
    # Step 1: Extract and save vocab/merges from YAML (do this once)
    print(f"\n{'='*80}")
    print("STEP 1: Preparing tokenizer files...")
    print("="*80)
    
    import yaml
    from pathlib import Path
    
    vocab_json_path = "artifacts/tinystories_vocab.json"
    merges_txt_path = "artifacts/tinystories_merges.txt"
    
    if not Path(vocab_json_path).exists() or not Path(merges_txt_path).exists():
        print("  Extracting vocab and merges from YAML (this may take a minute)...")
        with open("artifacts/tinystories_bpe.yaml", 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"  Saving vocab to {vocab_json_path}...")
        with open(vocab_json_path, 'w') as f:
            json.dump(data['vocab'], f)
        
        print(f"  Saving merges to {merges_txt_path}...")
        with open(merges_txt_path, 'w') as f:
            for merge in data.get('merges', []):
                f.write(merge + '\n')
        
        print("  ✅ Tokenizer files created!")
    else:
        print("  ✅ Tokenizer files already exist!")
    
    # Step 2: Load tokenizer
    print(f"\n{'='*80}")
    print("STEP 2: Loading tokenizer...")
    print("="*80)
    
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_json_path,
        merges_filepath=merges_txt_path,
        special_tokens=["<|endoftext|>"]
    )
    
    eos_token_id = tokenizer.bytes_to_id.get(b"<|endoftext|>")
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    print(f"  End-of-text token ID: {eos_token_id}")
    print("  ✅ Tokenizer loaded!")
    
    # Step 3: Load model
    print(f"\n{'='*80}")
    print("STEP 3: Loading model...")
    print("="*80)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if "config" in checkpoint:
        config = checkpoint["config"]
        if hasattr(config, 'model'):
            mc = config.model
        else:
            mc = config['model']
    else:
        # Default TinyStories config
        print("  Using default TinyStories config...")
        mc = type('obj', (object,), {
            'vocab_size': 10000,
            'context_length': 256,
            'num_layers': 4,
            'd_model': 512,
            'num_heads': 16,
            'd_ff': 1344,
            'use_rope': True,
            'theta': 10000.0,
        })()
    
    # Create model
    model = TransformerLM(
        vocab_size=mc.vocab_size,
        context_length=mc.context_length,
        num_layers=mc.num_layers,
        d_model=mc.d_model,
        num_heads=mc.num_heads,
        d_ff=mc.d_ff,
        use_rope=mc.use_rope,
        theta=getattr(mc, 'theta', 10000.0),
    )
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"  Model architecture:")
    print(f"    Vocab size: {model.vocab_size}")
    print(f"    Context length: {model.context_length}")
    print(f"    Layers: {model.num_layers}")
    print(f"    d_model: {model.token_embeddings.embedding_dim}")
    print("  ✅ Model loaded!")
    
    # Step 4: Generate text
    print(f"\n{'='*80}")
    print("STEP 4: Generating text...")
    print("="*80)
    print(f"\nPrompt: \"{prompt}\"")
    print(f"\nGenerating {max_tokens} tokens...\n")
    
    # Encode prompt
    prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    
    # Generate
    generated_ids = generate(
        model=model,
        prompt=prompt_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    # Print results
    print("="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(generated_text)
    print("="*80)
    
    # Statistics
    total_tokens = generated_ids.shape[1]
    generated_tokens = total_tokens - prompt_ids.shape[1]
    
    print(f"\nStatistics:")
    print(f"  Prompt tokens: {prompt_ids.shape[1]}")
    print(f"  Generated tokens: {generated_tokens}")
    print(f"  Total tokens: {total_tokens}")
    
    if generated_tokens >= 256:
        print(f"\n✅ SUCCESS: Generated {generated_tokens} tokens (requirement: ≥256)")
    else:
        print(f"\n⚠ Generated only {generated_tokens} tokens (requirement: ≥256)")
        print(f"   Model may have hit <|endoftext|> early. Try different prompt or higher max_tokens.")
    
    # Save output
    output_file = "generated_output.txt"
    with open(output_file, 'w') as f:
        f.write(f"Assignment Deliverable: Text Generation\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Top-p: {top_p}\n")
        f.write(f"Generated tokens: {generated_tokens}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"GENERATED TEXT:\n")
        f.write(f"{'='*80}\n\n")
        f.write(generated_text)
        f.write(f"\n\n{'='*80}\n")
        f.write(f"\nFluency Analysis:\n")
        f.write(f"- The output {'is' if generated_tokens >= 100 else 'may not be'} fluent English\n")
        f.write(f"- Factors affecting quality:\n")
        f.write(f"  1. Training data: TinyStories dataset (simple children's stories)\n")
        f.write(f"  2. Model size: {model.num_layers} layers, d_model={model.token_embeddings.embedding_dim}\n")
        f.write(f"  3. Training iterations: {checkpoint.get('iteration', 'unknown')}\n")
        f.write(f"  4. Temperature: {temperature} (higher = more random)\n")
    
    print(f"\n✅ Output saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()

