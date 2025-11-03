#!/usr/bin/env python3
"""
Extract vocab and merges from YAML file to separate JSON and TXT files.

This is a one-time script to convert the YAML format to separate files
that can be loaded quickly.
"""

import yaml
import json


def main():
    print("Extracting tokenizer from YAML...")
    
    # Load YAML with unsafe loader (needed for Python tuples)
    with open("artifacts/tinystories_bpe.yaml", 'r') as f:
        data = yaml.unsafe_load(f)
    
    print(f"Loaded vocab with {len(data['vocab'])} entries")
    print(f"Loaded {len(data.get('merges', []))} merges")

    # Save vocab as JSON
    # Note: Replace spaces with Ġ (U+0120) following GPT-2 convention
    vocab_path = "artifacts/tinystories_vocab.json"
    print(f"Saving vocab to {vocab_path}...")
    vocab_for_json = {}
    for token_id, token_str in data['vocab'].items():
        # Replace space with Ġ for display purposes
        token_str_encoded = token_str.replace(' ', 'Ġ')
        vocab_for_json[str(token_id)] = token_str_encoded
    with open(vocab_path, 'w') as f:
        json.dump(vocab_for_json, f)
    
    # Save merges as TXT
    # Note: We use Ġ (U+0120) to represent space characters, following GPT-2 convention
    # This allows us to use space as a delimiter in the merges file
    merges_path = "artifacts/tinystories_merges.txt"
    print(f"Saving merges to {merges_path}...")
    with open(merges_path, 'w') as f:
        for merge in data.get('merges', []):
            if isinstance(merge, tuple) and len(merge) == 2:
                # Replace space with Ġ for display purposes
                left = merge[0].replace(' ', 'Ġ')
                right = merge[1].replace(' ', 'Ġ')
                f.write(f"{left} {right}\n")
            else:
                # Unexpected format
                print(f"Warning: unexpected merge format: {merge}")
    
    print("✅ Done!")
    print(f"   Vocab: {vocab_path}")
    print(f"   Merges: {merges_path}")


if __name__ == "__main__":
    main()

