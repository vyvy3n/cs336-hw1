#!/usr/bin/env python3
"""
Convert YAML tokenizer to JSON vocab + TXT merges format.

Usage:
    python scripts/convert_yaml_to_json_txt.py \
        --input artifacts/owt_bpe.yaml \
        --vocab-output artifacts/owt_vocab.json \
        --merges-output artifacts/owt_merges.txt
"""

import argparse
import json
from scripts.encode_dataset import load_tokenizer_yaml


def main():
    parser = argparse.ArgumentParser(description="Convert YAML tokenizer to JSON/TXT format")
    parser.add_argument("--input", type=str, required=True, help="Input YAML file")
    parser.add_argument("--vocab-output", type=str, required=True, help="Output JSON vocab file")
    parser.add_argument("--merges-output", type=str, required=True, help="Output TXT merges file")
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.input}...")
    vocab, merges = load_tokenizer_yaml(args.input)
    
    # Save vocab as JSON
    # Use Ġ (U+0120) to represent spaces (GPT-2 convention)
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        token_str = token_bytes.decode("utf-8", errors="replace").replace(' ', 'Ġ')
        vocab_json[str(token_id)] = token_str
    
    with open(args.vocab_output, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(vocab_json)} tokens to {args.vocab_output}")
    
    # Save merges as TXT
    # Use Ġ (U+0120) to represent spaces (GPT-2 convention) to avoid ambiguity
    with open(args.merges_output, "w", encoding="utf-8") as f:
        for left, right in merges:
            left_str = left.decode("utf-8", errors="replace").replace(' ', 'Ġ')
            right_str = right.decode("utf-8", errors="replace").replace(' ', 'Ġ')
            f.write(f"{left_str} {right_str}\n")
    
    print(f"✓ Saved {len(merges)} merges to {args.merges_output}")


if __name__ == "__main__":
    main()

