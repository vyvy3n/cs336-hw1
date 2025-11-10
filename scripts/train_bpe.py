#!/usr/bin/env python3
"""
Train a BPE tokenizer on a dataset.

This script trains a byte-level BPE tokenizer using the custom implementation
from cs336_basics.bpe and saves it in YAML format for inspection.

Usage:
    # Train on TinyStories
    uv run python scripts/train_bpe.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --output artifacts/tinystories_bpe.yaml \
        --vocab_size 10000

    # Train on OpenWebText    
    uv run python scripts/train_bpe.py \
        --input data/owt_train.txt \
        --output artifacts/owt_bpe.yaml \
        --vocab_size 32000
"""

import argparse
import os
import time
import yaml
from cs336_basics.bpe import run_train_bpe


def save_tokenizer_yaml(vocab, merges, fname):
    """
    Save vocab and merges to a YAML file with UTF-8 decoding for readability.
    
    Args:
        vocab: Dictionary mapping token IDs to bytes
        merges: List of (bytes, bytes) merge pairs
        fname: Output file path
    """
    # Convert bytes → string for readability
    vocab_serializable = {
        k: v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
        for k, v in vocab.items()
    }
    merges_serializable = [
        (a.decode("utf-8", errors="replace"), b.decode("utf-8", errors="replace"))
        for a, b in merges
    ]

    # Ensure the parent directory exists before writing
    dirpath = os.path.dirname(fname)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump(
            {"vocab": vocab_serializable, "merges": merges_serializable},
            f,
            allow_unicode=True,
            sort_keys=False
        )
    print(f"✓ Tokenizer saved to: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text file (documents separated by <|endoftext|>)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output YAML file path"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Target vocabulary size"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: CPU count)"
    )
    
    args = parser.parse_args()
    
    # Determine number of processes
    if args.num_processes is None:
        args.num_processes = os.cpu_count()
    
    print("="*80)
    print(f"Training BPE tokenizer")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Parallel processes: {args.num_processes}")
    print()
    
    # Train tokenizer
    start = time.perf_counter()
    vocab, merges = run_train_bpe(
        args.input,
        args.vocab_size,
        ["<|endoftext|>"],
        args.num_processes
    )
    elapsed_s = time.perf_counter() - start
    
    print(f"\n✓ Training completed in {elapsed_s:.2f}s ({elapsed_s/60:.2f} minutes)")
    
    # Save tokenizer
    save_tokenizer_yaml(vocab, merges, args.output)
    
    # Find longest token
    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    longest_text = longest_bytes.decode('utf-8', 'replace')
    print(f"\nLongest token: id={longest_id}, len={len(longest_bytes)} bytes, text={longest_text!r}")
    
    print("\n" + "="*80)
    print("✓ BPE training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
