#!/usr/bin/env python3
"""
Prepare TinyStories dataset for training.

This script is a convenience wrapper that:
1. Trains a BPE tokenizer on TinyStories
2. Encodes the train and validation sets

It uses the custom BPE implementation from cs336_basics.bpe.

Usage:
    python scripts/prepare_tinystories.py --data_dir data --vocab_size 10000

Prerequisites:
    - data/TinyStoriesV2-GPT4-train.txt
    - data/TinyStoriesV2-GPT4-valid.txt

    Download from: https://huggingface.co/datasets/roneneldan/TinyStories
"""

import argparse
import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "="*80)
    print(description)
    print("="*80)
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n✗ {description} failed!")
        sys.exit(1)
    print(f"\n✓ {description} completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyStories dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing TinyStories text files"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Vocabulary size for BPE tokenizer"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: CPU count)"
    )

    args = parser.parse_args()

    # Check if input files exist
    train_file = os.path.join(args.data_dir, "TinyStoriesV2-GPT4-train.txt")
    valid_file = os.path.join(args.data_dir, "TinyStoriesV2-GPT4-valid.txt")

    if not os.path.exists(train_file):
        print(f"✗ Training file not found: {train_file}")
        print("\nPlease download TinyStories dataset from:")
        print("  https://huggingface.co/datasets/roneneldan/TinyStories")
        sys.exit(1)

    if not os.path.exists(valid_file):
        print(f"✗ Validation file not found: {valid_file}")
        print("\nPlease download TinyStories dataset from:")
        print("  https://huggingface.co/datasets/roneneldan/TinyStories")
        sys.exit(1)

    # Create artifacts directory
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Define output paths
    tokenizer_path = os.path.join(artifacts_dir, "tinystories_bpe.yaml")
    train_tokens_path = os.path.join(args.data_dir, "tinystories_train_tokens.npy")
    valid_tokens_path = os.path.join(args.data_dir, "tinystories_valid_tokens.npy")

    # Prepare num_processes argument
    num_proc_arg = ["--num_processes", str(args.num_processes)] if args.num_processes else []

    # Step 1: Train BPE tokenizer
    run_command(
        ["python", "scripts/train_bpe.py",
         "--input", train_file,
         "--output", tokenizer_path,
         "--vocab_size", str(args.vocab_size)] + num_proc_arg,
        "Step 1: Training BPE tokenizer"
    )

    # Step 2: Encode training set
    run_command(
        ["python", "scripts/encode_dataset.py",
         "--input", train_file,
         "--output", train_tokens_path,
         "--tokenizer", tokenizer_path] + num_proc_arg,
        "Step 2: Encoding training set"
    )

    # Step 3: Encode validation set
    run_command(
        ["python", "scripts/encode_dataset.py",
         "--input", valid_file,
         "--output", valid_tokens_path,
         "--tokenizer", tokenizer_path] + num_proc_arg,
        "Step 3: Encoding validation set"
    )

    print("\n" + "="*80)
    print("✓ TinyStories dataset preparation complete!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  - Tokenizer: {tokenizer_path}")
    print(f"  - Training tokens: {train_tokens_path}")
    print(f"  - Validation tokens: {valid_tokens_path}")
    print(f"\nYou can now train a model using:")
    print(f"  python scripts/train.py --dataset tinystories")


if __name__ == "__main__":
    main()
