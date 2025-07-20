"""
Script to train BPE tokenizer on TinyStories dataset.
"""

import cProfile
import json
import pickle
import pstats
import time
import traceback
import tracemalloc
from pathlib import Path
from typing import Any

from cs336_basics.tokenization.bpe import train_bpe


def train_bpe_on_tinystories() -> dict[str, Any]:
    """
    Train a BPE tokenizer on TinyStories dataset with vocab_size=10,000.

    Returns:
        Dictionary containing training results and statistics
    """
    # Use pathlib.Path relative to current working directory (project root)
    project_root = Path.cwd()
    input_path = project_root / "data" / "TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    print("Training BPE tokenizer on TinyStories dataset...")
    print(f"Input path: {input_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print()

    if not input_path.exists():
        raise FileNotFoundError(f"TinyStories dataset not found at {input_path}. Please run download script first.")

    tracemalloc.start()
    start_time = time.time()

    print("Starting BPE training...")
    vocab, merges = train_bpe(input_path=str(input_path), vocab_size=vocab_size, special_tokens=special_tokens)

    end_time = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    training_time_hours = (end_time - start_time) / 3600
    training_time_minutes = (end_time - start_time) / 60
    peak_memory_mb = peak / 1024 / 1024

    longest_token = max(vocab.values(), key=len)
    longest_token_length = len(longest_token)

    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    vocab_path = output_dir / "tinystories_vocab.json"
    merges_path = output_dir / "tinystories_merges.pkl"

    serializable_vocab = {str(k): v.hex() for k, v in vocab.items()}

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=2)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    results = {
        "training_time_hours": training_time_hours,
        "training_time_minutes": training_time_minutes,
        "peak_memory_mb": peak_memory_mb,
        "vocab_size_actual": len(vocab),
        "num_merges": len(merges),
        "longest_token": longest_token,
        "longest_token_length": longest_token_length,
        "longest_token_decoded": longest_token.decode("utf-8", errors="replace"),
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
    }

    print("\n" + "=" * 60)
    print("BPE TRAINING RESULTS")
    print("=" * 60)
    print(f"Training time: {training_time_hours:.4f} hours ({training_time_minutes:.2f} minutes)")
    print(f"Peak memory usage: {peak_memory_mb:.2f} MB")
    print(f"Actual vocabulary size: {len(vocab):,}")
    print(f"Number of merges: {len(merges):,}")
    print()
    print(f"Longest token length: {longest_token_length} bytes")
    print(f"Longest token (hex): {longest_token.hex()}")
    print(f"Longest token (decoded): '{longest_token.decode('utf-8', errors='replace')}'")
    print()
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"Merges saved to: {merges_path}")
    print("=" * 60)

    return results


def profile_bpe_training() -> None:
    """
    Profile the BPE training process to identify bottlenecks.
    """
    print("\n" + "=" * 60)
    print("PROFILING BPE TRAINING")
    print("=" * 60)

    project_root = Path.cwd()
    sample_path = project_root / "tests" / "fixtures" / "tinystories_sample_5M.txt"
    if sample_path.exists():
        print(f"Profiling with sample file: {sample_path}")

        profiler = cProfile.Profile()
        profiler.enable()

        _, _ = train_bpe(input_path=str(sample_path), vocab_size=1000, special_tokens=["<|endoftext|>"])

        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats("tottime")

        print("\nTop 10 functions by total time:")
        stats.print_stats(10)

        print("\nFunctions with '_pretokenize' in name:")
        stats.print_stats(".*_pretokenize.*")

        print("\nFunctions with '_merge' in name:")
        stats.print_stats(".*_merge.*")

        print("\nFunctions with '_get_stats' in name:")
        stats.print_stats(".*_get_stats.*")

    else:
        print(f"Sample file not found at {sample_path}, skipping detailed profiling")


def main() -> None:
    """Main function to run BPE training on TinyStories."""
    try:
        results = train_bpe_on_tinystories()

        profile_bpe_training()

        print("\n" + "=" * 60)
        print("ANSWERS TO PROBLEM QUESTIONS")
        print("=" * 60)

        print(
            f"(a) Training took {results['training_time_hours']:.4f} hours and used "
            f"{results['peak_memory_mb']:.2f} MB of memory."
        )

        print(f"    The longest token in the vocabulary is {results['longest_token_length']} bytes long:")
        print(f"    '{results['longest_token_decoded']}'")

        longest_decoded = results["longest_token_decoded"]
        if any(char.isalpha() for char in longest_decoded):
            print("    This token makes sense as it appears to contain meaningful text.")
        else:
            print("    This token might be punctuation, whitespace, or special characters.")

        print("\n(b) Run with profiling enabled to see the most time-consuming parts.")
        print("    Based on the implementation, pre-tokenization is likely the main bottleneck")
        print("    for large files, while merge computation dominates for smaller files with many merges.")

    except Exception as e:
        print(f"Error during BPE training: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
