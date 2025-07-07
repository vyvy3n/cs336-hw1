"""
Script to train BPE tokenizer on OpenWebText dataset.
"""

import json
import pickle
import time
import tracemalloc
import cProfile
import pstats
import traceback
from pathlib import Path
from typing import Any

from cs336_basics.bpe_tokenizer import train_bpe


def train_bpe_on_openwebtext() -> dict[str, Any]:
    """
    Train a BPE tokenizer on OpenWebText dataset with vocab_size=32,000.

    Returns:
        Dictionary containing training results and statistics
    """
    input_path = "data/owt_train.txt"
    vocab_size = 32_000
    special_tokens = ["<|endoftext|>"]

    print("Training BPE tokenizer on OpenWebText dataset...")
    print(f"Input path: {input_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print()

    if not Path(input_path).exists():
        raise FileNotFoundError(f"OpenWebText dataset not found at {input_path}. Please run the download script first.")

    file_size_gb = Path(input_path).stat().st_size / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")

    tracemalloc.start()
    start_time = time.time()

    print("Starting BPE training...")
    vocab, merges = train_bpe(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)

    end_time = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    training_time_hours = (end_time - start_time) / 3600
    training_time_minutes = (end_time - start_time) / 60
    peak_memory_mb = peak / 1024 / 1024
    peak_memory_gb = peak / 1024 / 1024 / 1024

    longest_token = max(vocab.values(), key=len)
    longest_token_length = len(longest_token)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    vocab_path = output_dir / "openwebtext_vocab.json"
    merges_path = output_dir / "openwebtext_merges.pkl"

    serializable_vocab = {str(k): v.hex() for k, v in vocab.items()}

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=2)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    results = {
        "training_time_hours": training_time_hours,
        "training_time_minutes": training_time_minutes,
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_gb": peak_memory_gb,
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
    print(f"Peak memory usage: {peak_memory_mb:.2f} MB ({peak_memory_gb:.2f} GB)")
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


def compare_with_tinystories() -> None:
    """
    Compare OpenWebText tokenizer with TinyStories tokenizer if available.
    """
    print("\n" + "=" * 60)
    print("COMPARISON WITH TINYSTORIES TOKENIZER")
    print("=" * 60)

    ts_vocab_path = Path("output/tinystories_vocab.json")
    ts_merges_path = Path("output/tinystories_merges.pkl")
    owt_vocab_path = Path("output/openwebtext_vocab.json")
    owt_merges_path = Path("output/openwebtext_merges.pkl")

    if not (ts_vocab_path.exists() and ts_merges_path.exists()):
        print("TinyStories tokenizer not found. Please train it first using train_bpe_tinystories.py")
        return

    if not (owt_vocab_path.exists() and owt_merges_path.exists()):
        print("OpenWebText tokenizer not found. Please train it first.")
        return

    with open(ts_vocab_path, "r") as f:
        ts_vocab_serialized = json.load(f)
    ts_vocab = {int(k): bytes.fromhex(v) for k, v in ts_vocab_serialized.items()}

    with open(ts_merges_path, "rb") as f:
        ts_merges = pickle.load(f)

    with open(owt_vocab_path, "r") as f:
        owt_vocab_serialized = json.load(f)
    owt_vocab = {int(k): bytes.fromhex(v) for k, v in owt_vocab_serialized.items()}

    with open(owt_merges_path, "rb") as f:
        owt_merges = pickle.load(f)

    ts_longest = max(ts_vocab.values(), key=len)
    owt_longest = max(owt_vocab.values(), key=len)

    print(f"TinyStories vocab size: {len(ts_vocab):,}")
    print(f"OpenWebText vocab size: {len(owt_vocab):,}")
    print(f"TinyStories merges: {len(ts_merges):,}")
    print(f"OpenWebText merges: {len(owt_merges):,}")
    print()
    print(f"TinyStories longest token: {len(ts_longest)} bytes - '{ts_longest.decode('utf-8', errors='replace')}'")
    print(f"OpenWebText longest token: {len(owt_longest)} bytes - '{owt_longest.decode('utf-8', errors='replace')}'")
    print()

    ts_tokens = {token for token in ts_vocab.values() if len(token) > 1 and b"<|" not in token}
    owt_tokens = {token for token in owt_vocab.values() if len(token) > 1 and b"<|" not in token}

    common_tokens = ts_tokens.intersection(owt_tokens)
    ts_only = ts_tokens - owt_tokens
    owt_only = owt_tokens - ts_tokens

    print(f"Common multi-byte tokens: {len(common_tokens):,}")
    print(f"TinyStories-only tokens: {len(ts_only):,}")
    print(f"OpenWebText-only tokens: {len(owt_only):,}")

    print("\nSample TinyStories-specific tokens:")
    for _, token in enumerate(sorted(ts_only)[:10]):
        decoded = token.decode("utf-8", errors="replace")
        print(f"  '{decoded}'")

    print("\nSample OpenWebText-specific tokens:")
    for i, token in enumerate(sorted(owt_only)[:10]):
        decoded = token.decode("utf-8", errors="replace")
        print(f"  '{decoded}'")


def profile_bpe_training() -> None:
    """
    Profile the BPE training process to identify bottlenecks.
    """
    print("\n" + "=" * 60)
    print("PROFILING BPE TRAINING")
    print("=" * 60)

    sample_path = "data/owt_valid.txt"
    if Path(sample_path).exists():
        print(f"Profiling with validation file: {sample_path}")

        profiler = cProfile.Profile()
        profiler.enable()

        _, _ = train_bpe(input_path=sample_path, vocab_size=2000, special_tokens=["<|endoftext|>"])

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
        print(f"Validation file not found at {sample_path}, skipping detailed profiling")


def main() -> None:
    """Main function to run BPE training on OpenWebText."""
    try:
        results = train_bpe_on_openwebtext()

        compare_with_tinystories()
        profile_bpe_training()

        print("\n" + "=" * 60)
        print("ANSWERS TO PROBLEM QUESTIONS")
        print("=" * 60)

        print(
            f"(a) Training took {results['training_time_hours']:.4f} hours and used "
            f"{results['peak_memory_gb']:.2f} GB of memory."
        )

        print(f"    The longest token in the vocabulary is {results['longest_token_length']} bytes long:")
        print(f"    '{results['longest_token_decoded']}'")

        longest_decoded = results["longest_token_decoded"]
        if any(char.isalpha() for char in longest_decoded):
            if len(longest_decoded.split()) > 1:
                print("    This token makes sense as it appears to contain meaningful multi-word text,")
                print("    which is common in web text where phrases and URLs occur frequently.")
            else:
                print("    This token makes sense as it appears to contain a meaningful word.")
        elif any(char.isdigit() for char in longest_decoded):
            print("    This token appears to contain numbers, which is reasonable for web text")
            print("    that contains dates, IDs, or other numeric content.")
        else:
            print("    This token might be punctuation, whitespace, or special characters")
            print("    which can be common in web text with varied formatting.")

        print("\n(b) OpenWebText tokenizer likely produces longer, more diverse tokens than TinyStories")
        print("    due to the varied web content including technical terms, URLs, and complex vocabulary,")
        print("    while TinyStories focuses on simple children's story language.")

    except Exception as e:
        print(f"Error during BPE training: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
