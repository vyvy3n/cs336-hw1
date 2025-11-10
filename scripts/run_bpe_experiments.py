#!/usr/bin/env python3
"""
Run BPE training experiments with profiling and analysis.

This script trains BPE tokenizers and provides detailed profiling information
including time, memory usage, and performance bottlenecks.

Usage:
    # Train TinyStories tokenizer with profiling
    python scripts/run_bpe_experiments.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --output artifacts/tinystories_bpe.yaml \
        --vocab-size 10000 \
        --profile

    # Train OpenWebText tokenizer
    python scripts/run_bpe_experiments.py \
        --input data/owt_train.txt \
        --output artifacts/owt_bpe.yaml \
        --vocab-size 32000 \
        --num-processes 40
"""

import argparse
import os
import time
import tracemalloc
import resource
import cProfile
import pstats
import io
from cs336_basics.bpe import run_train_bpe
from scripts.train_bpe import save_tokenizer_yaml


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m ({seconds:.1f}s)"
    else:
        return f"{seconds/3600:.2f}h ({seconds/60:.1f}m)"


def format_memory(bytes_val: float) -> str:
    """Format bytes into human-readable string."""
    if bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/(1024**2):.1f} MB"
    else:
        return f"{bytes_val/(1024**3):.2f} GB"


def train_with_profiling(input_file: str, vocab_size: int, num_processes: int, 
                        enable_profiling: bool = False):
    """
    Train BPE tokenizer with optional profiling.
    
    Args:
        input_file: Path to input text file
        vocab_size: Target vocabulary size
        num_processes: Number of parallel processes
        enable_profiling: Whether to enable cProfile profiling
        
    Returns:
        Tuple of (vocab, merges, elapsed_time, peak_memory, profiling_stats)
    """
    print("="*80)
    print("BPE TRAINING")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Parallel processes: {num_processes}")
    print(f"Profiling: {'enabled' if enable_profiling else 'disabled'}")
    print()
    
    # Start profiling
    profiler = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Train tokenizer
    vocab, merges = run_train_bpe(
        input_file,
        vocab_size,
        ["<|endoftext|>"],
        num_processes
    )
    
    # Stop profiling
    elapsed_time = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    if enable_profiling:
        profiler.disable()
    
    # Get RSS memory
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Convert to bytes
    
    # Print results
    print("\n" + "="*80)
    print("TRAINING RESULTS")
    print("="*80)
    print(f"Time: {format_time(elapsed_time)}")
    print(f"Peak Python heap: {format_memory(peak_mem)}")
    print(f"Peak RSS: {format_memory(peak_rss)}")
    print(f"Vocabulary size: {len(vocab)} tokens")
    print(f"Number of merges: {len(merges)}")
    
    # Find longest token
    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    longest_text = longest_bytes.decode('utf-8', 'replace')
    print(f"\nLongest token:")
    print(f"  ID: {longest_id}")
    print(f"  Length: {len(longest_bytes)} bytes")
    print(f"  Text: {longest_text!r}")
    
    # Print profiling stats if enabled
    if enable_profiling and profiler:
        print("\n" + "="*80)
        print("PROFILING RESULTS (Top 30 functions by cumulative time)")
        print("="*80)
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumtime')
        ps.print_stats(30)
        print(s.getvalue())
    
    return vocab, merges, elapsed_time, peak_mem, profiler


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer with profiling")
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
        "--vocab-size",
        type=int,
        required=True,
        help="Target vocabulary size"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: CPU count)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed profiling with cProfile"
    )
    
    args = parser.parse_args()
    
    # Determine number of processes
    if args.num_processes is None:
        args.num_processes = os.cpu_count()
    
    # Train tokenizer
    vocab, merges, elapsed_time, peak_mem, profiler = train_with_profiling(
        args.input,
        args.vocab_size,
        args.num_processes,
        args.profile
    )
    
    # Save tokenizer
    print("\n" + "="*80)
    print("SAVING TOKENIZER")
    print("="*80)
    save_tokenizer_yaml(vocab, merges, args.output)
    print(f"✓ Tokenizer saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Training time: {format_time(elapsed_time)}")
    print(f"Peak memory: {format_memory(peak_mem)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Output file: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()

