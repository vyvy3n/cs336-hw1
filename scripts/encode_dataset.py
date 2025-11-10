#!/usr/bin/env python3
"""
Encode a dataset using a trained BPE tokenizer.

This script tokenizes a text dataset and saves the token IDs as a numpy array
for efficient loading during training.

Usage:
    # Encode TinyStories training set
    python scripts/encode_dataset.py \\
        --input data/TinyStoriesV2-GPT4-train.txt \\
        --output data/tinystories_train_tokens.npy \\
        --tokenizer artifacts/tinystories_bpe.yaml

    # Encode TinyStories validation set
    python scripts/encode_dataset.py \\
        --input data/TinyStoriesV2-GPT4-valid.txt \\
        --output data/tinystories_valid_tokens.npy \\
        --tokenizer artifacts/tinystories_bpe.yaml
"""

import argparse
import os
import time
import numpy as np
import yaml
from multiprocessing import Process, Queue

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization import find_chunk_boundaries


def load_tokenizer_yaml(fname):
    """
    Load vocab and merges from a YAML file, handling Python tuple tags safely.
    
    Guarantees the 0..255 byte symbols exist (so no KeyError on non-ASCII).
    Uses any recoverable ASCII higher-id tokens and merges.
    Skips unrecoverable tokens/merges that contain �, which were destroyed on save.
    
    Args:
        fname: Path to YAML file
        
    Returns:
        Tuple of (vocab dict, merges list)
    """
    # Custom constructor for Python tuples
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
        lambda l, n: tuple(l.construct_sequence(n)))
    
    d = yaml.safe_load(open(fname, "r", encoding="utf-8"))

    vocab = {i: bytes([i]) for i in range(256)}
    for k, v in d["vocab"].items():
        i = int(k)
        if i < 256:  # skip base bytes already added
            continue

        if isinstance(v, (list, tuple)):
            b = bytes(v)
        elif isinstance(v, str):
            b = v.encode("utf-8", "ignore")
            if b:  # skip empty bytes
                vocab[i] = b
    
    merges = []
    for a, b in d["merges"]:
        if isinstance(a, (list, tuple)): 
            merges.append((bytes(a), bytes(b)))
        else:
            merge_a, merge_b = a.encode("utf-8", "ignore"), b.encode("utf-8", "ignore")
            if merge_a and merge_b:
                merges.append((merge_a, merge_b))
    return vocab, merges


def encode_worker(start: int, end: int, input_path: str, tokenizer, q: Queue):
    """Worker function to encode a chunk of the file."""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # Encode chunk and put tokens in queue
        tokens = []
        for token_id in tokenizer.encode_iterable([chunk]):
            tokens.append(token_id)
        q.put(tokens)


def encode_dataset_parallel(input_path, output_path, tokenizer, num_chunks):
    """
    Encode dataset using chunking approach from pretokenization.py and
    parallel processing following run_train_bpe() pattern.
    
    Args:
        input_path: Input text file path
        output_path: Output numpy file path
        tokenizer: Tokenizer instance
        num_chunks: Number of parallel processes
    """
    print(f"Starting parallel encoding with {num_chunks} processes")

    # Create processes and queue (same pattern as run_train_bpe)
    processes = []
    q = Queue()
    
    with open(input_path, "rb") as f:
        # Use the same chunking logic as pretokenization.py
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        
        # Process each chunk separately
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            p = Process(target=encode_worker, args=(start, end, input_path, tokenizer, q))
            p.start()
            processes.append(p)
    
        # Collect and merge tokens from workers
        all_tokens = []
        for _ in range(len(processes)):
            all_tokens.extend(q.get())
    
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Convert to uint16 numpy array
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        np.save(output_path, tokens_array)
        print(f"✓ Saved {len(tokens_array):,} tokens to {output_path}")
        return tokens_array


def main():
    parser = argparse.ArgumentParser(description="Encode dataset with BPE tokenizer")
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
        help="Output numpy file path (.npy)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer YAML file"
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
    print(f"Encoding dataset")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Parallel processes: {args.num_processes}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    vocab, merges = load_tokenizer_yaml(args.tokenizer)
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    print(f"✓ Tokenizer loaded (vocab size: {len(vocab)})")
    
    # Encode dataset
    start = time.perf_counter()
    encode_dataset_parallel(args.input, args.output, tokenizer, args.num_processes)
    elapsed_s = time.perf_counter() - start
    
    print(f"\n✓ Encoding completed in {elapsed_s:.2f}s ({elapsed_s/60:.2f} minutes)")
    
    # Calculate throughput
    file_size_mb = os.path.getsize(args.input) / (1024 * 1024)
    throughput_mb_s = file_size_mb / elapsed_s
    print(f"Throughput: {throughput_mb_s:.1f} MB/s")
    
    print("\n" + "="*80)
    print("✓ Dataset encoding complete!")
    print("="*80)


if __name__ == "__main__":
    main()
