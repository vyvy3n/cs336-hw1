"""
Tokenizer Experiments
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np

from cs336_basics.bpe_tokenizer import Tokenizer


def sample_documents_from_file(
    file_path: str | Path, num_docs: int = 10, delimiter: str = "<|endoftext|>", min_doc_length: int = 100
) -> list[str]:
    """
    Sample random documents from a text file that uses delimiter tokens.

    Args:
        file_path: Path to the input file
        num_docs: Number of documents to sample
        delimiter: Document delimiter token
        min_doc_length: Minimum document length in characters

    Returns:
        List of sampled document strings
    """
    print(f"Sampling {num_docs} documents from {file_path}...")

    documents: list[str] = []
    current_doc = ""

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            current_doc += line
            if delimiter in line:
                parts = current_doc.split(delimiter)
                for i, part in enumerate(parts[:-1]):
                    doc = part.strip()
                    if len(doc) >= min_doc_length:
                        documents.append(doc)

                current_doc = parts[-1]

    if current_doc.strip() and len(current_doc.strip()) >= min_doc_length:
        documents.append(current_doc.strip())

    print(f"Found {len(documents)} documents meeting minimum length requirement")

    if len(documents) >= num_docs:
        sampled = random.sample(documents, num_docs)
    else:
        print(f"Warning: Only {len(documents)} documents available, using all")
        sampled = documents

    return sampled


def calculate_compression_ratio(text: str, token_ids: list[int]) -> float:
    """
    Calculate compression ratio as bytes per token.

    Args:
        text: Original text string
        token_ids: Encoded token IDs

    Returns:
        Compression ratio (bytes per token)
    """
    text_bytes = len(text.encode("utf-8"))
    num_tokens = len(token_ids)

    if num_tokens == 0:
        return 0.0

    return text_bytes / num_tokens


def measure_tokenizer_throughput(tokenizer: Tokenizer, test_text: str, num_iterations: int = 5) -> tuple[float, float]:
    """
    Measure tokenizer throughput in bytes per second.

    Args:
        tokenizer: The tokenizer to benchmark
        test_text: Text to use for benchmarking
        num_iterations: Number of iterations for timing

    Returns:
        Tuple of (bytes_per_second, tokens_per_second)
    """
    text_size_bytes = len(test_text.encode("utf-8"))

    token_ids = tokenizer.encode(test_text)
    num_tokens = len(token_ids)

    times: list[float] = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        tokenizer.encode(test_text)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    bytes_per_second = text_size_bytes / avg_time
    tokens_per_second = num_tokens / avg_time

    return bytes_per_second, tokens_per_second


def encode_dataset_to_numpy(
    tokenizer: Tokenizer,
    file_path: str | Path,
    output_path: str | Path,
    chunk_size: int = 8 * 1024 * 1024,  # 8MB chunks
    max_vocab_size: int = 65536,  # uint16 max value
) -> None:
    """
    Encode a large dataset to numpy array with memory-efficient streaming.

    Args:
        tokenizer: The tokenizer to use
        file_path: Path to input text file
        output_path: Path to save encoded numpy array
        chunk_size: Size of chunks to process at once (bytes)
        max_vocab_size: Maximum vocabulary size (for uint16 validation)
    """
    print(f"Encoding dataset {file_path} to {output_path}...")

    all_token_ids: list[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        token_count = 0
        for token_id in tokenizer.encode_iterable(f):
            all_token_ids.append(token_id)
            token_count += 1

            if token_count % 1_000_000 == 0:
                print(f"  Processed {token_count:,} tokens...")

    print(f"Total tokens: {len(all_token_ids):,}")

    max_token_id = max(all_token_ids)
    if max_token_id >= max_vocab_size:
        raise ValueError(f"Token ID {max_token_id} exceeds uint16 range [0, {max_vocab_size - 1}]")

    token_array = np.array(all_token_ids, dtype=np.uint16)

    np.save(output_path, token_array)

    print(f"Saved {len(token_array):,} tokens to {output_path}")
    print(f"Array shape: {token_array.shape}")
    print(f"Array dtype: {token_array.dtype}")
    print(f"File size: {os.path.getsize(str(output_path) + '.npy') / (1024 * 1024):.1f} MB")


def run_experiments() -> None:
    """Run all tokenizer experiments and print results."""

    print("=" * 80)
    print("TOKENIZER EXPERIMENTS")
    print("=" * 80)

    random.seed(42)
    np.random.seed(42)

    print("\n1. Loading tokenizers...")

    ts_tokenizer = Tokenizer.from_files(
        vocab_filepath="../output/tinystories_vocab.json",
        merges_filepath="../output/tinystories_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    print(f"  TinyStories tokenizer loaded (vocab size: {len(ts_tokenizer.vocab)})")

    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath="../output/openwebtext_vocab.json",
        merges_filepath="../output/openwebtext_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    print(f"  OpenWebText tokenizer loaded (vocab size: {len(owt_tokenizer.vocab)})")

    print("\n2. Part (a): Sampling documents and calculating compression ratios...")

    ts_docs = sample_documents_from_file("../data/TinyStoriesV2-GPT4-train.txt", num_docs=10)

    owt_docs = sample_documents_from_file("../data/owt_train.txt", num_docs=10)

    ts_ratios: list[float] = []
    total_ts_chars = 0
    total_ts_tokens = 0

    print("\n  TinyStories documents with TinyStories tokenizer:")
    for i, doc in enumerate(ts_docs):
        tokens = ts_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, tokens)
        ts_ratios.append(ratio)
        total_ts_chars += len(doc.encode("utf-8"))
        total_ts_tokens += len(tokens)
        print(f"    Doc {i + 1:2d}: {len(doc):5d} chars -> {len(tokens):4d} tokens = {ratio:.2f} bytes/token")

    avg_ts_ratio = sum(ts_ratios) / len(ts_ratios)
    overall_ts_ratio = total_ts_chars / total_ts_tokens
    print(f"    Average: {avg_ts_ratio:.2f} bytes/token")
    print(f"    Overall: {overall_ts_ratio:.2f} bytes/token")

    owt_ratios: list[float] = []
    total_owt_chars = 0
    total_owt_tokens = 0

    print("\n  OpenWebText documents with OpenWebText tokenizer:")
    for i, doc in enumerate(owt_docs):
        tokens = owt_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, tokens)
        owt_ratios.append(ratio)
        total_owt_chars += len(doc.encode("utf-8"))
        total_owt_tokens += len(tokens)
        print(f"    Doc {i + 1:2d}: {len(doc):5d} chars -> {len(tokens):4d} tokens = {ratio:.2f} bytes/token")

    avg_owt_ratio = sum(owt_ratios) / len(owt_ratios)
    overall_owt_ratio = total_owt_chars / total_owt_tokens
    print(f"    Average: {avg_owt_ratio:.2f} bytes/token")
    print(f"    Overall: {overall_owt_ratio:.2f} bytes/token")

    print("\n3. Part (b): Cross-tokenizer comparison...")
    print("  OpenWebText documents with TinyStories tokenizer:")

    cross_ratios: list[float] = []
    total_cross_chars = 0
    total_cross_tokens = 0

    for i, doc in enumerate(owt_docs):
        tokens = ts_tokenizer.encode(doc)
        ratio = calculate_compression_ratio(doc, tokens)
        cross_ratios.append(ratio)
        total_cross_chars += len(doc.encode("utf-8"))
        total_cross_tokens += len(tokens)
        print(f"    Doc {i + 1:2d}: {len(doc):5d} chars -> {len(tokens):4d} tokens = {ratio:.2f} bytes/token")

    avg_cross_ratio = sum(cross_ratios) / len(cross_ratios)
    overall_cross_ratio = total_cross_chars / total_cross_tokens
    print(f"    Average: {avg_cross_ratio:.2f} bytes/token")
    print(f"    Overall: {overall_cross_ratio:.2f} bytes/token")

    print(f"\n  Comparison:")
    print(f"    OpenWebText with OpenWebText tokenizer: {overall_owt_ratio:.2f} bytes/token")
    print(f"    OpenWebText with TinyStories tokenizer: {overall_cross_ratio:.2f} bytes/token")
    print(f"    Degradation: {(overall_cross_ratio - overall_owt_ratio):.2f} bytes/token")
    print(f"    Relative increase: {(overall_cross_ratio / overall_owt_ratio - 1) * 100:.1f}%")

    print("\n4. Part (c): Measuring tokenizer throughput...")

    test_text = "\n".join(owt_docs[:5])
    print(f"  Test text size: {len(test_text.encode('utf-8')):,} bytes")

    ts_bytes_per_sec, ts_tokens_per_sec = measure_tokenizer_throughput(ts_tokenizer, test_text)
    print(f"  TinyStories tokenizer:")
    print(f"    Throughput: {ts_bytes_per_sec:,.0f} bytes/second ({ts_bytes_per_sec / (1024 * 1024):.1f} MB/s)")
    print(f"    Throughput: {ts_tokens_per_sec:,.0f} tokens/second")

    owt_bytes_per_sec, owt_tokens_per_sec = measure_tokenizer_throughput(owt_tokenizer, test_text)
    print(f"  OpenWebText tokenizer:")
    print(f"    Throughput: {owt_bytes_per_sec:,.0f} bytes/second ({owt_bytes_per_sec / (1024 * 1024):.1f} MB/s)")
    print(f"    Throughput: {owt_tokens_per_sec:,.0f} tokens/second")

    pile_size_bytes = 825 * 1024 * 1024 * 1024  # 825 GB in bytes

    ts_pile_time_hours = pile_size_bytes / ts_bytes_per_sec / 3600
    owt_pile_time_hours = pile_size_bytes / owt_bytes_per_sec / 3600

    print(f"  Time to tokenize Pile dataset (825GB):")
    print(f"    TinyStories tokenizer: {ts_pile_time_hours:.1f} hours ({ts_pile_time_hours / 24:.1f} days)")
    print(f"    OpenWebText tokenizer: {owt_pile_time_hours:.1f} hours ({owt_pile_time_hours / 24:.1f} days)")

    print("\n5. Part (d): Encoding full datasets...")

    os.makedirs("../data/encoded", exist_ok=True)

    print("  Why uint16 is appropriate:")
    print(f"    TinyStories vocab size: {len(ts_tokenizer.vocab)} (fits in uint16: 0-65535)")
    print(f"    OpenWebText vocab size: {len(owt_tokenizer.vocab)} (fits in uint16: 0-65535)")
    print(f"    uint16 saves 50% memory vs uint32, enables larger datasets in memory")
    print(f"    All modern tokenizers have vocab sizes < 65536, making uint16 sufficient")

    print("\n  Encoding TinyStories training dataset...")
    try:
        encode_dataset_to_numpy(
            ts_tokenizer, "../data/TinyStoriesV2-GPT4-train.txt", "../data/encoded/tinystories_train_tokens"
        )
    except Exception as e:
        print(f"    Error encoding TinyStories: {e}")

    print("\n  Encoding OpenWebText training dataset...")
    try:
        encode_dataset_to_numpy(owt_tokenizer, "../data/owt_train.txt", "../data/encoded/owt_train_tokens")
    except Exception as e:
        print(f"    Error encoding OpenWebText training: {e}")

    print("\n  Encoding OpenWebText validation dataset...")
    try:
        encode_dataset_to_numpy(owt_tokenizer, "../data/owt_valid.txt", "../data/encoded/owt_valid_tokens")
    except Exception as e:
        print(f"    Error encoding OpenWebText validation: {e}")

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED")
    print("=" * 80)

    print(f"\nSUMMARY:")
    print(f"  Compression ratios (bytes/token):")
    print(f"    TinyStories with TinyStories tokenizer: {overall_ts_ratio:.2f}")
    print(f"    OpenWebText with OpenWebText tokenizer: {overall_owt_ratio:.2f}")
    print(f"    OpenWebText with TinyStories tokenizer: {overall_cross_ratio:.2f}")
    print(f"  Cross-domain degradation: {(overall_cross_ratio / overall_owt_ratio - 1) * 100:.1f}%")
    print(f"  Tokenizer throughput: {owt_bytes_per_sec / (1024 * 1024):.1f} MB/s (OpenWebText)")


if __name__ == "__main__":
    run_experiments()
