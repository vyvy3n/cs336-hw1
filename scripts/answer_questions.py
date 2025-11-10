#!/usr/bin/env python3
"""
Answer all BPE and tokenizer-related questions from CS336 Assignment 1.

This script provides a unified interface to answer questions about:
- BPE training (time, memory, longest token)
- Tokenizer compression ratios
- Cross-domain tokenization
- Throughput estimation

Usage:
    # Answer all questions at once
    python scripts/answer_questions.py all \
        --tinystories-tokenizer artifacts/tinystories_bpe.yaml \
        --owt-tokenizer artifacts/owt_bpe.yaml \
        --tinystories-data data/TinyStoriesV2-GPT4-train.txt \
        --owt-data data/owt_train.txt

    # Answer specific question
    python scripts/answer_questions.py compression-ratio \
        --tinystories-tokenizer artifacts/tinystories_bpe.yaml \
        --owt-tokenizer artifacts/owt_bpe.yaml \
        --tinystories-data data/TinyStoriesV2-GPT4-train.txt \
        --owt-data data/owt_train.txt
"""

import argparse
from scripts.analyze_tokenizer import (
    sample_documents,
    calculate_compression_ratio,
    Tokenizer
)
from scripts.encode_dataset import load_tokenizer_yaml


def answer_compression_ratio(args):
    """
    Answer: What is each tokenizer's compression ratio (bytes/token)?
    
    Problem 2.7(a): Sample 10 documents from TinyStories and OpenWebText.
    Using your previously-trained TinyStories and OpenWebText tokenizers
    (10K and 32K vocabulary size, respectively), encode these sampled
    documents into integer IDs. What is each tokenizer's compression ratio?
    """
    print("\n" + "="*80)
    print("QUESTION: Compression Ratios (Problem 2.7a)")
    print("="*80)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    ts_vocab, ts_merges = load_tokenizer_yaml(args.tinystories_tokenizer)
    owt_vocab, owt_merges = load_tokenizer_yaml(args.owt_tokenizer)
    
    ts_tokenizer = Tokenizer(ts_vocab, ts_merges, special_tokens=["<|endoftext|>"])
    owt_tokenizer = Tokenizer(owt_vocab, owt_merges, special_tokens=["<|endoftext|>"])
    
    # Sample documents
    print(f"Sampling {args.num_samples} documents from each dataset...")
    ts_docs = sample_documents(args.tinystories_data, num_samples=args.num_samples)
    owt_docs = sample_documents(args.owt_data, num_samples=args.num_samples)
    
    # Calculate compression ratios
    ts_on_ts_ratio, ts_valid, ts_total = calculate_compression_ratio(ts_docs, ts_tokenizer)
    owt_on_owt_ratio, owt_valid, owt_total = calculate_compression_ratio(owt_docs, owt_tokenizer)
    
    print("\nANSWER:")
    print(f"  TinyStories tokenizer (10K vocab) on TinyStories docs:")
    print(f"    {ts_on_ts_ratio:.3f} bytes/token (used {ts_valid}/{ts_total} documents)")
    print(f"  OpenWebText tokenizer (32K vocab) on OpenWebText docs:")
    print(f"    {owt_on_owt_ratio:.3f} bytes/token (used {owt_valid}/{owt_total} documents)")


def answer_cross_domain(args):
    """
    Answer: What happens if you tokenize OpenWebText with TinyStories tokenizer?
    
    Problem 2.7(b): What happens if you tokenize your OpenWebText sample
    with the TinyStories tokenizer? Compare the compression ratio and/or
    qualitatively describe what happens.
    """
    print("\n" + "="*80)
    print("QUESTION: Cross-Domain Tokenization (Problem 2.7b)")
    print("="*80)
    
    # Load tokenizers
    ts_vocab, ts_merges = load_tokenizer_yaml(args.tinystories_tokenizer)
    owt_vocab, owt_merges = load_tokenizer_yaml(args.owt_tokenizer)
    
    ts_tokenizer = Tokenizer(ts_vocab, ts_merges, special_tokens=["<|endoftext|>"])
    owt_tokenizer = Tokenizer(owt_vocab, owt_merges, special_tokens=["<|endoftext|>"])
    
    # Sample documents
    ts_docs = sample_documents(args.tinystories_data, num_samples=args.num_samples)
    owt_docs = sample_documents(args.owt_data, num_samples=args.num_samples)
    
    # Calculate cross-domain ratios
    ts_on_owt_ratio, ts_owt_valid, ts_owt_total = calculate_compression_ratio(owt_docs, ts_tokenizer)
    owt_on_ts_ratio, owt_ts_valid, owt_ts_total = calculate_compression_ratio(ts_docs, owt_tokenizer)
    
    # Also get in-domain for comparison
    ts_on_ts_ratio, _, _ = calculate_compression_ratio(ts_docs, ts_tokenizer)
    owt_on_owt_ratio, _, _ = calculate_compression_ratio(owt_docs, owt_tokenizer)
    
    print("\nANSWER:")
    print(f"  TinyStories tokenizer on OpenWebText docs:")
    print(f"    {ts_on_owt_ratio:.3f} bytes/token (used {ts_owt_valid}/{ts_owt_total} documents)")
    print(f"    vs in-domain (OWT tokenizer on OWT): {owt_on_owt_ratio:.3f} bytes/token")
    print(f"    Efficiency loss: {(1 - ts_on_owt_ratio/owt_on_owt_ratio)*100:.1f}%")
    print(f"\n  OpenWebText tokenizer on TinyStories docs:")
    print(f"    {owt_on_ts_ratio:.3f} bytes/token (used {owt_ts_valid}/{owt_ts_total} documents)")
    print(f"    vs in-domain (TS tokenizer on TS): {ts_on_ts_ratio:.3f} bytes/token")
    print(f"    Efficiency change: {(owt_on_ts_ratio/ts_on_ts_ratio - 1)*100:+.1f}%")
    
    print("\n  INTERPRETATION:")
    print("    - TinyStories tokenizer on OpenWebText: LESS efficient (lower bytes/token)")
    print("      because TinyStories vocab doesn't have common OpenWebText tokens")
    print("    - OpenWebText tokenizer on TinyStories: Similar efficiency")
    print("      because larger vocab (32K) covers TinyStories vocabulary well")


def answer_throughput(args):
    """
    Answer: Estimate throughput and time to tokenize Pile dataset.
    
    Problem 2.7(c): Estimate the throughput of your tokenizer (e.g., in
    bytes/second). How long would it take to tokenize the Pile dataset (825GB)?
    """
    print("\n" + "="*80)
    print("QUESTION: Tokenizer Throughput (Problem 2.7c)")
    print("="*80)
    
    # Use provided encoding statistics
    ts_bytes_per_token = 4.0
    owt_bytes_per_token = 4.0
    
    ts_throughput = (args.ts_train_tokens * ts_bytes_per_token) / args.ts_train_time
    owt_throughput = (args.owt_train_tokens * owt_bytes_per_token) / args.owt_train_time
    avg_throughput = (ts_throughput + owt_throughput) / 2
    
    pile_bytes = args.pile_size_gb * 1024**3
    pile_time_hours = pile_bytes / avg_throughput / 3600
    
    print("\nANSWER:")
    print(f"  TinyStories throughput: {ts_throughput/(1024*1024):.1f} MB/s")
    print(f"    ({args.ts_train_tokens:,} tokens in {args.ts_train_time:.1f}s)")
    print(f"  OpenWebText throughput: {owt_throughput/(1024*1024):.1f} MB/s")
    print(f"    ({args.owt_train_tokens:,} tokens in {args.owt_train_time:.1f}s)")
    print(f"  Average throughput: {avg_throughput/(1024*1024):.1f} MB/s")
    print(f"\n  Time to tokenize Pile dataset ({args.pile_size_gb}GB): {pile_time_hours:.1f} hours")


def answer_uint16(args):
    """
    Answer: Why is uint16 an appropriate choice for token IDs?
    
    Problem 2.7(d): We recommend serializing the token IDs as a NumPy array
    of datatype uint16. Why is uint16 an appropriate choice?
    """
    print("\n" + "="*80)
    print("QUESTION: Why uint16 for Token IDs? (Problem 2.7d)")
    print("="*80)
    
    print("\nANSWER:")
    print("  uint16 is appropriate because:")
    print("    - Range: 0 to 65,535 (2^16 - 1)")
    print("    - TinyStories vocab: 10,000 tokens (fits comfortably)")
    print("    - OpenWebText vocab: 32,000 tokens (fits comfortably)")
    print("    - Memory efficient: 2 bytes per token vs 4 bytes (uint32) or 8 bytes (uint64)")
    print("    - For 2.85B tokens (TinyStories): saves 5.7GB vs uint32, 17.1GB vs uint64")
    print("\n  uint16 is sufficient for most practical tokenizers (vocab < 65K)")


def answer_all(args):
    """Answer all questions."""
    print("\n" + "="*80)
    print("CS336 ASSIGNMENT 1: BPE & TOKENIZER QUESTIONS")
    print("="*80)
    
    answer_compression_ratio(args)
    answer_cross_domain(args)
    answer_throughput(args)
    answer_uint16(args)
    
    print("\n" + "="*80)
    print("ALL QUESTIONS ANSWERED")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Answer BPE and tokenizer questions")
    subparsers = parser.add_subparsers(dest='command', help='Question to answer')
    
    # Common arguments for all subcommands
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument('--tinystories-tokenizer', type=str,
                            default='artifacts/tinystories_bpe.yaml',
                            help='Path to TinyStories tokenizer YAML')
    common_args.add_argument('--owt-tokenizer', type=str,
                            default='artifacts/owt_bpe.yaml',
                            help='Path to OpenWebText tokenizer YAML')
    common_args.add_argument('--tinystories-data', type=str,
                            default='data/TinyStoriesV2-GPT4-train.txt',
                            help='Path to TinyStories dataset')
    common_args.add_argument('--owt-data', type=str,
                            default='data/owt_train.txt',
                            help='Path to OpenWebText dataset')
    common_args.add_argument('--num-samples', type=int, default=10,
                            help='Number of documents to sample (default: 10)')
    common_args.add_argument('--ts-train-tokens', type=int, default=2850391059,
                            help='TinyStories train tokens (default: 2850391059)')
    common_args.add_argument('--ts-train-time', type=float, default=447.5,
                            help='TinyStories train time in seconds (default: 447.5)')
    common_args.add_argument('--owt-train-tokens', type=int, default=542447487,
                            help='OpenWebText train tokens (default: 542447487)')
    common_args.add_argument('--owt-train-time', type=float, default=76.9,
                            help='OpenWebText train time in seconds (default: 76.9)')
    common_args.add_argument('--pile-size-gb', type=int, default=825,
                            help='Pile dataset size in GB (default: 825)')
    
    # Subcommands
    subparsers.add_parser('all', parents=[common_args], help='Answer all questions')
    subparsers.add_parser('compression-ratio', parents=[common_args], help='Answer compression ratio question')
    subparsers.add_parser('cross-domain', parents=[common_args], help='Answer cross-domain question')
    subparsers.add_parser('throughput', parents=[common_args], help='Answer throughput question')
    subparsers.add_parser('uint16', parents=[common_args], help='Answer uint16 question')
    
    args = parser.parse_args()
    
    if args.command == 'all':
        answer_all(args)
    elif args.command == 'compression-ratio':
        answer_compression_ratio(args)
    elif args.command == 'cross-domain':
        answer_cross_domain(args)
    elif args.command == 'throughput':
        answer_throughput(args)
    elif args.command == 'uint16':
        answer_uint16(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
