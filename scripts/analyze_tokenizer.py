#!/usr/bin/env python3
"""
Analyze tokenizer performance and compression ratios.

This script provides utilities to:
1. Calculate compression ratios (bytes/token) for different tokenizers
2. Estimate tokenizer throughput (bytes/second)
3. Sample documents from datasets for analysis

Usage:
    # Calculate compression ratios
    python scripts/analyze_tokenizer.py compression \
        --tinystories-tokenizer artifacts/tinystories_bpe.yaml \
        --owt-tokenizer artifacts/owt_bpe.yaml \
        --tinystories-data data/TinyStoriesV2-GPT4-train.txt \
        --owt-data data/owt_train.txt \
        --num-samples 10

    # Estimate throughput
    python scripts/analyze_tokenizer.py throughput \
        --ts-train-tokens 2850391059 \
        --ts-train-time 447.5 \
        --owt-train-tokens 542447487 \
        --owt-train-time 76.9 \
        --pile-size-gb 825
"""

import argparse
import random
from typing import List, Tuple
from cs336_basics.tokenizer import Tokenizer
from scripts.encode_dataset import load_tokenizer_yaml


def sample_documents(file_path: str, num_samples: int = 10, seed: int = 42) -> List[str]:
    """
    Sample documents from a dataset file separated by <|endoftext|>.
    
    Args:
        file_path: Path to dataset file
        num_samples: Number of documents to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled document strings
    """
    random.seed(seed)
    
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_doc = ""
        for line in f:
            if '<|endoftext|>' in line:
                parts = line.split('<|endoftext|>')
                current_doc += parts[0]
                if current_doc.strip():
                    documents.append(current_doc.strip())
                
                for part in parts[1:-1]:
                    if part.strip():
                        documents.append(part.strip())
                
                current_doc = parts[-1]
                
                # Stop if we have enough documents for sampling
                if len(documents) >= num_samples * 10:
                    break
            else:
                current_doc += line
    
    # Add final document if exists
    if current_doc.strip():
        documents.append(current_doc.strip())
    
    # Sample random documents
    sampled_docs = random.sample(documents, min(num_samples, len(documents)))
    return sampled_docs


def can_tokenize(text: str, tokenizer: Tokenizer) -> bool:
    """Check if a text can be tokenized without errors."""
    try:
        tokenizer.encode(text)
        return True
    except KeyError as e:
        print(f"  Warning: Document failed tokenization, missing byte: {e}")
        return False


def calculate_compression_ratio(documents: List[str], tokenizer: Tokenizer) -> Tuple[float, int, int]:
    """
    Calculate compression ratio (bytes/token) for a list of documents.
    
    Args:
        documents: List of document strings
        tokenizer: Tokenizer to use
        
    Returns:
        Tuple of (compression_ratio, valid_docs, total_docs)
    """
    total_bytes = 0
    total_tokens = 0
    valid_docs = 0
    
    for doc in documents:
        if not can_tokenize(doc, tokenizer):
            continue
            
        doc_bytes = len(doc.encode('utf-8'))
        total_bytes += doc_bytes
        
        tokens = tokenizer.encode(doc)
        total_tokens += len(tokens)
        valid_docs += 1
    
    ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    return ratio, valid_docs, len(documents)


def analyze_compression(args):
    """Analyze compression ratios for different tokenizer/dataset combinations."""
    print("="*80)
    print("TOKENIZER COMPRESSION ANALYSIS")
    print("="*80)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    ts_vocab, ts_merges = load_tokenizer_yaml(args.tinystories_tokenizer)
    owt_vocab, owt_merges = load_tokenizer_yaml(args.owt_tokenizer)
    
    ts_tokenizer = Tokenizer(ts_vocab, ts_merges, special_tokens=["<|endoftext|>"])
    owt_tokenizer = Tokenizer(owt_vocab, owt_merges, special_tokens=["<|endoftext|>"])
    
    print(f"  TinyStories tokenizer: {len(ts_vocab)} tokens")
    print(f"  OpenWebText tokenizer: {len(owt_vocab)} tokens")
    
    # Sample documents
    print(f"\nSampling {args.num_samples} documents from each dataset...")
    ts_docs = sample_documents(args.tinystories_data, num_samples=args.num_samples)
    owt_docs = sample_documents(args.owt_data, num_samples=args.num_samples)
    
    print(f"  Sampled {len(ts_docs)} TinyStories documents")
    print(f"  Sampled {len(owt_docs)} OpenWebText documents")
    
    # Calculate compression ratios
    print("\n" + "="*80)
    print("IN-DOMAIN COMPRESSION RATIOS")
    print("="*80)
    
    ts_on_ts_ratio, ts_on_ts_valid, ts_on_ts_total = calculate_compression_ratio(ts_docs, ts_tokenizer)
    print(f"\nTinyStories tokenizer on TinyStories docs:")
    print(f"  Compression ratio: {ts_on_ts_ratio:.3f} bytes/token")
    print(f"  Valid documents: {ts_on_ts_valid}/{ts_on_ts_total}")
    
    owt_on_owt_ratio, owt_on_owt_valid, owt_on_owt_total = calculate_compression_ratio(owt_docs, owt_tokenizer)
    print(f"\nOpenWebText tokenizer on OpenWebText docs:")
    print(f"  Compression ratio: {owt_on_owt_ratio:.3f} bytes/token")
    print(f"  Valid documents: {owt_on_owt_valid}/{owt_on_owt_total}")
    
    # Cross-domain evaluation
    print("\n" + "="*80)
    print("CROSS-DOMAIN COMPRESSION RATIOS")
    print("="*80)
    
    ts_on_owt_ratio, ts_on_owt_valid, ts_on_owt_total = calculate_compression_ratio(owt_docs, ts_tokenizer)
    print(f"\nTinyStories tokenizer on OpenWebText docs:")
    print(f"  Compression ratio: {ts_on_owt_ratio:.3f} bytes/token")
    print(f"  Valid documents: {ts_on_owt_valid}/{ts_on_owt_total}")
    print(f"  Efficiency vs in-domain: {(ts_on_owt_ratio/owt_on_owt_ratio - 1)*100:+.1f}%")
    
    owt_on_ts_ratio, owt_on_ts_valid, owt_on_ts_total = calculate_compression_ratio(ts_docs, owt_tokenizer)
    print(f"\nOpenWebText tokenizer on TinyStories docs:")
    print(f"  Compression ratio: {owt_on_ts_ratio:.3f} bytes/token")
    print(f"  Valid documents: {owt_on_ts_valid}/{owt_on_ts_total}")
    print(f"  Efficiency vs in-domain: {(owt_on_ts_ratio/ts_on_ts_ratio - 1)*100:+.1f}%")
    
    print("\n" + "="*80)


def analyze_throughput(args):
    """Estimate tokenizer throughput and time to tokenize Pile dataset."""
    print("="*80)
    print("TOKENIZER THROUGHPUT ANALYSIS")
    print("="*80)
    
    # Calculate throughput from encoding results
    ts_bytes_per_token = 4.0  # Typical for TinyStories
    owt_bytes_per_token = 4.0  # Typical for OpenWebText
    
    ts_throughput = (args.ts_train_tokens * ts_bytes_per_token) / args.ts_train_time
    owt_throughput = (args.owt_train_tokens * owt_bytes_per_token) / args.owt_train_time
    avg_throughput = (ts_throughput + owt_throughput) / 2
    
    print(f"\nTinyStories encoding:")
    print(f"  Tokens: {args.ts_train_tokens:,}")
    print(f"  Time: {args.ts_train_time:.1f}s ({args.ts_train_time/60:.1f}m)")
    print(f"  Throughput: {ts_throughput/(1024*1024):.1f} MB/s")
    
    print(f"\nOpenWebText encoding:")
    print(f"  Tokens: {args.owt_train_tokens:,}")
    print(f"  Time: {args.owt_train_time:.1f}s ({args.owt_train_time/60:.1f}m)")
    print(f"  Throughput: {owt_throughput/(1024*1024):.1f} MB/s")
    
    print(f"\nAverage throughput: {avg_throughput/(1024*1024):.1f} MB/s")
    
    # Estimate time for Pile dataset
    pile_bytes = args.pile_size_gb * 1024**3
    pile_time_hours = pile_bytes / avg_throughput / 3600
    
    print(f"\nEstimated time to tokenize Pile dataset ({args.pile_size_gb}GB):")
    print(f"  {pile_time_hours:.1f} hours")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze tokenizer performance")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Compression analysis subcommand
    compression_parser = subparsers.add_parser('compression', help='Analyze compression ratios')
    compression_parser.add_argument('--tinystories-tokenizer', type=str, required=True,
                                   help='Path to TinyStories tokenizer YAML')
    compression_parser.add_argument('--owt-tokenizer', type=str, required=True,
                                   help='Path to OpenWebText tokenizer YAML')
    compression_parser.add_argument('--tinystories-data', type=str, required=True,
                                   help='Path to TinyStories dataset')
    compression_parser.add_argument('--owt-data', type=str, required=True,
                                   help='Path to OpenWebText dataset')
    compression_parser.add_argument('--num-samples', type=int, default=10,
                                   help='Number of documents to sample (default: 10)')
    
    # Throughput analysis subcommand
    throughput_parser = subparsers.add_parser('throughput', help='Estimate tokenizer throughput')
    throughput_parser.add_argument('--ts-train-tokens', type=int, required=True,
                                  help='Number of tokens in TinyStories train set')
    throughput_parser.add_argument('--ts-train-time', type=float, required=True,
                                  help='Time to encode TinyStories train set (seconds)')
    throughput_parser.add_argument('--owt-train-tokens', type=int, required=True,
                                  help='Number of tokens in OpenWebText train set')
    throughput_parser.add_argument('--owt-train-time', type=float, required=True,
                                  help='Time to encode OpenWebText train set (seconds)')
    throughput_parser.add_argument('--pile-size-gb', type=int, default=825,
                                  help='Size of Pile dataset in GB (default: 825)')
    
    args = parser.parse_args()
    
    if args.command == 'compression':
        analyze_compression(args)
    elif args.command == 'throughput':
        analyze_throughput(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
