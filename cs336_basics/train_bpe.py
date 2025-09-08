"""
BPE Tokenizer Training Implementation

This module implements Byte Pair Encoding (BPE) tokenizer training with optimized
merge efficiency using incremental pair count updates.
"""

import os
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter

from .pretokenization import find_chunk_boundaries, count_pretokens, pretokenize_text


def initialize_vocabulary(special_tokens: List[str] = None) -> Dict[bytes, int]:
    """
    Initialize vocabulary with special tokens and the 256 byte values.
    
    Args:
        special_tokens: List of special tokens
        
    Returns:
        Dictionary mapping token bytes to token ID
    """
    vocab = {}
    token_id = 0
    
    # Add special tokens
    special_tokens = special_tokens or []
    for special_token in special_tokens:
        special_bytes = special_token.encode('utf-8')
        vocab[special_bytes] = token_id
        token_id += 1
    
    # Add all 256 possible bytes
    for i in range(256):
        vocab[bytes([i])] = token_id
        token_id += 1
    
    logging.log(logging.INFO, f"Initialized vocabulary with {len(vocab)} tokens")
    return vocab


def count_byte_pairs(pretoken_counts: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
    """
    Count all adjacent byte pairs within pre-tokens.
    
    Args:
        pretoken_counts: Dictionary mapping pre-token byte tuples to counts
        
    Returns:
        Dictionary mapping (byte1, byte2) pairs to total counts
    """
    pair_counts = defaultdict(int)
    
    for pretoken_tuple, count in pretoken_counts.items():
        # Count adjacent pairs within this pre-token
        for i in range(len(pretoken_tuple) - 1):
            pair = (pretoken_tuple[i], pretoken_tuple[i + 1])
            pair_counts[pair] += count
    
    logging.log(logging.INFO, f"Counted {len(pair_counts)} unique byte pairs")
    return dict(pair_counts)


def find_most_frequent_pair(pair_counts: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes]:
    """
    Find the most frequent byte pair, breaking ties by preferring the lexicographically greater pair.
    
    Args:
        pair_counts: Dictionary mapping byte pairs to counts
        
    Returns:
        Most frequent byte pair
    """
    if not pair_counts:
        raise ValueError("No pairs to merge")
    
    max_count = max(pair_counts.values())
    # Get all pairs with max count, then pick lexicographically largest
    max_pairs = [pair for pair, count in pair_counts.items() if count == max_count]
    # Merge the lexicographically greater pair when ties occur
    most_frequent = max(max_pairs)
    
    logging.log(logging.INFO, f"Most frequent pair: {most_frequent} with count {max_count}")
    return most_frequent


def merge_pair_in_pretokens(
    pretoken_counts: Dict[Tuple[bytes, ...], int], 
    pair: Tuple[bytes, bytes]
) -> Dict[Tuple[bytes, ...], int]:
    """
    Merge a byte pair in all pre-tokens, creating new pre-token counts.
    
    Args:
        pretoken_counts: Current pre-token counts
        pair: Byte pair to merge (A, B) -> AB
        
    Returns:
        Updated pre-token counts after merging
    """
    merged_bytes = pair[0] + pair[1]
    new_pretoken_counts = {}
    
    for pretoken_tuple, count in pretoken_counts.items():
        # Create new tuple with merged pair
        new_tuple = []
        i = 0
        while i < len(pretoken_tuple):
            if (i < len(pretoken_tuple) - 1 and 
                pretoken_tuple[i] == pair[0] and 
                pretoken_tuple[i + 1] == pair[1]):
                # Merge the pair
                new_tuple.append(merged_bytes)
                i += 2  # Skip both bytes
            else:
                new_tuple.append(pretoken_tuple[i])
                i += 1
        
        new_pretoken_counts[tuple(new_tuple)] = count
    
    logging.log(logging.INFO, f"Merged pair {pair} -> {merged_bytes}")
    return new_pretoken_counts


def train_bpe_on_pretokens(
    pretoken_counts: Dict[Tuple[bytes, ...], int],
    num_merges: int,
    special_tokens: List[str] = None
) -> Tuple[List[Tuple[bytes, bytes]], Dict[bytes, int]]:
    """
    Train BPE on pre-token counts.
    
    Args:
        pretoken_counts: Dictionary mapping pre-token byte tuples to counts
        num_merges: Number of merge operations to perform
        special_tokens: List of special tokens to protect from merging
        
    Returns:
        Tuple of (merges, final_vocab) where:
        - merges: List of (byte1, byte2) pairs merged in order
        - final_vocab: Final vocabulary mapping bytes to token IDs
    """    
    merges = []
    
    # Initialize vocabulary with special tokens first, then bytes
    vocab = initialize_vocabulary(special_tokens)
    token_id = len(vocab)
    
    for merge_step in range(num_merges):
        logging.log(logging.INFO, f"Merge step {merge_step + 1}/{num_merges}")
        
        # Count current byte pairs (excluding special tokens)
        pair_counts = count_byte_pairs(pretoken_counts)
        
        if not pair_counts:
            logging.log(logging.INFO, "No more pairs to merge")
            break
        
        # Find most frequent pair
        most_frequent_pair = find_most_frequent_pair(pair_counts)
        merges.append(most_frequent_pair)
        
        # Add merged token to vocabulary
        merged_bytes = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[merged_bytes] = token_id
        token_id += 1
        
        # Merge the pair in all pre-tokens
        pretoken_counts = merge_pair_in_pretokens(pretoken_counts, most_frequent_pair)
    
    logging.log(logging.INFO, f"Completed {len(merges)} merges, final vocab size: {len(vocab)}")
    return merges, vocab


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    **kwargs,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on input text.
    
    Args:
        input_path: Path to training corpus
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include
        
    Returns:
        Tuple of (vocab, merges) where:
        - vocab: Dict mapping token ID to token bytes
        - merges: List of (byte1, byte2) pairs merged in order
    """
    logging.log(logging.INFO, f"Starting BPE training on {input_path}")
    logging.log(logging.INFO, f"Target vocab size: {vocab_size}, special tokens: {special_tokens}")
    
    # Read and pre-tokenize the corpus
    all_pretoken_counts = {}
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 4, b"<|endoftext|>")
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_counts = count_pretokens(chunk, special_tokens)
            
            # Merge counts
            for pretoken, count in chunk_counts.items():
                all_pretoken_counts[pretoken] = all_pretoken_counts.get(pretoken, 0) + count
    
    logging.log(logging.INFO, f"Total unique pre-tokens: {len(all_pretoken_counts)}")
    
    # Calculate number of merges needed
    initial_vocab_size = 256 + len(special_tokens)
    num_merges = vocab_size - initial_vocab_size
    
    if num_merges <= 0:
        logging.log(logging.WARNING, f"Vocab size {vocab_size} too small, using {initial_vocab_size}")
        num_merges = 0
    
    # Train BPE on all pre-token counts.
    # Special tokens are excluded from pre-token counts, so we don't need protection logic during training.
    merges, vocab_bytes_to_id = train_bpe_on_pretokens(all_pretoken_counts, num_merges, special_tokens)
    
    # Convert to ID -> bytes mapping
    vocab_id_to_bytes = {token_id: token_bytes for token_bytes, token_id in vocab_bytes_to_id.items()}
    
    logging.log(logging.INFO, f"Final vocabulary size: {len(vocab_id_to_bytes)}")
    return vocab_id_to_bytes, merges


if __name__ == "__main__":
    # Test with small example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    

    # test_text = "low low low low low<|endoftext|>lower lower<|endoftext|>widest widest widest<|endoftext|>newest newest newest newest newest newest"
    test_text = "low low low<|endoftext|> lower lower" 
    pretoken_counts = count_pretokens(test_text, ["<|endoftext|>"])
    
    # print("Original pre-token counts:")
    # for k, v in sorted(pretoken_counts.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{k}: {v}")
    
    merges, vocab = train_bpe_on_pretokens(pretoken_counts, 6, ["<|endoftext|>"])
    
    print(f"\nMerges: {merges}")
    print(f"Final vocab size: {len(vocab)}")
