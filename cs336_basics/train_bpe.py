"""
BPE Tokenizer Training Implementation

This module implements Byte Pair Encoding (BPE) tokenizer training with optimized
merge efficiency using incremental pair count updates.
"""

import os
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from collections import Counter
from multiprocessing import Process, Queue

from .pretokenization import find_chunk_boundaries, count_pretokens


def initialize_vocabulary(special_tokens: List[str] = None) -> Dict[bytes, int]:
    """
    Initialize vocabulary with special tokens and the 256 byte values.
    """
    vocab = {}

    # Add all 256 possible bytes
    vocab = {bytes([x]): x for x in range(0,256)}
    token_id = len(vocab)

    # Add special tokens
    special_tokens = special_tokens or []
    for special_token in special_tokens:
        special_bytes = special_token.encode('utf-8')
        vocab[special_bytes] = token_id
        token_id += 1

    logging.log(logging.INFO, f"Initialized vocabulary with {len(vocab)} tokens")
    return vocab


def build_pair_counts_and_index(
    pretoken_counts: Dict[Tuple[bytes, ...], int]
) -> Tuple[Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]]]:
    """
    Count all adjacent byte pairs within pre-tokens, and build its inverted index mapping.

    Args:
        pretoken_counts: Dictionary mapping pre-token byte tuples to counts

    Returns:
        pair_counts: (byte1, byte2) pair -> its total counts
        pair_index: (byte1, byte2) pair -> set of pretoken tuples that contain it
    """
    pair_counts = {}
    # pair_index is a defaultdict where each missing key
    # automatically creates an empty set as its default value.
    pair_index = defaultdict(set)

    for pretoken_tuple, count in pretoken_counts.items():
        # Count adjacent pairs within this pre-token, and index it
        for i in range(len(pretoken_tuple) - 1):
            pair = (pretoken_tuple[i], pretoken_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_index[pair].add(pretoken_tuple)

    logging.log(logging.INFO, f"Initialized pair index with {len(pair_index)} unique byte pairs")
    return pair_counts, pair_index


def find_most_frequent_pair(pair_counts: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes]:
    """
    Find the most frequent byte pair, breaking ties lexicographically.
    """
    if not pair_counts:
        raise ValueError("No pairs to merge")
    # Pick by (count, pair) so ties prefer lexicographically larger pair
    most_frequent_pair, max_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
    logging.log(logging.INFO, f"Most frequent pair: {most_frequent_pair} with count {max_count}")
    return most_frequent_pair


def new_pretoken_after_merge_pair(
    old_pretoken: Tuple[bytes, ...],
    pair: Tuple[bytes, bytes],
) -> Tuple[bytes, ...]:
    """
    Build the new pretoken tuple list by merging the target pair (left-to-right)
    """
    new_tuple = []
    i = 0
    while i < len(old_pretoken):
        if (i < len(old_pretoken) - 1 and old_pretoken[i:i+2] == pair):
            new_tuple.append(pair[0] + pair[1])  # Merged pair bytes
            i += 2  # Skip both bytes
        else:
            new_tuple.append(old_pretoken[i])
            i += 1
    return tuple(new_tuple)
    

def merge_pair_and_update_counts(
    pretoken_counts: Dict[Tuple[bytes, ...], int],
    pair: Tuple[bytes, bytes],
    pair_counts: Dict[Tuple[bytes, bytes], int],
    pair_index: Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]],
) -> Tuple[Dict[Tuple[bytes, ...], int], Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]]]:
    """
    Merge a byte pair in all pre-tokens, creating new pre-token counts.

    Uses an inverted index (pair -> set of pretoken tuples) to touch only
    pretokens that actually contain the merged pair.

    Args:
        pretoken_counts: Current pre-token counts
        pair: Byte pair to merge (A, B) -> AB
        pair_counts: Current pair counts
        pair_index: Inverted index mapping pair -> set of pretokens containing it

    Returns:
        Tuple of (updated_pretoken_counts, updated_pair_counts, updated_pair_index)
    """
    # Copy; only updated pairs count and index in affected tokens
    new_pair_counts = pair_counts.copy()
    new_pair_counts.pop(pair, None)
    new_pretoken_counts = pretoken_counts.copy()

    # Take affected pretokens (and clear this entry) for early exit and simpler logic
    affected_pretokens = pair_index.pop(pair, set())
    if not affected_pretokens:
        return pretoken_counts, pair_counts, pair_index

    # Process affected pretokens only
    # e.g. affected pretoken (A, B, C, D) as old tuple, merge pair (B, C) -> new tuple (A, BC, D) 
    for old_tuple in affected_pretokens:
        count = pretoken_counts.get(old_tuple, 0)
        if not count:
            continue

        # Remove old pretoken tuple from the map
        new_pretoken_counts.pop(old_tuple, None)

        # Subtract ALL old pairs of this pretoken
        for j in range(len(old_tuple) - 1):
            # Decrement old pair count and delete if <= 0; no-op if key missing
            old_pair = (old_tuple[j], old_tuple[j + 1])
            if old_pair in new_pair_counts.keys():
                new_pair_counts[old_pair] -= count
                if new_pair_counts[old_pair] <= 0:
                    new_pair_counts.pop(old_pair, None)
        
            # Update inverted index: remove old membership
            old_pair_index_set = pair_index.get(old_pair)
            if old_pair_index_set:
                old_pair_index_set.discard(old_tuple)
                if not old_pair_index_set:
                    pair_index.pop(old_pair, None)
            
        # Build the new pretoken tuple list by merging the target pair
        new_tuple = new_pretoken_after_merge_pair(old_tuple, pair)

        # Add ALL new pairs of this pretoken
        for j in range(len(new_tuple) - 1):
            # Increment new pair count (create if absent)
            new_pair = (new_tuple[j], new_tuple[j + 1])
            new_pair_counts[new_pair] = new_pair_counts.get(new_pair, 0) + count

            # Update inverted index: add new membership
            pair_index.setdefault(new_pair, set()).add(new_tuple)

        # Add/aggregate the rebuilt pretoken
        new_pretoken_counts[new_tuple] = new_pretoken_counts.get(new_tuple, 0) + count

    logging.log(logging.INFO, f"Merged pair {pair} -> {pair[0] + pair[1]}")
    return new_pretoken_counts, new_pair_counts, pair_index


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
    current_pretokens = pretoken_counts.copy()

    # Build initial pair counts and inverted index
    pair_counts, pair_index = build_pair_counts_and_index(pretoken_counts)

    for merge_step in range(num_merges):
        logging.log(logging.INFO, f"Merge step {merge_step + 1}/{num_merges}")

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

        # ✅ OPTIMIZED: Merge only affected pretokens using the inverted index and update counts
        current_pretokens, pair_counts, pair_index = merge_pair_and_update_counts(
            current_pretokens, most_frequent_pair, pair_counts, pair_index
        )

    logging.log(logging.INFO, f"Completed {len(merges)} merges, final vocab size: {len(vocab)}")
    return merges, vocab


def worker(start: int, end: int, input_path: str, special_tokens: list[str], q: Queue):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        q.put(count_pretokens(chunk, special_tokens))


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 4,
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

    # Read and pre-tokenize the corpus (parallelization)
    processes = []
    q = Queue()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # avoid copying big chunks by passing start/end so workers read their own slice
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            p = Process(target=worker, args=(start, end, input_path, special_tokens, q))
            p.start()
            processes.append(p)
    
    all_pretoken_counts = Counter(d := q.get()) 

    # Merge counts
    for _ in range(1, len(processes)): 
        all_pretoken_counts.update(q.get())

    for p in processes:
        p.join()
    
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
