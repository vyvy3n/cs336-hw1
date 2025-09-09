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

    pair_counts: Dict[Tuple[bytes, bytes], int] = {}
    # pair_index is a defaultdict where each missing key
    # automatically creates an empty set as its default value.
    pair_index: Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]] = defaultdict(set)

    for pretoken_tuple, count in pretoken_counts.items():
        # Count adjacent pairs within this pre-token, and index it
        for i in range(len(pretoken_tuple) - 1):
            pair = (pretoken_tuple[i], pretoken_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_index[pair].add(pretoken_tuple)
    
    logging.log(logging.INFO, f"Initialized pair index with {len(pair_index)} unique byte pairs")
    return pair_counts, pair_index


def pretoken_pairs_set(pretoken_tuple: Tuple[bytes, ...]) -> Set[Tuple[bytes, bytes]]:
    """Return the set of adjacent pairs present in the given pretoken tuple."""
    return {
        (pretoken_tuple[i], pretoken_tuple[i + 1])
        for i in range(len(pretoken_tuple) - 1)
    }


def find_most_frequent_pair(pair_counts: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes]:
    """
    Find the most frequent byte pair, breaking ties by preferring the lexicographically greater pair.
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


def contains_pair(pretoken_tuple: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> bool:
    """
    Check if a pre-token tuple contains the specified pair.
    """
    for i in range(len(pretoken_tuple) - 1):
        if pretoken_tuple[i] == pair[0] and pretoken_tuple[i + 1] == pair[1]:
            return True
    return False


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
    merged_bytes = pair[0] + pair[1]
    new_pretoken_counts: Dict[Tuple[bytes, ...], int] = {}
    new_pair_counts: Dict[Tuple[bytes, bytes], int] = pair_counts.copy()

    # Remove the merged pair from counts entirely (no need to decrement per occurrence)
    if pair in new_pair_counts:
        del new_pair_counts[pair]

    # Determine which pretokens are affected using the inverted index
    affected_pretokens: Set[Tuple[bytes, ...]] = set(pair_index.get(pair, set()))

    # Copy unaffected pretokens without scanning their contents
    for pretoken_tuple, count in pretoken_counts.items():
        if pretoken_tuple not in affected_pretokens:
            new_pretoken_counts[pretoken_tuple] = count

    # Process affected pretokens
    for pretoken_tuple in affected_pretokens:
        count = pretoken_counts[pretoken_tuple]

        # Collect all old pairs from this pretoken (for index updates only)
        old_pairs = pretoken_pairs_set(pretoken_tuple)

        # Create new tuple with merged pair and update pair counts incrementally
        new_tuple_list: List[bytes] = []
        i = 0
        while i < len(pretoken_tuple):
            if (
                i < len(pretoken_tuple) - 1
                and pretoken_tuple[i] == pair[0]
                and pretoken_tuple[i + 1] == pair[1]
            ):
                # Remove affected neighbor pairs from counts
                if i > 0:
                    old_pair_before = (pretoken_tuple[i - 1], pretoken_tuple[i])
                    if old_pair_before in new_pair_counts:
                        new_pair_counts[old_pair_before] -= count
                        if new_pair_counts[old_pair_before] <= 0:
                            del new_pair_counts[old_pair_before]
                if i < len(pretoken_tuple) - 2:
                    old_pair_after = (pretoken_tuple[i + 1], pretoken_tuple[i + 2])
                    if old_pair_after in new_pair_counts:
                        new_pair_counts[old_pair_after] -= count
                        if new_pair_counts[old_pair_after] <= 0:
                            del new_pair_counts[old_pair_after]

                # Add new neighbor pairs formed by the merge
                if i > 0:
                    new_pair_before = (pretoken_tuple[i - 1], merged_bytes)
                    new_pair_counts[new_pair_before] = new_pair_counts.get(new_pair_before, 0) + count
                if i < len(pretoken_tuple) - 2:
                    new_pair_after = (merged_bytes, pretoken_tuple[i + 2])
                    new_pair_counts[new_pair_after] = new_pair_counts.get(new_pair_after, 0) + count

                # Emit merged token
                new_tuple_list.append(merged_bytes)
                i += 2
            else:
                new_tuple_list.append(pretoken_tuple[i])
                i += 1

        new_tuple = tuple(new_tuple_list)
        # Aggregate counts if multiple pretokens collapse to the same new tuple
        new_pretoken_counts[new_tuple] = new_pretoken_counts.get(new_tuple, 0) + count

        # Update inverted index: remove old pretoken membership
        for p in old_pairs:
            s = pair_index.get(p)
            if s is not None:
                s.discard(pretoken_tuple)
                if not s:
                    # keep the dict clean
                    pair_index.pop(p, None)

        # Update inverted index: add new pretoken membership
        new_pairs = pretoken_pairs_set(new_tuple)
        for p in new_pairs:
            s = pair_index.get(p)
            if s is None:
                s = set()
                pair_index[p] = s
            s.add(new_tuple)

    # The merged pair no longer exists; ensure its index set is cleared
    pair_index.pop(pair, None)

    logging.log(logging.INFO, f"Merged pair {pair} -> {merged_bytes}")
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
