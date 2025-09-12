"""
BPE Tokenizer Training Implementation

This module implements Byte Pair Encoding (BPE) tokenizer training with optimized
merge efficiency using incremental pair count updates.
"""

import os
import logging
import heapq
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


# Tie-break helper: cached descending key for lexicographically-greater-first ordering.
#
# We map each bytes object b to inv(b) = bytes(255 - x for x in b). Because Python's tuple
# comparison is lexicographic ascending, sorting (inv(a), inv(b)) ascending is equivalent
# to sorting (a, b) descending. We cache inv(b) to avoid recomputation and allocations.
_inv_cache: Dict[bytes, bytes] = {}

def _inv(b: bytes) -> bytes:
    """
    Return inverted bytes to break ties by preferring lexicographically greater pair.

    Rationale:
    1.  bytes(255 - x for x in b) is not enough since it preserves the “shorter is smaller” tie-break, 
        which leads to incorrect merge ordering like choosing (b' ', b'd') before (b' a', b'nd').

    2.  Append a 0xFF sentinel after inverting the bytes. 
        This flips the prefix tie-break as well, 
        so ascending order on the mapped key matches descending order on the original bytes.
    """
    inv = _inv_cache.get(b)
    if inv is None:
        inv = bytes(255 - x for x in b) + b"\xff"
        _inv_cache[b] = inv
    return inv

def _desc_key(pair: Tuple[bytes, bytes]) -> Tuple[bytes, bytes]:
    """
    Secondary heap key to break ties by preferring lexicographically greater pair.
    """
    return (_inv(pair[0]), _inv(pair[1]))


def build_pair_counts_and_index(
    pretoken_counts: Dict[Tuple[bytes, ...], int]
) -> Tuple[
    Dict[Tuple[bytes, bytes], int],
    Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]],
    List[Tuple[int, Tuple[bytes, bytes], Tuple[bytes, bytes]]],  # heap entries: (-count, desc_key, pair)
]:
    """
    Count all adjacent byte pairs within pre-tokens, and build inverted index, initialize heap.

    Args:
        pretoken_counts: Dictionary mapping pre-token byte tuples to counts

    Returns:
        pair_counts: (byte1, byte2) pair -> its total counts
        pair_index: (byte1, byte2) pair -> set of pretoken tuples that contain it
        pair_heap: max-heap of (-count, pair) for efficient most frequent pair lookup
    """
    pair_counts = {}
    pair_index = defaultdict(set)

    for pretoken_tuple, count in pretoken_counts.items():
        # Count adjacent pairs within this pretoken, and index it
        for i in range(len(pretoken_tuple) - 1):
            pair = (pretoken_tuple[i], pretoken_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_index[pair].add(pretoken_tuple)

    # Initialize heap with all pairs: (-count, tie-break key, pair)
    # Tie-breaking: prefer lexicographically greater pair via cached descending key
    pair_heap = [(-count, _desc_key(pair), pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)

    logging.log(logging.INFO, f"Initialized pair index with {len(pair_index)} unique byte pairs")
    return pair_counts, pair_index, pair_heap


def find_most_frequent_pair(
        pair_counts: Dict[Tuple[bytes, bytes], int],
        pair_heap: List[Tuple[int, Tuple[bytes, bytes], Tuple[bytes, bytes]]]
    ) -> Tuple[bytes, bytes]:
    """
    Find the most frequent byte pair, breaking ties lexicographically. 
    Use heap with lazy deletion.

    ATTENTION: 
        Taking max() over the entire pair counts dict at every merge step takes O(n).
        Maintain a max-heap of pair counts instead, can reduces it to O(log n).
        This is the major performance bottleneck in BPE. 
        ```
        most_frequent_pair, max_count = \
            max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        ```
    """
    if not pair_counts:
        raise ValueError("No pairs to merge")

    # ✅ OPTIMIZED: max-heap with lazy deletion, reduces max() O(n) to O(log n)
    while pair_heap:
        neg_count, _, pair = heapq.heappop(pair_heap)
        current_count = pair_counts.get(pair, 0)
        if current_count > 0 and neg_count == -current_count:
            logging.log(logging.INFO, f"Most frequent pair: {pair} with count {current_count}")
            return pair
    
    raise ValueError("No pairs to merge")


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
    pair_heap: List[Tuple[int, Tuple[bytes, bytes], Tuple[bytes, bytes]]],
):
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
    pair_counts.pop(pair, None)

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
        pretoken_counts.pop(old_tuple, None)

        # Subtract ALL old pairs of this pretoken
        for j in range(len(old_tuple) - 1):
            # Decrement old pair count and delete if <= 0; no-op if key missing
            old_pair = (old_tuple[j], old_tuple[j + 1])
            if old_pair in pair_counts.keys():
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] <= 0:
                    pair_counts.pop(old_pair, None)
                else:
                    heapq.heappush(pair_heap, (-pair_counts[old_pair], _desc_key(old_pair), old_pair))

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
            pair_counts[new_pair] = pair_counts.get(new_pair, 0) + count

            # Update inverted index: add new membership
            pair_index.setdefault(new_pair, set()).add(new_tuple)

            # Push updated pair to heap (lazy deletion handles stale entries)
            heapq.heappush(pair_heap, (-pair_counts[new_pair], _desc_key(new_pair), new_pair))

        # Add/aggregate the rebuilt pretoken
        pretoken_counts[new_tuple] = pretoken_counts.get(new_tuple, 0) + count

    logging.log(logging.INFO, f"Merged pair {pair} -> {pair[0] + pair[1]}")
    return pretoken_counts, pair_counts, pair_index, pair_heap


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

    # Build initial pair counts and inverted index
    pair_counts, pair_index, pair_heap = build_pair_counts_and_index(pretoken_counts)

    for merge_step in range(num_merges):
        logging.log(logging.INFO, f"Merge step {merge_step + 1}/{num_merges}")

        if not pair_counts:
            logging.log(logging.INFO, "No more pairs to merge")
            break

        # Find most frequent pair
        most_frequent_pair = find_most_frequent_pair(pair_counts, pair_heap)
        merges.append(most_frequent_pair)

        # Add merged token to vocabulary
        merged_bytes = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[merged_bytes] = token_id
        token_id += 1

        # ✅ OPTIMIZED: Merge only affected pretokens using the inverted index and update counts
        pretoken_counts, pair_counts, pair_index, pair_heap = merge_pair_and_update_counts(
            pretoken_counts, most_frequent_pair, pair_counts, pair_index, pair_heap
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
