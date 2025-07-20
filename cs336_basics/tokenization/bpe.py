"""
Byte-Pair Encoding (BPE) tokenizer training.

This module implements BPE tokenizer training.
It includes pre-tokenization, merge computation, and vocabulary construction.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, Iterator

import regex

from .utils import find_chunk_boundaries

# GPT-2 pre-tokenization pattern from tiktoken
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = regex.compile(PAT)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the given corpus.

    Args:
        input_path: Path to the input text file for training
        vocab_size: Maximum vocabulary size (including special tokens and byte tokens)
        special_tokens: List of special tokens to add to vocabulary

    Returns:
        vocab: Dictionary mapping token IDs to bytes
        merges: List of merge operations in order of creation
    """
    vocab: dict[int, bytes] = {}
    next_token_id = 0

    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    for i in range(256):
        vocab[next_token_id] = bytes([i])
        next_token_id += 1

    word_freqs = _pretokenize_corpus(input_path, special_tokens)

    word_splits = {}
    for word, _ in word_freqs.items():
        word_splits[word] = [bytes([b]) for b in word.encode("utf-8")]

    pair_counts = _get_stats_bytes(word_splits, word_freqs)
    pair_to_words = _build_pair_index(word_splits)

    merges: list[tuple[bytes, bytes]] = []
    max_merges = vocab_size - len(vocab)

    for i in range(max_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merged_token = best_pair[0] + best_pair[1]

        _merge_vocab_bytes_with_index(best_pair, word_splits, word_freqs, pair_counts, pair_to_words, merged_token)
        merges.append(best_pair)

        vocab[next_token_id] = merged_token
        next_token_id += 1

    return vocab, merges


def _pretokenize_corpus(input_path: str, special_tokens: list[str]) -> dict[str, int]:
    """
    Pre-tokenize the corpus using GPT-2 regex pattern with parallel processing.

    Args:
        input_path: Path to input corpus
        special_tokens: List of special tokens to split on

    Returns:
        Dictionary mapping pre-tokens to their frequencies
    """
    num_processes = cpu_count()

    with open(input_path, "rb") as f:
        if special_tokens:
            split_token = special_tokens[0].encode("utf-8")
            boundaries = find_chunk_boundaries(f, num_processes, split_token)
        else:
            file_size = Path(input_path).stat().st_size
            chunk_size = file_size // num_processes
            boundaries = [i * chunk_size for i in range(num_processes + 1)]
            boundaries[-1] = file_size

    chunk_args = []
    for i in range(len(boundaries) - 1):
        start_pos = boundaries[i]
        end_pos = boundaries[i + 1]
        chunk_args.append((input_path, start_pos, end_pos, special_tokens))

    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(_process_chunk, chunk_args)

    combined_freqs = Counter()
    for chunk_freqs in chunk_results:
        combined_freqs.update(chunk_freqs)

    return dict(combined_freqs)


def _process_chunk(args: tuple[str, int, int, list[str]]) -> Counter:
    """Process a single chunk of the corpus for pre-tokenization."""
    input_path, start_pos, end_pos, special_tokens = args

    word_freqs = Counter()

    with open(input_path, "rb") as f:
        f.seek(start_pos)
        chunk_bytes = f.read(end_pos - start_pos)
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        special_pattern = "|".join(escaped_tokens)
        special_regex = re.compile(f"({special_pattern})")
    else:
        special_regex = None

    _process_text_chunk(chunk_text, word_freqs, special_tokens, special_regex)

    return word_freqs


def _process_text_chunk(
    text: str, word_freqs: Counter, special_tokens: list[str], special_regex: re.Pattern | None
) -> None:
    """Process a text chunk and update word frequencies"""
    if special_regex:
        text_chunks = special_regex.split(text)
    else:
        text_chunks = [text]

    for chunk in text_chunks:
        if chunk in special_tokens:
            continue

        if chunk.strip():
            for match in COMPILED_PAT.finditer(chunk):
                token = match.group()
                word_freqs[token] += 1


def _get_stats_bytes(word_splits: dict[str, list[bytes]], word_freqs: dict[str, int]) -> dict[tuple[bytes, bytes], int]:
    """
    Count frequency of all consecutive byte pairs across the corpus.

    Args:
        word_splits: Mapping from words to their byte sequences
        word_freqs: Frequency of each word in the corpus

    Returns:
        Dictionary mapping byte pairs to their frequencies
    """
    pairs = defaultdict(int)

    for word, freq in word_freqs.items():
        symbols = word_splits[word]
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq

    return dict(pairs)


def _build_pair_index(word_splits: dict[str, list[bytes]]) -> dict[tuple[bytes, bytes], set[str]]:
    """
    Build a reverse index mapping pairs to the words that contain them.

    Args:
        word_splits: Mapping from words to their byte sequences

    Returns:
        Dictionary mapping pairs to sets of words that contain them
    """
    pair_to_words = defaultdict(set)

    for word, symbols in word_splits.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_to_words[pair].add(word)

    return dict(pair_to_words)


def _merge_vocab_bytes_with_index(
    pair: tuple[bytes, bytes],
    word_splits: dict[str, list[bytes]],
    word_freqs: dict[str, int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[str]],
    merged_token: bytes,
) -> None:
    """
    Merge all occurences of a byte pair using reverse index for maximum efficiency.

    Args:
        pair: The byte pair to merge
        word_splits: Current word splits
        word_freqs: Frequency of each word
        pair_counts: Current pair counts
        pair_to_words: Reverse index from pairs to words
        merged_token: The merged token to replace pair with
    """
    affected_words = list(pair_to_words.get(pair, set()))

    if pair in pair_counts:
        del pair_counts[pair]
    if pair in pair_to_words:
        del pair_to_words[pair]

    pair0, pair1 = pair

    for word in affected_words:
        old_symbols = word_splits[word]
        freq = word_freqs[word]

        for i in range(len(old_symbols) - 1):
            old_pair = (old_symbols[i], old_symbols[i + 1])
            if old_pair in pair_counts:
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]

            if old_pair in pair_to_words:
                pair_to_words[old_pair].discard(word)
                if not pair_to_words[old_pair]:
                    del pair_to_words[old_pair]

        new_symbols = []
        i = 0
        while i < len(old_symbols):
            if i < len(old_symbols) - 1 and old_symbols[i] == pair0 and old_symbols[i + 1] == pair1:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(old_symbols[i])
                i += 1

        word_splits[word] = new_symbols

        for i in range(len(new_symbols) - 1):
            new_pair = (new_symbols[i], new_symbols[i + 1])
            if new_pair in pair_counts:
                pair_counts[new_pair] += freq
            else:
                pair_counts[new_pair] = freq

            if new_pair not in pair_to_words:
                pair_to_words[new_pair] = set()
            pair_to_words[new_pair].add(word)
