"""
Byte-Pair Encoding (BPE) Tokenizer Implementation

This module implements BPE tokenizer training.
It includes pre-tokenization, merge computation, and vocabulary construction.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import Iterable, Iterator

import regex

from cs336_basics.pretokenization_example import find_chunk_boundaries

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
            file_size = os.path.getsize(input_path)
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


class Tokenizer:
    """
    BPE tokenizer for encoding and decoding text.
    """

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        """
        Initialize the tokenizer with vocabulary, merges, and special tokens.

        Args:
            vocab: Dictionary mapping token IDs to bytes
            merges: List of BPE merges in order of creation
            special_tokens: Optional list of special tokens to preserve
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = special_tokens or []

        self.vocab_reverse: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        next_token_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in self.vocab_reverse:
                self.vocab[next_token_id] = special_bytes
                self.vocab_reverse[special_bytes] = next_token_id
                next_token_id += 1

        self.merge_ranks: dict[tuple[bytes, bytes], int] = {merge: i for i, merge in enumerate(self.merges)}

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
            self.special_pattern = "|".join(escaped_tokens)
            self.special_regex = regex.compile(f"({self.special_pattern})")
        else:
            self.special_regex = None

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> Tokenizer:
        """
        Create a tokenizer from saved vocabulary and merges files.

        Args:
            vocab_filepath: Path to vocabulary file
            merges_filepath: Path to merges file
            special_tokens: Optional list of special tokens

        Returns:
            Initialized Tokenizer instance
        """
        if vocab_filepath.endswith(".json"):
            with open(vocab_filepath, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                vocab = {}
                for k, v in vocab_data.items():
                    token_id = int(k)
                    if isinstance(v, str):
                        vocab[token_id] = v.encode("utf-8")
                    elif isinstance(v, list):
                        vocab[token_id] = bytes(v)
                    else:
                        vocab[token_id] = v
        else:
            with open(vocab_filepath, "rb") as f:
                vocab = pickle.load(f)

        if merges_filepath.endswith(".json"):
            with open(merges_filepath, "r", encoding="utf-8") as f:
                merges_data = json.load(f)
                merges = []
                for merge in merges_data:
                    if isinstance(merge[0], str):
                        merge_tuple = (merge[0].encode("utf-8"), merge[1].encode("utf-8"))
                    elif isinstance(merge[0], list):
                        merge_tuple = (bytes(merge[0]), bytes(merge[1]))
                    else:
                        merge_tuple = merge
                    merges.append(merge_tuple)
        else:
            with open(merges_filepath, "rb") as f:
                merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text into a sequence of token IDs.

        Args:
            text: Input text to encode
        
        Returns:
            List of token IDs
        """
        if not text:
            return []

        if self.special_regex:
            text_parts = self.special_regex.split(text)
        else:
            text_parts = [text]
        
        token_ids: list[int] = []

        for part in text_parts:
            if part in self.special_tokens:
                special_bytes = part.encode("utf-8")
                token_ids.append(self.vocab_reverse[special_bytes])
            elif part:
                part_ids = self._encode_text_part(part)
                token_ids.extend(part_ids)
        
        return token_ids
    
    def _encode_text_part(self, text: str) -> list[int]:
        """
        Encode a text part (not containing special tokens) using BPE.

        Args:
            text: Text part to encode
        
        Returns:
            List of token IDs for this text part
        """
        if not text:
            return []
        
        token_ids: list[int] = []

        for match in COMPILED_PAT.finditer(text):
            pre_token = match.group()
            if pre_token:
                byte_sequence = pre_token.encode("utf-8")
                word_tokens = self._apply_bpe(byte_sequence)

                for token in word_tokens:
                    if token in self.vocab_reverse:
                        token_ids.append(self.vocab_reverse[token])
                    else:
                        for byte_val in token:
                            byte_token = bytes([byte_val])
                            if byte_token in self.vocab_reverse:
                                token_ids.append(self.vocab_reverse[byte_token])
        
        return token_ids
    
    def _apply_bpe(self, byte_sequence: bytes) -> list[bytes]:
        """
        Apply BPE merges to a byte sequence

        Args:
            byte_sequence: Input bytes to apply BPE to
        
        Returns:
            List of byte tokens after applying BPE merges
        """
        if len(byte_sequence) <= 1:
            return [byte_sequence]
        
        word = [bytes([b]) for b in byte_sequence]

        while True:
            pairs = []
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in self.merge_ranks:
                    pairs.append((self.merge_ranks[pair], i, pair))
            
            if not pairs:
                break

            pairs.sort()
            _, _, (first, second) = pairs[0]

            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and
                    word[i] == first and
                    word[i + 1] == second):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
        
        return word
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back to text.

        Args:
            ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        if not ids:
            return ""
        
        byte_tokens: list[bytes] = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_tokens.append(self.vocab[token_id])
        
        combined_bytes = b''.join(byte_tokens)

        try:
            return combined_bytes.decode("utf-8", errors="replace")
        except Exception:
            return combined_bytes.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.

        This method processes the input line by line to minimize memory usage,
        making it suitable for large files that don't fit in memory.

        Args:
            iterable: An iterable of strings (e.g., file handle)
        
        Yields:
            Token IDs one at a time
        """
        buffer = ""

        for line in iterable:
            buffer += line

            while buffer:
                if len(buffer) > 8192: # Process in 8KB chunks
                    # Find last newline in first 8KB
                    chunk_end = buffer.rfind('\n', 0, 8192)
                    if chunk_end == -1:
                        # No newline found, take a smaller chunk at word boundary
                        chunk_end = buffer.rfind(' ', 0, 4096)
                        if chunk_end == -1:
                            chunk_end = 4096 # Force split if no word bounary
                    
                    chunk = buffer[:chunk_end + 1]
                    buffer = buffer[chunk_end + 1:]

                    token_ids = self.encode(chunk)
                    yield from token_ids
                else:
                    break
        
        if buffer:
            token_ids = self.encode(buffer)
            yield from token_ids
