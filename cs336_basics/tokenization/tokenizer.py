"""
BPE Tokenizer for encoding and decoding text.
"""

from __future__ import annotations

import json
import pickle
import re
from typing import Iterable, Iterator

import regex

# GPT-2 pre-tokenization pattern from tiktoken
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = regex.compile(PAT)


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
                        try:
                            vocab[token_id] = bytes.fromhex(v)
                        except ValueError:
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
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
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

        combined_bytes = b"".join(byte_tokens)

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
                if len(buffer) > 8192:  # Process in 8KB chunks
                    # Find last newline in first 8KB
                    chunk_end = buffer.rfind("\n", 0, 8192)
                    if chunk_end == -1:
                        # No newline found, take a smaller chunk at word boundary
                        chunk_end = buffer.rfind(" ", 0, 4096)
                        if chunk_end == -1:
                            chunk_end = 4096  # Force split if no word bounary

                    chunk = buffer[: chunk_end + 1]
                    buffer = buffer[chunk_end + 1 :]

                    token_ids = self.encode(chunk)
                    yield from token_ids
                else:
                    break

        if buffer:
            token_ids = self.encode(buffer)
            yield from token_ids
