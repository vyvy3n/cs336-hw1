from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Dict, List, Tuple, Optional

import regex as re  
import json

from .pretokenization import pretokenize, split_on_special_tokens


class Tokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer compatible with GPT-2 style pre-tokenization.

    Given a vocabulary and a list of merges, supports:
    - encodes text into integer IDs
    - decodes integer IDs into text 

    Also support to add user-provided special tokens to vocabulary.
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        self.id_to_bytes = vocab
        self.bytes_to_id = {b: i for i, b in self.id_to_bytes.items()}

        # Add special tokens to vocab if not already present
        new_id = len(self.id_to_bytes)

        self.special_tokens = special_tokens or []
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.bytes_to_id:
                self.id_to_bytes[new_id] = tok_bytes
                self.bytes_to_id[tok_bytes] = new_id
                new_id += 1

        # Merge order matters. Build rank: map pair(bytes, bytes) -> rank
        # Lower rank = earlier merge = higher priority
        self.bpe_ranks = {
            (left, right): rank for rank, (left, right) in enumerate(merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        # Load vocab (expects JSON mapping of token_id -> string token). Convert to bytes.
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        # Keys may be strings; normalize to int
        vocab: Dict[int, bytes] = {}
        for k, v in raw_vocab.items():
            token_id = int(k)
            if isinstance(v, str):
                vocab[token_id] = v.encode("utf-8")
            elif isinstance(v, list):  # optionally support list of byte values
                vocab[token_id] = bytes(v)
            else:
                raise ValueError("Unsupported vocab value type; expected str or list[int]")

        # Load merges as plain text: each line "token1 token2" (strings) -> bytes via utf-8
        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                left, right = parts
                merges.append((left.encode("utf-8"), right.encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings, yielding token IDs as they are produced.
        """
        for chunk in iterable:  # chunk: strings
            if not chunk:
                continue
            for segment in self._split_on_special_tokens(chunk):
                if not segment:
                    continue
                if segment in self.special_tokens:
                    # Special token: encode as its ID directly
                    yield self.bytes_to_id[segment.encode("utf-8")]
                else:
                    # Regular text: pre-tokenize then BPE encoding
                    for pretoken in self._pretokenize(segment):
                        pretoken_bytes = pretoken.encode("utf-8")
                        yield from self._bpe_encode_bytes(pretoken_bytes)

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        out_bytes = b"".join(self.id_to_bytes[i] for i in ids)
        return out_bytes.decode("utf-8", errors="replace")

    # -----------------
    # Internal helpers
    # -----------------
    def _split_on_special_tokens(self, text: str) -> List[str]:
        return split_on_special_tokens(text, self.special_tokens)

    def _pretokenize(self, text: str) -> List[str]:
        return pretokenize(text)
    
    def _bpe_encode_bytes(self, pretoken_bytes: bytes) -> Iterator[int]:
        """
        Apply BPE merges to a single pretoken (given as UTF-8 bytes) and yield IDs.
        
        Example:
            string  = 'the cat ate'
            suppose vocabulary is: {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
            suppose learned merges are [(b't', b'h'), (b' ', b'c'), (b' ', 'a'), (b'th', b'e'), (b' a', b't')]

        ->  pretokens = ['the', ' cat', ' ate']
        ->  pretokens stored as bytes = [
            (b't', b'h', b'e'), 
            (b' ', b'c', b'a', b't'), 
            (b' ', b'a', b't', b'e')
            ] 
        ->  for each pretoken:
            e.g. (b't', b'h', b'e')
        ->      identify 1st merge (b't', b'h') -> pretoken [b'th', b'e']
        ->      identify 2nd merge (b'th', b'e') -> pretoken into [b'the']
        ->      done applying merges, encode as[9]

            e.g. (b' ', b'c', b'a', b't')
        ->      identify only one merge (b' ', b'c') 
        ->      encode as [7, 1, 5]
        """
        if not pretoken_bytes:
            return
        
        byte_list = [bytes([byte]) for byte in pretoken_bytes]
        if len(byte_list) == 1:
            yield self.bytes_to_id[byte_list[0]]
            return

        while True:
            # Find best (lowest-rank) pair to merge
            best_rank = None
            best_index = -1
            for i in range(len(byte_list) - 1):
                pair = (byte_list[i], byte_list[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is None:  # the pair is not in merge list
                    continue
                if (best_rank is None) or (rank < best_rank):
                    best_rank = rank
                    best_index = i
            if best_rank is None:  # no more merges
                break

            # Merge best pair
            merged = byte_list[best_index] + byte_list[best_index + 1]
            # Replace indices best_index and best_index+1 with merged symbol
            byte_list = byte_list[:best_index] + [merged] + byte_list[best_index + 2 :]
            if len(byte_list) == 1:
                break

        for byte in byte_list:
            yield self.bytes_to_id[byte]
