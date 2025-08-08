from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, BinaryIO

BytePair = tuple[bytes, bytes]
Vocab = dict[int, bytes]

@dataclass
class BPETokenizer:
    id_to_bytes: Vocab
    merges: list[BytePair]
    special_tokens: list[str] | None = None

    def __post_init__(self):
        # Derived structures
        self.bytes_to_id: dict[bytes, int] = {b: i for i, b in self.id_to_bytes.items()}
        # Rank: lower index = higher priority
        self.ranks: dict[BytePair, int] = {pair: i for i, pair in enumerate(self.merges)}
        # Specials in bytes, plus data structure for longest-first matching
        self.special_bytes: list[bytes] = [s.encode("utf-8") for s in (self.special_tokens or [])]
        # Optionally build a trie for specials
        # self._special_trie = ...

    # Public API required by tests

    def encode(self, text: str) -> list[int]:
        """
        - Greedy longest-first special token matching (preserve specials).
        - Convert non-special spans to bytes and run BPE using self.ranks.
        """
        # 1) split into segments: [(is_special, span_bytes), ...]
        # segments = self._split_specials(text)
        # 2) for each segment:
        #    if special -> [bytes_to_id[special_bytes]]
        #    else -> self._encode_bytes(span_bytes)
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str] | Iterable[bytes] | BinaryIO) -> Iterator[int]:
        """
        Memory-efficient streaming version:
        - Iterate line-by-line/chunk-by-chunk, yield ids incrementally.
        - Must preserve special tokens across chunk boundaries (buffer tail).
        """
        # Maintain rolling buffer to avoid splitting specials across boundaries
        # for chunk in self._chunks(iterable):
        #     for _id in self._encode_chunk_with_carry(chunk):
        #         yield _id
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """
        - Join bytes for all ids in order.
        - Decode with UTF-8 to Python str.
        """
        # data = b"".join(self.id_to_bytes[i] for i in ids)
        # return data.decode("utf-8")
        raise NotImplementedError

    # Internal helpers (implement as needed)

    def _split_specials(self, text: str) -> list[tuple[bool, bytes]]:
        """
        Return segments as (is_special, bytes). Choose longest matching special first.
        Overlapping specials must prefer the longer one.
        """
        # Implement greedy longest-first scan (e.g., trie or sorted specials by length).
        raise NotImplementedError

    def _encode_bytes(self, data: bytes) -> list[int]:
        """
        Classic BPE:
        - Start as a list of single-byte tokens (each element is a bytes of length 1).
        - Repeatedly merge the lowest-rank adjacent pair until no pair exists in ranks.
        - Map final byte chunks to ids.
        """
        # seq: list[bytes] = [bytes([b]) for b in data]
        # while True:
        #     pair_positions = self._best_pair_positions(seq)
        #     if pair_positions is None: break
        #     seq = self._apply_merge(seq, pair_positions)
        # return [self.bytes_to_id[s] for s in seq]
        raise NotImplementedError

    def _best_pair_positions(self, seq: list[bytes]):
        """
        Find the adjacent pair with the best (lowest) rank, return its index(es).
        Efficient implementations use a heap + linked structure; a simple version scans.
        """
        raise NotImplementedError

    def _apply_merge(self, seq: list[bytes], pos: int) -> list[bytes]:
        """
        Merge seq[pos] and seq[pos+1] into a single bytes element, return new seq.
        """
        raise NotImplementedError


# Training API expected by tests
def train_bpe(
    input_path: str | bytes | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    min_frequency: int = 2,
    max_merges: int | None = None,
) -> tuple[Vocab, list[BytePair]]:
    """
    Learn BPE merges from corpus:
    - Initialize vocab with all single bytes (0..255) plus special tokens appended.
    - Count pair frequencies over corpus (respect specials; treat them as atomic).
    - Iteratively select most frequent pair, add merge, update counts, stop at vocab_size.
    - Return (id_to_bytes, merges). ids should be contiguous [0..N-1].
    """
    raise NotImplementedError