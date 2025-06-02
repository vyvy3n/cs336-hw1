import os
import regex as re
from collections import Counter
from collections.abc import Iterable, Iterator
import heapq
from typing import BinaryIO


# Parsing pattern used in GPT2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Segmenter:
    """This class scans the raw input string, finds occurrences of any of the “special tokens”
    (by working in raw UTF-8 bytes), and returns a sequence of segments"""

    def __init__(self, special_tokens: list[str] | None = None, in_bytes: bool = False):
        self.special_tokens = special_tokens or []
        self.in_string = not in_bytes

        # Build a simple byte‐trie for special tokens
        # (each key is a UTF‐8 string; store bytes-list for matching)
        self._special_trie = {}
        for tok in self.special_tokens:
            node = self._special_trie
            tok_bytes = tok.encode("utf-8")
            for b in tok_bytes:
                node = node.setdefault(b, {})
            node["_end"] = tok  # mark end‐of‐special‐token at this node

    def __call__(self, text: str | bytes) -> list[tuple[bool, str | bytes]]:
        """
        Get the list of segments (bool, str) from the text.
        """
        if self.in_string:
            data = text.encode("utf-8")  # work in bytes for exact matches
        else:
            data = text

        i = 0
        nbytes = len(data)
        result: list[tuple[bool, str]] = []
        buffer_bytes = bytearray()

        while i < nbytes:
            node = self._special_trie
            j = i
            last_match = None
            last_pos = i
            # walk as far as possible in the trie
            while j < nbytes and data[j] in node:
                node = node[data[j]]
                j += 1
                if "_end" in node:
                    last_match = node["_end"]
                    last_pos = j

            if last_match is not None:
                # flush any buffered “normal” bytes so far
                if buffer_bytes:
                    if self.in_string:
                        buffer_bytes = buffer_bytes.decode("utf-8", "ignore")
                    result.append((False, buffer_bytes))
                    buffer_bytes = bytearray()
                # append the special token
                result.append((True, last_match))
                i = last_pos
            else:
                # no special token starts here
                buffer_bytes.append(data[i])
                i += 1

        if buffer_bytes:
            if self.in_string:
                buffer_bytes = buffer_bytes.decode("utf-8", "ignore")
            result.append((False, buffer_bytes))
        return result


class Merger:
    """Class that keeps a list of byte‐pair merges and a rank‐lookup dict."""

    def __init__(self, merges: list[tuple[bytes, bytes]] | None = None):
        # Store merges in a plain list
        self.merges: list[tuple[bytes, bytes]] = []
        # The rank‐lookup dictionary
        self.merges_dict: dict[bytes, dict[bytes, int]] = {}

        # If an initial list was given, add each through our helper
        if merges:
            for a, b in merges:
                self.add_merge(a, b)

    def add_merge(self, a: bytes, b: bytes) -> None:
        """
        Append the pair (a,b) to self.merges and register it in self.merges_dict.
        The “rank” is simply the index in self.merges.
        """
        self.merges.append((a, b))
        rank = len(self.merges) - 1
        self.merges_dict.setdefault(a, {})[b] = rank

    def __call__(self, word_bytes: list[bytes]) -> list[bytes]:
        """
        Given a list of byte‐tokens, repeatedly merge the pair (a, b) with the lowest rank
        (according to self.merges_dict) until no more mergeable adjacent pairs remain.

        Uses a min‐heap to track only the candidate pairs, leading to roughly O(n log n)
        """
        n = len(word_bytes)
        if n < 2:
            return word_bytes[:]  # nothing to merge

        # Build helper arrays for a doubly‐linked structure over indices 0..n-1
        prev_idx = list(range(-1, n - 1))  # prev_idx[i] = i-1, except prev_idx[0] = -1
        next_idx = list(range(1, n + 1))  # next_idx[i] = i+1, except next_idx[n-1] = n (sentinel)
        alive = [True] * n  # alive[i] = whether token i is still “in play”

        # A small helper to push (rank, left_index) if (i, j) is a valid merge pair
        def push_pair(i: int, j: int):
            """If (word_bytes[i], word_bytes[j]) is in merges_dict, push (rank, i) into heap."""
            key_a = word_bytes[i]
            key_b = word_bytes[j]
            # merges_dict maps:   merges_dict[a][b] = rank (an integer)
            # if a not in merges_dict or b not a valid neighbor, skip
            rank = self.merges_dict.get(key_a, {}).get(key_b)
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        # Build the initial heap of all adjacent (i, i+1) pairs that exist in merges_dict
        heap: list[tuple[int, int]] = []
        for i in range(n - 1):
            push_pair(i, i + 1)

        # This counter keeps track of how many “tokens” are still alive
        # so we know when we’re down to a single dummy sentinel or no merges left.
        alive_count = n

        while heap:
            rank, i = heapq.heappop(heap)

            # If i is no longer alive, or its “next” neighbor is gone, skip
            if not alive[i]:
                continue
            j = next_idx[i]
            if j >= n or not alive[j]:
                # either j is the sentinel (n) or j has been merged away already
                continue

            # Double‐check that they still form a mergeable pair of the same rank:
            a, b = word_bytes[i], word_bytes[j]
            current_rank = self.merges_dict.get(a, {}).get(b)
            if current_rank is None or current_rank != rank:
                # either they’re no longer mergeable or their rank changed
                continue

            # ↓—— Perform the merge of indices i and j ———————————————————↓

            # 1) Concatenate b onto a, store in position i
            word_bytes[i] = a + b

            # 2) “Kill” index j by marking alive[j]=False
            alive[j] = False
            alive_count -= 1

            # 3) Splice j out of the linked structure:
            nxt = next_idx[j]
            next_idx[i] = nxt
            if nxt < n:
                prev_idx[nxt] = i

            # 4) Now i’s new neighbors might form new mergeable pairs. Push them:

            # 4a) Check if i has a “prev” neighbor
            p = prev_idx[i]
            if p >= 0 and alive[p]:
                push_pair(p, i)

            # 4b) Check if i has a “next” neighbor
            q = next_idx[i]
            if q < n and alive[q]:
                push_pair(i, q)

            # If we dropped below 2 alive tokens, there can’t be more merges
            if alive_count < 2:
                break

        # ———— Extract the surviving tokens in order ——————
        result: list[bytes] = []
        idx = 0
        # Find the first “alive” index
        while idx < n and not alive[idx]:
            idx += 1
        # Walk forward via next_idx and collect bytes
        while idx < n:
            if alive[idx]:
                result.append(word_bytes[idx])
            idx = next_idx[idx]

        return result


class BPETokenizer:
    """A BPE tokenizer that uses the provided vocab, merges, and special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merger = Merger(merges=merges)
        self.segmenter = Segmenter(special_tokens=special_tokens)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pretoken_to_id = {}
        self.special_bytes = {}
        for tok in self.segmenter.special_tokens:
            tok_bytes = tok.encode("utf-8")
            self.special_bytes[tok] = self.reverse_vocab[tok_bytes]

        self._PAT_RE = re.compile(PAT)

    def encode(self, text: str) -> list[int]:
        segments = self.segmenter(text)
        append = []
        pretoken_to_id = self.pretoken_to_id
        special_bytes = self.special_bytes
        reverse_vocab = self.reverse_vocab
        pattern = self._PAT_RE.finditer
        apply_merges = self.merger

        for is_special, token in segments:
            if is_special:
                append.append(special_bytes[token])
                continue

            for match in pattern(token):
                word = match.group(0)

                cached = pretoken_to_id.get(word)
                if cached is not None:
                    append.extend(cached)
                    continue

                # Encode UTF-8 and wrap bytes as needed
                word_bytes = [bytes([b]) for b in word.encode("utf-8")]
                merged = apply_merges(word_bytes)
                ids = [reverse_vocab[b] for b in merged]

                pretoken_to_id[word] = ids
                append.extend(ids)

        return append

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        output_bytes = b"".join(map(self.vocab.get, ids))
        return output_bytes.decode("utf-8", errors="replace")


class BPETrainer:
    """Train BPE tokenizer"""

    def __init__(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        split_special_token: bytes = b"<|endoftext|>",
        desired_num_chunks: int = 1,
    ):
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        self.input_path = input_path
        self.split_special_token = split_special_token
        self.desired_num_chunks = desired_num_chunks
        self.segmenter = Segmenter(special_tokens=special_tokens, in_bytes=True)
        self.merger = Merger()

        self.vocab = {i: bytes([i]) for i in range(256)}

        # add special tokens
        for token in self.segmenter.special_tokens:
            token_bytes = token.encode("utf-8")
            self.vocab[len(self.vocab)] = token_bytes

        self.num_merges = vocab_size - len(self.vocab)

        with open(file=input_path, mode="rb") as f:
            self.boundaries = self.find_chunk_boundaries(f)

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // self.desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(self.desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(self.split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
        """
        if self.num_merges <= 0:
            return self.vocab, []

        with open(file=self.input_path, mode="rb") as f:
            for _ in range(self.num_merges):
                counter = Counter()
                for start, end in zip(self.boundaries[:-1], self.boundaries[1:]):
                    f.seek(start)
                    chunk = f.read(end - start)
                    segments = self.segmenter(chunk)
                    for is_special, segment in segments:
                        if not is_special:
                            word_bytes = [bytes([b]) for b in segment]
                            merged = self.merger(word_bytes)
                            counter.update(zip(merged, merged[1:]))
                pair = counter.most_common(1)[0][0]
                self.merger.add_merge(*pair)
                self.vocab[len(self.vocab)] = pair[0] + pair[1]

        return self.vocab, self.merger.merges


if __name__ == "__main__":
    trainer = BPETrainer(
        input_path="tests/fixtures/tinystories_sample_5M.txt",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
        # desired_num_chunks=10,
    )
    res = trainer.train()
    print(res[1])
    # m = Merger()
    # segment = "ameli"
    # word_bytes = [bytes([b]) for b in segment.encode("utf-8")]
    # print(word_bytes)
    # o = m(word_bytes)
    # print(o)
