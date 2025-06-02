import regex as re
from collections.abc import Iterable, Iterator
import heapq


# Parsing pattern used in GPT2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Segmenter:
    """This class scans the raw input string, finds occurrences of any of the “special tokens”
    (by working in raw UTF-8 bytes), and returns a sequence of segments"""

    def __init__(self, special_tokens: list[str] | None = None):
        self.special_tokens = special_tokens or []

        # Build a simple byte‐trie for special tokens
        # (each key is a UTF‐8 string; store bytes-list for matching)
        self._special_trie = {}
        for tok in self.special_tokens:
            node = self._special_trie
            tok_bytes = tok.encode("utf-8")
            for b in tok_bytes:
                node = node.setdefault(b, {})
            node["_end"] = tok  # mark end‐of‐special‐token at this node

    def __call__(self, text: str) -> list[tuple[bool, str]]:
        """
        Get the list of segments (bool, str) from the text.
        """
        data = text.encode("utf-8")  # work in bytes for exact matches

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
                    result.append((False, buffer_bytes.decode("utf-8", "replace")))
                    buffer_bytes = bytearray()
                # append the special token
                result.append((True, last_match))
                i = last_pos
            else:
                # no special token starts here
                buffer_bytes.append(data[i])
                i += 1

        if buffer_bytes:
            result.append((False, buffer_bytes.decode("utf-8", "replace")))
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
