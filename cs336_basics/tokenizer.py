import regex as re
from collections.abc import Iterable, Iterator


# Parsing pattern used in GPT2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_bytes_temp = {}
        self.pretoken_to_id = {}
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            self.special_bytes_temp[tok] = self.reverse_vocab[tok_bytes]

        # need to reverse special bytes so that we consider largest special tokens first
        self.special_bytes = dict(sorted(self.special_bytes_temp.items(), key=lambda item: len(item[0]), reverse=True))

        self.merges_dict = {}
        for rank, (a, b) in enumerate(self.merges):
            self.merges_dict.setdefault(a, {})[b] = rank

        self._PAT_RE = re.compile(PAT)

    def segment_special_tokens(self, text: str) -> list[tuple[bool, str]]:
        i = 0
        result = []
        buffer = []
        n = len(text)

        while i < n:
            match = False
            for tok in self.special_bytes:
                if text.startswith(tok, i):
                    # first flush normal text
                    if buffer:
                        result.append((False, "".join(buffer)))
                        buffer = []
                    # then add the special token
                    result.append((True, tok))
                    i += len(tok)
                    match = True
                    break

            if not match:
                buffer.append(text[i])
                i += 1
        if buffer:
            result.append((False, "".join(buffer)))
        return result

    def apply_merges(self, word_bytes: list[bytes]) -> list[bytes]:
        while True:
            best_rank = None
            best_pos = None
            for i in range(len(word_bytes) - 1):
                a, b = word_bytes[i], word_bytes[i + 1]
                rank = self.merges_dict.get(a, {}).get(b)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos = rank, i
            if best_pos is None:
                break
            word_bytes[best_pos : best_pos + 2] = [word_bytes[best_pos] + word_bytes[best_pos + 1]]
        return word_bytes

    def encode(self, text: str) -> list[int]:
        # first segment the special tokens
        segments = self.segment_special_tokens(text)
        # then encode the text
        encoded = []
        for is_special, token in segments:
            if is_special:
                tok_id = self.special_bytes[token]
                encoded.append(tok_id)
            else:
                for word_match in self._PAT_RE.finditer(token):
                    word = word_match.group(0)
                    cached_ids = self.pretoken_to_id.get(word)
                    if cached_ids is not None:
                        encoded.extend(cached_ids)
                        continue
                    word_bytes = [bytes([b]) for b in word.encode("utf-8")]
                    word_bytes = self.apply_merges(word_bytes)
                    tok_ids = [self.reverse_vocab[b] for b in word_bytes]
                    self.pretoken_to_id[word] = tok_ids
                    encoded.extend(tok_ids)
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        output_bytes = b"".join(map(self.vocab.get, ids))
        return output_bytes.decode("utf-8", errors="replace")
