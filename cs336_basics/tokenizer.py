import pickle
import json
from tqdm import tqdm
import regex as re
from typing import Iterable

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):

        self.vocab = vocab
        self.merges = merges

        self.special_tokens = special_tokens or []
        self.PRETOKENIZE_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[next_id] = token_bytes
                next_id += 1

        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):

        if vocab_filepath.endswith('.pkl'):
            with open(vocab_filepath, 'rb') as f:
                vocab = pickle.load(f)
        elif vocab_filepath.endswith('.json'):
            with open(vocab_filepath, 'r', encoding='utf-8') as f:
                vocab_json = json.load(f)
                vocab = {int(k): bytes.fromhex(v) for k, v in vocab_json.items()}
        else:
            raise ValueError("Vocabulary file must be .pkl or .json")

        if merges_filepath.endswith('.pkl'):
            with open(merges_filepath, 'rb') as f:
                merges = pickle.load(f)
        elif merges_filepath.endswith('.txt'):
            merges = []
            with open(merges_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        a_hex, b_hex = line.split()
                        merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))
        else:
            raise ValueError("Merges file must be .pkl or .txt")

        tokenizer = cls(vocab, merges, special_tokens)
        return tokenizer

    def encode(self, text) -> list[int] :

        token_ids = []

        if self.special_tokens:
            pattern = "(" + "|".join(
                re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                token_ids.append(self.vocab_reverse[part.encode("utf-8")])
                continue

            for m in re.finditer(self.PRETOKENIZE_REGEX, part):
                text_bytes = m.group(0).encode("utf-8")
                tokens = [text_bytes[i:i+1] for i in range(len(text_bytes))]

                for a, b in self.merges:
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i] == a and tokens[i+1] == b:
                            merged = a + b
                            tokens[i:i+2] = [merged]
                        i += 1

                
                for token in tokens:
                    if token in self.vocab_reverse:
                        token_ids.append(self.vocab_reverse[token])
                    # For Safety
                    else:
                        for byte in token:
                            if byte in self.vocab_reverse:
                                token_ids.append(self.vocab_reverse[byte])
                            else:
                                raise ValueError(f"Unknown token: {token}")
            
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:

        for text in tqdm(iterable, desc="Streaming encoding"):
            token_ids = self.encode(text)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:

        bytes_sequence = []
        for id in ids:
            if id in self.vocab:
                bytes_sequence.append(self.vocab[id])
            else:
                raise ValueError(f"Unknown token ID: {id}")

        full_bytes = b"".join(bytes_sequence)
        return full_bytes.decode("utf-8", errors="replace")




                

        