import regex as re
from typing import Iterator, Iterable


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.encoder = {v: k for k, v in vocab.items()}
        
    @classmethod    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t', 1)
                    token_id = int(parts[0])
                    token_bytes = parts[1].encode('utf-8').decode('unicode_escape').encode('latin1')
                    vocab[token_id] = token_bytes

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    left = parts[0].encode('utf-8').decode('unicode_escape').encode('latin1')
                    right = parts[1].encode('utf-8').decode('unicode_escape').encode('latin1')
                    merges.append((left, right))
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        regexed = re.findall(PAT, text)
        
        result = []
        for token in regexed:
            encoded_token = list(token.encode('utf-8'))
            
            for left_bytes, right_bytes in self.merges:
                left_id = self.encoder.get(left_bytes)
                right_id = self.encoder.get(right_bytes)
                merged_bytes = left_bytes + right_bytes
                merged_id = self.encoder.get(merged_bytes)
                
                if left_id is not None and right_id is not None and merged_id is not None:
                    i = 0
                    while i < len(encoded_token) - 1:
                        if encoded_token[i] == left_id and encoded_token[i + 1] == right_id:
                            encoded_token = encoded_token[:i] + [merged_id] + encoded_token[i + 2:]
                        else:
                            i += 1
            
            result.extend(encoded_token)
        
        return result
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
        
        all_bytes = b''.join(byte_chunks)
        try:
            return all_bytes.decode('utf-8', errors='replace')
        except:
            return all_bytes.decode('utf-8', errors='replace')
    