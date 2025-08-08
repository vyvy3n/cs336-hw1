import tiktoken
from typing import Iterator, Iterable


class OpenAITokenizer:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.model_name = model_name
        self.vocab_size = self.tokenizer.n_vocab
        
    @property
    def vocab(self):
        """Return a dict-like object that supports len() and basic operations."""
        class VocabProxy:
            def __init__(self, size):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def items(self):
                return []
                
        return VocabProxy(self.vocab_size)
    
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self, ids: list[int]) -> str:

        try:
            return self.tokenizer.decode(ids)
        except Exception:
            # Handle potential decoding errors gracefully
            return self.tokenizer.decode(ids, errors='replace')
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size
    
    def __repr__(self):
        return f"OpenAITokenizer(model='{self.model_name}', vocab_size={self.vocab_size})"