# the base Tokenizer class
class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
        
    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError
    
    def _build_vocab(self):
        vocab =  {i : chr(i) for i in range(256)}
        for (id1, id2), merged_id in self.merges.items():
            vocab[merged_id] = vocab[id1]+vocab[id2]
        for text, id in self.special_tokens:
            vocab[id] = text.encode("utf-8")
        return vocab