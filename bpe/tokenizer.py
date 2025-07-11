
class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.vocab = self._build_vocab()
    
    def _build_vocab(self):
        # The base vocab is the 256 bytes.
        vocab = {i: bytes([i]) for i in range(256)}
        
        # This part handles rebuilding from existing merges if you were loading a saved tokenizer.
        for (id1, id2), merged_id in self.merges.items():
            vocab[merged_id] = vocab[id1] + vocab[id2]
            
        # The inverse vocab should always be built after the vocab is finalized.
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        return vocab
    
    def add_special_tokens(self, tokens: list[str]):
        """
        Adds special tokens to the vocabulary in-place and rebuilds the inverse vocabulary.
        """
        for token in tokens:
            # Add the new token if it doesn't already exist.
            if token.encode("utf-8") not in self.vocab.values():
                 self.vocab[len(self.vocab)] = token.encode("utf-8")
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}