"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py

But:
- Does not handle the regular expression splitting pattern.
"""

from .tokenizer import Tokenizer
import regex as re

class RegexTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)
        

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256 - len(self.special_tokens)
        
    def add_special_tokens(self, tokens: list[str]):
        """
        Creates a dictionary for special tokens, assigning them a unique integer ID.

        Args:
            tokens: A list of special token strings.

        Returns:
            A dictionary mapping the integer ID to the special token.
        """
        # TODO: vocab_size need to be smaller than 100257
        self.special_tokens = {i + 100257: token for i, token in enumerate(tokens)}