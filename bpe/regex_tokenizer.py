"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py

But:
- Does not handle the regular expression splitting pattern.
"""

from .tokenizer import Tokenizer
from .merge_common_pair import (
    get_stats,
    merge,
    update_cache,
    remove_from_cache,
    generate_global_pair_cnt,
)
import regex as re
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import Counter
import operator


class RegexTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)
        self.text_chunks = []

    def train(
        self,
        vocab_size: int,
        global_subword_count: Dict[Tuple[int, ...], int],
        verbose=False,
    ):
        assert vocab_size >= 256
        num_merges = vocab_size - len(self.vocab)
        subword_to_pair_cnt_cache = {}
        global_pair_count = Counter()
        for subword, subword_freq in global_subword_count.items():
            update_cache(
                subword=subword,
                subword_to_pair_cnt_cache=subword_to_pair_cnt_cache,
                subword_freq=subword_freq,
                global_pair_cnt=global_pair_count,
            )
        for i in tqdm(range(num_merges), desc="Training BPE"):
            pair = max(
                global_pair_count,
                key=lambda p: (
                    global_pair_count[p],
                    self.vocab[p[0]],
                    self.vocab[p[1]],
                ),
            )
            old_occurance = global_pair_count[pair]
            idx = len(self.vocab)
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges[pair] = idx

            affected_words = [
                word
                for word, pairs in subword_to_pair_cnt_cache.items()
                if pair in pairs
            ]

            for word_to_update in affected_words:
                word_freq = global_subword_count.get(word_to_update, 0)
                remove_from_cache(
                    subword=word_to_update,
                    subword_to_pair_cnt_cache=subword_to_pair_cnt_cache,
                    subword_freq=word_freq,
                    global_pair_cnt=global_pair_count,
                )
                merged_word = tuple(merge(ids=word_to_update, pair=pair, idx=idx))
                global_subword_count[merged_word] = word_freq
                update_cache(
                    subword=merged_word,
                    subword_to_pair_cnt_cache=subword_to_pair_cnt_cache,
                    subword_freq=word_freq,
                    global_pair_cnt=global_pair_count,
                )

            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: ({self.vocab[pair[0]]},{self.vocab[pair[1]]}) -> {idx} ({self.vocab[idx]}) had {old_occurance} occurrences"
                )
