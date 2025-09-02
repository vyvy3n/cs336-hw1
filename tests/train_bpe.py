from __future__ import annotations

import os

end_of_text_str = "<|endoftext|>"
end_of_text_bin = end_of_text_str.encode()

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

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

    from cs336_basics.pretokenization_example import find_chunk_boundaries
    import regex as re

    def init_vocab():
        # return a init vocab with basic bytes from 0 ~ 255
        # append special tokens at the end of the list in sequence
        vocab = {i: bytes([i]) for i in range(256)}
        idx = len(vocab)
        for token in special_tokens:
            vocab[idx] = token.encode("utf-8")
            idx += 1
        return vocab
    
    def update_vocab(vocab: dict, pair: tuple[int]) -> tuple[dict, int]:
        # pair contains token IDs, so just get their bytes from vocab
        bytes1 = vocab[pair[0]]
        bytes2 = vocab[pair[1]]
        merged_bytes = bytes1 + bytes2  # Concatenate bytes
        idx = len(vocab)
        vocab[idx] = merged_bytes
        return vocab, idx
    
    def split_by_special_tokens(text, special_tokens):
        # Split text into segments, isolating special tokens as their own segments
        import re
        if not special_tokens:
            return [text]
        # Build a regex pattern to match any special token
        pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
        segments = []
        last_end = 0
        for match in re.finditer(pattern, text):
            start, end = match.span()
            if start > last_end:
                segments.append(text[last_end:start])
            segments.append(text[start:end])
            last_end = end
        if last_end < len(text):
            segments.append(text[last_end:])
        # Remove segments with special tokens
        return [seg for seg in segments if seg not in special_tokens]

    def split_and_count(text: str, vocab) -> dict:
        pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE,
        )
        word_count = {}
        for match in pattern.finditer(text):
            word = match.group(0)
            word_bytes = tuple(word.encode("utf-8"))
            word_count[word_bytes] = word_count.get(word_bytes, 0) + 1
        return word_count

    def get_byte_pairs(word: tuple[int]) -> list[tuple[int]]:
        pairs = []
        for i in range(1, len(word)):
            pairs.append((word[i - 1], word[i]))
        return pairs
    
    def get_byte_pair_count(word_count: dict) -> dict:
        # words: dict[word_bytes, count]
        pair_counts = {}
        for word, count in word_count.items():
            pairs = get_byte_pairs(word)
            for pair in pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        return pair_counts
            
    def get_top_pair(pair_count: dict[tuple[int], int], vocab) -> tuple[int]:
        # Find the pair(s) with the highest count
        if not pair_count:
            return None
        max_count = max(pair_count.values())
        # Get all pairs with max_count
        top_pairs = [pair for pair, count in pair_count.items() if count == max_count]
        # Sort by the concatenated bytes value from vocab, descending
        top_pairs.sort(key=lambda pair: (vocab[pair[0]], vocab[pair[1]]), reverse=True)
        return top_pairs[0]
    
    def update_word_encode(word_count: dict, top_pair: tuple[int], idx: int) -> dict:
        new_word_count = {}
        for word, count in word_count.items():
            new_word = []
            i = 0
            while i < len(word):
                # Check if the next two elements match top_pair
                if i < len(word) - 1 and (word[i], word[i + 1]) == top_pair:
                    new_word.append(idx)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)
            new_word_count[new_word_tuple] = new_word_count.get(new_word_tuple, 0) + count
        return new_word_count

    # Open the file in binary mode
    with open(input_path, "rb") as f:
        vocab = init_vocab()
        merges = []
        word_count = {}
        
        num_chunks = 10
        boundaries = find_chunk_boundaries(f, num_chunks, end_of_text_bin)
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            texts = split_by_special_tokens(chunk, special_tokens)
            for text in texts:
                for word, count in split_and_count(text, vocab).items():
                    word_count[word] = word_count.get(word, 0) + count

        from tqdm import trange
        for _ in trange(len(vocab), vocab_size, desc="BPE Training Progress"):
            pair_count = get_byte_pair_count(word_count)
            top_pair = get_top_pair(pair_count, vocab)
            vocab, idx = update_vocab(vocab, top_pair)
            word_count = update_word_encode(word_count, top_pair, idx)
            merges.append((vocab[top_pair[0]], vocab[top_pair[1]]))

    return vocab, merges


if __name__ == "__main__":
    import argparse
    from common import FIXTURES_PATH

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to BPE tokenizer training data.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=500,
        help="Total number of items in the tokenizer's vocabulary (including special tokens).",
    )
    parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="List of special tokens to add to the tokenizer vocabulary.",
    )

    args = parser.parse_args()

    # Use fixed input if not given
    # input_path = args.input_path if args.input_path is not None else (FIXTURES_PATH / "corpus.en")
    input_path = (
        args.input_path
        if args.input_path is not None
        else (FIXTURES_PATH / "tinystories_sample_5M.txt")
    )

    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    print("Vocab:", vocab)
    print("Merges:", merges)
    print("Merges:", merges)
