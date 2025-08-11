import numpy as np
import pathlib
import random
import time

from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics.pretokenization import find_chunk_boundaries

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data" 
TINY_STORIES_PATH = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
OWT_PATH = DATA_PATH / "owt_train.txt"

TINY_STORIES_TOKENIZER_VOCAB_PATH = DATA_PATH / "TinyStoriesV2_vocab.txt"
TINY_STORIES_TOKENIZER_MERGES_PATH = DATA_PATH / "TinyStoriesV2_merges.txt"
OWT_TOKENIZER_VOCAB_PATH = DATA_PATH / "owt_train_vocab.txt"
OWT_TOKENIZER_MERGES_PATH = DATA_PATH / "owt_train_merges.txt"

TINY_STORIES_TRAIN_PATH = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
TINY_STORIES_TRAIN_TOKENIZED_PATH = DATA_PATH / "TinyStoriesV2-GPT4-train-tokenized.npy"
TINY_STORIES_VALID_PATH = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
TINY_STORIES_VALID_TOKENIZED_PATH = DATA_PATH / "TinyStoriesV2-GPT4-valid-tokenized.npy"

OWT_TRAIN_PATH = DATA_PATH / "owt_train.txt"
OWT_VALID_PATH = DATA_PATH / "owt_valid.txt"


def sample_ten_documents(
    file_path: str,
    tokenizer: BpeTokenizer,
    verbose: bool = False,
    num_slices: int = 1,
):
    with open(file_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_slices, ["<|endoftext|>".encode("utf-8")])
        text = f.read(boundaries[1] - boundaries[0]).decode('utf-8')
        chunks = text.split("<|endoftext|>")
        for doc in random.sample(chunks, 10):
            tokens = tokenizer.encode(doc)
            if verbose:
                print(doc[0:100], '...')
                print(tokens[0:20], '...')
            bytes_len = len(doc.encode("utf-8"))
            print(f'len of doc = {len(doc)}, len of bytes = {bytes_len}, len of tokens = {len(tokens)}, compression_ratio = {bytes_len * 1.0 / len(tokens)}\n')


def sample_runs(tiny_stories_tokenizer: BpeTokenizer, owt_tokenizer: BpeTokenizer):
    print('tokenizing tiny stories with tiny stories tokenizer...')
    sample_ten_documents(TINY_STORIES_PATH, tiny_stories_tokenizer, num_slices=5)
    print('tokenizing owt with owt tokenizer...')
    sample_ten_documents(OWT_PATH, owt_tokenizer, num_slices=20)

    # mixed and match...
    print('tokenizing tiny stories with owt tokenizer...')
    sample_ten_documents(TINY_STORIES_PATH, owt_tokenizer, num_slices=5)
    print('tokenizing owt with tiny stories tokenizer...')
    sample_ten_documents(OWT_PATH, tiny_stories_tokenizer, num_slices=20)


def tokenize_dataset(
    input_path: str,
    output_path: str,
    tokenizer: BpeTokenizer
):
    start_time = time.time()
    print(f"tokenizing {input_path} and saving results to {output_path}...")
    with open(input_path, 'r') as f:
        tokens = [token for token in tokenizer.encode_iterable(f)]
        np.save(output_path, np.array(tokens, dtype = np.uint16))
    end_time = time.time()
    print(f'runtime = {end_time - start_time}')


if __name__ == '__main__':
    tiny_stories_tokenizer = BpeTokenizer.from_files(
        vocab_filepath=TINY_STORIES_TOKENIZER_VOCAB_PATH,
        merges_filepath=TINY_STORIES_TOKENIZER_MERGES_PATH,
        special_tokens=["<|endoftext|>"])
    
    owt_tokenizer = BpeTokenizer.from_files(
        vocab_filepath=OWT_TOKENIZER_VOCAB_PATH,
        merges_filepath=OWT_TOKENIZER_MERGES_PATH,
        special_tokens=["<|endoftext|>"])

    # sample_runs(tiny_stories_tokenizer, owt_tokenizer)

    tokenize_dataset(TINY_STORIES_VALID_PATH, TINY_STORIES_VALID_TOKENIZED_PATH, tiny_stories_tokenizer)
    tokenize_dataset(TINY_STORIES_TRAIN_PATH, TINY_STORIES_TRAIN_TOKENIZED_PATH, tiny_stories_tokenizer)