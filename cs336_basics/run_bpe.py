import pathlib
import time

from tests.adapters import run_train_bpe

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data" 

def train_on_tiny_stories():
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    with open('TinyStoriesV2_dump.txt', 'w') as f:
        f.write(str(vocab) + '\n')
        f.write(str(merges) + '\n')
        f.write(f'longest vocab = {max(vocab.values(), key = len)}')
    return (vocab, merges)

def train_on_owt():
    input_path = DATA_PATH / "owt_train.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    with open('owt_train_dump.txt', 'w') as f:
        f.write(str(vocab) + '\n')
        f.write(str(merges) + '\n')
        f.write(f'longest vocab = {max(vocab.values(), key = len)}')
    return (vocab, merges)

if __name__ == '__main__':
    start_time = time.time()
    (vocab, merges) = train_on_tiny_stories()
    end_time = time.time()
    print(f'runtime = {end_time - start_time}')

    start_time = time.time()
    (vocab, merges) = train_on_owt()
    end_time = time.time()
    print(f'runtime = {end_time - start_time}')

