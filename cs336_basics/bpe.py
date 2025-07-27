import os
from typing import BinaryIO
import regex as re
from collections import Counter
from cs336_basics.common import write_vocab_to_file, read_vocab_from_file, write_merges_to_file, read_merges_from_file
from tests.common import FIXTURES_PATH

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def merge_key(key: tuple[bytes], bytes_pair: tuple[bytes, bytes], new_bytes: bytes) -> tuple[bytes]:
    i = 0
    L = len(key)
    result = []
    while i < L:
        if i < L - 1 and key[i] == bytes_pair[0] and key[i + 1] == bytes_pair[1]:
            result.append(new_bytes)
            i += 2
        else:
            result.append(key[i])
            i += 1
    return tuple(result)


def update_word_counts(word_counts: dict[tuple[bytes, ...], int], bytes_pair: tuple[bytes, bytes], bytes_pair_counts: Counter) -> None:
    new_word_counts = {}
    new_bytes = bytes_pair[0] + bytes_pair[1]

    for key, count in word_counts.items():
        new_key = merge_key(key, bytes_pair, new_bytes)
        if new_key != key:
            for i in range(len(key) - 1):
                pair = (key[i], key[i + 1])
                bytes_pair_counts[pair] -= count
            for i in range(len(new_key) - 1):
                pair = (new_key[i], new_key[i + 1])
                bytes_pair_counts[pair] += count
        new_word_counts[new_key] = new_word_counts.get(new_key, 0) + count

    word_counts.clear()
    word_counts.update(new_word_counts)
    

def get_bytes_pair_counts(word_counts: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]:
    """
    Get counts of all byte pairs in the word counts.
    """
    bytes_pair_counts = Counter()
    for key, value in word_counts.items():
        for i in range(len(key) - 1):
            bytes_pair = (key[i], key[i + 1])
            bytes_pair_counts[bytes_pair] += value
    return bytes_pair_counts


def remove_special_tokens(chunk:str, special_tokens:list[str])->str:
    escaped = [re.escape(t) for t in special_tokens]
    return re.split("|".join(escaped), chunk)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_processes = 2
    vocab = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    
    word_counts = {}
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            for sub_chunk in remove_special_tokens(chunk, special_tokens):
                words = re.findall(PAT, sub_chunk)
                for word in words:
                    key = tuple([bytes([x]) for x in word.encode('utf-8')])
                    word_counts[key] = word_counts.get(key, 0) + 1

    # merge until size of vocab reach vocab_size
    merged_tuples = []
    bytes_pair_counts = get_bytes_pair_counts(word_counts)
    while len(vocab) < vocab_size:
        # find the most freq tuple
        best_tuple = max(bytes_pair_counts.items(), key = lambda x: (x[1], x[0]))[0]
        merged_tuples.append(best_tuple)
        # update vocab
        vocab[len(vocab)] = best_tuple[0] + best_tuple[1]
        # replace best_tuple with new bytes
        update_word_counts(word_counts, best_tuple, bytes_pair_counts)            
    return vocab, merged_tuples


def test_read_write_vocab_merges():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    
    # Write vocab and merges to files
    write_vocab_to_file(vocab, "vocab.txt")
    write_merges_to_file(merges, "merges.txt")
    
    # Read back from files
    vocab2 = read_vocab_from_file("vocab.txt")
    merges2 = read_merges_from_file("merges.txt")
    
    assert vocab == vocab2
    assert merges == merges2


if __name__=="__main__":
    test_read_write_vocab_merges()
    print("Training BPE completed successfully.")
