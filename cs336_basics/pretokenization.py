from functools import reduce
import os
import regex as re
from typing import BinaryIO

PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
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
            found_at = -1
            for token in split_special_tokens:
                found_at = mini_chunk.find(token)
                if found_at != -1:
                    break
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_pretokens(
    chunk: str
) -> dict[tuple[bytes], int]:
    ret = {}
    for pretoken in re.finditer(PRETOKENIZATION_PATTERN, chunk):
        s = pretoken.group(0).encode("utf-8")
        if s in ret:
            ret[s] += 1
        else:
            ret[s] = 1
    return ret


def merge_pretoken_counts(
    d1: dict[tuple[bytes]],
    d2: dict[tuple[bytes]],
) -> dict[tuple[bytes], int]:
    for k, v in d2.items():
        if k in d1:
            d1[k] += v
        else:
            d1[k] = v
    return d1


def get_pretokenizaiton_counts(
    chunk: str,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    chunk_splitted = re.split('|'.join(special_tokens), chunk)
    return reduce(merge_pretoken_counts, map(count_pretokens, chunk_splitted))
