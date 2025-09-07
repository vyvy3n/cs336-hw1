import os
import regex as re
import logging
from typing import BinaryIO, Dict, List, Tuple


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    logging.log(logging.INFO, f"File size: {file_size} bytes, desired chunks: {desired_num_chunks}")

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
    final_boundaries = sorted(set(chunk_boundaries))
    logging.log(logging.INFO, f"Found {len(final_boundaries)-1} chunks with boundaries: {final_boundaries}")
    return final_boundaries


def pretokenize_text(text: str) -> List[str]:
    """
    Pre-tokenize text using GPT-2 regex pattern.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = [match.group() for match in re.finditer(PAT, text)]
    logging.log(logging.INFO, f"Pre-tokenized text into {len(pretokens)} tokens")
    return pretokens


def count_pretokens(text: str) -> Dict[Tuple[bytes, ...], int]:
    """
    Pre-tokenize text and count frequency of each pre-token as bytes.
    
    Returns:
        Dictionary mapping (byte1, byte2, ...) tuples to counts
    """
    pretokens = pretokenize_text(text)
    counts = {}
    
    for pretoken in pretokens:
        # Convert to bytes
        pretoken_bytes = pretoken.encode('utf-8')
        # Convert to tuple of individual bytes
        byte_tuple = tuple(bytes([b]) for b in pretoken_bytes)
        counts[byte_tuple] = counts.get(byte_tuple, 0) + 1
    
    logging.log(logging.INFO, f"Counted {len(counts)} unique pre-tokens")
    return counts


## Usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
        num_processes = 4
        logging.log(logging.INFO, f"Starting chunking with {num_processes} processes")
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        all_counts = {}
        chunk_sizes = []
        
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_sizes.append(len(chunk))
            logging.log(logging.INFO, f"Processing chunk {i+1}/{len(boundaries)-1}, size: {len(chunk)} chars")
            
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_counts = count_pretokens(chunk)
            
            # Merge counts from this chunk into the total
            for pretoken, count in chunk_counts.items():
                all_counts[pretoken] = all_counts.get(pretoken, 0) + count
        
        logging.log(logging.INFO, f"Processed {len(boundaries)-1} chunks, total unique pre-tokens: {len(all_counts)}")
        logging.log(logging.INFO, f"Chunk sizes: {chunk_sizes}")

    logging.log(logging.INFO, "Top 10 most frequent pre-tokens:")
    for k, v in sorted(all_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(k, v)

    """
    2025-09-07 22:14:54,793 - INFO - Top 10 most frequent pre-tokens:
    (b'.',) 421616
    (b',',) 235432
    (b' ', b't', b'h', b'e') 211031
    (b' ', b'a', b'n', b'd') 196057
    (b' ', b'a') 152161
    (b'\n',) 152067
    (b' ', b't', b'o') 150493
    (b' ', b'w', b'a', b's') 108019
    (b' ', b'T', b'h', b'e', b'y') 52425
    (b' ', b'i', b't') 51670
    """