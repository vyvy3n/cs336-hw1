import os
import multiprocessing as mp
from collections import Counter
import datetime
import regex as re

class BPE:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str] = None, num_processes: int = 10, mini_chunk_size: int = 4096,
                 desired_num_chunks: int = 40):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = ['<|endoftext|>'] + (special_tokens or [])
        self.split_special = re.compile('|'.join([re.escape(token) for token in self.special_tokens]))
        self.vocab = {}
        self.merges = []
        self.num_processes = num_processes
        self.desired_num_chunks = desired_num_chunks
        self.mini_chunk_size = mini_chunk_size
        self.data_chunks_boundaries = self._split_chunks()
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", flags=re.UNICODE)

        self._init_fill_vocab()

    def _init_fill_vocab(self):
        for i in range(256):
            self.vocab[i] = bytes([i])

        next_id = len(self.vocab)

        for token_str in self.special_tokens:
            self.vocab[next_id] = token_str.encode("utf-8")
            next_id += 1


    def get_vocab(self) -> dict[int, bytes]:
        return self.vocab

    def _split_chunks(self) -> list[int]:
        start_time = datetime.datetime.now()
        split_special_token = self.special_tokens[0].encode('utf-8')
        print('START SPLITTING INPUT FILE INTO CHUNKS')
        print(f'Input file: {self.input_path}')
        print(f'Desired number of chunks: {self.desired_num_chunks}')
        print(f'Mini chunk size: {self.mini_chunk_size}')
        print(f'Special token: {split_special_token}')
        assert isinstance(split_special_token, bytes), 'Must represent special token as a bytestring'
        with open(self.input_path, 'rb') as file:

            # Get total file size in bytes
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            chunk_size = file_size // self.desired_num_chunks

            # Initial guesses for chunk boundary locations, uniformly spaced
            # Chunks start on previous index, don't include last index
            chunk_boundaries = [i * chunk_size for i in range(self.desired_num_chunks + 1)]
            chunk_boundaries[-1] = file_size

            for bi in range(1, len(chunk_boundaries) - 1):
                initial_position = chunk_boundaries[bi]
                file.seek(initial_position)  # Start at boundary guess
                while True:
                    mini_chunk = file.read(self.mini_chunk_size)  # Read a mini chunk

                    # If EOF, this boundary should be at the end of the file
                    if mini_chunk == b"":
                        chunk_boundaries[bi] = file_size
                        break

                    # Find the special token in the mini chunk
                    found_at = mini_chunk.find(split_special_token)
                    if found_at != -1:
                        chunk_boundaries[bi] = initial_position + found_at
                        break
                    initial_position += self.mini_chunk_size

            end_time = datetime.datetime.now()
            print(f'Time to split input file into chunks: {end_time - start_time}')
            # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
            return sorted(set(chunk_boundaries))


    def _process_chunk_and_get_pairs(self, args):
        """
        This worker does everything: reads a chunk, gets word counts,
        and returns the final pair counts for that chunk.
        """
        start, end = args

        # --- Part 1: Get word counts (from your _process_chunk) ---
        start_time = datetime.datetime.now()
        word_counts = Counter()
        with open(self.input_path, 'rb') as file:
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            chunks_by_special_tokens = self.split_special.split(chunk)
            for special_chunk in chunks_by_special_tokens:
                words_generator = (m.group(0) for m in self.PAT.finditer(special_chunk))
                word_counts.update(words_generator)
        end_time = datetime.datetime.now()
        print(f'Time to process chunk and get word counts: {end_time - start_time}')

        # --- Part 2: Get pair counts (from your _get_pairs_from_chunk) ---
        start_time = datetime.datetime.now()
        pair_counts = Counter()
        for word_str, count in word_counts.items():
            word_bytes = word_str.encode('utf-8')
            if len(word_bytes) < 2:
                continue
            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                pair_counts[pair] += count
        end_time = datetime.datetime.now()
        print(f'Time to get pairs from chunk: {end_time - start_time}')

        return pair_counts

    def run_BPE(self):
        print(f'Number of actual chunks {len(self.data_chunks_boundaries)-1} chunks.')
        print(f'Chunk boundaries: {self.data_chunks_boundaries}')

        available_cores = mp.cpu_count()
        print(f'Available cores: {available_cores}')
        print(f'Desired number of processes: {self.num_processes}')

        chunk_args = []
        for start, end in zip(self.data_chunks_boundaries[:-1], self.data_chunks_boundaries[1:]):
            chunk_args.append((start, end))

        print("\nStarting combined processing pool...")
        start_time = datetime.datetime.now()
        with mp.Pool(self.num_processes) as pool:
            list_of_pair_counters = pool.map(self._process_chunk_and_get_pairs, chunk_args)
        end_time = datetime.datetime.now()
        print(f'Time to process all chunks and get pairs: {end_time - start_time}')

        start_time = datetime.datetime.now()
        final_pair_counts = Counter()
        for counter in list_of_pair_counters:
            final_pair_counts.update(counter)
        end_time = datetime.datetime.now()
        print(f'Most common pairs: {final_pair_counts.most_common(15)}')
        print(f'Time to combine all pair counts: {end_time - start_time}')




if __name__ == '__main__':
    start = datetime.datetime.now()
    # test = BPE(input_path='../data/debug_small_text.txt', vocab_size=256)
    test = BPE(input_path='../data/owt_train.txt', vocab_size=256)
    # print(test.get_vocab())
    test.run_BPE()
    end = datetime.datetime.now()
    print(f'Total time: {end - start}')



