import multiprocessing as mp
import os
from cs336_basics.pretokenization import find_chunk_boundaries, get_pretokenizaiton_counts, merge_pretoken_counts
from functools import partial, reduce


def get_chunk_pretoken_counts(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    range: tuple[int],
):
    with open(input_path, "rb") as f:
        f.seek(range[0])
        chunk = f.read(range[1] - range[0]).decode("utf-8", errors="ignore")
        return get_pretokenizaiton_counts(chunk, special_tokens)


def get_pretoken_counts(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        num_processes = 20
        boundaries = find_chunk_boundaries(f, num_processes, [a.encode("utf-8") for a in special_tokens])

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(partial(get_chunk_pretoken_counts, input_path, special_tokens), zip(boundaries[:-1], boundaries[1:]))
            pretoken_counts = reduce(lambda a, b: merge_pretoken_counts(a, b), results)
            print(list(pretoken_counts.items())[:100])
            return pretoken_counts


class Bpe:
    def __init__(
        self
    ):
        pass


    class TokenPointer:
        def __init__(
            self,
            t: int,
            count: int,
        ):
            self.t = t
            self.count = count
            self.prev = None
            self.next = None


    class TokenPointerPair:
        def __init__(
            self,
            token_pointer1,
            token_pointer2,
        ):
            self.token_pointer1 = token_pointer1
            self.token_pointer2 = token_pointer2


        def __hash__(self):
            return hash((self.token_pointer1, self.token_pointer2))
        
        
        def __eq__(self, other):
            return self.token_pointer1 == other.token_pointer1 and self.token_pointer2 == other.token_pointer2


    def encode(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        pretoken_counts = get_pretoken_counts(input_path, special_tokens)

        merges = []

        # initiate vocab
        vocab = {}
        vocab_idx = {}
        for t in special_tokens:
            vocab[len(vocab)] = t.encode("utf-8")
        for i in range(256):
            vocab[len(vocab)] = bytes([i])
        vocab_idx = {v: k for k, v in vocab.items()}
        
        token_pair_counts = {}
        token_pair_to_pointers = {}
        for pretoken, pc in pretoken_counts.items():
            token_list = []
            for t in pretoken:
                token_list.append(self.TokenPointer(vocab_idx[bytes([t])], pc))
            for i in range(len(token_list) - 1):
                token_list[i + 1].prev = token_list[i]
                token_list[i].next = token_list[i + 1]
                k = (vocab_idx[bytes([pretoken[i]])], vocab_idx[bytes([pretoken[i + 1]])])
                if k not in token_pair_to_pointers:
                    token_pair_to_pointers[k] = set()
                token_pair_to_pointers[k].add(self.TokenPointerPair(token_list[i], token_list[i + 1]))
                if k not in token_pair_counts:
                    token_pair_counts[k] = 0
                token_pair_counts[k] += pc

        while len(vocab) < vocab_size:
            t1, t2 = 0, 0
            max_cnt = 0
            for k, cnt in token_pair_counts.items():
                if cnt > max_cnt or (cnt == max_cnt and (vocab[t1], vocab[t2]) < (vocab[k[0]], vocab[k[1]])):
                    t1, t2 = k
                    max_cnt = cnt
            if max_cnt == 0:
                print('Early exiting...')
                break
            new_token = len(vocab)
            vocab[new_token] = vocab[t1] + vocab[t2]
            merges.append((vocab[t1], vocab[t2]))

            # print(f'#{len(merges)}: merging {t1} and {t2}, count={max_cnt}, yielding: {vocab[t1], vocab[t2]}')
            for token_pointer_pair in token_pair_to_pointers[(t1, t2)].copy():
                r1 = token_pointer_pair.token_pointer1
                r2 = token_pointer_pair.token_pointer2
                token_pair_counts[(t1, t2)] -= r1.count
                if r1.next != r2 or r2.prev != r1:
                    # when t1 is equal to t2 and there are consecutive elements, this could happen.
                    continue
                r0 = r1.prev
                r3 = r2.next
                new_pointer = self.TokenPointer(new_token, r1.count)
                if r0:
                    token_pair_counts[(r0.t, r1.t)] -= r1.count
                    token_pair_to_pointers[(r0.t, r1.t)].remove(self.TokenPointerPair(r0, r1))
                    new_k = (r0.t, new_token)
                    if new_k not in token_pair_to_pointers:
                        token_pair_to_pointers[new_k] = set()
                        token_pair_counts[new_k] = 0
                    token_pair_counts[new_k] += r1.count
                    token_pair_to_pointers[new_k].add(self.TokenPointerPair(r0, new_pointer))
                    r0.next = new_pointer
                    new_pointer.prev = r0
                if r3:
                    token_pair_counts[(r2.t, r3.t)] -= r1.count
                    token_pair_to_pointers[(r2.t, r3.t)].remove(self.TokenPointerPair(r2, r3))
                    new_k = (new_token, r3.t)
                    if new_k not in token_pair_to_pointers:
                        token_pair_to_pointers[new_k] = set()
                        token_pair_counts[new_k] = 0
                    token_pair_counts[new_k] += r1.count
                    token_pair_to_pointers[new_k].add(self.TokenPointerPair(new_pointer, r3))
                    r3.prev = new_pointer
                    new_pointer.next = r3

            token_pair_to_pointers.pop((t1, t2))
            # print(f'count remaining: {token_pair_counts[(t1, t2)]}')

        return (vocab, merges)
    