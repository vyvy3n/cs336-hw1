from cs336_basics.bpe import Bpe
from collections.abc import Iterable, Iterator
from cs336_basics.pretokenization import PRETOKENIZATION_PATTERN
import ast
from functools import reduce
import json
import multiprocessing as mp
import regex as re


ENCODE_ITERABLE_CHUNK_SIZE = 2**22


class BpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key = len, reverse = True) if special_tokens else None
        self.vocab_index = {v : k for k, v in vocab.items()}
        self.indexed_merges = [(self.vocab_index[b1], self.vocab_index[b2]) for b1, b2 in self.merges]


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab, merges = None, None
        with open(vocab_filepath, 'r') as vocab_f:
            vocab = ast.literal_eval(vocab_f.read())
        with open(merges_filepath) as merges_f:
            merges = ast.literal_eval(merges_f.read())

        return BpeTokenizer(
            vocab = vocab,
            merges = merges,
            special_tokens = special_tokens
        )


    class BytePointer:
        def __init__(
            self,
            value: int,
            is_token_start: bool = False,
            is_token_end: bool = False,
        ):
            self.value = value
            self.is_token_start = is_token_start
            self.is_token_end = is_token_end
            self.next = None
            self.prev = None


    class BytePair:
        def __init__(self, p1, p2):
            self.p1 = p1
            self.p2 = p2


        def __hash__(self):
            return hash((self.p1, self.p2))
        

        def __eq__(self, other):
                return self.p1 == other.p1 and self.p2 == other.p2


    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            chunks_splitted = re.split('(' + '|'.join(map(re.escape, self.special_tokens)) + ')', text)
        else:
            chunks_splitted = [text]

        first_pointer  = None
        pair_sets = {}
        last_end_pointer = None
        for chunk in chunks_splitted:
            if self.special_tokens and chunk in self.special_tokens:
                # no merge inside each special token.
                p = self.BytePointer(self.vocab_index[chunk.encode('utf-8')], is_token_start = True, is_token_end = True)
                if last_end_pointer:
                    last_end_pointer.next = p
                    p.prev = last_end_pointer
                last_end_pointer = p
                if not first_pointer:
                    first_pointer = p
            else:
                # firstly do the pretokenization.
                for pretoken in re.finditer(PRETOKENIZATION_PATTERN, chunk):
                    if len(pretoken) == 0:
                        continue
                    pretoken_bytes = pretoken.group(0).encode('utf_8')

                    # the linked list is comprised of each bytes in the pretoken.
                    li = [self.BytePointer(self.vocab_index[bytes([b])]) for b in pretoken_bytes]
                    li[0].is_token_start = True
                    li[-1].is_token_end = True

                    # maintain the helper pointers.
                    if not first_pointer:
                        first_pointer = li[0]
                    if last_end_pointer:
                        last_end_pointer.next = li[0]
                        li[0].prev = last_end_pointer
                    last_end_pointer = li[-1]

                    # link within the list.
                    for b1, b2 in zip(li[:-1], li[1:]):
                        b1.next = b2
                        b2.prev = b1
                        k = (b1.value, b2.value)
                        if k not in pair_sets:
                            pair_sets[k] = []
                        pair_sets[k].append(self.BytePair(b1, b2))

        for b1, b2 in self.indexed_merges:
            if (b1, b2) in pair_sets:
                for byte_pair in pair_sets[(b1, b2)]:
                    p1 = byte_pair.p1
                    p2 = byte_pair.p2
                    if p1.next != p2 or p2.prev != p1:
                        # one element of this pair is already merged with others.
                        continue

                    new_value = self.vocab_index[self.vocab[b1] + self.vocab[b2]]
                    new_p = self.BytePointer(
                        value = new_value,
                        is_token_start= p1.is_token_start, 
                        is_token_end = p2.is_token_end)
                    
                    # maintain the linked list.
                    if p2.next:
                        new_p.next = p2.next
                        p2.next.prev = new_p
                    if p1.prev:
                        new_p.prev = p1.prev
                        p1.prev.next = new_p

                    # update the list head.
                    if first_pointer == p1:
                        first_pointer = new_p

                    # create new pairs within the same pretoken.
                    if not new_p.is_token_start:
                        new_k = (new_p.prev.value, new_value)
                        if new_k not in pair_sets:
                            pair_sets[new_k] = []
                        pair_sets[new_k].append(self.BytePair(new_p.prev, new_p))
                    if not new_p.is_token_end:
                        new_k = (new_value, new_p.next.value)
                        if new_k not in pair_sets:
                            pair_sets[new_k] = []
                        pair_sets[new_k].append(self.BytePair(new_p, new_p.next))

        # Construct the resulting token list.
        ret = []
        p = first_pointer
        while p:
            ret.append(p.value)
            # print(p.value, self.vocab[p.value])
            p = p.next
        return ret
    

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        acc_s = ''
        for s in iterable:
            acc_s += s
            if len(acc_s) > ENCODE_ITERABLE_CHUNK_SIZE:
                token_ids = self.encode(acc_s)
                acc_s = ''
                for token_id in token_ids:
                    yield token_id
        if len(acc_s) > 0:
            token_ids = self.encode(acc_s)
            acc_s = ''
            for token_id in token_ids:
                yield token_id
    
    def decode(
        self,
        ids: list[int]
    ) -> str:
        if ids and len(ids) > 0:
            return reduce(lambda a, b: a + b, map(lambda x: self.vocab[x], ids)).decode('utf-8', errors = 'replace')
        else:
            return ""