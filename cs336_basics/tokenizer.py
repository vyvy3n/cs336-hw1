from dataclasses import dataclass, field
from collections import defaultdict

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None
    bytes_id : dict[bytes,int] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "bytes_id", {b: i for i, b in self.vocab.items()})



def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  
    new_indices.index
    i = 0  
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


class BPETokenizer:
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  
        # Note: this is a very slow implementation
        self.params.vocab
        for pair in self.params.merges:
            new_index = self.params.bytes_id[pair[0]+pair[1]]
            pair = (int(pair[0]),int(pair[1]))
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(lambda x: self.params.vocab.get(x,b""), indices))  
        string = b"".join(bytes_list).decode("utf-8")  
        return string



def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  
    # Start with the list of bytes of `string`.
    indices = list(map(int, string.encode("utf-8")))  
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1  

        # Find the most common pair.
        pair = max(counts, key=counts.get)  
        index1, index2 = pair

        # "Merge that pair."
        new_index = 256 + i  
        merges[pair] = new_index  
        vocab[new_index] = vocab[index1] + vocab[index2]  
        indices = merge(indices, pair, new_index)  

    return BPETokenizerParams(vocab=vocab, merges=merges)

if __name__ == "__main__":
    # tk = train_bpe("sam go to france. sam came from france.",num_merges=10)
    # print(tk)
    # with open(r"tests\fixtures\tinystories_sample.txt",encoding="utf-8") as f:
    #     print(type(f))
    #     try:
    #         # some_object_iterator = iter(f)
    #         # print(list(some_object_iterator))
    #         for i,x in enumerate(f):
    #             print(i)
    #             print(x)
    #     except TypeError as te:
    #         print(f, 'is not iterable',te)
    pass