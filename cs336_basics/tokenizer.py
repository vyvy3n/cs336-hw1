from typing import Iterator, Iterable, Union
from cs336_basics.common import write_vocab_to_file, read_vocab_from_file, write_merges_to_file, read_merges_from_file
import regex as re
from cs336_basics.bpe import PAT


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens. This function accepts
        the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges

        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes in self.vocab.values():
                print(f"Special token '{token}' already exists in vocabulary.")
            else:
                self.vocab[len(self.vocab)] = token_bytes
                print(f"Added special token '{token}' to vocabulary with ID {len(self.vocab)-1}.")
        
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.id_to_bytes = {k: v for k, v in vocab.items()}

        self.special_tokens_to_id = {}
        for id, token in self.vocab.items():
            if token.startswith(b"<|") and token.endswith(b"|>"):
                self.special_tokens_to_id[token] = id
    

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        vocab = read_vocab_from_file(vocab_filepath)
        merges = read_merges_from_file(merges_filepath)
        return cls(vocab, merges, special_tokens)


    def tokenize_special_tokens(self, text)->list[Union[str, int]]:
        results = [text]
        for special_token in self.special_tokens_to_id.keys():
            new_results = []
            token_str = special_token.decode('utf-8')
            for elem in results:
                if isinstance(elem, str):
                    while elem.find(token_str) != -1:
                        idx = elem.find(token_str)
                        if idx > 0:
                            new_results.append(elem[:idx])
                        new_results.append(self.special_tokens_to_id[special_token])
                        elem = elem[idx + len(token_str):]
                    if elem:
                        new_results.append(elem)
                elif isinstance(elem, int):
                    new_results.append(elem)
                else:
                    raise ValueError(f"Unexpected type {type(elem)} in results.")
            results = new_results
        return results


    def tokenize_word(self, word:str) -> list[int]:
        word_bytes = word.encode('utf-8').replace(b"\r\n", b"\n")
        bytes_len = len(word_bytes)
        i = 0
        token_ids = []
        while i < bytes_len:
            current_bytes = bytes([word_bytes[i]])
            j = i + 1
            while j < bytes_len and current_bytes + bytes([word_bytes[j]]) in self.bytes_to_id:
                current_bytes += bytes([word_bytes[j]])
                j += 1
            token_ids.append(self.bytes_to_id[current_bytes])
            i = j
        return token_ids
        

    def tokenize_normal_text(self, text:bytes) -> list[int]:
        words = re.findall(PAT, text)
        token_ids = []
        for word in words:
            token_ids.extend(self.tokenize_word(word))
        return token_ids


    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        # tokenize the text, find all special tokens, and replace them with their IDs
        results = self.tokenize_special_tokens(text)

        # the results in previous step will make the text into a list of Union[str, int]
        token_ids = []
        for elem in results:
            if isinstance(elem, str):
                token_ids.extend(self.tokenize_normal_text(elem))
            elif isinstance(elem, int):
                token_ids.append(elem)
            else:
                raise TypeError(f"Unexpected type {type(elem)} in results.")

        return token_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        """
        remaining_text = ""
        for text in iterable:
            text = remaining_text + text
            match = re.search(PAT, text, flags=re.REVERSE)
            if match:
                text = text[:match.end()]
                remaining_text = text[match.end():]
                for id in self.encode(text):
                    yield id
            else:
                remaining_text = text
        # If there's any remaining text after the loop, tokenize it
        if remaining_text:
            for id in self.encode(remaining_text):
                yield id


    def decode(self, ids: list[int]) -> str: 
        """
        Decode a sequence of token IDs into text.
        """
        tokens = [self.id_to_bytes[id] for id in ids]
        text = b"".join(tokens).decode('utf-8', errors='replace')
        return text


def test_encode():
    vocab = {0: b"<|endoftext|>", 1: b"<|startoftext|>", 2: b"ab", 3: b"a", 4: b"b", 5: b"c"}
    merges = [(b"a", b"b"), (b"ab", b"c")]
    special_tokens = ["<|endoftext|>", "<|startoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    results = tokenizer.encode("<|startoftext|>abacaba<|endoftext|>")
    print(tokenizer.encode("<|startoftext|>abacaba<|endoftext|>"))
    assert results == [1, 2, 3, 5, 2, 3, 0]


if __name__ == "__main__":
    # test_encode()
    text = "Hello, world! This is a test. <|endoftext|> Let's see how it works"
    match = re.search(PAT, text, flags=re.REVERSE)
    if match:
        print(f"最后一个匹配值: {match.start()}, 位置: {match.end()}")
    words = re.findall(PAT, text)
    print(f"Words found: {words}")
    total_len = 0
    for word in words[:-1]:
        total_len += len(word)
    print(f"Total length of words: {total_len}")
