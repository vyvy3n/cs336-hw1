#!/usr/bin/env python3

import json
import os
from tests.adapters import get_tokenizer
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

def debug_tokenizer():
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    # Load vocab
    with open(VOCAB_PATH) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    
    # Load merges
    gpt2_bpe_merges = []
    with open(MERGES_PATH) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    
    # Create vocab
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    
    # Add special token
    special_token = "<|endoftext|>"
    byte_encoded_special_token = special_token.encode("utf-8")
    if byte_encoded_special_token not in set(vocab.values()):
        vocab[len(vocab)] = byte_encoded_special_token
    
    # Create merges
    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    
    tokenizer = get_tokenizer(vocab, merges, [special_token])
    
    # Test with a simple string
    test_string = "the"
    print(f"Testing with: '{test_string}'")
    
    # Encode
    token_ids = tokenizer.encode(test_string)
    print(f"Encoded: {token_ids}")
    
    # Decode
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: '{decoded}'")
    
    # Let's also look at the first few merges
    print(f"\nFirst 10 merges:")
    for i, (t1, t2) in enumerate(merges[:10]):
        print(f"  {i}: {t1} + {t2} -> {t1 + t2}")
    
    # Let's look at what tokens we have for "the"
    print(f"\nTokens containing 'the':")
    for token_id, token_bytes in vocab.items():
        if b'the' in token_bytes:
            print(f"  {token_id}: {token_bytes}")

if __name__ == "__main__":
    debug_tokenizer() 