#!/usr/bin/env python3

import pickle
import json

# Load the pickle file
with open('data/bpe_results_vocab10000.pkl', 'rb') as f:
    data = pickle.load(f)

print("Type:", type(data))
print("Keys:" if hasattr(data, 'keys') else "Length:", 
      list(data.keys()) if hasattr(data, 'keys') else len(data) if hasattr(data, '__len__') else 'No length')

if hasattr(data, 'items'):
    for k, v in data.items():
        print(f'{k}: {type(v)}')
        if hasattr(v, '__len__'):
            print(f'  Length: {len(v)}')
        if isinstance(v, dict) and len(v) < 20:
            print(f'  Sample keys: {list(v.keys())[:10]}')
        elif isinstance(v, list) and len(v) < 50:
            print(f'  Sample items: {v[:5]}')
        print()

# Try to save vocab and merges as text files
if hasattr(data, 'get'):
    vocab = data.get('vocab', None)
    merges = data.get('merges', None)
    
    if vocab:
        print(f"Found vocab with {len(vocab)} items")
        print("Sample vocab items:")
        for i, (k, v) in enumerate(vocab.items()):
            if i < 10:
                print(f"  {k} -> {repr(v)}")
        
        # Convert vocab to JSON-serializable format
        # Vocab structure: {index: bytes_token} -> {token_string: index}
        json_vocab = {}
        for index, token_bytes in vocab.items():
            if isinstance(token_bytes, bytes):
                # Convert bytes to string using latin-1 encoding to preserve all byte values
                token_str = token_bytes.decode('latin-1')
            else:
                token_str = str(token_bytes)
            json_vocab[token_str] = index
        
        # Save vocab as JSON
        with open('data/bpe_vocab.json', 'w', encoding='utf-8') as f:
            json.dump(json_vocab, f, ensure_ascii=False, indent=2)
        print("Saved vocab to data/bpe_vocab.json")
    
    if merges:
        print(f"Found merges with {len(merges)} items")
        print("Sample merge items:")
        for i, merge in enumerate(merges[:10]):
            print(f"  {repr(merge)}")
        
        # Save merges as text file
        with open('data/bpe_merges.txt', 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")  # BPE format header
            for merge in merges:
                if isinstance(merge, (tuple, list)) and len(merge) == 2:
                    # Convert bytes to strings if needed
                    token1 = merge[0].decode('latin-1') if isinstance(merge[0], bytes) else str(merge[0])
                    token2 = merge[1].decode('latin-1') if isinstance(merge[1], bytes) else str(merge[1])
                    f.write(f"{token1} {token2}\n")
                else:
                    merge_str = merge.decode('latin-1') if isinstance(merge, bytes) else str(merge)
                    f.write(f"{merge_str}\n")
        print("Saved merges to data/bpe_merges.txt") 