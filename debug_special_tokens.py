from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH

def debug_special_tokens():
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    
    print("Special token check:")
    print(f"Special token in vocab: {b'<|endoftext|>' in vocab.values()}")
    
    # Find all tokens that contain '<|'
    problematic_tokens = []
    for token_id, token_bytes in vocab.items():
        if b'<|' in token_bytes and token_bytes != b'<|endoftext|>':
            problematic_tokens.append((token_id, token_bytes))
    
    print(f"\nProblematic tokens (contain '<|' but not the special token):")
    for token_id, token_bytes in problematic_tokens:
        print(f"  ID {token_id}: {token_bytes}")
    
    # Check if the special token appears in the input text
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\nSpecial token in input text: {'<|endoftext|>' in text}")
    if '<|endoftext|>' in text:
        print(f"Number of occurrences: {text.count('<|endoftext|>')}")
        # Find the context around the special token
        idx = text.find('<|endoftext|>')
        if idx != -1:
            start = max(0, idx - 20)
            end = min(len(text), idx + 30)
            print(f"Context: ...{text[start:end]}...")

if __name__ == "__main__":
    debug_special_tokens() 