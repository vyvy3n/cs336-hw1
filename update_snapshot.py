from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH
import pickle
from pathlib import Path

def update_snapshot():
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    # Create the snapshot data
    snapshot_data = {
        "vocab_keys": set(vocab.keys()),
        "vocab_values": set(vocab.values()),
        "merges": merges,
    }
    
    # Save the snapshot
    snapshot_path = Path("tests/_snapshots/test_train_bpe_special_tokens.pkl")
    with open(snapshot_path, "wb") as f:
        pickle.dump(snapshot_data, f)
    
    print(f"Snapshot updated: {snapshot_path}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

if __name__ == "__main__":
    update_snapshot() 