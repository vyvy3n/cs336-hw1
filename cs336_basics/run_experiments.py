import random
import time
from pathlib import Path
import numpy as np
from cs336_basics.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]  # assignment1-basics/
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# Ensure save dir exists
(OUT_DIR / "encoded_datasets").mkdir(parents=True, exist_ok=True)

def sample_docs(path, n=10):
    with open(path, "r", encoding="utf-8") as f:
        docs = f.read().split("<|endoftext|>")
    return random.sample(docs, n)

def compression_ratio(tokenizer, docs):
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        b = len(doc.encode("utf-8"))
        ids = tokenizer.encode(doc)
        total_bytes += b
        total_tokens += len(ids)
    return total_bytes / total_tokens


ts_dir = OUT_DIR / "tinystories_bpe_10k"
owt_dir = OUT_DIR / "owt_bpe_32k"

ts_vocab_path = ts_dir / "vocab.pkl"
ts_merges_path = ts_dir / "merges.pkl"

owt_vocab_path = owt_dir / "vocab.pkl"
owt_merges_path = owt_dir / "merges.pkl"

tiny_tok = Tokenizer.from_files(str(ts_vocab_path), str(ts_merges_path), ["<|endoftext|>"])
tiny_docs = sample_docs(DATA_DIR / "TinyStoriesV2-GPT4-train.txt")

owt_tok = Tokenizer.from_files(str(owt_vocab_path), str(owt_merges_path), ["<|endoftext|>"])
owt_docs = sample_docs(DATA_DIR / "owt_train.txt")

# (a) Compression ratios
tiny_ratio = compression_ratio(tiny_tok, tiny_docs)
owt_ratio = compression_ratio(owt_tok, owt_docs)
print("TinyStories tokenizer ratio:", tiny_ratio)
print("OpenWebText tokenizer ratio:", owt_ratio)

# (b) OWT with TinyStories tokenizer
owt_with_tiny_ratio = compression_ratio(tiny_tok, owt_docs)
print("Compression Ratio of OWT with TinyStories Tokenizer:", owt_with_tiny_ratio)

# (c) Throughput estimate
docs_for_tp = sample_docs(DATA_DIR / "TinyStoriesV2-GPT4-train.txt", n=1000)
start = time.time()
bytes_processed = sum(len(doc.encode("utf-8")) for doc in docs_for_tp)
_ = [tiny_tok.encode(doc) for doc in docs_for_tp]
end = time.time()

throughput = bytes_processed / (end - start)  # bytes/sec
pile_bytes = 825 * 1024**3  # 825 GB
pile_time_sec = pile_bytes / throughput

print("Throughput:", throughput, "bytes/sec")
print("Pile time (hours):", pile_time_sec / 3600)

# (d) Encode datasets
def encode_dataset(tokenizer, input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    np.save(output_path, ids)

encode_dataset(tiny_tok, DATA_DIR / "TinyStoriesV2-GPT4-train.txt", OUT_DIR / "encoded_datasets" / "tinystories_train.npy")
encode_dataset(owt_tok, DATA_DIR / "owt_train.txt", OUT_DIR / "encoded_datasets" / "owt_train.npy")