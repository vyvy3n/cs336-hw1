import os, json, time, psutil, pickle
from cs336_basics.train_bpe import train_bpe

def save_vocab_merges(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Save pickle versions for efficient processing (preserves bytes exactly)
    with open(os.path.join(out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(out_dir, "merges.pkl"), "wb") as f:
        pickle.dump(merges, f)
    
    # Save JSON/TXT versions for human inspection
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v.hex() for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")

def save_metrics(metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def run_job(input_path, vocab_size, out_dir, num_workers=None, special_tokens=["<|endoftext|>"]):
    proc = psutil.Process(os.getpid())
    t0 = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_workers=num_workers, profile=True)
    t1 = time.time()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    save_vocab_merges(vocab, merges, out_dir)
    longest_id = max(vocab, key=lambda i: len(vocab[i]))
    save_metrics({
        "time_sec": round(t1 - t0, 2),
        "time_hr": round((t1 - t0)/3600, 3),
        "max_rss_mb": round(rss_mb, 1),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "longest_token_id": int(longest_id),
        "longest_token_len": len(vocab[longest_id]),
    }, out_dir)

if __name__ == "__main__":
    # TinyStories
    # run_job(
    #     input_path="./data/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10_000,
    #     out_dir="./outputs/tinystories_bpe_10k",
    #     num_workers=12,  # adjust for your CPU; for tiny runs use None
    # )
    # OpenWebText
    run_job(
        input_path="./data/owt_train.txt",
        vocab_size=32_000,
        out_dir="./outputs/owt_bpe_32k",
        num_workers=20,
    )