#!/usr/bin/env python3
"""
Prepare TinyStories dataset for training.

This script downloads the TinyStories dataset and tokenizes it using a BPE tokenizer.
The tokenized data is saved as memory-mapped numpy arrays for efficient loading.

Usage:
    python scripts/prepare_tinystories.py --output_dir data --vocab_size 10000
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_tinystories(output_dir: str):
    """
    Download TinyStories dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    import urllib.request
    
    os.makedirs(output_dir, exist_ok=True)
    
    # TinyStories dataset URLs (from Hugging Face)
    urls = {
        "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz",
    }
    
    print("Downloading TinyStories dataset...")
    print("Note: This is a large file (~2GB). This may take a while.")
    
    for split, url in urls.items():
        output_path = os.path.join(output_dir, f"TinyStories_{split}.tar.gz")
        
        if os.path.exists(output_path):
            print(f"✓ {split} split already downloaded: {output_path}")
            continue
        
        print(f"Downloading {split} split from {url}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ Downloaded: {output_path}")
        except Exception as e:
            print(f"✗ Failed to download {split} split: {e}")
            print(f"\nPlease download manually from:")
            print(f"  {url}")
            print(f"And save to: {output_path}")
            return False
    
    return True


def extract_tinystories(data_dir: str):
    """
    Extract TinyStories tar.gz files.
    
    Args:
        data_dir: Directory containing the tar.gz files
    """
    import tarfile
    
    print("\nExtracting TinyStories dataset...")
    
    tar_path = os.path.join(data_dir, "TinyStories_train.tar.gz")
    
    if not os.path.exists(tar_path):
        print(f"✗ Tar file not found: {tar_path}")
        return False
    
    extract_dir = os.path.join(data_dir, "TinyStories_raw")
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)
    
    print(f"✓ Extracted to: {extract_dir}")
    return True


def load_tinystories_text(data_dir: str):
    """
    Load TinyStories text data from extracted files.
    
    Args:
        data_dir: Directory containing extracted files
    
    Returns:
        List of text strings
    """
    import json
    
    raw_dir = os.path.join(data_dir, "TinyStories_raw")
    
    if not os.path.exists(raw_dir):
        print(f"✗ Raw data directory not found: {raw_dir}")
        return None
    
    print("\nLoading TinyStories text data...")
    
    texts = []
    
    # Find all JSON files
    json_files = list(Path(raw_dir).rglob("*.json"))
    
    if not json_files:
        # Try looking for txt files
        txt_files = list(Path(raw_dir).rglob("*.txt"))
        if txt_files:
            print(f"Found {len(txt_files)} text files")
            for txt_file in tqdm(txt_files, desc="Loading text files"):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        else:
            print("✗ No JSON or TXT files found in extracted directory")
            return None
    else:
        print(f"Found {len(json_files)} JSON files")
        for json_file in tqdm(json_files, desc="Loading JSON files"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'story' in item:
                            texts.append(item['story'])
                        elif isinstance(item, str):
                            texts.append(item)
                elif isinstance(data, dict) and 'story' in data:
                    texts.append(data['story'])
    
    print(f"✓ Loaded {len(texts)} stories")
    return texts


def train_tokenizer(texts: list[str], vocab_size: int, output_path: str):
    """
    Train a BPE tokenizer on the text data.
    
    Args:
        texts: List of text strings
        vocab_size: Vocabulary size
        output_path: Path to save the tokenizer
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    
    print(f"\nTraining BPE tokenizer with vocab_size={vocab_size}...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
        show_progress=True,
    )
    
    # Train on texts
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # Save tokenizer
    tokenizer.save(output_path)
    print(f"✓ Tokenizer saved to: {output_path}")
    
    return tokenizer


def tokenize_and_save(texts: list[str], tokenizer, output_path: str, split_ratio: float = 0.95):
    """
    Tokenize texts and save as numpy arrays.
    
    Args:
        texts: List of text strings
        tokenizer: Trained tokenizer
        output_path: Base path for output files (will create _train.npy and _valid.npy)
        split_ratio: Ratio of data to use for training (rest for validation)
    """
    print(f"\nTokenizing {len(texts)} stories...")
    
    # Tokenize all texts
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        encoding = tokenizer.encode(text)
        all_tokens.extend(encoding.ids)
    
    print(f"✓ Total tokens: {len(all_tokens):,}")
    
    # Convert to numpy array
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # Split into train and validation
    split_idx = int(len(all_tokens) * split_ratio)
    train_tokens = all_tokens[:split_idx]
    valid_tokens = all_tokens[split_idx:]
    
    # Save as memory-mapped arrays
    train_path = output_path.replace(".npy", "_train.npy")
    valid_path = output_path.replace(".npy", "_valid.npy")
    
    np.save(train_path, train_tokens)
    np.save(valid_path, valid_tokens)
    
    print(f"✓ Training tokens: {len(train_tokens):,} saved to {train_path}")
    print(f"✓ Validation tokens: {len(valid_tokens):,} saved to {valid_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyStories dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Vocabulary size for BPE tokenizer"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading (use existing files)"
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Skip extraction (use existing extracted files)"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Download
    if not args.skip_download:
        print("="*80)
        print("Step 1: Downloading TinyStories dataset")
        print("="*80)
        success = download_tinystories(args.output_dir)
        if not success:
            print("\n⚠ Download failed. Please download manually or use --skip_download if files exist.")
            return
    
    # Step 2: Extract
    if not args.skip_extract:
        print("\n" + "="*80)
        print("Step 2: Extracting dataset")
        print("="*80)
        success = extract_tinystories(args.output_dir)
        if not success:
            print("\n⚠ Extraction failed.")
            return
    
    # Step 3: Load text data
    print("\n" + "="*80)
    print("Step 3: Loading text data")
    print("="*80)
    texts = load_tinystories_text(args.output_dir)
    if texts is None:
        print("\n⚠ Failed to load text data.")
        return
    
    # Step 4: Train tokenizer
    print("\n" + "="*80)
    print("Step 4: Training tokenizer")
    print("="*80)
    tokenizer_path = os.path.join(args.output_dir, f"tokenizer_v{args.vocab_size}.json")
    tokenizer = train_tokenizer(texts, args.vocab_size, tokenizer_path)
    
    # Step 5: Tokenize and save
    print("\n" + "="*80)
    print("Step 5: Tokenizing and saving")
    print("="*80)
    output_path = os.path.join(args.output_dir, "TinyStories.npy")
    tokenize_and_save(texts, tokenizer, output_path)
    
    print("\n" + "="*80)
    print("✓ Dataset preparation complete!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  - Tokenizer: {tokenizer_path}")
    print(f"  - Training data: {args.output_dir}/TinyStories_train.npy")
    print(f"  - Validation data: {args.output_dir}/TinyStories_valid.npy")


if __name__ == "__main__":
    main()

