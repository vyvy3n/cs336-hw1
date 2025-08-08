#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import torch
from torch.nn import functional as F

from cs336_basics.model.gptzero import GPTZero
from cs336_basics.training.AdamW import AdamW
from cs336_basics.training.data_loading import data_loading
from cs336_basics.training.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.cross_entropy import cross_entropy
from cs336_basics.training.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.training.mlflow_utils import (
    setup_mlflow, log_hyperparameters, log_training_metrics, 
    log_validation_metrics, log_model_checkpoint, finish_mlflow_run
)
from cs336_basics.tokenizer.Tokenizer import Tokenizer
from cs336_basics.tokenizer.OpenAITokenizer import OpenAITokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT model')
    
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--context_length', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--d_ff', type=int, default=3072)
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--min_learning_rate', type=float, default=3e-5)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, help='Path to validation data')
    parser.add_argument('--vocab_file', type=str, default='data/bpe_vocab.json', help='Path to BPE vocabulary file')
    parser.add_argument('--merges_file', type=str, default='data/bpe_merges.txt', help='Path to BPE merges file')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, help='Resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=5000)
    
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # MLflow arguments
    parser.add_argument('--use_mlflow', action='store_true', help='Enable MLflow tracking')
    parser.add_argument('--mlflow_experiment', type=str, default='gpt-training', help='MLflow experiment name')
    parser.add_argument('--mlflow_tracking_uri', type=str, help='MLflow tracking URI (optional)')
    
    # Tokenizer arguments
    parser.add_argument('--use_openai_tokenizer', action='store_true', help='Use OpenAI tiktoken tokenizer instead of custom BPE')
    parser.add_argument('--max_vocab_size', type=int, help='Limit vocab size (useful for reducing model size)')
    
    return parser.parse_args()

def get_tokenizer(use_openai_tokenizer=False, max_vocab_size=None):
    """Get tokenizer based on configuration."""
    if use_openai_tokenizer:
        if max_vocab_size:
            print(f"Using OpenAI tiktoken tokenizer (gpt2) with vocab_size limited to {max_vocab_size}")
            return OpenAITokenizer("gpt2", max_vocab_size=max_vocab_size)
        else:
            print("Using OpenAI tiktoken tokenizer (gpt2)")
            return OpenAITokenizer("gpt2")
    else:
        print("Using custom BPE tokenizer")
        import pickle
        with open("data/bpe_results_vocab10000.pkl", "rb") as f:
            bpe_data = pickle.load(f)
        vocab = bpe_data["vocab"] 
        merges = bpe_data["merges"]
        return Tokenizer(vocab, merges)

def load_data(data_path, tokenizer):
    if data_path.endswith('.npy'):
        return np.load(data_path, mmap_mode='r')
    else:
        with open(data_path, 'r') as f:
            text = f.read()
        tokens = tokenizer.encode(text) 
        
        return np.array(tokens, dtype=np.int32)


def evaluate_model(model, val_data, args):
    model.eval()
    total_loss = 0
    num_batches = 10  
    
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = data_loading(val_data, args.batch_size, args.context_length, args.device)
            logits = model(inputs)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


def main():
    args = parse_args()
    
    torch.manual_seed(42)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.use_mlflow:
        setup_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri)
        print(f"MLflow tracking enabled for experiment: {args.mlflow_experiment}")
    
    print(f"Using device: {args.device}")
    print(f"Training parameters: {vars(args)}")
    
    tokenizer = get_tokenizer(args.use_openai_tokenizer, args.max_vocab_size)
    actual_vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {actual_vocab_size}")
    
    print("Loading training data...")
    train_data = load_data(args.train_data, tokenizer)
    val_data = load_data(args.val_data, tokenizer) if args.val_data else None
    print(f"Training data size: {len(train_data)} tokens")
    
    model = GPTZero(
        vocab_size=actual_vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=args.device
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay
    )
    
    # Log hyperparameters to MLflow
    if args.use_mlflow:
        log_hyperparameters(args, actual_vocab_size)
    
    start_step = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from step {start_step}")
    
    model.train()
    step_times = []
    
    print("Starting training...")
    for step in range(start_step, args.max_steps):
        step_start = time.time()
        
        lr = get_lr_cosine_schedule(
            step, args.learning_rate, args.min_learning_rate, 
            args.warmup_steps, args.max_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        inputs, targets = data_loading(train_data, args.batch_size, args.context_length, args.device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        
        gradient_clipping(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        if step % args.log_every == 0:
            avg_time = np.mean(step_times[-100:]) if step_times else step_time
            print(f"Step {step:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                  f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s")
            
            if args.use_mlflow:
                log_training_metrics(step, loss.item(), lr, avg_time)
        
        if val_data is not None and step % args.eval_every == 0 and step > 0:
            val_loss = evaluate_model(model, val_data, args)
            print(f"Step {step:6d} | Validation Loss: {val_loss:.4f}")
            
            if args.use_mlflow:
                log_validation_metrics(step, val_loss)
        
        if step % args.save_every == 0 and step > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{step}.pt")
            save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            if args.use_mlflow:
                log_model_checkpoint(model, checkpoint_path, step)
    
    # Final checkpoint
    final_path = os.path.join(args.checkpoint_dir, f"final_checkpoint.pt")
    save_checkpoint(model, optimizer, args.max_steps, final_path)
    print(f"Training completed! Final checkpoint saved: {final_path}")
    
    # End MLflow run
    if args.use_mlflow:
        log_model_checkpoint(model, final_path, args.max_steps)
        finish_mlflow_run()
        print("MLflow run completed!")


if __name__ == "__main__":
    main()