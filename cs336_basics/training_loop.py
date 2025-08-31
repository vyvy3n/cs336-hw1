import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import time

from .training import cross_entropy, get_cosine_lr, gradient_clipping, save_checkpoint, load_checkpoint
from .optimizer import AdamW
from .data_loader import data_loader
from .tokenizer import Tokenizer
from .transformer import TransformerLM
from .logger import ExperimentLogger
from .train_bpe import _pretokenize_parallel # Corrected import for parallel pretokenization


def pretokenize_data_helper(
    tokenizer: Tokenizer, 
    logger: ExperimentLogger, 
    input_path: str, 
    output_path: str, 
    use_parallel: bool,
    num_workers: int,
    special_tokens: list[str]
) -> None:
    """
    Pretokenize a text file and save the token IDs as a numpy array.
    Handles parallel processing and logging.
    """
    print(f"Creating fresh pretokenized data from {input_path}...")
    start_time = time.time()
    
    all_token_ids = []

    if use_parallel:
        print("Starting parallel pretokenization...")
        pretoken_freq = _pretokenize_parallel(input_path, special_tokens, num_workers) # Corrected function call
        print(f"Found {len(pretoken_freq)} unique pretokens")
        
        total_pretokens = sum(pretoken_freq.values())
        print(f"Processing {total_pretokens} total pretokens")
        
        with tqdm(total=len(pretoken_freq), desc="Encoding pretokens") as pbar:
            for pretoken, freq in pretoken_freq.items():
                # pretoken is a tuple of integers (byte IDs) from _pretokenize_parallel
                # Convert tuple of ints to bytes object, then decode to string
                text = bytes(list(pretoken)).decode('utf-8')
                token_ids = tokenizer.encode(text)
                all_token_ids.extend(token_ids * freq)
                pbar.update(1)
        
    else:
        print("Using simple sequential pretokenization...")
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Pretokenizing {os.path.basename(input_path)} data"):
                all_token_ids.extend(tokenizer.encode(line))
    
    np.save(output_path, np.array(all_token_ids, dtype=np.int32))
    
    end_time = time.time()
    print(f"Pretokenization completed in {end_time - start_time:.2f} seconds")
    print(f"Saved {len(all_token_ids)} tokens to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    
    # data paths
    parser.add_argument("--train_path", type=str, default = "data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid_path", type=str, default = "data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--pretokens_train_path", type=str, default="outputs/tinystories_train.npy")
    parser.add_argument("--pretokens_valid_path", type=str, default="outputs/tinystories_valid.npy")
    parser.add_argument("--vocab_path", type=str, default = "outputs/tinystories_bpe_10k/vocab.pkl")
    parser.add_argument("--merges_path", type=str, default = "outputs/tinystories_bpe_10k/merges.pkl")

    # data loading params
    parser.add_argument("--vocab_size", type=int, default = 10000)
    parser.add_argument("--context_length", type=int, default = 256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--use_memmap", action="store_true", help="Use memory-mapped files for data loading")
    parser.add_argument("--reuse_pretokens", action="store_true", help="Reuse existing pretokenized data if available")
    parser.add_argument("--use_parallel_pretokenize", action="store_true") # Default to parallel for full dataset

    parser.add_argument("--special_tokens", type=str, nargs='+', default=["<|endoftext|>"])
    parser.add_argument("--pad_token", type=str, default="<|pad|>")

    # model params
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--max_seq_len", type=int, default=256)

    # training and optimizer params
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--gradient_clip_M", type=float, default=5.0)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=5e-3)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--min_loss_threshold", type=float, default=2.0) 
    parser.add_argument("--warmup_steps", type=int, default=2000)

    # logging and checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--checkpoint_freq", type=int, default=1000)
    parser.add_argument("--save_best_only", action="store_true", help="Only save the best model checkpoint")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from")

    # device and compilation
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_compile", action="store_true", help="Enable torch.compile()")
    
    return parser.parse_args()

def load_data_memmap(file_path, dtype=np.int32):
    """Load data using memory mapping for efficient memory usage."""
    print(f"Loading data from {file_path} using memory mapping...")
    # Use np.load with mmap_mode if the file was saved with np.save
    return np.load(file_path, mmap_mode='r')

def load_data_regular(file_path, dtype=np.int32):
    """Load data into regular memory."""
    print(f"Loading data from {file_path} into regular memory...")
    data = np.load(file_path)
    print(f"Loaded {len(data)} tokens")
    return data

def train_step(model, optimizer, train_data, args, device):
    inputs, targets = data_loader(train_data, args.batch_size, args.context_length, device) 
    logits = model(inputs)
    loss = cross_entropy(logits, targets) 
    
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), args.gradient_clip_M)
    optimizer.step()
    
    return loss.item()

def evaluate(model, valid_data, args, device, n_batches=10):
    model.eval()
    losses = []
    with torch.no_grad():
        max_start = max(0, len(valid_data) - args.context_length - 1) # Ensure max_start is not negative
        if max_start == 0 and len(valid_data) < args.context_length + 1:
            print("Warning: Validation data too short for context length. Skipping evaluation.")
            return float('inf') # Return inf if cannot evaluate
        
        for _ in range(n_batches):
            start = np.random.randint(0, max_start + 1)
            input_seq = valid_data[start:start + args.context_length]
            target_seq = valid_data[start + 1:start + args.context_length + 1]
            inputs = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(target_seq, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def main(args=None):
    if args is None:
        args = parse_args()
    
    # setup logger
    logger = ExperimentLogger(
        project_name=args.wandb_project,
        experiment_name=args.experiment_name,
        config=vars(args),
        use_wandb=args.use_wandb
    )
    
    # setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)

    # only enable TF32‐style "high" matmul precision on cuda, never on mps
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")
    
    # load tokenizer from files
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=args.special_tokens)


    s = "Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, “no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players.” With a few clicks here and there, you can findout that Noah Syndergaard’s fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game’s highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball’s culture: the analytical hobbyist."
    ids = tokenizer.encode(s)
    print("Decoder Output:")
    print(tokenizer.decode(ids))


    # Pretokenization (using helper function)
    # For training data
    if args.reuse_pretokens and os.path.exists(args.pretokens_train_path):
        print(f"Reusing existing pretokenized training data from: {args.pretokens_train_path}")
    else:
        print("Pretokenization required for training data.")
        pretokenize_data_helper(
            tokenizer=tokenizer,
            logger=logger,
            input_path=args.train_path,
            output_path=args.pretokens_train_path,
            use_parallel=args.use_parallel_pretokenize,
            num_workers=args.num_workers,
            special_tokens=args.special_tokens
        )

    # For validation data
    if args.reuse_pretokens and os.path.exists(args.pretokens_valid_path):
        print(f"Reusing existing pretokenized validation data from: {args.pretokens_valid_path}")
    else:
        print("Pretokenization required for validation data.")
        pretokenize_data_helper(
            tokenizer=tokenizer,
            logger=logger,
            input_path=args.valid_path,
            output_path=args.pretokens_valid_path,
            use_parallel=args.use_parallel_pretokenize,
            num_workers=args.num_workers,
            special_tokens=args.special_tokens
        )

    # load data based on the specified method
    if not args.use_memmap:
        print("Loading data into regular memory...")
        train_data = load_data_regular(args.pretokens_train_path)
        valid_data = load_data_regular(args.pretokens_valid_path)
    else:
        print("Loading data using memory mapping...")
        train_data = load_data_memmap(args.pretokens_train_path)
        valid_data = load_data_memmap(args.pretokens_valid_path)
    
    # create model
    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        theta=args.rope_theta,
        device=device
    )
    model = model.to(device) 
    
    print("Created model.")

    print("Max token ID in valid_data:", valid_data.max())
    print("Min token ID in valid_data:", valid_data.min())
    print("Model vocab size:", model.vocab_size)

    print("Max token ID in train_data:", train_data.max())
    print("Min token ID in train_data:", train_data.min())
    print("Model vocab size:", model.vocab_size)
    
    # compile model if enabled
    if args.use_compile:
        print("Compiling model for better performance...")
        model = torch.compile(model)
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.epsilon,
        betas=(args.beta1, args.beta2)
    )
    
    print("Created optimizer, starting training loop")
    print(f"Will run for {args.max_steps} steps")
    
    # training loop
    initial_step = 0
    best_val_loss = float('inf')
    patience = 50  # number of evaluations without improvement before stopping
    no_improvement_count = 0
    min_loss_threshold = args.min_loss_threshold  # stop if loss gets below this threshold
    
    # Resume from checkpoint if specified
    if args.resume_from:
        initial_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed training from iteration {initial_step}")
        # Log resume step via logger
        logger.log_metrics({'resumed_from_step': initial_step}, step=initial_step)

    # stability tracking
    val_losses = []  
    stability_window = 5  # num recent validation losses to check
    stability_threshold = 0.01  # max allowed relative change between consecutive losses
    
    # divergence detection
    divergence_threshold = 100.0  # loss threshold for divergence
    last_losses = []  
    window_size = 10  # increased from 5 to 10
    min_increase = 0.1  # min relative increase to consider as divergence
    
    pbar = tqdm(range(initial_step, args.max_steps), desc="Training")
    for step in pbar:
        # training
        train_loss = train_step(model, optimizer, train_data, args, device)
        
        # check for divergence
        last_losses.append(train_loss)
        if len(last_losses) > window_size:
            last_losses.pop(0)
        
        # if loss is too high or increasing rapidly, consider it diverged
        if train_loss > divergence_threshold:
            print(f"\nTraining diverged at step {step} with loss {train_loss:.4f}")
            logger.log_metrics({'diverged': True, 'divergence_reason': 'loss_too_high'}, step=step)
            return {
                'final_val_loss': float('inf'),
                'best_val_loss': float('inf'),
                'final_step': step,
                'diverged': True,
                'divergence_reason': 'loss_too_high'
            }
        
        # check for consistent increase with minimum threshold
        if len(last_losses) == window_size:
            increases = [last_losses[i+1] - last_losses[i] for i in range(len(last_losses)-1)]
            relative_increases = [inc / last_losses[i] for i, inc in enumerate(increases)]
            if all(inc > min_increase for inc in relative_increases):
                print(f"\nTraining diverged at step {step} with loss {train_loss:.4f}")
                logger.log_metrics({'diverged': True, 'divergence_reason': 'loss_increasing'}, step=step)
                return {
                    'final_val_loss': float('inf'),
                    'best_val_loss': float('inf'),
                    'final_step': step,
                    'diverged': True,
                    'divergence_reason': 'loss_increasing'
                }
        
        # LR scheduling
        lr = get_cosine_lr(
            step,
            args.max_lr, # alpha_max
            args.min_lr, # alpha_min
            args.warmup_steps, # Tw
            args.max_steps # Tc
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # logging
        if step % args.log_freq == 0:
            logger.log_train_step(train_loss, step, lr) # Pass lr to log_train_step
            pbar.write(f"Step {step}: Train Loss = {train_loss:.4f}, LR = {lr:.2e}")
        
        # eval and checkpointing
        if step % args.eval_freq == 0:
            val_loss = evaluate(model, valid_data, args, device, n_batches=10)
            logger.log_validation(val_loss, step)
            pbar.write(f"Step {step}: Val Loss = {val_loss:.4f}")
            
            # track validation losses for stability check
            val_losses.append(val_loss)
            if len(val_losses) > stability_window:
                val_losses.pop(0)
            
            # check for both low loss and stability
            if val_loss < min_loss_threshold and len(val_losses) == stability_window:
                #relative changes between consecutive losses
                relative_changes = [abs(val_losses[i+1] - val_losses[i]) / val_losses[i] 
                                 for i in range(len(val_losses)-1)]
                # check if all changes are below threshold
                if all(change < stability_threshold for change in relative_changes):
                    pbar.write(f"\nEarly stopping at step {step} - reached minimum loss threshold of {min_loss_threshold} with stable loss")
                    logger.log_metrics({'early_stop': True, 'reason': 'min_loss_threshold_stable'}, step=step)
                    break
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                save_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                run_info = {
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "wandb_run_name": getattr(logger.run, "name", None) if logger.use_wandb else None
                }
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    save_path,
                    run_info=run_info
                )
                logger.log_metrics({'best_val_loss': best_val_loss}, step=step)
                logger.log_artifact(save_path, artifact_name="best_model", artifact_type="model") # Log artifact via logger

            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    pbar.write(f"\nEarly stopping at step {step} - no improvement for {patience} evaluations")
                    logger.log_metrics({'early_stop': True, 'reason': f'no_improvement_for_{patience}_evals'}, step=step)
                break
    
        # periodic checkpointing
        if not args.save_best_only and step % args.checkpoint_freq == 0:
            save_path = os.path.join(args.checkpoint_dir, f"checkpoint_{step}.pt")
            run_info = {
                "save_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "wandb_run_name": getattr(logger.run, "name", None) if logger.use_wandb else None
            }
            save_checkpoint(
                model,
                optimizer,
                step,
                save_path,
                run_info=run_info
            )
            logger.log_artifact(save_path, artifact_name=f"checkpoint_{step}", artifact_type="model") # Log artifact via logger
    
    # save final model
    final_save_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    run_info = {
        "save_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "wandb_run_name": getattr(logger.run, "name", None) if logger.use_wandb else None
    }
    save_checkpoint(
        model,
        optimizer,
        step, 
        final_save_path,
        run_info=run_info
    )
    logger.log_artifact(final_save_path, artifact_name="final_model", artifact_type="model") # Log artifact via logger
    
    # finish logging
    logger.finish()
    
    # Return final metrics
    return {
        'final_val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'final_step': step,
        'diverged': False
    }

if __name__ == "__main__":
    main()