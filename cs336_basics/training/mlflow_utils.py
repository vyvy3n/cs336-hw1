"""Minimal MLflow utilities for tracking model training."""

import mlflow
import mlflow.pytorch
import torch
import os


def setup_mlflow(experiment_name="gpt-training", tracking_uri=None):
    """Setup MLflow experiment and tracking."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()


def log_hyperparameters(args, actual_vocab_size=None):
    """Log training hyperparameters to MLflow."""
    params = {
        'vocab_size': actual_vocab_size or args.vocab_size,
        'context_length': args.context_length,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'min_learning_rate': args.min_learning_rate,
        'warmup_steps': args.warmup_steps,
        'max_steps': args.max_steps,
        'weight_decay': args.weight_decay,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'grad_clip': args.grad_clip,
        'use_openai_tokenizer': getattr(args, 'use_openai_tokenizer', False),
    }
    mlflow.log_params(params)


def log_training_metrics(step, loss, lr, step_time=None):
    """Log training metrics to MLflow."""
    metrics = {
        'train_loss': loss,
        'learning_rate': lr,
    }
    if step_time:
        metrics['step_time'] = step_time
    
    mlflow.log_metrics(metrics, step=step)


def log_validation_metrics(step, val_loss):
    """Log validation metrics to MLflow."""
    mlflow.log_metrics({'val_loss': val_loss}, step=step)


def log_model_checkpoint(model, checkpoint_path, step):
    """Log model checkpoint as artifact to MLflow."""
    if os.path.exists(checkpoint_path):
        mlflow.log_artifact(checkpoint_path, f"checkpoints/step_{step}")


def finish_mlflow_run():
    """End the current MLflow run."""
    mlflow.end_run()