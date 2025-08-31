import wandb
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentLogger:
    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = True, 
        log_dir: str = "logs"
    ):
        self.start_time = time.time()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.use_wandb = use_wandb # Store the flag

        self.run = None
        if self.use_wandb:
            try:
                self.run = wandb.init(
                    project=project_name,
                    name=experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S"),
                    config=config or {}
                )
                # save config locally only if wandb initialized
                self.config_path = os.path.join(log_dir, f"{self.run.name}_config.json")
                with open(self.config_path, 'w') as f:
                    json.dump(config or {}, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to initialize Weights & Biases: {e}. Disabling wandb logging.")
                self.use_wandb = False # Disable if init fails

        self.metrics = {
            'train': {'loss': [], 'lr': [], 'steps': [], 'time': []}, # Add lr here
            'val': {'loss': [], 'steps': [], 'time': []}
        }

    def log_metrics(self, metrics_dict: Dict[str, Any], step: int):
        if self.use_wandb and self.run:
            wandb.log(metrics_dict, step=step)

    def log_train_step(self, loss: float, step: int, lr: float):
        current_time = time.time() - self.start_time

        self.metrics['train']['loss'].append(loss)
        self.metrics['train']['lr'].append(lr)
        self.metrics['train']['steps'].append(step)
        self.metrics['train']['time'].append(current_time)

        if self.use_wandb and self.run:
            wandb.log({
                'train/loss': loss,
                'train/learning_rate': lr,
                'train/wallclock_time': current_time
            }, step=step)

    def log_validation(self, loss: float, step: int):
        current_time = time.time() - self.start_time

        self.metrics['val']['loss'].append(loss)
        self.metrics['val']['steps'].append(step)
        self.metrics['val']['time'].append(current_time)

        if self.use_wandb and self.run:
            wandb.log({
                'val/loss': loss,
                'val/wallclock_time': current_time
            }, step=step)

    def log_artifact(self, file_path: str, artifact_name: str, artifact_type: str):
        if self.use_wandb and self.run:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
            print(f"Logged artifact '{artifact_name}' from {file_path} to WandB.")

    def save_metrics(self):
        """save metrics to local file"""
        if self.run: # Only save if wandb run was created (implies log_dir setup)
            metrics_path = os.path.join(self.log_dir, f"{self.run.name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)

    def finish(self):
        """finish the experiment and save all data"""
        self.save_metrics()
        if self.use_wandb and self.run:
            wandb.finish()

    def create_experiment_log(experiments: list[Dict[str, Any]], output_file: str = "experiment_log.md"):
        """
        create a single file documenting all experiments
        """
        with open(output_file, 'w') as f:
            f.write("# Experiment Log\n\n")
            
            for exp in experiments:
                f.write(f"## {exp['name']}\n\n")
                f.write("### Configuration\n")
                f.write("```json\n")
                f.write(json.dumps(exp['config'], indent=2))
                f.write("\n```\n\n")
                
                f.write("### Results\n")
                f.write(f"- Training Steps: {exp['train_steps']}\n")
                f.write(f"- Validation Steps: {exp['val_steps']}\n")
                f.write(f"- Total Time: {exp['total_time']:.2f} seconds\n")
                f.write(f"- Best Validation Loss: {exp['best_val_loss']:.4f}\n\n")
                
                f.write("### Learning Curves\n")
                f.write(f"![Learning Curves]({exp['name']}_metrics.png)\n\n")
                
            f.write("---\n\n")
