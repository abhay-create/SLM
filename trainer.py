"""
Training module with comprehensive logging and checkpointing.
Supports both curriculum and direct learning modes.
Features:
  - Automatic Mixed Precision (AMP) for fast GPU training
  - Gradient norm tracking to detect instability
  - Frequent checkpointing (every 500 steps by default)
  - Robust evaluation with error handling
  - Checkpoint resume support
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional
import time
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    Trainer class with full logging, AMP, and checkpointing support.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        log_dir: str = 'logs',
        checkpoint_dir: str = 'checkpoints',
        eval_interval: int = 500,
        save_interval: int = 500,
        gradient_clip: float = 1.0,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip

        # AMP scaler - only on CUDA
        self.use_amp = use_amp and (device == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)

        # Logging setup
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Intervals
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.step_times = []
        self.grad_norms = []

        # Log file
        self.log_file = self.log_dir / "training_log.jsonl"

        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  AMP enabled: {self.use_amp}")
        print(f"  Eval interval: {eval_interval} steps")
        print(f"  Save interval: {save_interval} steps")
        print(f"  Log dir: {log_dir}")
        print(f"  Checkpoint dir: {checkpoint_dir}")

    def train_epoch(self):
        """Train for one epoch with AMP and gradient norm tracking."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, (x, y) in enumerate(pbar):
            start_time = time.time()

            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            # Forward pass with AMP
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                logits, loss = self.model(x, y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠ NaN/Inf loss at step {self.global_step}, skipping batch.")
                continue

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Unscale before gradient clipping
            self.scaler.unscale_(self.optimizer)

            # Compute gradient norm (always cast to Python float)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            ))

            if np.isnan(grad_norm) or np.isinf(grad_norm):
                print(f"\n⚠ NaN/Inf gradient norm at step {self.global_step}, skipping update.")
                self.scaler.update()
                continue

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            # Track metrics — all stored as Python floats to avoid JSON issues
            step_time = float(time.time() - start_time)
            loss_val = float(loss.item())

            self.step_times.append(step_time)
            self.train_losses.append(loss_val)
            self.grad_norms.append(float(grad_norm))
            self.learning_rates.append(float(self.optimizer.param_groups[0]['lr']))

            epoch_loss += loss_val
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'grad': f'{grad_norm:.2f}',
                'tok/s': f'{x.numel() / step_time:.0f}',
            })

            # Evaluation
            if self.global_step % self.eval_interval == 0:
                val_loss, val_ppl = self._safe_evaluate()

                if val_loss is not None:
                    self.val_losses.append((self.global_step, val_loss))

                    self.log_metrics({
                        'step': int(self.global_step),
                        'epoch': int(self.epoch),
                        'train_loss': float(loss_val),
                        'val_loss': float(val_loss),
                        'val_perplexity': float(val_ppl),
                        'lr': float(self.optimizer.param_groups[0]['lr']),
                        'grad_norm': float(grad_norm),
                        'tokens_per_sec': float(x.numel() / step_time),
                    })

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._safe_save_checkpoint('best_model.pt', is_best=True)
                        print(f"\n✓ New best val loss: {val_loss:.4f} "
                              f"(ppl={val_ppl:.2f}) at step {self.global_step}")

            # Save checkpoint every save_interval steps
            if self.global_step % self.save_interval == 0:
                ckpt_name = f'checkpoint_step_{self.global_step}.pt'
                self._safe_save_checkpoint(ckpt_name)
                print(f"\n💾 Checkpoint saved: {ckpt_name}")

        return epoch_loss / max(num_batches, 1)

    def _safe_evaluate(self):
        """Evaluate with error handling."""
        try:
            val_loss, val_ppl = self.evaluate()
            return val_loss, val_ppl
        except Exception as e:
            print(f"\n⚠ Evaluation failed: {e}")
            return None, None

    def _safe_save_checkpoint(self, filename: str, is_best: bool = False):
        """Save checkpoint with error handling."""
        try:
            self.save_checkpoint(filename, is_best)
        except Exception as e:
            print(f"\n⚠ Checkpoint save failed ({filename}): {e}")

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with autocast(enabled=self.use_amp):
                logits, loss = self.model(x, y)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            return float('inf'), float('inf')

        avg_loss = total_loss / num_batches
        perplexity = float(np.exp(min(avg_loss, 20)))  # cap at exp(20) to avoid overflow

        self.model.train()
        return avg_loss, perplexity

    def train(self, num_epochs: int):
        """Train for specified number of epochs."""
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nStarting training:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Parameters: {total_params:,}")
        print(f"  Steps per epoch: {len(self.train_loader)}")
        print(f"  Total steps: {num_epochs * len(self.train_loader):,}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = self.train_epoch()

            print(f"\nEpoch {epoch} complete:")
            print(f"  Avg train loss: {epoch_loss:.4f}")
            print(f"  Best val loss:  {self.best_val_loss:.4f}")

            # Save epoch checkpoint
            self._safe_save_checkpoint(f'epoch_{epoch}.pt')

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        self.save_metrics()

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses[-2000:],  # Keep last 2000
            'val_losses': self.val_losses[-200:],
            'learning_rates': self.learning_rates[-2000:],
            'grad_norms': self.grad_norms[-2000:],
        }

        filepath = self.checkpoint_dir / filename
        # Write to temp file first, then rename (atomic write)
        tmp_path = filepath.with_suffix('.tmp')
        torch.save(checkpoint, tmp_path)
        tmp_path.rename(filepath)

        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            tmp_best = self.checkpoint_dir / 'best.tmp'
            torch.save(checkpoint, tmp_best)
            tmp_best.rename(best_path)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint and resume training."""
        print(f"Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = list(checkpoint.get('train_losses', []))
        self.val_losses = list(checkpoint.get('val_losses', []))
        self.learning_rates = list(checkpoint.get('learning_rates', []))
        self.grad_norms = list(checkpoint.get('grad_norms', []))

        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")
        print(f"Best val loss so far: {self.best_val_loss:.4f}")

    def log_metrics(self, metrics: Dict):
        """Log metrics to JSONL file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        except Exception:
            pass

    def save_metrics(self):
        """Save all metrics to JSON file."""
        metrics = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [(int(step), float(loss)) for step, loss in self.val_losses],
            'learning_rates': [float(x) for x in self.learning_rates],
            'grad_norms': [float(x) for x in self.grad_norms],
            'step_times': [float(x) for x in self.step_times],
            'best_val_loss': float(self.best_val_loss),
            'total_steps': self.global_step,
            'total_epochs': self.epoch,
        }

        metrics_file = self.log_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {metrics_file}")


class CurriculumTrainer(Trainer):
    """
    Extended trainer for curriculum learning with stage progression.
    """
    def __init__(self, *args, curriculum_dataset=None, stage_epochs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum = curriculum_dataset
        self.stage_epochs = stage_epochs or [5, 5, 5]
        self.current_stage = 0

    def train_curriculum(self):
        """Train through all curriculum stages."""
        n_stages = len(self.curriculum.stages)
        print(f"\nStarting curriculum training with {n_stages} stages...")
        print(f"Stage epochs: {self.stage_epochs}")

        for stage_idx in range(n_stages):
            self.current_stage = stage_idx
            stage = self.curriculum.stages[stage_idx]

            print(f"\n{'='*60}")
            print(f"STAGE {stage_idx + 1}/{n_stages}: {stage['name']}")
            print(f"{'='*60}")

            # Update data loader for this stage
            batch_size = self.train_loader.batch_size
            self.train_loader = self.curriculum.get_stage_dataloader(
                stage_idx, batch_size, shuffle=True
            )

            print(f"Stage data: {len(self.train_loader)} batches")

            # Train for stage epochs
            num_epochs = (self.stage_epochs[stage_idx]
                         if stage_idx < len(self.stage_epochs) else 5)

            for epoch in range(num_epochs):
                self.epoch = epoch
                epoch_loss = self.train_epoch()
                print(f"Stage {stage_idx+1}, Epoch {epoch}: loss = {epoch_loss:.4f}")

            # Save stage checkpoint
            ckpt_name = f'stage_{stage_idx+1}_of_{n_stages}_final.pt'
            self._safe_save_checkpoint(ckpt_name)
            print(f"Stage {stage_idx+1} complete. Checkpoint: {ckpt_name}")

        print("\nCurriculum training completed!")
        self.save_metrics()


if __name__ == "__main__":
    print("Trainer module loaded successfully")