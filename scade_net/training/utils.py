import os
import math
import torch
import random
import numpy as np
from typing import Dict, Optional


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class LRScheduler:
    """
    Learning rate scheduler that divides LR by factor every N iterations.

    This implements the MesoNet paper's LR schedule:
        - Start at initial_lr (1e-3)
        - Divide by 10 every 1000 iterations
        - Minimum LR: 1e-6

    Args:
        optimizer: PyTorch optimizer
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        decay_step: Number of iterations between LR reductions
        decay_factor: Factor to multiply LR by (e.g., 0.1 for divide by 10)
    """

    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float = 1e-3,
                 min_lr: float = 1e-6, decay_step: int = 1000, decay_factor: float = 0.1):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_step = decay_step
        self.decay_factor = decay_factor

        self.iteration = 0
        self.current_lr = initial_lr

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr

    def step(self):
        """Update learning rate based on iteration count"""
        self.iteration += 1
        if self.iteration % self.decay_step == 0:
            new_lr = max(self.current_lr * self.decay_factor, self.min_lr)

            if new_lr != self.current_lr:
                self.current_lr = new_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                return True
        return False

    def get_lr(self) -> float:
        # Get current learning rate
        return self.current_lr

    def state_dict(self) -> Dict:
        """Return state for checkpointing"""
        return {'iteration': self.iteration,
                'current_lr': self.current_lr
                }

    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint"""
        self.iteration = state_dict['iteration']
        self.current_lr = state_dict['current_lr']

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr


class CosineAnnealingWarmupScheduler:
    """
    Cosine Annealing LR scheduler with optional warmup.

    This scheduler provides smooth LR decay using a cosine function:
        - Warmup: Linear increase from 0 to initial_lr over warmup_epochs
        - Cosine decay: Smooth decrease to min_lr following cosine curve
        - Better convergence compared to step decay

    Args:
        optimizer: PyTorch optimizer
        initial_lr: Initial learning rate (reached after warmup)
        min_lr: Minimum learning rate
        warmup_epochs: Number of warmup epochs (default: 0)
        max_epochs: Total number of training epochs
        batches_per_epoch: Number of batches per epoch (for iteration-based updates)
    """

    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float = 1e-3, min_lr: float = 1e-7,
                 warmup_epochs: int = 0, max_epochs: int = 50, batches_per_epoch: int = 1000):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.batches_per_epoch = batches_per_epoch

        self.warmup_iters = warmup_epochs * batches_per_epoch
        self.max_iters = max_epochs * batches_per_epoch

        self.iteration = 0
        self.current_lr = 0.0 if warmup_epochs > 0 else initial_lr

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def step(self):
        """Update learning rate based on iteration count"""
        self.iteration += 1

        if self.iteration < self.warmup_iters:
            # Warmup phase: linear increase
            self.current_lr = (self.initial_lr * self.iteration) / self.warmup_iters
        else:
            # Cosine annealing phase
            progress = (self.iteration - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            progress = min(progress, 1.0)  # Clamp to [0, 1]

            # Cosine decay from initial_lr to min_lr
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * \
                              0.5 * (1 + math.cos(math.pi * progress))

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def get_lr(self) -> float:
        return self.current_lr

    def state_dict(self) -> Dict:
        """Return state for checkpointing"""
        return {
            'iteration': self.iteration,
            'current_lr': self.current_lr
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint"""
        self.iteration = state_dict['iteration']
        self.current_lr = state_dict['current_lr']

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a metric and stops training if it doesn't improve
    for a specified number of epochs.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'max' for metrics like AUC, 'min' for losses
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int = 0) -> bool:
        """
        Check if training should stop for none improvement

        Args:
            score: Current metric value
            epoch: Current epoch number

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0


class AverageMeter:
    """
    Computes and stores the average and current value
    Useful for tracking metrics during training process
    """

    def __init__(self, name: str = 'Metric'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self) -> str:
        return f'{self.name}: {self.avg:.4f}'


def save_checkpoint(filepath: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[LRScheduler] = None,
                    scl_criterion: Optional[torch.nn.Module] = None, epoch: int = 0, best_auc: float = 0.0, **kwargs):
    """
    Save training checkpoint

    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state (optional)
        scheduler: LR scheduler state (optional)
        scl_criterion: Single-Center Loss module (optional)
        epoch: Current epoch
        best_auc: Best validation AUC
        **kwargs: Additional items to save
    """
    checkpoint = {'model': model.state_dict(),
                  'epoch': epoch,
                  'best_auc': best_auc,
                  }

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    if scl_criterion is not None:
        checkpoint['scl'] = scl_criterion.state_dict()
    checkpoint.update(kwargs)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[LRScheduler] = None, scl_criterion: Optional[torch.nn.Module] = None,
                    device: Optional[torch.device] = None) -> Dict:
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if scl_criterion is not None and 'scl' in checkpoint:
        scl_criterion.load_state_dict(checkpoint['scl'])
    return checkpoint


if __name__ == '__main__':
    print("Training utilities test:")

    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = LRScheduler(optimizer, initial_lr=1e-3, decay_step=100)
    print(f"  Initial LR: {scheduler.get_lr()}")

    for i in range(250):
        scheduler.step()
    print(f"  LR after 250 iters: {scheduler.get_lr()}")

    early_stopping = EarlyStopping(patience=3)
    scores = [0.80, 0.82, 0.81, 0.80, 0.79, 0.78]

    print("\n  Early stopping test:")
    for i, score in enumerate(scores):
        should_stop = early_stopping(score, i)
        print(f"    Epoch {i}: score={score:.2f}, stop={should_stop}")
        if should_stop:
            break
