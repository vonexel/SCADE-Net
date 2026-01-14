import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from ..models import SCADENet
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score
from ..models.losses import CombinedLoss, SingleCenterLoss, FocalLoss
from .utils import (AverageMeter,
                    EarlyStopping,
                    LRScheduler,
                    CosineAnnealingWarmupScheduler,
                    save_checkpoint,
                    )


class SCADENetTrainer:
    """
    Trainer for SCADE-Net.

    Implements the full training pipeline with combined BCE + SCL loss.

    Args:
        model: SCADE-Net model
        config: Configuration dictionary or object with attributes:
            - initial_lr: Initial learning rate (default: 1e-3)
            - min_lr: Minimum learning rate (default: 1e-6)
            - lr_decay_step: Iterations between LR decay (default: 1000)
            - lr_decay_factor: LR decay factor (default: 0.1)
            - scl_lambda: Weight for SCL loss (default: 0.1)
            - scl_margin: SCL margin (default: 0.5)
            - scl_alpha: SCL balance coefficient (default: 0.5)
            - use_mixed_precision: Enable FP16 training (default: True)
            - gradient_clip_norm: Max gradient norm (default: 1.0)
            - early_stopping_patience: Epochs without improvement (default: 5)
        device: Torch device
    """
    def __init__(self, model: SCADENet, config: Dict, device: Optional[torch.device] = None, pos_weight: Optional[float] = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.model = model.to(device)

        # Helper to get value from either dict or object
        def get_config_value(key: str, default):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        # Learning rate
        self.initial_lr = float(get_config_value('initial_lr', 1e-3))
        self.min_lr = float(get_config_value('min_lr', 1e-6))

        # LR Scheduler config
        self.lr_scheduler_type = get_config_value('lr_scheduler', 'step')
        self.lr_decay_step = get_config_value('lr_decay_step', 1000)
        self.lr_decay_factor = float(get_config_value('lr_decay_factor', 0.1))
        self.warmup_epochs = get_config_value('warmup_epochs', 0)
        self.num_epochs = get_config_value('num_epochs', 50)

        # SCL config (ensure float conversion)
        self.scl_lambda = float(get_config_value('scl_lambda', 0.1))
        self.scl_margin = float(get_config_value('scl_margin', 0.5))
        self.scl_alpha = float(get_config_value('scl_alpha', 0.5))

        # Loss config
        self.use_focal_loss = get_config_value('use_focal_loss', False)
        self.focal_alpha = float(get_config_value('focal_alpha', 0.25))
        self.focal_gamma = float(get_config_value('focal_gamma', 2.0))

        # Training config
        self.use_mixed_precision = get_config_value('use_mixed_precision', True)
        self.gradient_clip_norm = float(get_config_value('gradient_clip_norm', 1.0))
        self.patience = get_config_value('early_stopping_patience', 5)
        self.early_stopping_min_delta = float(get_config_value('early_stopping_min_delta', 0.001))
        self.weight_decay = float(get_config_value('weight_decay', 0.0))

        # Initialize loss functions
        if self.use_focal_loss:
            self.bce_criterion = FocalLoss(
                alpha=self.focal_alpha,
                gamma=self.focal_gamma
            ).to(device)
            print(f"  Using Focal Loss (alpha={self.focal_alpha}, gamma={self.focal_gamma})")
        else:
            # Use BCEWithLogitsLoss (combines sigmoid + BCE, safe for autocast)
            # Apply pos_weight if provided for class imbalance handling
            if pos_weight is not None and pos_weight != 1.0:
                pos_weight_tensor = torch.tensor([pos_weight], device=device)
                self.bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                print(f"  Using weighted BCE loss (pos_weight={pos_weight:.3f})")
            else:
                self.bce_criterion = nn.BCEWithLogitsLoss()

        self.scl_criterion = SingleCenterLoss(
            feature_dim=1024,
            margin=self.scl_margin,
            alpha=self.scl_alpha
        ).to(device)

        # Initialize optimizer (include SCL center in parameters)
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.scl_criterion.parameters()),
            lr=self.initial_lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )

        # Store batches_per_epoch for cosine scheduler (will be set in train())
        self.batches_per_epoch = None

        # Initialize LR scheduler
        if self.lr_scheduler_type == 'cosine':
            # Will be created in train() when we know batches_per_epoch
            self.lr_scheduler = None
            self.scheduler_type_name = 'Cosine Annealing'
        else:
            self.lr_scheduler = LRScheduler(
                self.optimizer,
                initial_lr=self.initial_lr,
                min_lr=self.min_lr,
                decay_step=self.lr_decay_step,
                decay_factor=self.lr_decay_factor
            )
            self.scheduler_type_name = 'Step Decay'

        # Mixed precision
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            min_delta=self.early_stopping_min_delta,
            mode='max'
        )

        # Training history
        self.history = defaultdict(list)
        self.best_val_auc = 0.0
        self.best_model_state = None
        self.best_epoch = 0

    def train_epoch(self, train_loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        self.model.train()

        loss_meter = AverageMeter('Loss')
        bce_meter = AverageMeter('BCE')
        scl_meter = AverageMeter('SCL')

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                with autocast('cuda'):
                    logits, features = self.model(images, return_features=True)
                    bce_loss = self.bce_criterion(logits.squeeze(), labels)
                    scl_loss, scl_metrics = self.scl_criterion(features, labels.long())
                    total_loss = bce_loss + self.scl_lambda * scl_loss

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, features = self.model(images, return_features=True)
                bce_loss = self.bce_criterion(logits.squeeze(), labels)
                scl_loss, scl_metrics = self.scl_criterion(features, labels.long())
                total_loss = bce_loss + self.scl_lambda * scl_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm
                )
                self.optimizer.step()

            # Update LR
            lr_changed = self.lr_scheduler.step()

            # Update meters
            batch_size = images.size(0)
            loss_meter.update(total_loss.item(), batch_size)
            bce_meter.update(bce_loss.item(), batch_size)
            scl_meter.update(scl_loss.item(), batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': f'{self.lr_scheduler.get_lr():.2e}'
            })

        return {
            'loss': loss_meter.avg,
            'bce_loss': bce_meter.avg,
            'scl_loss': scl_meter.avg,
            'lr': self.lr_scheduler.get_lr()
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        loss_meter = AverageMeter('Loss')

        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]", leave=False)

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.bce_criterion(logits.squeeze(), labels)

            # Apply sigmoid to logits to get probabilities for metrics
            predictions = torch.sigmoid(logits.squeeze())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss_meter.update(loss.item(), images.size(0))

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        auc = roc_auc_score(all_labels, all_predictions)
        accuracy = accuracy_score(all_labels, (all_predictions > 0.5).astype(int))

        return {'loss': loss_meter.avg,
                'auc': auc,
                'accuracy': accuracy
                }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, output_dir: Optional[str] = None) -> Dict[str, List[float]]:
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Cosine scheduler
        if self.lr_scheduler_type == 'cosine' and self.lr_scheduler is None:
            self.batches_per_epoch = len(train_loader)
            self.lr_scheduler = CosineAnnealingWarmupScheduler(
                self.optimizer,
                initial_lr=self.initial_lr,
                min_lr=self.min_lr,
                warmup_epochs=self.warmup_epochs,
                max_epochs=num_epochs,
                batches_per_epoch=self.batches_per_epoch
            )
            print(f"Using {self.scheduler_type_name} LR scheduler")
            print(f"  Warmup epochs: {self.warmup_epochs}")
            print(f"  Batches per epoch: {self.batches_per_epoch}")

        print(f"Training for {num_epochs} epochs...")
        print(f"  Initial LR: {self.initial_lr}")
        print(f"  SCL weight: {self.scl_lambda}")
        print(f"  Mixed precision: {self.use_mixed_precision}")
        print()

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_bce'].append(train_metrics['bce_loss'])
            self.history['train_scl'].append(train_metrics['scl_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(train_metrics['lr'])

            # Print epoch summary with enhanced metrics
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train: loss={train_metrics['loss']:.4f} "
                  f"(BCE={train_metrics['bce_loss']:.4f}, "
                  f"SCL={train_metrics['scl_loss']:.4f}), "
                  f"LR={train_metrics['lr']:.2e}")
            print(f"  Val:   loss={val_metrics['loss']:.4f}, "
                  f"AUC={val_metrics['auc']:.4f}, "
                  f"Acc={val_metrics['accuracy']:.4f}")

            # Show improvement or degradation
            if epoch > 0:
                auc_delta = val_metrics['auc'] - self.history['val_auc'][-2]
                if abs(auc_delta) < 0.0001:
                    print(f"  → AUC unchanged ({auc_delta:+.5f})")
                elif auc_delta > 0:
                    print(f"  → AUC improved ({auc_delta:+.5f})")
                else:
                    print(f"  → AUC decreased ({auc_delta:+.5f})")

            # Check for best model
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.best_epoch = epoch
                self.best_model_state = {
                    'model': self.model.state_dict(),
                    'scl': self.scl_criterion.state_dict(),
                    'epoch': epoch,
                    'auc': val_metrics['auc']
                }
                print(f"  -> New best model (AUC: {val_metrics['auc']:.4f})")

                if output_dir:
                    save_checkpoint(
                        output_dir / 'best_model.pth',
                        self.model,
                        self.optimizer,
                        self.lr_scheduler,
                        self.scl_criterion,
                        epoch,
                        val_metrics['auc']
                    )

            # Early stopping check
            if self.early_stopping(val_metrics['auc'], epoch):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            print()

        # Save final model
        if output_dir:
            save_checkpoint(
                output_dir / 'final_model.pth',
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.scl_criterion,
                epoch,
                val_metrics['auc']
            )

        print(f"\nTraining complete!")
        print(f"  Best AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch + 1}")

        return dict(self.history)

    def load_best_model(self):
        """Load the best model state saved during training."""
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state['model'])
            self.scl_criterion.load_state_dict(self.best_model_state['scl'])
            print(f"Loaded best model from epoch {self.best_model_state['epoch'] + 1}")


if __name__ == '__main__':
    from ..models import SCADENet
    model = SCADENet()
    config = {'initial_lr': 1e-3,
              'scl_lambda': 0.1,
              'use_mixed_precision': True
              }

    trainer = SCADENetTrainer(model, config)
    print("Trainer initialized successfully.")
    print(f"  Device: {trainer.device}")
    print(f"  Initial LR: {trainer.initial_lr}")
    print(f"  SCL lambda: {trainer.scl_lambda}")