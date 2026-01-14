import torch
import torch.nn as nn
from typing import Dict, Tuple


class SingleCenterLoss(nn.Module):
    """
    Single-Center Loss for deepfake detection.

    Creates a hyperspherical decision boundary by learning a single
    center point representing "real" face features. Real samples are
    pulled toward this center while fake samples are pushed away.

    Loss formulation:
        L_SCL = max(0, margin - α × (mean_dist_fake - mean_dist_real))

    The goal is to maximize the distance between fake samples and the
    center while minimizing the distance between real samples and the
    center.

    Args:
        feature_dim: Dimension of feature vectors (1024 for SCADE-Net)
        margin: Desired margin between classes (default: 0.5)
        alpha: Balance coefficient (default: 0.5)
    """

    def __init__(self, feature_dim: int = 1024, margin: float = 0.5, alpha: float = 0.5):
        super(SingleCenterLoss, self).__init__()

        self.feature_dim = feature_dim
        self.margin = margin
        self.alpha = alpha

        # Learnable center for "real" class initialized with small random values
        self.center = nn.Parameter(torch.randn(feature_dim) * 0.01)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Single-Center Loss.

        Args:
            features: Feature vectors from flatten layer [B, feature_dim]
            labels: Binary labels [B] where 0=real, 1=fake

        Returns:
            loss: Scalar loss value
            metrics: Dictionary with distance statistics
        """
        # Compute Euclidean distances to center
        distances = torch.norm(features - self.center, dim=1)

        # Separate real and fake samples
        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        # Compute mean distances for each class
        if real_mask.sum() > 0:
            mean_dist_real = distances[real_mask].mean()
        else:
            mean_dist_real = torch.tensor(0.0, device=features.device)

        if fake_mask.sum() > 0:
            mean_dist_fake = distances[fake_mask].mean()
        else:
            mean_dist_fake = torch.tensor(0.0, device=features.device)

        # SCL objective: maximize (mean_dist_fake - mean_dist_real)
        # Loss = max(0, margin - alpha * (dist_fake - dist_real))
        loss = torch.clamp(
            self.margin - self.alpha * (mean_dist_fake - mean_dist_real),
            min=0.0
        )

        # Collect metrics
        metrics = {'dist_real': mean_dist_real.item(),
                   'dist_fake': mean_dist_fake.item(),
                   'separation': (mean_dist_fake - mean_dist_real).item(),
                   'scl_loss': loss.item()
                   }
        return loss, metrics

    def extra_repr(self) -> str:
        return (f'feature_dim={self.feature_dim}, '
                f'margin={self.margin}, '
                f'alpha={self.alpha}'
                )


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss down-weights easy examples and focuses training on hard
    negatives. This is particularly useful for imbalanced datasets where
    one class is much more prevalent than the other.

    Formula:
        FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    where:
        p_t = p if y = 1, else (1 - p)
        α_t = α if y = 1, else (1 - α)

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
               Higher alpha gives more weight to positive (fake) samples
        gamma: Focusing parameter (default: 2.0)
               Higher gamma gives more focus to hard examples
        reduction: Specifies the reduction to apply to output
                  ('mean', 'sum', 'none')
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure targets are float
        targets = targets.float()

        # Ensure logits are 1D
        if logits.dim() > 1:
            logits = logits.squeeze()

        # Compute BCE loss without reduction
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def extra_repr(self) -> str:
        return (
            f'alpha={self.alpha}, '
            f'gamma={self.gamma}, '
            f'reduction={self.reduction}'
        )


class CombinedLoss(nn.Module):
    """
    Combined loss function for SCADE-Net training.

    Combines Binary Cross-Entropy loss with Single-Center Loss:
        L_total = L_BCE + λ_SCL × L_SCL

    Args:
        scl_feature_dim: Feature dimension for SCL (default: 1024)
        scl_margin: Margin for SCL (default: 0.5)
        scl_alpha: Balance coefficient for SCL (default: 0.5)
        scl_lambda: Weight for SCL in total loss (default: 0.1)
    """
    def __init__(self, scl_feature_dim: int = 1024, scl_margin: float = 0.5, scl_alpha: float = 0.5, scl_lambda: float = 0.1):
        super(CombinedLoss, self).__init__()
        self.scl_lambda = scl_lambda
        self.bce = nn.BCELoss()
        self.scl = SingleCenterLoss(
            feature_dim=scl_feature_dim,
            margin=scl_margin,
            alpha=scl_alpha
        )

    def forward(self, predictions: torch.Tensor, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # BCE loss
        bce_loss = self.bce(predictions.squeeze(), labels.float())

        # SCL loss
        scl_loss, scl_metrics = self.scl(features, labels.long())

        # Combined loss
        total_loss = bce_loss + self.scl_lambda * scl_loss

        # Collect all metrics
        metrics = {'bce_loss': bce_loss.item(),
                   'scl_loss': scl_loss.item(),
                   'total_loss': total_loss.item(),
                   **scl_metrics
                   }
        return total_loss, metrics


if __name__ == '__main__':
    scl = SingleCenterLoss(feature_dim=1024, margin=0.5, alpha=0.5)
    features = torch.randn(10, 1024)
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    loss, metrics = scl(features, labels)

    print("SingleCenterLoss Test:")
    print(f"  Parameters:  {sum(p.numel() for p in scl.parameters())}")
    print(f"  Loss:        {loss.item():.4f}")
    print(f"  Dist (real): {metrics['dist_real']:.4f}")
    print(f"  Dist (fake): {metrics['dist_fake']:.4f}")
    print(f"  Separation:  {metrics['separation']:.4f}")

    print("\nCombinedLoss Test:")
    combined = CombinedLoss()
    predictions = torch.sigmoid(torch.randn(10, 1))
    total_loss, all_metrics = combined(predictions, features, labels)
    print(f"  Total loss: {total_loss.item():.4f}")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}")