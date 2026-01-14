import torch
from .eca import ECA
import torch.nn as nn
from typing import Optional, Tuple, Union
from .constrained_conv import ConstrainedConv2d


class SCADENet(nn.Module):
    def __init__(self, input_channels: int = 4, constrained_conv_option: str = 'A', eca_kernel_size: int = 3,
                 dropout_rate: float = 0.5):
        super(SCADENet, self).__init__()
        self.input_channels = input_channels

        self.block1_conv = ConstrainedConv2d(
            in_channels=input_channels,
            out_channels=8,
            kernel_size=3,
            option=constrained_conv_option
        )
        self.block1_act = nn.SiLU()
        self.block1_bn = nn.BatchNorm2d(8)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2_conv = nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=5,
            padding=2
        )
        self.block2_act = nn.SiLU()
        self.block2_bn = nn.BatchNorm2d(8)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 64x64x8 -> 32x32x16 (with ECA)
        self.block3_conv = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5,
            padding=2
        )
        self.block3_act = nn.SiLU()
        self.block3_eca = ECA(channels=16, k=eca_kernel_size)
        self.block3_bn = nn.BatchNorm2d(16)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 32x32x16 -> 8x8x16 (with ECA)
        self.block4_conv = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            padding=2
        )
        self.block4_act = nn.SiLU()
        self.block4_eca = ECA(channels=16, k=eca_kernel_size)
        self.block4_bn = nn.BatchNorm2d(16)
        self.block4_pool = nn.MaxPool2d(kernel_size=4, stride=4)

        # Classifier Head
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(1024, 16)
        self.fc1_act = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(16, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 4, 256, 256] (RGBD)
            return_features: If True, also return features from flatten layer

        Returns:
            logits: Raw prediction logits [B, 1] (use sigmoid for probabilities)
            features: (optional) Feature vector [B, 1024]
        """
        # Block 1: ConstrainedConv + SiLU + BN + Pool
        x = self.block1_conv(x)
        x = self.block1_act(x)
        x = self.block1_bn(x)
        x = self.block1_pool(x)

        # Block 2: Conv + SiLU + BN + Pool
        x = self.block2_conv(x)
        x = self.block2_act(x)
        x = self.block2_bn(x)
        x = self.block2_pool(x)

        # Block 3: Conv + SiLU + ECA + BN + Pool
        x = self.block3_conv(x)
        x = self.block3_act(x)
        x = self.block3_eca(x)
        x = self.block3_bn(x)
        x = self.block3_pool(x)

        # Block 4: Conv + SiLU + ECA + BN + Pool
        x = self.block4_conv(x)
        x = self.block4_act(x)
        x = self.block4_eca(x)
        x = self.block4_bn(x)
        x = self.block4_pool(x)

        # Flatten -> [B, 1024]
        features = self.flatten(x)

        # Classifier
        x = self.dropout1(features)
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        if return_features:
            return logits, features
        return logits

    def get_feature_extractor(self) -> nn.Sequential:
        return nn.Sequential(
            # Block 1
            self.block1_conv,
            self.block1_act,
            self.block1_bn,
            self.block1_pool,
            # Block 2
            self.block2_conv,
            self.block2_act,
            self.block2_bn,
            self.block2_pool,
            # Block 3
            self.block3_conv,
            self.block3_act,
            self.block3_eca,
            self.block3_bn,
            self.block3_pool,
            # Block 4
            self.block4_conv,
            self.block4_act,
            self.block4_eca,
            self.block4_bn,
            self.block4_pool,
            # Flatten
            self.flatten
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self):
        """Print model summary."""
        print("=" * 60)
        print("SCADE-Net Model Summary")
        print("=" * 60)
        print(f"Input channels: {self.input_channels}")
        print(f"Input size: 256x256x{self.input_channels}")
        print("-" * 60)
        print("Architecture:")
        print("  Block 1: ConstrainedConv(4->8, 3x3) -> SiLU -> BN -> Pool(2)")
        print("  Block 2: Conv(8->8, 5x5) -> SiLU -> BN -> Pool(2)")
        print("  Block 3: Conv(8->16, 5x5) -> SiLU -> ECA -> BN -> Pool(2)")
        print("  Block 4: Conv(16->16, 5x5) -> SiLU -> ECA -> BN -> Pool(4)")
        print("  Classifier: Flatten(1024) -> FC(16) -> FC(1) (logits)")
        print("-" * 60)
        print(f"Total parameters: {self.count_parameters():,}")
        print(f"Parameter budget: 30,000")
        print(f"Within budget: {self.count_parameters() < 30000}")
        print("=" * 60)


def create_model(input_channels: int = 4, constrained_conv_option: str = 'A', eca_kernel_size: int = 3,
                 dropout_rate: float = 0.5, pretrained_path: Optional[str] = None,
                 device: Optional[torch.device] = None) -> SCADENet:
    model = SCADENet(input_channels=input_channels,
                     constrained_conv_option=constrained_conv_option,
                     eca_kernel_size=eca_kernel_size,
                     dropout_rate=dropout_rate
                     )

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained weights from {pretrained_path}")

    if device is not None:
        model = model.to(device)

    return model


if __name__ == '__main__':
    model = SCADENet(input_channels=4, constrained_conv_option='A')
    model.print_summary()

    print("\nForward pass test:")
    x = torch.randn(2, 4, 256, 256)
    out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")

    out, feat = model(x, return_features=True)
    print(f"  Features: {feat.shape}")

    print(f"\n  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")