import torch
import torch.nn as nn


class ECA(nn.Module):
    """
    Efficient Channel Attention module.

    Uses 1D convolution instead of fully connected layers for
    channel attention, adding only k parameters per module (where
    k is the kernel size, typically 3).

    Args:
        channels: Number of input channels
        k: Kernel size for 1D convolution (default: 3)

    Architecture:
        Input [B, C, H, W]
            -> Global Avg Pool -> [B, C, 1, 1]
            -> Squeeze/Transpose -> [B, 1, C]
            -> Conv1D -> [B, 1, C]
            -> Transpose/Unsqueeze -> [B, C, 1, 1]
            -> Sigmoid
            -> Element-wise multiply with input
        Output [B, C, H, W]

    Parameters per module: k (kernel_size, default 3 = 3 params)
    """
    def __init__(self, channels: int, k: int = 3):
        super(ECA, self).__init__()
        self.channels = channels
        self.k = k
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D convolution for channel attention (no bias for efficiency)
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=1,
                              kernel_size=k,
                              padding=k // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Attention-weighted tensor of shape [B, C, H, W]
        """
        # Global average pooling: [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)

        # Squeeze and transpose: [B, C, 1, 1] -> [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)

        # 1D convolution: [B, 1, C] -> [B, 1, C]
        y = self.conv(y)

        # Transpose back and unsqueeze: [B, 1, C] -> [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Apply channel attention: [B, C, H, W] * [B, C, 1, 1]
        return x * self.sigmoid(y)

    def extra_repr(self) -> str:
        return f'channels={self.channels}, k={self.k}'


if __name__ == '__main__':
    eca = ECA(channels=16, k=3)
    x = torch.randn(2, 16, 32, 32)
    out = eca(x)

    print(f"ECA Module Test:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters:   {sum(p.numel() for p in eca.parameters())}")