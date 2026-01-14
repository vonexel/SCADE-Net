import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedConv2d(nn.Module):
    """
    Constrained Convolutional Layer with sum-to-zero constraint.
    The constraint enforces that each 2D kernel sums to zero, effectively
    making each filter a high-pass filter that emphasizes edges and
    manipulation artifacts while suppressing DC/content.

    Two implementation options:
        Option A (default): Sum-to-zero via center projection
            - Center weight = -(sum of all other weights)
            - Full kernel is trainable, constraint applied during forward

        Option B: Bayar & Stamm canonical
            - Center weight fixed at -1
            - Other 8 weights normalized to sum to 1

    Args:
        in_channels: Number of input channels (default: 4 for RGBD)
        out_channels: Number of output filters (default: 8)
        kernel_size: Kernel size, must be odd (default: 3)
        option: Constraint implementation ('A' or 'B')
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 8, kernel_size: int = 3, option: str = 'A'):
        super(ConstrainedConv2d, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert option in ['A', 'B'], "Option must be 'A' or 'B'"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.option = option
        self.padding = kernel_size // 2
        self.center_idx = kernel_size // 2

        if option == 'A':
            # Full kernel tensor, constraint applied via projection
            self.weight = nn.Parameter(
                torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
            )
        else:
            # Only non-center weights are trainable (k*k - 1 per filter per channel)
            num_non_center = kernel_size * kernel_size - 1
            self.weight = nn.Parameter(
                torch.randn(out_channels, in_channels, num_non_center) * 0.01
            )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _get_constrained_kernel(self) -> torch.Tensor:
        """
        Apply constraint to get final kernel weights.

        Returns:
            Constrained kernel tensor [out_channels, in_channels, k, k]
        """
        if self.option == 'A':
            # Option A: Sum-to-zero constraint
            # Center = -(sum of all other weights)
            kernel = self.weight.clone()
            c = self.center_idx

            # Create masks for center and non-center positions
            mask = torch.ones_like(kernel)
            mask[:, :, c, c] = 0

            center_mask = torch.zeros_like(kernel)
            center_mask[:, :, c, c] = 1

            # Apply constraint: kernel_new = kernel * mask + center_correction
            # where center = -sum(non-center weights)
            non_center_sum = (kernel * mask).sum(dim=(2, 3), keepdim=True)
            kernel = kernel * mask - non_center_sum * center_mask

            return kernel

        else:
            # Option B: Bayar & Stamm canonical
            # Center = -1, others normalized to sum to 1
            # Softmax ensures non-center weights sum to 1
            non_center_weights = F.softmax(self.weight, dim=-1)

            # Build full kernel
            kernel = torch.zeros(
                self.out_channels, self.in_channels,
                self.kernel_size, self.kernel_size,
                device=self.weight.device, dtype=self.weight.dtype
            )

            idx = 0
            c = self.center_idx
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    if i == c and j == c:
                        kernel[:, :, i, j] = -1.0
                    else:
                        kernel[:, :, i, j] = non_center_weights[:, :, idx]
                        idx += 1

            return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with constrained kernel.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, out_channels, H, W]
        """
        kernel = self._get_constrained_kernel()
        return F.conv2d(x, kernel, self.bias, padding=self.padding)

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'option={self.option}'
        )


if __name__ == '__main__':
    for option in ['A', 'B']:
        conv = ConstrainedConv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            option=option
        )
        x = torch.randn(2, 4, 256, 256)
        out = conv(x)

        kernel = conv._get_constrained_kernel()
        kernel_sums = kernel.sum(dim=(2, 3))

        print(f"\nConstrainedConv2d (Option {option}):")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Parameters:   {sum(p.numel() for p in conv.parameters())}")
        print(f"  Kernel sums (should be ~0):")
        print(f"    min: {kernel_sums.min().item():.6f}")
        print(f"    max: {kernel_sums.max().item():.6f}")
        print(f"    mean: {kernel_sums.mean().item():.6f}")