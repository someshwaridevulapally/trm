"""
CNN Encoder for 2-D grid inputs.

Architecture:
  Conv2d(C_in, 32, 3, padding=1) → ReLU → Conv2d(32, 64, 3, padding=1) → ReLU
  → AdaptiveAvgPool2d(4) → Flatten → Linear(64*4*4, hidden_dim)

The encoder maps an arbitrary-sized 2-D grid to a fixed-length feature vector.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    2-layer CNN encoder that converts a (B, C, H, W) grid tensor
    into a (B, hidden_dim) feature vector.

    Args:
        in_channels (int):  Number of input channels. Default 1.
        hidden_dim  (int):  Dimensionality of the output feature vector. Default 128.
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # spatial output always 4×4
        )
        self.fc = nn.Linear(64 * 4 * 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input grid tensor, shape (B, C, H, W).

        Returns:
            Feature vector, shape (B, hidden_dim).
        """
        features = self.cnn(x)                    # (B, 64, 4, 4)
        features = features.view(features.size(0), -1)  # (B, 64*16)
        return self.fc(features)                  # (B, hidden_dim)
