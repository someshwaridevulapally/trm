"""
Grid Encoder for TRM.

Maps a 2-D grid tensor (B, C, H, W) to a sequence of token embeddings
(B, 1, hidden_dim) that can be fed into the TinyTransformer core.

For grid tasks (maze, puzzle, ARC) the spatial content is first processed
by a lightweight CNN to extract local features, then projected to the
model's hidden dimension via a linear layer.

The output shape is (B, 1, hidden_dim) — a single "summary token" —
which is what the TRM paper uses for flat (non-sequential) reasoning tasks
where the entire grid is encoded into one latent vector fed as x.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    CNN-based grid encoder that produces a single summary token per sample.

    Architecture:
        Conv2d(C_in, 32, 3, pad=1) → ReLU
        Conv2d(32, 64, 3, pad=1)   → ReLU
        AdaptiveAvgPool2d(4)        → Flatten
        Linear(64*4*4, hidden_dim)  → output token (B, 1, hidden_dim)

    Args:
        in_channels (int): Number of input channels. Default 1.
        hidden_dim  (int): Output embedding dimensionality. Default 512.
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),          # fixed spatial output: 4×4
        )
        self.fc = nn.Linear(64 * 4 * 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Grid input, shape (B, C, H, W).

        Returns:
            Token embedding, shape (B, 1, hidden_dim).
            The leading 1 is the "sequence" dimension expected by TRMCore.
        """
        feat = self.cnn(x)                          # (B, 64, 4, 4)
        feat = feat.view(feat.size(0), -1)          # (B, 1024)
        feat = self.fc(feat)                        # (B, hidden_dim)
        return feat.unsqueeze(1)                    # (B, 1, hidden_dim)
