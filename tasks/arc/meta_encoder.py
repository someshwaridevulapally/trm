"""
Meta-encoder for ARC demonstration pairs.

Processes each (input, output) demonstration pair through a shared CNN,
then mean-pools across all demos to produce a fixed-size task embedding
vector.  This embedding captures the "rule" demonstrated by the examples
and is concatenated into the RecCore input at every iteration.

Architecture:
    For each demo pair:
        concat(input_onehot, output_onehot) → (20, H, W)
        → 2-layer CNN → AdaptiveAvgPool2d(4) → flatten → linear → pair_embedding
    Mean-pool pair embeddings → task_embedding (context_dim,)
"""

import torch
import torch.nn as nn


class MetaEncoder(nn.Module):
    """
    Encodes ARC demonstration (input, output) pairs into a task embedding.

    Args:
        grid_channels (int):  Channels per grid (10 for one-hot colours).
        context_dim (int):    Output embedding dimensionality.
    """

    def __init__(self, grid_channels: int = 10, context_dim: int = 128):
        super().__init__()
        # Input is concatenation of input + output one-hot grids → 2 * grid_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(2 * grid_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(64 * 4 * 4, context_dim)

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        demo_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode demonstration pairs into a task embedding.

        Args:
            demo_inputs:  (B, N_demos, C, H, W) one-hot input grids.
            demo_outputs: (B, N_demos, C, H, W) one-hot output grids.
            demo_mask:    (B, N_demos) binary mask for valid demo slots.

        Returns:
            task_embedding: (B, context_dim)
        """
        B, N, C, H, W = demo_inputs.shape

        # Concatenate input + output along channel dim → (B, N, 2C, H, W)
        pairs = torch.cat([demo_inputs, demo_outputs], dim=2)

        # Reshape to process all pairs through CNN at once
        pairs_flat = pairs.view(B * N, 2 * C, H, W)       # (B*N, 2C, H, W)
        features = self.cnn(pairs_flat)                     # (B*N, 64, 4, 4)
        features = features.view(B * N, -1)                 # (B*N, 1024)
        pair_embeddings = self.fc(features)                 # (B*N, context_dim)
        pair_embeddings = pair_embeddings.view(B, N, -1)    # (B, N, context_dim)

        # Mean-pool across demos, respecting the mask
        if demo_mask is not None:
            # Expand mask: (B, N, 1)
            mask_expanded = demo_mask.unsqueeze(-1)
            pair_embeddings = pair_embeddings * mask_expanded
            # Sum and divide by number of valid demos
            n_valid = demo_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            task_embedding = pair_embeddings.sum(dim=1) / n_valid      # (B, context_dim)
        else:
            task_embedding = pair_embeddings.mean(dim=1)               # (B, context_dim)

        return task_embedding
