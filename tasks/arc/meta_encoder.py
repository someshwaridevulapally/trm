"""
Meta-encoder for ARC demonstration pairs — upgraded to Transformer.

Replaces the old CNN + mean-pool approach with a proper Transformer encoder
that processes concatenated (input, output) demo pairs as token sequences.

Architecture:
    For each demo pair:
        Flatten input_onehot (10, H, W) + output_onehot (10, H, W) → (20, H, W)
        CNN patch embed → (B*N, num_patches, hidden_dim)
        Transformer encoder (2 layers, shared) → pooled pair embedding
    Mean-pool across N valid demos → task_embedding (B, context_dim)
"""

import torch
import torch.nn as nn
import math
from model.transformer_block import TinyTransformer


class MetaEncoder(nn.Module):
    """
    Transformer-based ARC demo-pair encoder.

    Each (input, output) demo pair is:
      1. Encoded by a CNN into a set of spatial patch tokens.
      2. Processed by a 2-layer Transformer for within-pair reasoning.
      3. Mean-pooled into a single pair embedding.
    All pair embeddings are then masked-mean-pooled into a task embedding.

    Args:
        grid_channels (int): Channels per grid (10 for one-hot ARC colours).
        context_dim   (int): Output task-embedding dimensionality.
        patch_size    (int): Spatial patch size for CNN tokenisation. Default 4.
    """

    def __init__(
        self,
        grid_channels: int = 10,
        context_dim: int = 512,
        patch_size: int = 4,
    ):
        super().__init__()
        self.context_dim = context_dim

        # CNN tokeniser: (B*N, 2*C, H, W) → (B*N, hidden, H/p, W/p) → tokens
        self.patch_embed = nn.Sequential(
            nn.Conv2d(2 * grid_channels, context_dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(inplace=True),
        )

        # Shared Transformer applied to the patch tokens of each demo pair
        self.transformer = TinyTransformer(
            d_model=context_dim,
            n_heads=max(1, context_dim // 64),  # e.g. 8 heads for dim=512
            n_layers=2,
        )

        # Final projection from pooled token to context_dim (may be identity if shapes match)
        self.out_proj = nn.Linear(context_dim, context_dim)
        self.norm = nn.LayerNorm(context_dim)

    def forward(
        self,
        demo_inputs:  torch.Tensor,          # (B, N, C, H, W)
        demo_outputs: torch.Tensor,          # (B, N, C, H, W)
        demo_mask:    torch.Tensor | None,   # (B, N)
    ) -> torch.Tensor:
        """
        Returns:
            task_embedding: (B, context_dim)
        """
        B, N, C, H, W = demo_inputs.shape

        # Stack input + output channels → (B*N, 2C, H, W)
        pairs = torch.cat([demo_inputs, demo_outputs], dim=2)          # (B, N, 2C, H, W)
        pairs_flat = pairs.view(B * N, 2 * C, H, W)

        # CNN patch embed → (B*N, context_dim, Ph, Pw)
        patches = self.patch_embed(pairs_flat)                          # (B*N, D, Ph, Pw)
        D, Ph, Pw = patches.shape[1], patches.shape[2], patches.shape[3]

        # Flatten spatial patches → tokens: (B*N, Ph*Pw, D)
        tokens = patches.view(B * N, D, Ph * Pw).permute(0, 2, 1)      # (B*N, Ph*Pw, D)

        # Transformer → (B*N, Ph*Pw, D)
        tokens = self.transformer(tokens)

        # Mean-pool over patch dimension → (B*N, D)
        pair_emb = tokens.mean(dim=1)                                   # (B*N, D)
        pair_emb = self.norm(self.out_proj(pair_emb))                   # (B*N, D)
        pair_emb = pair_emb.view(B, N, -1)                             # (B, N, D)

        # Masked mean-pool over demo dimension → (B, D)
        if demo_mask is not None:
            mask_exp  = demo_mask.unsqueeze(-1)                         # (B, N, 1)
            pair_emb  = pair_emb * mask_exp
            n_valid   = demo_mask.sum(dim=1, keepdim=True).clamp(min=1) # (B, 1)
            task_emb  = pair_emb.sum(dim=1) / n_valid                  # (B, D)
        else:
            task_emb  = pair_emb.mean(dim=1)                           # (B, D)

        return task_emb
