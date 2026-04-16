"""
Rotary Position Embeddings (RoPE).

Implements the rotation-based positional encoding used in the TRM paper:
  "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871v1)

RoPE encodes position information by rotating query and key vectors in
complex space, enabling relative position awareness without explicit
positional tokens.

Reference: Su et al., "RoFormer: Enhanced with Rotary Position Embedding" (2021)
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """
    Precomputes the cosine and sine rotation matrices for RoPE.

    Args:
        dim (int):        Head dimension (must be even). Usually head_dim = hidden_dim // n_heads.
        max_seq_len (int): Maximum sequence length to precompute. Default 2048.
        base (int):       Frequency base. Default 10000 (standard RoPE).
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}.")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies: θ_i = 1 / base^(2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Eagerly compute cos/sin tables up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])  # (1,1,seq_len,dim)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])  # (1,1,seq_len,dim)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns precomputed cos and sin tensors for the given sequence length.

        Returns:
            cos: (1, 1, seq_len, dim)
            sin: (1, 1, seq_len, dim)
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension in pairs: [x1, x2, ...] → [-x2, x1, ...]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q:   Query tensor,  shape (B, n_heads, seq_len, head_dim).
        k:   Key tensor,    shape (B, n_heads, seq_len, head_dim).
        cos: Cosine table,  shape (1, 1, seq_len, head_dim).
        sin: Sine table,    shape (1, 1, seq_len, head_dim).

    Returns:
        q_rot, k_rot: Rotated query and key tensors, same shapes as input.
    """
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
