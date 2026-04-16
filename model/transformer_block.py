"""
Tiny 2-Layer Transformer Block with SwiGLU activation and RoPE.

This is the shared network f(·) used for BOTH the micro-update (f_L)
and macro-update (f_H) in the Tiny Recursion Model (TRM):

  "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871v1)

Architecture per block:
    Pre-LayerNorm → Multi-Head Self-Attention (with RoPE) → residual
    Pre-LayerNorm → SwiGLU FFN                            → residual

The full Transformer consists of 2 stacked blocks sharing a single
RotaryEmbedding module, as described in the paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model.rope import RotaryEmbedding, apply_rotary_pos_emb


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network as used in the TRM paper.

    SwiGLU(x) = Swish(W1·x) ⊗ (W2·x)  followed by W3

    The hidden dimension is scaled to 8/3 × d_model to maintain
    a similar parameter count to a standard 4× FFN.

    Args:
        d_model  (int): Input / output dimensionality.
        ffn_mult (float): Multiplier for the intermediate dimension.
                         Default 8/3 (standard SwiGLU scale).
    """

    def __init__(self, d_model: int, ffn_mult: float = 8 / 3):
        super().__init__()
        # Round to nearest multiple of 64 for hardware efficiency
        hidden = int(d_model * ffn_mult)
        hidden = (hidden + 63) // 64 * 64

        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate
        self.w2 = nn.Linear(d_model, hidden, bias=False)  # value
        self.w3 = nn.Linear(hidden, d_model, bias=False)  # output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) * value
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ---------------------------------------------------------------------------
# Transformer Attention Block
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embeddings (RoPE).

    Args:
        d_model   (int): Model dimensionality.
        n_heads   (int): Number of attention heads.
        rope      (RotaryEmbedding): Shared RoPE module.
        dropout   (float): Attention dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.rope = rope
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, seq_len, d_model).

        Returns:
            Output tensor, shape (B, seq_len, d_model).
        """
        B, S, D = x.shape
        qkv = self.qkv(x)                              # (B, S, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)                 # each (B, S, D)

        # Reshape to (B, n_heads, S, head_dim)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(S)
        q, k = apply_rotary_pos_emb(q, k, cos.to(x.device), sin.to(x.device))

        # Scaled dot-product attention
        att = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, n_heads, S, head_dim)

        # Merge heads
        att = att.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(att)


# ---------------------------------------------------------------------------
# Single Transformer Layer (Pre-norm, Attn + SwiGLU)
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """
    A single pre-norm Transformer layer with SwiGLU FFN and RoPE attention.

    Args:
        d_model (int):  Model dimensionality.
        n_heads (int):  Number of attention heads.
        rope    (RotaryEmbedding): Shared RoPE module.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rope, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full 2-Layer Tiny Transformer (the shared f in TRM)
# ---------------------------------------------------------------------------

class TinyTransformer(nn.Module):
    """
    The shared tiny 2-layer Transformer network f(·) used in the TRM.

    Both the micro-update f_L and macro-update f_H call this same network.
    It maps a sequence of token embeddings to an output of the same shape.

    Args:
        d_model   (int):  Hidden dimensionality. Default 512 (from paper).
        n_heads   (int):  Number of attention heads. Default 8.
        n_layers  (int):  Number of Transformer layers. Default 2 (from paper).
        dropout   (float): Dropout probability.
        max_seq_len (int): Max sequence length for RoPE cache.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model

        # Shared RoPE across all layers
        head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, self.rope, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens, shape (B, seq_len, d_model).

        Returns:
            Output tokens, shape (B, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm_out(x)
