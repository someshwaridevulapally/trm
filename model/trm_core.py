"""
TRM Core — Two-Level Recursive Computation Engine.

Implements the nested micro/macro recursion described in:
  "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871v1)

The TRMCore maintains two latent states:
  z_L  — abstract reasoning / scratchpad state (never decoded directly)
  z_H  — embedded solution             (decoded to output at every macro step)

Update equations (shared network f):
  Micro:  z_L ← f(z_L + z_H + x)    [repeated n times per macro step]
  Macro:  z_H ← f(z_L + z_H)         [once per macro step, no x]

Training — Bootstrapped Backpropagation:
  • Macro steps 0 … T-2 are run under torch.no_grad() (bootstrapping).
  • Only the final macro step uses full gradient tracking.
  • This is cheaper than full-BPTT and avoids IFT / memory explosion.

Deep Supervision:
  z_H is decoded at EVERY macro step so trainers can compute loss at each
  step.  TRMCore returns z_H_list (all intermediate + final z_H values).
"""

import torch
import torch.nn as nn
from contextlib import nullcontext
from typing import Optional, Tuple, List

from model.transformer_block import TinyTransformer


class TRMCore(nn.Module):
    """
    Tiny Recursion Model core with nested micro/macro update loops.

    Args:
        hidden_dim  (int):   Dimensionality of z_L, z_H, and x. Default 512.
        T           (int):   Number of macro steps.  Default 3 (from paper).
        n           (int):   Number of micro steps per macro step. Default 6.
        n_heads     (int):   Attention heads in the shared Transformer. Default 8.
        n_layers    (int):   Transformer layers (2 = from paper).
        dropout     (float): Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        T: int = 3,
        n: int = 6,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.T = T          # macro steps
        self.n = n          # micro steps per macro step

        # Single shared Transformer used for BOTH f_L (micro) and f_H (macro)
        self.f = TinyTransformer(
            d_model=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Internal update helpers
    # ------------------------------------------------------------------

    def _micro_update(self, z_L: torch.Tensor, z_H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        z_L ← f(z_L + z_H + x)

        All three tensors must have shape (B, seq_len, hidden_dim).
        Returns updated z_L.
        """
        inp = z_L + z_H + x
        return self.f(inp)

    def _macro_update(self, z_L: torch.Tensor, z_H: torch.Tensor) -> torch.Tensor:
        """
        z_H ← f(z_L + z_H)

        x is NOT used in the macro update (paper eq. 4).
        Returns updated z_H.
        """
        inp = z_L + z_H
        return self.f(inp)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run the full TRM recursion.

        Args:
            x: Encoded input, shape (B, seq_len, hidden_dim).
               Typically seq_len = 1 for flat (non-sequential) tasks.

        Returns:
            z_H:       Final macro-step hidden state, shape (B, seq_len, hidden_dim).
            z_H_list:  List of z_H tensors after each macro step (for deep supervision).
                       Length = T.  All elements have shape (B, seq_len, hidden_dim).
        """
        B, S, D = x.shape
        device = x.device

        # Initialise latent states to zeros
        z_L = torch.zeros(B, S, D, device=device)
        z_H = torch.zeros(B, S, D, device=device)

        z_H_list: List[torch.Tensor] = []

        for t in range(self.T):
            # ── Bootstrapping: run all but the last macro step with no_grad ──
            is_last = (t == self.T - 1)
            ctx = nullcontext() if is_last else torch.no_grad()

            with ctx:
                # ── Micro loop: update z_L n times ──
                for _ in range(self.n):
                    z_L = self._micro_update(z_L, z_H, x)

                # ── Macro update: update z_H once ──
                z_H = self._macro_update(z_L, z_H)

            # Collect z_H for deep supervision
            # Detach non-final steps so only the last contributes gradients
            z_H_list.append(z_H if is_last else z_H.detach())

        return z_H, z_H_list
