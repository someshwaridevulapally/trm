"""
RecursiveNet — Main TRM Model.

Pipeline:
    Encoder  →  TRMCore (micro/macro loops)  →  Decoder

This replaces the old GRU-based RecursiveNet and now wires together
the Encoder, TRMCore, and Decoder as described in:
  "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871v1)

Key differences from the old model:
  • Core is TRMCore (2-level Transformer recursion), not GRUCell.
  • Returns (final_logits, z_H_list) for deep supervision.
  • No explicit "context" parameter — ARC uses ARCModel wrapper.
  • hidden_dim default is 512 (paper setting).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from model.encoder import Encoder
from model.trm_core import TRMCore
from model.decoder import Decoder


class RecursiveNet(nn.Module):
    """
    Full TRM: Encoder → TRMCore → Decoder.

    Args:
        in_channels (int):             Input grid channels.
        hidden_dim  (int):             d_model throughout the model. Default 512.
        head_sizes  (Dict[str, int]):  task_name → output_size.
        T           (int):             Macro steps. Default 3.
        n           (int):             Micro steps per macro. Default 6.
        n_heads     (int):             Attention heads in Transformer. Default 8.
        n_layers    (int):             Transformer layers. Default 2.
        dropout     (float):           Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 512,
        head_sizes: Optional[Dict[str, int]] = None,
        T: int = 3,
        n: int = 6,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if head_sizes is None:
            head_sizes = {"maze": 4, "puzzle": 9}

        self.encoder = Encoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.trm_core = TRMCore(
            hidden_dim=hidden_dim,
            T=T,
            n=n,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(hidden_dim=hidden_dim, head_sizes=head_sizes)

    def forward(
        self,
        x: torch.Tensor,
        task: str,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Full forward pass.

        Args:
            x:          Grid input, shape (B, C, H, W).
            task:       Decoder head to use (e.g. "maze", "puzzle", "arc").
            return_all: If True, also return decoded logits for all intermediate
                        macro steps (for deep supervision during training).
                        If False, only the final-step logits are returned.

        Returns:
            logits:       Final macro-step logits. Shape depends on task.
            logits_list:  List of T logit tensors (all macro steps).
                          If return_all=False, this is [logits] (length 1).
        """
        # 1. Encode grid → (B, 1, hidden_dim)
        enc = self.encoder(x)

        # 2. Run TRMCore → z_H (B, 1, D), z_H_list [T × (B, 1, D)]
        z_H, z_H_list = self.trm_core(enc)

        # 3. Decode final z_H
        logits = self.decoder(z_H, task)

        if return_all:
            # Decode all intermediate z_H for deep supervision
            logits_list = self.decoder.decode_all(z_H_list, task)
        else:
            logits_list = [logits]

        return logits, logits_list
