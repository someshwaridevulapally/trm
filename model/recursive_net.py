"""
RecursiveNet – the main model.

Pipeline:   Encoder  →  Iterative RecCore  →  Decoder

The model accepts a 2-D grid tensor, encodes it with a CNN, runs the encoded
representation through a GRU-based recursive core for up to `max_iters` steps
(with convergence-based early stopping), and finally decodes the hidden state
through a task-specific output head.

For the ARC task an optional `context` vector (task embedding from
MetaEncoder) is concatenated into the RecCore input at every iteration.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from model.encoder import Encoder
from model.rec_core import RecCore
from model.decoder import Decoder


class RecursiveNet(nn.Module):
    """
    Full recursive neural network.

    Args:
        in_channels  (int):             Input grid channels (1 for binary mazes, etc.).
        hidden_dim   (int):             Hidden-state dimensionality throughout the model.
        head_sizes   (Dict[str, int]):  Task-name → output-size mapping for the decoder.
        max_iters    (int):             Maximum recurrence iterations.
        epsilon      (float):           Convergence threshold.
        context_dim  (int):             Extra context size for ARC (0 = disabled).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 128,
        head_sizes: Optional[Dict[str, int]] = None,
        max_iters: int = 10,
        epsilon: float = 1e-3,
        context_dim: int = 0,
    ):
        super().__init__()

        if head_sizes is None:
            head_sizes = {"maze": 4, "puzzle": 9}

        self.encoder = Encoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.rec_core = RecCore(
            hidden_dim=hidden_dim,
            input_dim=hidden_dim,
            context_dim=context_dim,
            max_iters=max_iters,
            epsilon=epsilon,
        )
        self.decoder = Decoder(hidden_dim=hidden_dim, head_sizes=head_sizes)

    def forward(
        self,
        x: torch.Tensor,
        task: str,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        """
        End-to-end forward pass.

        Args:
            x:       Grid input, shape (B, C, H, W).
            task:    Which decoder head to use ("maze", "puzzle", "arc").
            context: Optional ARC task embedding, shape (B, context_dim).

        Returns:
            logits:     Output logits from the selected decoder head.
            num_iters:  How many RecCore iterations were executed.
        """
        encoded = self.encoder(x)                               # (B, hidden_dim)
        h, num_iters = self.rec_core(encoded, context=context)  # (B, hidden_dim)
        logits = self.decoder(h, task=task)                     # (B, out_size)
        return logits, num_iters
