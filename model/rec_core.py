"""
GRU-based Recursive Core with an iteration loop.

The RecCore runs a GRUCell up to `max_iters` times, feeding the encoded input
at every step and optionally concatenating an external context vector
(e.g., ARC task embedding).  Early stopping is triggered when the hidden state
converges (see utils.convergence).
"""

import torch
import torch.nn as nn
from utils.convergence import ConvergenceChecker


class RecCore(nn.Module):
    """
    Recursive processing core built on a GRUCell.

    At each iteration the cell receives:
        input  = concat(encoded_input, context)   [if context is provided]
        input  = encoded_input                     [otherwise]
        h_next = GRUCell(input, h_prev)

    Args:
        hidden_dim (int):   Dimensionality of the hidden state. Default 128.
        input_dim  (int):   Dimensionality of the encoded input.  Default 128.
                            (Must equal hidden_dim when no context is used.)
        context_dim (int):  Extra context vector size (0 = none). Default 0.
        max_iters  (int):   Maximum number of recurrence steps. Default 10.
        epsilon    (float): Convergence threshold for early stopping. Default 1e-3.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        input_dim: int = 128,
        context_dim: int = 0,
        max_iters: int = 10,
        epsilon: float = 1e-3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_iters = max_iters

        # GRUCell input size = encoded input + optional context
        self.gru_cell = nn.GRUCell(input_dim + context_dim, hidden_dim)
        self.convergence = ConvergenceChecker(epsilon=epsilon)

    def forward(
        self,
        encoded: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """
        Run the iterative GRU loop.

        Args:
            encoded: Encoded input features, shape (B, input_dim).
            context: Optional context vector, shape (B, context_dim).

        Returns:
            h:          Final hidden state, shape (B, hidden_dim).
            num_iters:  Number of iterations actually executed.
        """
        batch_size = encoded.size(0)
        device = encoded.device

        # Initialise hidden state to zeros
        h = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Build the per-step input (same every iteration)
        if context is not None:
            gru_input = torch.cat([encoded, context], dim=-1)  # (B, input_dim + context_dim)
        else:
            gru_input = encoded  # (B, input_dim)

        num_iters = 0
        for t in range(self.max_iters):
            h_prev = h
            h = self.gru_cell(gru_input, h_prev)  # (B, hidden_dim)
            num_iters = t + 1

            # Early stopping on convergence (only at eval time to keep
            # training gradients flowing through all iterations)
            if not self.training and self.convergence.check(h_prev, h):
                break

        return h, num_iters
