"""
Convergence checker for the recursive core.

Monitors the L2 norm of the difference between successive hidden states
and signals early stopping when the change drops below a threshold (epsilon).
"""

import torch


class ConvergenceChecker:
    """
    Checks whether the recursive core has converged by comparing
    consecutive hidden states: ||h_t - h_{t-1}||_2 < epsilon.

    Args:
        epsilon (float): Convergence threshold. Default 1e-3.
    """

    def __init__(self, epsilon: float = 1e-3):
        self.epsilon = epsilon

    def check(self, h_prev: torch.Tensor, h_curr: torch.Tensor) -> bool:
        """
        Compare two hidden states and return True if converged.

        Args:
            h_prev: Previous hidden state, shape (B, hidden_dim).
            h_curr: Current hidden state, shape (B, hidden_dim).

        Returns:
            True if the max per-sample L2 norm of the difference is below epsilon.
        """
        with torch.no_grad():
            # Per-sample L2 norm of the difference
            diff = torch.norm(h_curr - h_prev, dim=-1)  # (B,)
            # Converged only if ALL samples in the batch are below threshold
            return bool(diff.max().item() < self.epsilon)
