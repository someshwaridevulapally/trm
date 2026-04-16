"""
Task-switchable Decoder for TRM.

Decodes z_H (the embedded solution from TRMCore) into task-specific logits.
Supports multiple tasks via named output heads.

Unlike the original GRU-based decoder, this version:
  1. Accepts z_H of shape (B, 1, hidden_dim) — the TRMCore token format.
  2. Is called at EVERY macro step for deep supervision.
  3. Uses two linear layers with a GELU activation between them.

Task → output size mapping:
    maze   →  4        (action: up / down / left / right)
    puzzle →  9        (slide-tile position classification)
    arc    →  H*W*10   (per-cell 10-class colour prediction)
    sudoku →  81*9     (per-cell 9-digit prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class Decoder(nn.Module):
    """
    Multi-head decoder: selects the appropriate output head by task name.

    Args:
        hidden_dim (int):          Size of the z_H embedding (d_model).
        head_sizes (Dict[str, int]): Mapping task_name → output_size.
            Example: {"maze": 4, "puzzle": 9, "arc": 300, "sudoku": 729}
    """

    def __init__(self, hidden_dim: int, head_sizes: Dict[str, int]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = nn.ModuleDict()
        for task_name, out_size in head_sizes.items():
            self.heads[task_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_size),
            )

    def forward(self, z_H: torch.Tensor, task: str) -> torch.Tensor:
        """
        Decode z_H into logits for the given task.

        Args:
            z_H:  TRMCore output, shape (B, 1, hidden_dim).
                  The sequence dimension (1) is squeezed before decoding.
            task: Name of the task head to use.

        Returns:
            Logits whose last dimension equals head_sizes[task].
            Shape: (B, head_sizes[task])
        """
        if task not in self.heads:
            raise ValueError(
                f"Unknown task '{task}'. Available: {list(self.heads.keys())}"
            )
        # z_H: (B, 1, D) → squeeze → (B, D)
        h = z_H.squeeze(1)
        return self.heads[task](h)

    def decode_all(self, z_H_list: list, task: str) -> list:
        """
        Convenience helper: decode every z_H in a list (for deep supervision).

        Args:
            z_H_list: List of T tensors, each (B, 1, hidden_dim).
            task:     Task key.

        Returns:
            List of T logit tensors.
        """
        return [self.forward(z_H, task) for z_H in z_H_list]
