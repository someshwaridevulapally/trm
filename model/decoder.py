"""
Task-switchable Decoder for TRM.

Decodes z_H (the embedded solution from TRMCore) into task-specific logits.
Supports multiple tasks via named output heads.

z_H is now (B, seq_len, hidden_dim) where seq_len = num_cells (e.g. 9 for puzzle).
We mean-pool across the sequence dimension before applying the output head so
that the decoder works regardless of seq_len and aggregates all cell information.

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
    Multi-head decoder: pools z_H tokens then applies task-specific head.

    Args:
        hidden_dim (int):            Size of the z_H embedding (d_model).
        head_sizes (Dict[str, int]): Mapping task_name → output_size.
            Example: {"maze": 4, "puzzle": 9, "arc": 300, "sudoku": 729}
    """

    def __init__(self, hidden_dim: int, head_sizes: Dict[str, int]):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared pooled-feature normalisation (applied after mean-pool)
        self.pool_norm = nn.LayerNorm(hidden_dim)

        self.heads = nn.ModuleDict()
        for task_name, out_size in head_sizes.items():
            # Deeper head: Linear → GELU → LayerNorm → Linear → GELU → Linear(out)
            self.heads[task_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, out_size),
            )

    def _pool(self, z_H: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool z_H over the sequence dimension.

        Args:
            z_H: (B, seq_len, hidden_dim)  — seq_len = 9 for puzzle, 1 for maze.

        Returns:
            pooled: (B, hidden_dim)
        """
        pooled = z_H.mean(dim=1)          # (B, hidden_dim)
        return self.pool_norm(pooled)

    def forward(self, z_H: torch.Tensor, task: str) -> torch.Tensor:
        """
        Decode z_H into logits for the given task.

        Args:
            z_H:  TRMCore output, shape (B, seq_len, hidden_dim).
                  Mean-pooled over seq_len before the head.
            task: Name of the task head to use.

        Returns:
            Logits, shape (B, head_sizes[task]).
        """
        if task not in self.heads:
            raise ValueError(
                f"Unknown task '{task}'. Available: {list(self.heads.keys())}"
            )
        h = self._pool(z_H)               # (B, hidden_dim)
        return self.heads[task](h)         # (B, out_size)

    def decode_all(self, z_H_list: list, task: str) -> list:
        """
        Convenience helper: decode every z_H in a list (for deep supervision).

        Args:
            z_H_list: List of T tensors, each (B, seq_len, hidden_dim).
            task:     Task key.

        Returns:
            List of T logit tensors, each (B, out_size).
        """
        return [self.forward(z_H, task) for z_H in z_H_list]
