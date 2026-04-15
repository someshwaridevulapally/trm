"""
Task-switchable decoder (output head).

Holds a dictionary of named linear heads so the model can produce outputs
of different sizes depending on the active task:

    maze   → 4  (up / down / left / right)
    puzzle → 9  (slide tile 1–8 or blank-position classification)
    arc    → H * W * 10  (per-cell 10-class colour prediction)
"""

import torch
import torch.nn as nn
from typing import Dict


class Decoder(nn.Module):
    """
    Multi-head decoder that selects the appropriate linear head based
    on the `task` key provided at forward time.

    Args:
        hidden_dim  (int):                Size of the hidden-state input.
        head_sizes  (Dict[str, int]):     Mapping from task name → output size.
            Example: {"maze": 4, "puzzle": 9, "arc": 300}
    """

    def __init__(self, hidden_dim: int, head_sizes: Dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict()
        for task_name, out_size in head_sizes.items():
            self.heads[task_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_size),
            )

    def forward(self, h: torch.Tensor, task: str) -> torch.Tensor:
        """
        Args:
            h:    Hidden state from RecCore, shape (B, hidden_dim).
            task: Key that selects the output head (e.g. "maze").

        Returns:
            Logits whose last-dimension size equals head_sizes[task].
        """
        if task not in self.heads:
            raise ValueError(
                f"Unknown task '{task}'. Available: {list(self.heads.keys())}"
            )
        return self.heads[task](h)
