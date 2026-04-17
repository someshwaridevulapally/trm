"""
PyTorch Dataset built from BFS-solved mazes.

Each sample is a (grid_tensor, action) pair where the grid encodes
the maze plus the agent's current position, and the action is the
next BFS-optimal move.

For training we unroll the full BFS path: for a path of length L the
maze produces L training samples (one per step along the path).
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm

from tasks.maze.maze_env import generate_solvable_maze, ACTION_DELTAS


class MazeDataset(Dataset):
    """
    On-the-fly maze dataset.

    Each item:
        grid_tensor:  (2, H, W) float tensor
                        channel 0 = maze walls (0/1)
                        channel 1 = agent position (one-hot)
        action:       int in {0,1,2,3} = next optimal action

    Args:
        num_mazes (int):   Number of mazes to generate. Default 10_000.
        maze_h (int):      Logical maze height. Default 10.
        maze_w (int):      Logical maze width.  Default 10.
        seed (int):        Base seed for reproducibility.
    """

    def __init__(
        self,
        num_mazes: int = 5_000,
        maze_h: int = 10,
        maze_w: int = 10,
        seed: int = 42,
    ):
        super().__init__()
        self.samples: List[Tuple[np.ndarray, int, int, int]] = []  # (grid, pos_r, pos_c, action)

        print(f"Generating {num_mazes} mazes ({maze_h}×{maze_w})…")
        for i in tqdm(range(num_mazes), desc="Mazes"):
            grid, start, goal, actions = generate_solvable_maze(
                height=maze_h, width=maze_w, seed=seed + i,
            )
            # Unroll path into per-step samples
            r, c = start
            for a in actions:
                self.samples.append((grid, r, c, a))
                dr, dc = ACTION_DELTAS[a]
                r, c = r + dr, c + dc

        print(f"  Total training samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        grid, pos_r, pos_c, action = self.samples[idx]
        H, W = grid.shape

        # Channel 0: maze layout
        ch_maze = grid.astype(np.float32)

        # Channel 1: agent position indicator
        ch_pos = np.zeros((H, W), dtype=np.float32)
        ch_pos[pos_r, pos_c] = 1.0

        grid_tensor = torch.tensor(np.stack([ch_maze, ch_pos], axis=0))  # (2, H, W)
        return grid_tensor, action
