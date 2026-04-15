"""
PyTorch Dataset for the 8-puzzle task.

Each sample consists of:
  - A 9-channel 3×3 tensor (one-hot encoding of tile positions)
  - The next optimal action (flat index 0–8 of the cell the blank swaps with)

Supports curriculum learning: difficulty (scramble depth) can be set externally
and increased across epochs.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from tasks.puzzle.puzzle_env import scramble_puzzle


def _state_to_onehot(state: np.ndarray) -> np.ndarray:
    """
    Convert a 3×3 puzzle state (values 0–8) to a (9, 3, 3) one-hot tensor.
    Channel k has a 1 at the position of tile k.
    """
    oh = np.zeros((9, 3, 3), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            oh[state[r, c], r, c] = 1.0
    return oh


class PuzzleDataset(Dataset):
    """
    8-puzzle dataset with curriculum support.

    Generates puzzles scrambled `min_moves` to `max_moves` random steps from
    the goal and stores per-step (state, action) pairs from the A* solution.

    Args:
        num_puzzles (int):  Number of puzzles to generate. Default 10_000.
        min_moves (int):    Minimum scramble depth. Default 5.
        max_moves (int):    Maximum scramble depth. Default 30.
        seed (int):         Base RNG seed.
    """

    def __init__(
        self,
        num_puzzles: int = 10_000,
        min_moves: int = 5,
        max_moves: int = 30,
        seed: int = 42,
    ):
        super().__init__()
        self.samples: List[Tuple[np.ndarray, int]] = []  # (state, action)

        print(f"Generating {num_puzzles} puzzles (scramble {min_moves}–{max_moves})…")
        for i in tqdm(range(num_puzzles), desc="Puzzles"):
            import random as _rng
            _rng.seed(seed + i)
            depth = _rng.randint(min_moves, max_moves)
            state, actions = scramble_puzzle(num_moves=depth, seed=seed + i)

            # Unroll the A* solution into per-step samples
            for a in actions:
                self.samples.append((state.copy(), a))
                # Apply the action to get the next state
                br, bc = int(np.argwhere(state == 0)[0, 0]), int(np.argwhere(state == 0)[0, 1])
                tr, tc = a // 3, a % 3  # position of the tile that moves
                state[br, bc] = state[tr, tc]
                state[tr, tc] = 0

        print(f"  Total training samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        state, action = self.samples[idx]
        tensor = torch.tensor(_state_to_onehot(state))  # (9, 3, 3)
        return tensor, action
