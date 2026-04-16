"""
Sudoku PyTorch dataset.

Each sample is a (puzzle, solution) pair where:
  - puzzle:   (9, 9, 10) one-hot tensor — 10 classes (0=blank, 1–9=digit)
  - solution: (9, 9) long tensor of class indices 0–9
              (0 = class for "blank placeholder", 1–9 = actual digit)

Difficulty controls `num_clues` per puzzle:
  easy    = 36 clues
  medium  = 27 clues
  hard    = 22 clues
  extreme = 17–21 clues  (paper benchmark)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple

from tasks.sudoku.sudoku_env import generate_sudoku


DIFFICULTY_CLUES = {
    "easy":    36,
    "medium":  27,
    "hard":    22,
    "extreme": 20,   # ~17–21, we use 20 as default extreme
}


def board_to_onehot(board: np.ndarray) -> np.ndarray:
    """
    Convert a (9, 9) integer board to a (10, 9, 9) one-hot float32 tensor.
    Channel 0 = blank (value 0), channels 1–9 = digit 1–9.
    Shape is (C, H, W) to match CNN convention.
    """
    out = np.zeros((10, 9, 9), dtype=np.float32)
    for r in range(9):
        for c in range(9):
            out[board[r, c], r, c] = 1.0
    return out


class SudokuDataset(Dataset):
    """
    Generates Sudoku (puzzle, solution) pairs on-the-fly.

    Args:
        num_puzzles  (int):  Number of puzzles to generate. Default 10 000.
        difficulty   (str):  'easy', 'medium', 'hard', or 'extreme'. Default 'extreme'.
        num_clues    (int):  Override difficulty with explicit clue count (17–45).
        seed         (int):  Base random seed.
    """

    def __init__(
        self,
        num_puzzles: int = 10_000,
        difficulty: str = "extreme",
        num_clues: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.num_puzzles = num_puzzles
        self.seed        = seed
        self.clues       = num_clues if num_clues is not None else DIFFICULTY_CLUES[difficulty]

        # Pre-generate all puzzles for reproducible batches
        self.puzzles   = []    # list of (9, 9) int arrays
        self.solutions = []    # list of (9, 9) int arrays

        for i in range(num_puzzles):
            puzzle, solution = generate_sudoku(num_clues=self.clues, seed=seed + i)
            self.puzzles.append(puzzle)
            self.solutions.append(solution)

    def __len__(self) -> int:
        return self.num_puzzles

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            puzzle_tensor:   (10, 9, 9) float32 — one-hot encoded puzzle input
            solution_tensor: (9, 9) long — digit class labels 1–9
        """
        puzzle   = self.puzzles[idx]
        solution = self.solutions[idx]

        puzzle_tensor   = torch.tensor(board_to_onehot(puzzle),    dtype=torch.float32)
        solution_tensor = torch.tensor(solution,                   dtype=torch.long)
        return puzzle_tensor, solution_tensor
