"""
Sudoku environment: generator and constraint-propagation solver.

Generates valid Sudoku puzzles at configurable difficulty levels by:
  1. Building a complete valid board via backtracking.
  2. Removing cells according to difficulty (number of clues retained).

The solver uses backtracking with MRV (Minimum Remaining Values) heuristic.

Difficulty levels (clues retained):
    easy   : 36-45 clues
    medium : 27-35 clues
    hard   : 22-26 clues
    extreme: 17-21 clues  (paper's "Sudoku-Extreme" benchmark)
"""

import numpy as np
import random
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Complete board generator
# ---------------------------------------------------------------------------

def _is_valid(board: np.ndarray, r: int, c: int, d: int) -> bool:
    """Check if digit d can be placed at (r, c) without conflict."""
    if d in board[r]:
        return False
    if d in board[:, c]:
        return False
    br, bc = (r // 3) * 3, (c // 3) * 3
    if d in board[br:br + 3, bc:bc + 3]:
        return False
    return True


def _generate_complete_board(seed: Optional[int] = None) -> np.ndarray:
    """
    Return a fully filled, valid 9x9 Sudoku board (values 1-9).
    Uses backtracking with random digit ordering.
    """
    rng = random.Random(seed)
    board = np.zeros((9, 9), dtype=np.int8)

    def _fill(pos: int) -> bool:
        if pos == 81:
            return True
        r, c = divmod(pos, 9)
        digits = list(range(1, 10))
        rng.shuffle(digits)
        for d in digits:
            if _is_valid(board, r, c, d):
                board[r, c] = d
                if _fill(pos + 1):
                    return True
                board[r, c] = 0
        return False

    _fill(0)
    return board


# ---------------------------------------------------------------------------
# Puzzle creator
# ---------------------------------------------------------------------------

def generate_sudoku(
    num_clues: int = 25,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Sudoku puzzle with exactly `num_clues` revealed cells.

    Args:
        num_clues: Number of given clues (17-81). 17 = hardest, 81 = trivial.
        seed:      Random seed.

    Returns:
        puzzle:   (9, 9) int8 array, 0 = blank, 1-9 = given clue.
        solution: (9, 9) int8 array, fully filled solution.
    """
    rng = random.Random(seed)
    solution = _generate_complete_board(seed=seed)
    puzzle = solution.copy()

    cells_to_remove = 81 - num_clues
    positions = list(range(81))
    rng.shuffle(positions)

    for pos in positions[:cells_to_remove]:
        r, c = divmod(pos, 9)
        puzzle[r, c] = 0

    return puzzle, solution


# ---------------------------------------------------------------------------
# Solver — backtracking with MRV heuristic
# ---------------------------------------------------------------------------

def _candidates(board: np.ndarray, r: int, c: int) -> set:
    """Return the set of valid digits for blank cell (r, c)."""
    used = set(board[r].tolist())
    used |= set(board[:, c].tolist())
    br, bc = (r // 3) * 3, (c // 3) * 3
    used |= set(board[br:br + 3, bc:bc + 3].flatten().tolist())
    used.discard(0)
    return set(range(1, 10)) - used


def solve_sudoku(puzzle: np.ndarray) -> Optional[np.ndarray]:
    """
    Solve a Sudoku puzzle using backtracking with MRV heuristic.

    Args:
        puzzle: (9, 9) int array, 0=blank, 1-9=given.

    Returns:
        Solved (9, 9) board, or None if unsolvable.
    """
    board = puzzle.astype(np.int8).copy()

    def _solve(board: np.ndarray) -> bool:
        # Find the empty cell with fewest valid candidates (MRV)
        best_r, best_c = -1, -1
        best_cands = None

        for r in range(9):
            for c in range(9):
                if board[r, c] == 0:
                    cands = _candidates(board, r, c)
                    if len(cands) == 0:
                        return False   # contradiction — backtrack
                    if best_cands is None or len(cands) < len(best_cands):
                        best_r, best_c = r, c
                        best_cands = cands
                        if len(best_cands) == 1:
                            break       # can't do better than 1 candidate
            if best_cands is not None and len(best_cands) == 1:
                break

        if best_r == -1:
            return True   # no blanks left — solved!

        for d in best_cands:
            board[best_r, best_c] = d
            if _solve(board):
                return True
            board[best_r, best_c] = 0   # undo

        return False   # exhausted all candidates — backtrack

    if _solve(board):
        return board
    return None
