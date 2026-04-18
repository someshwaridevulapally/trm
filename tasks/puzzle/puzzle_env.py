import heapq
import random
from typing import List, Optional, Tuple

import numpy as np

# Goal configuration
GOAL_STATE = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 0]], dtype=np.int32)

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_NAMES = ["up", "down", "left", "right"]

# --- Optimized 1D Tuple internals for fast A* ---

GOAL_TUPLE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Precompute Manhattan distances for all tile values (1-8) at all positions (0-8)
MD_TABLE = {}
for val in range(1, 9):
    MD_TABLE[val] = {}
    goal_r, goal_c = (val - 1) // 3, (val - 1) % 3
    for pos in range(9):
        pos_r, pos_c = pos // 3, pos % 3
        MD_TABLE[val][pos] = abs(pos_r - goal_r) + abs(pos_c - goal_c)

# Precompute valid moves (neighbors) for each blank position (0-8)
NEIGHBOR_MAP = {}
for pos in range(9):
    r, c = pos // 3, pos % 3
    moves = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            moves.append(nr * 3 + nc)
    NEIGHBOR_MAP[pos] = moves

def _manhattan_distance_tuple(state_tuple: tuple) -> int:
    dist = 0
    for pos, val in enumerate(state_tuple):
        if val != 0:
            dist += MD_TABLE[val][pos]
    return dist

def _get_neighbours_tuple(state_tuple: tuple) -> List[Tuple[tuple, int]]:
    b_pos = state_tuple.index(0)
    neighbours = []
    state_list = list(state_tuple)
    for n_pos in NEIGHBOR_MAP[b_pos]:
        # Swap blank with neighbor
        state_list[b_pos], state_list[n_pos] = state_list[n_pos], state_list[b_pos]
        neighbours.append((tuple(state_list), n_pos))
        # Revert
        state_list[b_pos], state_list[n_pos] = state_list[n_pos], state_list[b_pos]
    return neighbours


# --- External/Legacy API (used by app.py and puzzle_trainer.py) ---

def _find_blank(state: np.ndarray) -> Tuple[int, int]:
    """Return (row, col) of the blank (0) cell."""
    pos = np.argwhere(state == 0)
    return int(pos[0, 0]), int(pos[0, 1])

def _state_to_tuple(state: np.ndarray) -> tuple:
    """Convert 3×3 array to a hashable tuple."""
    return tuple(state.flatten())


def solve_puzzle(state: np.ndarray) -> Optional[List[int]]:
    """
    Solve the 8-puzzle from `state` using A* with Manhattan distance.
    Returns: List of actions (flat indices 0–8 of the cell swapped with blank)
    """
    start = _state_to_tuple(state)
    goal = GOAL_TUPLE

    if start == goal:
        return []

    # Priority queue: (f_score, counter, state_tuple, action_list)
    counter = 0
    h = _manhattan_distance_tuple(start)
    open_set = [(h, counter, start, [])]
    g_scores = {start: 0}

    while open_set:
        f, _, current_tuple, actions = heapq.heappop(open_set)

        if current_tuple == goal:
            return actions

        # Standard A* duplicate filtering
        g = g_scores[current_tuple]

        for new_tuple, action in _get_neighbours_tuple(current_tuple):
            new_g = g + 1

            if new_tuple not in g_scores or new_g < g_scores[new_tuple]:
                g_scores[new_tuple] = new_g
                h = _manhattan_distance_tuple(new_tuple)
                counter += 1
                heapq.heappush(open_set, (new_g + h, counter, new_tuple, actions + [action]))

    return None  # unsolvable


def scramble_puzzle(
    num_moves: int = 20,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Generate a puzzle and its optimal A* solution."""
    if seed is not None:
        random.seed(seed)

    state_tuple = GOAL_TUPLE

    # Fast scramble using precomputed tuple neighbors
    for _ in range(num_moves):
        neighbours = _get_neighbours_tuple(state_tuple)
        state_tuple, _ = random.choice(neighbours)

    state = np.array(state_tuple, dtype=np.int32).reshape(3, 3)
    actions = solve_puzzle(state)
    assert actions is not None, "Scrambled puzzle must be solvable."
    return state, actions
