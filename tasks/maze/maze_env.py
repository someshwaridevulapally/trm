"""
Random maze generator and BFS solver.

Generates 10×10 mazes using randomised recursive back-tracking (guaranteed
to be perfect mazes – exactly one path between any two cells). Then uses
BFS to find the shortest path from start (0, 0) to goal (H-1, W-1) and
converts it into an action sequence (0=up, 1=down, 2=left, 3=right).
"""

import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

# Action encoding
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_DELTAS = {
    UP:    (-1,  0),
    DOWN:  ( 1,  0),
    LEFT:  ( 0, -1),
    RIGHT: ( 0,  1),
}


def generate_maze(height: int = 10, width: int = 10, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random maze via recursive back-tracking.

    The returned grid has shape (2*height+1, 2*width+1) where:
      - 1 = wall
      - 0 = open passage

    Cell (r, c) in the logical grid maps to position (2*r+1, 2*c+1) in the
    full grid.  Walls between cells are the even-indexed rows/columns.

    Args:
        height: Number of logical rows.
        width:  Number of logical columns.
        seed:   Optional RNG seed for reproducibility.

    Returns:
        2-D numpy array of 0s and 1s.
    """
    if seed is not None:
        random.seed(seed)

    H, W = 2 * height + 1, 2 * width + 1
    grid = np.ones((H, W), dtype=np.int32)

    # Mark all logical cells as open
    for r in range(height):
        for c in range(width):
            grid[2 * r + 1, 2 * c + 1] = 0

    visited = np.zeros((height, width), dtype=bool)
    stack: List[Tuple[int, int]] = []

    # Start from (0, 0)
    start_r, start_c = 0, 0
    visited[start_r, start_c] = True
    stack.append((start_r, start_c))

    while stack:
        r, c = stack[-1]
        # Find unvisited neighbours
        neighbours = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc]:
                neighbours.append((nr, nc, dr, dc))

        if neighbours:
            nr, nc, dr, dc = random.choice(neighbours)
            # Remove wall between current cell and neighbour
            wall_r = 2 * r + 1 + dr
            wall_c = 2 * c + 1 + dc
            grid[wall_r, wall_c] = 0
            visited[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    return grid


def bfs_solve(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[int]]:
    """
    BFS shortest-path solver on the grid.

    Operates on the raw grid (not the logical grid), where 0 = passable.

    Args:
        grid:  2-D array with 0 = open, 1 = wall.
        start: (row, col) start position in the raw grid.
        goal:  (row, col) goal position in the raw grid.

    Returns:
        List of actions [0=up, 1=down, 2=left, 3=right] from start to goal,
        or None if no path exists.
    """
    H, W = grid.shape
    queue: deque = deque()
    queue.append((start, []))
    visited = set()
    visited.add(start)

    while queue:
        (r, c), actions = queue.popleft()
        if (r, c) == goal:
            return actions

        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), actions + [action]))

    return None  # no path found


def generate_solvable_maze(
    height: int = 10,
    width: int = 10,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], List[int]]:
    """
    Generate a maze and return it together with a BFS solution.

    Start = top-left logical cell → raw grid (1, 1).
    Goal  = bottom-right logical cell → raw grid (2*height-1, 2*width-1).

    Args:
        height: Logical rows.
        width:  Logical columns.
        seed:   RNG seed.

    Returns:
        (grid, start, goal, actions)
    """
    grid = generate_maze(height, width, seed=seed)
    start = (1, 1)
    goal = (2 * height - 1, 2 * width - 1)
    actions = bfs_solve(grid, start, goal)
    assert actions is not None, "Generated maze is unsolvable – this should never happen."
    return grid, start, goal, actions
