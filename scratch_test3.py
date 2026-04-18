import time
import sys
import numpy as np
sys.path.append('.')
from tasks.puzzle.puzzle_env import solve_puzzle, GOAL_TUPLE

# The hardest 8-puzzle configuration requires 31 moves.
# One such state is:
hardest_state = np.array([
    [8, 6, 7],
    [2, 5, 4],
    [3, 0, 1]
])

print("Solving the hardest state...")
start = time.time()
sol = solve_puzzle(hardest_state)
print(f"Time: {time.time() - start:.2f}s, steps: {len(sol)}")
