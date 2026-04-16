"""Quick verification: Sudoku solver + 1-epoch maze mini-train."""
import sys

print("=== Sudoku solver verification ===")
from tasks.sudoku.sudoku_env import generate_sudoku, solve_sudoku
import numpy as np

all_ok = True
for seed in range(5):
    puzzle, solution = generate_sudoku(num_clues=25, seed=seed)
    solved = solve_sudoku(puzzle)
    if solved is None:
        print(f"  seed={seed}: SOLVER RETURNED NONE")
        all_ok = False
        continue
    rows_ok  = all(set(solved[r].tolist()) == set(range(1, 10)) for r in range(9))
    cols_ok  = all(set(solved[:, c].tolist()) == set(range(1, 10)) for c in range(9))
    boxes_ok = all(
        set(solved[r*3:(r+1)*3, c*3:(c+1)*3].flatten().tolist()) == set(range(1, 10))
        for r in range(3) for c in range(3)
    )
    clues_ok = all(
        puzzle[r, c] == 0 or puzzle[r, c] == solved[r, c]
        for r in range(9) for c in range(9)
    )
    ok = rows_ok and cols_ok and boxes_ok and clues_ok
    status = "OK" if ok else "FAIL"
    print(f"  seed={seed}: rows={rows_ok} cols={cols_ok} boxes={boxes_ok} clues={clues_ok} -> {status}")
    all_ok = all_ok and ok

if all_ok:
    print("Solver: ALL CORRECT")
else:
    print("Solver: HAS BUGS")
    sys.exit(1)

print()
print("=== 1-epoch maze mini-train ===")
from tasks.maze.maze_trainer import train_maze
results = train_maze(
    hidden_dim=64,   # small for speed
    T=2, n=2,
    epochs=1,
    batch_size=32,
    lr=1e-3,
    num_mazes=200,
    device="cpu",
    seed=0,
)
print("Results:", results)
print()
print("=== 1-epoch puzzle mini-train ===")
from tasks.puzzle.puzzle_trainer import train_puzzle
results = train_puzzle(
    hidden_dim=64,
    T=2, n=2,
    epochs=1,
    batch_size=32,
    lr=1e-3,
    num_puzzles=200,
    device="cpu",
    seed=0,
)
print("Results:", results)
print()
print("ALL VERIFICATION PASSED")
