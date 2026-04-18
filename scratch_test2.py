import time
import sys
sys.path.append('.')
from tasks.puzzle.puzzle_dataset import PuzzleDataset

for depth in [20, 25, 30, 35, 40]:
    start = time.time()
    # just 1 puzzle to see if it even finishes
    try:
        dataset = PuzzleDataset(num_puzzles=1, min_moves=depth, max_moves=depth)
        print(f"Time for 1 puzzle at depth {depth}: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Exception at depth {depth}: {e}")
