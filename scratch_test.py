import time
import sys
sys.path.append('.')
from tasks.puzzle.puzzle_dataset import PuzzleDataset

start = time.time()
dataset = PuzzleDataset(num_puzzles=10, min_moves=15, max_moves=15)
print(f"Time for 10 puzzles at depth 15: {time.time() - start:.2f}s")
