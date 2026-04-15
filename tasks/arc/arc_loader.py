"""
ARC-AGI task loader.

Loads tasks from the ARC-AGI JSON format (as published at
https://github.com/fchollet/ARC-AGI).  Each JSON file contains a single
task with 'train' (demonstration) pairs and 'test' pairs.

Expected directory layout:
    data/arc/training/   ← JSON files like 00d62c1b.json
    data/arc/evaluation/ ← JSON files (optional)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np


def load_arc_task(json_path: str) -> Dict[str, Any]:
    """
    Load a single ARC task from a JSON file.

    Returns:
        {
            "task_id":     str,
            "train":       [{"input": np.ndarray, "output": np.ndarray}, ...],
            "test":        [{"input": np.ndarray, "output": np.ndarray}, ...],
        }
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    task_id = Path(json_path).stem

    def _parse_pairs(pairs: List[dict]) -> List[Dict[str, np.ndarray]]:
        result = []
        for p in pairs:
            inp = np.array(p["input"], dtype=np.int32)
            out = np.array(p["output"], dtype=np.int32)
            result.append({"input": inp, "output": out})
        return result

    return {
        "task_id": task_id,
        "train": _parse_pairs(raw["train"]),
        "test": _parse_pairs(raw["test"]),
    }


def load_arc_dataset(
    data_dir: str,
    split: str = "training",
    max_tasks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load all ARC tasks from a directory.

    Args:
        data_dir:   Root directory containing 'training/' and 'evaluation/' subdirs.
        split:      Which split to load ('training' or 'evaluation').
        max_tasks:  If set, load at most this many tasks (useful for debugging).

    Returns:
        List of task dicts (see load_arc_task).
    """
    split_dir = os.path.join(data_dir, split)

    if not os.path.isdir(split_dir):
        # Try the data_dir directly (maybe user pointed straight at the folder)
        if os.path.isdir(data_dir) and any(f.endswith(".json") for f in os.listdir(data_dir)):
            split_dir = data_dir
        else:
            raise FileNotFoundError(
                f"ARC data directory not found: {split_dir}\n"
                f"Download ARC-AGI from https://github.com/fchollet/ARC-AGI "
                f"and place JSON files in {split_dir}"
            )

    json_files = sorted([
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".json")
    ])

    if max_tasks is not None:
        json_files = json_files[:max_tasks]

    tasks = []
    for jf in json_files:
        try:
            tasks.append(load_arc_task(jf))
        except Exception as e:
            print(f"  Warning: skipping {jf}: {e}")

    print(f"Loaded {len(tasks)} ARC tasks from {split_dir}")
    return tasks


def pad_grid(grid: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    """
    Pad a grid to (max_h, max_w) with -1 (will be masked in loss).
    """
    h, w = grid.shape
    padded = np.full((max_h, max_w), -1, dtype=np.int32)
    padded[:h, :w] = grid
    return padded


def grid_to_tensor_channels(grid: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    """
    Convert a grid to a one-hot (10, max_h, max_w) float tensor.
    Cells with value -1 (padding) will be all-zeros.
    """
    padded = pad_grid(grid, max_h, max_w)
    one_hot = np.zeros((10, max_h, max_w), dtype=np.float32)
    for c in range(10):
        one_hot[c] = (padded == c).astype(np.float32)
    return one_hot
