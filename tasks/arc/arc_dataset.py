"""
Few-shot ARC dataset.

Each sample represents one ARC task and contains:
  - demo_inputs:  list of (10, H, W) one-hot tensors for demonstration inputs
  - demo_outputs: list of (10, H, W) one-hot tensors for demonstration outputs
  - test_input:   (10, H, W) one-hot tensor
  - test_output:  (H, W) integer tensor (class labels 0–9) for cross-entropy
  - mask:         (H, W) binary mask (1 = valid cell, 0 = padding)

All grids are padded to a common (max_h, max_w) within the batch.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple

import numpy as np

from tasks.arc.arc_loader import load_arc_dataset, grid_to_tensor_channels, pad_grid


# Default max grid dimensions (ARC grids are at most 30×30)
MAX_H = 30
MAX_W = 30


class ARCDataset(Dataset):
    """
    PyTorch Dataset for ARC-AGI tasks.

    Each item corresponds to one (task, test_pair) tuple.  If a task has
    multiple test pairs, each becomes a separate item.

    Args:
        data_dir (str):       Path to the ARC data root.
        split (str):          'training' or 'evaluation'.
        max_tasks (int|None): Limit number of tasks for debugging.
        max_h (int):          Pad height.
        max_w (int):          Pad width.
    """

    def __init__(
        self,
        data_dir: str = "data/arc",
        split: str = "training",
        max_tasks: int | None = None,
        max_h: int = MAX_H,
        max_w: int = MAX_W,
    ):
        super().__init__()
        self.max_h = max_h
        self.max_w = max_w
        self.items: List[Dict[str, Any]] = []

        tasks = load_arc_dataset(data_dir, split=split, max_tasks=max_tasks)

        for task in tasks:
            demo_inputs = task["train"]
            # Each test pair becomes a separate dataset item
            for test_pair in task["test"]:
                self.items.append({
                    "task_id": task["task_id"],
                    "demos": demo_inputs,
                    "test_input": test_pair["input"],
                    "test_output": test_pair["output"],
                })

        print(f"  ARCDataset: {len(self.items)} samples from {len(tasks)} tasks")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]

        # Encode demo pairs
        demo_input_tensors = []
        demo_output_tensors = []
        for demo in item["demos"]:
            di = grid_to_tensor_channels(demo["input"], self.max_h, self.max_w)
            do = grid_to_tensor_channels(demo["output"], self.max_h, self.max_w)
            demo_input_tensors.append(torch.tensor(di))   # (10, H, W)
            demo_output_tensors.append(torch.tensor(do))   # (10, H, W)

        # Stack demos: (num_demos, 10, H, W)
        demo_inputs = torch.stack(demo_input_tensors)
        demo_outputs = torch.stack(demo_output_tensors)

        # Encode test input
        test_in = torch.tensor(
            grid_to_tensor_channels(item["test_input"], self.max_h, self.max_w)
        )  # (10, H, W)

        # Target: padded integer grid for cross-entropy
        test_out_padded = pad_grid(item["test_output"], self.max_h, self.max_w)
        test_out = torch.tensor(test_out_padded, dtype=torch.long)  # (H, W)

        # Mask: 1 where valid, 0 where padding (-1 in the padded grid)
        mask = torch.tensor((test_out_padded >= 0).astype(np.float32))  # (H, W)

        # Clamp -1 to 0 so cross-entropy doesn't crash (masked cells ignored via mask)
        test_out = test_out.clamp(min=0)

        return {
            "demo_inputs": demo_inputs,    # (N_demo, 10, H, W)
            "demo_outputs": demo_outputs,  # (N_demo, 10, H, W)
            "test_input": test_in,         # (10, H, W)
            "test_output": test_out,       # (H, W)
            "mask": mask,                  # (H, W)
        }


def arc_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    Custom collate that handles variable-length demo lists.

    Pads demo counts to the maximum in the batch.
    """
    max_demos = max(b["demo_inputs"].size(0) for b in batch)
    B = len(batch)
    _, C, H, W = batch[0]["demo_inputs"].shape

    demo_inputs = torch.zeros(B, max_demos, C, H, W)
    demo_outputs = torch.zeros(B, max_demos, C, H, W)
    demo_mask = torch.zeros(B, max_demos)  # which demo slots are real

    test_inputs = []
    test_outputs = []
    masks = []

    for i, b in enumerate(batch):
        nd = b["demo_inputs"].size(0)
        demo_inputs[i, :nd] = b["demo_inputs"]
        demo_outputs[i, :nd] = b["demo_outputs"]
        demo_mask[i, :nd] = 1.0
        test_inputs.append(b["test_input"])
        test_outputs.append(b["test_output"])
        masks.append(b["mask"])

    return {
        "demo_inputs": demo_inputs,       # (B, max_demos, 10, H, W)
        "demo_outputs": demo_outputs,     # (B, max_demos, 10, H, W)
        "demo_mask": demo_mask,           # (B, max_demos)
        "test_input": torch.stack(test_inputs),   # (B, 10, H, W)
        "test_output": torch.stack(test_outputs), # (B, H, W)
        "mask": torch.stack(masks),               # (B, H, W)
    }
