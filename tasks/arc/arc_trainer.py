"""
Training and evaluation loop for the ARC-AGI task.

The ARC pipeline is different from maze/puzzle because it requires:
  1. A MetaEncoder to produce task embeddings from demo pairs.
  2. The RecursiveNet receives the task embedding as context.
  3. Output is a per-cell 10-class classification (colours 0–9).
  4. Loss is cross-entropy, masked to ignore padded cells.
  5. Evaluation metric is exact grid match accuracy.
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Dict, Any

import numpy as np

from model.recursive_net import RecursiveNet
from tasks.arc.arc_dataset import ARCDataset, arc_collate_fn, MAX_H, MAX_W
from tasks.arc.meta_encoder import MetaEncoder


class ARCModel(nn.Module):
    """
    Wrapper combining MetaEncoder + RecursiveNet for ARC tasks.

    The MetaEncoder produces a context vector from demo pairs,
    which is fed into the RecursiveNet's RecCore at every iteration.
    """

    def __init__(self, hidden_dim: int = 128, max_iters: int = 10):
        super().__init__()
        self.context_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.output_size = MAX_H * MAX_W * 10  # per-cell 10-class

        self.meta_encoder = MetaEncoder(
            grid_channels=10,
            context_dim=self.context_dim,
        )

        self.recursive_net = RecursiveNet(
            in_channels=10,  # one-hot colour channels
            hidden_dim=hidden_dim,
            head_sizes={"arc": self.output_size},
            max_iters=max_iters,
            context_dim=self.context_dim,
        )

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        demo_mask: torch.Tensor,
        test_input: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """
        Args:
            demo_inputs:  (B, N_demos, 10, H, W)
            demo_outputs: (B, N_demos, 10, H, W)
            demo_mask:    (B, N_demos)
            test_input:   (B, 10, H, W)

        Returns:
            logits:     (B, H, W, 10)  per-cell class logits
            num_iters:  number of RecCore iterations
        """
        # Get task embedding from demos
        context = self.meta_encoder(demo_inputs, demo_outputs, demo_mask)  # (B, context_dim)

        # Run through recursive net with context
        flat_logits, num_iters = self.recursive_net(test_input, task="arc", context=context)
        # flat_logits: (B, H*W*10)

        B = test_input.size(0)
        logits = flat_logits.view(B, MAX_H, MAX_W, 10)  # (B, H, W, 10)
        return logits, num_iters


def train_arc(
    hidden_dim: int = 128,
    max_iters: int = 10,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    data_dir: str = "data/arc",
    device: str = "cpu",
    max_tasks: int | None = None,
) -> Dict[str, Any]:
    """
    Full training + evaluation pipeline for ARC.
    """
    os.makedirs("checkpoints/arc", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device(device)

    # ── Dataset ───────────────────────────────────────────────────────────
    try:
        full_dataset = ARCDataset(data_dir=data_dir, split="training", max_tasks=max_tasks)
    except FileNotFoundError as e:
        print(f"\n⚠ ARC data not found: {e}")
        print("To use ARC training, download the dataset:")
        print("  git clone https://github.com/fchollet/ARC-AGI.git data/arc")
        print("  (or provide --arc_data_dir pointing to your ARC data)")
        return {"error": "ARC data not found"}

    if len(full_dataset) == 0:
        print("⚠ No ARC tasks loaded. Check your data directory.")
        return {"error": "No tasks loaded"}

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(full_dataset) - 1
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=arc_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=arc_collate_fn, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = ARCModel(hidden_dim=hidden_dim, max_iters=max_iters).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    csv_path = "logs/arc_training.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_cell_acc", "val_grid_match"])

    best_grid_match = 0.0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [ARC train]")
        for batch in pbar:
            demo_in = batch["demo_inputs"].to(device)
            demo_out = batch["demo_outputs"].to(device)
            demo_m = batch["demo_mask"].to(device)
            test_in = batch["test_input"].to(device)
            test_out = batch["test_output"].to(device)   # (B, H, W) long
            mask = batch["mask"].to(device)               # (B, H, W)

            logits, _ = model(demo_in, demo_out, demo_m, test_in)  # (B, H, W, 10)

            # Reshape for cross-entropy: (B*H*W, 10) vs (B*H*W,)
            B, H, W, C = logits.shape
            loss_per_cell = criterion(
                logits.reshape(-1, C),
                test_out.reshape(-1),
            )  # (B*H*W,)
            loss_per_cell = loss_per_cell.view(B, H, W)

            # Mask out padding cells
            masked_loss = (loss_per_cell * mask).sum() / mask.sum().clamp(min=1)

            optimiser.zero_grad()
            masked_loss.backward()
            optimiser.step()

            total_loss += masked_loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{masked_loss.item():.4f}")

        avg_train_loss = total_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        cell_correct = 0
        cell_total = 0
        grid_matches = 0
        grid_total = 0

        with torch.no_grad():
            for batch in val_loader:
                demo_in = batch["demo_inputs"].to(device)
                demo_out = batch["demo_outputs"].to(device)
                demo_m = batch["demo_mask"].to(device)
                test_in = batch["test_input"].to(device)
                test_out = batch["test_output"].to(device)
                mask = batch["mask"].to(device)

                logits, _ = model(demo_in, demo_out, demo_m, test_in)
                B, H, W, C = logits.shape

                loss_per_cell = criterion(
                    logits.reshape(-1, C), test_out.reshape(-1),
                ).view(B, H, W)
                masked_loss = (loss_per_cell * mask).sum() / mask.sum().clamp(min=1)
                val_loss += masked_loss.item()

                # Per-cell accuracy (on valid cells only)
                preds = logits.argmax(dim=-1)  # (B, H, W)
                correct_cells = ((preds == test_out) * mask).sum().item()
                valid_cells = mask.sum().item()
                cell_correct += correct_cells
                cell_total += valid_cells

                # Exact grid match: all valid cells must be correct
                for b in range(B):
                    valid = mask[b].bool()
                    if (preds[b][valid] == test_out[b][valid]).all():
                        grid_matches += 1
                    grid_total += 1

        avg_val_loss = val_loss / max(len(val_loader), 1)
        cell_acc = cell_correct / max(cell_total, 1)
        grid_match_rate = grid_matches / max(grid_total, 1)

        print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  cell_acc={cell_acc:.4f}  "
              f"grid_match={grid_match_rate:.4f}")

        csv_writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
                             f"{cell_acc:.6f}", f"{grid_match_rate:.6f}"])
        csv_file.flush()

        if grid_match_rate > best_grid_match:
            best_grid_match = grid_match_rate
            torch.save(model.state_dict(), "checkpoints/arc/best_model.pt")
            print(f"  ✓ Saved best model (grid_match={grid_match_rate:.4f})")

    csv_file.close()

    return {
        "best_grid_match": best_grid_match,
        "final_cell_acc": cell_acc,
    }
