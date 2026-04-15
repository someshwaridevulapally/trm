"""
Training and evaluation loop for the maze-solving task.

Training:
  - Cross-entropy loss on per-step action predictions.
  - Adam optimiser with configurable LR.

Evaluation:
  - End-to-end maze solving: feed the initial grid, take the predicted action,
    update position, repeat until the goal is reached or max steps exceeded.
  - Metrics: % mazes solved, average steps taken.
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Dict, Any

from model.recursive_net import RecursiveNet
from tasks.maze.maze_dataset import MazeDataset
from tasks.maze.maze_env import generate_solvable_maze, ACTION_DELTAS
import numpy as np


def train_maze(
    hidden_dim: int = 128,
    max_iters: int = 10,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_mazes: int = 10_000,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full training + evaluation pipeline for the maze task.

    Returns:
        Dictionary with final metrics.
    """
    # ── Setup ─────────────────────────────────────────────────────────────
    os.makedirs("checkpoints/maze", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device(device)

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = MazeDataset(num_mazes=num_mazes, seed=seed)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = RecursiveNet(
        in_channels=2,  # maze walls + agent position
        hidden_dim=hidden_dim,
        head_sizes={"maze": 4},
        max_iters=max_iters,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ── CSV logger ────────────────────────────────────────────────────────
    csv_path = "logs/maze_training.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

    best_val_acc = 0.0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for grids, actions in pbar:
            grids = grids.to(device)           # (B, 2, H, W)
            actions = actions.to(device).long() # (B,)

            logits, _ = model(grids, task="maze")  # (B, 4)
            loss = criterion(logits, actions)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for grids, actions in val_loader:
                grids = grids.to(device)
                actions = actions.to(device).long()

                logits, _ = model(grids, task="maze")
                loss = criterion(logits, actions)
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                correct += (preds == actions).sum().item()
                total += actions.size(0)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = correct / max(total, 1)

        print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.4f}")

        csv_writer.writerow([epoch, f"{avg_train_loss:.6f}",
                             f"{avg_val_loss:.6f}", f"{val_acc:.6f}"])
        csv_file.flush()

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/maze/best_model.pt")
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    csv_file.close()

    # ── End-to-end evaluation ─────────────────────────────────────────────
    e2e_metrics = evaluate_maze_end_to_end(model, device=device, num_mazes=200, seed=99999)
    print(f"\n  End-to-end eval: solved={e2e_metrics['solve_rate']:.2%}  "
          f"avg_steps={e2e_metrics['avg_steps']:.1f}")

    return {
        "best_val_acc": best_val_acc,
        **e2e_metrics,
    }


def evaluate_maze_end_to_end(
    model: RecursiveNet,
    device: torch.device,
    num_mazes: int = 200,
    seed: int = 99999,
    max_steps: int = 200,
) -> Dict[str, float]:
    """
    Evaluate the model by rolling out actions autoregressively on fresh mazes.

    Returns:
        Dictionary with 'solve_rate' and 'avg_steps'.
    """
    model.eval()
    solved = 0
    total_steps = 0

    for i in tqdm(range(num_mazes), desc="End-to-end eval"):
        grid, start, goal, _ = generate_solvable_maze(height=10, width=10, seed=seed + i)
        r, c = start
        H, W = grid.shape

        for step in range(max_steps):
            if (r, c) == goal:
                solved += 1
                total_steps += step
                break

            # Build input tensor
            ch_maze = grid.astype(np.float32)
            ch_pos = np.zeros((H, W), dtype=np.float32)
            ch_pos[r, c] = 1.0
            grid_tensor = torch.tensor(
                np.stack([ch_maze, ch_pos], axis=0),
            ).unsqueeze(0).to(device)  # (1, 2, H, W)

            with torch.no_grad():
                logits, _ = model(grid_tensor, task="maze")
            action = logits.argmax(dim=-1).item()

            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            # Only move if new position is open
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                r, c = nr, nc
        else:
            total_steps += max_steps  # didn't solve

    return {
        "solve_rate": solved / num_mazes,
        "avg_steps": total_steps / max(num_mazes, 1),
    }
