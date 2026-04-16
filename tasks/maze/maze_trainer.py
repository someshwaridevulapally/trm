"""
Training and evaluation loop for the maze-solving task.

Updated for TRM (arXiv:2510.04871v1):
  - Deep supervision: loss averaged over all T macro-step outputs.
  - EMA (Exponential Moving Average) on model weights.
  - AdamW optimiser with cosine LR schedule.
  - New API: model returns (logits, logits_list); train with return_all=True.
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Any

import numpy as np

from model.recursive_net import RecursiveNet
from tasks.maze.maze_dataset import MazeDataset
from tasks.maze.maze_env import generate_solvable_maze, ACTION_DELTAS
from utils.deep_supervision import deep_supervision_loss
from utils.ema import EMA


def train_maze(
    hidden_dim: int = 512,
    T: int = 3,
    n: int = 6,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_mazes: int = 10_000,
    device: str = "cpu",
    seed: int = 42,
    ema_decay: float = 0.999,
    # Legacy alias kept for backwards-compat with old train.py callers
    max_iters: int = None,
) -> Dict[str, Any]:
    """
    Full training + evaluation pipeline for the maze task with TRM.

    Returns:
        Dictionary with final metrics.
    """
    if max_iters is not None and T == 3:
        # honour old --max_iters flag by mapping it to T
        T = max_iters

    os.makedirs("checkpoints/maze", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    torch.manual_seed(seed)
    device = torch.device(device)

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = MazeDataset(num_mazes=num_mazes, seed=seed)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = RecursiveNet(
        in_channels=2,          # maze walls + agent position
        hidden_dim=hidden_dim,
        head_sizes={"maze": 4},
        T=T,
        n=n,
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss(reduction="none")
    ema = EMA(model, decay=ema_decay)

    csv_path = "logs/maze_training.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "lr"])

    best_val_acc = 0.0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Maze train]")
        for grids, actions in pbar:
            grids   = grids.to(device)            # (B, 2, H, W)
            actions = actions.to(device).long()   # (B,)

            # Deep supervision: returns logits from all T macro steps
            _, logits_list = model(grids, task="maze", return_all=True)

            loss = deep_supervision_loss(
                logits_list=logits_list,
                targets=actions,
                criterion=criterion,
            )

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            ema.update()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)

        # ── Validation (with EMA weights) ─────────────────────────────────
        val_loss  = 0.0
        correct   = 0
        total_val = 0

        with ema.average_parameters():
            model.eval()
            with torch.no_grad():
                for grids, actions in val_loader:
                    grids   = grids.to(device)
                    actions = actions.to(device).long()

                    logits, _ = model(grids, task="maze")
                    loss_val = criterion(logits, actions).mean()
                    val_loss += loss_val.item()

                    preds    = logits.argmax(dim=-1)
                    correct  += (preds == actions).sum().item()
                    total_val += actions.size(0)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc      = correct / max(total_val, 1)
        cur_lr       = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.4f}  lr={cur_lr:.2e}")

        csv_writer.writerow([epoch, f"{avg_train_loss:.6f}",
                             f"{avg_val_loss:.6f}", f"{val_acc:.6f}", f"{cur_lr:.6f}"])
        csv_file.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/maze/best_model.pt")
            print(f"  [SAVED] best model (val_acc={val_acc:.4f})")

    csv_file.close()

    # ── End-to-end evaluation ─────────────────────────────────────────────
    with ema.average_parameters():
        e2e = evaluate_maze_end_to_end(model, device=device, num_mazes=200, seed=99999)

    print(f"\n  End-to-end eval: solved={e2e['solve_rate']:.2%}  "
          f"avg_steps={e2e['avg_steps']:.1f}")

    return {"best_val_acc": best_val_acc, **e2e}


def evaluate_maze_end_to_end(
    model: RecursiveNet,
    device: torch.device,
    num_mazes: int = 200,
    seed: int = 99999,
    max_steps: int = 200,
) -> Dict[str, float]:
    """Autoregressively roll out model actions on fresh mazes."""
    model.eval()
    solved      = 0
    total_steps = 0

    for i in tqdm(range(num_mazes), desc="End-to-end eval"):
        grid, start, goal, _ = generate_solvable_maze(height=10, width=10, seed=seed + i)
        r, c = start
        H, W = grid.shape

        for step in range(max_steps):
            if (r, c) == goal:
                solved      += 1
                total_steps += step
                break

            ch_maze = grid.astype(np.float32)
            ch_pos  = np.zeros((H, W), dtype=np.float32)
            ch_pos[r, c] = 1.0
            tensor = torch.tensor(np.stack([ch_maze, ch_pos])).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(tensor, task="maze")
            action = logits.argmax(dim=-1).item()

            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                r, c = nr, nc
        else:
            total_steps += max_steps

    return {
        "solve_rate": solved / num_mazes,
        "avg_steps":  total_steps / max(num_mazes, 1),
    }
