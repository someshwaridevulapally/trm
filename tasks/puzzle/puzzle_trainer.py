"""
Training and evaluation loop for the 8-puzzle task.

Features:
  - Curriculum learning: scramble depth increases linearly over epochs
    (starts at 5, grows to 30 by the final epoch).
  - Cross-entropy loss on per-step action predictions.
  - End-to-end evaluation: autoregressively solve fresh puzzles, report
    solve rate and average solution length vs. optimal.
"""

import os
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Dict, Any

import numpy as np

from model.recursive_net import RecursiveNet
from tasks.puzzle.puzzle_dataset import PuzzleDataset, _state_to_onehot
from tasks.puzzle.puzzle_env import scramble_puzzle, solve_puzzle, GOAL_STATE, _state_to_tuple, _find_blank


def train_puzzle(
    hidden_dim: int = 128,
    max_iters: int = 10,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_puzzles: int = 10_000,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full training + evaluation pipeline for the 8-puzzle task
    with curriculum learning.
    """
    os.makedirs("checkpoints/puzzle", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device(device)

    # ── Model ─────────────────────────────────────────────────────────────
    model = RecursiveNet(
        in_channels=9,  # one-hot tile channels
        hidden_dim=hidden_dim,
        head_sizes={"puzzle": 9},
        max_iters=max_iters,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    csv_path = "logs/puzzle_training.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "curriculum_depth"])

    best_val_acc = 0.0

    # ── Training loop with curriculum ─────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # Curriculum: linearly increase max scramble depth from 5 to 30
        progress = (epoch - 1) / max(epochs - 1, 1)
        curr_min = 5
        curr_max = int(5 + 25 * progress)
        curr_max = max(curr_max, curr_min)

        print(f"\nEpoch {epoch}/{epochs}  curriculum: scramble {curr_min}–{curr_max}")

        # Regenerate dataset with current curriculum difficulty
        # Use fewer puzzles per epoch for speed, scale with curriculum
        n_epoch = min(num_puzzles, 2000 + epoch * 400)
        dataset = PuzzleDataset(
            num_puzzles=n_epoch,
            min_moves=curr_min,
            max_moves=curr_max,
            seed=seed + epoch * 10000,
        )

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        if val_size == 0:
            val_size = 1
            train_size = len(dataset) - 1
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"  Training")
        for states, actions in pbar:
            states = states.to(device)           # (B, 9, 3, 3)
            actions = actions.to(device).long()  # (B,)

            logits, _ = model(states, task="puzzle")  # (B, 9)
            loss = criterion(logits, actions)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device)
                actions = actions.to(device).long()

                logits, _ = model(states, task="puzzle")
                loss = criterion(logits, actions)
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                correct += (preds == actions).sum().item()
                total += actions.size(0)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = correct / max(total, 1)

        print(f"  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

        csv_writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
                             f"{val_acc:.6f}", curr_max])
        csv_file.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/puzzle/best_model.pt")
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    csv_file.close()

    # ── End-to-end evaluation ─────────────────────────────────────────────
    e2e = evaluate_puzzle_end_to_end(model, device=device, num_puzzles=200, seed=77777)
    print(f"\n  End-to-end eval: solved={e2e['solve_rate']:.2%}  "
          f"avg_steps={e2e['avg_steps']:.1f}  avg_optimal={e2e['avg_optimal']:.1f}")

    return {"best_val_acc": best_val_acc, **e2e}


def evaluate_puzzle_end_to_end(
    model: RecursiveNet,
    device: torch.device,
    num_puzzles: int = 200,
    seed: int = 77777,
    max_steps: int = 100,
) -> Dict[str, float]:
    """
    Autoregressively solve fresh puzzles and measure performance.

    Returns dict with 'solve_rate', 'avg_steps', 'avg_optimal'.
    """
    model.eval()
    solved = 0
    total_steps = 0
    total_optimal = 0
    goal_tuple = _state_to_tuple(GOAL_STATE)

    for i in tqdm(range(num_puzzles), desc="Puzzle end-to-end eval"):
        random.seed(seed + i)
        depth = random.randint(10, 30)
        state, optimal_actions = scramble_puzzle(num_moves=depth, seed=seed + i)
        optimal_len = len(optimal_actions)
        total_optimal += optimal_len

        visited = set()
        for step in range(max_steps):
            st = _state_to_tuple(state)
            if st == goal_tuple:
                solved += 1
                total_steps += step
                break
            if st in visited:
                # Stuck in a loop
                total_steps += max_steps
                break
            visited.add(st)

            # Build input
            oh = _state_to_onehot(state)
            tensor = torch.tensor(oh).unsqueeze(0).to(device)  # (1, 9, 3, 3)

            with torch.no_grad():
                logits, _ = model(tensor, task="puzzle")
            action = logits.argmax(dim=-1).item()

            # Apply action: swap blank with cell at flat index `action`
            br, bc = _find_blank(state)
            tr, tc = action // 3, action % 3
            # Validate that the swap is with an adjacent cell
            if abs(br - tr) + abs(bc - tc) == 1:
                state[br, bc] = state[tr, tc]
                state[tr, tc] = 0
            else:
                # Invalid action → count as failure
                total_steps += max_steps
                break
        else:
            total_steps += max_steps

    return {
        "solve_rate": solved / max(num_puzzles, 1),
        "avg_steps": total_steps / max(num_puzzles, 1),
        "avg_optimal": total_optimal / max(num_puzzles, 1),
    }
