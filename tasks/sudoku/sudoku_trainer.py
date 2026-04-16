"""
Training and evaluation loop for the Sudoku task.

Sudoku-Extreme is the hardest benchmark in the TRM paper (87.4% solve rate).
The model has to predict the complete 9×9 solution from a partial board.

Task framing:
  - Input:  (10, 9, 9) one-hot tensor — channels 0–9 per cell
  - Output: (9, 9) prediction of digit class (1–9) for every cell
  - Loss:   Masked cross-entropy (only on blank cells in the puzzle)
  - Metric: Exact board solve rate (all 81 cells correct)
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Any

from model.recursive_net import RecursiveNet
from tasks.sudoku.sudoku_dataset import SudokuDataset
from utils.deep_supervision import deep_supervision_loss
from utils.ema import EMA


_HEAD_SIZE = 9 * 9 * 10   # 9×9 cells, 10 classes (0=blank placeholder, 1–9=digit)


def train_sudoku(
    hidden_dim: int = 512,
    T: int = 3,
    n: int = 6,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    difficulty: str = "extreme",
    num_puzzles: int = 10_000,
    device: str = "cpu",
    seed: int = 42,
    ema_decay: float = 0.999,
    max_iters: int = None,   # legacy alias
) -> Dict[str, Any]:
    """
    Full training + evaluation pipeline for the Sudoku task.

    Returns a dict with 'best_solve_rate' and 'best_cell_acc'.
    """
    if max_iters is not None and T == 3:
        T = max_iters

    os.makedirs("checkpoints/sudoku", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    torch.manual_seed(seed)
    device = torch.device(device)

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"Generating {num_puzzles} {difficulty} Sudoku puzzles…")
    dataset = SudokuDataset(
        num_puzzles=num_puzzles,
        difficulty=difficulty,
        seed=seed,
    )

    train_size = int(0.9 * len(dataset))
    val_size   = max(len(dataset) - train_size, 1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = RecursiveNet(
        in_channels=10,           # one-hot digit channels (0=blank, 1–9)
        hidden_dim=hidden_dim,
        head_sizes={"sudoku": _HEAD_SIZE},
        T=T,
        n=n,
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss(reduction="none")
    ema       = EMA(model, decay=ema_decay)

    csv_path = "logs/sudoku_training.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_cell_acc", "val_solve_rate", "lr"])

    best_solve_rate = 0.0
    best_cell_acc   = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Sudoku train]")
        for puzzles, solutions in pbar:
            puzzles   = puzzles.to(device)          # (B, 10, 9, 9)
            solutions = solutions.to(device).long() # (B, 9, 9)

            # Deep-supervision: get logits from all T macro steps
            _, logits_list = model(puzzles, task="sudoku", return_all=True)

            # Reshape each flat logit (B, 9*9*10) → (B, 9, 9, 10) for deep_supervision_loss
            logits_list_grid = [l.view(puzzles.size(0), 9, 9, 10) for l in logits_list]

            loss = deep_supervision_loss(
                logits_list=logits_list_grid,
                targets=solutions,
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

        # ── Validation (EMA weights) ───────────────────────────────────────
        val_loss     = 0.0
        cell_correct = 0
        cell_total   = 0
        solved       = 0
        total_val    = 0

        with ema.average_parameters():
            model.eval()
            with torch.no_grad():
                for puzzles, solutions in val_loader:
                    puzzles   = puzzles.to(device)
                    solutions = solutions.to(device).long()
                    B         = puzzles.size(0)

                    logits, _ = model(puzzles, task="sudoku")
                    logits    = logits.view(B, 9, 9, 10)   # (B, 9, 9, 10)

                    # Per-cell cross-entropy
                    loss_per_cell = criterion(
                        logits.reshape(-1, 10), solutions.reshape(-1),
                    ).view(B, 9, 9)
                    val_loss += loss_per_cell.mean().item()

                    preds = logits.argmax(dim=-1)           # (B, 9, 9)
                    cell_correct += (preds == solutions).sum().item()
                    cell_total   += B * 81

                    # Exact board solve rate
                    for b in range(B):
                        if (preds[b] == solutions[b]).all():
                            solved += 1
                    total_val += B

        avg_val_loss = val_loss / max(len(val_loader), 1)
        cell_acc     = cell_correct / max(cell_total, 1)
        solve_rate   = solved / max(total_val, 1)
        cur_lr       = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch}: train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  "
              f"cell_acc={cell_acc:.4f}  solve_rate={solve_rate:.4f}  lr={cur_lr:.2e}")

        csv_writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
                             f"{cell_acc:.6f}", f"{solve_rate:.6f}", f"{cur_lr:.6f}"])
        csv_file.flush()

        if solve_rate > best_solve_rate:
            best_solve_rate = solve_rate
            best_cell_acc   = cell_acc
            torch.save(model.state_dict(), "checkpoints/sudoku/best_model.pt")
            print(f"  [SAVED] best model (solve_rate={solve_rate:.4f})")

    csv_file.close()
    return {
        "best_solve_rate": best_solve_rate,
        "best_cell_acc":   best_cell_acc,
    }
