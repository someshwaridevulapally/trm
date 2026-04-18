"""
Training and evaluation loop for the 8-puzzle task.

Updated for TRM (arXiv:2510.04871v1):
  - Tile-tokenized encoder (9 tokens) — critical fix for spatial reasoning.
  - Deep supervision with RAMPED weights (later steps weighted more).
  - Extended curriculum: max scramble 40 moves (covers all 8-puzzle states).
  - T=5 macro steps, n=8 micro steps by default.
  - LR warmup (5 epochs linear) + cosine decay.
  - EMA (Exponential Moving Average) on model weights.
  - AdamW optimiser.
"""

import os
import csv
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Any

import numpy as np

from model.recursive_net import RecursiveNet
from tasks.puzzle.puzzle_dataset import PuzzleDataset, _state_to_onehot
from tasks.puzzle.puzzle_env import scramble_puzzle, solve_puzzle, GOAL_STATE, _state_to_tuple, _find_blank
from utils.deep_supervision import deep_supervision_loss, make_ramp_weights
from utils.ema import EMA


def _warmup_cosine_schedule(warmup_epochs: int, total_epochs: int):
    """LR lambda: linear warmup for `warmup_epochs`, then cosine decay."""
    def _schedule(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return _schedule


def train_puzzle(
    hidden_dim: int = 512,
    T: int = 5,
    n: int = 8,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_puzzles: int = 10_000,
    device: str = "cpu",
    seed: int = 42,
    ema_decay: float = 0.999,
    max_iters: int = None,   # legacy alias
    warmup_epochs: int = 5,
) -> Dict[str, Any]:
    """
    Full training + evaluation pipeline for the 8-puzzle task
    with tile-tokenized encoder, ramped deep supervision, and extended curriculum.
    """
    if max_iters is not None and T == 5:
        T = max_iters

    os.makedirs("checkpoints/puzzle", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    torch.manual_seed(seed)
    device = torch.device(device)

    # ── Model ─────────────────────────────────────────────────────────────
    # encoder_mode='tile_embed': 9-token encoder, one token per board cell.
    # This is the key fix — the Transformer can now attend across tile positions.
    model = RecursiveNet(
        in_channels=9,           # one-hot tile channels  (= num tile values)
        hidden_dim=hidden_dim,
        head_sizes={"puzzle": 9},
        T=T,
        n=n,
        encoder_mode="tile_embed",
        grid_h=3,
        grid_w=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine decay schedule
    scheduler = LambdaLR(
        optimiser,
        lr_lambda=_warmup_cosine_schedule(warmup_epochs, epochs),
    )

    criterion = nn.CrossEntropyLoss(reduction="none")
    ema = EMA(model, decay=ema_decay)

    # Ramped deep supervision weights: later macro steps get higher weight
    ds_weights = make_ramp_weights(T)
    print(f"Deep supervision weights (T={T}): {[f'{w:.3f}' for w in ds_weights]}")

    csv_path = "logs/puzzle_training.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "curriculum_depth", "lr"])

    best_val_acc = 0.0

    # ── Training loop with extended curriculum ─────────────────────────────
    for epoch in range(1, epochs + 1):
        # Extended curriculum: scramble depth 5 → 40 (covers all 8-puzzle configs)
        progress = (epoch - 1) / max(epochs - 1, 1)
        curr_min = 5
        curr_max = int(5 + 35 * progress)   # was 25, now 35 → reaches 40
        curr_max = max(curr_max, curr_min)

        print(f"\nEpoch {epoch}/{epochs}  curriculum: scramble {curr_min}–{curr_max}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        n_epoch = min(num_puzzles, 2000 + epoch * 400)
        dataset = PuzzleDataset(
            num_puzzles=n_epoch,
            min_moves=curr_min,
            max_moves=curr_max,
            seed=seed + epoch * 10000,
        )

        train_size = int(0.9 * len(dataset))
        val_size   = max(len(dataset) - train_size, 1)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc="  Training")
        for states, actions in pbar:
            states  = states.to(device)            # (B, 9, 3, 3)
            actions = actions.to(device).long()    # (B,)

            _, logits_list = model(states, task="puzzle", return_all=True)

            # Use ramped weights: final macro step gets the most gradient
            loss = deep_supervision_loss(
                logits_list=logits_list,
                targets=actions,
                criterion=criterion,
                weights=ds_weights,
            )

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            ema.update()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(n_batches, 1)

        # ── Validate (EMA weights) ─────────────────────────────────────────
        val_loss  = 0.0
        correct   = 0
        total_val = 0

        with ema.average_parameters():
            model.eval()
            with torch.no_grad():
                for states, actions in val_loader:
                    states  = states.to(device)
                    actions = actions.to(device).long()

                    logits, _ = model(states, task="puzzle")
                    loss_v    = criterion(logits, actions).mean()
                    val_loss += loss_v.item()

                    preds     = logits.argmax(dim=-1)
                    correct  += (preds == actions).sum().item()
                    total_val += actions.size(0)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc      = correct / max(total_val, 1)
        cur_lr       = scheduler.get_last_lr()[0]

        scheduler.step()
        print(f"  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  lr={cur_lr:.2e}")

        csv_writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
                             f"{val_acc:.6f}", curr_max, f"{cur_lr:.6f}"])
        csv_file.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/puzzle/best_model.pt")
            print(f"  [SAVED] best model (val_acc={val_acc:.4f})")

    csv_file.close()

    # ── End-to-end evaluation ─────────────────────────────────────────────
    with ema.average_parameters():
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
    """Autoregressively solve fresh puzzles and measure performance."""
    model.eval()
    solved        = 0
    total_steps   = 0
    total_optimal = 0
    goal_tuple    = _state_to_tuple(GOAL_STATE)

    for i in tqdm(range(num_puzzles), desc="Puzzle end-to-end eval"):
        random.seed(seed + i)
        depth = random.randint(10, 40)    # test up to 40-move scrambles
        state, optimal_actions = scramble_puzzle(num_moves=depth, seed=seed + i)
        total_optimal += len(optimal_actions)

        visited = set()
        for step in range(max_steps):
            st = _state_to_tuple(state)
            if st == goal_tuple:
                solved      += 1
                total_steps += step
                break
            if st in visited:
                total_steps += max_steps
                break
            visited.add(st)

            oh     = _state_to_onehot(state)
            tensor = torch.tensor(oh).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(tensor, task="puzzle")
            action = logits.argmax(dim=-1).item()

            br, bc = _find_blank(state)
            tr, tc = action // 3, action % 3
            if abs(br - tr) + abs(bc - tc) == 1:
                state[br, bc] = state[tr, tc]
                state[tr, tc] = 0
            else:
                total_steps += max_steps
                break
        else:
            total_steps += max_steps

    return {
        "solve_rate":  solved / max(num_puzzles, 1),
        "avg_steps":   total_steps / max(num_puzzles, 1),
        "avg_optimal": total_optimal / max(num_puzzles, 1),
    }
