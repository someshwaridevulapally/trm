"""
Training and evaluation loop for ARC-AGI.

Updated for TRM (arXiv:2510.04871v1):
  - Deep supervision over all T macro-step outputs.
  - EMA on model weights.
  - AdamW optimiser with cosine LR schedule.
  - MetaEncoder replaced by Transformer-based demo encoder.
  - ARCModel wraps MetaEncoder context into TRMCore's encoded input
    by adding the context token to x before passing to TRMCore.
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
from tasks.arc.arc_dataset import ARCDataset, arc_collate_fn, MAX_H, MAX_W
from tasks.arc.meta_encoder import MetaEncoder
from utils.deep_supervision import deep_supervision_loss
from utils.ema import EMA


class ARCModel(nn.Module):
    """
    Wrapper combining MetaEncoder + RecursiveNet for ARC tasks.

    The MetaEncoder encodes (input, output) demo pairs into a context vector.
    This context is added element-wise to the encoded test-input token before
    it enters TRMCore, so the task identity modulates all micro/macro updates.
    """

    def __init__(self, hidden_dim: int = 512, T: int = 3, n: int = 6):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.output_size = MAX_H * MAX_W * 10  # per-cell 10-class

        self.meta_encoder = MetaEncoder(
            grid_channels=10,
            context_dim=hidden_dim,
        )
        self.recursive_net = RecursiveNet(
            in_channels=10,
            hidden_dim=hidden_dim,
            head_sizes={"arc": self.output_size},
            T=T,
            n=n,
        )

    def _encode_with_context(
        self,
        test_input: torch.Tensor,
        context: torch.Tensor,
        return_all: bool = False,
    ):
        """Encode test_input, add context token, then run TRMCore + decode."""
        # Encode test grid → (B, 1, hidden_dim)
        enc = self.recursive_net.encoder(test_input)
        # Add context (B, hidden_dim) → unsqueeze → (B, 1, hidden_dim)
        enc = enc + context.unsqueeze(1)
        # Run TRM core
        z_H, z_H_list = self.recursive_net.trm_core(enc)
        # Decode
        logits = self.recursive_net.decoder(z_H, "arc")
        if return_all:
            logits_list = self.recursive_net.decoder.decode_all(z_H_list, "arc")
        else:
            logits_list = [logits]
        return logits, logits_list

    def forward(
        self,
        demo_inputs:  torch.Tensor,   # (B, N, 10, H, W)
        demo_outputs: torch.Tensor,   # (B, N, 10, H, W)
        demo_mask:    torch.Tensor,   # (B, N)
        test_input:   torch.Tensor,   # (B, 10, H, W)
        return_all:   bool = False,
    ):
        """
        Returns:
            logits:       (B, MAX_H, MAX_W, 10)
            logits_list:  List of T flat logit tensors for deep supervision.
        """
        context = self.meta_encoder(demo_inputs, demo_outputs, demo_mask)  # (B, hidden_dim)
        flat_logits, flat_logits_list = self._encode_with_context(test_input, context, return_all)

        B = test_input.size(0)
        logits = flat_logits.view(B, MAX_H, MAX_W, 10)
        logits_list = [l.view(B, MAX_H, MAX_W, 10) for l in flat_logits_list]
        return logits, logits_list


def train_arc(
    hidden_dim: int = 512,
    T: int = 3,
    n: int = 6,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    data_dir: str = "data/arc",
    device: str = "cpu",
    max_tasks: int | None = None,
    ema_decay: float = 0.999,
    max_iters: int = None,   # legacy alias
) -> Dict[str, Any]:
    """Full training + evaluation pipeline for ARC."""
    if max_iters is not None and T == 3:
        T = max_iters

    os.makedirs("checkpoints/arc", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = torch.device(device)

    # ── Dataset ───────────────────────────────────────────────────────────
    try:
        full_dataset = ARCDataset(data_dir=data_dir, split="training", max_tasks=max_tasks)
    except FileNotFoundError as e:
        print(f"\n⚠ ARC data not found: {e}")
        print("  git clone https://github.com/fchollet/ARC-AGI.git data/arc")
        return {"error": "ARC data not found"}

    if len(full_dataset) == 0:
        return {"error": "No tasks loaded"}

    train_size = int(0.8 * len(full_dataset))
    val_size   = max(len(full_dataset) - train_size, 1)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=arc_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=arc_collate_fn, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model     = ARCModel(hidden_dim=hidden_dim, T=T, n=n).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss(reduction="none")
    ema       = EMA(model, decay=ema_decay)

    csv_path = "logs/arc_training.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_cell_acc", "val_grid_match", "lr"])

    best_grid_match = 0.0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [ARC train]")
        for batch in pbar:
            demo_in  = batch["demo_inputs"].to(device)
            demo_out = batch["demo_outputs"].to(device)
            demo_m   = batch["demo_mask"].to(device)
            test_in  = batch["test_input"].to(device)
            test_out = batch["test_output"].to(device)   # (B, H, W) long
            mask     = batch["mask"].to(device)           # (B, H, W)

            _, logits_list = model(demo_in, demo_out, demo_m, test_in, return_all=True)

            loss = deep_supervision_loss(
                logits_list=logits_list,
                targets=test_out,
                criterion=criterion,
                mask=mask,
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
        grid_matches = 0
        grid_total   = 0

        with ema.average_parameters():
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    demo_in  = batch["demo_inputs"].to(device)
                    demo_out = batch["demo_outputs"].to(device)
                    demo_m   = batch["demo_mask"].to(device)
                    test_in  = batch["test_input"].to(device)
                    test_out = batch["test_output"].to(device)
                    mask     = batch["mask"].to(device)

                    logits, _ = model(demo_in, demo_out, demo_m, test_in)
                    B, H, W, C = logits.shape

                    loss_per_cell = criterion(
                        logits.reshape(-1, C), test_out.reshape(-1),
                    ).view(B, H, W)
                    masked_loss = (loss_per_cell * mask).sum() / mask.sum().clamp(min=1)
                    val_loss   += masked_loss.item()

                    preds = logits.argmax(dim=-1)
                    cell_correct += ((preds == test_out) * mask).sum().item()
                    cell_total   += mask.sum().item()

                    for b in range(B):
                        valid = mask[b].bool()
                        if (preds[b][valid] == test_out[b][valid]).all():
                            grid_matches += 1
                        grid_total += 1

        avg_val_loss    = val_loss / max(len(val_loader), 1)
        cell_acc        = cell_correct / max(cell_total, 1)
        grid_match_rate = grid_matches / max(grid_total, 1)
        cur_lr          = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch}: train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  "
              f"cell_acc={cell_acc:.4f}  grid_match={grid_match_rate:.4f}  lr={cur_lr:.2e}")

        csv_writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
                             f"{cell_acc:.6f}", f"{grid_match_rate:.6f}", f"{cur_lr:.6f}"])
        csv_file.flush()

        if grid_match_rate > best_grid_match:
            best_grid_match = grid_match_rate
            torch.save(model.state_dict(), "checkpoints/arc/best_model.pt")
            print(f"  [SAVED] best model (grid_match={grid_match_rate:.4f})")

    csv_file.close()
    return {
        "best_grid_match": best_grid_match,
        "final_cell_acc":  cell_acc,
    }
