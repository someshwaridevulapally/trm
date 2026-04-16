"""
Deep Supervision Loss Utility.

Computes the loss across ALL intermediate macro-step predictions
(z_H decoded at every macro step T), as described in:
  "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871v1)

The loss is averaged uniformly across all macro steps.  This provides
a learning signal at every level of the macro recursion and prevents
vanishing gradients through deep recursion.
"""

import torch
import torch.nn as nn
from typing import List, Optional


def deep_supervision_loss(
    logits_list: List[torch.Tensor],
    targets: torch.Tensor,
    criterion: nn.Module,
    mask: Optional[torch.Tensor] = None,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Compute the deep supervision loss across all macro-step predictions.

    Each element of `logits_list` is the decoder output at one macro step.
    The loss is (optionally weighted) average of per-step losses.

    Args:
        logits_list: List of T tensors, each of shape (B, num_classes) or
                     (B, H, W, C) for grid tasks.  One per macro step.
        targets:     Ground-truth labels, shape (B,) or (B, H, W).
        criterion:   Loss function (e.g. nn.CrossEntropyLoss(reduction='none')).
        mask:        Optional boolean/float mask of shape (B,) or (B, H, W)
                     indicating which positions/cells to count in the loss.
                     Useful for ARC padded grids.
        weights:     Optional per-step loss weights (length == len(logits_list)).
                     If None, all steps are weighted equally.

    Returns:
        Scalar loss tensor (with gradient).
    """
    T = len(logits_list)
    if weights is None:
        weights = [1.0 / T] * T
    else:
        assert len(weights) == T, "weights must have same length as logits_list"
        total = sum(weights)
        weights = [w / total for w in weights]

    total_loss = torch.tensor(0.0, device=logits_list[0].device)

    for logits, w in zip(logits_list, weights):
        step_loss = _compute_step_loss(logits, targets, criterion, mask)
        total_loss = total_loss + w * step_loss

    return total_loss


def _compute_step_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Compute masked cross-entropy loss for a single macro step.

    Handles both:
      - 1-D classification: logits (B, C), targets (B,)
      - Grid prediction:    logits (B, H, W, C), targets (B, H, W)
    """
    if logits.dim() == 2:
        # Classification: (B, C) vs (B,)
        raw_loss = criterion(logits, targets)  # (B,) when reduction='none'
        if mask is not None:
            raw_loss = raw_loss * mask
            return raw_loss.sum() / mask.sum().clamp(min=1)
        return raw_loss.mean()

    elif logits.dim() == 4:
        # Grid: (B, H, W, C) → reshape to (B*H*W, C) vs (B*H*W,)
        B, H, W, C = logits.shape
        raw_loss = criterion(
            logits.reshape(-1, C),
            targets.reshape(-1),
        )  # (B*H*W,)
        raw_loss = raw_loss.view(B, H, W)

        if mask is not None:
            raw_loss = raw_loss * mask
            return raw_loss.sum() / mask.sum().clamp(min=1)
        return raw_loss.mean()

    else:
        raise ValueError(
            f"Unsupported logits shape: {logits.shape}. "
            "Expected (B, C) or (B, H, W, C)."
        )
