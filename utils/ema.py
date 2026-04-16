"""
Exponential Moving Average (EMA) for model weights.

Used in TRM training as described in:
  "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871v1)

EMA smooths model weights across training steps to improve generalisation
and reduce the impact of noisy gradient updates.  At evaluation time, the
EMA weights are used instead of the live training weights.

EMA update rule (per parameter):
    θ_ema ← decay * θ_ema + (1 - decay) * θ_live
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Iterator


class EMA:
    """
    Maintains an exponential moving average copy of a model's parameters.

    Args:
        model (nn.Module): The model whose weights to track.
        decay (float):     EMA decay factor. Default 0.999 (from the TRM paper).
        device:            Device for the shadow copy. Defaults to model's device.

    Usage::

        ema = EMA(model, decay=0.999)

        for batch in dataloader:
            loss = criterion(model(batch), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()          # update shadow weights after optimizer step

        # Evaluate with EMA weights:
        with ema.average_parameters():
            val_loss = criterion(model(val_batch), val_targets)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device

        # Create a deep copy of model parameters as "shadow" weights
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow weights using the current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data.to(self.device)
                )

    def apply_shadow(self) -> None:
        """Replace model weights with EMA weights (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original (live) weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    class _EMAContext:
        """Context manager that temporarily applies EMA weights."""

        def __init__(self, ema: "EMA"):
            self.ema = ema

        def __enter__(self):
            self.ema.apply_shadow()
            return self.ema.model

        def __exit__(self, *args):
            self.ema.restore()

    def average_parameters(self) -> "_EMAContext":
        """
        Context manager: temporarily swap in EMA weights.

        Usage::
            with ema.average_parameters():
                outputs = model(inputs)
        """
        return self._EMAContext(self)

    def state_dict(self) -> dict:
        """Return shadow weights for checkpointing."""
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        """Restore EMA state from a checkpoint."""
        self.decay = state["decay"]
        self.shadow = {k: v.to(self.device) for k, v in state["shadow"].items()}
