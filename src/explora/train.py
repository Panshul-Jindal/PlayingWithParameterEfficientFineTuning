"""Phase 5b: the Stage 1 training loop (MAE loss, fp16 mixed precision).

ViTMAEForPreTraining computes the MAE loss itself: call model(pixel_values=x).loss.
It masks 75% of patches at random (config.mask_ratio), encodes the visible 25%,
decodes all positions, and returns the MSE on the MASKED patches only.

Mixed precision, the standard recipe (docs: https://pytorch.org/docs/stable/amp.html):
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        loss = model(...).loss                 # forward in fp16 where it's safe
    scaler.scale(loss).backward()              # scale up so small fp16 grads don't underflow to 0
    scaler.unscale_(optimizer)                 # back to true grads, needed before clipping
    clip_grad_norm_(params, max_grad_norm)
    scaler.step(optimizer)                     # skips the step if grads overflowed (inf/nan)
    scaler.update()                            # adjust the scale factor
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


def build_optimizer(
    model: nn.Module, lr: float, weight_decay: float = 0.05, betas: tuple[float, float] = (0.9, 0.95)
) -> torch.optim.AdamW:
    """AdamW over the TRAINABLE parameters only, in two param groups:
      * weight decay on matrices (p.ndim >= 2: Linear weights, lora_A, lora_B)
      * NO weight decay on vectors (p.ndim < 2: biases, LayerNorm weights and biases)
    (Decaying LayerNorm gains toward 0 shrinks activations. That's a common bug.)"""
    raise NotImplementedError("Phase 5: build_optimizer")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    use_amp: bool = True,
    max_grad_norm: float | None = 1.0,
    log_fn: Callable[[dict, int], None] | None = None,
    log_every: int = 20,
    max_steps: int | None = None,
) -> float:
    """One pass over `loader` (batches are (images, labels). Ignore the labels: this is
    self-supervised). Return the mean training loss over the steps taken.

    * model.train() first.
    * Per step: autocast forward -> scaled backward -> (unscale + clip if max_grad_norm) ->
      scaler.step -> scaler.update -> scheduler.step() if given. Zero the grads every step.
    * Every `log_every` steps (including step 0) call
      log_fn({"train/loss": float, "train/lr": float}, step) with step = index within this epoch.
    * Stop early after `max_steps` steps if given (for quick timing and debugging).
    """
    raise NotImplementedError("Phase 5: train_one_epoch")
