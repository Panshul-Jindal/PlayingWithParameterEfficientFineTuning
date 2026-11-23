"""Plumbing (scaffold, already written): seeding, devices, parameter counts,
checkpoints, LR schedule, wandb setup.

Read it: it's short, and the viva may ask why checkpoints only store the
*trainable* parameters.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
# A full ViT-MAE checkpoint is ~450 MB. Stage 1 trains only ~8% of the
# weights; the rest are the frozen pretrained weights, which anyone can
# re-download. So we save only the trainable parameters (plus optimizer state
# for resuming), which keeps each per-epoch checkpoint to tens of MB.
# To load one, rebuild the same structure first (from_pretrained ->
# configure_explora with the same config), then load_state_dict(strict=False).


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    """Atomic save: write to a temp file, then rename. A Colab disconnect
    mid-write can't leave a corrupt checkpoint on Drive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_trainable_weights(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a trainable-only state dict and check it covers every trainable param."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Checkpoint has keys the model doesn't: {unexpected[:5]} ...")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    not_loaded = trainable - set(state)
    if not_loaded:
        raise RuntimeError(f"Trainable params missing from checkpoint: {sorted(not_loaded)[:5]} ...")


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup from 0 to the base LR, then cosine decay to min_lr_ratio * base LR.
    Call scheduler.step() once per optimizer step (not per epoch)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# wandb + provenance: every result must point back to a commit and a W&B run
# ---------------------------------------------------------------------------


def git_provenance() -> dict[str, Any]:
    """The commit this code runs from, and whether tracked files have uncommitted changes."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def init_wandb(project: str, job_type: str, config: dict, mode: str = "online", name: str | None = None,
               tags: list[str] | None = None, resume_id: str | None = None):
    """mode: 'online' (default), 'offline' (sync later with `wandb sync`), or 'disabled'.

    Also stores the git commit in the run's config, and tags the run "dirty" if tracked files
    had uncommitted changes, because such a run can't be reproduced from any commit."""
    import wandb

    prov = git_provenance()
    run = wandb.init(
        project=project,
        job_type=job_type,
        config=config,
        mode=mode,
        name=name,
        tags=(tags or []) + (["dirty"] if prov["git_dirty"] else []),
        id=resume_id,
        resume="allow" if resume_id else None,
    )
    run.config.update(prov, allow_val_change=True)
    print(f"W&B run {run.id} · commit {(prov['git_commit'] or 'none')[:7]}"
          + (" + UNCOMMITTED CHANGES" if prov["git_dirty"] else " (clean)"))
    if prov["git_dirty"]:
        print("  !! This run can't be reproduced from a commit. For a result you report: commit first, then re-run.")
    return run


def record_run(script: str, tag: str, run, metrics: dict[str, Any]) -> dict[str, Any]:
    """Append one row to results/runs.csv (when, what, which commit, which W&B run, headline numbers)
    and return the same provenance, to store next to the result."""
    import csv
    import json
    from datetime import datetime

    prov = {"git_commit": run.config.get("git_commit"), "git_dirty": run.config.get("git_dirty"),
            "wandb_run_id": run.id, "wandb_url": getattr(run, "url", None) or ""}
    path = Path("results/runs.csv")
    path.parent.mkdir(exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["finished_at", "script", "tag", "git_commit", "git_dirty", "wandb_run_id", "wandb_url", "metrics"])
        w.writerow([f"{datetime.now():%Y-%m-%d %H:%M}", script, tag, prov["git_commit"], prov["git_dirty"],
                    prov["wandb_run_id"], prov["wandb_url"], json.dumps(metrics)])
    print(f"Recorded in results/runs.csv: commit {(prov['git_commit'] or 'none')[:7]} · W&B {prov['wandb_url'] or prov['wandb_run_id']}")
    return prov
