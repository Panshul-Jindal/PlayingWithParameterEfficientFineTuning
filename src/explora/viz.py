"""Reconstruction figures (scaffold, already written).

The assignment asks for reconstructions of a few masked test images before and
after Stage 1. For a fair before/after comparison, the SAME images must get the
SAME mask. ViTMAEForPreTraining.forward takes a `noise` tensor
(B, num_patches): patches with the smallest noise are kept. So a fixed noise
tensor gives a fixed mask.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from explora.data import IMAGENET_MEAN, IMAGENET_STD


def fixed_noise(n_images: int, num_patches: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_images, num_patches, generator=g)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


@torch.no_grad()
def reconstruct(model: nn.Module, images: torch.Tensor, noise: torch.Tensor) -> dict[str, torch.Tensor]:
    """Returns images in [0, 1], each (B, 3, H, W):
    original, masked (masked patches greyed out), recon (the raw decoder output),
    pasted (visible patches from the original, masked ones from the decoder)."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    images, noise = images.to(device), noise.to(device)
    out = model(pixel_values=images, noise=noise)
    if model.config.norm_pix_loss:
        raise NotImplementedError("norm_pix_loss=True needs per-patch de-normalisation; vit-mae-base uses False")
    pred = model.unpatchify(out.logits)  # (B, 3, H, W) in ImageNet-normalised space
    p = model.config.patch_size
    mask = out.mask.unsqueeze(-1).repeat(1, 1, p * p * 3)  # (B, N, p*p*3); 1 = masked
    mask = model.unpatchify(mask)  # (B, 3, H, W)
    model.train(was_training)

    original = _denorm(images)
    recon = _denorm(pred)
    return {
        "original": original.cpu(),
        "masked": (original * (1 - mask) + 0.5 * mask).cpu(),
        "recon": recon.cpu(),
        "pasted": (original * (1 - mask) + recon * mask).cpu(),
    }


def recon_figure(model: nn.Module, images: torch.Tensor, noise: torch.Tensor, title: str,
                 row_labels: list[str] | None = None, loss: float | None = None):
    r = reconstruct(model, images, noise)
    cols = ["original", "masked", "recon", "pasted"]
    n = images.shape[0]
    fig, axes = plt.subplots(n, len(cols), figsize=(2.2 * len(cols), 2.2 * n))
    axes = axes.reshape(n, len(cols))
    for i in range(n):
        for j, c in enumerate(cols):
            axes[i, j].imshow(r[c][i].permute(1, 2, 0).numpy())
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(c, fontsize=10)
        if row_labels:
            axes[i, 0].set_ylabel(row_labels[i][:14], fontsize=8)
    fig.suptitle(title if loss is None else f"{title}  (masked-patch MSE on these: {loss:.4f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


@torch.no_grad()
def masked_loss(model: nn.Module, images: torch.Tensor, noise: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    loss = model(pixel_values=images.to(device), noise=noise.to(device)).loss.item()
    model.train(was_training)
    return loss


def masked_loss_batched(model: nn.Module, images: torch.Tensor, noise: torch.Tensor, batch_size: int = 64) -> float:
    """masked_loss over many images, in chunks. Every image masks the same number of patches,
    so weighting each chunk by its size gives the exact mean over all images."""
    total, n = 0.0, 0
    for i in range(0, len(images), batch_size):
        k = len(images[i:i + batch_size])
        total += masked_loss(model, images[i:i + batch_size], noise[i:i + batch_size]) * k
        n += k
    return total / n


def loss_curve_figure(steps: list[int], losses: list[float], epoch_losses: list[float], steps_per_epoch: int,
                      val_losses: list[float] | None = None):
    """val_losses[0] is measured before training (step 0), then one value after each epoch."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, alpha=0.35, label="train loss (logged steps)")
    ax.plot([(i + 1) * steps_per_epoch for i in range(len(epoch_losses))], epoch_losses, "o-", label="train, epoch mean")
    if val_losses:
        ax.plot([i * steps_per_epoch for i in range(len(val_losses))], val_losses, "s--",
                label="val (256 fixed images, fixed mask)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("MAE loss (MSE on masked patches)")
    ax.set_title("ExPLoRA Stage 1 reconstruction loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
