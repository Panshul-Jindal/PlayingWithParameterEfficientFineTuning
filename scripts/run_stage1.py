"""Task 2: ExPLoRA Stage 1 (scaffold, already written; it calls YOUR functions).

Time one epoch first:
    uv run python scripts/run_stage1.py --time-only
Full run (resumable, one checkpoint per epoch):
    uv run python scripts/run_stage1.py --epochs 10 --out-dir checkpoints/stage1
On Colab, point --out-dir at Google Drive and add --resume to continue after a disconnect.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import ViTMAEForPreTraining

from explora.data import CLASS_NAMES, build_dataset, build_loader
from explora.explora import configure_explora
from explora.train import build_optimizer, train_one_epoch
from explora.utils import (cosine_with_warmup, get_device, init_wandb, load_trainable_weights, save_checkpoint,
                           seed_everything, trainable_state_dict)
from explora.viz import fixed_noise, loss_curve_figure, masked_loss, recon_figure

MODEL_ID = "facebook/vit-mae-base"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--no-augment", action="store_true", help="turn off random-resized-crop + flips")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="checkpoints/stage1")
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt if it exists")
    p.add_argument("--time-only", action="store_true", help="run ONE epoch, report its time, and exit")
    p.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch (debugging)")
    p.add_argument("--wandb-project", type=str, default="explora-bloodmnist")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def recon_samples(img_size: int) -> tuple[torch.Tensor, list[str]]:
    """The first test image of each class: 8 fixed images, one per class."""
    ds = build_dataset("test", img_size=img_size, augment=False)
    picked: dict[int, torch.Tensor] = {}
    for i in range(len(ds)):
        x, y = ds[i]
        if y not in picked:
            picked[y] = x
        if len(picked) == len(CLASS_NAMES):
            break
    ys = sorted(picked)
    return torch.stack([picked[y] for y in ys]), [CLASS_NAMES[y] for y in ys]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    use_amp = (not args.no_amp) and device.type == "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("outputs")
    fig_dir.mkdir(exist_ok=True)

    resume_ckpt = None
    if args.resume and (out_dir / "last.pt").exists():
        resume_ckpt = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        print(f"Resuming from {out_dir / 'last.pt'} (finished epoch {resume_ckpt['epoch'] + 1})")

    explora_cfg = dict(r=args.r, alpha=args.alpha)
    run = init_wandb(
        args.wandb_project, "stage1", config=vars(args) | {"explora": explora_cfg}, mode=args.wandb_mode,
        name=args.run_name or ("time-epoch" if args.time_only else f"stage1-r{args.r}"),
        tags=["stage1", "time-only" if args.time_only else "full"],
        resume_id=resume_ckpt["wandb_id"] if resume_ckpt else None,
    )
    import wandb

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("epoch/*", step_metric="epoch")

    # ---- model + "before" reconstructions (pure pretrained weights) -----------------------
    model = ViTMAEForPreTraining.from_pretrained(MODEL_ID).to(device)
    images, labels = recon_samples(args.img_size)
    noise = fixed_noise(len(images), (args.img_size // model.config.patch_size) ** 2, seed=args.seed)
    if resume_ckpt is None:
        before_loss = masked_loss(model, images, noise)
        fig = recon_figure(model, images, noise, "Before Stage 1 (pretrained ViT-MAE)", labels, before_loss)
        fig.savefig(fig_dir / "recon_before.png", dpi=120)
        wandb.log({"recon/before": wandb.Image(fig), "recon/before_loss": before_loss})

    # ---- ExPLoRA: choose what trains (YOUR configure_explora) --------------------------------
    summary = configure_explora(model, **explora_cfg)
    model.to(device)
    print(f"Trainable: {summary['trainable']:,} / {summary['total']:,} ({summary['trainable_pct']:.2f}%)")
    print(f"LoRA modules: {len(summary['lora_modules'])} | unfrozen blocks: {summary['unfrozen_blocks']}")
    run.summary.update({"params/trainable": summary["trainable"], "params/total": summary["total"],
                        "params/trainable_pct": summary["trainable_pct"]})

    loader = build_loader("train", batch_size=args.batch_size, img_size=args.img_size, for_training=True,
                          augment=not args.no_augment, num_workers=args.num_workers)
    steps_per_epoch = min(len(loader), args.max_steps or len(loader))
    epochs = 1 if args.time_only else args.epochs
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_with_warmup(optimizer, int(args.warmup_epochs * steps_per_epoch), epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, history = 0, {"steps": [], "losses": [], "epoch_losses": [], "epoch_times": []}
    if resume_ckpt is not None:
        load_trainable_weights(model, resume_ckpt["trainable"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        scaler.load_state_dict(resume_ckpt["scaler"])
        start_epoch, history = resume_ckpt["epoch"] + 1, resume_ckpt["history"]

    # ---- training ---------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        def log_fn(metrics: dict, step: int, _epoch=epoch) -> None:
            g = _epoch * steps_per_epoch + step
            history["steps"].append(g)
            history["losses"].append(metrics["train/loss"])
            wandb.log(metrics | {"global_step": g})

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        mean_loss = train_one_epoch(model, loader, optimizer, scaler, device, scheduler=scheduler, use_amp=use_amp,
                                    log_fn=log_fn, max_steps=args.max_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        history["epoch_losses"].append(mean_loss)
        history["epoch_times"].append(dt)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        val_loss = masked_loss(model, images, noise)
        print(f"epoch {epoch + 1}/{epochs}  loss {mean_loss:.4f}  fixed-test loss {val_loss:.4f}  "
              f"{dt:.0f}s  peak {peak_gb:.2f} GB")
        wandb.log({"epoch": epoch + 1, "epoch/loss": mean_loss, "epoch/time_s": dt, "epoch/peak_mem_gb": peak_gb,
                   "epoch/fixed_test_loss": val_loss,
                   "epoch/recon": wandb.Image(recon_figure(model, images, noise, f"Epoch {epoch + 1}", labels, val_loss))})

        if args.time_only:
            print(f"\nOne epoch = {dt:.0f}s ({steps_per_epoch} steps of {args.batch_size}). "
                  f"Projected: 10 epochs = {10 * dt / 60:.0f} min, 20 epochs = {20 * dt / 60:.0f} min.")
            run.summary.update({"time/epoch_s": dt})
            wandb.finish()
            return

        save_checkpoint(out_dir / "last.pt", trainable=trainable_state_dict(model), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                        history=history, explora_cfg=explora_cfg, args=vars(args), wandb_id=run.id)

    # ---- after Stage 1 -----------------------------------------------------------------------
    after_loss = masked_loss(model, images, noise)
    fig = recon_figure(model, images, noise, "After Stage 1 (ExPLoRA)", labels, after_loss)
    fig.savefig(fig_dir / "recon_after.png", dpi=120)
    curve = loss_curve_figure(history["steps"], history["losses"], history["epoch_losses"], steps_per_epoch)
    curve.savefig(fig_dir / "stage1_loss.png", dpi=120)
    wandb.log({"recon/after": wandb.Image(fig), "recon/after_loss": after_loss, "loss_curve": wandb.Image(curve)})
    save_checkpoint(out_dir / "final.pt", trainable=trainable_state_dict(model), explora_cfg=explora_cfg,
                    args=vars(args), history=history)
    (fig_dir / "stage1_summary.json").write_text(json.dumps(
        {"params": {k: summary[k] for k in ("trainable", "total", "trainable_pct")},
         "epoch_losses": history["epoch_losses"], "epoch_times": history["epoch_times"]}, indent=2))
    print(f"Done. Final checkpoint: {out_dir / 'final.pt'}")
    wandb.finish()


if __name__ == "__main__":
    main()
