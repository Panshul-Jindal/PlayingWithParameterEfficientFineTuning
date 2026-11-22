"""Task 1 (baseline) and Task 3 (ExPLoRA) linear probes (scaffold; calls YOUR functions).

Baseline, the pretrained encoder:
    uv run python scripts/run_probe.py --tag baseline
After Stage 1, merge LoRA into the encoder and probe it the same way:
    uv run python scripts/run_probe.py --tag explora --checkpoint checkpoints/stage1/final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import ViTMAEForPreTraining

from explora.data import CLASS_NAMES, build_loader
from explora.explora import configure_explora
from explora.lora import LoRALinear, merge_lora
from explora.probe import extract_features, fit_linear_probe
from explora.utils import get_device, init_wandb, load_trainable_weights, seed_everything

MODEL_ID = "facebook/vit-mae-base"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", type=str, required=True, help="e.g. baseline | explora")
    p.add_argument("--checkpoint", type=str, default=None, help="Stage 1 checkpoint (final.pt or last.pt)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the logistic regression")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", type=str, default="explora-bloodmnist")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


@torch.no_grad()
def merge_sanity_check(model, device) -> float:
    """The merged encoder must compute the same function as the LoRA encoder."""
    x = torch.randn(4, 3, 224, 224, device=device)
    noise = torch.rand(4, (224 // model.config.patch_size) ** 2, device=device)
    model.eval()
    before = model.vit(pixel_values=x, noise=noise).last_hidden_state.float()
    merge_lora(model)
    after = model.vit(pixel_values=x, noise=noise).last_hidden_state.float()
    return (before - after).abs().max().item()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run = init_wandb(args.wandb_project, "probe", config=vars(args), mode=args.wandb_mode,
                     name=f"probe-{args.tag}", tags=["probe", args.tag])
    import wandb

    model = ViTMAEForPreTraining.from_pretrained(MODEL_ID)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        configure_explora(model, **ckpt["explora_cfg"])  # rebuild the same structure ...
        load_trainable_weights(model, ckpt["trainable"])  # ... then load the trained deltas
        model.to(device)
        diff = merge_sanity_check(model, device)  # W = W0 + (alpha/r) B A, for every LoRA layer
        assert not any(isinstance(m, LoRALinear) for m in model.modules()), "LoRA layers left after merge"
        print(f"Merged LoRA into the encoder. Max |before - after| on random input: {diff:.2e}")
        run.summary["merge/max_abs_diff"] = diff
    encoder = model.vit.to(device)

    train_loader = build_loader("train", batch_size=args.batch_size, img_size=args.img_size, for_training=False,
                                num_workers=args.num_workers)
    test_loader = build_loader("test", batch_size=args.batch_size, img_size=args.img_size, for_training=False,
                               num_workers=args.num_workers)
    print("Extracting train features ...")
    f_tr, y_tr = extract_features(encoder, train_loader, device)
    print("Extracting test features ...")
    f_te, y_te = extract_features(encoder, test_loader, device)
    Path("outputs").mkdir(exist_ok=True)
    np.savez_compressed(f"outputs/features_{args.tag}.npz", **{f"train_{k}": v for k, v in f_tr.items()},
                        **{f"test_{k}": v for k, v in f_te.items()}, y_train=y_tr, y_test=y_te)

    results = {}
    for pool in ("cls", "mean"):
        res = fit_linear_probe(f_tr[pool], y_tr, f_te[pool], y_te, C=args.C)
        results[pool] = {k: v for k, v in res.items() if k != "test_pred"}
        print(f"[{args.tag}] {pool:>4}: train acc {res['train_acc']:.4f} | test acc {res['test_acc']:.4f} | "
              f"test macro-F1 {res['test_macro_f1']:.4f}")
        run.summary.update({f"probe/{pool}/test_acc": res["test_acc"], f"probe/{pool}/train_acc": res["train_acc"],
                            f"probe/{pool}/test_macro_f1": res["test_macro_f1"]})
        wandb.log({f"confusion/{pool}": wandb.plot.confusion_matrix(
            y_true=y_te.tolist(), preds=res["test_pred"].tolist(), class_names=CLASS_NAMES)})

    Path("results").mkdir(exist_ok=True)
    Path(f"results/probe_{args.tag}.json").write_text(json.dumps(results, indent=2))

    # Side-by-side table once both runs exist (also logged to wandb)
    files = sorted(Path("results").glob("probe_*.json"))
    rows = [[f.stem.removeprefix("probe_"), pool, r[pool]["test_acc"], r[pool]["test_macro_f1"]]
            for f in files for r in [json.loads(f.read_text())] for pool in ("cls", "mean")]
    wandb.log({"probe/comparison": wandb.Table(columns=["encoder", "pooling", "test_acc", "test_macro_f1"], data=rows)})
    print("\n| encoder | pooling | test acc | test macro-F1 |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.4f} | {r[3]:.4f} |")
    wandb.finish()


if __name__ == "__main__":
    main()
