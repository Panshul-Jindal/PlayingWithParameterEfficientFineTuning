"""Task 1 (baseline) and Task 3 (ExPLoRA) linear probes (scaffold; calls YOUR functions).

Protocol, identical for every encoder:
  1. Extract frozen features (no masking) for the train, val and test splits.
  2. For each pooling (CLS, mean): fit the probe on TRAIN for every C in --C-grid and score it on VAL.
     Keep the C with the best val accuracy (ties -> the smaller C, i.e. stronger regularisation).
  3. Evaluate on TEST once, at that C. The test set never influences any choice.
  Also reported, for reference only: test accuracy at C = 1 (the original fixed-C protocol).

Baseline, the pretrained encoder:
    uv run python scripts/run_probe.py --tag baseline
After Stage 1, merge LoRA into the encoder and probe it the same way:
    uv run python scripts/run_probe.py --tag explora --checkpoint checkpoints/stage1/final.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed, parallel_config
from transformers import ViTMAEForPreTraining

from explora.data import CLASS_NAMES, build_loader
from explora.explora import configure_explora
from explora.lora import LoRALinear, merge_lora
from explora.probe import extract_features, fit_linear_probe
from explora.utils import get_device, init_wandb, load_trainable_weights, record_run, seed_everything

MODEL_ID = "facebook/vit-mae-base"
POOLS = ("cls", "mean")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", type=str, required=True, help="e.g. baseline | explora")
    p.add_argument("--checkpoint", type=str, default=None, help="Stage 1 checkpoint (final.pt or last.pt)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--C-grid", type=str, default="0.001,0.003,0.01,0.03,0.1,0.3,1",
                   help="comma-separated values of C (inverse L2 strength) to choose from on the val split")
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


def fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


def fit_many(jobs: list[tuple]) -> list[dict]:
    """Run independent fit_linear_probe calls in parallel processes, 2 BLAS threads each
    (about 10x faster than one after another on a 20-thread CPU). jobs = [(train_x, train_y, eval_x, eval_y, C)]."""
    if not jobs:
        return []
    n = min(len(jobs), max(1, (os.cpu_count() or 2) // 2))
    with parallel_config(backend="loky", inner_max_num_threads=2):
        return Parallel(n_jobs=n)(delayed(fit_linear_probe)(a, b, c, d, C=C) for a, b, c, d, C in jobs)


def main() -> None:
    args = parse_args()
    grid = sorted(float(c) for c in args.C_grid.split(","))
    seed_everything(args.seed)
    device = get_device()
    run = init_wandb(args.wandb_project, "probe", config=vars(args) | {"C_grid": grid}, mode=args.wandb_mode,
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

    feats, labels = {}, {}
    for split in ("train", "val", "test"):
        loader = build_loader(split, batch_size=args.batch_size, img_size=args.img_size, for_training=False,
                              num_workers=args.num_workers)
        print(f"Extracting {split} features ...")
        feats[split], labels[split] = extract_features(encoder, loader, device)
    Path("outputs").mkdir(exist_ok=True)
    np.savez_compressed(f"outputs/features_{args.tag}.npz",
                        **{f"{s}_{k}": v for s in feats for k, v in feats[s].items()},
                        **{f"y_{s}": labels[s] for s in labels})
    (xtr, ytr), (yva, yte) = (feats["train"], labels["train"]), (labels["val"], labels["test"])

    # 1) choose C on VAL: one fit per (pooling, C), all in parallel. In these calls the "test" arguments
    #    of fit_linear_probe are the VAL split. Also the C = 1 reference on test (never used for a choice).
    print(f"Fitting {len(POOLS) * (len(grid) + 1)} probes in parallel ...")
    jobs = [(xtr[p], ytr, feats["val"][p], yva, c) for p in POOLS for c in grid]
    jobs += [(xtr[p], ytr, feats["test"][p], yte, 1.0) for p in POOLS]
    out = fit_many(jobs)
    curves = {p: {c: out[i * len(grid) + j]["test_acc"] for j, c in enumerate(grid)} for i, p in enumerate(POOLS)}
    refs = {p: out[len(POOLS) * len(grid) + i] for i, p in enumerate(POOLS)}
    best = {p: max(grid, key=lambda c: (curves[p][c], -c)) for p in POOLS}  # best val accuracy; ties -> smaller C
    # 2) evaluate on TEST once per pooling, at its chosen C
    finals = fit_many([(xtr[p], ytr, feats["test"][p], yte, best[p]) for p in POOLS if best[p] != 1.0])
    finals = iter(finals)
    results = {}
    for pool in POOLS:
        best_c, ref = best[pool], refs[pool]
        res = ref if best_c == 1.0 else next(finals)
        print(f"[{args.tag}] {pool:>4} val acc by C: "
              + "  ".join(f"{c:g}:{curves[pool][c]:.4f}" for c in grid) + f"  -> C = {best_c:g}")
        results[pool] = {"C": best_c, "val_acc": curves[pool][best_c], "train_acc": res["train_acc"],
                         "test_acc": res["test_acc"], "test_macro_f1": res["test_macro_f1"],
                         "test_acc_C1": ref["test_acc"], "test_macro_f1_C1": ref["test_macro_f1"],
                         "val_curve": {f"{c:g}": curves[pool][c] for c in grid}}
        print(f"[{args.tag}] {pool:>4}: C {best_c:g} | val acc {curves[pool][best_c]:.4f} | train acc "
              f"{res['train_acc']:.4f} | test acc {res['test_acc']:.4f} | test macro-F1 {res['test_macro_f1']:.4f} "
              f"(C = 1 reference: test acc {ref['test_acc']:.4f})")
        run.summary.update({f"probe/{pool}/{k}": v for k, v in results[pool].items() if k != "val_curve"})
        wandb.log({f"confusion/{pool}": wandb.plot.confusion_matrix(
            y_true=yte.tolist(), preds=res["test_pred"].tolist(), class_names=CLASS_NAMES)})
    wandb.log({"probe/val_acc_vs_C": wandb.plot.line_series(
        xs=[math.log10(c) for c in grid], ys=[[curves[p][c] for c in grid] for p in POOLS], keys=list(POOLS),
        title="Probe: val accuracy vs log10(C)", xname="log10(C)")})

    Path("results").mkdir(exist_ok=True)
    results["run"] = record_run("run_probe", args.tag, run, {f"{p}_{k}": round(results[p][k], 4)
                                                             for p in POOLS for k in ("C", "val_acc", "test_acc")})
    Path(f"results/probe_{args.tag}.json").write_text(json.dumps(results, indent=2))

    # Side-by-side table once several runs exist (also logged to wandb). Older files may lack the val columns.
    files = sorted(Path("results").glob("probe_*.json"))
    rows = [[f.stem.removeprefix("probe_"), pool, r[pool].get("C", 1.0), r[pool].get("val_acc"), r[pool]["test_acc"],
             r[pool]["test_macro_f1"], r[pool].get("test_acc_C1", r[pool]["test_acc"] if "C" not in r[pool] else None)]
            for f in files for r in [json.loads(f.read_text())] for pool in POOLS]
    cols = ["encoder", "pooling", "C (picked on val)", "val_acc", "test_acc", "test_macro_f1", "test_acc_at_C1"]
    wandb.log({"probe/comparison": wandb.Table(columns=cols, data=rows)})
    print("\n| encoder | pooling | C (picked on val) | val acc | test acc | test macro-F1 | test acc at C=1 |"
          "\n|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:g} | {fmt(r[3])} | {fmt(r[4])} | {fmt(r[5])} | {fmt(r[6])} |")
    wandb.finish()


if __name__ == "__main__":
    main()
