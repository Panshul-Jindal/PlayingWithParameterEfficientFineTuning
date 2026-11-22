"""Phase 5b checks: uv run pytest tests/test_p5_train.py -v"""

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from explora.explora import configure_explora
from explora.train import build_optimizer, train_one_epoch

CPU = torch.device("cpu")


def _scaler():
    return torch.amp.GradScaler("cpu", enabled=False)


def test_optimizer_holds_only_trainable_params(tiny_mae):
    configure_explora(tiny_mae, r=2)
    opt = build_optimizer(tiny_mae, lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    trainable = {id(p) for p in tiny_mae.parameters() if p.requires_grad}
    assert in_opt == trainable, "optimizer must hold exactly the trainable params (no frozen ones, none missing)"


def test_no_weight_decay_on_vectors(tiny_mae):
    configure_explora(tiny_mae, r=2)
    opt = build_optimizer(tiny_mae, lr=1e-3, weight_decay=0.05)
    for g in opt.param_groups:
        for p in g["params"]:
            want = 0.0 if p.ndim < 2 else 0.05
            assert g["weight_decay"] == want, f"param of shape {tuple(p.shape)} has weight_decay {g['weight_decay']}"


def test_one_epoch_updates_only_trainable(tiny_mae, tiny_loader):
    configure_explora(tiny_mae, r=2)
    before = {n: p.detach().clone() for n, p in tiny_mae.named_parameters()}
    opt = build_optimizer(tiny_mae, lr=1e-2)
    loss = train_one_epoch(tiny_mae, tiny_loader, opt, _scaler(), CPU, use_amp=False)
    assert isinstance(loss, float) and math.isfinite(loss)
    for n, p in tiny_mae.named_parameters():
        if p.requires_grad:
            assert not torch.equal(p, before[n]), f"trainable {n} did not change"
        else:
            assert torch.equal(p, before[n]), f"FROZEN {n} changed: check what the optimizer holds"
    assert tiny_mae.training, "train_one_epoch should leave the model in train mode"


def test_logging_contract(tiny_mae, tiny_loader):
    configure_explora(tiny_mae, r=2)
    opt = build_optimizer(tiny_mae, lr=1e-3)
    calls = []
    train_one_epoch(tiny_mae, tiny_loader, opt, _scaler(), CPU, use_amp=False,
                    log_fn=lambda m, s: calls.append((s, m)), log_every=2)
    steps = [s for s, _ in calls]
    assert steps == [0, 2], f"20 samples / batch 8 = 3 steps, log_every=2 -> log at steps [0, 2]; got {steps}"
    assert {"train/loss", "train/lr"} <= set(calls[0][1])
    assert all(isinstance(v, float) for v in calls[0][1].values()), "log plain floats (.item()), not tensors"


def test_scheduler_steps_once_per_batch(tiny_mae, tiny_loader):
    configure_explora(tiny_mae, r=2)
    opt = build_optimizer(tiny_mae, lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    train_one_epoch(tiny_mae, tiny_loader, opt, _scaler(), CPU, scheduler=sched, use_amp=False)
    assert sched.last_epoch == 3, f"expected 3 scheduler steps (one per batch), got {sched.last_epoch}"


def test_max_steps(tiny_mae, tiny_loader):
    configure_explora(tiny_mae, r=2)
    opt = build_optimizer(tiny_mae, lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    train_one_epoch(tiny_mae, tiny_loader, opt, _scaler(), CPU, scheduler=sched, use_amp=False, max_steps=1)
    assert sched.last_epoch == 1


def test_loss_goes_down(tiny_mae):
    # Each image is one flat colour, so masked patches ARE predictable from visible ones.
    # (Pure-noise images would be unlearnable: loss stays ~1.0 whatever you do.)
    g = torch.Generator().manual_seed(0)
    colours = torch.randn(32, 3, 1, 1, generator=g)
    # The whole tiny model trains here: this checks that your loop really optimises.
    loader = DataLoader(TensorDataset(colours.expand(32, 3, 32, 32).clone(), torch.zeros(32)), batch_size=8)
    opt = build_optimizer(tiny_mae, lr=3e-3, weight_decay=0.0)
    losses = [train_one_epoch(tiny_mae, loader, opt, _scaler(), CPU, use_amp=False) for _ in range(25)]
    first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
    assert last < 0.7 * first, f"loss should fall on a tiny fixed dataset: first {first:.3f} -> last {last:.3f}"
