"""Phase 3 checks: uv run pytest tests/test_p3_probe.py -v"""

import numpy as np
import torch

from explora.probe import extract_features, fit_linear_probe

CPU = torch.device("cpu")


def test_feature_shapes_and_labels(tiny_mae, tiny_loader):
    feats, labels = extract_features(tiny_mae.vit, tiny_loader, CPU)
    assert set(feats) >= {"cls", "mean"}
    for k in ("cls", "mean"):
        assert feats[k].shape == (20, 32), f"{k}: expected (N=20, D=32), got {feats[k].shape}"
        assert feats[k].dtype == np.float32
    assert labels.shape == (20,)
    np.testing.assert_array_equal(labels, np.arange(20) % 4, err_msg="labels must stay in loader order")


def test_features_are_deterministic_so_masking_is_off(tiny_mae, tiny_loader):
    a, _ = extract_features(tiny_mae.vit, tiny_loader, CPU)
    b, _ = extract_features(tiny_mae.vit, tiny_loader, CPU)
    for k in ("cls", "mean"):
        assert np.allclose(a[k], b[k], atol=1e-5), (
            f"'{k}' features differ between two identical calls: random masking is still on. "
            "ViTMAEModel masks 75% of patches unless you change config.mask_ratio."
        )


def test_mask_ratio_is_restored(tiny_mae, tiny_loader):
    extract_features(tiny_mae.vit, tiny_loader, CPU)
    assert tiny_mae.config.mask_ratio == 0.75, (
        "mask_ratio was not restored. The config object is shared with the pretraining model, "
        "so Stage 1 would now train with NO masking."
    )


def test_cls_differs_from_mean(tiny_mae, tiny_loader):
    feats, _ = extract_features(tiny_mae.vit, tiny_loader, CPU)
    assert not np.allclose(feats["cls"], feats["mean"]), "cls and mean pooling should not be identical"


def test_mean_feature_matches_manual_unmasked_forward(tiny_mae, tiny_loader):
    feats, _ = extract_features(tiny_mae.vit, tiny_loader, CPU)
    x, _ = next(iter(tiny_loader))
    tiny_mae.config.mask_ratio = 0.0
    tiny_mae.eval()
    with torch.no_grad():
        h = tiny_mae.vit(pixel_values=x).last_hidden_state
    tiny_mae.config.mask_ratio = 0.75
    assert h.shape[1] == 17  # CLS + all 16 patches
    assert np.allclose(feats["mean"][:8], h[:, 1:].mean(1).numpy(), atol=1e-5), "mean = average of PATCH tokens (skip CLS)"
    assert np.allclose(feats["cls"][:8], h[:, 0].numpy(), atol=1e-5)


def test_training_mode_restored(tiny_mae, tiny_loader):
    tiny_mae.train()
    extract_features(tiny_mae.vit, tiny_loader, CPU)
    assert tiny_mae.vit.training, "restore the encoder's previous train/eval mode"


def test_linear_probe_on_separable_blobs():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 5, size=(3, 10))
    ytr = rng.integers(0, 3, 300)
    yte = rng.integers(0, 3, 100)
    xtr = (centers[ytr] + rng.normal(0, 1, size=(300, 10))).astype(np.float32)
    xte = (centers[yte] + rng.normal(0, 1, size=(100, 10))).astype(np.float32)
    res = fit_linear_probe(xtr, ytr, xte, yte)
    for key in ("train_acc", "test_acc", "test_macro_f1", "test_pred"):
        assert key in res, f"missing key {key!r}"
    assert res["test_acc"] > 0.95 and res["train_acc"] > 0.95
    assert res["test_pred"].shape == (100,)
    assert 0.0 <= res["test_macro_f1"] <= 1.0


def test_linear_probe_is_scale_invariant():
    """With standardisation, multiplying a feature by 1000 must not change the result much."""
    rng = np.random.default_rng(3)
    centers = rng.normal(0, 2, size=(4, 6))
    ytr, yte = rng.integers(0, 4, 400), rng.integers(0, 4, 200)
    xtr = centers[ytr] + rng.normal(0, 1, size=(400, 6))
    xte = centers[yte] + rng.normal(0, 1, size=(200, 6))
    scale = np.array([1000.0, 1, 1, 1, 1, 0.001])
    a = fit_linear_probe(xtr, ytr, xte, yte)["test_acc"]
    b = fit_linear_probe(xtr * scale, ytr, xte * scale, yte)["test_acc"]
    assert abs(a - b) < 0.02, "did you standardise the features (fit on train, apply to both)?"
