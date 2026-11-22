"""Phase 1 checks: uv run pytest tests/test_p1_data.py -v"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from explora.data import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, build_dataset, build_loader, build_transform, label_to_int

HAVE_DATA = (Path.home() / ".medmnist" / "bloodmnist_64.npz").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="BloodMNIST 64px not downloaded yet")


def _pil(color=(255, 0, 128)):
    return Image.new("RGB", (64, 64), color)


def test_transform_shape_and_dtype():
    t = build_transform(224)
    x = t(_pil())
    assert isinstance(x, torch.Tensor), "transform must return a tensor"
    assert x.shape == (3, 224, 224), f"expected (3, 224, 224), got {tuple(x.shape)}"
    assert x.dtype == torch.float32


def test_transform_normalises_with_imagenet_stats():
    x = build_transform(224)(_pil((255, 0, 128)))
    expected = [(c / 255 - m) / s for c, m, s in zip((255, 0, 128), IMAGENET_MEAN, IMAGENET_STD)]
    got = x.mean(dim=(1, 2)).tolist()
    assert np.allclose(got, expected, atol=1e-2), f"per-channel values {got}, expected ~{expected}"


def test_transform_other_size():
    assert build_transform(128)(_pil()).shape == (3, 128, 128)


def test_eval_transform_is_deterministic():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    t = build_transform(224, augment=False)
    assert torch.equal(t(img), t(img)), "augment=False must be deterministic (probing depends on it)"


def test_augment_transform_shape_and_randomness():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    t = build_transform(224, augment=True)
    outs = [t(img) for _ in range(6)]
    assert all(o.shape == (3, 224, 224) for o in outs)
    assert any(not torch.equal(outs[0], o) for o in outs[1:]), "augment=True should be random"


def test_label_to_int():
    y = label_to_int(np.array([5]))
    assert y == 5 and type(y) is int, "labels must come out as plain Python ints"


@needs_data
def test_dataset_items():
    ds = build_dataset("test")
    assert len(ds) == 3421
    x, y = ds[0]
    assert x.shape == (3, 224, 224)
    assert type(y) is int and 0 <= y < NUM_CLASSES


@needs_data
def test_training_loader_shuffles_and_drops_last():
    dl = build_loader("val", batch_size=64, for_training=True, num_workers=0)
    assert isinstance(dl.sampler, RandomSampler), "training loader must shuffle"
    assert dl.drop_last, "training loader should drop the incomplete last batch"
    x, y = next(iter(dl))
    assert x.shape == (64, 3, 224, 224) and y.shape == (64,) and y.dtype == torch.int64


@needs_data
def test_eval_loader_keeps_every_sample_in_order():
    dl = build_loader("val", batch_size=100, for_training=False, num_workers=0)
    assert isinstance(dl.sampler, SequentialSampler), "eval loader must NOT shuffle (features must align with labels)"
    assert not dl.drop_last, "eval loader must keep the last partial batch"
    assert len(dl) == int(np.ceil(1712 / 100))
