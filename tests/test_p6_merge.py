"""Phase 6 checks: uv run pytest tests/test_p6_merge.py -v"""

import pytest
import torch
from torch import nn
from transformers import ViTMAEConfig, ViTMAEForPreTraining

from explora.explora import configure_explora
from explora.lora import LoRALinear, merge_lora
from explora.probe import extract_features

from conftest import TINY


@pytest.fixture
def base():
    torch.manual_seed(0)
    return nn.Linear(12, 20)


def _perturb_lora(model, std=0.1):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, std)


def test_merged_linear_matches(base):
    m = LoRALinear(base, r=4, alpha=8)
    with torch.no_grad():
        m.lora_B.normal_()
    merged = m.merged_linear()
    assert type(merged) is nn.Linear, "merged_linear must return a plain nn.Linear"
    assert merged is not base, "return a NEW layer (don't silently edit the frozen base in place)"
    x = torch.randn(6, 12)
    assert torch.allclose(merged(x), m(x), atol=1e-5)
    assert torch.allclose(merged.weight, base.weight + 2.0 * m.lora_B @ m.lora_A, atol=1e-6)
    assert torch.equal(merged.bias, base.bias)


def test_merged_linear_without_bias():
    torch.manual_seed(0)
    m = LoRALinear(nn.Linear(5, 7, bias=False), r=2)
    with torch.no_grad():
        m.lora_B.normal_()
    merged = m.merged_linear()
    assert merged.bias is None
    x = torch.randn(3, 5)
    assert torch.allclose(merged(x), m(x), atol=1e-5)


def test_merge_restores_original_architecture(tiny_mae):
    original_keys = set(ViTMAEForPreTraining(ViTMAEConfig(**TINY)).state_dict())
    configure_explora(tiny_mae, r=2)
    n = merge_lora(tiny_mae)
    assert n == (3 + 2) * 2, f"expected 10 LoRA layers merged, got {n}"
    assert not any(isinstance(m, LoRALinear) for m in tiny_mae.modules())
    assert set(tiny_mae.state_dict()) == original_keys, "after merging, state_dict keys must match a fresh model"


def test_merge_preserves_the_function(tiny_mae, fixed_noise_tiny):
    configure_explora(tiny_mae, r=2)
    _perturb_lora(tiny_mae)
    x = torch.randn(4, 3, 32, 32)
    tiny_mae.eval()
    with torch.no_grad():
        before = tiny_mae(pixel_values=x, noise=fixed_noise_tiny).logits
        merge_lora(tiny_mae)
        after = tiny_mae(pixel_values=x, noise=fixed_noise_tiny).logits
    assert torch.allclose(before, after, atol=1e-5), "the merged model must compute the same function"


def test_probe_features_identical_after_merge(tiny_mae, tiny_loader):
    configure_explora(tiny_mae, r=2)
    _perturb_lora(tiny_mae)
    cpu = torch.device("cpu")
    a, _ = extract_features(tiny_mae.vit, tiny_loader, cpu)
    merge_lora(tiny_mae)
    b, _ = extract_features(tiny_mae.vit, tiny_loader, cpu)
    for k in ("cls", "mean"):
        assert abs(a[k] - b[k]).max() < 1e-4
