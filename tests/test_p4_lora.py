"""Phase 4 checks: uv run pytest tests/test_p4_lora.py -v"""

import math

import pytest
import torch
from torch import nn

from explora.lora import LoRALinear, inject_lora


@pytest.fixture
def base():
    torch.manual_seed(0)
    return nn.Linear(12, 20)


def test_shapes_names_and_scaling(base):
    m = LoRALinear(base, r=4, alpha=8)
    assert m.base is base, "keep the original layer as self.base"
    assert isinstance(m.lora_A, nn.Parameter) and isinstance(m.lora_B, nn.Parameter)
    assert m.lora_A.shape == (4, 12), f"lora_A should be (r, in_features), got {tuple(m.lora_A.shape)}"
    assert m.lora_B.shape == (20, 4), f"lora_B should be (out_features, r), got {tuple(m.lora_B.shape)}"
    assert m.r == 4 and m.alpha == 8 and math.isclose(m.scaling, 2.0)


def test_only_A_and_B_are_trainable(base):
    m = LoRALinear(base, r=4, alpha=8)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}, f"trainable params should be exactly lora_A, lora_B; got {trainable}"
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 4 * (12 + 20)


def test_init_B_zero_A_nonzero(base):
    m = LoRALinear(base, r=4)
    assert torch.count_nonzero(m.lora_B) == 0, "B must start at zero"
    assert torch.count_nonzero(m.lora_A) > 0, "A must NOT start at zero (else grads to B are zero forever)"
    bound = 1 / math.sqrt(12)  # kaiming_uniform_(a=sqrt(5)) on fan_in=12 gives U(-1/sqrt(fan_in), +)
    assert m.lora_A.abs().max() <= bound + 1e-6, "use nn.init.kaiming_uniform_(A, a=math.sqrt(5))"


def test_starts_as_exactly_the_pretrained_layer(base):
    x = torch.randn(5, 7, 12)  # (batch, tokens, features): must work on 3-D inputs too
    m = LoRALinear(base, r=4)
    assert torch.equal(m(x), base(x)), "with B = 0 the output must equal the base layer's output exactly"


def test_forward_matches_formula(base):
    m = LoRALinear(base, r=4, alpha=8)
    with torch.no_grad():
        m.lora_B.normal_()
    x = torch.randn(3, 12)
    expected = x @ (base.weight + 2.0 * m.lora_B @ m.lora_A).T + base.bias
    assert torch.allclose(m(x), expected, atol=1e-5), "forward != W0 x + b + (alpha/r) B A x"


def test_dropout_only_on_lora_branch(base):
    m = LoRALinear(base, r=4, dropout=0.5)
    m.train()
    x = torch.randn(64, 12)
    assert torch.equal(m(x), base(x)), "with B = 0 dropout must not touch the base path"


def test_gradients_reach_A_and_B_not_base(base):
    m = LoRALinear(base, r=4)
    with torch.no_grad():
        m.lora_B.normal_()
    m(torch.randn(3, 12)).sum().backward()
    assert m.lora_A.grad is not None and m.lora_B.grad is not None
    assert base.weight.grad is None and base.bias.grad is None


def test_inject_wraps_exactly_the_named_layers(tiny_mae):
    names = ["vit.layers.0.attention.q_proj", "decoder.decoder_layers.1.attention.v_proj"]
    out = inject_lora(tiny_mae, names, r=2, alpha=4)
    assert list(out) == names
    wrapped = sorted(n for n, m in tiny_mae.named_modules() if isinstance(m, LoRALinear))
    assert wrapped == sorted(names)
    assert isinstance(tiny_mae.vit.layers[0].attention.k_proj, nn.Linear), "don't wrap layers you weren't asked to"
    assert tiny_mae.vit.layers[0].attention.q_proj.lora_A.shape == (2, 32)


def test_inject_keeps_model_output_at_init(tiny_mae, fixed_noise_tiny):
    x = torch.randn(4, 3, 32, 32)
    tiny_mae.eval()
    with torch.no_grad():
        before = tiny_mae(pixel_values=x, noise=fixed_noise_tiny).logits
        inject_lora(tiny_mae, [f"vit.layers.{i}.attention.{t}" for i in range(4) for t in ("q_proj", "v_proj")])
        after = tiny_mae(pixel_values=x, noise=fixed_noise_tiny).logits
    assert torch.allclose(before, after, atol=1e-6), "injecting LoRA must not change the model at init"


def test_inject_rejects_non_linear_and_double_wrap(tiny_mae):
    with pytest.raises(TypeError):
        inject_lora(tiny_mae, ["vit.layers.0.layernorm_before"])
    inject_lora(tiny_mae, ["vit.layers.0.attention.q_proj"])
    with pytest.raises(TypeError):
        inject_lora(tiny_mae, ["vit.layers.0.attention.q_proj"])
