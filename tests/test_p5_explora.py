"""Phase 5a checks: uv run pytest tests/test_p5_explora.py -v

The tiny model has 4 encoder blocks (0-3) and 2 decoder blocks, so the default config is:
block 3 fully unfrozen, LoRA on q/v of encoder blocks 0-2 and decoder blocks 0-1, all LayerNorms trainable.
"""

import torch
from torch import nn

from explora.explora import configure_explora
from explora.lora import LoRALinear


def _expected_trainable(model, unfrozen):
    names = set()
    for n, _ in model.named_parameters():
        if n.endswith(("lora_A", "lora_B")):
            names.add(n)
        elif any(n.startswith(f"vit.layers.{i}.") for i in unfrozen):
            names.add(n)
        elif "layernorm" in n or "decoder_norm" in n:
            names.add(n)
    return names


def test_lora_placement_default(tiny_mae):
    summary = configure_explora(tiny_mae, r=2, alpha=4)
    wrapped = {n for n, m in tiny_mae.named_modules() if isinstance(m, LoRALinear)}
    expected = {f"vit.layers.{i}.attention.{t}" for i in range(3) for t in ("q_proj", "v_proj")}
    expected |= {f"decoder.decoder_layers.{j}.attention.{t}" for j in range(2) for t in ("q_proj", "v_proj")}
    assert wrapped == expected, f"LoRA in the wrong places.\n missing: {expected - wrapped}\n extra: {wrapped - expected}"
    assert set(summary["lora_modules"]) == expected
    assert list(summary["unfrozen_blocks"]) == [3]


def test_unfrozen_block_has_no_lora_and_is_fully_trainable(tiny_mae):
    configure_explora(tiny_mae, r=2)
    last = tiny_mae.vit.layers[3]
    assert not any(isinstance(m, LoRALinear) for m in last.modules()), "the unfrozen block shouldn't also get LoRA"
    assert all(p.requires_grad for p in last.parameters()), "every param in the last block must train"


def test_exact_trainable_set(tiny_mae):
    configure_explora(tiny_mae, r=2)
    got = {n for n, p in tiny_mae.named_parameters() if p.requires_grad}
    want = _expected_trainable(tiny_mae, unfrozen=[3])
    assert got == want, f"\n should train but frozen: {sorted(want - got)[:8]}\n trains but shouldn't: {sorted(got - want)[:8]}"


def test_things_that_must_stay_frozen(tiny_mae):
    configure_explora(tiny_mae, r=2)
    frozen = [tiny_mae.vit.embeddings.patch_embeddings, tiny_mae.decoder.decoder_embed, tiny_mae.decoder.decoder_pred,
              tiny_mae.vit.layers[0].mlp, tiny_mae.vit.layers[0].attention.k_proj, tiny_mae.vit.layers[0].attention.o_proj]
    for mod in frozen:
        assert not any(p.requires_grad for p in mod.parameters()), f"{type(mod).__name__} should be frozen"
    assert not tiny_mae.decoder.mask_token.requires_grad
    assert not tiny_mae.vit.layers[0].attention.q_proj.base.weight.requires_grad, "the LoRA base weight stays frozen"


def test_all_layernorms_trainable(tiny_mae):
    configure_explora(tiny_mae, r=2)
    lns = [m for m in tiny_mae.modules() if isinstance(m, nn.LayerNorm)]
    assert len(lns) == 4 * 2 + 1 + 2 * 2 + 1  # 2 per block + final encoder norm + decoder_norm
    assert all(p.requires_grad for ln in lns for p in ln.parameters())


def test_summary_counts(tiny_mae):
    s = configure_explora(tiny_mae, r=2)
    trainable = sum(p.numel() for p in tiny_mae.parameters() if p.requires_grad)
    total = sum(p.numel() for p in tiny_mae.parameters())
    assert s["trainable"] == trainable and s["total"] == total
    assert abs(s["trainable_pct"] - 100 * trainable / total) < 1e-6


def test_custom_unfrozen_blocks_and_no_decoder_lora(tiny_mae):
    s = configure_explora(tiny_mae, r=2, unfrozen_blocks=[0, 3], lora_decoder=False)
    wrapped = {n for n, m in tiny_mae.named_modules() if isinstance(m, LoRALinear)}
    assert wrapped == {f"vit.layers.{i}.attention.{t}" for i in (1, 2) for t in ("q_proj", "v_proj")}
    assert sorted(s["unfrozen_blocks"]) == [0, 3]


def test_backward_only_touches_trainable(tiny_mae):
    configure_explora(tiny_mae, r=2)
    with torch.no_grad():  # make B nonzero so every LoRA layer matters
        for m in tiny_mae.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.1)
    tiny_mae(pixel_values=torch.randn(2, 3, 32, 32)).loss.backward()
    for n, p in tiny_mae.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{n} is trainable but got no gradient"
        else:
            assert p.grad is None, f"{n} is frozen but got a gradient"
