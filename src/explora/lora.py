"""Phase 4: LoRA from scratch (no `peft` library). This is viva question (b).

For a frozen pretrained linear layer y = W0 x + b, LoRA learns a low-rank update:

    y = W0 x + b + (alpha / r) * B A x          A: (r, in_features)   B: (out_features, r)

  * W0 and b are frozen. Only A and B train: r * (in + out) parameters instead of in * out.
  * B starts at zero, so at step 0 the wrapped layer computes exactly W0 x + b
    (training starts from the pretrained model, not a perturbed one).
  * A gets a Kaiming-uniform init (as nn.Linear does), so gradients reach B from step 1.
  * After training, merge: W = W0 + (alpha / r) * B @ A. Then the layer is a plain nn.Linear
    again, with zero extra inference cost (Phase 6, viva question (c)/(d)).

Your vault note [[LoRA]] has the maths. This file is the code.
Docs: nn.Module / nn.Parameter / nn.init: https://pytorch.org/docs/stable/nn.html
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear and adds a trainable low-rank update.

    Contract (the tests rely on these names):
      self.base     the original nn.Linear, with all its params frozen (requires_grad=False)
      self.lora_A   nn.Parameter, shape (r, in_features), Kaiming-uniform init (a=sqrt(5))
      self.lora_B   nn.Parameter, shape (out_features, r), zero init
      self.r, self.alpha, self.scaling (= alpha / r)
      self.dropout  nn.Dropout(dropout) applied to x on the LoRA branch only (nn.Identity if dropout == 0)
    lora_A and lora_B must be on the same device and dtype as base.weight.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("Phase 4: LoRALinear.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 4: LoRALinear.forward")

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Return a NEW nn.Linear with weight W0 + scaling * B @ A and the same bias.
        Do the maths in float32, then cast back to base.weight's dtype."""
        raise NotImplementedError("Phase 6: LoRALinear.merged_linear")


def set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace the submodule at dotted path `name` (e.g. 'vit.layers.3.attention.q_proj')."""
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, new_module)


def inject_lora(
    model: nn.Module, target_names: Iterable[str], r: int = 8, alpha: float = 16.0, dropout: float = 0.0
) -> list[str]:
    """Wrap each named nn.Linear in a LoRALinear, in place. Return the list of wrapped names.
    Raise TypeError if a target is not an nn.Linear (that includes an already-wrapped one)."""
    raise NotImplementedError("Phase 4: inject_lora")


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear in `model` with its merged nn.Linear, in place.
    Return how many were merged. Afterwards the model has exactly the original
    architecture (same state_dict keys as a fresh from_pretrained model)."""
    raise NotImplementedError("Phase 6: merge_lora")
