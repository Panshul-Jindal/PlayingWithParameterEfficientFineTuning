"""Phase 5a: choosing what trains in ExPLoRA Stage 1. This is viva question (c).

The assignment's configuration for facebook/vit-mae-base (12 encoder blocks, 8 decoder blocks):
  1. Freeze everything.
  2. Encoder block 12 (index 11, the last one) fully unfrozen.
  3. LoRA (rank 8) on query and value in encoder blocks 1-11 (indices 0-10) and in ALL decoder blocks.
  4. All LayerNorms in the encoder and decoder trainable.
Then train with the ordinary MAE loss (mask ratio 0.75) on unlabeled BloodMNIST images.

Before writing this, find the real module names yourself:
    m = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    for name, mod in m.named_modules(): print(name, type(mod).__name__)
Tutorials (and LLMs) often say `attention.attention.query`. That was transformers v4.
Trust the installed source, not memory.
"""

from __future__ import annotations

from typing import Sequence

from torch import nn

from explora.lora import inject_lora
from explora.utils import count_parameters


def configure_explora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    unfrozen_blocks: Sequence[int] | None = None,
    lora_targets: Sequence[str] = ("q_proj", "v_proj"),
    lora_decoder: bool = True,
    train_layernorms: bool = True,
) -> dict:
    """Set up a ViTMAEForPreTraining for ExPLoRA Stage 1, in place.

    unfrozen_blocks: encoder block indices (0-based) to unfreeze fully. None means the last block only.
                     These blocks get NO LoRA (they're already fully trainable).
    lora_targets:    names of the attention projections to wrap, inside each block's `.attention`.
    lora_decoder:    also add LoRA to the same projections in every decoder block.

    Must work for any depth (the tests use a tiny 4-block encoder / 2-block decoder),
    so read the depths from model.config instead of hard-coding 12 and 8.

    Returns a summary dict with at least:
      "trainable": int, "total": int, "trainable_pct": float (0-100),
      "lora_modules": list[str], "unfrozen_blocks": list[int]
    """
    raise NotImplementedError("Phase 5: configure_explora")
