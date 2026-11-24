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
    ## Freeze Everything
    model.requires_grad_(False)
    
    num_encoder_blocks = model.config.num_hidden_layers
    num_decoder_blocks = model.config.decoder_num_hidden_layers
    
    ## Unfreeze encoder blocks
    if unfrozen_blocks is None:
        last_block_idx= num_encoder_blocks-1
        unfrozen_blocks=[last_block_idx]
        model.vit.layers[last_block_idx].requires_grad_(True)
    else:
      for block_idx in unfrozen_blocks:
        model.vit.layers[block_idx].requires_grad_(True)
        
    target_names =[]
    
    ## Lora targets for non frozen blocks of encoder
    for block_idx in range(num_encoder_blocks):
      if block_idx not in unfrozen_blocks:
          for t in lora_targets:
              target_names.append(f"vit.layers.{block_idx}.attention.{t}")

                
    
    ## For decoder 
    if lora_decoder:
      for block_idx in range(num_decoder_blocks):
          for t in lora_targets:
              target_names.append(f"decoder.decoder_layers.{block_idx}.attention.{t}")
    
    lora_modules = inject_lora(model,target_names,r=r,alpha=alpha,dropout=dropout)


    if train_layernorms:
      for mod in model.modules():
          if isinstance(mod,nn.LayerNorm):
              mod.requires_grad_(True)

    trainable_parameters,total_parameters = count_parameters(model)

    return {
      "trainable":trainable_parameters,
      "total":total_parameters,
      "trainable_pct":(trainable_parameters/total_parameters)*100,
      "lora_modules":lora_modules,
      "unfrozen_blocks":unfrozen_blocks
    }
