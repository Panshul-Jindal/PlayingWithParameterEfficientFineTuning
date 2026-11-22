"""Shared fixtures: a TINY randomly-initialised ViT-MAE, so tests run on CPU in seconds.

Same architecture as facebook/vit-mae-base, just smaller:
4 encoder blocks, 2 decoder blocks, width 32, 32x32 images, 8x8 patches -> 16 patches.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTMAEConfig, ViTMAEForPreTraining

TINY = dict(
    image_size=32, patch_size=8, num_channels=3,
    hidden_size=32, num_hidden_layers=4, num_attention_heads=4, intermediate_size=64,
    decoder_hidden_size=32, decoder_num_hidden_layers=2, decoder_num_attention_heads=4,
    decoder_intermediate_size=64, mask_ratio=0.75,
)


@pytest.fixture
def tiny_mae() -> ViTMAEForPreTraining:
    torch.manual_seed(0)
    return ViTMAEForPreTraining(ViTMAEConfig(**TINY))


@pytest.fixture
def tiny_loader() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 3, 32, 32, generator=g)
    y = torch.arange(20) % 4
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


@pytest.fixture
def fixed_noise_tiny() -> torch.Tensor:
    return torch.rand(4, 16, generator=torch.Generator().manual_seed(1))
