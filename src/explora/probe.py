"""Phase 3: the linear probe (Task 1 and Task 3). This is viva question (a).

Protocol (keep it IDENTICAL for the baseline and the ExPLoRA encoder):
  1. Freeze the encoder. Run every train and test image through it once with NO masking.
  2. Two candidate image embeddings from last_hidden_state (B, 1 + num_patches, D):
       "cls"  = the [CLS] token, last_hidden_state[:, 0]
       "mean" = the average of the patch tokens, last_hidden_state[:, 1:].mean(1)
  3. Standardise the features (fit on TRAIN only), fit multinomial logistic regression
     on the train features, report accuracy on the test features.
The probe never changes the encoder. It measures how linearly separable the
classes already are in the representation.

Docs: ViTMAEModel: https://huggingface.co/docs/transformers/model_doc/vit_mae
      sklearn:     LogisticRegression, StandardScaler, make_pipeline, accuracy_score, f1_score
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(
    encoder: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = True
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run a ViTMAEModel (model.vit) over `loader` and collect embeddings.

    Returns ({"cls": (N, D) float32, "mean": (N, D) float32}, labels (N,) int64),
    in the loader's order.

    Gotchas the tests check:
      * ViTMAEModel MASKS 75% OF PATCHES BY DEFAULT, even in eval mode. Turn masking off
        while extracting (look at how embeddings.random_masking decides how many to keep),
        then put the original setting back. The config object is SHARED with the full
        pretraining model, and Stage 1 must still mask at 0.75.
      * Use eval mode and restore the encoder's previous train/eval mode afterwards.
      * Use fp16 autocast only when use_amp is True AND the device is CUDA. Return float32 numpy.
    """
    raise NotImplementedError("Phase 3: extract_features")


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    C: float = 1.0,
    max_iter: int = 2000,
) -> dict:
    """StandardScaler (fit on train only) + LogisticRegression(C, max_iter).

    Returns a dict with at least:
      "train_acc": float, "test_acc": float, "test_macro_f1": float, "test_pred": np.ndarray (N_test,)
    """
    raise NotImplementedError("Phase 3: fit_linear_probe")
