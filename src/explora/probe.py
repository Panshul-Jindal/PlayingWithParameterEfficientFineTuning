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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
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
    
    old, was_training = encoder.config.mask_ratio, encoder.training
    encoder.config.mask_ratio = 0.0; encoder.eval()
    
    enable_autocast = use_amp and device.type=='cuda'
    cls_embs = []
    mean_embs = []
    targets = []
    try:
      for x,y in loader:
          x = x.to(device)
          with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enable_autocast):
            output  = encoder(x)
            output_tokens  = output.last_hidden_state 
            
            cls_emb = output_tokens[:,0].float().cpu().numpy()
            mean_emb = output_tokens[:,1:].float().mean(dim=1).cpu().numpy()
            labels = y.numpy()
            cls_embs.append(cls_emb)
            mean_embs.append(mean_emb) 
            targets.append(labels)
    finally:
            encoder.config.mask_ratio = old; encoder.train(was_training)
    
    cls_embs = np.concatenate(cls_embs,axis = 0)
    mean_embs = np.concatenate(mean_embs,axis =0)
    targets = np.concatenate(targets,axis = 0 )
    
    return ({"cls":cls_embs,"mean":mean_embs},targets)



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
    
    ## Standardize the train dataset
    
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x= scaler.transform(train_x)
    
    test_x = scaler.transform(test_x)
    
    clf = LogisticRegression(random_state=0,C=C,max_iter=max_iter).fit(train_x,train_y)
    train_predictions = clf.predict(train_x)
    

    train_accuracy = accuracy_score(train_predictions,train_y)
    test_predictions = clf.predict(test_x)
    test_accuracy = accuracy_score(test_predictions,test_y)
    test_macro_f1_score = f1_score(test_predictions,test_y,average='macro')
    
    
    
    return {"train_acc":train_accuracy,"test_acc":test_accuracy,"test_macro_f1":test_macro_f1_score ,
            "test_pred":test_predictions}
