ExPLoRA on BloodMNIST
Setup
Dataset: BloodMNIST from MedMNIST (pip install medmnist), at 64×64 resolution. Use the train split for pretraining and probing, and the test split for evaluation.
Backbone: facebook/vit-mae-base, loaded with Hugging Face ViTMAEForPreTraining (encoder and decoder).
Run everything on Colab/Kaggle with fp16 mixed precision, and save checkpoints to Google Drive every epoch.
```
In [1]: import matplotlib.pyplot as plt

In [2]: from medmnist import BloodMNIST

In [3]: dataset = BloodMNIST(split='train',download=True,size=128)

In [4]: x = dataset[0]
```
Task 1: Baselines
Linear probe classification accuracy on BlodMNIST test set.
Task 2: ExPLoRA
(Stage 1)
Freeze all weights, then make these trainable: (a) LoRA (rank 8) on query and value in encoder blocks 1–11 and in all decoder blocks. (b) encoder block 12, fully unfrozen (c) all LayerNorms in the encoder and decoder. You are free to pick any other set of things to apply LoRA on. You can as well try multiple configurations. But given the resources, I will suggest try one reasonable configuration. Alternatively, try multiple LoRA configurations for small number of epochs, and pick the most promising configuration for longer training.
Train with the MAE loss (masking ratio 0.75) on the train images without labels.
Time one epoch first, then choose the number of epochs to fit your compute budget.
Plot the reconstruction loss over training, and show reconstructions of a few masked test images before and after
Stage 1.

Task 3: Evaluation (Stage 2)
Merge the LoRA weights into the encoder(W=W0 + BA, for each W).
Repeat Task 1 with the adapted encoder, using the exact same probing protocol, and compare with the baseline.
As with previous assignments, please make use of LLMs to help code these parts. In the 15-min one-on-one, you will need to explain (a) where in code Linear probe classification is happening, (b) how lora is implemented, (c) how explora is implemented, (d) and the results.




** Special Things that will be seen during grading (Applied AI things):**

Github integration.

Wandb integration. (The experiment results should be on wandb/tensorboard or something similar). Showing a standalone performance
table will not be sufficient.

uv or some other package manager used.