# ExPLoRA on BloodMNIST

Parameter-efficient extended MAE pretraining (ExPLoRA) of `facebook/vit-mae-base` on unlabeled BloodMNIST images, evaluated with a linear probe before and after.

**Report:** [PDF](report/ExPLoRA_report.pdf) · [Markdown](report/ExPLoRA_report.md) · **Tracking:** [W&B project](https://forge.coreweave.com/wandb/models-indian-institute-of-technology-hyderabad/explora-bloodmnist)

## Results (test set, linear probe, C chosen on val)

| Encoder | CLS | Mean |
|---|---|---|
| Baseline (pretrained MAE) | 97.75% | 97.75% |
| ExPLoRA (20 epochs, LoRA merged) | **98.25%** | **98.33%** |

ExPLoRA trains 7.54M of 112M parameters (6.72%): LoRA r=8 on Q,V of encoder blocks 1–11 and all decoder blocks, encoder block 12, and all LayerNorms.

## Run

```bash
uv sync
uv run pytest -q
uv run python scripts/run_probe.py --tag baseline
uv run python scripts/run_stage1.py --epochs 20
uv run python scripts/run_probe.py --tag explora --checkpoint checkpoints/stage1/final.pt
```

## Code

| What | Where |
|---|---|
| Linear probe | `src/explora/probe.py` |
| LoRA (from scratch) and merge | `src/explora/lora.py` |
| ExPLoRA setup and training loop | `src/explora/explora.py`, `src/explora/train.py` |
| Every run's commit and W&B ID | `results/runs.csv` |
