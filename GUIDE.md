# ExPLoRA on BloodMNIST: your guided build

> **In one breath.** Take an ImageNet-pretrained ViT-MAE. Probe it on blood-cell images (Task 1). Continue its *own* MAE pretraining on unlabeled blood cells while training only ~7% of its weights: the last block, LoRA on Q/V everywhere else, and the LayerNorms (Task 2). Merge LoRA back into the weights and probe again with the identical protocol (Task 3). You write every piece the viva asks about. The plumbing is already done.

**Theory is already in your vault:** the ExPLoRA, LoRA and Masked Autoencoders notes (`Vault_713/Super Intelligence/Self-Supervised Learning/`). This guide is about **code**.

---

## How this works

| | |
|---|---|
| **You write** | `src/explora/data.py` · `probe.py` · `lora.py` · `explora.py` · `train.py`. Each function has a docstring that is its contract. |
| **Already written** | `utils.py` (checkpoints, LR schedule, wandb) · `viz.py` (reconstructions) · `scripts/run_stage1.py` · `scripts/run_probe.py` · the notebook |
| **Your checker** | `uv run pytest tests/test_p3_probe.py -v`. 47 tests across 5 files. They're the hidden tests, like `./drill check`. |

**The rules (same as your Coding Drills):**
1. **Docs first, no LLM.** PyTorch docs, HF docs, `help()`, and reading installed source are allowed. This is exactly the interview condition ("code with documentation").
2. **Stuck for 10 minutes?** Open the next hint (they're folded, in order). Still stuck? Tell me: phase, what you tried, the exact error. I'll give you the next rung, not the answer.
3. **Tests green → say "review P3" (etc.).** I'll review your code like a senior engineer would: correctness, idioms, what an interviewer would poke at.
4. **Commit after every phase.** GitHub history is graded, and small commits look professional.
5. **Every lookup goes into your error log** (the API call you had to search for, the gotcha that bit you). Those are your interview flashcards.

### Viva map (the 15-minute one-on-one)
| They ask | You point at |
|---|---|
| (a) Where is linear-probe classification happening? | `probe.py: extract_features` (frozen, unmasked encoder → CLS/mean embeddings) and `fit_linear_probe` (StandardScaler + LogisticRegression) |
| (b) How is LoRA implemented? | `lora.py: LoRALinear`, `inject_lora` |
| (c) How is ExPLoRA implemented? | `explora.py: configure_explora` (what trains) + `train.py: train_one_epoch` (MAE loss, fp16) + `lora.py: merge_lora` |
| (d) Results | wandb project `explora-bloodmnist`: loss curve, reconstructions before/after, the probe comparison table |

---

## 3-day schedule

| Day | Phases | Time | End state |
|---|---|---|---|
| **Day 1 (Sun 22)** | P0 setup → P1 data → P2 explore the model → P3 probe | ~3 h | **Baseline accuracy on wandb** (Task 1 ✓) |
| **Day 2 (Mon 23)** | P4 LoRA → P5 ExPLoRA + training loop → time an epoch → full run | ~3.5 h | **Stage 1 trained**, loss curve + reconstructions on wandb (Task 2 ✓) |
| **Day 3 (Tue 24)** | P6 merge + probe → P7 results → P8 README + viva rehearsal | ~3 h | **Comparison table**, README, you can explain every line (Task 3 ✓) |

A full epoch takes **~1 min on your RTX 3060** (batch 64, fp16, 3.4 GB). Compute isn't the bottleneck; your understanding is.

---

## P0: Setup (15 min)

```bash
cd ~/AI/projects/PlayingWithParameterEfficientFineTuning
uv sync                      # creates .venv from pyproject.toml + uv.lock
uv run wandb login           # paste your API key from wandb.ai/authorize
uv run pytest -q             # expect 47 failures (NotImplementedError). That's the starting line.
```

Read `pyproject.toml`, `src/explora/utils.py` and `scripts/run_stage1.py` once, top to bottom (10 min). Know what the scaffold does before you fill it in.

> [!question] Explain it
> Why does `save_checkpoint` save only the **trainable** parameters? How big is each checkpoint vs a full model, and what must you do before you can load one?

```bash
git add -A && git commit -m "Scaffold: uv project, tests, training/probe scripts"
```

---

## P1: Data (25 min) · `data.py`

**Goal:** BloodMNIST 64×64 → tensors the ImageNet-pretrained ViT-MAE expects.
**Docs:** [torchvision transforms](https://pytorch.org/vision/stable/transforms.html) · [DataLoader](https://pytorch.org/docs/stable/data.html) · `uv run python -c "from medmnist import BloodMNIST; help(BloodMNIST)"`
**Check:** `uv run pytest tests/test_p1_data.py -v`

**Decide before you code:** the model was pretrained at 224×224 with 16×16 patches. You have 64×64 images. What breaks if you feed 64×64 directly? (Count the patches, then count how many survive 75% masking.) Why upsample to 224?

<details><summary>Hint 1: what goes in the transform</summary>

Eval: resize to (img_size, img_size) → to tensor in [0, 1] → normalise with ImageNet mean/std. Train (augment=True): replace the resize with a random resized crop (scale (0.5, 1)) and add horizontal + vertical flips.
</details>

<details><summary>Hint 2: API names</summary>

`transforms.Compose`, `transforms.Resize((s, s), interpolation=transforms.InterpolationMode.BICUBIC)`, `transforms.RandomResizedCrop(s, scale=(0.5, 1.0))`, `transforms.RandomHorizontalFlip()`, `transforms.RandomVerticalFlip()`, `transforms.ToTensor()`, `transforms.Normalize(mean, std)`.
`BloodMNIST(split=..., download=True, size=64, transform=..., target_transform=...)`.
`DataLoader(ds, batch_size, shuffle, drop_last, num_workers, pin_memory, persistent_workers)`.
</details>

<details><summary>Hint 3: the shape of build_loader</summary>

```python
ds = build_dataset(split, img_size, augment)
return DataLoader(ds, batch_size=batch_size, shuffle=for_training, drop_last=for_training, ...)
```
</details>

> [!question] Explain it
> Why must the **probe** loader not shuffle and not drop the last batch? Why is `label_to_int` a named function and not a lambda?

---

## P2: Explore the model with docs only (30 min). No tests; tell me your answers

The skill under test is **finding things in an unfamiliar library fast.** That's the interview skill. Open a REPL (`uv run python`) and answer:

```python
from transformers import ViTMAEForPreTraining
m = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
import inspect, transformers.models.vit_mae.modeling_vit_mae as src; print(src.__file__)  # the source, open it
```

1. Exact module names of the **query** and **value** projections in encoder block 1 and decoder block 1. (`named_modules()`)
2. Parameter count: total, encoder (`m.vit`), decoder. What % of the encoder is **one** block?
3. `out = m(pixel_values=torch.randn(2,3,224,224))`. Shapes of `out.logits`, `out.mask`, `out.ids_restore`. **Why is the last dim of logits 768?** (Careful: it's not the hidden size.)
4. Find the loss in the source. Is it averaged over all patches or only masked ones? What's `norm_pix_loss` for this checkpoint?
5. `m.vit(pixel_values=x).last_hidden_state.shape` is `(2, 50, 768)`. Why 50?
6. Find `random_masking`. How would you turn masking **off**? What is the `noise` argument for?
7. Which parameters have `requires_grad=False` out of the box? (Older transformers versions froze the position embeddings. What does yours do?)

<details><summary>Answers (only after you've tried)</summary>

1. `vit.layers.0.attention.q_proj` / `v_proj`; `decoder.decoder_layers.0.attention.q_proj` / `v_proj`. **Trap:** blogs and LLMs say `attention.attention.query`. That's transformers v4. You have v5.
2. 111.9M total; encoder 85.8M; decoder ≈ 26.1M. One encoder block = 7.09M ≈ 8.3% of the encoder.
3. logits (2, 196, 768); mask (2, 196); ids_restore (2, 196). 768 = 16·16·3 = the **pixels of one patch**. It equals the hidden size only by coincidence.
4. MSE per patch → `(loss * mask).sum() / mask.sum()` → **masked patches only**. `norm_pix_loss=False` here (raw normalised pixels are the target).
5. 1 CLS + 49 visible patches (25% of 196).
6. `len_keep = int(seq_length * (1 - self.config.mask_ratio))` → set `config.mask_ratio = 0`. `noise` decides **which** patches are kept (smallest noise kept), so a fixed noise gives a fixed mask (`viz.py` uses this for fair before/after pictures).
7. None. In v5 everything is trainable by default, even the sin-cos position embeddings. That's why "freeze everything first" matters.
</details>

---

## P3: Linear probe + baseline run (45 min) · `probe.py` · **Task 1**

**Goal:** frozen encoder → embeddings → logistic regression → test accuracy. Viva question (a).
**Docs:** [ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae) · [AMP autocast](https://pytorch.org/docs/stable/amp.html) · sklearn `make_pipeline`, `StandardScaler`, `LogisticRegression`, `accuracy_score`, `f1_score`
**Check:** `uv run pytest tests/test_p3_probe.py -v`

<details><summary>Hint 1: the two traps</summary>

(1) The encoder masks 75% of patches **even in eval mode**, so your features would be random. Set `encoder.config.mask_ratio = 0.0` while extracting. (2) That config object is **shared** with the full pretraining model, so restore it afterwards (use `try/finally`), or Stage 1 silently trains with no masking.
</details>

<details><summary>Hint 2: API names</summary>

`encoder.eval()`, `encoder.training` (to remember the mode), `torch.autocast(device_type=device.type, dtype=torch.float16, enabled=...)`, `out.last_hidden_state`, `h[:, 0]`, `h[:, 1:].mean(dim=1)`, `.float().cpu()`, `torch.cat(list).numpy()`.
</details>

<details><summary>Hint 3: skeleton</summary>

```python
old, was_training = encoder.config.mask_ratio, encoder.training
encoder.config.mask_ratio = 0.0; encoder.eval()
try:
    for x, y in loader:
        ...  # forward under autocast, collect cls / mean / labels
finally:
    encoder.config.mask_ratio = old; encoder.train(was_training)
```
</details>

**Run Task 1** (~1.5 min):
```bash
uv run python scripts/run_probe.py --tag baseline
```

> [!question] Explain it
> Why fit the StandardScaler on train only? CLS vs mean pooling: which did better, and why might MAE's CLS token be weak? (MAE never puts a loss on CLS.) **Look at the train accuracy next to the test accuracy, and remember both for P7.**

```bash
git commit -am "Task 1: linear probe baseline"
```

---

## P4: LoRA from scratch (60 min) · `lora.py` (`__init__`, `forward`, `inject_lora`) · viva (b)

**Goal:** a `LoRALinear` you could write on a whiteboard. This is the single most likely interview coding question in this whole project.
**Docs:** [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) · [nn.init](https://pytorch.org/docs/stable/nn.init.html) · `Module.get_submodule`
**Check:** `uv run pytest tests/test_p4_lora.py -v` (`merged_linear` / `merge_lora` wait for P6)

**Predict first (numerical):** rank 8 on one 768×768 projection is how many trainable parameters, as a % of that layer?

<details><summary>Hint 1: shapes</summary>

`x: (..., in)`. `x @ A.T: (..., r)`. `@ B.T: (..., out)`. Multiply by `alpha / r`, add `base(x)`. Writing it with `@ .T` (not `F.linear`) keeps it working for 2-D and 3-D inputs.
</details>

<details><summary>Hint 2: API names</summary>

`for p in base.parameters(): p.requires_grad_(False)`; `nn.Parameter(torch.empty(r, in_f, device=..., dtype=...))`; `nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))`; `torch.zeros(out_f, r, ...)`. In `inject_lora`: `model.get_submodule(name)`, `isinstance(m, nn.Linear)`, and `set_submodule` (given).
</details>

<details><summary>Hint 3: forward</summary>

```python
return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
```
</details>

> [!question] Explain it
> Why B = 0 and not A = 0? (What's the gradient of B when A = 0?) What does `alpha / r` buy you when you change r? Why can `setattr(parent, "3", new)` replace element 3 of a `ModuleList`?

```bash
git commit -am "LoRA from scratch: LoRALinear + inject_lora"
```

---

## P5: ExPLoRA + the training loop (90 min) · `explora.py`, `train.py` · **Task 2** · viva (c)

**Check:** `uv run pytest tests/test_p5_explora.py tests/test_p5_train.py -v`

### 5a. `configure_explora`: what trains

**Predict first (your favourite kind of numerical):** with r = 8, how many parameters will train? Build it up:
1. one fully unfrozen encoder block (q, k, v, o + fc1, fc2 + 2 LayerNorms, all with biases)
2. LoRA on q, v of encoder blocks 1–11 (dim 768)
3. LoRA on q, v of all 8 decoder blocks (dim 512)
4. LayerNorms: 2 per block in blocks 1–11, the final encoder norm, 2 per decoder block, and the decoder norm

Write your number down, then compare it with what `run_stage1.py` prints.

<details><summary>The full answer</summary>

| Piece | Count |
|---|---|
| Block 12: attention 4·(768²+768) = 2,362,368 · fc1 768·3072+3072 = 2,362,368 · fc2 3072·768+768 = 2,360,064 · LN 3,072 | **7,087,872** |
| LoRA encoder: 11 blocks × 2 × 8·(768+768) | **270,336** |
| LoRA decoder: 8 blocks × 2 × 8·(512+512) | **131,072** |
| LayerNorms: 11·3,072 + 1,536 + 8·2,048 + 1,024 | **52,736** |
| **Total trainable** | **7,542,016 = 6.72% of 112.3M** |

94% of the trainable budget is block 12. LoRA itself is only 0.36% of the model.
</details>

<details><summary>Hint: order of operations</summary>

Freeze all → unfreeze the chosen blocks → build the list of target names (skip unfrozen blocks) → `inject_lora` (new LoRA params are born trainable) → unfreeze every `nn.LayerNorm`. If you freeze *after* injecting, you freeze LoRA too.
</details>

### 5b. `build_optimizer` + `train_one_epoch`: the fp16 loop
The recipe is in `train.py`'s module docstring. Two traps: (1) the optimizer must only see trainable params, with no weight decay on 1-D params (biases, LN); (2) `scaler.unscale_` **before** `clip_grad_norm_`.

<details><summary>Hint: one step</summary>

```python
with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
    loss = model(pixel_values=images).loss
optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()
# unscale + clip, then scaler.step / scaler.update / scheduler.step
```
</details>

### 5c. Time an epoch, then train
```bash
uv run python scripts/run_stage1.py --time-only            # the assignment says: time one epoch first
uv run python scripts/run_stage1.py --epochs 20            # ~20 min locally, checkpoint every epoch
```
For the course's "Colab/Kaggle + Drive" requirement, use `notebooks/stage1_colab_kaggle.ipynb` (same scripts, `--out-dir` on Drive, `--resume` after disconnects).

Watch on wandb: `train/loss` should fall fast for the first ~2 epochs, then flatten. `epoch/recon` shows the reconstructions sharpening epoch by epoch.

> [!question] Explain it
> Why is the gradient still computed through *all 12* encoder blocks even though block 1 is frozen? (Hint: where does LoRA live?) So why is ExPLoRA cheap at all? (Optimizer state, checkpoint size, and less overfitting with few trainable params.) Why the same MAE loss as the original pretraining?

```bash
git commit -am "ExPLoRA Stage 1: configure_explora + fp16 training loop"
```

---

## P6: Merge + probe again (30 min) · `merged_linear`, `merge_lora` · **Task 3**

**Check:** `uv run pytest tests/test_p6_merge.py -v`, then the whole suite: `uv run pytest -q` → 47 passed.

```bash
uv run python scripts/run_probe.py --tag explora --checkpoint checkpoints/stage1/final.pt
```
The script rebuilds the ExPLoRA structure, loads your trained deltas, **checks that merging doesn't change the function** (prints max |before − after|), then runs the identical probe and prints the comparison table.

> [!question] Explain it
> Why merge in float32? After merging, what's the inference overhead of LoRA? Why does `merge_lora` produce the exact original `state_dict` keys, and why does that matter?

```bash
git commit -am "Task 3: merge LoRA (W = W0 + BA) and re-probe"
```

---

## P7: Read your results like a researcher (30 min)

Answer these in the README before the viva:
1. Baseline vs ExPLoRA, CLS vs mean. What's the gap, and is it bigger than run-to-run noise? (How would you find out?)
2. **Train accuracy.** If the probe already fits the train set perfectly with frozen ImageNet features, how much room is left for *any* encoder to show it's better? What does that say about this probe as a ruler?
3. ExPLoRA's pitch is "**labels are scarce**, unlabeled data is plentiful." Your probe uses all 11,959 labels. What evaluation would actually test the pitch?
4. The reconstruction loss fell. Does lower reconstruction loss *guarantee* better linear-probe features? (MAE's own paper has a famous answer.)

Bring your answers to me. Depending on what your numbers show, there's a stretch experiment (low-label probing, which is pandas/sklearn practice for your interviews) I'll unlock.

---

## P8: README + viva rehearsal (45 min)
- README: setup (uv), how to run, **wandb link**, results table, the reconstruction images (`outputs/recon_before.png`, `outputs/recon_after.png`), the loss curve, your P7 answers.
- Then tell me **"viva"**. I'll run a 15-minute mock one-on-one: questions (a)–(d), plus the follow-ups examiners love.

---

## Interview reps (the crossover)
After a phase is green, do its rep: **blank file, docs allowed, no LLM, timer on.** Over the limit counts as a miss.

| Rep | Task | Limit |
|---|---|---|
| R1 | `LoRALinear` (init, forward, merge) from a blank file | 10 min |
| R2 | fp16 training loop with GradScaler, clipping, scheduler | 10 min |
| R3 | Load `bert-base-uncased`, freeze all, add LoRA to every query/value. **Find the module names yourself** (they differ from ViT-MAE's), print trainable % | 12 min |
| R4 | Frozen-encoder feature extraction + sklearn probe for any HF vision model | 12 min |
| R5 | `Dataset` + `DataLoader` + transforms over a raw numpy array of images | 8 min |

R3 is the real interview skill: unfamiliar model, docs only, find the right names, get it working.
