# MultiBypassTriplets2026 — Ensemble-Distilled 8-Model Ensemble

Surgical action triplet recognition (`<instrument, verb, target>`) on **MultiBypass-4C-T40**,
for the [MICCAI 2026 MultiBypassTriplets challenge](https://camma-public.github.io/mbt40_challenge/).

**4-fold CV (videowise ivt mAP, leak-free): 0.5181**

| Component | 4-fold CV |
|---|---|
| SPIRIT student (alone) | 0.5007 |
| Swin student (alone) | 0.4527 |
| **Ensemble `0.7·SPIRIT + 0.3·Swin`** | **0.5181** |

Per fold: f0 0.5321 / f1 0.4790 / f2 0.5223 / f3 0.5392

---

## Method

Two complementary architectures, each trained as **4 fold models**, then averaged
(8 models total at inference):

1. **SPIRIT student** — DINOv3 ViT-L/16 backbone (lower 16 blocks frozen) + TUF / PIC / TGR
   heads. Causal clip of 8 frames at stride 1, 224px. Uses raw frames only.
2. **Swin student** — Swin3D-B (Kinetics-400 init, then CholecT50 verb+tool pretraining) fused
   with two frozen front-end signals: ConvNeXt-B target features and a Mask2Former
   instrument presence/mask. Clip of 16 frames sampled at exponential offsets
   `[-181 … -1, 0]` (~3 min of context), 224px.

Both are **distilled from an ensemble teacher** rather than trained on ground truth alone.

### Ensemble distillation (the key step)

The teacher is an **out-of-fold ensemble** of three earlier models
(dense SPIRIT + dense Swin + keyframe SPIRIT), evaluated fold-wise so that every frame's
soft label comes from models that never trained on that frame (CV 0.492).
Its predictions over all 76,774 annotated frames become `teach/ensemble_soft_all.npz`.

Students then re-train on **all frames (dense)** with a distillation loss that follows the
SPIRIT paper's sample weighting:

```
w_n ∝ TS_n · exp(−TG_n / β)
```

where `TS_n` is teacher–student disagreement and `TG_n` is teacher–GT disagreement, so
frames where the teacher itself contradicts the (noisy) ground truth are down-weighted.

**Effect** — re-distilling from the ensemble teacher instead of a single-model teacher
raised the 4-fold CV from **0.4888 → 0.5181 (+0.029)**, and the 2-model student ensemble
matches the much larger raw teacher ensemble while being far cheaper to run.

Adding the teachers back into the final ensemble does **not** help
(students 0.5181 > students+teachers 0.5167 > teachers alone 0.4959): the students have
already absorbed the teacher's knowledge, and the weaker teachers only dilute it.

---

## Repository layout

```
configs/
  spirit/ensdistill_dense_f{0..3}.yaml   # the exact configs the submitted students were trained with
  swin/ensdistill_dense_f{0..3}.yaml
  swin/pretrain_swin3d_verbtool.yaml     # CholecT50 verb+tool pretraining
  frontend/target_convnext.yaml          # frozen ConvNeXt target model
src/
  spirit/    train_spirit_ft.py, model_spirit_ft.py, model_spirit_full.py, model_spirit.py
  swin/      train_verb_fusion.py, cache_frozen_feats.py, pretrain_swin3d_cholec.py, ensemble_eval.py
  frontend/  train_target.py, train_target_mtl.py, train_mask2former.py, build_stage.py, infer_multibypass.py
  common/utils/                          # triplet mappings + metrics (from the official starter kit)
folds/folds.csv                          # video-level 4-fold split (GroupKFold by video)
submit/                                  # inference container (what was submitted)
scripts/                                 # end-to-end reproduction driver
docs/                                    # method report material
```

Configs refer to `${REPO}` (this repository), `${WORK}` (a scratch dir for caches and
checkpoints) and `${MB_DATA}` (the challenge data root). Set them before training —
see `scripts/env.sh`.

---

## Reproduction

### 0. Environment

The project is managed with [uv](https://docs.astral.sh/uv/) and ships a lockfile, so the
exact dependency set that produced the submitted container is reproducible:

```bash
uv sync                 # inference + evaluation
uv sync --extra train   # add the training-only extras (wandb, albumentations, matplotlib)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`uv sync` creates `.venv/` and installs torch 2.6.0 / torchvision 0.21.0 from the
**CUDA 11.8** index (pinned in `pyproject.toml`, matching the evaluation server). Prefix
commands with `uv run`, or activate `.venv` and call `python` directly.

<details>
<summary>pip alternative</summary>

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
</details>

The SPIRIT branch additionally needs the DINOv3 backbone code and weights:

```bash
git clone https://github.com/facebookresearch/dinov3 third_party/dinov3
# put dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth under ${WORK}/weights/
# Keep that exact filename — dinov3's hubconf parses the weights filename.
```

Data: obtain **MultiBypass-4C-T40** from the challenge organisers and point `${MB_DATA}`
at the directory holding `videos/<VID>/<6-digit>.jpg` and `label_files_challenge/<VID>.json`.
This repository ships no data and no trained weights.

### 1. Folds

`folds/folds.csv` is the split used for every number reported here: **GroupKFold over
videos** (16 videos → 4 folds × 4 videos), so no video appears in both train and val.

### 2. Front-end models (needed by the Swin branch only)

```bash
bash scripts/01_frontend.sh     # ConvNeXt-B target model + Mask2Former instrument model, per fold
bash scripts/02_frozen_cache.sh # cache their outputs for every frame -> ${WORK}/frozen_cache_oof_dense
```

The cache is built **out-of-fold**: each video's features come from the fold model that did
not train on it. At test time there is no leak concern, so the inference container uses a
single front-end set.

### 3. Base models and the ensemble teacher

```bash
bash scripts/03_base_models.sh  # dense SPIRIT, dense Swin, keyframe SPIRIT (per fold)
bash scripts/04_teacher.sh      # OOF-ensemble their predictions -> ${WORK}/teach/ensemble_soft_all.npz
```

### 4. The submitted students

```bash
bash scripts/05_students.sh     # 4 SPIRIT + 4 Swin students, dense, distilled from the teacher
bash scripts/06_evaluate.sh     # 4-fold CV of the 2-student ensemble -> expect ~0.5181
```

### 5. Inference container

```bash
cd submit
bash build.sh
bash test.sh /path/to/data_root      # mounts /data, writes /results, checks the output contract
bash export.sh <team>_<CC>_<initials>_<date>_<version>
```

The container reads `/data/MultiBypass-4C-T40/...` and writes
`/results/multibypass_triplet_predictions.json` as `{video_id: {frame_id: [85 scores]}}`.

---

## Inference cost

Measured on 512 frames, model loading included, against the organisers' reported timing for
an earlier 3-model submission (114 s on the evaluation server):

| Pipeline | Eval server | Local RTX 4090 |
|---|---|---|
| earlier 3-model submission | 114 s | 113.6 s |
| **this 8-model ensemble** | — | **121.3 s** |

The local box reproduces the server timing to within 0.4%, so the 8-model ensemble is
**≈1.07× the earlier submission ≈ 0.235 s/frame**, i.e. about **5.1 h** for a 78,000-frame
hidden test — inside the 8 h budget. Peak usage is 6.4 GB GPU / 7.1 GB CPU RAM.

Eight models cost only ~7% more than three because the pipeline removes the redundant work:

* **DINOv3 lower trunk is shared.** With `freeze_blocks=16` the patch embedding and blocks
  0–15 are bit-identical across the 4 SPIRIT students, so they run **once**; only blocks
  16–23 and the heads run per fold. Verified to change the output by ≤3e-5 (rounding).
* **The front-end runs once per frame** and feeds all 4 Swin students.
* Frames are decoded once per video into a rolling buffer, and everything is batched.

---

## Notes for anyone extending this

* The inference container **imports the training model definitions verbatim**
  (`submit/src/`) instead of re-implementing them. A hand-written copy silently diverged
  (a feature-concat ordering difference) and cost ~0.15 mAP before it was caught by
  comparing against stored OOF predictions. Always diff new inference code against the
  training path on real frames.
* **DINOv3 ViT-L must not be cast with `.half()`** — it overflows to NaN. Use autocast and
  keep fp32 weights.
* The container must run **offline**. `torchvision`'s `swin3d_b(weights=...)` downloads
  Kinetics weights at construction time, so they are bundled under `$TORCH_HOME`; verify
  with `docker run --network none`.

---

## Acknowledgements

Built on the official [starter kit](https://github.com/CAMMA-public/multibypasstriplets2026_starter_kit)
(triplet mappings, videowise metrics) and on
[SPIRIT](https://arxiv.org/abs/2608.02188) for the head design and the distillation recipe.
Backbones: DINOv3 (Meta AI), Swin3D and ConvNeXt (torchvision / timm), Mask2Former (HuggingFace).
