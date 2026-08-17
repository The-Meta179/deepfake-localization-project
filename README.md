# Temporal Deepfake Localization — Supervision Trade-off Study

Companion code for `deepfake_localization_project_scope.md`. Follow this README top to bottom — each step assumes the previous one is done.

## 0. Accounts & access (do this FIRST — longest lead time)

1. **Google account** for Colab + Drive (15GB free storage — should be enough for the subsampled dataset; if not, Kaggle also gives you scratch disk during a session).
2. **Kaggle account** — gives ~30 GPU-hours/week (T4 x2 or P100), separate quota from Colab. Verify your phone number to unlock GPU access (required by Kaggle).
3. **LAV-DF dataset access** — request it now: https://github.com/ControlNet/LAV-DF (follow the repo's access instructions; some deepfake datasets gate downloads behind a form — approval can take a few days). While waiting, you can start Phase 0/1 scaffolding with a handful of placeholder videos.
4. **GitHub repo** — create one now (e.g. `deepfake-localization-project`), both teammates as collaborators. Push everything in this folder as your first commit.

## 1. Repo structure

```
deepfake-project/
├── requirements.txt
├── src/
│   ├── data_prep.py      # Phase 1: subsample + extract frames + build labels
│   ├── dataset.py         # PyTorch Dataset classes
│   ├── models.py          # Backbone + weak (MIL) head + full (per-frame) head
│   ├── train_weak.py      # Phase 2a
│   ├── train_full.py      # Phase 2b
│   └── evaluate.py        # Phase 3
└── notebooks/
    └── colab_setup.ipynb  # Run this first in Colab
```

## 2. Colab setup (run once per session — sessions don't persist installs)

Open `notebooks/colab_setup.ipynb` in Colab, or paste this into a new cell:

```python
# Mount Drive — this is where checkpoints and data live, NOT Colab's local disk
from google.colab import drive
drive.mount('/content/drive')

# Clone your repo
!git clone https://github.com/YOUR_USERNAME/deepfake-localization-project.git
%cd deepfake-localization-project

# Install deps
!pip install -q -r requirements.txt

# Point all scripts at Drive for persistence
import os
os.makedirs('/content/drive/MyDrive/deepfake_project/data', exist_ok=True)
os.makedirs('/content/drive/MyDrive/deepfake_project/checkpoints', exist_ok=True)
DATA_DIR = '/content/drive/MyDrive/deepfake_project/data'
CKPT_DIR = '/content/drive/MyDrive/deepfake_project/checkpoints'
```

For **Kaggle**: create a new Notebook, add the LAV-DF dataset (or your uploaded subset) via "Add Data," turn on GPU (Settings → Accelerator → GPU T4x2), and `!pip install -r requirements.txt` the same way — Kaggle's `/kaggle/working` persists for the session but download outputs before it ends.

## 3. Run order

```bash
# Phase 1 — once LAV-DF access is approved and downloaded to DATA_DIR/raw
python src/data_prep.py --raw_dir $DATA_DIR/raw --out_dir $DATA_DIR/processed --n_videos 400 --fps 2

# Phase 2a — weakly-supervised (video-level labels only)
python src/train_weak.py --data_dir $DATA_DIR/processed --ckpt_dir $CKPT_DIR --epochs 15

# Phase 2b — fully-supervised (frame-level labels), run in parallel by your teammate
python src/train_full.py --data_dir $DATA_DIR/processed --ckpt_dir $CKPT_DIR --epochs 15

# Phase 3 — compare both
python src/evaluate.py --data_dir $DATA_DIR/processed --ckpt_dir $CKPT_DIR
```

## 4. Checkpointing discipline (avoid losing work to disconnects)

Both `train_weak.py` and `train_full.py` save a checkpoint every epoch to `$CKPT_DIR` and **auto-resume** from the latest checkpoint if one exists — just re-run the same command after a disconnect. Don't rely on Colab's local `/content` disk for anything you can't afford to lose.

## 5. Sanity-check before scaling up

The first time you run `data_prep.py` and `train_weak.py`/`train_full.py`, use `--n_videos 20` to confirm the whole pipeline runs end-to-end in a few minutes before committing GPU hours to the full ~400-video run.

## 6. What's a stub vs. what's ready to run

- `data_prep.py`: the frame-extraction and stratified-sampling logic is ready to run, but the **LAV-DF metadata field names are marked with `# VERIFY:` comments** — open the actual `metadata.json` LAV-DF ships and confirm key names match before running on real data (dataset metadata schemas do shift between releases).
- `dataset.py`, `models.py`, `train_weak.py`, `train_full.py`, `evaluate.py`: fully implemented and should run as-is once `data_prep.py`'s output CSV exists.
