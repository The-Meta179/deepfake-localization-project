# Implementation Walkthrough — Step 1 to Done

Follow these in order. Each step says what to do, what "done" looks like, and what commonly goes wrong.

---

## STEP 1 — Create accounts (Day 1, ~30 min)

1. Google account → open https://colab.research.google.com and confirm it loads.
2. Kaggle account → https://www.kaggle.com → Settings → Phone Verification → enable it (Kaggle won't give you GPU access without this).
3. GitHub account (if you don't have one) → create a **new empty repo** named e.g. `deepfake-localization-project`, set to Private if you prefer, add your teammate as a Collaborator (Settings → Collaborators).

**Done when:** you and your teammate both have GitHub, Colab, and Kaggle access confirmed.

---

## STEP 2 — Push the starter code to GitHub (Day 1, ~15 min)

On your own machine (or directly in Colab's terminal):

```bash
cd deepfake-project        # the folder you unzipped
git init
git remote add origin https://github.com/YOUR_USERNAME/deepfake-localization-project.git
git add .
git commit -m "Initial project scaffold"
git branch -M main
git push -u origin main
```

Have your teammate `git clone` the repo to confirm they can see the same files.

**Done when:** both of you can see `README.md`, `src/`, `notebooks/` on GitHub.

---

## STEP 3 — Get access to LAV-DF via Kaggle (Day 1, ~15 min — no waiting required)

LAV-DF's full research release (~135,000 clips total) is mirrored on Kaggle. Use that instead of the original gated form — it avoids any multi-day approval wait, and you never download the raw pile to your own Drive at all:

1. Search Kaggle for "Localized Audio Visual DeepFake Dataset LAV-DF" and open the dataset page.
2. Don't download it to your computer. Instead, create a **new Kaggle Notebook**, and in the right-hand panel click **Add Data** → search for the same dataset → attach it. Kaggle mounts it read-only at `/kaggle/input/...` inside the notebook — no download, no storage used on your end.
3. Turn on GPU for this notebook too (Settings → Accelerator → GPU T4x2), since Phase 1 (frame extraction) also benefits from being run here.
4. `!pip install -r requirements.txt` in this Kaggle notebook the same way you would in Colab.

**Done when:** you can run `!ls /kaggle/input/` in a Kaggle notebook cell and see the LAV-DF folder contents (videos + metadata file).

**Why Kaggle and not Drive for the raw data:** you truly don't need the full ~135K-clip dataset — only your ~400-video subsample. Keeping the huge raw pile on Kaggle's mounted storage (not yours) and only exporting your small processed subsample to Drive keeps your storage and download time under control.

---

## STEP 4 — Sanity-check the pipeline with fake data (Day 2, while waiting on Step 3)

This proves your code runs end-to-end before you burn GPU hours on real data.

1. Create a tiny throwaway folder with 2–3 short `.mp4` clips you have lying around (any video works for this test).
2. Hand-write a matching `metadata.json`:
   ```json
   [
     {"file": "videos/test1.mp4", "n_fakes": 1, "fake_segments": [[1.0, 2.5]]},
     {"file": "videos/test2.mp4", "n_fakes": 0, "fake_segments": []}
   ]
   ```
3. In Colab, run Step 5 below but with `--raw_dir` pointed at this test folder and `--n_videos 2`.

**Done when:** `data_prep.py` runs without errors and `labels.csv` is created with sensible rows (open it and eyeball it).

---

## STEP 5 — Open Colab and run setup (Day 2)

1. Open `notebooks/colab_setup.ipynb` in Colab (File → Upload notebook, or open directly from GitHub via File → Open notebook → GitHub tab).
2. Runtime → Change runtime type → **T4 GPU** → Save.
3. Run the first 4 cells (mount Drive, clone repo, install requirements, create Drive folders).

**Common error:** `git clone` fails with "repository not found" → check the URL matches your actual repo, and that it's public or you're authenticated (private repos need a GitHub token — see Step 5a below).

**Step 5a (only if repo is private):** generate a GitHub Personal Access Token (Settings → Developer settings → Tokens) and clone with:
```
!git clone https://<TOKEN>@github.com/YOUR_USERNAME/deepfake-localization-project.git
```

**Done when:** `!pip install -q -r requirements.txt` finishes with no red errors, and `DATA_DIR`/`CKPT_DIR` print correctly.

---

## STEP 6 — Run Phase 1 directly inside the Kaggle notebook (using Step 3's mount)

Since the raw data is already mounted read-only at `/kaggle/input/...` (Step 3), you don't upload or copy anything — you point `data_prep.py` straight at it, run frame extraction there, and only copy the small OUTPUT to Drive afterward.

1. In your Kaggle notebook, clone your GitHub repo the same way as Colab (`!git clone ...`).
2. Set `RAW_DIR` to the mounted path, e.g.:
   ```python
   RAW_DIR = '/kaggle/input/localized-audio-visual-deepfake-dataset-lav-df'  # match the exact folder name shown by !ls /kaggle/input/
   ```
3. **Open the actual `metadata.json` inside that mounted folder and compare field names against the `# VERIFY:` comments in `src/data_prep.py`.** If names differ (e.g. `"video_path"` instead of `"file"`), edit `data_prep.py`'s `load_lavdf_metadata` / `extract_frames_and_labels` functions to match — this is the one place real-world data may not match the template exactly.

**Done when:** you can `!ls {RAW_DIR}` in the Kaggle notebook and see the video files and `metadata.json`.

---

## STEP 7 — Run Phase 1 (data prep) for real, then move the small output to Drive (Day 3–4)

Sanity run first:
```python
!python src/data_prep.py --raw_dir {RAW_DIR} --out_dir {DATA_DIR}/processed --n_videos 20 --fps 2
```
Check `labels.csv` looks right (open it via `pandas.read_csv` in a cell, check `frame_label`/`video_label` columns have both 0s and 1s, check `split` column has train/val/test).

Then scale up:
```python
!python src/data_prep.py --raw_dir {RAW_DIR} --out_dir {DATA_DIR}/processed --n_videos 400 --fps 2
```
This will take a while (video decoding is slow) — expect it to run for an hour or more depending on video lengths. It's safe to let it run in the background; if Colab disconnects mid-way, just re-run the same command with a smaller `--n_videos` first to confirm nothing corrupted, then resume with the full count (note: `data_prep.py` doesn't auto-resume like the training scripts — if it's interrupted, just re-run it from scratch, since frame extraction is idempotent and fairly fast to redo per video).

**Done when:** `labels.csv` exists with your target video count, split roughly stratified real/fake.

**Getting it out of Kaggle and into Drive for training:** Kaggle notebooks don't have Drive access directly. Easiest path: zip your `processed/` output (`!zip -r processed.zip processed/`), download the zip from Kaggle's output panel (it's small now — just frames from ~400 short clips, not the original 135K-clip dataset), then upload that zip to your Drive folder and unzip it there before Step 8. This is a one-time transfer of a few GB, not the full dataset.

---

## STEP 8 — Run Phase 2 (train both arms) (Day 5–3 weeks, parallelized)

**Teammate A**, in their own Colab session:
```python
!python src/train_weak.py --data_dir {DATA_DIR}/processed --ckpt_dir {CKPT_DIR} --epochs 15
```

**Teammate B**, in their own Colab session (same Drive if shared, or their own Drive with data copied over):
```python
!python src/train_full.py --data_dir {DATA_DIR}/processed --ckpt_dir {CKPT_DIR} --epochs 15
```

Watch the printed `train_loss`/`val_loss` each epoch — val_loss should trend down. If it's flat or increasing after several epochs, that's a real signal to flag (see Step 8a).

**If Colab disconnects:** just re-run the exact same command — it auto-resumes from the last saved checkpoint epoch.

**Step 8a — if training loss won't go down:**
- Confirm `frame_label`/`video_label` in `labels.csv` actually contain both classes (not all 0 or all 1).
- Try a smaller `--lr` (e.g. `1e-4`) if loss is oscillating wildly.
- Try more epochs — 15 is a starting point, not a guarantee.

**Done when:** both `full_model.pt` and `weak_model.pt` exist in `{CKPT_DIR}`.

---

## STEP 9 — Run Phase 3 (evaluate & compare) (Day: after Step 8 finishes)

```python
!python src/evaluate.py --data_dir {DATA_DIR}/processed --ckpt_dir {CKPT_DIR}
```

This prints precision/recall/F1/AP@0.5 for both models and saves `tradeoff_plot.png` to `{CKPT_DIR}` — download it (right-click in Drive, or `!cp` to `/content` and use Colab's file browser to download).

**Done when:** you have concrete numbers for both arms and the plot image.

---

## STEP 10 — Analysis & write-up (final days)

Answer these explicitly in your report — this is where "compare both" becomes a real contribution rather than just two trained models:

1. What's your measured accuracy gap (full F1 − weak F1)? How does it compare to the survey's reported 5–10%?
2. What's your annotation-cost ratio (printed by `evaluate.py`)? Compare to the survey's 30–60×.
3. If your gap is much bigger/smaller than the survey's, why? (Likely candidates: smaller dataset, frozen backbone vs. fine-tuned, fewer epochs, simpler temporal head than transformer-based SOTA — say this explicitly, it's a legitimate and expected limitation, not a flaw to hide.)
4. Show the `tradeoff_plot.png` as your figure, alongside the survey's Fig. 2, and discuss the comparison.

**Done when:** you have a report/slide draft. Come back here and I can help structure that write-up or turn it into a formatted document once you have real numbers.

---

## Quick troubleshooting index

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch size too big for free-tier GPU | Lower `--batch_size` to 2 |
| Training loss is `nan` | Learning rate too high, or a bad batch | Lower `--lr`, check for corrupted frames |
| `decord` fails to read a video | Corrupted/unsupported video file | Wrap extraction in try/except (already done in `data_prep.py`) and skip |
| Colab disconnects mid-training | Free-tier idle/session limits | Just re-run the same `train_*.py` command — it resumes automatically |
| `git clone` fails in Colab | Private repo without auth | Use the token method in Step 5a |
| Val loss much worse than train loss | Overfitting on a small dataset | Expected at this scale — mention it as a limitation in Step 10, don't chase it endlessly |
