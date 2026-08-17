"""
Phase 1: Data preparation for LAV-DF.

Reads LAV-DF's metadata, stratified-subsamples N videos, extracts frames at a
fixed fps using decord, and writes a labels CSV containing BOTH:
  - frame-level labels (for the fully-supervised arm)
  - video-level labels (for the weakly-supervised arm, derived by collapsing
    frame-level labels — if ANY frame is fake, the video label is fake)

Output layout:
  out_dir/
    frames/<video_id>/frame_%05d.jpg
    labels.csv   (columns: video_id, frame_idx, frame_path, frame_label, video_label, split)

Usage:
  python data_prep.py --raw_dir <path to LAV-DF raw> --out_dir <processed output dir> \
      --n_videos 400 --fps 2
"""
import argparse
import json
import os
import random

import cv2
import pandas as pd
from decord import VideoReader, cpu
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_lavdf_metadata(raw_dir):
    """
    Loads LAV-DF's metadata.json.

    VERIFY: LAV-DF's metadata.json is a list of records. Confirm these key
    names against the actual file you download — dataset schemas can change
    between releases. Expected shape (as of the original LAV-DF release):
      {
        "file": "videos/000001.mp4",
        "n_fakes": 1,
        "fake_segments": [[1.2, 3.4]],   # seconds
        "video_frames": 150,
        "video_fps": 25,
        "split": "train" | "val" | "test"
      }
    """
    meta_path = os.path.join(raw_dir, "metadata.json")
    with open(meta_path, "r") as f:
        records = json.load(f)
    return records


def stratified_subsample(records, n_videos, seed=42):
    """Keep a roughly balanced mix of real (n_fakes == 0) and fake videos."""
    random.seed(seed)
    real = [r for r in records if r.get("n_fakes", 0) == 0]
    fake = [r for r in records if r.get("n_fakes", 0) > 0]

    n_real = min(len(real), n_videos // 2)
    n_fake = min(len(fake), n_videos - n_real)

    random.shuffle(real)
    random.shuffle(fake)
    return real[:n_real] + fake[:n_fake]


def extract_frames_and_labels(record, raw_dir, out_dir, fps):
    video_rel_path = record["file"]  # VERIFY: key name
    video_path = os.path.join(raw_dir, video_rel_path)
    video_id = os.path.splitext(os.path.basename(video_rel_path))[0]

    frame_out_dir = os.path.join(out_dir, "frames", video_id)
    os.makedirs(frame_out_dir, exist_ok=True)

    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration_sec = total_frames / native_fps

    step = max(1, round(native_fps / fps))
    sample_indices = list(range(0, total_frames, step))

    fake_segments = record.get("fake_segments", [])  # list of [start_sec, end_sec]

    rows = []
    for i, frame_idx in enumerate(sample_indices):
        frame = vr[frame_idx].asnumpy()  # RGB
        timestamp_sec = frame_idx / native_fps

        frame_label = 0  # real
        for seg_start, seg_end in fake_segments:
            if seg_start <= timestamp_sec <= seg_end:
                frame_label = 1  # fake
                break

        frame_path = os.path.join(frame_out_dir, f"frame_{i:05d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        rows.append({
            "video_id": video_id,
            "frame_idx": i,
            "frame_path": frame_path,
            "frame_label": frame_label,
        })

    video_label = 1 if len(fake_segments) > 0 else 0
    for r in rows:
        r["video_label"] = video_label

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="Path to downloaded LAV-DF raw data")
    parser.add_argument("--out_dir", required=True, help="Where processed frames/labels go")
    parser.add_argument("--n_videos", type=int, default=400)
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading metadata...")
    records = load_lavdf_metadata(args.raw_dir)

    print(f"Subsampling to ~{args.n_videos} videos (stratified real/fake)...")
    subset = stratified_subsample(records, args.n_videos, seed=args.seed)

    all_rows = []
    for record in tqdm(subset, desc="Extracting frames"):
        try:
            rows = extract_frames_and_labels(record, args.raw_dir, args.out_dir, args.fps)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Skipping {record.get('file')} due to error: {e}")

    df = pd.DataFrame(all_rows)

    # Split at the VIDEO level, not frame level, so no video's frames leak across splits
    video_ids = df["video_id"].unique()
    train_ids, temp_ids = train_test_split(video_ids, test_size=0.3, random_state=args.seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=args.seed)

    split_map = {vid: "train" for vid in train_ids}
    split_map.update({vid: "val" for vid in val_ids})
    split_map.update({vid: "test" for vid in test_ids})
    df["split"] = df["video_id"].map(split_map)

    out_csv = os.path.join(args.out_dir, "labels.csv")
    df.to_csv(out_csv, index=False)

    print(f"Done. {df['video_id'].nunique()} videos, {len(df)} frames written to {out_csv}")
    print(df.groupby("split")["video_id"].nunique())


if __name__ == "__main__":
    main()
