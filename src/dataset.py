
"""
Final LAV-DF dataset loader.

IMPORTANT:
The trained checkpoints were trained with:
    uint8 [0,255]
        -> float32 / 255.0
        -> HWC -> CHW

NO ImageNet normalization is applied here.

Both weak and fully-supervised models use the exact same
frame preprocessing and 16-frame sampling.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMG_SIZE = 224
N_FRAMES_PER_CLIP = 16


def uniform_sample_indices(n_available, n_target):
    """Uniformly sample n_target frames."""
    if n_available <= 0:
        raise ValueError("No frames available.")

    if n_available >= n_target:
        step = n_available / n_target
        return [int(i * step) for i in range(n_target)]

    # Repeat frames cyclically if fewer than 16 are available.
    return [i % n_available for i in range(n_target)]


def prepare_frame(frame):
    """
    Convert stored uint8 HWC RGB frame to:
        float32 CHW
        range [0,1]
    """

    frame = np.asarray(frame)

    if frame.ndim != 3:
        raise ValueError(
            f"Expected HWC frame, got shape {frame.shape}"
        )

    if frame.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB frame, got shape {frame.shape}"
        )

    # EXACT training preprocessing
    frame = frame.astype(np.float32) / 255.0

    # HWC -> CHW
    frame = np.transpose(frame, (2, 0, 1))

    tensor = torch.from_numpy(frame)

    # Resize only if needed.
    if tensor.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return tensor


class VideoClipDataset(Dataset):

    def __init__(self, labels_csv, split, mode):

        if mode not in ("weak", "full"):
            raise ValueError(
                "mode must be 'weak' or 'full'"
            )

        if split == "val":
            split = "dev"

        self.labels_csv = labels_csv
        self.mode = mode
        self.split = split

        df = pd.read_csv(labels_csv)

        required = {
            "file",
            "split",
            "npz_file",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"labels.csv missing columns: {sorted(missing)}"
            )

        split_df = (
            df[df["split"] == split]
            .copy()
            .reset_index(drop=True)
        )

        self.video_records = (
            split_df[
                ["file", "npz_file"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if len(self.video_records) == 0:
            available = sorted(
                df["split"]
                .dropna()
                .unique()
                .tolist()
            )

            raise ValueError(
                f"No videos found for split='{split}'. "
                f"Available splits: {available}"
            )

        self.frames_dir = os.path.join(
            os.path.dirname(labels_csv),
            "frames",
        )

    def __len__(self):
        return len(self.video_records)

    def __getitem__(self, idx):

        record = self.video_records.iloc[idx]

        npz_name = str(record["npz_file"])

        npz_path = os.path.join(
            self.frames_dir,
            npz_name,
        )

        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"NPZ file not found: {npz_path}"
            )

        data = np.load(
            npz_path,
            allow_pickle=False,
        )

        required_keys = {
            "frames",
            "frame_labels",
            "frame_times",
            "video_label",
        }

        missing = (
            required_keys - set(data.files)
        )

        if missing:
            raise ValueError(
                f"{npz_name} missing keys: "
                f"{sorted(missing)}"
            )

        frames = data["frames"]
        frame_labels_all = data["frame_labels"]
        video_label = data["video_label"]

        sample_indices = uniform_sample_indices(
            len(frames),
            N_FRAMES_PER_CLIP,
        )

        clip_frames = [
            prepare_frame(frames[i])
            for i in sample_indices
        ]

        clip = torch.stack(
            clip_frames,
            dim=0,
        )

        frame_labels = torch.tensor(
            frame_labels_all[sample_indices],
            dtype=torch.float32,
        )

        video_label = torch.tensor(
            float(video_label),
            dtype=torch.float32,
        )

        if self.mode == "full":
            return clip, frame_labels

        return clip, video_label
